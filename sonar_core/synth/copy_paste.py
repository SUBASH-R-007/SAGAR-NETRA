"""Copy-paste target synthesis for ground-range tiles.

Rare debris classes (aircraft, human_body, mine_like) have almost no real
survey examples, so training data is manufactured by compositing a *chip* —
a real crop from another survey, or a synthetic render — into a background
tile. A naive paste fails in two sonar-specific ways that this module fixes:

* **Radiometry**: backscatter level depends on the sonar, TVG state and
  seabed type, so a chip's absolute intensities mean nothing in a new tile.
  The chip is rescaled by matching medians against the local background, so
  the chip's *internal* highlight-to-background contrast (the physically
  meaningful quantity) is what survives the paste.
* **Shadow**: a highlight without its acoustic shadow is a giveaway artifact
  and teaches the detector the wrong cue. After blending, a geometrically
  consistent shadow is ray-cast behind the pasted footprint via
  :func:`sonar_core.synth.shadow_render.render_shadow`, using the tile's own
  altitude and ground-range geometry.

Out-of-swath NaN pixels are sacrosanct: they mark ground range the ping never
insonified, so nothing may ever be pasted onto them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from sonar_core.synth.shadow_render import render_shadow

#: Sentinel bbox for a paste that landed entirely off-tile or on NaN fill.
EMPTY_BBOX: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class PasteResult:
    """One composited target and the label geometry it produces.

    ``bbox`` is half-open ``(r0, c0, r1, c1)`` around the pasted footprint
    (the visible highlight only — the shadow is context, not object, so it is
    deliberately outside the box, mirroring how real targets are labelled).
    """

    image: np.ndarray  # (h, w) float32 tile with chip + shadow composited
    mask: np.ndarray  # (h, w) bool pasted footprint, clipped to tile and swath
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1), half-open; EMPTY_BBOX if nothing landed
    height_m: float  # proud height used for the shadow (label for PhysiCheck)


def _local_background_median(patch: np.ndarray, fg: np.ndarray, full_tile: np.ndarray) -> float:
    """Median intensity of the seabed the chip lands on.

    Prefers the destination patch outside the footprint (the most local
    estimate), widens to the whole tile if the patch is fully covered or all
    NaN, and returns NaN only when the tile holds no finite pixel at all.
    """
    local = patch[~fg & np.isfinite(patch)]
    if local.size:
        return float(np.median(local))
    finite = full_tile[np.isfinite(full_tile)]
    return float(np.median(finite)) if finite.size else float("nan")


def paste_target(
    background: np.ndarray,
    chip: np.ndarray,
    chip_mask: np.ndarray,
    position_rc: tuple[int, int],
    ground_res: float,
    altitude_m: float | np.ndarray,
    nadir_col0: int,
    height_m: float,
    rng: np.random.Generator,
    blend_sigma: float = 1.0,
    shadow_level: float = 0.12,
    shadow_feather_px: float = 1.0,
    brightness_jitter: float = 0.1,
    fallback_ratio: float = 4.0,
    min_context_px: int = 16,
) -> PasteResult:
    """Composite *chip* into *background* with brightness matching and shadow.

    Parameters
    ----------
    background:
        ``(h, w)`` float ground-range tile; NaN marks out-of-swath fill.
        Never modified.
    chip:
        ``(ch, cw)`` float target crop (same intensity domain as any sonar
        image; absolute level is irrelevant — it is rescaled).
    chip_mask:
        ``(ch, cw)`` bool foreground footprint within the chip. Non-mask chip
        pixels are the chip's own local seabed context, used to measure its
        internal highlight contrast.
    position_rc:
        ``(row, col)`` of the chip's top-left corner in tile coordinates. May
        be negative or beyond the tile — the chip is clipped.
    ground_res, altitude_m, nadir_col0:
        Ground geometry of the tile, forwarded to
        :func:`~sonar_core.synth.shadow_render.render_shadow`; ``nadir_col0``
        is the absolute ground-range column of tile column 0 (``Tile.col0``).
    height_m:
        Proud height assigned to the pasted object; sets its shadow length
        and is returned as ground truth for shadow-physics validation.
    rng:
        Source of the brightness jitter, so paste batches are reproducible.
    blend_sigma:
        Gaussian feather (pixels) applied to the footprint alpha so the chip
        edge blends into the background instead of cutting a seam the
        detector could key on.
    shadow_level, shadow_feather_px:
        Forwarded to :func:`render_shadow`.
    brightness_jitter:
        Uniform relative jitter applied to the matched brightness scale
        (+-fraction), decorrelating pasted highlight levels across a dataset.
    fallback_ratio:
        Highlight-over-background ratio assumed when the chip carries too few
        context pixels to measure its own (e.g. a tightly cropped synthetic
        chip); the default matches the scene renderer's typical target
        reflectivity.
    min_context_px:
        Minimum finite non-foreground chip pixels required to trust the
        chip's own context median instead of *fallback_ratio*.

    Returns
    -------
    PasteResult
        With ``mask``/``bbox`` describing where the footprint actually landed
        (clipped to the tile and to finite-swath pixels). A paste that lands
        entirely off-tile or on NaN returns the untouched background copy
        with an all-False mask and :data:`EMPTY_BBOX`.
    """
    bg = np.asarray(background, dtype=np.float32)
    chp = np.asarray(chip, dtype=np.float64)
    cmask = np.asarray(chip_mask, dtype=bool)
    if bg.ndim != 2 or chp.ndim != 2:
        raise ValueError("background and chip must both be 2-D")
    if cmask.shape != chp.shape:
        raise ValueError(f"chip_mask shape {cmask.shape} != chip shape {chp.shape}")
    if not cmask.any():
        raise ValueError("chip_mask has no foreground pixel — nothing to paste")
    if blend_sigma < 0:
        raise ValueError(f"blend_sigma must be non-negative, got {blend_sigma}")

    out = bg.copy()
    h, w = out.shape
    ch, cw = chp.shape
    r_pos, c_pos = position_rc

    # Destination window clipped to the tile, and the matching chip window.
    dr0, dr1 = max(r_pos, 0), min(r_pos + ch, h)
    dc0, dc1 = max(c_pos, 0), min(c_pos + cw, w)
    if dr0 >= dr1 or dc0 >= dc1:  # chip entirely off-tile
        return PasteResult(out, np.zeros((h, w), dtype=bool), EMPTY_BBOX, float(height_m))
    sr0, sc0 = dr0 - r_pos, dc0 - c_pos
    chip_c = chp[sr0 : sr0 + (dr1 - dr0), sc0 : sc0 + (dc1 - dc0)]
    mask_c = cmask[sr0 : sr0 + (dr1 - dr0), sc0 : sc0 + (dc1 - dc0)]

    patch = out[dr0:dr1, dc0:dc1]

    # -- radiometric matching (chip statistics use the FULL chip so clipping
    # never changes the brightness of the visible part) --------------------
    fg_vals = chp[cmask & np.isfinite(chp)]
    ctx_vals = chp[~cmask & np.isfinite(chp)]
    bg_med = _local_background_median(patch, mask_c, bg)
    if np.isfinite(bg_med) and fg_vals.size:
        if ctx_vals.size >= min_context_px and np.median(ctx_vals) > 0:
            # Map the chip's own seabed level onto the local seabed level;
            # the highlight then lands at the chip's native contrast ratio.
            scale = bg_med / float(np.median(ctx_vals))
        else:
            # No trustworthy context: place the highlight at an assumed
            # ratio over the local background.
            fg_med = float(np.median(fg_vals))
            scale = (bg_med * fallback_ratio / fg_med) if fg_med > 0 else 1.0
    else:
        scale = 1.0
    scale *= 1.0 + rng.uniform(-brightness_jitter, brightness_jitter)

    # -- feathered alpha blend ---------------------------------------------
    alpha = gaussian_filter(mask_c.astype(np.float64), sigma=blend_sigma, mode="constant")
    peak = float(alpha.max())
    if peak > 0:
        alpha /= peak  # thin footprints still reach full opacity at their core
    alpha[~np.isfinite(chip_c)] = 0.0
    alpha[~np.isfinite(patch)] = 0.0  # NaN background is never pasted over
    chip_vals = np.where(np.isfinite(chip_c), chip_c * scale, 0.0)
    blended = alpha * chip_vals + (1.0 - alpha) * patch.astype(np.float64)
    out[dr0:dr1, dc0:dc1] = blended.astype(np.float32)

    # -- footprint bookkeeping and shadow ----------------------------------
    placed = np.zeros((h, w), dtype=bool)
    placed[dr0:dr1, dc0:dc1] = mask_c & np.isfinite(patch) & np.isfinite(chip_c)
    if not placed.any():
        return PasteResult(bg.copy(), placed, EMPTY_BBOX, float(height_m))

    out = render_shadow(
        out,
        placed,
        ground_res=ground_res,
        altitude_m=altitude_m,
        nadir_col0=nadir_col0,
        height_m=height_m,
        shadow_level=shadow_level,
        feather_px=shadow_feather_px,
    )

    rows = np.flatnonzero(placed.any(axis=1))
    cols = np.flatnonzero(placed.any(axis=0))
    bbox = (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
    return PasteResult(out, placed, bbox, float(height_m))
