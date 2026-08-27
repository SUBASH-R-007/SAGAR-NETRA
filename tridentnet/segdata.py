"""Synthetic segmentation-mask dataset for TridentNet Brain B (blueprint N-01/L2).

The scene renderer (:mod:`sonar_core.synth.scene`) seeds every
:class:`SynthTarget` with exact geometry — centre ping, ground range,
along-track length and a per-row across-track half-width
(:meth:`SynthTarget.half_width_at`) — so pixel-accurate masks are *free*:
no annotation, no weak labels. This module rasterizes those footprints onto
the same ground-range grid the M2 pipeline produces (rows = pings, columns =
ground range, column 0 at nadir on BOTH sides), then cuts image/mask chip
pairs from the **enhanced** ground image — exactly the imagery the segmenter
sees at inference time, since detector tiles are cut from that image.

**Why only nets and ropes** (:data:`FOREGROUND_CLASSES`): Brain B is a
filamentous-target specialist. Ghost nets and pipelines/ropes are thin,
sinuous and sprawling, so an axis-aligned bounding box wildly overestimates
their area and misplaces their centroid — yet their pixel footprint is the
quantity that matters (entanglement hazard extent, recovery planning, tight
geo-corners). Compact solid debris (drums, tires, containers, wrecks) is
already well served by Brain A boxes plus shadow physics; training the
segmenter on those blocky footprints would dilute the specialist and teach it
to fire on every bright blob. Non-foreground targets still appear in chips —
with *empty* masks — as explicit negatives, so the segmenter learns that a
drum highlight is not net.

Masks follow the ground-image convention everywhere: never mirror across
columns (shadows extend toward increasing column; a flip is physically
impossible imagery).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

from sonar_core.preprocess.pipeline import preprocess
from sonar_core.preprocess.slant_range import GroundImage
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.data import CLASS_SPECS, random_targets

SIDES: tuple[str, str] = ("port", "starboard")

#: Classes Brain B segments (mask = 255). Everything else — including the
#: hard negatives — rasterizes to background (mask = 0). See module docstring
#: for why the segmenter is deliberately a net/rope specialist.
FOREGROUND_CLASSES: Final[tuple[str, ...]] = ("ghost_net", "pipeline")


def rasterize_target_masks(
    gi: GroundImage,
    targets: list[SynthTarget],
    cfg: SceneConfig,
    classes: tuple[str, ...] = FOREGROUND_CLASSES,
) -> dict[str, np.ndarray]:
    """Binary footprint masks per side, aligned with the ground image grid.

    Reproduces the renderer's illumination geometry exactly: a target spans
    pings ``t.ping +/- t.length / (2 * speed * ping_interval)`` and, at each
    ping row, the across-track interval ``t.ground_range +/-
    t.half_width_at(dp)`` (rect/ellipse/irregular all reduce to that per-row
    half-width — the irregular texture only modulates *brightness*, never the
    footprint). A ground column belongs to the mask when its centre ground
    range ``(j + 0.5) * ground_res`` falls inside that interval, mirroring the
    ``in_obj`` test the renderer applies before slant-to-ground resampling.
    Rows whose footprint is thinner than one column still get their nearest
    column — a rope must never vanish from its own mask. Only the *highlight*
    footprint is masked, never the acoustic shadow: the shadow is absence of
    ensonification, not target material.

    Returns ``{side: (n_pings, n_cols) bool}`` with True only for targets
    whose class is in *classes*.
    """
    masks = {side: np.zeros(gi.side(side).shape, dtype=bool) for side in SIDES}
    for t in targets:
        if t.cls not in classes:
            continue
        mask = masks[t.side]
        n_rows, n_cols = mask.shape
        if n_rows == 0 or n_cols == 0:
            continue
        half_len = max(t.length / (2.0 * cfg.speed * cfg.ping_interval), 1.0)
        r_lo = max(int(np.ceil(t.ping - half_len)), 0)
        r_hi = min(int(np.floor(t.ping + half_len)), n_rows - 1)
        if r_hi < r_lo:
            continue
        rows = np.arange(r_lo, r_hi + 1)
        dp = (rows - t.ping) / half_len
        half_w = np.array([t.half_width_at(float(v)) for v in dp], dtype=np.float64)
        c_lo = np.ceil(gi.col_of_ground_range(t.ground_range - half_w)).astype(np.int64)
        c_hi = np.floor(gi.col_of_ground_range(t.ground_range + half_w)).astype(np.int64)
        # Sub-column-thin rows: snap to the single nearest column.
        centre = int(np.round(gi.col_of_ground_range(t.ground_range)))
        thin = c_hi < c_lo
        c_lo = np.where(thin, centre, c_lo)
        c_hi = np.where(thin, centre, c_hi)
        # Zero-width rows (ellipse tips) render nothing; off-swath rows clip away.
        inside = (half_w > 0.0) & (c_hi >= 0) & (c_lo <= n_cols - 1)
        c_lo = np.clip(c_lo, 0, n_cols - 1)
        c_hi = np.clip(c_hi, 0, n_cols - 1)
        cols = np.arange(n_cols)
        band = inside[:, None] & (cols[None, :] >= c_lo[:, None]) & (cols[None, :] <= c_hi[:, None])
        mask[r_lo : r_hi + 1] |= band
    return masks


def _boost_foreground(
    targets: list[SynthTarget], rng: np.random.Generator, net_weight: float
) -> list[SynthTarget]:
    """Convert sampled man-made non-foreground targets into nets/ropes.

    :func:`tridentnet.data.random_targets` already oversamples ``ghost_net``
    3x, but a mask training set needs foreground *pixels* to dominate or the
    BCE+Dice objective collapses to predicting empty masks. Each sampled
    man-made non-foreground target is therefore re-drawn as a foreground class
    with probability ``1 - 1 / net_weight`` — keeping its placement (which
    already satisfies the separation constraints) and redrawing its physical
    parameters from the foreground class's :class:`ClassSpec`. Natural hard
    negatives (rocks) are never converted: they are exactly the confusers the
    segmenter must learn to ignore. ``net_weight <= 1`` disables the boost.
    """
    if net_weight <= 1:
        return targets
    p_convert = 1.0 - 1.0 / float(net_weight)
    out: list[SynthTarget] = []
    for t in targets:
        if t.cls in FOREGROUND_CLASSES or t.natural or rng.random() >= p_convert:
            out.append(t)
            continue
        name = FOREGROUND_CLASSES[int(rng.integers(len(FOREGROUND_CLASSES)))]
        spec = CLASS_SPECS[name]
        out.append(
            SynthTarget(
                name,
                t.side,
                t.ping,
                t.ground_range,
                length=float(rng.uniform(*spec.length_m)),
                width=float(rng.uniform(*spec.width_m)),
                height=float(rng.uniform(*spec.height_m)),
                reflectivity=float(rng.uniform(*spec.reflectivity)),
                shape=spec.shapes[int(rng.integers(len(spec.shapes)))],
            )
        )
    return out


def _window_origin(centre: float, win: int, extent: int) -> int:
    """Origin of a *win*-pixel window centred on *centre*, clamped in-image."""
    return int(np.clip(int(round(centre - win / 2.0)), 0, max(extent - win, 0)))


def _write_chip(view: np.ndarray, path: Path) -> None:
    """Enhanced ground pixels ([0, 1], NaN beyond swath) -> 8-bit gray PNG.

    NaN maps to 0: out-of-swath fill becomes black, the same level as true
    acoustic shadow — no invented texture (same convention as detector chips
    and inference-time tile conversion)."""
    arr = np.clip(np.nan_to_num(view, nan=0.0), 0.0, 1.0)
    Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="L").save(path)


def _write_mask(view: np.ndarray, path: Path) -> None:
    """Boolean footprint -> PNG with 0 background / 255 target."""
    Image.fromarray(np.where(view, 255, 0).astype(np.uint8), mode="L").save(path)


def _safe_progress(
    progress: Callable[[str, float], None] | None, stage: str, fraction: float
) -> None:
    """Invoke the progress observer; a broken observer must never abort a build."""
    if progress is None:
        return
    try:
        progress(stage, fraction)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


def build_mask_dataset(
    out_dir: str | Path,
    n_scenes: int = 12,
    chip: int = 256,
    seed: int = 0,
    val_frac: float = 0.2,
    net_weight: float = 3,
    progress: Callable[[str, float], None] | None = None,
    *,
    n_pings_range: tuple[int, int] = (300, 500),
    n_samples: int = 512,
    altitude_range: tuple[float, float] = (6.0, 12.0),
    slant_range_range: tuple[float, float] = (40.0, 60.0),
    n_min_targets: int = 3,
    n_max_targets: int = 6,
    backgrounds_per_scene: int = 2,
    targets_fn: Callable[[SceneConfig, np.random.Generator], list[SynthTarget]] | None = None,
    preprocess_config: dict[str, Any] | None = None,
) -> Path:
    """Build an image/mask chip dataset from *n_scenes* simulated surveys.

    Per scene: draw a random :class:`SceneConfig` (altitude, slant range and
    survey length vary so the segmenter sees the spread of ground resolutions
    the simulator produces — scenes are shorter than the detector dataset's
    because a mask specialist needs many net examples, not long surveys), seed
    a debris field with :func:`tridentnet.data.random_targets`, boost the
    foreground share via *net_weight* (see :func:`_boost_foreground`), render
    with :func:`make_scene`, run the full M2 :func:`preprocess` chain, and
    rasterize exact footprint masks with :func:`rasterize_target_masks`.

    One ``chip x chip`` window is then cut per target, *centred* on the target
    (clamped inside the image; an image smaller than *chip* yields a smaller
    full-extent window — never padding, which would inject phantom edges).
    The image PNG comes from the enhanced ground image, the mask PNG (0
    background, 255 target) from the scene-level foreground mask, so a chip
    centred on a drum still carries any net that strays into its window, and
    non-foreground chips are explicit empty-mask negatives.

    *backgrounds_per_scene* additional chips per scene are cut at random
    positions with the scene-level mask crop (usually empty). The FIRST one is
    forced to start at column 0: the near-nadir strip — bright first-return
    residue and slant-stretch smearing — appears in no target-centred chip
    (targets are seeded metres from nadir), and a segmenter that never saw it
    hallucinates net all along the nadir line at inference. Same reasoning as
    Brain C's ``nadir_guard_cols``, solved with data instead of a mask-out.

    The train/val split assigns whole SCENES (chips from one scene share the
    same speckle realization and seabed texture; splitting chips would leak).
    ``val_frac`` of scenes go to val via a seed-deterministic permutation, at
    least one when positive and never all scenes. The build is fully
    deterministic for a given *seed* (per-scene generators are
    ``default_rng([seed, scene_idx])``).

    Chip filenames encode provenance:
    ``s{scene}_{side}_t{k}_{cls}_r{row}_c{col}`` with ``r``/``c`` the global
    (ping, ground-column) of the chip's top-left pixel, so any mask pixel maps
    back to survey coordinates. *targets_fn* overrides target sampling for
    tests (the *net_weight* boost is then skipped — the caller has full
    control); *preprocess_config* is deep-merged over the M2 defaults.

    Returns the dataset root (containing ``images/{train,val}`` and
    ``masks/{train,val}``).
    """
    out_dir = Path(out_dir).resolve()
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "masks" / split).mkdir(parents=True, exist_ok=True)

    if val_frac <= 0.0:
        n_val = 0
    else:
        n_val = max(1, round(val_frac * n_scenes))
        n_val = min(n_val, n_scenes - 1)  # the train split must never be empty
    val_scenes = {int(i) for i in np.random.default_rng(seed).permutation(n_scenes)[:n_val]}

    for scene_idx in range(n_scenes):
        rng = np.random.default_rng([seed, scene_idx])
        cfg = SceneConfig(
            n_pings=int(rng.integers(n_pings_range[0], n_pings_range[1] + 1)),
            n_samples=int(n_samples),
            slant_range=float(rng.uniform(*slant_range_range)),
            altitude=float(rng.uniform(*altitude_range)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        if targets_fn is not None:
            targets = targets_fn(cfg, rng)
        else:
            targets = random_targets(cfg, rng, n_min=n_min_targets, n_max=n_max_targets)
            targets = _boost_foreground(targets, rng, net_weight)

        pa, targets = make_scene(cfg, targets)
        pre = preprocess(pa, config=preprocess_config)
        gi = pre.ground
        masks = rasterize_target_masks(gi, targets, cfg)

        split = "val" if scene_idx in val_scenes else "train"
        img_dir = out_dir / "images" / split
        msk_dir = out_dir / "masks" / split

        for k, t in enumerate(targets):
            img = gi.side(t.side)
            n_rows, n_cols = img.shape
            if n_rows == 0 or n_cols == 0:
                continue
            win_h, win_w = min(chip, n_rows), min(chip, n_cols)
            r_off = _window_origin(t.ping + 0.5, win_h, n_rows)
            c_off = _window_origin(
                float(gi.col_of_ground_range(t.ground_range)) + 0.5, win_w, n_cols
            )
            stem = f"s{scene_idx:03d}_{t.side}_t{k:02d}_{t.cls}_r{r_off}_c{c_off}"
            _write_chip(
                img[r_off : r_off + win_h, c_off : c_off + win_w], img_dir / f"{stem}.png"
            )
            _write_mask(
                masks[t.side][r_off : r_off + win_h, c_off : c_off + win_w],
                msk_dir / f"{stem}.png",
            )

        for b in range(backgrounds_per_scene):
            side = SIDES[int(rng.integers(len(SIDES)))]
            img = gi.side(side)
            n_rows, n_cols = img.shape
            if n_rows == 0 or n_cols == 0:
                continue
            win_h, win_w = min(chip, n_rows), min(chip, n_cols)
            r_off = int(rng.integers(0, n_rows - win_h + 1))
            # First background chip pinned to the nadir strip (see docstring).
            c_off = 0 if b == 0 else int(rng.integers(0, n_cols - win_w + 1))
            stem = f"s{scene_idx:03d}_{side}_bg{b:02d}_r{r_off}_c{c_off}"
            _write_chip(
                img[r_off : r_off + win_h, c_off : c_off + win_w], img_dir / f"{stem}.png"
            )
            _write_mask(
                masks[side][r_off : r_off + win_h, c_off : c_off + win_w],
                msk_dir / f"{stem}.png",
            )

        _safe_progress(progress, f"scene {scene_idx + 1}/{n_scenes}", (scene_idx + 1) / n_scenes)

    _safe_progress(progress, "done", 1.0)
    return out_dir
