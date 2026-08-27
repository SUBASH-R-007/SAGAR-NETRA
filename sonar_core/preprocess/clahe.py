"""Contrast-limited adaptive histogram equalization (CLAHE) for display.

Side-scan imagery is dominated by two competing dynamic ranges: bright
near-nadir and specular target returns sit orders of magnitude above the weak
grazing-angle backscatter at far range, and the diagnostic detail — shadow
edges, highlight texture, ripple micro-relief — lives in small *local*
intensity differences. A single global transfer curve (histogram equalization
or plain min/max stretch) lets the dominant background mode dictate the
mapping, crushing exactly the local contrast a detector needs. CLAHE instead
equalizes each tile of the image independently while a clip limit caps the
slope of every tile's transfer function, so shadow/highlight signatures are
boosted without amplifying Rayleigh speckle into salt-and-pepper noise.

The pipeline here is: robust percentile stretch (raw amplitudes are
heavy-tailed — specular flashes would otherwise consume most of the code
space) to uint16, OpenCV CLAHE on the full 16-bit histogram, then rescale to
float32 ``[0, 1]`` for display or detector input. Out-of-swath pixels (NaN
after slant-range correction) are filled with the low percentile for the
operation so they read as "darkest valid data" and cannot inject a spurious
histogram mode, then restored to NaN on output.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Full-scale code value of the uint16 working image handed to OpenCV.
UINT16_MAX = 65535

#: Display value returned for images with no dynamic range (constant input):
#: neutral mid-gray, so a featureless swath renders as such rather than black.
FLAT_FILL = 0.5


def clahe(
    img: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: tuple[int, int] = (8, 8),
    p_low: float = 1.0,
    p_high: float = 99.7,
) -> np.ndarray:
    """Percentile-stretch *img* and apply 16-bit CLAHE; return float32 [0, 1].

    Parameters
    ----------
    img:
        2-D float32 intensity image (waterfall or ground-range). NaN marks
        out-of-swath pixels; those come back NaN and do not distort the
        histogram (they are filled with the low percentile for the operation).
    clip_limit:
        OpenCV CLAHE contrast limit — the cap on any tile's histogram bins
        (relative to a uniform histogram) before excess is redistributed.
        Higher values allow stronger local amplification, at the cost of
        boosting speckle.
    tile_grid:
        ``(tiles_x, tiles_y)`` grid passed to OpenCV: tiles across the
        range (column) axis and along the ping (row) axis. Each tile gets its
        own equalization curve; more tiles adapt to finer gain structure.
        Clamped so a tile is never smaller than one pixel on tiny images.
    p_low, p_high:
        Percentiles of the finite pixels mapped to 0 and 65535 before CLAHE.
        The asymmetric default keeps the shadow floor while clipping only the
        rare specular flashes at the top of the amplitude distribution.

    Returns
    -------
    np.ndarray
        float32 image, same shape as *img*, finite values in ``[0, 1]``, NaN
        exactly where *img* was non-finite. Degenerate inputs (all-NaN, or a
        constant image with no stretchable range) return without exceptions:
        all-NaN in, all-NaN out; constant in, :data:`FLAT_FILL` out.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"img must be 2-D, got shape {img.shape}")

    finite = np.isfinite(img)
    if not finite.any():
        return np.full(img.shape, np.nan, dtype=np.float32)

    lo, hi = np.percentile(img[finite], (p_low, p_high))
    if not hi > lo:
        # No dynamic range to equalize (constant swath, dead channel).
        out = np.full(img.shape, FLAT_FILL, dtype=np.float32)
        out[~finite] = np.nan
        return out

    filled = np.where(finite, img, np.float32(lo))
    stretched = np.clip((filled - lo) / (hi - lo), 0.0, 1.0)
    u16 = np.rint(stretched * UINT16_MAX).astype(np.uint16)

    n_rows, n_cols = img.shape
    grid = (
        int(np.clip(tile_grid[0], 1, n_cols)),  # tiles_x: across-range
        int(np.clip(tile_grid[1], 1, n_rows)),  # tiles_y: along-track
    )
    equalized = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid).apply(u16)

    out = equalized.astype(np.float32) / np.float32(UINT16_MAX)
    out[~finite] = np.nan
    return out
