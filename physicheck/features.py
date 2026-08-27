"""Physics feature extraction for the Stage-2 ML verifier.

The Stage-1 gate (:mod:`physicheck.calibrate`) applies hand-set multipliers to
three binary cues; Stage 2 hands a *learned* classifier the full continuous
cue vector so it can trade cues off against each other (a weak highlight with
a razor-straight shadow edge is a pipe, not noise). All features are computed
from the **unenhanced** ground image (``PreprocessResult.ground_raw``) — the
same radiometry :func:`physicheck.shadow.analyze_shadow` measures, because
CLAHE/despeckle would distort every ratio and entropy below.

Feature rationale, cue by cue:

* ``shadow_linearity`` — man-made objects (pipes, hulls, containers) have
  machined edges that cast a *straight* shadow boundary along track; rock
  piles cast ragged ones. Per-ping dark-run end columns down-range of the box
  are fit with a line; the feature is the fit's R² with a per-row variance
  floor (a perfectly straight across-track edge has near-zero end-position
  variance, where raw R² is degenerate — the floor makes "constant within
  speckle jitter" score high instead of undefined).
* ``contour_regularity`` — thresholding the box interior at 1.5x local
  background and measuring the largest blob's fill of its own bounding box:
  man-made highlights trend compact/rectilinear, natural clutter is patchy.
* ``texture_entropy_delta`` — Shannon entropy inside the box minus a
  surrounding ring: debris breaks the seabed texture statistics, while sand
  ripples continue through a false box unchanged (delta near zero).
* ``ping_persistence`` — real seabed objects persist across scan lines;
  single-ping spikes are electrical/acoustic noise.

The dict returned by :func:`extract_features` is insertion-ordered to match
:data:`FEATURE_NAMES` exactly — the verifier serializes on that order.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from physicheck.shadow import ShadowAnalysis
from sonar_core.preprocess.slant_range import GroundImage

#: Stable feature order; the trained verifier stores (and re-checks) this.
FEATURE_NAMES: tuple[str, ...] = (
    "highlight_ratio",
    "shadow_ratio",
    "shadow_len_m",
    "height_m",
    "has_height",
    "shadow_linearity",
    "contour_regularity",
    "texture_entropy_delta",
    "ping_persistence",
    "aspect_ratio",
    "area_px",
    "range_frac",
    "score_raw",
)

#: Sentinel replacing NaN in analysis-derived ratio/height features, paired
#: with the explicit ``has_height`` flag so the model can tell "no shadow"
#: from "short shadow" instead of reading the sentinel as a magnitude.
_NAN_SENTINEL = -1.0


def _finite(value: float, sentinel: float = 0.0) -> float:
    """NaN/inf-safe scalar: any non-finite value becomes *sentinel*."""
    v = float(value)
    return v if np.isfinite(v) else sentinel


def _dark_run_end(
    row: np.ndarray,
    dark_level: float,
    lead_in_cols: int,
    tolerance_cols: int,
    min_run_cols: int,
) -> int | None:
    """End index (into *row*) of the first dark run, or None when unusable.

    Mirrors the column-wise run logic of :func:`~physicheck.shadow.analyze_shadow`
    per ping row: the run may start up to *lead_in_cols* after the box edge
    (box-quantization slop), tolerates up to *tolerance_cols* bright speckle
    interruptions, and must reach *min_run_cols* to count as shadow at all.
    NaN samples (beyond the swath) terminate the run — nothing was ensonified
    there, so no shadow evidence exists.
    """
    if row.size == 0:
        return None
    finite = np.isfinite(row)
    dark = finite & (row < dark_level)
    start = -1
    for j in range(min(max(lead_in_cols, 1), row.size)):
        if dark[j]:
            start = j
            break
    if start < 0:
        return None
    end = start
    misses = 0
    for j in range(start + 1, row.size):
        if not finite[j]:
            break
        if dark[j]:
            end = j
            misses = 0
        else:
            misses += 1
            if misses > tolerance_cols:
                break
    return end if end - start + 1 >= min_run_cols else None


def _shadow_linearity(
    img: np.ndarray,
    ping0: int,
    ping1: int,
    col1: int,
    analysis: ShadowAnalysis,
    gi: GroundImage,
    shadow_thresh: float,
    lead_in_cols: int,
    tolerance_cols: int,
    min_run_cols: int,
    min_rows: int,
    floor_std_cols: float,
    mad_gate: float,
    mad_gate_min_cols: float,
) -> float:
    """Floored R² of a line fit through per-ping shadow end columns in [0, 1].

    The search window is sized exactly like ``analyze_shadow``'s (max plausible
    height at the measured altitude). An object tapers toward its along-track
    ends, so box-edge rows carry stub runs (a column or two of speckle read as
    "shadow") that would swamp the fit with box-quantization artefacts; a
    median/MAD gate rejects ends farther than ``max(mad_gate * MAD,
    mad_gate_min_cols)`` from the median end before fitting, keeping the rows
    that actually see the shadow edge. Fewer than *min_rows* surviving rows —
    no shadow, or one shredded by texture — scores 0: too little edge to call
    straight. The R² denominator is floored at ``n * floor_std_cols²`` so a
    straight, constant-column edge (near-zero end variance, the man-made
    ideal) maps to ~1 instead of the raw-R² degeneracy 0/0, while a curved or
    ragged edge (rock pile) keeps residuals well above the floor and scores
    low.
    """
    background = analysis.background
    altitude = analysis.altitude_m
    if not np.isfinite(background) or background <= 0:
        return 0.0
    n_cols = img.shape[1]
    x_far = analysis.x_far_m
    h_cap = min(20.0, 0.9 * altitude) if altitude > 0 else 20.0
    if altitude > 0 and h_cap < altitude:
        x_end_max = x_far * altitude / (altitude - h_cap)
    else:
        x_end_max = x_far * 4.0
    search_hi = int(np.clip(np.ceil(gi.col_of_ground_range(x_end_max)), col1 + 1, n_cols))
    if search_hi <= col1 + 1:
        return 0.0

    dark_level = shadow_thresh * background
    pings: list[int] = []
    ends: list[int] = []
    for p in range(ping0, ping1 + 1):
        end = _dark_run_end(
            img[p, col1 + 1 : search_hi], dark_level, lead_in_cols, tolerance_cols, min_run_cols
        )
        if end is not None:
            pings.append(p)
            ends.append(end)
    if len(ends) < min_rows:
        return 0.0

    x = np.asarray(pings, dtype=np.float64)
    y = np.asarray(ends, dtype=np.float64)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    keep = np.abs(y - med) <= max(mad_gate * mad, mad_gate_min_cols)
    x, y = x[keep], y[keep]
    if len(y) < min_rows:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    denom = max(ss_tot, len(y) * floor_std_cols**2)
    return float(np.clip(1.0 - ss_res / denom, 0.0, 1.0))


def _contour_regularity(box: np.ndarray, background: float, highlight_thresh: float) -> float:
    """Largest bright blob's fill fraction of its own bounding box, in [0, 1].

    Man-made highlights are compact and convex-ish (fill near 1); rock piles
    and clutter fragment into patchy blobs (fill well below 1). Bright means
    above ``highlight_thresh x local background``.
    """
    if not np.isfinite(background) or background <= 0 or box.size == 0:
        return 0.0
    bright = np.isfinite(box) & (box > highlight_thresh * background)
    if not bright.any():
        return 0.0
    labels, n = ndimage.label(bright)
    if n == 0:
        return 0.0
    areas = ndimage.sum_labels(bright, labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(areas)) + 1
    ys, xs = np.nonzero(labels == biggest)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(areas[biggest - 1] / bbox_area) if bbox_area > 0 else 0.0


def _entropy_bits(values: np.ndarray, lo: float, hi: float, n_bins: int) -> float:
    """Shannon entropy (bits) of *values* over *n_bins* bins spanning [lo, hi]."""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or hi <= lo:
        return 0.0
    hist, _ = np.histogram(finite, bins=n_bins, range=(lo, hi))
    p = hist / finite.size
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _texture_entropy_delta(
    img: np.ndarray, ping0: int, ping1: int, col0: int, col1: int, ring_px: int, n_bins: int
) -> float:
    """Box entropy minus surrounding-ring entropy (shared bin edges, bits).

    Debris disrupts the local texture statistics (positive delta); a ripple
    field continues through a false box unchanged (delta near zero). The ring
    is the *ring_px*-expanded box minus the box itself, clipped to the image.
    """
    n_rows, n_cols = img.shape
    box = img[ping0 : ping1 + 1, col0 : col1 + 1]
    r0, r1 = max(ping0 - ring_px, 0), min(ping1 + ring_px, n_rows - 1)
    c0, c1 = max(col0 - ring_px, 0), min(col1 + ring_px, n_cols - 1)
    outer = img[r0 : r1 + 1, c0 : c1 + 1].copy()
    outer[ping0 - r0 : ping1 - r0 + 1, col0 - c0 : col1 - c0 + 1] = np.nan  # cut the box out
    both = np.concatenate([box[np.isfinite(box)].ravel(), outer[np.isfinite(outer)].ravel()])
    if both.size == 0:
        return 0.0
    lo, hi = float(both.min()), float(both.max())
    return _entropy_bits(box, lo, hi, n_bins) - _entropy_bits(outer, lo, hi, n_bins)


def extract_features(
    gi_raw: GroundImage,
    det,
    analysis: ShadowAnalysis,
    *,
    shadow_thresh: float = 0.5,
    lead_in_cols: int = 4,
    tolerance_cols: int = 2,
    min_run_cols: int = 2,
    min_linearity_rows: int = 4,
    linearity_floor_std_cols: float = 3.0,
    linearity_mad_gate: float = 6.0,
    linearity_mad_gate_min_cols: float = 6.0,
    highlight_thresh: float = 1.5,
    ring_px: int = 8,
    entropy_bins: int = 16,
) -> dict[str, float]:
    """Feature vector for one detection, keyed and ordered by FEATURE_NAMES.

    *gi_raw* must be the unenhanced ground image (``PreprocessResult.ground_raw``)
    and *analysis* the :func:`~physicheck.shadow.analyze_shadow` result already
    computed for this box — its ratios/height/background are reused verbatim,
    never recomputed, so Stage 1 and Stage 2 argue from identical evidence.
    *det* is duck-typed (``side, ping0, ping1, col0, col1, score``). Every
    value is finite: NaN analysis ratios/height become the -1 sentinel with
    ``has_height`` flagging a usable height estimate.
    """
    img = gi_raw.side(det.side)
    n_pings, n_cols = img.shape
    ping0 = int(np.clip(det.ping0, 0, n_pings - 1))
    ping1 = int(np.clip(det.ping1, ping0, n_pings - 1))
    col0 = int(np.clip(det.col0, 0, n_cols - 1))
    col1 = int(np.clip(det.col1, col0, n_cols - 1))
    box = img[ping0 : ping1 + 1, col0 : col1 + 1]

    box_pings = ping1 - ping0 + 1
    box_cols = col1 - col0 + 1
    features: dict[str, float] = {
        "highlight_ratio": _finite(analysis.highlight_ratio, _NAN_SENTINEL),
        "shadow_ratio": _finite(analysis.shadow_ratio, _NAN_SENTINEL),
        "shadow_len_m": _finite(analysis.shadow_len_m, _NAN_SENTINEL),
        "height_m": _finite(analysis.height_m, _NAN_SENTINEL),
        "has_height": 1.0 if np.isfinite(analysis.height_m) else 0.0,
        "shadow_linearity": _shadow_linearity(
            img, ping0, ping1, col1, analysis, gi_raw,
            shadow_thresh, lead_in_cols, tolerance_cols, min_run_cols,
            min_linearity_rows, linearity_floor_std_cols,
            linearity_mad_gate, linearity_mad_gate_min_cols,
        ),
        "contour_regularity": _contour_regularity(box, analysis.background, highlight_thresh),
        "texture_entropy_delta": _texture_entropy_delta(
            img, ping0, ping1, col0, col1, ring_px, entropy_bins
        ),
        "ping_persistence": float(box_pings),
        "aspect_ratio": float(box_pings) / float(box_cols),
        "area_px": float(box_pings * box_cols),
        "range_frac": ((col0 + col1) / 2.0) / float(n_cols),
        "score_raw": _finite(det.score),
    }
    # Belt and braces: no non-finite value may ever reach the classifier.
    return {name: _finite(features[name]) for name in FEATURE_NAMES}
