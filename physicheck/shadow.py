"""Highlight–shadow verification and height-from-shadow estimation.

A real object proud of the seabed shows an acoustic *highlight* (strong
backscatter) with a dark *shadow* stretching down-range (away from nadir),
because the object blocks the fan beam. In ground-range imagery the shadow
of an object whose far edge sits at ground range ``x_far`` ends at
``x_end = x_far * A / (A - H)`` for towfish altitude ``A`` and object height
``H``; inverting gives the height estimate

    H = A * (x_end - x_far) / x_end

which is the ground-domain form of the classic slant-range rule H = L*A/R.
Analysis runs on the *unenhanced* ground image (``PreprocessResult.ground_raw``)
— CLAHE/despeckle would distort the ratio statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonar_core.preprocess.slant_range import GroundImage


@dataclass(frozen=True)
class ShadowAnalysis:
    """Acoustic cues for one detection box."""

    has_highlight: bool
    has_shadow: bool
    highlight_ratio: float  # box median / background (>= 1 means brighter)
    shadow_ratio: float  # shadow median / background (<= 1 means darker)
    shadow_len_m: float  # ground-range length of the detected shadow
    height_m: float  # estimated object height; NaN when no usable shadow
    background: float  # local seabed reference level
    altitude_m: float  # altitude used for the height estimate
    x_far_m: float  # ground range of the object's far edge
    x_end_m: float  # ground range where the shadow ends (NaN if none)

    def cues(self) -> dict[str, bool | float]:
        """JSON-ready cue list for the Evidence Card."""
        return {
            "highlight": self.has_highlight,
            "shadow": self.has_shadow,
            "highlight_ratio": round(self.highlight_ratio, 3),
            "shadow_ratio": round(self.shadow_ratio, 3),
            "shadow_len_m": round(self.shadow_len_m, 2),
            "height_m": None if np.isnan(self.height_m) else round(self.height_m, 2),
        }


def _robust_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def analyze_shadow(
    gi: GroundImage,
    side: str,
    ping0: int,
    ping1: int,
    col0: int,
    col1: int,
    shadow_thresh: float = 0.5,
    min_dark_frac: float = 0.6,
    highlight_min_ratio: float = 1.4,
    max_height_m: float = 20.0,
    bg_width_cols: int = 40,
    bg_gap_cols: int = 4,
    tolerance_cols: int = 2,
) -> ShadowAnalysis:
    """Verify highlight+shadow for the box [ping0..ping1] x [col0..col1].

    The shadow is searched down-range (increasing column) of the box's far
    edge: a run of columns whose fraction of rows darker than
    ``shadow_thresh * background`` stays above ``min_dark_frac``, allowing
    up to ``tolerance_cols`` bright interruptions (speckle).
    """
    img = gi.side(side)
    n_pings, n_cols = img.shape
    ping0 = int(np.clip(ping0, 0, n_pings - 1))
    ping1 = int(np.clip(ping1, ping0, n_pings - 1))
    col0 = int(np.clip(col0, 0, n_cols - 1))
    col1 = int(np.clip(col1, col0, n_cols - 1))

    rows = img[ping0 : ping1 + 1]
    altitude = float(np.nanmean(gi.altitude_m[ping0 : ping1 + 1]))

    # Background: near-side strip before the box (unaffected by the shadow,
    # which falls down-range). Fall back to a far strip if the box hugs nadir.
    bg_hi = max(col0 - bg_gap_cols, 0)
    bg_lo = max(bg_hi - bg_width_cols, 0)
    background = _robust_median(rows[:, bg_lo:bg_hi])
    if not np.isfinite(background) or background <= 0:
        far_lo = min(col1 + bg_gap_cols, n_cols)
        background = _robust_median(rows[:, far_lo : far_lo + bg_width_cols])
    if not np.isfinite(background) or background <= 0:
        return ShadowAnalysis(
            has_highlight=False,
            has_shadow=False,
            highlight_ratio=float("nan"),
            shadow_ratio=float("nan"),
            shadow_len_m=0.0,
            height_m=float("nan"),
            background=float("nan"),
            altitude_m=altitude,
            x_far_m=float(gi.ground_range_of_col(col1)),
            x_end_m=float("nan"),
        )

    highlight_ratio = _robust_median(rows[:, col0 : col1 + 1]) / background
    has_highlight = bool(highlight_ratio >= highlight_min_ratio)

    # Shadow search window sized by the maximum plausible height.
    x_far = float(gi.ground_range_of_col(col1))
    h_cap = min(max_height_m, 0.9 * altitude) if altitude > 0 else max_height_m
    if altitude > 0 and h_cap < altitude:
        x_end_max = x_far * altitude / (altitude - h_cap)
    else:
        x_end_max = x_far * 4.0
    search_hi = int(np.clip(np.ceil(gi.col_of_ground_range(x_end_max)), col1 + 1, n_cols))

    window = rows[:, col1 + 1 : search_hi]
    shadow_cols = 0
    if window.shape[1] > 0:
        dark = window < shadow_thresh * background
        finite = np.isfinite(window)
        with np.errstate(invalid="ignore"):
            dark_frac = np.where(
                finite.sum(axis=0) > 0, dark.sum(axis=0) / np.maximum(finite.sum(axis=0), 1), 0.0
            )
        is_shadow = dark_frac >= min_dark_frac
        misses = 0
        for flag in is_shadow:
            if flag:
                shadow_cols += 1 + misses if shadow_cols else 1
                misses = 0
            else:
                if not shadow_cols:
                    break  # shadow must start immediately behind the object
                misses += 1
                if misses > tolerance_cols:
                    break

    shadow_len_m = shadow_cols * gi.ground_res
    has_shadow = shadow_cols >= 2
    if has_shadow:
        shadow_slice = window[:, :shadow_cols]
        shadow_ratio = _robust_median(shadow_slice) / background
        x_end = float(gi.ground_range_of_col(col1 + shadow_cols))
        height = altitude * (x_end - x_far) / x_end if x_end > 0 else float("nan")
    else:
        shadow_ratio = float("nan")
        x_end = float("nan")
        height = float("nan")

    return ShadowAnalysis(
        has_highlight=has_highlight,
        has_shadow=has_shadow,
        highlight_ratio=float(highlight_ratio),
        shadow_ratio=float(shadow_ratio),
        shadow_len_m=float(shadow_len_m),
        height_m=float(height),
        background=background,
        altitude_m=altitude,
        x_far_m=x_far,
        x_end_m=x_end,
    )
