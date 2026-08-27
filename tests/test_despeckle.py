"""Despeckling: variance reduction on flat seabed, shadow-edge preservation,
impulse removal, NaN safety, and dispatcher errors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from sonar_core.preprocess.despeckle import (
    RAYLEIGH_CU,
    adaptive_median,
    despeckle,
    lee_filter,
)


def _speckle(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Unit-mean Rayleigh multiplicative speckle, matching the scene renderer."""
    return rng.rayleigh(scale=np.sqrt(2 / np.pi), size=shape)


def _descending_crossing(profile: np.ndarray, level: float) -> float:
    """Interpolated index where a descending profile first drops below *level*."""
    below = np.flatnonzero(profile < level)
    assert below.size, f"profile never drops below {level}"
    i = int(below[0])
    if i == 0:
        return 0.0
    f0, f1 = float(profile[i - 1]), float(profile[i])
    return (i - 1) + (f0 - level) / (f0 - f1)


def _rise_width(profile: np.ndarray, lo_frac: float = 0.1, hi_frac: float = 0.9) -> float:
    """10-90 percent transition width (pixels) of a descending step profile."""
    return _descending_crossing(profile, lo_frac) - _descending_crossing(profile, hi_frac)


def test_lee_reduces_variance_on_flat_field() -> None:
    rng = np.random.default_rng(11)
    img = (1000.0 * _speckle(rng, (200, 256))).astype(np.float32)
    out = lee_filter(img, size=5)
    cv_in = float(img.std() / img.mean())
    cv_out = float(out.std() / out.mean())
    assert cv_out < 0.5 * cv_in, f"cv only went {cv_in:.3f} -> {cv_out:.3f}"
    # Radiometry must be preserved: Lee is (locally) unbiased in the mean.
    assert abs(float(out.mean()) - float(img.mean())) / float(img.mean()) < 0.02


def test_lee_preserves_shadow_edge() -> None:
    """A bright->dark step (seabed into shadow) must not be smeared: the 10-90
    width of the along-track-averaged profile grows by at most 2 pixels."""
    n_pings, n_samples, edge = 600, 256, 128
    bright, dark = 1000.0, 100.0
    clean = np.full((n_pings, n_samples), bright, dtype=np.float64)
    clean[:, edge:] = dark
    rng = np.random.default_rng(17)
    img = (clean * _speckle(rng, clean.shape)).astype(np.float32)

    out = lee_filter(img, size=5)
    profile = out.mean(axis=0)
    # Normalize by measured plateau levels, away from the transition.
    high = float(profile[edge - 108 : edge - 28].mean())
    low = float(profile[edge + 28 : edge + 108].mean())
    window = slice(edge - 20, edge + 20)
    norm = (profile[window] - low) / (high - low)
    clean_norm = (clean[0, window] - dark) / (bright - dark)

    width_clean = _rise_width(clean_norm)
    width_filtered = _rise_width(norm)
    assert width_filtered <= width_clean + 2.0, (
        f"shadow edge smeared: {width_clean:.2f} -> {width_filtered:.2f} px"
    )


def test_adaptive_median_removes_impulses_only() -> None:
    rng = np.random.default_rng(23)
    img = (1000.0 * _speckle(rng, (200, 256))).astype(np.float32)
    impulse_value = 25_000.0
    flat_idx = rng.choice(img.size, size=60, replace=False)
    rows, cols = np.unravel_index(flat_idx, img.shape)
    img[rows, cols] = impulse_value

    # Threshold 4.0: raw single-look Rayleigh speckle has a heavy upper tail,
    # so a plain 3-sigma rule false-triggers a few percent of the time.
    out = adaptive_median(img, size=3, threshold=4.0)
    # Every seeded impulse is knocked back to a plausible local level.
    assert np.all(out[rows, cols] < impulse_value / 3)
    # Everything else passes through bit-identical, bar rare speckle-tail hits.
    other = np.ones(img.shape, dtype=bool)
    other[rows, cols] = False
    unchanged = float((out[other] == img[other]).mean())
    assert unchanged > 0.98, f"only {unchanged:.4f} of non-impulse pixels unchanged"


def test_lee_nan_in_nan_out_and_local_influence_only() -> None:
    rng = np.random.default_rng(31)
    base = (1000.0 * _speckle(rng, (120, 160))).astype(np.float32)
    holed = base.copy()
    holed[[30, 31, 77], [40, 41, 90]] = np.nan
    nan_mask = ~np.isfinite(holed)

    size = 5
    # Fixed cu so the two runs differ only through local statistics.
    out_ref = lee_filter(base, size=size, cu=RAYLEIGH_CU)
    out = lee_filter(holed, size=size, cu=RAYLEIGH_CU)

    assert np.isnan(out[nan_mask]).all()
    assert np.isfinite(out[~nan_mask]).all()
    # Beyond the filter window footprint, results are bit-identical.
    influenced = binary_dilation(nan_mask, structure=np.ones((size, size), dtype=bool))
    np.testing.assert_array_equal(out[~influenced], out_ref[~influenced])

    # adaptive_median likewise preserves NaN as NaN.
    med = adaptive_median(holed)
    assert np.isnan(med[nan_mask]).all()
    assert np.isfinite(med[~nan_mask]).all()


def test_despeckle_dispatch() -> None:
    rng = np.random.default_rng(41)
    img = (1000.0 * _speckle(rng, (40, 50))).astype(np.float32)
    np.testing.assert_array_equal(despeckle(img), lee_filter(img))
    np.testing.assert_array_equal(
        despeckle(img, method="median", size=3), adaptive_median(img, size=3)
    )
    with pytest.raises(ValueError, match="frost"):
        despeckle(img, method="frost")
