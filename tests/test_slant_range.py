"""Slant-range correction: geometry invertibility and correctness."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.parsers.base import NAV_DTYPE, PingArray
from sonar_core.preprocess.bottom_track import track_bottom
from sonar_core.preprocess.slant_range import slant_to_ground


def _flat_pa(n_pings: int = 8, n_samples: int = 200, alt: float = 6.0) -> PingArray:
    nav = np.zeros(n_pings, dtype=NAV_DTYPE)
    nav["slant_range"] = 50.0
    nav["altitude"] = alt
    nav["sound_velocity"] = 1500.0
    ramp = np.tile(np.arange(n_samples, dtype=np.float32), (n_pings, 1))
    return PingArray(port=ramp.copy(), starboard=ramp.copy(), nav=nav)


def test_known_geometry() -> None:
    """A sample at slant r must land at ground sqrt(r^2 - A^2)."""
    pa = _flat_pa(alt=6.0)
    gi = slant_to_ground(pa)
    res = pa.slant_resolution("starboard")  # 0.25 m
    # Pick corrected column j: value interpolated from source ramp equals the
    # fractional source sample index, which must match the geometry.
    j = gi.n_cols("starboard") // 2
    g = gi.ground_range_of_col(j)
    expected_sample = np.hypot(g, 6.0) / res - 0.5
    got = gi.starboard[0, j]
    assert got == pytest.approx(expected_sample, abs=1e-3)


def test_roundtrip_within_one_pixel(small_scene) -> None:
    """M2 acceptance: corrected pixel -> slant sample -> corrected pixel < 1 px."""
    pa, _ = small_scene
    bt = track_bottom(pa)
    gi = slant_to_ground(pa, bt)
    rng = np.random.default_rng(0)
    for side in ("port", "starboard"):
        n_cols = gi.n_cols(side)
        pings = rng.integers(0, gi.n_pings, size=200)
        cols = rng.integers(0, n_cols, size=200)
        for ping, col in zip(pings, cols, strict=True):
            s = gi.to_slant_sample(side, int(ping), int(col))
            back = gi.to_ground_col(side, int(ping), s)
            assert abs(back - col) < 1.0


def test_water_column_not_in_output() -> None:
    """Ground range starts at 0 at nadir: no water-column gap in output."""
    pa = _flat_pa(alt=10.0)
    gi = slant_to_ground(pa)
    # Column 0 (ground ~ 0.125 m) maps to slant ~ altitude -> sample ~ alt/res.
    s = gi.to_slant_sample("starboard", 0, 0)
    assert s == pytest.approx(10.0 / 0.25 - 0.5, abs=0.1)


def test_nadir_blend_columns_are_nan() -> None:
    """Columns whose source sample precedes the first return would blend
    water-column fill with real data; they must be NaN, not a silent blend."""
    pa = _flat_pa(alt=20.0)
    gi = slant_to_ground(pa)
    # Ground col 0 maps to source sample ~79.5 < first return (80) -> masked.
    assert np.isnan(gi.starboard[0, 0])
    # First column whose source sample clears the first return is finite.
    first_valid = np.flatnonzero(np.isfinite(gi.starboard[0]))[0]
    assert gi.to_slant_sample("starboard", 0, int(first_valid)) >= np.round(20.0 / 0.25)


def test_far_range_is_nan_beyond_swath() -> None:
    """Pixels beyond sqrt(R^2-A^2) have no source data and must be NaN."""
    pa = _flat_pa(alt=20.0)
    gi = slant_to_ground(pa)
    max_ground = np.sqrt(50.0**2 - 20.0**2)
    far_col = int(gi.col_of_ground_range(max_ground)) + 2
    if far_col < gi.n_cols("starboard"):
        assert np.isnan(gi.starboard[0, far_col])
    first_valid = np.flatnonzero(np.isfinite(gi.starboard[0]))[0]
    inner = gi.starboard[0, first_valid : int(gi.col_of_ground_range(max_ground)) - 1]
    assert np.isfinite(inner).all()


def test_rejects_mosaic_input() -> None:
    pa = _flat_pa()
    pa.meta["ground_range"] = True
    with pytest.raises(ValueError, match="already ground-range"):
        slant_to_ground(pa)
