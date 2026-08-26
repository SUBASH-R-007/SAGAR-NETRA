"""PingArray model: validation, bookkeeping, slicing."""

from __future__ import annotations

import numpy as np
import pytest

from sonar_core.parsers.base import NAV_DTYPE, PingArray


def _nav(n: int) -> np.ndarray:
    nav = np.zeros(n, dtype=NAV_DTYPE)
    nav["slant_range"] = 50.0
    nav["sound_velocity"] = 1500.0
    return nav


def test_construction_and_props() -> None:
    pa = PingArray(
        port=np.zeros((10, 32)), starboard=np.zeros((10, 48)), nav=_nav(10)
    )
    assert pa.n_pings == 10
    assert pa.n_samples("port") == 32
    assert pa.n_samples("starboard") == 48
    assert pa.port.dtype == np.float32
    assert pa.slant_resolution("port") == pytest.approx(50.0 / 32)
    assert pa.sample_to_slant("port", 0) == pytest.approx(50.0 / 64)


def test_ping_count_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="ping-count mismatch"):
        PingArray(port=np.zeros((10, 32)), starboard=np.zeros((9, 32)), nav=_nav(10))


def test_nav_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="one record per ping"):
        PingArray(port=np.zeros((10, 32)), starboard=np.zeros((10, 32)), nav=_nav(9))


def test_bad_nav_dtype_rejected() -> None:
    with pytest.raises(ValueError, match="NAV_DTYPE"):
        PingArray(
            port=np.zeros((4, 8)), starboard=np.zeros((4, 8)), nav=np.zeros(4)
        )


def test_slice_pings_tracks_offset() -> None:
    pa = PingArray(port=np.zeros((10, 8)), starboard=np.zeros((10, 8)), nav=_nav(10))
    sub = pa.slice_pings(3, 7)
    assert sub.n_pings == 4
    assert sub.meta["ping_offset"] == 3
    subsub = sub.slice_pings(1, 3)
    assert subsub.meta["ping_offset"] == 4


def test_zero_width_side_allowed() -> None:
    pa = PingArray(port=np.zeros((5, 0)), starboard=np.zeros((5, 16)), nav=_nav(5))
    assert pa.slant_resolution("port") == 0.0
    assert pa.n_samples("starboard") == 16
