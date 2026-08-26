"""Bottom tracking: recovers the true altitude series from rendered pings."""

from __future__ import annotations

import numpy as np

from sonar_core.preprocess.bottom_track import blank_water_column, track_bottom
from sonar_core.synth.scene import SceneConfig, make_scene


def test_tracks_true_altitude(small_scene) -> None:
    pa, _ = small_scene
    bt = track_bottom(pa)
    true_alt = pa.nav["altitude"]
    err = np.abs(bt.altitude_m - true_alt)
    # First-return picking is quantised to ~res (0.15 m here); allow 3 samples.
    assert np.median(err) < 3.5 * pa.slant_resolution("starboard")
    assert bt.valid.mean() > 0.9


def test_tracks_through_altitude_wobble() -> None:
    cfg = SceneConfig(n_pings=300, n_samples=512, slant_range=40.0,
                      altitude=9.0, altitude_wobble=1.2, seed=5)
    pa, _ = make_scene(cfg, targets=[])
    bt = track_bottom(pa)
    corr = np.corrcoef(bt.altitude_m, pa.nav["altitude"])[0, 1]
    assert corr > 0.95, "tracked altitude should follow the heave"


def test_blank_water_column(small_scene) -> None:
    pa, _ = small_scene
    bt = track_bottom(pa)
    blanked = blank_water_column(pa, bt)
    i = pa.n_pings // 2
    fr = int(bt.first_return["starboard"][i])
    assert blanked.starboard[i, : fr].max() == 0.0
    # Seabed beyond the first return is untouched.
    np.testing.assert_array_equal(
        blanked.starboard[i, fr + 1 :], pa.starboard[i, fr + 1 :]
    )
    assert blanked.meta["water_column_blanked"] is True
    np.testing.assert_array_equal(blanked.nav["altitude"], bt.altitude_m)


def test_header_fallback_when_untrackable() -> None:
    """All-flat data has no bottom return; tracking must fall back to header."""
    cfg = SceneConfig(n_pings=50, n_samples=128, seed=2)
    pa, _ = make_scene(cfg, targets=[])
    flat = pa.port * 0 + 100.0
    pa.port[:] = flat
    pa.starboard[:] = flat
    bt = track_bottom(pa)
    assert not bt.valid.any()
    np.testing.assert_allclose(bt.altitude_m, pa.nav["altitude"], atol=1e-3)
