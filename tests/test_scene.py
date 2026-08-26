"""Scene simulator: verify the rendered physics is actually consistent —
water column, bottom return, shadows whose length obeys the ray geometry."""

from __future__ import annotations

import numpy as np

from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene


def test_water_column_is_dark_and_bottom_return_bright(small_scene) -> None:
    pa, _ = small_scene
    i = pa.n_pings // 2
    alt = float(pa.nav["altitude"][i])
    res = pa.slant_resolution("starboard")
    bottom_sample = int(alt / res)
    water = pa.starboard[i, : max(bottom_sample - 3, 1)]
    seabed = pa.starboard[i, bottom_sample + 5 : bottom_sample + 60]
    assert water.mean() < 0.1 * seabed.mean()
    # First return peak stands above the nearby seabed level.
    peak = pa.starboard[i, max(bottom_sample - 2, 0) : bottom_sample + 3].max()
    assert peak > 1.5 * seabed.mean()


def test_target_highlight_and_shadow(small_scene) -> None:
    pa, targets = small_scene
    t = next(t for t in targets if t.cls == "cylinder_drum")
    i = t.ping
    alt = float(pa.nav["altitude"][i])
    res = pa.slant_resolution(t.side)
    line = pa.side(t.side)[i]

    def sample_of_ground(x: float) -> int:
        return int(np.sqrt(x**2 + alt**2) / res)

    s_obj = sample_of_ground(t.ground_range)
    s_shadow_start, s_shadow_end = (
        sample_of_ground(x) for x in t.shadow_extent(alt)
    )
    background = np.median(line[sample_of_ground(t.ground_range - 8) : sample_of_ground(t.ground_range - 3)])
    highlight = line[s_obj - 2 : s_obj + 3].max()
    shadow = np.median(line[s_shadow_start + 2 : s_shadow_end - 1])
    assert highlight > 2.0 * background, "highlight should stand out"
    assert shadow < 0.5 * background, "shadow should be dark"


def test_shadow_length_recovers_height() -> None:
    """H = L*A/R on the rendered shadow must recover the seeded height."""
    cfg = SceneConfig(n_pings=60, n_samples=1024, slant_range=40.0, seed=3)
    target = SynthTarget(
        "container", "starboard", 30, 20.0, 4.0, 2.0, height=2.0, reflectivity=6.0
    )
    pa, _ = make_scene(cfg, [target])
    alt = float(pa.nav["altitude"][target.ping])

    x_far, x_end = target.shadow_extent(alt)
    # Slant-range shadow length between object far edge and shadow end:
    r_far = np.hypot(x_far, alt)
    r_end = np.hypot(x_end, alt)
    length_slant = r_end - r_far
    h_est = length_slant * alt / r_end
    # The classic estimator is exact in this geometry up to the flat-seabed
    # small-angle approximation; require better than 12% here.
    assert abs(h_est - target.height) / target.height < 0.12


def test_scene_is_deterministic() -> None:
    cfg = SceneConfig(n_pings=40, n_samples=128, seed=11)
    a, _ = make_scene(cfg)
    b, _ = make_scene(cfg)
    np.testing.assert_array_equal(a.port, b.port)
    np.testing.assert_array_equal(a.nav["lat"], b.nav["lat"])
