"""Tests for side-scan beam geometry.

These pin the four sensor relationships the pipeline now depends on, against
the textbook figures for a 455 kHz towfish: 7.5 cm across-track resolution,
0.22 m along-track at 25 m and 0.65 m at 75 m, 0.75 m of range error from a
1% sound-speed error at 75 m, and a second bottom return at 1.73 altitudes.

The two resolution properties that matter operationally are asserted as
*behaviour*, not just arithmetic: across-track must not change with range, and
along-track must grow in proportion to it. Those are the reason a far-range
length is a softer number than a near-range one, and a regression in either
would quietly mislead every report.
"""

from __future__ import annotations

import math

import pytest

from sonar_core.geometry import (
    DEFAULT_SOUND_VELOCITY,
    SonarGeometry,
    across_track_resolution_m,
    along_track_resolution_m,
    is_multipath_candidate,
    multipath_ground_range_m,
    resolution_cell_m,
    slant_range_from_time,
    sound_speed_range_error_m,
)


def test_range_from_time_halves_the_round_trip() -> None:
    """R = c*t/2 — the pulse covers the distance twice."""
    assert slant_range_from_time(0.1, 1500.0) == pytest.approx(75.0)
    assert slant_range_from_time(0.0) == 0.0


def test_range_from_time_matches_the_jsf_parser() -> None:
    """The JSF adapter derives range this way; the two must not drift apart."""
    n_samples, interval_ns, sv = 1024, 100_000, 1500.0
    expected = n_samples * interval_ns * 1e-9 * sv / 2.0
    assert slant_range_from_time(n_samples * interval_ns * 1e-9, sv) == pytest.approx(expected)


def test_across_track_resolution_is_7_5_cm() -> None:
    assert across_track_resolution_m(1500.0, 1.0e-4) == pytest.approx(0.075)


def test_across_track_resolution_does_not_degrade_with_range() -> None:
    """It depends on pulse length alone — the far swath edge is as sharp as nadir."""
    fixed = across_track_resolution_m(1500.0, 1.0e-4)
    for range_m in (5.0, 25.0, 75.0, 200.0):
        _, across = resolution_cell_m(range_m, 0.5, 1500.0, 1.0e-4)
        assert across == pytest.approx(fixed)


@pytest.mark.parametrize(
    ("range_m", "expected"), [(25.0, 0.218), (50.0, 0.436), (75.0, 0.654)]
)
def test_along_track_resolution_matches_the_textbook_figures(
    range_m: float, expected: float
) -> None:
    assert along_track_resolution_m(0.5, range_m) == pytest.approx(expected, abs=5e-3)


def test_along_track_resolution_grows_linearly_with_range() -> None:
    """Tripling the range must triple the smear — this is why far contacts blur."""
    near = along_track_resolution_m(0.5, 25.0)
    far = along_track_resolution_m(0.5, 75.0)
    assert far == pytest.approx(3.0 * near)


def test_along_track_resolution_is_never_negative() -> None:
    assert along_track_resolution_m(0.5, -10.0) == 0.0


def test_sound_speed_error_is_a_scale_error() -> None:
    """1% of assumed c is 1% of every range: 0.75 m at 75 m."""
    assert sound_speed_range_error_m(75.0, 0.01) == pytest.approx(0.75)
    assert sound_speed_range_error_m(150.0, 0.01) == pytest.approx(1.5)
    assert sound_speed_range_error_m(75.0, 0.0) == 0.0


def test_second_bottom_return_sits_at_root_three_altitudes() -> None:
    """g = sqrt((2A)^2 - A^2) = A*sqrt(3), the same triangle as slant correction."""
    assert multipath_ground_range_m(10.0) == pytest.approx(10.0 * math.sqrt(3))
    assert multipath_ground_range_m(10.0, order=3) == pytest.approx(10.0 * math.sqrt(8))


def test_multipath_range_rejects_impossible_geometry() -> None:
    assert multipath_ground_range_m(0.0) == 0.0
    assert multipath_ground_range_m(-5.0) == 0.0
    assert multipath_ground_range_m(10.0, order=1) == 0.0


def test_multipath_flag_fires_only_inside_the_band() -> None:
    altitude = 10.0
    centre = multipath_ground_range_m(altitude)  # 17.32 m
    assert is_multipath_candidate(centre, altitude, 0.15)
    assert is_multipath_candidate(centre * 1.10, altitude, 0.15)
    assert not is_multipath_candidate(centre * 1.30, altitude, 0.15)
    assert not is_multipath_candidate(5.0, altitude, 0.15)


def test_multipath_flag_never_fires_on_unknown_geometry() -> None:
    """Missing altitude must not manufacture a suspicion out of nothing."""
    assert not is_multipath_candidate(17.3, 0.0)
    assert not is_multipath_candidate(17.3, float("nan"))
    assert not is_multipath_candidate(float("nan"), 10.0)


def test_config_loads_and_falls_back_cleanly(tmp_path) -> None:
    shipped = SonarGeometry.load()
    assert shipped.along_track_beam_deg == pytest.approx(0.5)
    assert shipped.pulse_length_s == pytest.approx(1.0e-4)

    missing = SonarGeometry.load(tmp_path / "absent.yaml")
    assert missing == SonarGeometry()

    partial = tmp_path / "partial.yaml"
    partial.write_text("geometry:\n  along_track_beam_deg: 1.0\n", encoding="utf-8")
    loaded = SonarGeometry.load(partial)
    assert loaded.along_track_beam_deg == pytest.approx(1.0)
    # Unspecified keys keep their defaults rather than becoming zero.
    assert loaded.pulse_length_s == pytest.approx(SonarGeometry().pulse_length_s)


def test_shipped_config_reproduces_the_quoted_figures() -> None:
    """The numbers in the README come from this config, not from prose."""
    g = SonarGeometry.load()
    assert across_track_resolution_m(DEFAULT_SOUND_VELOCITY, g.pulse_length_s) == (
        pytest.approx(0.075)
    )
    assert along_track_resolution_m(g.along_track_beam_deg, 75.0) == pytest.approx(
        0.654, abs=5e-3
    )


# ------------------------------------------------------ pipeline integration ----


@pytest.fixture(scope="module")
def scene():
    """A small rendered scene with its preprocessed ground imagery."""
    import numpy as np

    from sonar_core.preprocess.pipeline import preprocess
    from sonar_core.synth.scene import SceneConfig, make_scene
    from tridentnet.data import random_targets

    cfg = SceneConfig(n_pings=260, n_samples=512, slant_range=50.0, altitude=10.0, seed=7)
    pa, _ = make_scene(cfg, random_targets(cfg, np.random.default_rng(7)))
    return preprocess(pa)


def _detection_at(pre, ground_range_m: float):
    """A detection box centred on the given ground range, on the starboard side."""
    from tridentnet.detector import Detection

    col = int(ground_range_m / pre.ground_raw.ground_res)
    return Detection(
        side="starboard", ping0=100, ping1=112, col0=col - 5, col1=col + 5,
        cls="cylinder_drum", score=0.8,
    )


def test_multipath_flag_fires_on_the_second_bottom_return(scene) -> None:
    """A box at A*sqrt(3) is flagged; one well inside it is not."""
    from physicheck.verify import verify_detections

    altitude = float(scene.ground_raw.altitude_m[100])
    at_multipath = _detection_at(scene, multipath_ground_range_m(altitude))
    elsewhere = _detection_at(scene, multipath_ground_range_m(altitude) * 0.5)

    by_box = {
        (v.det.col0, v.det.col1): v
        for v in verify_detections([at_multipath, elsewhere], scene, use_verifier=False)
    }
    assert by_box[(at_multipath.col0, at_multipath.col1)].multipath_suspect
    assert not by_box[(elsewhere.col0, elsewhere.col1)].multipath_suspect


def test_multipath_flag_never_changes_a_confidence(scene, monkeypatch) -> None:
    """The advisory must be inert.

    If flagging could move a score it would have silently changed every
    published table the moment it was added. Widening the band from "nothing
    is multipath" to "almost everything is" must leave confidences bit-identical.
    """
    from physicheck import verify as verify_mod
    from physicheck.verify import verify_detections

    altitude = float(scene.ground_raw.altitude_m[100])
    dets = [
        _detection_at(scene, multipath_ground_range_m(altitude)),
        _detection_at(scene, multipath_ground_range_m(altitude) * 0.6),
    ]

    def _with_tolerance(frac):
        monkeypatch.setattr(
            verify_mod.SonarGeometry, "load",
            classmethod(lambda cls, *a, **k: SonarGeometry(multipath_tolerance_frac=frac)),
        )
        return verify_detections(dets, scene, use_verifier=False)

    none_flagged = _with_tolerance(0.0)
    all_flagged = _with_tolerance(0.9)

    assert not any(v.multipath_suspect for v in none_flagged)
    assert any(v.multipath_suspect for v in all_flagged)
    assert [v.confidence_pct for v in none_flagged] == [
        v.confidence_pct for v in all_flagged
    ]
    assert [v.gate.multiplier for v in none_flagged] == [
        v.gate.multiplier for v in all_flagged
    ]
