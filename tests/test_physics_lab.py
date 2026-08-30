"""Tests for the Physics Lab endpoints.

The lab's whole claim is that its sliders call the *deployed* physics rather
than a browser re-derivation, so these tests check the properties that claim
depends on: the shadow inversion round-trips, the resolution curve behaves the
way the sensor does (across-track flat, along-track rising), and a rendered
scene's measured heights land near the truth the renderer was given.

They also pin the guard rails. The simulator runs inside a request, so a caller
must not be able to ask for an unbounded scene, place an object outside the
swath, or crash it with a class that does not exist.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.physics_lab import MAX_TARGETS, geometry_report, shadow_round_trip


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(require_auth=False))


# ------------------------------------------------------------ shadow model --


@pytest.mark.parametrize(
    ("altitude", "height", "ground"),
    [(10.0, 2.0, 20.0), (8.0, 0.5, 30.0), (15.0, 4.0, 12.0), (6.0, 1.0, 45.0)],
)
def test_shadow_inversion_round_trips(altitude, height, ground) -> None:
    """Forward-modelled shadow, inverted, must return the height it came from."""
    r = shadow_round_trip(altitude, height, ground)
    assert r["recovered_height_m"] == pytest.approx(height, abs=1e-3)
    assert r["round_trip_error_m"] < 1e-3


def test_shadow_is_longer_than_the_object_is_tall() -> None:
    """The shadow amplifies height — the reason it is measured instead of the object.

    At 10 m altitude a 2 m object throws a 5 m shadow: a 2.5x lever on a
    quantity that is otherwise only a few pixels of brightness.
    """
    r = shadow_round_trip(10.0, 2.0, 20.0)
    assert r["shadow_length_m"] == pytest.approx(5.0)
    assert r["shadow_gain"] == pytest.approx(2.5)


def test_shadow_length_grows_with_range_at_fixed_height() -> None:
    """Same object further out casts a longer shadow — a shallower grazing angle."""
    near = shadow_round_trip(10.0, 2.0, 15.0)["shadow_length_m"]
    far = shadow_round_trip(10.0, 2.0, 45.0)["shadow_length_m"]
    assert far > near


def test_object_taller_than_towfish_is_clamped_not_divided_by_zero() -> None:
    """H -> A sends x_end -> infinity; the panel must clamp and say it did."""
    r = shadow_round_trip(10.0, 50.0, 20.0)
    assert r["height_clamped"] is True
    assert r["height_m"] == pytest.approx(9.0)  # 0.9 * altitude
    assert r["shadow_end_m"] > r["shadow_start_m"]


def test_flat_object_casts_no_shadow() -> None:
    r = shadow_round_trip(10.0, 0.0, 20.0)
    assert r["shadow_length_m"] == 0.0
    assert r["recovered_height_m"] == 0.0


# --------------------------------------------------------------- geometry --


def test_across_track_flat_and_along_track_rising_across_the_swath() -> None:
    """The two resolution limits behave differently, and that is the point."""
    curve = geometry_report(8.0, 50.0)["curve"]
    across = {row["across_track_m"] for row in curve}
    assert len(across) == 1  # constant with range

    along = [row["along_track_m"] for row in curve]
    assert along == sorted(along)  # monotonically worse
    assert along[-1] > along[0]


def test_geometry_accepts_a_different_sonar() -> None:
    """A narrower beam must improve along-track; a shorter pulse, across-track."""
    base = geometry_report(8.0, 50.0)
    narrow = geometry_report(8.0, 50.0, beam_deg=0.25)
    short = geometry_report(8.0, 50.0, pulse_us=50.0)
    assert narrow["along_track_resolution_far_m"] < base["along_track_resolution_far_m"]
    assert short["across_track_resolution_m"] < base["across_track_resolution_m"]


def test_geometry_swath_closes_when_altitude_meets_range() -> None:
    """No swath exists where the first bottom return is the whole range."""
    assert geometry_report(50.0, 50.0)["max_ground_range_m"] == pytest.approx(0.0)


# ---------------------------------------------------------------- routes --


def test_geometry_and_classes_routes(client) -> None:
    assert client.get("/api/physics/geometry?altitude_m=8&range_m=50").status_code == 200
    classes = client.get("/api/physics/classes").json()
    assert classes and all("cls" in c and "height_m" in c for c in classes)
    assert any(c["natural"] for c in classes)  # rock_cluster is placeable too


def test_shadow_route_matches_the_function(client) -> None:
    body = {"altitude_m": 10.0, "height_m": 2.0, "ground_range_m": 20.0}
    assert client.post("/api/physics/shadow", json=body).json() == shadow_round_trip(
        10.0, 2.0, 20.0
    )


def test_simulate_measures_placed_targets(client) -> None:
    """Heights recovered from shadow must land near what was placed."""
    res = client.post(
        "/api/physics/simulate",
        json={
            "targets": [
                {"cls": "cylinder_drum", "ground_range_m": 18.0, "height_m": 1.2},
                {"cls": "container", "ground_range_m": 28.0, "height_m": 2.4},
            ],
            "altitude_m": 9.0, "slant_range_m": 50.0, "n_pings": 300,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["targets"], "placed targets must survive rendering"
    for t in body["targets"]:
        assert t["measured_height_m"] is not None, f"no shadow found for {t['cls']}"
        assert t["measured_height_m"] == pytest.approx(t["truth_height_m"], abs=0.6)
        assert t["shadow_len_m"] > 0

    # The waterfall must be a real decodable PNG, not a placeholder.
    png = base64.b64decode(body["waterfall_png_b64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_simulate_needs_at_least_one_target(client) -> None:
    assert client.post("/api/physics/simulate", json={"targets": []}).status_code == 422


def test_simulate_clamps_a_target_placed_outside_the_swath(client) -> None:
    """A placement that cannot be imaged is corrected, not rendered as a puzzle."""
    body = client.post(
        "/api/physics/simulate",
        json={
            "targets": [{"cls": "tire", "ground_range_m": 5000.0}],
            "altitude_m": 8.0, "slant_range_m": 40.0, "n_pings": 240,
        },
    ).json()
    assert body["targets"][0]["ground_range_m"] <= body["max_ground_range_m"]


def test_simulate_ignores_an_unknown_class(client) -> None:
    body = client.post(
        "/api/physics/simulate",
        json={"targets": [{"cls": "spaceship"}, {"cls": "tire"}], "n_pings": 240},
    ).json()
    assert [t["cls"] for t in body["targets"]] == ["tire"]


def test_simulate_caps_the_work_it_will_accept(client) -> None:
    """Bounded no matter what is asked for -- this runs inside a request."""
    body = client.post(
        "/api/physics/simulate",
        json={
            "targets": [
                {"cls": "tire", "ground_range_m": 10.0 + i} for i in range(MAX_TARGETS + 6)
            ],
            "n_pings": 99_999,
        },
    ).json()
    assert len(body["targets"]) <= MAX_TARGETS
    assert body["n_pings"] <= 700
