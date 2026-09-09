"""Cross-view corroboration: agreement boosts, single-view demotes and flags
for re-survey, class disagreement blocks the match, and class-agnostic
anomalies corroborate classed contacts."""

from __future__ import annotations

import pytest

from geoscribe.contact import Contact, Dimensions, PhysicsEvidence, PixelRef
from physicheck.crossview import cross_confirm
from tridentnet.classes import ANOMALY_CLASS

# ~0.00005 deg latitude is ~5.6 m: comfortably inside the 15 m default radius.
NEAR = 0.00005
# ~0.001 deg latitude is ~111 m: far outside it.
FAR = 0.001


def _contact(
    cid: str,
    lat: float,
    lon: float,
    cls: str = "container",
    confidence: float = 80.0,
    survey: str = "a",
) -> Contact:
    return Contact(
        id=cid, cls=cls, confidence=confidence, lat=lat, lon=lon,
        dims=Dimensions(length_m=2, width_m=2, height_m=1),
        physics=PhysicsEvidence(highlight=True, shadow=True),
        severity=60.0, survey=survey,
        pixel=PixelRef(side="starboard", ping0=0, ping1=1, col0=0, col1=1),
    )


def test_matching_pair_confirmed_and_boosted() -> None:
    a = [_contact("A1", 13.0, 80.3)]
    b = [_contact("B1", 13.0 + NEAR, 80.3, survey="b")]
    result = cross_confirm(a, b)

    assert len(result.confirmed) == 1
    assert result.a_only == [] and result.b_only == []
    a_dict, b_dict, dist = result.confirmed[0]
    assert a_dict["id"] == "A1" and b_dict["id"] == "B1"
    assert 0 < dist < 15.0
    assert a_dict["adjusted_confidence"] == pytest.approx(80.0 * 1.15)
    assert b_dict["adjusted_confidence"] == pytest.approx(80.0 * 1.15)
    # Stored contacts are never mutated: only the dumps carry the adjustment.
    assert a[0].confidence == 80.0 and b[0].confidence == 80.0


def test_boost_capped_at_100() -> None:
    a = [_contact("A1", 13.0, 80.3, confidence=95.0)]
    b = [_contact("B1", 13.0, 80.3, confidence=95.0, survey="b")]
    result = cross_confirm(a, b)
    a_dict, b_dict, _ = result.confirmed[0]
    assert a_dict["adjusted_confidence"] == 100.0
    assert b_dict["adjusted_confidence"] == 100.0


def test_unmatched_a_side_demoted_with_resurvey_flag() -> None:
    a = [_contact("A1", 13.0, 80.3), _contact("A2", 13.0 + FAR, 80.3)]
    b = [_contact("B1", 13.0 + NEAR, 80.3, survey="b")]
    result = cross_confirm(a, b)

    assert len(result.confirmed) == 1
    assert [c["id"] for c in result.a_only] == ["A2"]
    only = result.a_only[0]
    assert only["resurvey_recommended"] is True
    assert only["adjusted_confidence"] == pytest.approx(80.0 * 0.85)
    assert a[1].confidence == 80.0  # never mutated


def test_demotion_floors_at_one_percent() -> None:
    a = [_contact("A1", 13.0, 80.3, confidence=1.0)]
    result = cross_confirm(a, [])
    assert result.a_only[0]["adjusted_confidence"] == 1.0


def test_class_mismatch_at_same_position_not_confirmed() -> None:
    a = [_contact("A1", 13.0, 80.3, cls="tire")]
    b = [_contact("B1", 13.0, 80.3, cls="container", survey="b")]
    result = cross_confirm(a, b)

    assert result.confirmed == []
    assert [c["id"] for c in result.a_only] == ["A1"]
    assert [c["id"] for c in result.b_only] == ["B1"]
    assert all(c["resurvey_recommended"] for c in result.a_only + result.b_only)


def test_unknown_anomaly_corroborates_classed_contact() -> None:
    a = [_contact("A1", 13.0, 80.3, cls="container")]
    b = [_contact("B1", 13.0 + NEAR, 80.3, cls=ANOMALY_CLASS, survey="b")]
    result = cross_confirm(a, b)

    assert len(result.confirmed) == 1
    a_dict, b_dict, _ = result.confirmed[0]
    assert a_dict["cls"] == "container" and b_dict["cls"] == ANOMALY_CLASS


def test_outside_radius_not_confirmed() -> None:
    a = [_contact("A1", 13.0, 80.3)]
    b = [_contact("B1", 13.0 + FAR, 80.3, survey="b")]
    result = cross_confirm(a, b)
    assert result.confirmed == []
    assert len(result.a_only) == 1 and len(result.b_only) == 1


def test_to_dict_counts_and_shape() -> None:
    a = [_contact("A1", 13.0, 80.3), _contact("A2", 13.0 + FAR, 80.3)]
    b = [_contact("B1", 13.0, 80.3, survey="b")]
    payload = cross_confirm(a, b).to_dict()

    assert payload["n_confirmed"] == 1
    assert payload["n_a_only"] == 1 and payload["n_b_only"] == 0
    assert payload["radius_m"] == 15.0
    entry = payload["confirmed"][0]
    assert set(entry) == {"a", "b", "distance_m"}


def test_api_crossview_endpoint(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from api.db import ContactRepo
    from api.main import create_app

    repo = ContactRepo(tmp_path / "contacts.db")
    repo.add_contacts(
        [
            _contact("S1-1", 13.0, 80.3, survey="s1"),
            _contact("S1-2", 13.0 + FAR, 80.3, survey="s1"),
            _contact("S2-1", 13.0 + NEAR, 80.3, survey="s2"),
        ]
    )
    app = create_app(
        require_auth=False,  # exercise the pipeline, not the RBAC layer (tests/test_auth.py)
        repo=repo, upload_dir=tmp_path / "uploads", output_root=tmp_path / "outputs"
    )
    with TestClient(app) as client:
        payload = client.get(
            "/api/crossview",
            params={"survey_a": "s1", "survey_b": "s2", "radius_m": 15},
        ).json()

    assert payload["survey_a"] == "s1" and payload["survey_b"] == "s2"
    assert payload["n_confirmed"] == 1 and payload["n_a_only"] == 1
    assert payload["confirmed"][0]["a"]["id"] == "S1-1"
    assert payload["a_only"][0]["resurvey_recommended"] is True
