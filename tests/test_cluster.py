"""Retrieval-zone clustering: membership, eps sensitivity, severity-weighted
anchor points, and the two-level cluster route."""

from __future__ import annotations

import pytest

from geoscribe.cluster import cluster_contacts, plan_cluster_route
from geoscribe.contact import Contact, Dimensions, PhysicsEvidence, PixelRef


def _contact(cid: str, lat: float, lon: float, severity: float = 50.0) -> Contact:
    return Contact(
        id=cid, cls="cylinder_drum", confidence=80.0, lat=lat, lon=lon,
        dims=Dimensions(length_m=2, width_m=1, height_m=0.8),
        physics=PhysicsEvidence(highlight=True, shadow=True),
        severity=severity,
        pixel=PixelRef(side="port", ping0=0, ping1=1, col0=0, col1=1),
    )


def _three_groups() -> tuple[list[Contact], list[list[str]]]:
    """Three tight debris fields ~1.1 km apart, members within ~35 m."""
    contacts = [
        _contact("a1", 13.0000, 80.3000),
        _contact("a2", 13.0003, 80.3000),
        _contact("a3", 13.0000, 80.3003),
        _contact("b1", 13.0100, 80.3000),
        _contact("b2", 13.0103, 80.3000),
        _contact("c1", 13.0200, 80.3000),
        _contact("c2", 13.0203, 80.3003),
    ]
    groups = [["a1", "a2", "a3"], ["b1", "b2"], ["c1", "c2"]]
    return contacts, groups


def test_three_tight_groups_yield_three_clusters() -> None:
    contacts, groups = _three_groups()
    clusters = cluster_contacts(contacts, eps_m=150.0)

    assert len(clusters) == 3
    memberships = sorted(sorted(cl.contacts) for cl in clusters)
    assert memberships == groups
    for cl in clusters:
        assert cl.n == len(cl.contacts)
        assert cl.total_severity == pytest.approx(50.0 * cl.n)


def test_eps_sensitivity() -> None:
    contacts, _ = _three_groups()
    # Below the ~35 m member spacing: everything is a singleton.
    assert len(cluster_contacts(contacts, eps_m=5.0)) == 7
    # Above the ~1.1 km group spacing: single linkage chains one big zone.
    assert len(cluster_contacts(contacts, eps_m=2000.0)) == 1


def test_min_size_drops_small_zones() -> None:
    contacts, groups = _three_groups()
    clusters = cluster_contacts(contacts, eps_m=150.0, min_size=3)
    assert len(clusters) == 1
    assert sorted(clusters[0].contacts) == groups[0]


def test_severity_weighted_centroid_pulls_toward_severe_contact() -> None:
    heavy = _contact("H", 13.000, 80.300, severity=90.0)
    light = _contact("L", 13.001, 80.300, severity=10.0)
    (cluster,) = cluster_contacts([heavy, light], eps_m=200.0)

    expected_lat = (90.0 * 13.000 + 10.0 * 13.001) / 100.0
    assert cluster.centroid_lat == pytest.approx(expected_lat, abs=1e-9)
    assert cluster.centroid_lon == pytest.approx(80.300)
    # The anchor point lands much closer to the severe contact.
    assert abs(cluster.centroid_lat - heavy.lat) < abs(cluster.centroid_lat - light.lat)


def test_cluster_route_visits_all_once_in_centroid_order() -> None:
    contacts, groups = _three_groups()
    plan = plan_cluster_route(contacts, eps_m=150.0, start_lat=12.999, start_lon=80.30)

    ids = [w["id"] for w in plan["waypoints"]]
    assert sorted(ids) == sorted(c.id for c in contacts)
    assert len(ids) == len(set(ids))  # exactly once
    assert [w["seq"] for w in plan["waypoints"]] == list(range(1, 8))

    # Start just south of group a: nearest-centroid order is a -> b -> c.
    zone_sets = [set(z["contact_ids"]) for z in plan["clusters"]]
    assert zone_sets == [set(g) for g in groups]
    assert [z["seq"] for z in plan["clusters"]] == [1, 2, 3]
    # Waypoints stay contiguous per zone: work a field fully before moving on.
    assert [w["cluster"] for w in plan["waypoints"]] == [1, 1, 1, 2, 2, 3, 3]

    # Same shape as plan_route, plus the clusters list.
    assert set(plan) == {"waypoints", "legs_m", "total_m", "start", "clusters"}
    assert plan["start"] == {"lat": 12.999, "lon": 80.30}
    assert len(plan["legs_m"]) == 7  # start->first, then 6 chained hops
    assert plan["total_m"] == pytest.approx(sum(plan["legs_m"]), abs=0.5)


def test_cluster_route_without_start_and_empty() -> None:
    contacts, _ = _three_groups()
    plan = plan_cluster_route(contacts, eps_m=150.0)
    assert plan["start"] is None
    assert len(plan["waypoints"]) == 7
    assert len(plan["legs_m"]) == 6  # no anchor leg into the first zone

    empty = plan_cluster_route([], eps_m=150.0)
    assert empty["waypoints"] == [] and empty["clusters"] == []
    assert empty["total_m"] == 0.0


def test_api_route_cluster_param(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from api.db import ContactRepo
    from api.main import create_app

    contacts, groups = _three_groups()
    repo = ContactRepo(tmp_path / "contacts.db")
    repo.add_contacts(contacts)
    app = create_app(
        repo=repo, upload_dir=tmp_path / "uploads", output_root=tmp_path / "outputs"
    )
    with TestClient(app) as client:
        plain = client.get("/api/route", params={"review": "pending"}).json()
        clustered = client.get(
            "/api/route", params={"review": "pending", "cluster_eps_m": 150.0}
        ).json()

    assert "clusters" not in plain  # default behaviour unchanged
    assert len(plain["waypoints"]) == 7
    assert len(clustered["clusters"]) == 3
    assert sorted(w["id"] for w in clustered["waypoints"]) == sorted(
        c.id for c in contacts
    )
