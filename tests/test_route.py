"""Route planner: tour covers every contact exactly once and 2-opt beats a
pathological visiting order."""

from __future__ import annotations

from geoscribe.contact import Contact, Dimensions, PhysicsEvidence, PixelRef
from geoscribe.route import plan_route


def _contact(cid: str, lat: float, lon: float) -> Contact:
    return Contact(
        id=cid, cls="ghost_net", confidence=80.0, lat=lat, lon=lon,
        dims=Dimensions(length_m=2, width_m=2, height_m=1),
        physics=PhysicsEvidence(highlight=True, shadow=True),
        severity=70.0,
        pixel=PixelRef(side="starboard", ping0=0, ping1=1, col0=0, col1=1),
    )


def test_route_visits_all_once() -> None:
    contacts = [
        _contact("A", 13.00, 80.30),
        _contact("B", 13.02, 80.30),
        _contact("C", 13.01, 80.30),
        _contact("D", 13.03, 80.30),
    ]
    plan = plan_route(contacts)
    assert [w["id"] for w in plan["waypoints"]] != []
    assert sorted(w["id"] for w in plan["waypoints"]) == ["A", "B", "C", "D"]
    assert len(plan["legs_m"]) == 3
    # Points on a line: the optimal sweep A->C->B->D is ~3.3 km; a naive
    # zig-zag would exceed 5 km.
    assert plan["total_m"] < 3500


def test_route_with_start_anchor() -> None:
    contacts = [_contact("A", 13.00, 80.30), _contact("B", 13.01, 80.30)]
    plan = plan_route(contacts, start_lat=13.005, start_lon=80.30)
    assert plan["start"] == {"lat": 13.005, "lon": 80.30}
    assert len(plan["waypoints"]) == 2
    assert len(plan["legs_m"]) == 2  # start->first, first->second


def test_route_empty() -> None:
    plan = plan_route([])
    assert plan["waypoints"] == [] and plan["total_m"] == 0.0
