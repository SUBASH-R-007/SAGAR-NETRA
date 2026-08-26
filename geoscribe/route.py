"""Recovery-route planner: order confirmed contacts into a short visiting tour.

Nearest-neighbour construction followed by a 2-opt improvement pass over
geodesic distances — for the handful-to-dozens of contacts a recovery vessel
visits, this lands within a few percent of optimal in microseconds, with no
solver dependency.
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

from geoscribe.contact import Contact

_GEOD = Geod(ellps="WGS84")


def _distance_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, _, d = _GEOD.inv(points[i][1], points[i][0], points[j][1], points[j][0])
            dist[i, j] = dist[j, i] = d
    return dist


def _nearest_neighbour(dist: np.ndarray, start: int) -> list[int]:
    n = len(dist)
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: dist[last, j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def _two_opt(tour: list[int], dist: np.ndarray, max_rounds: int = 8) -> list[int]:
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[j + 1]
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-9:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
    return tour


def plan_route(
    contacts: list[Contact],
    start_lat: float | None = None,
    start_lon: float | None = None,
) -> dict:
    """Ordered waypoint list + leg/total distances (metres).

    With a start position given, a virtual start node anchors the tour there
    (the vessel's position); otherwise the tour starts at the first contact.
    """
    if not contacts:
        return {"waypoints": [], "total_m": 0.0, "legs_m": []}

    points = [(c.lat, c.lon) for c in contacts]
    has_start = start_lat is not None and start_lon is not None
    if has_start:
        points = [(float(start_lat), float(start_lon)), *points]

    dist = _distance_matrix(points)
    tour = _two_opt(_nearest_neighbour(dist, 0), dist)

    legs = [float(dist[tour[k], tour[k + 1]]) for k in range(len(tour) - 1)]
    order = tour[1:] if has_start else tour
    offset = 1 if has_start else 0
    waypoints = [
        {
            "seq": seq + 1,
            "id": contacts[idx - offset].id,
            "cls": contacts[idx - offset].cls,
            "lat": contacts[idx - offset].lat,
            "lon": contacts[idx - offset].lon,
            "severity": contacts[idx - offset].severity,
        }
        for seq, idx in enumerate(order)
    ]
    return {
        "waypoints": waypoints,
        "legs_m": [round(leg, 1) for leg in legs],
        "total_m": round(float(sum(legs)), 1),
        "start": {"lat": start_lat, "lon": start_lon} if has_start else None,
    }
