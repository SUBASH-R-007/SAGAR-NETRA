"""Retrieval-zone clustering for the mission planner (blueprint N-06).

A recovery vessel does not hop contact-to-contact across the whole survey
box — it anchors over a debris field and works the field. Grouping contacts
into retrieval zones turns the flat contact list into an operational plan:
which zones exist, where to drop anchor (the severity-weighted centroid, so
the hook splashes nearest the worst debris), and in what order to work them.

Clustering is geodesic single-linkage via union-find over the full pairwise
distance matrix: two contacts share a zone when *any* chain of hops of at
most ``eps_m`` connects them — exactly how debris fields present on sonar,
as strings of contacts shed along a drift or dumping track. Contact counts
are tens, not millions, so the O(n^2) matrix costs microseconds and buys an
exact result with zero dependencies; sklearn's DBSCAN would add a heavyweight
import for no observable gain at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoscribe.contact import Contact
from geoscribe.route import geodesic_matrix, order_points, plan_route


@dataclass(frozen=True)
class Cluster:
    """One retrieval zone: member contact ids plus where to anchor."""

    contacts: list[str]
    centroid_lat: float
    centroid_lon: float
    total_severity: float
    n: int


def _union_find_labels(dist: np.ndarray, eps_m: float) -> list[int]:
    """Single-linkage component label (root index) per point."""
    n = len(dist)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    ii, jj = np.nonzero(np.triu(dist <= eps_m, k=1))
    for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return [find(i) for i in range(n)]


def cluster_contacts(
    contacts: list[Contact], eps_m: float = 150.0, min_size: int = 1
) -> list[Cluster]:
    """Group contacts into retrieval zones (geodesic single-linkage).

    ``eps_m`` is the linkage radius — 150 m default, roughly the working
    radius of an anchored recovery vessel. ``min_size`` drops zones smaller
    than the threshold (an isolated low-value contact may not justify an
    anchor stop). Centroids are severity-weighted arithmetic means of
    lat/lon — valid because a zone spans at most a few hundred metres, where
    the geographic grid is locally planar; uniform weights are used if a
    zone's severities sum to zero. Zones are returned worst-first
    (descending total severity, id tiebreak) for a deterministic order.
    """
    if not contacts:
        return []
    dist = geodesic_matrix([(c.lat, c.lon) for c in contacts])
    labels = _union_find_labels(dist, eps_m)

    groups: dict[int, list[int]] = {}
    for idx, root in enumerate(labels):
        groups.setdefault(root, []).append(idx)

    clusters: list[Cluster] = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        lats = np.array([contacts[i].lat for i in members])
        lons = np.array([contacts[i].lon for i in members])
        sev = np.array([contacts[i].severity for i in members], dtype=float)
        weights = sev if sev.sum() > 0 else np.ones(len(members))
        clusters.append(
            Cluster(
                contacts=[contacts[i].id for i in members],
                centroid_lat=float(np.average(lats, weights=weights)),
                centroid_lon=float(np.average(lons, weights=weights)),
                total_severity=float(sev.sum()),
                n=len(members),
            )
        )
    clusters.sort(key=lambda cl: (-cl.total_severity, cl.contacts[0]))
    return clusters


def plan_cluster_route(
    contacts: list[Contact],
    eps_m: float,
    start_lat: float | None = None,
    start_lon: float | None = None,
) -> dict:
    """Two-level recovery tour: order the zones, then work each zone.

    Zone visiting order comes from the route planner run over the zone
    centroids (anchored at the vessel start when given); within each zone
    the members are ordered by the same planner, anchored wherever the
    vessel arrives from — the previous zone's last contact, or the start.
    Returns the same shape as :func:`geoscribe.route.plan_route` (waypoints
    carry a global ``seq`` and a 1-based ``cluster`` tag) plus a
    ``clusters`` list in visiting order.
    """
    clusters = cluster_contacts(contacts, eps_m=eps_m)
    if not clusters:
        empty = plan_route([], start_lat, start_lon)
        empty["clusters"] = []
        return empty

    by_id = {c.id: c for c in contacts}
    centroids = [(cl.centroid_lat, cl.centroid_lon) for cl in clusters]
    has_start = start_lat is not None and start_lon is not None
    if has_start:
        tour, _ = order_points([(float(start_lat), float(start_lon)), *centroids])
        visit = [k - 1 for k in tour[1:]]
    else:
        tour, _ = order_points(centroids)
        visit = tour

    waypoints: list[dict] = []
    legs: list[float] = []
    ordered: list[dict] = []
    prev: tuple[float, float] | None = (
        (float(start_lat), float(start_lon)) if has_start else None
    )
    for zone_seq, k in enumerate(visit, start=1):
        cluster = clusters[k]
        members = [by_id[cid] for cid in cluster.contacts]
        sub = plan_route(
            members,
            prev[0] if prev is not None else None,
            prev[1] if prev is not None else None,
        )
        for wp in sub["waypoints"]:
            waypoints.append({**wp, "seq": len(waypoints) + 1, "cluster": zone_seq})
        legs.extend(sub["legs_m"])
        last = sub["waypoints"][-1]
        prev = (last["lat"], last["lon"])
        ordered.append(
            {
                "seq": zone_seq,
                "contact_ids": [wp["id"] for wp in sub["waypoints"]],
                "centroid_lat": cluster.centroid_lat,
                "centroid_lon": cluster.centroid_lon,
                "total_severity": cluster.total_severity,
                "n": cluster.n,
            }
        )

    return {
        "waypoints": waypoints,
        "legs_m": legs,
        "total_m": round(float(sum(legs)), 1),
        "start": {"lat": start_lat, "lon": start_lon} if has_start else None,
        "clusters": ordered,
    }
