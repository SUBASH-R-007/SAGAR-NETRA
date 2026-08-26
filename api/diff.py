"""Change detection between two surveys of the same area.

A contact in survey B is *new* when no survey-A contact lies within the match
radius (geodesic metres). This is the M7 "diff mode" backend; 25 m default
matches typical SSS positioning error after layback correction.
"""

from __future__ import annotations

from pyproj import Geod

from api.db import ContactRepo
from geoscribe.contact import Contact

_GEOD = Geod(ellps="WGS84")


def geodesic_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(dist)


def diff_surveys(
    repo: ContactRepo, survey_a: str, survey_b: str, radius_m: float = 25.0
) -> dict:
    """Contacts of B partitioned into new vs matched against A."""
    a_contacts = repo.query(survey=survey_a, limit=10_000)
    b_contacts = repo.query(survey=survey_b, limit=10_000)

    new: list[Contact] = []
    matched: list[tuple[Contact, Contact, float]] = []
    for b in b_contacts:
        best: tuple[Contact, float] | None = None
        for a in a_contacts:
            dist = geodesic_m(b.lat, b.lon, a.lat, a.lon)
            if best is None or dist < best[1]:
                best = (a, dist)
        if best is not None and best[1] <= radius_m:
            matched.append((b, best[0], best[1]))
        else:
            new.append(b)

    return {
        "survey_a": survey_a,
        "survey_b": survey_b,
        "radius_m": radius_m,
        "n_a": len(a_contacts),
        "n_b": len(b_contacts),
        "new_contacts": [c.model_dump(mode="json") for c in new],
        "matched": [
            {
                "b": b.model_dump(mode="json"),
                "a": a.model_dump(mode="json"),
                "distance_m": round(d, 1),
            }
            for b, a, d in matched
        ],
    }
