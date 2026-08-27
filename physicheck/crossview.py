"""Multi-view cross-confirmation — the L3 cross-swath doctrine.

Overlapping surveys (or a repeat pass down the same line) image the same
patch of seabed from two vantage points. A real object persists: it returns
a highlight and casts a shadow on both passes, at the same geodesic position.
A fish school, surface-return artefact, or water-column ghost does not.
Agreement between passes is therefore independent acoustic corroboration and
earns a confidence boost; a contact seen from only one pass is *demoted and
flagged for re-survey* — never deleted — so the operator sees an honest
ranking, not a filtered world view (same doctrine as ``physicheck.verify``).

Class handling: a match requires the same class on both sides, with one
deliberate exception — ``unknown_anomaly``. Brain C's open-set anomalies are
reconstruction-error blobs with no class semantics, so an anomaly at the same
position as a classed contact corroborates it: both passes agree *something
non-seabed* sits there, which is precisely the evidence cross-view seeks.

Matching is greedy nearest-first one-to-one over the candidate pairs; with
tens of contacts per survey this is exact enough that a Hungarian assignment
would change nothing while adding a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api.diff import geodesic_m
from geoscribe.contact import Contact
from tridentnet.classes import ANOMALY_CLASS


def _classes_agree(cls_a: str, cls_b: str) -> bool:
    """Same class, or one side is the class-agnostic open-set anomaly."""
    return cls_a == cls_b or ANOMALY_CLASS in (cls_a, cls_b)


def _boosted(contact: Contact, boost: float) -> dict[str, Any]:
    """JSON-ready dump with corroboration-boosted confidence, capped at 100."""
    dump = contact.model_dump(mode="json")
    dump["adjusted_confidence"] = min(contact.confidence * boost, 100.0)
    return dump


def _demoted(contact: Contact, demote: float) -> dict[str, Any]:
    """JSON-ready dump with single-view-demoted confidence (floor 1.0) and
    the re-survey flag: absence on the other pass is a data-collection gap
    until proven otherwise."""
    dump = contact.model_dump(mode="json")
    dump["adjusted_confidence"] = max(contact.confidence * demote, 1.0)
    dump["resurvey_recommended"] = True
    return dump


@dataclass(frozen=True)
class CrossViewResult:
    """Outcome of comparing two overlapping passes.

    All entries are plain JSON-ready dicts carrying the cross-view verdict in
    ``adjusted_confidence`` (and ``resurvey_recommended`` on the singles);
    the stored :class:`~geoscribe.contact.Contact` objects are never mutated.
    """

    confirmed: list[tuple[dict[str, Any], dict[str, Any], float]]
    a_only: list[dict[str, Any]]
    b_only: list[dict[str, Any]]
    radius_m: float

    def to_dict(self) -> dict[str, Any]:
        """Counts + lists, shaped like the ``/api/diff`` payload."""
        return {
            "radius_m": self.radius_m,
            "n_confirmed": len(self.confirmed),
            "n_a_only": len(self.a_only),
            "n_b_only": len(self.b_only),
            "confirmed": [
                {"a": a, "b": b, "distance_m": round(dist, 1)}
                for a, b, dist in self.confirmed
            ],
            "a_only": self.a_only,
            "b_only": self.b_only,
        }


def cross_confirm(
    contacts_a: list[Contact],
    contacts_b: list[Contact],
    radius_m: float = 15.0,
    boost: float = 1.15,
    demote: float = 0.85,
) -> CrossViewResult:
    """Cross-confirm two overlapping surveys of the same area.

    A pair is *confirmed* when the geodesic separation is within ``radius_m``
    (15 m default: typical residual SSS positioning error after layback
    correction, tighter than the 25 m diff radius because both passes have
    been georeferenced) and the classes agree per :func:`_classes_agree`.
    Confirmed contacts carry ``adjusted_confidence = min(conf * boost, 100)``;
    unmatched ones carry ``adjusted_confidence = max(conf * demote, 1)`` plus
    ``resurvey_recommended: True``. One-to-one matching, nearest pairs first.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(contacts_a):
        for j, b in enumerate(contacts_b):
            dist = geodesic_m(a.lat, a.lon, b.lat, b.lon)
            if dist <= radius_m and _classes_agree(a.cls, b.cls):
                candidates.append((dist, i, j))
    candidates.sort()

    used_a: set[int] = set()
    used_b: set[int] = set()
    confirmed: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for dist, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        confirmed.append(
            (_boosted(contacts_a[i], boost), _boosted(contacts_b[j], boost), dist)
        )

    a_only = [_demoted(c, demote) for i, c in enumerate(contacts_a) if i not in used_a]
    b_only = [_demoted(c, demote) for j, c in enumerate(contacts_b) if j not in used_b]
    return CrossViewResult(
        confirmed=confirmed, a_only=a_only, b_only=b_only, radius_m=radius_m
    )
