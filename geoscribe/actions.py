"""Recommended actions: what to DO about a contact, from a rule file.

The previous version was a hard-coded (class, severity-band) table in
``report.py``. The wording of an operational instruction is domain knowledge —
a survey chief or a port liaison should be able to correct "notify DGCA" to
whatever their standing order actually says without editing Python — so the
rules moved to ``configs/actions.yaml`` and this module resolves them.

Resolution is most-specific-first, and the order is the interesting part:

1. ``always_override`` — the class alone decides. A low-severity human body is
   still a human body; softening that by score would be a category error, so
   these classes never reach the severity branches at all.
2. ``near_sensitive_zone`` — proximity to a mapped habitat or lane outranks
   severity, because it changes *who must be told* before anyone intervenes,
   not merely how urgent it is.
3. ``high_severity`` → 4. ``deep`` / ``shallow`` → 5. ``base``.

Every rule also names the permission required to carry it out. The console
shows the recommendation to every role — an analyst has to be able to see that
a contact needs an ROV — but only a role holding that permission is offered
the control. Advice and authority are different things.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "actions.yaml"

#: Used when a class has no entry at all — including the hard-negative classes,
#: which never reach an operator anyway.
FALLBACK_CLASS = "unknown_anomaly"
FALLBACK_ACTION = "Review manually - no standing instruction for this class"


@dataclass(frozen=True)
class Recommendation:
    """What to do, why that rule fired, and who may do it."""

    action: str
    #: Which branch produced it: base | high_severity | near_sensitive_zone |
    #: deep | shallow | always_override. Surfaced so an operator can see the
    #: rule that fired rather than trusting an unexplained sentence.
    rule: str
    #: Permission needed to action it (api/auth.py), or None when the rule
    #: names none.
    requires: str | None = None


@lru_cache(maxsize=4)
def load_rules(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Parse the rule file once per path.

    A missing or unreadable file yields empty rules rather than raising: the
    reporting pipeline must still produce contacts when an operator has broken
    the YAML, with the fallback action making the breakage visible.
    """
    config = Path(path)
    if not config.is_file():
        return {}
    try:
        return yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def recommend(
    cls: str,
    severity: float = 0.0,
    depth_m: float | None = None,
    nearest_layer_kind: str | None = None,
    nearest_layer_distance_m: float | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Recommendation:
    """Resolve one contact's recommended action.

    Takes scalars rather than a Contact so it can be unit-tested without
    building a full record, and so ``geoscribe.report`` does not import the
    contact model it is imported by.

    ``nearest_layer_kind`` is the layer's *kind* (``turtle_zone``), not its
    display name ("Olive Ridley Nesting Buffer"): the authority table is keyed
    by kind so renaming a feature on the map cannot silently drop the contact
    back to "the relevant authority".
    """
    rules = load_rules(config_path)
    classes = rules.get("classes") or {}
    rule = classes.get(cls) or classes.get(FALLBACK_CLASS)
    if not rule:
        return Recommendation(FALLBACK_ACTION, "fallback", None)

    requires = rule.get("requires")
    thresholds = rules.get("thresholds") or {}

    def pick(key: str) -> Recommendation | None:
        text = rule.get(key)
        return Recommendation(text, key, requires) if text else None

    # 1. Class-only override.
    if rule.get("always_override"):
        return Recommendation(rule.get("base", FALLBACK_ACTION), "always_override", requires)

    # 2. Sensitive-zone proximity. Checked before severity because it changes
    #    who must be notified, which severity alone never does.
    radius = float(thresholds.get("zone_radius_m", 500))
    if (
        nearest_layer_kind
        and nearest_layer_distance_m is not None
        and nearest_layer_distance_m < radius
        and rule.get("near_sensitive_zone")
    ):
        authorities = rules.get("zone_authorities") or {}
        return Recommendation(
            rule["near_sensitive_zone"].format(
                zone_authority=authorities.get(nearest_layer_kind, "the relevant authority")
            ),
            "near_sensitive_zone",
            requires,
        )

    # 3. Severity.
    if severity >= float(thresholds.get("high_severity_at", 75)) and (hit := pick("high_severity")):
        return hit

    # 4. Depth bands. Deep first: an object both deep and shallow is
    #    impossible, but an unset threshold should not silently match.
    if depth_m is not None:
        if depth_m > float(thresholds.get("deep_m", 40)) and (hit := pick("deep")):
            return hit
        if depth_m < float(thresholds.get("shallow_m", 10)) and (hit := pick("shallow")):
            return hit

    # 5. Fallback.
    return Recommendation(rule.get("base", FALLBACK_ACTION), "base", requires)
