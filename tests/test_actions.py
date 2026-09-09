"""Rule-based recommended actions (configs/actions.yaml).

The precedence order is the whole design, so it is tested as an order rather
than as isolated branches: always_override must beat a sensitive zone, a zone
must beat severity, and severity must beat depth. Getting that sequence wrong
would still produce plausible-looking advice, which is the failure mode worth
guarding against.
"""

from __future__ import annotations

import pytest

from geoscribe.actions import FALLBACK_ACTION, load_rules, recommend


def test_always_override_ignores_severity_and_zone() -> None:
    """A low-severity human body is still a human body. Under-reacting and
    over-reacting are not symmetric here, so score must not soften it."""
    for sev in (0, 40, 99):
        r = recommend("human_body", severity=sev, depth_m=5, nearest_layer_kind="mpa",
                      nearest_layer_distance_m=10)
        assert r.rule == "always_override"
        assert "SAR authority" in r.action


def test_zone_proximity_outranks_severity() -> None:
    """Proximity changes WHO must be notified, which severity never does."""
    r = recommend("ghost_net", severity=95, depth_m=20,
                  nearest_layer_kind="turtle_zone", nearest_layer_distance_m=100)
    assert r.rule == "near_sensitive_zone"
    assert "Wildlife Warden" in r.action


def test_zone_authority_is_substituted_per_layer() -> None:
    for layer, expected in (
        ("turtle_zone", "Wildlife Warden"),
        ("mpa", "marine protected area authority"),
    ):
        r = recommend("ghost_net", severity=10, nearest_layer_kind=layer,
                      nearest_layer_distance_m=50)
        assert expected in r.action
    # An unmapped layer must not leave a raw placeholder in operator-facing text.
    r = recommend("ghost_net", severity=10, nearest_layer_kind="unknown_kind",
                  nearest_layer_distance_m=50)
    assert "{" not in r.action and "relevant authority" in r.action


def test_zone_rule_ignores_a_distant_layer() -> None:
    r = recommend("ghost_net", severity=10, depth_m=20,
                  nearest_layer_kind="turtle_zone", nearest_layer_distance_m=5000)
    assert r.rule == "base"


def test_severity_outranks_depth() -> None:
    r = recommend("ghost_net", severity=90, depth_m=80)
    assert r.rule == "high_severity"


def test_depth_bands() -> None:
    assert recommend("ghost_net", severity=10, depth_m=80).rule == "deep"
    assert recommend("container", severity=10, depth_m=5).rule == "shallow"
    # Between the bands, neither fires.
    assert recommend("container", severity=10, depth_m=25).rule == "base"


def test_missing_depth_never_matches_a_depth_band() -> None:
    """None must not be treated as zero, which would read as 'shallow'."""
    assert recommend("container", severity=10, depth_m=None).rule == "base"


def test_unknown_class_falls_back_without_raising() -> None:
    r = recommend("reef", severity=30)
    assert r.action and "{" not in r.action


def test_every_rule_declares_the_permission_it_needs() -> None:
    """The console gates the control on this; a missing value would silently
    offer an action to a role that cannot perform it."""
    from api.auth import Permission

    valid = {p.value for p in Permission}
    for name, rule in (load_rules().get("classes") or {}).items():
        assert rule.get("requires") in valid, f"{name} declares no valid permission"


def test_reportable_classes_all_have_rules() -> None:
    """Any class an operator can see must carry an instruction."""
    from tridentnet.classes import CLASS_NAMES, is_reportable

    rules = (load_rules().get("classes") or {})
    missing = [c for c in CLASS_NAMES if is_reportable(c) and c not in rules]
    assert not missing, f"reportable classes without an action rule: {missing}"


def test_broken_config_degrades_instead_of_crashing(tmp_path) -> None:
    """An operator editing the YAML badly must not stop the pipeline."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("classes: [this is not: a mapping", encoding="utf-8")
    r = recommend("ghost_net", severity=50, config_path=bad)
    assert r.action == FALLBACK_ACTION

    missing = tmp_path / "absent.yaml"
    assert recommend("ghost_net", severity=50, config_path=missing).action == FALLBACK_ACTION


@pytest.mark.parametrize("cls", ["ghost_net", "mine_like", "container", "wreck"])
def test_actions_are_imperative_not_descriptive(cls: str) -> None:
    """These are instructions to an operator, not observations."""
    action = recommend(cls, severity=50).action
    assert len(action) > 20
    assert action[0].isupper() or action.startswith("URGENT")


def test_the_shipped_layers_all_name_an_authority() -> None:
    """Every mapped layer's `kind` must resolve to a named authority.

    This is the join that a unit test alone cannot check. ``severity.py``
    records the feature's display *name* ("Olive Ridley Nesting Buffer (demo)")
    for the operator and its *kind* (``turtle_zone``) for the rules; the
    authority table is keyed on kind. When those two were conflated, every
    zone recommendation silently degraded to "the relevant authority" while
    the unit tests above still passed, because they supplied kinds directly.
    """
    import json
    from pathlib import Path

    authorities = load_rules().get("zone_authorities") or {}
    layer_dir = Path(__file__).resolve().parents[1] / "data" / "layers"
    kinds = {
        feature["properties"]["kind"]
        for path in layer_dir.glob("*.geojson")
        for feature in json.loads(path.read_text(encoding="utf-8"))["features"]
    }
    assert kinds, "no mapped layers to check"
    missing = kinds - authorities.keys()
    assert not missing, f"layers with no authority in configs/actions.yaml: {missing}"

    for kind in kinds:
        r = recommend("ghost_net", severity=10, nearest_layer_kind=kind,
                      nearest_layer_distance_m=50)
        assert r.rule == "near_sensitive_zone"
        assert authorities[kind] in r.action
        assert "the relevant authority" not in r.action
