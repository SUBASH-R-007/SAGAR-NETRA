"""Strategy-PDF section 5.3 schema upgrades: triage priority / recommended
action rules, the honest position-accuracy budget, the contacts.json summary
block, and backward compatibility of stored contact JSON."""

from __future__ import annotations

import json

import pytest

from geoscribe.build import (
    build_contacts,
    nav_fix_uncertainty_m,
    position_accuracy,
    survey_stats,
)
from geoscribe.contact import Contact, contacts_json_schema
from geoscribe.report import (
    priority_for,
    recommended_action_for,
    write_all,
    write_contacts_csv,
)
from physicheck.verify import verify_detections
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.detector import Detection


class StubDetector:
    """One deterministic reportable box per side (test_api pattern)."""

    def detect_tiles(self, tiles, progress=None):
        by_side = {}
        for tile in tiles:
            by_side.setdefault(tile.side, tile)
        detections = []
        for side, tile in by_side.items():
            h, w = tile.image.shape
            r0, c0 = tile.row0 + h // 3, tile.col0 + w // 3
            detections.append(
                Detection(
                    side=side, ping0=r0, ping1=r0 + 12, col0=c0, col1=c0 + 18,
                    cls="container", score=0.82, tile_index=tile.index,
                )
            )
        if progress:
            progress("detect", 1.0)
        return detections


@pytest.fixture(scope="module")
def processed():
    """Small survey run through preprocess -> stub detect -> verify -> build."""
    cfg = SceneConfig(n_pings=200, n_samples=256, slant_range=40.0, seed=11)
    targets = [
        SynthTarget("container", "starboard", 100, 20.0, 3.0, 2.0, 1.5, reflectivity=5.5),
        SynthTarget("cylinder_drum", "port", 60, 14.0, 1.4, 0.9, 0.9, reflectivity=6.0),
    ]
    pa, _ = make_scene(cfg, targets)
    pre = preprocess(pa)
    verified = verify_detections(StubDetector().detect_tiles(pre.tiles), pre)
    contacts = build_contacts(verified, pre, survey="schema_test.xtf")
    return cfg, pre, contacts


# ------------------------------------------------------ priority / action ----


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (100.0, "HIGH"),
        (75.0, "HIGH"),
        (74.9, "MEDIUM"),
        (50.0, "MEDIUM"),
        (49.9, "LOW"),
        (25.0, "LOW"),
        (0.0, "LOW"),
    ],
)
def test_priority_thresholds(severity: float, expected: str) -> None:
    assert priority_for(severity) == expected


@pytest.mark.parametrize(
    ("cls", "severity", "expected"),
    [
        ("ghost_net", 80.0, "Entanglement hazard — flag for ROV recovery"),
        ("ghost_net", 55.0, "Entanglement hazard — flag for ROV recovery"),
        ("ghost_net", 40.0, "Log and monitor"),
        ("human_body", 5.0, "Notify SAR authority immediately"),
        ("human_body", 95.0, "Notify SAR authority immediately"),
        ("container", 60.0, "Navigation hazard — report to port authority"),
        ("wreck", 90.0, "Navigation hazard — report to port authority"),
        ("container", 30.0, "Log and monitor"),
        ("mine_like", 10.0, "Do NOT approach — notify naval EOD"),
        ("mine_like", 99.0, "Do NOT approach — notify naval EOD"),
        ("tire", 90.0, "Log and monitor"),
    ],
)
def test_recommended_action_rules(cls: str, severity: float, expected: str) -> None:
    assert recommended_action_for(cls, severity) == expected


def test_built_contacts_carry_triage_fields(processed) -> None:
    _, pre, contacts = processed
    assert contacts, "stub survey must yield reportable contacts"
    expected_accuracy = round(
        position_accuracy(
            pre.ground.ground_res,
            layback_known=True,  # synthetic nav records an explicit (zero) layback
            nav_uncertainty_m=nav_fix_uncertainty_m(),
        ),
        2,
    )
    for c in contacts:
        assert c.priority == priority_for(c.severity)
        assert c.recommended_action == recommended_action_for(c.cls, c.severity)
        assert c.position_accuracy_m == pytest.approx(expected_accuracy)


# ------------------------------------------------------- position accuracy ----


@pytest.mark.parametrize("ground_res", [0.05, 0.2, 1.0])
def test_position_accuracy_positive_and_scales_with_ground_res(ground_res: float) -> None:
    acc = position_accuracy(ground_res, layback_known=True, nav_uncertainty_m=2.0)
    assert acc == pytest.approx(2.0 * ground_res + 2.0)
    assert acc > 0
    # Halving the pixel size must shrink the budget by exactly one pixel worth.
    finer = position_accuracy(ground_res / 2, layback_known=True, nav_uncertainty_m=2.0)
    assert acc - finer == pytest.approx(ground_res)


def test_position_accuracy_layback_term() -> None:
    known = position_accuracy(0.1, True, nav_uncertainty_m=2.0, layback_uncertainty_m=5.0)
    unknown = position_accuracy(0.1, False, nav_uncertainty_m=2.0, layback_uncertainty_m=5.0)
    assert unknown - known == pytest.approx(5.0)


def test_nav_uncertainty_config_key(tmp_path) -> None:
    assert nav_fix_uncertainty_m() == pytest.approx(2.0)  # shipped default
    assert nav_fix_uncertainty_m(tmp_path / "missing.yaml") == pytest.approx(2.0)
    custom = tmp_path / "geoscribe.yaml"
    custom.write_text("position_accuracy: {nav_fix_uncertainty_m: 3.5}\n", encoding="utf-8")
    assert nav_fix_uncertainty_m(custom) == pytest.approx(3.5)


# ----------------------------------------------------------- summary block ----


def test_summary_block_arithmetic(processed, tmp_path) -> None:
    cfg, pre, contacts = processed
    stats = survey_stats(pre)
    paths = write_all(contacts, tmp_path, survey="schema_test.xtf", survey_stats=stats)
    summary = json.loads(paths["json"].read_text(encoding="utf-8"))["summary"]

    assert summary["total_detections"] == len(contacts) > 0
    assert summary["high_confidence"] == sum(1 for c in contacts if c.confidence >= 70.0)
    area = summary["area_surveyed_sqkm"]
    assert area == stats["area_surveyed_sqkm"] > 0
    assert summary["debris_density_per_sqkm"] == pytest.approx(len(contacts) / area, abs=0.01)
    sonar = summary["sonar_config"]
    assert sonar["n_pings"] == cfg.n_pings
    assert sonar["range_m"] == pytest.approx(cfg.slant_range, abs=0.5)
    assert sonar["altitude_m"] == pytest.approx(cfg.altitude, abs=1.0)


def test_survey_stats_geometry(processed) -> None:
    cfg, pre, _ = processed
    stats = survey_stats(pre)
    expected_track_km = (cfg.n_pings - 1) * cfg.speed * cfg.ping_interval / 1e3
    assert stats["track_length_km"] == pytest.approx(expected_track_km, rel=0.05)
    # Usable swath cannot exceed the full image width across both sides.
    max_swath = (
        pre.ground.n_cols("port") + pre.ground.n_cols("starboard")
    ) * pre.ground.ground_res
    assert 0 < stats["swath_width_m"] <= max_swath
    assert stats["area_surveyed_sqkm"] == pytest.approx(
        stats["track_length_km"] * 1e3 * stats["swath_width_m"] / 1e6, abs=5e-4
    )


def test_summary_without_stats_keeps_counts_and_nulls_coverage(processed, tmp_path) -> None:
    _, _, contacts = processed
    paths = write_all(contacts, tmp_path, survey="schema_test.xtf")
    summary = json.loads(paths["json"].read_text(encoding="utf-8"))["summary"]
    assert summary["total_detections"] == len(contacts)
    assert summary["area_surveyed_sqkm"] is None
    assert summary["debris_density_per_sqkm"] is None
    assert summary["sonar_config"] is None


# ------------------------------------------------- compatibility / formats ----

OLD_CONTACT = {  # a stored record from before the section-5.3 fields existed
    "id": "SN-20260101-0001",
    "cls": "tire",
    "confidence": 55.0,
    "lat": 13.0,
    "lon": 80.3,
    "dims": {"length_m": 1.0, "width_m": 1.0, "height_m": None},
    "physics": {"highlight": True, "shadow": False},
    "severity": 30.0,
    "pixel": {"side": "port", "ping0": 10, "ping1": 20, "col0": 5, "col1": 15},
}


def test_old_contact_json_still_validates() -> None:
    c = Contact.model_validate(OLD_CONTACT)
    assert c.priority == "LOW"
    assert c.recommended_action is None
    assert c.position_accuracy_m == 0.0
    dumped = c.model_dump(mode="json")
    assert dumped["ping_range"] == [10, 20]  # convenience alias in the dump
    assert Contact.model_validate(dumped).id == c.id  # new dump round-trips too


def test_csv_gains_triage_columns(processed, tmp_path) -> None:
    _, _, contacts = processed
    path = write_contacts_csv(contacts, tmp_path / "contacts.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in ("priority", "recommended_action", "position_accuracy_m"):
        assert column in header


def test_schema_documents_new_fields() -> None:
    schema = contacts_json_schema()
    assert "summary" in schema["properties"]
    assert "summary" not in schema["required"]  # old documents must validate
    item = schema["properties"]["contacts"]["items"]
    for name in ("priority", "recommended_action", "position_accuracy_m", "ping_range"):
        assert name in item["properties"]
