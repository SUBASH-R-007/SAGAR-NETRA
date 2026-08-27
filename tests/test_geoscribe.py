"""GeoScribe: geotagging geometry, severity index, contact building, and all
five report formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from geoscribe.build import build_contacts
from geoscribe.contact import Contact
from geoscribe.geotag import offset_position, pixel_to_wgs84, towfish_position
from geoscribe.report import severity_band, write_all
from geoscribe.severity import load_layers, severity_score
from physicheck.verify import verify_detections
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FakeDetection:
    side: str
    ping0: int
    ping1: int
    col0: int
    col1: int
    cls: str
    score: float
    brain: str = "A"
    tile_index: int = -1


# ---------------------------------------------------------------- geotag ----


def test_offset_position_cardinal_directions() -> None:
    lat, lon = 13.0, 80.35
    # Heading north: starboard is due east, port due west.
    lat_s, lon_s = offset_position(lat, lon, 0.0, "starboard", 100.0)
    lat_p, lon_p = offset_position(lat, lon, 0.0, "port", 100.0)
    assert lon_s > lon and lon_p < lon
    assert abs(lat_s - lat) < 1e-5 and abs(lat_p - lat) < 1e-5
    # 100 m ~ 0.00092 deg of longitude at 13 N.
    assert (lon_s - lon) * 111_320 * np.cos(np.deg2rad(lat)) == pytest.approx(100.0, rel=0.01)


def test_towfish_layback_moves_astern() -> None:
    lat, lon = 13.0, 80.35
    lat2, lon2 = towfish_position(lat, lon, 90.0, 50.0)  # heading east -> astern is west
    assert lon2 < lon and abs(lat2 - lat) < 1e-6


def test_pixel_to_wgs84_matches_seeded_target(container_pre) -> None:
    """The geotagged position of the seeded container must sit at the known
    across-track offset from the towfish track."""
    pre, target, box = container_pre
    gi = pre.ground
    ping_c = (box.ping0 + box.ping1) // 2
    lat, lon = pixel_to_wgs84(gi, "starboard", ping_c, (box.col0 + box.col1) / 2)
    rec = gi.nav[ping_c]
    # Heading 90 (east): starboard offset is due south.
    assert lat < rec["lat"]
    offset_m = (rec["lat"] - lat) * 111_320
    assert offset_m == pytest.approx(target.ground_range, rel=0.06)


# -------------------------------------------------------------- severity ----


def test_severity_class_hazard_ordering() -> None:
    common = dict(area_m2=4.0, height_m=0.5, depth_m=30.0, lat=13.0, lon=80.35)
    net, _ = severity_score("ghost_net", **common)
    tire, _ = severity_score("tire", **common)
    assert net > tire


def test_severity_layer_proximity() -> None:
    layers = load_layers(REPO / "data" / "layers")
    assert len(layers) == 3
    common = dict(cls="ghost_net", area_m2=4.0, height_m=0.5, depth_m=30.0)
    inside, bd_in = severity_score(
        **common, lat=13.05, lon=80.35, layers=layers
    )  # inside/near demo zones
    far, bd_far = severity_score(**common, lat=12.0, lon=81.5, layers=layers)
    assert inside > far
    assert bd_in.proximity > 0.5 and bd_far.proximity < 0.01
    assert bd_in.nearest_layer is not None


def test_severity_bounds() -> None:
    score, _ = severity_score(
        "ghost_net", area_m2=1e6, height_m=50.0, depth_m=1.0, lat=13.05, lon=80.35,
        layers=load_layers(REPO / "data" / "layers"),
    )
    assert 0.0 <= score <= 100.0


# ------------------------------------------------------- contacts/report ----


@pytest.fixture(scope="module")
def container_pre():
    cfg = SceneConfig(n_pings=80, n_samples=1024, slant_range=40.0, seed=3)
    target = SynthTarget(
        "container", "starboard", 40, 20.0, 6.0, 2.0, height=2.0, reflectivity=6.0
    )
    pa, _ = make_scene(cfg, [target])
    pre = preprocess(pa)
    gi = pre.ground_raw
    half_pings = int(target.length / (2 * cfg.speed * cfg.ping_interval))
    box = FakeDetection(
        side="starboard",
        ping0=target.ping - half_pings,
        ping1=target.ping + half_pings,
        col0=int(gi.col_of_ground_range(target.ground_range - target.width / 2)),
        col1=int(gi.col_of_ground_range(target.ground_range + target.width / 2)),
        cls="container",
        score=0.85,
    )
    return pre, target, box


@pytest.fixture(scope="module")
def contacts(container_pre, tmp_path_factory):
    pre, target, box = container_pre
    rock = FakeDetection(box.side, 10, 20, box.col0, box.col1, "rock_cluster", 0.9)
    verified = verify_detections([box, rock], pre)
    out = tmp_path_factory.mktemp("evidence")
    return build_contacts(
        verified,
        pre,
        survey="survey_test.xtf",
        layers=load_layers(REPO / "data" / "layers"),
        evidence_dir=out,
    )


def test_contacts_complete_and_hard_negatives_dropped(contacts, container_pre) -> None:
    pre, target, _ = container_pre
    assert len(contacts) == 1, "rock_cluster must never be reported"
    c = contacts[0]
    assert c.cls == "container"
    assert c.id.startswith("SN-2026")
    assert 0 <= c.confidence <= 100 and 0 <= c.severity <= 100
    assert c.dims.height_m == pytest.approx(target.height, rel=0.3)
    assert c.dims.width_m == pytest.approx(target.width, rel=0.3)
    assert c.dims.length_m == pytest.approx(target.length, rel=0.4)
    assert c.depth_m == pytest.approx(22.0 + 8.0, abs=1.5)
    assert c.detected_at is not None and c.detected_at.startswith("2026-01-01")
    assert len(c.corners) == 4
    assert c.evidence_png is not None and Path(c.evidence_png).exists()
    assert c.thumbnail_png is not None and Path(c.thumbnail_png).exists()


def test_all_report_formats(contacts, tmp_path) -> None:
    paths = write_all(contacts, tmp_path, survey="survey_test.xtf")
    assert set(paths) == {"json", "schema", "csv", "geojson", "kml", "pdf"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0

    # JSON round-trips through the pydantic contract.
    doc = json.loads(paths["json"].read_text())
    assert doc["survey"] == "survey_test.xtf"
    restored = [Contact.model_validate(c) for c in doc["contacts"]]
    assert restored[0].id == contacts[0].id

    # Schema documents the contract.
    schema = json.loads(paths["schema"].read_text())
    assert schema["title"].startswith("SAGAR-NETRA")

    # GeoJSON parses and places the point at the contact position.
    geo = json.loads(paths["geojson"].read_text())
    assert geo["features"][0]["geometry"]["coordinates"] == [
        contacts[0].lon,
        contacts[0].lat,
    ]

    # KML is well-formed XML containing the contact id.
    import xml.etree.ElementTree as ET

    root = ET.parse(paths["kml"]).getroot()
    assert contacts[0].id in ET.tostring(root, encoding="unicode")

    # CSV has a header plus one row.
    lines = paths["csv"].read_text().strip().splitlines()
    assert len(lines) == 2 and lines[1].startswith(contacts[0].id)


def test_severity_bands() -> None:
    assert severity_band(90)[0] == "critical"
    assert severity_band(60)[0] == "high"
    assert severity_band(30)[0] == "medium"
    assert severity_band(5)[0] == "low"
