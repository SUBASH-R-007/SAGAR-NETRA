"""Disaster-mode mission profiles (blueprint N-12): YAML loading, hazard
re-ranking (SAR puts a possible victim above big loud debris), and the API
surface (/api/missions + upload mission field)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.db import ContactRepo
from api.main import create_app
from geoscribe.severity import DEFAULT_HAZARD, list_missions, load_mission, severity_score
from tridentnet.detector import Detection

MISSION_NAMES = {"aircraft_search", "ghost_net_cleanup", "port_clearance", "sar"}


class TestProfiles:
    def test_all_profiles_load(self) -> None:
        for name in MISSION_NAMES:
            profile = load_mission(name)
            assert profile["name"] == name
            assert profile["description"]
            assert profile["reportable_extra_note"]
            assert set(DEFAULT_HAZARD) <= set(profile["hazard_table"])
            assert all(0.0 <= w <= 1.0 for w in profile["hazard_table"].values())

    def test_listing_matches_files(self) -> None:
        listed = list_missions()
        assert MISSION_NAMES <= {m["name"] for m in listed}
        assert all(m["description"] for m in listed)

    def test_unknown_mission_raises_with_available(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            load_mission("bogus")
        message = str(excinfo.value)
        assert "bogus" in message
        for name in MISSION_NAMES:
            assert name in message

    def test_sar_is_recall_first(self) -> None:
        sar = load_mission("sar")
        assert sar["detector_conf"] is not None and sar["detector_conf"] < 0.25
        assert sar["hazard_table"]["human_body"] == 1.0
        assert sar["hazard_table"]["tire"] < DEFAULT_HAZARD["tire"]

    def test_unmentioned_classes_keep_defaults(self) -> None:
        table = load_mission("port_clearance")["hazard_table"]
        assert table["human_body"] == DEFAULT_HAZARD["human_body"]  # never demoted


class TestSarRanking:
    """A big, shallow tire outranks a small deep body under default hazards;
    the SAR profile must flip that ordering."""

    TIRE = dict(cls="tire", area_m2=150.0, height_m=2.5, depth_m=8.0, lat=13.0, lon=80.3)
    BODY = dict(cls="human_body", area_m2=0.5, height_m=None, depth_m=150.0, lat=13.0, lon=80.3)

    def test_default_hazards_rank_tire_first(self) -> None:
        tire, _ = severity_score(**self.TIRE)
        body, _ = severity_score(**self.BODY)
        assert tire > body

    def test_sar_ranks_body_first(self) -> None:
        table = load_mission("sar")["hazard_table"]
        tire, tire_bd = severity_score(**self.TIRE, hazard_table=table)
        body, body_bd = severity_score(**self.BODY, hazard_table=table)
        assert body > tire
        assert body_bd.hazard == 1.0
        assert tire_bd.hazard == pytest.approx(table["tire"])


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
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("missions_api")
    app = create_app(
        repo=ContactRepo(root / "contacts.db"),
        upload_dir=root / "uploads",
        output_root=root / "outputs",
        detector_factory=StubDetector,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def tiny_xtf(tmp_path_factory):
    """A very short synthetic survey — CPU stays free for the trainings."""
    from sonar_core.parsers.xtf_writer import write_xtf
    from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene

    cfg = SceneConfig(n_pings=160, n_samples=256, slant_range=40.0, seed=17)
    targets = [
        SynthTarget("container", "starboard", 80, 20.0, 3.0, 2.0, 1.5, reflectivity=5.5)
    ]
    pa, _ = make_scene(cfg, targets)
    out = tmp_path_factory.mktemp("mission_survey")
    return write_xtf(pa, out / "survey_mission.xtf")


def test_missions_endpoint(client) -> None:
    listed = client.get("/api/missions").json()
    assert MISSION_NAMES <= {m["name"] for m in listed}
    assert all(m["description"] for m in listed)


def test_upload_unknown_mission_is_422(client) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("x.xtf", b"\0", "application/octet-stream")},
        data={"mission": "bogus"},
    )
    assert response.status_code == 422
    assert "sar" in response.json()["detail"]


def test_upload_with_sar_mission_processes(client, tiny_xtf) -> None:
    with tiny_xtf.open("rb") as fh:
        response = client.post(
            "/api/upload",
            files={"file": (tiny_xtf.name, fh, "application/octet-stream")},
            data={"mission": "sar"},
        )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", f"job failed: {job.get('error')}"
    assert job["n_contacts"] >= 1
    # the console's ingest ledger names the profile a survey ran under, so the
    # snapshot has to carry it — a reload must not silently drop the provenance
    assert job["mission"] == "sar"
