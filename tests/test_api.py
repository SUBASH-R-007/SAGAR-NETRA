"""DRISHTI Console API: upload -> job -> contacts -> review -> reports flow,
with a stub detector so no trained weights are needed."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.db import ContactRepo
from api.main import create_app
from tridentnet.detector import Detection


class StubDetector:
    """Deterministic detections: one reportable box on each side's mid-swath,
    plus a hard negative that must never surface as a contact."""

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
        first = next(iter(by_side.values()))
        detections.append(
            Detection(
                side=first.side, ping0=first.row0 + 2, ping1=first.row0 + 10,
                col0=first.col0 + 2, col1=first.col0 + 20,
                cls="rock_cluster", score=0.95, tile_index=first.index,
            )
        )
        if progress:
            progress("detect", 1.0)
        return detections


@pytest.fixture(scope="module")
def client(tmp_path_factory, sample_xtf):
    root = tmp_path_factory.mktemp("api")
    app = create_app(
        require_auth=False,  # exercise the pipeline, not the RBAC layer (tests/test_auth.py)
        repo=ContactRepo(root / "contacts.db"),
        upload_dir=root / "uploads",
        output_root=root / "outputs",
        detector_factory=StubDetector,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def processed_survey(client, sample_xtf) -> str:
    with sample_xtf.open("rb") as fh:
        response = client.post(
            "/api/upload", files={"file": (sample_xtf.name, fh, "application/octet-stream")}
        )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    deadline = time.time() + 180
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", f"job failed: {job.get('error')}"
    assert job["n_contacts"] >= 1
    return job["survey"]


def test_health(client) -> None:
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_rejects_unknown_type(client) -> None:
    response = client.post("/api/upload", files={"file": ("evil.exe", b"MZ", "app/exe")})
    assert response.status_code == 415


def test_contacts_and_filters(client, processed_survey) -> None:
    contacts = client.get("/api/contacts", params={"survey": processed_survey}).json()[
        "contacts"
    ]
    assert len(contacts) >= 1
    assert all(c["cls"] != "rock_cluster" for c in contacts), "hard negatives leaked"
    assert all(0 <= c["severity"] <= 100 for c in contacts)

    one = contacts[0]
    detail = client.get(f"/api/contacts/{one['id']}").json()
    assert detail["id"] == one["id"]
    assert detail["pixel"]["side"] in ("port", "starboard")

    none = client.get(
        "/api/contacts", params={"survey": processed_survey, "min_conf": 101}
    ).json()["contacts"]
    assert none == []


def test_review_flow_and_export(client, processed_survey) -> None:
    contact = client.get("/api/contacts").json()["contacts"][0]
    response = client.post(
        f"/api/contacts/{contact['id']}/review",
        json={"status": "confirmed", "notes": "verified on replay"},
    )
    assert response.status_code == 200
    assert response.json()["review"] == "confirmed"

    log = client.get("/api/reviews/export").json()
    assert any(
        entry["contact_id"] == contact["id"] and entry["status"] == "confirmed"
        for entry in log
    )


def test_all_report_formats_served(client, processed_survey) -> None:
    for fmt in ("json", "csv", "geojson", "kml", "pdf"):
        response = client.get(f"/api/report/{fmt}", params={"survey": processed_survey})
        assert response.status_code == 200, f"{fmt} report missing"
        assert len(response.content) > 0


def test_waterfall_and_meta(client, processed_survey) -> None:
    meta = client.get(f"/api/waterfall/{processed_survey}/meta").json()
    assert meta["n_pings"] == 600
    assert meta["n_port_cols"] > 0 and meta["n_stbd_cols"] > 0
    image = client.get(f"/api/waterfall/{processed_survey}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"


def test_survey_summary(client, processed_survey) -> None:
    """The overview dashboard reads the report's own summary block, so the two
    can never disagree about area surveyed or debris density."""
    summary = client.get("/api/summary", params={"survey": processed_survey}).json()
    assert summary["survey"] == processed_survey
    assert summary["total_detections"] >= 1
    assert summary["area_surveyed_sqkm"] > 0
    assert summary["debris_density_per_sqkm"] > 0
    assert summary["sonar_config"]["n_pings"] == 600
    assert summary["pipeline_version"]

    missing = client.get("/api/summary", params={"survey": "no-such-survey.xtf"})
    assert missing.status_code == 404


def test_evidence_and_thumb_served(client, processed_survey) -> None:
    contact = client.get("/api/contacts").json()["contacts"][0]
    for endpoint in ("evidence", "thumb"):
        response = client.get(f"/api/contacts/{contact['id']}/{endpoint}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


def test_copilot_rules_mode(client, processed_survey) -> None:
    response = client.post("/api/copilot", json={"question": "How many containers?"}).json()
    assert response["mode"] == "rules"
    assert "container contacts" in response["answer"]
    top = client.post(
        "/api/copilot", json={"question": "top 3 most severe contacts"}
    ).json()
    assert top["rows"] and len(top["rows"]) <= 3


def test_diff_survey_with_itself(client, processed_survey) -> None:
    diff = client.get(
        "/api/diff",
        params={"survey_a": processed_survey, "survey_b": processed_survey},
    ).json()
    assert diff["new_contacts"] == []
    assert len(diff["matched"]) == diff["n_b"]


def test_ws_progress_replays_final_state(client, processed_survey) -> None:
    jobs = client.get("/api/jobs").json()
    job_id = jobs[0]["id"]
    with client.websocket_connect(f"/api/jobs/{job_id}/progress") as ws:
        snapshot = ws.receive_json()
    assert snapshot["status"] == "done"
    assert snapshot["fraction"] == 1.0


def test_layers_endpoint(client) -> None:
    layers = client.get("/api/layers").json()
    assert set(layers) == {"marine_protected_area", "shipping_lane", "turtle_nesting_zone"}
    assert layers["shipping_lane"]["type"] == "FeatureCollection"
