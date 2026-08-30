"""Uploading a sonar image: declared geometry in, measured contacts out.

An image records no navigation, so the geometry a survey log carries per ping
has to be stated by the operator. These tests pin that contract — what is
required, what is rejected, and that a picture of the seabed yields the same
physics the log path yields.
"""

from __future__ import annotations

import time
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from api.db import ContactRepo
from api.main import create_app
from sonar_core.parsers.base import load
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.waterfall import save_waterfall_png

SCENE_RANGE_M = 40.0
SCENE_ALTITUDE_M = 8.0
GEOMETRY = {
    "altitude_m": SCENE_ALTITUDE_M,
    "range_m": SCENE_RANGE_M,
    "lat": 13.05,
    "lon": 80.35,
    "heading_deg": 90,
}


@pytest.fixture(scope="module")
def waterfall_png(tmp_path_factory):
    """A display waterfall PNG — what an operator would actually have.

    Rendered at survey resolution so acoustic shadows span enough samples for
    the height estimator to work, exactly as they would in a real capture.
    """
    from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene

    cfg = SceneConfig(
        n_pings=600, n_samples=1024, slant_range=SCENE_RANGE_M,
        altitude=SCENE_ALTITUDE_M, seed=31,
    )
    targets = [
        SynthTarget("container", "starboard", 150, 20.0, 6.0, 2.4, height=2.4,
                    reflectivity=6.0),
        SynthTarget("wreck", "port", 380, 24.0, 20.0, 5.0, height=3.5, reflectivity=5.5),
    ]
    pa, _ = make_scene(cfg, targets)
    return save_waterfall_png(pa, tmp_path_factory.mktemp("img") / "sonar_capture.png")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("imgapi")
    app = create_app(
        repo=ContactRepo(root / "c.db"),
        upload_dir=root / "up",
        output_root=root / "out",
    )
    with TestClient(app) as c:
        yield c


def _upload(client, png, **data):
    with png.open("rb") as fh:
        return client.post(
            "/api/upload", files={"file": (png.name, fh, "image/png")}, data=data
        )


def test_image_without_geometry_is_refused_with_a_reason(client, waterfall_png) -> None:
    r = _upload(client, waterfall_png)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "altitude_m" in detail and "range_m" in detail


def test_range_must_exceed_altitude(client, waterfall_png) -> None:
    """A swath only exists beyond the first bottom return."""
    r = _upload(client, waterfall_png, altitude_m=50, range_m=10)
    assert r.status_code == 422
    assert "must exceed" in r.json()["detail"]

    assert _upload(client, waterfall_png, altitude_m=-1, range_m=50).status_code == 422


def test_survey_log_needs_no_geometry(client, sample_xtf) -> None:
    """The requirement is scoped to nav-less formats; XTF carries its own."""
    with sample_xtf.open("rb") as fh:
        r = client.post(
            "/api/upload", files={"file": (sample_xtf.name, fh, "application/octet-stream")}
        )
    assert r.status_code == 200


def test_declared_geometry_yields_geotagged_measured_contacts(
    client, waterfall_png
) -> None:
    r = _upload(client, waterfall_png, **GEOMETRY)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    deadline = time.time() + 240
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")

    contacts = client.get(
        "/api/contacts", params={"survey": job["survey"], "limit": 500}
    ).json()["contacts"]
    assert contacts, "an image with geometry must yield contacts"

    # Geotagged onto the declared line, not left at NaN.
    assert all(abs(c["lat"] - GEOMETRY["lat"]) < 0.5 for c in contacts)
    assert all(abs(c["lon"] - GEOMETRY["lon"]) < 0.5 for c in contacts)
    # Height from shadow works, so declared altitude reached the physics stage.
    assert any(c["dims"]["height_m"] is not None for c in contacts)

    # And the result is readable: skipping the double gain normalisation is
    # what keeps this from being a 90-contact open-set anomaly flood.
    anomalies = Counter(c["cls"] for c in contacts)["unknown_anomaly"]
    assert anomalies <= len(contacts) / 2, (
        f"{anomalies}/{len(contacts)} contacts are open-set anomalies — the image "
        "is being normalised twice again"
    )


def test_display_image_is_not_gain_normalized_twice(waterfall_png, small_scene) -> None:
    """The regression behind a 92-contact false-positive flood.

    A waterfall written for display has already had its range falloff
    flattened. Running EGN over it again invents range structure that the
    open-set anomaly brain reads as debris.
    """
    pa = load(waterfall_png, altitude_m=SCENE_ALTITUDE_M, slant_range_m=SCENE_RANGE_M)
    assert pa.meta["gain_normalized"] is True

    auto = preprocess(pa)
    assert "egn" not in auto.timings, "EGN must not run on a pre-normalised image"

    forced = preprocess(pa, config={"egn": {"enabled": True}})
    assert "egn" in forced.timings, "an explicit caller override must still win"


def test_out_of_range_coordinates_are_refused(client, waterfall_png) -> None:
    """A typo like lat=91 or lon=999 must fail loudly at upload, not geotag
    contacts into an impossible ocean. (Found the hard way: an upload with
    swapped fields put 52 contacts in the Arctic, and the map rendered an
    empty Chennai coast.)"""
    r = _upload(client, waterfall_png, **{**GEOMETRY, "lat": 91.0})
    assert r.status_code == 422 and "lat" in r.json()["detail"]
    r = _upload(client, waterfall_png, **{**GEOMETRY, "lon": -190.0})
    assert r.status_code == 422 and "lon" in r.json()["detail"]


def test_half_supplied_position_is_refused(client, waterfall_png) -> None:
    """lat without lon (or vice versa) is always a form mistake."""
    geom = {k: v for k, v in GEOMETRY.items() if k != "lon"}
    r = _upload(client, waterfall_png, **geom)
    assert r.status_code == 422 and "together" in r.json()["detail"]


def test_survey_delete_removes_contacts_and_404s_unknown(client, waterfall_png) -> None:
    """Console hygiene: a mistaken upload must be removable without SQLite
    surgery, and deleting nonsense must say so."""
    name = "delete_me.png"
    with waterfall_png.open("rb") as fh:
        r = client.post("/api/upload", files={"file": (name, fh, "image/png")},
                        data=GEOMETRY)
    assert r.status_code == 200
    job = r.json()["job_id"]
    import time as _t
    for _ in range(600):
        snap = client.get(f"/api/jobs/{job}").json()
        if snap["status"] in ("done", "failed"):
            break
        _t.sleep(0.2)
    assert snap["status"] == "done"

    before = client.get(f"/api/contacts?survey={name}").json()["contacts"]
    assert before, "the fixture upload must produce contacts to delete"

    r = client.delete(f"/api/surveys/{name}")
    assert r.status_code == 200
    assert r.json()["contacts_removed"] == len(before)
    assert client.get(f"/api/contacts?survey={name}").json()["contacts"] == []
    assert name not in [s["name"] for s in client.get("/api/surveys").json()]

    assert client.delete(f"/api/surveys/{name}").status_code == 404
