"""Real-time streaming sink (api.realtime): ordered window/contact events,
batch-vs-stream geodesic parity, overlap dedup, the recovery-status workflow,
and the edge-telemetry health shape — all with stub detectors, no weights."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.db import ContactRepo
from api.diff import geodesic_m
from api.main import create_app
from api.processing import process_survey
from api.realtime import _window_starts, stream_survey
from geoscribe.contact import Contact, RecoveryStatus
from tridentnet.detector import Detection

N_PINGS = 300
WINDOW_PINGS = 200
OVERLAP_PINGS = 40  # -> window starts [0, 160]; pings 160..199 seen twice

#: (side, global ping row, ground col0, class). Ping 180 sits inside the
#: [160, 200) overlap of the two stream windows, so it is detected in BOTH
#: windows and must be collapsed by the geodesic dedup; the other two targets
#: each belong to exactly one window.
TARGETS: list[tuple[str, int, int, str]] = [
    ("starboard", 80, 400, "container"),
    ("port", 180, 350, "ghost_net"),
    ("starboard", 250, 420, "cylinder_drum"),
]


class GlobalTargetDetector:
    """StubDetector variant that fires boxes at fixed *global* ping rows.

    ``stream_survey`` runs its windows in schedule order through one detector
    instance, so counting ``detect_tiles`` calls recovers each window's start
    ping; batch mode is the one-call schedule where the start is always 0.
    A target whose box falls outside the current window is withheld — exactly
    what a live detector would (not) see. Detections are emitted in the
    window-local ping frame, as a real detector on window tiles would.
    """

    HALF_PINGS = 6  # box half-height: 13 pings, safely past the thin-ping gate
    COL_SPAN = 12

    def __init__(self, window_starts: tuple[int, ...] = (0,)) -> None:
        self.starts = list(window_starts)
        self.calls = 0

    def detect_tiles(self, tiles: list, progress=None) -> list[Detection]:
        start = self.starts[min(self.calls, len(self.starts) - 1)]
        self.calls += 1
        n_rows = max((t.row0 + t.image.shape[0] for t in tiles), default=0)
        detections = []
        for side, ping, col0, cls in TARGETS:
            row = ping - start
            if row - self.HALF_PINGS < 0 or row + self.HALF_PINGS >= n_rows:
                continue  # outside this window
            detections.append(
                Detection(
                    side=side, ping0=row - self.HALF_PINGS, ping1=row + self.HALF_PINGS,
                    col0=col0, col1=col0 + self.COL_SPAN, cls=cls, score=0.8,
                )
            )
        if progress:
            progress("detect", 1.0)
        return detections


def _stream_factory() -> GlobalTargetDetector:
    return GlobalTargetDetector(_window_starts(N_PINGS, WINDOW_PINGS, OVERLAP_PINGS))


@pytest.fixture(scope="module")
def survey_file(tmp_path_factory: pytest.TempPathFactory):
    from sonar_core.synth.sample import make_sample

    return make_sample(tmp_path_factory.mktemp("stream_sample"), n_pings=N_PINGS)


@pytest.fixture(scope="module")
def streamed(survey_file, tmp_path_factory: pytest.TempPathFactory):
    """One shared stream_survey run: (repo, ordered events, summary)."""
    repo = ContactRepo()
    events: list[dict] = []
    summary = stream_survey(
        survey_file, repo, emit=events.append,
        window_pings=WINDOW_PINGS, overlap_pings=OVERLAP_PINGS,
        detector_factory=_stream_factory,
        output_root=tmp_path_factory.mktemp("stream_out"),
    )
    return repo, events, summary


# ------------------------------------------------------------ (1) events --


def test_stream_events_ordered_and_contacts_stored(streamed) -> None:
    repo, events, summary = streamed
    windows = [e for e in events if e["type"] == "window"]
    assert len(windows) == summary["n_windows"] == 2

    done = [w["done_pings"] for w in windows]
    assert done == sorted(done), "window progress must be monotonic"
    assert done[-1] == N_PINGS == summary["n_pings"]
    covered: set[int] = set()
    for w in windows:
        covered.update(range(w["start"], w["stop"]))
    assert covered == set(range(N_PINGS)), "windows must cover every ping"

    contact_events = [e for e in events if e["type"] == "contact"]
    stored = {c.id for c in repo.query(survey=summary["survey"], limit=100)}
    assert {e["contact"]["id"] for e in contact_events} == stored
    assert summary["n_contacts"] == len(stored) == len(TARGETS)

    # ping offsets were added back: the window-1-only target reports its
    # GLOBAL ping rows, not window-local ones.
    drum = next(
        c for c in repo.query(survey=summary["survey"], limit=100)
        if c.cls == "cylinder_drum"
    )
    assert drum.pixel.ping0 == 250 - GlobalTargetDetector.HALF_PINGS
    assert drum.pixel.ping1 == 250 + GlobalTargetDetector.HALF_PINGS


# ---------------------------------------------- (2) batch/stream parity --


def test_stream_matches_batch_positions(survey_file, tmp_path_factory) -> None:
    batch_repo, stream_repo = ContactRepo(), ContactRepo()
    batch = process_survey(
        survey_file, batch_repo,
        detector_factory=GlobalTargetDetector,
        output_root=tmp_path_factory.mktemp("batch_out"),
    )
    stream = stream_survey(
        survey_file, stream_repo, emit=lambda _e: None,
        window_pings=WINDOW_PINGS, overlap_pings=OVERLAP_PINGS,
        detector_factory=_stream_factory,
        output_root=tmp_path_factory.mktemp("stream_out2"),
    )

    a = batch_repo.query(survey=batch["survey"], limit=100)
    b = stream_repo.query(survey=stream["survey"], limit=100)
    assert len(a) == len(b) == len(TARGETS)
    for contact in a:
        twin = min(
            (s for s in b if s.cls == contact.cls),
            key=lambda s: geodesic_m(contact.lat, contact.lon, s.lat, s.lon),
        )
        dist = geodesic_m(contact.lat, contact.lon, twin.lat, twin.lon)
        assert dist <= 10.0, f"{contact.cls}: batch/stream positions {dist:.1f} m apart"
        # nav was sliced with the pings, so the global ping frame is identical
        assert twin.pixel.ping0 == contact.pixel.ping0
        assert twin.pixel.side == contact.pixel.side


# ------------------------------------------------------ (3) overlap dedup --


def test_overlap_dedup_keeps_first_seen(streamed) -> None:
    repo, events, summary = streamed
    contacts = repo.query(survey=summary["survey"], limit=100)
    for i, a in enumerate(contacts):
        for b in contacts[i + 1:]:
            if a.cls == b.cls:
                dist = geodesic_m(a.lat, a.lon, b.lat, b.lon)
                assert dist > 5.0, f"duplicate {a.cls} contacts {dist:.1f} m apart"
    # the overlap target fired in both windows but was stored exactly once,
    # under its first-seen (window 0) global ping rows
    nets = [c for c in contacts if c.cls == "ghost_net"]
    assert len(nets) == 1
    assert nets[0].pixel.ping0 == 180 - GlobalTargetDetector.HALF_PINGS


class ClippingTargetDetector:
    """Emits the VISIBLE PART of each target — clipped at window edges, as a
    real detector on window tiles would — for the boundary-straddle case."""

    def __init__(self, targets, window_starts=(0,), half_pings=6, min_visible=3):
        self.targets = targets
        self.starts = list(window_starts)
        self.half = half_pings
        self.min_visible = min_visible
        self.calls = 0

    def detect_tiles(self, tiles: list, progress=None) -> list[Detection]:
        start = self.starts[min(self.calls, len(self.starts) - 1)]
        self.calls += 1
        n_rows = max((t.row0 + t.image.shape[0] for t in tiles), default=0)
        detections = []
        for side, ping, col0, cls in self.targets:
            row = ping - start
            r0 = max(row - self.half, 0)
            r1 = min(row + self.half, n_rows - 1)
            if r1 - r0 + 1 < self.min_visible or r1 < 0 or r0 >= n_rows:
                continue
            detections.append(
                Detection(side=side, ping0=r0, ping1=r1, col0=col0,
                          col1=col0 + 12, cls=cls, score=0.8)
            )
        return detections


def test_boundary_clipped_fragment_replaced_by_full_redetection(
    survey_file, tmp_path_factory
) -> None:
    """A target straddling the window boundary is stored clipped from window 0,
    then REPLACED (same id) by its whole re-detection in the overlap window —
    the fragment must never win (reviewer probe)."""
    targets = [("starboard", 195, 400, "wreck")]  # rows 189..201; window 0 ends at 200
    starts = _window_starts(N_PINGS, WINDOW_PINGS, OVERLAP_PINGS)
    repo = ContactRepo()
    events: list[dict] = []
    stream_survey(
        survey_file, repo, emit=events.append,
        window_pings=WINDOW_PINGS, overlap_pings=OVERLAP_PINGS,
        detector_factory=lambda: ClippingTargetDetector(targets, starts),
        output_root=tmp_path_factory.mktemp("clip_out"),
    )
    contacts = repo.query(limit=10)
    assert len(contacts) == 1, "one physical object, one stored contact"
    stored = contacts[0]
    assert stored.pixel.ping0 == 189 and stored.pixel.ping1 == 201, (
        f"stored extent {stored.pixel.ping0}-{stored.pixel.ping1} is the clipped "
        "fragment, not the full re-detection"
    )
    contact_events = [e for e in events if e["type"] == "contact"]
    assert len(contact_events) == 2, "initial store + replacement update"
    assert contact_events[0]["contact"]["id"] == contact_events[1]["contact"]["id"], (
        "the replacement must keep the operator-visible id"
    )


def test_distinct_same_class_neighbours_in_one_window_both_kept(
    survey_file, tmp_path_factory
) -> None:
    """Two real tires ~8 m apart inside ONE window are two contacts — the
    overlap dedup must never merge same-window neighbours (reviewer probe)."""
    # 8 m across-track at the sample's ~0.0488 m/col ground resolution.
    targets = [("starboard", 80, 400, "tire"), ("starboard", 80, 564, "tire")]
    repo = ContactRepo()
    stream_survey(
        survey_file, repo, emit=lambda _e: None,
        window_pings=WINDOW_PINGS, overlap_pings=OVERLAP_PINGS,
        detector_factory=lambda: ClippingTargetDetector(
            targets, _window_starts(N_PINGS, WINDOW_PINGS, OVERLAP_PINGS)
        ),
        output_root=tmp_path_factory.mktemp("pair_out"),
    )
    tires = [c for c in repo.query(limit=10) if c.cls == "tire"]
    assert len(tires) == 2, f"distinct neighbours merged: {len(tires)} stored"
    assert 4.0 < geodesic_m(tires[0].lat, tires[0].lon, tires[1].lat, tires[1].lon) < 12.0


def test_stream_survey_rejects_bad_windows(survey_file) -> None:
    with pytest.raises(ValueError):
        stream_survey(survey_file, ContactRepo(), emit=lambda _e: None, window_pings=1)
    with pytest.raises(ValueError):
        stream_survey(
            survey_file, ContactRepo(), emit=lambda _e: None,
            window_pings=100, overlap_pings=100,
        )


# --------------------------------------------------------- API-level flow --


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("realtime_api")
    app = create_app(
        require_auth=False,  # exercise the pipeline, not the RBAC layer (tests/test_auth.py)
        repo=ContactRepo(root / "contacts.db"),
        upload_dir=root / "uploads",
        output_root=root / "outputs",
        detector_factory=_stream_factory,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def streamed_job(client, survey_file) -> dict:
    with survey_file.open("rb") as fh:
        response = client.post(
            "/api/upload",
            files={"file": (survey_file.name, fh, "application/octet-stream")},
            data={"mode": "stream"},
        )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    deadline = time.time() + 180
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert job["status"] == "done", f"stream job failed: {job.get('error')}"
    return job


def test_upload_rejects_unknown_mode(client, survey_file) -> None:
    with survey_file.open("rb") as fh:
        response = client.post(
            "/api/upload",
            files={"file": (survey_file.name, fh, "application/octet-stream")},
            data={"mode": "warp"},
        )
    assert response.status_code == 422


def test_stream_job_snapshot_carries_recent_events(client, streamed_job) -> None:
    snap = client.get(f"/api/jobs/{streamed_job['id']}").json()
    assert snap["mode"] == "stream"
    events = snap["recent_events"]
    assert 0 < len(events) <= 50
    types = {e["type"] for e in events}
    assert types == {"window", "contact"}
    contact = next(e["contact"] for e in events if e["type"] == "contact")
    assert {"id", "cls", "confidence"} <= set(contact)


# --------------------------------------------------- (4) recovery workflow --


def test_recovery_cycle_and_audit_log(client, streamed_job) -> None:
    contact = client.get("/api/contacts").json()["contacts"][0]
    cid = contact["id"]
    assert contact["recovery"] == "flagged", "every contact starts flagged"

    for status in ("assigned", "retrieved"):
        response = client.post(f"/api/contacts/{cid}/recovery", json={"status": status})
        assert response.status_code == 200
        assert response.json()["recovery"] == status
    assert client.get(f"/api/contacts/{cid}").json()["recovery"] == "retrieved"

    log = [e for e in client.get("/api/recovery/log").json() if e["contact_id"] == cid]
    assert [e["status"] for e in log] == ["assigned", "retrieved"]

    assert client.post(
        f"/api/contacts/{cid}/recovery", json={"status": "lost"}
    ).status_code == 422
    assert client.post(
        "/api/contacts/no-such-id/recovery", json={"status": "assigned"}
    ).status_code == 404


def test_recovery_default_round_trips_legacy_json(client, streamed_job) -> None:
    """Contact JSON stored before the recovery field existed must validate."""
    contact = client.get("/api/contacts").json()["contacts"][0]
    legacy = {k: v for k, v in contact.items() if k not in ("recovery", "ping_range")}
    restored = Contact.model_validate(legacy)
    assert restored.recovery is RecoveryStatus.flagged


# ----------------------------------------------------- (5) health telemetry --


def test_health_edge_telemetry_shape(client, streamed_job) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["service"] == "sagar-netra"

    versions = health["versions"]
    assert set(versions) >= {
        "pipeline", "detector", "segmenter", "anomaly", "verifier_config", "verifier_model",
    }
    assert isinstance(versions["pipeline"], str) and versions["pipeline"]
    # sha1-8 of the weights file, or the explicit pretrained-fallback marker
    assert versions["detector"] == "pretrained-fallback" or len(versions["detector"]) == 8

    assert set(health["models_loaded"]) == {"detector", "segmenter", "anomaly", "verifier"}
    assert all(isinstance(v, bool) for v in health["models_loaded"].values())

    assert health["memory_mb"] > 0
    assert health["last_survey"]["name"] == streamed_job["survey"]
    assert health["last_survey"]["n_contacts"] == streamed_job["n_contacts"]
    assert health["last_survey"]["seconds"] > 0
    assert health["tiles_per_s_last"] > 0
