"""DRISHTI Console backend — FastAPI application.

App-factory pattern so tests inject a temp database, a stub detector, and a
temp output root; ``python -m api.main`` (or uvicorn) serves the real thing
plus the built React dashboard from ``web/dist`` when present.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import copilot as copilot_mod
from api import physics_lab
from api.db import ContactRepo
from api.diff import diff_surveys
from api.jobs import Job, JobRegistry
from api.processing import DEFAULT_OUTPUT_ROOT, REPO_ROOT, process_survey
from api.realtime import model_inventory, stream_survey
from geoscribe.contact import RecoveryStatus, ReviewStatus
from geoscribe.severity import list_missions, load_mission

#: .zip is the transport for Humminbird recordings (a .DAT plus its sibling
#: .SON/.IDX directory cannot travel as one plain file); the archive is
#: extracted server-side and its .DAT located.
UPLOAD_SUFFIXES = {
    ".xtf", ".jsf", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
    ".sl2", ".sl3", ".zip",
}
TILE_CACHE = REPO_ROOT / "data" / "tile_cache"
#: Basemap source. Esri World Imagery: satellite with native coverage to deep
#: zoom everywhere (Ocean Base runs out at ~z10 offshore), no API key,
#: permitted with attribution — unlike tile.openstreetmap.org, whose usage
#: policy blocks server-side proxies like this one ("access blocked" tiles).
#: Note Esri's {z}/{y}/{x} path order. Override with SAGAR_TILE_URL and keep
#: the frontend attribution in MapView.jsx in step with whatever you point at.
TILE_URL = os.environ.get(
    "SAGAR_TILE_URL",
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
)
#: Cache is namespaced per source so switching providers never mixes styles.
TILE_CACHE_SLUG = hashlib.sha1(TILE_URL.encode()).hexdigest()[:8]
WS_POLL_S = 0.25
#: Stream-mode events kept on a job snapshot (the WS forwards this window,
#: so a reconnecting dashboard replays the freshest detections, not megabytes).
RECENT_EVENTS_CAP = 50
UPLOAD_MODES = ("batch", "stream")
#: Formats that record no navigation. For these the operator declares the
#: survey geometry at upload time (the sonar's range/altitude setting and
#: where the line ran); without it slant-range correction, height-from-shadow
#: and geotagging have nothing to work from.
GEOMETRY_REQUIRED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ReviewRequest(BaseModel):
    status: ReviewStatus
    notes: str | None = None


class RecoveryRequest(BaseModel):
    status: RecoveryStatus


class CopilotRequest(BaseModel):
    question: str


class ShadowRequest(BaseModel):
    """One object on a flat seabed, for the shadow forward/inverse panel."""

    altitude_m: float = 10.0
    height_m: float = 2.0
    ground_range_m: float = 20.0


class SimTarget(BaseModel):
    cls: str = "cylinder_drum"
    ground_range_m: float = 20.0
    height_m: float | None = None
    length_m: float | None = None
    width_m: float | None = None
    ping: int | None = None
    side: str = "starboard"


class SimulateRequest(BaseModel):
    """A seabed a visitor built, to be rendered and then measured."""

    targets: list[SimTarget] = []
    altitude_m: float = 8.0
    slant_range_m: float = 50.0
    n_pings: int = 400
    seed: int = 26057


def create_app(
    repo: ContactRepo | None = None,
    upload_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    detector_factory: Any = None,
) -> FastAPI:
    app = FastAPI(title="SAGAR-NETRA DRISHTI Console", version="0.1.0")
    app.state.repo = repo or ContactRepo(REPO_ROOT / "data" / "contacts.db")
    app.state.jobs = JobRegistry()
    app.state.upload_dir = Path(upload_dir or REPO_ROOT / "data" / "uploads")
    app.state.output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    app.state.detector_factory = detector_factory
    #: job id -> {"mode", "events", "lock"} for stream-mode uploads only;
    #: kept outside Job so the dataclass snapshot contract stays untouched.
    app.state.stream_meta = {}
    #: edge telemetry updated after every processing run (batch or stream).
    app.state.last_run = None
    app.state.tiles_per_s_last = None

    def _job_snapshot(job: Job) -> dict[str, Any]:
        """Job snapshot plus stream-mode extras (mode + bounded event window)."""
        snap = job.snapshot()
        meta = app.state.stream_meta.get(job.id)
        if meta is not None:
            snap["mode"] = meta["mode"]
            with meta["lock"]:
                snap["recent_events"] = list(meta["events"])
        return snap

    # ------------------------------------------------------------- health --

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Edge telemetry: model fingerprints, memory, last-run throughput.

        Model state is reported from file existence/hashes only — a health
        probe must never pay (or hide) a model load.
        """
        import psutil

        rows = app.state.repo.surveys()
        last_survey = None
        if rows:
            row = rows[0]  # surveys() orders by processed_at DESC
            last_run = app.state.last_run
            last_survey = {
                "name": row["name"],
                "n_contacts": row["n_contacts"],
                "seconds": (
                    last_run["seconds"]
                    if last_run and last_run["survey"] == row["name"]
                    else None
                ),
            }
        return {
            "status": "ok",
            "service": "sagar-netra",
            **model_inventory(),
            "memory_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
            "last_survey": last_survey,
            "tiles_per_s_last": app.state.tiles_per_s_last,
        }

    # ------------------------------------------------------------- upload --

    @app.post("/api/upload")
    async def upload(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI idiom
        mission: str | None = Form(None),  # noqa: B008 - FastAPI idiom
        mode: str = Form("batch"),  # noqa: B008 - FastAPI idiom
        # Declared survey geometry — used only by nav-less formats (images).
        altitude_m: float | None = Form(None),  # noqa: B008 - FastAPI idiom
        range_m: float | None = Form(None),  # noqa: B008 - FastAPI idiom
        lat: float | None = Form(None),  # noqa: B008 - FastAPI idiom
        lon: float | None = Form(None),  # noqa: B008 - FastAPI idiom
        heading_deg: float | None = Form(None),  # noqa: B008 - FastAPI idiom
        sensor_depth_m: float | None = Form(None),  # noqa: B008 - FastAPI idiom
    ) -> dict[str, str]:
        suffix = Path(file.filename or "upload.bin").suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise HTTPException(415, f"unsupported file type {suffix!r}")
        if mode not in UPLOAD_MODES:
            raise HTTPException(422, f"mode must be one of {UPLOAD_MODES}, got {mode!r}")
        if mission and mode == "stream":
            raise HTTPException(422, "mission profiles apply to batch mode only")
        if mission:
            try:  # validate before accepting the upload; processing re-loads it
                load_mission(mission)
            except KeyError as exc:
                raise HTTPException(422, str(exc.args[0])) from exc
        parser_kwargs: dict[str, Any] = {}
        if suffix in GEOMETRY_REQUIRED_SUFFIXES:
            if altitude_m is None or range_m is None:
                raise HTTPException(
                    422,
                    f"{suffix} carries no navigation: altitude_m and range_m are "
                    "required so slant-range correction and height-from-shadow "
                    "have geometry to work from",
                )
            if altitude_m <= 0 or range_m <= 0:
                raise HTTPException(422, "altitude_m and range_m must be positive")
            if range_m <= altitude_m:
                raise HTTPException(
                    422,
                    f"range_m ({range_m}) must exceed altitude_m ({altitude_m}): a "
                    "swath only exists beyond the first bottom return",
                )
            parser_kwargs = {
                "altitude_m": float(altitude_m),
                "slant_range_m": float(range_m),
                "start_time": time.time(),
            }
            if sensor_depth_m is not None:
                parser_kwargs["sensor_depth_m"] = float(sensor_depth_m)
            if lat is not None and lon is not None:
                parser_kwargs["lat"] = float(lat)
                parser_kwargs["lon"] = float(lon)
            if heading_deg is not None:
                parser_kwargs["heading_deg"] = float(heading_deg)

        app.state.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = app.state.upload_dir / Path(file.filename).name
        with dest.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)

        if suffix == ".zip":
            dest = _extract_recording_zip(dest)

        job = app.state.jobs.create(dest.name, mission=mission or None)
        registry = app.state.jobs

        emit = None
        if mode == "stream":
            meta = {
                "mode": mode,
                "events": deque(maxlen=RECENT_EVENTS_CAP),
                "lock": threading.Lock(),
            }
            app.state.stream_meta[job.id] = meta

            def emit(event: dict[str, Any]) -> None:
                """Stream sink: keep the bounded event window, map window
                progress onto the job, and bump the job version so the WS
                forwards a fresh snapshot for every event."""
                with meta["lock"]:
                    meta["events"].append(event)
                fields: dict[str, Any] = {}
                if event.get("type") == "window" and event.get("total_pings"):
                    done, total = event["done_pings"], event["total_pings"]
                    fields = {
                        "stage": "stream",
                        "fraction": round(done / total, 3),
                        "message": f"streamed {done}/{total} pings",
                    }
                registry.update(job.id, **fields)

        def run() -> None:
            registry.update(job.id, status="running", stage="parse")
            t0 = time.perf_counter()
            try:
                if mode == "stream":
                    summary = stream_survey(
                        dest,
                        app.state.repo,
                        emit=emit,
                        detector_factory=app.state.detector_factory,
                        output_root=app.state.output_root,
                    )
                else:
                    summary = process_survey(
                        dest,
                        app.state.repo,
                        update=lambda **f: registry.update(job.id, **f),
                        detector_factory=app.state.detector_factory,
                        output_root=app.state.output_root,
                        mission=mission or None,
                        parser_kwargs=parser_kwargs or None,
                    )
                elapsed = time.perf_counter() - t0
                app.state.tiles_per_s_last = round(
                    summary["n_tiles"] / max(elapsed, 1e-9), 2
                )
                app.state.last_run = {
                    "survey": summary["survey"],
                    "n_contacts": summary["n_contacts"],
                    "seconds": round(elapsed, 2),
                }
                registry.update(
                    job.id, status="done", stage="done", fraction=1.0,
                    survey=summary["survey"], n_contacts=summary["n_contacts"],
                    message=f"{summary['n_contacts']} contacts",
                )
            except Exception as exc:  # noqa: BLE001 - job errors surface to the UI
                registry.update(job.id, status="error", error=str(exc), message=str(exc))

        threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
        return {"job_id": job.id}

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return [_job_snapshot(j) for j in app.state.jobs.all()]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return _job_snapshot(job)

    @app.websocket("/api/jobs/{job_id}/progress")
    async def job_progress(ws: WebSocket, job_id: str) -> None:
        await ws.accept()
        last_version = -1
        try:
            while True:
                job = app.state.jobs.get(job_id)
                if job is None:
                    await ws.send_json({"error": "unknown job"})
                    break
                if job.version != last_version:
                    last_version = job.version
                    await ws.send_json(_job_snapshot(job))
                if job.status in ("done", "error"):
                    break
                await asyncio.sleep(WS_POLL_S)
        except WebSocketDisconnect:
            return
        await ws.close()

    # ----------------------------------------------------------- contacts --

    @app.get("/api/surveys")
    def surveys() -> list[dict[str, Any]]:
        return app.state.repo.surveys()

    @app.get("/api/contacts")
    def contacts(
        survey: str | None = None,
        cls: str | None = None,
        min_conf: float | None = None,
        min_sev: float | None = None,
        review: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        found = app.state.repo.query(
            survey=survey, cls=cls, min_conf=min_conf, min_sev=min_sev,
            review=review, limit=limit,
        )
        return {"contacts": [c.model_dump(mode="json") for c in found]}

    @app.get("/api/contacts/{contact_id}")
    def contact(contact_id: str) -> dict[str, Any]:
        found = app.state.repo.get(contact_id)
        if found is None:
            raise HTTPException(404, "unknown contact")
        return found.model_dump(mode="json")

    @app.post("/api/contacts/{contact_id}/review")
    def review(contact_id: str, body: ReviewRequest) -> dict[str, Any]:
        updated = app.state.repo.set_review(contact_id, body.status, body.notes)
        if updated is None:
            raise HTTPException(404, "unknown contact")
        return updated.model_dump(mode="json")

    @app.post("/api/contacts/{contact_id}/recovery")
    def recovery(contact_id: str, body: RecoveryRequest) -> dict[str, Any]:
        """Advance the recovery workflow: flagged -> assigned -> retrieved."""
        updated = app.state.repo.set_recovery(contact_id, body.status)
        if updated is None:
            raise HTTPException(404, "unknown contact")
        return updated.model_dump(mode="json")

    @app.get("/api/reviews/export")
    def review_export() -> list[dict[str, Any]]:
        """The append-only review trail (future retraining label set)."""
        return app.state.repo.review_log()

    @app.get("/api/recovery/log")
    def recovery_export() -> list[dict[str, Any]]:
        """The append-only recovery audit trail (operations record)."""
        return app.state.repo.recovery_log()

    def _contact_image(contact_id: str, attr: str) -> FileResponse:
        found = app.state.repo.get(contact_id)
        if found is None:
            raise HTTPException(404, "unknown contact")
        path = getattr(found, attr)
        if not path or not Path(path).exists():
            raise HTTPException(404, f"no {attr} stored")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/contacts/{contact_id}/evidence")
    def evidence(contact_id: str) -> FileResponse:
        return _contact_image(contact_id, "evidence_png")

    @app.get("/api/contacts/{contact_id}/thumb")
    def thumb(contact_id: str) -> FileResponse:
        return _contact_image(contact_id, "thumbnail_png")

    # ------------------------------------------------------------ reports --

    @app.get("/api/report/{fmt}")
    def report(fmt: str, survey: str) -> FileResponse:
        names = {
            "json": "contacts.json", "csv": "contacts.csv",
            "geojson": "contacts.geojson", "kml": "contacts.kml", "pdf": "report.pdf",
        }
        if fmt not in names:
            raise HTTPException(404, f"unknown format {fmt!r}")
        row = next((s for s in app.state.repo.surveys() if s["name"] == survey), None)
        if row is None:
            raise HTTPException(404, "unknown survey")
        path = Path(row["outputs_dir"]) / names[fmt]
        if not path.exists():
            raise HTTPException(404, "report not generated")
        return FileResponse(path, filename=path.name)

    # ---------------------------------------------------------- waterfall --

    def _survey_outputs(survey: str) -> Path:
        row = next((s for s in app.state.repo.surveys() if s["name"] == survey), None)
        if row is None:
            raise HTTPException(404, "unknown survey")
        return Path(row["outputs_dir"])

    @app.get("/api/waterfall/{survey}")
    def waterfall(survey: str, raw: bool = False) -> FileResponse:
        name = "waterfall_raw.png" if raw else "waterfall.png"
        path = _survey_outputs(survey) / name
        if not path.exists():
            raise HTTPException(404, "waterfall not generated")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/waterfall/{survey}/meta")
    def waterfall_meta(survey: str) -> JSONResponse:
        path = _survey_outputs(survey) / "meta.json"
        if not path.exists():
            raise HTTPException(404, "meta not generated")
        return JSONResponse(json.loads(path.read_text()))

    @app.get("/api/summary")
    def survey_summary(survey: str) -> JSONResponse:
        """Survey-level statistics for the overview dashboard.

        Reuses the ``summary`` block already written into ``contacts.json`` by
        :func:`geoscribe.report.write_all` — area surveyed, debris density and
        the sonar configuration are derived from ping navigation during
        reporting, so recomputing them here would risk the two disagreeing.
        """
        path = _survey_outputs(survey) / "contacts.json"
        if not path.exists():
            raise HTTPException(404, "report not generated")
        doc = json.loads(path.read_text(encoding="utf-8"))
        return JSONResponse(
            {
                "survey": doc.get("survey", survey),
                "generated_at": doc.get("generated_at"),
                "pipeline_version": doc.get("pipeline_version"),
                **doc.get("summary", {}),
            }
        )

    # --------------------------------------------------- diff & copilot ----

    @app.get("/api/diff")
    def diff(survey_a: str, survey_b: str, radius_m: float = 25.0) -> dict[str, Any]:
        return diff_surveys(app.state.repo, survey_a, survey_b, radius_m)

    @app.get("/api/crossview")
    def crossview(survey_a: str, survey_b: str, radius_m: float = 15.0) -> dict[str, Any]:
        """Cross-swath corroboration between two overlapping surveys."""
        from physicheck.crossview import cross_confirm

        a = app.state.repo.query(survey=survey_a, limit=10_000)
        b = app.state.repo.query(survey=survey_b, limit=10_000)
        result = cross_confirm(a, b, radius_m=radius_m)
        return {"survey_a": survey_a, "survey_b": survey_b, **result.to_dict()}

    @app.get("/api/route")
    def route(
        survey: str | None = None,
        review: str = "confirmed",
        start_lat: float | None = None,
        start_lon: float | None = None,
        cluster_eps_m: float | None = None,
    ) -> dict[str, Any]:
        """Recovery tour over contacts (default: the confirmed ones);
        ``cluster_eps_m`` switches to the two-level retrieval-zone tour."""
        from geoscribe.route import plan_route

        found = app.state.repo.query(
            survey=survey, review=review or None, limit=500
        )
        if cluster_eps_m is not None:
            from geoscribe.cluster import plan_cluster_route

            return plan_cluster_route(found, cluster_eps_m, start_lat, start_lon)
        return plan_route(found, start_lat, start_lon)

    @app.post("/api/copilot")
    def ask_copilot(body: CopilotRequest) -> dict[str, Any]:
        return copilot_mod.ask(body.question, app.state.repo)

    # -------------------------------------------------------- physics lab --
    # Every route here calls the same functions that process real surveys, so
    # what the lab shows and what the pipeline does cannot drift apart.

    @app.get("/api/physics/geometry")
    def physics_geometry(
        altitude_m: float = 8.0,
        range_m: float = 50.0,
        beam_deg: float | None = None,
        pulse_us: float | None = None,
        sound_velocity_mps: float | None = None,
    ) -> dict[str, Any]:
        """Resolution, multipath range and sound-speed error for one sonar setup."""
        return physics_lab.geometry_report(
            altitude_m, range_m,
            beam_deg=beam_deg, pulse_us=pulse_us,
            sound_velocity_mps=sound_velocity_mps,
        )

    @app.post("/api/physics/shadow")
    def physics_shadow(body: ShadowRequest) -> dict[str, Any]:
        """Forward-model a shadow, then invert it with the deployed estimator."""
        return physics_lab.shadow_round_trip(
            body.altitude_m, body.height_m, body.ground_range_m
        )

    @app.get("/api/physics/classes")
    def physics_classes() -> list[dict[str, Any]]:
        """Target classes the scene simulator can place, with their size ranges."""
        return physics_lab.available_classes()

    @app.post("/api/physics/simulate")
    def physics_simulate(body: SimulateRequest) -> dict[str, Any]:
        """Render a placed seabed and measure every target from its shadow."""
        if not body.targets:
            raise HTTPException(422, "place at least one target to simulate")
        return physics_lab.simulate_scene(
            [t.model_dump() for t in body.targets],
            altitude_m=body.altitude_m,
            slant_range_m=body.slant_range_m,
            n_pings=body.n_pings,
            seed=body.seed,
        )

    # ----------------------------------------------------------- missions --

    @app.get("/api/missions")
    def missions() -> list[dict[str, str]]:
        """Disaster-mode mission profiles available for /api/upload."""
        return list_missions()

    # ------------------------------------------------------------- layers --

    @app.get("/api/layers")
    def layers() -> dict[str, Any]:
        layer_dir = REPO_ROOT / "data" / "layers"
        docs = {}
        if layer_dir.is_dir():
            for path in sorted(layer_dir.glob("*.geojson")):
                docs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        return docs

    # ------------------------------------------------- offline tile proxy --

    @app.get("/tiles/{z}/{x}/{y}.png")
    def tile(z: int, x: int, y: int) -> Response:
        """Basemap tile proxy with a disk cache: online once, offline forever."""
        cached = TILE_CACHE / TILE_CACHE_SLUG / str(z) / str(x) / f"{y}.png"
        if cached.exists():
            return Response(cached.read_bytes(), media_type="image/png")
        try:
            request = urllib.request.Request(
                TILE_URL.format(z=z, x=x, y=y),
                headers={"User-Agent": "SAGAR-NETRA/0.1 (offline tile cache)"},
            )
            with urllib.request.urlopen(request, timeout=10) as resp:
                data = resp.read()
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
            return Response(data, media_type="image/png")
        except Exception:  # noqa: BLE001 - offline: serve a neutral sea-grid tile
            return Response(_fallback_tile(), media_type="image/png")

    # ---------------------------------------------------------- frontend --

    @app.middleware("http")
    async def _cache_policy(request, call_next):
        """Cache the immutable, never the shell.

        Vite fingerprints every bundle under /assets/, so those files can be
        cached forever — but index.html references them BY HASH, and a cached
        shell pointing at bundles a later rebuild deleted renders a broken or
        frozen dashboard (the "nothing changed after the update" bug). HTML
        therefore always revalidates.
        """
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif "text/html" in content_type:
            response.headers["Cache-Control"] = "no-cache"
        return response

    dist = REPO_ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


def _extract_recording_zip(archive: Path) -> Path:
    """Extract an uploaded recording archive and return its survey entry file.

    Supports Humminbird recordings (a ``.DAT`` plus its sibling ``.SON``/
    ``.IDX`` directory) and, generically, any archive containing exactly one
    parseable survey file. Zip-slip is blocked by refusing member paths that
    escape the extraction directory.
    """
    import zipfile

    out_dir = archive.parent / archive.stem
    try:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (out_dir / member).resolve()
                if not str(target).startswith(str(out_dir.resolve())):
                    raise HTTPException(422, "archive contains unsafe paths")
            zf.extractall(out_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(422, "not a valid zip archive") from exc
    finally:
        archive.unlink(missing_ok=True)

    for pattern in ("*.dat", "*.DAT", "*.xtf", "*.jsf", "*.sl2", "*.sl3"):
        matches = sorted(out_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise HTTPException(422, "archive contains no supported survey file (.DAT/.xtf/.jsf/.sl2/.sl3)")


_FALLBACK_TILE: bytes | None = None


def _fallback_tile() -> bytes:
    """256x256 flat 'offline sea' tile with a faint grid, built once."""
    global _FALLBACK_TILE
    if _FALLBACK_TILE is None:
        import io

        import numpy as np
        from PIL import Image

        tile = np.full((256, 256, 3), (18, 38, 60), dtype=np.uint8)
        tile[::64, :, :] = (28, 52, 80)
        tile[:, ::64, :] = (28, 52, 80)
        buf = io.BytesIO()
        Image.fromarray(tile).save(buf, format="PNG")
        _FALLBACK_TILE = buf.getvalue()
    return _FALLBACK_TILE


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
