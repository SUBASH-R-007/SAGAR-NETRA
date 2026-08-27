"""DRISHTI Console backend — FastAPI application.

App-factory pattern so tests inject a temp database, a stub detector, and a
temp output root; ``python -m api.main`` (or uvicorn) serves the real thing
plus the built React dashboard from ``web/dist`` when present.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
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
from api.db import ContactRepo
from api.diff import diff_surveys
from api.jobs import JobRegistry
from api.processing import DEFAULT_OUTPUT_ROOT, REPO_ROOT, process_survey
from geoscribe.contact import ReviewStatus
from geoscribe.severity import list_missions, load_mission

#: .zip is the transport for Humminbird recordings (a .DAT plus its sibling
#: .SON/.IDX directory cannot travel as one plain file); the archive is
#: extracted server-side and its .DAT located.
UPLOAD_SUFFIXES = {
    ".xtf", ".jsf", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
    ".sl2", ".sl3", ".zip",
}
TILE_CACHE = REPO_ROOT / "data" / "tile_cache"
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
WS_POLL_S = 0.25


class ReviewRequest(BaseModel):
    status: ReviewStatus
    notes: str | None = None


class CopilotRequest(BaseModel):
    question: str


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

    # ------------------------------------------------------------- health --

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "sagar-netra"}

    # ------------------------------------------------------------- upload --

    @app.post("/api/upload")
    async def upload(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI idiom
        mission: str | None = Form(None),  # noqa: B008 - FastAPI idiom
    ) -> dict[str, str]:
        suffix = Path(file.filename or "upload.bin").suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise HTTPException(415, f"unsupported file type {suffix!r}")
        if mission:
            try:  # validate before accepting the upload; processing re-loads it
                load_mission(mission)
            except KeyError as exc:
                raise HTTPException(422, str(exc.args[0])) from exc
        app.state.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = app.state.upload_dir / Path(file.filename).name
        with dest.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)

        if suffix == ".zip":
            dest = _extract_recording_zip(dest)

        job = app.state.jobs.create(dest.name)

        def run() -> None:
            registry = app.state.jobs
            registry.update(job.id, status="running", stage="parse")
            try:
                summary = process_survey(
                    dest,
                    app.state.repo,
                    update=lambda **f: registry.update(job.id, **f),
                    detector_factory=app.state.detector_factory,
                    output_root=app.state.output_root,
                    mission=mission or None,
                )
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
        return [j.snapshot() for j in app.state.jobs.all()]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.snapshot()

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
                    await ws.send_json(job.snapshot())
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

    @app.get("/api/reviews/export")
    def review_export() -> list[dict[str, Any]]:
        """The append-only review trail (future retraining label set)."""
        return app.state.repo.review_log()

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
        """OSM tile proxy with a disk cache: online once, offline forever."""
        cached = TILE_CACHE / str(z) / str(x) / f"{y}.png"
        if cached.exists():
            return Response(cached.read_bytes(), media_type="image/png")
        try:
            request = urllib.request.Request(
                OSM_URL.format(z=z, x=x, y=y),
                headers={"User-Agent": "SAGAR-NETRA/0.1 (offline-cache)"},
            )
            with urllib.request.urlopen(request, timeout=10) as resp:
                data = resp.read()
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
            return Response(data, media_type="image/png")
        except Exception:  # noqa: BLE001 - offline: serve a neutral sea-grid tile
            return Response(_fallback_tile(), media_type="image/png")

    # ---------------------------------------------------------- frontend --

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
