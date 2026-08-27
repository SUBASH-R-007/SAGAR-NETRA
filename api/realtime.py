"""Real-time streaming sink: survey replay as a live towed stream, plus the
edge-telemetry model inventory behind GET /api/health.

Sink pattern (blueprint): the pipeline core is the *same* parse ->
preprocess -> detect -> physics-verify -> geotag chain as
:func:`api.processing.process_survey`; only the output destination differs.
Instead of one summary at the end, the survey is replayed in sliding ping
windows — exactly how a towed real-time deployment consumes pings — and each
window's contacts are emitted and stored the moment they are verified.

Window correctness rests on two properties of the existing pieces:

* :meth:`PingArray.slice_pings` slices *nav together with the pings*, so a
  window is a self-consistent sub-survey: window-local ping indices index
  window-local nav records, and per-window preprocessing/geotagging therefore
  produces correct WGS-84 positions with no offset arithmetic at all (a test
  asserts stream positions match batch positions geodesically). Only the
  stored pixel extents need the window's ping offset added back afterwards,
  so contact pixel refs are global, batch-identical survey coordinates.
* Windows overlap by ``overlap_pings`` so a target split by one window
  boundary appears whole in the next — the same reason detector tiles
  overlap. The flip side is duplicates: dedup is geodesic (reusing the diff
  module's WGS-84 helper) and keeps the first-seen contact id.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from api.db import ContactRepo
from api.diff import geodesic_m
from api.processing import (
    DEFAULT_LAYER_DIR,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    DetectorLike,
    _default_detector_factory,
)
from geoscribe.build import build_contacts
from geoscribe.contact import Contact
from geoscribe.report import PIPELINE_VERSION, write_all
from geoscribe.severity import Layer, load_layers
from physicheck.calibrate import PhysicsGate
from physicheck.verify import verify_detections
from sonar_core.parsers.base import PingArray
from sonar_core.parsers.base import load as load_survey
from sonar_core.preprocess.pipeline import DEFAULTS, _deep_merge, preprocess

EmitFn = Callable[[dict[str, Any]], None]

#: Model artefacts reported by the health endpoint, per TridentNet brain.
WEIGHTS_PATHS: dict[str, Path] = {
    "detector": REPO_ROOT / "weights" / "detector.pt",
    "segmenter": REPO_ROOT / "weights" / "segmenter.pt",
    "anomaly": REPO_ROOT / "weights" / "anomaly.pt",
}
#: The Stage-2 verifier's behaviour lives in TWO files: the physics-gate
#: tunables (config) and the trained checkpoint (weights) that
#: verify_detections auto-loads when present — both are fingerprinted so a
#: retrain is always visible in /api/health.
VERIFIER_CONFIG_PATH: Path = REPO_ROOT / "configs" / "physics.yaml"
VERIFIER_WEIGHTS_PATH: Path = REPO_ROOT / "weights" / "verifier.pkl"


def _safe_emit(emit: EmitFn, event: dict[str, Any]) -> None:
    """Deliver one event; a broken observer must never abort a live stream."""
    try:
        emit(event)
    except Exception:  # noqa: BLE001 - observer failures are deliberately swallowed
        pass


def _survey_date(pa: PingArray) -> str:
    """Contact-id date part from the first finite ping time (build_contacts rule)."""
    times = pa.nav["time"]
    finite = times[np.isfinite(times)]
    if not finite.size:
        return "unknown-date"
    return datetime.fromtimestamp(float(finite[0]), tz=UTC).strftime("%Y%m%d")


def _window_starts(n_pings: int, window_pings: int, overlap_pings: int) -> list[int]:
    """Sliding-window start pings; the last window always reaches ``n_pings``."""
    step = window_pings - overlap_pings
    starts = [0]
    while starts[-1] + window_pings < n_pings:
        starts.append(starts[-1] + step)
    return starts


def _window_preprocess_config(
    base: dict[str, Any] | None, window_pings: int, total_pings: int
) -> dict[str, Any]:
    """Preprocess config for one window, with CLAHE kept scale-invariant.

    CLAHE equalizes each cell of a fixed ``tile_grid`` independently, so the
    grid's *row* count decides how many pings share one transfer curve. A
    window is far shorter than the survey, so the stock ``(8, 8)`` grid gives
    it much thinner cells (25 pings instead of ~75 on a 600-ping line) and
    equalizes far more aggressively — which lifts Rayleigh speckle into
    structure the anomaly autoencoder was never trained on. Measured on a
    600-ping survey: 0.50 anomalies/tile whole, **19.8/tile** in 200-ping
    windows, i.e. a flood of spurious ``unknown_anomaly`` contacts.

    Scaling the row count by the window's share of the survey restores the
    same pings-per-cell the batch path uses (1.17/tile measured), so streamed
    and batch imagery are equalized alike and the calibrated anomaly
    threshold stays meaningful. Column count is untouched: swath width does
    not change window to window.
    """
    merged = _deep_merge(DEFAULTS, base or {})
    grid = merged["clahe"]["tile_grid"]
    cols, rows = int(grid[0]), int(grid[1])
    if total_pings > 0 and window_pings > 0:
        rows = max(1, round(rows * window_pings / total_pings))
    config = dict(base or {})
    clahe_cfg = dict(config.get("clahe", {}))
    clahe_cfg["tile_grid"] = (cols, rows)
    config["clahe"] = clahe_cfg
    return config


def _better_observation(a: Contact, b: Contact) -> bool:
    """True when *a* is the stronger observation of one physical object.

    More scan lines wins first — a window-clipped fragment must always lose
    to the whole re-detection seen through the overlap (the same rule
    cross-tile NMS applies spatially); confidence breaks ties.
    """
    span_a = a.pixel.ping1 - a.pixel.ping0
    span_b = b.pixel.ping1 - b.pixel.ping0
    if span_a != span_b:
        return span_a > span_b
    return a.confidence > b.confidence


def _find_overlap_duplicate(
    candidate: Contact, kept: list[Contact], window_start: int, radius_m: float
) -> Contact | None:
    """A previously-kept contact that is the same object re-seen through the
    window overlap, or None.

    Only CROSS-window pairs can be windowing duplicates: contacts from the
    current window are never compared against each other (two drums 8 m
    apart inside one window are two real contacts — the detector's own NMS
    already handled intra-window duplication), and a kept contact qualifies
    only when its global ping extent reaches into the current window
    (``ping1 >= window_start``), because the shared overlap is the only place
    one object can be ensonified twice. Match = same class within *radius_m*
    geodesic metres.
    """
    for k in kept:
        if k.cls != candidate.cls or k.pixel.ping1 < window_start:
            continue
        if geodesic_m(candidate.lat, candidate.lon, k.lat, k.lon) <= radius_m:
            return k
    return None


def _offset_pixel(contact: Contact, ping_offset: int) -> Contact:
    """Window-local pixel extents -> survey-global ping rows (geotags are
    already correct: nav was sliced with the pings)."""
    pixel = contact.pixel.model_copy(
        update={
            "ping0": contact.pixel.ping0 + ping_offset,
            "ping1": contact.pixel.ping1 + ping_offset,
        }
    )
    return contact.model_copy(update={"pixel": pixel})


def _rekey(contact: Contact, new_id: str) -> Contact:
    """Assign the survey-wide id; evidence assets are renamed on disk to
    match, so paths keep telling the truth."""
    updates: dict[str, Any] = {"id": new_id}
    for attr in ("evidence_png", "thumbnail_png"):
        old = getattr(contact, attr)
        if not old:
            continue
        old_path = Path(old)
        new_path = old_path.with_name(old_path.name.replace(contact.id, new_id, 1))
        if old_path.exists():
            old_path.rename(new_path)
        updates[attr] = str(new_path)
    return contact.model_copy(update=updates)


def _drop_evidence(contact: Contact) -> None:
    """Remove the rendered assets of a deduplicated contact (never reported)."""
    for attr in ("evidence_png", "thumbnail_png"):
        path = getattr(contact, attr)
        if path:
            Path(path).unlink(missing_ok=True)


def stream_survey(
    path: str | Path,
    repo: ContactRepo,
    emit: EmitFn,
    window_pings: int = 200,
    overlap_pings: int = 40,
    detector_factory: Callable[[], DetectorLike] | None = None,
    layers: list[Layer] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    preprocess_config: dict | None = None,
    dedup_radius_m: float = 10.0,
    id_prefix: str = "SN",
) -> dict[str, Any]:
    """Replay one survey file as a live stream of window/contact events.

    The file is parsed once; windows are zero-copy :meth:`PingArray.slice_pings`
    views run through the exact batch pipeline stages. ``emit`` receives, in
    order, ``{"type": "contact", "contact": <contact json>}`` for every new
    (non-duplicate) contact as soon as its window is verified, then
    ``{"type": "window", "window", "start", "stop", "done_pings",
    "total_pings"}`` when the window completes; ``done_pings`` is monotonic
    and ends at ``total_pings``. Contacts are stored to *repo* incrementally,
    so a dashboard query mid-stream already sees everything emitted so far.

    ``overlap_pings`` plays the tiler's overlap role along-track: a target cut
    by a window boundary is whole in the next window; ``dedup_radius_m``
    (geodesic metres, same-class, cross-window pairs only) then collapses the
    double detections, keeping the STRONGER observation under the first-seen
    id — a clipped fragment is replaced by its whole re-detection (an updated
    contact event is emitted), and two distinct same-class objects inside one
    window are never merged. The final summary matches
    :func:`api.processing.process_survey` (no waterfall assets — a live sink
    has no full-survey mosaic) plus honest throughput numbers.
    """
    if window_pings < 2:
        raise ValueError(f"window_pings must be >= 2, got {window_pings}")
    if not 0 <= overlap_pings < window_pings:
        raise ValueError(
            f"overlap_pings must be in [0, window_pings), got {overlap_pings}"
        )
    t_start = time.perf_counter()
    path = Path(path)
    survey = path.name
    out_dir = Path(output_root) / path.stem

    pa = load_survey(path)  # parse ONCE; every window below is a view
    total_pings = pa.n_pings
    date_part = _survey_date(pa)
    if layers is None:
        layers = load_layers(DEFAULT_LAYER_DIR)
    detector = (
        detector_factory() if detector_factory is not None else _default_detector_factory()
    )
    gate = PhysicsGate()

    repo.delete_survey(survey)  # idempotent re-processing, like batch mode

    kept: list[Contact] = []
    seq = 0
    n_tiles = 0
    n_detections = 0
    starts = _window_starts(total_pings, window_pings, overlap_pings)
    for w, start in enumerate(starts):
        stop = min(start + window_pings, total_pings)
        window = pa.slice_pings(start, stop)
        pre = preprocess(
            window,
            config=_window_preprocess_config(
                preprocess_config, stop - start, total_pings
            ),
        )
        detections = detector.detect_tiles(pre.tiles)
        n_tiles += len(pre.tiles)
        n_detections += len(detections)
        verified = verify_detections(detections, pre, gate=gate)
        window_contacts = build_contacts(
            verified,
            pre,
            survey=survey,
            layers=layers,
            evidence_dir=out_dir / "evidence",
            id_prefix=f"{id_prefix}W{w:03d}",  # collision-free scratch ids
        )

        fresh: list[Contact] = []
        replaced: list[Contact] = []
        for contact in window_contacts:
            candidate = _offset_pixel(contact, start)  # global frame, scratch id
            dup = _find_overlap_duplicate(candidate, kept, start, dedup_radius_m)
            if dup is None:
                seq += 1
                fresh.append(_rekey(candidate, f"{id_prefix}-{date_part}-{seq:04d}"))
                continue
            if _better_observation(candidate, dup):
                # The overlap re-detection sees the whole object; the stored
                # window-clipped fragment must not win. Keep the operator-
                # visible id (and any review/recovery already recorded on it)
                # but replace the measurement.
                _drop_evidence(dup)
                replacement = _rekey(candidate, dup.id).model_copy(
                    update={
                        "review": dup.review,
                        "recovery": dup.recovery,
                        "notes": dup.notes,
                    }
                )
                kept[kept.index(dup)] = replacement
                replaced.append(replacement)
            else:
                _drop_evidence(candidate)
        if fresh or replaced:
            # add_contacts upserts by id, so replacements update in place;
            # incremental: queryable before the stream ends.
            repo.add_contacts(fresh + replaced)
            kept.extend(fresh)
            for contact in [*fresh, *replaced]:
                _safe_emit(
                    emit, {"type": "contact", "contact": contact.model_dump(mode="json")}
                )
        _safe_emit(
            emit,
            {
                "type": "window",
                "window": w,
                "start": start,
                "stop": stop,
                "done_pings": stop,
                "total_pings": total_pings,
            },
        )

    report_paths = write_all(kept, out_dir, survey=survey)
    repo.upsert_survey(survey, str(path), total_pings, len(kept), str(out_dir))
    seconds = time.perf_counter() - t_start

    return {
        "survey": survey,
        "n_pings": total_pings,
        "n_windows": len(starts),
        "n_tiles": n_tiles,
        "n_detections": n_detections,
        "n_contacts": len(kept),
        "outputs_dir": str(out_dir),
        "reports": {k: str(v) for k, v in report_paths.items()},
        "seconds": round(seconds, 2),
        "tiles_per_s": round(n_tiles / seconds, 2) if seconds > 0 else None,
    }


# ------------------------------------------------------- edge telemetry ----

#: (path, mtime, size) -> sha1-8, so /api/health never re-reads unchanged files.
_VERSION_CACHE: dict[tuple[str, float, int], str] = {}


def file_version(path: str | Path) -> str | None:
    """sha1-8 fingerprint of a weights/config file; None when the file is absent.

    A content hash (not a semver) is the only version a trained artefact
    truly has: retraining changes behaviour whether or not anyone bumps a
    number. Cached by (path, mtime, size) so health checks stay cheap.
    """
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    key = (str(path), stat.st_mtime, stat.st_size)
    if key not in _VERSION_CACHE:
        digest = hashlib.sha1()  # fingerprint, not security
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                digest.update(chunk)
        _VERSION_CACHE[key] = digest.hexdigest()[:8]
    return _VERSION_CACHE[key]


def model_inventory() -> dict[str, Any]:
    """Versions + per-brain load state for GET /api/health.

    File existence only — a health check must never pay a model load (or hide
    one behind a cache miss); ``models_loaded`` therefore means "weights are
    on disk for this brain", matching how the processing stack decides which
    brains to run.
    """
    versions: dict[str, Any] = {
        "pipeline": PIPELINE_VERSION,
        "detector": file_version(WEIGHTS_PATHS["detector"]) or "pretrained-fallback",
        "segmenter": file_version(WEIGHTS_PATHS["segmenter"]),
        "anomaly": file_version(WEIGHTS_PATHS["anomaly"]),
        "verifier_config": file_version(VERIFIER_CONFIG_PATH),
        "verifier_model": file_version(VERIFIER_WEIGHTS_PATH),
    }
    models_loaded = {name: path.is_file() for name, path in WEIGHTS_PATHS.items()}
    models_loaded["verifier"] = VERIFIER_WEIGHTS_PATH.is_file()
    return {"versions": versions, "models_loaded": models_loaded}
