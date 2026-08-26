"""Survey processing orchestration: file -> contacts -> reports, with progress.

This is the one function behind both POST /upload and ``scripts/demo.py``:
parse -> preprocess -> detect -> physics-verify -> geotag/severity ->
reports, streaming stage/fraction updates through a callback the WebSocket
layer forwards to the browser. The detector is injected via a factory so
tests (and the anomaly-only mode) can substitute brains without loading
weights.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from api.db import ContactRepo
from geoscribe.build import build_contacts
from geoscribe.report import write_all
from geoscribe.severity import Layer, load_layers
from physicheck.calibrate import PhysicsGate
from physicheck.verify import verify_detections
from sonar_core.parsers.base import load as load_survey
from sonar_core.preprocess.pipeline import PreprocessResult, preprocess
from sonar_core.waterfall import normalize_u8

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "surveys"
DEFAULT_LAYER_DIR = REPO_ROOT / "data" / "layers"

#: (stage key, progress range start) — fractions interpolate inside each band.
STAGE_BANDS: dict[str, tuple[float, float]] = {
    "parse": (0.00, 0.08),
    "preprocess": (0.08, 0.45),
    "detect": (0.45, 0.70),
    "physics": (0.70, 0.80),
    "report": (0.80, 1.00),
}

ProgressFn = Callable[..., None]


class DetectorLike(Protocol):
    def detect_tiles(self, tiles: list, progress: Any = None) -> list: ...


def _default_detector_factory() -> DetectorLike:
    from tridentnet.detector import Detector

    return Detector()


def _band_progress(update: ProgressFn | None, stage: str, message: str = ""):
    """Progress fn for one stage, remapping [0,1] into the stage's band."""
    lo, hi = STAGE_BANDS[stage]

    def fn(_name: str = "", frac: float = 0.0) -> None:
        if update is not None:
            update(stage=stage, fraction=round(lo + (hi - lo) * min(max(frac, 0), 1), 3),
                   message=message or stage)

    return fn


def _write_waterfall_assets(pre: PreprocessResult, out_dir: Path) -> dict[str, Any]:
    """Combined enhanced waterfall PNG + geometry meta for the viewer overlay."""
    port = np.nan_to_num(pre.ground.port, nan=0.0)
    stbd = np.nan_to_num(pre.ground.starboard, nan=0.0)
    combined = np.hstack([port[:, ::-1], stbd])
    img = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(out_dir / "waterfall.png")

    raw = normalize_u8(np.hstack([
        np.nan_to_num(pre.ground_raw.port, nan=0.0)[:, ::-1],
        np.nan_to_num(pre.ground_raw.starboard, nan=0.0),
    ]))
    Image.fromarray(raw, mode="L").save(out_dir / "waterfall_raw.png")

    meta = {
        "n_pings": int(pre.ground.n_pings),
        "n_port_cols": int(pre.ground.n_cols("port")),
        "n_stbd_cols": int(pre.ground.n_cols("starboard")),
        "ground_res": float(pre.ground.ground_res),
    }
    import json

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def process_survey(
    path: str | Path,
    repo: ContactRepo,
    update: ProgressFn | None = None,
    detector_factory: Callable[[], DetectorLike] | None = None,
    layers: list[Layer] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    preprocess_config: dict | None = None,
) -> dict[str, Any]:
    """Run the full chain on one survey file; returns a summary dict."""
    path = Path(path)
    survey = path.name
    out_dir = Path(output_root) / path.stem

    stage = _band_progress(update, "parse", f"parsing {survey}")
    stage(frac=0.0)
    pa = load_survey(path)
    stage(frac=1.0)

    pre = preprocess(
        pa, config=preprocess_config,
        progress=lambda name, frac: _band_progress(update, "preprocess", name)(frac=frac),
    )

    stage = _band_progress(update, "detect", "running TridentNet")
    stage(frac=0.0)
    detector = (detector_factory or _default_detector_factory)()
    detections = detector.detect_tiles(
        pre.tiles, progress=lambda name, frac: _band_progress(update, "detect", name)(frac=frac)
    )
    stage(frac=1.0)

    gate = PhysicsGate()
    verified = verify_detections(
        detections, pre, gate=gate,
        progress=lambda name, frac: _band_progress(update, "physics", name)(frac=frac),
    )

    stage = _band_progress(update, "report", "geotagging and reporting")
    stage(frac=0.1)
    if layers is None:
        layers = load_layers(DEFAULT_LAYER_DIR)
    contacts = build_contacts(
        verified, pre, survey=survey, layers=layers, evidence_dir=out_dir / "evidence"
    )
    stage(frac=0.5)
    report_paths = write_all(contacts, out_dir, survey=survey)
    meta = _write_waterfall_assets(pre, out_dir)
    stage(frac=0.9)

    repo.delete_survey(survey)  # idempotent re-processing
    repo.add_contacts(contacts)
    repo.upsert_survey(
        survey, str(path), pa.n_pings, len(contacts), str(out_dir)
    )
    stage(frac=1.0)

    return {
        "survey": survey,
        "n_pings": pa.n_pings,
        "n_tiles": len(pre.tiles),
        "n_detections": len(detections),
        "n_contacts": len(contacts),
        "outputs_dir": str(out_dir),
        "reports": {k: str(v) for k, v in report_paths.items()},
        "waterfall_meta": meta,
    }
