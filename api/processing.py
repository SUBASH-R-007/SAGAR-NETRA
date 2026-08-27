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
from geoscribe.build import build_contacts, survey_stats
from geoscribe.report import write_all
from geoscribe.severity import Layer, load_layers, load_mission
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


class _CombinedBrains:
    """Default TridentNet stack: Brain A always, Brain C when weights exist,
    fused by the ensemble merger (corroboration + open-set anomalies).

    ``detector_config`` is a partial Brain-A config override (deep-merged over
    ``configs/detector.yaml``) — mission profiles use it to lower ``conf`` for
    recall-first operations like SAR without touching the deployed YAML.
    """

    def __init__(self, detector_config: dict[str, Any] | None = None) -> None:
        from tridentnet.deep_ensemble import build_brain_a

        self.brain_a = build_brain_a(detector_config)
        try:
            from tridentnet.segmenter import Segmenter

            self.brain_b: Any = Segmenter()
        except (FileNotFoundError, ImportError):
            self.brain_b = None  # segmenter weights not trained yet
        try:
            from tridentnet.anomaly import AnomalyDetector

            self.brain_c: Any = AnomalyDetector()
        except (FileNotFoundError, ImportError):
            self.brain_c = None  # anomaly weights not trained yet: A-only mode

    def detect_tiles(self, tiles: list, progress: Any = None) -> list:
        from tridentnet.ensemble import merge_brains

        detections = self.brain_a.detect_tiles(tiles, progress=progress)
        if self.brain_b is not None:
            # Brain B refines filamentous-class boxes to their pixel-mask
            # extent (nets/ropes: the bbox lies about size; the mask doesn't).
            detections = [det for det, _mask in self.brain_b.refine_detections(detections, tiles)]
        blobs = self.brain_c.detect_tiles(tiles) if self.brain_c is not None else None
        return merge_brains(detections, blobs)


def _default_detector_factory(detector_config: dict[str, Any] | None = None) -> DetectorLike:
    return _CombinedBrains(detector_config)


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
    mission: str | None = None,
) -> dict[str, Any]:
    """Run the full chain on one survey file; returns a summary dict.

    ``mission`` names a disaster-mode profile in ``configs/missions/``
    (blueprint N-12): its ``hazard_table`` re-weights the severity ranking
    and its ``detector_conf`` overrides the Brain-A confidence floor (only
    for the default detector stack — an injected ``detector_factory`` owns
    its own configuration). Unknown names raise ``KeyError`` before any
    processing starts.
    """
    path = Path(path)
    survey = path.name
    out_dir = Path(output_root) / path.stem

    profile: dict[str, Any] | None = None
    hazard_table: dict[str, float] | None = None
    detector_config: dict[str, Any] | None = None
    if mission is not None:
        profile = load_mission(mission)
        hazard_table = profile["hazard_table"]
        if profile["detector_conf"] is not None:
            detector_config = {"conf": float(profile["detector_conf"])}

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
    detector = (
        detector_factory() if detector_factory is not None
        else _default_detector_factory(detector_config)
    )
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
        verified, pre, survey=survey, layers=layers, evidence_dir=out_dir / "evidence",
        hazard_table=hazard_table,
    )
    stage(frac=0.5)
    report_paths = write_all(contacts, out_dir, survey=survey, survey_stats=survey_stats(pre))
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
        "mission": mission,
        "mission_note": profile["reportable_extra_note"] if profile else None,
    }
