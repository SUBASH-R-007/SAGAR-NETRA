"""Deep-ensemble uncertainty for Brain A (blueprint L3 / N-03).

Several detectors trained from different seeds vote on every survey: their
per-model detections are fused by consensus, and the fused score is the sum
of matched scores divided by the number of models — a model that misses a
box drags its confidence down. This turns "how much do independently trained
models agree?" into the statistically meaningful uncertainty the blueprint
asks for (chosen over MC-dropout: YOLOv8n has no dropout layers to sample,
so ensemble disagreement is the honest source of epistemic uncertainty; see
DECISIONS.md).

The single-model :class:`~tridentnet.detector.Detector` remains the default
path; :class:`EnsembleDetector` activates when ``configs/detector.yaml``
lists ``ensemble_weights`` and those files exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tridentnet.detector import Detection, Detector, box_iou, load_config


def fuse_ensemble(
    per_model: Sequence[Sequence[Detection]],
    iou_thresh: float = 0.55,
    min_models: int = 1,
) -> list[Detection]:
    """Consensus fusion of per-model detection lists.

    Detections are greedily clustered across models per ``(side, cls)`` at
    ``iou_thresh``; each cluster fuses to one detection whose box is the
    score-weighted mean of its members and whose score is
    ``sum(member scores) / n_models`` — full agreement keeps the mean score,
    a 1-of-3 lone find is cut to a third. Clusters seen by fewer than
    ``min_models`` models are dropped.
    """
    n_models = len(per_model)
    if n_models == 0:
        return []
    if n_models == 1:
        return list(per_model[0])

    # Flatten with model provenance, strongest first so cluster seeds are the
    # most confident members.
    flat: list[tuple[int, Detection]] = [
        (m, d) for m, dets in enumerate(per_model) for d in dets
    ]
    flat.sort(key=lambda pair: -pair[1].score)

    used = [False] * len(flat)
    fused: list[Detection] = []
    for i, (_, seed) in enumerate(flat):
        if used[i]:
            continue
        members: list[tuple[int, Detection]] = [flat[i]]
        used[i] = True
        for j in range(i + 1, len(flat)):
            if used[j]:
                continue
            model_j, cand = flat[j]
            if (
                cand.side == seed.side
                and cand.cls == seed.cls
                and box_iou(seed, cand) >= iou_thresh
            ):
                members.append(flat[j])
                used[j] = True

        model_ids = {m for m, _ in members}
        if len(model_ids) < min_models:
            continue
        scores = np.array([d.score for _, d in members], dtype=np.float64)
        boxes = np.array(
            [(d.ping0, d.ping1, d.col0, d.col1) for _, d in members], dtype=np.float64
        )
        weights = scores / scores.sum()
        p0, p1, c0, c1 = (weights @ boxes).round().astype(int)
        fused.append(
            replace(
                seed,
                ping0=int(p0),
                ping1=int(max(p1, p0)),
                col0=int(c0),
                col1=int(max(c1, c0)),
                score=float(min(scores.sum() / n_models, 0.99)),
            )
        )
    fused.sort(key=lambda d: -d.score)
    return fused


class EnsembleDetector:
    """N independently trained Brain-A detectors fused by consensus."""

    def __init__(
        self,
        weights: Sequence[str | Path],
        config: dict[str, Any] | None = None,
        fuse_iou: float = 0.55,
    ) -> None:
        paths = [Path(w) for w in weights]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"ensemble weights missing: {missing}")
        if not paths:
            raise ValueError("ensemble needs at least one weights file")
        self.members = [Detector(weights=p, config=config) for p in paths]
        self.fuse_iou = fuse_iou

    def detect_tiles(
        self,
        tiles: list,
        progress: Callable[[str, float], None] | None = None,
    ) -> list[Detection]:
        per_model: list[list[Detection]] = []
        n = len(self.members)
        for k, member in enumerate(self.members):
            def member_progress(stage: str, frac: float, _k: int = k) -> None:
                if progress is not None:
                    progress(f"ensemble {_k + 1}/{n}", (_k + frac) / n)

            per_model.append(member.detect_tiles(tiles, progress=member_progress))
        return fuse_ensemble(per_model, iou_thresh=self.fuse_iou)

    def class_names(self) -> list[str]:
        return self.members[0].class_names()


def build_brain_a(config: dict[str, Any] | None = None) -> Any:
    """The processing layer's Brain-A factory: ensemble when configured and
    all listed weights exist, single detector otherwise."""
    cfg = load_config(config)
    listed = cfg.get("ensemble_weights") or []
    repo_root = Path(__file__).resolve().parents[1]
    paths = [repo_root / w if not Path(w).is_absolute() else Path(w) for w in listed]
    if paths and all(p.exists() for p in paths):
        return EnsembleDetector(paths, config=config, fuse_iou=float(cfg["dedup_iou"]))
    return Detector(config=config)
