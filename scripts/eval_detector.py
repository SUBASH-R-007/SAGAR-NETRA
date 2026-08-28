"""Ablation + metrics table for the deployed detection stack (strategy-PDF section 9).

Usage:
    python scripts/eval_detector.py [--scenes 8] [--seed-base 12000] [--iou 0.30]
        [--conf-pct 50] [--any-class] [--out outputs/metrics/ablation.md]

Held-out synthetic scenes (seed base 12000 — far from the detector training
seeds and the calibration set's 9000 block) are rendered with known ground
truth, preprocessed with the real M2 chain, and detected ONCE with the
deployed detector stack (the exact object behind POST /upload: Brain-A
ensemble when configured, Brain-B mask refinement, Brain-C anomaly merge).
The SAME raw detections are then re-scored under four cumulative
configurations, so every delta in the table is attributable to exactly one
pipeline stage and never to detector nondeterminism:

    (a) raw detector    — temperature-calibrated score only (score >= 0.25)
    (b) + physics gate  — Stage-1 highlight/shadow plausibility multipliers
    (c) + ML verifier   — Stage-2 learned cue-vector multiplier
    (d) + temporal      — thin-detection persistence gate (full deployed config)

A detection is a true positive when it overlaps a same-class MAN-MADE truth
box at IoU >= the threshold (``--any-class`` relaxes the class match, keeping
localization-only credit); hits on natural targets (rock clusters) and on
background are false positives. Per configuration the table reports
precision/recall/F1 at calibrated confidence >= the floor (default 50%),
PR-AUC swept over all confidence thresholds, and false positives per km² of
surveyed seabed (area via :func:`geoscribe.build.survey_stats` — the same
arithmetic the contacts.json summary block uses).

Everything here is SYNTHETIC: the honest preamble is written into the output
markdown so the table can never be quoted without that caveat.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from geoscribe.build import survey_stats
from physicheck.calibrate import DEFAULT_CONFIG as PHYSICS_CONFIG_PATH
from physicheck.calibrate import GateResult, PhysicsGate
from physicheck.verifier import PhysicsVerifier
from physicheck.verify import verify_detections
from sonar_core.preprocess.pipeline import PreprocessResult, preprocess
from sonar_core.synth.scene import SceneConfig, SynthTarget, make_scene
from tridentnet.data import _target_bbox, random_targets
from tridentnet.detector import Detection, box_iou

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "metrics" / "ablation.md"

#: Held-out seed block: detector training uses small seeds, the calibration
#: validation set uses the 9000 block — 12000 shares texture with neither.
SEED_BASE = 12_000

#: (key, table label) in cumulative-ablation order.
CONFIGS: tuple[tuple[str, str], ...] = (
    ("a_raw", "(a) raw detector (calibrated score)"),
    ("b_gate", "(b) + physics gate (Stage-1)"),
    ("c_verifier", "(c) + ML verifier (Stage-2)"),
    ("d_full", "(d) + temporal persistence (deployed)"),
)

#: Multiplier-free gate result: config (a) reuses the deployed temperature
#: scaling (PhysicsGate.confidence_pct) with no physics evidence applied.
_NEUTRAL_GATE = GateResult(multiplier=1.0, violation=False, reason=None)


class DetectorLike(Protocol):
    def detect_tiles(self, tiles: list, progress: Any = None) -> list: ...


def deployed_detector() -> DetectorLike:
    """The exact stack POST /upload runs (``api.processing`` default factory)."""
    from api.processing import _default_detector_factory

    return _default_detector_factory()


def _no_temporal_gate() -> PhysicsGate:
    """The deployed gate with the persistence gate disabled.

    ``min_persistence_pings = 1`` can never demote (every inclusive box spans
    at least one ping), so configs (b) and (c) isolate the highlight/shadow
    and verifier stages from the temporal stage that (d) adds back.
    """
    import copy

    import yaml

    config = yaml.safe_load(PHYSICS_CONFIG_PATH.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["scoring"]["min_persistence_pings"] = 1
    return PhysicsGate(config)


def _truth_boxes(
    pre: PreprocessResult,
    targets: list[SynthTarget],
    cfg: SceneConfig,
    shadow_pad_cols: int,
) -> list[Detection]:
    """Man-made truth boxes in global inclusive pixel coordinates.

    Reuses the dataset builder's box geometry (:func:`tridentnet.data._target_bbox`
    with the same down-range shadow pad the detector was trained with), so
    "truth" here means exactly what a perfect detector would have been taught
    to output. Natural targets (rock clusters) are excluded on purpose: a hit
    on one is a false alarm by definition, not a recalled object.
    """
    truths: list[Detection] = []
    for t in targets:
        if t.natural:
            continue
        bbox = _target_bbox(pre.ground, t, cfg, shadow_pad_cols=shadow_pad_cols)
        if bbox is None:
            continue  # rendered entirely off the usable swath
        r0, r1, c0, c1 = bbox
        truths.append(
            Detection(
                side=t.side, ping0=int(r0), ping1=max(int(r1) - 1, int(r0)),
                col0=int(c0), col1=max(int(c1) - 1, int(c0)), cls=t.cls, score=1.0,
            )
        )
    return truths


def _match_scene(
    scored: list[tuple[Any, float]],
    truths: list[Detection],
    iou_thresh: float,
    any_class: bool,
) -> list[tuple[float, bool]]:
    """Greedy one-to-one matching, most confident detection first.

    Standard detection-evaluation doctrine: each truth box may be claimed by
    at most one detection, so duplicates of a found object count as false
    positives instead of free extra recall. Matching is confidence-ranked
    *per configuration* — a stage that reorders detections changes which copy
    claims the truth, exactly as it would change what the operator reviews
    first. Returns ``(confidence_pct, is_true_positive)`` per detection.
    """
    order = sorted(range(len(scored)), key=lambda i: -scored[i][1])
    taken = [False] * len(truths)
    labels: list[tuple[float, bool]] = []
    for i in order:
        det, conf = scored[i]
        best_j, best_iou = -1, 0.0
        for j, truth in enumerate(truths):
            if taken[j] or truth.side != det.side:
                continue
            if not any_class and truth.cls != det.cls:
                continue
            iou = box_iou(det, truth)
            if iou >= iou_thresh and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            taken[best_j] = True
        labels.append((conf, best_j >= 0))
    return labels


@dataclass(frozen=True)
class ConfigMetrics:
    """Scores for one ablation configuration over every held-out scene."""

    precision: float  # at confidence >= the floor
    recall: float
    f1: float
    pr_auc: float  # average precision swept over all confidence thresholds
    fp_per_km2: float  # false positives >= the floor per surveyed km²
    tp: int  # true positives at the floor
    fp: int  # false positives at the floor
    n_detections: int  # all detections scored (any confidence)


def _config_metrics(
    labels: list[tuple[float, bool]],
    n_truth: int,
    area_km2: float,
    conf_floor_pct: float,
) -> ConfigMetrics:
    """Point metrics at the confidence floor + threshold-swept PR-AUC.

    PR-AUC is standard average precision: detections ranked by confidence,
    ``AP = sum(dRecall * precision)`` — the area under the precision-recall
    curve as the confidence threshold sweeps from strict to permissive.
    """
    confs = np.asarray([c for c, _ in labels], dtype=np.float64)
    tps = np.asarray([t for _, t in labels], dtype=bool)

    sel = confs >= conf_floor_pct
    tp = int((tps & sel).sum())
    fp = int((~tps & sel).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pr_auc = 0.0
    if n_truth and confs.size:
        order = np.argsort(-confs, kind="stable")
        cum_tp = np.cumsum(tps[order].astype(np.float64))
        prec_k = cum_tp / np.arange(1, confs.size + 1)
        rec_k = cum_tp / n_truth
        d_rec = np.diff(np.concatenate([[0.0], rec_k]))
        pr_auc = float(np.sum(d_rec * prec_k))

    fp_per_km2 = fp / area_km2 if area_km2 > 0 else float("nan")
    return ConfigMetrics(
        precision=precision, recall=recall, f1=f1, pr_auc=pr_auc,
        fp_per_km2=fp_per_km2, tp=tp, fp=fp, n_detections=len(labels),
    )


def _write_markdown(
    out_path: Path,
    metrics: dict[str, ConfigMetrics],
    *,
    n_scenes: int,
    seed_base: int,
    n_truth: int,
    area_km2: float,
    iou_thresh: float,
    conf_floor_pct: float,
    any_class: bool,
    raw_score_floor: float,
    verifier: PhysicsVerifier,
    elapsed_s: float,
) -> Path:
    """The ablation table plus the honesty preamble it must never travel without."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    match_rule = "any man-made class (localization only)" if any_class else "same class required"
    lines = [
        "# Detection-stack ablation — SYNTHETIC held-out scenes",
        "",
        f"Generated {datetime.now(tz=UTC).isoformat(timespec='seconds')} by "
        f"`scripts/eval_detector.py` in {elapsed_s:.0f} s.",
        "",
        "**Read this before quoting any number.** Every scene below is rendered by",
        "the SAGAR-NETRA physics scene simulator — no real sonar data is involved.",
        "The scenes are *held out* (seed base "
        f"{seed_base}, disjoint from detector training and calibration seeds), so no",
        "model in the stack has seen their speckle, seabed texture or targets; but",
        "synthetic targets are inevitably easier than real debris in real clutter,",
        "so treat these numbers as an upper bound and, above all, as a *relative*",
        "comparison between pipeline stages measured on identical raw detections.",
        "",
        "## Protocol",
        "",
        f"- {n_scenes} scenes, {n_truth} man-made truth boxes, "
        f"{area_km2:.4f} km² surveyed (per `geoscribe.build.survey_stats`).",
        "- Deployed detector stack ran ONCE per scene; the four configurations",
        "  re-score the SAME raw detections "
        f"(raw score >= {raw_score_floor}).",
        f"- TP: IoU >= {iou_thresh} with a man-made truth box, {match_rule};",
        "  one detection per truth (greedy, confidence-ranked). Hits on natural",
        "  targets (rock clusters) or background count as FP.",
        f"- Point metrics at calibrated confidence >= {conf_floor_pct:.0f}%; PR-AUC is",
        "  average precision swept over all confidence thresholds.",
        f"- Stage-2 verifier checkpoint: held-out AUC {verifier.val_auc:.3f}, "
        f"accuracy {verifier.val_accuracy:.3f} (own scene-level held-out split).",
        "",
        "## Results",
        "",
        f"| configuration | P@{conf_floor_pct:.0f} | R@{conf_floor_pct:.0f} "
        f"| F1@{conf_floor_pct:.0f} | PR-AUC | FP/km² | TP | FP | dets |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, label in CONFIGS:
        m = metrics[key]
        lines.append(
            f"| {label} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} "
            f"| {m.pr_auc:.3f} | {m.fp_per_km2:.2f} | {m.tp} | {m.fp} "
            f"| {m.n_detections} |"
        )
    lines += [
        "",
        "Reading the ladder: (b) applies the Stage-1 highlight/shadow multipliers,",
        "(c) adds the learned Stage-2 cue-vector multiplier, (d) adds the",
        "thin-detection persistence gate — the full deployed configuration.",
        "Confidence multipliers demote rather than delete, so `dets` is constant",
        "by construction; what moves is how many false alarms stay above the",
        "operator's confidence floor. When rows (c) and (d) coincide, no scored",
        "detection in this set was thinner than the persistence minimum — the",
        "ensemble consensus already suppresses 1-2-ping impulsive returns, and",
        "the temporal gate is the deployed backstop for single-model operation.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


@dataclass(frozen=True)
class Scene:
    """One rendered held-out scene, with its truth boxes and surveyed area."""

    index: int
    cfg: SceneConfig
    pre: PreprocessResult
    truths: list[Detection]
    area_km2: float


def iter_scenes(
    n_scenes: int,
    seed_base: int = SEED_BASE,
    *,
    shadow_pad_cols: int = 3,
    n_pings_range: tuple[int, int] = (500, 800),
    n_samples: int = 1024,
    slant_range_range: tuple[float, float] = (40.0, 60.0),
    altitude_range: tuple[float, float] = (6.0, 12.0),
):
    """Render the held-out scene set, one scene at a time.

    Shared by the ablation and the classical-baseline comparison so both are
    measured on *the same seabed*. The draw order off the seeded generator is
    part of the contract: change it and previously published tables stop being
    reproducible, which is why both callers go through this one function
    instead of keeping their own copy of the loop.
    """
    rng = np.random.default_rng(seed_base)
    for i in range(n_scenes):
        cfg = SceneConfig(
            n_pings=int(rng.integers(*n_pings_range)),
            n_samples=int(n_samples),
            slant_range=float(rng.uniform(*slant_range_range)),
            altitude=float(rng.uniform(*altitude_range)),
            seed=seed_base + i,
        )
        targets = random_targets(cfg, rng)
        pa, targets = make_scene(cfg, targets)
        pre = preprocess(pa)
        yield Scene(
            index=i,
            cfg=cfg,
            pre=pre,
            truths=_truth_boxes(pre, targets, cfg, shadow_pad_cols),
            area_km2=float(survey_stats(pre)["area_surveyed_sqkm"]),
        )


def run_eval(
    n_scenes: int = 8,
    seed_base: int = SEED_BASE,
    iou_thresh: float = 0.30,
    conf_floor_pct: float = 50.0,
    any_class: bool = False,
    out_path: str | Path = DEFAULT_OUT,
    detector: DetectorLike | None = None,
    *,
    raw_score_floor: float = 0.25,
    shadow_pad_cols: int = 3,
    n_pings_range: tuple[int, int] = (500, 800),
    n_samples: int = 1024,
    slant_range_range: tuple[float, float] = (40.0, 60.0),
    altitude_range: tuple[float, float] = (6.0, 12.0),
) -> dict[str, ConfigMetrics]:
    """Run the four-way ablation and write the markdown table.

    *detector* defaults to the deployed stack; tests inject a stub so the
    evaluation plumbing is exercised without loading model weights. Scene
    kwargs mirror :func:`scripts.fit_calibration.collect_pairs` so the
    held-out geometry spread matches the calibration set's.
    """
    start = time.perf_counter()
    detector = detector if detector is not None else deployed_detector()
    verifier = PhysicsVerifier.load()  # missing weights must fail loudly, not skip (c)
    gate_full = PhysicsGate()
    gate_no_temporal = _no_temporal_gate()

    labels: dict[str, list[tuple[float, bool]]] = {key: [] for key, _ in CONFIGS}
    n_truth = 0
    area_km2 = 0.0
    for scene in iter_scenes(
        n_scenes,
        seed_base,
        shadow_pad_cols=shadow_pad_cols,
        n_pings_range=n_pings_range,
        n_samples=n_samples,
        slant_range_range=slant_range_range,
        altitude_range=altitude_range,
    ):
        i, pre, truths = scene.index, scene.pre, scene.truths
        area_km2 += scene.area_km2
        n_truth += len(truths)

        # Detect ONCE; every configuration re-scores these exact boxes.
        detections = [
            d for d in detector.detect_tiles(pre.tiles) if d.score >= raw_score_floor
        ]
        scored: dict[str, list[tuple[Any, float]]] = {
            "a_raw": [
                (d, gate_full.confidence_pct(float(d.score), _NEUTRAL_GATE))
                for d in detections
            ],
            "b_gate": [
                (v.det, v.confidence_pct)
                for v in verify_detections(
                    detections, pre, gate=gate_no_temporal, use_verifier=False
                )
            ],
            "c_verifier": [
                (v.det, v.confidence_pct)
                for v in verify_detections(
                    detections, pre, gate=gate_no_temporal, verifier=verifier
                )
            ],
            "d_full": [
                (v.det, v.confidence_pct)
                for v in verify_detections(detections, pre, gate=gate_full, verifier=verifier)
            ],
        }
        for key, pairs in scored.items():
            labels[key].extend(_match_scene(pairs, truths, iou_thresh, any_class))
        print(
            f"scene {i + 1}/{n_scenes}: {len(detections)} detections, "
            f"{len(truths)} man-made truths"
        )

    metrics = {
        key: _config_metrics(labels[key], n_truth, area_km2, conf_floor_pct)
        for key, _ in CONFIGS
    }
    written = _write_markdown(
        Path(out_path), metrics,
        n_scenes=n_scenes, seed_base=seed_base, n_truth=n_truth, area_km2=area_km2,
        iou_thresh=iou_thresh, conf_floor_pct=conf_floor_pct, any_class=any_class,
        raw_score_floor=raw_score_floor, verifier=verifier,
        elapsed_s=time.perf_counter() - start,
    )
    print(f"\nwrote {written}\n")
    print(written.read_text(encoding="utf-8"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--conf-pct", type=float, default=50.0)
    parser.add_argument(
        "--any-class", action="store_true",
        help="count any man-made truth overlap as TP (localization-only credit)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_eval(
        n_scenes=args.scenes,
        seed_base=args.seed_base,
        iou_thresh=args.iou,
        conf_floor_pct=args.conf_pct,
        any_class=args.any_class,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
