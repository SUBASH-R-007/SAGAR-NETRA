"""Fit the confidence temperature on a held-out synthetic validation set and
produce the reliability diagram (M4 acceptance artifact).

Usage:
    python scripts/fit_calibration.py [--weights weights/detector.pt]
        [--scenes 8] [--iou 0.30] [--out outputs/calibration] [--apply]

Validation scenes use seeds far from the training range, are rendered with
known ground truth, and a detection counts as *correct* when it overlaps a
same-class truth box at IoU >= the threshold. ``--apply`` writes the fitted
temperature into configs/physics.yaml in place (comment-preserving line edit).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from physicheck.calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_diagram,
)
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, make_scene
from tridentnet.data import _target_bbox, random_targets
from tridentnet.detector import Detection, Detector, box_iou

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_SEED_BASE = 9_000  # far away from training seeds


def collect_pairs(
    weights: Path, n_scenes: int, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    detector = Detector(weights=weights)
    scores: list[float] = []
    correct: list[bool] = []
    rng = np.random.default_rng(VAL_SEED_BASE)
    for i in range(n_scenes):
        cfg = SceneConfig(
            n_pings=int(rng.integers(500, 800)),
            n_samples=1024,
            slant_range=float(rng.uniform(40, 60)),
            altitude=float(rng.uniform(6, 12)),
            seed=VAL_SEED_BASE + i,
        )
        targets = random_targets(cfg, rng)
        pa, _ = make_scene(cfg, targets)
        pre = preprocess(pa)
        detections = detector.detect_tiles(pre.tiles)

        truths: list[Detection] = []
        for t in targets:
            gi = pre.ground
            bbox = _target_bbox(gi, t, cfg, shadow_pad_cols=2)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox
            truths.append(
                Detection(
                    side=t.side, ping0=int(r0), ping1=max(int(r1) - 1, int(r0)),
                    col0=int(c0), col1=max(int(c1) - 1, int(c0)), cls=t.cls, score=1.0,
                )
            )
        for det in detections:
            hit = any(
                det.cls == truth.cls
                and det.side == truth.side
                and box_iou(det, truth) >= iou_thresh
                for truth in truths
            )
            scores.append(float(det.score))
            correct.append(hit)
        print(f"scene {i + 1}/{n_scenes}: {len(detections)} detections, "
              f"{sum(correct)} correct so far")
    return np.asarray(scores), np.asarray(correct, dtype=float)


def apply_to_yaml(temperature: float) -> None:
    path = REPO_ROOT / "configs" / "physics.yaml"
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"^(\s*temperature:\s*)[0-9.]+", rf"\g<1>{temperature:.3f}", text, flags=re.M
    )
    if n != 1:
        raise SystemExit("could not find a unique 'temperature:' line in physics.yaml")
    path.write_text(new_text, encoding="utf-8")
    print(f"wrote temperature {temperature:.3f} into configs/physics.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "detector.pt")
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "calibration")
    parser.add_argument("--apply", action="store_true", help="write T into physics.yaml")
    args = parser.parse_args()

    scores, correct = collect_pairs(args.weights, args.scenes, args.iou)
    if scores.size < 20:
        raise SystemExit(
            f"only {scores.size} detections collected — train the detector first"
        )

    temperature = fit_temperature(scores, correct)
    ece_before = expected_calibration_error(scores, correct)
    calibrated = np.asarray(apply_temperature(scores, temperature))
    ece_after = expected_calibration_error(calibrated, correct)

    args.out.mkdir(parents=True, exist_ok=True)
    reliability_diagram(scores, correct, args.out / "reliability_raw.png",
                        title="Raw detector confidence")
    reliability_diagram(calibrated, correct, args.out / "reliability_calibrated.png",
                        title=f"Temperature-scaled (T={temperature:.2f})")
    (args.out / "calibration.txt").write_text(
        f"temperature={temperature:.4f}\nece_raw={ece_before:.4f}\n"
        f"ece_calibrated={ece_after:.4f}\nn_pairs={scores.size}\n"
        f"accuracy={correct.mean():.4f}\n",
        encoding="utf-8",
    )
    print(f"T = {temperature:.3f}  |  ECE {ece_before:.4f} -> {ece_after:.4f}  "
          f"({scores.size} pairs, accuracy {correct.mean():.2%})")
    print(f"diagrams in {args.out}")
    if args.apply:
        apply_to_yaml(temperature)


if __name__ == "__main__":
    main()
