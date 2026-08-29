"""Acceptance gates for a real-data-trained detector checkpoint.

Usage:
    python scripts/eval_gates_real.py [--candidate weights/detector_real.pt]
        [--baseline weights/detector.pt] [--out docs/real_training.md]

A checkpoint trained on the UATD+KLSG+synthetic mix must EARN its way into the
deployed stack. This script runs both the candidate and the currently deployed
baseline through identical measurements, so every claim about the new weights
is a same-harness comparison rather than a memory of an old number:

1. **Per-source val mAP50** on synth_xl / klsg_yolo / uatd_yolo. The UATD row
   is the first mAP in this project computed against real human-drawn boxes.
   The synth row is the regression gate: the deployed pipeline's published
   behaviour rests on synthetic performance, so a collapse there fails the
   candidate outright.
2. **KLSG class-reach.** On the KLSG *val* chips: how often does each model's
   top prediction land in {wreck, aircraft} - the folder-level truth. This is
   the number that was 13.8% for the untrained detector over the full corpus;
   computing it here for baseline AND candidate on the same split makes the
   before/after honest.
3. **Demo-survey sanity.** The bundled survey processed with the candidate as
   Brain A: contact count and class spread, next to the baseline's. Not a
   pass/fail metric - a smoke check that the new weights do not detonate the
   demo everyone will watch.

The script only reports. The decision to swap weights stays with a human,
because gate 1 and gate 2 can genuinely trade against each other.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "data" / "datasets"
DEFAULT_OUT = REPO_ROOT / "docs" / "real_training.md"

VAL_SETS = ("synth_xl", "klsg_yolo", "uatd_yolo")
WRECK_LIKE = {"wreck", "aircraft"}

#: Candidate must stay within this much mAP50 of the baseline on synthetic.
SYNTH_REGRESSION_TOLERANCE = 0.10


def _val_map50(weights: Path, data_yaml: Path) -> float:
    """One Ultralytics val run; returns mAP50."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), workers=0, plots=False, verbose=False
    )
    return float(metrics.box.map50)


def _klsg_reach(weights: Path) -> tuple[int, int]:
    """(hits, total): val chips whose top-1 prediction is wreck-like."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    val_dir = DATASETS / "klsg_yolo" / "images" / "val"
    chips = sorted(val_dir.iterdir())
    hits = 0
    for chip in chips:
        results = model.predict(str(chip), verbose=False, conf=0.10)
        best_cls, best_conf = None, 0.0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_cls = r.names[int(box.cls)]
        if best_cls in WRECK_LIKE:
            hits += 1
    return hits, len(chips)


def _demo_contacts(weights: Path) -> dict:
    """Process the bundled survey with these weights alone as Brain A."""
    import tempfile

    from api.db import ContactRepo
    from api.processing import process_survey
    from tridentnet.deep_ensemble import build_brain_a

    rel = str(weights.relative_to(REPO_ROOT)) if weights.is_absolute() else str(weights)

    class BrainAOnly:
        def __init__(self) -> None:
            self.a = build_brain_a({"ensemble_weights": [rel]})

        def detect_tiles(self, tiles, progress=None):
            return self.a.detect_tiles(tiles, progress=progress)

    tmp = Path(tempfile.mkdtemp(prefix="gates_"))
    repo = ContactRepo(tmp / "contacts.db")
    summary = process_survey(
        REPO_ROOT / "data" / "samples" / "survey_alpha.xtf",
        repo,
        detector_factory=BrainAOnly,
        output_root=tmp / "out",
    )
    contacts = repo.query(survey=summary["survey"], limit=200)
    return {
        "n_contacts": summary["n_contacts"],
        "classes": dict(Counter(c.cls for c in contacts).most_common()),
    }


def run(candidate: Path, baseline: Path, out_path: Path) -> None:
    start = time.perf_counter()
    for w in (candidate, baseline):
        if not w.is_file():
            raise SystemExit(f"missing weights: {w}")

    rows: dict[str, dict[str, float]] = {}
    for name in VAL_SETS:
        data_yaml = DATASETS / name / "data.yaml"
        if not data_yaml.is_file():
            raise SystemExit(f"missing dataset {data_yaml}")
        print(f"validating on {name} ...")
        rows[name] = {
            "baseline": _val_map50(baseline, data_yaml),
            "candidate": _val_map50(candidate, data_yaml),
        }

    print("measuring KLSG class-reach ...")
    reach = {
        "baseline": _klsg_reach(baseline),
        "candidate": _klsg_reach(candidate),
    }
    print("processing the demo survey with each ...")
    demo = {
        "baseline": _demo_contacts(baseline),
        "candidate": _demo_contacts(candidate),
    }

    synth_drop = rows["synth_xl"]["baseline"] - rows["synth_xl"]["candidate"]
    synth_ok = synth_drop <= SYNTH_REGRESSION_TOLERANCE
    reach_b = reach["baseline"][0] / max(reach["baseline"][1], 1)
    reach_c = reach["candidate"][0] / max(reach["candidate"][1], 1)

    lines = [
        "# Real-data training - acceptance gates",
        "",
        f"Generated {datetime.now(tz=UTC).isoformat(timespec='seconds')} by "
        f"`scripts/eval_gates_real.py` in {time.perf_counter() - start:.0f} s.",
        "",
        f"Candidate `{candidate.name}` vs deployed baseline `{baseline.name}`,",
        "measured through one harness in one run - no numbers quoted from memory.",
        "",
        "## Gate 1 - per-source val mAP50",
        "",
        "| val set | boxes | baseline | candidate |",
        "|---|---|---|---|",
        f"| synth_xl | synthetic truth | {rows['synth_xl']['baseline']:.3f} "
        f"| {rows['synth_xl']['candidate']:.3f} |",
        f"| klsg_yolo | weak (measured) | {rows['klsg_yolo']['baseline']:.3f} "
        f"| {rows['klsg_yolo']['candidate']:.3f} |",
        f"| uatd_yolo | **real, annotated** | {rows['uatd_yolo']['baseline']:.3f} "
        f"| {rows['uatd_yolo']['candidate']:.3f} |",
        "",
        f"Synthetic regression: {synth_drop:+.3f} against a tolerance of "
        f"{SYNTH_REGRESSION_TOLERANCE} - **{'PASS' if synth_ok else 'FAIL'}**.",
        "",
        "The KLSG row is against weak boxes and is reported for completeness, not",
        "compared against anything: its labels are measured approximations.",
        "",
        "## Gate 2 - KLSG class-reach (top-1 in {wreck, aircraft})",
        "",
        "| model | hits | share |",
        "|---|---|---|",
        f"| baseline | {reach['baseline'][0]} / {reach['baseline'][1]} | {100 * reach_b:.1f}% |",
        f"| candidate | {reach['candidate'][0]} / {reach['candidate'][1]} | {100 * reach_c:.1f}% |",
        "",
        "## Gate 3 - demo survey smoke check",
        "",
        f"- baseline: {demo['baseline']['n_contacts']} contacts, "
        f"classes {demo['baseline']['classes']}",
        f"- candidate: {demo['candidate']['n_contacts']} contacts, "
        f"classes {demo['candidate']['classes']}",
        "",
        "## Verdict",
        "",
        f"- Gate 1 (synthetic within tolerance): {'PASS' if synth_ok else 'FAIL'}",
        f"- Gate 2 (class-reach improved): "
        f"{'PASS' if reach_c > reach_b else 'FAIL'} ({100 * reach_b:.1f}% -> {100 * reach_c:.1f}%)",
        "- Gate 3 is judgement, not arithmetic - read the class spread above.",
        "",
        "Swapping the deployed weights is a human decision on these numbers;",
        "this script never copies a checkpoint anywhere.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path}\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path,
                        default=REPO_ROOT / "weights" / "detector_real.pt")
    parser.add_argument("--baseline", type=Path,
                        default=REPO_ROOT / "weights" / "detector.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.candidate, args.baseline, args.out)


if __name__ == "__main__":
    main()
