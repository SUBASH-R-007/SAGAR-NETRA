"""Train the Stage-2 physics-feature verifier (confidence scoring & noise filtering).

Usage:
    python scripts/train_verifier.py [--scenes 10] [--seed 0] [--out weights/verifier.pkl]
    python scripts/train_verifier.py --smoke    # 2 tiny scenes, well under 2 min

Labels come free from the physics scene simulator: seeded man-made targets are
positives; rock clusters, clear-seabed boxes and ripple-band boxes are
negatives (see physicheck.verifier). Held-out accuracy/AUC are computed on
whole held-out scenes and stored in the checkpoint.

Smoke mode writes ``weights/verifier_smoke.pkl`` and NEVER touches
``weights/verifier.pkl`` — a weak smoke model must not shadow real weights
(and its mere presence would switch verify_detections into verifier mode).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib

from physicheck.verifier import DEFAULT_WEIGHTS_PATH, train_verifier

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_WEIGHTS_PATH = REPO_ROOT / "weights" / "verifier_smoke.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--smoke", action="store_true", help="2 tiny scenes (<2 min)")
    args = parser.parse_args()

    out = args.out
    train_kwargs: dict = {}
    if args.smoke:
        args.scenes = 2
        train_kwargs = {"n_pings_range": (360, 420), "n_samples": 512}
        # Never overwrite the real checkpoint from a smoke run.
        if Path(out).resolve() == DEFAULT_WEIGHTS_PATH.resolve():
            out = SMOKE_WEIGHTS_PATH

    start = time.perf_counter()
    out = train_verifier(
        n_scenes=args.scenes,
        seed=args.seed,
        out_path=out,
        progress=lambda name, frac: print(f"[{frac * 100:5.1f}%] {name}"),
        **train_kwargs,
    )
    payload = joblib.load(out)
    print(
        f"saved {out} in {time.perf_counter() - start:.0f}s | "
        f"held-out AUC {payload['val_auc']:.3f} | "
        f"held-out accuracy {payload['val_accuracy']:.3f} | "
        f"train/val samples {payload['n_train']}/{payload['n_val']}"
    )


if __name__ == "__main__":
    main()
