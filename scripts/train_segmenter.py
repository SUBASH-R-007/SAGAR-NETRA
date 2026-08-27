"""Train TridentNet Brain B (net/rope U-Net segmenter) on synthetic mask chips.

Usage:
    python scripts/train_segmenter.py [--scenes 12] [--epochs 20] [--chip 256]
                                      [--data DIR] [--out weights/segmenter.pt]
    python scripts/train_segmenter.py --smoke    # tiny CPU run (<5 min)

The dataset comes free from the physics scene simulator: seeded targets carry
exact geometry, so pixel masks need no annotation (see tridentnet.segdata).
Chips are cut from the same enhanced ground imagery inference tiles use.

Smoke mode writes ``weights/segmenter_smoke.pt`` and NEVER touches
``weights/segmenter.pt`` — a weak smoke model must not shadow real weights.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from tridentnet.segdata import build_mask_dataset
from tridentnet.segmenter import DEFAULT_WEIGHTS_PATH, train_segmenter

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_WEIGHTS_PATH = REPO_ROOT / "weights" / "segmenter_smoke.pt"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "segmenter_synth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--chip", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=None,
                        help="dataset dir (reused when it already holds chips)")
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the dataset even if chips already exist")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run (<5 min)")
    args = parser.parse_args()

    epochs = args.epochs
    build_kwargs: dict = {}
    out = args.out
    data_dir = args.data
    if args.smoke:
        args.scenes, args.chip = 3, 160
        epochs = 40 if epochs is None else epochs
        build_kwargs = {"n_pings_range": (240, 300), "n_samples": 384}
        if data_dir is None:
            data_dir = REPO_ROOT / "data" / "segmenter_synth_smoke"
        # Never overwrite the real checkpoint from a smoke run.
        if Path(out).resolve() == DEFAULT_WEIGHTS_PATH.resolve():
            out = SMOKE_WEIGHTS_PATH
    elif data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    start = time.perf_counter()
    have_chips = any((data_dir / "images" / "train").glob("*.png")) if data_dir.exists() else False
    if args.rebuild or not have_chips:
        build_mask_dataset(
            data_dir,
            n_scenes=args.scenes,
            chip=args.chip,
            seed=args.seed,
            progress=lambda name, frac: print(f"[{frac * 100:5.1f}%] dataset: {name}"),
            **build_kwargs,
        )
    else:
        print(f"reusing existing chips in {data_dir} (pass --rebuild to regenerate)")

    config = {"epochs": epochs} if epochs is not None else None
    out = train_segmenter(
        data_dir,
        out_path=out,
        config=config,
        seed=args.seed,
        progress=lambda name, frac: print(f"[{frac * 100:5.1f}%] {name}"),
    )
    payload = torch.load(out, map_location="cpu", weights_only=False)
    print(
        f"saved {out} in {time.perf_counter() - start:.0f}s | "
        f"val Dice {payload['val_dice']:.3f} | "
        f"train loss {payload['train_losses'][0]:.3f} -> {payload['train_losses'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
