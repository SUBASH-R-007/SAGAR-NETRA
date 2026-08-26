"""Train TridentNet Brain C (anomaly autoencoder) on clean synthetic seabed.

Usage:
    python scripts/train_anomaly.py [--scenes 8] [--epochs 12] [--out weights/anomaly.pt]
    python scripts/train_anomaly.py --smoke     # tiny CPU run for CI

Backgrounds are target-free renders from the physics scene simulator, run
through the exact preprocessing chain inference uses, so the autoencoder
learns the statistics of what the detector actually sees.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, make_scene
from tridentnet.anomaly import DEFAULT_WEIGHTS_PATH, train_anomaly

REPO_ROOT = Path(__file__).resolve().parents[1]


def collect_background_tiles(
    n_scenes: int, n_pings: int, seed: int = 100
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    tiles: list[np.ndarray] = []
    for i in range(n_scenes):
        cfg = SceneConfig(
            n_pings=n_pings,
            n_samples=1024,
            slant_range=float(rng.uniform(40, 60)),
            altitude=float(rng.uniform(6, 12)),
            seed=seed + i,
        )
        pa, _ = make_scene(cfg, targets=[])
        pre = preprocess(pa)
        tiles.extend(t.image for t in pre.tiles)
        print(f"scene {i + 1}/{n_scenes}: {len(pre.tiles)} tiles")
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--pings", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run (<5 min)")
    args = parser.parse_args()

    if args.smoke:
        args.scenes, args.pings = 2, 300
        epochs = 3
    else:
        epochs = args.epochs

    start = time.perf_counter()
    tiles = collect_background_tiles(args.scenes, args.pings)
    config = {"epochs": epochs} if epochs is not None else None
    out = train_anomaly(
        tiles,
        out_path=args.out,
        config=config,
        progress=lambda name, frac: print(f"[{frac * 100:5.1f}%] {name}"),
    )
    print(f"saved {out} in {time.perf_counter() - start:.0f}s "
          f"({len(tiles)} background tiles)")


if __name__ == "__main__":
    main()
