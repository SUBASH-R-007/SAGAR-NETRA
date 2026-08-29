"""Train TridentNet Brain C (anomaly autoencoder) on clean seabed.

Usage:
    python scripts/train_anomaly.py [--scenes 8] [--epochs 12] [--out weights/anomaly.pt]
    python scripts/train_anomaly.py --klsg --out weights/anomaly_real.pt
    python scripts/train_anomaly.py --smoke     # tiny CPU run for CI

Synthetic backgrounds are target-free renders from the physics scene simulator,
run through the exact preprocessing chain inference uses, so the autoencoder
learns the statistics of what the detector actually sees.

``--klsg`` additionally mixes in **real** seabed from the KLSG corpus. Brain C
is an autoencoder: anything it reconstructs badly reads as anomalous, so an
autoencoder that has only ever seen simulated seabed flags real sand ripples,
rock fields and survey artefacts as targets. Measured on one real wreck image,
it produced 539 of 560 detections -- the third appearance of that failure class
after live-stream and image upload.

Real seabed is taken from the **borders** of KLSG chips. The corpus is
target-centred by construction, so peripheral bands are predominantly seabed
while the centre is the wreck; training on whole chips would teach the
autoencoder to reconstruct wrecks, which is precisely what it must not do.
Chips too small to leave a margin are skipped rather than sampled blindly.

Synthetic is mixed in rather than replaced: every published table depends on
synthetic behaviour, and an autoencoder retrained only on real texture would
regress there silently.
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

KLSG_ROOT = (
    REPO_ROOT / "data" / "datasets" / "klsg"
    / "SeabedObjects-Ship-and-Airplane-dataset-master"
)

#: A chip must be at least this wide and tall to yield a border band that is
#: plausibly clear of its centred target. Smaller chips are almost entirely
#: target and would poison the background set.
KLSG_MIN_DIM = 400

#: Border band thickness, matching the autoencoder's patch size so every band
#: can supply at least one crop.
KLSG_BAND = 128


def collect_real_background_tiles(
    root: Path = KLSG_ROOT, min_dim: int = KLSG_MIN_DIM, band: int = KLSG_BAND
) -> list[np.ndarray]:
    """Peripheral seabed strips from real KLSG imagery, L1-conditioned.

    Each eligible chip contributes four bands -- top, bottom, left, right --
    taken from outside the centred target. They run through the same
    preprocessing chain as inference, so the statistics the autoencoder learns
    are the statistics it will be scored against.
    """
    from sonar_core.parsers.base import load as load_survey

    if not root.is_dir():
        raise SystemExit(
            f"KLSG not found under {root}. "
            "Fetch it first:  python scripts/download_datasets.py --get klsg"
        )

    paths = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    tiles: list[np.ndarray] = []
    used = 0
    for path in paths:
        try:
            pa = load_survey(
                path, combined=False, slant_range_m=75.0, altitude_m=15.0,
                gain_normalized=True,
            )
            img = preprocess(pa).ground.starboard
        except Exception as exc:  # noqa: BLE001 - one bad chip must not stop training
            print(f"  skip {path.name}: {type(exc).__name__}")
            continue

        h, w = img.shape
        if min(h, w) < min_dim:
            continue
        used += 1
        tiles.extend([
            img[:band, :], img[h - band:, :],      # above and below the target
            img[:, :band], img[:, w - band:],      # either side of it
        ])

    print(f"real seabed: {used} chips >= {min_dim}px -> {len(tiles)} border bands")
    return tiles


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
    parser.add_argument(
        "--klsg", action="store_true",
        help="mix real KLSG seabed borders in with the synthetic backgrounds",
    )
    parser.add_argument(
        "--klsg-only", action="store_true",
        help="train on real seabed alone (ablation; regresses synthetic behaviour)",
    )
    args = parser.parse_args()

    if args.smoke:
        args.scenes, args.pings = 2, 300
        epochs = 3
    else:
        epochs = args.epochs

    start = time.perf_counter()
    tiles: list[np.ndarray] = []
    if not args.klsg_only:
        tiles.extend(collect_background_tiles(args.scenes, args.pings))
    if args.klsg or args.klsg_only:
        tiles.extend(collect_real_background_tiles())
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
