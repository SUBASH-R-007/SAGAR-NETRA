"""Turn KLSG chips into a weakly-labelled YOLO dataset for fine-tuning.

Usage:
    python scripts/build_klsg_dataset.py [--out data/datasets/klsg_yolo] [--val-frac 0.2]

KLSG ships 385 real shipwreck and 62 real aircraft side-scan images with
**folder-level class labels and no bounding boxes**, so a detector cannot be
trained on it directly. What it does have is a strong structural property: every
chip is cropped around its target. That makes a *weak* label recoverable.

The pseudo-box is measured per image rather than assumed. Pixels far from the
frame's median -- bright highlight and dark shadow alike, both being target
evidence -- are located, and the box spans their 5th-95th percentile extent.
Measured across the corpus that lands at a median of 0.79 x 0.73 of the frame,
with a real spread (p25 0.65, p75 0.86) and only 19% of chips exceeding 80% on
both axes. A constant whole-frame box would have taught the detector that a box
is always the size of its tile, which is degenerate at inference; a measured box
varies per image and does not.

**These labels are approximate and must be treated as such.** Any mAP computed
against them is not comparable to the synthetic numbers, and the honest question
this dataset can answer is narrower: after fine-tuning, does the detector reach
for the right *class* on held-out real wrecks more often than the 13.8% it
manages untrained?

Two properties of the corpus are worth stating because they work against us:
chips are arbitrary-orientation mosaics rather than nadir-first waterfalls, so
the shadow-direction prior the synthetic training teaches does not hold on them;
and 447 images is small. Both are reasons to measure synthetic regression after
fine-tuning rather than assume it away.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from tridentnet.classes import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
KLSG_ROOT = (
    REPO_ROOT / "data" / "datasets" / "klsg"
    / "SeabedObjects-Ship-and-Airplane-dataset-master"
)
DEFAULT_OUT = REPO_ROOT / "data" / "datasets" / "klsg_yolo"

#: Chips smaller than this in either axis are dropped: upscaled to the training
#: size they carry no detail the detector could learn from.
MIN_DIM = 96

#: Very large mosaics are downscaled before training. The detector sees 512-px
#: tiles, so retaining 3000-px chips only slows the run.
MAX_DIM = 1024

#: Deviation from the frame median, in standard deviations, that counts as
#: target evidence. Captures the bright highlight and the dark shadow alike.
DEVIATION_SIGMA = 1.2

#: Percentile span used for the box, trimming stray specks at either end.
LO_PCT, HI_PCT = 5.0, 95.0

#: Minimum pseudo-box side as a fraction of the frame, so a chip whose target
#: barely deviates cannot produce a degenerate sliver.
MIN_BOX_FRAC = 0.25

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def pseudo_box(image: np.ndarray) -> tuple[float, float, float, float] | None:
    """Normalized ``(cx, cy, w, h)`` spanning the target's evidence, or None."""
    med = float(np.median(image))
    sd = float(image.std()) + 1e-6
    mask = np.abs(image - med) > DEVIATION_SIGMA * sd
    if int(mask.sum()) < 50:
        return None

    ys, xs = np.nonzero(mask)
    h_img, w_img = image.shape
    y0, y1 = np.percentile(ys, LO_PCT), np.percentile(ys, HI_PCT)
    x0, x1 = np.percentile(xs, LO_PCT), np.percentile(xs, HI_PCT)

    w = max((x1 - x0) / w_img, MIN_BOX_FRAC)
    h = max((y1 - y0) / h_img, MIN_BOX_FRAC)
    cx = float(np.clip((x0 + x1) / 2 / w_img, w / 2, 1 - w / 2))
    cy = float(np.clip((y0 + y1) / 2 / h_img, h / 2, 1 - h / 2))
    return cx, cy, float(min(w, 1.0)), float(min(h, 1.0))


def klsg_items(root: Path = KLSG_ROOT) -> list[tuple[Path, str]]:
    """Every usable chip paired with the class its folder implies."""
    if not root.is_dir():
        raise SystemExit(
            f"KLSG not found under {root}. "
            "Fetch it first:  python scripts/download_datasets.py --get klsg"
        )
    out: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        out.append((path, "aircraft" if path.parent.name == "plane-real" else "wreck"))
    return out


def build(out_dir: Path = DEFAULT_OUT, val_frac: float = 0.2, seed: int = 0) -> dict:
    items = klsg_items()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = int(round(val_frac * len(items)))
    counts = {"train": 0, "val": 0}
    skipped = {"too_small": 0, "no_box": 0, "unreadable": 0}
    per_class: dict[str, int] = {}

    for rank, idx in enumerate(order):
        path, cls = items[int(idx)]
        split = "val" if rank < n_val else "train"
        try:
            im = Image.open(path).convert("L")
        except Exception:  # noqa: BLE001 - a corrupt chip must not stop the build
            skipped["unreadable"] += 1
            continue
        if min(im.size) < MIN_DIM:
            skipped["too_small"] += 1
            continue
        if max(im.size) > MAX_DIM:
            im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

        box = pseudo_box(np.asarray(im, dtype=np.float32))
        if box is None:
            skipped["no_box"] += 1
            continue

        stem = f"{cls}_{rank:04d}"
        im.save(out_dir / "images" / split / f"{stem}.png", optimize=True)
        cx, cy, w, h = box
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            f"{CLASS_INDEX[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8"
        )
        counts[split] += 1
        per_class[cls] = per_class.get(cls, 0) + 1

    (out_dir / "data.yaml").write_text(
        "# KLSG real side-scan chips, WEAKLY labelled.\n"
        "#\n"
        "# Boxes are measured from each chip's own highlight/shadow extent, not\n"
        "# annotated. They are approximate: mAP against them is NOT comparable to\n"
        "# the synthetic dataset's numbers. See scripts/build_klsg_dataset.py.\n"
        "#\n"
        "# AUGMENTATION WARNING: never mirror across columns - set fliplr: 0.0.\n"
        "# Acoustic shadows always extend down-range; a left-right flip produces\n"
        "# geometry no sonar can generate.\n"
        "#\n"
        "# Licence: released by the KLSG authors for academic use. Cite the paper.\n"
        "# Do not ship in a commercial build.\n"
        + yaml.safe_dump(
            {
                "path": str(out_dir).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(CLASS_NAMES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "out_dir": str(out_dir),
        "train": counts["train"],
        "val": counts["val"],
        "per_class": per_class,
        "skipped": skipped,
    }
    print(
        f"built {summary['train']} train + {summary['val']} val chips "
        f"({per_class}) into {out_dir}"
    )
    if any(skipped.values()):
        print(f"  skipped: {skipped}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.out, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
