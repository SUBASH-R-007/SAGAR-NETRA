"""Convert UATD (real annotated sonar) into a YOLO dataset in our taxonomy.

Usage:
    python scripts/build_uatd_dataset.py            # extract + inventory + convert
    python scripts/build_uatd_dataset.py --inventory-only

UATD — the Underwater Acoustic Target Detection dataset (figshare DOI
10.6084/m9.figshare.21331143.v3, **CC BY 4.0**) — is the one public corpus in
our list with real sonar imagery AND real human-annotated bounding boxes at
scale. That combination is what the detector has never seen: every box it has
trained on so far was synthetic.

Two properties must be stated up front, because they bound what training on
this can claim:

* **Domain.** UATD was collected with a Tritech Gemini 1200ik — a *multibeam
  forward-looking* sonar, not a side-scan towfish. Real acoustics, real
  speckle, real annotation noise; different imaging geometry. It teaches the
  detector what real sonar texture and real targets look like, and it says
  nothing about side-scan shadow geometry. KLSG covers that half.
* **Taxonomy.** UATD's ten classes are mapped into our frozen twelve below.
  Every mapping is written out with its justification; classes with no honest
  home are DROPPED and counted, never guessed. The converter hard-fails on an
  unmapped class name so a dataset revision cannot silently change the label
  space.

The official splits are preserved: ``Training`` becomes train, ``Test_1``
becomes val, and ``Test_2`` is deliberately **left out entirely** as an
untouched holdout for a final honest evaluation after all tuning is done.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from tridentnet.classes import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
UATD_ROOT = REPO_ROOT / "data" / "datasets" / "uatd"
DEFAULT_OUT = REPO_ROOT / "data" / "datasets" / "uatd_yolo"

#: UATD class -> ours. Each mapping is a judgement; the justification lives
#: here so it can be argued with rather than rediscovered.
CLASS_MAP: dict[str, str | None] = {
    "tyre": "tire",  # same object, different spelling
    "human body": "human_body",  # same object (mannequin targets in UATD)
    "plane": "aircraft",  # same object
    "cylinder": "cylinder_drum",  # a cylinder is the drum geometry
    "metal bucket": "cylinder_drum",  # open cylinder; nearest rigid class
    "cube": "container",  # boxy rigid body; container is our boxy class
    "ball": "mine_like",  # spherical proud target = the mine-like geometry
    "circle cage": "ghost_net",  # cages are fishing gear; ghost_net is our
    "square cage": "ghost_net",  # abandoned-fishing-gear class
    "rov": None,  # an ROV is none of our classes; teaching it as any of them
    # would be a lie the verifier cannot undo. DROPPED, counted.
}

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

#: Official archive split -> our split. Test_2 is reserved untouched.
SPLIT_MAP = {"Training": "train", "Test_1": "val"}


def extract_archives(root: Path = UATD_ROOT) -> None:
    """Unpack the figshare archive and any nested zips, idempotently."""
    for _round in range(3):  # archive-of-archives: a few passes flatten it
        zips = [p for p in root.rglob("*.zip") if not (p.parent / (p.stem + ".ok")).exists()]
        if not zips:
            return
        for z in zips:
            print(f"extracting {z.relative_to(root)}")
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent)
                (z.parent / (z.stem + ".ok")).write_text("extracted", encoding="utf-8")
            except zipfile.BadZipFile:
                print(f"  WARNING: {z.name} is not a valid zip; skipping")


def _voc_boxes(xml_path: Path) -> list[tuple[str, float, float, float, float]] | None:
    """Parse one VOC annotation: [(class, xmin, ymin, xmax, ymax)] or None."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    out = []
    for obj in root.iter("object"):
        name_el = obj.find("name")
        box = obj.find("bndbox")
        if name_el is None or box is None:
            continue
        try:
            coords = tuple(
                float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError:
            continue
        out.append((str(name_el.text).strip().lower(), *coords))
    return out


def find_pairs(root: Path = UATD_ROOT) -> dict[str, list[tuple[Path, Path]]]:
    """(image, annotation) pairs per official split, discovered not assumed.

    Two mistakes an earlier draft made, kept as constraints here:

    * Split membership is a *substring* test on each path component — the
      archive's directories are ``UATD_Training``, not ``Training``, so exact
      component equality matched nothing.
    * Annotations are resolved **relative to each image** (the sibling
      ``annotations/`` directory, then the image's own directory). A global
      stem->xml map would collide across splits — Training and Test_1 both
      number their files ``00001`` upward — silently pairing images with the
      other split's boxes.
    """
    pairs: dict[str, list[tuple[Path, Path]]] = {}
    for img in root.rglob("*"):
        if img.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg", ".tif"}:
            continue
        parts = [part.lower() for part in img.parts]
        if any("test_2" in part for part in parts):
            continue  # reserved holdout, untouched on purpose
        split = next(
            (ours for official, ours in SPLIT_MAP.items()
             if any(official.lower() in part for part in parts)),
            None,
        )
        if split is None:
            continue
        candidates = (
            img.parent.parent / "annotations" / f"{img.stem}.xml",
            img.with_suffix(".xml"),
        )
        xml = next((c for c in candidates if c.is_file()), None)
        if xml is None:
            continue
        pairs.setdefault(split, []).append((img, xml))
    for split in pairs:
        pairs[split].sort()
    return pairs


def inventory(pairs: dict[str, list[tuple[Path, Path]]]) -> Counter:
    """Print what is actually in the archive; return the raw class histogram."""
    classes: Counter = Counter()
    for split, items in sorted(pairs.items()):
        print(f"  {split}: {len(items)} image/annotation pairs")
    for items in pairs.values():
        for _img, xml in items:
            boxes = _voc_boxes(xml) or []
            classes.update(name for name, *_ in boxes)
    print("  raw class histogram:")
    for name, count in classes.most_common():
        mapped = CLASS_MAP.get(name, "!! UNMAPPED !!")
        print(f"    {name:14s} {count:6d}  -> {mapped}")
    unmapped = [n for n in classes if n not in CLASS_MAP]
    if unmapped:
        raise SystemExit(
            f"unmapped UATD classes {unmapped}: extend CLASS_MAP deliberately "
            "rather than letting a silent guess relabel the dataset"
        )
    return classes


def build(out_dir: Path = DEFAULT_OUT) -> dict:
    extract_archives()
    pairs = find_pairs()
    if not pairs.get("train"):
        raise SystemExit(
            f"no Training pairs found under {UATD_ROOT} - is the download complete?"
        )
    inventory(pairs)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0}
    dropped_boxes = 0
    kept_boxes: Counter = Counter()
    for split, items in pairs.items():
        for img_path, xml_path in items:
            boxes = _voc_boxes(xml_path)
            if not boxes:
                continue
            try:
                with Image.open(img_path) as im:
                    w_img, h_img = im.size
                    # BMP is uncompressed; grayscale PNG shrinks it without
                    # touching a pixel value. compress_level=3, NOT
                    # optimize=True: optimize forces level 9 plus extra
                    # filter passes and turned 8,400 conversions into a
                    # two-hour job for a few percent of disk.
                    out_img = out_dir / "images" / split / f"{img_path.stem}.png"
                    im.convert("L").save(out_img, compress_level=3)
            except Exception:  # noqa: BLE001 - skip corrupt frames, keep building
                continue

            lines = []
            for name, x0, y0, x1, y1 in boxes:
                ours = CLASS_MAP[name]
                if ours is None:
                    dropped_boxes += 1
                    continue
                # Clamp to the frame; some UATD boxes overhang by a pixel.
                x0, x1 = max(x0, 0.0), min(x1, float(w_img))
                y0, y1 = max(y0, 0.0), min(y1, float(h_img))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    dropped_boxes += 1
                    continue
                cx, cy = (x0 + x1) / 2 / w_img, (y0 + y1) / 2 / h_img
                bw, bh = (x1 - x0) / w_img, (y1 - y0) / h_img
                lines.append(
                    f"{CLASS_INDEX[ours]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
                kept_boxes[ours] += 1
            if not lines:
                out_img.unlink(missing_ok=True)  # image whose only box was dropped
                continue
            (out_dir / "labels" / split / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            written[split] += 1

    (out_dir / "data.yaml").write_text(
        "# UATD (figshare 10.6084/m9.figshare.21331143.v3, CC BY 4.0) in our taxonomy.\n"
        "#\n"
        "# REAL sonar imagery with REAL human-annotated boxes - the only corpus in\n"
        "# the collection with both. DOMAIN CAVEAT: multibeam forward-looking sonar\n"
        "# (Tritech Gemini 1200ik), not side-scan; it teaches real-sonar texture and\n"
        "# real target shapes, not side-scan shadow geometry.\n"
        "#\n"
        "# Official Training -> train, Test_1 -> val. Test_2 is reserved untouched\n"
        "# as a final holdout. 'rov' boxes are dropped (no honest class for them).\n"
        "# See scripts/build_uatd_dataset.py for the full mapping rationale.\n"
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
        "written": written,
        "kept_boxes": dict(kept_boxes),
        "dropped_boxes": dropped_boxes,
    }
    print(f"\nbuilt {written['train']} train + {written['val']} val images -> {out_dir}")
    print(f"kept boxes per class: {dict(kept_boxes.most_common())}")
    print(f"dropped boxes (rov/degenerate): {dropped_boxes}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        extract_archives()
        inventory(find_pairs())
    else:
        build(args.out)


if __name__ == "__main__":
    main()
