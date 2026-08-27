"""Active-learning flywheel export (blueprint N-08): review labels -> YOLO set.

Every operator verdict in the DRISHTI console is a free training label: a
*confirmed* contact is a vetted positive with a class and a box; a *rejected*
contact is a vetted **hard negative** — imagery the detector fired on but an
expert called seabed. This script turns the review trail in
``data/contacts.db`` into a YOLO-format dataset of chips + labels.

Why the chips are re-rendered instead of cropped from the stored waterfall:
``outputs/surveys/<stem>/waterfall.png`` is the port+starboard *combined*
image (port mirrored for display) — cropping it would both break the frozen
nadir-first column convention (column 0 at nadir, shadows toward increasing
column; the training data must never contain mirrored geometry) and bake in
whatever enhancement was tuned for on-screen viewing. Instead the source
survey file (``surveys.source_path``) is re-parsed and re-preprocessed
(cached once per survey within a run) and the chip is cropped from the
enhanced per-side :class:`GroundImage` — the exact detector-input domain the
contact's stored pixel box refers to.

Label semantics:

* confirmed -> ``labels/<id>.txt`` with one normalized YOLO box using the
  frozen class id from :data:`tridentnet.classes.CLASS_TO_ID`;
  ``unknown_anomaly`` (an ensemble-level open-set label, not a detector
  class) is skipped with a note — an empty label file would poison training
  by teaching the detector that a confirmed object is background.
* rejected -> chip written with an **empty** label file: a first-class hard
  negative for exactly the imagery that currently fools the detector.

Continue training on the export with:

    python scripts/train_detector.py --data <out>/data.yaml

(train_detector.py already enforces the sonar augmentation rules: never
mirror across columns — ``fliplr: 0.0``.)

Usage:
    python scripts/export_review_labels.py [--db PATH] [--out PATH] [--chip N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from api.db import ContactRepo
from geoscribe.contact import Contact
from sonar_core.parsers.base import load as load_survey
from sonar_core.preprocess.pipeline import PreprocessResult, preprocess
from tridentnet.classes import CLASS_NAMES, CLASS_TO_ID

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DB: Path = REPO_ROOT / "data" / "contacts.db"
DEFAULT_OUT: Path = REPO_ROOT / "data" / "datasets" / "review_export"
#: Chip side (pixels) — matches the detector's tile size so exported samples
#: sit in the same scale regime as inference-time tiles.
DEFAULT_CHIP: int = 512


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"contacts database (default {DEFAULT_DB})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"export directory for images/, labels/, data.yaml (default {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--chip", type=int, default=DEFAULT_CHIP,
        help=f"chip side in pixels, shrunk to the image when it is smaller "
        f"(default {DEFAULT_CHIP})",
    )
    return parser


def crop_chip(
    image: np.ndarray, ping0: int, ping1: int, col0: int, col1: int, chip: int
) -> tuple[np.ndarray, int, int]:
    """Crop a chip centred on the (inclusive) box, clamped inside the image.

    The window is ``min(chip, extent)`` per axis so short test surveys and
    narrow swaths still export (YOLO training letterboxes any chip size), and
    it is shifted — never padded — to stay inside the image: every exported
    pixel is a real ground sample, the same rule the tiler follows. Returns
    ``(chip_view, row_origin, col_origin)``.
    """
    n_rows, n_cols = image.shape
    ch = min(chip, n_rows)
    cw = min(chip, n_cols)
    r0 = int(round((ping0 + ping1 + 1) / 2 - ch / 2))
    c0 = int(round((col0 + col1 + 1) / 2 - cw / 2))
    r0 = max(0, min(r0, n_rows - ch))
    c0 = max(0, min(c0, n_cols - cw))
    return image[r0 : r0 + ch, c0 : c0 + cw], r0, c0


def yolo_line(
    class_id: int,
    ping0: int, ping1: int, col0: int, col1: int,
    row_origin: int, col_origin: int, chip_h: int, chip_w: int,
) -> str:
    """One YOLO label line for an inclusive global box inside a chip.

    x runs along ground-range columns and y along pings — the ground-image
    axis convention — and the box is clamped to the chip before normalizing
    (a box wider than the chip window trains on its visible part).
    """
    x0 = max(col0 - col_origin, 0)
    x1 = min(col1 - col_origin, chip_w - 1)
    y0 = max(ping0 - row_origin, 0)
    y1 = min(ping1 - row_origin, chip_h - 1)
    cx = (x0 + x1 + 1) / 2 / chip_w
    cy = (y0 + y1 + 1) / 2 / chip_h
    bw = (x1 - x0 + 1) / chip_w
    bh = (y1 - y0 + 1) / chip_h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_chip_png(chip: np.ndarray, path: Path) -> None:
    """Enhanced float [0,1] chip (NaN beyond swath) -> 8-bit grayscale PNG.

    NaN maps to 0 — the blanked-water-column level — so out-of-swath fill
    reads as dark seabed absence, matching what the detector sees at
    inference time (:func:`tridentnet.detector._tile_to_uint8_rgb`).
    """
    arr = np.clip(np.nan_to_num(chip, nan=0.0), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="L").save(path)


def write_data_yaml(out_dir: Path) -> Path:
    """``data.yaml`` with the frozen class map; same layout and no-mirror
    warning as the synthetic dataset builder, so both sets are drop-in
    interchangeable for ``train_detector.py --data``."""
    lines = [
        "# SAGAR-NETRA review-label export (active-learning flywheel, N-08).",
        "# generated by scripts/export_review_labels.py from data/contacts.db.",
        "#",
        "# AUGMENTATION WARNING: never mirror across columns — set fliplr: 0.0.",
        "# Both sides are stored nadir-first (column 0 at nadir) and acoustic",
        "# shadows always extend toward increasing column; a left-right flip",
        "# puts shadows up-range of their highlights, which no sonar can produce.",
        f"path: {out_dir.as_posix()}",
        "train: images",
        "val: images",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _reviewed_contacts(repo: ContactRepo) -> list[Contact]:
    """Contacts whose *current* status is confirmed or rejected. The
    append-only review log is the audit trail; the contact row carries the
    operator's final verdict, which is what trains the detector."""
    confirmed = repo.query(review="confirmed", limit=100_000)
    rejected = repo.query(review="rejected", limit=100_000)
    return [*confirmed, *rejected]


def export(db: Path, out: Path, chip: int) -> int:
    """Run the export; returns a process exit code (0 even when empty —
    an unreviewed database is not an error, just an empty flywheel)."""
    repo = ContactRepo(db)
    try:
        surveys = {s["name"]: s for s in repo.surveys()}
        reviewed_at = {entry["contact_id"]: entry["at"] for entry in repo.review_log()}
        contacts = _reviewed_contacts(repo)
    finally:
        repo.close()

    images_dir = out / "images"
    labels_dir = out / "labels"
    cache: dict[str, PreprocessResult] = {}
    rows: list[tuple[str, str, str, str, str]] = []
    n_boxes = n_negatives = n_skipped = 0

    for contact in contacts:
        status = contact.review.value
        when = reviewed_at.get(contact.id, "-")
        survey_row = surveys.get(contact.survey)
        if survey_row is None:
            rows.append((contact.id, contact.cls, status, when, "SKIP: no survey record"))
            n_skipped += 1
            continue
        source = Path(survey_row["source_path"] or "")
        if not source.is_file():
            rows.append((contact.id, contact.cls, status, when, f"SKIP: source gone ({source})"))
            n_skipped += 1
            continue
        if status == "confirmed" and contact.cls not in CLASS_TO_ID:
            # unknown_anomaly and friends: no detector class id exists, and an
            # empty label would mark a confirmed object as background.
            rows.append((contact.id, contact.cls, status, when, "SKIP: no detector class id"))
            n_skipped += 1
            continue

        if contact.survey not in cache:  # one parse+preprocess per survey per run
            cache[contact.survey] = preprocess(load_survey(source))
        pre = cache[contact.survey]

        px = contact.pixel
        side_image = pre.ground.side(px.side)
        chip_view, r0, c0 = crop_chip(side_image, px.ping0, px.ping1, px.col0, px.col1, chip)
        save_chip_png(chip_view, images_dir / f"{contact.id}.png")

        label_path = labels_dir / f"{contact.id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if status == "confirmed":
            line = yolo_line(
                CLASS_TO_ID[contact.cls],
                px.ping0, px.ping1, px.col0, px.col1,
                r0, c0, *chip_view.shape,
            )
            label_path.write_text(line + "\n", encoding="utf-8")
            rows.append((contact.id, contact.cls, status, when, f"box (id {CLASS_TO_ID[contact.cls]})"))
            n_boxes += 1
        else:
            label_path.write_text("", encoding="utf-8")  # hard negative
            rows.append((contact.id, contact.cls, status, when, "hard negative (empty label)"))
            n_negatives += 1

    out.mkdir(parents=True, exist_ok=True)
    yaml_path = write_data_yaml(out)

    header = ("contact", "class", "review", "reviewed at", "exported as")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(5)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)))
    print(
        f"\nexported {n_boxes} labelled box(es), {n_negatives} hard negative(s), "
        f"{n_skipped} skipped -> {out}"
    )
    print(f"\n{yaml_path}:\n{yaml_path.read_text(encoding='utf-8')}")
    print(f"continue training with: python scripts/train_detector.py --data {yaml_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return export(Path(args.db), Path(args.out), int(args.chip))


if __name__ == "__main__":
    raise SystemExit(main())
