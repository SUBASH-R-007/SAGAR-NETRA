"""Build the web-sized figures used by README.md (and usable as slide assets).

Usage:
    python scripts/make_docs_images.py

Reads the full-resolution artifacts produced by the demo/gallery/calibration
scripts under ``outputs/`` and writes downscaled, labelled composites to
``docs/images/``. Regenerate after re-running the pipeline so the figures in
the README always match the committed numbers.

Prerequisites (each printed as a hint if missing):
    python scripts/preprocess_gallery.py     -> outputs/gallery/*.png
    python scripts/demo.py                   -> outputs/surveys/<stem>/evidence/*
    python scripts/detect_demo.py            -> outputs/detections.png
    python scripts/fit_calibration.py        -> outputs/calibration/*.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "images"
OUTPUTS = REPO_ROOT / "outputs"

#: Composite chrome, matching the console's Government-of-India skin.
PAPER = (242, 244, 247)
INK = (27, 39, 51)
NAVY = (21, 56, 116)
LABEL_H = 26
GAP = 10
TARGET_W = 1100


def _load(path: Path) -> Image.Image | None:
    if not path.is_file():
        print(f"  missing: {path.relative_to(REPO_ROOT)}")
        return None
    return Image.open(path).convert("RGB")


def _fit(img: Image.Image, width: int, max_height: int | None = None) -> Image.Image:
    scale = width / img.width
    height = int(img.height * scale)
    if max_height is not None and height > max_height:
        scale = max_height / img.height
        width, height = int(img.width * scale), max_height
    return img.resize((width, height), Image.LANCZOS)


def _labelled_row(
    panels: list[tuple[str, Image.Image]], width: int = TARGET_W, max_h: int = 460
) -> Image.Image:
    """Panels side by side, each with a navy caption bar above it."""
    n = len(panels)
    panel_w = (width - GAP * (n - 1)) // n
    scaled = [(caption, _fit(img, panel_w, max_h)) for caption, img in panels]
    panel_h = max(img.height for _, img in scaled)

    canvas = Image.new("RGB", (width, panel_h + LABEL_H), PAPER)
    draw = ImageDraw.Draw(canvas)
    x = 0
    for caption, img in scaled:
        draw.rectangle([x, 0, x + panel_w, LABEL_H], fill=NAVY)
        draw.text((x + 8, 7), caption, fill=(255, 255, 255))
        canvas.paste(img, (x + (panel_w - img.width) // 2, LABEL_H))
        x += panel_w + GAP
    return canvas


def _labelled_stack(
    panels: list[tuple[str, Image.Image]], width: int = TARGET_W
) -> Image.Image:
    """Panels stacked vertically at full width — keeps in-image captions legible."""
    scaled = [(caption, _fit(img, width)) for caption, img in panels]
    height = sum(img.height + LABEL_H for _, img in scaled) + GAP * (len(scaled) - 1)

    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for caption, img in scaled:
        draw.rectangle([0, y, width, y + LABEL_H], fill=NAVY)
        draw.text((8, y + 7), caption, fill=(255, 255, 255))
        canvas.paste(img, (0, y + LABEL_H))
        y += LABEL_H + img.height + GAP
    return canvas


def build_pipeline_figure() -> None:
    """L1: raw slant-range waterfall vs the detector-ready ground-range image."""
    raw = _load(OUTPUTS / "gallery" / "01_raw_waterfall.png")
    ground = _load(OUTPUTS / "gallery" / "03_ground_range.png")
    enhanced = _load(OUTPUTS / "gallery" / "04_enhanced.png")
    if not (raw and ground and enhanced):
        return
    fig = _labelled_row(
        [
            ("1  RAW  slant range, TVG banding, water column", raw),
            ("2  GROUND RANGE  gain-flattened, nadir removed", ground),
            ("3  ENHANCED  despeckled + CLAHE, detector input", enhanced),
        ]
    )
    fig.save(OUT_DIR / "pipeline.png", optimize=True)
    print(f"  wrote docs/images/pipeline.png  {fig.size[0]}x{fig.size[1]}")


def build_detections_figure() -> None:
    """L2: the annotated waterfall from scripts/detect_demo.py."""
    img = _load(OUTPUTS / "detections.png")
    if not img:
        return
    fig = _fit(img, TARGET_W)
    fig.save(OUT_DIR / "detections.png", optimize=True)
    print(f"  wrote docs/images/detections.png  {fig.size[0]}x{fig.size[1]}")


def build_evidence_figure() -> None:
    """L3: two evidence cards, stacked so their in-image cue captions stay readable."""
    evidence_dir = OUTPUTS / "surveys" / "survey_alpha" / "evidence"
    cards = sorted(evidence_dir.glob("*_evidence.png"), key=lambda p: -p.stat().st_size)
    if not cards:
        print(f"  missing: {evidence_dir.relative_to(REPO_ROOT)}/*_evidence.png")
        return
    panels = [(p.stem.replace("_evidence", ""), _load(p)) for p in cards[:2]]
    panels = [(caption, img) for caption, img in panels if img is not None]
    if not panels:
        return
    fig = _labelled_stack(panels)
    fig.save(OUT_DIR / "evidence.png", optimize=True)
    print(f"  wrote docs/images/evidence.png  {fig.size[0]}x{fig.size[1]}")


def build_calibration_figure() -> None:
    """L3: reliability diagrams before and after temperature scaling."""
    raw = _load(OUTPUTS / "calibration" / "reliability_raw.png")
    cal = _load(OUTPUTS / "calibration" / "reliability_calibrated.png")
    if not (raw and cal):
        return
    fig = _labelled_row(
        [("BEFORE  raw detector confidence", raw), ("AFTER  temperature-scaled", cal)],
        width=820,
        max_h=420,
    )
    fig.save(OUT_DIR / "calibration.png", optimize=True)
    print(f"  wrote docs/images/calibration.png  {fig.size[0]}x{fig.size[1]}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"building README figures into {OUT_DIR.relative_to(REPO_ROOT)}")
    build_pipeline_figure()
    build_detections_figure()
    build_evidence_figure()
    build_calibration_figure()


if __name__ == "__main__":
    main()
