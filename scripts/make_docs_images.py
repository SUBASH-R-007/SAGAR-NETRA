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


#: Bar colours by row family: the classical baseline reads as neutral, the
#: un-verified detector as a warning, the deployed system as the house navy.
BAR_CLASSICAL = (150, 158, 168)
BAR_UNVERIFIED = (224, 124, 0)
BAR_DEPLOYED = NAVY


def _font(size: int):
    """A legible TrueType face, falling back to PIL's bitmap default."""
    from PIL import ImageFont

    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bar_panel(
    rows: list[tuple[str, float, tuple[int, int, int]]],
    title: str,
    value_fmt: str,
    width: int = TARGET_W,
) -> Image.Image:
    """One horizontal bar chart: label column, bar, value printed at the end."""
    label_w, pad, row_h = 340, 16, 44
    height = LABEL_H + pad + row_h * len(rows) + pad
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, width, LABEL_H], fill=NAVY)
    draw.text((8, 6), title, fill=(255, 255, 255), font=_font(14))

    bar_max = width - label_w - 130
    peak = max((v for _, v, _ in rows), default=1.0) or 1.0
    y = LABEL_H + pad
    for label, value, colour in rows:
        draw.text((8, y + 9), label, fill=INK, font=_font(14))
        bar_w = max(int(bar_max * value / peak), 2)
        draw.rectangle([label_w, y + 6, label_w + bar_w, y + row_h - 14], fill=colour)
        draw.text(
            (label_w + bar_w + 10, y + 9), value_fmt.format(value), fill=INK,
            font=_font(14),
        )
        y += row_h
    return canvas


def build_comparison_figure() -> None:
    """Classical CAD vs SAGAR-NETRA, from the committed comparison JSON.

    Reads the sidecar rather than re-running the evaluation, so the figure and
    `docs/baseline_comparison.md` can never disagree about what was measured.
    """
    import json

    source = REPO_ROOT / "docs" / "baseline_comparison.json"
    if not source.is_file():
        print("  missing: docs/baseline_comparison.json "
              "(run scripts/eval_baseline.py)")
        return
    data = json.loads(source.read_text(encoding="utf-8"))

    palette = {
        "blob": BAR_CLASSICAL,
        "blob_shadow": BAR_CLASSICAL,
        "sagar_raw": BAR_UNVERIFIED,
        "sagar_full": BAR_DEPLOYED,
        "sagar_tuned": BAR_DEPLOYED,
    }
    short = {
        "blob": "Classical: threshold + blob",
        "blob_shadow": "Classical: + shadow gate",
        "sagar_raw": "SAGAR-NETRA: detector only",
        "sagar_full": "SAGAR-NETRA: full stack",
        "sagar_tuned": "SAGAR-NETRA: full, tuned",
    }
    rows = [r for r in data["rows"] if r["key"] in palette]
    f1_rows = [(short[r["key"]], r["f1"], palette[r["key"]]) for r in rows]
    fp_rows = [(short[r["key"]], r["fp_per_km2"], palette[r["key"]]) for r in rows]

    panels = [
        _bar_panel(f1_rows, "F1 SCORE  (higher is better)", "{:.3f}"),
        _bar_panel(fp_rows, "FALSE ALARMS PER KM²  (lower is better)", "{:.0f}"),
    ]
    # 22px left the 13px caption clipped against the canvas edge; 34 clears its
    # descenders with a margin.
    height = sum(p.height for p in panels) + GAP + 34
    fig = Image.new("RGB", (TARGET_W, height), PAPER)
    y = 0
    for panel in panels:
        fig.paste(panel, (0, y))
        y += panel.height + GAP

    caption = (
        f"SYNTHETIC held-out scenes (n={data['n_truth']} targets, "
        f"{data['area_km2']:.3f} km²) · localization-only scoring · "
        "baseline tuned on a separate split"
    )
    ImageDraw.Draw(fig).text((8, y + 2), caption, fill=(90, 100, 112), font=_font(13))
    fig.save(OUT_DIR / "comparison.png", optimize=True)
    print(f"  wrote docs/images/comparison.png  {fig.size[0]}x{fig.size[1]}")


def _line_panel(
    series: list[tuple[str, list[float], list[float], tuple[int, int, int], bool]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 540,
    height: int = 300,
) -> Image.Image:
    """A small line chart. `series` is (name, xs, ys, colour, dashed)."""
    left, right, top, bottom = 54, 14, LABEL_H + 12, 44
    plot_w = width - left - right
    plot_h = height - top - bottom

    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, LABEL_H], fill=NAVY)
    draw.text((8, 6), title, fill=(255, 255, 255), font=_font(13))

    x_max = max(max(xs) for _, xs, _, _, _ in series) or 1
    # Precision is a fraction: fix the axis at 0..1 so the two panels are
    # directly comparable by eye instead of each auto-scaling to its own range.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - int(plot_h * frac)
        draw.line([left, y, left + plot_w, y], fill=(214, 219, 226))
        draw.text((8, y - 7), f"{frac:.2f}", fill=(110, 120, 132), font=_font(12))

    def px(x: float, y: float) -> tuple[int, int]:
        return (
            left + int(plot_w * x / x_max),
            top + plot_h - int(plot_h * min(max(y, 0.0), 1.0)),
        )

    for name, xs, ys, colour, dashed in series:
        points = [px(x, y) for x, y in zip(xs, ys, strict=True)]
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
            if dashed:  # sample the segment so the line reads as a dashed run
                steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                for s in range(0, steps, 8):
                    a = s / steps
                    b = min((s + 4) / steps, 1.0)
                    draw.line(
                        [x0 + (x1 - x0) * a, y0 + (y1 - y0) * a,
                         x0 + (x1 - x0) * b, y0 + (y1 - y0) * b],
                        fill=colour, width=3,
                    )
            else:
                draw.line([x0, y0, x1, y1], fill=colour, width=3)
        for cx, cy in points:
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=colour)
        draw.text((points[-1][0] - 76, points[-1][1] - 18), name, fill=colour,
                  font=_font(12))

    for x in series[0][1]:
        cx, _ = px(x, 0)
        draw.text((cx - 8, top + plot_h + 6), f"+{int(x)}", fill=INK, font=_font(12))
    # y_label rides in the title bar: at this panel size a rotated axis label
    # collides with the top gridline tick, and the ticks already read as 0..1.
    draw.text((left, height - 20), f"{x_label}   (y: {y_label})", fill=(110, 120, 132),
              font=_font(12))
    return canvas


def build_clutter_figure() -> None:
    """Precision vs natural clutter, with and without the brightness shortcut."""
    import json

    source = REPO_ROOT / "docs" / "clutter_sweep.json"
    if not source.is_file():
        print("  missing: docs/clutter_sweep.json (run scripts/eval_clutter.py)")
        return
    data = json.loads(source.read_text(encoding="utf-8"))

    panels = []
    for mode, blurb in (
        ("native", "NATIVE  simulator's brightness gap intact"),
        ("matched", "MATCHED  decoys as bright as real targets"),
    ):
        rows = data["modes"][mode]
        xs = [r["extra_rocks"] for r in rows]
        panels.append(
            _line_panel(
                [
                    ("classical", xs, [r["classical"]["precision"] for r in rows],
                     BAR_CLASSICAL, True),
                    ("SAGAR-NETRA", xs, [r["sagar"]["precision"] for r in rows],
                     BAR_DEPLOYED, False),
                ],
                blurb, "extra rock clusters per scene", "precision",
            )
        )

    fig = Image.new("RGB", (TARGET_W, panels[0].height + 24), PAPER)
    x = 0
    for panel in panels:
        fig.paste(panel, (x, 0))
        x += panel.width + GAP
    ImageDraw.Draw(fig).text(
        (8, panels[0].height + 4),
        f"SYNTHETIC · {data['n_scenes']} scenes, {data['n_truth']} targets · "
        "nested clutter, debris field held fixed · localization-only scoring",
        fill=(90, 100, 112), font=_font(13),
    )
    fig.save(OUT_DIR / "clutter.png", optimize=True)
    print(f"  wrote docs/images/clutter.png  {fig.size[0]}x{fig.size[1]}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"building README figures into {OUT_DIR.relative_to(REPO_ROOT)}")
    build_pipeline_figure()
    build_detections_figure()
    build_evidence_figure()
    build_calibration_figure()
    build_comparison_figure()
    build_clutter_figure()


if __name__ == "__main__":
    main()
