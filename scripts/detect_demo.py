"""End-to-end detection demo: survey file -> preprocess -> Brain A -> annotated PNG.

Loads a survey through the format-agnostic parser stack, runs the full M2
preprocessing chain, detects debris on the SAHI tiles with
:class:`tridentnet.detector.Detector`, and renders the detections on the
combined *enhanced* ground-range waterfall — the exact imagery the detector
saw. Layout follows survey-software convention: port is mirrored so its far
range is at the left edge, both nadirs meet at the centreline, starboard far
range at the right (port column ``c`` maps to ``x = n_port_cols - 1 - c``,
starboard to ``x = n_port_cols + c``). Boxes are drawn from the detector's
global inclusive ``(ping, ground-column)`` extents, so what is drawn is what
would be georeferenced — no separate pixel bookkeeping to drift.

Also prints an honest detection table (side, ping range, ground range in
metres, class, score) and the tile inference rate. Exit code is 0 even with
zero detections: an empty seabed is a valid survey result.

Usage:
    python scripts/detect_demo.py [--input PATH] [--weights PATH]
        [--out PATH] [--conf F]
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from sonar_core.parsers.base import load
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.preprocess.slant_range import GroundImage
from tridentnet.classes import CLASS_TO_ID
from tridentnet.detector import Detection, Detector

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT: Path = REPO_ROOT / "data" / "samples" / "survey_alpha.xtf"
DEFAULT_WEIGHTS: Path = REPO_ROOT / "weights" / "detector.pt"
SMOKE_WEIGHTS: Path = REPO_ROOT / "weights" / "detector_smoke.pt"
DEFAULT_OUT: Path = REPO_ROOT / "outputs" / "detections.png"
PREPROCESS_CONFIG: Path = REPO_ROOT / "configs" / "preprocess.yaml"

#: Full-scale value of an 8-bit display pixel.
U8_MAX = 255.0

#: Saturated, mutually distinguishable box colors. Sonar classes index this by
#: their frozen class id, so a class keeps its color across runs and surveys;
#: unknown names (COCO fallback weights) hash in deterministically via CRC32.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (66, 214, 146),  # ghost_net
    (255, 99, 71),  # wreck
    (64, 156, 255),  # aircraft
    (255, 195, 0),  # pipeline
    (255, 128, 0),  # cylinder_drum
    (186, 104, 255),  # tire
    (0, 206, 209),  # container
    (255, 64, 129),  # human_body
    (255, 235, 59),  # mine_like
    (156, 204, 101),  # rock_cluster
    (121, 134, 203),  # sand_ripple
    (240, 98, 146),  # reef
)

#: Box outline width (px) and label font size for the annotated PNG.
BOX_WIDTH_PX: int = 2
LABEL_FONT_SIZE: int = 12
LABEL_PAD_PX: int = 2


def class_color(name: str) -> tuple[int, int, int]:
    """Deterministic per-class color: frozen id for sonar classes, CRC otherwise."""
    idx = CLASS_TO_ID.get(name)
    if idx is None:
        idx = zlib.crc32(name.encode("utf-8"))
    return PALETTE[idx % len(PALETTE)]


def det_x_span(det: Detection, n_port_cols: int) -> tuple[int, int]:
    """Detection column extent -> inclusive x extent on the combined waterfall.

    Port is mirrored (nadir at the centreline, far range left), so its column
    order reverses: ``x = n_port_cols - 1 - c``. Starboard maps directly to
    ``x = n_port_cols + c``. Returned as ``(x0, x1)`` with ``x0 <= x1``.
    """
    if det.side == "port":
        return n_port_cols - 1 - det.col1, n_port_cols - 1 - det.col0
    return n_port_cols + det.col0, n_port_cols + det.col1


def combined_enhanced(gi: GroundImage) -> np.ndarray:
    """Port mirrored | starboard, nadirs meeting at the centreline."""
    return np.hstack([gi.port[:, ::-1], gi.starboard])


def _label_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: fixed-size bitmap font only
        return ImageFont.load_default()


def render_detections(gi: GroundImage, detections: list[Detection], out_path: Path) -> Path:
    """Draw every detection on the combined enhanced waterfall and save a PNG.

    The enhanced image is already display-normalized to [0, 1] by CLAHE; NaN
    (beyond-swath fill) maps to black, matching true acoustic shadow rather
    than inventing texture. Each box gets its class color and a "cls score"
    tag placed just above the box (below it when clipped by the image top).
    """
    img = np.nan_to_num(np.clip(combined_enhanced(gi), 0.0, 1.0), nan=0.0)
    canvas = Image.fromarray((img * U8_MAX + 0.5).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(LABEL_FONT_SIZE)
    n_port = gi.n_cols("port")

    for det in detections:
        x0, x1 = det_x_span(det, n_port)
        color = class_color(det.cls)
        draw.rectangle((x0, det.ping0, x1, det.ping1), outline=color, width=BOX_WIDTH_PX)
        text = f"{det.cls} {det.score:.2f}"
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        th = (ty1 - ty0) + 2 * LABEL_PAD_PX
        tw = (tx1 - tx0) + 2 * LABEL_PAD_PX
        lx = int(np.clip(x0, 0, max(canvas.width - tw, 0)))
        ly = det.ping0 - th
        if ly < 0:  # box touches the top edge: hang the tag below instead
            ly = min(det.ping1 + BOX_WIDTH_PX, canvas.height - th)
        draw.rectangle((lx, ly, lx + tw, ly + th), fill=color)
        draw.text((lx + LABEL_PAD_PX, ly + LABEL_PAD_PX - ty0), text, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def format_table(detections: list[Detection], gi: GroundImage) -> str:
    """Fixed-width detection table: side, ping range, ground range (m), class, score."""
    header = f"{'side':<10} {'pings':<12} {'ground range m':<15} {'class':<15} {'score':>5}"
    lines = [header, "-" * len(header)]
    for det in detections:
        g0 = float(gi.ground_range_of_col(det.col0))
        g1 = float(gi.ground_range_of_col(det.col1))
        lines.append(
            f"{det.side:<10} {f'{det.ping0}-{det.ping1}':<12} "
            f"{f'{g0:.1f}-{g1:.1f}':<15} {det.cls:<15} {det.score:>5.2f}"
        )
    return "\n".join(lines)


def resolve_weights(requested: Path | None) -> Path | None:
    """Pick the checkpoint to run: explicit > trained > smoke > COCO fallback.

    Returns ``None`` when no local checkpoint exists, letting the Detector
    fall back to the COCO-pretrained asset named in configs/detector.yaml
    (plumbing demo only — class names are then COCO's, not sonar classes).
    """
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"--weights {requested} does not exist")
        return requested
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    if SMOKE_WEIGHTS.is_file():
        print(
            f"note: {DEFAULT_WEIGHTS} not found; falling back to smoke checkpoint "
            f"{SMOKE_WEIGHTS} (3-epoch smoke training, expect weak detections)"
        )
        return SMOKE_WEIGHTS
    print(
        f"note: neither {DEFAULT_WEIGHTS} nor {SMOKE_WEIGHTS} exists; "
        "using the COCO-pretrained fallback from configs/detector.yaml "
        "(sonar classes unavailable)"
    )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="survey file to scan")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="detector checkpoint (default weights/detector.pt, then detector_smoke.pt)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="annotated PNG path")
    parser.add_argument(
        "--conf", type=float, default=None,
        help="confidence threshold override (default from configs/detector.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"{args.input} missing; generating the bundled sample survey...")
        from sonar_core.synth.sample import make_sample

        make_sample(args.input.parent)

    config = (
        yaml.safe_load(PREPROCESS_CONFIG.read_text(encoding="utf-8"))
        if PREPROCESS_CONFIG.is_file()
        else None
    )
    print(f"loading {args.input}")
    pa = load(args.input)
    print(f"preprocessing {pa.n_pings} pings...")
    result = preprocess(pa, config=config)
    print(f"{len(result.tiles)} tiles from the enhanced ground image")

    weights = resolve_weights(args.weights)
    overrides: dict[str, Any] = {}
    if args.conf is not None:
        overrides["conf"] = float(args.conf)
    detector = Detector(weights=weights, config=overrides or None)
    print(f"weights: {detector.weights} | conf: {detector.config['conf']}")

    detector.model  # noqa: B018 - force the lazy load so timing is inference only
    t0 = time.perf_counter()
    detections = detector.detect_tiles(result.tiles)
    elapsed = time.perf_counter() - t0
    rate = len(result.tiles) / elapsed if elapsed > 0 else float("inf")

    print(f"\n{len(detections)} detection(s):")
    if detections:
        print(format_table(detections, result.ground))
    else:
        print("(none above the confidence threshold — honest empty result)")
    print(f"\ninference: {len(result.tiles)} tiles in {elapsed:.2f} s = {rate:.2f} tiles/s")

    out = render_detections(result.ground, detections, args.out)
    print(f"annotated waterfall written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
