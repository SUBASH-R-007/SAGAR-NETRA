"""Render the M2 preprocessing gallery: one PNG per major pipeline stage.

Loads the bundled sample survey (generating it first if missing), runs the
full preprocessing pipeline with ``configs/preprocess.yaml``, and writes
8-bit grayscale quick-looks to ``outputs/gallery/``:

    01_raw_waterfall.png  raw slant-range waterfall (port mirrored | starboard)
    02_egn_slant.png      after empirical gain normalization, still slant range
    03_ground_range.png   slant-corrected, unenhanced (NaN swath shown black)
    04_enhanced.png       final despeckled + CLAHE detector input

Usage:
    python scripts/preprocess_gallery.py [--xtf PATH] [--config PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sonar_core.parsers.base import PingArray, load
from sonar_core.preprocess.pipeline import PreprocessResult, preprocess
from sonar_core.preprocess.slant_range import GroundImage
from sonar_core.synth.sample import make_sample
from sonar_core.waterfall import combine, normalize_u8

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Full-scale value of an 8-bit display pixel.
U8_MAX = 255.0


def _print_progress(stage: str, fraction: float) -> None:
    print(f"[{fraction:6.1%}] {stage}")


def _save_png(img_u8: np.ndarray, path: Path) -> None:
    Image.fromarray(img_u8, mode="L").save(path)
    print(f"wrote {path}")


def _combined_ground(gi: GroundImage) -> np.ndarray:
    """Port mirrored | starboard, nadirs meeting at the centreline."""
    return np.hstack([gi.port[:, ::-1], gi.starboard])


def _egn_slant_image(pa: PingArray, result: PreprocessResult) -> np.ndarray:
    """Reconstruct the EGN-normalized slant waterfall from the returned gain
    curves: seabed samples divided by the per-sample gain, water column (before
    the tracked first return) passed through untouched — exactly what the EGN
    stage applied inside the pipeline."""
    parts: list[np.ndarray] = []
    for side, mirror in (("port", True), ("starboard", False)):
        raw = pa.side(side).astype(np.float64)
        gain = result.egn_gain[side]
        norm = raw / gain[None, :] if gain.size else raw.copy()
        cols = np.arange(raw.shape[1])[None, :]
        first = result.bottom.first_return[side][:, None]
        norm = np.where(cols < first, raw, norm)
        parts.append(norm[:, ::-1] if mirror else norm)
    return np.hstack(parts).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xtf", type=Path, default=REPO_ROOT / "data" / "samples" / "survey_alpha.xtf"
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "preprocess.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "gallery")
    args = parser.parse_args()

    if not args.xtf.exists():
        print(f"{args.xtf} missing; generating the sample survey...")
        make_sample(args.xtf.parent)

    config = yaml.safe_load(args.config.read_text()) if args.config.exists() else None
    pa = load(args.xtf)
    result = preprocess(pa, config=config, progress=_print_progress)

    args.out.mkdir(parents=True, exist_ok=True)
    _save_png(normalize_u8(combine(pa)), args.out / "01_raw_waterfall.png")
    _save_png(normalize_u8(_egn_slant_image(pa, result)), args.out / "02_egn_slant.png")
    _save_png(normalize_u8(_combined_ground(result.ground_raw)), args.out / "03_ground_range.png")
    # CLAHE output is already display-normalized to [0, 1]; scale directly so
    # the PNG shows exactly what the detector ingests (NaN swath -> black).
    enhanced = np.nan_to_num(np.clip(_combined_ground(result.ground), 0.0, 1.0), nan=0.0)
    _save_png((enhanced * U8_MAX + 0.5).astype(np.uint8), args.out / "04_enhanced.png")

    print(f"\n{result.ground.n_pings} pings -> {len(result.tiles)} tiles; stage timings:")
    for stage, seconds in result.timings.items():
        print(f"  {stage:>18s}: {seconds * 1000.0:8.1f} ms")
    print(f"  {'total':>18s}: {sum(result.timings.values()) * 1000.0:8.1f} ms")


if __name__ == "__main__":
    main()
