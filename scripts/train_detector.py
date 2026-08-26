"""Train TridentNet Brain A (Ultralytics YOLO) on the synthetic side-scan dataset.

Builds (or reuses) a YOLO-format dataset rendered by the physics-consistent
scene simulator through the real M2 preprocessing chain — the exact imagery
Brain A sees at inference time — then fine-tunes a pretrained checkpoint.

Sonar-imposed training rules honoured here:

* ``fliplr`` is forced to 0: both sides are stored nadir-first (column 0 at
  nadir) and acoustic shadows always extend toward increasing column, so a
  left-right mirror would put shadows up-range of their highlights — geometry
  no sonar can produce. Along-track (vertical) flips remain physically valid
  and are enabled instead.
* ``workers=0`` always: Windows DataLoader worker processes re-import the
  training script under spawn semantics and deadlock.
* The train/val split is by scene (done inside the dataset builder), so val
  mAP is never inflated by shared speckle/seabed texture.

After training the best checkpoint is copied to ``weights/detector.pt``
(``weights/detector_smoke.pt`` with ``--smoke``), where
:class:`tridentnet.detector.Detector` picks it up by default.

Usage:
    python scripts/train_detector.py [--smoke] [--data PATH] [--scenes N]
        [--model NAME] [--epochs N] [--imgsz N] [--batch N] [--name RUN]
        [--rebuild]
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tridentnet.data import build_synthetic_dataset
from tridentnet.detector import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default dataset locations; the smoke build is kept separate so a tiny
#: 6-scene dataset can never silently shadow the full training set.
DATASET_DIR: Path = REPO_ROOT / "data" / "datasets" / "synth"
SMOKE_DATASET_DIR: Path = REPO_ROOT / "data" / "datasets" / "synth_smoke"

#: Where the deployable checkpoints live (Detector's default search path).
WEIGHTS_DIR: Path = REPO_ROOT / "weights"

#: Full-run defaults (mirrored by the argparse help text).
DEFAULT_SCENES: int = 24
DEFAULT_EPOCHS: int = 40
DEFAULT_IMGSZ: int = 640
DEFAULT_BATCH: int = 8

#: Smoke profile: just enough optimisation to prove the dataset -> train ->
#: checkpoint -> Detector loop end to end on CPU in minutes, not hours.
SMOKE_SCENES: int = 6
SMOKE_EPOCHS: int = 3
SMOKE_IMGSZ: int = 320
SMOKE_BATCH: int = 4
#: Short surveys for the smoke build (full builds use the builder's default
#: 700-1100 pings); enough pings for several chips per scene, cheap to render.
SMOKE_PINGS_RANGE: tuple[int, int] = (240, 360)

#: Probability of an along-track (vertical) flip during training. Reversing
#: the ping order is physically valid — a survey line run the other way — and
#: doubles apparent along-track diversity for free.
ALONG_TRACK_FLIP_P: float = 0.5

#: Fixed seed so runs are reproducible chip-for-chip and batch-for-batch.
TRAIN_SEED: int = 0


@dataclass(frozen=True)
class RunParams:
    """Fully resolved training parameters (smoke profile already applied)."""

    data: Path | None  # explicit data.yaml, or None to build/reuse the default
    scenes: int
    model: str
    epochs: int
    imgsz: int
    batch: int
    name: str
    smoke: bool
    rebuild: bool
    dataset_dir: Path
    pings_range: tuple[int, int] | None  # None -> dataset builder default
    dest: Path  # deployable checkpoint destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, default=None,
        help="existing data.yaml; default: build/reuse the synthetic dataset",
    )
    parser.add_argument(
        "--scenes", type=int, default=None,
        help=f"synthetic scenes to render when building (default {DEFAULT_SCENES})",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="base checkpoint (default: 'model' from configs/detector.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help=f"training epochs (default {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help=f"training image size (default {DEFAULT_IMGSZ})"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help=f"batch size (default {DEFAULT_BATCH})"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="run name under runs/ (default: detector / detector_smoke)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny end-to-end run: 6 short scenes, 3 epochs, imgsz 320, batch 4",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="rebuild the synthetic dataset even if its data.yaml already exists",
    )
    return parser


def resolve(args: argparse.Namespace) -> RunParams:
    """Fill unset CLI options from the full-run or smoke profile.

    Explicit CLI values always win, so ``--smoke --epochs 5`` runs the tiny
    dataset for 5 epochs instead of 3.
    """
    smoke = bool(args.smoke)

    def pick(value: int | None, full: int, tiny: int) -> int:
        if value is not None:
            return int(value)
        return tiny if smoke else full

    return RunParams(
        data=args.data,
        scenes=pick(args.scenes, DEFAULT_SCENES, SMOKE_SCENES),
        model=args.model or str(load_config()["model"]),
        epochs=pick(args.epochs, DEFAULT_EPOCHS, SMOKE_EPOCHS),
        imgsz=pick(args.imgsz, DEFAULT_IMGSZ, SMOKE_IMGSZ),
        batch=pick(args.batch, DEFAULT_BATCH, SMOKE_BATCH),
        name=args.name or ("detector_smoke" if smoke else "detector"),
        smoke=smoke,
        rebuild=bool(args.rebuild),
        dataset_dir=SMOKE_DATASET_DIR if smoke else DATASET_DIR,
        pings_range=SMOKE_PINGS_RANGE if smoke else None,
        dest=WEIGHTS_DIR / ("detector_smoke.pt" if smoke else "detector.pt"),
    )


def _print_progress(stage: str, fraction: float) -> None:
    print(f"[{fraction:6.1%}] {stage}", flush=True)


def ensure_dataset(params: RunParams) -> Path:
    """Return the data.yaml to train on, building the synthetic set if needed."""
    if params.data is not None:
        if not params.data.is_file():
            raise FileNotFoundError(f"--data {params.data} does not exist")
        return params.data
    yaml_path = params.dataset_dir / "data.yaml"
    if yaml_path.is_file() and not params.rebuild:
        print(f"reusing existing dataset {yaml_path} (pass --rebuild to regenerate)")
        return yaml_path
    print(f"building synthetic dataset: {params.scenes} scenes -> {params.dataset_dir}")
    build_kwargs: dict[str, Any] = {}
    if params.pings_range is not None:
        build_kwargs["n_pings_range"] = params.pings_range
    return build_synthetic_dataset(
        params.dataset_dir,
        n_scenes=params.scenes,
        progress=_print_progress,
        **build_kwargs,
    )


def extract_map50(metrics: Any) -> float | None:
    """Pull mAP@0.5 from an Ultralytics metrics object, tolerating API drift."""
    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "map50"):
        return float(box.map50)
    results = getattr(metrics, "results_dict", None)
    if isinstance(results, dict):
        value = results.get("metrics/mAP50(B)")
        if value is not None:
            return float(value)
    return None


def _best_checkpoint(model: Any, run_name: str) -> Path | None:
    """Locate the best (else last) checkpoint written by the finished trainer."""
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", REPO_ROOT / "runs" / run_name))
    candidates = [
        Path(str(getattr(trainer, "best", ""))) if trainer is not None else None,
        save_dir / "weights" / "best.pt",
        save_dir / "weights" / "last.pt",
    ]
    for cand in candidates:
        if cand is not None and str(cand) and cand.is_file():
            return cand
    return None


def train(params: RunParams) -> int:
    """Build/reuse the dataset, train, deploy the checkpoint. Returns exit code."""
    t_start = time.perf_counter()
    data_yaml = ensure_dataset(params)
    t_data = time.perf_counter() - t_start

    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"training {params.model} on {data_yaml} "
        f"({params.epochs} epochs, imgsz {params.imgsz}, batch {params.batch}, {device})"
    )
    model = YOLO(params.model)
    t0 = time.perf_counter()
    metrics = model.train(
        data=str(data_yaml),
        epochs=params.epochs,
        imgsz=params.imgsz,
        batch=params.batch,
        device=device,
        workers=0,  # Windows: DataLoader worker processes deadlock under spawn
        plots=False,
        val=True,
        seed=TRAIN_SEED,
        project=str(REPO_ROOT / "runs"),
        name=params.name,
        exist_ok=True,
        # NEVER mirror across columns: shadows must stay down-range of their
        # highlights (see module docstring). Along-track flips are valid.
        fliplr=0.0,
        flipud=ALONG_TRACK_FLIP_P,
    )
    t_train = time.perf_counter() - t0

    src = _best_checkpoint(model, params.name)
    if src is None:
        print("ERROR: training produced no best.pt/last.pt checkpoint")
        return 1
    params.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, params.dest)

    map50 = extract_map50(metrics)
    map50_txt = f"{map50:.4f}" if map50 is not None else "unavailable (metrics object empty)"
    print(f"\ncopied {src} -> {params.dest}")
    print(f"final val mAP50: {map50_txt}")
    print(
        f"wall time: dataset {t_data:.1f} s, train {t_train:.1f} s, "
        f"total {time.perf_counter() - t_start:.1f} s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    params = resolve(build_parser().parse_args(argv))
    return train(params)


if __name__ == "__main__":
    raise SystemExit(main())
