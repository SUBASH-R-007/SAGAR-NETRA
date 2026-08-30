"""Bundle everything the edge device needs into one archive.

Usage:
    python scripts/pack_edge.py                     # -> dist/sagar-netra-edge.tar.gz
    python scripts/pack_edge.py --out /tmp/pi.tar.gz --include-torch-detector

A Raspberry Pi checkout cannot simply be ``git clone``d: the two things it most
needs are the two things git does not carry. ``weights/`` is gitignored (model
files), and ``web/dist/`` is a build artefact that requires Node, which has no
business being installed on the target. Hand-assembling the transfer is how a
demo arrives missing its console or running yesterday's model.

So this produces one file, with a manifest, and refuses to build a bundle whose
parts disagree:

* **Stale-export guard.** ``detector.onnx`` is the model Brain A runs from on
  the Pi. If it is older than ``detector.pt`` it was exported from different
  weights, and the device would silently run a model the workstation retired.
  That is a hard error, not a warning.
* **Curated weights.** Smoke checkpoints, training backups and ensemble seeds
  the deployed config does not list are left behind — they are tens of
  megabytes of things the device will never load.
* **SHA-256 manifest** so the receiving end can prove the transfer was clean,
  and so a support question months later has an answer.

The archive is a ``.tar.gz`` because the target is Linux: it preserves modes
and unpacks with tooling already on the device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "dist" / "sagar-netra-edge.tar.gz"

#: Source trees the runtime imports. ``tests`` ships too: the fastest way to
#: know an ARM wheel set is sane is to run the suite on the device itself.
SOURCE_DIRS = (
    "sonar_core", "tridentnet", "physicheck", "geoscribe",
    "api", "edge", "scripts", "configs", "tests",
)

#: Files at the repo root worth carrying.
ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE", "DECISIONS.md")

#: Weights the deployed configuration actually loads. detector.onnx is Brain A
#: on the Pi (onnxruntime, no torch inference); the .pt files are Brain B and
#: Brain C, which have no ONNX path today.
CORE_WEIGHTS = (
    "detector.onnx",
    "detector_int8.onnx",
    "segmenter.pt",
    "anomaly.pt",
    "verifier.pkl",
)

#: Only with --include-torch-detector: the torch detector as a fallback if
#: onnxruntime misbehaves on the target. 6 MB for peace of mind.
OPTIONAL_WEIGHTS = ("detector.pt",)

#: Data the demo needs: the bundled survey and the sensitive-zone overlays.
DATA_PATHS = ("data/samples", "data/layers")

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def _iter_files(root: Path):
    """Every shippable file under *root*, skipping caches and bytecode."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        yield path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(include_torch_detector: bool = False) -> list[Path]:
    """Absolute paths of everything the bundle carries, in a stable order."""
    picked: list[Path] = []

    for name in SOURCE_DIRS:
        d = REPO_ROOT / name
        if d.is_dir():
            picked.extend(_iter_files(d))

    for name in ROOT_FILES:
        f = REPO_ROOT / name
        if f.is_file():
            picked.append(f)

    dist = REPO_ROOT / "web" / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit(
            "web/dist/index.html is missing - build the console first:\n"
            "    cd web && npm install && npm run build\n"
            "Without it the device serves the JSON API behind a blank page."
        )
    picked.extend(_iter_files(dist))

    wanted = list(CORE_WEIGHTS) + (list(OPTIONAL_WEIGHTS) if include_torch_detector else [])
    for name in wanted:
        f = REPO_ROOT / "weights" / name
        if not f.is_file():
            raise SystemExit(f"missing required weight: weights/{name}")
        picked.append(f)

    for rel in DATA_PATHS:
        d = REPO_ROOT / rel
        if d.is_dir():
            picked.extend(_iter_files(d))

    return picked


def check_export_freshness() -> None:
    """Refuse to ship an ONNX exported from weights that have since changed.

    Brain A on the device runs from detector.onnx. If the .pt was retrained
    after the last export, the Pi would run the previous model while every
    fingerprint and report claimed otherwise - the exact class of silent
    mismatch this project has already been bitten by once.
    """
    pt = REPO_ROOT / "weights" / "detector.pt"
    onnx = REPO_ROOT / "weights" / "detector.onnx"
    if not (pt.is_file() and onnx.is_file()):
        return
    if onnx.stat().st_mtime < pt.stat().st_mtime:
        raise SystemExit(
            "weights/detector.onnx is OLDER than weights/detector.pt - the export "
            "does not match the deployed model.\nRe-export before packing:\n"
            "    python edge/export_onnx.py --weights weights/detector.pt "
            "--imgsz 640 --data data/datasets/real_mix/data.yaml --int8"
        )


def build(out_path: Path, include_torch_detector: bool = False) -> Path:
    check_export_freshness()
    files = collect(include_torch_detector)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "bundle": out_path.name,
        "created": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "target": "linux-aarch64 (Raspberry Pi OS 64-bit, Bookworm)",
        "brain_a_runtime": "onnxruntime via weights/detector.onnx",
        "files": {},
    }
    total = 0
    for f in files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        size = f.stat().st_size
        total += size
        # Hash only the payload files: hashing every source file triples the
        # pack time for no operational benefit, since source integrity is
        # already covered by git on the sending side.
        if rel.startswith(("weights/", "web/dist/", "data/")):
            manifest["files"][rel] = {"sha256": _sha256(f), "bytes": size}
        else:
            manifest["files"][rel] = {"bytes": size}
    manifest["total_bytes"] = total
    manifest["file_count"] = len(files)

    started = time.perf_counter()
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f"sagar-netra/{f.relative_to(REPO_ROOT).as_posix()}")
        payload = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("sagar-netra/MANIFEST.json")
        info.size = len(payload)
        import io

        tar.addfile(info, io.BytesIO(payload))

    packed = out_path.stat().st_size
    print(f"packed {len(files)} files ({total / 1e6:.1f} MB) -> {out_path}")
    print(f"  archive {packed / 1e6:.1f} MB, {time.perf_counter() - started:.1f}s")
    print(f"  weights: {', '.join(CORE_WEIGHTS)}"
          + (f", {', '.join(OPTIONAL_WEIGHTS)}" if include_torch_detector else ""))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-torch-detector", action="store_true",
        help="also ship weights/detector.pt as a fallback if onnxruntime misbehaves",
    )
    args = parser.parse_args()
    build(args.out, args.include_torch_detector)


if __name__ == "__main__":
    main()
