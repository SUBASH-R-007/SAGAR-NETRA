"""Benchmark the detection stack on the current machine and write benchmark.md.

Usage:
    python edge/benchmark.py [--weights weights/detector.pt] [--imgsz 640]
                             [--tiles 24] [--out edge/benchmark.md]

Measures, honestly, on this hardware:
- L1 preprocessing throughput (pings/s) on a synthetic survey
- Detector latency/FPS: PyTorch CPU and ONNX Runtime (if an export exists)
- Anomaly autoencoder latency (if weights exist)
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def bench_preprocess(n_pings: int = 800) -> dict:
    from sonar_core.preprocess.pipeline import preprocess
    from sonar_core.synth.scene import SceneConfig, make_scene

    pa, _ = make_scene(SceneConfig(n_pings=n_pings, seed=1))
    start = time.perf_counter()
    pre = preprocess(pa)
    elapsed = time.perf_counter() - start
    return {
        "pings": n_pings,
        "seconds": round(elapsed, 2),
        "pings_per_s": round(n_pings / elapsed, 1),
        "n_tiles": len(pre.tiles),
    }


def bench_torch(weights: Path, imgsz: int, n_tiles: int) -> dict | None:
    if not weights.exists():
        return None
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rng = np.random.default_rng(0)
    tiles = [(rng.random((imgsz, imgsz, 3)) * 255).astype(np.uint8) for _ in range(n_tiles)]
    model.predict(tiles[0], imgsz=imgsz, device="cpu", verbose=False)  # warmup
    start = time.perf_counter()
    for tile in tiles:
        model.predict(tile, imgsz=imgsz, device="cpu", verbose=False)
    elapsed = time.perf_counter() - start
    return {
        "n": n_tiles,
        "ms_per_tile": round(1000 * elapsed / n_tiles, 1),
        "tiles_per_s": round(n_tiles / elapsed, 2),
    }


def bench_onnx(onnx_path: Path, imgsz: int, n_tiles: int) -> dict | None:
    if not onnx_path.exists():
        return None
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    rng = np.random.default_rng(0)
    batch = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
    session.run(None, {name: batch})  # warmup
    start = time.perf_counter()
    for _ in range(n_tiles):
        session.run(None, {name: batch})
    elapsed = time.perf_counter() - start
    return {
        "n": n_tiles,
        "ms_per_tile": round(1000 * elapsed / n_tiles, 1),
        "tiles_per_s": round(n_tiles / elapsed, 2),
    }


def bench_anomaly(weights: Path, imgsz: int, n_tiles: int) -> dict | None:
    if not weights.exists():
        return None
    from tridentnet.anomaly import AnomalyDetector

    detector = AnomalyDetector(weights=weights, device="cpu")
    rng = np.random.default_rng(0)
    tile = rng.random((imgsz, imgsz)).astype(np.float32)
    detector.error_map(tile)  # warmup
    start = time.perf_counter()
    for _ in range(n_tiles):
        detector.error_map(tile)
    elapsed = time.perf_counter() - start
    return {
        "n": n_tiles,
        "ms_per_tile": round(1000 * elapsed / n_tiles, 1),
        "tiles_per_s": round(n_tiles / elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "detector.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tiles", type=int, default=24)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "edge" / "benchmark.md")
    args = parser.parse_args()

    weights = args.weights
    if not weights.exists() and (REPO_ROOT / "weights" / "detector_smoke.pt").exists():
        weights = REPO_ROOT / "weights" / "detector_smoke.pt"

    results = {
        "preprocess": bench_preprocess(),
        "torch": bench_torch(weights, args.imgsz, args.tiles),
        "onnx": bench_onnx(weights.with_suffix(".onnx"), args.imgsz, args.tiles),
        "anomaly": bench_anomaly(REPO_ROOT / "weights" / "anomaly.pt", 512, 8),
    }

    lines = [
        "# SAGAR-NETRA edge benchmark",
        "",
        f"- Date: {datetime.now(tz=UTC).isoformat(timespec='seconds')}",
        f"- Machine: {platform.node()} — {platform.processor() or platform.machine()}",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}, weights: `{weights.name}`, imgsz {args.imgsz}",
        "",
        "| Stage | Metric | Value |",
        "|---|---|---|",
        f"| L1 preprocess | pings/s | {results['preprocess']['pings_per_s']} |",
        f"| L1 preprocess | full {results['preprocess']['pings']}-ping survey | "
        f"{results['preprocess']['seconds']} s -> {results['preprocess']['n_tiles']} tiles |",
    ]
    for key, label in (("torch", "Detector (PyTorch CPU)"), ("onnx", "Detector (ONNX RT CPU)"),
                       ("anomaly", "Anomaly AE (CPU)")):
        r = results[key]
        if r is None:
            lines.append(f"| {label} | — | not available (no weights/export) |")
        else:
            lines.append(f"| {label} | ms/tile | {r['ms_per_tile']} |")
            lines.append(f"| {label} | tiles/s | {r['tiles_per_s']} |")
    lines += [
        "",
        "TensorRT INT8 numbers require a Jetson-class device; see `edge/trt_int8.md`.",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
