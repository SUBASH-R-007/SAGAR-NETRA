"""Export the trained detector to ONNX and verify numerical parity.

Usage:
    python edge/export_onnx.py [--weights weights/detector.pt] [--imgsz 640]
                               [--data path/to/data.yaml]

Parity is verified two ways:
1. Raw-output check (always): both models run the same synthetic tiles and
   the top-k box tensors must agree within tolerance.
2. mAP parity (with ``--data``): ``ultralytics val`` runs on both the .pt and
   the .onnx; mAP50 must agree within 1 percentage point (the M8 acceptance
   bar). Printed honestly either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def export(weights: Path, imgsz: int, opset: int = 12) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False)
    return Path(onnx_path)


def raw_parity(weights: Path, onnx_path: Path, imgsz: int, n_tiles: int = 4) -> float:
    """Max abs deviation of raw output tensors between torch and ONNX."""
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    rng = np.random.default_rng(0)
    batch = rng.random((n_tiles, 3, imgsz, imgsz), dtype=np.float32)

    torch_model = YOLO(str(weights)).model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(batch))[0].numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: batch})[0]

    return float(np.max(np.abs(torch_out - onnx_out)))


def map_parity(weights: Path, onnx_path: Path, data_yaml: Path, imgsz: int) -> tuple[float, float]:
    from ultralytics import YOLO

    kwargs = dict(data=str(data_yaml), imgsz=imgsz, device="cpu", workers=0,
                  plots=False, verbose=False)
    map_pt = float(YOLO(str(weights)).val(**kwargs).box.map50)
    map_onnx = float(YOLO(str(onnx_path)).val(**kwargs).box.map50)
    return map_pt, map_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "detector.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--data", type=Path, default=None, help="data.yaml for mAP parity")
    args = parser.parse_args()

    weights = args.weights
    if not weights.exists():
        fallback = REPO_ROOT / "weights" / "detector_smoke.pt"
        if fallback.exists():
            print(f"note: {weights} missing, exporting {fallback}")
            weights = fallback
        else:
            raise SystemExit(f"no weights at {weights}; run scripts/train_detector.py first")

    onnx_path = export(weights, args.imgsz, args.opset)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    deviation = raw_parity(weights, onnx_path, args.imgsz)
    print(f"raw-output max abs deviation vs PyTorch: {deviation:.2e}")
    if deviation > 1e-3:
        print("WARNING: raw outputs deviate more than 1e-3 — inspect the export")

    if args.data is not None and args.data.exists():
        map_pt, map_onnx = map_parity(weights, onnx_path, args.data, args.imgsz)
        delta = abs(map_pt - map_onnx)
        print(f"mAP50  pytorch={map_pt:.4f}  onnx={map_onnx:.4f}  |delta|={delta:.4f}")
        print("PASS: within 1% mAP" if delta <= 0.01 else "FAIL: mAP parity outside 1%")


if __name__ == "__main__":
    main()
