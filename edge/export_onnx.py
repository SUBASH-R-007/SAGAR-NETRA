"""Export the trained detector to ONNX and verify numerical parity.

Usage:
    python edge/export_onnx.py [--weights weights/detector.pt] [--imgsz 640]
                               [--data path/to/data.yaml] [--int8]

Parity is verified two ways:
1. Raw-output check (always): both models run the same synthetic tiles and
   the top-k box tensors must agree within tolerance.
2. mAP parity (with ``--data``): ``ultralytics val`` runs on both the .pt and
   the .onnx; mAP50 must agree within 1 percentage point (the M8 acceptance
   bar). Printed honestly either way.

``--int8`` additionally writes a dynamically weight-quantized model
(``*_int8.onnx``) and runs the SAME probes against it. INT8 weights *will*
deviate from FP32 in the raw check — that is the point of quantization, not a
bug — so the honest acceptance number is the mAP delta on real data, printed
as "INT8 cost" when ``--data`` is given.
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
    torch_model = YOLO(str(weights)).model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # The export is static batch-1; compare tile by tile.
    worst = 0.0
    for _ in range(n_tiles):
        tile = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(tile))[0].numpy()
        onnx_out = session.run(None, {input_name: tile})[0]
        worst = max(worst, float(np.max(np.abs(torch_out - onnx_out))))
    return worst


def quantize_int8(onnx_path: Path) -> Path:
    """Dynamic (weight-only) INT8 quantization: ``detector.onnx`` ->
    ``detector_int8.onnx``. Weights are stored INT8 and dequantized per layer
    at run time — no calibration dataset needed, and the file shrinks ~4x,
    which is what matters for shipping to an edge box over a boat's uplink."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = onnx_path.with_name(f"{onnx_path.stem}_int8.onnx")
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def val_map50(model_path: Path, data_yaml: Path, imgsz: int) -> float:
    """mAP50 of one model (.pt or .onnx) via ``ultralytics val``."""
    from ultralytics import YOLO

    # rect=False: the .pt path would otherwise use rectangular val batches
    # while the static ONNX runs square inputs — an apples-to-oranges mAP.
    kwargs = dict(data=str(data_yaml), imgsz=imgsz, device="cpu", workers=0,
                  plots=False, verbose=False, rect=False)
    return float(YOLO(str(model_path)).val(**kwargs).box.map50)


def map_parity(weights: Path, onnx_path: Path, data_yaml: Path, imgsz: int) -> tuple[float, float]:
    return val_map50(weights, data_yaml, imgsz), val_map50(onnx_path, data_yaml, imgsz)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "detector.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--data", type=Path, default=None, help="data.yaml for mAP parity")
    parser.add_argument("--int8", action="store_true",
                        help="also write a dynamically weight-quantized *_int8.onnx")
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

    int8_path = None
    if args.int8:
        int8_path = quantize_int8(onnx_path)
        print(
            f"quantized {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB, "
            f"fp32 was {onnx_path.stat().st_size / 1e6:.1f} MB)"
        )
        int8_dev = raw_parity(weights, int8_path, args.imgsz)
        print(f"INT8 raw-output max abs deviation vs PyTorch: {int8_dev:.2e} "
              "(deviation is expected — INT8 trades exactness for size)")

    if args.data is not None and args.data.exists():
        map_pt = val_map50(weights, args.data, args.imgsz)
        map_onnx = val_map50(onnx_path, args.data, args.imgsz)
        delta = abs(map_pt - map_onnx)
        print(f"mAP50  pytorch={map_pt:.4f}  onnx={map_onnx:.4f}  |delta|={delta:.4f}")
        print("PASS: within 1% mAP" if delta <= 0.01 else "FAIL: mAP parity outside 1%")
        if int8_path is not None:
            map_int8 = val_map50(int8_path, args.data, args.imgsz)
            int8_cost = map_pt - map_int8
            print(f"mAP50  int8={map_int8:.4f}  INT8 cost vs pytorch={int8_cost:+.4f}")


if __name__ == "__main__":
    main()
