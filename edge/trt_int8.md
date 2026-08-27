# TensorRT INT8 conversion (Jetson-class edge devices)

The detector deploys to Jetson (Orin Nano/NX, Xavier) as a TensorRT engine with
INT8 quantization. This roughly triples throughput vs FP16 with negligible mAP
loss **when calibrated on in-domain sonar tiles** — never calibrate on natural
photos.

## 1. Export ONNX on the workstation

```bash
python edge/export_onnx.py --weights weights/detector.pt --imgsz 640
```

Verify the printed parity numbers before moving on (raw deviation ~1e-5, mAP
within 1%).

## 2. Build the calibration set (on-domain!)

INT8 needs ~500 representative tiles. Generate them from the synthetic factory
plus any real surveys you have processed:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
from sonar_core.preprocess.pipeline import preprocess
from sonar_core.synth.scene import SceneConfig, make_scene

out = Path("edge/calib"); out.mkdir(exist_ok=True)
count = 0
for seed in range(12):
    pa, _ = make_scene(SceneConfig(n_pings=600, seed=seed))
    for tile in preprocess(pa).tiles:
        img = (np.nan_to_num(tile.image, nan=0.0) * 255).astype("uint8")
        Image.fromarray(img).convert("RGB").save(out / f"tile_{count:05d}.png")
        count += 1
print(count, "calibration tiles")
PY
```

## 3. Build the engine on the Jetson

Copy `weights/detector.onnx` and `edge/calib/` to the device, then either:

**trtexec (quick):**

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=detector.onnx \
  --saveEngine=detector_int8.engine \
  --int8 --fp16 \
  --calib=calib_cache.bin \
  --shapes=images:1x3x640x640
```

**Ultralytics (managed, does calibration for you):**

```bash
yolo export model=detector.pt format=engine int8=True data=path/to/data.yaml imgsz=640
```

## 4. Validate on-device

```bash
yolo val model=detector_int8.engine data=path/to/data.yaml imgsz=640
python edge/benchmark.py --weights detector_int8.engine
```

Acceptance: mAP50 within 1–2% of the FP32 PyTorch number recorded in
`edge/benchmark.md`; if it drops more, the calibration set was not
representative — regenerate it with more real survey tiles.

## Expected envelope (from Ultralytics-published Jetson numbers; measure your own)

| Device | Precision | ~FPS @640 |
|---|---|---|
| Orin Nano 8GB | FP16 | ~60 |
| Orin Nano 8GB | INT8 | ~100+ |
| Xavier NX | INT8 | ~60 |

These are order-of-magnitude planning figures for YOLOv8n-class models, not
our measurements; `edge/benchmark.py` writes the real numbers for whatever
hardware it runs on.
