# Hailo AI HAT+ deployment (Raspberry Pi 5)

The AI HAT+ carries a Hailo-8L (13 TOPS) or Hailo-8 (26 TOPS) NPU on the Pi 5's
PCIe lane. This runbook takes `weights/detector.onnx` (exported at imgsz 640
from the real-data-trained checkpoint — regenerate with `edge/export_onnx.py`
after any retrain, never ship a stale export) to a compiled `.hef` running on
the HAT.

**Status honesty:** steps 1–2 are executed practice in this repo; steps 3–5 are
documented from Hailo's toolchain requirements and have not been run here,
because the Dataflow Compiler does not run on Windows or on the Pi (see below).
Update this line when a `.hef` has actually been built and measured.

**The zero-friction fallback works today:** the Pi 5's CPU runs the full
pipeline with Brain A on `detector.onnx` via onnxruntime (see
`raspberry_pi.md`). The HAT accelerates Brain A only; everything else —
preprocessing, physics, reports, console — is CPU either way. Bring the system
up on CPU first, then add the HAT.

## 1. Export fresh ONNX (workstation)

```bash
python edge/export_onnx.py --weights weights/detector.pt --imgsz 640 --data data/datasets/real_mix/data.yaml
```

Do not proceed unless the printed parity is clean (raw deviation ~1e-5, mAP
delta within 0.01). A model that is wrong at this stage is wrong on the HAT
with three extra layers of toolchain between you and the bug.

## 2. Build an in-domain calibration set (workstation)

Hailo's quantizer needs ~500 representative images. Sonar tiles, never natural
photos — calibration statistics from the wrong domain quantize the wrong
dynamic range:

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
        Image.fromarray(img).convert("RGB").resize((640, 640)).save(
            out / f"tile_{count:05d}.png")
        count += 1
print(count, "calibration tiles")
PY
```

If real surveys have been processed, add tiles from them — real seabed texture
in the calibration set is strictly better than synthetic alone.

## 3. Compile ONNX -> HEF (x86-64 Linux ONLY)

The Hailo Dataflow Compiler runs on **x86-64 Ubuntu only** — not Windows, not
the Pi. WSL2 on the training laptop satisfies it. Install the DFC and Model
Zoo from Hailo's Developer Zone (registration required), then let the model
zoo's YOLOv8 recipe handle layer cutting and NMS placement rather than
hand-picking end nodes:

```bash
# hw-arch: hailo8l for the 13 TOPS HAT+, hailo8 for the 26 TOPS one
hailomz compile yolov8n \
    --ckpt weights/detector.onnx \
    --hw-arch hailo8l \
    --calib-path edge/calib \
    --classes 12 \
    --performance
```

Output: `yolov8n.hef`. Copy it to the Pi.

Known sharp edge: the compiler consumes the *backbone+head*; the sigmoid/DFL
decode either runs on-chip via the recipe's NMS config or falls back to host
code. If compilation rejects ops, re-export the ONNX with `--opset 11` before
touching anything else — opset mismatches are the usual culprit.

## 4. Pi-side install

```bash
sudo apt update && sudo apt install -y hailo-all   # HailoRT + firmware (Pi OS Bookworm)
sudo reboot
hailortcli fw-control identify                     # must print the device; if not, check
                                                   # the FFC cable and PCIe gen3 setting
```

Note: the HAT occupies the PCIe connector — no NVMe SSD alongside it. The
microSD card is the only storage.

## 5. Run inference against the HEF

HailoRT's Python API (`hailo_platform`) feeds uint8 640x640x3 frames — the
same letterboxed tiles the ONNX path uses, grayscale replicated to three
channels:

```python
from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams

hef = HEF("yolov8n.hef")
with VDevice() as device:
    network_group = device.configure(hef)[0]
    with InferVStreams(network_group,
                       InputVStreamParams.make(network_group),
                       OutputVStreamParams.make(network_group)) as pipeline:
        results = pipeline.infer({"input": batch})   # batch: (N, 640, 640, 3) uint8
```

Integration point: subclass or shim `tridentnet.detector.Detector.detect_tiles`
to route the forward pass through the HEF while keeping the existing tile
bookkeeping, class map and downstream physics untouched — the contract is just
"tiles in, `Detection` boxes out", and nothing downstream knows what silicon
produced them.

## 6. Measure before claiming

Run `edge/benchmark.py` equivalents on-device and record: tiles/s on Pi CPU
(onnxruntime) vs HAT, and mAP50 of the HEF against `data/datasets/real_mix`
val — quantization to the HAT is a *third* set of weights, and its accuracy is
a measurement, not an assumption. Publish both numbers in `edge/benchmark.md`
next to the x86 ones.
