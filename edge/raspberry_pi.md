# Raspberry Pi 5 bring-up (full stack, CPU-first)

Gets the complete SAGAR-NETRA system — API, console, all five layers — running
on a Pi 5 with **no Hailo dependency**, so the demo works the moment the board
boots. The AI HAT is an acceleration upgrade on top (`hailo.md`), never a
prerequisite.

## 0. What actually ships to the Pi

The Pi needs the repo, the trained weights, and a **pre-built** `web/dist` —
do not install Node on the Pi:

| item | source | note |
|---|---|---|
| repo checkout | `git clone` | code + configs |
| `weights/*.pt`, `weights/*.onnx`, `weights/verifier.pkl` | copy from workstation | gitignored, never in the clone |
| `web/dist/` | build on the workstation: `cd web && npm run build`, copy the folder | gitignored |
| `data/samples/survey_alpha.xtf` | `python scripts/make_sample_xtf.py` on-device | deterministic |

Weights sanity: `detector.onnx` must be the export of the *deployed*
`detector.pt` (same content epoch). `api/health` fingerprints both — if the
System tab shows a `.pt`/`.onnx` pair from different trainings, re-export
before demoing.

## 1. OS + system packages

Raspberry Pi OS Bookworm 64-bit. Then:

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git libgl1
```

`libgl1` is for opencv-headless's transitive import; without it cv2 dies at
import time with a misleading error.

## 2. Python environment

```bash
cd SAGAR-NETRA
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,api]"
pip install onnxruntime          # Brain A without torch-heavy inference
```

Notes for ARM64:

* `torch` CPU wheels exist for aarch64 and install cleanly; they are needed by
  Brain C (autoencoder) and the segmenter regardless of how Brain A runs.
* `onnxruntime` has aarch64 wheels; it is the fast Brain A path on Pi CPU.
* If pip is slow, `--extra-index-url https://www.piwheels.org/simple` serves
  prebuilt ARM wheels for most of the stack.

## 3. Point Brain A at ONNX (recommended on Pi)

Ultralytics loads `.onnx` directly through onnxruntime, and the `Detector`
class passes the path straight through. One config line:

```yaml
# configs/detector.yaml on the Pi
ensemble_weights:
  - weights/detector.onnx
```

Everything downstream — physics gate, verifier, severity, reports — is
identical; the detector contract is "tiles in, boxes out" and does not care
what executes the forward pass.

## 4. Run

```bash
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` because the operator's browser is on another machine: the
console is at `http://<pi-address>:8000`, API at `/api/health`. Headless is
the intended mode — no monitor, keyboard or mouse on the Pi.

To survive reboots, a minimal systemd unit:

```ini
# /etc/systemd/system/sagar-netra.service
[Unit]
Description=SAGAR-NETRA DRISHTI console
After=network.target

[Service]
WorkingDirectory=/home/pi/SAGAR-NETRA
ExecStart=/home/pi/SAGAR-NETRA/.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now sagar-netra
```

## 5. Acceptance checks (run all four, in order)

```bash
# 1 - environment sane, weights fingerprinted
curl -s http://localhost:8000/api/health | python3 -m json.tool

# 2 - full pipeline end-to-end on the bundled survey
.venv/bin/python scripts/demo.py

# 3 - console serves (needs web/dist copied in step 0)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/    # expect 200

# 4 - throughput on this hardware, recorded not assumed
.venv/bin/python edge/benchmark.py
```

Expectation setting: the Pi 5 CPU runs the demo survey in minutes, not the
workstation's ~10 s — preprocessing is NumPy/SciPy at ~2,700 pings/s per core
on x86 and slower here, and Brain A on onnxruntime is single-digit tiles/s.
That is the number the AI HAT exists to change (`hailo.md`), and the honest
comparison requires recording the CPU figure first.

## 6. Power note (from the hardware checklist)

Under sustained inference the Pi 5 + HAT wants the full 5 V / 5 A PD profile.
A bank or supply that only offers 20 V profiles caps USB current and the board
browns out under load — symptoms are throttling and random USB resets, not a
clean crash. `vcgencmd get_throttled` should read `0x0` during the demo.
