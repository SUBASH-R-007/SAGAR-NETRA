#!/usr/bin/env bash
# SAGAR-NETRA — Raspberry Pi OS 64-bit (Bookworm, aarch64) bring-up.
#
#   bash edge/setup_pi64.sh
#
# Run from inside the unpacked bundle. Idempotent: safe to re-run after a
# failure without undoing anything.
#
# CPU-first by design. Brain A runs from weights/detector.onnx through
# onnxruntime; the Hailo AI HAT is an accelerator added afterwards
# (edge/hailo.md), never a prerequisite. Bring the system up on CPU, prove it
# works, then make it faster.
set -euo pipefail

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mFAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. checks --
say "Checking platform"
ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || die "expected aarch64, found $ARCH.
This bundle targets Raspberry Pi OS 64-bit. A 32-bit (armv7l) install has no
PyTorch or onnxruntime wheels: reflash with the 64-bit image."
echo "  arch      : $ARCH"
echo "  model     : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "  python    : $(python3 --version)"
[ -f pyproject.toml ] || die "run this from the unpacked bundle root (pyproject.toml not found)"

# ---------------------------------------------------------- 1. system deps --
# libgl1 + libglib2.0-0: OpenCV's transitive shared-library needs. Without
# them cv2 fails at import with a message that names neither package.
say "Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev libgl1 libglib2.0-0

# ------------------------------------------------------------ 2. venv+deps --
say "Creating virtualenv and installing Python packages (several minutes)"
[ -d .venv ] || python3 -m venv .venv
# piwheels serves prebuilt aarch64 wheels for the ARM-heavy parts of the stack;
# without it pip compiles scipy/opencv from source, which takes hours on a Pi.
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --extra-index-url https://www.piwheels.org/simple \
    -e ".[ml,api]" onnxruntime

# ------------------------------------------------ 3. point Brain A at ONNX --
# The deployed config lists weights/detector.pt (a torch checkpoint). On the Pi
# we run the ONNX export instead: same model, no torch in the forward pass,
# roughly twice the throughput on CPU. Everything downstream is unchanged --
# the detector contract is "tiles in, boxes out".
say "Configuring Brain A for onnxruntime"
if grep -q 'weights/detector.pt' configs/detector.yaml; then
  cp configs/detector.yaml configs/detector.yaml.bak
  sed -i 's|- weights/detector.pt|- weights/detector.onnx|' configs/detector.yaml
  echo "  configs/detector.yaml -> weights/detector.onnx (backup: .bak)"
else
  echo "  already pointed at an ONNX model, leaving alone"
fi

# ------------------------------------------------------------ 4. acceptance --
say "Acceptance checks"

echo "-- imports"
.venv/bin/python - <<'PY'
import importlib
missing = [m for m in ("numpy","scipy","cv2","onnxruntime","torch","fastapi","pyxtf")
           if importlib.util.find_spec(m) is None]
raise SystemExit(f"missing modules: {missing}" if missing else 0)
PY
echo "   all runtime modules import"

echo "-- weights present"
for w in detector.onnx segmenter.pt anomaly.pt verifier.pkl; do
  [ -f "weights/$w" ] || die "weights/$w missing from the bundle"
done
echo "   all four model files present"

echo "-- full pipeline on the bundled survey (slow on a Pi; this is the point)"
time .venv/bin/python scripts/demo.py

echo "-- console assets"
[ -f web/dist/index.html ] || die "web/dist missing - rebuild on the workstation and repack"
echo "   web/dist present"

# --------------------------------------------------------------- 5. serve --
say "Bring-up complete"
cat <<'EOF'

Start the console (headless — reach it from another machine's browser):

    .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Then open   http://<this-pi-address>:8000   and check:
  - System tab shows the detector fingerprint
  - Overview shows the bundled survey's contacts

Survive reboots:
    sudo cp edge/sagar-netra.service /etc/systemd/system/
    sudo systemctl enable --now sagar-netra

Record the throughput for the record (the number the AI HAT will improve on):
    .venv/bin/python edge/benchmark.py

Power sanity under load — must print throttled=0x0:
    vcgencmd get_throttled

EOF
