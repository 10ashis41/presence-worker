#!/usr/bin/env bash
# One-shot setup for a fresh rented GPU pod (RunPod / Vast.ai).
#
#   export WORKER_TOKEN=...            # from ~/avatar-presenter/api/.env on the VM
#   bash setup_pod.sh
#
# Installs MuseTalk + Chatterbox, fetches weights, then starts the worker.
# Safe to re-run: every step is idempotent.
#
# Versions are pinned to what MuseTalk's own README specifies (Python 3.10,
# torch 2.0.1 / cu118, mmcv 2.0.1). Do NOT "upgrade" these — the mmlab stack is
# extremely version-sensitive and newer torch silently breaks mmcv's compiled ops.
#
# BASE IMAGE MATTERS: use runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04.
# An Ubuntu 24.04 image gives Python 3.12, for which torch 2.0.1 has no wheels at
# all — pip fails with "Could not find a version that satisfies torch==2.0.1"
# (verified on a real pod, 2026-09-19).
set -euo pipefail

: "${WORKER_TOKEN:?set WORKER_TOKEN first (see api/.env on the VM)}"
API_BASE="${API_BASE:-https://api.aiguyonthefly.com/presenter}"
MUSETALK_DIR="${MUSETALK_DIR:-/workspace/MuseTalk}"
WORKER_DIR="${WORKER_DIR:-/workspace/presence-worker}"

say(){ printf "\n\033[1;32m==> %s\033[0m\n" "$1"; }

PYV=$(python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYV" != "3.10" ]; then
  echo "FATAL: Python $PYV detected, but MuseTalk needs 3.10." >&2
  echo "       torch 2.0.1 has no wheels for $PYV, so the install cannot succeed." >&2
  echo "       Use base image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" >&2
  exit 1
fi
echo "python $PYV — ok"

say "1/6  system packages"
apt-get update -qq
apt-get install -y -qq ffmpeg git wget fonts-dejavu-core >/dev/null

say "2/6  MuseTalk"
if [ ! -d "$MUSETALK_DIR" ]; then
  git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
fi
cd "$MUSETALK_DIR"

# Only install torch if it's missing or the wrong major version — RunPod's
# PyTorch templates often already ship a compatible build.
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.0') else 1)" 2>/dev/null; then
  say "     installing torch 2.0.1 + cu118 (MuseTalk's pinned version)"
  pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118
fi

pip install -q -r requirements.txt
pip install -q --no-cache-dir -U openmim
mim install -q mmengine
mim install -q "mmcv==2.0.1"
mim install -q "mmdet==3.1.0"
mim install -q "mmpose==1.1.0"

say "3/6  MuseTalk weights (~10 GB — slowest step)"
if [ ! -f "$MUSETALK_DIR/models/musetalkV15/unet.pth" ]; then
  sh ./download_weights.sh
else
  echo "     already present, skipping"
fi

say "4/6  Chatterbox TTS (multilingual: en / he / ar)"
pip install -q chatterbox-tts

say "5/6  worker files"
mkdir -p "$WORKER_DIR"
if [ ! -f "$WORKER_DIR/run_worker.py" ]; then
  echo "     !! Copy run_worker.py, tts_chatterbox.py and watermark.sh into $WORKER_DIR"
  echo "        then re-run this script."
  exit 1
fi
chmod +x "$WORKER_DIR/watermark.sh"

say "6/6  smoke check"
python - <<'EOF'
import torch
print(f"     torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"     gpu: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
else:
    raise SystemExit("     NO GPU VISIBLE — the render will fail. Check the pod.")
EOF
ffmpeg -version | head -1 | sed 's/^/     /'

say "starting worker — it will poll $API_BASE for jobs"
cd "$WORKER_DIR"
exec env \
  API_BASE="$API_BASE" \
  WORKER_TOKEN="$WORKER_TOKEN" \
  MUSETALK_DIR="$MUSETALK_DIR" \
  TTS_BACKEND="${TTS_BACKEND:-chatterbox}" \
  LIPSYNC_BACKEND="${LIPSYNC_BACKEND:-musetalk}" \
  IDLE_EXIT="${IDLE_EXIT:-0}" \
  python run_worker.py
