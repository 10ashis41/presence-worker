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
WORKER_DIR_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ printf "\n\033[1;32m==> %s\033[0m\n" "$1"; }

PYV=$(python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYV" != "3.10" ]; then
  echo "FATAL: Python $PYV detected, but MuseTalk needs 3.10." >&2
  echo "       torch 2.0.1 has no wheels for $PYV, so the install cannot succeed." >&2
  echo "       Use base image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" >&2
  exit 1
fi
echo "python $PYV — ok"

# Keep HuggingFace's cache on the persistent volume — otherwise every pod
# restart re-downloads Chatterbox's models (~4.5 min observed).
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# pip's cache also defaults to the container disk, which RunPod wipes on every
# stop/restart — so torch (2.3 GB), mmcv and friends were being re-downloaded on
# every single boot. That reinstall IS most of the ~10-minute provisioning time.
# Pointing the cache at the persistent volume turns reinstalls into disk copies.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"

say "1/6  system packages"
apt-get update -qq
apt-get install -y -qq ffmpeg git wget fonts-dejavu-core libgl1 libglib2.0-0 >/dev/null

say "2/6  MuseTalk"
# MuseTalk's quality was REJECTED (256px mouth region = "melting"), and LatentSync is the
# lip-sync backend in production. Its weights are ~10 GB of a 50 GB volume that must now
# also hold EchoMimic's ~28 GB, so it is opt-out rather than deleted-outright:
# INSTALL_MUSETALK=0 skips steps 2 and 3 entirely. (The `if/else` below is deliberately
# unindented — an indentation-only diff over 30 lines is how shell scripts get broken.)
if [ "${INSTALL_MUSETALK:-1}" != "1" ]; then
  echo "     SKIPPED (INSTALL_MUSETALK=${INSTALL_MUSETALK:-1}) — LatentSync is the backend in use"
else
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
# MuseTalk's own download_weights.sh is broken (deprecated huggingface-cli,
# removed `gdown --id`, a China mirror, and NO exit-code checks — it prints
# success while downloading nothing). Verified failing 2026-09-19. We use our
# own downloader, which verifies each file and exits non-zero on any miss.
# Pin huggingface_hub<1.0: `-U` pulls 1.x, which breaks transformers 4.39.2
# that MuseTalk imports —
#   ImportError: huggingface-hub>=0.19.3,<1.0 is required ... found 1.32.0
# (observed on a real pod, 2026-09-19). We only need hf_hub_download, which
# the 0.x line provides.
pip install -q "huggingface_hub<1.0" gdown
python "$WORKER_DIR_SRC/download_weights.py" --dir "$MUSETALK_DIR/models"
fi   # end of the INSTALL_MUSETALK guard covering steps 2 and 3

say "4/6  Chatterbox TTS — in its OWN venv"
# Chatterbox and MuseTalk require INCOMPATIBLE torch versions and cannot share
# an environment. Installing chatterbox-tts into the system python upgrades
# torch 2.0.1 -> 2.6.0, which breaks mmcv's precompiled ops at import:
#   ImportError: mmcv/_ext...so: undefined symbol: _ZN2at4_ops10zeros_like4call
# (observed on a real pod, 2026-09-19). mmcv 2.0.1 is compiled against the
# torch 2.0.1 C++ ABI; nothing short of matching that ABI fixes it.
#
# So Chatterbox gets an isolated venv with its own torch. MuseTalk keeps the
# system python untouched. The worker invokes the venv via CHATTERBOX_PYTHON.
CB_VENV="${CB_VENV:-/workspace/cbvenv}"
if [ ! -x "$CB_VENV/bin/python" ]; then
  python -m venv "$CB_VENV"
fi
"$CB_VENV/bin/pip" install -q --upgrade pip
"$CB_VENV/bin/pip" install -q chatterbox-tts

echo "     chatterbox venv: $("$CB_VENV/bin/python" -c 'import torch;print("torch",torch.__version__)')"
echo "     system python  : $(python -c 'import torch;print("torch",torch.__version__)')"

# Guard: the system torch MUST still match what mmcv was built against.
# Only meaningful when MuseTalk is actually installed — mmcv's compiled ops are the thing
# that breaks. With INSTALL_MUSETALK=0 the system torch is just the base image's, and
# asserting 2.0.x would abort the whole boot (set -e) for no reason.
if [ "${INSTALL_MUSETALK:-1}" = "1" ]; then
python - <<'EOF'
import sys, torch
if not torch.__version__.startswith("2.0"):
    sys.exit(
        f"FATAL: system torch is {torch.__version__}, but mmcv was compiled for 2.0.x.\n"
        "       Something upgraded torch in the system env — MuseTalk will fail at\n"
        "       'mmcv/_ext...so: undefined symbol'. Keep new-torch packages in venvs."
    )
print("     system torch still 2.0.x — mmcv ABI intact")
EOF
else
  echo "     system-torch/mmcv guard skipped (MuseTalk not installed)"
fi

say "5/7  LatentSync (higher quality lip sync)"
# LatentSync pins torch 2.5.1 — incompatible with MuseTalk's 2.0.1, so it gets
# its own venv too. It runs at 512px (vs MuseTalk's 256) and has temporal
# layers + TREPA, which is what fixes MuseTalk's frame-to-frame "melting".
# Apache-2.0. VRAM: 8GB for v1.5, 18GB for v1.6.
LS_DIR="${LS_DIR:-/workspace/LatentSync}"
LS_VENV="${LS_VENV:-/workspace/lsvenv}"
if [ "${INSTALL_LATENTSYNC:-1}" = "1" ]; then
  if [ ! -d "$LS_DIR/.git" ]; then
    git clone --depth 1 https://github.com/bytedance/LatentSync.git "$LS_DIR"
  fi
  if [ ! -x "$LS_VENV/bin/python" ]; then
    python -m venv "$LS_VENV"
  fi
  "$LS_VENV/bin/pip" install -q --upgrade pip
  "$LS_VENV/bin/pip" install -q -r "$LS_DIR/requirements.txt"

  mkdir -p "$LS_DIR/checkpoints/whisper"
  "$LS_VENV/bin/python" - <<EOF
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
root = Path("$LS_DIR/checkpoints")
want = [("whisper/tiny.pt", "whisper/tiny.pt", 30), ("latentsync_unet.pt", "latentsync_unet.pt", 1000)]
missing = []
for remote, local, min_mb in want:
    dest = root / local
    have = dest.stat().st_size/1048576 if dest.exists() else 0
    if have >= min_mb:
        print(f"     have {local} ({have:.0f} MB)"); continue
    print(f"     get  {remote}")
    try:
        got = hf_hub_download(repo_id="ByteDance/LatentSync-1.6", filename=remote)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(got).read_bytes())
    except Exception as e:
        missing.append(f"{remote}: {e}"); continue
    sz = dest.stat().st_size/1048576
    if sz < min_mb: missing.append(f"{local} too small ({sz:.0f} MB)")
if missing:
    raise SystemExit("LatentSync weights failed:\n  " + "\n  ".join(missing))
print("     latentsync weights ok")
EOF
  echo "     latentsync venv: $("$LS_VENV/bin/python" -c 'import torch;print("torch",torch.__version__)')"
else
  echo "     skipped (INSTALL_LATENTSYNC=0)"
fi

say "5b/7  BEN2 matting (MIT, commercial-safe)"
# Cuts the finished clone out of its background so it can be placed over any
# scene. Installed into the LatentSync venv on purpose: that venv already has a
# CUDA torch 2.5.1 and torchvision, so ben2 adds only timm/einops instead of a
# second ~2.5 GB torch download (provisioning time is the scarce resource here —
# a restart already costs ~10 minutes).
#
# Why not RVM: it is the obvious human video-matting model, but the repo was
# re-released under GPL-3.0. Copyleft is not something to ship inside a product
# we sell. BEN2 is MIT. Licence table in COMPOSITING.md.
if [ "${INSTALL_BEN2:-1}" = "1" ]; then
  "$LS_VENV/bin/pip" install -q -e "git+https://github.com/PramaLLC/BEN2.git#egg=ben2"
  # ben2's requirements list omits opencv, but segment_video uses cv2. LatentSync
  # already pulls opencv in, so only install if it is genuinely missing.
  if ! "$LS_VENV/bin/python" -c "import cv2" 2>/dev/null; then
    "$LS_VENV/bin/pip" install -q opencv-python-headless
  fi
  # Guard: installing ben2 must not have moved torch. The lip-sync path depends
  # on this venv's torch, and a silent upgrade would break it in a way that only
  # shows up mid-render.
  "$LS_VENV/bin/python" - <<'EOF'
import sys, torch
if not torch.__version__.startswith("2.5"):
    sys.exit(f"FATAL: LatentSync venv torch became {torch.__version__} (expected 2.5.x) — "
             "installing BEN2 upgraded it. LatentSync will fail. Pin the install.")
print(f"     ben2 ok; LatentSync venv torch still {torch.__version__}")
EOF
  # Fetch weights into the persistent HF cache so a pod restart does not re-download.
  HF_HOME="$HF_HOME" "$LS_VENV/bin/python" - <<'EOF'
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import snapshot_download
print("     BEN2 weights:", snapshot_download("PramaLLC/BEN2"))
EOF
else
  echo "     skipped (INSTALL_BEN2=0)"
fi

say "6/7  worker files"
mkdir -p "$WORKER_DIR"
if [ ! -f "$WORKER_DIR/run_worker.py" ]; then
  echo "     !! Copy run_worker.py, tts_chatterbox.py and watermark.sh into $WORKER_DIR"
  echo "        then re-run this script."
  exit 1
fi
chmod +x "$WORKER_DIR/watermark.sh"

say "6b/7  EchoMimicV3 (Apache-2.0) — photo -> talking video"
# Optional, opt-in via INSTALL_ECHOMIMIC=1, and non-fatal: the lip-sync pipeline is what
# earns money today, so this experiment must never be able to stop the pod from booting.
# It pulls ~28 GB and refuses to start if the volume cannot hold it, because a
# half-downloaded weight file looks installed on the next boot and fails mid-render.
# FREE_SPACE=1 additionally deletes MuseTalk's dead weights and the pip cache first.
if [ "${INSTALL_ECHOMIMIC:-0}" = "1" ]; then
  if [ -f "$WORKER_DIR/install_echomimic.sh" ]; then
    MUSETALK_DIR="$MUSETALK_DIR" WORKER_DIR="$WORKER_DIR" \
      bash "$WORKER_DIR/install_echomimic.sh" \
      || echo "     !! EchoMimic install failed (non-fatal — continuing to start the worker)"
  else
    echo "     !! install_echomimic.sh not in $WORKER_DIR — skipping"
  fi
else
  echo "     skipped (INSTALL_ECHOMIMIC=0)"
fi

say "7/7  smoke check"
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
  CHATTERBOX_PYTHON="${CB_VENV:-/workspace/cbvenv}/bin/python" \
  HF_HOME="$HF_HOME" \
  LATENTSYNC_DIR="$LS_DIR" \
  LATENTSYNC_PYTHON="$LS_VENV/bin/python" \
  LIPSYNC_BACKEND="${LIPSYNC_BACKEND:-latentsync}" \
  EM_REPO="${EM_DIR:-/workspace/echomimic_v3}" \
  EM_VENV="${EM_VENV:-/opt/emvenv}" \
  EM_PYTHON="${EM_VENV:-/opt/emvenv}/bin/python" \
  IDLE_EXIT="${IDLE_EXIT:-0}" \
  python run_worker.py
