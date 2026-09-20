#!/usr/bin/env bash
# Install EchoMimicV3 (Apache-2.0) — the photo-to-talking-video path.
#
# Deliberately NOT run under `set -e`: by contract this script must never take the pod down.
# The lip-sync pipeline is the product that works today; EchoMimic is an experiment sitting
# beside it. If this fails — no disk, no network, upstream moved — the worker must still
# boot and keep serving renders. Callers invoke it as:
#
#     bash install_echomimic.sh || echo "non-fatal failure"
#
# THE DISK IS THE WHOLE PROBLEM HERE (learned the hard way, 2026-09-19/20).
# The volume is 50 GB and the existing pipeline already spends ~40 GB of it (LatentSync, two
# venvs, the HF cache). EchoMimic needs ~19 GB of WEIGHTS and ~7 GB of PYTHON ENV, and the
# first two attempts died at "free 10 GB, need 28 GB" / "free 26, need 28" — 2 GB short.
#
# Splitting those two budgets is the fix:
#   * weights -> the persistent volume, because 19 GB must not be re-downloaded;
#   * venv    -> the CONTAINER disk (EM_VENV=/opt/emvenv), which is wiped on every restart
#                but rebuilds in minutes from the pip cache.
# That moves ~7 GB off the volume and turns "always 2 GB short" into "fits with room".
#
# Expect on every restart: the venv is rebuilt, the weights are NOT re-downloaded. That is
# why the two steps are separate below — a missing venv must never trigger a 19 GB
# re-verification.
set -uo pipefail

EM_DIR="${EM_DIR:-/workspace/echomimic_v3}"
EM_VENV="${EM_VENV:-/opt/emvenv}"
EM_MODELS="$EM_DIR/models"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HOME HF_HUB_DISABLE_TELEMETRY=1

# Only the weights must fit on the volume now: ~19 GB, plus headroom for the HF cache.
NEED_GB="${EM_NEED_GB:-24}"

echo "     == disk =="
df -h /workspace / 2>/dev/null | sed 's/^/       /'

# FREE_SPACE=1 removes MuseTalk before measuring. Its quality was rejected and LatentSync is
# the backend in production, so its weights are pure dead weight on a volume that must also
# hold EchoMimic. The PIP CACHE IS DELIBERATELY KEPT: it costs space but is what turns a venv
# rebuild into a disk copy instead of a download — and the venv now rebuilds every restart.
if [ "${FREE_SPACE:-0}" = "1" ]; then
  echo "     == freeing space (FREE_SPACE=1) =="
  for d in "${MUSETALK_DIR:-/workspace/MuseTalk}"; do
    if [ -e "$d" ]; then
      sz="$(du -sh "$d" 2>/dev/null | cut -f1)"
      rm -rf "$d"
      echo "       removed $d ($sz)"
    fi
  done
  echo "     == what is using /workspace now =="
  du -sh /workspace/* 2>/dev/null | sort -rh | head -12 | sed 's/^/       /'
fi

free_gb="$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')"
echo "       free ${free_gb:-?} GB on /workspace, need ~${NEED_GB} GB for the weights"

ensure_venv() {
  [ -x "$EM_VENV/bin/python" ] && { echo "     venv present at $EM_VENV — skipping"; return 0; }
  echo "     == venv at $EM_VENV (container disk; rebuilt each restart) =="
  python -m venv "$EM_VENV" || { echo "       !! venv creation failed (non-fatal)"; return 1; }
  "$EM_VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -2 | sed 's/^/       /'
  # Pin torch from the cu124 index FIRST. requirements.txt only says torch>=2.1.2, so the
  # resolver would otherwise choose a wheel built against a different CUDA and the first CUDA
  # call would fail deep inside a render. EchoMimic was tested on torch 2.5.1.
  echo "     == torch 2.5.1 + cu124 =="
  "$EM_VENV/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 | sed 's/^/       /'
  echo "     == requirements (tensorflow 2.15 + moviepy — the slow part) =="
  "$EM_VENV/bin/pip" install -q -r "$EM_DIR/requirements.txt" 2>&1 | tail -5 | sed 's/^/       /'
}

download_weights() {
  echo "     == weights (~19 GB: umt5-xxl 10.8, CLIP 4.6, VAE 0.5, transformer 3.3, wav2vec 0.4) =="
  "$EM_VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
base = os.environ["EM_MODELS"]
jobs = [
    # Wan 2.1 Fun 1.3B inpainting base: VAE, umt5-xxl text encoder, CLIP image encoder.
    # The 2.98 GB base DiT is deliberately NOT requested — we supply EchoMimic's
    # transformer instead (see transformer_path in em_driver.py).
    ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", f"{base}/Wan2.1-Fun-V1.1-1.3B-InP",
     ["*.pth", "*.json", "*.model", "*.txt", "*/tokenizer.json",
      "*/special_tokens_map.json", "*/tokenizer_config.json",
      "*/sentencepiece.bpe.model", "*/spiece.model"]),
    # EchoMimic's own transformer. Preview, not flash-pro: flash-pro's audio encoder is a
    # Chinese wav2vec2 and our narration is English.
    ("BadToBest/EchoMimicV3", f"{base}/transformer",
     ["transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"]),
    ("facebook/wav2vec2-base-960h", f"{base}/wav2vec2-base-960h", None),
]
for repo, dest, allow in jobs:
    print(f"       {repo} -> {dest}", flush=True)
    kw = dict(local_dir=dest, allow_patterns=allow) if allow else dict(local_dir=dest)
    snapshot_download(repo, **kw)
print("       weights complete")
PY
}

# ---------------------------------------------------------------- 1. repo
if [ ! -d "$EM_DIR/.git" ]; then
  echo "     == repo =="
  git clone --depth 1 https://github.com/antgroup/echomimic_v3 "$EM_DIR" 2>&1 | tail -3 | sed 's/^/       /'
fi
mkdir -p "$EM_MODELS"

# ------------------------------------------------------- 2. weights, then venv
# The weights check comes first and on its own, so the common restart case (weights on the
# volume, container disk wiped) is a fast no-op rather than a 19 GB re-verification. The
# space check is before the venv build so a doomed run fails in seconds, not minutes.
if [ -f "$EM_MODELS/transformer/diffusion_pytorch_model.safetensors" ] \
   && [ -f "$EM_MODELS/wav2vec2-base-960h/config.json" ] \
   && [ -d "$EM_MODELS/Wan2.1-Fun-V1.1-1.3B-InP" ]; then
  echo "     weights already on the volume — skipping download"
else
  if [ "${free_gb:-0}" -lt "$NEED_GB" ]; then
    echo "       !! NOT ENOUGH DISK: ${free_gb:-0} GB free, need ${NEED_GB} GB."
    echo "          The venv now lives on the container disk ($EM_VENV), so this is the"
    echo "          weight budget alone. Free /workspace or enlarge the volume, then restart."
    exit 0
  fi
  ensure_venv || exit 0
  download_weights || { echo "       !! weight download failed (non-fatal)"; exit 0; }
fi

# A restart wipes the container disk but keeps the weights — rebuild the venv for them.
ensure_venv || exit 0

# ---------------------------------------------------------------- 3. driver
if [ -f "$WORKER_DIR/em_driver.py" ]; then
  cp "$WORKER_DIR/em_driver.py" "$EM_DIR/em_driver.py"
  echo "     driver: copied em_driver.py into $EM_DIR (must sit beside infer_preview.py so its 'src' imports resolve)"
else
  echo "       !! em_driver.py missing from $WORKER_DIR (non-fatal)"
  exit 0
fi

"$EM_VENV/bin/python" -c "import torch, moviepy, transformers; print('       torch', torch.__version__, '| moviepy ok | transformers', transformers.__version__)"
echo "     == EchoMimicV3 ready =="
