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
# and it exits 0 for every condition it can explain, so the log tells the story instead of
# the pod dying silently.
set -uo pipefail

EM_DIR="${EM_DIR:-/workspace/echomimic_v3}"
EM_VENV="${EM_VENV:-/workspace/emvenv}"
EM_MODELS="$EM_DIR/models"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HOME HF_HUB_DISABLE_TELEMETRY=1

# Wan2.1-Fun-1.3B-InP is ~15.5 GB with our allow-patterns (10.8 of it the T5 text encoder;
# the 2.98 GB base DiT is skipped because we supply EchoMimic's transformer instead), the
# EchoMimic transformer is 3.3 GB, wav2vec2 is 0.4 GB, and the venv lands around 8 GB with
# torch + tensorflow + moviepy. Call it 28 GB. Running out mid-download leaves a
# half-written file that looks installed on the next boot, so measure first and refuse.
NEED_GB="${EM_NEED_GB:-28}"

# FREE_SPACE=1 removes the two things we know are dead weight before measuring. Explicit
# and opt-in rather than silent housekeeping: deleting 10 GB of a colleague's model weights
# inside an "install" script is the kind of thing that should be visible in the log.
if [ "${FREE_SPACE:-0}" = "1" ]; then
  echo "     == freeing space (FREE_SPACE=1) =="
  for d in "${MUSETALK_DIR:-/workspace/MuseTalk}" "${PIP_CACHE_DIR:-/workspace/pip-cache}"; do
    if [ -e "$d" ]; then
      sz="$(du -sh "$d" 2>/dev/null | cut -f1)"
      rm -rf "$d"
      echo "       removed $d ($sz)"
    fi
  done
  # MuseTalk's weights may also be split into the HF cache; report what is left so the next
  # shortage is diagnosable from the log instead of by guessing.
  echo "     == what is using /workspace now =="
  du -sh /workspace/* 2>/dev/null | sort -rh | head -12 | sed 's/^/       /'
fi

echo "     == disk =="
df -h /workspace | sed 's/^/       /'
free_gb="$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')"
echo "       free ${free_gb:-?} GB, need ~${NEED_GB} GB"

# Already installed? Then this is a no-op re-run on a later restart, which is the common case.
if [ -f "$EM_MODELS/transformer/diffusion_pytorch_model.safetensors" ] \
   && [ -f "$EM_MODELS/wav2vec2-base-960h/config.json" ] \
   && [ -x "$EM_VENV/bin/python" ] \
   && [ -d "$EM_MODELS/Wan2.1-Fun-V1.1-1.3B-InP" ]; then
  echo "       echoMimic already present — skipping download"
else
  if [ "${free_gb:-0}" -lt "$NEED_GB" ]; then
    echo "       !! NOT ENOUGH DISK: ${free_gb:-0} GB free, need ${NEED_GB} GB."
    echo "          Enlarge the pod volume (RunPod: volume size) and restart. Skipping EchoMimic."
    exit 0
  fi

  echo "     == repo =="
  if [ ! -d "$EM_DIR/.git" ]; then
    git clone --depth 1 https://github.com/antgroup/echomimic_v3 "$EM_DIR" 2>&1 | tail -3 | sed 's/^/       /'
  fi
  mkdir -p "$EM_MODELS"

  echo "     == venv =="
  [ -x "$EM_VENV/bin/python" ] || python -m venv "$EM_VENV"
  "$EM_VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -2 | sed 's/^/       /'

  # Pin torch from the cu124 index FIRST. requirements.txt only says torch>=2.1.2, so the
  # resolver would otherwise pick a wheel built against a different CUDA and the first
  # CUDA call would fail deep inside a render. EchoMimic was tested on torch 2.5.1.
  echo "     == torch 2.5.1 + cu124 =="
  "$EM_VENV/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 | sed 's/^/       /'

  echo "     == requirements (includes tensorflow 2.15 + moviepy) — this is the slow part =="
  "$EM_VENV/bin/pip" install -q -r "$EM_DIR/requirements.txt" 2>&1 | tail -5 | sed 's/^/       /'

  echo "     == weights =="
  "$EM_VENV/bin/python" - <<'EOF'
import os
from huggingface_hub import snapshot_download
base = os.environ["EM_MODELS"]
jobs = [
    # the Wan 2.1 Fun 1.3B inpainting base: VAE, umt5-xxl text encoder, CLIP image encoder
    ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", f"{base}/Wan2.1-Fun-V1.1-1.3B-InP",
     ["*.pth", "*.json", "*.model", "*.txt", "*/tokenizer.json", "*/special_tokens_map.json",
      "*/tokenizer_config.json", "*/sentencepiece.bpe.model", "*/spiece.model"]),
    # EchoMimic's own transformer. Only the preview transformer: the flash-pro variant is
    # built on a Chinese wav2vec2 and we narrate in English.
    ("BadToBest/EchoMimicV3", f"{base}/transformer", ["transformer/config.json",
                                                      "transformer/diffusion_pytorch_model.safetensors"]),
    # English audio encoder for the preview path
    ("facebook/wav2vec2-base-960h", f"{base}/wav2vec2-base-960h", None),
]
for repo, dest, allow in jobs:
    print(f"       {repo} -> {dest}", flush=True)
    kw = dict(local_dir=dest, allow_patterns=allow) if allow else dict(local_dir=dest)
    snapshot_download(repo, **kw)
print("       weights complete")
EOF
  if [ $? -ne 0 ]; then
    echo "       !! weight download failed — EchoMimic not usable this boot (non-fatal)"
    exit 0
  fi
fi

echo "     == driver =="
if [ -f "$WORKER_DIR/em_driver.py" ]; then
  cp "$WORKER_DIR/em_driver.py" "$EM_DIR/em_driver.py"
  echo "       copied em_driver.py into $EM_DIR (must live beside infer_preview.py so its 'src' imports resolve)"
else
  echo "       !! em_driver.py missing from $WORKER_DIR"
  exit 0
fi

"$EM_VENV/bin/python" -c "import torch, moviepy, transformers; print('       torch', torch.__version__, '| moviepy ok | transformers', transformers.__version__)"
echo "     == EchoMimicV3 ready =="
