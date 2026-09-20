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
# EM_MODELS must be EXPORTED: the weights downloader below is a separate `python` child
# (heredoc) that reads it via os.environ, and a shell variable set but not exported reaches
# no child at all — the download then dies with KeyError: 'EM_MODELS' and the install
# silently reports "weights failed (non-fatal)", leaving the photo path dead.
export HF_HOME HF_HUB_DISABLE_TELEMETRY=1 EM_MODELS EM_VENV EM_DIR

# Only the weights must fit on the volume now: ~19 GB. Measured on a freshly provisioned
# 50 GB volume: ~20 GB free after LatentSync, both venvs and the HF cache exist. The venv
# (~7 GB) lives on the container disk and the pip cache no longer sits on the volume — those
# two moves are what create the room.
NEED_GB="${EM_NEED_GB:-23}"

echo "     == disk =="
df -h /workspace / 2>/dev/null | sed 's/^/       /'

# FREE_SPACE=1 removes MuseTalk before measuring. Its quality was rejected and LatentSync is
# the backend in production, so its weights are pure dead weight on a volume that must also
# hold EchoMimic. The PIP CACHE IS DELIBERATELY KEPT: it costs space but is what turns a venv
# rebuild into a disk copy instead of a download — and the venv now rebuilds every restart.
if [ "${FREE_SPACE:-0}" = "1" ]; then
  echo "     == freeing space (FREE_SPACE=1) =="
  # /workspace/pip-cache is the OLD cache location: the cache now lives on the container disk
  # (/opt/pip-cache), so any copy left on the volume from an earlier boot is dead weight.
  for d in "${MUSETALK_DIR:-/workspace/MuseTalk}" /workspace/pip-cache; do
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
  # RE-PIN TORCH LAST. requirements.txt only wants torch>=2.1.2, so its resolver installs a
  # much newer build on top of the cu124 trio above (observed: torch 2.14.0+cu130 while
  # torchaudio stayed 2.5.1+cu124). The mismatch is invisible until the driver imports, and
  # then it is fatal:
  #   OSError: libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
  #   -> "EchoMimic driver exited 1" AFTER 4 minutes of TTS work. EchoMimic is tested on
  # 2.5.1, so the pinned trio is restored as the final word.
  echo "     == re-pinning torch 2.5.1 (requirements.txt installs a newer, incompatible one) =="
  "$EM_VENV/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2 | sed 's/^/       /'
  "$EM_VENV/bin/python" -c "import torch, torchaudio; print('       emvenv torch', torch.__version__, '| torchaudio', torchaudio.__version__)"

  # SAME BUG CLASS AS THE TORCH RE-PIN ABOVE, different package (found 2026-09-20).
  # EchoMimic's face detector is retina-face, which declares only `tensorflow>=1.9.0`, so
  # pip happily installs TF 2.16+ over the 2.15 requirements.txt asked for. In 2.16 Keras 3
  # became the default and the `tensorflow.keras` module was removed, so the driver dies at
  # import — 4 minutes of TTS work already spent:
  #   src/face_detect.py -> retinaface -> ModuleNotFoundError: No module named 'tensorflow.keras'
  # tf-keras restores the Keras 2 API, and TF_USE_LEGACY_KERAS=1 (set in em_driver.py, so it
  # holds no matter how this venv was built) routes `tensorflow.keras` back to it. Installing
  # tf-keras is harmless on TF 2.15, so this is safe whichever version the resolver picks.
  echo "     == tf-keras (retina-face imports tensorflow.keras, removed in TF 2.16) =="
  "$EM_VENV/bin/pip" install -q tf-keras 2>&1 | tail -2 | sed 's/^/       /'

  # THIRD INSTANCE OF THE SAME PATTERN (found 2026-09-20): a co-installed package quietly
  # replacing something torch was built against. TensorFlow ships its own nvidia-cudnn-cu12
  # pin, which lands on top of the one torch 2.5.1 needs. Torch then loads a cuDNN whose
  # symbols it does not match, and the FIRST conv3d — inside the Wan VAE, ~4 minutes into a
  # render — dies with:
  #     RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
  # It is NOT out of memory: instrumentation showed 44.2 of 44.4 GiB free at that moment.
  # The Chatterbox venv proves the diagnosis — same GPU, same CUDA, torch works there, and
  # the only difference is that it has no TensorFlow in it.
  #
  # Ask torch which cuDNN it declares rather than hard-coding a version, so this keeps
  # working when the pinned torch changes.
  echo "     == restoring the cuDNN torch was built against (TensorFlow overwrote it) =="
  want_cudnn="$("$EM_VENV/bin/python" - <<'PY' 2>/dev/null
try:
    import importlib.metadata as md
    for r in md.requires("torch") or []:
        head = r.split(";")[0].strip()
        if head.startswith("nvidia-cudnn-cu12"):
            print(head.replace(" ", "")); break
except Exception:
    pass
PY
)"
  if [ -n "${want_cudnn:-}" ]; then
    echo "       torch wants: $want_cudnn"
    "$EM_VENV/bin/pip" install -q --force-reinstall --no-deps "$want_cudnn" 2>&1 | tail -2 | sed 's/^/       /'
  else
    echo "       !! could not read torch's cuDNN pin — leaving as installed"
  fi

  # Prove cuDNN actually works HERE, at install time, with the operation that was failing.
  # Every bug in this path so far has surfaced four minutes into a render, after TTS had
  # already run. A two-second conv3d now is worth a great deal.
  "$EM_VENV/bin/python" - <<'PY' 2>&1 | sed 's/^/       /'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | cudnn {torch.backends.cudnn.version()}")
if not torch.cuda.is_available():
    print("!! no CUDA visible at install time — cannot verify cuDNN")
else:
    try:
        x = torch.randn(1, 4, 4, 16, 16, device="cuda")
        w = torch.randn(4, 4, 3, 3, 3, device="cuda")
        torch.nn.functional.conv3d(x, w, padding=1)
        torch.cuda.synchronize()
        print("cuDNN conv3d smoke test OK — the Wan VAE path will run")
    except Exception as e:
        print(f"!! cuDNN conv3d FAILED: {type(e).__name__}: {e}")
        print("!! a render would die in wan_vae.py; fix the venv before queueing a job")
PY
  TF_USE_LEGACY_KERAS=1 "$EM_VENV/bin/python" - <<'PY' 2>&1 | sed 's/^/       /'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model  # the exact import retina-face makes
    print(f"emvenv tensorflow {tf.__version__} | tensorflow.keras import OK")
except Exception as e:
    print(f"!! tensorflow.keras still broken: {type(e).__name__}: {e}")
PY
}

download_weights() {
  echo "     == weights (~19 GB: umt5-xxl 10.8, CLIP 4.6, VAE 0.5, transformer 3.3, wav2vec 0.4) =="
  "$EM_VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
base = os.environ["EM_MODELS"]
jobs = [
    # Wan 2.1 Fun 1.3B inpainting base: VAE, umt5-xxl text encoder, CLIP image encoder,
    # AND the 2.98 GB base DiT (diffusion_pytorch_model.safetensors).
    #
    # The DiT was deliberately skipped here until 2026-09-20 on the theory that EchoMimic's
    # own transformer replaces it. It does not — it is loaded ON TOP of this one. Upstream
    # does `WanTransformerAudioMask3DModel.from_pretrained(<this dir>)` first and immediately
    # compares shapes:
    #     if model.state_dict()['patch_embedding.weight'].size() != state_dict[...].size()
    # With the DiT absent that state_dict is empty and inference dies with
    #     KeyError: 'patch_embedding.weight'
    # — after TTS has already run. Note the allow_patterns below had no "*.safetensors", so
    # the omission was silent: the directory existed and looked populated.
    ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", f"{base}/Wan2.1-Fun-V1.1-1.3B-InP",
     ["*.pth", "*.json", "*.model", "*.txt", "diffusion_pytorch_model.safetensors",
      "*/tokenizer.json", "*/special_tokens_map.json", "*/tokenizer_config.json",
      "*/sentencepiece.bpe.model", "*/spiece.model"]),
    # EchoMimic's own transformer. Preview, not flash-pro: flash-pro's audio encoder is a
    # Chinese wav2vec2 and our narration is English.
    #
    # local_dir is the MODELS ROOT and the allow_patterns keep the repo's own `transformer/`
    # prefix, so the files land at models/transformer/<file> — which is where em_driver.py
    # looks (EM_TRANSFORMER) and what the "already downloaded" check above tests for.
    # local_dir=$base/transformer ALSO strips the prefix target-wise but keeps it in the
    # pattern, which wrote models/transformer/transformer/diffusion_pytorch_model.safetensors:
    # the 3.4 GB downloaded on every boot, the check never saw it, and the install then died
    # on the free-space guard claiming the weights were missing.
    ("BadToBest/EchoMimicV3", base,
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

# Self-heal a volume written by the older local_dir bug: the transformer files landed one
# level too deep (models/transformer/transformer/...). Move them up so an existing 3.4 GB
# download is reused instead of re-fetched — and so the check below sees a complete set.
if [ -f "$EM_MODELS/transformer/transformer/diffusion_pytorch_model.safetensors" ]; then
  echo "     == repairing transformer path (older layout) =="
  mv -f "$EM_MODELS/transformer/transformer/"* "$EM_MODELS/transformer/" 2>/dev/null
  rmdir "$EM_MODELS/transformer/transformer" 2>/dev/null
fi

# ------------------------------------------------------- 2. weights, then venv
# The weights check comes first and on its own, so the common restart case (weights on the
# volume, container disk wiped) is a fast no-op rather than a 19 GB re-verification. The
# space check is before the venv build so a doomed run fails in seconds, not minutes.
# The Wan check tests for the BASE DiT FILE, not just the directory. Testing the directory
# is what let the missing 2.98 GB DiT survive every restart: the dir existed (VAE, T5, CLIP
# were all there), the guard passed, the download was skipped, and the failure only showed up
# mid-render as KeyError: 'patch_embedding.weight'. A guard must check the thing that breaks.
if [ -f "$EM_MODELS/transformer/diffusion_pytorch_model.safetensors" ] \
   && [ -f "$EM_MODELS/wav2vec2-base-960h/config.json" ] \
   && [ -f "$EM_MODELS/Wan2.1-Fun-V1.1-1.3B-InP/diffusion_pytorch_model.safetensors" ] \
   && [ -f "$EM_MODELS/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth" ]; then
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
