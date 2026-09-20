#!/usr/bin/env python3
"""Generate a talking video from a still photo with EchoMimicV3 (Apache-2.0).

Why this exists: EchoMimic's `infer_preview.py` has no CLI. Every path it needs lives in a
`Config` class, and its inputs come from a fixed
`datasets/echomimicv3_demos/{imgs,audios,prompts,masks}/<name>` layout. So there is nothing
to call — we lay our inputs out in that shape, patch the Config's defaults at runtime, and
run their script.

Patch at RUNTIME rather than vendoring a modified copy of their file: a vendored copy would
silently stop receiving upstream fixes, and the anchor we patch (`self.save_path`) is checked
so a refactor upstream fails loudly here instead of producing a video of the wrong length.

Env in -> env out:
  EM_IMAGE   still photo of the person (jpg/png)
  EM_AUDIO   narration wav; the video length is int(duration * 25 fps)
  EM_OUT     output directory; the video lands at <EM_OUT>/final.mp4
  EM_PROMPT  optional text prompt (defaults to a neutral speaking prompt)
  EM_REPO    EchoMimic checkout (default /workspace/echomimic_v3)
  EM_STEPS   sampling steps; 5 suits talking head, 15-25 for talking body
  EM_PARTIAL frames per generation chunk (default 113); larger = more VRAM
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(os.environ.get("EM_REPO", "/workspace/echomimic_v3"))
SCRIPT = REPO / "infer_preview.py"

# The exact line whose presence proves the Config body is still where we expect it. If
# upstream renames or moves this, we abort with a clear message rather than generating
# something subtly wrong (e.g. a clip that is one chunk long and stops mid-sentence).
ANCHOR = 'self.save_path = "outputs"'

# Second anchor: the import that pulls in retina-face (and therefore TensorFlow). The shim
# below has to land BEFORE it, because TensorFlow's visible-device list can only be changed
# while no GPU has been initialised yet.
FACE_ANCHOR = "from src.face_detect import get_mask_coord"

TF_CPU_SHIM = """# ---- injected by em_driver.py: keep TensorFlow on the CPU ----
# retina-face runs on TensorFlow, and TF's cuDNN does not match this container's CUDA 11.8 /
# driver 570.x. The moment it touches the GPU it dies with:
#     INTERNAL: No DNN support for stream [[{{node model/bn_data/FusedBatchNormV3}}]]
# — after the Wan transformer has already loaded, so ~3.5 minutes in.
#
# Face detection is ONE forward pass over ONE still image; on CPU that costs a second or two.
# The video diffusion that actually needs the GPU is PyTorch and is untouched by this.
#
# This must NOT be done with CUDA_VISIBLE_DEVICES: that variable would blind torch as well
# and move the entire render onto the CPU.
import tensorflow as _tf
try:
    _tf.config.set_visible_devices([], "GPU")
    print("     em: TensorFlow pinned to CPU (retina-face only)", flush=True)
except Exception as _e:  # already initialised, or no GPU to hide — harmless either way
    print(f"     em: could not pin TensorFlow to CPU ({_e})", flush=True)
"""

OVERRIDES = """
        # ---- injected by em_driver.py ----
        import os as _os
        self.base_dir = _os.environ["EM_BASE_DIR"]
        self.test_name_list = [_os.environ["EM_NAME"]]
        self.save_path = _os.environ["EM_SAVE"]
        self.model_name = _os.environ.get("EM_MODEL", self.model_name)
        self.transformer_path = _os.environ.get("EM_TRANSFORMER", self.transformer_path)
        self.wav2vec_model_dir = _os.environ.get("EM_WAV2VEC", self.wav2vec_model_dir)
        self.num_inference_steps = int(_os.environ.get("EM_STEPS", self.num_inference_steps))
        self.partial_video_length = int(_os.environ.get("EM_PARTIAL", self.partial_video_length))
        self.guidance_scale = float(_os.environ.get("EM_GUIDANCE", self.guidance_scale))
        self.audio_guidance_scale = float(
            _os.environ.get("EM_AUDIO_GUIDANCE", self.audio_guidance_scale))
        self.seed = int(_os.environ.get("EM_SEED", self.seed))
        self.fps = int(_os.environ.get("EM_FPS", self.fps))
        _size = int(_os.environ.get("EM_SIZE", 768))
        self.sample_size = [_size, _size]
        self.enable_teacache = _os.environ.get("EM_TEACACHE", "1") == "1"
"""


def log(msg: str) -> None:
    print(f"     em: {msg}", flush=True)


def main() -> int:
    image = pathlib.Path(os.environ["EM_IMAGE"])
    audio = pathlib.Path(os.environ["EM_AUDIO"])
    out_dir = pathlib.Path(os.environ["EM_OUT"])
    prompt = os.environ.get(
        "EM_PROMPT",
        "A person looks at the camera and speaks naturally, with calm facial expressions "
        "and small natural head movements.")
    name = "clip"

    for p in (image, audio):
        if not p.exists():
            log(f"FATAL missing input {p}")
            return 2
    if not SCRIPT.exists():
        log(f"FATAL {SCRIPT} not found — is EM_REPO right?")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the dataset layout their loader expects. No masks/<name>.npy on purpose: the
    # script then derives the head box itself with its own face detector.
    work = pathlib.Path(tempfile.mkdtemp(prefix="em-in-"))
    for sub in ("imgs", "audios", "prompts", "masks"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    ext = image.suffix.lower().lstrip(".") or "jpg"
    if ext not in ("png", "jpg", "jpeg"):
        ext = "jpg"
    shutil.copy(image, work / "imgs" / f"{name}.{ext}")
    shutil.copy(audio, work / "audios" / f"{name}.wav")
    (work / "prompts" / f"{name}.txt").write_text(prompt)

    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(audio)],
        capture_output=True, text=True).stdout.strip() or 0)
    log(f"photo={image.name} audio={duration:.2f}s -> {int(duration*25)} frames at 25fps")

    src = SCRIPT.read_text()
    if ANCHOR not in src:
        log("FATAL could not find the Config anchor in infer_preview.py — upstream changed; "
            "refusing to guess at paths")
        return 3
    if FACE_ANCHOR not in src:
        log("FATAL could not find the face_detect import in infer_preview.py — upstream "
            "changed; refusing to run with TensorFlow loose on the GPU")
        return 3

    # The patched copy must sit INSIDE the repo: the script does `from src... import ...`,
    # and python puts the *script's* directory on sys.path — not the working directory. A
    # copy in /tmp would fail to import src/ no matter what cwd we set.
    patched = REPO / "infer_eric.py"
    body = src.replace(ANCHOR, ANCHOR + OVERRIDES, 1)
    body = body.replace(FACE_ANCHOR, TF_CPU_SHIM + FACE_ANCHOR, 1)
    patched.write_text(body)
    log(f"patched Config + TF-CPU shim -> {patched.name}")

    env = dict(os.environ)
    env.update({
        # retina-face (EchoMimic's face detector) does `from tensorflow.keras.models import
        # Model`. That module was removed in TF 2.16 when Keras 3 became the default, and
        # retina-face only pins `tensorflow>=1.9.0` so the resolver is free to install it.
        # tf-keras (installed by install_echomimic.sh) provides the Keras 2 API and this flag
        # is what makes `tensorflow.keras` resolve to it. Set here rather than only at install
        # time so it holds even on a venv someone built by hand.
        "TF_USE_LEGACY_KERAS": "1",
        "EM_BASE_DIR": str(work), "EM_NAME": name, "EM_SAVE": str(out_dir),
        "EM_MODEL": str(REPO / "models/Wan2.1-Fun-V1.1-1.3B-InP"),
        "EM_TRANSFORMER": str(REPO / "models/transformer/diffusion_pytorch_model.safetensors"),
        "EM_WAV2VEC": str(REPO / "models/wav2vec2-base-960h"),
        "PYTHONUNBUFFERED": "1",
    })
    os.environ.pop("EM_SAVE", None)

    log("running inference (chunked; long audio takes a while)")
    proc = subprocess.run([sys.executable, str(patched)], cwd=str(REPO), env=env)
    if proc.returncode != 0:
        log(f"FATAL inference exited {proc.returncode}")
        return proc.returncode

    # Their script writes into save_path/<timestamp>_gs.._ags../ — find the newest video and
    # normalise its name so callers have one stable path.
    vids = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vids:
        log("FATAL no mp4 produced")
        return 4
    final = out_dir / "final.mp4"
    if vids[0] != final:
        shutil.copy(vids[0], final)
    log(f"done -> {final} ({final.stat().st_size/1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
