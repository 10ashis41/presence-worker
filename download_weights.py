#!/usr/bin/env python3
"""
Fetch MuseTalk's model weights — replacing MuseTalk's own download_weights.sh,
which is broken as of 2026-09-19 and, worse, fails silently:

  * calls the deprecated `huggingface-cli` (now `hf`), which errors out
  * calls `gdown --id`, a flag removed from current gdown
  * points HF_ENDPOINT at hf-mirror.com, a China mirror that is slow or
    unreachable from US/EU pods
  * checks no exit codes, then unconditionally prints
    "✅ All weights have been downloaded successfully!"

That last one is the dangerous part: the install looks fine and the failure
only surfaces much later, mid-render, as "weights missing".

This version uses the stable huggingface_hub Python API, verifies every file
exists at a plausible size, and exits non-zero if anything is missing.

    python download_weights.py [--dir /workspace/MuseTalk/models]
"""
import argparse, subprocess, sys
from pathlib import Path

# repo_id -> [(remote path, local path relative to models/, min MB)]
HF_FILES = {
    "TMElyralab/MuseTalk": [
        ("musetalk/musetalk.json",      "musetalk/musetalk.json",      0.0001),
        ("musetalk/pytorch_model.bin",  "musetalk/pytorch_model.bin",  100),
        ("musetalkV15/musetalk.json",   "musetalkV15/musetalk.json",   0.0001),
        ("musetalkV15/unet.pth",        "musetalkV15/unet.pth",        100),
    ],
    "stabilityai/sd-vae-ft-mse": [
        ("config.json",                      "sd-vae/config.json",                      0.0001),
        ("diffusion_pytorch_model.bin",      "sd-vae/diffusion_pytorch_model.bin",      100),
    ],
    "openai/whisper-tiny": [
        ("config.json",              "whisper/config.json",              0.0001),
        ("pytorch_model.bin",        "whisper/pytorch_model.bin",        10),
        ("preprocessor_config.json", "whisper/preprocessor_config.json", 0.0001),
    ],
    "yzd-v/DWPose": [
        ("dw-ll_ucoco_384.pth", "dwpose/dw-ll_ucoco_384.pth", 50),
    ],
    "ByteDance/LatentSync": [
        ("latentsync_syncnet.pt", "syncnet/latentsync_syncnet.pt", 50),
    ],
}

# Not on HuggingFace — face-parse-bisent ships via Google Drive + torchvision.
GDRIVE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
GDRIVE_DEST = "face-parse-bisent/79999_iter.pth"
RESNET_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
RESNET_DEST = "face-parse-bisent/resnet18-5c106cde.pth"


def mb(p: Path) -> float:
    return p.stat().st_size / 1_048_576 if p.exists() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/MuseTalk/models")
    args = ap.parse_args()
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    failures = []

    for repo, files in HF_FILES.items():
        for remote, local, min_mb in files:
            dest = root / local
            if mb(dest) >= min_mb:
                print(f"  have  {local} ({mb(dest):.1f} MB)", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  get   {repo}:{remote}", flush=True)
            try:
                got = hf_hub_download(repo_id=repo, filename=remote)
                dest.write_bytes(Path(got).read_bytes())
            except Exception as e:
                failures.append(f"{repo}:{remote} — {e}")
                continue
            if mb(dest) < min_mb:
                failures.append(f"{local} too small ({mb(dest):.2f} MB < {min_mb} MB)")

    # face-parse-bisent: current gdown takes the id positionally, not --id
    dest = root / GDRIVE_DEST
    if mb(dest) < 10:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("  get   face-parse-bisent/79999_iter.pth (gdrive)", flush=True)
        r = subprocess.run(["gdown", GDRIVE_ID, "-O", str(dest)])
        if r.returncode != 0 or mb(dest) < 10:
            failures.append("face-parse-bisent/79999_iter.pth via gdown")
    else:
        print(f"  have  {GDRIVE_DEST} ({mb(dest):.1f} MB)", flush=True)

    dest = root / RESNET_DEST
    if mb(dest) < 10:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("  get   resnet18-5c106cde.pth", flush=True)
        r = subprocess.run(["curl", "-fsSL", RESNET_URL, "-o", str(dest)])
        if r.returncode != 0 or mb(dest) < 10:
            failures.append("resnet18-5c106cde.pth")
    else:
        print(f"  have  {RESNET_DEST} ({mb(dest):.1f} MB)", flush=True)

    if failures:
        print("\nFAILED to fetch:", file=sys.stderr)
        for f in failures:
            print("  -", f, file=sys.stderr)
        sys.exit(1)

    print("\nall weights verified present:", flush=True)
    for p in sorted(root.rglob("*")):
        if p.is_file() and mb(p) > 1:
            print(f"  {mb(p):8.1f} MB  {p.relative_to(root)}", flush=True)


if __name__ == "__main__":
    main()
