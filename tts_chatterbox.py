#!/usr/bin/env python3
"""
Voice-cloned narration with Chatterbox (Resemble AI, MIT licence).

Uses the presenter's own recorded take as the voice reference — no training
step, no per-character fee. Multilingual V3 covers 23 languages including
**Arabic (ar) and Hebrew (he)**, and clones the same voice across all of them.

    python tts_chatterbox.py --reference take.mp4 --text-file script.txt \
        --out narration.wav --language he

Language handling:
  * `--language auto` (default) detects Hebrew/Arabic from the script's script
    block and falls back to English.
  * English uses the English-only checkpoint, which is slightly better for
    English than the multilingual one; everything else uses Multilingual V3.
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

import torch
import torchaudio

MAX_CHARS = 280  # keep each pass short enough for stable prosody

# Chatterbox Multilingual V3 language ids
SUPPORTED = {"ar","da","de","el","en","es","fi","fr","he","hi","it","ja","ko",
             "ms","nl","no","pl","pt","ru","sv","sw","tr","zh"}

HEBREW = re.compile(r"[֐-׿]")
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def detect_language(text: str) -> str:
    """Script-block detection. Deliberately simple: Hebrew and Arabic use
    distinct Unicode ranges, so counting characters is reliable — far more so
    than a statistical detector on a short script."""
    he, ar = len(HEBREW.findall(text)), len(ARABIC.findall(text))
    if he or ar:
        return "he" if he >= ar else "ar"
    return "en"


def extract_reference(src: Path, dest: Path):
    """Pull clean mono 24k audio out of whatever the phone recorded."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "24000", str(dest)],
        check=True,
    )


def chunk(text: str):
    # Arabic and Hebrew use the same .!? terminators, plus Arabic's ؟ and ۔
    sentences = re.split(r"(?<=[.!?؟۔])\s+", " ".join(text.split()))
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= MAX_CHARS:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="video/audio of the presenter's voice")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="auto",
                    help="ISO code (en, he, ar, …) or 'auto' to detect from the script")
    a = ap.parse_args()

    text = Path(a.text_file).read_text(encoding="utf-8")
    lang = detect_language(text) if a.language == "auto" else a.language.lower()
    if lang not in SUPPORTED:
        sys.exit(f"language '{lang}' is not supported by Chatterbox Multilingual "
                 f"(supported: {', '.join(sorted(SUPPORTED))})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"chatterbox on {device}, language={lang}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.wav"
        extract_reference(Path(a.reference), ref)

        if lang == "en":
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
            gen = lambda t: model.generate(t, audio_prompt_path=str(ref))
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
            gen = lambda t: model.generate(t, language_id=lang, audio_prompt_path=str(ref))

        pieces = chunk(text)
        print(f"{len(pieces)} chunk(s)", flush=True)

        waves = []
        for i, piece in enumerate(pieces, 1):
            print(f"  [{i}/{len(pieces)}] {piece[:60]}…", flush=True)
            waves.append(gen(piece))
            # a short gap keeps sentences from colliding at the seams
            waves.append(torch.zeros(1, int(model.sr * 0.28)))

        torchaudio.save(a.out, torch.cat(waves, dim=-1), model.sr)
        print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
