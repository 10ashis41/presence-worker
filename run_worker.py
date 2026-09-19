#!/usr/bin/env python3
"""
Presence worker — claims render jobs from the API and produces the two output
files (clean final + watermarked preview).

Two stages are pluggable so the same worker runs with or without a GPU:

  TTS_BACKEND      chatterbox  GPU, free, clones the presenter's own voice   [production]
                   elevenlabs  API, works anywhere, costs per character      [no-GPU fallback]

  LIPSYNC_BACKEND  latentsync  GPU, 512px + temporal layers                  [DEFAULT]
                   musetalk    GPU, 256px — visibly "melts", rejected        [speed only]
                   passthrough ffmpeg only — muxes narration over the take   [PIPELINE TEST ONLY]

`passthrough` exists so the whole system (API, queueing, watermarking, upload,
paywall, front end) can be exercised on a machine with no GPU. Its output is NOT
a clone — the mouth does not match the words — so it burns an unmissable label
into the video. Never ship it to a client.

The worker PULLS work: the pod needs no inbound network, public IP or port
forwarding, and can be started and killed freely.

Env:
    API_BASE        https://api.aiguyonthefly.com/presenter
    WORKER_TOKEN    bearer token from the API's .env
    TTS_BACKEND     chatterbox | elevenlabs          (default chatterbox)
    LIPSYNC_BACKEND latentsync | musetalk | passthrough  (default latentsync)
    ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID         (elevenlabs backend only)
    MUSETALK_DIR    /opt/MuseTalk                    (musetalk backend only)
    NARRATION_LUFS  loudness target for the narration BEFORE lip sync, in LUFS.
                    The mouth shapes are generated from this audio, so a quiet
                    track starves the model of signal — and it fails worst at
                    lip closures (m/b/p), where the mouth blotches. 0/off keeps
                    the raw TTS level.                        (default -16)
    LATENTSYNC_STEPS      20-50, higher = finer detail + slower   (default 20)
    LATENTSYNC_GUIDANCE   1.0-3.0, higher = tighter sync, more jitter (1.5)
    LATENTSYNC_SEED       pins the sampling seed so A/B tests compare the
                          settings and not luck                     (1247)
    LATENTSYNC_DEEPCACHE  1 = fast but an approximation that can smear
                          high-frequency mouth detail; 0 = quality pass (1)
    POLL_SECONDS    idle poll interval               (default 20)
    IDLE_EXIT       exit after N idle seconds so a rented pod can shut itself
                    down instead of billing you to sit idle (0 = never)
    ONESHOT         1 = process a single job then exit (used by the smoke test)
"""
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import urllib.request, urllib.error

API = os.environ.get("API_BASE", "https://api.aiguyonthefly.com/presenter").rstrip("/")
TOKEN = os.environ["WORKER_TOKEN"]
POLL = int(os.environ.get("POLL_SECONDS", "20"))
IDLE_EXIT = int(os.environ.get("IDLE_EXIT", "0"))
ONESHOT = os.environ.get("ONESHOT") == "1"
TTS_BACKEND = os.environ.get("TTS_BACKEND", "chatterbox")
LIPSYNC_BACKEND = os.environ.get("LIPSYNC_BACKEND", "latentsync")
HERE = Path(__file__).parent


def log(*a):
    print(*a, flush=True)


def call(method, path, data=None, raw=None, timeout=900):
    req = urllib.request.Request(API + path, method=method)
    req.add_header("Authorization", "Bearer " + TOKEN)
    body = None
    if raw is not None:
        body = raw
        req.add_header("Content-Type", "application/octet-stream")
    elif data is not None:
        body = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, body, timeout=timeout) as r:
        if r.status == 204:
            return None
        payload = r.read()
        return json.loads(payload) if payload else None


def download(path, dest):
    req = urllib.request.Request(API + path)
    req.add_header("Authorization", "Bearer " + TOKEN)
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def run(cmd, **kw):
    log("  $", " ".join(str(c) for c in cmd)[:160])
    subprocess.run(cmd, check=True, **kw)


def probe_duration(p: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(p)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


# ---------------------------------------------------------------- TTS
HEBREW_RE = __import__("re").compile(r"[\u0590-\u05FF]")
ARABIC_RE = __import__("re").compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def detect_language(text: str) -> str:
    he, ar = len(HEBREW_RE.findall(text)), len(ARABIC_RE.findall(text))
    if he or ar:
        return "he" if he >= ar else "ar"
    return "en"


def tts_elevenlabs(script: str, out: Path):
    key = os.environ["ELEVENLABS_API_KEY"]
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "rQOBu7YxCDxGiFdTm28w")
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
        data=json.dumps({
            "text": script,
            # multilingual_v2 covers Arabic and Hebrew; the model infers the
            # language from the text itself, so no language flag is needed.
            "model_id": os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            "voice_settings": {"stability": 0.6, "similarity_boost": 0.8, "speed": 0.95},
        }).encode(),
        headers={"xi-api-key": key, "content-type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)


def tts_chatterbox(script: str, reference: Path, out: Path, work: Path, lang: str):
    # Chatterbox lives in its own venv: it needs a much newer torch than
    # MuseTalk, and sharing one environment breaks mmcv's compiled ops
    # (undefined symbol at import). CHATTERBOX_PYTHON points at that venv.
    py = os.environ.get("CHATTERBOX_PYTHON", sys.executable)
    txt = work / "script.txt"
    txt.write_text(script, encoding="utf-8")
    run([py, str(HERE / "tts_chatterbox.py"),
         "--reference", str(reference), "--text-file", str(txt),
         "--out", str(out), "--language", lang])


def normalize_narration(src: Path, lufs: str) -> None:
    """Loudness-normalise the narration in place (single-pass loudnorm).

    This is NOT cosmetic. The lip-sync model derives mouth shapes from this
    audio, and a quiet track gives its audio encoder weak features. That fails
    hardest exactly where it matters: at lip closures (m/b/p), the model has
    the least information to work with, and the mouth collapses into a blotch.

    Measured on the first LatentSync render: mean -27.9 dB, peak -9.0 dB — quiet
    for speech and ~10 dB below a normal delivery. Set NARRATION_LUFS=0 (or
    "off") to disable and keep the raw TTS level.
    """
    if lufs in ("", "0", "off", "none"):
        return
    out = src.with_name("narration_norm" + src.suffix)
    acodec = ["-c:a", "pcm_s16le"] if src.suffix.lower() == ".wav" else \
             ["-c:a", "libmp3lame", "-b:a", "192k"]
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
         "-vn", *acodec, str(out)])
    shutil.move(str(out), str(src))
    log(f"  narration normalised to {lufs} LUFS")


# ------------------------------------------------------------ lip sync
def lipsync_musetalk(take: Path, narration: Path, work: Path) -> Path:
    """MuseTalk 1.5 does NOT accept --video_path/--audio_path. It reads a YAML
    task config and needs explicit unet paths + a version flag (see its
    inference.sh). Verified against the upstream repo 2026-09-18."""
    musetalk = Path(os.environ.get("MUSETALK_DIR", "/opt/MuseTalk"))
    version = os.environ.get("MUSETALK_VERSION", "v15")     # v15 (1.5) or v1
    model_dir = musetalk / ("models/musetalkV15" if version == "v15" else "models/musetalk")
    unet_path = model_dir / ("unet.pth" if version == "v15" else "pytorch_model.bin")
    unet_cfg = model_dir / "musetalk.json"
    for p_ in (unet_path, unet_cfg):
        if not p_.exists():
            raise RuntimeError(f"MuseTalk weights missing: {p_} — run download_weights.sh")

    cfg = work / "task.yaml"
    # MuseTalk resolves these relative to its own cwd, so pass absolute paths.
    cfg.write_text(
        "task_0:\n"
        f" video_path: \"{take.resolve()}\"\n"
        f" audio_path: \"{narration.resolve()}\"\n",
        encoding="utf-8")

    outdir = work / "musetalk_out"
    outdir.mkdir(exist_ok=True)
    run([sys.executable, "-m", "scripts.inference",
         "--inference_config", str(cfg),
         "--result_dir", str(outdir),
         "--unet_model_path", str(unet_path),
         "--unet_config", str(unet_cfg),
         "--version", version],
        cwd=str(musetalk))

    produced = sorted(outdir.rglob("*.mp4"), key=lambda f: f.stat().st_mtime)
    if not produced:
        raise RuntimeError(f"MuseTalk wrote no .mp4 under {outdir}")
    return produced[-1]


def lipsync_latentsync(take: Path, narration: Path, work: Path) -> Path:
    """LatentSync 1.6 (ByteDance, Apache-2.0) — the quality option.

    Runs at 512px vs MuseTalk's 256, and its temporal layers + TREPA are what
    fix MuseTalk's frame-to-frame "melting". Much slower in exchange.

    Lives in its own venv: it pins torch 2.5.1 against MuseTalk's 2.0.1.
    Args verified against upstream inference.sh, 2026-09-19.
    """
    ls_dir = Path(os.environ.get("LATENTSYNC_DIR", "/workspace/LatentSync"))
    py = os.environ.get("LATENTSYNC_PYTHON", sys.executable)
    ckpt = ls_dir / "checkpoints" / "latentsync_unet.pt"
    if not ckpt.exists():
        raise RuntimeError(f"LatentSync checkpoint missing: {ckpt}")

    out = work / "latentsync_out.mp4"
    steps = os.environ.get("LATENTSYNC_STEPS", "20")        # 20-50, higher = better + slower
    guidance = os.environ.get("LATENTSYNC_GUIDANCE", "1.5")  # 1.0-3.0, higher = tighter sync, more jitter
    # Pin the seed so an A/B between settings compares the settings, not luck,
    # and so a good render can be reproduced. 1247 is LatentSync's own default.
    seed = os.environ.get("LATENTSYNC_SEED", "1247")
    # DeepCache caches UNet features to go faster. It is an APPROXIMATION, and
    # it lands on exactly the high-frequency detail we care about — lips and
    # teeth. Default keeps today's behaviour (on); set LATENTSYNC_DEEPCACHE=0
    # for a quality pass. It is the cheapest A/B against mouth blotching.
    deepcache = os.environ.get("LATENTSYNC_DEEPCACHE", "1") not in ("0", "false", "off", "")
    args = [py, "-m", "scripts.inference",
            "--unet_config_path", str(ls_dir / "configs/unet/stage2_512.yaml"),
            "--inference_ckpt_path", str(ckpt),
            "--inference_steps", steps,
            "--guidance_scale", guidance,
            "--seed", seed]
    if deepcache:
        args.append("--enable_deepcache")
    args += ["--video_path", str(take.resolve()),
             "--audio_path", str(narration.resolve()),
             "--video_out_path", str(out)]
    log(f"  latentsync: steps={steps} guidance={guidance} seed={seed} deepcache={deepcache}")
    run(args, cwd=str(ls_dir))
    if not out.exists():
        raise RuntimeError("LatentSync produced no output file")
    return out


def lipsync_passthrough(take: Path, narration: Path, work: Path) -> Path:
    """No lip sync. Loops the take to cover the narration and burns in a label
    so this can never be mistaken for a finished clone."""
    out = work / "passthrough.mp4"
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    dur = probe_duration(narration)
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-stream_loop", "-1", "-i", str(take),
         "-i", str(narration),
         "-map", "0:v:0", "-map", "1:a:0", "-t", f"{dur:.2f}",
         "-vf", f"drawtext=fontfile={font}:text='PIPELINE TEST — NO LIP SYNC':"
                "fontcolor=yellow:fontsize=w/22:box=1:boxcolor=black@0.6:boxborderw=8:"
                "x=(w-tw)/2:y=h*0.06",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
    return out


# ---------------------------------------------------------------- job
def render(job, work: Path):
    jid = job["id"]
    take = work / ("take." + job.get("ext", "mp4"))
    download(f"/jobs/{jid}/take", take)
    log(f"  take {take.stat().st_size/1e6:.1f} MB, {probe_duration(take):.1f}s")

    # 1. narration
    call("POST", f"/jobs/{jid}/progress", {"step": 2})
    narration = work / ("narration.mp3" if TTS_BACKEND == "elevenlabs" else "narration.wav")
    lang = job.get("language") or detect_language(job["script"])
    log(f"  tts: {TTS_BACKEND}, language={lang}")
    if TTS_BACKEND == "elevenlabs":
        tts_elevenlabs(job["script"], narration)
    else:
        tts_chatterbox(job["script"], take, narration, work, lang)
    # Loudness-normalise before lip sync: the mouth shapes are generated from
    # THIS audio, and a quiet track starves the sync model of signal — worst at
    # lip closures, where it has the least to work with. NARRATION_LUFS=0 skips.
    normalize_narration(narration, os.environ.get("NARRATION_LUFS", "-16"))
    log(f"  narration {probe_duration(narration):.1f}s")

    # 2. lip sync
    call("POST", f"/jobs/{jid}/progress", {"step": 3})
    log(f"  lipsync: {LIPSYNC_BACKEND}")
    backends = {"musetalk": lipsync_musetalk,
                "latentsync": lipsync_latentsync,
                "passthrough": lipsync_passthrough}
    if LIPSYNC_BACKEND not in backends:
        raise RuntimeError(f"unknown LIPSYNC_BACKEND: {LIPSYNC_BACKEND}")
    produced = backends[LIPSYNC_BACKEND](take, narration, work)
    final = work / "final.mp4"
    shutil.move(str(produced), final)
    log(f"  final {final.stat().st_size/1e6:.1f} MB, {probe_duration(final):.1f}s")

    # 3. watermarked preview
    call("POST", f"/jobs/{jid}/progress", {"step": 4})
    preview = work / "preview.mp4"
    run(["bash", str(HERE / "watermark.sh"), str(final), str(preview)])

    # 4. upload both, then mark done
    for kind, p in (("final", final), ("preview", preview)):
        with open(p, "rb") as f:
            call("PUT", f"/jobs/{jid}/result/{kind}", raw=f.read(), timeout=1800)
        log(f"  uploaded {kind} ({p.stat().st_size/1e6:.1f} MB)")
    call("POST", f"/jobs/{jid}/done", {})
    log(f"  DONE {jid}")


def main():
    log(f"worker up — api={API} tts={TTS_BACKEND} lipsync={LIPSYNC_BACKEND}")
    if LIPSYNC_BACKEND == "passthrough":
        log("  !! passthrough mode: output is NOT a clone. Pipeline testing only.")
    idle = 0
    while True:
        try:
            job = call("GET", "/work", timeout=60)
        except urllib.error.HTTPError as e:
            log("api error", e.code); time.sleep(POLL); continue
        except Exception as e:
            log("unreachable:", e); time.sleep(POLL); continue

        if not job:
            if ONESHOT:
                log("no queued work"); return 0
            idle += POLL
            if IDLE_EXIT and idle >= IDLE_EXIT:
                log(f"idle {idle}s — exiting so the pod can shut down"); return 0
            time.sleep(POLL); continue

        idle = 0
        log("claimed", job["id"])
        work = Path(tempfile.mkdtemp(prefix="presence-"))
        try:
            render(job, work)
            ok = True
        except Exception as e:
            log("FAILED:", repr(e))
            ok = False
            try: call("POST", f"/jobs/{job['id']}/done", {"error": str(e)[:300]})
            except Exception: pass
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if ONESHOT:
            return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
