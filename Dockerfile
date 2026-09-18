# Presence GPU worker — build once, run on any rented GPU pod.
#
#   docker build -t presence-worker .
#   docker run --gpus all -e WORKER_TOKEN=... -e API_BASE=... presence-worker
#
# CUDA 12.1 + cuDNN runtime matches the torch build below. Everything in here
# is MIT/Apache licensed, which is what makes this resellable — see LICENSES.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip git ffmpeg fonts-dejavu-core wget \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Torch first so the heavy layer caches independently of app code.
RUN pip3 install --no-cache-dir \
      torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# --- MuseTalk (MIT — code and weights, commercial use explicitly allowed) ---
ENV MUSETALK_DIR=/opt/MuseTalk
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git $MUSETALK_DIR
WORKDIR $MUSETALK_DIR
RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install --no-cache-dir --no-deps openmim \
 && mim install "mmengine" "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0" || true

# Model weights are fetched at build time so a cold pod doesn't pay to download
# them on every start. Expect this layer to be large (~10GB).
RUN bash ./download_weights.sh || echo "WARN: fetch weights at runtime instead"

# --- Chatterbox TTS (MIT, Resemble AI) -------------------------------------
RUN pip3 install --no-cache-dir chatterbox-tts

WORKDIR /app
COPY run_worker.py tts_chatterbox.py watermark.sh ./
RUN chmod +x watermark.sh

ENV API_BASE=https://api.aiguyonthefly.com/presenter \
    POLL_SECONDS=20 \
    IDLE_EXIT=0
CMD ["python", "/app/run_worker.py"]
