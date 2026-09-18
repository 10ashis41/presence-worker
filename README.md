# presence-worker

GPU worker for the AI Clone Presenter (aiguyonthefly.com/aiclone).
Pulls render jobs from the API, produces the clean final + watermarked preview.

## On a fresh RunPod / Vast.ai pod

```bash
git clone https://github.com/10ashis41/presence-worker.git /workspace/presence-worker
cd /workspace/presence-worker
export WORKER_TOKEN=...        # from api/.env on the VM
bash setup_pod.sh
```

Contains no secrets. WORKER_TOKEN is supplied at runtime.
