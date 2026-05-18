# SceneWorks Worker

Python sidecar package for Diffusers/PyTorch-backed image and video inference.

In Docker Compose this service polls the Rust API job queue, advertises only GPU
image/video generation capabilities, reports progress and heartbeats over HTTP,
and writes generated assets into the mounted project data directory. CPU utility
queues are owned by `sceneworks-rust-worker`; keep that service running
alongside the API.

Runtime dependencies live in `requirements.txt`. Test-only dependencies live in
`requirements-dev.txt` and are intentionally excluded from the Docker image.
The Dockerfile preinstalls CPU-only PyTorch by default to avoid accidentally
vendoring the full CUDA wheel stack; override `PYTORCH_INDEX_URL` and
`PYTORCH_SPEC` at build time when a CUDA-enabled image is required.

Use a local smoke check without contacting the API:

```powershell
python -m scene_worker --check
```
