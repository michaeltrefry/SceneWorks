# SceneWorks Worker

Python sidecar package for Diffusers/PyTorch-backed image and video inference.

In Docker Compose this service polls the Rust API job queue, advertises only GPU
image/video generation capabilities, reports progress and heartbeats over HTTP,
and writes generated assets into the mounted project data directory. CPU utility
queues are owned by `sceneworks-rust-worker`; keep that service running
alongside the API.

Runtime dependencies live in `requirements.txt`. Test-only dependencies live in
`requirements-dev.txt` and are intentionally excluded from the Docker image.
The Dockerfile defaults to CPU-only PyTorch for direct image builds, while
`docker-compose.yml` passes CUDA PyTorch build args for the inference worker.
Override `PYTORCH_INDEX_URL` / `PYTORCH_SPEC` for direct `docker build`, or
`SCENEWORKS_PYTORCH_INDEX_URL` / `SCENEWORKS_PYTORCH_SPEC` for Compose builds.
GPU workers with CPU-only PyTorch intentionally register without generation
capabilities so jobs stay queued instead of crawling on CPU with an idle GPU.

Native LTX-2.3 work is staged behind `SCENEWORKS_VIDEO_ADAPTER=ltx_pipelines`.
The base worker does not install that stack by default; install
`requirements-ltx.txt` into an LTX-specific worker environment when validating
the native pipeline adapter.

Native LTX-2.3 text-to-video and image-to-video require these local resources:

- `checkpointPath`
- `spatialUpscalerPath`
- `distilledLoraPath`
- `gemmaRoot`

By default the worker resolves those from the model manifest `resources` block
under `data/models`. A job can override them through matching advanced settings.
Use `advanced.mockNativeInference=true` only for adapter smoke tests; otherwise
the native adapter loads `ltx-pipelines` and writes MP4 video assets.

Use a local smoke check without contacting the API:

```powershell
python -m scene_worker --check
```
