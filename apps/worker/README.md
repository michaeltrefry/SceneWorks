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
Override `PYTORCH_INDEX_URL` / `PYTORCH_SPEC` / `PYTORCH_AUDIO_SPEC` for direct
`docker build`, or `SCENEWORKS_PYTORCH_INDEX_URL` / `SCENEWORKS_PYTORCH_SPEC` /
`SCENEWORKS_PYTORCH_AUDIO_SPEC` for Compose builds.
GPU workers with CPU-only PyTorch intentionally register without generation
capabilities so jobs stay queued instead of crawling on CPU with an idle GPU.

Video adapter selection follows the selected model by default: LTX-2.3 uses the
native `ltx_pipelines` adapter, while Diffusers-compatible video models such as
Wan2.2 use `diffusers_video`. Docker Compose builds the worker with
`requirements-ltx.txt` by default. Set `SCENEWORKS_VIDEO_ADAPTER` only when you
explicitly want to force one adapter for all video jobs.

Native LTX-2.3 text-to-video and image-to-video require these local resources:

- `checkpointPath`
- `spatialUpscalerPath`
- `distilledLoraPath`
- `gemmaRoot`

Image-conditioned native LTX-2.3 modes such as image-to-video and first/last
frame route through `ltx_pipelines.ic_lora` when the selected preset includes
an installed LTX-compatible IC-LoRA. Without an IC-LoRA they fall back to the
standard distilled/two-stage LTX pipeline. Video-conditioned modes such as
extend require an installed IC-LoRA preset.

By default the worker resolves those from the model manifest `resources` block,
preferring SceneWorks-managed imports under `data/models` and then the shared
Hugging Face cache (`HUGGINGFACE_HUB_CACHE` or `HF_HOME/hub`). A job can
override them through matching advanced settings.
Use `advanced.mockNativeInference=true` only for adapter smoke tests; otherwise
the native adapter loads `ltx-pipelines` and writes MP4 video assets.

Use a local smoke check without contacting the API:

```powershell
python -m scene_worker --check
```
