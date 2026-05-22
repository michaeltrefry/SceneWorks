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

## Replace Person: detection, tracking, segmentation (`person_detect` / `person_track`)

Real person detection and tracking run here, in the GPU worker, replacing the
Rust utility worker's procedural placeholders (epic sc-1090). The backends live
in `scene_worker/person_adapters.py` and are imported lazily, so the worker stays
importable without them:

- **Detection (`person_detect`)** runs Ultralytics YOLO over a representative
  frame and returns pixel-derived candidate boxes with confidence and
  adapter/model/runtime metadata. No-person frames return no candidates.
- **Tracking (`person_track`)** follows the selected detection through real
  source-video frames with YOLO + ByteTrack/BoT-SORT, samples the track, and
  writes a reusable person-track sidecar. Lost/low-confidence frames are recorded
  honestly (`detected: false`, flagged) rather than fabricated.
- **Segmentation** generates per-frame person masks with SAM2 (or MatAnyone),
  stored under `person-tracks/{id}/masks/`. Without a segmenter the track records
  `maskState: degraded` and replacement falls back to box masks.

Install these backends with `apps/worker/requirements-person.txt` (Docker Compose
builds them by default; opt out with `INCLUDE_PERSON_BACKENDS=0`). A GPU worker
advertises `person_detect` / `person_track` / `person_segment` only when the
corresponding backend is installed, so a real job never routes to a worker that
cannot run it. The Rust utility worker advertises `person_detect_preview` /
`person_track_preview` and serves the procedural preview only for jobs created
with `preview: true`.

**Replace Person (`replace_person`)** on LTX-2.3 runs through the native
`ltx_pipelines` adapter: it builds a masked control clip from the source frames +
person-track masks + masking strength and injects it through the IC-LoRA
video-conditioning path. The output sidecar sets `replacementActive: true` only
when that real masked-control path ran (mocked/preview runs leave it false). Wan
remains a secondary VACE/masked-conditioning path. Replacement requires a source
clip, a saved person track, an approved character reference, and an installed
LTX IC-LoRA; the job fails clearly when any are missing.

## Native LoRA training (`lora_train`)

Training jobs carry a fully normalized, Rust-resolved `TrainingPlan` (see
`crates/sceneworks-core/src/training.rs`). The worker's kernels in
`scene_worker/training_adapters.py` consume only that plan — they never read
SceneWorks storage, config defaults, or the target registry directly.

- **Dry run** (`payload.dryRun == true`, the default) validates the plan and that
  its dataset images exist on the worker, then reports what a real run would
  produce. It loads no model and needs no inference backend, so a GPU worker
  advertises `lora_train` even without torch/CUDA.
- **Real run** (`payload.dryRun == false`) routes to the kernel named by
  `plan.target.kernel`. The first kernel is `z_image_lora`
  (:class:`ZImageLoraTrainer`), an image LoRA trainer for Z-Image-Turbo. It loads
  the diffusers `ZImagePipeline` components (transformer, VAE, Qwen3 text encoder)
  via `from_pretrained`, attaches a PEFT LoRA to the transformer, caches per-item
  latents and prompt embeddings, runs a flow-matching loop (the raw transformer-output
  target is `latents - noise` — the negated velocity, since the pipeline negates the
  output before the scheduler — at timestep `(1000 - t) / 1000`), and writes a `.safetensors`
  adapter with `ZImagePipeline.save_lora_weights`. It reports preparing,
  loading, caching, training, checkpointing (every `saveEvery` steps), and saving
  stages, and honors cancellation between steps. A real run requires the
  inference backend; the kernel reports clearly when it is missing.
- **Native MLX video LoRA** (`ltx_mlx_lora`, :class:`LtxMlxLoraTrainer`, Apple
  Silicon only — gated by `target.requiresAppleSilicon`) trains an LTX-2.3 video
  LoRA from a still-image dataset entirely in MLX. It loads the quantized
  AudioVideo transformer (`notapalindrome/ltx23-mlx-av-q4`) plus the LTX VAE
  encoder and gemma text encoder, freezes the base, injects rank-r LoRA into the
  `attn1`/`attn2` projections, caches each still as a single-frame latent
  (`encode_image`, already per-channel normalized) and a caption context embed,
  then runs a rectified-flow loop. The raw transformer output is regressed to
  `noise - clean` at timestep `sigma` (no sign flip — unlike the diffusers
  Z-Image path above, the LTX `to_denoised` consumes the output directly). The
  adapter is saved keyed by the real module paths (`{module}.lora_A.weight` /
  `.lora_B.weight` + scalar `.alpha`) so `mlx_video.lora` round-trips at
  inference with no key remap. Validated end-to-end: a rank-32 / 1500-step run on
  ~76 stills (res 512, trigger-focused captions) produces a clearly attributable
  identity effect through the real `MlxVideoAdapter` generation path. Practical
  footprint: ~1.35 s/step, **peak ~59 GB during training (needs a 64 GB+ Mac)**
  and ~34 GB during generation; the gemma text encoder stays resident through the
  loop, which dominates the training peak.

The kernel produces the weights file and a result summary. Registering the
produced adapter as a usable SceneWorks LoRA (with provenance and Image Studio
compatibility) is owned by the Rust API: on a completed real run it recomputes
the scope's LoRA manifest path from trusted inputs and upserts the entry (story
1418). See [documents/TRAINING_QUICKSTART.md](../../documents/TRAINING_QUICKSTART.md)
for the end-to-end flow and troubleshooting.

Training reuses the runtime dependencies in `requirements.txt` (torch, diffusers,
transformers, peft, accelerate, safetensors). The `adamw8bit` optimizer uses
`bitsandbytes` when present and falls back to torch AdamW otherwise. The
`prodigyopt` optimizer uses the Prodigy package and follows the ai-toolkit
convention of raising very small configured learning rates to `1.0`. During real
training, `config.advanced.sampleEvery` renders up to four prompts from
`config.advanced.samplePrompts` into the LoRA output directory and streams their
paths through job progress. Override PEFT target modules per job via
`config.advanced.loraTargetModules` when a base model names its attention layers
differently.

Use a local smoke check without contacting the API:

```powershell
python -m scene_worker --check
```
