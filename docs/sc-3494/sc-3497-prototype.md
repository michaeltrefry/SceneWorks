# sc-3497 — Candle SDXL txt2img prototype behind a worker-shaped seam: PASS

Epic 3494. A narrow Candle-backed SDXL txt2img prototype runs on Windows/CUDA
(Blackwell sm_120), produces a real PNG, emits worker-style progress, and writes
through a SceneWorks-worker-shaped asset/reporting contract. Unsupported request
shapes fail loudly. Production-routing untouched.

Code: `src/bin/candle_sdxl.rs` (spike crate). Build/run: `run_sdxl.ps1`.

## What was proven

- **Real image on Blackwell:** base `sdxl` (stabilityai/stable-diffusion-xl-base-1.0),
  1024×1024, 30 steps, guidance 7.0, from cached HF weights → coherent, well-exposed
  SDXL image (`out/assets/images/genset_01/asset_57649fa60.png`).
- **Performance:** ~0.32 s/step (~9.7 s of UNet denoise for 30 steps); ~33 s end-to-end
  including model load + first-run downloads (fp16-VAE-fix + 2 tokenizers). f16 on cuda:0.
- **Worker-shaped contract output** (`out/result.json`): `generationSetId`,
  `expectedCount`, `adapter="candle_sdxl"`, `model`, and per-image `assetWrites`
  (assetId, mediaPath, mimeType, width/height, normalizedWidth/Height, seed, index,
  model, adapter, prompt, negativePrompt, `rawAdapterSettings`{repo, numInferenceSteps,
  guidanceScale, scheduler, vaeScale, realModelInference, backend}).
- **Progress events** mirroring the worker: `image_inference_start`, per-step `step`
  {current,total,stepMs,backend:"cuda"}, `image_inference_complete`.
- **Gating (fails clearly, no silent drops):**
  - `--lora` → "does not support LoRA yet (DEFERRED) — refusing to silently drop it"
  - `--source-asset`/`--mask-asset`/`--reference-asset` → txt2img-only rejections
  - unknown model (e.g. `kolors`) → "supports only sdxl/realvisxl txt2img"
  - width/height not %8 → rejected.

## Pipeline (Candle building blocks used)
`candle_transformers::models::stable_diffusion`: dual CLIP (CLIP-L + CLIP-bigG) via
`build_clip_transformer`, `StableDiffusionConfig::sdxl`, `build_unet`, `build_vae`,
Euler-ancestral scheduler (`build_scheduler`). Seed via `Device::set_seed`.

## Decisions / corrections baked in
- **VAE scale = 0.13025** (diffusers-correct SDXL value). Candle's example hardcodes
  0.18215 for `Xl` (the SD1.5 value); 0.13025 produced correctly-exposed output here.
- **fp16 VAE fix** (`madebyollin/sdxl-vae-fp16-fix`) required for f16 to avoid NaNs;
  matches the Python adapter / candle example behavior.
- CLIP weights loaded as F32 even though the fp16 safetensors file is used (matches
  candle's reference); UNet/VAE in F16.

## Honest gap (sc-3497 acceptance nuance)
The prototype emits the worker **contract shapes** but is a **standalone binary**, not
yet linked into the production `rust-worker` crate. True production wiring — an external
`candle-gen` sibling crate (mirroring `mlx-gen`) registered behind a `cfg(target_os=
"windows")` + feature-flag lane in `crates/sceneworks-worker/src/image_jobs.rs`, with a
`candle` GPU capability in `gpu.rs` — is intentionally deferred to a follow-up (sc-3499).
This keeps production routing untouched during the spike, as the story requires.

## Known parity gaps to carry into sc-3498
- Scheduler is Euler-**ancestral** (Candle) vs Euler-**discrete** (SceneWorks/diffusers).
- Candle RNG ≠ torch RNG → no pixel-exact match to Python for a given seed.
- `realvisxl` mapped (same architecture) but not yet run; verify its HF layout exposes
  the diffusers component safetensors (it may ship single-file only).
