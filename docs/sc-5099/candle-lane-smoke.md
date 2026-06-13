# sc-5099 — candle (Windows/CUDA) worker lane: labeling, Python fallback, and live smoke

Epic 5095 wired seven candle-gen providers into the worker beyond SDXL:
**z-image, flux, flux2, qwen-image** (sc-5096, image), **wan, ltx** (sc-5097, video), and
**joycaption** (sc-5098, caption). This story (sc-5099) closes the labeling + Python-fallback surface
and defines the live deployed-worker smoke.

## Two-level gating (unchanged)

1. **Build feature** `backend-candle` — pulls the optional `candle-gen-*` provider crates (CUDA). Off
   by default so the Desktop/non-CUDA builds never touch candle/cudarc/nvcc.
2. **Runtime flag** `SCENEWORKS_BACKEND_CANDLE_ENABLED=1` (`Settings.backend_candle_enabled`) — until
   set, the worker links the providers but routes nothing to them (production routing unchanged).

Build the worker with the **VS2022 BuildTools MSVC 14.44** toolset (CUDA 12.9 rejects the newer
VS18/14.51), `CUDA_COMPUTE_CAP=120` for Blackwell sm_120. See the build-toolchain note.

## What runs on candle vs Python (the fallback matrix)

The candle lane is **txt2img / txt2video-only** + JoyCaption. The router
(`jobs_store::worker_supports_job` via the `candle` marker capability) confines the candle worker to
the advertised surface; everything else falls back to the co-resident Python torch worker, with **no
silently-dropped controls**.

| Job | Candle-eligible | Falls back to Python |
| --- | --- | --- |
| `image_generate` | `sdxl`, `realvisxl`, `z_image_turbo`, `flux_schnell`, `flux_dev`, `flux2_klein_9b`, `qwen_image` — plain txt2img | `edit_image` / `sourceAssetId` / `referenceAssetId` / `maskAssetId` / `advanced.poses` / `loras` / **`advanced.mlxQuantize > 0`**; any non-listed model id (incl. `*_edit`, `flux2_klein_9b_kv`, chroma, kolors, sensenova) |
| `video_generate` | `wan_2_2` (→ `wan2_2_ti2v_5b`), `ltx_2_3` (→ `ltx_2_3_distilled`) — `mode=text_to_video` | any other mode (i2v / first_last_frame / extend / bridge / replace), source/reference/mask, `loras`, `mlxQuantize > 0`; `wan_2_2_t2v_14b` / `wan_2_2_i2v_14b` / `svd` / `ltx_2_3_eros`; the advanced video job types |
| `training_caption` | `captioner=joy_caption` | any other captioner |

Per-asset backend labels (`candle_<family>`, distinct from the MLX `mlx_<family>`):
`candle_sdxl`, `candle_z_image`, `candle_flux`, `candle_flux2`, `candle_qwen` (image);
`candle_wan`, `candle_ltx` (video). Caption records `captioner=joy_caption` + `modelNameOrPath`.

The candle descriptors advertise `supported_quants: &[]` (dense bf16/fp16) and no LoRA/conditioning,
so the gates above mirror the descriptor surface 1:1.

## Live deployed-worker smoke (run on the Windows/CUDA box)

Goal: one real job per modality through the **deployed** worker (not a local `cargo run --example`),
confirming `assetWrites` streaming, progress, and mid-generation cancellation on the production path.

### Prerequisites
- Worker built `--features backend-candle` (MSVC 14.44 / CUDA 12.9) and deployed.
- `SCENEWORKS_BACKEND_CANDLE_ENABLED=1` in the worker environment.
- Model weights present in the HF cache (download via Model Manager):
  - image: `stabilityai/stable-diffusion-xl-base-1.0` (or any one wired image repo)
  - video: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` **or** `Lightricks/LTX-2.3` (+ `google/gemma-3-12b-it`
    for LTX; the worker sets `LTX_GEMMA_DIR` from that snapshot, else honors an explicit override)
  - caption: `fancyfeast/llama-joycaption-beta-one-hf-llava`

### Steps (pick one engine per modality)
1. **Image** — submit a plain txt2img job for a candle model (e.g. `sdxl`, prompt only, no
   conditioning/LoRA/quant). Expect: streamed `assetWrites` (PNG lands in the gallery), step progress
   with backend `candle`, asset `adapter = candle_sdxl`. Re-run and cancel mid-denoise → job ends
   `Canceled`, no partial asset indexed.
2. **Video** — submit a `text_to_video` job for `wan_2_2` (or `ltx_2_3`). Expect: a playable mp4
   asset, step progress (backend `candle`), `adapter = candle_wan` (`candle_ltx`). Cancel mid-denoise
   → `Canceled`.
3. **Caption** — submit a `training_caption` job with `captioner=joy_caption` over a small dataset.
   Expect: per-item progress, caption sidecars persisted, result `captioner=joy_caption`. Cancel
   mid-run → `Canceled`.
4. **Fallback spot-check** — submit one unsupported shape (e.g. an `sdxl` `edit_image`, or a
   `qwen_image` with `advanced.poses`, or `mlxQuantize: 8`) and confirm it runs on the **Python torch
   worker** (not the candle worker) and produces the expected torch-labeled asset — i.e. the control
   was honored, not dropped.

### Pass criteria
- One image, one video, and one caption engine each complete end-to-end on candle with correct
  `candle_<x>` labeling, streamed assetWrites, progress, and honored cancellation.
- The fallback spot-check runs on Python with the control honored.

> Status: code + routing + labeling + unit/routing tests landed in sc-5099. The **live GPU smoke is
> executed on the deployed Windows/CUDA box** (hardware lane) — record the run results here when done.
