# sc-3496 — SceneWorks SDXL contract → Candle provider mapping

Epic 3494 (Candle SDXL Windows feasibility spike). This is the paper-mapping
deliverable: it maps the current SceneWorks SDXL/RealVisXL txt2img behavior onto
the smallest Candle implementation surface needed for a useful prototype, and
marks each behavior `REQUIRED` (for the prototype), `DEFERRED`, or `NOT NEEDED`.

Sources:
- Rust worker contract: `crates/sceneworks-core/src/image_request.rs`,
  `crates/sceneworks-worker/src/image_jobs.rs` (MLX_MODELS, generate_mlx_stream),
  mlx-gen `Generator` / `GenerationRequest` / `GenerationOutput`.
- Python reference: `apps/worker/scene_worker/image_adapters.py`
  (`SdxlDiffusersAdapter`, `MODEL_TARGETS`).
- Candle building blocks: `candle-transformers/src/models/stable_diffusion/*`
  and `candle-examples/examples/stable-diffusion/main.rs`
  (`StableDiffusionVersion::Xl` → `stabilityai/stable-diffusion-xl-base-1.0`).

---

## 1. Smallest useful parity target

**Base `sdxl` (stabilityai/stable-diffusion-xl-base-1.0), txt2img, 1024×1024,
single image, 30 steps, guidance 7.0, real CFG with negative prompt.**

Why this target:
- It is the canonical SDXL checkpoint and is the exact repo Candle's reference
  example already loads, so the prototype starts from a known-good Candle path.
- `realvisxl` (SG161222/RealVisXL_V5.0) is the **same UNet/VAE/text-encoder
  architecture** — in the Rust worker it even shares `engine_id = "sdxl"`. Once
  base SDXL runs, RealVisXL is a weights swap, not new code. Prove `sdxl` first,
  then point the loader at the RealVisXL snapshot to confirm.

---

## 2. Request fields (txt2img) — REQUIRED / DEFERRED / NOT NEEDED

`ImageRequest` (Rust) / `ImageRequest` (Python) fields, scoped to the prototype:

| Field | Prototype status | Notes |
|---|---|---|
| `prompt` | **REQUIRED** | Dual-CLIP encode (CLIP-L + CLIP-G/OpenCLIP bigG). |
| `negative_prompt` | **REQUIRED** | SDXL is real-CFG; negative honored. |
| `model` (`sdxl`/`realvisxl`) | **REQUIRED** | Map to repo + (later) snapshot dir. |
| `width` / `height` | **REQUIRED** | Must be %8==0; default 1024×1024. |
| `seed` (+ per-index) | **REQUIRED** | `seed + index`; see RNG caveat below. |
| `count` | **REQUIRED** | Loop N single-image generations (matches MLX path). |
| steps (`advanced.steps`) | **REQUIRED** | Default 30. |
| guidance (`advanced.guidanceScale`) | **REQUIRED** | Default 7.0. |
| dtype/precision | **REQUIRED** | f16 weights + SDXL fp16-VAE-fix (see §4). |
| sampler/scheduler selection | **DEFERRED** | Prototype uses one fixed scheduler (§3). |
| `loras` | **DEFERRED** | Candle has no turnkey SDXL LoRA loader; out of scope. |
| `source_asset_id` (img2img/edit) | **DEFERRED** | txt2img only this spike. |
| `mask_asset_id` (inpaint/outpaint) | **DEFERRED** | Excluded by epic. |
| `reference_asset_id` (IP-Adapter) | **DEFERRED** | Excluded by epic. |
| tile-ControlNet detail | **DEFERRED** | Excluded by epic. |
| `style_preset` | **NOT NEEDED** | Prompt-only concern, no model effect here. |
| `character_id` / `character_look_id` | **NOT NEEDED** | Higher-level orchestration. |
| `fit_mode` | **NOT NEEDED** | Only relevant to edit/outpaint. |

---

## 3. Component inventory — what Candle provides vs what we implement

| Component | Candle provides? | Work for prototype |
|---|---|---|
| CLIP-L + CLIP-G tokenizers | yes (`clip.rs` + HF tokenizers) | wire up both tokenizers |
| Dual text encoders + pooled embed | yes (`clip.rs`) | assemble SDXL added-cond (pooled + time-ids) |
| UNet2DConditionModel (SDXL) | yes (`unet_2d.rs`, `sdxl_` config) | load weights from safetensors |
| VAE (AutoEncoderKL) | yes (`vae.rs`) | use SDXL scale factor 0.13025 (§4) |
| Scheduler | partial (`euler_ancestral_discrete`, `ddim`, `uni_pc`, `ddpm`) | **parity gap, see §3a** |
| safetensors loading | yes (`candle_core::safetensors`) | resolve component files |
| HF cache resolution | reuse worker helper | feed snapshot dir to loader |
| RGB8 image encode → PNG | trivial (image crate) | match worker `Image{w,h,pixels}` |
| Progress / cancellation | n/a (worker-side) | adapt loop to emit Step events + check CancelFlag |

### 3a. Scheduler parity gap (flag for sc-3497/sc-3498)
SceneWorks/diffusers SDXL defaults to **EulerDiscreteScheduler** (non-ancestral,
deterministic). Candle ships **EulerAncestralDiscreteScheduler** (adds noise each
step → not identical, less reproducible), plus DDIM and UniPC. The prototype
should pick the closest available (likely DDIM or UniPC for determinism, or
Euler-ancestral for visual similarity) and **document the divergence** — this is
a known, acceptable spike gap, not a blocker. Exact-match Euler-discrete would be
a small follow-up port.

---

## 4. Parity details that must be correct

- **VAE scaling factor = 0.13025** for SDXL (NOT 0.18215, which is SD1.5). Candle's
  SDXL config carries the correct value; just don't hardcode the SD1.5 number.
- **fp16 VAE fix:** loading SDXL in f16 needs `madebyollin/sdxl-vae-fp16-fix` to
  avoid NaNs/black images — Candle's example already special-cases this for `Xl`.
- **Resolution constraint:** width/height must be divisible by 8 (Candle asserts).
- **Default dims 1024×1024**, steps 30, guidance 7.0 (match `MLX_MODELS` /
  `MODEL_TARGETS` `sdxl` entry).

---

## 5. RNG / determinism expectation (important for sc-3498)

Python SDXL seeds a `torch.Generator` (CUDA). Candle uses its **own RNG**, so
Candle output will **not bit-match** Python for the same seed. Expectation:
- Candle is **internally deterministic** given a fixed seed (same seed → same image).
- Candle vs Python is a **distributional / qualitative** comparison, never pixel-exact.
This must be stated explicitly so validation (sc-3498) compares correctness and
aesthetics, not byte equality.

---

## 6. Output / sidecar metadata (REQUIRED)

The Candle path must emit the same streaming result the MLX path does
(`image_jobs.rs` streaming_result + assetWrites). Minimum per-image facts:
`assetId, mediaPath (PNG), mimeType, width, height, normalizedWidth/Height,
seed, index, model, adapter, prompt, negativePrompt, rawAdapterSettings
{repo, numInferenceSteps, guidanceScale, scheduler, realModelInference:true}`.
Use a distinct adapter label, e.g. **`candle_sdxl`**, parallel to `mlx_sdxl`.

---

## 7. Explicit exclusions for this spike

IP-Adapter / reference, img2img / inpaint / outpaint, tile-ControlNet detail,
arbitrary LoRA/LyCORIS, scheduler/sampler selection UI, training. All `DEFERRED`
or `NOT NEEDED` above; the prototype must **fail clearly** on these shapes rather
than silently dropping the control (sc-3497 acceptance).
