# Train your first LoRA (Z-Image-Turbo)

SceneWorks can train an image LoRA for **Z-Image-Turbo** entirely on your own
machine. This guide walks through the first run: preparing a dataset, validating
the plan with a dry run, training for real, and using the result in Image Studio.

Native LoRA training is Rust-first: Rust owns the dataset store, training
contracts, the target registry, queue routing, and LoRA registration. The Python
worker is a narrow execution kernel that consumes a fully resolved plan and
produces weights. See `crates/sceneworks-core/src/training.rs` and
`apps/worker/scene_worker/training_adapters.py`.

## Prerequisites

- **A CUDA GPU worker.** Real training requires a GPU worker that advertises the
  `lora_train_execute` capability — i.e. a worker with CUDA-enabled PyTorch. A
  CPU-only worker can still validate a plan (dry run) but cannot train.
- **The base model installed.** Install **Z-Image-Turbo**
  (`Tongyi-MAI/Z-Image-Turbo`) from the Model Manager before a real run. A dry
  run does not need it. If it is missing, a real run is rejected at submit with
  *"Base model 'z_image_turbo' is not installed."*
- **Free disk space.** Keep at least ~2 GiB free on the volume that holds the
  output (checkpoints + the final adapter). The base model download is larger and
  is handled separately by the Model Manager.

## 1. Build a dataset

Create a dataset from images in your project, either in the **Train** screen or
via the API:

```http
POST /api/v1/projects/{projectId}/training/datasets
{
  "name": "Aurora style",
  "items": [
    { "assetId": "asset_…", "caption": { "text": "auroraStyle portrait", "triggerWords": ["auroraStyle"] } }
  ]
}
```

Recommended for a first run:

- **Size.** 10–30 images for a character or style is plenty; more is not always
  better. Start small and iterate.
- **Quality over quantity.** Sharp, varied, well-lit images. Avoid duplicates and
  heavy compression artifacts.
- **Captions.** One short caption per image describing the subject, and a single
  **trigger word** you will later type to invoke the LoRA (e.g. `auroraStyle`).
  Put the trigger word in the caption and in the training config's `triggerWord`.
  - **Identity / person LoRAs:** prefer a short, **unique non-word token** as the
    trigger (e.g. `kls_woman`, not a common name) and keep captions minimal —
    the trigger plus only what varies between shots (pose, outfit, setting). Long,
    fully-descriptive captions bind the person's features (hair colour, face) to
    ordinary words instead of the trigger, so a different inference prompt yields a
    different person.
  - **Style LoRAs:** detailed descriptive captions are appropriate — you *want* the
    style attached broadly rather than to one token.
- Datasets are versioned; provenance pins the exact version a run trained on, so a
  retrain is reproducible.

The Rust store owns dataset layout under `<project>/training/datasets/<id>/` and
exports caption sidecars; you never hand-edit trainer files.

## 2. Choose a training preset

Training configuration starts from a named built-in preset. For
`z_image_turbo_lora`, the default is **Character balanced**, which sets rank,
alpha, learning rate, optimizer, steps, resolution, checkpoint cadence, and
sample cadence from the Rust preset registry.

The preset registry is available at:

```http
GET /api/v1/training/presets
```

Initial Z-Image-Turbo presets include:

- `Character balanced` (`adamw8bit`, rank 16, 3000 steps, 1024px)
- `Character conservative` (`adamw8bit`, rank 8, learning rate `0.00005`)
- `Character balanced (AdamW)` (`adamw`)
- `Prodigy character (experimental)` (`prodigyopt`, learning rate `1.0`)
- `Style balanced` (rank 32, 3000 steps)
- `Low VRAM character` (rank 8, 2000 steps, 768px)

In the **Train** screen, changing the training target applies that target's
default preset. Changing the optimizer while the draft is still pristine applies
the matching optimizer preset. Manual edits are shown as customizations and are
preserved during normal refreshes. Turbo presets use sigmoid/high-noise
timestep sampling, MSE loss, weight decay `0.0001`, gradient checkpointing,
and 8-step CFG-0 sample previews. They also select the
`ostris/zimage_turbo_training_adapter` **de-distill adapter**
(`trainingAdapterRepo`/`trainingAdapterVersion`), which the worker fuses into the
base transformer for training and excludes from the saved LoRA — see §4. The
adapter version is a **De-distill adapter** dropdown in the Advanced panel
(`v1` — stable, smaller; `v2` — experimental, heavier de-distill), so you can A/B
the two on the same dataset.

## Network type — LoRA vs LoKr

Most targets train a standard **LoRA** (a low-rank `B·A` update). The
**Z-Image-Turbo** and **SDXL** targets also offer **LoKr** (LyCORIS
Kronecker-product), selectable as a **Network type** dropdown in the Advanced
panel on targets that advertise it (`limits.networkTypes`). LoKr parameterizes
the weight delta as a Kronecker product (`W₁ ⊗ W₂`, with one factor optionally
low-rank), which produces a **much smaller adapter** — e.g. a rank-16 SDXL LoKr
is ~2.7 MB vs tens of MB for the LoRA equivalent — at comparable or better
fidelity. A single **LoKr factor** (`decomposeFactor`, `-1` = auto) controls the
block split; leave it on auto to start.

Because LoKr changes the saved file, it also changes how inference loads it:

- diffusers' `load_lora_weights` understands only LoRA keys, so a LoKr adapter
  (`lokr_w1`/`lokr_w2`) is applied by rebuilding its `peft.LoKrConfig` from the
  file's safetensors metadata and **injecting it** into the UNet/transformer
  (`peft.inject_adapter_in_model`). The worker does this automatically — the
  trainer stamps `networkType` / `rank` / `alpha` / `decomposeFactor` /
  `targetModules` into the file header so generation can reconstruct the network.
- **Torch backends only.** The MLX image backends apply LoRA via a Kronecker-free
  merge, so they **reject** a LoKr adapter with a clear error rather than
  mis-applying it — train LoKr only where you will generate on torch.

Everything else — datasets, presets, dry run, output registration — is identical
to LoRA. `networkType` / `decomposeFactor` live in the config's free-form
`advanced` bag (gated per target by `limits.networkTypes`), not as first-class
config fields.

## 3. Dry run (validate the plan)

A dry run resolves and validates the full plan — dataset items exist, config is
in range, paths are absolute — **without** loading the model or producing
weights. It is the fast way to catch problems before spending GPU time.

```http
POST /api/v1/projects/{projectId}/training/jobs
{
  "targetId": "z_image_turbo_lora",
  "presetId": "z_image_turbo_lora.character.adamw8bit.balanced",
  "presetVersion": 1,
  "datasetId": "ds_…",
  "config": { …target defaults… },
  "outputName": "Aurora Style",
  "dryRun": true
}
```

In the **Train** screen this is the *"Validate (dry run)"* action (the default).
The job completes with a summary of what a real run would produce.

The dry-run plan records `presetId`, `presetVersion`, `presetName`, the preset
config snapshot, and the effective config snapshot. Existing direct submissions
without a preset still work; when you provide a preset id, the API validates that
the preset exists, matches the target, and matches the pinned version.

## 4. Train for real

Submit the same request with `"dryRun": false` (the *"Run training (beta)"*
toggle in the Train screen). The job routes to a GPU worker's `z_image_lora`
kernel, which loads the Z-Image-Turbo pipeline, **fuses the configured de-distill
training adapter into the transformer**, attaches a PEFT LoRA to the
(now de-distilled) transformer, caches latents and prompt embeddings, runs the
flow-matching loop with the configured timestep sampler/loss settings, checkpoints
every `saveEvery` steps, and writes a `.safetensors` adapter that contains only the
trained LoRA (the de-distill adapter is not saved).

**Two "schedulers" — keep them distinct.** The Advanced panel exposes two
independent settings that the word *scheduler* is used for elsewhere:

- The **noise scheduler** (`timestepType` / `timestepBias`) controls *which*
  flow-matching timesteps the loop trains on — the `flowmatch`-style sigmoid
  sampler. For Z-Image this denoising methodology is fixed; the knobs only bias
  where in the noise range sampling concentrates.
- The **learning-rate scheduler** (`lrScheduler`) controls how the *optimizer
  learning rate* changes over the run. `constant` (the default) holds the LR fixed
  for the whole run; `linear` and `cosine` decay it toward zero, optionally after
  an `lrWarmupSteps` linear warmup. It is a worker-backed control — selecting a
  non-constant value genuinely changes the LR during real training, and an
  unsupported name is rejected at submit.

Progress reports flow through the job stream (`preparing → caching → training →
checkpointing → saving → completed`) and the run honors cancellation between
steps.

### VRAM and time

Training is more memory-intensive than generation. The validated baseline is a
96 GB card. If you hit out-of-memory errors, lower the resolution (768 or 512),
keep batch size at 1, and reduce rank. Fewer steps train faster but may
under-fit; use earlier checkpoints if a 3000-step balanced run starts overfitting.

> **Note:** Z-Image-Turbo is a step-distilled (few-step) model. Training a LoRA
> *directly* on the distilled weights makes the distillation break down
> unpredictably — the LoRA learns (loss falls, weights move) but converges to
> generic, off-identity output. The worker therefore fuses the
> `ostris/zimage_turbo_training_adapter` de-distill adapter into the base before
> training and ships only your LoRA, which inference applies to the plain distilled
> model. Watch the worker `training_lora_weight_norm` event: a growing `loraBNorm`
> across checkpoints confirms the adapter is learning.

> **Apple Silicon — LTX-2.3 video LoRA:** the `ltx_mlx_lora` kernel
> (`target.kernel`, gated to Apple Silicon) trains an LTX-2.3 *video* LoRA
> natively in MLX from the same still-image dataset, registered under the
> `ltx-video` family. Validated recipe: rank 32 / alpha 32 / lr 1e-4 / weight
> decay 0.01 / res 512 with short trigger-focused captions
> (`"a photo of <trigger>"`), ~1500 steps
> (~1.35 s/step). The gemma text encoder (~28 GB) is freed after caption
> caching, so the training loop peaks ~27 GB; the whole-run ceiling is the
> dataset-caching phase at ~42 GB (text encoder still resident), which fits a
> **48 GB Mac** (64 GB+ comfortable). Generation peaks ~34 GB. Set `sampleEvery`
> to render a preview clip from the in-progress adapter at intervals — the loss
> alone won't tell you if it's working — but previews reload the full inference
> stack (peak rises to ~46 GB; a 64 GB+ Mac is recommended with them on), and the
> distilled sampler ignores `sampleSteps`/`sampleGuidanceScale`.

> **Lens LoRA (base Lens → Lens-Turbo):** the `lens_lora` kernel
> (`target.kernel`) trains an image LoRA for Microsoft Lens. Lens-Turbo is a
> 4-step *distillation* of base Lens; training a LoRA directly on the
> distilled velocity field drifts (the same failure mode the Z-Image de-distill
> adapter exists to avoid). So — unlike Z-Image — Lens needs **no de-distill
> adapter**: it trains on the non-distilled base (20-step, CFG 5.0,
> the direct weight-parent of Turbo) and the resulting `lens`-family LoRA applies
> cleanly to **Lens-Turbo** at inference. The training base is the
> **`SceneWorks/Lens`** diffusers rehost (sc-8797; Microsoft pulled the original
> `microsoft/Lens*` repos). On Windows/Linux the **Lens (base)** install already
> fetches it; on macOS install the Lens **training** variant from the Model
> Manager before a real run. Another source is selectable via
> `advanced.baseModelRepo`.
>
> Like the Lens inference path, training runs in the isolated **Lens sidecar venv**
> (`/opt/lens-venv`: transformers 5.x + diffusers 0.38), not the main worker venv,
> via `scene_worker/lens_train_runner.py`. The gpt-oss-20b text encoder + Flux.2
> VAE encode the dataset once (latents + the 4 selected-layer text features are
> cached), then the flow-matching loop trains a PEFT LoRA on the transformer's
> **fused-QKV** projections (`img_qkv`/`txt_qkv`/`to_out`/`to_add_out` — Z-Image's
> `to_q`/`to_k`/`to_v` would match nothing here). The 96 GB card is the validated
> baseline; the text encoder (~16 GB mxfp4) stays resident, so lower-VRAM runs
> should cut resolution (768) and rank first.
>
> **In-training previews render on the base Lens** (the loaded model), so the
> defaults use the base's `sampleSteps: 20` / `sampleGuidanceScale: 5.0` — they are
> a proxy for the concept on the base, **not** a Lens-Turbo (4-step) preview. They
> add real time, so the default `sampleEvery` is wider (500); raise it or disable
> previews to train faster.

### Lens parameter defaults

These are **principled starting points**, not empirically validated optima — Microsoft
publishes no Lens LoRA-training guidance, so they are derived from the tuned
Z-Image recipe plus general flow-matching DiT practice. Treat the right-hand
column as the first knobs to sweep in a real validation pass.

| Parameter | Default | Rationale / sweep note |
| --- | --- | --- |
| `rank` / `alpha` | 16 / 16 | Standard for style/character LoRAs (scale 1.0). Raise rank to 32 for complex styles; lower to 8 for tight identity or low VRAM. |
| `learningRate` | `1e-4` | Typical for rank-16 flow-matching LoRAs on a non-distilled base. Lower to `5e-5` if identity drifts; Prodigy (`lr 1.0`) auto-tunes. |
| `steps` | 3000 | Good for 10–30 images. Use earlier checkpoints if it overfits. |
| `resolution` | 1024 | Lens snaps to 1024/1440 buckets; 768 for lower VRAM. |
| `optimizer` | `adamw8bit` | bitsandbytes 8-bit AdamW; falls back to AdamW if unavailable. |
| `timestepType` / `timestepBias` | `sigmoid` / `high_noise` | **Deliberate, and a sweep candidate.** `high_noise` biases training toward the high-noise sigmas where the 4-step *Turbo* deployment target actually evaluates the adapter. If outputs come out over-baked, try `balanced`. |
| `lossType` | `mse` | Standard flow-matching objective. |
| `lrScheduler` | `constant` | Holds LR fixed; `linear`/`cosine` (+ optional `lrWarmupSteps`) also honored. |

The flow-matching target is **`noise - latents`** (note the sign — Lens feeds the
transformer output to the scheduler without negation, the opposite of Z-Image),
and the transformer timestep is the noise fraction directly. These are fixed in
the kernel, not user knobs.

AI Toolkit features such as EMA and Differential Output Preservation are not
exposed as SceneWorks presets yet because the current worker does not implement
their extra training passes.

> **Stable Diffusion XL LoRA (`sdxl_lora`):** the `sdxl_lora` kernel
> (`target.kernel`) trains an image LoRA for Stable Diffusion XL base 1.0, and is
> the **generic SDXL-UNet trainer** — the shared foundation epic 1929 (Kolors
> LoRA training) extends by swapping the pipeline class + text encoder. It is the
> first **U-Net (non-DiT)** trainer in the repo: unlike the flow-matching
> transformer kernels (Z-Image / Lens / LTX), it runs the SDXL **ε/v-prediction**
> objective on a **DDPM** noise schedule (integer timesteps) with the SDXL
> `added_cond_kwargs` (pooled CLIP text embeds + `add_time_ids`). SDXL base is
> **not** step-distilled, so — unlike Z-Image — it needs **no de-distill adapter**;
> and it uses real classifier-free guidance, so in-training previews render at a
> positive `sampleGuidanceScale` (7.0). The LoRA injects into the UNet attention
> projections (`to_q`/`to_k`/`to_v`/`to_out.0`) and is saved as an `sdxl`-family
> diffusers safetensors the SDXL adapter loads at generation. Install **Stable
> Diffusion XL** (`stabilityai/stable-diffusion-xl-base-1.0`) from the Model
> Manager before a real run; defaults are rank 16 / alpha 16 / lr 1e-4 / ~1500
> steps / 1024px, with character / style / low-VRAM presets. CreativeML
> OpenRAIL++-M (commercial-OK, ungated). Watch the worker
> `training_lora_weight_norm` event — a growing `loraBNorm` across checkpoints
> confirms the adapter is learning.

> **Wan2.2 video LoRA (`wan_lora`, `wan_moe_lora`):** train a *video* LoRA for
> the Wan2.2 family — applied at generation in **Video Studio** under the
> `wan-video` family. Unlike the MLX-only LTX kernel, these are torch/diffusers
> kernels that run on **CUDA *and* Apple-Silicon MPS**. Like the LTX video LoRA
> they train from a **still-image dataset** (each item encodes to a single
> Wan-VAE latent frame, `numFrames: 1`; the 5D latent shape is kept so a future
> clip dataset can pass `numFrames > 1`). The loop is flow-matching velocity on
> the `WanTransformer3DModel` attention projections
> (`to_q`/`to_k`/`to_v`/`to_out.0`); the target is **`noise - latents`** (Wan
> feeds the transformer output to the scheduler without negation — the opposite
> of Z-Image). Wan is **not** step-distilled, so there is **no de-distill
> adapter**. Defaults: rank 32 / alpha 32 / lr 1e-4 / 1500 steps / 512px / plain
> `adamw` (cross-platform; `adamw8bit` is CUDA-only and falls back), balanced
> sigmoid timestep sampling, MSE loss; in-training previews are **off**
> (`sampleEvery: 0`) because per-step Wan video gen is too expensive for the
> first cut. **Wan2.2 weights are Apache-2.0 (commercial-OK, ungated).**
>
> **Apple Silicon needs fp32.** Wan's Conv3d patch embedding has no bf16 Metal
> kernel, so the kernel forces fp32 on MPS (CUDA training stays bf16). This
> inflates end-to-end memory well beyond the quantized weights.
>
> Two size targets, three base models:
>
> - **`wan_lora` — Wan2.2-TI2V-5B** (`wan_2_2`,
>   `Wan-AI/Wan2.2-TI2V-5B-Diffusers`): the dense 5B. A single transformer →
>   one LoRA file. MPS-feasible (~32 GB peak in the spike).
> - **`wan_moe_lora` — A14B MoE** (`wan_t2v_14b_lora` →
>   `Wan-AI/Wan2.2-T2V-A14B-Diffusers`; `wan_i2v_14b_lora` →
>   `Wan-AI/Wan2.2-I2V-A14B-Diffusers`): A14B is a **two-expert mixture** — a
>   high-noise expert (`transformer`) and a low-noise expert (`transformer_2`),
>   split at the pipeline `boundary_ratio` (0.875). The kernel trains a
>   **separate LoRA on each expert** (alternating per step, each sampling
>   timesteps only within its band) and saves **two files**,
>   `<name>.high_noise.safetensors` + `<name>.low_noise.safetensors`. The
>   inference loader applies high→`transformer`, low→`transformer_2`.
>
> The bf16 A14B base (~56 GB of transformers) is GPU-only. To train it on a
> memory-bound host (e.g. a 128 GB Mac), point the base at a **Q8_0 GGUF**
> quantized expert pair via `config.advanced.baseQuantization`:
>
> ```json
> { "baseQuantization": { "format": "gguf",
>     "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
>     "highNoiseFile": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
>     "lowNoiseFile":  "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf" } }
> ```
>
> A LoRA trains fine on a GGUF-quantized base — PEFT attaches to the dequantizing
> `GGUFLinear` layers, gradients flow, and the saved adapter applies to the full
> base at inference. (`gguf>=0.10.0` must be installed in the worker venv.)
>
> **5B and 14B are different architectures** despite the shared `wan-video`
> family — the 5B uses the Wan2.2-VAE (48 latent channels), the A14B uses the
> Wan2.1-VAE (16 channels) — so a LoRA is **not** interchangeable between them.
> The trained adapter records its `baseModel`, and the inference loader gates by
> exact base-model match (family alone is insufficient). Watch the worker
> `training_lora_weight_norm` event — a growing `loraBNorm` across checkpoints
> confirms the adapter is learning (for MoE, both experts report).

## 5. Where the output goes

On a successful real run the adapter is registered as a normal SceneWorks LoRA:

- **Project scope** (default): weights at `<project>/loras/<loraId>/<name>.safetensors`,
  registered in `<project>/loras/manifest.jsonc`.
- **Global scope** (`config.advanced.outputScope = "global"`): weights under
  `data/loras/<loraId>/`, registered in `config/manifests/user.loras.jsonc`.

The entry carries provenance back to the dataset (id + version), training target,
selected preset metadata, job id, base model, and a config snapshot. It appears in `GET /api/v1/loras`
(add `?projectId=…` for project-scoped) and is selectable in **Image Studio**
under the `z-image` family. Type your trigger word in the prompt to invoke it.

The job result records the outcome: `loraRegistered: true` with `loraId` and
`loraManifestPath` on success, or `loraRegistered: false` with
`loraRegistrationError` if registration could not complete. Failed, canceled, or
weight-less runs never leave a broken catalog entry.

## Troubleshooting

SceneWorks fails fast at submit and maps common runtime errors to actionable
messages:

| Symptom | Cause | Fix |
| --- | --- | --- |
| *"Base model 'z_image_turbo' is not installed."* (submit) | Z-Image-Turbo not downloaded | Install it from the Model Manager, then retry the real run. Dry runs work without it. |
| *"Base model 'lens' is not installed."* (submit) | `SceneWorks/Lens` (the Lens training base) not downloaded | Install **Lens (base)** from the Model Manager (on macOS, its **training** variant — the MLX inference tiers alone don't include the flat training base). Lens trains on the non-distilled base and applies the LoRA to Lens-Turbo. |
| *"LoRA adapter attached no trainable parameters…"* (runtime) | `loraTargetModules` matched nothing | For Lens use the fused-QKV names (`img_qkv`, `txt_qkv`, `to_out`, `to_add_out`), not Z-Image's `to_q`/`to_k`/`to_v`. |
| *"Lens LoRA training requires the isolated Lens sidecar venv…"* (submit) | Worker built without the Lens sidecar | Rebuild with `INCLUDE_LENS=1`, or set `SCENEWORKS_LENS_PYTHON`. |
| *"… cannot target CPU workers."* (submit) | `requestedGpu` was `cpu` | Training is GPU-only. Use `auto` or a GPU id. |
| *"… cannot apply the LoKr adapter …"* (runtime) | A LoKr adapter was sent to the MLX generation backend | Generate the LoKr LoRA on a torch backend; MLX supports only standard LoRA (see *Network type*). |
| *"Not enough free disk space to train…"* (submit) | Output volume low on space | Free space (model weights, checkpoints, cached latents are the largest consumers). |
| Job stays **queued** | No GPU worker advertises `lora_train_execute` | Start a CUDA-enabled worker. A torch-less worker can claim dry runs only. |
| *"… GPU ran out of memory."* (runtime) | VRAM exhausted | Lower resolution, batch size, or rank. |
| *"… missing CUDA-enabled PyTorch."* (runtime) | Worker has CPU-only torch | Rebuild the worker with a CUDA wheel and restart. |
| *"… required model files were not available."* (runtime) | Model files missing/incomplete | Re-install the base model; ensure the utility worker is running; set `HF_TOKEN` for gated repos. |
| *"… disk ran out of space."* (runtime) | Volume filled mid-run | Free space and retry. |

See `apps/worker/README.md` for the kernel's runtime dependencies and a local
smoke check (`python -m scene_worker --check`).
