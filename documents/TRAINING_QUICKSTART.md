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

AI Toolkit features such as EMA and Differential Output Preservation are not
exposed as SceneWorks presets yet because the current worker does not implement
their extra training passes.

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
| *"… cannot target CPU workers."* (submit) | `requestedGpu` was `cpu` | Training is GPU-only. Use `auto` or a GPU id. |
| *"Not enough free disk space to train…"* (submit) | Output volume low on space | Free space (model weights, checkpoints, cached latents are the largest consumers). |
| Job stays **queued** | No GPU worker advertises `lora_train_execute` | Start a CUDA-enabled worker. A torch-less worker can claim dry runs only. |
| *"… GPU ran out of memory."* (runtime) | VRAM exhausted | Lower resolution, batch size, or rank. |
| *"… missing CUDA-enabled PyTorch."* (runtime) | Worker has CPU-only torch | Rebuild the worker with a CUDA wheel and restart. |
| *"… required model files were not available."* (runtime) | Model files missing/incomplete | Re-install the base model; ensure the utility worker is running; set `HF_TOKEN` for gated repos. |
| *"… disk ran out of space."* (runtime) | Volume filled mid-run | Free space and retry. |

See `apps/worker/README.md` for the kernel's runtime dependencies and a local
smoke check (`python -m scene_worker --check`).
