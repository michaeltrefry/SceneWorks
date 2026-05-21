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
- Datasets are versioned; provenance pins the exact version a run trained on, so a
  retrain is reproducible.

The Rust store owns dataset layout under `<project>/training/datasets/<id>/` and
exports caption sidecars; you never hand-edit trainer files.

## 2. Dry run (validate the plan)

A dry run resolves and validates the full plan — dataset items exist, config is
in range, paths are absolute — **without** loading the model or producing
weights. It is the fast way to catch problems before spending GPU time.

```http
POST /api/v1/projects/{projectId}/training/jobs
{
  "targetId": "z_image_turbo_lora",
  "datasetId": "ds_…",
  "config": { …target defaults… },
  "outputName": "Aurora Style",
  "dryRun": true
}
```

In the **Train** screen this is the *"Validate (dry run)"* action (the default).
The job completes with a summary of what a real run would produce.

Config defaults for `z_image_turbo_lora` (from the Rust registry): rank 16,
alpha 16, learning rate 1e-4, 2000 steps, batch size 1, resolution 1024, save
every 250 steps, `adamw8bit` optimizer. Advanced fields stay collapsed by
default; `outputScope` defaults to `project`.

## 3. Train for real

Submit the same request with `"dryRun": false` (the *"Run training (beta)"*
toggle in the Train screen). The job routes to a GPU worker's `z_image_lora`
kernel, which loads the Z-Image-Turbo pipeline, attaches a PEFT LoRA to the
transformer, caches latents and prompt embeddings, runs the flow-matching loop,
checkpoints every `saveEvery` steps, and writes a `.safetensors` adapter.

Progress reports flow through the job stream (`preparing → caching → training →
checkpointing → saving → completed`) and the run honors cancellation between
steps.

### VRAM and time

Training is more memory-intensive than generation. The validated baseline is a
96 GB card. If you hit out-of-memory errors, lower the resolution (768 or 512),
keep batch size at 1, and reduce rank. Fewer steps train faster but may
under-fit; 1500–2500 steps is a reasonable first range.

> **Note:** Z-Image-Turbo is a distilled (few-step) model. A LoRA trained
> directly on it can work, but may need extra care for quality on some subjects.
> Treat the first result as a baseline and iterate on dataset and steps.

## 4. Where the output goes

On a successful real run the adapter is registered as a normal SceneWorks LoRA:

- **Project scope** (default): weights at `<project>/loras/<loraId>/<name>.safetensors`,
  registered in `<project>/loras/manifest.jsonc`.
- **Global scope** (`config.advanced.outputScope = "global"`): weights under
  `data/loras/<loraId>/`, registered in `config/manifests/user.loras.jsonc`.

The entry carries provenance back to the dataset (id + version), training target,
job id, base model, and a config snapshot. It appears in `GET /api/v1/loras`
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
