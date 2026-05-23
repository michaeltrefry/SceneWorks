# Epic: Training Presets And Model Defaults

Shortcut epic: `1546`

## Goal

Make LoRA training configuration model-aware and preset-driven so users start from known-good defaults instead of hand-tuning every field.

When a user selects a training model, SceneWorks should load recommended configuration values for that model. When a user selects an optimizer, SceneWorks should adjust optimizer-sensitive defaults such as learning rate, steps, and sample cadence. Advanced users can still override individual values, but the default path should feel opinionated and reliable.

## Problem Statement

Training configuration currently starts from a single target default block in the Rust training registry. That works for the first Z-Image-Turbo LoRA path, but it does not scale well as we add models, optimizers, quality presets, and hardware profiles.

Some settings are not globally optimal:

- Z-Image-Turbo may need different sample settings than other image models.
- `adamw8bit`, `adamw`, and `prodigyopt` can require different learning-rate assumptions.
- Character, style, and object LoRAs may want different rank, step, and caption/sample defaults.
- VRAM-constrained users may need a lower-resolution preset without having to understand every field.

The app needs a first-class preset layer that captures recommended combinations while keeping the resolved training plan explicit and reproducible.

## Non-Goals

- No automated hyperparameter search.
- No hosted recommendation service.
- No hidden mutation after a job is submitted.
- No guarantee that presets are universally optimal for every dataset.
- No removal of advanced manual controls.
- No migration to an external trainer config format as the source of truth.

## User Outcomes

- Selecting a training target applies a recommended default preset for that model.
- Selecting an optimizer updates optimizer-sensitive fields to the recommended values for that model/optimizer pair.
- Users can choose from named presets such as `Character balanced`, `Style balanced`, `Low VRAM`, or `Prodigy character`.
- The UI explains presets through short labels and concrete field values, not long prose.
- Overrides are visible: if the user edits a field, SceneWorks does not silently overwrite it unless they explicitly apply another preset.
- Dry-run and real training jobs record the selected preset id, preset version, and effective config snapshot.
- Training results record enough preset metadata to reproduce or compare runs later.

## Architecture

Add a training preset registry alongside the existing training target registry.

The Rust core remains the source of truth for built-in training targets and built-in preset metadata. The web UI consumes the registry and builds config drafts from presets. The API resolves the submitted config into the existing `TrainingPlan`; the worker still receives only concrete values.

Recommended shape:

```jsonc
{
  "schemaVersion": 1,
  "presets": [
    {
      "id": "z_image_turbo_lora.character.adamw8bit.balanced",
      "version": 1,
      "targetId": "z_image_turbo_lora",
      "name": "Character balanced",
      "recommendedFor": ["character"],
      "optimizer": "adamw8bit",
      "qualityPreset": "balanced",
      "config": {
        "rank": 16,
        "alpha": 16,
        "learningRate": 0.0001,
        "steps": 3000,
        "batchSize": 1,
        "gradientAccumulation": 1,
        "resolution": 1024,
        "saveEvery": 250,
        "seed": 42,
        "optimizer": "adamw8bit",
        "advanced": {
          "mixedPrecision": "bf16",
          "weightDecay": 0.0001,
          "lrScheduler": "constant",
          "timestepType": "sigmoid",
          "timestepBias": "high_noise",
          "lossType": "mse",
          "gradientCheckpointing": true,
          "cacheTextEmbeddings": true,
          "sampleEvery": 250,
          "sampleSteps": 8,
          "sampleGuidanceScale": 0.0,
          "outputScope": "project"
        }
      }
    }
  ]
}
```

Preset config should be a partial or complete `TrainingConfig`. The resolver should merge in this order:

1. Training target hard defaults.
2. Selected training preset config.
3. User-edited field overrides.
4. Required contextual values, such as `triggerWord`, `requestedGpu`, and output scope.

## UI Behavior

Training Studio should treat the selected model, preset, and optimizer as a connected control group:

- Training model select chooses the target.
- Preset select lists presets recommended for that target.
- Optimizer select can either filter presets or switch to the recommended preset variant for that optimizer.
- Applying a preset updates the form fields in one deliberate action.
- Editing any field marks the config as customized.
- A compact "Preset values" summary shows the important resolved values: rank, alpha, learning rate, steps, resolution, optimizer, sample cadence, sample guidance.
- Advanced controls stay available for manual edits.

Avoid surprising resets:

- Changing model should apply that model's default preset.
- Changing optimizer should offer the matching preset and apply it only when the user confirms or when the field is still pristine.
- If the user has custom edits, show that the preset no longer exactly matches.

## API And Contracts

Extend the training target registry response or add a sibling route:

- `GET /api/v1/training/targets`
- `GET /api/v1/training/presets`

The training job request should optionally include:

```json
{
  "presetId": "z_image_turbo_lora.character.adamw8bit.balanced",
  "presetVersion": 1
}
```

The resolved plan and job result should preserve:

- `presetId`
- `presetVersion`
- `presetName`
- `presetConfigSnapshot`
- effective `config`

If a preset is missing or incompatible with the selected target, the API should reject the request with a clear validation error.

## Initial Built-In Presets

Start with Z-Image-Turbo LoRA presets:

- `Character balanced / adamw8bit`
- `Character conservative / adamw8bit`
- `Character balanced / adamw`
- `Prodigy character (experimental) / prodigyopt`
- `Style balanced / adamw8bit`
- `Low VRAM character / adamw8bit`

The initial values can be conservative and should be easy to revise as we validate real runs. The key is establishing the registry and UI contract before adding more models.

## Implementation Tasks

1. Add Rust contract types for training preset registry entries.
2. Add built-in preset registry and snapshot fixture tests.
3. Add API route or extend the training targets route to expose presets.
4. Add preset fields to training job request, plan provenance, and job result summaries.
5. Update Training Studio state so model selection initializes from the default preset.
6. Add a preset selector in the training configuration sidebar.
7. Make optimizer selection choose or suggest matching presets.
8. Track customized fields so preset application does not unexpectedly erase user edits.
9. Include selected preset metadata in dry-run and real training submissions.
10. Update `TRAINING_QUICKSTART.md` with the new preset workflow.
11. Add tests for preset merge order, invalid preset rejection, UI defaulting, and optimizer-sensitive defaults.

## Acceptance Criteria

- A fresh Z-Image-Turbo training config loads from a named built-in preset.
- Changing to `prodigyopt` can apply a Prodigy-specific preset with a Prodigy-appropriate learning-rate configuration.
- Manual field edits survive normal UI refreshes and are visible as customizations.
- Dry-run output shows the selected preset and effective config.
- Real training result metadata includes the selected preset and effective config.
- Existing direct config submissions without a preset continue to work.
- The worker receives concrete config values only; it does not need to know preset resolution rules.

## Open Questions

- Should presets be global built-ins only for the first version, or should project/user presets land in the same epic?
- Should optimizer changes auto-apply matching presets only while the draft is pristine, or always ask?
- Do we want separate presets for character/style/object, or a single `recommendedFor` field with filter chips?
- Should sample prompts become part of presets, or remain generated from the trigger phrase?
- Should hardware profiles such as `24GB`, `48GB`, and `96GB` be explicit preset dimensions?
