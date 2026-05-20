# Epic: Native LTX-2.3 Video Adapter

Shortcut epic: `sc-1394`

## Goal

Implement first-class local LTX-2.3 video generation in SceneWorks without depending on ComfyUI or hosted LTX APIs.

This epic replaces the current Diffusers-only LTX path for LTX-2.3 with an adapter backed by the official Lightricks `ltx-pipelines` stack. The adapter must support text-to-video first, then image-to-video, and must save generated clips as normal SceneWorks video assets with recipe, lineage, preview, and Library integration.

## Problem Statement

SceneWorks currently routes `ltx_2_3` through `DiffusionPipeline.from_pretrained(...)`. That works only for Hugging Face repositories that publish a Diffusers component layout with `model_index.json`.

`Lightricks/LTX-2.3` currently publishes raw LTX-2.3 checkpoint files, upscalers, and LoRAs, but not a root Diffusers `model_index.json`. As a result, SceneWorks text-to-video attempts fail with a Hugging Face 404 for:

```text
https://huggingface.co/Lightricks/LTX-2.3/resolve/main/model_index.json
```

LTX-2.3 itself does support text-to-video and image-to-video. The missing piece is a native SceneWorks adapter that loads the same model family through the official Lightricks pipeline stack instead of generic Diffusers loading.

## Non-Goals

- No ComfyUI dependency.
- No hosted LTX API dependency.
- No network rendering.
- No silent fallback from LTX-2.3 to LTX-2.
- No attempt to build a general ComfyUI workflow runner.
- No broad redesign of SceneWorks jobs, assets, or Library contracts.

## User Outcomes

- A user can select LTX-2.3 and generate a short text-to-video clip locally.
- A user can select an image asset and generate a short image-to-video clip locally.
- A generated clip appears in the Library as a video asset.
- The asset recipe records the LTX-2.3 checkpoint, upscaler, LoRA, prompt, seed, resolution, frame count, FPS, duration, and adapter settings.
- Queue progress reports useful stages instead of generic Python loader errors.
- If required model files are missing, the job fails before inference with a clear list of missing files and where SceneWorks expected to find them.

## Architecture

Add a new worker adapter, tentatively `LtxPipelinesVideoAdapter`, separate from `DiffusersVideoAdapter`.

The adapter should satisfy the existing `VideoGenerationAdapter` interface in `apps/worker/scene_worker/video_adapters.py`:

- `prepare`
- `ensure_models`
- `estimate_requirements`
- `run`
- `cancel`
- `cleanup`

Adapter selection should be explicit:

- `SCENEWORKS_VIDEO_ADAPTER=ltx_pipelines` uses the native LTX-2.3 stack.
- `SCENEWORKS_VIDEO_ADAPTER=diffusers_video` remains available for Wan and any Diffusers-compatible video models.
- `SCENEWORKS_VIDEO_ADAPTER=procedural_video` remains available for tests and lightweight development.

Longer term, model-level adapter routing can replace the environment toggle, but the first implementation should keep the change small and predictable.

## Runtime Strategy

Use the official Lightricks `LTX-2` repository packages:

- `ltx-core`
- `ltx-pipelines`

The recommended production-quality pipeline is `ltx_pipelines.ti2vid_two_stages`. The faster path is `ltx_pipelines.distilled`.

The first SceneWorks implementation should support two quality modes:

- `fast`: distilled pipeline, 8-step-style settings where supported.
- `balanced`: two-stage pipeline with distilled LoRA and spatial upscaler.

`best` can initially map to `balanced` with a higher step budget, then become a separate HQ pipeline once validated.

## Model Files

SceneWorks must manage these as model resources, not hidden one-off paths:

- LTX-2.3 model checkpoint:
  - `ltx-2.3-22b-dev.safetensors`
  - or `ltx-2.3-22b-distilled-1.1.safetensors`
- Spatial upscaler:
  - `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
  - or a validated lower-memory upscaler variant.
- Distilled LoRA for two-stage pipelines:
  - `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`
- IC-LoRA for IC-conditioned pipelines:
  - one or more installed LTX-compatible IC-LoRA `.safetensors` files selected through a preset.
- Gemma text encoder assets:
  - full local Gemma 3-12B repo or local text encoder path required by `ltx-pipelines`.

The adapter should not hardcode absolute user paths. Add advanced settings and manifest fields for:

- `checkpointPath`
- `spatialUpscalerPath`
- `distilledLoraPath`
- `gemmaRoot`
- optional `temporalUpscalerPath`

## Manifest Changes

Extend `config/manifests/builtin.models.jsonc` for `ltx_2_3` with resource entries for the native pipeline files.

Recommended shape:

```jsonc
{
  "id": "ltx_2_3",
  "adapter": "ltx_pipelines",
  "resources": {
    "checkpoint": {
      "repo": "Lightricks/LTX-2.3",
      "file": "ltx-2.3-22b-distilled-1.1.safetensors"
    },
    "spatialUpscaler": {
      "repo": "Lightricks/LTX-2.3",
      "file": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    },
    "distilledLora": {
      "repo": "Lightricks/LTX-2.3",
      "file": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    },
    "gemma": {
      "repo": "google/gemma-3-12b-it-qat-q4_0-unquantized"
    }
  }
}
```

The official `ltx-pipelines` stack expects the Gemma 3-12B text encoder family; using Gemma 3-4B causes tensor shape mismatches during text encoder load. Image-conditioned modes should route through `ltx_pipelines.ic_lora.ICLoraPipeline` when an IC-LoRA preset is selected so references can use IC-LoRA attention conditioning for better identity consistency, and otherwise fall back to the standard distilled/two-stage LTX pipelines. Video-conditioned modes should require IC-LoRA because their video conditioning path depends on the IC-LoRA pipeline.

## Worker Dependency Plan

Avoid bloating the default worker until the adapter is enabled.

Preferred first approach:

- Add optional install target or Docker build arg for LTX native dependencies.
- Keep base image compatible with existing Diffusers image/video jobs.
- Document that native LTX-2.3 requires the LTX-enabled worker image.

Possible implementation options:

- Add `apps/worker/requirements-ltx.txt`.
- Add `docker/worker-ltx.Dockerfile` or `ARG INCLUDE_LTX_PIPELINES=1`.
- Install LTX packages from a pinned Git commit of `https://github.com/Lightricks/LTX-2`.
- Record the pinned commit in this repository.

Do not install from an unpinned branch for production use.

## Adapter Implementation Tasks

1. Add `LtxPipelinesVideoAdapter`.
2. Add adapter selection for `SCENEWORKS_VIDEO_ADAPTER=ltx_pipelines`.
3. Resolve model resource paths from manifest/download markers or advanced overrides.
4. Validate required files in `ensure_models`.
5. Map SceneWorks request fields:
   - `prompt`
   - `negativePrompt`
   - `seed`
   - `duration`
   - `fps`
   - `width`
   - `height`
   - `quality`
   - `sourceAssetId` for image-to-video
6. Normalize dimensions to LTX-compatible values.
7. Normalize frame counts to supported pipeline constraints.
8. Execute the selected `ltx-pipelines` pipeline.
9. Stream coarse progress:
   - preparing
   - validating model files
   - loading checkpoint
   - encoding prompt
   - stage 1 generation
   - stage 2 upscaling/refinement
   - saving
10. Save output to `assets/videos`.
11. Write generation set, asset sidecar, recipe, and project DB index.
12. Clean up temporary files on cancellation or failure.
13. Keep loaded model reporting meaningful enough for worker heartbeat.

## UI And Product Behavior

The Video Studio can keep the current simple controls:

- Mode: Text to Video / Image to Video.
- Prompt.
- Duration.
- Aspect/resolution.
- Quality.
- Seed.
- Advanced model settings.

Add LTX-native advanced fields only if needed:

- checkpoint variant.
- text encoder mode/local path.
- distilled LoRA weight.
- upscaler variant.
- prompt enhancement toggle.

Default UI should remain small. The adapter should pick conservative defaults from manifest metadata.

## Recommended Defaults

Initial defaults should favor successful local runs over maximum quality:

- Duration: 4 seconds.
- FPS: 25.
- Resolution: 768x512 landscape and 512x768 portrait.
- Quality: balanced.
- Seed: generated when absent.
- Output: MP4.

If memory pressure is high, first fallback should be shorter duration or lower resolution, not switching models.

## Testing Plan

Unit tests:

- Adapter selection returns `LtxPipelinesVideoAdapter` for `SCENEWORKS_VIDEO_ADAPTER=ltx_pipelines`.
- Required model path validation reports all missing files.
- Request mapping preserves prompt, seed, duration, FPS, dimensions, quality, and source image.
- Frame count normalization is deterministic.
- Output sidecar preserves native LTX lineage.
- Cancellation deletes temp outputs.

Integration tests with mocked `ltx_pipelines`:

- Text-to-video job completes and writes a video asset.
- Image-to-video job loads source asset and passes image conditioning.
- Missing Gemma root fails with clear user-facing message.
- Missing spatial upscaler fails before inference.

Manual validation:

- Run one 4-second T2V generation on the target NVIDIA machine.
- Run one 4-second I2V generation from a Library image.
- Confirm output previews in Library.
- Confirm recipe can be reused.
- Confirm GPU memory returns after job completion/cancel.

## Acceptance Criteria

- LTX-2.3 text-to-video works locally without ComfyUI and without hosted API calls.
- LTX-2.3 image-to-video works locally without ComfyUI and without hosted API calls.
- SceneWorks fails fast with actionable messages when any required native LTX model file is missing.
- Generated outputs are normal SceneWorks video assets.
- Recipes preserve enough settings to reproduce the generation.
- Existing Diffusers video paths and procedural video tests still pass.
- The worker test suite covers adapter selection, validation, request mapping, and asset writing.

## Risks And Open Questions

- `ltx-pipelines` dependency weight may make the worker image much larger.
- The LTX package may require Python/CUDA/PyTorch versions that diverge from the current worker.
- Gemma local text encoder requirements need exact validation.
- Two-stage pipeline defaults may require more VRAM than expected.
- Audio generation support should be deferred unless it comes for free without destabilizing video-only output.
- The current Model Manager may not express multi-file model resources cleanly enough; this may require a small manifest/resource schema expansion.
- License prompts and gated model access must be handled clearly if any dependency requires Hugging Face auth or license acceptance.

## Implementation Slices

### Slice 1: Native Adapter Skeleton

Shortcut story: `sc-1395`

- Add optional LTX dependencies.
- Add `LtxPipelinesVideoAdapter`.
- Implement validation and mocked run path.
- No real inference yet.

### Slice 2: Model Resource Resolution

Shortcut story: `sc-1396`

- Extend manifest resource metadata.
- Resolve checkpoint/upscaler/LoRA/Gemma paths.
- Add missing-file diagnostics.

### Slice 3: Real Text-To-Video

Shortcut story: `sc-1397`

- Wire `ltx_pipelines` text-to-video.
- Save MP4 asset.
- Validate one short local generation.

### Slice 4: Real Image-To-Video

Shortcut story: `sc-1398`

- Load source image from SceneWorks asset.
- Pass image conditioning into the pipeline.
- Validate one short local generation.

### Slice 5: Polish And Hardening

Shortcut story: `sc-1399`

- Progress stages.
- Cancellation.
- GPU cleanup.
- Better model install UX.
- Documentation.

## Sources

- LTX-2.3 model page: https://huggingface.co/Lightricks/LTX-2.3
- LTX-2 repository: https://github.com/Lightricks/LTX-2
- `ltx-pipelines` README: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
- LTX open-source quick start: https://docs.ltx.video/open-source-model/getting-started/quick-start
- LTX-2.3 product FAQ: https://ltx.io/model/ltx-2-3
