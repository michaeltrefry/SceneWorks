# SceneWorks Implementation Plan

This document turns the SceneWorks product plan into a practical implementation sequence. It assumes the current goal is to build a local Docker-based AI image/video generation and editing app with a React + Vite frontend, FastAPI backend, Python workers, SQLite metadata, file-backed assets, and model adapters.

## Guiding Principles

- Build vertical slices, not isolated demo screens.
- Keep every generated/imported output as a reusable Asset.
- Treat generation, download, tracking, replacement, extraction, and export as Jobs.
- Keep model-specific details behind adapter boundaries.
- Keep project folders inspectable and portable.
- Optimize for short generated shots assembled into longer videos.
- Favor stable local workflows over early feature breadth.

## Phase 0: Repository And Runtime Skeleton

Goal: SceneWorks boots locally with frontend, backend, and shared config volumes.

Deliverables:

- Docker Compose layout.
- Frontend app using React + Vite.
- Backend app using FastAPI.
- Python worker package scaffold.
- Shared config and data folders.
- Health checks.
- Basic dev scripts.

Suggested structure:

```text
SceneWorks/
  apps/
    web/
    api/
    worker/
  packages/
    schemas/
    shared/
  config/
    manifests/
      builtin.models.jsonc
      builtin.loras.jsonc
      user.models.jsonc
      user.loras.jsonc
  data/
    projects/
    models/
    loras/
    cache/
  docker/
  docker-compose.yml
  SCENEWORKS_PLAN.md
  IMPLEMENTATION_PLAN.md
```

Acceptance criteria:

- `docker compose up` starts web and API services.
- Web app can call `/api/v1/health`.
- API can see configured data directories.
- Worker can start in placeholder mode.

## Phase 1: Project And Asset Spine

Goal: users can create/open projects, import media, and see assets in the Library.

Backend:

- Project creation/open APIs.
- Recent projects registry.
- Project folder creation.
- `project.json`.
- `project.db`.
- Asset import endpoint.
- Sidecar JSON writer.
- SQLite indexing.
- Trash support.

Frontend:

- Project list/create/open screen.
- Project shell with left nav.
- Library screen.
- Asset grid.
- Asset detail panel.
- Rating/favorite/reject/trash controls.
- Asset tray shell.

Storage:

```text
MyProject.sceneworks/
  project.json
  project.db
  assets/
    images/
    videos/
    uploads/
    frames/
    renders/
  characters/
  loras/
  recipes/
  timelines/
  trash/
  cache/
```

Acceptance criteria:

- User creates a project.
- User imports an image and video.
- Imported files are copied into the project.
- Assets appear in Library.
- Asset sidecars are written.
- Asset metadata survives app restart.
- Rating/favorite/reject/trash works.

## Phase 2: Schemas And Persistence Contracts

Goal: make project, asset, recipe, timeline, manifest, and job structures explicit before generation complexity arrives.

Deliverables:

- JSON schemas or typed models for:
  - Project.
  - Asset.
  - GenerationSet.
  - Recipe.
  - Job.
  - ModelManifest.
  - LoRAManifest.
  - Character.
  - Timeline.
  - TimelineItem.
- Schema version fields.
- App version fields in sidecars.
- Basic migration placeholder.
- Reindex command placeholder.

Acceptance criteria:

- Sidecars validate against schema.
- API responses use typed models.
- Invalid sidecar/manifest data produces clear errors.
- Reindex command can scan a project and report assets, even if full rebuild is later.

## Phase 3: Job Queue And Worker Foundation

Goal: long-running work is queued, visible, cancelable, retryable, and survives browser refresh.

Backend:

- SQLite-backed job table.
- Job create/list/detail APIs.
- Job status transitions.
- Job progress events via WebSocket or SSE.
- Cancel job endpoint.
- Retry job endpoint.
- Duplicate with changes endpoint.
- Startup handling for interrupted jobs.

Worker:

- Worker process loop.
- Job claim/heartbeat.
- Placeholder job execution.
- Progress updates.
- Cancellation checks.

Frontend:

- Global Queue screen.
- Project filter.
- Job status/progress display.
- Cancel/retry/duplicate controls.
- Queue status in top bar.

Acceptance criteria:

- User starts a placeholder job.
- Job appears in Queue.
- Progress updates live.
- Browser refresh does not lose job state.
- Cancel works.
- Retry works.
- Backend restart marks running jobs interrupted.

## Phase 4: GPU Awareness And Dispatch

Goal: support one worker per GPU and explicit GPU selection, even before real generation.

Backend/worker:

- GPU discovery abstraction.
- Worker registration with GPU ID.
- Exclusive GPU reservation per job.
- Requested GPU field.
- Assigned GPU field.
- Loaded-model hint field.

Frontend:

- GPU selector in advanced generation drawer placeholder.
- Active GPU display in top bar.
- GPU status display in Queue or Settings.

Acceptance criteria:

- Multiple workers can register as different GPUs.
- Job can request a GPU.
- Dispatcher assigns only one running GPU job per GPU.
- User override wins over auto selection.

## Phase 5: Manifests And Model Manager

Goal: model and LoRA configuration is text-configurable and visible in the app.

Backend:

- Load built-in and user model manifests.
- Load built-in and user LoRA manifests.
- Merge global and project overrides.
- Validate manifest entries.
- Resolve local paths and Hugging Face cache paths.
- Model Manager APIs.
- Model download job type.
- LoRA import job type placeholder.

Frontend:

- Model Manager screen.
- Installed/missing/downloadable states.
- Download size when known.
- Predownload button.
- LoRA list and compatibility display.

Acceptance criteria:

- App reads JSONC manifests.
- User model manifest can override paths for known adapters.
- Missing model can create a download job.
- Download jobs do not consume GPU slots.
- LoRA compatibility can be filtered by model family.

## Phase 6: First Image Generation Adapter

Goal: generate real images and save them as assets with recipes.

Research first:

- Confirm best first image model target from the supported v1 image list.
- Validate dependency footprint.
- Validate LoRA support path.
- Validate 24GB VRAM expectations.
- Identify Apple/MPS feasibility separately.
- Validate image editing support versus pure text-to-image support.
- Validate licensing and redistribution/download constraints.

Supported v1 image model targets:

- [HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- [Qwen Image](https://github.com/QwenLM/Qwen-Image)
- [LightX2V Qwen-Image-Lightning](https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning)
- [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

Implementation:

- Image generation adapter interface.
- First image adapter.
- Text-to-image job.
- Image-edit job if supported by chosen model.
- GenerationSet creation.
- Output asset writing.
- Recipe sidecars.
- Batch review UI.
- Fullscreen preview.
- Send to Video and Send to Editor buttons.

Acceptance criteria:

- User can generate a batch of images from text.
- All outputs become assets.
- Bad fresh outputs can be discarded.
- Ratings/favorites work from generation review.
- Recipe is stored and visible in asset detail.
- User can rerun/duplicate a generation job.

## Phase 7: Image Studio Simple And Advanced UX

Goal: make image generation feel like SceneWorks, not a raw model form.

Frontend:

- Modes:
  - Text to Image.
  - Edit Image.
  - Character Image.
  - Style Variations.
- Prompt field.
- Style preset selector.
- Character/LoRA selectors.
- Count selector.
- Advanced drawer:
  - Model.
  - Seed.
  - Negative prompt.
  - Resolution.
  - Adapter-supported settings.
- Last settings remembered per project.

Acceptance criteria:

- Simple mode can generate without exposing model internals.
- Advanced drawer can override model/seed/negative prompt.
- Settings persist per project.
- LoRA selection respects compatibility and simple-mode limits.

## Phase 8: First Video Generation Adapter

Goal: generate short video clips, especially image-to-video, and save them as reusable assets.

Research first:

- Validate LTX2.3 install path and current best inference repo.
- Validate I2V and T2V support.
- Validate best output lengths, with target guidance around 15 seconds or less.
- Validate FPS/resolution constraints.
- Validate LoRA support.
- Validate Wan2.2 path separately.

Implementation:

- Video generation adapter interface.
- LTX2.3 or chosen first video adapter.
- Image-to-video job.
- Text-to-video job if supported.
- Video asset writing.
- Recipe and lineage.
- Clip preview.
- Send clip to Editor.

Acceptance criteria:

- User can generate a short clip from an image asset.
- User can generate a short clip from text if supported.
- Output becomes a video asset.
- Recipe includes model, prompt, source image, duration, FPS, resolution, seed, LoRAs, and raw adapter settings.
- Clip can be previewed in the Library.

## Phase 9: Video Studio UX

Goal: expose video workflows through clean modes with short-shot assumptions.

Frontend:

- Modes:
  - Image to Video.
  - Text to Video.
  - First/Last Frame.
  - Extend Clip.
  - Replace Person placeholder.
- Prompt field.
- Input slots based on mode.
- Character/style/LoRA selectors.
- Duration selector with model-aware recommended limits.
- Aspect ratio/resolution.
- Quality preset: Fast, Balanced, Best.
- Advanced drawer:
  - Model.
  - Seed.
  - Negative prompt.
  - FPS if supported.
  - Adapter-specific controls.
  - GPU selection.

Acceptance criteria:

- User can switch between video modes easily.
- UI discourages unrealistic long generations.
- Generated clips appear in current review area and asset tray.

## Phase 10: Basic Editor And Export

Goal: assemble generated/imported clips into a short MP4.

Research first:

- Evaluate React timeline/editor libraries.
- Confirm timeline data model fit.
- Confirm FFmpeg export strategy.

Frontend:

- Timeline screen.
- Create multiple timelines per project.
- Project aspect ratio: 16:9, 9:16, 1:1.
- Add generated/imported clips.
- Add still images.
- Trim/arrange.
- Speed controls.
- Basic transitions:
  - cut
  - crossfade
  - fade from black
  - fade to black
- Frontend undo/redo for active session.

Backend:

- Timeline JSON save/load.
- SQLite timeline index.
- Timeline export job.
- FFmpeg renderer.
- Render output asset creation.

Acceptance criteria:

- User can assemble clips and stills.
- User can save/reopen timeline.
- User can export MP4.
- Export becomes a render asset.
- Final export is backend-rendered, not browser-recorded.

## Phase 11: Timeline Generation Hooks

Goal: make the editor AI-aware.

Features:

- Extract frame from clip at playhead.
- Save extracted frame as image asset with source timestamp.
- Send frame to Image Studio or Video Studio.
- Extend selected clip.
- Generate bridge from gap between clips.
- Replace selected timeline item nondestructively.
- Timeline item version history.

Acceptance criteria:

- User extracts a frame and reuses it.
- User extends a clip by generating a short continuation.
- User selects a gap and generates a bridge clip using left last frame and right first frame.
- Generated bridge lands in the gap.
- Prior timeline versions remain available.

## Phase 12: Character Studio

Goal: create reusable character identities before full LoRA training exists.

Backend:

- Character model.
- Character sidecars.
- Reference image import.
- Approved reference handling.
- Look model.
- Character-linked LoRAs.

Frontend:

- Character list.
- Character detail.
- Type: Person, Creature, Object.
- Reference images.
- Approved references.
- Looks.
- Imported LoRAs.
- Test Character workflow.
- Send Character to image/video generation.

Acceptance criteria:

- User creates a Character.
- User adds reference images.
- User marks approved references.
- User creates a Look.
- Character can be selected in generation forms.
- Test Character produces/rates sample outputs once image adapter supports it.

## Phase 13: Person Tracking Foundation

Goal: support selecting and tracking a person in a video clip.

Research first:

- Person detection options.
- Segmentation options.
- Video object tracking options.
- Mask storage size and performance.

Implementation:

- Representative frame extraction.
- Person detection job.
- Person selection UI.
- Person track data model.
- Tracking job.
- Store boxes/masks/confidence per sampled frame.
- Minimal correction UI if feasible.

Acceptance criteria:

- User selects a person in a source clip.
- App creates a reusable person track.
- Person track can be named.
- Track metadata is stored in the project.

## Phase 14: Replace Person V1

Goal: ship the core differentiator on selected short clips.

Research first:

- Face-only replacement pipeline.
- Full-person keep-outfit pipeline.
- Full-person replace-outfit pipeline.
- Identity/reference conditioning.
- Temporal consistency approach.
- Practical limits for clip length and resolution.

Product modes:

- Face Only.
- Full Person, Keep Outfit.
- Full Person, Replace Outfit.

Implementation:

- Replace Person job.
- Character input.
- Person track input.
- Mode selector.
- Quality preset: Fast, Balanced, Best.
- Replacement adapter.
- Output clip asset.
- Lineage to source clip, person track, Character, mode, model, and recipe.
- A/B toggle.
- Side-by-side comparison.

Acceptance criteria:

- User imports/selects a short clip.
- User selects one person.
- User selects a Character.
- User runs one replacement mode.
- App produces a new replacement clip.
- User can compare against original.
- Output is usable in timeline.

## Phase 15: Multi-GPU Polish

Goal: make two-GPU systems feel deliberate and predictable.

Features:

- Worker per GPU.
- Exclusive GPU job reservation.
- User-selected GPU.
- Auto GPU option.
- Loaded-model preference.
- Queue status per GPU.
- Clear blocked/waiting states.
- Better failure messages for OOM and missing models.

Acceptance criteria:

- Two GPU workers can run different jobs concurrently.
- User can choose where a job runs.
- App does not schedule two GPU jobs onto the same GPU at once.
- Queue UI clearly explains what is running and waiting.

## Phase 16: LoRA And Preset Polish

Goal: make styles, enhancements, and character LoRAs pleasant and constrained.

Features:

- Global LoRAs.
- Project LoRAs.
- LoRA import flow.
- Metadata inspection.
- Category editing.
- Compatibility editing.
- Trigger words.
- Default weight.
- Built-in recipe presets.
- Simple mode limit: built-ins plus up to 2 user LoRAs.
- Advanced override for unknown/incompatible LoRAs.

Acceptance criteria:

- User imports a LoRA globally or into a project.
- App filters compatible LoRAs by selected model.
- User can apply style/enhance/character LoRAs in generation.
- Built-in presets set LoRAs and defaults without exposing internals.

## Phase 17: Next Version Preparation

Goal: leave hooks for next-priority features without blocking v1.

Prepare for:

- Character LoRA training via `ai-toolkit`.
- Storyboard/shot planning view.
- Inpainting.
- Better audio support.
- Captions/speech-to-text.
- Multiple person replacement.
- Style application to clips.
- Apple runtime work.

Acceptance criteria:

- Character data model can attach future trained LoRAs.
- Job system can support training jobs.
- Model manifests can describe training-compatible base models.
- UI has no hard assumptions that block storyboard or training later.

## Research Tracks

These should run in parallel with implementation when possible.

### Image Model Research

Questions:

- Which supported image model should be the first adapter?
- What is the best current inference path for HunyuanImage-3.0?
- What is the best current inference path for Qwen Image/Image Edit?
- How should LightX2V Qwen-Image-Lightning fit as a faster Qwen path?
- How should FLUX.2-dev and FLUX.2-klein-4B fit into v1?
- Which model gives the best quality/control tradeoff on 24GB+ VRAM?
- What is the Apple/MPS story?

Output:

- Recommended first image adapter.
- Recommended adapter order after the first adapter.
- Dependency notes.
- VRAM notes.
- LoRA compatibility notes.
- Image editing support notes.
- License/download constraint notes.

### Video Model Research

Questions:

- Current best LTX2.3 inference path.
- Current best Wan2.2 inference path.
- Practical duration/resolution/FPS limits.
- LoRA support for each.
- First/last frame support maturity.

Output:

- Recommended first video adapter.
- Shot length recommendations.
- Manifest entries.
- Adapter settings map.

### Timeline Library Research

Questions:

- Which React timeline library best supports trims, clips, stills, speed, and transitions?
- Can it support our timeline JSON model?
- How much custom work is required?

Output:

- Recommended library.
- Known gaps.
- Export mapping to FFmpeg.

### Replace Person Research

Questions:

- Best face-only replacement pipeline.
- Best full-person replacement pipeline.
- Best detection/tracking/segmentation stack.
- Temporal consistency strategy.
- Clip length limits.

Output:

- Recommended v1 replacement path.
- Fallback plan if full-person replacement is not good enough.
- Minimum quality bar.

### Apple Runtime Research

Questions:

- Which adapters can run on Apple Silicon/MPS?
- Which models fail due to FP8/FP16 or unsupported ops?
- Is Docker viable for Apple acceleration, or is a native runtime needed?

Output:

- Apple feasibility matrix.
- Runtime recommendation.
- Required adapter abstractions.

## Suggested First Sprint

Sprint 1 should avoid model integration and build the skeleton correctly.

Tasks:

1. Create repo structure.
2. Add Docker Compose with web, api, worker placeholders.
3. Scaffold React + Vite app.
4. Scaffold FastAPI app.
5. Add `/api/v1/health`.
6. Add project directory config.
7. Add basic project create/open APIs.
8. Add basic web shell with project list.
9. Add placeholder Queue and Library routes.
10. Add initial JSON schema files.

Done means:

- App boots locally.
- A project can be created on disk.
- The frontend can show that project.
- The API and worker architecture has a place to grow.

## V1 Completion Definition

V1 is credible when a user can:

1. Create a project.
2. Import media.
3. Generate images.
4. Generate short videos from images and text.
5. Store all outputs as assets with recipes.
6. Assemble clips/stills into a timeline.
7. Export MP4.
8. Create Characters.
9. Run at least one useful Replace Person workflow on a selected short clip.
10. Use a multi-GPU queue deliberately.

V1 does not need:

- Character LoRA training.
- Storyboard view.
- Full audio mixing.
- Captions.
- Inpainting.
- Multiple person replacement in one video.
- Commercial content guardrails.
