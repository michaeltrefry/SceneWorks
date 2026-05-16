# SceneWorks Plan

## Product Summary

SceneWorks is a local Docker-based AI image and video generation web app for semi-technical AI hobbyists who want ComfyUI-class power without managing node graphs. It is also being designed first for the creator's own workflow.

The app should feel like a creative studio, but use practical tool wording. The video editor should feel closer to CapCut, while image and video generation should stay much simpler and more opinionated than ComfyUI.

Core promise:

> You do not need to manage a bunch of random nodes you know nothing about.

SceneWorks should support high-quality local generation, project-based asset management, character workflows, person replacement, and short-video assembly.

## Target User

- Semi-technical AI hobbyist.
- Comfortable with local Docker and GPUs.
- Finds ComfyUI too complex or too node-heavy.
- Wants strong defaults, but still wants advanced control when needed.
- Initial hardware target is 24GB+ VRAM.
- Windows/NVIDIA support can land first, but Apple support should remain a serious design target.

## Product Feel

- Creative studio look.
- Practical wording.
- Simple screens with advanced drawers.
- No node graph in the primary UI.
- Clear project library shared across generation and editing surfaces.
- Opinionated model and LoRA choices rather than "everything to everyone."

## V1 Core Workflows

The first working vertical slice should be:

1. Create a project.
2. Generate an image.
3. Save the image as an asset with full recipe metadata.
4. Send the image to image-to-video.
5. Generate a video clip.
6. Add the clip to the timeline.
7. Export an MP4.

This validates the most important foundation: projects, assets, recipes, models, jobs, GPU routing, generation, timeline integration, and export.

V1 must support:

- Text to image.
- Image edit from a starting image.
- Image to video.
- Text to video.
- First-frame/last-frame video generation.
- Clip extension.
- Bridge generation between timeline clips.
- Replace one selected person in a video.
- Project asset library.
- Basic-to-medium video editing.
- MP4 export.

## Main V1 Screens

### Projects

- Create and open projects.
- Projects are inspectable folders on disk.
- Projects should be designed for future portability.
- Project portability is desirable, but not a v1 showstopper.

### Image Studio

Modes:

- Text to Image.
- Edit Image.
- Character Image.
- Style Variations.

Simple mode:

- Prompt.
- Style/enhancement/character LoRA selections.
- Number of images to generate.
- Large thumbnails for generated images.
- Fullscreen preview.
- Easy discard of bad generations.
- Rating/favorite for kept generations.

Advanced drawer:

- Model selection.
- Seed.
- Negative prompt.
- More detailed generation settings.

Inpainting can come later.

### Video Studio

Modes:

- Image to Video.
- Text to Video.
- First/Last Frame.
- Extend Clip.
- Replace Person.

Image to video is expected to be the friendliest and most common starting point, but switching between I2V, T2V, and FFLF should be easy.

### Character Studio

Characters are first-class project objects.

Character types:

- Person.
- Creature.
- Object, if useful for consistent object workflows.

A Character can include:

- Uploaded reference images.
- Generated images.
- Extracted video frames.
- Approved reference images.
- Imported character LoRAs.
- Future trained LoRAs.
- Saved looks.
- Recipes that worked well.

Character LoRA training should be designed into the system, but does not need to ship in the first v1 slice. It is a top priority for the next version and should likely use the `ai-toolkit` repo.

### Editor

The editor should be CapCut-like and support assembling 5-10 minute short videos.

V1 editor scope:

- Imported and generated clips.
- Multiple timelines per project.
- Main video track plus recommended overlay support.
- Trim and arrange clips.
- Transitions.
- Still images.
- Speed changes.
- Side-by-side comparison.
- A/B toggle comparison.
- Timeline integration with generation tools.
- MP4 export.

Skip captions until audio support is more complete.

Timeline edits should be nondestructive. Replacing or regenerating a clip should keep the previous version available.

### Library

The Library is the shared project asset system and should be available across Image Studio, Video Studio, Character Studio, and Editor.

Assets include:

- Generated images.
- Generated videos.
- Uploaded images.
- Uploaded videos.
- Extracted still frames.
- Character references.
- Replacement outputs.
- Exported renders.

Assets should store:

- Full generation recipe.
- Source relationships.
- Ratings.
- Favorite state.
- Rejected/discarded state where useful.
- Notes.

Generated batches should immediately become assets, but users should be able to quickly hard-delete bad image generations.

### Models & LoRAs

The app should mostly download models automatically as needed, but include a Model Manager for visibility into installed and downloadable models.

Model and LoRA configuration should be manifest-driven, using JSON or JSONC.

Recommended manifest split:

- `builtin.models.json`
- `user.models.json`
- `builtin.loras.json`
- `user.loras.json`

User config should survive app upgrades.

Default models are downloaded from Hugging Face where possible. Some LoRAs may come from Civitai. User-supplied custom models and LoRAs are allowed, but custom arbitrary pipelines are out of scope for v1.

Advanced users should be able to swap model paths for known adapter families through text configuration.

### Queue

The job queue should be:

- FIFO.
- Pausable.
- Cancelable.
- Multi-GPU aware.
- Exclusive GPU reservation per job.

Users should be able to choose the GPU for a job. The app can prefer a GPU that already has the needed model loaded, but users can override that.

## Models

Initial model interests:

Images:

- HunyuanImage-3.0.
- HunyuanImage-3.0-Instruct / Distil if useful for image-to-image and faster deployment.
- Qwen Image.
- Qwen Image Edit.
- LightX2V Qwen-Image-Lightning.
- FLUX.2-dev.
- FLUX.2-klein-4B.
- Wan2.2 image generation via 1-frame video.

Video:

- Wan2.2.
- LTX2.3.

LTX2.3 is currently the preferred first-class video model target.

Image model support should include:

- [Tencent HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- [Qwen Image](https://github.com/QwenLM/Qwen-Image)
- [LightX2V Qwen-Image-Lightning](https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning)
- [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

The image implementation order still needs a research pass. The likely decision should be based on quality, image editing support, LoRA support, speed, 24GB VRAM behavior, license constraints, and Apple/MPS feasibility.

Working video-model assumption:

- Wan2.2 may start to loop around roughly 7 seconds.
- LTX2.3 is best at 15 seconds or less and may degrade heavily on longer clips.
- These limits should be validated during implementation, but the product should be designed around short generated shots rather than single long generations.
- SceneWorks should help users assemble 5-10 minute videos from many shorter clips.
- The editor, Library, bridge generation, extend generation, and timeline replacement flows are therefore core to making longer stories practical.

## LoRA System

LoRAs can be:

- Global.
- Project-only.

Global LoRAs are useful for built-in styles and enhancements:

- Fantasy styling.
- Anime styling.
- Cinematic styling.
- Detail enhancement.
- Color enhancement.

Project LoRAs are useful for:

- Character-specific LoRAs.
- Project-specific styles.
- User experiments.

LoRA categories:

- Style.
- Enhance.
- Character.
- Motion.
- Clothing/Object.
- Experimental.

Simple mode should allow up to 2 user-selected LoRAs in addition to built-in preset LoRAs.

LoRA compatibility should be filtered by model where possible. Built-in LoRAs should ship with trusted compatibility metadata. User-uploaded LoRAs should be inspected for metadata, but if compatibility is unclear, the user should choose the compatible model family manually.

An advanced override can allow unknown or incompatible LoRAs to be shown.

## Recipe Presets

Recipe presets are recommended.

Examples:

- Cinematic.
- Anime.
- Fantasy.
- Photoreal.
- Product Shot.

These should ultimately set model defaults, prompt fragments, negative prompt defaults, and built-in LoRAs.

Users should eventually be able to save custom presets from advanced settings.

## Character And Person Replacement

Replace Person is core v1.

User-facing modes:

- Face Only.
- Full Person, Keep Outfit.
- Full Person, Replace Outfit.

V1 should support replacing one selected person in a video. Future versions may support assigning multiple different Characters to different people in the same source video.

Recommended v1 flow:

1. User imports or selects a video clip.
2. App shows a representative frame.
3. App detects possible people.
4. User clicks/selects the person.
5. App tracks that person through the clip.
6. User can correct tracking drift if needed.
7. User selects a Character and replacement mode.
8. User chooses Fast, Balanced, or Best quality.
9. App generates a new replacement clip.
10. User can compare original and replacement side by side or with A/B toggle.

The replacement output should be a new clip asset. It should retain lineage back to the source video, selected person track, Character, mode, model, recipe, seed, and settings.

The source clip should remain available for comparison.

## Keyframe And Frame Extraction

Do not create a separate keyframe asset type in v1.

All extracted frames should be normal image assets with metadata describing their source and intended use.

An image can be used as:

- First frame.
- Last frame.
- Character reference.
- Style reference.
- Timeline still.
- Generation input.

Metadata should track source relationships such as:

- Extracted from clip.
- Source timestamp.
- Source timeline.
- Used as first frame.
- Used as last frame.

## Timeline To Generation Integration

Important editor integrations:

- Select clip and extend it.
- Extract still frame from any point in a video.
- Use a still frame for another generation.
- Generate bridge clip between two timeline clips.
- Use last frame of one clip and first frame of another as FFLF inputs.

For bridge generation:

- If clips are adjacent with no gap, tell the user to create space.
- If there is a gap, infer bridge duration from the timeline gap.
- Extract last frame of left clip.
- Extract first frame of right clip.
- Generate a bridge clip.
- Place the generated clip into the gap while preserving prior versions.

## Export

V1 export:

- MP4.
- Resolution presets: 640, 720, 1024, 1280.
- Aspect ratio presets: 16:9, 9:16, 1:1, and likely source/as-is.
- FPS depends on model-native output and export settings.

## Technical Architecture

Recommended stack:

- Frontend: React + Vite.
- Backend API: FastAPI.
- Workers: Python generation workers.
- Queue/dispatcher: SQLite-backed job system.
- Storage: SQLite metadata plus files on disk plus sidecar JSON.
- Editing/export: FFmpeg-backed.
- Runtime: Docker Compose with GPU support.
- Models: manifest-driven adapters.
- ComfyUI: not a runtime dependency.

Python is acceptable because the AI ecosystem largely depends on it. The app should keep Python generation code behind clean backend and worker boundaries.

## Backend Shape

The backend should expose stable internal APIs early.

Likely API areas:

- Projects.
- Assets.
- Recipes.
- Jobs.
- Models.
- LoRAs.
- Characters.
- Timelines.
- Editor/export.
- Generation modes.

Generation should be job-based, not direct blocking API calls.

Jobs should support:

- Queuing.
- Canceling.
- Progress.
- Preview updates when supported.
- Failure messages.
- GPU assignment.
- Model-loading awareness.

API boundary decisions:

- Frontend creates jobs for generation/export work and never calls model adapters directly.
- Every long-running action becomes a job.
- Job outputs become assets automatically, except failed/canceled outputs and unpromoted temp previews.
- Project open/create is file-path based, and the app also keeps a lightweight recent-projects registry.
- Backend owns all filesystem writes.
- Frontend uploads files or requests actions; backend writes project files, sidecars, and database rows.
- Use REST for normal operations.
- Use WebSocket or Server-Sent Events for job/queue progress.
- Version the API from the start with `/api/v1/...`.

Likely API groups:

- `/api/v1/projects`
- `/api/v1/assets`
- `/api/v1/generation-sets`
- `/api/v1/recipes`
- `/api/v1/jobs`
- `/api/v1/models`
- `/api/v1/loras`
- `/api/v1/characters`
- `/api/v1/timelines`
- `/api/v1/exports`

## Storage

Use:

- SQLite for searchable metadata.
- Files on disk for media assets.
- Sidecar JSON as the reliable portable recipe source.
- Embedded metadata where useful, but do not rely on it exclusively because media tools may strip it.

Source-of-truth decisions:

- Media files plus sidecars are the portable source of truth for assets and recipes.
- SQLite is authoritative for active app state such as jobs, current timeline index data, recent UI state, and fast indexes.
- Add a rebuild/reindex command later so project folders can be repaired or re-imported from sidecars.
- Timelines should be stored as both JSON files and SQLite-indexed records.
- Timeline JSON files are portable snapshots.
- Completed job history can stay with the project.
- Queued/running jobs are machine-local state and should not be resumed when a project is moved.
- Model and LoRA manifests live globally, with project overrides inside the project.
- Project assets refer to global models by manifest ID rather than absolute path.
- Raw recipe metadata can also capture resolved model paths for reproducibility and debugging.
- Sidecars include schema version and app version for future migrations.

Project folders should be inspectable.

Suggested structure:

```text
MyProject.sceneworks/
  project.json
  assets/
    images/
    videos/
    uploads/
    frames/
  recipes/
  timelines/
  renders/
  characters/
  cache/
```

Generated filenames should be human-readable.

## Model Storage

Model files should be mapped to disk by Docker Compose.

Default:

- A subfolder from where Docker Compose is run.

Users can change storage paths by editing Docker Compose or config.

Hugging Face cache reuse is important:

- If `HF_HOME` or the Hugging Face cache is available, models downloaded from Hugging Face should use that cache.
- Avoid duplicating large models unnecessarily.

The app should support:

- App-managed model folder.
- Hugging Face cache.
- Local paths.
- Manual downloaded models.
- User-edited manifest paths.
- Optional Hugging Face auth token support.

## Core Data Model

SceneWorks should be anchored around these objects:

- Project.
- Asset.
- GenerationSet.
- Recipe.
- Job.
- Model.
- LoRA.
- Character.
- Look.
- Timeline.
- TimelineItem.
- Render.

Important decisions:

- Imported media is copied into the project by default so projects do not break when original files move.
- Assets are grouped by type on disk.
- Asset recipes live beside assets as sidecar JSON, and SQLite indexes them.
- Reusable saved recipes live under `recipes/`.
- Generated asset filenames include date, model family, short prompt slug, and sequence.
- Users can rename assets in the UI without renaming files on disk.
- Normal Library deletion uses project trash.
- Immediate discard from a fresh generation can hard-delete if the asset has not been used anywhere.

### GenerationSet

A GenerationSet groups outputs from one generation request. For example, four images from one prompt should be one GenerationSet, so the UI can show siblings, compare them, and preserve the original batch context.

### Recipe

Recipes should store both normalized settings and raw adapter-specific settings.

- Normalized settings support UI, search, filtering, and cross-model concepts.
- Raw adapter settings preserve exact reproduction when possible.

### Asset Ratings

Assets support:

- 1-5 star rating.
- Favorite.
- Rejected.
- Notes.

Rejected assets should be hidden by default, with a filter to show them.

Timeline exports should become normal render assets under `assets/renders/`, with recipe and lineage pointing back to the timeline.

Project-local LoRAs can be copied or referenced. Default to copying character/project LoRAs into the project for portability.

### Timeline Data Model

Timeline decisions:

- Timeline items reference `assetId`, not file paths.
- Timeline items store source trim separately from timeline placement.
- `sourceIn` and `sourceOut` define which part of the source asset is used.
- `timelineStart` and `timelineEnd` define where the item appears in the edit.
- Replacement generation creates a new asset.
- Timeline items keep version history of asset IDs.
- The current timeline item version points to the active asset.
- Transitions are modeled as separate `Transition` objects between timeline items.
- Speed changes live on timeline items, not assets.
- The same clip can be used at different speeds in different places.
- V1 editor undo/redo can be frontend-only for the active session.
- Timeline saves/revisions should be explicit.
- Full persistent undo stack can wait.

## Asset Sidecars

Asset sidecars should sit next to the media file:

```text
assets/images/2026-05-16_qwen_noir-alley_0001.png
assets/images/2026-05-16_qwen_noir-alley_0001.sceneworks.json
```

Example sidecar shape:

```json
{
  "schemaVersion": 1,
  "id": "asset_...",
  "projectId": "project_...",
  "generationSetId": "genset_...",
  "type": "image",
  "displayName": "Noir alley",
  "createdAt": "2026-05-16T12:00:00Z",
  "file": {
    "path": "assets/images/2026-05-16_qwen_noir-alley_0001.png",
    "mimeType": "image/png",
    "width": 1024,
    "height": 1024,
    "duration": null,
    "fps": null
  },
  "status": {
    "favorite": false,
    "rating": 0,
    "rejected": false,
    "trashed": false
  },
  "recipe": {
    "mode": "text_to_image",
    "model": "qwen_image",
    "prompt": "",
    "negativePrompt": "",
    "seed": 123,
    "loras": [],
    "normalizedSettings": {},
    "rawAdapterSettings": {}
  },
  "lineage": {
    "parents": [],
    "sourceAssetId": null,
    "sourceTimestamp": null,
    "jobId": "job_..."
  }
}
```

## Job System

Most meaningful actions in SceneWorks should be jobs:

- `model_download`
- `image_generate`
- `image_edit`
- `video_generate`
- `video_extend`
- `video_bridge`
- `person_track`
- `person_replace`
- `frame_extract`
- `timeline_export`
- `lora_import`

Future jobs:

- `character_lora_train`
- `audio_generate`
- `caption_transcribe`

Recommended lifecycle:

- `queued`
- `preparing`
- `downloading`
- `loading_model`
- `running`
- `saving`
- `completed`
- `failed`
- `canceled`
- `interrupted`

Job UX decisions:

- Queue screen is global, with project filtering.
- Running jobs survive browser refresh because state is backend-owned.
- Running jobs interrupted by backend restart are marked failed/interrupted in v1.
- Canceling a generation discards temp previews unless they were promoted to assets.
- Failed jobs support `Retry` and `Duplicate with changes`.
- Show elapsed time and progress stage.
- Show ETA only when an adapter can provide a reliable estimate.

## Worker Model

- FastAPI receives job requests.
- Job goes into SQLite.
- Dispatcher assigns job to an available worker/GPU.
- Worker owns one GPU exclusively while a job runs.
- Worker streams progress back to backend.
- Backend updates SQLite and notifies UI.
- Worker writes output files and sidecars.
- Backend indexes final assets.

For multi-GPU:

- One worker process per GPU is the cleanest default.
- UI lets the user choose GPU.
- Default GPU can be `auto`.
- Dispatcher can prefer a GPU with the requested model already loaded.
- User override wins.

## Model And LoRA Manifests

SceneWorks should use known adapter families with manifest-driven model paths. V1 should not allow arbitrary custom pipelines from the UI. Advanced users can swap paths and variants within known adapter families through text configuration.

Manifest decisions:

- Show model download size before starting a job when known.
- Allow users to predownload models from Model Manager.
- If a model is missing, generation should automatically queue a `model_download` job first.
- Missing model downloads block only that job path, not the whole queue.
- Model download jobs do not consume GPU slots.
- LoRA import reads safetensors metadata when possible.
- After LoRA metadata inspection, users can edit category, compatibility, trigger words, default weight, and scope.

Example model manifest entry:

```jsonc
{
  "id": "ltx_2_3_i2v",
  "name": "LTX 2.3",
  "family": "ltx",
  "type": "video",
  "adapter": "ltx_video",
  "capabilities": ["text_to_video", "image_to_video"],
  "downloads": [
    {
      "provider": "huggingface",
      "repo": "example/ltx-2.3",
      "files": ["model.safetensors"]
    }
  ],
  "paths": {
    "model": "${HF_CACHE}/example/ltx-2.3/model.safetensors"
  },
  "defaults": {
    "resolution": 720,
    "fps": 24,
    "durationSeconds": 5,
    "quality": "balanced"
  },
  "limits": {
    "resolutions": [640, 720, 1024, 1280],
    "durationsSeconds": [3, 5, 8, 10],
    "nativeFps": [24]
  },
  "loraCompatibility": {
    "families": ["ltx"],
    "types": ["style", "enhance", "character", "motion"]
  },
  "ui": {
    "label": "LTX 2.3",
    "description": "Fast high-quality local video generation.",
    "recommendedFor": ["image_to_video", "text_to_video"]
  }
}
```

Example LoRA manifest entry:

```jsonc
{
  "id": "cinematic_detail_global",
  "name": "Cinematic Detail",
  "scope": "global",
  "category": "enhance",
  "compatibleFamilies": ["ltx", "wan", "qwen"],
  "path": "loras/global/cinematic_detail.safetensors",
  "triggerWords": [],
  "defaultWeight": 0.6,
  "builtIn": true,
  "ui": {
    "description": "Adds sharper cinematic detail."
  }
}
```

## Adapter Layer

Adapters should implement a common generation interface conceptually like:

```python
class GenerationAdapter:
    def prepare(self, job): ...
    def ensure_models(self, manifest_entry): ...
    def estimate_requirements(self, job): ...
    def run(self, job, progress): ...
    def cancel(self, job): ...
    def cleanup(self, job): ...
```

This should stay lightweight early, but adapters should exist from the start so Qwen, Wan, LTX, replacement pipelines, and future Apple runtimes do not tangle together.

## UI Structure

Once inside a project:

```text
Top bar:
  Project name | Active GPU | Queue status | Settings

Left nav:
  Library
  Image
  Video
  Characters
  Editor
  Queue

Asset tray:
  Recent assets, selected assets, drag/drop targets
```

Design notes:

- Darker neutral canvas.
- Strong media previews.
- Restrained controls.
- Avoid a sci-fi purple dashboard.
- Calm, capable, creative studio feel.

Asset tray behavior:

- Visible by default on desktop.
- Collapsible on desktop.
- Collapsed by default on smaller screens.

Generated results appear both in the main review workspace and in the persistent asset tray/library.

Each screen remembers its last settings per project.

Advanced drawer open/closed state is remembered per screen/user.

Buttons should be reliable first. Drag/drop should follow soon after, but does not need to block v1.

V1 keyboard shortcuts should be limited to editor basics:

- Play/pause.
- Delete.
- Undo.
- Redo.
- Maybe split later if implemented.

## Library UX

Asset grid supports:

- Filter by type: image, video, upload, frame, render, character reference.
- Filter by rating/favorite.
- Hide rejected by default.
- Search by prompt/model/tag.
- Sort by newest, rating, model, type.
- Multi-select.
- Send to Image, Video, Character, or Editor.
- Drag asset into generation inputs or timeline.

Asset detail panel shows:

- Preview.
- Display name.
- Rating/favorite.
- Prompt summary.
- Model.
- LoRAs.
- Source lineage.
- Buttons: `Reuse`, `Variations`, `Send to Video`, `Send to Editor`, `Extract Frame`, `Trash`.

## Screen UX Details

### Image Studio UX

Simple panel:

```text
Mode: Text to Image | Edit Image | Character Image | Style Variations
Prompt
Style preset
Character / LoRA selections
Count
Generate
```

Advanced drawer:

```text
Model
Seed
Negative prompt
Resolution
Adapter-supported settings
```

Generated results:

- Large grid.
- Rate/favorite/discard directly on each result.
- Fullscreen view.
- Send selected result to Video Studio or Editor.

### Video Studio UX

Simple panel:

```text
Mode: Image to Video | Text to Video | First/Last Frame | Extend Clip | Replace Person
Prompt
Input image/video/frame slots depending on mode
Character, style, LoRA selections
Duration
Aspect ratio/resolution
Quality: Fast | Balanced | Best
Generate
```

Advanced drawer:

```text
Model
Seed
Negative prompt
FPS if supported
Adapter-specific controls
GPU selection
```

### Character Studio UX

Character detail includes:

- Name.
- Type: Person, Creature, Object.
- Reference images.
- Approved references.
- Looks.
- Imported LoRAs.
- Test Character button.
- Send Character to image/video generation.

### Editor UX

Core interactions:

- Drag assets to timeline.
- Trim clips.
- Add transitions.
- Change speed.
- Add still images.
- Right-click clip: `Regenerate`, `Extend`, `Extract Frame`, `Replace Person`, `Show Source`, `Duplicate`.
- Select gap between clips: `Generate Bridge`.
- Export MP4.

Comparison:

- A/B toggle.
- Side-by-side view for replacement outputs.

## Editor Scope

V1 should support multiple video tracks only if the chosen timeline library makes it straightforward. Otherwise, use:

- One main video track.
- One overlay track.
- One audio placeholder/import track.

Editor scope decisions:

- Imported audio files can be supported as passthrough if easy, but no audio mixing UI yet.
- Still images on the timeline have duration and basic fit modes: fit, fill/crop, and possibly stretch in advanced.
- Initial transitions: cut, crossfade, fade from black, fade to black.
- Speed controls: 0.25x, 0.5x, 1x, 2x, custom in advanced.
- Timeline export renders from timeline JSON through FFmpeg.
- Do not use browser canvas recording for final export.
- Timeline preview can be approximate in-browser.
- Final export is backend-rendered.
- Skip multi-layer compositing effects beyond overlay track opacity/position if easy.
- Timeline creation supports project aspect ratios: 16:9, 9:16, 1:1.
- Custom aspect ratios can come later.
- No collaborative editing in v1.
- No cloud sync in v1.
- No fancy media bins beyond the local project Library in v1.

## Apple Support

Apple support is desired and should be treated as a serious target, but Windows/NVIDIA may land first.

Design implication:

- Do not bake CUDA assumptions into the product model.
- Keep generation adapters isolated.
- Allow runtime/backend variants.
- Expect Apple/MPS support to require separate testing and possibly separate packaging from NVIDIA Docker.

## Security And Access

The app should be usable from another device on the LAN.

Because it may contain private media, large local model management, and long-running GPU jobs, plan for a simple local password or pairing token. This is for privacy and control, not content moderation.

No product-level content guardrails are required.

## Next Version Priorities

High-priority next version:

- Local Character LoRA training using `ai-toolkit`.

Other next-version candidates:

- Storyboard/shot planning view.
- Inpainting.
- More advanced audio support.
- Captions and speech-to-text.
- Multiple person replacement in one video.
- User-created presets.
- More detailed tracking correction tools.
- Style application to clips.

## Open Questions

- Which image model should be the first implementation target?
- Which existing React timeline/editor library best fits the editor?
- What exact video/person replacement pipeline is stable enough for v1?
- How much of Apple support is feasible inside Docker versus a separate local runtime?
- What should the exact JSON/JSONC manifest schema look like?
- What are the minimum viable FFmpeg timeline features for the first export path?

## Current Build Order Recommendation

### Milestone 0: Project Skeleton

Goal: app boots locally.

- Docker Compose.
- Frontend React + Vite.
- Backend FastAPI.
- Shared config folders.
- Basic health checks.
- Basic app shell.

### Milestone 1: Project And Asset Spine

Goal: media can be imported, stored, viewed, and described.

- Project creation/open.
- Project folder structure.
- SQLite metadata.
- Asset import.
- Asset grid.
- Asset detail panel.
- Sidecar JSON.
- Rating/favorite/trash.
- Human-readable filenames.

### Milestone 2: Jobs And Workers

Goal: long-running work is reliable.

- Job queue.
- Worker process.
- Progress updates.
- Cancel/retry.
- Global queue screen.
- GPU selection placeholder.
- Job survives browser refresh.
- Interrupted job handling on backend restart.

### Milestone 3: Model Manifests And Downloads

Goal: models are config-driven.

- Built-in/user JSONC manifests.
- Model Manager.
- Hugging Face cache support.
- Auto-download jobs.
- Download progress.
- Path swapping for known adapters.

### Milestone 4: First Image Generation

Goal: produce real generated images.

- Pick first image model after research.
- Implement image adapter.
- Save outputs as assets.
- Store full recipes.
- Batch review UI.
- Simple LoRA selection shape, even if built-ins are minimal.

### Milestone 5: First Video Generation

Goal: image-to-video works end to end.

- Implement LTX2.3 or chosen first video adapter.
- I2V flow.
- T2V flow if same adapter supports it.
- Save clip outputs as assets.
- Preview and review generated clips.

### Milestone 6: Basic Editor And Export

Goal: make a simple short video.

- Existing timeline library.
- Import/generated clips on timeline.
- Trim/arrange.
- Still images.
- Basic transitions if library supports it cleanly.
- Speed changes.
- FFmpeg MP4 export.
- Render assets.

### Milestone 7: Timeline Generation Hooks

Goal: editor becomes AI-aware.

- Extract frame.
- Extend clip.
- FFLF bridge generation.
- Send selected frame/clip to Video Studio.
- Replace timeline clip nondestructively.

### Milestone 8: Characters

Goal: reusable identity objects.

- Character creation.
- Person/Creature/Object type.
- Reference images.
- Approved references.
- Looks.
- Imported character LoRAs.
- Test Character workflow.

### Milestone 9: Replace Person

Goal: core differentiator lands.

- Import source video.
- Detect people in representative frame.
- Select one person.
- Track through clip.
- Optional correction if feasible.
- Replacement modes: Face Only, Full Person Keep Outfit, Full Person Replace Outfit.
- Fast/Balanced/Best.
- Output replacement clip.
- Side-by-side and A/B compare.

### Milestone 10: Multi-GPU Polish

Goal: two-GPU machines behave properly.

- Worker per GPU.
- Exclusive GPU reservation.
- User-selected GPU.
- Loaded-model preference.
- Queue awareness.
- Better GPU status display.

## Top Risks

1. Replace Person quality.
2. Video model integration churn.
3. Apple support.
4. Timeline editor complexity.
5. Model storage size.
6. LoRA compatibility metadata.
