// Single source of truth for job-type, job-status, and generation-quality enums
// shared across the web UI (sc-1657). Previously these literals were re-declared
// — and drifted — across QueueScreen, PresetManagerScreen, VideoStudio,
// ImageStudio, JobProgress, and constants.js.

// Job types that require a GPU worker. Keep in sync with the backend:
//   crates/sceneworks-core/src/jobs_store.rs::job_requires_gpu
//   apps/worker/scene_worker/runtime.py (SUPPORTED_JOB_TYPES + TRAINING_JOB_TYPES + CAPTION_JOB_TYPES)
export const GPU_REQUIRED_JOB_TYPES = new Set([
  "image_generate",
  "image_edit",
  "image_vqa",
  "image_interleave",
  "video_generate",
  "video_extend",
  "video_bridge",
  "person_replace",
  "lora_train",
  "training_caption",
]);

// Utility job types that run on any worker (no GPU required).
export const NON_GPU_JOB_TYPES = new Set([
  "model_download",
  "model_import",
  "model_convert",
  "lora_import",
  "prompt_refine",
]);

// Terminal job statuses (no further progress expected).
export const terminalStatuses = new Set(["completed", "failed", "canceled", "interrupted"]);

// Statuses that surface job actions (retry/repeat/cancel) in the queue.
export const actionStatuses = new Set(["failed", "canceled", "interrupted", "completed"]);

// Terminal statuses that represent an error/abnormal end (terminal minus completed).
export const errorStatuses = new Set(["failed", "canceled", "interrupted"]);

// Generation quality enum. The VALUES are what the worker honors
// (apps/worker/scene_worker/video_adapters.py step maps: fast | balanced | best;
// unknown values fall through to balanced). Labels are the user-facing names.
// Shared by Video Studio and the Preset Manager so a saved preset's quality
// always matches the studio control.
export const qualityChoices = [
  ["fast", "Draft"],
  ["balanced", "Balanced"],
  ["best", "Final"],
];
