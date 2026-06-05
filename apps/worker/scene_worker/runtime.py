from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import httpx

from sceneworks_shared import find_project_path, utc_now

from .caption_adapters import run_training_caption_job
from .gpu import cpu_worker_id, discover_gpu, discover_gpus, gpu_utilization, gpu_worker_id
from .image_adapters import (
    ChromaDiffusersAdapter,
    FluxDiffusersAdapter,
    KolorsDiffusersAdapter,
    LensTurboAdapter,
    MlxFlux2Adapter,
    MlxFluxAdapter,
    MlxQwenAdapter,
    MlxZImageAdapter,
    ProceduralImageAdapter,
    QwenImageAdapter,
    SdxlDiffusersAdapter,
    SenseNovaU1Adapter,
    ZImageDiffusersAdapter,
    create_image_adapter,
    evict_other_image_adapters,
    image_request_from_job,
    release_image_worker_memory,
    run_image_detail,
    run_image_upscale,
    torch_inference_backend_available,
)
from .instantid_adapter import InstantIDAdapter
from .pulid_flux_adapter import PuLIDFluxAdapter
from .person_adapters import (
    detector_backend_available,
    run_person_detect,
    run_person_track,
    segmenter_backend_available,
    tracker_backend_available,
)
from .pose_adapters import pose_detector_backend_available, run_pose_detect
from .prompt_refine import PromptRefineError, PromptRefiner
from .settings import WorkerSettings
from .training_adapters import (
    SUPPORTED_TRAINING_PLAN_VERSION,
    create_training_kernel,
    dry_run_training_summary,
    validate_training_plan,
)
from .video_adapters import create_video_adapter


LoadedModelsSource = Callable[[], list[str]] | None
# Predicate adapters call to learn whether the running job has been canceled.
CancelCallback = Callable[[], bool]
IMAGE_JOB_TYPES = ("image_generate", "image_edit")
VIDEO_JOB_TYPES = ("video_generate", "video_extend", "video_bridge", "person_replace")
# Real person detection/tracking run in the Python GPU worker (YOLO/ByteTrack/SAM2),
# replacing the Rust CPU procedural placeholders. Advertised only when the backend
# is installed; the Rust utility worker keeps procedural previews under the distinct
# person_detect_preview / person_track_preview capabilities. Keep in sync with
# crates/sceneworks-core/src/contracts.rs::WorkerCapability and jobs_store::worker_supports_job.
PERSON_JOB_TYPES = ("person_detect", "person_track")
# DWPose whole-body keypoint detection for the Pose Library (epic 2282). onnxruntime
# (rtmlib) like person_detect — advertised by the Python worker only when the backend
# is installed; routed via requested_gpu (not in NON_GPU_JOB_TYPES). Keep in sync with
# crates/sceneworks-core/src/contracts.rs::JobType / WorkerCapability.
POSE_JOB_TYPES = ("pose_detect",)
# Keep GPU-required job types in sync with
# crates/sceneworks-core/src/jobs_store.rs::job_requires_gpu and
# apps/web/src/screens/QueueScreen.jsx::gpuRequiredJobTypes.
SUPPORTED_JOB_TYPES = IMAGE_JOB_TYPES + VIDEO_JOB_TYPES
# Training is GPU-required like generation, but a GPU worker advertises lora_train
# even without an inference backend: the dry-run path only validates a Rust-resolved
# plan (no torch needed). Real (non-dry-run) execution needs the backend, so it is
# advertised as a distinct capability (TRAINING_EXECUTE_CAPABILITIES) only when the
# backend is available — the Rust dispatcher routes dryRun:false jobs only to workers
# that advertise it, instead of letting a torch-less worker claim and fail one. Keep
# in sync with crates/sceneworks-core/src/contracts.rs::WorkerCapability and
# jobs_store::worker_supports_job.
TRAINING_JOB_TYPES = ("lora_train",)
TRAINING_EXECUTE_CAPABILITIES = ("lora_train_execute",)
CAPTION_JOB_TYPES = ("training_caption",)
# Visual question answering: text output (not an image asset), GPU-required like
# generation. Keep in sync with crates/sceneworks-core/src/contracts.rs::JobType /
# WorkerCapability, jobs_store::job_requires_gpu, and QueueScreen gpuRequiredJobTypes.
VQA_JOB_TYPES = ("image_vqa",)
# Interleaved text-image generation: mixed output (ordered text + generated images),
# GPU-required. sc-1604 wires the job type + routing + capability; the real worker
# path (vendored interleave_gen) lands in sc-1606, where run_interleave_job's stub is
# replaced. Keep in sync with contracts.rs::JobType/WorkerCapability,
# jobs_store::job_requires_gpu, and QueueScreen gpuRequiredJobTypes.
INTERLEAVE_JOB_TYPES = ("image_interleave",)
# Prompt refinement (sc-2041): rewrites a user's prompt with a small instruction
# LLM loaded in-process (like JoyCaption captioning). It needs the torch inference
# backend, so it is advertised only by a backend-capable Python worker — the Rust
# utility worker never claims it. Keep in sync with
# crates/sceneworks-core/src/contracts.rs::JobType/WorkerCapability.
PROMPT_REFINE_JOB_TYPES = ("prompt_refine",)
# Standalone image upscale (Image Editor, epic 2427): Real-ESRGAN / AuraSR on an
# existing asset → one child asset. Torch-backed, GPU-required like generation;
# advertised by a backend-capable Python worker. Keep in sync with
# contracts.rs::JobType/WorkerCapability, jobs_store::job_requires_gpu, and
# apps/web/src/jobTypes.js::GPU_REQUIRED_JOB_TYPES.
IMAGE_UPSCALE_JOB_TYPES = ("image_upscale",)
# Standalone tile-ControlNet detail refine (Image Editor, epic 2427; spike sc-2437):
# SDXL/RealVisXL img2img + a tile ControlNet over feathered tiles on an existing
# asset → one child asset. Torch-backed, GPU-required like generation; advertised by
# a backend-capable Python worker. Keep in sync with contracts.rs::JobType/
# WorkerCapability, jobs_store::job_requires_gpu, and apps/web/src/jobTypes.js.
IMAGE_DETAIL_JOB_TYPES = ("image_detail",)
# Every runtime-dispatchable job-type group, in a stable order, for the
# ``--check`` diagnostic. Derived from the groups above so run_check can't drift
# out of sync and underreport readiness (sc-1635: VQA + interleave were
# advertised and dispatched but missing from the check output).
# ``lora_train_execute`` is a capability flag, not a dispatchable job type, so it
# is intentionally absent here (it still appears in the ``capabilities`` field).
ALL_JOB_TYPES = (
    SUPPORTED_JOB_TYPES
    + PERSON_JOB_TYPES
    + POSE_JOB_TYPES
    + TRAINING_JOB_TYPES
    + CAPTION_JOB_TYPES
    + VQA_JOB_TYPES
    + INTERLEAVE_JOB_TYPES
    + PROMPT_REFINE_JOB_TYPES
    + IMAGE_UPSCALE_JOB_TYPES
    + IMAGE_DETAIL_JOB_TYPES
)


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


class ApiClient:
    def __init__(self, settings: WorkerSettings) -> None:
        headers = {}
        if settings.access_token:
            headers["X-SceneWorks-Token"] = settings.access_token
        self.client = httpx.Client(base_url=settings.api_url, headers=headers, timeout=20)

    def post(self, path: str, payload: dict) -> dict:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def get(self, path: str) -> dict:
        response = self.client.get(path)
        response.raise_for_status()
        return response.json()


def worker_capabilities(gpu: dict) -> list[str]:
    gpu_capabilities = set(gpu["capabilities"])
    capabilities = set(gpu["capabilities"]) - {"placeholder"}
    is_gpu_worker = "cpu" not in gpu_capabilities and "gpu" in gpu_capabilities
    if is_gpu_worker:
        # Dry-run plan validation needs no inference backend, so a GPU worker can
        # claim lora_train even before torch/CUDA is installed (e.g. on Mac).
        capabilities |= set(TRAINING_JOB_TYPES)
        if torch_inference_backend_available():
            capabilities |= set(SUPPORTED_JOB_TYPES)
            capabilities |= set(CAPTION_JOB_TYPES)
            capabilities |= set(VQA_JOB_TYPES)
            capabilities |= set(INTERLEAVE_JOB_TYPES)
            # Standalone image upscale reuses the torch-backed engines (sc-2431).
            capabilities |= set(IMAGE_UPSCALE_JOB_TYPES)
            # Standalone tile-ControlNet detail refine: SDXL/RealVisXL img2img + tile
            # CN, so it needs the same torch inference backend (sc-2437/sc-2438).
            capabilities |= set(IMAGE_DETAIL_JOB_TYPES)
            # Prompt refinement loads a small LLM in-process (like captioning), so
            # it needs the inference backend and is advertised alongside it.
            capabilities |= set(PROMPT_REFINE_JOB_TYPES)
            # Only a backend-capable worker advertises real training execution, so
            # the queue won't route a dryRun:false job to a worker that can't train.
            capabilities |= set(TRAINING_EXECUTE_CAPABILITIES)
        # Real detection/tracking/segmentation are advertised per installed backend
        # so the queue never routes a real person job to a worker that can't run it.
        if detector_backend_available():
            capabilities.add("person_detect")
        if tracker_backend_available():
            capabilities.add("person_track")
        if segmenter_backend_available():
            capabilities.add("person_segment")
        # DWPose whole-body keypoints (Pose Library) — onnxruntime/rtmlib, advertised
        # only when installed so the queue never routes a pose job to a worker that
        # can't run it.
        if pose_detector_backend_available():
            capabilities.add("pose_detect")
    return sorted(capabilities)


def loaded_models_from_adapter(adapter: object, *, job_id: str | None = None) -> list[str]:
    loaded_models = getattr(adapter, "loaded_models", None)
    if not callable(loaded_models):
        return []
    return resolve_loaded_models(loaded_models, job_id=job_id)


def loaded_models_from_adapters(adapters: dict[str, object]) -> list[str]:
    models: set[str] = set()
    for adapter in adapters.values():
        models.update(loaded_models_from_adapter(adapter))
    return sorted(models)


def register_worker(api: ApiClient, settings: WorkerSettings, gpu: dict, loaded_models: list[str] | None = None) -> None:
    payload = {
        "workerId": settings.worker_id,
        "gpuId": gpu["id"],
        "gpuName": gpu["name"],
        "capabilities": worker_capabilities(gpu),
        "loadedModels": loaded_models or [],
    }
    if gpu.get("utilization"):
        payload["utilization"] = gpu["utilization"]
    worker = api.post("/api/v1/workers/register", payload)
    emit({"event": "registered", "worker": worker, "reportedAt": utc_now()})


def heartbeat(
    api: ApiClient,
    settings: WorkerSettings,
    status: str,
    current_job_id: str | None = None,
    loaded_models: list[str] | None = None,
) -> None:
    payload = {"status": status, "currentJobId": current_job_id, "loadedModels": loaded_models or []}
    utilization = gpu_utilization(getattr(settings, "gpu_id", "cpu"))
    if utilization:
        payload["utilization"] = utilization
    api.post(
        f"/api/v1/workers/{settings.worker_id}/heartbeat",
        payload,
    )


def adapter_backend(adapter: object | None, settings: WorkerSettings) -> str:
    """Resolve the runtime backend label the WorkerProgressCard's arch pill
    should display for this job (sc-2086 follow-up).

    The previous arch pill was a heuristic on the worker's gpu_name, which
    couldn't tell an MLX run from a Diffusers-on-MPS run on the same Apple
    Silicon worker. This walks the actual adapter that ran the job:

      - Any adapter whose class starts with "Mlx" runs through Apple's MLX
        runtime — that's mlx.
      - settings.gpu_id == "mps" → torch on MPS (Diffusers / Wan torch path).
      - settings.gpu_id == "cpu" → CPU-only path (utility jobs, prompt refine
        falling back to CPU, etc.).
      - Everything else (NVIDIA gpu ids like "gpu-0") → cuda.

    Lets the card report what *actually* ran, not what the worker advertised.
    """
    if adapter is not None and type(adapter).__name__.startswith("Mlx"):
        return "mlx"
    gpu_id = getattr(settings, "gpu_id", "cpu")
    if gpu_id == "mps":
        return "mps"
    if gpu_id == "cpu":
        return "cpu"
    return "cuda"


def _sample_into_peaks(peaks: dict, settings: WorkerSettings) -> None:
    """Sample current GPU utilization and ratchet the running max into a
    per-job peaks dict (sc-2086).

    Shared by `track_job_peaks` (sampled at each progress() call boundary, the
    one-shot path) and by `keep_job_alive` (sampled continuously during long
    blocking diffusion phases that don't tick progress() themselves — without
    this the video adapters miss the actual GPU-heavy window entirely because
    their progress() callbacks fire at boundary points only).

    Defensive against minimal test settings (SimpleNamespace) that may not
    carry a gpu_id attribute — falls back to "cpu" the same way heartbeat()
    does. gpu_utilization() returns None for unknown gpu ids, so the helper
    no-ops cleanly when there's nothing to sample.
    """
    gpu_id = getattr(settings, "gpu_id", "cpu")
    utilization = gpu_utilization(gpu_id)
    if not utilization:
        return
    mem_used = utilization.get("memoryUsedMb")
    mem_total = utilization.get("memoryTotalMb")
    if mem_used and mem_total and mem_total > 0:
        pct = min(100.0, (float(mem_used) / float(mem_total)) * 100.0)
        peaks["memory"] = max(peaks.get("memory", 0.0), pct)
    load = utilization.get("gpuLoadPercent")
    if load is not None:
        peaks["load"] = max(peaks.get("load", 0.0), min(100.0, float(load)))


def track_job_peaks(
    payload: dict,
    peaks: dict,
    settings: WorkerSettings,
    backend: str | None = None,
) -> None:
    """Sample GPU utilization, ratchet the running max into `peaks`, and fold
    the peaks into a progress payload (sc-2086).

    `peaks` is a per-job dict (`{"memory": float, "load": float}`) that the
    caller carries across every progress() call inside one job, so the
    completed-row hardware meters show the highest values observed during the
    run instead of whatever the last sample happened to be. The same dict is
    shared with `keep_job_alive` (via `run_blocking_job_step`) so peaks are
    captured both at progress() boundaries and during blocking work.

    `backend` is the runtime label the WorkerProgressCard's arch pill shows
    ("mlx" / "mps" / "cuda" / "cpu") — see `adapter_backend()`. Pass once per
    progress call; the API coalesces first-non-null so subsequent calls are
    idempotent and the value persists across the run.
    """
    _sample_into_peaks(peaks, settings)
    if peaks.get("memory"):
        payload["peakGpuMemoryPct"] = peaks["memory"]
    if peaks.get("load"):
        payload["peakGpuLoadPct"] = peaks["load"]
    if backend is not None:
        payload["backend"] = backend


def resolve_loaded_models(source: LoadedModelsSource, *, job_id: str | None = None) -> list[str]:
    if source is None:
        return []
    try:
        return source()
    except Exception as exc:
        payload = {"event": "loaded_models_failed", "error": str(exc), "reportedAt": utc_now()}
        if job_id:
            payload["jobId"] = job_id
        emit(payload)
        return []


def heartbeat_with_loaded_models(
    api: ApiClient,
    settings: WorkerSettings,
    status: str,
    current_job_id: str,
    loaded_models: LoadedModelsSource,
) -> None:
    heartbeat(
        api,
        settings,
        status,
        current_job_id,
        loaded_models=resolve_loaded_models(loaded_models, job_id=current_job_id),
    )


def update_job(api: ApiClient, job_id: str, payload: dict[str, Any]) -> dict:
    job = api.post(f"/api/v1/jobs/{job_id}/progress", payload)
    emit({"event": "job_progress", "jobId": job_id, "status": job["status"], "stage": job["stage"]})
    return job


def friendly_failure(job_kind: str, exc: Exception) -> tuple[str, str]:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    exc_name = exc.__class__.__name__.lower()
    if "outofmemory" in exc_name or "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return (
            f"{job_kind} failed because the GPU ran out of memory.",
            (
                "GPU memory was exhausted. Try a lower resolution, shorter clip, smaller batch count, "
                f"or a different GPU. Technical detail: {detail}"
            ),
        )
    if "cuda-enabled pytorch" in lowered or "torch.cuda.is_available" in lowered:
        return (
            f"{job_kind} failed because the worker is missing CUDA-enabled PyTorch.",
            (
                "The worker claimed a GPU inference job, but PyTorch cannot use CUDA in that environment. "
                "Rebuild the worker image with CUDA PyTorch support, then restart the worker and retry. "
                f"Technical detail: {detail}"
            ),
        )
    disk_full_markers = (
        "no space left on device",
        "not enough space on the disk",
        "there is not enough space",
        "disk full",
        "errno 28",
        "enospc",
    )
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        disk_full = True
    else:
        disk_full = any(marker in lowered for marker in disk_full_markers)
    if disk_full:
        return (
            f"{job_kind} failed because the disk ran out of space.",
            (
                "The volume holding the model cache, dataset, or output ran out of space while writing. "
                "Free up disk space (model weights, checkpoints, and cached latents are the largest consumers) and retry. "
                f"Technical detail: {detail}"
            ),
        )
    ltx_frame_markers = (
        "num_frames",
        "frame count",
        "divisible by 8",
        "multiple of 8",
        "8 + 1",
        "8n+1",
    )
    if ("ltx" in lowered or job_kind.lower().startswith("video")) and any(marker in lowered for marker in ltx_frame_markers):
        return (
            f"{job_kind} failed because LTX requires a compatible frame count.",
            (
                "LTX video frame counts must satisfy (frames - 1) being divisible by 8. "
                "Try a standard SceneWorks duration/FPS preset or shorten the clip. "
                f"Technical detail: {detail}"
            ),
        )
    peft_markers = (
        "peft backend",
        "requires peft",
        "requires the peft backend",
        "peft is required",
        "install peft",
        "no module named 'peft'",
        'no module named "peft"',
    )
    if any(marker in lowered for marker in peft_markers):
        return (
            f"{job_kind} failed because the selected preset or LoRA needs PEFT support.",
            (
                "The worker needs the PEFT backend to apply the selected preset LoRAs. "
                "For bare-metal workers, run `pip install -r apps/worker/requirements.txt`; "
                "for Docker Compose, run `docker compose build worker --no-cache`, then restart the worker and retry. "
                "You can also choose a preset without LoRAs. "
                f"Technical detail: {detail}"
            ),
        )
    tokenizer_backend_markers = (
        "protobuf",
        "protocolbuffers",
        "requires the protobuf library",
        "sentencepiece",
        "tokenization_t5",
        "t5.tokenization",
        "does not seem to have any of the loading methods",
    )
    if any(marker in lowered for marker in tokenizer_backend_markers):
        return (
            f"{job_kind} failed because the worker is missing a tokenizer backend.",
            (
                "The selected model needs the tokenizer support libraries from the worker requirements. "
                "For bare-metal workers, run `pip install -r apps/worker/requirements.txt`; "
                "for Docker Compose, run `docker compose build worker --no-cache`, then restart the worker and retry. "
                f"Technical detail: {detail}"
            ),
        )
    missing_model_markers = (
        "repo id",
        "repository not found",
        "entry not found",
        "is not a local folder",
        "couldn't connect to 'https://huggingface.co'",
        "model_index.json",
        "no file named model_index.json",
        "error no file named",
        "cannot load model",
        "missing model file",
        "missing resources",
    )
    if any(marker in lowered for marker in missing_model_markers):
        return (
            f"{job_kind} failed because required model files were not available.",
            (
                "The worker could not find or download the model files. Check that the model is installed, "
                "open Model Manager, ensure the Rust utility worker is running for downloads, and verify HF_TOKEN for gated repos. "
                f"Technical detail: {detail}"
            ),
        )
    return (f"{job_kind} failed.", detail)


def job_cancel_requested(api: ApiClient, job_id: str) -> bool:
    return bool(api.get(f"/api/v1/jobs/{job_id}")["cancelRequested"])


class JobCancelMonitor:
    """Watches a running job's cancel flag on a background thread.

    Two responsibilities:

    1. Cache the cancel state so hot paths (per-step pipe callbacks, tight
       per-frame loops) can ask "was this canceled?" without an HTTP round-trip
       on every check. ``requested`` returns the most recently polled value.
    2. Enforce the hard-stop backstop: if the cancel flag stays set for longer
       than ``settings.force_cancel_seconds`` while the job is still running, the
       cooperative path failed to interrupt a wedged native call in time — so we
       mark the job canceled and force-terminate the worker process. Its
       supervisor (Tauri on desktop, the child supervisor / Docker on the web
       build) respawns a clean worker. The worker runs one job at a time, so
       killing the process cancels exactly the stuck job (at the cost of the
       loaded model, which must reload on the next job).
    """

    def __init__(
        self,
        api: ApiClient,
        settings: WorkerSettings,
        job_id: str,
        *,
        poll_interval: float = 1.0,
        force_cancel_seconds: float | None = None,
        on_force_terminate: Callable[[], None] | None = None,
    ) -> None:
        self._api = api
        self._settings = settings
        self._job_id = job_id
        self._poll_interval = max(0.05, poll_interval)
        # Best-effort hook run just before os._exit (which skips adapter cleanup).
        # Must be filesystem-only — see _force_terminate.
        self._on_force_terminate = on_force_terminate
        deadline = (
            force_cancel_seconds
            if force_cancel_seconds is not None
            else getattr(settings, "force_cancel_seconds", 0)
        )
        self._deadline = max(0, deadline)
        self._requested = False
        self._requested_monotonic: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def requested(self) -> bool:
        """The most recently polled cancel state (cheap; no HTTP)."""
        return self._requested

    def start(self) -> "JobCancelMonitor":
        self._thread = threading.Thread(
            target=self._run,
            name=f"cancel-monitor-{self._job_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            try:
                requested = job_cancel_requested(self._api, self._job_id)
            except httpx.HTTPError as exc:
                emit({"event": "cancel_poll_failed", "jobId": self._job_id, "error": str(exc), "reportedAt": utc_now()})
                continue
            if not requested:
                continue
            self._requested = True
            if self._requested_monotonic is None:
                self._requested_monotonic = time.monotonic()
                continue
            if self._deadline and (time.monotonic() - self._requested_monotonic) >= self._deadline:
                self._force_terminate()
                return

    def _force_terminate(self) -> None:
        # Lost the race with a clean finish? Don't kill a process that's done.
        if self._stop.is_set():
            return
        emit(
            {
                "event": "cancel_force_kill",
                "jobId": self._job_id,
                "workerId": getattr(self._settings, "worker_id", None),
                "afterSeconds": self._deadline,
                "reportedAt": utc_now(),
            }
        )
        # Mark the job canceled *before* exiting so the UI resolves immediately
        # instead of waiting for a stale-worker reaper to notice the dead worker.
        try:
            update_job(
                self._api,
                self._job_id,
                {
                    "status": "canceled",
                    "stage": "canceled",
                    "progress": 1,
                    "message": f"Force-stopped: cancellation was not acknowledged within {self._deadline}s.",
                },
            )
        except httpx.HTTPError as exc:
            emit(
                {
                    "event": "cancel_force_kill_finalize_failed",
                    "jobId": self._job_id,
                    "error": str(exc),
                    "reportedAt": utc_now(),
                }
            )
        # os._exit skips the cooperative adapter.cancel()/cleanup() path, so a
        # force-killed job would otherwise orphan its temp files. Give the caller
        # one best-effort hook to reap them first. It must be filesystem-only: the
        # main thread is wedged in a native call, so touching torch/GPU here is
        # unsafe (the hook removes tracked temp files, not the resident pipeline,
        # which os._exit frees anyway).
        if self._on_force_terminate is not None:
            try:
                self._on_force_terminate()
            except Exception as exc:  # noqa: BLE001 - never block the hard stop
                emit(
                    {
                        "event": "cancel_force_kill_cleanup_failed",
                        "jobId": self._job_id,
                        "error": str(exc),
                        "reportedAt": utc_now(),
                    }
                )
        # The main thread is wedged in a native call we cannot interrupt from
        # Python, so os._exit is the only way to stop it now. It skips interpreter
        # cleanup by design — the supervisor restarts a fresh worker.
        os._exit(FORCED_CANCEL_EXIT_CODE)


@contextmanager
def job_cancel_monitor(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    *,
    on_force_terminate: Callable[[], None] | None = None,
):
    """Run a JobCancelMonitor for the duration of a job's blocking work."""
    monitor = JobCancelMonitor(api, settings, job_id, on_force_terminate=on_force_terminate)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()


def keep_job_alive(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    status: str,
    stop_event: threading.Event,
    loaded_models: LoadedModelsSource,
    peaks: dict | None = None,
) -> None:
    interval = max(5, min(settings.heartbeat_seconds, 30))
    while not stop_event.wait(interval):
        try:
            heartbeat_with_loaded_models(api, settings, status, job_id, loaded_models)
        except httpx.HTTPError as exc:
            emit({"event": "heartbeat_failed", "jobId": job_id, "error": str(exc), "reportedAt": utc_now()})
        # sc-2086 — keep the per-job peak GPU mem/load tracking current during
        # long blocking phases (e.g. video diffusion) that don't tick
        # progress() themselves. The next progress() call then folds the
        # ratcheted peaks into its payload via track_job_peaks.
        if peaks is not None:
            _sample_into_peaks(peaks, settings)


def run_blocking_job_step(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    status: str,
    callback: Callable[[CancelCallback], Any],
    *,
    loaded_models: LoadedModelsSource,
    on_force_terminate: Callable[[], None] | None = None,
    peaks: dict | None = None,
) -> Any:
    """Run a job's blocking work while keeping it alive and cancelable.

    Spawns a heartbeat thread and a JobCancelMonitor for the duration of the
    work, then invokes ``callback`` with a cached cancel predicate so adapters
    can poll cancellation cheaply (and so the monitor can force-stop the worker
    if a cancel goes unacknowledged past the deadline). ``on_force_terminate`` is
    a best-effort, filesystem-only hook run just before the hard-stop os._exit.

    ``peaks`` is the per-job sc-2086 dict — when supplied the keepalive thread
    samples GPU utilization on every heartbeat and ratchets the running max
    into the dict, so adapters whose progress() doesn't tick during their
    blocking phase still capture peak meters."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=keep_job_alive,
        args=(api, settings, job_id, status, stop_event, loaded_models, peaks),
        daemon=True,
    )
    thread.start()
    try:
        with job_cancel_monitor(api, settings, job_id, on_force_terminate=on_force_terminate) as monitor:
            return callback(monitor.requested)
    finally:
        stop_event.set()
        thread.join(timeout=1)


def is_cuda_oom(exc: BaseException) -> bool:
    """True if exc is a CUDA out-of-memory error (torch.OutOfMemoryError or a
    RuntimeError carrying 'out of memory')."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


# Exit code used when a worker child restarts itself after a CUDA OOM so the
# supervisor respawns it with a fresh (non-poisoned) CUDA context.
OOM_RESTART_EXIT_CODE = 75

# Exit code used when the worker force-terminates itself because a cancellation
# went unacknowledged past force_cancel_seconds (the hard-stop backstop). The
# supervisor respawns a clean worker.
FORCED_CANCEL_EXIT_CODE = 76


def restart_worker_after_oom(settings: WorkerSettings, job_id: str) -> None:
    """Exit the worker child after a CUDA OOM so the supervisor respawns it with a
    fresh CUDA context — releasing VRAM the poisoned context can't reclaim in place.
    Raises SystemExit, which propagates out of the claim loop and ends the process."""
    emit(
        {
            "event": "worker_restart_after_oom",
            "workerId": getattr(settings, "worker_id", None),
            "gpuId": getattr(settings, "gpu_id", None),
            "jobId": job_id,
            "reportedAt": utc_now(),
        }
    )
    raise SystemExit(OOM_RESTART_EXIT_CODE)


def should_skip_claim_low_vram(settings: WorkerSettings) -> bool:
    """True if this GPU worker should defer claiming because its card is nearly
    full — typically another process (e.g. ComfyUI) is using it — so jobs flow to
    a free GPU instead. Never gates the CPU worker or when the threshold is 0."""
    threshold = getattr(settings, "min_free_vram_mb", 0)
    # Never gate the CPU worker; never gate MPS either — it's the single
    # accelerator on a Mac (no alternate card to defer to), and gating its
    # unified "free memory" against the NVIDIA-tuned threshold would just stall
    # all work. The free-memory figure is still reported for display.
    if threshold <= 0 or settings.gpu_id in ("cpu", "mps"):
        return False
    utilization = gpu_utilization(settings.gpu_id)
    if not utilization:
        return False
    free_mb = utilization.get("memoryFreeMb")
    if free_mb is None:
        return False
    # The configured gate (24 GB default) is NVIDIA-tuned for large cards. On a
    # smaller card an absolute MB threshold can exceed what the card could ever
    # report free, deferring every claim forever with no obvious signal. Clamp it
    # to a high fraction of this card's capacity so a small idle card still
    # claims, while keeping the "defer when another tool is hogging the card"
    # intent on cards big enough to satisfy the configured value.
    total_mb = utilization.get("memoryTotalMb")
    effective = threshold
    if isinstance(total_mb, int) and total_mb > 0:
        effective = min(threshold, int(total_mb * 0.9))
    if free_mb >= effective:
        return False
    emit(
        {
            "event": "claim_skipped_low_vram",
            "gpuId": settings.gpu_id,
            "memoryFreeMb": free_mb,
            "memoryTotalMb": total_mb,
            "thresholdMb": effective,
            "configuredThresholdMb": threshold,
            "reportedAt": utc_now(),
        }
    )
    return True


def run_image_job(api: ApiClient, settings: WorkerSettings, job: dict, image_adapters: dict[str, object]) -> None:
    job_id = job["id"]
    adapter = create_image_adapter(job, image_adapters)
    # Free any other family's resident model before this one loads, so only one
    # image model stays resident at a time.
    evict_other_image_adapters(image_adapters, getattr(adapter, "id", ""))
    needs_oom_restart = False

    def adapter_loaded_models() -> list[str]:
        return loaded_models_from_adapter(adapter, job_id=job_id)

    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str, result: dict[str, Any] | None = None) -> None:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, adapter_loaded_models)
        payload = {
            "status": status,
            "stage": stage,
            "progress": value,
            "message": message,
        }
        if result is not None:
            payload["result"] = result
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(adapter, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.08, "Preparing Image Studio request.")
        progress("loading_model", "loading_model", 0.16, "Resolving image adapter target.")
        # Resolve the request + project path once per job (sc-1678); the adapter, the
        # asset writer, and source-image loading reuse these instead of each re-parsing
        # the job and re-scanning recent-projects.json.
        request = image_request_from_job(job)
        project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        # Some adapters (e.g. the Lens sidecar) hold a filesystem scratch dir to reap
        # if the cancel backstop force-kills the worker — os._exit skips the adapter's
        # own finally (sc-1719). Filesystem-only; no-op for adapters without one.
        discard_scratch = getattr(adapter, "discard_temp_outputs", None)
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: adapter.generate(
                settings=settings,
                job=job,
                request=request,
                project_path=project_path,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=adapter_loaded_models,
            on_force_terminate=(lambda: discard_scratch(job_id)) if callable(discard_scratch) else None,
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Image generation assets saved.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Image generation", exc)
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": message,
                "error": error,
            },
        )
    finally:
        # Return the (potentially tens of GB) generation activation pool to the OS
        # so an idle worker doesn't sit at peak memory. Cached models stay resident
        # for fast reuse; the just-finished job's tensors are already unreferenced.
        release_image_worker_memory()
        heartbeat(api, settings, "idle", loaded_models=loaded_models_from_adapters(image_adapters))
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


def run_vqa_job(api: ApiClient, settings: WorkerSettings, job: dict, image_adapters: dict[str, object]) -> None:
    job_id = job["id"]
    adapter = image_adapters["sensenova_u1"]
    # Free any other family's resident model before this one loads.
    evict_other_image_adapters(image_adapters, getattr(adapter, "id", ""))
    needs_oom_restart = False

    def adapter_loaded_models() -> list[str]:
        return loaded_models_from_adapter(adapter, job_id=job_id)

    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str, result: dict[str, Any] | None = None) -> None:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, adapter_loaded_models)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        if result is not None:
            payload["result"] = result
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(adapter, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.08, "Preparing visual question.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: adapter.answer_question(
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=adapter_loaded_models,
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Answer ready.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(api, job_id, {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)})
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Visual question answering", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
    finally:
        # Return the (potentially tens of GB) generation activation pool to the OS
        # so an idle worker doesn't sit at peak memory. Cached models stay resident
        # for fast reuse; the just-finished job's tensors are already unreferenced.
        release_image_worker_memory()
        heartbeat(api, settings, "idle", loaded_models=loaded_models_from_adapters(image_adapters))
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


def run_interleave_job(api: ApiClient, settings: WorkerSettings, job: dict, image_adapters: dict[str, object]) -> None:
    job_id = job["id"]
    adapter = image_adapters["sensenova_u1"]
    # Free any other family's resident model before this one loads.
    evict_other_image_adapters(image_adapters, getattr(adapter, "id", ""))
    needs_oom_restart = False

    def adapter_loaded_models() -> list[str]:
        return loaded_models_from_adapter(adapter, job_id=job_id)

    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str, result: dict[str, Any] | None = None) -> None:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, adapter_loaded_models)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        if result is not None:
            payload["result"] = result
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(adapter, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.08, "Preparing interleaved document.")
        # Resolve the request + project path once per job (sc-1678); generate_interleaved
        # threads them into input-image loading, the asset writer, and the document write
        # instead of re-scanning recent-projects.json at each step.
        request = image_request_from_job(job)
        project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: adapter.generate_interleaved(
                settings=settings,
                job=job,
                request=request,
                project_path=project_path,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=adapter_loaded_models,
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Interleaved document ready.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(api, job_id, {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)})
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Interleaved generation", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
    finally:
        # Return the (potentially tens of GB) generation activation pool to the OS
        # so an idle worker doesn't sit at peak memory. Cached models stay resident
        # for fast reuse; the just-finished job's tensors are already unreferenced.
        release_image_worker_memory()
        heartbeat(api, settings, "idle", loaded_models=loaded_models_from_adapters(image_adapters))
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


def run_video_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    adapter = create_video_adapter(job)
    job_failed = False
    needs_oom_restart = False

    def adapter_loaded_models() -> list[str]:
        return loaded_models_from_adapter(adapter, job_id=job_id)

    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, adapter_loaded_models)
        payload = {
            "status": status,
            "stage": stage,
            "progress": value,
            "message": message,
        }
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(adapter, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.06, "Preparing Video Studio request.")
        request = adapter.prepare(settings=settings, job=job)
        progress("loading_model", "loading_model", 0.14, "Resolving video adapter target.")
        adapter.ensure_models(request)
        requirements = adapter.estimate_requirements(request)
        estimated_frames = (
            requirements.get("previewFrames")
            or requirements.get("estimatedFrames")
            or requirements.get("requestedFrames")
        )
        frame_label = "preview frames" if "previewFrames" in requirements else "frames"
        estimate_message = (
            f"Estimated {estimated_frames} {frame_label} for this clip."
            if estimated_frames
            else "Estimated video generation requirements."
        )
        progress(
            "running",
            "estimating",
            0.18,
            estimate_message,
        )
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: adapter.run(
                settings=settings,
                job=job,
                request=request,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=adapter_loaded_models,
            # Reap the job's partial .tmp.mp4/.control.mp4 outputs if the hard-stop
            # backstop force-kills the worker (os._exit skips adapter cleanup).
            on_force_terminate=lambda: adapter.discard_temp_outputs(job_id),
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Video generation asset saved.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        adapter.cancel(job_id)
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        job_failed = True
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Video generation", exc)
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": message,
                "error": error,
            },
        )
    finally:
        # Free GPU memory only after the except block exits: while the handler
        # runs, the interpreter keeps the active exception (and its traceback)
        # alive, and that traceback references the OOM tensors — so an
        # empty_cache() inside the handler reclaims nothing and the memory stays
        # held until the worker restarts.
        if job_failed:
            adapter.cleanup(job_id)
        heartbeat(api, settings, "idle", loaded_models=adapter_loaded_models())
        # A CUDA OOM can leave the allocator/context unable to reclaim VRAM in
        # place; restart the child so the supervisor gives it a fresh context.
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


# The supported training plan version and shared plan validation/summary helpers
# live in training_adapters; the kernels there consume the Rust-resolved plan.


def run_person_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    """Run a real person_detect or person_track job in the GPU worker.

    Detection and tracking now use model-backed adapters (YOLO/ByteTrack with
    SAM2 masks) instead of the Rust procedural placeholders, so completed jobs
    report active detection/tracking metadata and content-derived results.
    """
    job_id = job["id"]
    job_type = job["type"]
    needs_oom_restart = False
    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(None, settings))
        update_job(api, job_id, payload)

    runner = run_person_detect if job_type == "person_detect" else run_person_track
    done_message = "Person candidates detected." if job_type == "person_detect" else "Reusable person track saved."

    try:
        progress("preparing", "preparing", 0.06, "Preparing person analysis.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: runner(
                settings,
                job,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=lambda: [],
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {"status": "completed", "stage": "completed", "progress": 1, "message": done_message, "result": result},
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)},
        )
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        kind = "Person detection" if job_type == "person_detect" else "Person tracking"
        message, error = friendly_failure(kind, exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
    finally:
        heartbeat(api, settings, "idle")
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


def run_pose_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    """Run a pose_detect job: DWPose whole-body keypoints from one or more photos.

    onnxruntime-backed (rtmlib), so no torch model is loaded — the worker advertises
    pose_detect only when the backend is installed (worker_capabilities). Returns one
    pose candidate per detected person plus a rendered skeleton preview per pose.
    """
    job_id = job["id"]
    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(None, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.06, "Preparing pose detection.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: run_pose_detect(
                settings,
                job,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=lambda: [],
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {"status": "completed", "stage": "completed", "progress": 1,
             "message": "Pose candidates detected.", "result": result},
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)},
        )
    except Exception as exc:
        message, error = friendly_failure("Pose detection", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
    finally:
        heartbeat(api, settings, "idle")


def run_upscale_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    """Run an image_upscale job: Real-ESRGAN / AuraSR on one existing asset.

    Torch-backed (the engine is loaded lazily by run_image_upscale), so the worker
    advertises image_upscale only when the inference backend is installed
    (worker_capabilities). Writes one child asset with lineage to the source.
    """
    job_id = job["id"]
    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(None, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.06, "Preparing image upscale.")
        project_id = str(job.get("payload", {}).get("projectId") or "")
        project_path = find_project_path(settings.data_dir / "recent-projects.json", project_id)
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: run_image_upscale(
                settings,
                job,
                project_path=project_path,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=lambda: [],
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {"status": "completed", "stage": "completed", "progress": 1,
             "message": "Upscaled image saved.", "result": result},
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)},
        )
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Image upscale", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
        if needs_oom_restart:
            release_image_worker_memory()
    finally:
        heartbeat(api, settings, "idle")


def run_detail_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    """Run an image_detail job: tile-ControlNet detail refine on one existing asset.

    Torch-backed (the SDXL/RealVisXL img2img pipe + tile ControlNet are loaded lazily
    by run_image_detail), so the worker advertises image_detail only when the
    inference backend is installed (worker_capabilities). Writes one child asset with
    lineage to the source. Mirrors run_upscale_job (sc-2431); composes after it.
    """
    job_id = job["id"]
    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(None, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.06, "Preparing detail enhancement.")
        project_id = str(job.get("payload", {}).get("projectId") or "")
        project_path = find_project_path(settings.data_dir / "recent-projects.json", project_id)
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: run_image_detail(
                settings,
                job,
                project_path=project_path,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=lambda: [],
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {"status": "completed", "stage": "completed", "progress": 1,
             "message": "Detail-enhanced image saved.", "result": result},
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)},
        )
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Detail enhancement", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
        if needs_oom_restart:
            release_image_worker_memory()
    finally:
        heartbeat(api, settings, "idle")


def run_lora_train_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    """Run a lora_train job: validate the plan, then either report a dry-run
    summary or execute the real training kernel.

    Dry-run (the default) validates the Rust-resolved plan and its dataset inputs
    and reports what a real run would produce — no model load, no backend needed
    (so a GPU worker without torch can still validate). A real run loads the
    target's narrow execution kernel and trains an actual LoRA, reporting staged
    progress and honoring cancellation.
    """
    payload = job.get("payload") or {}
    if bool(payload.get("dryRun", True)):
        _run_lora_train_dry_run(api, settings, job, payload)
    else:
        _run_lora_train_execution(api, settings, job, payload)


def _run_lora_train_dry_run(api: ApiClient, settings: WorkerSettings, job: dict, payload: dict) -> None:
    job_id = job["id"]
    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        update_payload = {"status": status, "stage": stage, "progress": value, "message": message}
        track_job_peaks(update_payload, peaks, settings, backend=adapter_backend(None, settings))
        update_job(api, job_id, update_payload)

    try:
        progress("preparing", "preparing", 0.1, "Validating training plan.")
        plan = payload.get("plan")
        items = validate_training_plan(plan, require_images=True)
        progress("running", "running", 0.5, f"Checked {len(items)} dataset item(s).")
        summary = dry_run_training_summary(plan, dry_run=True)
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": f"Dry run validated {len(items)} dataset item(s); training plan is ready.",
                "result": summary,
            },
        )
    except Exception as exc:  # noqa: BLE001 - report any validation failure cleanly
        message, error = friendly_failure("Training dry run", exc)
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": message,
                "error": error,
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def _run_lora_train_execution(api: ApiClient, settings: WorkerSettings, job: dict, payload: dict) -> None:
    job_id = job["id"]
    trainer_holder: dict[str, Any] = {"trainer": None}
    needs_oom_restart = False

    def trainer_loaded_models() -> list[str]:
        trainer = trainer_holder["trainer"]
        return trainer.loaded_models() if trainer is not None else []

    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str, result: dict | None = None) -> None:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, trainer_loaded_models)
        update_payload = {"status": status, "stage": stage, "progress": value, "message": message}
        if result is not None:
            update_payload["result"] = result
        track_job_peaks(update_payload, peaks, settings, backend=adapter_backend(trainer_holder["trainer"], settings))
        update_job(api, job_id, update_payload)

    try:
        plan = payload.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("Training job payload is missing a resolved plan.")
        kernel_id = (plan.get("target") or {}).get("kernel")
        trainer = create_training_kernel(kernel_id)
        trainer_holder["trainer"] = trainer
        # The Lens trainer holds a sidecar scratch dir to reap if the cancel backstop
        # force-kills the worker (os._exit skips train()'s finally) — sc-1719.
        # Filesystem-only; no-op for kernels without a scratch dir.
        discard_scratch = getattr(trainer, "discard_temp_outputs", None)
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: trainer.train(
                settings=settings,
                plan=plan,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=trainer_loaded_models,
            on_force_terminate=(lambda: discard_scratch(job_id)) if callable(discard_scratch) else None,
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": f"Trained LoRA saved as {result.get('fileName')}.",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(
            api,
            job_id,
            {
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": str(exc),
            },
        )
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("LoRA training", exc)
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": message,
                "error": error,
            },
        )
    finally:
        heartbeat(api, settings, "idle")
        # A CUDA OOM can leave the allocator/context unable to reclaim VRAM in
        # place; restart the child so the supervisor gives it a fresh context.
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


def run_training_caption_worker_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    needs_oom_restart = False
    peaks: dict[str, float] = {}

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        payload = {"status": status, "stage": stage, "progress": value, "message": message}
        track_job_peaks(payload, peaks, settings, backend=adapter_backend(None, settings))
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.04, "Preparing training caption job.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda cancel: run_training_caption_job(
                api=api,
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=cancel,
            ),
            loaded_models=lambda: [],
            peaks=peaks,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": f"Created captions for {result.get('captionedItemCount', 0)} training item(s).",
                "result": result,
            },
        )
    except InterruptedError as exc:
        update_job(api, job_id, {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)})
    except Exception as exc:
        needs_oom_restart = is_cuda_oom(exc)
        message, error = friendly_failure("Training captioning", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
    finally:
        heartbeat(api, settings, "idle")
        if needs_oom_restart:
            restart_worker_after_oom(settings, job_id)


def run_prompt_refine_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    payload = job.get("payload") or {}
    prompt = (payload.get("prompt") or "").strip()
    refiner = PromptRefiner(
        model_name_or_path=payload.get("model") or getattr(settings, "prompt_refine_model", "") or "",
        gpu_id=getattr(settings, "gpu_id", "auto"),
        max_new_tokens=getattr(settings, "prompt_refine_max_new_tokens", 512),
    )
    try:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, refiner.loaded_models)
        update_job(
            api,
            job_id,
            {"status": "loading_model", "stage": "loading_model", "progress": 0.1, "message": "Loading refinement model."},
        )
        refiner.load()
        update_job(
            api,
            job_id,
            {"status": "running", "stage": "running", "progress": 0.4, "message": "Refining prompt…"},
        )
        refined = refiner.refine(prompt, guide=payload.get("guide"), workflow=payload.get("workflow"))
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Prompt refined.",
                "result": {"originalPrompt": prompt, "refinedPrompt": refined},
            },
        )
    except InterruptedError as exc:
        update_job(api, job_id, {"status": "canceled", "stage": "canceled", "progress": 1, "message": str(exc)})
    except PromptRefineError as exc:
        # Expected, user-facing failures (model couldn't load, empty rewrite) carry
        # a clear message already.
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": str(exc), "error": str(exc)},
        )
    except Exception as exc:
        message, error = friendly_failure("Prompt refinement", exc)
        update_job(
            api,
            job_id,
            {"status": "failed", "stage": "failed", "progress": 1, "message": message, "error": error},
        )
    finally:
        heartbeat(api, settings, "idle")


def run_worker_loop(settings: WorkerSettings) -> None:
    gpu = discover_gpu(settings.gpu_id)
    api = ApiClient(settings)
    image_adapters: dict[str, object] = {
        "procedural_preview": ProceduralImageAdapter(),
        "qwen_image": QwenImageAdapter(),
        "mlx_qwen": MlxQwenAdapter(),
        "z_image_diffusers": ZImageDiffusersAdapter(),
        "mlx_z_image": MlxZImageAdapter(),
        "lens_turbo": LensTurboAdapter(),
        "sensenova_u1": SenseNovaU1Adapter(),
        "flux_diffusers": FluxDiffusersAdapter(),
        "mlx_flux": MlxFluxAdapter(),
        # FLUX.2-klein family (sc-2164). Registered here so create_image_adapter's
        # `adapters.get("mlx_flux2")` dispatch resolves in the real runtime; the
        # earlier omission made every FLUX.2 job crash with "'NoneType' object
        # has no attribute 'generate'" (sc-2203).
        "mlx_flux2": MlxFlux2Adapter(),
        "kolors_diffusers": KolorsDiffusersAdapter(),
        "sdxl_diffusers": SdxlDiffusersAdapter(),
        "chroma_diffusers": ChromaDiffusersAdapter(),
        "instantid_sdxl": InstantIDAdapter(),
        "pulid_flux": PuLIDFluxAdapter(),
    }
    max_registration_attempts = 20

    for attempt in range(1, max_registration_attempts + 1):
        try:
            register_worker(api, settings, gpu, loaded_models_from_adapters(image_adapters))
            break
        except httpx.HTTPError as exc:
            delay = min(30, settings.poll_seconds * (2 ** (attempt - 1)))
            emit(
                {
                    "event": "register_failed",
                    "attempt": attempt,
                    "maxAttempts": max_registration_attempts,
                    "retryInSeconds": delay,
                    "error": str(exc),
                    "reportedAt": utc_now(),
                }
            )
            if attempt == max_registration_attempts:
                raise RuntimeError(f"Worker registration failed after {max_registration_attempts} attempts.") from exc
            time.sleep(delay)

    while True:
        try:
            heartbeat(api, settings, "idle", loaded_models=loaded_models_from_adapters(image_adapters))
            if should_skip_claim_low_vram(settings):
                time.sleep(settings.poll_seconds)
                continue
            claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})
            job = claimed.get("job")
            if job is None:
                time.sleep(settings.poll_seconds)
                continue

            emit({"event": "claimed", "jobId": job["id"], "gpuId": job["assignedGpu"], "reportedAt": utc_now()})
            if job["type"] in IMAGE_JOB_TYPES:
                run_image_job(api, settings, job, image_adapters)
            elif job["type"] in IMAGE_UPSCALE_JOB_TYPES:
                run_upscale_job(api, settings, job)
            elif job["type"] in IMAGE_DETAIL_JOB_TYPES:
                run_detail_job(api, settings, job)
            elif job["type"] in VQA_JOB_TYPES:
                run_vqa_job(api, settings, job, image_adapters)
            elif job["type"] in INTERLEAVE_JOB_TYPES:
                run_interleave_job(api, settings, job, image_adapters)
            elif job["type"] in VIDEO_JOB_TYPES:
                run_video_job(api, settings, job)
            elif job["type"] in PERSON_JOB_TYPES:
                run_person_job(api, settings, job)
            elif job["type"] in POSE_JOB_TYPES:
                run_pose_job(api, settings, job)
            elif job["type"] in TRAINING_JOB_TYPES:
                run_lora_train_job(api, settings, job)
            elif job["type"] in CAPTION_JOB_TYPES:
                run_training_caption_worker_job(api, settings, job)
            elif job["type"] in PROMPT_REFINE_JOB_TYPES:
                run_prompt_refine_job(api, settings, job)
            else:
                update_job(
                    api,
                    job["id"],
                    {
                        "status": "failed",
                        "stage": "failed",
                        "progress": 1,
                        "message": "No adapter exists for this job type yet.",
                        "error": f"Unsupported job type: {job['type']}",
                    },
                )
        except httpx.HTTPError as exc:
            emit({"event": "api_error", "error": str(exc), "reportedAt": utc_now()})
            time.sleep(settings.poll_seconds)


def child_environment(settings: WorkerSettings, *, worker_id: str, gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["SCENEWORKS_WORKER_CHILD"] = "1"
    env["SCENEWORKS_WORKER_ID"] = worker_id
    env["SCENEWORKS_GPU_ID"] = gpu_id
    if gpu_id == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu_id == "mps":
        # MPS is not a CUDA device; leave CUDA_VISIBLE_DEVICES untouched rather
        # than pinning it to a bogus "mps" value (sc-1335).
        pass
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    return env


def start_child_worker(settings: WorkerSettings, *, worker_id: str, gpu_id: str) -> subprocess.Popen:
    emit({"event": "starting_worker", "workerId": worker_id, "gpuId": gpu_id, "reportedAt": utc_now()})
    return subprocess.Popen(
        [sys.executable, "-m", "scene_worker"],
        env=child_environment(settings, worker_id=worker_id, gpu_id=gpu_id),
    )


def supervise_auto_workers(settings: WorkerSettings) -> None:
    gpus = discover_gpus()
    if not gpus:
        run_worker_loop(settings.for_worker(worker_id=cpu_worker_id(settings.worker_id), gpu_id="cpu"))
        return

    worker_specs = [(gpu_worker_id(settings.worker_id, gpu["id"]), gpu["id"]) for gpu in gpus]
    worker_specs.append((cpu_worker_id(settings.worker_id), "cpu"))
    children = {
        worker_id: start_child_worker(settings, worker_id=worker_id, gpu_id=gpu_id)
        for worker_id, gpu_id in worker_specs
    }
    shutting_down = False

    def stop_children(_signum: int, _frame: object) -> None:
        nonlocal shutting_down
        shutting_down = True
        for child in children.values():
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    while True:
        for worker_id, child in list(children.items()):
            exit_code = child.poll()
            if exit_code is None:
                continue
            if shutting_down:
                children.pop(worker_id)
                continue
            gpu_id = next(gpu_id for candidate_worker_id, gpu_id in worker_specs if candidate_worker_id == worker_id)
            emit(
                {
                    "event": "worker_exited",
                    "workerId": worker_id,
                    "gpuId": gpu_id,
                    "exitCode": exit_code,
                    "restartInSeconds": settings.poll_seconds,
                    "reportedAt": utc_now(),
                }
            )
            time.sleep(settings.poll_seconds)
            children[worker_id] = start_child_worker(settings, worker_id=worker_id, gpu_id=gpu_id)

        if shutting_down and not children:
            return
        time.sleep(1)


def run_check(settings: WorkerSettings) -> None:
    gpu = discover_gpu(settings.gpu_id)
    capabilities = worker_capabilities(gpu)
    emit(
        {
            "event": "worker_check",
            "workerId": settings.worker_id,
            "gpu": gpu,
            "capabilities": capabilities,
            "jobTypes": [
                job_type for job_type in ALL_JOB_TYPES if job_type in capabilities
            ],
            "supportedJobTypes": list(ALL_JOB_TYPES),
            "reportedAt": utc_now(),
        }
    )


def _force_utf8_stdio() -> None:
    """Force stdout/stderr to UTF-8 so the worker survives non-ASCII library output.

    On Windows the worker's stdout/stderr default to the locale code page (cp1252
    for en-US), so any dependency that ``print()``s a non-Latin-1 character raises
    UnicodeEncodeError and kills the process. transformers' ``@auto_docstring``
    decorator unconditionally prints an "undocumented parameters" developer notice
    containing a 🚨 emoji while decorating the vendored SenseNova-U1 model classes
    (and would for other models too), which crashed ``import sensenova_u1`` on
    Windows. UTF-8 is already the default on Linux/macOS, so this is a no-op there.
    The JSON events the worker emits on stdout are ASCII (json.dumps ensure_ascii),
    so widening the encoding never changes the IPC bytes.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # Access denied still means the process exists; invalid parameter is
            # the usual Windows error for a PID that no longer resolves.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — still alive for our purposes.
        return True
    except OSError:
        return False
    return True


def start_parent_death_watchdog() -> None:
    """Terminate this process when the launching desktop shell goes away.

    macOS has no ``PR_SET_PDEATHSIG``, so a sidecar whose parent force-quits or
    crashes orphans to launchd (PPID=1) and keeps its multi-GB model resident
    indefinitely — the desktop only tears sidecars down on a clean
    ``RunEvent::ExitRequested``. When ``SCENEWORKS_PARENT_PID`` is set (only the
    desktop shell sets it), a daemon thread watches that pid and signals this
    process to stop once it disappears, covering force-quit, crash, and any exit
    path that skips graceful teardown. Unset in Docker/server runs, so it is a
    no-op there.

    SIGTERM is sent first so a supervisor parent runs its ``stop_children``
    handler; a leaf worker has no SIGTERM handler, so the default disposition
    terminates it immediately (releasing the loaded model) even mid-inference,
    since signal delivery is handled by the kernel regardless of the blocked
    thread. SIGKILL escalates if the process lingers past the grace window.
    """
    raw = os.getenv("SCENEWORKS_PARENT_PID", "").strip()
    if not raw:
        return
    try:
        parent_pid = int(raw)
    except ValueError:
        return
    if parent_pid <= 1:
        return

    def watch() -> None:
        while _pid_alive(parent_pid):
            time.sleep(3)
        emit({"event": "parent_exited", "parentPid": parent_pid, "pid": os.getpid(), "reportedAt": utc_now()})
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except OSError:
            pass
        time.sleep(5)
        try:
            os.kill(os.getpid(), signal.SIGKILL)
        except OSError:
            pass

    threading.Thread(target=watch, name="parent-death-watchdog", daemon=True).start()


def main(argv: list[str] | None = None) -> None:
    _force_utf8_stdio()
    args = list(sys.argv[1:] if argv is None else argv)
    settings = WorkerSettings()
    if args == ["--check"]:
        run_check(settings)
        return
    if args:
        raise SystemExit(f"Unsupported scene_worker arguments: {' '.join(args)}")
    start_parent_death_watchdog()
    if settings.gpu_id == "auto" and os.getenv("SCENEWORKS_WORKER_CHILD") != "1":
        supervise_auto_workers(settings)
        return
    run_worker_loop(settings)
