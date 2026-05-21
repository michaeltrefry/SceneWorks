from __future__ import annotations

from datetime import UTC, datetime
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import httpx

from .gpu import cpu_worker_id, discover_gpu, discover_gpus, gpu_utilization, gpu_worker_id
from .image_adapters import ProceduralImageAdapter, QwenImageAdapter, ZImageDiffusersAdapter, create_image_adapter
from .settings import WorkerSettings
from .video_adapters import create_video_adapter


LoadedModelsSource = Callable[[], list[str]] | None
IMAGE_JOB_TYPES = ("image_generate", "image_edit")
VIDEO_JOB_TYPES = ("video_generate", "video_extend", "video_bridge", "person_replace")
# Keep GPU-required job types in sync with
# crates/sceneworks-core/src/jobs_store.rs::job_requires_gpu and
# apps/web/src/screens/QueueScreen.jsx::gpuRequiredJobTypes.
SUPPORTED_JOB_TYPES = IMAGE_JOB_TYPES + VIDEO_JOB_TYPES
# Training is GPU-required like generation, but a GPU worker advertises it even
# without an inference backend: the story 1416 dry-run only validates a Rust-
# resolved plan (no torch needed). Real execution is gated per platform in 1417.
TRAINING_JOB_TYPES = ("lora_train",)


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    return sorted(capabilities)


def torch_inference_backend_available() -> bool:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    try:
        if bool(torch.cuda.is_available()):
            return True
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


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
    emit({"event": "registered", "worker": worker, "reportedAt": now()})


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


def resolve_loaded_models(source: LoadedModelsSource, *, job_id: str | None = None) -> list[str]:
    if source is None:
        return []
    try:
        return source()
    except Exception as exc:
        payload = {"event": "loaded_models_failed", "error": str(exc), "reportedAt": now()}
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
        "sentencepiece",
        "tokenization_t5",
        "t5.tokenization",
        "does not seem to have any of the loading methods",
    )
    if any(marker in lowered for marker in tokenizer_backend_markers):
        return (
            f"{job_kind} failed because the worker is missing a tokenizer backend.",
            (
                "The selected video model needs the SentencePiece tokenizer runtime. "
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


def keep_job_alive(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    status: str,
    stop_event: threading.Event,
    loaded_models: LoadedModelsSource,
) -> None:
    interval = max(5, min(settings.heartbeat_seconds, 30))
    while not stop_event.wait(interval):
        try:
            heartbeat_with_loaded_models(api, settings, status, job_id, loaded_models)
        except httpx.HTTPError as exc:
            emit({"event": "heartbeat_failed", "jobId": job_id, "error": str(exc), "reportedAt": now()})


def run_blocking_job_step(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    status: str,
    callback: Any,
    *,
    loaded_models: LoadedModelsSource,
) -> Any:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=keep_job_alive,
        args=(api, settings, job_id, status, stop_event, loaded_models),
        daemon=True,
    )
    thread.start()
    try:
        return callback()
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
            "reportedAt": now(),
        }
    )
    raise SystemExit(OOM_RESTART_EXIT_CODE)


def should_skip_claim_low_vram(settings: WorkerSettings) -> bool:
    """True if this GPU worker should defer claiming because its card is nearly
    full — typically another process (e.g. ComfyUI) is using it — so jobs flow to
    a free GPU instead. Never gates the CPU worker or when the threshold is 0."""
    threshold = getattr(settings, "min_free_vram_mb", 0)
    if threshold <= 0 or settings.gpu_id == "cpu":
        return False
    utilization = gpu_utilization(settings.gpu_id)
    free_mb = utilization.get("memoryFreeMb") if utilization else None
    if free_mb is None or free_mb >= threshold:
        return False
    emit(
        {
            "event": "claim_skipped_low_vram",
            "gpuId": settings.gpu_id,
            "memoryFreeMb": free_mb,
            "thresholdMb": threshold,
            "reportedAt": now(),
        }
    )
    return True


def run_image_job(api: ApiClient, settings: WorkerSettings, job: dict, image_adapters: dict[str, object]) -> None:
    job_id = job["id"]
    adapter = create_image_adapter(job, image_adapters)
    needs_oom_restart = False

    def adapter_loaded_models() -> list[str]:
        return loaded_models_from_adapter(adapter, job_id=job_id)

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
        update_job(api, job_id, payload)

    try:
        progress("preparing", "preparing", 0.08, "Preparing Image Studio request.")
        progress("loading_model", "loading_model", 0.16, "Resolving image adapter target.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda: adapter.generate(
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=lambda: job_cancel_requested(api, job_id),
            ),
            loaded_models=adapter_loaded_models,
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

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat_with_loaded_models(api, settings, "busy", job_id, adapter_loaded_models)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": value,
                "message": message,
            },
        )

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
            lambda: adapter.run(
                settings=settings,
                job=job,
                request=request,
                progress=progress,
                cancel_requested=lambda: job_cancel_requested(api, job_id),
            ),
            loaded_models=adapter_loaded_models,
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


# Highest training plan version this worker understands. Plans are resolved in
# Rust (crates/sceneworks-core/src/training.rs::TRAINING_PLAN_VERSION); the worker
# rejects any version it cannot interpret rather than guessing.
SUPPORTED_TRAINING_PLAN_VERSION = 1


def run_lora_train_dry_run_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    """Validate a Rust-resolved training plan and complete with a dry-run summary.

    This is the story 1416 stub: it claims a lora_train job, checks the embedded
    plan and that its dataset inputs exist on the worker, and reports what a real
    run would produce — without loading a model or training. Story 1417 replaces
    the body with the narrow Z-Image LoRA execution kernel.
    """
    job_id = job["id"]

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {"status": status, "stage": stage, "progress": value, "message": message},
        )

    try:
        progress("preparing", "preparing", 0.1, "Validating training plan.")
        payload = job.get("payload") or {}
        # Real training execution does not exist yet; never let a non-dry-run job
        # report success without producing weights (the API also rejects this).
        if not payload.get("dryRun", True):
            raise ValueError(
                "Real LoRA training execution is not available yet; this worker only validates dry-run plans."
            )
        plan = payload.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("Training job payload is missing a resolved plan.")
        plan_version = plan.get("planVersion")
        if plan_version != SUPPORTED_TRAINING_PLAN_VERSION:
            raise ValueError(
                f"Unsupported training plan version {plan_version!r}; this worker "
                f"understands version {SUPPORTED_TRAINING_PLAN_VERSION}."
            )

        dataset = plan.get("dataset") or {}
        items = dataset.get("items") or []
        if not items:
            raise ValueError("Training plan dataset has no items to train on.")

        progress("running", "running", 0.5, f"Checking {len(items)} dataset item(s).")
        missing = [
            item.get("imagePath")
            for item in items
            if not (item.get("imagePath") and os.path.exists(item["imagePath"]))
        ]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} dataset image(s) are missing on the worker, e.g. {preview}."
            )

        output = plan.get("output") or {}
        target = plan.get("target") or {}
        base_model_path = target.get("baseModelPath")
        summary = {
            "mode": "dry_run",
            "validated": True,
            "dryRun": bool(payload.get("dryRun", True)),
            "datasetItemCount": len(items),
            "datasetId": dataset.get("datasetId"),
            "datasetVersion": dataset.get("datasetVersion"),
            "targetId": target.get("targetId"),
            "kernel": target.get("kernel"),
            "loraId": output.get("loraId"),
            "outputDir": output.get("outputDir"),
            "fileName": output.get("fileName"),
            "baseModelPath": base_model_path,
            "baseModelInstalled": bool(base_model_path and os.path.exists(base_model_path)),
            "planVersion": plan_version,
            "completedAt": now(),
        }
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


def run_worker_loop(settings: WorkerSettings) -> None:
    gpu = discover_gpu(settings.gpu_id)
    api = ApiClient(settings)
    image_adapters: dict[str, object] = {
        "procedural_preview": ProceduralImageAdapter(),
        "qwen_image": QwenImageAdapter(),
        "z_image_diffusers": ZImageDiffusersAdapter(),
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
                    "reportedAt": now(),
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

            emit({"event": "claimed", "jobId": job["id"], "gpuId": job["assignedGpu"], "reportedAt": now()})
            if job["type"] in IMAGE_JOB_TYPES:
                run_image_job(api, settings, job, image_adapters)
            elif job["type"] in VIDEO_JOB_TYPES:
                run_video_job(api, settings, job)
            elif job["type"] in TRAINING_JOB_TYPES:
                run_lora_train_dry_run_job(api, settings, job)
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
            emit({"event": "api_error", "error": str(exc), "reportedAt": now()})
            time.sleep(settings.poll_seconds)


def child_environment(settings: WorkerSettings, *, worker_id: str, gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["SCENEWORKS_WORKER_CHILD"] = "1"
    env["SCENEWORKS_WORKER_ID"] = worker_id
    env["SCENEWORKS_GPU_ID"] = gpu_id
    if gpu_id == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    return env


def start_child_worker(settings: WorkerSettings, *, worker_id: str, gpu_id: str) -> subprocess.Popen:
    emit({"event": "starting_worker", "workerId": worker_id, "gpuId": gpu_id, "reportedAt": now()})
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
                    "reportedAt": now(),
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
            "jobTypes": [job_type for job_type in SUPPORTED_JOB_TYPES if job_type in capabilities],
            "supportedJobTypes": list(SUPPORTED_JOB_TYPES),
            "reportedAt": now(),
        }
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    settings = WorkerSettings()
    if args == ["--check"]:
        run_check(settings)
        return
    if args:
        raise SystemExit(f"Unsupported scene_worker arguments: {' '.join(args)}")
    if settings.gpu_id == "auto" and os.getenv("SCENEWORKS_WORKER_CHILD") != "1":
        supervise_auto_workers(settings)
        return
    run_worker_loop(settings)
