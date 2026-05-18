from __future__ import annotations

from datetime import UTC, datetime
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import httpx
from sceneworks_shared import find_project_path

from .gpu import cpu_worker_id, discover_gpu, discover_gpus, gpu_worker_id
from .image_adapters import ProceduralImageAdapter, ZImageDiffusersAdapter, create_image_adapter
from .settings import WorkerSettings
from .timeline_exporter import run_timeline_export
from .video_adapters import ProceduralVideoAdapter, run_frame_extract, run_person_detect, run_person_track


LoadedModelsSource = Callable[[], list[str]] | None
LORA_MANIFEST_LOCK = threading.Lock()


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
    utility_jobs_enabled = os.getenv("SCENEWORKS_UTILITY_JOBS", "1").strip() != "0"
    legacy_model_lora_jobs_enabled = os.getenv("SCENEWORKS_LEGACY_MODEL_LORA_JOBS", "0").strip() != "0"
    legacy_ffmpeg_jobs_enabled = os.getenv("SCENEWORKS_LEGACY_FFMPEG_JOBS", "0").strip() != "0"
    capabilities = set(gpu["capabilities"])
    if utility_jobs_enabled:
        capabilities |= {"person_detect", "person_track"}
        if legacy_ffmpeg_jobs_enabled:
            capabilities |= {"timeline_export", "frame_extract"}
        if legacy_model_lora_jobs_enabled:
            capabilities |= {"model_download", "lora_import"}
    if "cpu" not in gpu_capabilities and "gpu" in gpu_capabilities:
        capabilities |= {"image_generate", "image_edit", "video_generate", "video_extend", "video_bridge", "person_replace"}
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
    worker = api.post("/api/v1/workers/register", payload)
    emit({"event": "registered", "worker": worker, "reportedAt": now()})


def heartbeat(
    api: ApiClient,
    settings: WorkerSettings,
    status: str,
    current_job_id: str | None = None,
    loaded_models: list[str] | None = None,
) -> None:
    api.post(
        f"/api/v1/workers/{settings.worker_id}/heartbeat",
        {"status": status, "currentJobId": current_job_id, "loadedModels": loaded_models or []},
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
    missing_model_markers = (
        "repo id",
        "repository not found",
        "is not a local folder",
        "couldn't connect to 'https://huggingface.co'",
        "no file named model_index.json",
        "error no file named",
        "cannot load model",
        "missing model file",
    )
    if any(marker in lowered for marker in missing_model_markers):
        return (
            f"{job_kind} failed because required model files were not available.",
            (
                "The worker could not find or download the model files. Check that the model is installed, "
                "open Model Manager or queue a model_download job to install it, and verify HF_TOKEN for gated repos. "
                f"Technical detail: {detail}"
            ),
        )
    return (f"{job_kind} failed.", detail)


def job_cancel_requested(api: ApiClient, job_id: str) -> bool:
    return bool(api.get(f"/api/v1/jobs/{job_id}")["cancelRequested"])


def safe_download_dir(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "._-" else "__" for char in value)
    return normalized.strip("_") or "download"


def strip_jsonc_comments(value: str) -> str:
    output = []
    index = 0
    in_string = False
    escaped = False
    while index < len(value):
        char = value[index]
        next_char = value[index + 1] if index + 1 < len(value) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(value) and value[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(value) and not (value[index] == "*" and value[index + 1] == "/"):
                index += 1
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def resolve_lora_import_target(settings: WorkerSettings, payload: dict[str, Any], fallback_target: Path) -> Path:
    target = Path(payload.get("targetDir") or fallback_target).expanduser().resolve()
    allowed_roots = [(settings.data_dir / "loras").resolve()]
    project_id = payload.get("projectId")
    if project_id:
        project_path = find_project_path(settings.data_dir / "recent-projects.json", project_id)
        allowed_roots.append((project_path / "loras" / "imports").resolve())
    for root in allowed_roots:
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue
    raise ValueError("LoRA import targetDir must be inside app-managed data/loras or project/loras/imports")


def lora_manifest_target(settings: WorkerSettings, payload: dict[str, Any]) -> Path | None:
    manifest_path_text = payload.get("manifestPath")
    manifest_entry = payload.get("manifestEntry")
    if not manifest_path_text or not isinstance(manifest_entry, dict):
        return None
    manifest_path = Path(manifest_path_text).expanduser().resolve()
    allowed = [(settings.config_dir / "manifests" / "user.loras.jsonc").resolve()]
    project_id = payload.get("projectId")
    if project_id:
        project_path = find_project_path(settings.data_dir / "recent-projects.json", project_id)
        allowed.append((project_path / "loras" / "manifest.jsonc").resolve())
    if manifest_path not in allowed:
        raise ValueError("LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest")
    return manifest_path


def upsert_lora_manifest_entry(path: Path, entry: dict[str, Any]) -> None:
    with LORA_MANIFEST_LOCK:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.loads(strip_jsonc_comments(handle.read()))
        else:
            payload = {"schemaVersion": 1, "loras": []}
        payload.setdefault("schemaVersion", 1)
        lora_id = entry["id"]
        loras = []
        found = False
        for item in payload.get("loras", []):
            if item.get("id") == lora_id:
                found = True
                loras.append({**item, **entry, "createdAt": item.get("createdAt", entry.get("createdAt"))})
            else:
                loras.append(item)
        if not found:
            loras.append(entry)
        payload["loras"] = loras
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
                tmp_path = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            tmp_path.replace(path)
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def write_model_install_marker(target_dir: Path, payload: dict, repo: str, job_id: str) -> None:
    marker = {
        "repo": repo,
        "modelId": payload.get("modelId"),
        "modelName": payload.get("modelName"),
        "jobId": job_id,
        "completedAt": now(),
    }
    with (target_dir / ".sceneworks-download-complete.json").open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file() or item.name == ".sceneworks-download-complete.json":
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def allow_pattern_matches(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch(path, pattern) for pattern in patterns)


def estimate_huggingface_repo_size(repo: str, files: list[str] | None = None) -> int | None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception as exc:
        emit({"event": "download_size_unavailable", "repo": repo, "error": str(exc), "reportedAt": now()})
        return None

    patterns = files or []
    total = 0
    found_size = False
    for sibling in getattr(info, "siblings", []):
        filename = getattr(sibling, "rfilename", "")
        if not allow_pattern_matches(filename, patterns):
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        found_size = True
        total += int(size)
    return total if found_size else None


def download_progress_payload(
    repo: str,
    downloaded_bytes: int,
    total_bytes: int | None,
    *,
    started_bytes: int,
    started_at: float,
) -> dict[str, Any]:
    elapsed_seconds = max(0.001, time.monotonic() - started_at)
    transferred_bytes = max(0, downloaded_bytes - started_bytes)
    rate = transferred_bytes / elapsed_seconds
    eta_seconds = None
    if total_bytes and rate > 0:
        eta_seconds = max(0, (max(0, total_bytes - downloaded_bytes)) / rate)

    if total_bytes:
        ratio = min(max(downloaded_bytes / total_bytes, 0), 1)
        progress = 0.1 + ratio * 0.85
        remaining_bytes = max(0, total_bytes - downloaded_bytes)
        message = (
            f"Downloading {repo}: {format_bytes(downloaded_bytes)} of {format_bytes(total_bytes)} "
            f"({format_bytes(remaining_bytes)} left)."
        )
    else:
        progress = 0.1
        message = f"Downloading {repo}: {format_bytes(downloaded_bytes)} written."

    return {
        "status": "downloading",
        "stage": "downloading",
        "progress": progress,
        "message": message,
        "etaSeconds": eta_seconds,
    }


def snapshot_huggingface_repo(repo: str, target_dir: Path, files: list[str] | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for model and LoRA downloads") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=target_dir,
        allow_patterns=files or None,
        local_dir_use_symlinks=False,
    )
    return target_dir


def monitor_download_progress(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    repo: str,
    target_dir: Path,
    total_bytes: int | None,
    stop_event: threading.Event,
) -> None:
    interval = max(5, min(settings.heartbeat_seconds, 15))
    started_bytes = directory_size(target_dir)
    started_at = time.monotonic()
    while not stop_event.wait(interval):
        try:
            heartbeat(api, settings, "busy", job_id)
            update_job(
                api,
                job_id,
                download_progress_payload(
                    repo,
                    directory_size(target_dir),
                    total_bytes,
                    started_bytes=started_bytes,
                    started_at=started_at,
                ),
            )
        except httpx.HTTPError as exc:
            emit({"event": "download_progress_failed", "jobId": job_id, "error": str(exc), "reportedAt": now()})


def run_monitored_download(
    api: ApiClient,
    settings: WorkerSettings,
    job_id: str,
    repo: str,
    target_dir: Path,
    files: list[str] | None,
    total_bytes: int | None,
) -> Path:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=monitor_download_progress,
        args=(api, settings, job_id, repo, target_dir, total_bytes, stop_event),
        daemon=True,
    )
    thread.start()
    try:
        return snapshot_huggingface_repo(repo, target_dir, files)
    finally:
        stop_event.set()
        thread.join(timeout=1)


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


def run_model_download_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    payload = job["payload"]
    repo = payload.get("repo")
    if not repo:
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Model download is missing a repository.",
                "error": "Missing payload.repo",
            },
        )
        heartbeat(api, settings, "idle")
        return

    target_dir = Path(payload.get("targetDir") or settings.data_dir / "models" / safe_download_dir(repo))
    try:
        heartbeat(api, settings, "busy", job_id)
        total_bytes = estimate_huggingface_repo_size(repo, payload.get("files") or [])
        update_job(
            api,
            job_id,
            {
                "status": "downloading",
                "stage": "downloading",
                "progress": 0.1,
                "message": (
                    f"Downloading {repo}: 0 B of {format_bytes(total_bytes)}."
                    if total_bytes
                    else f"Downloading {repo}: estimating size."
                ),
            },
        )
        if job_cancel_requested(api, job_id):
            raise InterruptedError("Model download canceled before transfer started.")
        run_monitored_download(api, settings, job_id, repo, target_dir, payload.get("files") or [], total_bytes)
        write_model_install_marker(target_dir, payload, repo, job_id)
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Model download completed.",
                "result": {
                    "modelId": payload.get("modelId"),
                    "repo": repo,
                    "path": str(target_dir),
                    "completedAt": now(),
                },
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
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Model download failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_lora_import_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    payload = job["payload"]
    repo = payload.get("repo")
    source_path = payload.get("sourcePath")
    target_name = safe_download_dir(payload.get("loraId") or payload.get("name") or repo or Path(source_path or "lora").stem)
    target_dir = resolve_lora_import_target(settings, payload, settings.data_dir / "loras" / target_name)

    try:
        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": "downloading",
                "stage": "importing",
                "progress": 0.1,
                "message": "Importing LoRA.",
            },
        )
        if job_cancel_requested(api, job_id):
            raise InterruptedError("LoRA import canceled before transfer started.")
        if repo:
            run_blocking_job_step(
                api,
                settings,
                job_id,
                "busy",
                lambda: snapshot_huggingface_repo(repo, target_dir, payload.get("files") or []),
                loaded_models=None,
            )
        elif source_path:
            source = Path(source_path).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(f"LoRA source not found: {source}")
            target_dir.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target_dir / source.name)
        else:
            raise ValueError("Provide repo or sourcePath for LoRA import")
        manifest_path = lora_manifest_target(settings, payload)
        if manifest_path:
            upsert_lora_manifest_entry(manifest_path, payload["manifestEntry"])
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "LoRA import completed.",
                "result": {"repo": repo, "path": str(target_dir), "completedAt": now()},
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
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "LoRA import failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_placeholder_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    stages = [
        ("preparing", "preparing", 0.1, "Preparing placeholder job."),
        ("running", "running", 0.35, "Running placeholder step 1."),
        ("running", "running", 0.65, "Running placeholder step 2."),
        ("saving", "saving", 0.9, "Saving placeholder result."),
    ]

    for status, stage, progress, message in stages:
        if job_cancel_requested(api, job_id):
            update_job(
                api,
                job_id,
                {
                    "status": "canceled",
                    "stage": "canceled",
                    "progress": progress,
                    "message": "Worker canceled the job before completion.",
                },
            )
            heartbeat(api, settings, "idle")
            return

        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
            },
        )
        time.sleep(1.5)

    update_job(
        api,
        job_id,
        {
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Placeholder job completed.",
            "result": {"completedAt": now(), "output": "placeholder"},
        },
    )
    heartbeat(api, settings, "idle")


def run_image_job(api: ApiClient, settings: WorkerSettings, job: dict, image_adapters: dict[str, object]) -> None:
    job_id = job["id"]
    adapter = create_image_adapter(job, image_adapters)

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


def run_video_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    adapter = ProceduralVideoAdapter()

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
        progress(
            "running",
            "estimating",
            0.18,
            f"Estimated {requirements['previewFrames']} preview frames for this clip.",
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
        adapter.cleanup(job_id)
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
        heartbeat(api, settings, "idle", loaded_models=adapter_loaded_models())


def run_timeline_export_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
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
        progress("preparing", "preparing", 0.06, "Preparing timeline export.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda: run_timeline_export(
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=lambda: job_cancel_requested(api, job_id),
            ),
            loaded_models=None,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Timeline MP4 export saved.",
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
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Timeline export failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_frame_extract_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
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
        progress("preparing", "preparing", 0.08, "Preparing frame extraction.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda: run_frame_extract(
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=lambda: job_cancel_requested(api, job_id),
            ),
            loaded_models=None,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Timeline frame saved as an asset.",
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
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Frame extraction failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_person_detect_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
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
        progress("preparing", "preparing", 0.08, "Preparing representative frame analysis.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda: run_person_detect(
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=lambda: job_cancel_requested(api, job_id),
            ),
            loaded_models=None,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Person candidates detected.",
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
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Person detection failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_person_track_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]

    def progress(status: str, stage: str, value: float, message: str) -> None:
        heartbeat(api, settings, "busy", job_id)
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
        progress("preparing", "preparing", 0.08, "Preparing selected-person tracking.")
        result = run_blocking_job_step(
            api,
            settings,
            job_id,
            "busy",
            lambda: run_person_track(
                settings=settings,
                job=job,
                progress=progress,
                cancel_requested=lambda: job_cancel_requested(api, job_id),
            ),
            loaded_models=None,
        )
        update_job(
            api,
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Reusable person track saved.",
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
        update_job(
            api,
            job_id,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Person tracking failed.",
                "error": str(exc),
            },
        )
    finally:
        heartbeat(api, settings, "idle")


def run_worker_loop(settings: WorkerSettings) -> None:
    gpu = discover_gpu(settings.gpu_id)
    api = ApiClient(settings)
    image_adapters: dict[str, object] = {
        "procedural_preview": ProceduralImageAdapter(),
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
            claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})
            job = claimed.get("job")
            if job is None:
                time.sleep(settings.poll_seconds)
                continue

            emit({"event": "claimed", "jobId": job["id"], "gpuId": job["assignedGpu"], "reportedAt": now()})
            if job["type"] == "placeholder":
                run_placeholder_job(api, settings, job)
            elif job["type"] in ("image_generate", "image_edit"):
                run_image_job(api, settings, job, image_adapters)
            elif job["type"] in ("video_generate", "video_extend", "video_bridge", "person_replace"):
                run_video_job(api, settings, job)
            elif job["type"] == "person_detect":
                run_person_detect_job(api, settings, job)
            elif job["type"] == "person_track":
                run_person_track_job(api, settings, job)
            elif job["type"] == "frame_extract":
                run_frame_extract_job(api, settings, job)
            elif job["type"] == "timeline_export":
                run_timeline_export_job(api, settings, job)
            elif job["type"] == "model_download":
                run_model_download_job(api, settings, job)
            elif job["type"] == "lora_import":
                run_lora_import_job(api, settings, job)
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
    # Compose exposes this as SCENEWORKS_PYTHON_UTILITY_JOBS, then maps it to
    # the in-worker SCENEWORKS_UTILITY_JOBS flag for parent/child consistency.
    utility_jobs = os.getenv("SCENEWORKS_UTILITY_JOBS")
    if gpu_id == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["SCENEWORKS_UTILITY_JOBS"] = utility_jobs if utility_jobs is not None else "1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["SCENEWORKS_UTILITY_JOBS"] = "0"
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


def main() -> None:
    settings = WorkerSettings()
    if settings.gpu_id == "auto" and os.getenv("SCENEWORKS_WORKER_CHILD") != "1":
        supervise_auto_workers(settings)
        return
    run_worker_loop(settings)
