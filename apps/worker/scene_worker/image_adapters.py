from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
import sys
import warnings
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable, Protocol
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from sceneworks_shared import (
    find_asset_sidecar_path,
    find_project_path as shared_find_project_path,
    index_asset,
    read_json,
    safe_int,
    slugify,
    utc_now,
    write_json,
)

from .adapter_utils import filter_call_kwargs
from .lora_adapters import LoraPipelineState, apply_loras_to_pipeline, reject_loras_if_unsupported
from .settings import WorkerSettings


Image.MAX_IMAGE_PIXELS = 64_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

CancelCallback = Callable[[], bool]


class ProgressCallback(Protocol):
    def __call__(
        self,
        status: str,
        stage: str,
        value: float,
        message: str,
        result: dict[str, Any] | None = None,
    ) -> None: ...


def huggingface_repo_cache_exists(repo: str) -> bool:
    repo_cache = huggingface_repo_cache_path(repo)
    if repo_cache is None:
        return False
    return (repo_cache / "snapshots").is_dir() or (repo_cache / "blobs").is_dir()


def huggingface_repo_cache_path(repo: str) -> Path | None:
    default_home = Path.home() / ".cache" / "huggingface"
    hf_home = Path(os.getenv("HF_HOME") or default_home)
    cache_root = Path(os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE") or hf_home / "hub")
    safe_repo = "".join(char if char.isalnum() or char in "._-" else "--" for char in repo).strip("-")
    if not safe_repo:
        return None
    try:
        root = cache_root.resolve()
        repo_cache = (root / f"models--{safe_repo}").resolve()
        repo_cache.relative_to(root)
    except (OSError, ValueError):
        return None
    return repo_cache


def emit_worker_event(event: str, **payload: Any) -> None:
    """Emit a structured JSON diagnostic event on the worker's stdout.

    Mirrors `scene_worker.runtime.emit` so adapter-level phase markers
    (pipeline load, device placement, per-image inference) land in the
    same operator log stream as worker lifecycle events. Keeps phases
    distinguishable when a generation job appears to hang.
    """

    payload["event"] = event
    payload["reportedAt"] = utc_now()
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def gpu_memory_snapshot(torch: Any, device: str) -> dict[str, Any] | None:
    if not isinstance(device, str) or not device.startswith("cuda"):
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return None
    try:
        if not bool(cuda.is_available()):
            return None
    except Exception:
        return None
    snapshot: dict[str, Any] = {"device": device}
    index = None
    if ":" in device:
        try:
            index = int(device.split(":", 1)[1])
        except ValueError:
            index = None
    try:
        allocated = int(cuda.memory_allocated(index) if index is not None else cuda.memory_allocated())
        snapshot["allocatedMb"] = round(allocated / (1024 * 1024), 2)
    except Exception:
        pass
    try:
        reserved = int(cuda.memory_reserved(index) if index is not None else cuda.memory_reserved())
        snapshot["reservedMb"] = round(reserved / (1024 * 1024), 2)
    except Exception:
        pass
    return snapshot


def pipeline_component_devices(pipe: Any) -> list[str]:
    """Return the sorted, unique torch device strings of a pipeline's submodules."""

    devices: set[str] = set()
    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        candidates = list(components.values())
    else:
        candidate_names = ("transformer", "unet", "text_encoder", "text_encoder_2", "vae")
        candidates = [getattr(pipe, name, None) for name in candidate_names]
    for component in candidates:
        if component is None:
            continue
        device = getattr(component, "device", None)
        if device is None:
            parameters = getattr(component, "parameters", None)
            if callable(parameters):
                try:
                    first = next(parameters())
                except StopIteration:
                    first = None
                except Exception:
                    first = None
                if first is not None:
                    device = getattr(first, "device", None)
        if device is None:
            continue
        devices.add(str(device))
    return sorted(devices)


def verify_pipeline_on_device(
    pipe: Any,
    *,
    requested_device: str,
    model_label: str,
    allow_offload: bool,
) -> list[str]:
    """Confirm a GPU-bound pipeline actually landed on the requested CUDA device.

    Returns the observed component device strings. Raises RuntimeError when
    the worker asked for a CUDA device but no pipeline component is on a
    matching CUDA device — that path is the most common source of jobs that
    look "running" while the GPU stays idle.
    """

    devices = pipeline_component_devices(pipe)
    if allow_offload or not requested_device.startswith("cuda"):
        return devices
    if not devices:
        return devices
    target_index = requested_device.split(":", 1)[1] if ":" in requested_device else None
    unexpected_devices = []
    for device in devices:
        if target_index is None:
            if device == "cuda" or device.startswith("cuda:"):
                continue
        elif device == "cuda" or device == requested_device:
            continue
        unexpected_devices.append(device)
    if not unexpected_devices:
        return devices
    observed = ", ".join(devices) or "no detected device"
    raise RuntimeError(
        f"{model_label} did not move onto {requested_device}; pipeline components are on {observed}. "
        "Check CUDA driver compatibility and worker GPU assignment, then retry."
    )


def format_batch_running_message(label: str, index: int, total: int) -> str:
    """Build a per-iteration "Running" progress message that names the actual
    saved count alongside the in-flight index, so users do not see "Running 3
    of 4" without prior images being durable."""

    prefix = f"Generated {index} of {total}. " if index > 0 else ""
    return f"{prefix}Running {label} {index + 1} of {total}."


MODEL_TARGETS = {
    "z_image_turbo": {
        "label": "Z-Image-Turbo",
        "family": "z-image",
        "supportsEdit": False,
        "steps": 8,
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "adapter": "z_image_diffusers",
    },
    "z_image_edit": {
        "label": "Z-Image-Edit",
        "family": "z-image",
        "supportsEdit": True,
        "steps": 8,
        # Uses Turbo weights via ZImageImg2ImgPipeline until the dedicated Edit checkpoint is released.
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "adapter": "z_image_diffusers",
    },
    "qwen_image": {
        "label": "Qwen Image",
        "family": "qwen-image",
        "supportsEdit": False,
        "steps": 20,
        "repo": "Qwen/Qwen-Image",
        "adapter": "qwen_image",
    },
    "qwen_image_edit": {
        "label": "Qwen Image Edit",
        "family": "qwen-image",
        "supportsEdit": True,
        "steps": 20,
        "repo": "Qwen/Qwen-Image-Edit",
        "adapter": "qwen_image",
    },
}


@dataclass(frozen=True)
class ImageRequest:
    project_id: str
    mode: str
    prompt: str
    negative_prompt: str
    model: str
    count: int
    seed: int | None
    seeds: list[int]
    width: int
    height: int
    style_preset: str
    loras: list[dict[str, Any]]
    character_id: str | None
    character_look_id: str | None
    source_asset_id: str | None
    advanced: dict[str, Any]


def load_registry(data_dir: Path) -> list[dict[str, Any]]:
    registry_path = data_dir / "recent-projects.json"
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_request_from_job(job: dict[str, Any]) -> ImageRequest:
    payload = job["payload"]
    return ImageRequest(
        project_id=payload["projectId"],
        mode=payload.get("mode", "text_to_image"),
        prompt=payload.get("prompt", ""),
        negative_prompt=payload.get("negativePrompt", ""),
        model=payload.get("model", "z_image_turbo"),
        count=safe_int(payload.get("count"), 4, 1, 8),
        seed=payload.get("seed"),
        seeds=[int(seed) for seed in payload.get("seeds", []) if seed is not None],
        width=safe_int(payload.get("width"), 1024, 256, 2048),
        height=safe_int(payload.get("height"), 1024, 256, 2048),
        style_preset=payload.get("stylePreset", "cinematic"),
        loras=payload.get("loras", []),
        character_id=payload.get("characterId"),
        character_look_id=payload.get("characterLookId"),
        source_asset_id=payload.get("sourceAssetId"),
        advanced=payload.get("advanced", {}),
    )


class ImageAssetWriter:
    def write_outputs(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        images: list[Image.Image],
        adapter_id: str,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        raw_settings: dict[str, Any],
    ) -> dict[str, Any]:
        return self.write_incremental_outputs(
            settings=settings,
            job=job,
            image_count=len(images),
            image_at_index=lambda index: images[index],
            adapter_id=adapter_id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings=raw_settings,
        )

    def write_incremental_outputs(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        image_count: int,
        image_at_index: Callable[[int], Image.Image],
        adapter_id: str,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        raw_settings: dict[str, Any],
    ) -> dict[str, Any]:
        request = image_request_from_job(job)
        project_path = shared_find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        for folder in ("assets/images", "generation-sets", "recipes"):
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        created_at = utc_now()
        generation_set_id = f"genset_{uuid4().hex}"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        prompt_slug = slugify(request.prompt, fallback="image", max_length=42)
        date_slug = created_at[:10]
        assets = []

        generation_set = {
            "schemaVersion": 1,
            "id": generation_set_id,
            "projectId": request.project_id,
            "jobId": job["id"],
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": image_count,
            "createdAt": created_at,
        }
        write_json(project_path / "generation-sets" / f"{generation_set_id}.json", generation_set)

        for index in range(image_count):
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            image = image_at_index(index)
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            asset_id = f"asset_{uuid4().hex}"
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            filename = f"{date_slug}_{request.model}_{prompt_slug}_{index + 1:04d}.png"
            media_rel = f"assets/images/{filename}"
            media_path = project_path / media_rel
            sidecar_path = media_path.with_suffix(".sceneworks.json")
            image.save(media_path, "PNG")

            asset = build_asset_sidecar(
                asset_id=asset_id,
                project_id=request.project_id,
                generation_set_id=generation_set_id,
                request=request,
                job_id=job["id"],
                media_rel=media_rel,
                created_at=created_at,
                seed=seed,
                index=index,
                model_target=model_target,
                adapter_id=adapter_id,
                raw_settings=raw_settings,
            )
            write_json(sidecar_path, asset)
            write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
            index_asset(project_path, asset, sidecar_path)
            assets.append(asset)
            progress(
                "saving",
                "saving",
                image_batch_progress(index + 1, image_count),
                f"Saved image asset {index + 1} of {image_count}.",
                {
                    "generationSetId": generation_set_id,
                    "assetIds": [item["id"] for item in assets],
                    "assets": assets,
                    "expectedCount": image_count,
                    "adapter": adapter_id,
                    "model": request.model,
                },
            )

        return {
            "generationSetId": generation_set_id,
            "assetIds": [asset["id"] for asset in assets],
            "assets": assets,
            "expectedCount": image_count,
            "adapter": adapter_id,
            "model": request.model,
        }


class ZImageDiffusersAdapter:
    id = "z_image_diffusers"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._img2img_pipe: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        request = image_request_from_job(job)
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a Z-Image Diffusers target.")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
        total = request.count
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = "Z-Image"

        def image_at_index(index: int) -> Image.Image:
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            progress(
                "running",
                "generating",
                image_batch_progress(index, total),
                format_batch_running_message(label, index, total),
            )
            emit_worker_event(
                "image_inference_start",
                jobId=job["id"],
                adapter=self.id,
                model=request.model,
                imageIndex=index,
                imageCount=total,
                device=device,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                image = self._run_pipeline(settings, pipe, request, seed)
            except Exception as exc:
                emit_worker_event(
                    "image_inference_failed",
                    jobId=job["id"],
                    adapter=self.id,
                    imageIndex=index,
                    error=str(exc),
                    errorType=exc.__class__.__name__,
                )
                raise
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return image

        return ImageAssetWriter().write_incremental_outputs(
            settings=settings,
            job=job,
            image_count=total,
            image_at_index=image_at_index,
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={
                **request.advanced,
                "repo": model_target["repo"],
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request),
                "realModelInference": True,
            },
        )

    def _load_pipeline(
        self,
        settings: WorkerSettings,
        request: ImageRequest,
        model_target: dict[str, Any],
        progress: ProgressCallback,
        *,
        job_id: str,
    ) -> Any:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_img2img = request.mode == "edit_image"
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        cached_pipe = self._img2img_pipe if use_img2img else self._text_pipe
        if cached_pipe is not None and self._loaded_repo == repo:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(cached_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return cached_pipe

        if self._loaded_repo and self._loaded_repo != repo:
            self._evict_pipelines(torch)
        elif use_img2img and self._text_pipe is not None:
            self._text_pipe = None
            self._forget_loaded_loras("text")
            self._empty_cuda_cache(torch)
        elif not use_img2img and self._img2img_pipe is not None:
            self._img2img_pipe = None
            self._forget_loaded_loras("img2img")
            self._empty_cuda_cache(torch)

        pipeline_name = "ZImageImg2ImgPipeline" if use_img2img else "ZImagePipeline"
        pipeline_class = getattr(diffusers, pipeline_name, None)
        if pipeline_class is None and use_img2img:
            raise RuntimeError(
                "The installed diffusers package does not expose ZImageImg2ImgPipeline. "
                "Install the latest diffusers build for Z-Image edit support."
            )
        if pipeline_class is None:
            pipeline_class = getattr(diffusers, "DiffusionPipeline")

        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=use_img2img,
            cpuOffload=cpu_offload,
            cached=cache_action == "Loading cached",
        )
        pipe = pipeline_class.from_pretrained(
            repo,
            torch_dtype=dtype,
            low_cpu_mem_usage=bool(request.advanced.get("lowCpuMemUsage", False)),
        )
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        progress("loading_model", "loading_model", 0.22, f"Moving {model_target['label']} to {device}.")
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

        if use_img2img:
            self._img2img_pipe = pipe
        else:
            self._text_pipe = pipe
        self._loaded_repo = repo
        self._loaded_model = request.model
        return pipe

    def _evict_pipelines(self, torch: Any) -> None:
        self._text_pipe = None
        self._img2img_pipe = None
        self._loaded_repo = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(torch)

    def _empty_cuda_cache(self, torch: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_pipeline(self, settings: WorkerSettings, pipe: Any, request: ImageRequest, seed: int) -> Image.Image:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        generator_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(generator_device).manual_seed(seed)
        kwargs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, MODEL_TARGETS[request.model]),
            "guidance_scale": self._guidance_scale(request),
            "generator": generator,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.mode == "edit_image":
            kwargs["image"] = load_source_image(settings, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
        output = pipe(**kwargs)
        image = output.images[0]
        return image.convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        key = "img2img" if request.mode == "edit_image" else "text"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        self._loaded_lora_states[key] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get(key),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"] + 1, 1, 80)

    def _guidance_scale(self, request: ImageRequest) -> float:
        try:
            return float(request.advanced.get("guidanceScale", 0.0))
        except (TypeError, ValueError):
            return 0.0


class QwenImageAdapter:
    id = "qwen_image"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._edit_pipe: Any | None = None
        self._text_repo: str | None = None
        self._edit_repo: str | None = None
        self._loaded_model: str | None = None
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._text_repo, self._edit_repo, self._loaded_model) if value})

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        request = image_request_from_job(job)
        model_target = MODEL_TARGETS.get(request.model, {})
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a Qwen Image target.")
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = "Qwen Image"

        def image_at_index(index: int) -> Image.Image:
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            progress(
                "running",
                "generating",
                image_batch_progress(index, request.count),
                format_batch_running_message(label, index, request.count),
            )
            emit_worker_event(
                "image_inference_start",
                jobId=job["id"],
                adapter=self.id,
                model=request.model,
                imageIndex=index,
                imageCount=request.count,
                device=device,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                image = self._run_pipeline(settings, pipe, request, seed)
            except Exception as exc:
                emit_worker_event(
                    "image_inference_failed",
                    jobId=job["id"],
                    adapter=self.id,
                    imageIndex=index,
                    error=str(exc),
                    errorType=exc.__class__.__name__,
                )
                raise
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return image

        return ImageAssetWriter().write_incremental_outputs(
            settings=settings,
            job=job,
            image_count=request.count,
            image_at_index=image_at_index,
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={
                **request.advanced,
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request),
                "realModelInference": True,
            },
        )

    def _load_pipeline(
        self,
        settings: WorkerSettings,
        request: ImageRequest,
        model_target: dict[str, Any],
        progress: ProgressCallback,
        *,
        job_id: str,
    ) -> Any:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        repo = self._repo_for_request(request, model_target)
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_edit = request.mode == "edit_image"
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        cached_pipe = self._edit_pipe if use_edit else self._text_pipe
        cached_repo = self._edit_repo if use_edit else self._text_repo
        if cached_pipe is not None and cached_repo == repo:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(cached_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return cached_pipe
        if cached_pipe is not None:
            if use_edit:
                self._edit_pipe = None
                self._edit_repo = None
            else:
                self._text_pipe = None
                self._text_repo = None
            self._empty_cuda_cache(torch)
            self._forget_loaded_loras("edit" if use_edit else "text")

        pipeline_name = "QwenImageEditPipeline" if use_edit else "QwenImagePipeline"
        pipeline_class = getattr(diffusers, pipeline_name, None)
        if pipeline_class is None:
            raise RuntimeError(f"The installed diffusers package does not expose {pipeline_name}. Install the latest diffusers build.")

        progress("loading_model", "loading_model", 0.2, f"Loading {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=use_edit,
            cpuOffload=cpu_offload,
            cached=huggingface_repo_cache_exists(repo),
        )
        pipe = pipeline_class.from_pretrained(repo, torch_dtype=dtype)
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

        if use_edit:
            self._edit_pipe = pipe
            self._edit_repo = repo
        else:
            self._text_pipe = pipe
            self._text_repo = repo
        self._loaded_model = request.model
        return pipe

    def _empty_cuda_cache(self, torch: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_pipeline(self, settings: WorkerSettings, pipe: Any, request: ImageRequest, seed: int) -> Image.Image:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        kwargs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, MODEL_TARGETS[request.model]),
            "guidance_scale": self._guidance_scale(request),
            "generator": generator,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.mode == "edit_image":
            kwargs["image"] = load_source_image(settings, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
            kwargs["true_cfg_scale"] = float(request.advanced.get("trueCfgScale", 4.0))
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        return output.images[0].convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        key = "edit" if request.mode == "edit_image" else "text"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["qwen_image"])
        self._loaded_lora_states[key] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get(key),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest) -> float:
        try:
            return float(request.advanced.get("guidanceScale", 4.0))
        except (TypeError, ValueError):
            return 4.0


class ProceduralImageAdapter:
    id = "procedural_preview"

    def loaded_models(self) -> list[str]:
        return []

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        request = image_request_from_job(job)
        reject_loras_if_unsupported(request.loras, self.id)
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        def image_at_index(index: int) -> Image.Image:
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            progress(
                "running",
                "generating",
                image_batch_progress(index, request.count),
                f"Generated preview image {index + 1} of {request.count}.",
            )
            return render_preview_image(request, model_target, seed, index)

        return ImageAssetWriter().write_incremental_outputs(
            settings=settings,
            job=job,
            image_count=request.count,
            image_at_index=image_at_index,
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={
                **request.advanced,
                "targetSteps": model_target["steps"],
                "previewRenderer": True,
            },
        )


def create_image_adapter(
    job: dict[str, Any],
    adapters: dict[str, object] | None = None,
) -> ProceduralImageAdapter | ZImageDiffusersAdapter | QwenImageAdapter:
    payload = job.get("payload", {})
    requested = os.getenv("SCENEWORKS_IMAGE_ADAPTER", payload.get("adapter", "")).strip()
    if requested == "auto":
        requested = ""
    if requested in {"procedural", "procedural_preview"}:
        return adapters.get("procedural_preview") if adapters else ProceduralImageAdapter()
    if requested and requested not in {ZImageDiffusersAdapter.id, QwenImageAdapter.id}:
        raise RuntimeError(f"Unsupported SCENEWORKS_IMAGE_ADAPTER value: {requested}.")
    if requested == ZImageDiffusersAdapter.id:
        return adapters.get("z_image_diffusers") if adapters else ZImageDiffusersAdapter()
    if requested == QwenImageAdapter.id:
        return adapters.get("qwen_image") if adapters else QwenImageAdapter()
    model_target = MODEL_TARGETS.get(payload.get("model", "z_image_turbo"), {})
    if model_target.get("adapter") == ZImageDiffusersAdapter.id:
        return adapters.get("z_image_diffusers") if adapters else ZImageDiffusersAdapter()
    if model_target.get("adapter") == QwenImageAdapter.id:
        return adapters.get("qwen_image") if adapters else QwenImageAdapter()
    return adapters.get("procedural_preview") if adapters else ProceduralImageAdapter()


def model_supports_edit(model_id: str) -> bool:
    return bool(MODEL_TARGETS.get(model_id, {}).get("supportsEdit"))


def resolve_seed(seed: int | None, prompt: str, index: int, seeds: list[int] | None = None) -> int:
    if seed is not None:
        return int(seed) + index
    if seeds and index < len(seeds):
        return int(seeds[index])
    digest = hashlib.sha256(f"{prompt}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def image_batch_progress(completed_count: int, total: int) -> float:
    safe_total = max(1, total)
    bounded_count = min(max(0, completed_count), safe_total)
    return 0.78 + (bounded_count / safe_total) * 0.17


def select_torch_device(torch: Any, gpu_id: str | None = None) -> str:
    if torch.cuda.is_available():
        gpu_id = str(gpu_id or "").strip()
        if gpu_id.isdigit():
            try:
                device_count = int(torch.cuda.device_count())
            except (AttributeError, TypeError, ValueError):
                device_count = 0
            physical_index = int(gpu_id)
            if device_count > 1 and physical_index < device_count:
                return f"cuda:{physical_index}"
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_inference_backend_available(torch: Any) -> bool:
    if torch.cuda.is_available():
        return True
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps and mps.is_available())


def require_inference_backend_for_gpu_worker(torch: Any, gpu_id: str | None) -> None:
    requested = str(gpu_id or "").strip().lower()
    if requested != "cpu" and not torch_inference_backend_available(torch):
        raise RuntimeError(
            "CUDA-enabled PyTorch is not available in this GPU worker. "
            "Rebuild the worker with a CUDA PyTorch wheel, for example "
            "`docker compose build worker --no-cache`, then restart the worker."
        )


def activate_torch_device(torch: Any, device: str) -> None:
    if device.startswith("cuda:") and hasattr(torch.cuda, "set_device"):
        torch.cuda.set_device(device)


def select_torch_dtype(torch: Any, device: str, requested: Any) -> Any:
    if requested == "float16":
        return torch.float16
    if requested == "float32" or device == "cpu":
        return torch.float32
    return torch.bfloat16


def load_source_image(settings: WorkerSettings, request: ImageRequest) -> Image.Image:
    source_path = request.advanced.get("sourceImagePath")
    if not source_path and request.source_asset_id:
        project_path = shared_find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        source_path = find_asset_media_path(project_path, request.source_asset_id)
    if not source_path:
        raise RuntimeError("Image edit jobs require a source image asset.")
    try:
        image = Image.open(source_path).convert("RGB")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise RuntimeError(f"Source image could not be loaded safely: {source_path}") from exc
    return image.resize((request.width, request.height))


def find_asset_media_path(project_path: Path, asset_id: str) -> Path:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is not None:
        asset = read_json(sidecar_path)
        media_path = project_path / asset.get("file", {}).get("path", "")
        if media_path.exists():
            return media_path
        raise RuntimeError(f"Source image file is missing for asset {asset_id}.")
    raise RuntimeError(f"Source image asset not found: {asset_id}.")


def render_preview_image(request: ImageRequest, model_target: dict[str, Any], seed: int, index: int) -> Image.Image:
    import numpy as np

    width = min(request.width, 1280)
    height = min(request.height, 1280)
    digest = hashlib.sha256(f"{request.prompt}:{request.style_preset}:{seed}".encode("utf-8")).digest()
    base = np.array([digest[0], digest[1], digest[2]], dtype=np.float32)
    accent = np.array([digest[9], digest[10], digest[11]], dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    mix = x * 0.56 + y * 0.44
    xi = np.arange(width, dtype=np.uint32)[None, :]
    yi = np.arange(height, dtype=np.uint32)[:, None]
    wave = ((xi * digest[3] + yi * digest[4] + seed) % 255).astype(np.float32) / 255
    pixels = base * (1 - mix[..., None]) + accent * mix[..., None] * 0.85 + wave[..., None] * 34
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, height * 0.68, width, height), fill=(12, 12, 12, 168))
    draw.rectangle((0, 0, width, 84), fill=(12, 12, 12, 118))
    font = ImageFont.load_default()
    title = f"{model_target['label']} preview #{index + 1}"
    draw.text((28, 26), title, fill=(250, 241, 220, 255), font=font)
    draw.text((28, 50), f"{request.mode.replace('_', ' ')} | seed {seed}", fill=(194, 235, 226, 255), font=font)

    text = request.prompt.strip() or "Untitled prompt"
    y = int(height * 0.7) + 24
    for line in wrap(text, width=max(28, width // 14))[:8]:
        draw.text((28, y), line, fill=(255, 255, 255, 242), font=font)
        y += 18
    return image


def build_asset_sidecar(
    *,
    asset_id: str,
    project_id: str,
    generation_set_id: str,
    request: ImageRequest,
    job_id: str,
    media_rel: str,
    created_at: str,
    seed: int,
    index: int,
    model_target: dict[str, Any],
    adapter_id: str,
    raw_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": project_id,
        "generationSetId": generation_set_id,
        "type": "image",
        "displayName": f"{request.prompt[:56] or 'Generated image'} #{index + 1}",
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": min(request.width, 1280),
            "height": min(request.height, 1280),
            "duration": None,
            "fps": None,
        },
        "status": {
            "favorite": False,
            "rating": 0,
            "rejected": False,
            "trashed": False,
        },
        "recipe": {
            "mode": request.mode,
            "model": request.model,
            "adapter": adapter_id,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "seed": seed,
            "loras": request.loras,
            "stylePreset": request.style_preset,
            "normalizedSettings": {
                "width": request.width,
                "height": request.height,
                "count": request.count,
                "family": model_target["family"],
                "characterId": request.character_id,
                "characterLookId": request.character_look_id,
                "characterConditioningActive": False,
                "characterConditioningNote": "Character metadata is recorded, but adapter-level character conditioning is not active in this build.",
            },
            "rawAdapterSettings": raw_settings,
        },
        "lineage": {
            "parents": [request.source_asset_id] if request.source_asset_id else [],
            "sourceAssetId": request.source_asset_id,
            "sourceTimestamp": None,
            "jobId": job_id,
        },
    }


