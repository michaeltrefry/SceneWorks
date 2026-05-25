from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable, Iterable, Protocol, TypeVar
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

from .adapter_utils import cancel_step_callback, filter_call_kwargs
from .hf_cache import huggingface_repo_cache_path
from .lora_adapters import (
    LoraPipelineState,
    apply_loras_to_pipeline,
    normalize_lora_specs,
    reject_loras_if_unsupported,
    validate_lora_compatibility,
)
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
    if isinstance(device, str) and device.startswith("mps"):
        # MPS uses unified memory, so allocated/driver figures double as the
        # process's accelerator footprint — the only built-in signal for the Mac
        # memory growth (CUDA-style per-device stats don't exist here).
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is None:
            return None
        snapshot: dict[str, Any] = {"device": "mps"}
        for key, attr in (
            ("allocatedMb", "current_allocated_memory"),
            ("driverAllocatedMb", "driver_allocated_memory"),
        ):
            fn = getattr(mps_backend, attr, None)
            if callable(fn):
                try:
                    snapshot[key] = round(int(fn()) / (1024 * 1024), 2)
                except Exception:
                    pass
        return snapshot
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
    "lens": {
        "label": "Lens",
        "family": "lens",
        "supportsEdit": False,
        # Non-distilled base: 20 steps, CFG 5.0. Also the LoRA training base.
        "steps": 20,
        "guidanceScale": 5.0,
        "repo": "microsoft/Lens",
        "adapter": "lens_turbo",
    },
    "lens_turbo": {
        "label": "Lens-Turbo",
        "family": "lens",
        "supportsEdit": False,
        # Distilled 4-step variant; the base Lens model uses 20-50 steps.
        "steps": 4,
        "repo": "microsoft/Lens-Turbo",
        "adapter": "lens_turbo",
    },
    "sensenova_u1_8b": {
        "label": "SenseNova-U1 8B",
        "family": "sensenova-u1",
        # Unified model: same weights do text-to-image and instruction editing (it2i).
        "supportsEdit": True,
        # Base 8B-MoT uses ~50 steps; an 8-step distill LoRA exists (cfg 1.0).
        "steps": 50,
        "repo": "sensenova/SenseNova-U1-8B-MoT",
        "adapter": "sensenova_u1",
    },
    "sensenova_u1_8b_fast": {
        "label": "SenseNova-U1 8B Fast",
        "family": "sensenova-u1",
        # Distilled editing (it2i) at 8 steps; the it2i path merges the same LoRA.
        "supportsEdit": True,
        # 8-step distill LoRA (cfg 1.0): shares the base weights, ~5-6x faster.
        "steps": 8,
        "guidanceScale": 1.0,
        "repo": "sensenova/SenseNova-U1-8B-MoT",
        "adapter": "sensenova_u1",
        "distillLora": {
            "repo": "sensenova/SenseNova-U1-8B-MoT-LoRAs",
            "file": "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors",
        },
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
        # Backstop only — per-model resolution is governed by manifest limits + the UI.
        # SenseNova-U1's trained buckets reach 3456 (the adapter snaps by aspect ratio),
        # so this clamp must allow the requested ratio through rather than truncate it.
        width=safe_int(payload.get("width"), 1024, 256, 4096),
        height=safe_int(payload.get("height"), 1024, 256, 4096),
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
        request: ImageRequest,
        project_path: Path,
        images: list[Image.Image],
        adapter_id: str,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        raw_settings: dict[str, Any],
    ) -> dict[str, Any]:
        return self.write_incremental_outputs(
            request=request,
            project_path=project_path,
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
        request: ImageRequest,
        project_path: Path,
        image_count: int,
        image_at_index: Callable[[int], Image.Image],
        adapter_id: str,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        raw_settings: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = utc_now()
        generation_set_id = f"genset_{uuid4().hex}"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        prompt_slug = slugify(request.prompt, fallback="image", max_length=42)
        date_slug = created_at[:10]
        # Each generation set saves its PNGs into its own subfolder so two jobs that
        # share the same date + model + prompt + image index cannot collide on a flat
        # `<date>_<model>_<prompt>_<index>.png` name and clobber each other's PNGs.
        # The folder carries the uniqueness (a full UUID), so the per-image filenames
        # stay short and readable. Asset discovery is rglob-based and paths are stored
        # in the sidecar/DB, so nesting is transparent downstream.
        images_dir = project_path / "assets" / "images" / generation_set_id
        images_dir.mkdir(parents=True, exist_ok=True)

        # Rust is the single project-store writer now (story 1656): the worker saves
        # only the PNG bytes and reports flat facts; the Rust API builds + writes the
        # sidecar / generation-set / recipe and indexes project.db on each progress
        # update, then re-injects the built assets into the result. We still emit a
        # progress update per image so multi-image batches keep streaming into the UI.
        generation_set = {
            "id": generation_set_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": image_count,
            "createdAt": created_at,
        }
        asset_writes: list[dict[str, Any]] = []

        for index in range(image_count):
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            image = image_at_index(index)
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            asset_id = f"asset_{uuid4().hex}"
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            filename = f"{date_slug}_{request.model}_{prompt_slug}_{index + 1:04d}.png"
            media_rel = f"assets/images/{generation_set_id}/{filename}"
            image.save(project_path / media_rel, "PNG")

            asset_writes.append(
                {
                    "assetId": asset_id,
                    "mediaPath": media_rel,
                    "mimeType": "image/png",
                    # True saved pixel dimensions, not the request's. SenseNova-U1 (and
                    # any model that snaps to a trained bucket) saves at a size that
                    # differs from request.width/height.
                    "width": image.width,
                    "height": image.height,
                    "normalizedWidth": request.width,
                    "normalizedHeight": request.height,
                    "count": request.count,
                    "family": model_target["family"],
                    "seed": seed,
                    "index": index,
                    "displayName": f"{request.prompt[:56] or 'Generated image'} #{index + 1}",
                    "createdAt": created_at,
                    "mode": request.mode,
                    "model": request.model,
                    "adapter": adapter_id,
                    "prompt": request.prompt,
                    "negativePrompt": request.negative_prompt,
                    "loras": request.loras,
                    "stylePreset": request.style_preset,
                    "characterId": request.character_id,
                    "characterLookId": request.character_look_id,
                    "sourceAssetId": request.source_asset_id,
                    "rawAdapterSettings": raw_settings,
                }
            )
            progress(
                "saving",
                "saving",
                image_batch_progress(index + 1, image_count),
                f"Saved image asset {index + 1} of {image_count}.",
                {
                    "generationSetId": generation_set_id,
                    "expectedCount": image_count,
                    "adapter": adapter_id,
                    "model": request.model,
                    "generationSet": generation_set,
                    "assetWrites": list(asset_writes),
                },
            )

        return {
            "generationSetId": generation_set_id,
            "expectedCount": image_count,
            "adapter": adapter_id,
            "model": request.model,
            "generationSet": generation_set,
            "assetWrites": asset_writes,
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

    def unload(self) -> bool:
        """Free any resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None and self._img2img_pipe is None:
            return False
        self._evict_pipelines(importlib.import_module("torch"))
        return True

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
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
                image = self._run_pipeline(settings, pipe, request, seed, cancel_requested=cancel_requested)
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
            request=request,
            project_path=project_path,
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
        # gc.collect() first: the pipeline we just dropped is held alive by its
        # nn.Module reference cycles until the cyclic collector runs, so a bare
        # empty_cache() would reclaim nothing on MPS.
        release_inference_memory(torch)

    def _run_pipeline(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
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
            kwargs["image"] = load_source_image(project_path, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**kwargs)
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
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

    def unload(self) -> bool:
        """Free any resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None and self._edit_pipe is None:
            return False
        self._text_pipe = None
        self._edit_pipe = None
        self._text_repo = None
        self._edit_repo = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(importlib.import_module("torch"))
        return True

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
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
                image = self._run_pipeline(settings, pipe, request, seed, cancel_requested=cancel_requested)
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
            request=request,
            project_path=project_path,
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
        # gc.collect() first so the just-dropped pipeline is actually collected
        # before empty_cache() asks the allocator to return its blocks.
        release_inference_memory(torch)

    def _run_pipeline(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
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
            kwargs["image"] = load_source_image(project_path, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
            kwargs["true_cfg_scale"] = float(request.advanced.get("trueCfgScale", 4.0))
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
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


# Lens trains on two base resolutions crossed with nine aspect ratios and expects
# `base_resolution` + `aspect_ratio` rather than free width/height. These mirror
# scene_worker/_vendor/lens/resolution.py so we can snap a SceneWorks W×H request
# onto the nearest trained bucket without importing the (diffusers-injecting) lens
# package just to read the table.
_LENS_BASE_RESOLUTIONS = (1024, 1440)
_LENS_ASPECT_RATIOS = (
    ("1:2", 1 / 2),
    ("9:16", 9 / 16),
    ("2:3", 2 / 3),
    ("3:4", 3 / 4),
    ("1:1", 1.0),
    ("4:3", 4 / 3),
    ("3:2", 3 / 2),
    ("16:9", 16 / 9),
    ("2:1", 2.0),
)
# (aspect_ratio, label) buckets for snap_to_aspect_bucket; preserves table order.
_LENS_ASPECT_BUCKETS = [(ratio, label) for label, ratio in _LENS_ASPECT_RATIOS]


_BucketT = TypeVar("_BucketT")


def snap_to_aspect_bucket(
    width: int, height: int, buckets: Iterable[tuple[float, _BucketT]]
) -> _BucketT:
    """Return the value of the bucket whose aspect ratio is closest to width/height
    in log-space. ``buckets`` is an iterable of (aspect_ratio, value) pairs; ties
    resolve to the first matching bucket, so callers pass tables in priority order.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    target = math.log(width / height)
    return min(buckets, key=lambda bucket: abs(target - math.log(bucket[0])))[1]


def lens_resolution_for(width: int, height: int) -> tuple[int, str]:
    """Snap a requested W×H to the nearest Lens (base_resolution, aspect_ratio).

    The base is chosen by total area against the geometric midpoint of the two
    bases' square areas (1024² and 1440²); the aspect ratio by closest log-ratio
    so portrait/landscape requests land on the matching bucket.
    """
    base = 1440 if max(1, int(width)) * max(1, int(height)) >= 1024 * 1440 else 1024
    return base, snap_to_aspect_bucket(width, height, _LENS_ASPECT_BUCKETS)


class LensTurboAdapter:
    """Microsoft Lens / Lens-Turbo text-to-image, run OUT-OF-PROCESS.

    Lens needs transformers 5.x (gpt-oss text encoder) + diffusers 0.38, which are
    incompatible with the main worker venv's transformers 4.x stack that native
    LTX-2.3 (ltx-core's Gemma-3 integration) requires. So Lens runs in a dedicated
    sidecar venv (``/opt/lens-venv``) via ``scene_worker/lens_runner.py``; this
    adapter only orchestrates that subprocess and writes the resulting PNGs through
    the shared asset writer. The vendored ``lens`` package (scene_worker/_vendor)
    is imported by the runner, not here.

    Text-to-image only (no edit/img2img). LoRAs (the `lens` family, trained by
    the `lens_lora` kernel) are resolved here and applied to the transformer in
    the sidecar via PeftAdapterMixin (sc-1587).
    """

    id = "lens_turbo"

    def loaded_models(self) -> list[str]:
        # The sidecar process loads and frees the model per job; nothing stays
        # resident in this (main-venv) process.
        return []

    @staticmethod
    def _lens_python() -> str:
        return os.getenv("SCENEWORKS_LENS_PYTHON", "/opt/lens-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        return Path(__file__).resolve().parent / "lens_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._lens_python()).exists() and self._runner_path().exists()

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        if request.mode == "edit_image":
            raise RuntimeError(f"{request.model} does not support image editing.")
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["lens_turbo"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a Lens target.")
        # Lens LoRAs (sc-1587) are trained on the base and applied to the
        # transformer inside the sidecar. Resolve + validate them in the main venv
        # so a bad path or incompatible family fails before we spawn the
        # subprocess; the sidecar only sees concrete file paths + weights.
        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id
        )
        lora_specs = normalize_lora_specs(request.loras)
        if not self._sidecar_available():
            raise RuntimeError(
                "Lens generation requires the isolated Lens sidecar venv. Rebuild the worker image with "
                "INCLUDE_LENS=1 (the Docker Compose default), or set SCENEWORKS_LENS_PYTHON to a Python "
                f"interpreter that has the lens stack installed (looked for {self._lens_python()})."
            )

        total = request.count
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        steps = self._num_inference_steps(request, model_target)
        guidance_scale = self._guidance_scale(request, model_target)
        base_resolution, aspect_ratio = lens_resolution_for(request.width, request.height)
        seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, getattr(settings, "gpu_id", None))
        # mxfp4 keeps the gpt-oss-20b text encoder small but needs CUDA + Triton
        # kernels, which exist only on NVIDIA. On MPS/CPU the encoder must load
        # dequantized to bf16 (transformers auto-falls back, but force it here so
        # a non-CUDA host never reaches the Triton path).
        disable_mxfp4 = bool(request.advanced.get("disableMxfp4", False)) or not device.startswith("cuda")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (sidecar venv).")
        work_dir = Path(tempfile.mkdtemp(prefix="lens_sidecar_"))
        try:
            images = self._run_sidecar(
                job_id=job["id"],
                work_dir=work_dir,
                label=model_target["label"],
                total=total,
                spec={
                    "repo": repo,
                    "prompt": request.prompt,
                    "negativePrompt": request.negative_prompt,
                    "baseResolution": base_resolution,
                    "aspectRatio": aspect_ratio,
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance_scale,
                    "seeds": seeds,
                    "disableMxfp4": disable_mxfp4,
                    "cpuOffload": bool(request.advanced.get("cpuOffload", False)),
                    "dtype": request.advanced.get("dtype"),
                    "device": device,
                    "loras": [
                        {"path": lora.path, "weight": lora.weight, "name": lora.adapter_name}
                        for lora in lora_specs
                    ],
                },
                progress=progress,
                cancel_requested=cancel_requested,
            )

            def image_at_index(index: int) -> Image.Image:
                progress(
                    "running",
                    "generating",
                    image_batch_progress(index, total),
                    format_batch_running_message("Lens-Turbo", index, total),
                )
                with Image.open(images[index]) as handle:
                    return handle.convert("RGB")

            return ImageAssetWriter().write_incremental_outputs(
                request=request,
                project_path=project_path,
                image_count=total,
                image_at_index=image_at_index,
                adapter_id=self.id,
                progress=progress,
                cancel_requested=cancel_requested,
                raw_settings={
                    **request.advanced,
                    "repo": repo,
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance_scale,
                    "baseResolution": base_resolution,
                    "aspectRatio": aspect_ratio,
                    "textEncoderMxfp4": not disable_mxfp4,
                    "sidecarVenv": self._lens_python(),
                    "realModelInference": True,
                },
            )
        finally:
            # The writer has read every PNG into the project by now; drop the
            # sidecar's scratch dir regardless of success/failure.
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_sidecar(
        self,
        *,
        job_id: str,
        work_dir: Path,
        label: str,
        total: int,
        spec: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> list[str]:
        spec = {**spec, "outDir": str(work_dir)}
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        stdout_log = work_dir / "stdout.log"
        cmd = [self._lens_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "lens_sidecar_start",
            jobId=job_id,
            adapter=self.id,
            repo=spec["repo"],
            imageCount=total,
            device=spec["device"],
            mxfp4=not spec["disableMxfp4"],
            sidecar=self._lens_python(),
        )
        progress("running", "generating", image_batch_progress(0, total), f"Running {label} ({total} image(s)).")
        # stdout -> file (avoids any pipe-fill deadlock); stderr inherits to the
        # worker log for diagnostics. Poll so the job stays cancelable; the
        # heartbeat thread keeps it alive during the (minutes-long) run.
        with stdout_log.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=None)
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("Image generation canceled by user.")
        result = self._read_result(work_dir, stdout_log)
        if proc.returncode != 0 or "error" in result:
            error = result.get("error") or f"Lens sidecar exited with code {proc.returncode}."
            emit_worker_event(
                "lens_sidecar_failed",
                jobId=job_id,
                adapter=self.id,
                error=error,
                returnCode=proc.returncode,
            )
            raise RuntimeError(f"Lens generation failed in the sidecar venv: {error}")
        images = [str(path) for path in result.get("images", [])]
        if len(images) != total:
            raise RuntimeError(f"Lens sidecar produced {len(images)} image(s); expected {total}.")
        emit_worker_event("lens_sidecar_complete", jobId=job_id, adapter=self.id, imageCount=len(images))
        return images

    @staticmethod
    def _read_result(work_dir: Path, stdout_log: Path) -> dict[str, Any]:
        result_path = work_dir / "result.json"
        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        try:
            lines = [line for line in stdout_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            lines = []
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
        return {"error": "Lens sidecar produced no parseable result."}

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        # Lens-Turbo is distilled for guidance_scale ~1.0 (no CFG); the base Lens
        # model uses ~5.0. The per-model default comes from MODEL_TARGETS so each
        # variant gets the right CFG when the request does not override it.
        default = model_target.get("guidanceScale", 1.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)


class ProceduralImageAdapter:
    id = "procedural_preview"

    def loaded_models(self) -> list[str]:
        return []

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
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
            request=request,
            project_path=project_path,
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


_SENSENOVA_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}
_SENSENOVA_ASPECT_BUCKETS = [(w / h, (w, h)) for (w, h) in _SENSENOVA_RESOLUTIONS.values()]


def sensenova_resolution_for(width: int, height: int) -> tuple[int, int]:
    """Snap a requested W×H to the nearest SenseNova-U1 trained bucket (by aspect ratio).

    SenseNova-U1 only renders well at its trained resolutions; off-bucket sizes
    degrade (upstream warns). Pick the bucket whose aspect ratio is closest in
    log-space so portrait/landscape requests land on the matching orientation.
    """
    return snap_to_aspect_bucket(width, height, _SENSENOVA_ASPECT_BUCKETS)


# SenseNova-U1 interleaved generation was trained at smaller buckets than plain
# text-to-image (e.g. 1:1 = 1536² vs 2048², 16:9 = 2048×1152 vs 2720×1536), so it
# has its own bucket set. Mirrors upstream examples/interleave/inference.py.
_INTERLEAVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1536, 1536),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "3:2": (1888, 1248),
    "2:3": (1248, 1888),
    "4:3": (1760, 1312),
    "3:4": (1312, 1760),
    "1:2": (1088, 2144),
    "2:1": (2144, 1088),
    "1:3": (864, 2592),
    "3:1": (2592, 864),
}
_INTERLEAVE_ASPECT_BUCKETS = [(w / h, (w, h)) for (w, h) in _INTERLEAVE_RESOLUTIONS.values()]


def interleave_resolution_for(width: int, height: int) -> tuple[int, int]:
    """Snap a requested W×H to the nearest SenseNova-U1 *interleave* bucket by
    aspect ratio (log-space). Off-bucket sizes degrade, as upstream warns."""
    return snap_to_aspect_bucket(width, height, _INTERLEAVE_ASPECT_BUCKETS)


# Interleave inference requires a system prompt describing the think/no-think
# protocol the model was trained with; without it the model won't interleave
# correctly. Verbatim from upstream examples/interleave/inference.py (238d6cf).
_INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without "
    "reasoning. Do not use tags like <image1>, <image2>; present any images naturally "
    "alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. The answer "
    "may include text, images, or both. Match the user's language in both reasoning and the "
    "final answer."
)


class SenseNovaU1Adapter:
    """SenseNova-U1 unified multimodal model — text-to-image, run IN-PROCESS.

    NEO-unify (Qwen3-based Mixture-of-Transformers; no separate VAE/encoder).
    Unlike Lens, its deps (torch 2.8 / transformers 4.57.x / accelerate) match
    the main worker venv, so it loads in-process via the vendored ``sensenova_u1``
    package (scene_worker/_vendor): importing it registers the ``neo_chat`` model
    type, so ``AutoModel.from_pretrained`` resolves it with no trust_remote_code.
    Attention uses torch SDPA (flash-attn optional), so it runs on CUDA and MPS.

    Supports text-to-image and instruction-based editing (it2i); VQA and
    interleaved generation are not wired yet. The ``sensenova_u1_8b_fast``
    variant merges the upstream 8-step distill LoRA at load (see ``distillLora``
    in MODEL_TARGETS); user-supplied LoRAs are still rejected.
    """

    id = "sensenova_u1"

    # Pixel normalization from the upstream T2I example (examples/t2i/inference.py).
    _NORM_MEAN = (0.5, 0.5, 0.5)
    _NORM_STD = (0.5, 0.5, 0.5)

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._repo: str | None = None
        self._loaded_model: str | None = None
        # Identity of the distill LoRA merged into the cached model (or None for
        # the base model). The merge mutates weights in place, so this must be
        # part of the cache key — otherwise a fast-variant model would be reused
        # for the base 50-step variant (and vice versa).
        self._distill_lora_key: str | None = None

    def loaded_models(self) -> list[str]:
        return [self._loaded_model] if self._loaded_model else []

    def unload(self) -> bool:
        """Free the resident model so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._model is None:
            return False
        self._model = None
        self._tokenizer = None
        self._repo = None
        self._distill_lora_key = None
        self._loaded_model = None
        release_inference_memory(importlib.import_module("torch"))
        return True

    @staticmethod
    def _ensure_vendor_on_path() -> None:
        vendor = str(Path(__file__).resolve().parent / "_vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("numInferenceSteps"), int(model_target.get("steps", 50)), 1, 100)

    @staticmethod
    def _guidance_scale(request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = float(model_target.get("guidanceScale", 4.0))
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _image_guidance_scale(request: ImageRequest) -> float:
        # Image-conditioning guidance for editing (it2i); upstream default is 1.0.
        try:
            return float(request.advanced.get("imageGuidanceScale", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        reject_loras_if_unsupported(request.loras, self.id)
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["sensenova_u1_8b"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a SenseNova-U1 target.")
        is_edit = request.mode == "edit_image"
        if is_edit and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        torch = importlib.import_module("torch")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        distill_lora = model_target.get("distillLora") if isinstance(model_target.get("distillLora"), dict) else None
        steps = self._num_inference_steps(request, model_target)
        guidance_scale = self._guidance_scale(request, model_target)
        timestep_shift = float(request.advanced.get("timestepShift", 3.0) or 3.0)
        img_guidance_scale = self._image_guidance_scale(request)
        width, height = sensenova_resolution_for(request.width, request.height)
        source_image = load_source_image(project_path, request) if is_edit else None

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        model, tokenizer = self._load_model(torch, repo, device, dtype, distill_lora=distill_lora, job_id=job["id"])
        self._loaded_model = request.model
        label = model_target["label"]

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
                resolution=f"{width}x{height}",
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                if is_edit:
                    image = self._run_edit_inference(
                        torch, model, tokenizer, request.prompt, source_image,
                        width, height, steps, guidance_scale, img_guidance_scale, timestep_shift, seed,
                    )
                else:
                    image = self._run_inference(
                        torch, model, tokenizer, request.prompt,
                        width, height, steps, guidance_scale, timestep_shift, seed,
                    )
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
            request=request,
            project_path=project_path,
            image_count=request.count,
            image_at_index=image_at_index,
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={
                **request.advanced,
                "repo": repo,
                "numInferenceSteps": steps,
                "guidanceScale": guidance_scale,
                **({"imageGuidanceScale": img_guidance_scale} if is_edit else {}),
                "timestepShift": timestep_shift,
                "resolution": f"{width}x{height}",
                "realModelInference": True,
            },
        )

    def _load_model(
        self,
        torch: Any,
        repo: str,
        device: str,
        dtype: Any,
        *,
        distill_lora: dict[str, Any] | None = None,
        job_id: str,
    ) -> tuple[Any, Any]:
        lora_key = f"{distill_lora['repo']}/{distill_lora['file']}" if distill_lora else None
        if self._model is not None and self._repo == repo and self._distill_lora_key == lora_key:
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                repo=repo,
                device=device,
                distillLora=lora_key,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return self._model, self._tokenizer
        if self._model is not None:
            self._model = None
            self._tokenizer = None
            self._repo = None
            self._distill_lora_key = None
            # gc.collect() before empty_cache(): the 8B-MoT we just dropped is
            # kept alive by its nn.Module reference cycles until the cyclic
            # collector runs, so on MPS a bare empty_cache() leaves the old model
            # resident alongside the replacement (base↔fast switches stacked to
            # tens of GB before this fix).
            release_inference_memory(torch)
        self._ensure_vendor_on_path()
        import sensenova_u1  # noqa: F401 — import registers the neo_chat model type
        from sensenova_u1.utils import load_model_and_tokenizer

        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            device=device,
            dtype=str(dtype),
            distillLora=lora_key,
            cached=huggingface_repo_cache_exists(repo),
        )
        model, tokenizer = load_model_and_tokenizer(repo, dtype=dtype, device=device)
        if distill_lora:
            self._merge_distill_lora(model, distill_lora, job_id=job_id)
        self._model = model
        self._tokenizer = tokenizer
        self._repo = repo
        self._distill_lora_key = lora_key
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            device=device,
            distillLora=lora_key,
        )
        return model, tokenizer

    def _merge_distill_lora(self, model: Any, distill_lora: dict[str, Any], *, job_id: str) -> None:
        """Resolve and merge the distill LoRA into the loaded model (in place).

        The ~0.4GB LoRA lives in a separate HF repo from the base weights, so it
        is fetched on demand: the local cache is checked first, then the hub. The
        vendored merge folds the delta into the model weights, so it survives the
        model cache (keyed by ``self._distill_lora_key``) with no per-call cost.
        """
        from sensenova_u1.utils import load_and_merge_lora_weight_from_safetensors

        repo = str(distill_lora["repo"])
        file_name = str(distill_lora["file"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job_id,
            adapter=self.id,
            loraRepo=repo,
            loraFile=file_name,
        )
        lora_path = self._resolve_distill_lora_path(repo, file_name)
        load_and_merge_lora_weight_from_safetensors(model, str(lora_path))
        emit_worker_event(
            "image_lora_apply_complete",
            jobId=job_id,
            adapter=self.id,
            loraRepo=repo,
            loraFile=file_name,
            loraPath=str(lora_path),
        )

    @staticmethod
    def _resolve_distill_lora_path(repo: str, file_name: str) -> str:
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(repo_id=repo, filename=file_name, local_files_only=True)
        except Exception:
            return hf_hub_download(repo_id=repo, filename=file_name)

    def _run_inference(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        timestep_shift: float,
        seed: int,
    ) -> Image.Image:
        with torch.inference_mode():
            tensor = model.t2i_generate(
                tokenizer,
                prompt,
                image_size=(width, height),
                cfg_scale=guidance_scale,
                cfg_norm="none",
                timestep_shift=timestep_shift,
                cfg_interval=(0.0, 1.0),
                num_steps=steps,
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return self._to_pil(torch, tensor)[0]

    def _run_edit_inference(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt: str,
        source_image: Image.Image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        img_guidance_scale: float,
        timestep_shift: float,
        seed: int,
    ) -> Image.Image:
        # Instruction-based editing (it2i): the source image is the conditioning
        # input; `image_size` is the output bucket. Defaults mirror the upstream
        # editing example (cfg 4.0 text / 1.0 image, 50 steps, shift 3.0).
        with torch.inference_mode():
            tensor = model.it2i_generate(
                tokenizer,
                prompt,
                [source_image],
                image_size=(width, height),
                cfg_scale=guidance_scale,
                img_cfg_scale=img_guidance_scale,
                cfg_norm="none",
                timestep_shift=timestep_shift,
                cfg_interval=(0.0, 1.0),
                num_steps=steps,
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return self._to_pil(torch, tensor)[0]

    def answer_question(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        """Visual question answering (VQA): a text answer about a source image.

        Reuses the cached base model (the understanding side) via the model's
        ``chat`` path. Output is text, not an image asset, so this does not go
        through ImageAssetWriter — the answer is returned in the job result.
        """
        payload = job["payload"]
        project_id = payload["projectId"]
        source_asset_id = payload.get("sourceAssetId")
        question = str(payload.get("question") or "").strip()
        if not question:
            raise RuntimeError("Visual question answering requires a question.")
        model_id = payload.get("model", "sensenova_u1_8b")
        model_target = MODEL_TARGETS.get(model_id, MODEL_TARGETS["sensenova_u1_8b"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{model_id} is not a SenseNova-U1 target.")
        advanced = payload.get("advanced", {}) if isinstance(payload.get("advanced"), dict) else {}
        # VQA latency ~ output tokens (one model pass each) + input vision tokens
        # (prefill). Both default low for responsiveness and are tunable per request.
        max_new_tokens = safe_int(payload.get("maxNewTokens"), 256, 16, 2048)
        # Downscale the understanding input — vision tokens (and prefill cost) scale
        # with pixel count (~pixels/1024 tokens), and there's little perceptible
        # difference for question answering between ~768px and ~1024px. Default ~768²
        # (~576 tokens vs ~1024 at 1024²); tunable up via payload.maxImagePixels when a
        # question needs fine detail or in-image text.
        max_image_pixels = safe_int(payload.get("maxImagePixels"), 768 * 768, 256 * 256, 2048 * 2048)

        torch = importlib.import_module("torch")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, advanced.get("dtype"))
        repo = advanced.get("modelRepo") or model_target["repo"]

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        # VQA uses the base understanding path — never the distilled generation LoRA.
        model, tokenizer = self._load_model(torch, repo, device, dtype, distill_lora=None, job_id=job["id"])
        self._loaded_model = model_id

        source_path = self._resolve_source_path(settings, project_id, source_asset_id)
        try:
            image = Image.open(source_path).convert("RGB")
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise RuntimeError(f"Source image could not be loaded safely: {source_path}") from exc

        if cancel_requested():
            raise InterruptedError("Visual question answering canceled by user.")
        progress("running", "generating", 0.6, "Analyzing image.")
        emit_worker_event(
            "image_vqa_start",
            jobId=job["id"],
            adapter=self.id,
            model=model_id,
            sourceAssetId=source_asset_id,
            device=device,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        answer = self._run_vqa(torch, model, tokenizer, image, question, device, max_new_tokens, max_image_pixels)
        emit_worker_event(
            "image_vqa_complete",
            jobId=job["id"],
            adapter=self.id,
            answerChars=len(answer),
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        return {
            "answer": answer,
            "question": question,
            "sourceAssetId": source_asset_id,
            "model": model_id,
            "realModelInference": True,
        }

    def _resolve_source_path(
        self,
        settings: WorkerSettings,
        project_id: str,
        source_asset_id: str | None,
    ) -> str:
        # Resolve only through the project sidecar/DB (find_asset_media_path constrains
        # the result to the project root). There is deliberately no client-supplied path
        # escape hatch: an arbitrary sourceImagePath would let a job read any file the
        # worker can open and, for VQA, return its contents to the caller.
        if not source_asset_id:
            raise RuntimeError("Visual question answering requires a source image asset.")
        project_path = shared_find_project_path(settings.data_dir / "recent-projects.json", project_id)
        return str(find_asset_media_path(project_path, source_asset_id))

    def _run_vqa(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        image: Image.Image,
        question: str,
        device: str,
        max_new_tokens: int,
        max_image_pixels: int,
    ) -> str:
        self._ensure_vendor_on_path()
        from sensenova_u1.models.neo_unify.utils import load_image_native

        pixel_values, grid_hw = load_image_native(image, max_pixels=int(max_image_pixels))
        pixel_values = pixel_values.to(device, dtype=model.dtype)
        grid_hw = grid_hw.to(device)
        generation_config = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        with torch.inference_mode():
            # think=False skips the model's chain-of-thought so the budget goes to
            # the answer (otherwise reasoning fills the output and can truncate it).
            response = model.chat(tokenizer, pixel_values, question, generation_config, grid_hw=grid_hw, think=False)
        return self._strip_reasoning(str(response))

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Drop any ``<think>…</think>`` reasoning so only the answer is returned.

        Defensive backstop for the no-think prime: removes complete think blocks
        and any dangling/unclosed one (e.g. reasoning truncated by max_new_tokens).
        """
        import re

        cleaned = re.sub(r"(?s)<think>.*?</think>", "", text)
        cleaned = re.sub(r"(?s)<think>.*$", "", cleaned)
        return cleaned.strip()

    def _to_pil(self, torch: Any, batch: Any) -> list[Image.Image]:
        import numpy as np

        mean = torch.tensor(self._NORM_MEAN, device=batch.device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._NORM_STD, device=batch.device, dtype=torch.float32).view(1, 3, 1, 1)
        arr = (batch.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        arr = (arr * 255.0).round().astype(np.uint8)
        return [Image.fromarray(a) for a in arr]

    @staticmethod
    def _advanced_float(advanced: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(advanced.get(key, default))
        except (TypeError, ValueError):
            return default

    def generate_interleaved(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        """Interleaved text-image generation (sc-1576): one model pass yields
        ordered text + images, persisted as a ``document`` asset. Reuses the cached
        base understanding+generation model — never the distilled generation LoRA.
        """
        payload = job["payload"]
        project_id = payload["projectId"]
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("Interleaved generation requires a prompt.")
        model_id = payload.get("model", "sensenova_u1_8b")
        model_target = MODEL_TARGETS.get(model_id, MODEL_TARGETS["sensenova_u1_8b"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{model_id} is not a SenseNova-U1 target.")
        advanced = payload.get("advanced", {}) if isinstance(payload.get("advanced"), dict) else {}
        max_images = safe_int(payload.get("maxImages"), 6, 1, 10)
        width, height = interleave_resolution_for(
            safe_int(payload.get("width"), 2048, 256, 4096),
            safe_int(payload.get("height"), 1152, 256, 4096),
        )
        source_asset_ids = [str(asset) for asset in (payload.get("sourceAssetIds") or []) if asset]
        # Upstream interleave defaults (examples/interleave/inference.py @238d6cf).
        steps = safe_int(advanced.get("numInferenceSteps"), 50, 1, 100)
        cfg_scale = self._advanced_float(advanced, "guidanceScale", 4.0)
        img_cfg_scale = self._advanced_float(advanced, "imageGuidanceScale", 1.0)
        timestep_shift = self._advanced_float(advanced, "timestepShift", 3.0)
        max_new_tokens = safe_int(advanced.get("maxNewTokens"), 2048, 64, 8192)
        # Non-Think by default: the document is the deliverable, so skip the model's
        # chain-of-thought (mirrors the VQA think=False choice — "present images
        # naturally alongside the text"). Tunable; confirm on a real MPS run.
        think_mode = bool(advanced.get("thinkMode", False))
        # The think/no-think system prompt is exposed in the UI (prefilled with the
        # default); a blank/absent value falls back to _INTERLEAVE_SYSTEM_MESSAGE.
        system_message = str(advanced.get("systemMessage") or "").strip() or _INTERLEAVE_SYSTEM_MESSAGE
        seed = resolve_seed(payload.get("seed"), prompt, 0, None)

        torch = importlib.import_module("torch")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, advanced.get("dtype"))
        repo = advanced.get("modelRepo") or model_target["repo"]

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        model, tokenizer = self._load_model(torch, repo, device, dtype, distill_lora=None, job_id=job["id"])
        self._loaded_model = model_id

        input_images = self._load_input_images(project_path, source_asset_ids)

        if cancel_requested():
            raise InterruptedError("Interleaved generation canceled by user.")
        progress("running", "generating", 0.45, "Composing interleaved document.")
        emit_worker_event(
            "image_interleave_start",
            jobId=job["id"],
            adapter=self.id,
            model=model_id,
            device=device,
            resolution=f"{width}x{height}",
            maxImages=max_images,
            inputImages=len(input_images),
            thinkMode=think_mode,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        generated_text, images = self._run_interleave(
            torch, model, tokenizer, prompt, input_images,
            width, height, steps, cfg_scale, img_cfg_scale, timestep_shift,
            max_images, max_new_tokens, think_mode, system_message, seed,
        )
        emit_worker_event(
            "image_interleave_complete",
            jobId=job["id"],
            adapter=self.id,
            imageCount=len(images),
            textChars=len(generated_text),
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

        return self._write_interleaved_document(
            project_path=project_path,
            request=request,
            job=job,
            project_id=project_id,
            model_id=model_id,
            prompt=prompt,
            seed=seed,
            generated_text=generated_text,
            images=images,
            cancel_requested=cancel_requested,
            progress=progress,
            raw_settings={
                **advanced,
                "repo": repo,
                "numInferenceSteps": steps,
                "guidanceScale": cfg_scale,
                "imageGuidanceScale": img_cfg_scale,
                "timestepShift": timestep_shift,
                "maxImages": max_images,
                "maxNewTokens": max_new_tokens,
                "thinkMode": think_mode,
                "resolution": f"{width}x{height}",
                "realModelInference": True,
            },
        )

    def _load_input_images(
        self,
        project_path: Path,
        source_asset_ids: list[str],
    ) -> list[Image.Image]:
        if not source_asset_ids:
            return []
        images: list[Image.Image] = []
        for asset_id in source_asset_ids:
            path = find_asset_media_path(project_path, asset_id)
            try:
                images.append(Image.open(path).convert("RGB"))
            except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise RuntimeError(f"Source image could not be loaded safely: {path}") from exc
        return images

    def _run_interleave(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt: str,
        input_images: list[Image.Image],
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        img_cfg_scale: float,
        timestep_shift: float,
        max_images: int,
        max_new_tokens: int,
        think_mode: bool,
        system_message: str,
        seed: int,
    ) -> tuple[str, list[Image.Image]]:
        generation_config = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        with torch.inference_mode():
            text, image_tensors = model.interleave_gen(
                tokenizer,
                prompt,
                images=list(input_images),
                generation_config=generation_config,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                cfg_norm="none",
                max_images=int(max_images),
                enable_timestep_shift=True,
                timestep_shift=timestep_shift,
                image_size=(width, height),
                cfg_interval=(0.0, 1.0),
                num_steps=int(steps),
                system_message=system_message,
                think_mode=think_mode,
                seed=int(seed),
            )
        pil_images: list[Image.Image] = []
        for tensor in image_tensors:
            pil_images.extend(self._to_pil(torch, tensor))
        return str(text), pil_images

    @staticmethod
    def _build_interleaved_segments(
        generated_text: str,
        image_writes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Split the model output on its inline ``<image>`` markers and slot the
        generated image assets in order: text[0], image[0], text[1], image[1], ….
        Reads the worker-reported image facts (assetId + mediaPath); Rust builds
        the image sidecars from the same facts (story 1656)."""
        parts = (generated_text or "").split("<image>")
        segments: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            text = part.strip()
            if text:
                segments.append({"type": "text", "text": text})
            if index < len(image_writes):
                write = image_writes[index]
                segments.append({"type": "image", "assetId": write["assetId"], "path": write["mediaPath"]})
        return segments

    def _write_interleaved_document(
        self,
        *,
        project_path: Path,
        request: ImageRequest,
        job: dict[str, Any],
        project_id: str,
        model_id: str,
        prompt: str,
        seed: int,
        generated_text: str,
        images: list[Image.Image],
        cancel_requested: CancelCallback,
        progress: ProgressCallback,
        raw_settings: dict[str, Any],
    ) -> dict[str, Any]:
        (project_path / "assets" / "documents").mkdir(parents=True, exist_ok=True)

        # Generated images persist as ordinary image assets — the worker saves the
        # PNG bytes + reports facts, and the Rust API builds + indexes their
        # sidecars (story 1656). The document references them in order.
        image_result = ImageAssetWriter().write_outputs(
            request=request,
            project_path=project_path,
            images=images,
            adapter_id=self.id,
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=cancel_requested,
            raw_settings={**raw_settings, "interleaved": True},
        )
        image_writes = image_result.get("assetWrites", [])
        generation_set_id = image_result.get("generationSetId")
        generation_set = image_result.get("generationSet")
        image_asset_ids = [write["assetId"] for write in image_writes]

        segments = self._build_interleaved_segments(generated_text, image_writes)

        created_at = utc_now()
        document_id = f"doc_{uuid4().hex}"
        media_rel = f"assets/documents/{document_id}.json"
        # The worker saves the document body (the "media"); the Rust API builds the
        # document sidecar + indexes project.db from the document fact below.
        write_json(
            project_path / media_rel,
            {
                "schemaVersion": 1,
                "id": document_id,
                "projectId": project_id,
                "jobId": job["id"],
                "model": model_id,
                "prompt": prompt,
                "createdAt": created_at,
                "segments": segments,
            },
        )

        asset_id = f"asset_{uuid4().hex}"
        document_write = {
            "type": "document",
            "assetId": asset_id,
            "mediaPath": media_rel,
            "mimeType": "application/json",
            "displayName": prompt[:56] or "Interleaved document",
            "createdAt": created_at,
            "mode": "interleave",
            "model": model_id,
            "adapter": self.id,
            "prompt": prompt,
            "negativePrompt": "",
            "seed": int(seed),
            "loras": [],
            "rawAdapterSettings": raw_settings,
            "maxImages": raw_settings.get("maxImages"),
            "resolution": raw_settings.get("resolution"),
            "imageCount": len(image_asset_ids),
            "parents": list(image_asset_ids),
        }
        asset_writes = [*image_writes, document_write]
        result = {
            "documentId": document_id,
            "documentAssetId": asset_id,
            "imageAssetIds": image_asset_ids,
            "segments": segments,
            "model": model_id,
            "realModelInference": True,
            "generationSetId": generation_set_id,
            "expectedCount": len(asset_writes),
            "generationSet": generation_set,
            "assetWrites": asset_writes,
        }
        progress("saving", "saving", 1.0, "Interleaved document saved.", result)
        return result


def create_image_adapter(
    job: dict[str, Any],
    adapters: dict[str, object] | None = None,
) -> ProceduralImageAdapter | ZImageDiffusersAdapter | QwenImageAdapter | LensTurboAdapter | SenseNovaU1Adapter:
    payload = job.get("payload", {})
    requested = os.getenv("SCENEWORKS_IMAGE_ADAPTER", payload.get("adapter", "")).strip()
    if requested == "auto":
        requested = ""
    if requested in {"procedural", "procedural_preview"}:
        return adapters.get("procedural_preview") if adapters else ProceduralImageAdapter()
    if requested and requested not in {ZImageDiffusersAdapter.id, QwenImageAdapter.id, LensTurboAdapter.id, SenseNovaU1Adapter.id}:
        raise RuntimeError(f"Unsupported SCENEWORKS_IMAGE_ADAPTER value: {requested}.")
    if requested == ZImageDiffusersAdapter.id:
        return adapters.get("z_image_diffusers") if adapters else ZImageDiffusersAdapter()
    if requested == QwenImageAdapter.id:
        return adapters.get("qwen_image") if adapters else QwenImageAdapter()
    if requested == LensTurboAdapter.id:
        return adapters.get("lens_turbo") if adapters else LensTurboAdapter()
    if requested == SenseNovaU1Adapter.id:
        return adapters.get("sensenova_u1") if adapters else SenseNovaU1Adapter()
    model_target = MODEL_TARGETS.get(payload.get("model", "z_image_turbo"), {})
    if model_target.get("adapter") == ZImageDiffusersAdapter.id:
        return adapters.get("z_image_diffusers") if adapters else ZImageDiffusersAdapter()
    if model_target.get("adapter") == QwenImageAdapter.id:
        return adapters.get("qwen_image") if adapters else QwenImageAdapter()
    if model_target.get("adapter") == LensTurboAdapter.id:
        return adapters.get("lens_turbo") if adapters else LensTurboAdapter()
    if model_target.get("adapter") == SenseNovaU1Adapter.id:
        return adapters.get("sensenova_u1") if adapters else SenseNovaU1Adapter()
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
    # An explicit "cpu" forces CPU on any platform, including Apple Silicon where
    # MPS would otherwise be picked (honors SCENEWORKS_GPU_ID=cpu, sc-1335).
    if str(gpu_id or "").strip().lower() == "cpu":
        return "cpu"
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
        # Route the few diffusers ops that lack an MPS kernel to CPU instead of
        # erroring out. Set only when MPS is actually selected — never on CUDA
        # or CPU hosts (sc-1332).
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def empty_torch_cache(torch: Any) -> None:
    """Release cached accelerator memory on whichever backend is active.

    Clears the CUDA allocator cache on NVIDIA and the MPS allocator cache on
    Apple Silicon; a no-op on CPU-only hosts (sc-1332).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps and mps.is_available():
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "empty_cache"):
            mps_backend.empty_cache()


def release_inference_memory(torch: Any) -> None:
    """Collect dropped references, then return cached accelerator blocks to the OS.

    A bare ``empty_cache()`` reclaims nothing while a just-dropped model/pipeline
    is still kept alive by reference cycles in its ``nn.Module`` graph (the cyclic
    collector has not run yet), and the MPS/CUDA caching allocator only returns
    freed blocks to the OS on ``empty_cache()``. Both steps are required, in order:
    collect first, then empty — otherwise an evicted multi-GB model lingers
    alongside its replacement (the cross-model accumulation that pins MPS memory).
    """
    gc.collect()
    empty_torch_cache(torch)


def release_image_worker_memory() -> None:
    """Drop transient inference buffers after a job WITHOUT evicting cached models.

    Returns the post-generation activation pool (tens of GB at large resolutions)
    to the OS so an idle worker does not sit at peak memory, while cached
    pipelines/models stay resident for fast reuse. The just-finished job's tensors
    are already unreferenced here, so ``empty_cache()`` reclaims them; the model,
    still held by the adapter, is untouched. No-op when torch is unavailable.
    """
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return
    release_inference_memory(torch)


def evict_other_image_adapters(adapters: dict[str, object], keep_id: str) -> None:
    """Enforce a single resident image model: unload every adapter except keep_id.

    Each adapter is a long-lived singleton that only evicts its OWN previous model
    on a within-family switch, so without this, running e.g. Qwen then SenseNova
    leaves both multi-GB models resident. Called before a job loads its model so
    the previous family's weights are freed first (no transient double residency).
    Adapters with nothing resident (procedural, out-of-process Lens) expose no
    unload and are skipped.
    """
    freed: list[str] = []
    for adapter_id, adapter in adapters.items():
        if adapter_id == keep_id:
            continue
        unload = getattr(adapter, "unload", None)
        if not callable(unload):
            continue
        try:
            if unload():
                freed.append(adapter_id)
        except Exception as exc:  # noqa: BLE001 - never let cleanup abort a job
            emit_worker_event("image_adapter_unload_failed", adapter=adapter_id, error=str(exc))
    if freed:
        emit_worker_event("image_adapters_evicted", keep=keep_id, evicted=freed)


def torch_inference_backend_available(torch: Any | None = None) -> bool:
    """True when a CUDA or MPS inference backend is usable.

    Callers that already imported torch pass it in; callers that haven't (e.g.
    worker capability registration before torch setup) omit it, so we import torch
    defensively and treat any failure as "no backend".
    """
    if torch is None:
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
    # bfloat16 for both CUDA and MPS. On MPS, float16 overflows to NaN and yields
    # all-black images — the denoiser latents go NaN before the VAE, so upcasting
    # only the VAE does not help. bfloat16 keeps float32's exponent range (no
    # overflow) at half float32's memory and renders correctly + fast on Apple
    # Silicon. Verified on an M-series Mac with Z-Image-Turbo: fp16 = black,
    # bf16 = OK (~18s), fp32 = OK but ~2x memory and slower. Explicit
    # float16/float32 requests above still win (sc-1336).
    return torch.bfloat16


def load_source_image(project_path: Path, request: ImageRequest) -> Image.Image:
    # Resolve only through the project sidecar/DB (find_asset_media_path constrains the
    # result to the project root). No client-supplied path escape hatch: an arbitrary
    # sourceImagePath would let an edit job read any file the worker can open.
    if not request.source_asset_id:
        raise RuntimeError("Image edit jobs require a source image asset.")
    source_path = find_asset_media_path(project_path, request.source_asset_id)
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


