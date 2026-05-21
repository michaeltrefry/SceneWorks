from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
import warnings
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, UnidentifiedImageError

from sceneworks_shared import (
    find_asset_sidecar_path,
    find_project_path,
    index_asset,
    read_json,
    safe_float,
    safe_int,
    slugify,
    utc_now,
)

from .adapter_utils import filter_call_kwargs
from .image_adapters import emit_worker_event, empty_torch_cache, gpu_memory_snapshot, require_inference_backend_for_gpu_worker, select_torch_device, select_torch_dtype, write_json
from .lora_adapters import (
    LoraPipelineState,
    LoraSpec,
    apply_loras_to_pipeline,
    lora_cache_key_for_specs,
    lora_looks_like_ic_lora,
    normalize_lora_specs,
    reject_loras_if_unsupported,
    validate_lora_compatibility,
)
from .settings import WorkerSettings


Image.MAX_IMAGE_PIXELS = 64_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]
_DelegatingBuilderT = TypeVar("_DelegatingBuilderT")


def _ltx_inference_mode():
    """torch.inference_mode() when torch is importable, else a no-op context.

    ltx-core's pipeline __call__ runs with autograd enabled (only its CLI main()
    is decorated), so direct callers must disable grad themselves or the per-step
    activation graph is retained and OOMs. Falls back to nullcontext where torch
    is unavailable (e.g. unit tests) so the adapter stays importable.
    """
    try:
        import torch

        return torch.inference_mode()
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


VIDEO_MODEL_TARGETS: dict[str, dict[str, Any]] = {
    "ltx_2_3": {
        "label": "LTX-2.3",
        "family": "ltx-video",
        "adapter": "ltx_video",
        "repo": "Lightricks/LTX-2.3",
        "fallbackRepo": "Lightricks/LTX-Video",
        "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
        "recommendedMaxDuration": 10,
        "hardMaxDuration": 15,
        "steps": {"fast": 6, "balanced": 8, "best": 20},
        "durationHint": "Best as short shots. Start with 4-8 seconds; 10 seconds is the current workflow ceiling.",
    },
    "wan_2_2": {
        "label": "Wan2.2",
        "family": "wan-video",
        "adapter": "wan_video",
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
        "recommendedMaxDuration": 7,
        "hardMaxDuration": 8,
        "steps": {"fast": 12, "balanced": 20, "best": 30},
        "durationHint": "Keep clips shorter until local looping behavior is validated.",
    },
    # Wan2.2 A14B is a mixture-of-experts checkpoint split into separate text-to-video and
    # image-to-video repos, registered here as two single-repo models. Each repo ships a
    # high-noise expert (transformer) and a low-noise expert (transformer_2); diffusers' Wan
    # pipelines load both and switch at the config boundary_ratio, so no custom dual-model
    # sampling loop is required.
    "wan_2_2_t2v_14b": {
        "label": "Wan2.2 14B (T2V)",
        "family": "wan-video",
        "adapter": "wan_video",
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "capabilities": ["text_to_video"],
        "recommendedMaxDuration": 5,
        "hardMaxDuration": 5,
        "steps": {"fast": 20, "balanced": 30, "best": 40},
        "guidanceScale": 4.0,
        "durationHint": "A14B is heavier than 5B; keep clips at 5s or less. Native cadence is 16fps.",
    },
    "wan_2_2_i2v_14b": {
        "label": "Wan2.2 14B (I2V)",
        "family": "wan-video",
        "adapter": "wan_video",
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "capabilities": ["image_to_video", "first_last_frame", "extend_clip", "video_bridge"],
        "recommendedMaxDuration": 5,
        "hardMaxDuration": 5,
        "steps": {"fast": 20, "balanced": 30, "best": 40},
        "guidanceScale": 4.0,
        # Wan2.2 A14B conditions image modes on VAE latents rather than a CLIP image encoder,
        # so the repo ships no image_encoder subfolder to pre-load.
        "noImageEncoder": True,
        "durationHint": "A14B is heavier than 5B; keep clips at 5s or less. Native cadence is 16fps.",
    },
}


@dataclass(frozen=True)
class VideoRequest:
    project_id: str
    mode: str
    prompt: str
    negative_prompt: str
    model: str
    duration: float
    fps: int
    width: int
    height: int
    quality: str
    seed: int | None
    loras: list[dict[str, Any]]
    character_id: str | None
    character_look_id: str | None
    person_track_id: str | None
    replacement_mode: str
    source_asset_id: str | None
    last_frame_asset_id: str | None
    source_clip_asset_id: str | None
    bridge_right_clip_asset_id: str | None
    advanced: dict[str, Any]


@dataclass(frozen=True)
class LtxPipelinesResources:
    checkpoint_path: Path
    spatial_upsampler_path: Path
    distilled_lora_path: Path
    gemma_root: Path
    temporal_upsampler_path: Path | None = None


@dataclass(frozen=True)
class LtxReplacementControl:
    """The shared source/control/mask package the native LTX replacement path
    consumes (sc-1483). It is the model-agnostic product object the epic calls
    for: source frames + per-frame person masks + masking strength, plus the
    masked control clip built from them. ``mask_mode`` is ``"segmentation"`` when
    real masks were loaded or ``"degraded_box"`` for the explicit box fallback.
    """

    track_id: str
    masked_clip_path: Path
    mask_mode: str
    mask_state: str
    person_tracking_active: bool
    masking_strength: float
    frame_count: int
    character_reference_count: int


def install_ltx_pipelines_multigpu_compat() -> None:
    if "ltx_pipelines.multigpu.delegating_builder" in sys.modules:
        return

    class DelegatingBuilder(Generic[_DelegatingBuilderT]):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "The installed ltx-pipelines package references an optional multigpu DelegatingBuilder "
                "that is not shipped by the package."
            )

    multigpu_module = types.ModuleType("ltx_pipelines.multigpu")
    multigpu_module.__path__ = []
    delegating_builder_module = types.ModuleType("ltx_pipelines.multigpu.delegating_builder")
    delegating_builder_module.DelegatingBuilder = DelegatingBuilder
    sys.modules.setdefault("ltx_pipelines.multigpu", multigpu_module)
    sys.modules.setdefault("ltx_pipelines.multigpu.delegating_builder", delegating_builder_module)


class VideoGenerationAdapter(ABC):
    id: str

    @abstractmethod
    def prepare(self, *, settings: WorkerSettings, job: dict[str, Any]) -> VideoRequest:
        raise NotImplementedError

    @abstractmethod
    def ensure_models(self, request: VideoRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    def estimate_requirements(self, request: VideoRequest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: VideoRequest,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, job_id: str) -> None:
        raise NotImplementedError


class ProceduralVideoAdapter(VideoGenerationAdapter):
    id = "procedural_video"

    def __init__(self) -> None:
        self._temporary_outputs: dict[str, list[Path]] = {}

    def prepare(self, *, settings: WorkerSettings, job: dict[str, Any]) -> VideoRequest:
        return video_request_from_job(job)

    def ensure_models(self, request: VideoRequest) -> None:
        reject_loras_if_unsupported(request.loras, self.id)
        target = model_target(request.model)
        if request.mode not in target["capabilities"]:
            raise RuntimeError(f"{target['label']} does not support {request.mode.replace('_', ' ')}.")
        if request.duration > target["hardMaxDuration"]:
            raise RuntimeError(f"{target['label']} is limited to {target['hardMaxDuration']}s clips in this adapter.")

    def estimate_requirements(self, request: VideoRequest) -> dict[str, Any]:
        pixels = request.width * request.height
        raw_frames = max(1, int(round(request.duration * request.fps)))
        return {
            "estimatedFrames": raw_frames,
            "previewFrames": preview_frame_count(request),
            "pixelCount": pixels,
            "recommendedMaxDuration": model_target(request.model)["recommendedMaxDuration"],
            "gpuPreference": request.advanced.get("gpuPreference", "auto"),
        }

    def run(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: VideoRequest,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        for folder in ("assets/videos", "generation-sets", "recipes"):
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        generation_set_id = f"genset_{uuid4().hex}"
        asset_id = f"asset_{uuid4().hex}"
        created_at = utc_now()
        seed = resolve_seed(request.seed, request.prompt)
        target = model_target(request.model)
        raw_settings = self.map_settings(request, target)
        filename_base = f"{created_at[:10]}_{request.model}_{slugify(request.prompt, fallback='video', max_length=42)}"
        media_rel = f"assets/videos/{filename_base}.webp"
        media_path = project_path / media_rel
        temp_path = media_path.with_suffix(".tmp.webp")
        self._temporary_outputs.setdefault(job["id"], []).append(temp_path)

        first_image = load_source_image(project_path, request.source_asset_id, request.width, request.height)
        last_image = load_source_image(project_path, request.last_frame_asset_id, request.width, request.height)
        context = request.advanced.get("timelineContext", {})
        if request.mode == "extend_clip" and first_image is None:
            timestamp = safe_float(context.get("endpointTimestamp"), 0, 0, 3600)
            first_image = load_source_frame(project_path, request.source_clip_asset_id, timestamp, request.width, request.height)
        if request.mode == "video_bridge":
            left_timestamp = safe_float(context.get("leftTimestamp"), 0, 0, 3600)
            right_timestamp = safe_float(context.get("rightTimestamp"), 0, 0, 3600)
            first_image = load_source_frame(project_path, request.source_clip_asset_id, left_timestamp, request.width, request.height)
            last_image = load_source_frame(project_path, request.bridge_right_clip_asset_id, right_timestamp, request.width, request.height)
        if request.mode == "replace_person" and first_image is None:
            first_image = load_source_frame(project_path, request.source_clip_asset_id, 0, request.width, request.height)
        if cancel_requested():
            raise InterruptedError("Video generation canceled before rendering.")

        frames = render_preview_frames(request, target, seed, first_image, last_image, progress, cancel_requested)
        save_animated_preview(frames, temp_path, request.duration)
        temp_path.replace(media_path)

        sidecar_path = media_path.with_suffix(".sceneworks.json")
        generation_set = {
            "schemaVersion": 1,
            "id": generation_set_id,
            "projectId": request.project_id,
            "jobId": job["id"],
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": 1,
            "createdAt": created_at,
        }
        asset = build_video_asset_sidecar(
            asset_id=asset_id,
            project_id=request.project_id,
            generation_set_id=generation_set_id,
            request=request,
            job_id=job["id"],
            media_rel=media_rel,
            created_at=created_at,
            seed=seed,
            target=target,
            adapter_id=self.id,
            mime_type="image/webp",
            raw_settings=raw_settings,
        )

        progress("saving", "saving", 0.9, "Saving video asset and recipe.")
        if cancel_requested():
            media_path.unlink(missing_ok=True)
            raise InterruptedError("Video generation canceled before asset promotion.")

        write_json(project_path / "generation-sets" / f"{generation_set_id}.json", generation_set)
        write_json(sidecar_path, asset)
        write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
        index_asset(project_path, asset)
        return {
            "generationSetId": generation_set_id,
            "assetIds": [asset_id],
            "assets": [asset],
            "adapter": self.id,
            "model": request.model,
            "requirements": self.estimate_requirements(request),
        }

    def cancel(self, job_id: str) -> None:
        self.cleanup(job_id)

    def cleanup(self, job_id: str) -> None:
        for path in self._temporary_outputs.pop(job_id, []):
            path.unlink(missing_ok=True)

    def map_settings(self, request: VideoRequest, target: dict[str, Any]) -> dict[str, Any]:
        steps = target["steps"].get(request.quality, target["steps"]["balanced"])
        return {
            **request.advanced,
            "adapterFamily": target["family"],
            "targetAdapter": target["adapter"],
            "steps": steps,
            "frameCount": max(1, int(round(request.duration * request.fps))),
            "previewFrameCount": preview_frame_count(request),
            "recommendedMaxDuration": target["recommendedMaxDuration"],
            "previewRenderer": True,
        }


class LtxPipelinesVideoAdapter(ProceduralVideoAdapter):
    id = "ltx_pipelines"

    _supported_modes = {"text_to_video", "image_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"}

    # ltx-core builds each stage (text encoder, transformer, upscaler, VAE) inside
    # a gpu_model() context that frees it before the next stage, so the components
    # are NOT all resident simultaneously. offload_mode controls how each stage
    # loads while active: "none" builds it fully on GPU (fast, higher per-stage
    # peak), "cpu" layer-streams it (lower peak but ~30x slower and, on WSL2 without
    # expandable_segments, prone to per-step allocator growth that OOMs mid-loop).
    # Default to resident; callers opt into streaming via advanced.offloadMode.
    _default_offload_mode = "none"

    # FP8 quantizes the transformer weights (~half their VRAM). It coexists with
    # offloading in ltx-core (layer streaming explicitly requires fp8_cast), so the
    # only real conflict is torch.compile, which streaming cannot do — see
    # _load_ltx_pipeline. Works on the bf16 checkpoints we ship (cast at load,
    # which transiently holds bf16+fp8 — a known load-time peak we're profiling).
    _default_precision = "fp8"

    def __init__(self) -> None:
        super().__init__()
        self._loaded_models: set[str] = set()
        self._settings: WorkerSettings | None = None
        self._resources_by_model: dict[str, LtxPipelinesResources] = {}
        self._lora_specs_by_request_id: dict[int, list[LoraSpec]] = {}
        self._pipeline: Any | None = None
        self._pipeline_key_value: str | None = None

    def loaded_models(self) -> list[str]:
        return sorted(self._loaded_models)

    def prepare(self, *, settings: WorkerSettings, job: dict[str, Any]) -> VideoRequest:
        self._settings = settings
        self._lora_specs_by_request_id.clear()
        return video_request_from_job(job)

    def ensure_models(self, request: VideoRequest) -> None:
        target = model_target(request.model)
        if target["adapter"] != "ltx_video":
            raise RuntimeError("The native LTX pipelines adapter only supports LTX-family video models.")
        if request.mode not in self._supported_modes:
            supported = ", ".join(sorted(mode.replace("_", " ") for mode in self._supported_modes))
            raise RuntimeError(f"{target['label']} native pipelines currently support {supported}.")
        if request.duration > target["hardMaxDuration"]:
            raise RuntimeError(f"{target['label']} is limited to {target['hardMaxDuration']}s clips in this adapter.")
        validate_lora_compatibility(request.loras, model_family=target["family"], adapter_id=self.id)
        self._ltx_lora_specs(request)
        if request.mode == "replace_person" and not self._mock_inference_enabled(request):
            self._validate_replacement_inputs(request)
        if self._uses_ic_lora_pipeline(request) and not self._has_ic_lora(request) and not self._mock_inference_enabled(request):
            raise RuntimeError(
                "Native LTX IC-LoRA video conditioning requires at least one installed LTX-compatible LoRA. "
                "Add an IC-LoRA to the selected preset before running source-video conditioning."
            )
        resources = self.resolve_resources(request)
        missing = self._missing_resources(request, resources)
        if missing:
            details = "\n".join(f"- {label}: {path}" for label, path in missing)
            search_details = self._missing_resource_search_details(request, resources, missing)
            if search_details:
                details = f"{details}\nSearched Hugging Face cache paths:\n{search_details}"
            override_keys = ["checkpointPath", "spatialUpscalerPath", "gemmaRoot"]
            if self._pipeline_module(request) == "ltx_pipelines.ti2vid_two_stages":
                override_keys.insert(2, "distilledLoraPath")
            raise RuntimeError(
                "Native LTX-2.3 requires local model resources before generation. "
                "Install the LTX-2.3 model resources in Model Manager or set advanced overrides "
                f"for {', '.join(override_keys)}.\n"
                f"Missing resources:\n{details}"
            )
        self._resources_by_model[request.model] = resources
        if not self._mock_inference_enabled(request) and not self._dependencies_available(request):
            raise RuntimeError(
                "Native LTX-2.3 generation requires optional worker dependencies. "
                "Install apps/worker/requirements-ltx.txt in this worker environment, rebuild the worker image, or use "
                "advanced.mockNativeInference for local adapter smoke tests."
            )

    def estimate_requirements(self, request: VideoRequest) -> dict[str, Any]:
        target = model_target(request.model)
        raw_frames = max(1, int(round(request.duration * request.fps)))
        resources = self._resources_by_model.get(request.model) or self.resolve_resources(request)
        mocked = self._mock_inference_enabled(request)
        return {
            "estimatedFrames": ltx_frame_count(raw_frames),
            "requestedFrames": raw_frames,
            "previewFrames": preview_frame_count(request),
            "pixelCount": request.width * request.height,
            "recommendedMaxDuration": target["recommendedMaxDuration"],
            "gpuPreference": request.advanced.get("gpuPreference", "auto"),
            "adapter": self.id,
            "pipeline": self._pipeline_module(request),
            "resources": self._resource_summary(resources),
            "nativeDependenciesAvailable": self._dependencies_available(request),
            "mockedInference": mocked,
        }

    def run(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: VideoRequest,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        if not self._mock_inference_enabled(request):
            return self._run_real_ltx_video(
                settings=settings,
                job=job,
                request=request,
                progress=progress,
                cancel_requested=cancel_requested,
            )

        self._loaded_models.update({request.model, self._pipeline_module(request)})
        progress("running", "mocking_native_ltx", 0.2, "Rendering mocked native LTX-2.3 preview clip.")
        return super().run(
            settings=settings,
            job=job,
            request=request,
            progress=progress,
            cancel_requested=cancel_requested,
        )

    def cleanup(self, job_id: str) -> None:
        super().cleanup(job_id)
        self._loaded_models.clear()
        self._evict_pipeline()

    def map_settings(self, request: VideoRequest, target: dict[str, Any]) -> dict[str, Any]:
        steps = target["steps"].get(request.quality, target["steps"]["balanced"])
        resources = self._resources_by_model.get(request.model) or self.resolve_resources(request)
        mocked = self._mock_inference_enabled(request)
        return {
            **request.advanced,
            "adapterFamily": target["family"],
            "targetAdapter": self.id,
            "pipeline": self._pipeline_module(request),
            "resources": self._resource_summary(resources),
            "icLoraConditioning": self._uses_ic_lora_pipeline(request),
            "steps": safe_int(request.advanced.get("steps"), steps, 1, 80),
            "frameCount": ltx_frame_count(max(1, int(round(request.duration * request.fps)))),
            "previewFrameCount": preview_frame_count(request),
            "recommendedMaxDuration": target["recommendedMaxDuration"],
            "previewRenderer": mocked,
            "mockedNativeInference": mocked,
            "realModelInference": not mocked,
        }

    def _ltx_mem_profile_begin(self, job_id: str) -> None:
        """Reset peak GPU stats before a run; start allocation history if profiling."""
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is None or not cuda.is_available():
                return
            cuda.reset_peak_memory_stats()
            if os.environ.get("SCENEWORKS_LTX_MEM_PROFILE"):
                try:
                    cuda.memory._record_memory_history(max_entries=200_000)
                except Exception:
                    pass
        except Exception:
            pass

    def _ltx_mem_event(
        self,
        stage: str,
        job_id: str,
        *,
        request: VideoRequest | None = None,
        num_frames: int | None = None,
        error: str | None = None,
    ) -> None:
        """Emit current + peak GPU memory so we can localize the LTX footprint."""
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is None or not cuda.is_available():
                return
            mb = lambda b: round(b / (1024 * 1024), 2)
            gpu_memory = {
                "allocatedMb": mb(cuda.memory_allocated()),
                "reservedMb": mb(cuda.memory_reserved()),
                "maxAllocatedMb": mb(cuda.max_memory_allocated()),
                "maxReservedMb": mb(cuda.max_memory_reserved()),
            }
            dims = None
            if request is not None:
                dims = {
                    "width": request.width,
                    "height": request.height,
                    "frames": num_frames,
                    "precision": str(request.advanced.get("precision", self._default_precision)),
                    "offload": str(request.advanced.get("offloadMode", self._default_offload_mode)),
                }
            emit_worker_event("ltx_gpu_memory", jobId=job_id, stage=stage, gpuMemory=gpu_memory, dims=dims, error=error)
        except Exception:
            pass

    def _ltx_mem_dump_snapshot(self, settings: WorkerSettings, job_id: str) -> None:
        """Dump a CUDA allocation snapshot (analyzable at pytorch.org/memory_viz)."""
        if not os.environ.get("SCENEWORKS_LTX_MEM_PROFILE"):
            return
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is None or not cuda.is_available():
                return
            out = Path(settings.data_dir) / "cache" / f"ltx_mem_{job_id}.pickle"
            out.parent.mkdir(parents=True, exist_ok=True)
            cuda.memory._dump_snapshot(str(out))
            emit_worker_event("ltx_mem_snapshot", jobId=job_id, path=str(out))
        except Exception:
            pass

    def _run_real_ltx_video(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: VideoRequest,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        for folder in ("assets/videos", "generation-sets", "recipes"):
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        target = model_target(request.model)
        resources = self._resources_by_model.get(request.model) or self.resolve_resources(request)
        seed = resolve_seed(request.seed, request.prompt)
        num_frames = ltx_frame_count(max(1, int(round(request.duration * request.fps))))
        generation_set_id = f"genset_{uuid4().hex}"
        asset_id = f"asset_{uuid4().hex}"
        created_at = utc_now()
        filename_base = f"{created_at[:10]}_{request.model}_{slugify(request.prompt, fallback='video', max_length=42)}"
        media_rel = f"assets/videos/{filename_base}.mp4"
        media_path = project_path / media_rel
        temp_path = media_path.with_suffix(".tmp.mp4")
        self._temporary_outputs.setdefault(job["id"], []).append(temp_path)

        progress("preparing", "validating_inputs", 0.2, "Validating native LTX-2.3 inputs.")
        replacement_control: LtxReplacementControl | None = None
        replacement_status: dict[str, Any] | None = None
        if request.mode == "replace_person":
            control_clip = media_path.with_suffix(".control.mp4")
            self._temporary_outputs.setdefault(job["id"], []).append(control_clip)
            progress("preparing", "building_control", 0.24, "Building masked control clip from person track.")
            replacement_control = self._ltx_replacement_control(project_path, request, num_frames, control_clip)
            conditioning_images = []
            video_conditioning = [(str(replacement_control.masked_clip_path), replacement_control.masking_strength)]
        else:
            conditioning_images = self._ltx_conditioning_images(project_path, request, num_frames)
            video_conditioning = self._ltx_video_conditioning(project_path, request)
        if cancel_requested():
            raise InterruptedError("Video generation canceled before native LTX pipeline load.")

        self._ltx_mem_profile_begin(job["id"])
        progress("loading_model", "loading_model", 0.28, "Loading native LTX-2.3 pipeline.")
        try:
            pipeline = self._load_ltx_pipeline(request, resources)
            self._ltx_mem_event("after_load", job["id"], request=request, num_frames=num_frames)
            if cancel_requested():
                raise InterruptedError("Video generation canceled before native LTX inference.")

            progress("running", "generating", 0.4, "Running native LTX-2.3 inference.")
            video, audio, video_chunks_number, encode_video = self._run_ltx_pipeline(
                pipeline=pipeline,
                request=request,
                resources=resources,
                seed=seed,
                num_frames=num_frames,
                conditioning_images=conditioning_images,
                video_conditioning=video_conditioning,
            )
            self._ltx_mem_event("after_run", job["id"], request=request, num_frames=num_frames)
        except InterruptedError:
            raise
        except Exception as exc:
            self._ltx_mem_event("on_failure", job["id"], request=request, num_frames=num_frames, error=str(exc))
            self._ltx_mem_dump_snapshot(settings, job["id"])
            raise
        if cancel_requested():
            raise InterruptedError("Video generation canceled before saving.")

        progress("saving", "saving", 0.9, "Saving native LTX-2.3 MP4 asset and recipe.")
        # The decoded video is a lazy iterator of inference tensors produced under
        # inference_mode in _run_ltx_pipeline; decode/encode it under the same mode
        # so the VAE decode stays grad-free and the inference tensors remain usable.
        with _ltx_inference_mode():
            encode_video(
                video=video,
                fps=request.fps,
                audio=audio,
                output_path=str(temp_path),
                video_chunks_number=video_chunks_number,
            )
        temp_path.replace(media_path)

        if replacement_control is not None:
            # replacementActive is true ONLY here: the masked-control package was
            # built from a real person track and run through the native LTX path.
            replacement_status = {
                "personDetectionActive": True,
                "personTrackingActive": replacement_control.person_tracking_active,
                "replacementActive": True,
                "replacementAdapter": self.id,
                "maskMode": replacement_control.mask_mode,
                "maskState": replacement_control.mask_state,
                "maskingStrength": replacement_control.masking_strength,
                "personTrackId": replacement_control.track_id,
                "characterReferenceCount": replacement_control.character_reference_count,
                "controlFrameCount": replacement_control.frame_count,
            }

        generation_set = {
            "schemaVersion": 1,
            "id": generation_set_id,
            "projectId": request.project_id,
            "jobId": job["id"],
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": 1,
            "createdAt": created_at,
        }
        asset = build_video_asset_sidecar(
            asset_id=asset_id,
            project_id=request.project_id,
            generation_set_id=generation_set_id,
            request=request,
            job_id=job["id"],
            media_rel=media_rel,
            created_at=created_at,
            seed=seed,
            target=target,
            adapter_id=self.id,
            mime_type="video/mp4",
            raw_settings=self.map_settings(request, target),
            replacement_status=replacement_status,
        )
        write_json(project_path / "generation-sets" / f"{generation_set_id}.json", generation_set)
        write_json(media_path.with_suffix(".sceneworks.json"), asset)
        write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
        index_asset(project_path, asset)
        self._loaded_models.update({request.model, self._pipeline_module(request), str(resources.checkpoint_path)})
        return {
            "generationSetId": generation_set_id,
            "assetIds": [asset_id],
            "assets": [asset],
            "adapter": self.id,
            "model": request.model,
            "requirements": self.estimate_requirements(request),
        }

    def _ltx_conditioning_images(self, project_path: Path, request: VideoRequest, num_frames: int) -> list[Any]:
        if request.mode in {"text_to_video", "extend_clip", "video_bridge"}:
            return []
        if request.mode not in {"image_to_video", "first_last_frame"}:
            raise RuntimeError(f"Native LTX-2.3 does not support {request.mode.replace('_', ' ')} yet.")

        media_path = source_asset_media_path(project_path, request.source_asset_id)
        if media_path is None:
            raise RuntimeError(f"{request.mode.replace('_', ' ').title()} requires a readable source image.")
        try:
            with Image.open(media_path) as image:
                image.verify()
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise RuntimeError("Image to Video requires a readable source image.") from exc

        install_ltx_pipelines_multigpu_compat()
        args_module = importlib.import_module("ltx_pipelines.utils.args")
        condition_class = getattr(args_module, "ImageConditioningInput")
        images = [
            condition_class(
                str(media_path),
                safe_int(request.advanced.get("imageFrameIndex"), 0, 0, 1_000_000),
                self._advanced_float(request, "imageConditioningStrength", 1.0),
            )
        ]
        if request.mode == "first_last_frame":
            last_media_path = source_asset_media_path(project_path, request.last_frame_asset_id)
            if last_media_path is None:
                raise RuntimeError("First/Last Frame requires a readable last frame image.")
            try:
                with Image.open(last_media_path) as image:
                    image.verify()
            except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise RuntimeError("First/Last Frame requires a readable last frame image.") from exc
            images.append(
                condition_class(
                    str(last_media_path),
                    max(0, num_frames - 1),
                    self._advanced_float(request, "lastFrameConditioningStrength", 1.0),
                )
            )
        return images

    def _ltx_video_conditioning(self, project_path: Path, request: VideoRequest) -> list[tuple[str, float]]:
        if request.mode not in {"extend_clip", "video_bridge"}:
            return []
        conditionings: list[tuple[str, float]] = []
        left_path = source_asset_media_path(project_path, request.source_clip_asset_id)
        if left_path is None:
            raise RuntimeError(f"{request.mode.replace('_', ' ').title()} requires a readable source clip.")
        conditionings.append((str(left_path), self._advanced_float(request, "videoConditioningStrength", 1.0)))
        if request.mode == "video_bridge":
            right_path = source_asset_media_path(project_path, request.bridge_right_clip_asset_id)
            if right_path is None:
                raise RuntimeError("Video Bridge requires a readable right-side source clip.")
            conditionings.append((str(right_path), self._advanced_float(request, "bridgeRightVideoConditioningStrength", 1.0)))
        return conditionings

    def _validate_replacement_inputs(self, request: VideoRequest) -> None:
        """Fail clearly when the required Replace Person inputs are missing,
        before loading the LTX pipeline."""
        settings = self._settings or WorkerSettings()
        project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        if source_asset_media_path(project_path, request.source_clip_asset_id) is None:
            raise RuntimeError("Replace Person requires a readable source clip.")
        track = read_person_track(project_path, request.person_track_id)
        if not track:
            raise RuntimeError(
                "Replace Person requires a saved person track. Run detection and tracking on the source clip first."
            )
        if not track.get("frames") and not track.get("selectedDetection", {}).get("box"):
            raise RuntimeError("The selected person track has no usable boxes or masks.")
        if not character_reference_images(
            project_path, request.character_id, request.character_look_id, request.width, request.height
        ):
            raise RuntimeError("Replace Person requires at least one approved character reference image.")

    def _ltx_replacement_control(
        self,
        project_path: Path,
        request: VideoRequest,
        num_frames: int,
        clip_path: Path,
    ) -> LtxReplacementControl:
        """Assemble the source/control/mask package and write a masked control
        clip that bakes the person masks into the source frames (the masked
        region is neutralized by ``maskingStrength`` so the IC-LoRA path
        regenerates it). This is the SceneWorks-native equivalent of Wan2GP's
        masked control-video injection."""
        from .person_adapters import load_track_masks  # local: avoids an import cycle

        track = read_person_track(project_path, request.person_track_id)
        if not track:
            raise RuntimeError("Replace Person requires a saved person track.")
        source_frames = load_source_video_frames(
            project_path, request.source_clip_asset_id, request.width, request.height, num_frames
        )
        masks, mask_mode = load_track_masks(project_path, track, request.width, request.height, len(source_frames))
        references = character_reference_images(
            project_path, request.character_id, request.character_look_id, request.width, request.height
        )
        if not references:
            raise RuntimeError("Replace Person requires at least one approved character reference image.")
        masking_strength = self._advanced_float(request, "maskingStrength", 1.0)
        masked_frames = [
            self._apply_replacement_mask(frame, mask, masking_strength)
            for frame, mask in zip(source_frames, masks)
        ]
        self._write_control_clip(masked_frames, clip_path, request.fps)
        status = track.get("status", {}) if isinstance(track.get("status"), dict) else {}
        return LtxReplacementControl(
            track_id=str(track.get("id") or request.person_track_id),
            masked_clip_path=clip_path,
            mask_mode=mask_mode,
            mask_state=str(status.get("maskState", "missing")),
            person_tracking_active=bool(status.get("personTrackingActive", False)),
            masking_strength=masking_strength,
            frame_count=len(masked_frames),
            character_reference_count=len(references),
        )

    def _write_control_clip(self, frames: list[Image.Image], path: Path, fps: int) -> None:
        """Write the masked control frames as a clip the native LTX video
        conditioner can decode. LTX reads ``video_conditioning`` with PyAV/ffmpeg,
        which cannot frame-decode an animated WebP, so real runs (``.mp4``) are
        encoded with imageio/ffmpeg; tests pass ``.webp`` for a Pillow-only host.
        """
        if path.suffix.lower() in {".webp", ".gif"}:
            save_animated_preview(frames, path, max(1.0, len(frames) / max(1, fps)))
            return
        import imageio
        import numpy as np

        writer = imageio.get_writer(str(path), fps=max(1, int(fps)), codec="libx264", macro_block_size=None)
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame.convert("RGB")))
        finally:
            writer.close()

    def _apply_replacement_mask(self, frame: Image.Image, mask: Image.Image, strength: float) -> Image.Image:
        """Blend the masked person region toward neutral gray by ``strength`` so
        the control video preserves the background and clears the replacement
        region for regeneration."""
        strength = max(0.0, min(1.0, strength))
        neutral = Image.new("RGB", frame.size, (118, 118, 118))
        gate = mask.convert("L").resize(frame.size).point(lambda value: int(value * strength))
        return Image.composite(neutral, frame.convert("RGB"), gate)

    def _load_ltx_pipeline(self, request: VideoRequest, resources: LtxPipelinesResources) -> Any:
        key = self._pipeline_key(request, resources)
        if self._pipeline is not None and self._pipeline_key_value == key:
            return self._pipeline

        install_ltx_pipelines_multigpu_compat()
        loader = importlib.import_module("ltx_core.loader")
        quantization = self._quantization(request)
        torch_compile = bool(request.advanced.get("compile", False))
        # FP8 and CPU/disk offload coexist: at the pinned ltx-core commit, layer
        # streaming explicitly requires QuantizationPolicy.fp8_cast() (DiffusionStage
        # chains it into the streaming ops) and the Gemma encoder has its own streaming
        # builder, so offloading streams the fp8 transformer and frees the ~23 GB text
        # encoder after prompt encoding instead of pinning it resident. Streaming is the
        # one thing torch.compile cannot do, so only force offload off when compile is on.
        offload_mode = self._offload_mode(request, override="none" if torch_compile else None)
        loras = self._ltx_loras(loader, request)
        # The native Lightricks constructors expose `loras` across distilled,
        # two-stage, and IC-LoRA pipelines; `distilled_lora` remains separate.
        common_kwargs = {
            "gemma_root": str(resources.gemma_root),
            "spatial_upsampler_path": str(resources.spatial_upsampler_path),
            "loras": loras,
            "quantization": quantization,
            "torch_compile": torch_compile,
            "offload_mode": offload_mode,
        }
        if self._pipeline_module(request) == "ltx_pipelines.ic_lora":
            pipeline_module = importlib.import_module("ltx_pipelines.ic_lora")
            pipeline = pipeline_module.ICLoraPipeline(
                distilled_checkpoint_path=str(resources.checkpoint_path),
                **common_kwargs,
            )
        elif self._pipeline_module(request) == "ltx_pipelines.distilled":
            pipeline_module = importlib.import_module("ltx_pipelines.distilled")
            pipeline = pipeline_module.DistilledPipeline(
                distilled_checkpoint_path=str(resources.checkpoint_path),
                **common_kwargs,
            )
        else:
            pipeline_module = importlib.import_module("ltx_pipelines.ti2vid_two_stages")
            lora_spec = loader.LoraPathStrengthAndSDOps(
                str(resources.distilled_lora_path),
                float(request.advanced.get("distilledLoraStrength", 0.8)),
                loader.LTXV_LORA_COMFY_RENAMING_MAP,
            )
            pipeline = pipeline_module.TI2VidTwoStagesPipeline(
                checkpoint_path=str(resources.checkpoint_path),
                distilled_lora=[lora_spec],
                **common_kwargs,
            )
        self._pipeline = pipeline
        self._pipeline_key_value = key
        self._loaded_models.update({request.model, self._pipeline_module(request), str(resources.checkpoint_path)})
        return pipeline

    def _run_ltx_pipeline(
        self,
        *,
        pipeline: Any,
        request: VideoRequest,
        resources: LtxPipelinesResources,
        seed: int,
        num_frames: int,
        conditioning_images: list[Any],
        video_conditioning: list[tuple[str, float]],
    ) -> tuple[Any, Any, int, Any]:
        install_ltx_pipelines_multigpu_compat()
        video_vae = importlib.import_module("ltx_core.model.video_vae")
        media_io = importlib.import_module("ltx_pipelines.utils.media_io")
        tiling_config = video_vae.TilingConfig.default()
        video_chunks_number = video_vae.get_video_chunks_number(num_frames, tiling_config)
        base_kwargs = {
            "prompt": request.prompt,
            "seed": seed,
            "height": request.height,
            "width": request.width,
            "num_frames": num_frames,
            "frame_rate": request.fps,
            "images": conditioning_images,
            "tiling_config": tiling_config,
            "enhance_prompt": bool(request.advanced.get("enhancePrompt", False)),
        }
        # ltx-core's pipeline __call__ does NOT disable autograd (only its CLI main()
        # is @torch.inference_mode()); calling it directly with grad enabled retains
        # the full activation graph across every diffusion step (~80-100GB/step) and
        # OOMs. Run inference under inference_mode, matching the diffusers adapter.
        with _ltx_inference_mode():
            if self._pipeline_module(request) == "ltx_pipelines.ic_lora":
                video, audio = pipeline(
                    **base_kwargs,
                    video_conditioning=video_conditioning,
                    conditioning_attention_strength=self._advanced_float(request, "conditioningAttentionStrength", 1.0),
                    skip_stage_2=bool(request.advanced.get("skipStage2", False)),
                    conditioning_attention_mask=None,
                )
            elif self._pipeline_module(request) == "ltx_pipelines.distilled":
                video, audio = pipeline(**base_kwargs)
            else:
                guiders = importlib.import_module("ltx_core.components.guiders")
                video, audio = pipeline(
                    **base_kwargs,
                    negative_prompt=request.negative_prompt or default_negative_prompt(model_target(request.model)),
                    num_inference_steps=self._num_inference_steps(request, model_target(request.model)),
                    video_guider_params=guiders.MultiModalGuiderParams(
                        cfg_scale=self._advanced_float(request, "videoCfgGuidanceScale", 4.0),
                        stg_scale=self._advanced_float(request, "videoStgGuidanceScale", 0.0),
                        rescale_scale=self._advanced_float(request, "videoRescaleScale", 0.7),
                        modality_scale=self._advanced_float(request, "a2vGuidanceScale", 1.0),
                        skip_step=safe_int(request.advanced.get("videoSkipStep"), 0, 0, 80),
                        stg_blocks=request.advanced.get("videoStgBlocks", []),
                    ),
                    audio_guider_params=guiders.MultiModalGuiderParams(
                        cfg_scale=self._advanced_float(request, "audioCfgGuidanceScale", 1.0),
                        stg_scale=self._advanced_float(request, "audioStgGuidanceScale", 0.0),
                        rescale_scale=self._advanced_float(request, "audioRescaleScale", 0.0),
                        modality_scale=self._advanced_float(request, "v2aGuidanceScale", 1.0),
                        skip_step=safe_int(request.advanced.get("audioSkipStep"), 0, 0, 80),
                        stg_blocks=request.advanced.get("audioStgBlocks", []),
                    ),
                    max_batch_size=safe_int(request.advanced.get("maxBatchSize"), 1, 1, 16),
                )
        return video, audio, video_chunks_number, media_io.encode_video

    def _pipeline_module(self, request: VideoRequest) -> str:
        if self._uses_ic_lora_pipeline(request):
            return "ltx_pipelines.ic_lora"
        override = str(request.advanced.get("ltxPipeline", "auto")).strip().lower()
        if override == "distilled":
            return "ltx_pipelines.distilled"
        if override in {"two_stage", "two-stage", "ti2vid", "ti2vid_two_stages"}:
            return "ltx_pipelines.ti2vid_two_stages"
        if request.quality == "fast":
            return "ltx_pipelines.distilled"
        return "ltx_pipelines.ti2vid_two_stages"

    def _uses_ic_lora_pipeline(self, request: VideoRequest) -> bool:
        if bool(request.advanced.get("useIcLoraPipeline", False)):
            return True
        # Replace Person rides the IC-LoRA video-conditioning path: the masked
        # control clip is injected as video_conditioning during denoising.
        if request.mode in {"extend_clip", "video_bridge", "replace_person"}:
            return True
        return request.mode in {"image_to_video", "first_last_frame"} and self._has_ic_lora(request)

    def _has_ic_lora(self, request: VideoRequest) -> bool:
        return any(lora_looks_like_ic_lora(lora if isinstance(lora, dict) else {"id": str(lora)}) for lora in request.loras)

    def _num_inference_steps(self, request: VideoRequest, target: dict[str, Any]) -> int:
        default_steps = target["steps"].get(request.quality, target["steps"]["balanced"])
        return safe_int(request.advanced.get("steps"), default_steps, 1, 80)

    def _mock_inference_enabled(self, request: VideoRequest) -> bool:
        return bool(request.advanced.get("mockNativeInference", False))

    def _pipeline_key(self, request: VideoRequest, resources: LtxPipelinesResources) -> str:
        return ":".join(
            [
                self._pipeline_module(request),
                str(resources.checkpoint_path),
                str(resources.spatial_upsampler_path),
                str(resources.distilled_lora_path),
                str(resources.gemma_root),
                str(request.advanced.get("offloadMode", self._default_offload_mode)),
                str(request.advanced.get("precision", self._default_precision)),
                lora_cache_key_for_specs(self._ltx_lora_specs(request)) if request.loras else "",
            ]
        )

    def _ltx_lora_specs(self, request: VideoRequest) -> list[LoraSpec]:
        cache_key = id(request)
        specs = self._lora_specs_by_request_id.get(cache_key)
        if specs is None:
            specs = normalize_lora_specs(request.loras)
            self._lora_specs_by_request_id[cache_key] = specs
        return specs

    def _ltx_loras(self, loader: Any, request: VideoRequest) -> tuple[Any, ...]:
        specs = self._ltx_lora_specs(request)
        return tuple(
            loader.LoraPathStrengthAndSDOps(
                spec.path,
                spec.weight,
                loader.LTXV_LORA_COMFY_RENAMING_MAP,
            )
            for spec in specs
        )

    def _distilled_variant(self, request: VideoRequest) -> str | None:
        value = request.advanced.get("distilledVariant")
        if value in (None, ""):
            return None
        return str(value).strip()

    def _apply_distilled_variant(self, resources: dict[str, Any], variant: str | None) -> dict[str, Any]:
        if not variant:
            return resources
        updated = dict(resources)
        for key in ("distilledCheckpoint", "distilledLora"):
            entry = updated.get(key)
            if not isinstance(entry, dict):
                continue
            variants = entry.get("variants")
            if isinstance(variants, dict) and variant in variants:
                updated[key] = {**entry, "file": variants[variant]}
        return updated

    def _offload_mode(self, request: VideoRequest, *, override: str | None = None) -> Any:
        raw = override if override is not None else request.advanced.get("offloadMode", self._default_offload_mode)
        offload_value = str(raw).strip().lower()
        install_ltx_pipelines_multigpu_compat()
        types_module = importlib.import_module("ltx_pipelines.utils.types")
        offload_mode = getattr(types_module, "OffloadMode")
        if offload_value == "cpu":
            return offload_mode.CPU
        if offload_value == "disk":
            return offload_mode.DISK
        return offload_mode.NONE

    def _quantization(self, request: VideoRequest) -> Any:
        precision = str(request.advanced.get("precision", self._default_precision)).strip().lower()
        if precision != "fp8":
            return None
        install_ltx_pipelines_multigpu_compat()
        quant_module = importlib.import_module("ltx_core.quantization")
        return quant_module.QuantizationPolicy.fp8_cast()

    def _advanced_float(self, request: VideoRequest, key: str, fallback: float) -> float:
        try:
            return float(request.advanced.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    def _evict_pipeline(self) -> None:
        self._pipeline = None
        self._pipeline_key_value = None
        # Drop reference cycles (e.g. frames retained by a failed run) before
        # asking the CUDA allocator to release blocks — empty_cache() can only
        # reclaim memory that is no longer referenced by any live Python object.
        gc.collect()
        try:
            torch = importlib.import_module("torch")
            empty_torch_cache(torch)
        except Exception:
            return

    def resolve_resources(self, request: VideoRequest) -> LtxPipelinesResources:
        settings = self._settings or WorkerSettings()
        entry = ltx_model_manifest_entry(settings, request.model)
        resources = entry.get("resources", {}) if isinstance(entry.get("resources"), dict) else {}
        resources = self._apply_distilled_variant(resources, self._distilled_variant(request))
        checkpoint_resource_key = "distilledCheckpoint" if self._pipeline_module(request) in {"ltx_pipelines.distilled", "ltx_pipelines.ic_lora"} else "checkpoint"
        return LtxPipelinesResources(
            checkpoint_path=self._resource_path(
                settings,
                request,
                resources,
                checkpoint_resource_key,
                "checkpointPath",
            ),
            spatial_upsampler_path=self._resource_path(
                settings,
                request,
                resources,
                "spatialUpscaler",
                "spatialUpscalerPath",
            ),
            distilled_lora_path=self._resource_path(
                settings,
                request,
                resources,
                "distilledLora",
                "distilledLoraPath",
            ),
            gemma_root=self._resource_path(
                settings,
                request,
                resources,
                "gemma",
                "gemmaRoot",
                expect_file=False,
            ),
            temporal_upsampler_path=self._optional_resource_path(
                settings,
                request,
                resources,
                "temporalUpscaler",
                "temporalUpscalerPath",
            ),
        )

    def _resource_path(
        self,
        settings: WorkerSettings,
        request: VideoRequest,
        resources: dict[str, Any],
        resource_key: str,
        advanced_key: str,
        *,
        expect_file: bool = True,
    ) -> Path:
        override = request.advanced.get(advanced_key)
        if override:
            return resolve_worker_path(settings, override)
        resource = resources.get(resource_key) if isinstance(resources.get(resource_key), dict) else {}
        if not resource and resource_key == "distilledCheckpoint":
            resource = resources.get("checkpoint") if isinstance(resources.get("checkpoint"), dict) else {}
        configured_path = resource.get("path")
        if configured_path:
            return resolve_manifest_path(settings, configured_path)
        repo = str(resource.get("repo") or VIDEO_MODEL_TARGETS["ltx_2_3"]["repo"])
        root = settings.data_dir / "models" / safe_download_dir(repo)
        file_name = resource.get("file")
        if expect_file and file_name:
            local_path = root / str(file_name)
            if local_path.is_file():
                return local_path
            cached_path = huggingface_cached_resource_file(settings, repo, str(file_name))
            if cached_path is not None:
                return cached_path
            return local_path
        if not expect_file and root.is_dir():
            return root
        cached_root = huggingface_cached_snapshot_dir(settings, repo)
        if cached_root is not None:
            return cached_root
        return root

    def _optional_resource_path(
        self,
        settings: WorkerSettings,
        request: VideoRequest,
        resources: dict[str, Any],
        resource_key: str,
        advanced_key: str,
    ) -> Path | None:
        override = request.advanced.get(advanced_key)
        if override:
            return resolve_worker_path(settings, override)
        resource = resources.get(resource_key) if isinstance(resources.get(resource_key), dict) else None
        if not resource:
            return None
        return self._resource_path(settings, request, resources, resource_key, advanced_key)

    def _missing_resources(self, request: VideoRequest, resources: LtxPipelinesResources) -> list[tuple[str, Path]]:
        required = [
            ("checkpointPath", resources.checkpoint_path, "file"),
            ("spatialUpscalerPath", resources.spatial_upsampler_path, "file"),
            ("gemmaRoot", resources.gemma_root, "dir"),
        ]
        if self._pipeline_module(request) == "ltx_pipelines.ti2vid_two_stages":
            required.insert(2, ("distilledLoraPath", resources.distilled_lora_path, "file"))
        missing = [
            (label, path)
            for label, path, kind in required
            if not (path.is_file() if kind == "file" else path.is_dir())
        ]
        if resources.temporal_upsampler_path is not None and not resources.temporal_upsampler_path.is_file():
            missing.append(("temporalUpscalerPath", resources.temporal_upsampler_path))
        return missing

    def _missing_resource_search_details(
        self,
        request: VideoRequest,
        resources: LtxPipelinesResources,
        missing: list[tuple[str, Path]],
    ) -> str:
        settings = self._settings or WorkerSettings()
        entry = ltx_model_manifest_entry(settings, request.model)
        manifest_resources = entry.get("resources", {}) if isinstance(entry.get("resources"), dict) else {}
        resource_names = {
            "checkpointPath": "distilledCheckpoint" if self._pipeline_module(request) in {"ltx_pipelines.distilled", "ltx_pipelines.ic_lora"} else "checkpoint",
            "spatialUpscalerPath": "spatialUpscaler",
            "distilledLoraPath": "distilledLora",
            "temporalUpscalerPath": "temporalUpscaler",
        }
        details: list[str] = []
        for label, _path in missing:
            if label == "gemmaRoot":
                repo = self._resource_repo(manifest_resources, "gemma")
                paths = huggingface_cached_snapshot_search_paths(settings, repo)
            else:
                resource_name = resource_names.get(label)
                if resource_name is None:
                    continue
                resource = manifest_resources.get(resource_name) if isinstance(manifest_resources.get(resource_name), dict) else {}
                if not resource and resource_name == "distilledCheckpoint":
                    resource = manifest_resources.get("checkpoint") if isinstance(manifest_resources.get("checkpoint"), dict) else {}
                file_name = resource.get("file")
                if not file_name:
                    continue
                repo = self._resource_repo(manifest_resources, resource_name)
                paths = huggingface_cached_resource_search_paths(settings, repo, str(file_name))
            details.extend(f"- {label}: {path}" for path in paths)
        return "\n".join(details)

    def _resource_repo(self, resources: dict[str, Any], resource_key: str) -> str:
        resource = resources.get(resource_key) if isinstance(resources.get(resource_key), dict) else {}
        if not resource and resource_key == "distilledCheckpoint":
            resource = resources.get("checkpoint") if isinstance(resources.get("checkpoint"), dict) else {}
        return str(resource.get("repo") or VIDEO_MODEL_TARGETS["ltx_2_3"]["repo"])

    def _resource_summary(self, resources: LtxPipelinesResources) -> dict[str, str | None]:
        return {
            "checkpointPath": str(resources.checkpoint_path),
            "spatialUpscalerPath": str(resources.spatial_upsampler_path),
            "distilledLoraPath": str(resources.distilled_lora_path),
            "gemmaRoot": str(resources.gemma_root),
            "temporalUpscalerPath": str(resources.temporal_upsampler_path) if resources.temporal_upsampler_path else None,
        }

    def _dependencies_available(self, request: VideoRequest | None = None) -> bool:
        try:
            if importlib.util.find_spec("ltx_core") is None or importlib.util.find_spec("ltx_pipelines") is None:
                return False
            install_ltx_pipelines_multigpu_compat()
            importlib.import_module(self._pipeline_module(request) if request is not None else "ltx_pipelines.distilled")
            return True
        except (ImportError, ValueError):
            return False


class DiffusersVideoAdapter(VideoGenerationAdapter):
    id = "diffusers_video"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._pipeline_key_value: str | None = None
        self._loaded_models: set[str] = set()
        self._loaded_lora_state = LoraPipelineState()

    def loaded_models(self) -> list[str]:
        return sorted(self._loaded_models)

    def prepare(self, *, settings: WorkerSettings, job: dict[str, Any]) -> VideoRequest:
        return video_request_from_job(job)

    def ensure_models(self, request: VideoRequest) -> None:
        target = model_target(request.model)
        if request.mode not in target["capabilities"]:
            raise RuntimeError(f"{target['label']} does not support {request.mode.replace('_', ' ')}.")
        if request.duration > target["hardMaxDuration"]:
            raise RuntimeError(f"{target['label']} is limited to {target['hardMaxDuration']}s clips in this adapter.")
        if (
            target["adapter"] == "ltx_video"
            and request.mode == "text_to_video"
            and not request.advanced.get("modelRepo")
            and not target.get("diffusersTextRepo")
        ):
            raise RuntimeError(
                "LTX-2.3 text-to-video is supported by the model, but this worker's Diffusers adapter cannot "
                "load the raw LTX-2.3 checkpoint repo because it does not publish a Diffusers model_index.json. "
                "Use an advanced modelRepo override that points to a Diffusers-compatible LTX-2.3 conversion, "
                "or use an adapter backed by the official Lightricks LTX pipeline stack."
            )

    def estimate_requirements(self, request: VideoRequest) -> dict[str, Any]:
        target = model_target(request.model)
        raw_frames = max(1, int(round(request.duration * request.fps)))
        estimated_frames = self._num_frames(request)
        return {
            "estimatedFrames": estimated_frames,
            "requestedFrames": raw_frames,
            "pixelCount": request.width * request.height,
            "recommendedMaxDuration": target["recommendedMaxDuration"],
            "gpuPreference": request.advanced.get("gpuPreference", "auto"),
            "adapter": target["adapter"],
            "repo": self._repo_for_request(request, target),
        }

    def run(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: VideoRequest,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
        for folder in ("assets/videos", "generation-sets", "recipes"):
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        target = model_target(request.model)
        progress("loading_model", "loading_model", 0.2, f"Loading {target['label']} Diffusers pipeline.")
        pipe = self._load_pipeline(settings, request, target)
        self._apply_loras(pipe, request, target)

        first_image = self._first_condition_image(project_path, request)
        last_image = self._last_condition_image(project_path, request)
        self._validate_inputs(project_path, request, first_image, last_image)
        if cancel_requested():
            raise InterruptedError("Video generation canceled before inference.")

        seed = resolve_seed(request.seed, request.prompt)
        num_frames = self._num_frames(request)
        kwargs = self._pipeline_kwargs(
            pipe=pipe,
            project_path=project_path,
            request=request,
            target=target,
            first_image=first_image,
            last_image=last_image,
            seed=seed,
            num_frames=num_frames,
        )
        progress("running", "generating", 0.32, f"Running {target['label']} inference.")
        torch = importlib.import_module("torch")
        with torch.inference_mode():
            output = pipe(**kwargs)
        if cancel_requested():
            raise InterruptedError("Video generation canceled before saving.")

        frames = frames_from_output(output)
        if not frames:
            raise RuntimeError(f"{target['label']} returned no video frames.")

        generation_set_id = f"genset_{uuid4().hex}"
        asset_id = f"asset_{uuid4().hex}"
        created_at = utc_now()
        raw_settings = self.map_settings(request, target)
        filename_base = f"{created_at[:10]}_{request.model}_{slugify(request.prompt, fallback='video', max_length=42)}"
        media_rel = f"assets/videos/{filename_base}.mp4"
        media_path = project_path / media_rel
        temp_path = media_path.with_suffix(".tmp.mp4")

        progress("saving", "saving", 0.9, "Saving generated MP4 asset and recipe.")
        diffusers_utils = importlib.import_module("diffusers.utils")
        diffusers_utils.export_to_video(frames, str(temp_path), fps=request.fps)
        temp_path.replace(media_path)

        generation_set = {
            "schemaVersion": 1,
            "id": generation_set_id,
            "projectId": request.project_id,
            "jobId": job["id"],
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": 1,
            "createdAt": created_at,
        }
        asset = build_video_asset_sidecar(
            asset_id=asset_id,
            project_id=request.project_id,
            generation_set_id=generation_set_id,
            request=request,
            job_id=job["id"],
            media_rel=media_rel,
            created_at=created_at,
            seed=seed,
            target=target,
            adapter_id=target["adapter"],
            mime_type="video/mp4",
            raw_settings=raw_settings,
        )

        write_json(project_path / "generation-sets" / f"{generation_set_id}.json", generation_set)
        write_json(media_path.with_suffix(".sceneworks.json"), asset)
        write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
        index_asset(project_path, asset)
        return {
            "generationSetId": generation_set_id,
            "assetIds": [asset_id],
            "assets": [asset],
            "adapter": target["adapter"],
            "model": request.model,
            "requirements": self.estimate_requirements(request),
        }

    def cancel(self, _job_id: str) -> None:
        return

    def cleanup(self, _job_id: str) -> None:
        return

    def map_settings(self, request: VideoRequest, target: dict[str, Any]) -> dict[str, Any]:
        return {
            **request.advanced,
            "adapterFamily": target["family"],
            "targetAdapter": target["adapter"],
            "repo": self._repo_for_request(request, target),
            "steps": self._num_inference_steps(request, target),
            "frameCount": self._num_frames(request),
            "recommendedMaxDuration": target["recommendedMaxDuration"],
            "previewRenderer": False,
            "realModelInference": True,
        }

    def _load_pipeline(self, settings: WorkerSettings, request: VideoRequest, target: dict[str, Any]) -> Any:
        key = self._pipeline_key(request, target)
        repo = self._repo_for_request(request, target)
        if self._pipeline is not None and self._pipeline_key_value == key:
            self._loaded_models.update({request.model, self._repo_for_request(request, target)})
            return self._pipeline

        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        self._evict_pipeline(torch)
        pipeline_class = self._pipeline_class(diffusers, request, target)
        kwargs: dict[str, Any] = {"torch_dtype": dtype}

        if target["adapter"] == "wan_video":
            vae_class = getattr(diffusers, "AutoencoderKLWan", None)
            if vae_class is not None:
                kwargs["vae"] = vae_class.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
            if request.mode != "text_to_video" and not target.get("noImageEncoder"):
                transformers = importlib.import_module("transformers")
                image_encoder_class = getattr(transformers, "CLIPVisionModel", None)
                if image_encoder_class is not None:
                    try:
                        kwargs["image_encoder"] = image_encoder_class.from_pretrained(repo, subfolder="image_encoder", torch_dtype=dtype)
                    except (OSError, ValueError):
                        # Repos that condition on VAE latents instead of CLIP (e.g. Wan2.2 A14B)
                        # ship no image_encoder subfolder; the pipeline loads any components it needs.
                        pass

        pipe = pipeline_class.from_pretrained(repo, **kwargs)
        if bool(request.advanced.get("cpuOffload", False)) and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        self._pipeline = pipe
        self._pipeline_key_value = key
        self._loaded_models.update({request.model, repo})
        return pipe

    def _evict_pipeline(self, torch: Any) -> None:
        self._pipeline = None
        self._pipeline_key_value = None
        self._loaded_models.clear()
        self._loaded_lora_state = LoraPipelineState()
        empty_torch_cache(torch)

    def _apply_loras(self, pipe: Any, request: VideoRequest, target: dict[str, Any]) -> None:
        self._loaded_lora_state = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=target["adapter"],
            model_family=target.get("family"),
            previous_state=self._loaded_lora_state,
        )

    def _pipeline_class(self, diffusers: Any, request: VideoRequest, target: dict[str, Any]) -> Any:
        if target["adapter"] == "wan_video":
            if request.mode == "text_to_video":
                return getattr(diffusers, "WanPipeline")
            if request.mode == "replace_person" and hasattr(diffusers, "WanVACEPipeline"):
                return getattr(diffusers, "WanVACEPipeline")
            return getattr(diffusers, "WanImageToVideoPipeline")

        if request.mode in {"first_last_frame", "video_bridge"}:
            try:
                ltx_condition = importlib.import_module("diffusers.pipelines.ltx.pipeline_ltx_condition")
                return getattr(ltx_condition, "LTXConditionPipeline")
            except (ImportError, AttributeError):
                pass
        if request.mode == "text_to_video":
            pipeline_class = getattr(diffusers, "LTX2Pipeline", None) or getattr(diffusers, "LTXPipeline", None)
            if pipeline_class is None:
                raise RuntimeError("The installed diffusers package does not expose LTX2Pipeline or LTXPipeline.")
            return pipeline_class
        return getattr(diffusers, "LTXImageToVideoPipeline")

    def _pipeline_key(self, request: VideoRequest, target: dict[str, Any]) -> str:
        return f"{self._repo_for_request(request, target)}:{self._pipeline_kind(request, target)}"

    def _pipeline_kind(self, request: VideoRequest, target: dict[str, Any]) -> str:
        if target["adapter"] == "wan_video":
            if request.mode == "text_to_video":
                return "text"
            if request.mode == "replace_person":
                return "vace"
            return "image"
        if request.mode in {"first_last_frame", "video_bridge"}:
            return "condition"
        if request.mode == "text_to_video":
            return "text"
        return "image"

    def _repo_for_request(self, request: VideoRequest, target: dict[str, Any]) -> str:
        if request.advanced.get("modelRepo"):
            return str(request.advanced["modelRepo"])
        if target["adapter"] == "ltx_video" and request.mode == "text_to_video" and target.get("diffusersTextRepo"):
            return target["diffusersTextRepo"]
        if target["adapter"] == "ltx_video" and request.mode != "text_to_video":
            return target.get("fallbackRepo") or target["repo"]
        return target["repo"]

    def _num_inference_steps(self, request: VideoRequest, target: dict[str, Any]) -> int:
        default_steps = target["steps"].get(request.quality, target["steps"]["balanced"])
        return safe_int(request.advanced.get("steps"), default_steps, 1, 80)

    def _guidance_scale(self, request: VideoRequest, fallback: float) -> float:
        try:
            return float(request.advanced.get("guidanceScale", fallback))
        except (TypeError, ValueError):
            return fallback

    def _num_frames(self, request: VideoRequest) -> int:
        raw_frames = max(1, int(round(request.duration * request.fps)))
        adapter = model_target(request.model)["adapter"]
        if adapter == "ltx_video":
            return ltx_frame_count(raw_frames)
        if adapter == "wan_video":
            return max(5, raw_frames - ((raw_frames - 1) % 4))
        return raw_frames

    def _pipeline_kwargs(
        self,
        *,
        pipe: Any,
        project_path: Path,
        request: VideoRequest,
        target: dict[str, Any],
        first_image: Image.Image | None,
        last_image: Image.Image | None,
        seed: int,
        num_frames: int,
    ) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch)
        generator = torch.Generator("cuda" if device == "cuda" else "cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or default_negative_prompt(target),
            "height": request.height,
            "width": request.width,
            "num_frames": num_frames,
            "num_inference_steps": self._num_inference_steps(request, target),
            "generator": generator,
        }
        if target["adapter"] == "wan_video":
            kwargs["guidance_scale"] = self._guidance_scale(request, float(target.get("guidanceScale", 5.0)))
            # A14B exposes a second CFG scale for the low-noise expert and the high/low expert
            # boundary. Pass them through when supplied; filter_call_kwargs drops them for
            # single-expert pipelines (e.g. the 5B TI2V model) that don't accept them.
            guidance_scale_2 = request.advanced.get("guidanceScale2", target.get("guidanceScale2"))
            if guidance_scale_2 is not None:
                kwargs["guidance_scale_2"] = float(guidance_scale_2)
            boundary_ratio = request.advanced.get("boundaryRatio", target.get("boundaryRatio"))
            if boundary_ratio is not None:
                kwargs["boundary_ratio"] = float(boundary_ratio)
            if request.mode == "replace_person":
                kwargs["video"] = load_source_video_frames(project_path, request.source_clip_asset_id, request.width, request.height, num_frames)
                kwargs["mask"] = person_track_masks(project_path, request.person_track_id, request.width, request.height, num_frames)
                reference_images = character_reference_images(project_path, request.character_id, request.character_look_id, request.width, request.height)
                if reference_images:
                    kwargs["reference_images"] = reference_images
                kwargs["conditioning_scale"] = float(request.advanced.get("conditioningScale", 1.0))
            elif request.mode != "text_to_video" and first_image is not None:
                kwargs["image"] = first_image.resize((request.width, request.height))
            if last_image is not None:
                kwargs["last_image"] = last_image.resize((request.width, request.height))
        else:
            kwargs["guidance_scale"] = self._guidance_scale(request, 3.0)
            if request.mode in {"first_last_frame", "video_bridge"}:
                conditions = ltx_conditions(first_image, last_image, num_frames, request.width, request.height)
                if conditions:
                    kwargs["conditions"] = conditions
            elif first_image is not None:
                kwargs["image"] = first_image.resize((request.width, request.height))
            kwargs["frame_rate"] = request.fps
        return filter_call_kwargs(pipe, kwargs)

    def _first_condition_image(self, project_path: Path, request: VideoRequest) -> Image.Image | None:
        if request.mode in {"image_to_video", "first_last_frame"}:
            return load_source_image(project_path, request.source_asset_id, request.width, request.height)
        if request.mode in {"extend_clip", "video_bridge", "replace_person"}:
            context = request.advanced.get("timelineContext", {})
            timestamp = safe_float(context.get("endpointTimestamp") or context.get("leftTimestamp"), 0, 0, 3600)
            return load_source_frame(project_path, request.source_clip_asset_id, timestamp, request.width, request.height)
        return None

    def _last_condition_image(self, project_path: Path, request: VideoRequest) -> Image.Image | None:
        if request.mode == "first_last_frame":
            return load_source_image(project_path, request.last_frame_asset_id, request.width, request.height)
        if request.mode == "video_bridge":
            context = request.advanced.get("timelineContext", {})
            timestamp = safe_float(context.get("rightTimestamp"), 0, 0, 3600)
            return load_source_frame(project_path, request.bridge_right_clip_asset_id, timestamp, request.width, request.height)
        return None

    def _validate_inputs(self, project_path: Path, request: VideoRequest, first_image: Image.Image | None, last_image: Image.Image | None) -> None:
        if request.mode in {"image_to_video", "first_last_frame"} and first_image is None:
            raise RuntimeError(f"{request.mode.replace('_', ' ').title()} requires a readable source image.")
        if request.mode == "first_last_frame" and last_image is None:
            raise RuntimeError("First/Last Frame requires a readable last frame image.")
        if request.mode in {"extend_clip", "video_bridge", "replace_person"} and first_image is None:
            raise RuntimeError(f"{request.mode.replace('_', ' ').title()} requires a readable source clip frame.")
        if request.mode == "video_bridge" and last_image is None:
            raise RuntimeError("Bridge generation requires a readable right clip frame.")
        if request.mode == "replace_person":
            if not request.person_track_id:
                raise RuntimeError("Replace Person requires a selected person track.")
            if not request.character_id:
                raise RuntimeError("Replace Person requires a character.")
            if read_person_track(project_path, request.person_track_id) is None:
                raise RuntimeError(f"Replace Person track not found: {request.person_track_id}.")
            if read_character(project_path, request.character_id) is None:
                raise RuntimeError(f"Replace Person character not found: {request.character_id}.")
            if not character_reference_images(project_path, request.character_id, request.character_look_id, request.width, request.height):
                raise RuntimeError("Replace Person requires at least one readable approved character reference image.")


def create_video_adapter(job: dict[str, Any] | None = None) -> VideoGenerationAdapter:
    requested = os.getenv("SCENEWORKS_VIDEO_ADAPTER", "").strip()
    if requested in {"ltx", "ltx_pipelines", "native_ltx"}:
        return LtxPipelinesVideoAdapter()
    if requested == "diffusers_video":
        return DiffusersVideoAdapter()
    if requested in {"procedural", "procedural_video"}:
        return ProceduralVideoAdapter()
    if not requested:
        model = str((job or {}).get("payload", {}).get("model", "ltx_2_3"))
        target = model_target(model)
        if target["adapter"] == "ltx_video":
            return LtxPipelinesVideoAdapter()
        return DiffusersVideoAdapter()
    raise RuntimeError(f"Unsupported SCENEWORKS_VIDEO_ADAPTER value: {requested}.")


def video_request_from_job(job: dict[str, Any]) -> VideoRequest:
    payload = job["payload"]
    width, height = normalized_dimensions(payload.get("width", 768), payload.get("height", 512))
    return VideoRequest(
        project_id=payload["projectId"],
        mode=payload.get("mode", "image_to_video"),
        prompt=payload.get("prompt", ""),
        negative_prompt=payload.get("negativePrompt", ""),
        model=payload.get("model", "ltx_2_3"),
        duration=safe_float(payload.get("duration"), 6, 1, 30),
        fps=safe_int(payload.get("fps"), 25, 1, 60),
        width=width,
        height=height,
        quality=payload.get("quality", "balanced"),
        seed=payload.get("seed"),
        loras=payload.get("loras", []),
        character_id=payload.get("characterId"),
        character_look_id=payload.get("characterLookId"),
        person_track_id=payload.get("personTrackId"),
        replacement_mode=payload.get("replacementMode", "face_only"),
        source_asset_id=payload.get("sourceAssetId"),
        last_frame_asset_id=payload.get("lastFrameAssetId"),
        source_clip_asset_id=payload.get("sourceClipAssetId"),
        bridge_right_clip_asset_id=payload.get("bridgeRightClipAssetId"),
        advanced=payload.get("advanced", {}),
    )


def model_target(model_id: str) -> dict[str, Any]:
    return VIDEO_MODEL_TARGETS.get(model_id, VIDEO_MODEL_TARGETS["ltx_2_3"])


def normalized_dimensions(width: Any, height: Any) -> tuple[int, int]:
    parsed_width = safe_int(width, 768, 256, 1920)
    parsed_height = safe_int(height, 512, 256, 1920)
    return max(256, parsed_width - (parsed_width % 32)), max(256, parsed_height - (parsed_height % 32))


def resolve_seed(seed: int | None, prompt: str) -> int:
    if seed is not None:
        return int(seed)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def preview_frame_count(request: VideoRequest) -> int:
    raw_frames = int(round(request.duration * request.fps))
    if request.quality == "fast":
        return max(12, min(40, raw_frames // 2))
    if request.quality == "best":
        return max(18, min(80, raw_frames))
    return max(16, min(60, raw_frames))


def ltx_frame_count(raw_frames: int) -> int:
    frame_count = max(9, int(raw_frames))
    lower = frame_count - ((frame_count - 1) % 8)
    upper = lower + 8
    if lower < 9:
        return upper
    lower_delta = abs(frame_count - lower)
    upper_delta = abs(upper - frame_count)
    return lower if lower_delta <= upper_delta else upper


def ltx_model_manifest_entry(settings: WorkerSettings, model_id: str) -> dict[str, Any]:
    config_dir = getattr(settings, "config_dir", Path("config").resolve())
    builtin_entry: dict[str, Any] = {}
    user_entry: dict[str, Any] = {}
    for manifest_name in ("builtin.models.jsonc", "user.models.jsonc"):
        manifest_path = config_dir / "manifests" / manifest_name
        try:
            payload = json.loads(strip_jsonc_comments(manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        models = payload.get("models", [])
        if not isinstance(models, list):
            continue
        for entry in models:
            if isinstance(entry, dict) and entry.get("id") == model_id:
                if manifest_name.startswith("builtin"):
                    builtin_entry = entry
                else:
                    user_entry = entry
    if not user_entry:
        return builtin_entry
    merged = {**builtin_entry, **user_entry}
    for nested_key in ("paths", "resources", "defaults", "limits", "loraCompatibility", "ui"):
        builtin_nested = builtin_entry.get(nested_key) if isinstance(builtin_entry.get(nested_key), dict) else {}
        user_nested = user_entry.get(nested_key) if isinstance(user_entry.get(nested_key), dict) else {}
        if builtin_nested or user_nested:
            merged[nested_key] = {**builtin_nested, **user_nested}
    return merged


def strip_jsonc_comments(text: str) -> str:
    output = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
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
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def resolve_worker_path(settings: WorkerSettings, value: Any) -> Path:
    raw_path = str(value).strip()
    path = Path(raw_path)
    return path if path.is_absolute() else settings.data_dir / path


def resolve_manifest_path(settings: WorkerSettings, value: Any) -> Path:
    raw_path = str(value).strip()
    if "${DATA_DIR}" in raw_path:
        raw_path = raw_path.replace("${DATA_DIR}", str(settings.data_dir))
    if "${HF_CACHE}" in raw_path:
        repo_ref = raw_path.split("${HF_CACHE}", 1)[1].strip("/\\").replace("\\", "/")
        snapshot = huggingface_cached_snapshot_dir(settings, repo_ref) if repo_ref else None
        if snapshot is not None:
            return snapshot
        raw_path = raw_path.replace("${HF_CACHE}", str(huggingface_cache_root()))
    path = Path(raw_path)
    return path if path.is_absolute() else settings.data_dir / path


def huggingface_cache_root() -> Path:
    default_home = Path.home() / ".cache" / "huggingface"
    hf_home = Path(os.getenv("HF_HOME") or default_home)
    return Path(os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE") or hf_home / "hub")


def huggingface_cache_roots(settings: WorkerSettings | None = None) -> list[Path]:
    roots: list[Path] = []
    for value in (os.getenv("HF_HUB_CACHE"), os.getenv("HUGGINGFACE_HUB_CACHE")):
        if value:
            roots.append(Path(value))
    if os.getenv("HF_HOME"):
        roots.append(Path(os.getenv("HF_HOME", "")) / "hub")
    if settings is not None:
        roots.append(settings.data_dir / "cache" / "huggingface" / "hub")
    roots.append(huggingface_cache_root())
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def huggingface_repo_cache_path_for_root(cache_root: Path, repo: str) -> Path | None:
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


def huggingface_cached_snapshot_dir(settings: WorkerSettings, repo: str) -> Path | None:
    for root in huggingface_cache_roots(settings):
        repo_cache = huggingface_repo_cache_path_for_root(root, repo)
        if repo_cache is None:
            continue
        snapshot = newest_huggingface_snapshot(repo_cache)
        if snapshot is not None:
            return snapshot
    return None


def huggingface_cached_snapshot_search_paths(settings: WorkerSettings, repo: str) -> list[Path]:
    paths = []
    for root in huggingface_cache_roots(settings):
        repo_cache = huggingface_repo_cache_path_for_root(root, repo)
        if repo_cache is None:
            continue
        snapshot = newest_huggingface_snapshot(repo_cache)
        if snapshot is not None:
            paths.append(snapshot)
        else:
            paths.append(repo_cache / "snapshots" / "<revision>")
    return paths


def newest_huggingface_snapshot(repo_cache: Path) -> Path | None:
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    except OSError:
        return None
    snapshots.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    return snapshots[0] if snapshots else None


def huggingface_cached_resource_file(settings: WorkerSettings, repo: str, file_name: str) -> Path | None:
    relative = Path(file_name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        return None
    for root in huggingface_cache_roots(settings):
        repo_cache = huggingface_repo_cache_path_for_root(root, repo)
        if repo_cache is None:
            continue
        snapshot = newest_huggingface_snapshot(repo_cache)
        if snapshot is None:
            continue
        candidate = snapshot / relative
        if candidate.is_file():
            return candidate
    return None


def huggingface_cached_resource_search_paths(settings: WorkerSettings, repo: str, file_name: str) -> list[Path]:
    relative = Path(file_name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        return []
    paths = []
    for root in huggingface_cache_roots(settings):
        repo_cache = huggingface_repo_cache_path_for_root(root, repo)
        if repo_cache is None:
            continue
        snapshot = newest_huggingface_snapshot(repo_cache)
        if snapshot is not None:
            paths.append(snapshot / relative)
        else:
            paths.append(repo_cache / "snapshots" / "<revision>" / relative)
    return paths


def safe_download_dir(repo: str) -> str:
    output = []
    in_replacement = False
    for character in repo:
        if character.isalnum() or character in "_.-":
            output.append(character)
            in_replacement = False
        elif not in_replacement:
            output.append("__")
            in_replacement = True
    safe = "".join(output).strip("_")
    return safe or "download"


def source_asset_media_path(project_path: Path, asset_id: str | None) -> Path | None:
    if not asset_id:
        return None
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        return None
    payload = read_json(sidecar_path)
    media_rel = payload.get("file", {}).get("path", "")
    media_path = project_path / media_rel
    return media_path if media_path.exists() else None


def load_source_image(project_path: Path, asset_id: str | None, width: int, height: int) -> Image.Image | None:
    if not asset_id:
        return None
    media_path = source_asset_media_path(project_path, asset_id)
    if media_path is None:
        return None
    try:
        image = Image.open(media_path).convert("RGB")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return None
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (18, 17, 15))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def load_source_frame(project_path: Path, asset_id: str | None, timestamp: float, width: int, height: int) -> Image.Image | None:
    if not asset_id:
        return None
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        return None
    payload = read_json(sidecar_path)
    media_path = project_path / payload.get("file", {}).get("path", "")
    if not media_path.exists():
        return None

    image = load_seekable_image_frame(media_path, timestamp, payload.get("file", {}).get("duration"))
    if image is None:
        return None

    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (18, 17, 15))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def load_source_video_frames(project_path: Path, asset_id: str | None, width: int, height: int, count: int) -> list[Image.Image]:
    if not asset_id:
        raise RuntimeError("Replace Person requires a source clip asset.")
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        raise RuntimeError(f"Source clip asset not found: {asset_id}.")
    payload = read_json(sidecar_path)
    media_path = project_path / payload.get("file", {}).get("path", "")
    if not media_path.exists():
        raise RuntimeError(f"Source clip file is missing for asset {asset_id}.")

    frames = load_pil_video_frames(media_path, width, height, count)
    if not frames:
        frames = load_imageio_video_frames(media_path, width, height, count)
    if not frames:
        raise RuntimeError(f"Source clip frames could not be read for asset {asset_id}.")
    if len(frames) < count:
        frames.extend(frames[-1].copy() for _ in range(count - len(frames)))
    return frames[:count]


def load_pil_video_frames(media_path: Path, width: int, height: int, count: int) -> list[Image.Image]:
    try:
        image = Image.open(media_path)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise RuntimeError(f"Source media exceeds the safe image pixel limit: {media_path}") from exc
    except (UnidentifiedImageError, OSError):
        return []
    frame_total = max(1, getattr(image, "n_frames", 1))
    frames = []
    for index in evenly_spaced_indices(frame_total, count):
        try:
            image.seek(index)
            frames.append(fit_frame(image.convert("RGB"), width, height))
        except (EOFError, OSError):
            break
    return frames


def load_imageio_video_frames(media_path: Path, width: int, height: int, count: int) -> list[Image.Image]:
    try:
        imageio = importlib.import_module("imageio.v3")
        metadata = imageio.immeta(media_path)
        duration = safe_float(metadata.get("duration"), 0, 0, 3600)
        fps = safe_float(metadata.get("fps"), 24, 1, 240)
        frame_total = max(count, int(round(duration * fps))) if duration > 0 else count
        target_indices = set(evenly_spaced_indices(frame_total, count))
        frames = []
        for index, frame in enumerate(imageio.imiter(media_path)):
            if index in target_indices:
                frames.append(fit_frame(Image.fromarray(frame).convert("RGB"), width, height))
                if len(frames) == len(target_indices):
                    break
        return frames
    except Exception:
        return []


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    return [min(total - 1, max(0, round((total - 1) * index / (count - 1)))) for index in range(count)]


def fit_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (18, 17, 15))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def load_seekable_image_frame(media_path: Path, timestamp: float, duration: Any = None) -> Image.Image | None:
    try:
        image = Image.open(media_path)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return None
    except UnidentifiedImageError:
        return load_seekable_video_frame(media_path, timestamp)
    except OSError:
        return None

    try:
        frame_count = getattr(image, "n_frames", 1)
        if frame_count > 1:
            total_duration = safe_float(duration, 0, 0, 3600)
            if total_duration > 0:
                frame_index = min(frame_count - 1, max(0, int(round((timestamp / total_duration) * (frame_count - 1)))))
                image.seek(frame_index)
        return image.convert("RGB")
    except (EOFError, OSError):
        return None


def load_seekable_video_frame(media_path: Path, timestamp: float) -> Image.Image | None:
    try:
        imageio = importlib.import_module("imageio.v3")
        metadata = imageio.immeta(media_path)
        fps = safe_float(metadata.get("fps"), 24, 1, 240)
        frame_index = max(0, int(round(timestamp * fps)))
        frame = imageio.imread(media_path, index=frame_index)
        return Image.fromarray(frame).convert("RGB")
    except Exception:
        return None


def render_preview_frames(
    request: VideoRequest,
    target: dict[str, Any],
    seed: int,
    first_image: Image.Image | None,
    last_image: Image.Image | None,
    progress: ProgressCallback,
    cancel_requested: CancelCallback,
) -> list[Image.Image]:
    frame_count = preview_frame_count(request)
    digest = hashlib.sha256(f"{request.prompt}:{request.mode}:{seed}".encode("utf-8")).digest()
    base = first_image or gradient_frame(request.width, request.height, digest)
    end = last_image or ImageEnhance.Color(base.copy()).enhance(1.35)
    frames: list[Image.Image] = []

    for index in range(frame_count):
        if cancel_requested():
            raise InterruptedError("Video generation canceled by user.")

        t = index / max(1, frame_count - 1)
        frame = Image.blend(base, end, t * 0.55 if last_image else t * 0.24)
        draw_motion(frame, digest, index, frame_count)
        if request.mode == "replace_person":
            draw_replacement_overlay(frame, request, digest, index, frame_count)
        draw_caption(frame, request, target, seed, index, frame_count)
        frames.append(frame)
        if index % max(1, frame_count // 8) == 0 or index == frame_count - 1:
            progress("running", "generating", 0.2 + ((index + 1) / frame_count) * 0.62, "Rendering preview clip frames.")
    return frames


def gradient_frame(width: int, height: int, digest: bytes) -> Image.Image:
    import numpy as np

    base = np.array([digest[0], digest[1], digest[2]], dtype=np.float32)
    accent = np.array([digest[10], digest[11], digest[12]], dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    mix = x * 0.58 + y * 0.42
    xi = np.arange(width, dtype=np.uint32)[None, :]
    yi = np.arange(height, dtype=np.uint32)[:, None]
    wave = ((xi * digest[3] + yi * digest[4]) % 255).astype(np.float32) / 255
    pixels = base * (1 - mix[..., None]) + accent * mix[..., None] * 0.88 + wave[..., None] * 28
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def draw_motion(frame: Image.Image, digest: bytes, index: int, frame_count: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    width, height = frame.size
    t = index / max(1, frame_count - 1)
    sweep_x = int((width + 180) * t) - 120
    draw.rectangle((sweep_x, 0, sweep_x + 70, height), fill=(255, 255, 255, 26))
    draw.ellipse(
        (
            int(width * (0.18 + 0.58 * t)),
            int(height * (0.22 + (digest[5] % 20) / 100)),
            int(width * (0.3 + 0.58 * t)),
            int(height * (0.34 + (digest[6] % 20) / 100)),
        ),
        outline=(110, 198, 184, 150),
        width=max(2, width // 220),
    )
    draw.line(
        (
            int(width * 0.08),
            int(height * (0.78 - t * 0.16)),
            int(width * 0.92),
            int(height * (0.68 + t * 0.12)),
        ),
        fill=(248, 215, 140, 90),
        width=max(2, width // 180),
    )


def draw_replacement_overlay(frame: Image.Image, request: VideoRequest, digest: bytes, index: int, frame_count: int) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    width, height = frame.size
    t = index / max(1, frame_count - 1)
    drift = ((digest[7] % 21) - 10) / 1000
    box_width = int(width * 0.24)
    box_height = int(height * 0.58)
    center_x = int(width * (0.48 + (t - 0.5) * 0.08 + drift))
    top = int(height * 0.18)
    left = max(8, min(width - box_width - 8, center_x - box_width // 2))
    right = left + box_width
    bottom = min(height - 8, top + box_height)
    colors = {
        "face_only": (88, 214, 179, 92),
        "full_person_keep_outfit": (248, 213, 112, 82),
        "full_person_replace_outfit": (222, 143, 246, 82),
    }
    fill = colors.get(request.replacement_mode, colors["face_only"])
    outline = (255, 255, 255, 190)
    draw.rounded_rectangle((left, top, right, bottom), radius=max(8, width // 80), fill=fill, outline=outline, width=max(2, width // 240))
    if request.replacement_mode == "face_only":
        face_top = top + int(box_height * 0.08)
        draw.ellipse((left + box_width * 0.3, face_top, right - box_width * 0.3, face_top + box_width * 0.38), fill=(255, 244, 214, 130))
    else:
        draw.line((left + 12, top + box_height * 0.52, right - 12, top + box_height * 0.52), fill=(255, 255, 255, 95), width=max(2, width // 220))
    label = request.replacement_mode.replace("_", " ")
    draw.text((left + 10, max(8, top - 20)), label, fill=(255, 255, 255, 230), font=ImageFont.load_default())


def draw_caption(
    frame: Image.Image,
    request: VideoRequest,
    target: dict[str, Any],
    seed: int,
    index: int,
    frame_count: int,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    width, height = frame.size
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, 72), fill=(12, 12, 12, 124))
    draw.rectangle((0, int(height * 0.72), width, height), fill=(12, 12, 12, 156))
    draw.text((24, 18), f"{target['label']} preview | {request.mode.replace('_', ' ')}", fill=(255, 244, 214, 255), font=font)
    draw.text((24, 42), f"{request.duration:g}s {request.fps}fps {request.width}x{request.height} seed {seed}", fill=(194, 235, 226, 255), font=font)
    draw.text((width - 92, 18), f"{index + 1}/{frame_count}", fill=(255, 255, 255, 220), font=font)

    y = int(height * 0.74)
    for line in wrap(request.prompt.strip() or "Untitled video prompt", width=max(28, width // 14))[:5]:
        draw.text((24, y), line, fill=(255, 255, 255, 238), font=font)
        y += 18


def save_animated_preview(frames: list[Image.Image], path: Path, duration: float) -> None:
    frame_ms = max(40, int((duration * 1000) / max(1, len(frames))))
    frames[0].save(
        path,
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        quality=82,
        method=4,
    )


def _frame_to_image(frame: Any) -> Image.Image:
    if hasattr(frame, "convert"):
        return frame.convert("RGB")
    import numpy as np

    array = np.asarray(frame)
    # Diffusers video pipelines default to output_type="np", which returns
    # float32 frames in [0, 1]. PIL cannot build an RGB image from float data
    # ("Cannot handle this data type: (1, 1, 3), <f4"), so scale to uint8 the
    # same way diffusers.utils.export_to_video does internally.
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def frames_from_output(output: Any) -> list[Image.Image]:
    frames = getattr(output, "frames", None)
    if frames is None and isinstance(output, tuple) and output:
        frames = output[0]
    if frames is None:
        frames = getattr(output, "images", None)
    if frames is None:
        return []
    if hasattr(frames, "ndim"):
        import numpy as np

        array = np.asarray(frames)
        if array.ndim == 5:
            array = array[0]
        return [_frame_to_image(frame) for frame in array]
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        return [_frame_to_image(frame) for frame in frames[0]]
    if isinstance(frames, list):
        return [_frame_to_image(frame) for frame in frames]
    return []


def ltx_conditions(
    first_image: Image.Image | None,
    last_image: Image.Image | None,
    num_frames: int,
    width: int,
    height: int,
) -> list[Any]:
    try:
        ltx_condition = importlib.import_module("diffusers.pipelines.ltx.pipeline_ltx_condition")
        condition_class = getattr(ltx_condition, "LTXVideoCondition")
    except (ImportError, AttributeError):
        return []

    conditions = []
    if first_image is not None:
        conditions.append(condition_class(image=first_image.resize((width, height)), frame_index=0))
    if last_image is not None:
        conditions.append(condition_class(image=last_image.resize((width, height)), frame_index=max(0, num_frames - 1)))
    return conditions


def person_track_masks(project_path: Path, track_id: str | None, width: int, height: int, count: int) -> list[Image.Image]:
    """Load per-frame replacement masks for a track, preferring stored
    segmentation masks and only falling back to rectangular box masks in the
    explicit degraded path (sc-1482)."""
    from .person_adapters import load_track_masks  # local: avoids an import cycle

    track = read_person_track(project_path, track_id)
    if not track:
        raise RuntimeError(f"Person track not found: {track_id}.")
    if not track.get("frames"):
        selected_box = track.get("selectedDetection", {}).get("box")
        if not isinstance(selected_box, dict):
            raise RuntimeError(f"Person track has no usable boxes: {track_id}.")
        track = {**track, "frames": [{"box": selected_box, "mask": None}]}
    masks, _mode = load_track_masks(project_path, track, width, height, count)
    return masks


def read_person_track(project_path: Path, track_id: str | None) -> dict[str, Any] | None:
    if not track_id:
        return None
    tracks_dir = project_path / "person-tracks"
    candidates = [tracks_dir / f"{track_id}.sceneworks.person-track.json"]
    if tracks_dir.exists():
        candidates.extend(tracks_dir.glob("*.sceneworks.person-track.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if payload.get("id") == track_id:
            return payload
    return None


def character_reference_images(project_path: Path, character_id: str | None, look_id: str | None, width: int, height: int) -> list[Image.Image]:
    character = read_character(project_path, character_id)
    if not character:
        return []
    approved_ids = []
    for look in character.get("looks", []):
        if isinstance(look, dict) and look.get("id") == look_id:
            approved_ids.extend(look.get("approvedReferenceIds", []))
    if not approved_ids:
        approved_ids.extend(
            reference.get("assetId")
            for reference in character.get("references", [])
            if isinstance(reference, dict) and reference.get("approved")
        )
    images = []
    for asset_id in [asset_id for asset_id in approved_ids if asset_id][:4]:
        image = load_source_image(project_path, asset_id, width, height)
        if image is not None:
            images.append(image)
    return images


def read_character(project_path: Path, character_id: str | None) -> dict[str, Any] | None:
    if not character_id:
        return None
    characters_dir = project_path / "characters"
    candidates = [characters_dir / f"{character_id}.sceneworks.character.json"]
    if characters_dir.exists():
        candidates.extend(characters_dir.glob("*.sceneworks.character.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if payload.get("id") == character_id:
            return payload
    return None


def default_negative_prompt(target: dict[str, Any]) -> str:
    if target["adapter"] == "wan_video":
        return (
            "Bright tones, overexposed, static, blurred details, subtitles, paintings, still picture, "
            "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "deformed, disfigured, messy background"
        )
    return "worst quality, inconsistent motion, blurry, jittery, distorted"


def build_video_asset_sidecar(
    *,
    asset_id: str,
    project_id: str,
    generation_set_id: str,
    request: VideoRequest,
    job_id: str,
    media_rel: str,
    created_at: str,
    seed: int,
    target: dict[str, Any],
    raw_settings: dict[str, Any],
    adapter_id: str,
    mime_type: str,
    replacement_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parents = [
        asset_id
        for asset_id in [
            request.source_asset_id,
            request.last_frame_asset_id,
            request.source_clip_asset_id,
            request.bridge_right_clip_asset_id,
        ]
        if asset_id
    ]
    timeline_context = request.advanced.get("timelineContext", {})
    normalized_settings = {
        "duration": request.duration,
        "fps": request.fps,
        "width": request.width,
        "height": request.height,
        "quality": request.quality,
        "family": target["family"],
        "sourceAssetId": request.source_asset_id,
        "lastFrameAssetId": request.last_frame_asset_id,
        "sourceClipAssetId": request.source_clip_asset_id,
        "bridgeRightClipAssetId": request.bridge_right_clip_asset_id,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "personTrackId": request.person_track_id,
        "replacementMode": request.replacement_mode,
        "timelineContextRef": "lineage.timeline",
    }
    if request.mode == "replace_person":
        # Honest defaults: a replacement asset only claims active detection/
        # tracking/replacement when the adapter actually ran a real masked-control
        # path and reported it via replacement_status (sc-1483/sc-1487). Adapters
        # that did not (mocked previews, not-yet-upgraded paths) leave these false.
        replacement_settings = {
            "personDetectionActive": False,
            "personTrackingActive": False,
            "replacementActive": False,
        }
        if replacement_status:
            replacement_settings.update(replacement_status)
        normalized_settings.update(replacement_settings)
    return {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": project_id,
        "generationSetId": generation_set_id,
        "type": "video",
        "displayName": f"{request.prompt[:56] or 'Generated video'}",
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": mime_type,
            "width": request.width,
            "height": request.height,
            "duration": request.duration,
            "fps": request.fps,
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
            "normalizedSettings": normalized_settings,
            "rawAdapterSettings": raw_settings,
        },
        "lineage": {
            "parents": parents,
            "sourceAssetId": request.source_asset_id,
            "lastFrameAssetId": request.last_frame_asset_id,
            "sourceClipAssetId": request.source_clip_asset_id,
            "bridgeRightClipAssetId": request.bridge_right_clip_asset_id,
            "personTrackId": request.person_track_id,
            "characterId": request.character_id,
            "characterLookId": request.character_look_id,
            "replacementMode": request.replacement_mode,
            "sourceTimestamp": None,
            "timeline": timeline_context,
            "jobId": job_id,
        },
    }
