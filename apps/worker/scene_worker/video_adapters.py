from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import importlib
import os
import warnings
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable
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
from .image_adapters import select_torch_device, select_torch_dtype, write_json
from .lora_adapters import apply_loras_to_pipeline, assert_loras_supported
from .settings import WorkerSettings


Image.MAX_IMAGE_PIXELS = 64_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]


VIDEO_MODEL_TARGETS: dict[str, dict[str, Any]] = {
    "ltx_2_3": {
        "label": "LTX-2.3",
        "family": "ltx-video",
        "adapter": "ltx_video",
        "repo": "Lightricks/LTX-2.3",
        "fallbackRepo": "Lightricks/LTX-Video",
        "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge"],
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
        assert_loras_supported(request.loras, self.id)
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


class DiffusersVideoAdapter(VideoGenerationAdapter):
    id = "diffusers_video"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._pipeline_key_value: str | None = None
        self._loaded_models: set[str] = set()
        self._loaded_lora_key = ""
        self._loaded_lora_names: list[str] = []

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
        pipe = self._load_pipeline(request, target)
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

    def _load_pipeline(self, request: VideoRequest, target: dict[str, Any]) -> Any:
        key = self._pipeline_key(request, target)
        repo = self._repo_for_request(request, target)
        if self._pipeline is not None and self._pipeline_key_value == key:
            self._loaded_models.update({request.model, self._repo_for_request(request, target)})
            return self._pipeline

        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        device = select_torch_device(torch)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        self._evict_pipeline(torch)
        pipeline_class = self._pipeline_class(diffusers, request, target)
        kwargs: dict[str, Any] = {"torch_dtype": dtype}

        if target["adapter"] == "wan_video":
            vae_class = getattr(diffusers, "AutoencoderKLWan", None)
            if vae_class is not None:
                kwargs["vae"] = vae_class.from_pretrained(repo, subfolder="vae", torch_dtype=dtype)
            if request.mode != "text_to_video":
                transformers = importlib.import_module("transformers")
                image_encoder_class = getattr(transformers, "CLIPVisionModel", None)
                if image_encoder_class is not None:
                    kwargs["image_encoder"] = image_encoder_class.from_pretrained(repo, subfolder="image_encoder", torch_dtype=dtype)

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
        self._loaded_lora_key = ""
        self._loaded_lora_names = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_loras(self, pipe: Any, request: VideoRequest, target: dict[str, Any]) -> None:
        loaded_key, loaded_names = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=target["adapter"],
            previous_key=self._loaded_lora_key,
            previous_adapter_names=self._loaded_lora_names,
        )
        self._loaded_lora_key = loaded_key
        self._loaded_lora_names = loaded_names

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
            kwargs["guidance_scale"] = self._guidance_scale(request, 5.0)
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


def create_video_adapter() -> VideoGenerationAdapter:
    requested = os.getenv("SCENEWORKS_VIDEO_ADAPTER", "").strip()
    if not requested or requested == "diffusers_video":
        return DiffusersVideoAdapter()
    if requested in {"procedural", "procedural_video"}:
        return ProceduralVideoAdapter()
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


def load_source_image(project_path: Path, asset_id: str | None, width: int, height: int) -> Image.Image | None:
    if not asset_id:
        return None
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        return None
    payload = read_json(sidecar_path)
    media_path = project_path / payload.get("file", {}).get("path", "")
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
        return [Image.fromarray(frame).convert("RGB") for frame in array]
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        return [frame.convert("RGB") if hasattr(frame, "convert") else Image.fromarray(frame).convert("RGB") for frame in frames[0]]
    if isinstance(frames, list):
        return [frame.convert("RGB") if hasattr(frame, "convert") else Image.fromarray(frame).convert("RGB") for frame in frames]
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
    track = read_person_track(project_path, track_id)
    boxes = []
    if track:
        boxes = [frame.get("box") for frame in track.get("frames", []) if isinstance(frame, dict) and isinstance(frame.get("box"), dict)]
        selected_box = track.get("selectedDetection", {}).get("box")
        if not boxes and isinstance(selected_box, dict):
            boxes = [selected_box]
    if not boxes:
        raise RuntimeError(f"Person track has no usable boxes: {track_id}.")

    masks = []
    for index in range(count):
        box = boxes[min(len(boxes) - 1, round(index * (len(boxes) - 1) / max(1, count - 1)))]
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        left = int(safe_float(box.get("x"), 0, 0, 1) * width)
        top = int(safe_float(box.get("y"), 0, 0, 1) * height)
        right = int((safe_float(box.get("x"), 0, 0, 1) + safe_float(box.get("width"), 0, 0, 1)) * width)
        bottom = int((safe_float(box.get("y"), 0, 0, 1) + safe_float(box.get("height"), 0, 0, 1)) * height)
        padding_x = max(8, int(width * 0.03))
        padding_y = max(8, int(height * 0.03))
        draw.rectangle(
            (
                max(0, left - padding_x),
                max(0, top - padding_y),
                min(width, right + padding_x),
                min(height, bottom + padding_y),
            ),
            fill=255,
        )
        masks.append(mask)
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
        normalized_settings.update(
            {
                "personDetectionActive": False,
                "personTrackingActive": False,
                "replacementActive": False,
            }
        )
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
