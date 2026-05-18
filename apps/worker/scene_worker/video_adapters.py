from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable
from uuid import uuid4

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

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

from .image_adapters import write_json
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
        "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
        "recommendedMaxDuration": 7,
        "hardMaxDuration": 8,
        "steps": {"fast": 12, "balanced": 20, "best": 30},
        "durationHint": "Keep clips shorter until local looping behavior is validated.",
    },
}


ASSET_FOLDERS = ("assets/images", "assets/videos", "assets/uploads", "assets/frames", "assets/renders")


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
        image = load_ffmpeg_frame(media_path, timestamp)
    if image is None:
        return None

    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (18, 17, 15))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def load_seekable_image_frame(media_path: Path, timestamp: float, duration: Any = None) -> Image.Image | None:
    try:
        image = Image.open(media_path)
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
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


def load_ffmpeg_frame(media_path: Path, timestamp: float) -> Image.Image | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not available for frame extraction.")
    with tempfile.TemporaryDirectory(prefix="sceneworks-frame-") as temp_dir:
        frame_path = Path(temp_dir) / "frame.png"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0, timestamp):.3f}",
            "-i",
            str(media_path),
            "-frames:v",
            "1",
            "-f",
            "image2",
            str(frame_path),
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0 or not frame_path.exists():
            tail = "\n".join((result.stderr or "").splitlines()[-6:]).strip()
            if tail:
                raise RuntimeError(f"ffmpeg could not extract a frame at {timestamp:.3f}s: {tail}")
            return None
        try:
            return Image.open(frame_path).convert("RGB")
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
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
            "mimeType": "image/webp",
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
            "adapter": "procedural_video",
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


def run_frame_extract(
    *,
    settings: WorkerSettings,
    job: dict[str, Any],
    progress: ProgressCallback,
    cancel_requested: CancelCallback,
) -> dict[str, Any]:
    payload = job["payload"]
    project_id = payload["projectId"]
    project_path = find_project_path(settings.data_dir / "recent-projects.json", project_id)
    frames_dir = project_path / "assets" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (project_path / "recipes").mkdir(parents=True, exist_ok=True)

    source_asset_id = payload.get("sourceAssetId")
    timestamp = safe_float(payload.get("sourceTimestamp"), 0, 0, 3600)
    source_sidecar_path = find_asset_sidecar_path(project_path, source_asset_id)
    if source_sidecar_path is None:
        raise FileNotFoundError(f"Source asset not found: {source_asset_id}")
    source_asset = read_json(source_sidecar_path)
    source_media_path = project_path / source_asset.get("file", {}).get("path", "")
    if not source_media_path.exists():
        raise FileNotFoundError(f"Source media not found: {source_media_path}")

    progress("running", "extracting", 0.25, "Extracting timeline frame.")
    if cancel_requested():
        raise InterruptedError("Frame extraction canceled before reading media.")

    image = load_source_frame(project_path, source_asset_id, timestamp, 1920, 1080)
    if image is None:
        raise RuntimeError("Could not decode a frame from the selected clip.")

    asset_id = f"asset_{uuid4().hex}"
    created_at = utc_now()
    filename = f"{created_at[:10]}_frame_{asset_id[-8:]}.png"
    media_rel = f"assets/frames/{filename}"
    media_path = project_path / media_rel
    temp_path = media_path.with_suffix(".tmp.png")
    image.save(temp_path, "PNG")
    temp_path.replace(media_path)

    timeline_id = payload.get("timelineId")
    timeline_item_id = payload.get("timelineItemId")
    intended_use = payload.get("intendedUse", "reuse")
    asset = {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": project_id,
        "generationSetId": None,
        "type": "frame",
        "displayName": f"Frame {timestamp:.2f}s from {source_asset.get('displayName', 'clip')}",
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": image.width,
            "height": image.height,
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
            "mode": "frame_extract",
            "model": "timeline-frame-extract",
            "adapter": "ffmpeg-frame-extract",
            "prompt": f"Extract frame at {timestamp:.2f}s",
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "timelineId": timeline_id,
                "timelineItemId": timeline_item_id,
                "playheadSeconds": payload.get("playheadSeconds"),
                "sourceTimestamp": timestamp,
                "intendedUse": intended_use,
            },
            "rawAdapterSettings": {"sourcePath": str(source_media_path.relative_to(project_path)).replace("\\", "/")},
        },
        "lineage": {
            "parents": [source_asset_id],
            "sourceAssetId": source_asset_id,
            "sourceTimestamp": timestamp,
            "timelineId": timeline_id,
            "timelineItemId": timeline_item_id,
            "intendedUse": intended_use,
            "jobId": job["id"],
        },
    }
    sidecar_path = media_path.with_suffix(".sceneworks.json")
    progress("saving", "saving", 0.85, "Saving extracted frame asset.")
    if cancel_requested():
        media_path.unlink(missing_ok=True)
        raise InterruptedError("Frame extraction canceled before asset promotion.")
    write_json(sidecar_path, asset)
    write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
    index_asset(project_path, asset)
    return {
        "assetIds": [asset_id],
        "assets": [asset],
        "sourceAssetId": source_asset_id,
        "sourceTimestamp": timestamp,
        "timelineId": timeline_id,
        "timelineItemId": timeline_item_id,
    }
