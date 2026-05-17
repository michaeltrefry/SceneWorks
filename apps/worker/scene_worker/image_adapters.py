from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from .settings import WorkerSettings


ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]


MODEL_TARGETS = {
    "z_image_turbo": {
        "label": "Z-Image-Turbo",
        "family": "z-image",
        "supportsEdit": False,
        "steps": 8,
    },
    "z_image_edit": {
        "label": "Z-Image-Edit",
        "family": "z-image",
        "supportsEdit": True,
        "steps": 8,
    },
    "qwen_image_edit": {
        "label": "Qwen Image Edit",
        "family": "qwen-image",
        "supportsEdit": True,
        "steps": 20,
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
    width: int
    height: int
    style_preset: str
    loras: list[dict[str, Any]]
    source_asset_id: str | None
    advanced: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return (slug or "image")[:42]


def safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def load_registry(data_dir: Path) -> list[dict[str, Any]]:
    registry_path = data_dir / "recent-projects.json"
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_project_path(settings: WorkerSettings, project_id: str) -> Path:
    for project in load_registry(settings.data_dir):
        if project.get("id") == project_id:
            return Path(project["path"])
    raise RuntimeError(f"Project not found: {project_id}")


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
        width=safe_int(payload.get("width"), 1024, 256, 2048),
        height=safe_int(payload.get("height"), 1024, 256, 2048),
        style_preset=payload.get("stylePreset", "cinematic"),
        loras=payload.get("loras", []),
        source_asset_id=payload.get("sourceAssetId"),
        advanced=payload.get("advanced", {}),
    )


class ProceduralImageAdapter:
    id = "procedural_preview"

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        request = image_request_from_job(job)
        project_path = find_project_path(settings, request.project_id)
        for folder in ("assets/images", "generation-sets", "recipes"):
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        created_at = utc_now()
        generation_set_id = f"genset_{uuid4().hex}"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        prompt_slug = slugify(request.prompt)
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
            "count": request.count,
            "createdAt": created_at,
        }
        write_json(project_path / "generation-sets" / f"{generation_set_id}.json", generation_set)

        for index in range(request.count):
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            asset_id = f"asset_{uuid4().hex}"
            seed = resolve_seed(request.seed, request.prompt, index)
            filename = f"{date_slug}_{request.model}_{prompt_slug}_{index + 1:04d}.png"
            media_rel = f"assets/images/{filename}"
            media_path = project_path / media_rel
            sidecar_path = media_path.with_suffix(".sceneworks.json")
            preview = render_preview_image(request, model_target, seed, index)
            preview.save(media_path, "PNG")

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
            )
            write_json(sidecar_path, asset)
            write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
            index_project_db(project_path, asset)
            assets.append(asset)
            progress(
                "running",
                "generating",
                0.2 + ((index + 1) / request.count) * 0.55,
                f"Generated preview image {index + 1} of {request.count}.",
            )

        return {
            "generationSetId": generation_set_id,
            "assetIds": [asset["id"] for asset in assets],
            "assets": assets,
            "adapter": self.id,
            "model": request.model,
        }


def model_supports_edit(model_id: str) -> bool:
    return bool(MODEL_TARGETS.get(model_id, {}).get("supportsEdit"))


def resolve_seed(seed: int | None, prompt: str, index: int) -> int:
    if seed is not None:
        return int(seed) + index
    digest = hashlib.sha256(f"{prompt}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def render_preview_image(request: ImageRequest, model_target: dict[str, Any], seed: int, index: int) -> Image.Image:
    width = min(request.width, 1280)
    height = min(request.height, 1280)
    digest = hashlib.sha256(f"{request.prompt}:{request.style_preset}:{seed}".encode("utf-8")).digest()
    base = (digest[0], digest[1], digest[2])
    accent = (digest[9], digest[10], digest[11])
    image = Image.new("RGB", (width, height), base)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            mix = (x / max(1, width - 1) * 0.56) + (y / max(1, height - 1) * 0.44)
            wave = ((x * digest[3] + y * digest[4] + seed) % 255) / 255
            pixels[x, y] = tuple(
                int(base[channel] * (1 - mix) + accent[channel] * mix * 0.85 + wave * 34)
                for channel in range(3)
            )

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
            "adapter": "procedural_preview",
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
            },
            "rawAdapterSettings": {
                **request.advanced,
                "targetSteps": model_target["steps"],
                "previewRenderer": True,
            },
        },
        "lineage": {
            "parents": [request.source_asset_id] if request.source_asset_id else [],
            "sourceAssetId": request.source_asset_id,
            "sourceTimestamp": None,
            "jobId": job_id,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def index_project_db(project_path: Path, asset: dict[str, Any]) -> None:
    db_path = project_path / "project.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            create table if not exists assets (
              id text primary key,
              type text not null,
              display_name text not null,
              file_path text not null,
              generation_set_id text,
              created_at text not null,
              favorite integer not null default 0,
              rating integer not null default 0,
              rejected integer not null default 0,
              trashed integer not null default 0
            )
            """
        )
        connection.execute(
            """
            insert or replace into assets (
              id, type, display_name, file_path, generation_set_id, created_at,
              favorite, rating, rejected, trashed
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset["id"],
                asset["type"],
                asset["displayName"],
                asset["file"]["path"],
                asset["generationSetId"],
                asset["createdAt"],
                int(asset["status"]["favorite"]),
                int(asset["status"]["rating"]),
                int(asset["status"]["rejected"]),
                int(asset["status"]["trashed"]),
            ),
        )
