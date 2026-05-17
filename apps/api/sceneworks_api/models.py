from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field


router = APIRouter(prefix="/models", tags=["models"])
loras_router = APIRouter(prefix="/loras", tags=["loras"])


class ModelDownloadRequest(BaseModel):
    requestedGpu: str = "auto"


class LoraImportRequest(BaseModel):
    loraId: str | None = None
    name: str | None = None
    repo: str | None = None
    sourcePath: str | None = None
    files: list[str] = Field(default_factory=list)
    family: str | None = None


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


def manifest_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def load_manifest_cached(path_text: str, signature: tuple[int, int], key: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_text)
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.loads(strip_jsonc_comments(handle.read()))
    return tuple(payload.get(key, []))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [dict(item) for item in load_manifest_cached(str(path), manifest_signature(path), "models")]


def load_lora_manifest(path: Path) -> list[dict[str, Any]]:
    return [dict(item) for item in load_manifest_cached(str(path), manifest_signature(path), "loras")]


def safe_download_dir(repo: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", repo).strip("_") or "download"


def model_download(model: dict[str, Any]) -> dict[str, Any] | None:
    for download in model.get("downloads", []):
        if download.get("provider") == "huggingface" and download.get("repo"):
            return download
    return None


def model_catalog(request: Request) -> list[dict[str, Any]]:
    settings = request.app.state.settings
    manifest_dir = settings.config_dir / "manifests"
    builtin = load_manifest(manifest_dir / "builtin.models.jsonc")
    user = load_manifest(manifest_dir / "user.models.jsonc")
    by_id = {model["id"]: model for model in builtin if "id" in model}
    for model in user:
        model_id = model.get("id")
        if model_id:
            by_id[model_id] = {**by_id.get(model_id, {}), **model}

    models = []
    for model in by_id.values():
        download = model_download(model)
        installed_path = None
        installed = False
        if download:
            installed_path = settings.data_dir / "models" / safe_download_dir(download["repo"])
            installed = installed_path.is_dir() and any(installed_path.iterdir())
        models.append(
            {
                **model,
                "downloadable": bool(download),
                "installState": "installed" if installed else "missing",
                "installedPath": str(installed_path) if installed_path else None,
            }
        )
    return sorted(models, key=lambda model: (model.get("type", ""), model.get("name", "")))


def lora_catalog(request: Request) -> list[dict[str, Any]]:
    settings = request.app.state.settings
    manifest_dir = settings.config_dir / "manifests"
    builtin = load_lora_manifest(manifest_dir / "builtin.loras.jsonc")
    user = load_lora_manifest(manifest_dir / "user.loras.jsonc")
    by_id = {lora["id"]: lora for lora in builtin if "id" in lora}
    for lora in user:
        lora_id = lora.get("id")
        if lora_id:
            by_id[lora_id] = {**by_id.get(lora_id, {}), **lora}
    return sorted(by_id.values(), key=lambda lora: (lora.get("family", ""), lora.get("name", "")))


def lora_families(lora: dict[str, Any]) -> set[str]:
    compatibility = lora.get("compatibility", {})
    values = (
        lora.get("families")
        or lora.get("compatibleFamilies")
        or lora.get("modelFamilies")
        or compatibility.get("families")
        or ([lora["family"]] if lora.get("family") else [])
    )
    values = values if isinstance(values, list) else [values]
    return {str(value) for value in values}


@router.get("")
def list_models(request: Request) -> list[dict[str, Any]]:
    return model_catalog(request)


@router.post("/{model_id}/download", status_code=201)
def create_model_download_job(model_id: str, payload: ModelDownloadRequest, request: Request) -> dict[str, Any]:
    model = next((item for item in model_catalog(request) if item.get("id") == model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    download = model_download(model)
    if not download:
        raise HTTPException(status_code=400, detail="Model does not define a Hugging Face download")

    job = request.app.state.jobs_store.create_job(
        job_type="model_download",
        project_id=None,
        project_name=None,
        payload={
            "modelId": model_id,
            "modelName": model.get("name", model_id),
            "provider": download["provider"],
            "repo": download["repo"],
            "files": download.get("files", []),
            "targetDir": str(request.app.state.settings.data_dir / "models" / safe_download_dir(download["repo"])),
        },
        requested_gpu=payload.requestedGpu or "auto",
    )
    request.app.state.event_hub.publish("job.updated", job)
    from .jobs import queue_summary

    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job


@loras_router.get("")
def list_loras(request: Request, modelFamily: str | None = Query(default=None)) -> list[dict[str, Any]]:
    items = lora_catalog(request)
    if modelFamily:
        items = [item for item in items if modelFamily in lora_families(item)]
    return items


@loras_router.post("/import", status_code=201)
def create_lora_import_job(payload: LoraImportRequest, request: Request) -> dict[str, Any]:
    if not payload.repo and not payload.sourcePath:
        raise HTTPException(status_code=400, detail="Provide a Hugging Face repo or source path")
    job = request.app.state.jobs_store.create_job(
        job_type="lora_import",
        project_id=None,
        project_name=None,
        payload=payload.model_dump(exclude_none=True),
        requested_gpu="auto",
    )
    request.app.state.event_hub.publish("job.updated", job)
    from .jobs import queue_summary

    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job
