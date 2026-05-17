from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from .jobs import publish, queue_summary
from .settings import Settings


NON_GPU_JOB_TYPES = {"model_download", "lora_import"}
MODEL_MANIFEST_FILES = ("builtin.models.jsonc", "user.models.jsonc")
LORA_MANIFEST_FILES = ("builtin.loras.jsonc", "user.loras.jsonc")
PROJECT_MODEL_MANIFEST = "manifests/project.models.jsonc"
PROJECT_LORA_MANIFEST = "manifests/project.loras.jsonc"

router = APIRouter(tags=["models"])


class DownloadSpec(BaseModel):
    provider: str = Field(min_length=1)
    repo: str | None = None
    files: list[str] = Field(default_factory=list)
    sizeBytes: int | None = Field(default=None, ge=0)
    sizeLabel: str | None = None
    url: str | None = None


class ManifestModel(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=160)
    family: str = Field(min_length=1, max_length=80)
    type: str = Field(min_length=1, max_length=40)
    adapter: str = Field(min_length=1, max_length=80)
    capabilities: list[str] = Field(default_factory=list)
    downloads: list[DownloadSpec] = Field(default_factory=list)
    paths: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    loraCompatibility: dict[str, list[str]] = Field(default_factory=dict)
    ui: dict[str, Any] = Field(default_factory=dict)
    builtIn: bool = False
    scope: str = "global"


class ManifestLora(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=160)
    scope: str = "global"
    category: str = "style"
    compatibleFamilies: list[str] = Field(default_factory=list)
    path: str = Field(min_length=1)
    triggerWords: list[str] = Field(default_factory=list)
    defaultWeight: float = 1
    builtIn: bool = False
    ui: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelDownloadRequest(BaseModel):
    projectId: str | None = None


class LoraImportRequest(BaseModel):
    path: str = Field(min_length=1)
    projectId: str | None = None
    scope: str = "global"
    id: str | None = None
    name: str | None = None
    category: str = "style"
    compatibleFamilies: list[str] = Field(default_factory=list)
    triggerWords: list[str] = Field(default_factory=list)
    defaultWeight: float = 1


def strip_jsonc(source: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
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
            index = source.find("\n", index)
            if index == -1:
                break
            output.append("\n")
            index += 1
            continue
        if char == "/" and next_char == "*":
            end = source.find("*/", index + 2)
            index = len(source) if end == -1 else end + 2
            continue
        output.append(char)
        index += 1

    return re.sub(r",(\s*[}\]])", r"\1", "".join(output))


def load_jsonc(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(strip_jsonc(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSONC: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion", 0) < 1:
        raise ValueError(f"{path} must include schemaVersion >= 1")
    entries = payload.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a {key} array")
    return entries


def write_manifest(path: Path, key: str, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "$schema": f"../../packages/schemas/{'model' if key == 'models' else 'lora'}-manifest.schema.json",
        "schemaVersion": 1,
        key: entries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower() or "lora"


def format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return None


class ManifestService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def storage_info(self) -> dict[str, Any]:
        return {
            "appManagedModels": str(self.settings.models_dir),
            "appManagedLoras": str(self.settings.loras_dir),
            "hfHome": str(self.settings.hf_home),
            "hfCache": str(self.settings.hf_cache_dir),
            "hfTokenConfigured": bool(self.settings.huggingface_token),
            "hfTokenSources": ["SCENEWORKS_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
            "userManifests": {
                "models": str(self.settings.manifests_dir / "user.models.jsonc"),
                "loras": str(self.settings.manifests_dir / "user.loras.jsonc"),
            },
        }

    def project_path(self, project_id: str | None) -> Path | None:
        if not project_id or not self.settings.registry_path.exists():
            return None
        with self.settings.registry_path.open("r", encoding="utf-8") as handle:
            for item in json.load(handle):
                if item.get("id") == project_id:
                    path = Path(item["path"])
                    return path if path.exists() else None
        return None

    def list_models(self, project_id: str | None = None) -> list[dict[str, Any]]:
        project_path = self.project_path(project_id)
        entries: dict[str, dict[str, Any]] = {}
        for filename in MODEL_MANIFEST_FILES:
            source = filename.removesuffix(".jsonc").replace(".", "-")
            for item in load_jsonc(self.settings.manifests_dir / filename, "models"):
                item = {**item, "source": source}
                entries[item["id"]] = {**entries.get(item["id"], {}), **item}
        if project_path:
            for item in load_jsonc(project_path / PROJECT_MODEL_MANIFEST, "models"):
                item = {**item, "source": "project", "scope": "project"}
                entries[item["id"]] = {**entries.get(item["id"], {}), **item}
        return [self._model_response(item, project_path) for item in entries.values()]

    def list_loras(
        self,
        *,
        project_id: str | None = None,
        model_id: str | None = None,
        include_unknown: bool = False,
        include_incompatible: bool = False,
    ) -> list[dict[str, Any]]:
        project_path = self.project_path(project_id)
        model_family = None
        if model_id:
            model = next((item for item in self.list_models(project_id) if item["id"] == model_id), None)
            model_family = model["family"] if model else None

        entries: dict[str, dict[str, Any]] = {}
        for filename in LORA_MANIFEST_FILES:
            source = filename.removesuffix(".jsonc").replace(".", "-")
            for item in load_jsonc(self.settings.manifests_dir / filename, "loras"):
                item = {**item, "source": source}
                entries[item["id"]] = {**entries.get(item["id"], {}), **item}
        if project_path:
            for item in load_jsonc(project_path / PROJECT_LORA_MANIFEST, "loras"):
                item = {**item, "source": "project", "scope": "project"}
                entries[item["id"]] = {**entries.get(item["id"], {}), **item}

        loras = [self._lora_response(item, project_path) for item in entries.values()]
        if not model_family:
            return loras

        filtered = []
        for item in loras:
            families = item["compatibleFamilies"]
            if model_family in families:
                filtered.append({**item, "compatibility": "compatible"})
            elif not families and include_unknown:
                filtered.append({**item, "compatibility": "unknown"})
            elif include_incompatible:
                filtered.append({**item, "compatibility": "incompatible"})
        return filtered

    def find_model(self, model_id: str, project_id: str | None = None) -> dict[str, Any] | None:
        return next((item for item in self.list_models(project_id) if item["id"] == model_id), None)

    def upsert_lora(self, request: LoraImportRequest, metadata: dict[str, Any]) -> dict[str, Any]:
        project_path = self.project_path(request.projectId) if request.scope == "project" else None
        target = project_path / PROJECT_LORA_MANIFEST if project_path else self.settings.manifests_dir / "user.loras.jsonc"
        entries = load_jsonc(target, "loras")
        lora_id = request.id or slugify(Path(request.path).stem)
        entry = {
            "id": lora_id,
            "name": request.name or Path(request.path).stem.replace("_", " ").replace("-", " ").title(),
            "scope": "project" if project_path else "global",
            "category": request.category,
            "compatibleFamilies": request.compatibleFamilies,
            "path": request.path,
            "triggerWords": request.triggerWords,
            "defaultWeight": request.defaultWeight,
            "builtIn": False,
            "metadata": metadata,
        }
        entries = [item for item in entries if item.get("id") != lora_id]
        entries.append(entry)
        write_manifest(target, "loras", entries)
        return self._lora_response({**entry, "source": "project" if project_path else "user-loras"}, project_path)

    def _model_response(self, item: dict[str, Any], project_path: Path | None) -> dict[str, Any]:
        try:
            model = ManifestModel.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"Invalid model manifest entry {item.get('id', '<missing id>')}: {exc}") from exc
        paths = {key: str(self.resolve_path(value, project_path, "model")) for key, value in model.paths.items()}
        installed = bool(paths) and all(Path(path).exists() for path in paths.values())
        size_bytes = sum(download.sizeBytes or 0 for download in model.downloads) or None
        if installed:
            status = "installed"
        elif model.downloads:
            status = "downloadable"
        else:
            status = "missing"
        return {
            **model.model_dump(),
            "source": item.get("source", "unknown"),
            "builtIn": model.builtIn or str(item.get("source", "")).startswith("builtin"),
            "resolvedPaths": paths,
            "status": status,
            "downloadSizeBytes": size_bytes,
            "downloadSizeLabel": format_bytes(size_bytes),
        }

    def _lora_response(self, item: dict[str, Any], project_path: Path | None) -> dict[str, Any]:
        try:
            lora = ManifestLora.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"Invalid LoRA manifest entry {item.get('id', '<missing id>')}: {exc}") from exc
        resolved = self.resolve_path(lora.path, project_path, "lora")
        return {
            **lora.model_dump(),
            "source": item.get("source", "unknown"),
            "builtIn": lora.builtIn or str(item.get("source", "")).startswith("builtin"),
            "resolvedPath": str(resolved),
            "status": "installed" if resolved.exists() else "missing",
            "compatibility": "unfiltered",
        }

    def resolve_path(self, raw: str, project_path: Path | None, kind: str) -> Path:
        replacements = {
            "DATA_DIR": str(self.settings.data_dir),
            "CONFIG_DIR": str(self.settings.config_dir),
            "MODEL_DIR": str(self.settings.models_dir),
            "LORA_DIR": str(self.settings.loras_dir),
            "HF_HOME": str(self.settings.hf_home),
            "HF_CACHE": str(self.settings.hf_cache_dir),
            "PROJECT_DIR": str(project_path or ""),
        }
        value = raw
        for key, replacement in replacements.items():
            value = value.replace(f"${{{key}}}", replacement)
        value = os.path.expandvars(os.path.expanduser(value))
        path = Path(value)
        if path.is_absolute():
            return path
        if project_path and kind == "lora" and raw.startswith(("loras/", "assets/", "cache/")):
            return (project_path / path).resolve()
        if raw.startswith(("models/", "loras/", "cache/")):
            return (self.settings.data_dir / path).resolve()
        base = self.settings.models_dir if kind == "model" else self.settings.loras_dir
        return (base / path).resolve()


def get_service(request: Request) -> ManifestService:
    return request.app.state.manifest_service


def model_download_payload(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "modelId": model["id"],
        "modelName": model["name"],
        "family": model["family"],
        "downloads": model["downloads"],
        "resolvedPaths": model["resolvedPaths"],
        "downloadSizeBytes": model["downloadSizeBytes"],
        "downloadSizeLabel": model["downloadSizeLabel"],
    }


def inspect_safetensors_metadata(path: Path) -> dict[str, Any]:
    if not path.exists() or path.suffix.lower() != ".safetensors":
        return {}
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        if len(header_length_raw) != 8:
            return {}
        header_length = struct.unpack("<Q", header_length_raw)[0]
        if header_length > 50_000_000:
            raise ValueError("Safetensors header is unexpectedly large")
        header = json.loads(handle.read(header_length).decode("utf-8"))
    metadata = header.get("__metadata__", {})
    return metadata if isinstance(metadata, dict) else {}


@router.get("/models")
def list_models(request: Request, projectId: str | None = Query(default=None)) -> dict[str, Any]:
    try:
        return {"models": get_service(request).list_models(projectId), "storage": get_service(request).storage_info()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/{model_id}/download", status_code=201)
def download_model(model_id: str, payload: ModelDownloadRequest, request: Request) -> dict[str, Any]:
    service = get_service(request)
    model = service.find_model(model_id, payload.projectId)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if model["status"] == "installed":
        return {"job": None, "model": model, "message": "Model is already installed."}
    if not model["downloads"]:
        raise HTTPException(status_code=400, detail="Model has no downloadable source")

    job = request.app.state.jobs_store.create_job(
        job_type="model_download",
        project_id=payload.projectId,
        project_name=None,
        payload=model_download_payload(model),
        requested_gpu="none",
        requires_gpu=False,
        message="Queued model download. This job does not reserve a GPU worker.",
    )
    job = request.app.state.jobs_store.update_job_progress(
        job["id"],
        status="completed",
        stage="completed",
        progress=1,
        message="Model download job placeholder completed. File transfer adapter is not installed yet.",
        result={"modelId": model["id"], "placeholder": True},
    )
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return {"job": job, "model": model}


@router.get("/loras")
def list_loras(
    request: Request,
    projectId: str | None = Query(default=None),
    modelId: str | None = Query(default=None),
    includeUnknown: bool = Query(default=False),
    includeIncompatible: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        return {
            "loras": get_service(request).list_loras(
                project_id=projectId,
                model_id=modelId,
                include_unknown=includeUnknown,
                include_incompatible=includeIncompatible,
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/loras/import", status_code=201)
def import_lora(payload: LoraImportRequest, request: Request) -> dict[str, Any]:
    service = get_service(request)
    path = service.resolve_path(payload.path, service.project_path(payload.projectId), "lora")
    try:
        metadata = inspect_safetensors_metadata(path)
        lora = service.upsert_lora(payload, metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = request.app.state.jobs_store.create_job(
        job_type="lora_import",
        project_id=payload.projectId,
        project_name=None,
        payload={"path": payload.path, "resolvedPath": str(path), "metadata": metadata, "loraId": lora["id"]},
        requested_gpu="none",
        requires_gpu=False,
        message="Inspected LoRA metadata.",
    )
    job = request.app.state.jobs_store.update_job_progress(
        job["id"],
        status="completed",
        stage="completed",
        progress=1,
        message="LoRA import metadata inspection completed.",
        result={"loraId": lora["id"], "metadata": metadata},
    )
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return {"job": job, "lora": lora, "metadata": metadata}
