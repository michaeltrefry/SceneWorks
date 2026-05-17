from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .projects import find_project_path


router = APIRouter(prefix="/projects/{project_id}", tags=["assets"])

ASSET_SIDECAR_PATTERN = "*.sceneworks.json"
MEDIA_FOLDERS = ("assets/images", "assets/videos", "assets/uploads", "assets/frames", "assets/renders")


class AssetStatusUpdate(BaseModel):
    favorite: bool | None = None
    rating: int | None = Field(default=None, ge=0, le=5)
    rejected: bool | None = None
    trashed: bool | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def normalize_asset(project_id: str, project_path: Path, sidecar_path: Path) -> dict[str, Any]:
    asset = read_json(sidecar_path)
    rel_media = asset.get("file", {}).get("path", "")
    if rel_media:
        normalized_path = rel_media.replace("\\", "/")
        asset["url"] = f"/api/v1/projects/{project_id}/files/{normalized_path}"
    asset["sidecarPath"] = str(sidecar_path.relative_to(project_path)).replace("\\", "/")
    return asset


def find_asset_sidecar(project_path: Path, asset_id: str) -> Path:
    for folder in MEDIA_FOLDERS:
        for sidecar_path in (project_path / folder).glob(ASSET_SIDECAR_PATTERN):
            try:
                payload = read_json(sidecar_path)
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("id") == asset_id:
                return sidecar_path
    raise HTTPException(status_code=404, detail="Asset not found")


@router.get("/assets")
def list_assets(
    project_id: str,
    request: Request,
    includeRejected: bool = Query(default=False),
    includeTrashed: bool = Query(default=False),
) -> list[dict[str, Any]]:
    project_path = find_project_path(request.app.state.settings, project_id)
    assets = []
    for folder in MEDIA_FOLDERS:
        for sidecar_path in (project_path / folder).glob(ASSET_SIDECAR_PATTERN):
            try:
                asset = normalize_asset(project_id, project_path, sidecar_path)
            except (OSError, json.JSONDecodeError):
                continue
            status = asset.get("status", {})
            if status.get("rejected") and not includeRejected:
                continue
            if status.get("trashed") and not includeTrashed:
                continue
            assets.append(asset)

    return sorted(assets, key=lambda item: item.get("createdAt", ""), reverse=True)


@router.patch("/assets/{asset_id}/status")
def update_asset_status(
    project_id: str,
    asset_id: str,
    payload: AssetStatusUpdate,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    sidecar_path = find_asset_sidecar(project_path, asset_id)
    asset = read_json(sidecar_path)
    status = asset.setdefault("status", {})
    changes = payload.model_dump(exclude_none=True)
    status.update(changes)
    write_json(sidecar_path, asset)
    return normalize_asset(project_id, project_path, sidecar_path)


@router.delete("/assets/{asset_id}")
def delete_asset(project_id: str, asset_id: str, request: Request) -> dict[str, str]:
    project_path = find_project_path(request.app.state.settings, project_id)
    sidecar_path = find_asset_sidecar(project_path, asset_id)
    asset = read_json(sidecar_path)
    media_path = project_path / asset.get("file", {}).get("path", "")

    if media_path.exists() and media_path.is_file():
        media_path.unlink()
    sidecar_path.unlink()
    return {"id": asset_id, "status": "deleted"}


@router.get("/files/{relative_path:path}")
def get_project_file(project_id: str, relative_path: str, request: Request) -> FileResponse:
    project_path = find_project_path(request.app.state.settings, project_id)
    target = (project_path / relative_path).resolve()
    try:
        target.relative_to(project_path.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project file path") from None
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)
