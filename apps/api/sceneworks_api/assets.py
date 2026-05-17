from __future__ import annotations

import json
import mimetypes
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sceneworks_shared import (
    ensure_project_db_ready,
    find_asset_sidecar_path,
    index_asset,
    purge_asset,
    read_json,
    reindex_project,
    utc_now,
    write_json,
)

from .projects import find_project_path


router = APIRouter(prefix="/projects/{project_id}", tags=["assets"])

ASSET_SIDECAR_PATTERN = "*.sceneworks.json"
MEDIA_FOLDERS = ("assets/images", "assets/videos", "assets/uploads", "assets/frames", "assets/renders")
ALLOWED_IMPORT_PREFIXES = ("image/", "video/")


class AssetStatusUpdate(BaseModel):
    favorite: bool | None = None
    rating: int | None = Field(default=None, ge=0, le=5)
    rejected: bool | None = None
    trashed: bool | None = None


def safe_filename(value: str, fallback: str = "upload") -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem.strip()).strip("-").lower()
    return slug[:64] or fallback


def media_type_for_mime(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    raise HTTPException(status_code=400, detail="Only image and video uploads are supported")


def project_has_sidecars(project_path: Path) -> bool:
    for folder in MEDIA_FOLDERS:
        if any((project_path / folder).glob(ASSET_SIDECAR_PATTERN)):
            return True
    return False


def normalize_asset(project_id: str, project_path: Path, sidecar_path: Path) -> dict[str, Any]:
    asset = read_json(sidecar_path)
    rel_media = asset.get("file", {}).get("path", "")
    if rel_media:
        normalized_path = rel_media.replace("\\", "/")
        asset["url"] = f"/api/v1/projects/{project_id}/files/{normalized_path}"
    asset["sidecarPath"] = str(sidecar_path.relative_to(project_path)).replace("\\", "/")
    return asset


def find_asset_sidecar(project_path: Path, asset_id: str) -> Path:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is not None:
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
    seen_asset_ids = set()
    ensure_project_db_ready(project_path)
    with sqlite3.connect(project_path / "project.db") as connection:
        total = connection.execute("select count(*) from assets").fetchone()[0]
    if total == 0 and project_has_sidecars(project_path):
        reindex_project(project_path)
    with sqlite3.connect(project_path / "project.db") as connection:
        rows = connection.execute(
            """
            select sidecar_path, file_path
              from assets
             where (? or rejected = 0)
               and (? or trashed = 0)
             order by created_at desc
            """,
            (int(includeRejected), int(includeTrashed)),
        ).fetchall()

    for sidecar_rel, file_rel in rows:
        candidates = []
        if sidecar_rel:
            candidates.append(project_path / sidecar_rel)
        if file_rel:
            candidates.append((project_path / file_rel).with_suffix(".sceneworks.json"))
        for sidecar_path in candidates:
            if not sidecar_path.exists():
                continue
            try:
                asset = normalize_asset(project_id, project_path, sidecar_path)
            except (OSError, json.JSONDecodeError):
                continue
            if asset.get("id") in seen_asset_ids:
                break
            seen_asset_ids.add(asset.get("id"))
            assets.append(asset)
            break

    return assets


@router.post("/assets", status_code=201)
def import_asset(project_id: str, request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    upload_dir = project_path / "assets" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    guessed_mime, _ = mimetypes.guess_type(file.filename or "")
    content_type = file.content_type or ""
    mime_type = guessed_mime if content_type in {"", "application/octet-stream"} else content_type
    mime_type = mime_type or "application/octet-stream"
    if not mime_type.startswith(ALLOWED_IMPORT_PREFIXES):
        raise HTTPException(status_code=400, detail="Only image and video uploads are supported")

    asset_id = f"asset_{uuid4().hex}"
    created_at = utc_now()
    extension = Path(file.filename or "").suffix.lower() or mimetypes.guess_extension(mime_type) or ".bin"
    filename = f"{safe_filename(file.filename or '', asset_id)}-{asset_id[-8:]}{extension}"
    media_path = upload_dir / filename
    media_rel = str(media_path.relative_to(project_path)).replace("\\", "/")

    try:
        with media_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    finally:
        file.file.close()

    if media_path.stat().st_size == 0:
        media_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    asset = {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": project_id,
        "generationSetId": None,
        "type": media_type_for_mime(mime_type),
        "displayName": Path(file.filename or filename).name,
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": mime_type,
            "width": None,
            "height": None,
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
            "mode": "upload",
            "model": "manual-import",
            "adapter": "api-upload",
            "prompt": file.filename or filename,
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {},
            "rawAdapterSettings": {"contentType": mime_type},
        },
        "lineage": {
            "parents": [],
            "sourceAssetId": None,
            "sourceTimestamp": None,
            "jobId": None,
        },
    }
    sidecar_path = media_path.with_suffix(".sceneworks.json")
    write_json(sidecar_path, asset)
    index_asset(project_path, asset)
    return normalize_asset(project_id, project_path, sidecar_path)


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
    index_asset(project_path, asset)
    return normalize_asset(project_id, project_path, sidecar_path)


@router.delete("/assets/{asset_id}")
def delete_asset(project_id: str, asset_id: str, request: Request) -> dict[str, str]:
    project_path = find_project_path(request.app.state.settings, project_id)
    sidecar_path = find_asset_sidecar(project_path, asset_id)
    asset = read_json(sidecar_path)
    media_path = project_path / asset.get("file", {}).get("path", "")
    status = asset.setdefault("status", {})
    status["trashed"] = True

    trash_dir = project_path / "trash" / asset_id
    trash_dir.mkdir(parents=True, exist_ok=True)
    if media_path.exists() and media_path.is_file():
        trashed_media_path = trash_dir / media_path.name
        shutil.move(str(media_path), trashed_media_path)
        asset["file"]["path"] = str(trashed_media_path.relative_to(project_path)).replace("\\", "/")
    trashed_sidecar_path = trash_dir / sidecar_path.name
    write_json(trashed_sidecar_path, asset)
    if sidecar_path != trashed_sidecar_path:
        sidecar_path.unlink(missing_ok=True)
    index_asset(project_path, asset, trashed_sidecar_path)
    return {"id": asset_id, "status": "trashed"}


@router.delete("/assets/{asset_id}/purge")
def purge_deleted_asset(project_id: str, asset_id: str, request: Request) -> dict[str, str]:
    project_path = find_project_path(request.app.state.settings, project_id)
    sidecar_path = find_asset_sidecar(project_path, asset_id)
    asset = read_json(sidecar_path)
    media_path = project_path / asset.get("file", {}).get("path", "")

    if media_path.exists() and media_path.is_file():
        media_path.unlink()
    sidecar_path.unlink(missing_ok=True)
    parent = sidecar_path.parent
    if parent.name == asset_id and parent.parent == project_path / "trash":
        shutil.rmtree(parent, ignore_errors=True)
    purge_asset(project_path, asset_id)
    return {"id": asset_id, "status": "purged"}


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
