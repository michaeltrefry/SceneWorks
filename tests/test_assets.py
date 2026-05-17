from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sceneworks_api.assets import (
    AssetStatusUpdate,
    delete_asset,
    get_project_file,
    purge_deleted_asset,
    update_asset_status,
)
from sceneworks_shared import index_asset, write_json


def request_for_project(tmp_path, project_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_path = data_dir / "recent-projects.json"
    registry_path.write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    settings = SimpleNamespace(registry_path=registry_path)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))


def test_project_file_rejects_path_traversal(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        get_project_file("project-1", "../outside.txt", request_for_project(tmp_path, project_path))

    assert exc_info.value.status_code == 400


def test_status_patch_updates_project_db(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    image_dir = project_path / "assets" / "images"
    image_dir.mkdir(parents=True)
    asset = {
        "id": "asset-1",
        "type": "image",
        "displayName": "Image",
        "createdAt": "2026-05-17T00:00:00Z",
        "generationSetId": None,
        "file": {"path": "assets/images/image.png"},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
    }
    write_json(image_dir / "image.sceneworks.json", asset)
    index_asset(project_path, asset)

    update_asset_status(
        "project-1",
        "asset-1",
        AssetStatusUpdate(favorite=True, rating=4, rejected=True),
        request_for_project(tmp_path, project_path),
    )

    with sqlite3.connect(project_path / "project.db") as connection:
        row = connection.execute("select favorite, rating, rejected, trashed from assets where id = ?", ("asset-1",)).fetchone()

    assert row == (1, 4, 1, 0)


def test_delete_soft_trashes_asset_and_purge_removes_it(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    image_dir = project_path / "assets" / "images"
    image_dir.mkdir(parents=True)
    media_path = image_dir / "image.png"
    media_path.write_bytes(b"image")
    asset = {
        "id": "asset-1",
        "projectId": "project-1",
        "type": "image",
        "displayName": "Image",
        "createdAt": "2026-05-17T00:00:00Z",
        "generationSetId": None,
        "file": {"path": "assets/images/image.png"},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
    }
    sidecar_path = image_dir / "image.sceneworks.json"
    write_json(sidecar_path, asset)
    index_asset(project_path, asset)
    request = request_for_project(tmp_path, project_path)

    deleted = delete_asset("project-1", "asset-1", request)

    assert deleted == {"id": "asset-1", "status": "trashed"}
    assert not media_path.exists()
    trash_sidecar = project_path / "trash" / "asset-1" / "image.sceneworks.json"
    assert trash_sidecar.exists()
    with sqlite3.connect(project_path / "project.db") as connection:
        row = connection.execute("select file_path, trashed, sidecar_path from assets where id = ?", ("asset-1",)).fetchone()
    assert row[0] == "trash/asset-1/image.png"
    assert row[1] == 1
    assert row[2] == "trash/asset-1/image.sceneworks.json"

    purged = purge_deleted_asset("project-1", "asset-1", request)

    assert purged == {"id": "asset-1", "status": "purged"}
    assert not (project_path / "trash" / "asset-1").exists()
    with sqlite3.connect(project_path / "project.db") as connection:
        assert connection.execute("select count(*) from assets where id = ?", ("asset-1",)).fetchone()[0] == 0
