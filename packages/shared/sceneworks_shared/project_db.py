from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .utils import read_json


ASSET_SIDECAR_PATTERN = "*.sceneworks.json"
ASSET_FOLDERS = (
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "assets/documents",
    "assets/poses",
    "trash",
)


def apply_project_migrations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        create table if not exists project_metadata (
          key text primary key,
          value text not null
        )
        """
    )
    connection.execute("insert or replace into project_metadata (key, value) values (?, ?)", ("schemaVersion", "1"))
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
        create table if not exists generation_sets (
          id text primary key,
          mode text not null,
          model text not null,
          prompt text not null,
          created_at text not null,
          job_id text
        )
        """
    )
    connection.execute(
        """
        create table if not exists timelines (
          id text primary key,
          name text not null,
          file_path text not null,
          aspect_ratio text not null,
          width integer not null,
          height integer not null,
          fps integer not null,
          duration real not null default 0,
          created_at text not null,
          updated_at text not null
        )
        """
    )
    _ensure_column(connection, "assets", "sidecar_path", "text")


def ensure_project_db_ready(project_path: Path) -> None:
    with sqlite3.connect(project_path / "project.db") as connection:
        apply_project_migrations(connection)


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in connection.execute(f"pragma table_info({table})").fetchall()}
    if column not in columns:
        connection.execute(f"alter table {table} add column {column} {definition}")


def index_asset(project_path: Path, asset: dict[str, Any], sidecar_path: Path | None = None) -> None:
    if sidecar_path is None:
        sidecar_path = (project_path / asset["file"]["path"]).with_suffix(".sceneworks.json")
    sidecar_rel = str(sidecar_path.relative_to(project_path)).replace("\\", "/")
    with sqlite3.connect(project_path / "project.db") as connection:
        apply_project_migrations(connection)
        _index_asset_on_connection(connection, asset, sidecar_rel)


def _index_asset_on_connection(connection: sqlite3.Connection, asset: dict[str, Any], sidecar_rel: str | None) -> None:
    connection.execute(
        """
        insert or replace into assets (
          id, type, display_name, file_path, generation_set_id, created_at,
          favorite, rating, rejected, trashed, sidecar_path
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset["id"],
            asset["type"],
            asset["displayName"],
            asset["file"]["path"],
            asset.get("generationSetId"),
            asset["createdAt"],
            int(asset.get("status", {}).get("favorite", False)),
            int(asset.get("status", {}).get("rating", 0)),
            int(asset.get("status", {}).get("rejected", False)),
            int(asset.get("status", {}).get("trashed", False)),
            sidecar_rel,
        ),
    )


def find_asset_record(project_path: Path, asset_id: str) -> dict[str, Any] | None:
    ensure_project_db_ready(project_path)
    with sqlite3.connect(project_path / "project.db") as connection:
        row = connection.execute(
            """
            select id, type, display_name, file_path, generation_set_id, created_at,
                   favorite, rating, rejected, trashed, sidecar_path
              from assets
             where id = ?
            """,
            (asset_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "type": row[1],
        "displayName": row[2],
        "filePath": row[3],
        "generationSetId": row[4],
        "createdAt": row[5],
        "favorite": bool(row[6]),
        "rating": row[7],
        "rejected": bool(row[8]),
        "trashed": bool(row[9]),
        "sidecarPath": row[10],
    }


def find_asset_records(project_path: Path, asset_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_ids = list(dict.fromkeys(asset_ids))
    if not unique_ids:
        return {}
    ensure_project_db_ready(project_path)
    placeholders = ",".join("?" for _ in unique_ids)
    with sqlite3.connect(project_path / "project.db") as connection:
        rows = connection.execute(
            f"""
            select id, type, display_name, file_path, generation_set_id, created_at,
                   favorite, rating, rejected, trashed, sidecar_path
              from assets
             where id in ({placeholders})
            """,
            unique_ids,
        ).fetchall()
    return {
        row[0]: {
            "id": row[0],
            "type": row[1],
            "displayName": row[2],
            "filePath": row[3],
            "generationSetId": row[4],
            "createdAt": row[5],
            "favorite": bool(row[6]),
            "rating": row[7],
            "rejected": bool(row[8]),
            "trashed": bool(row[9]),
            "sidecarPath": row[10],
        }
        for row in rows
    }


def find_asset_sidecar_path(project_path: Path, asset_id: str) -> Path | None:
    record = find_asset_record(project_path, asset_id)
    if record is not None:
        candidates = []
        if record.get("sidecarPath"):
            candidates.append(project_path / record["sidecarPath"])
        if record.get("filePath"):
            candidates.append((project_path / record["filePath"]).with_suffix(".sceneworks.json"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return _find_asset_sidecar_by_glob(project_path, asset_id)


def resolve_project_relative_path(project_path: Path, relative_path: str) -> Path | None:
    if not relative_path:
        return None
    try:
        project_root = project_path.resolve()
        resolved = (project_root / relative_path).resolve()
        resolved.relative_to(project_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _find_asset_sidecar_by_glob(project_path: Path, asset_id: str) -> Path | None:
    for folder in ASSET_FOLDERS:
        for sidecar_path in (project_path / folder).rglob(ASSET_SIDECAR_PATTERN):
            try:
                if read_json(sidecar_path).get("id") == asset_id:
                    return sidecar_path
            except (OSError, json.JSONDecodeError):
                continue
    return None


def purge_asset(project_path: Path, asset_id: str) -> None:
    with sqlite3.connect(project_path / "project.db") as connection:
        apply_project_migrations(connection)
        connection.execute("delete from assets where id = ?", (asset_id,))


def reindex_project(project_path: Path) -> dict[str, int]:
    db_path = project_path / "project.db"
    counts = {"assets": 0, "generationSets": 0, "timelines": 0}
    with sqlite3.connect(db_path) as connection:
        apply_project_migrations(connection)
        connection.execute("delete from assets")
        connection.execute("delete from generation_sets")
        connection.execute("delete from timelines")

        for sidecar_path in _asset_sidecars(project_path):
            try:
                asset = read_json(sidecar_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not asset.get("id") or not asset.get("file", {}).get("path"):
                continue
            sidecar_rel = str(sidecar_path.relative_to(project_path)).replace("\\", "/")
            _index_asset_on_connection(connection, asset, sidecar_rel)
            counts["assets"] += 1

        for genset_path in (project_path / "generation-sets").glob("*.json"):
            try:
                generation_set = read_json(genset_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not generation_set.get("id"):
                continue
            connection.execute(
                """
                insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_set["id"],
                    generation_set.get("mode", "unknown"),
                    generation_set.get("model", "unknown"),
                    generation_set.get("prompt", ""),
                    generation_set.get("createdAt", ""),
                    generation_set.get("jobId"),
                ),
            )
            counts["generationSets"] += 1

        for timeline_path in (project_path / "timelines").glob("*.sceneworks.timeline.json"):
            try:
                timeline = read_json(timeline_path)
            except (OSError, json.JSONDecodeError):
                continue
            if not timeline.get("id"):
                continue
            rel_path = str(timeline_path.relative_to(project_path)).replace("\\", "/")
            connection.execute(
                """
                insert or replace into timelines (
                  id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timeline["id"],
                    timeline.get("name", "Timeline"),
                    rel_path,
                    timeline.get("aspectRatio", "16:9"),
                    int(timeline.get("width") or 1280),
                    int(timeline.get("height") or 720),
                    int(timeline.get("fps") or 30),
                    float(timeline.get("duration") or 0),
                    timeline.get("createdAt") or "",
                    timeline.get("updatedAt") or timeline.get("createdAt") or "",
                ),
            )
            counts["timelines"] += 1
    return counts


def _asset_sidecars(project_path: Path) -> list[Path]:
    sidecars: list[Path] = []
    for folder in ASSET_FOLDERS:
        sidecars.extend((project_path / folder).rglob(ASSET_SIDECAR_PATTERN))
    timeline_dir = project_path / "timelines"
    return [path for path in sidecars if timeline_dir not in path.parents]
