from datetime import UTC, datetime
import json
import re
import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .settings import Settings


router = APIRouter(prefix="/projects", tags=["projects"])

PROJECT_FOLDERS = [
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "characters",
    "generation-sets",
    "loras",
    "recipes",
    "timelines",
    "trash",
    "cache",
]


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectSummary(BaseModel):
    id: str
    name: str
    path: str
    createdAt: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "project"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


def ensure_data_dirs(settings: Settings) -> None:
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "loras", "cache"):
        (settings.data_dir / folder).mkdir(parents=True, exist_ok=True)


def load_registry(settings: Settings) -> list[dict]:
    if not settings.registry_path.exists():
        return []

    with settings.registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_registry(settings: Settings, projects: list[dict]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with settings.registry_path.open("w", encoding="utf-8") as handle:
        json.dump(projects, handle, indent=2)
        handle.write("\n")


def create_project_db(project_path: Path) -> None:
    db_path = project_path / "project.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            create table if not exists project_metadata (
              key text primary key,
              value text not null
            )
            """
        )
        connection.execute(
            "insert or replace into project_metadata (key, value) values (?, ?)",
            ("schemaVersion", "1"),
        )
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


def write_project_file(settings: Settings, project_path: Path, project_id: str, name: str) -> dict:
    created_at = utc_now()
    project_file = {
        "schemaVersion": 1,
        "appVersion": settings.app_version,
        "id": project_id,
        "name": name,
        "createdAt": created_at,
        "folders": {folder.replace("/", "_"): folder for folder in PROJECT_FOLDERS},
    }
    with (project_path / "project.json").open("w", encoding="utf-8") as handle:
        json.dump(project_file, handle, indent=2)
        handle.write("\n")
    return project_file


def read_project_summary(project_path: Path) -> ProjectSummary:
    project_file = project_path / "project.json"
    if not project_file.exists():
        raise HTTPException(status_code=404, detail="Project file not found")

    with project_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return ProjectSummary(
        id=payload["id"],
        name=payload["name"],
        path=str(project_path),
        createdAt=payload["createdAt"],
    )


def find_project_path(settings: Settings, project_id: str) -> Path:
    for item in load_registry(settings):
        if item.get("id") == project_id:
            project_path = Path(item["path"])
            if project_path.exists():
                return project_path
            break

    raise HTTPException(status_code=404, detail="Project not found")


@router.get("", response_model=list[ProjectSummary])
def list_projects(request: Request) -> list[ProjectSummary]:
    settings = get_settings_from_request(request)
    ensure_data_dirs(settings)
    projects = []
    for item in load_registry(settings):
        path = Path(item["path"])
        if path.exists():
            projects.append(read_project_summary(path))
    return projects


@router.post("", response_model=ProjectSummary, status_code=201)
def create_project(payload: ProjectCreateRequest, request: Request) -> ProjectSummary:
    settings = get_settings_from_request(request)
    ensure_data_dirs(settings)

    project_id = f"project_{uuid4().hex}"
    folder_name = f"{slugify(payload.name)}.sceneworks"
    project_path = settings.projects_dir / folder_name
    if project_path.exists():
        project_path = settings.projects_dir / f"{slugify(payload.name)}-{project_id[-8:]}.sceneworks"

    project_path.mkdir(parents=True)
    for folder in PROJECT_FOLDERS:
        (project_path / folder).mkdir(parents=True, exist_ok=True)

    write_project_file(settings, project_path, project_id, payload.name)
    create_project_db(project_path)

    registry = [item for item in load_registry(settings) if item.get("id") != project_id]
    registry.insert(0, {"id": project_id, "name": payload.name, "path": str(project_path)})
    save_registry(settings, registry)

    return read_project_summary(project_path)


@router.get("/{project_id}", response_model=ProjectSummary)
def get_project(project_id: str, request: Request) -> ProjectSummary:
    settings = get_settings_from_request(request)
    return read_project_summary(find_project_path(settings, project_id))
