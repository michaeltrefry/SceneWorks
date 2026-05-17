import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sceneworks_shared import (
    ProjectNotFound,
    apply_project_migrations,
    find_project_path as shared_find_project_path,
    load_registry as shared_load_registry,
    reindex_project,
    slugify,
    utc_now,
)

from .settings import Settings


router = APIRouter(prefix="/projects", tags=["projects"])
REGISTRY_LOCK = threading.Lock()

PROJECT_FOLDERS = [
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "characters",
    "generation-sets",
    "loras",
    "person-tracks",
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


class ReindexResult(BaseModel):
    projectId: str
    assets: int
    generationSets: int
    timelines: int


def get_settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


def ensure_data_dirs(settings: Settings) -> None:
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "loras", "cache"):
        (settings.data_dir / folder).mkdir(parents=True, exist_ok=True)


def load_registry(settings: Settings) -> list[dict]:
    return shared_load_registry(settings.registry_path)


def save_registry(settings: Settings, projects: list[dict]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=settings.data_dir, encoding="utf-8") as handle:
            tmp_path = Path(handle.name)
            json.dump(projects, handle, indent=2)
            handle.write("\n")
        tmp_path.replace(settings.registry_path)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def create_project_db(project_path: Path) -> None:
    db_path = project_path / "project.db"
    with sqlite3.connect(db_path) as connection:
        apply_project_migrations(connection)


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
    try:
        return shared_find_project_path(settings.registry_path, project_id)
    except ProjectNotFound:
        raise HTTPException(status_code=404, detail="Project not found") from None


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

    with REGISTRY_LOCK:
        project_id = f"project_{uuid4().hex}"
        folder_name = f"{slugify(payload.name, fallback='project')}.sceneworks"
        project_path = settings.projects_dir / folder_name
        if project_path.exists():
            project_path = settings.projects_dir / f"{slugify(payload.name, fallback='project')}-{project_id[-8:]}.sceneworks"

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


@router.post("/{project_id}/reindex", response_model=ReindexResult)
def reindex_project_endpoint(project_id: str, request: Request) -> ReindexResult:
    settings = get_settings_from_request(request)
    counts = reindex_project(find_project_path(settings, project_id))
    return ReindexResult(projectId=project_id, **counts)
