from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from sceneworks_shared import read_json

from .jobs import queue_summary
from .projects import find_project_path


router = APIRouter(prefix="/projects/{project_id}/person-tracks", tags=["person-tracking"])


class PersonDetectionJobRequest(BaseModel):
    sourceAssetId: str = Field(min_length=1)
    sourceTimestamp: float | None = Field(default=None, ge=0)
    requestedGpu: str = "auto"


class PersonTrackJobRequest(BaseModel):
    sourceAssetId: str = Field(min_length=1)
    representativeFrameAssetId: str = Field(min_length=1)
    detection: dict[str, Any]
    trackName: str = Field(default="Selected person", min_length=1, max_length=120)
    requestedGpu: str = "auto"


def tracks_dir(project_path: Path) -> Path:
    return project_path / "person-tracks"


def normalize_track(project_path: Path, track_path: Path) -> dict[str, Any]:
    track = read_json(track_path)
    track["path"] = str(track_path.relative_to(project_path)).replace("\\", "/")
    return track


@router.get("")
def list_person_tracks(project_id: str, request: Request) -> list[dict[str, Any]]:
    project_path = find_project_path(request.app.state.settings, project_id)
    folder = tracks_dir(project_path)
    if not folder.exists():
        return []

    tracks = []
    for track_path in folder.glob("*.sceneworks.person-track.json"):
        try:
            tracks.append(normalize_track(project_path, track_path))
        except OSError:
            continue
    return sorted(tracks, key=lambda track: track.get("createdAt", ""), reverse=True)


@router.get("/{track_id}")
def get_person_track(project_id: str, track_id: str, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    track_path = tracks_dir(project_path) / f"{track_id}.sceneworks.person-track.json"
    try:
        resolved_path = track_path.resolve()
        resolved_path.relative_to(tracks_dir(project_path).resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person track ID") from None
    if not track_path.exists():
        raise HTTPException(status_code=404, detail="Person track not found")
    return normalize_track(project_path, track_path)


@router.post("/detections", status_code=201)
def create_person_detection_job(
    project_id: str,
    payload: PersonDetectionJobRequest,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    if not payload.sourceAssetId:
        raise HTTPException(status_code=400, detail="Source clip is required")

    job = request.app.state.jobs_store.create_job(
        job_type="person_detect",
        project_id=project_id,
        project_name=project_path.stem,
        payload={
            "projectId": project_id,
            "sourceAssetId": payload.sourceAssetId,
            "sourceTimestamp": payload.sourceTimestamp,
        },
        requested_gpu=payload.requestedGpu or "auto",
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job


@router.post("/jobs", status_code=201)
def create_person_track_job(
    project_id: str,
    payload: PersonTrackJobRequest,
    request: Request,
) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    if not payload.detection.get("id"):
        raise HTTPException(status_code=400, detail="Selected detection metadata is required")

    job = request.app.state.jobs_store.create_job(
        job_type="person_track",
        project_id=project_id,
        project_name=project_path.stem,
        payload={
            "projectId": project_id,
            "sourceAssetId": payload.sourceAssetId,
            "representativeFrameAssetId": payload.representativeFrameAssetId,
            "detection": payload.detection,
            "trackName": payload.trackName,
        },
        requested_gpu=payload.requestedGpu or "auto",
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job
