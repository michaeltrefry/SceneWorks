from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from sceneworks_shared import ensure_project_db_ready, read_json, slugify, utc_now, write_json

from .jobs import queue_summary
from .projects import find_project_path


router = APIRouter(prefix="/projects/{project_id}/timelines", tags=["timelines"])

AspectRatio = Literal["16:9", "9:16", "1:1"]
TrackKind = Literal["video", "overlay", "audio"]
TimelineItemKind = Literal["video", "image", "audio"]
TransitionKind = Literal["cut", "crossfade", "fade_from_black", "fade_to_black"]

ASPECT_DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1024, 1024),
}


class Transition(BaseModel):
    id: str = Field(default_factory=lambda: f"transition_{uuid4().hex}")
    type: TransitionKind = "cut"
    fromItemId: str | None = None
    toItemId: str | None = None
    duration: float = Field(default=0, ge=0, le=10)


class TimelineItemVersion(BaseModel):
    assetId: str = Field(min_length=1)
    createdAt: str | None = None
    source: Literal["original", "replacement", "extension", "bridge", "restore", "manual"] = "manual"
    jobId: str | None = None
    note: str | None = None


class TimelineItem(BaseModel):
    id: str = Field(default_factory=lambda: f"item_{uuid4().hex}")
    trackId: str
    assetId: str = Field(min_length=1)
    type: TimelineItemKind = "video"
    displayName: str = Field(min_length=1, max_length=160)
    sourceIn: float = Field(default=0, ge=0)
    sourceOut: float = Field(default=4, gt=0)
    timelineStart: float = Field(default=0, ge=0)
    timelineEnd: float = Field(default=4, gt=0)
    speed: float = Field(default=1, ge=0.1, le=8)
    fit: Literal["fit", "fill", "stretch"] = "fit"
    volume: float = Field(default=1, ge=0, le=2)
    versionAssetIds: list[str] = Field(default_factory=list)
    currentVersionAssetId: str | None = None
    versionHistory: list[TimelineItemVersion] = Field(default_factory=list)
    transitionIn: Transition | None = None
    transitionOut: Transition | None = None

    @model_validator(mode="after")
    def validate_ranges_and_populate_version_metadata(self) -> "TimelineItem":
        if self.sourceOut <= self.sourceIn:
            raise ValueError("sourceOut must be greater than sourceIn.")
        if self.timelineEnd <= self.timelineStart:
            raise ValueError("timelineEnd must be greater than timelineStart.")
        if not self.currentVersionAssetId:
            self.currentVersionAssetId = self.assetId
        if self.assetId not in self.versionAssetIds:
            self.versionAssetIds.append(self.assetId)
        if not self.versionHistory:
            self.versionHistory.append(TimelineItemVersion(assetId=self.assetId, source="original"))
        return self


class TimelineTrack(BaseModel):
    id: str
    name: str
    kind: TrackKind
    locked: bool = False
    muted: bool = False
    items: list[TimelineItem] = Field(default_factory=list)


class TimelineDocument(BaseModel):
    schemaVersion: int = 1
    id: str = Field(default_factory=lambda: f"timeline_{uuid4().hex}")
    projectId: str
    name: str = Field(min_length=1, max_length=120)
    aspectRatio: AspectRatio = "16:9"
    width: int = Field(default=1280, ge=256, le=3840)
    height: int = Field(default=720, ge=256, le=3840)
    fps: int = Field(default=30, ge=1, le=60)
    duration: float = Field(default=0, ge=0)
    tracks: list[TimelineTrack] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    createdAt: str | None = None
    updatedAt: str | None = None


class TimelineCreateRequest(BaseModel):
    name: str = Field(default="Main timeline", min_length=1, max_length=120)
    aspectRatio: AspectRatio = "16:9"
    fps: int = Field(default=30, ge=1, le=60)


class TimelineSaveRequest(BaseModel):
    timeline: TimelineDocument


class TimelineExportRequest(BaseModel):
    resolution: int = Field(default=720)
    fps: int = Field(default=30, ge=1, le=60)
    requestedGpu: str = "auto"

    @model_validator(mode="after")
    def validate_resolution(self) -> "TimelineExportRequest":
        if self.resolution not in {640, 720, 1024, 1280}:
            raise ValueError("Resolution must be one of 640, 720, 1024, or 1280.")
        return self


class FrameExtractRequest(BaseModel):
    playheadSeconds: float = Field(ge=0)
    intendedUse: Literal["reuse", "first_frame", "last_frame", "video_studio", "image_studio", "bridge", "extension"] = "reuse"
    requestedGpu: str = "auto"


def timeline_file_path(project_path: Path, timeline_id: str, name: str) -> Path:
    return project_path / "timelines" / f"{slugify(name, fallback='timeline', max_length=48)}-{timeline_id[-8:]}.sceneworks.timeline.json"


def ensure_timeline_db(project_path: Path) -> None:
    ensure_project_db_ready(project_path)


def default_tracks() -> list[TimelineTrack]:
    return [
        TimelineTrack(id="track_main", name="Main", kind="video"),
        TimelineTrack(id="track_overlay", name="Overlay", kind="overlay"),
        TimelineTrack(id="track_audio", name="Audio", kind="audio"),
    ]


def compute_duration(timeline: TimelineDocument) -> float:
    ends = [item.timelineEnd for track in timeline.tracks for item in track.items]
    return max(ends, default=0)


def find_timeline_item(timeline: TimelineDocument, item_id: str) -> TimelineItem:
    for track in timeline.tracks:
        for item in track.items:
            if item.id == item_id:
                return item
    raise HTTPException(status_code=404, detail="Timeline item not found")


def source_timestamp_for_item(item: TimelineItem, playhead_seconds: float) -> float:
    clamped = min(max(playhead_seconds, item.timelineStart), item.timelineEnd)
    return item.sourceIn + ((clamped - item.timelineStart) * item.speed)


def index_timeline(project_path: Path, timeline: TimelineDocument, rel_path: str) -> None:
    ensure_timeline_db(project_path)
    with sqlite3.connect(project_path / "project.db") as connection:
        connection.execute(
            """
            insert or replace into timelines (
              id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timeline.id,
                timeline.name,
                rel_path,
                timeline.aspectRatio,
                timeline.width,
                timeline.height,
                timeline.fps,
                timeline.duration,
                timeline.createdAt,
                timeline.updatedAt,
            ),
        )


def find_timeline_file(project_path: Path, timeline_id: str) -> Path:
    ensure_timeline_db(project_path)
    indexed_path = None
    with sqlite3.connect(project_path / "project.db") as connection:
        row = connection.execute("select file_path from timelines where id = ?", (timeline_id,)).fetchone()
    if row is not None:
        indexed_path = row[0]
        path = project_path / indexed_path
        if path.exists():
            return path
    for candidate in (project_path / "timelines").glob("*.sceneworks.timeline.json"):
        try:
            if read_json(candidate).get("id") == timeline_id:
                rel_path = str(candidate.relative_to(project_path)).replace("\\", "/")
                with sqlite3.connect(project_path / "project.db") as connection:
                    connection.execute("update timelines set file_path = ? where id = ?", (rel_path, timeline_id))
                return candidate
        except (OSError, json.JSONDecodeError):
            continue
    if indexed_path:
        raise HTTPException(
            status_code=404,
            detail=f"Timeline file not found at indexed path {indexed_path}; reindex required",
        )
    raise HTTPException(status_code=404, detail="Timeline not found")


def save_timeline(project_path: Path, timeline: TimelineDocument) -> TimelineDocument:
    now = utc_now()
    if timeline.createdAt is None:
        timeline.createdAt = now
    timeline.updatedAt = now
    timeline.duration = compute_duration(timeline)
    if not timeline.tracks:
        timeline.tracks = default_tracks()
    path = timeline_file_path(project_path, timeline.id, timeline.name)
    rel_path = str(path.relative_to(project_path)).replace("\\", "/")
    write_json(path, timeline.model_dump())
    index_timeline(project_path, timeline, rel_path)
    return timeline


@router.get("")
def list_timelines(project_id: str, request: Request) -> list[dict[str, Any]]:
    project_path = find_project_path(request.app.state.settings, project_id)
    ensure_timeline_db(project_path)
    with sqlite3.connect(project_path / "project.db") as connection:
        rows = connection.execute(
            """
            select id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
              from timelines
             order by updated_at desc
            """
        ).fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "filePath": row[2],
            "aspectRatio": row[3],
            "width": row[4],
            "height": row[5],
            "fps": row[6],
            "duration": row[7],
            "createdAt": row[8],
            "updatedAt": row[9],
        }
        for row in rows
    ]


@router.post("", status_code=201)
def create_timeline(project_id: str, payload: TimelineCreateRequest, request: Request) -> TimelineDocument:
    project_path = find_project_path(request.app.state.settings, project_id)
    width, height = ASPECT_DIMENSIONS[payload.aspectRatio]
    timeline = TimelineDocument(
        projectId=project_id,
        name=payload.name,
        aspectRatio=payload.aspectRatio,
        width=width,
        height=height,
        fps=payload.fps,
        tracks=default_tracks(),
    )
    return save_timeline(project_path, timeline)


@router.get("/{timeline_id}")
def get_timeline(project_id: str, timeline_id: str, request: Request) -> dict[str, Any]:
    project_path = find_project_path(request.app.state.settings, project_id)
    return read_json(find_timeline_file(project_path, timeline_id))


@router.put("/{timeline_id}")
def update_timeline(
    project_id: str,
    timeline_id: str,
    payload: TimelineSaveRequest,
    request: Request,
) -> TimelineDocument:
    project_path = find_project_path(request.app.state.settings, project_id)
    timeline = payload.timeline
    if timeline_id != timeline.id:
        raise HTTPException(status_code=400, detail="Timeline ID mismatch")
    if timeline.projectId != project_id:
        raise HTTPException(status_code=400, detail="Project ID mismatch")
    return save_timeline(project_path, timeline)


@router.post("/{timeline_id}/exports", status_code=201)
def create_timeline_export(
    project_id: str,
    timeline_id: str,
    payload: TimelineExportRequest,
    request: Request,
) -> dict:
    project_path = find_project_path(request.app.state.settings, project_id)
    timeline_path = find_timeline_file(project_path, timeline_id)
    timeline = read_json(timeline_path)
    job = request.app.state.jobs_store.create_job(
        job_type="timeline_export",
        project_id=project_id,
        project_name=None,
        payload={
            "projectId": project_id,
            "timelineId": timeline_id,
            "timelineName": timeline.get("name", "Timeline"),
            "timelinePath": str(timeline_path.relative_to(project_path)).replace("\\", "/"),
            "resolution": payload.resolution,
            "fps": payload.fps,
        },
        requested_gpu=payload.requestedGpu,
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job


@router.post("/{timeline_id}/items/{item_id}/frames", status_code=201)
def extract_timeline_frame(
    project_id: str,
    timeline_id: str,
    item_id: str,
    payload: FrameExtractRequest,
    request: Request,
) -> dict:
    project_path = find_project_path(request.app.state.settings, project_id)
    timeline_path = find_timeline_file(project_path, timeline_id)
    timeline = TimelineDocument.model_validate(read_json(timeline_path))
    item = find_timeline_item(timeline, item_id)
    timestamp = source_timestamp_for_item(item, payload.playheadSeconds)
    job = request.app.state.jobs_store.create_job(
        job_type="frame_extract",
        project_id=project_id,
        project_name=None,
        payload={
            "projectId": project_id,
            "timelineId": timeline_id,
            "timelineName": timeline.name,
            "timelinePath": str(timeline_path.relative_to(project_path)).replace("\\", "/"),
            "timelineItemId": item_id,
            "sourceAssetId": item.assetId,
            "sourceTimestamp": timestamp,
            "playheadSeconds": payload.playheadSeconds,
            "intendedUse": payload.intendedUse,
        },
        requested_gpu=payload.requestedGpu,
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job
