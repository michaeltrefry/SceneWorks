from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from .jobs import queue_summary


router = APIRouter(prefix="/video", tags=["video"])

VideoMode = Literal["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"]
VideoQuality = Literal["fast", "balanced", "best"]
ReplacementMode = Literal["face_only", "full_person_keep_outfit", "full_person_replace_outfit"]


class VideoJobRequest(BaseModel):
    projectId: str = Field(min_length=1)
    projectName: str | None = None
    mode: VideoMode = "image_to_video"
    prompt: str = Field(min_length=1, max_length=4000)
    negativePrompt: str = ""
    model: str = "ltx_2_3"
    duration: float = Field(default=6, ge=1, le=30)
    fps: int = Field(default=25, ge=1, le=60)
    width: int = Field(default=768, ge=256, le=1920)
    height: int = Field(default=512, ge=256, le=1920)
    quality: VideoQuality = "balanced"
    seed: int | None = None
    loras: list[dict[str, Any]] = Field(default_factory=list)
    characterId: str | None = None
    characterLookId: str | None = None
    personTrackId: str | None = None
    replacementMode: ReplacementMode = "face_only"
    sourceAssetId: str | None = None
    lastFrameAssetId: str | None = None
    sourceClipAssetId: str | None = None
    bridgeRightClipAssetId: str | None = None
    requestedGpu: str = "auto"
    advanced: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> "VideoJobRequest":
        if self.mode == "image_to_video" and not self.sourceAssetId:
            raise ValueError("Image to Video requires a source image.")
        if self.mode == "first_last_frame" and (not self.sourceAssetId or not self.lastFrameAssetId):
            raise ValueError("First/Last Frame requires first and last image assets.")
        if self.mode == "extend_clip" and not self.sourceClipAssetId:
            raise ValueError("Extend Clip requires a source clip.")
        if self.mode == "video_bridge" and (not self.sourceClipAssetId or not self.bridgeRightClipAssetId):
            raise ValueError("Bridge generation requires left and right source clips.")
        if self.mode == "replace_person":
            if not self.sourceClipAssetId:
                raise ValueError("Replace Person requires a source clip.")
            if not self.personTrackId:
                raise ValueError("Replace Person requires a selected person track.")
            if not self.characterId:
                raise ValueError("Replace Person requires a Character.")
        return self


@router.post("/jobs", status_code=201)
def create_video_job(payload: VideoJobRequest, request: Request) -> dict:
    job_type = "video_generate"
    if payload.mode == "extend_clip":
        job_type = "video_extend"
    elif payload.mode == "video_bridge":
        job_type = "video_bridge"
    elif payload.mode == "replace_person":
        # User-facing modes stay verb-first; backend job types group person workflows together.
        job_type = "person_replace"
    job = request.app.state.jobs_store.create_job(
        job_type=job_type,
        project_id=payload.projectId,
        project_name=payload.projectName,
        payload=payload.model_dump(exclude={"requestedGpu"}),
        requested_gpu=payload.requestedGpu,
    )
    request.app.state.event_hub.publish("job.updated", job)
    request.app.state.event_hub.publish("queue.updated", queue_summary(request))
    return job
