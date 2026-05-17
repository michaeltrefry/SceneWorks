from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .events import encode_sse
from .jobs_store import JOB_STATUSES, MAX_JOB_ATTEMPTS, JobsStore


router = APIRouter(tags=["jobs"])

JobType = Literal[
    "placeholder",
    "image_generate",
    "image_edit",
    "video_generate",
    "video_extend",
    "video_bridge",
    "person_detect",
    "person_track",
    "person_replace",
    "frame_extract",
    "timeline_export",
    "model_download",
    "lora_import",
]


class JobCreateRequest(BaseModel):
    type: JobType = "placeholder"
    projectId: str | None = None
    projectName: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    requestedGpu: str = "auto"


class DuplicateJobRequest(BaseModel):
    payloadChanges: dict[str, Any] = Field(default_factory=dict)
    requestedGpu: str | None = None


class WorkerRegisterRequest(BaseModel):
    workerId: str = Field(min_length=1, max_length=120)
    gpuId: str = Field(default="auto", min_length=1, max_length=120)
    gpuName: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    loadedModels: list[str] = Field(default_factory=list)


class WorkerHeartbeatRequest(BaseModel):
    status: str = "idle"
    currentJobId: str | None = None
    loadedModels: list[str] = Field(default_factory=list)


class ClaimRequest(BaseModel):
    workerId: str = Field(min_length=1, max_length=120)


class ProgressRequest(BaseModel):
    status: str
    stage: str
    progress: float = Field(ge=0, le=1)
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    etaSeconds: float | None = None


def get_store(request: Request) -> JobsStore:
    return request.app.state.jobs_store


def publish(request: Request, event: str, data: dict) -> None:
    request.app.state.event_hub.publish(event, data)


def sweep_stale_workers(request: Request) -> None:
    swept = get_store(request).mark_stale_workers_interrupted(request.app.state.settings.worker_timeout_seconds)
    for job in swept["jobs"]:
        publish(request, "job.updated", job)
    for worker in swept["workers"]:
        publish(request, "worker.updated", worker)
    if swept["jobs"] or swept["workers"]:
        publish(request, "queue.updated", queue_summary(request))


@router.get("/jobs")
def list_jobs(
    request: Request,
    projectId: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    sweep_stale_workers(request)
    if status and status not in JOB_STATUSES:
        raise HTTPException(status_code=400, detail="Unsupported job status")
    return get_store(request).list_jobs(project_id=projectId, status=status, limit=limit)


@router.post("/jobs", status_code=201)
def create_job(payload: JobCreateRequest, request: Request) -> dict:
    job = get_store(request).create_job(
        job_type=payload.type,
        project_id=payload.projectId,
        project_name=payload.projectName,
        payload=payload.payload,
        requested_gpu=payload.requestedGpu,
    )
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return job


@router.post("/jobs/events/ticket")
def create_event_ticket(request: Request) -> dict[str, Any]:
    return request.app.state.event_ticket_store.issue()


@router.get("/jobs/events")
async def job_events(request: Request, ticket: str = Query(default="")) -> StreamingResponse:
    if request.app.state.settings.access_token:
        request.app.state.event_ticket_store.consume(ticket)
    queue = await request.app.state.event_hub.subscribe()

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield encode_sse(message)
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            request.app.state.event_hub.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/jobs/claim")
def claim_job(payload: ClaimRequest, request: Request) -> dict:
    sweep_stale_workers(request)
    try:
        job = get_store(request).claim_next_job(payload.workerId)
    except KeyError:
        raise HTTPException(status_code=404, detail="Worker not found") from None
    if job is None:
        return {"job": None}
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return {"job": job}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    try:
        return get_store(request).get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    try:
        job = get_store(request).cancel_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return job


@router.post("/jobs/{job_id}/retry", status_code=201)
def retry_job(job_id: str, request: Request) -> dict:
    try:
        job = get_store(request).retry_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return job


@router.post("/jobs/{job_id}/duplicate", status_code=201)
def duplicate_job(job_id: str, payload: DuplicateJobRequest, request: Request) -> dict:
    try:
        job = get_store(request).duplicate_job(
            job_id,
            payload_changes=payload.payloadChanges,
            requested_gpu=payload.requestedGpu,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return job


@router.get("/queue")
def queue_summary(request: Request) -> dict:
    sweep_stale_workers(request)
    jobs = get_store(request).list_jobs(limit=500)
    workers = get_store(request).list_workers()
    counts = {status: 0 for status in JOB_STATUSES}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    return {
        "counts": counts,
        "activeJobs": [job for job in jobs if job["status"] not in ("completed", "failed", "canceled", "interrupted")],
        "workers": workers,
        "maxJobAttempts": MAX_JOB_ATTEMPTS,
    }


@router.get("/workers")
def list_workers(request: Request) -> list[dict]:
    sweep_stale_workers(request)
    return get_store(request).list_workers()


@router.post("/workers/register")
def register_worker(payload: WorkerRegisterRequest, request: Request) -> dict:
    worker = get_store(request).register_worker(
        worker_id=payload.workerId,
        gpu_id=payload.gpuId,
        gpu_name=payload.gpuName,
        capabilities=payload.capabilities,
        loaded_models=payload.loadedModels,
    )
    publish(request, "worker.updated", worker)
    publish(request, "queue.updated", queue_summary(request))
    return worker


@router.post("/workers/{worker_id}/heartbeat")
def heartbeat_worker(worker_id: str, payload: WorkerHeartbeatRequest, request: Request) -> dict:
    try:
        worker = get_store(request).heartbeat_worker(
            worker_id=worker_id,
            status=payload.status,
            current_job_id=payload.currentJobId,
            loaded_models=payload.loadedModels,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Worker not found") from None
    publish(request, "worker.updated", worker)
    return worker


@router.post("/jobs/{job_id}/progress")
def update_job_progress(job_id: str, payload: ProgressRequest, request: Request) -> dict:
    try:
        job = get_store(request).update_job_progress(
            job_id,
            status=payload.status,
            stage=payload.stage,
            progress=payload.progress,
            message=payload.message,
            error=payload.error,
            result=payload.result,
            eta_seconds=payload.etaSeconds,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    publish(request, "job.updated", job)
    publish(request, "queue.updated", queue_summary(request))
    return job
