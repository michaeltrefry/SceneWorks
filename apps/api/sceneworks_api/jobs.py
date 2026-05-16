from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .events import encode_sse
from .jobs_store import JOB_STATUSES, JobsStore


router = APIRouter(tags=["jobs"])


class JobCreateRequest(BaseModel):
    type: str = Field(default="placeholder", min_length=1, max_length=80)
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


@router.get("/jobs")
def list_jobs(
    request: Request,
    projectId: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
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


@router.get("/jobs/events")
async def job_events(request: Request) -> StreamingResponse:
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
    jobs = get_store(request).list_jobs(limit=500)
    workers = get_store(request).list_workers()
    counts = {status: 0 for status in JOB_STATUSES}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    return {
        "counts": counts,
        "activeJobs": [job for job in jobs if job["status"] not in ("completed", "failed", "canceled", "interrupted")],
        "workers": workers,
    }


@router.get("/workers")
def list_workers(request: Request) -> list[dict]:
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
