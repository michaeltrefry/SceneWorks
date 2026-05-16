from __future__ import annotations

from datetime import UTC, datetime
import json
import time
from typing import Any

import httpx

from .gpu import discover_gpu
from .settings import WorkerSettings


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


class ApiClient:
    def __init__(self, settings: WorkerSettings) -> None:
        headers = {}
        if settings.access_token:
            headers["X-SceneWorks-Token"] = settings.access_token
        self.client = httpx.Client(base_url=settings.api_url, headers=headers, timeout=20)

    def post(self, path: str, payload: dict) -> dict:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def get(self, path: str) -> dict:
        response = self.client.get(path)
        response.raise_for_status()
        return response.json()


def register_worker(api: ApiClient, settings: WorkerSettings, gpu: dict) -> None:
    payload = {
        "workerId": settings.worker_id,
        "gpuId": gpu["id"],
        "gpuName": gpu["name"],
        "capabilities": gpu["capabilities"],
        "loadedModels": [],
    }
    worker = api.post("/api/v1/workers/register", payload)
    emit({"event": "registered", "worker": worker, "reportedAt": now()})


def heartbeat(
    api: ApiClient,
    settings: WorkerSettings,
    status: str,
    current_job_id: str | None = None,
) -> None:
    api.post(
        f"/api/v1/workers/{settings.worker_id}/heartbeat",
        {"status": status, "currentJobId": current_job_id, "loadedModels": []},
    )


def update_job(api: ApiClient, job_id: str, payload: dict[str, Any]) -> dict:
    job = api.post(f"/api/v1/jobs/{job_id}/progress", payload)
    emit({"event": "job_progress", "jobId": job_id, "status": job["status"], "stage": job["stage"]})
    return job


def job_cancel_requested(api: ApiClient, job_id: str) -> bool:
    return bool(api.get(f"/api/v1/jobs/{job_id}")["cancelRequested"])


def run_placeholder_job(api: ApiClient, settings: WorkerSettings, job: dict) -> None:
    job_id = job["id"]
    stages = [
        ("preparing", "preparing", 0.1, "Preparing placeholder job."),
        ("running", "running", 0.35, "Running placeholder step 1."),
        ("running", "running", 0.65, "Running placeholder step 2."),
        ("saving", "saving", 0.9, "Saving placeholder result."),
    ]

    for status, stage, progress, message in stages:
        if job_cancel_requested(api, job_id):
            update_job(
                api,
                job_id,
                {
                    "status": "canceled",
                    "stage": "canceled",
                    "progress": progress,
                    "message": "Worker canceled the job before completion.",
                },
            )
            heartbeat(api, settings, "idle")
            return

        heartbeat(api, settings, "busy", job_id)
        update_job(
            api,
            job_id,
            {
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
            },
        )
        time.sleep(1.5)

    update_job(
        api,
        job_id,
        {
            "status": "completed",
            "stage": "completed",
            "progress": 1,
            "message": "Placeholder job completed.",
            "result": {"completedAt": now(), "output": "placeholder"},
        },
    )
    heartbeat(api, settings, "idle")


def main() -> None:
    settings = WorkerSettings()
    gpu = discover_gpu(settings.gpu_id)
    api = ApiClient(settings)

    while True:
        try:
            register_worker(api, settings, gpu)
            break
        except httpx.HTTPError as exc:
            emit({"event": "register_failed", "error": str(exc), "reportedAt": now()})
            time.sleep(settings.poll_seconds)

    while True:
        try:
            heartbeat(api, settings, "idle")
            claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})
            job = claimed.get("job")
            if job is None:
                time.sleep(settings.poll_seconds)
                continue

            emit({"event": "claimed", "jobId": job["id"], "gpuId": job["assignedGpu"], "reportedAt": now()})
            if job["type"] == "placeholder":
                run_placeholder_job(api, settings, job)
            else:
                update_job(
                    api,
                    job["id"],
                    {
                        "status": "failed",
                        "stage": "failed",
                        "progress": 1,
                        "message": "No adapter exists for this job type yet.",
                        "error": f"Unsupported job type: {job['type']}",
                    },
                )
        except httpx.HTTPError as exc:
            emit({"event": "api_error", "error": str(exc), "reportedAt": now()})
            time.sleep(settings.poll_seconds)
