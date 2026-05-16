from datetime import UTC, datetime
import json
import time

from .settings import WorkerSettings


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def readiness_payload(settings: WorkerSettings) -> dict:
    return {
        "event": "ready",
        "workerId": settings.worker_id,
        "gpuId": settings.gpu_id,
        "apiUrl": settings.api_url,
        "capabilities": ["placeholder"],
        "directories": {
            "data": str(settings.data_dir),
            "config": str(settings.config_dir),
        },
        "reportedAt": now(),
    }


def heartbeat_payload(settings: WorkerSettings) -> dict:
    return {
        "event": "heartbeat",
        "workerId": settings.worker_id,
        "gpuId": settings.gpu_id,
        "status": "idle",
        "currentJobId": None,
        "loadedModels": [],
        "reportedAt": now(),
    }


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    settings = WorkerSettings()
    emit(readiness_payload(settings))

    while True:
        emit(heartbeat_payload(settings))
        time.sleep(settings.heartbeat_seconds)
