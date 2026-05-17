from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

import httpx
import pytest

from scene_worker.runtime import ApiClient, heartbeat, job_cancel_requested, register_worker, update_job


ROOT = Path(__file__).resolve().parents[1]
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe"
    b"\x02\xfeA\xe2&\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"Rust API exited early with code {process.returncode}")
        try:
            response = httpx.get(f"{base_url}/api/v1/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(f"Rust API did not become healthy: {last_error}")


def wait_for_job_status(base_url: str, job_id: str, status: str, process: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 30
    last_job: dict | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"Rust worker exited early with code {process.returncode}: {stderr}")
        response = httpx.get(f"{base_url}/api/v1/jobs/{job_id}", timeout=5)
        response.raise_for_status()
        last_job = response.json()
        if last_job["status"] == status:
            return last_job
        if last_job["status"] in {"failed", "canceled", "interrupted"}:
            raise AssertionError(f"Job reached terminal status {last_job['status']}: {last_job}")
        time.sleep(0.25)
    raise AssertionError(f"Job did not reach {status}: {last_job}")


@pytest.fixture()
def rust_api(tmp_path):
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for the Rust API smoke test")

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_HOST": "127.0.0.1",
            "SCENEWORKS_API_PORT": str(port),
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_CONFIG_DIR": str(tmp_path / "config"),
            "SCENEWORKS_JOBS_DB_PATH": str(tmp_path / "cache" / "jobs.db"),
            "SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE": "1",
        }
    )
    process = subprocess.Popen(
        ["cargo", "run", "-q", "-p", "sceneworks-rust-api"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_health(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_python_worker_protocol_round_trips_against_rust_api_binary(rust_api):
    settings = SimpleNamespace(
        api_url=rust_api,
        access_token="",
        worker_id="live-test-worker",
    )
    api = ApiClient(settings)

    register_worker(
        api,
        settings,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]},
        loaded_models=[],
    )
    created = httpx.post(
        f"{rust_api}/api/v1/image/jobs",
        json={
            "projectId": "project-1",
            "prompt": "mist over hills",
            "model": "z_image_turbo",
            "requestedGpu": "gpu-0",
        },
        timeout=5,
    )
    created.raise_for_status()
    job = created.json()

    claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})["job"]
    assert claimed["id"] == job["id"]
    assert claimed["workerId"] == settings.worker_id
    assert claimed["assignedGpu"] == "gpu-0"

    heartbeat(api, settings, "busy", claimed["id"], loaded_models=["Tongyi-MAI/Z-Image-Turbo"])
    workers = httpx.get(f"{rust_api}/api/v1/workers", timeout=5).json()
    worker = next(worker for worker in workers if worker["id"] == settings.worker_id)
    assert worker["loadedModels"] == ["Tongyi-MAI/Z-Image-Turbo"]

    canceled = httpx.post(f"{rust_api}/api/v1/jobs/{claimed['id']}/cancel", timeout=5)
    canceled.raise_for_status()
    assert job_cancel_requested(api, claimed["id"]) is True

    completed = update_job(
        api,
        claimed["id"],
        {
            "status": "canceled",
            "stage": "canceled",
            "progress": 1,
            "message": "Worker canceled the job before completion.",
        },
    )
    assert completed["status"] == "canceled"
    assert completed["cancelRequested"] is True


def test_rust_worker_claims_and_completes_lora_import_against_rust_api_binary(rust_api, tmp_path):
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for the Rust worker smoke test")

    source = tmp_path / "tiny.safetensors"
    source.write_bytes(b"lora")
    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_URL": rust_api,
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_WORKER_ID": "rust-worker-smoke",
            "SCENEWORKS_GPU_ID": "cpu",
            "SCENEWORKS_POLL_SECONDS": "1",
            "SCENEWORKS_HEARTBEAT_SECONDS": "5",
        }
    )
    worker = subprocess.Popen(
        ["cargo", "run", "-q", "-p", "sceneworks-rust-worker"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        created = httpx.post(
            f"{rust_api}/api/v1/loras/import",
            json={"sourcePath": str(source), "name": "Smoke LoRA"},
            timeout=5,
        )
        created.raise_for_status()
        job = created.json()

        completed = wait_for_job_status(rust_api, job["id"], "completed", worker)

        assert completed["workerId"] == "rust-worker-smoke"
        assert completed["result"]["repo"] is None
        assert completed["result"]["path"].endswith("Smoke__LoRA")
        assert (tmp_path / "data" / "loras" / "Smoke__LoRA" / "tiny.safetensors").read_bytes() == b"lora"
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)


def test_rust_worker_completes_ffmpeg_frame_and_timeline_jobs_against_rust_api_binary(rust_api, tmp_path):
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for the Rust worker smoke test")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the FFmpeg worker smoke test")

    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_URL": rust_api,
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_WORKER_ID": "rust-ffmpeg-smoke",
            "SCENEWORKS_GPU_ID": "cpu",
            "SCENEWORKS_POLL_SECONDS": "1",
            "SCENEWORKS_HEARTBEAT_SECONDS": "5",
        }
    )
    worker = subprocess.Popen(
        ["cargo", "run", "-q", "-p", "sceneworks-rust-worker"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        created_project = httpx.post(f"{rust_api}/api/v1/projects", json={"name": "FFmpeg Smoke"}, timeout=5)
        created_project.raise_for_status()
        project_id = created_project.json()["id"]

        uploaded = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/assets",
            files={"file": ("source.png", PNG_1X1, "image/png")},
            timeout=5,
        )
        uploaded.raise_for_status()
        asset = uploaded.json()
        asset_id = asset["id"]

        created_timeline = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/timelines",
            json={"name": "Main timeline", "aspectRatio": "16:9", "fps": 24},
            timeout=5,
        )
        created_timeline.raise_for_status()
        timeline = created_timeline.json()
        timeline_id = timeline["id"]
        timeline["tracks"][0]["items"] = [
            {
                "id": "item-1",
                "trackId": "track_main",
                "assetId": asset_id,
                "type": "image",
                "displayName": "Still",
                "sourceIn": 0,
                "sourceOut": 1,
                "timelineStart": 0,
                "timelineEnd": 1,
                "speed": 1,
                "fit": "fit",
                "volume": 1,
            }
        ]
        saved_timeline = httpx.put(
            f"{rust_api}/api/v1/projects/{project_id}/timelines/{timeline_id}",
            json={"timeline": timeline},
            timeout=5,
        )
        saved_timeline.raise_for_status()

        frame_job = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/timelines/{timeline_id}/items/item-1/frames",
            json={"playheadSeconds": 0.5, "intendedUse": "reuse"},
            timeout=5,
        )
        frame_job.raise_for_status()
        export_job = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/timelines/{timeline_id}/exports",
            json={"resolution": 240, "fps": 24, "requestedGpu": "auto"},
            timeout=5,
        )
        export_job.raise_for_status()

        frame_completed = wait_for_job_status(rust_api, frame_job.json()["id"], "completed", worker)
        export_completed = wait_for_job_status(rust_api, export_job.json()["id"], "completed", worker)

        assert frame_completed["workerId"] == "rust-ffmpeg-smoke"
        assert frame_completed["result"]["assets"][0]["type"] == "frame"
        assert frame_completed["result"]["assets"][0]["recipe"]["mode"] == "frame_extract"
        assert export_completed["workerId"] == "rust-ffmpeg-smoke"
        assert export_completed["result"]["assets"][0]["type"] == "render"
        assert export_completed["result"]["assets"][0]["file"]["mimeType"] == "video/mp4"
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
