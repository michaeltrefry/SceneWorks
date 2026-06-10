from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

import httpx
import pytest

from scene_worker.image_adapters import (
    CHARACTER_ANGLE_SET_ORDER,
    ImageAssetWriter,
    MODEL_TARGETS,
    render_preview_image,
)
from scene_worker.runtime import (
    ApiClient,
    heartbeat,
    job_cancel_requested,
    register_worker,
    run_image_job,
    update_job,
)
from scene_worker.settings import WorkerSettings

# Every test here drives the compiled Rust API binary (the `rust_api` fixture
# spawns `cargo run -p sceneworks-rust-api`), so the whole module is e2e: it
# must run in the CI step that follows the Rust build, not in the lightweight
# worker-suite step (sc-4180).
pytestmark = pytest.mark.e2e


ROOT = Path(__file__).resolve().parents[1]


def _minimal_safetensors() -> bytes:
    # Smallest valid safetensors: 8-byte little-endian header length + JSON header.
    # The import path inspects this header for architecture detection, so a stub
    # like b"lora" is rejected with an invalid-header 400.
    header = b'{"__metadata__":{"format":"pt"}}'
    return len(header).to_bytes(8, "little") + header


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

    # An image_generate job is only offered to workers advertising the
    # image_generate capability, so register with it (mirrors the procedural e2e
    # worker) or the claim below returns no job.
    register_worker(
        api,
        settings,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "image_generate"]},
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


def test_python_worker_completes_procedural_image_job_against_rust_api_binary(
    rust_api, tmp_path, monkeypatch
):
    # The rust_api fixture runs the API with SCENEWORKS_DATA_DIR=tmp_path/"data"; the
    # in-process worker below shares that same data dir. recent-projects.json is owned
    # by the desktop/web app (the Rust API never writes it), so seed it directly the
    # way the worker resolves project paths.
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    project_path = tmp_path / "project"
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )

    monkeypatch.setenv("SCENEWORKS_API_URL", rust_api)
    monkeypatch.setenv("SCENEWORKS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SCENEWORKS_WORKER_ID", "image-e2e-worker")
    monkeypatch.setenv("SCENEWORKS_GPU_ID", "gpu-0")
    # The procedural adapter renders a deterministic preview with no model weights, so
    # the whole image pipeline runs in CI without the multi-GB diffusion checkpoints.
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "procedural")

    settings = WorkerSettings()
    api = ApiClient(settings)
    register_worker(
        api,
        settings,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "image_generate"]},
        loaded_models=[],
    )

    created = httpx.post(
        f"{rust_api}/api/v1/image/jobs",
        json={
            "projectId": "project-1",
            "prompt": "mist over rolling hills",
            "model": "z_image_turbo",
            "count": 1,
            "requestedGpu": "gpu-0",
        },
        timeout=5,
    )
    created.raise_for_status()
    job = created.json()
    assert job["type"] == "image_generate"

    claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})["job"]
    assert claimed["id"] == job["id"]
    assert claimed["type"] == "image_generate"
    assert claimed["payload"]["projectId"] == "project-1"

    # Drives the real dispatch path: create_image_adapter -> ProceduralImageAdapter ->
    # ImageAssetWriter, reporting completion back to the Rust API over HTTP.
    run_image_job(api, settings, claimed, {})

    completed = httpx.get(f"{rust_api}/api/v1/jobs/{job['id']}", timeout=5).json()
    assert completed["status"] == "completed"
    assert completed["workerId"] == settings.worker_id
    assets = completed["result"]["assets"]
    assert len(assets) == 1
    asset = assets[0]
    assert asset["type"] == "image"
    assert asset["file"]["mimeType"] == "image/png"
    assert asset["recipe"]["adapter"] == "procedural_preview"
    written = project_path / asset["file"]["path"]
    assert written.exists()
    assert written.suffix == ".png"


class _RecordingImageAdapter:
    """Records the ImageRequest the worker hands the adapter, then writes weightless
    preview images so the job completes. Unlike the procedural adapter it does NOT
    reject loras (that is the point — sc-2226 proves loras survive the API + worker
    boundary into request.loras for character_image angle/pose sets)."""

    id = "recording-e2e"

    def __init__(self, sink: list[dict]) -> None:
        self._sink = sink

    def loaded_models(self) -> list[str]:
        return []

    def generate(self, *, settings, job, request, project_path, progress, cancel_requested):
        self._sink.append(
            {"mode": request.mode, "loras": list(request.loras), "advanced": dict(request.advanced)}
        )
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        if request.advanced.get("angleSet"):
            count = len(CHARACTER_ANGLE_SET_ORDER)
        elif isinstance(request.advanced.get("poses"), list):
            count = len(request.advanced["poses"])
        else:
            count = request.count
        return ImageAssetWriter().write_incremental_outputs(
            request=request,
            project_path=project_path,
            image_count=count,
            image_at_index=lambda index: render_preview_image(request, model_target, index, index),
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={**request.advanced, "recordingE2E": True},
            settings=settings,
            job_id=job["id"],
        )


def test_character_image_angle_and_pose_sets_carry_loras_through_worker(rust_api, tmp_path, monkeypatch):
    """sc-2226: a character_image angle set AND pose set, each carrying a `loras` array,
    submitted through the Rust API binary and run by the in-process worker, deliver those
    loras to the adapter as request.loras (the payload -> catalog-normalized -> worker ->
    ImageRequest.loras boundary). z_image_turbo stands in as a weightless backbone; the
    angle/pose LoRA path itself is unit-covered in sc-2224/2225."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # The Rust API resolves a job's project through its own project store (for project
    # LoRAs), so create the project via the API and mirror its path into recent-projects
    # .json (how the in-process worker resolves the same project).
    created_project = httpx.post(f"{rust_api}/api/v1/projects", json={"name": "Lora E2E"}, timeout=5)
    created_project.raise_for_status()
    project = created_project.json()
    project_id = project["id"]
    project_path = Path(project["path"])
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": project_id, "path": str(project_path)}]), encoding="utf-8"
    )

    # The Rust API rejects loras not present in the catalog (it hydrates + normalizes
    # submitted specs against installed LoRAs). Seed one user LoRA whose family matches
    # the z_image_turbo backbone so the submission validates and reaches the worker.
    lora_id = "kelsie-zit"
    lora_dir = data_dir / "loras" / lora_id
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "kelsie.safetensors").write_bytes(_minimal_safetensors())
    manifests = tmp_path / "config" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    # The lora compatibility check resolves the model + builtin loras from the catalog,
    # which reads these manifests; copy the real ones so z_image_turbo (family z-image)
    # is known. Size estimation is disabled by the rust_api fixture (no network).
    for manifest_name in ("builtin.models.jsonc", "builtin.loras.jsonc"):
        shutil.copy(ROOT / "config" / "manifests" / manifest_name, manifests / manifest_name)
    (manifests / "user.loras.jsonc").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "loras": [
                    {
                        "id": lora_id,
                        "name": "Kelsie ZIT",
                        "family": "z-image",
                        "scope": "global",
                        "files": ["kelsie.safetensors"],
                        "source": {"path": f"loras/{lora_id}", "provider": "local", "repo": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("SCENEWORKS_API_URL", rust_api)
    monkeypatch.setenv("SCENEWORKS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SCENEWORKS_WORKER_ID", "lora-e2e-worker")
    monkeypatch.setenv("SCENEWORKS_GPU_ID", "gpu-0")

    # Inject a recording adapter (the procedural adapter rejects loras), so the worker's
    # real dispatch path runs but no model weights are needed.
    recorded: list[dict] = []
    monkeypatch.setattr(
        "scene_worker.runtime.create_image_adapter",
        lambda job, image_adapters: _RecordingImageAdapter(recorded),
    )

    settings = WorkerSettings()
    api = ApiClient(settings)
    register_worker(
        api,
        settings,
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "image_generate"]},
        loaded_models=[],
    )

    loras = [{"id": lora_id, "weight": 0.8}]
    pose_keypoints = [[0.5, index / 18] for index in range(18)]
    base = {
        "projectId": project_id,
        "mode": "character_image",
        "model": "z_image_turbo",
        "prompt": "the character",
        "referenceAssetId": "ref-1",
        "count": 1,
        "width": 256,
        "height": 256,
        "requestedGpu": "gpu-0",
        "loras": loras,
    }
    jobs = {
        "angle": {**base, "advanced": {"angleSet": True, "ipAdapterScale": 0.8}},
        "pose": {
            **base,
            "advanced": {"poses": [{"id": "standing_01", "keypoints": pose_keypoints}], "ipAdapterScale": 0.8},
        },
    }

    for kind, body in jobs.items():
        created = httpx.post(f"{rust_api}/api/v1/image/jobs", json=body, timeout=5)
        assert created.status_code == 201, (kind, created.status_code, created.text)
        job = created.json()

        claimed = api.post("/api/v1/jobs/claim", {"workerId": settings.worker_id})["job"]
        assert claimed["id"] == job["id"], kind
        # The Rust API persisted + served the (normalized) loras on the claimed payload.
        assert [lora["id"] for lora in claimed["payload"]["loras"]] == [lora_id], kind

        run_image_job(api, settings, claimed, {})

        completed = httpx.get(f"{rust_api}/api/v1/jobs/{job['id']}", timeout=5).json()
        assert completed["status"] == "completed", (kind, completed)

    # The adapter received request.loras for BOTH the angle set and the pose set.
    assert len(recorded) == 2
    assert all([lora["id"] for lora in entry["loras"]] == [lora_id] for entry in recorded)
    angle_entry = next(entry for entry in recorded if entry["advanced"].get("angleSet"))
    pose_entry = next(entry for entry in recorded if entry["advanced"].get("poses"))
    assert angle_entry["loras"][0]["weight"] == 0.8
    assert pose_entry["loras"][0]["weight"] == 0.8


def test_rust_worker_claims_and_completes_lora_import_against_rust_api_binary(rust_api, tmp_path):
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for the Rust worker smoke test")

    # The Rust API only imports a sourcePath from app-managed roots (data/loras,
    # project loras, or staged uploads) for path safety; stage the source inside
    # the worker's data/loras dir so the import isn't rejected.
    source = tmp_path / "data" / "loras" / "tiny.safetensors"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(_minimal_safetensors())
    env = os.environ.copy()
    env.update(
        {
            "SCENEWORKS_API_URL": rust_api,
            "SCENEWORKS_DATA_DIR": str(tmp_path / "data"),
            "SCENEWORKS_CONFIG_DIR": str(tmp_path / "config"),
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

        # The supervisor spawns per-device child workers (e.g. <id>-cpu-2), so the
        # configured WORKER_ID is a prefix of the claiming worker's id.
        assert completed["workerId"].startswith("rust-worker-smoke")
        assert completed["result"]["repo"] is None
        assert completed["result"]["path"].endswith("smoke_lora")
        assert (
            tmp_path / "data" / "loras" / "smoke_lora" / "tiny.safetensors"
        ).read_bytes() == _minimal_safetensors()
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
            "SCENEWORKS_CONFIG_DIR": str(tmp_path / "config"),
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
        detection_jobs = []
        for index in range(5):
            detection_job = httpx.post(
                f"{rust_api}/api/v1/projects/{project_id}/person-tracks/detections",
                json={"sourceAssetId": asset_id, "sourceTimestamp": index * 0.1},
                timeout=5,
            )
            detection_job.raise_for_status()
            detection_jobs.append(detection_job.json())

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
        detection_completed = [
            wait_for_job_status(rust_api, job["id"], "completed", worker) for job in detection_jobs
        ]
        first_detection = detection_completed[0]["result"]["detections"][0]
        track_job = httpx.post(
            f"{rust_api}/api/v1/projects/{project_id}/person-tracks/jobs",
            json={
                "sourceAssetId": asset_id,
                "representativeFrameAssetId": detection_completed[0]["result"]["frameAssetId"],
                "detection": first_detection,
                "trackName": "Hero",
            },
            timeout=5,
        )
        track_job.raise_for_status()
        track_completed = wait_for_job_status(rust_api, track_job.json()["id"], "completed", worker)

        assert frame_completed["workerId"] == "rust-ffmpeg-smoke"
        assert frame_completed["result"]["assets"][0]["type"] == "frame"
        assert frame_completed["result"]["assets"][0]["recipe"]["mode"] == "frame_extract"
        assert export_completed["workerId"] == "rust-ffmpeg-smoke"
        assert export_completed["result"]["assets"][0]["type"] == "render"
        assert export_completed["result"]["assets"][0]["file"]["mimeType"] == "video/mp4"
        assert {job["workerId"] for job in detection_completed} == {"rust-ffmpeg-smoke"}
        assert all(job["result"]["detections"] for job in detection_completed)
        assert track_completed["workerId"] == "rust-ffmpeg-smoke"
        assert track_completed["result"]["track"]["recipe"]["mode"] == "person_track"
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
