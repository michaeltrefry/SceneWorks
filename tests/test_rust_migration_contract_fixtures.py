from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from scene_worker.image_adapters import ImageAssetWriter
from scene_worker.runtime import (
    worker_capabilities,
    write_model_install_marker,
)
from scene_worker.video_adapters import (
    VIDEO_MODEL_TARGETS,
    build_video_asset_sidecar,
    run_person_track,
    video_request_from_job,
)
from sceneworks_api.assets import ASSET_SIDECAR_PATTERN, get_project_file
from sceneworks_api.characters import CHARACTER_SIDECAR_PATTERN, CharacterCreate, create_character
from sceneworks_api.jobs import (
    JobCreateRequest,
    JobType,
    ProgressRequest,
    WorkerHeartbeatRequest,
    WorkerRegisterRequest,
)
from sceneworks_api.jobs_store import ACTIVE_STATUSES, JOB_STATUSES, JobsStore, NON_GPU_JOB_TYPES, TERMINAL_STATUSES
from sceneworks_api.main import create_app
from sceneworks_api.models import load_manifest, model_install_marker
from sceneworks_api.projects import PROJECT_FOLDERS, write_project_file
from sceneworks_api.settings import Settings
from sceneworks_api.timelines import TimelineCreateRequest, create_timeline
from sceneworks_shared import index_asset, write_json


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rust_migration_contracts"
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCENEWORKS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCENEWORKS_JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    return create_app(Settings())


def openapi_surface(app) -> list[dict[str, str]]:
    surface = []
    for path, operations in app.openapi()["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        for method in operations:
            if method.upper() in HTTP_METHODS:
                surface.append({"path": path, "method": method.upper()})
    return sorted(surface, key=lambda item: (item["path"], item["method"]))


def fixture_surface() -> list[dict[str, str]]:
    fixture = load_fixture("api_surface.json")
    surface = []
    for endpoint in fixture["endpoints"]:
        for method in endpoint["methods"]:
            surface.append({"path": endpoint["path"], "method": method})
    return sorted(surface, key=lambda item: (item["path"], item["method"]))


def openapi_component_schemas_hash(app) -> str:
    schemas = app.openapi().get("components", {}).get("schemas", {})
    payload = json.dumps(schemas, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_api_surface_fixture_matches_python_openapi(tmp_path, monkeypatch):
    fixture = load_fixture("api_surface.json")
    app = build_app(tmp_path, monkeypatch)

    assert fixture_surface() == openapi_surface(app)
    assert fixture["openapiComponentSchemasSha256"] == openapi_component_schemas_hash(app)
    for endpoint in fixture["endpoints"]:
        assert endpoint["family"]
        assert endpoint["usedBy"]


def source_text(*relative_paths: str) -> str:
    return "\n".join((ROOT / path).read_text(encoding="utf-8") for path in relative_paths)


def progress_stages_from_sources() -> set[str]:
    text = source_text(
        "apps/worker/scene_worker/runtime.py",
        "apps/worker/scene_worker/image_adapters.py",
        "apps/worker/scene_worker/video_adapters.py",
        "apps/worker/scene_worker/timeline_exporter.py",
        "apps/api/sceneworks_api/jobs_store.py",
    )
    stages = set(re.findall(r'"stage"\s*:\s*"([^"]+)"', text))
    stages.update(re.findall(r"stage\s*=\s*'([^']+)'", text))
    stages.update(re.findall(r'progress\(\s*"[^"]+"\s*,\s*"([^"]+)"', text))
    return stages


def sse_events_from_sources() -> set[str]:
    text = source_text(
        "apps/api/sceneworks_api/events.py",
        "apps/api/sceneworks_api/jobs.py",
        "apps/api/sceneworks_api/models.py",
        "apps/api/sceneworks_api/image_generation.py",
        "apps/api/sceneworks_api/video_generation.py",
        "apps/api/sceneworks_api/timelines.py",
        "apps/api/sceneworks_api/person_tracking.py",
        "apps/api/sceneworks_api/characters.py",
    )
    events = set(re.findall(r'\.publish\(\s*"([^"]+)"', text))
    events.update(re.findall(r'publish\([^,\n]+,\s*"([^"]+)"', text))
    events.update(re.findall(r'"event"\s*:\s*"([^"]+)"', text))
    events.update(re.findall(r"event:\s*([A-Za-z0-9_.-]+)\\n", text))
    return events


def worker_statuses_from_store(tmp_path: Path) -> set[str]:
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    worker = store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name="GPU 0",
        capabilities=["image_generate"],
        loaded_models=[],
    )
    idle = worker["status"]
    job = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={},
        requested_gpu="auto",
    )
    store.claim_next_job("worker-1")
    busy = store.get_worker("worker-1")["status"]
    with sqlite3.connect(tmp_path / "jobs.db") as connection:
        connection.execute("update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = 'worker-1'")
        connection.execute("update jobs set last_heartbeat_at = '2000-01-01T00:00:00Z' where id = ?", (job["id"],))
    store.mark_stale_workers_interrupted(1)
    offline = store.get_worker("worker-1")["status"]
    return {idle, busy, offline}


def test_job_protocol_fixture_matches_python_contracts(tmp_path, monkeypatch):
    fixture = load_fixture("job_protocol.json")

    assert fixture["jobTypes"] == list(JobType.__args__)
    assert fixture["statuses"] == list(JOB_STATUSES)
    assert fixture["activeStatuses"] == list(ACTIVE_STATUSES)
    assert fixture["terminalStatuses"] == list(TERMINAL_STATUSES)
    assert fixture["nonGpuJobTypes"] == list(NON_GPU_JOB_TYPES)
    assert set(fixture["progressStages"]) >= progress_stages_from_sources()
    assert set(fixture["sseEvents"]) == sse_events_from_sources()
    assert set(fixture["workerStatuses"]) == worker_statuses_from_store(tmp_path)

    monkeypatch.setenv("SCENEWORKS_UTILITY_JOBS", "1")
    assert fixture["workerCapabilityProfiles"]["cpu"] == worker_capabilities(
        {"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]}
    )

    monkeypatch.setenv("SCENEWORKS_LEGACY_MODEL_LORA_JOBS", "1")
    assert fixture["workerCapabilityProfiles"]["cpuLegacy"] == worker_capabilities(
        {"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]}
    )

    monkeypatch.delenv("SCENEWORKS_LEGACY_MODEL_LORA_JOBS")
    monkeypatch.setenv("SCENEWORKS_UTILITY_JOBS", "0")
    assert fixture["workerCapabilityProfiles"]["gpuChild"] == worker_capabilities(
        {"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]}
    )

    JobCreateRequest.model_validate(fixture["requests"]["createJob"])
    WorkerRegisterRequest.model_validate(fixture["requests"]["registerWorker"])
    WorkerHeartbeatRequest.model_validate(fixture["requests"]["heartbeat"])
    ProgressRequest.model_validate(fixture["requests"]["progress"])

    store = JobsStore(tmp_path / "snapshot.db")
    store.initialize()
    snapshot = store.create_job(
        job_type="image_generate",
        project_id="project_fixture",
        project_name="Fixture Project",
        payload={"prompt": "mist over hills", "model": "z_image_turbo"},
        requested_gpu="auto",
    )
    assert set(fixture["jobSnapshot"]) == set(snapshot)


def request_for_project(tmp_path: Path, project_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    registry_path = data_dir / "recent-projects.json"
    registry_path.write_text(
        json.dumps([{"id": "project_fixture", "path": str(project_path)}]),
        encoding="utf-8",
    )
    state = SimpleNamespace(
        settings=SimpleNamespace(
            data_dir=data_dir,
            registry_path=registry_path,
            app_version="0.1.0",
            worker_timeout_seconds=120,
        )
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def create_live_sidecars(tmp_path: Path) -> dict[str, dict]:
    data_dir = tmp_path / "data"
    project_path = data_dir / "projects" / "fixture.sceneworks"
    for folder in PROJECT_FOLDERS:
        (project_path / folder).mkdir(parents=True, exist_ok=True)
    request = request_for_project(tmp_path, project_path)
    worker_settings = SimpleNamespace(data_dir=data_dir)

    project = write_project_file(request.app.state.settings, project_path, "project_fixture", "Fixture Project")

    image_result = ImageAssetWriter().write_outputs(
        settings=worker_settings,
        job={
            "id": "job_fixture",
            "payload": {
                "projectId": "project_fixture",
                "mode": "text_to_image",
                "prompt": "mist over hills",
                "negativePrompt": "",
                "model": "z_image_turbo",
                "count": 1,
                "seed": 101,
                "width": 1024,
                "height": 1024,
                "stylePreset": "cinematic",
                "loras": [],
                "advanced": {},
            },
        },
        images=[Image.new("RGB", (16, 16), color=(12, 34, 56))],
        adapter_id="procedural_preview",
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
        raw_settings={},
    )
    image_asset = image_result["assets"][0]
    generation_set = load_json(project_path / "generation-sets" / f"{image_result['generationSetId']}.json")
    recipe = load_json(project_path / "recipes" / f"{image_asset['id']}.recipe.json")

    video_job = {
        "id": "job_fixture",
        "payload": {
            "projectId": "project_fixture",
            "mode": "replace_person",
            "prompt": "Hero walks through rain",
            "model": "wan_2_2",
            "sourceClipAssetId": "asset_source_clip",
            "personTrackId": "track_fixture",
            "characterId": "character_fixture",
            "characterLookId": "look_fixture",
            "replacementMode": "full_person_keep_outfit",
            "advanced": {},
        },
    }
    video_asset = build_video_asset_sidecar(
        asset_id="asset_video_fixture",
        project_id="project_fixture",
        generation_set_id="genset_fixture",
        request=video_request_from_job(video_job),
        job_id="job_fixture",
        media_rel="assets/videos/replacement.webp",
        created_at="2026-05-17T13:00:00Z",
        seed=44,
        target=VIDEO_MODEL_TARGETS["wan_2_2"],
        raw_settings={},
    )

    character = create_character(
        "project_fixture",
        CharacterCreate(name="Mira", type="person", description="Lead character reference."),
        request,
    )
    timeline = create_timeline(
        "project_fixture",
        TimelineCreateRequest(name="Main timeline", aspectRatio="16:9", fps=30),
        request,
    ).model_dump()

    source_media = project_path / "assets" / "videos" / "source.webp"
    Image.new("RGB", (16, 16), color=(32, 48, 64)).save(source_media, "WEBP")
    source_asset = {
        "schemaVersion": 1,
        "id": "asset_source_clip",
        "projectId": "project_fixture",
        "type": "video",
        "displayName": "Source clip",
        "createdAt": "2026-05-17T13:00:00Z",
        "generationSetId": None,
        "file": {"path": "assets/videos/source.webp", "duration": 2},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
    }
    source_sidecar = source_media.with_suffix(".sceneworks.json")
    write_json(source_sidecar, source_asset)
    index_asset(project_path, source_asset, source_sidecar)
    person_track = run_person_track(
        settings=worker_settings,
        job={
            "id": "job_fixture",
            "payload": {
                "projectId": "project_fixture",
                "sourceAssetId": "asset_source_clip",
                "representativeFrameAssetId": "asset_frame_fixture",
                "detection": {
                    "id": "person_1",
                    "box": {"x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6},
                    "confidence": 0.92,
                },
                "trackName": "Hero",
            },
        },
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )["track"]

    marker_dir = tmp_path / "model"
    marker_dir.mkdir()
    write_model_install_marker(
        marker_dir,
        {"modelId": "z_image_turbo", "modelName": "Z-Image-Turbo"},
        "Tongyi-MAI/Z-Image-Turbo",
        "job_fixture",
    )

    return {
        "project": project,
        "imageAsset": image_asset,
        "videoAsset": video_asset,
        "generationSet": generation_set,
        "recipe": recipe,
        "character": character,
        "timeline": timeline,
        "personTrack": person_track,
        "modelManifestEntry": load_manifest(ROOT / "config" / "manifests" / "builtin.models.jsonc")[0],
        "modelInstallMarker": load_json(model_install_marker(marker_dir)),
    }


def test_resource_sidecar_fixtures_match_live_writer_shapes(tmp_path):
    fixture = load_fixture("resource_sidecars.json")
    live_payloads = create_live_sidecars(tmp_path)

    assert fixture["projectFolders"] == PROJECT_FOLDERS
    assert fixture["sidecarPatterns"]["asset"] == ASSET_SIDECAR_PATTERN
    assert fixture["sidecarPatterns"]["character"] == CHARACTER_SIDECAR_PATTERN
    assert fixture["sidecarPatterns"]["timeline"] == "*.sceneworks.timeline.json"
    assert fixture["sidecarPatterns"]["personTrack"] == "*.sceneworks.person-track.json"
    assert fixture["sidecarPatterns"]["generationSet"] == "generation-sets/*.json"
    assert fixture["sidecarPatterns"]["recipe"] == "recipes/*.recipe.json"
    assert fixture["sidecarPatterns"]["project"] == "project.json"
    assert fixture["sidecarPatterns"]["modelInstallMarker"] == model_install_marker(Path()).name

    for sidecar in fixture["fixtures"]:
        payload_path = FIXTURE_DIR / sidecar["path"]
        assert payload_path.exists(), f"Missing fixture: {payload_path}"
        fixture_payload = load_json(payload_path)
        assert set(sidecar["requiredTopLevelKeys"]).issubset(fixture_payload.keys()), sidecar["name"]
        if sidecar["name"] in live_payloads:
            assert set(sidecar["requiredTopLevelKeys"]).issubset(live_payloads[sidecar["name"]].keys()), sidecar["name"]
        if sidecar["name"] == "personTrack":
            assert "timestamp" in fixture_payload["frames"][0]
            assert "time" not in fixture_payload["frames"][0]
            assert "mask" in fixture_payload["frames"][0]
            assert "timestamp" in live_payloads["personTrack"]["frames"][0]
            assert "time" not in live_payloads["personTrack"]["frames"][0]
            assert "mask" in live_payloads["personTrack"]["frames"][0]


def test_security_behavior_fixture_matches_python_files_endpoint(tmp_path):
    fixture = load_fixture("api_surface.json")
    cases = {case["name"]: case for case in fixture["securityBehaviors"]}
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    request = request_for_project(tmp_path, project_path)

    with pytest.raises(HTTPException) as traversal:
        get_project_file("project_fixture", "../outside.txt", request)
    assert traversal.value.status_code == cases["projectFilePathTraversal"]["statusCode"]
    assert traversal.value.detail == cases["projectFilePathTraversal"]["detail"]

    with pytest.raises(HTTPException) as missing:
        get_project_file("project_fixture", "assets/images/missing.png", request)
    assert missing.value.status_code == cases["projectFileMissing"]["statusCode"]
    assert missing.value.detail == cases["projectFileMissing"]["detail"]
