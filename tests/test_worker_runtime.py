from __future__ import annotations

from types import SimpleNamespace

from scene_worker.image_adapters import (
    MODEL_TARGETS,
    ZImageDiffusersAdapter,
    build_asset_sidecar,
    huggingface_repo_cache_path,
    image_request_from_job,
    resolve_seed,
)
from scene_worker.runtime import (
    child_environment,
    friendly_failure,
    heartbeat,
    keep_job_alive,
    loaded_models_from_adapters,
    main,
    resolve_loaded_models,
    run_check,
    run_video_job,
    worker_capabilities,
)
from scene_worker.video_adapters import VIDEO_MODEL_TARGETS, build_video_asset_sidecar, video_request_from_job


def test_cpu_worker_does_not_advertise_gpu_generation_capabilities():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert capabilities == ["cpu"]
    assert "placeholder" not in capabilities


def test_gpu_worker_advertises_generation_capabilities():
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "video_generate" in capabilities
    assert "person_replace" in capabilities
    assert "placeholder" not in capabilities


def test_python_worker_only_advertises_inference_job_capabilities():
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    job_capabilities = [capability for capability in capabilities if capability != "gpu"]

    assert job_capabilities == [
        "image_edit",
        "image_generate",
        "person_replace",
        "video_bridge",
        "video_extend",
        "video_generate",
    ]


def test_python_cpu_worker_does_not_advertise_person_tracking_jobs():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert capabilities == ["cpu"]


def test_python_cpu_child_disables_cuda():
    env = child_environment(SimpleNamespace(), worker_id="python-inference-worker-cpu", gpu_id="cpu")

    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_python_gpu_child_selects_cuda_device():
    env = child_environment(SimpleNamespace(), worker_id="worker-gpu-0", gpu_id="0")

    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_loaded_models_are_collected_from_adapter_cache():
    class Adapter:
        def loaded_models(self):
            return ["Tongyi-MAI/Z-Image-Turbo"]

    assert loaded_models_from_adapters({"z": Adapter()}) == ["Tongyi-MAI/Z-Image-Turbo"]


def test_z_image_loaded_models_include_repo_and_model_id():
    adapter = ZImageDiffusersAdapter()
    adapter._loaded_repo = "Tongyi-MAI/Z-Image-Turbo"
    adapter._loaded_model = "z_image_turbo"

    assert set(adapter.loaded_models()) == {"Tongyi-MAI/Z-Image-Turbo", "z_image_turbo"}


def test_huggingface_repo_cache_path_stays_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    path = huggingface_repo_cache_path(r"..\outside/../../model")

    assert path is not None
    path.relative_to((tmp_path / "hub").resolve())
    assert path.name.startswith("models--")


def test_friendly_failure_identifies_gpu_oom():
    message, error = friendly_failure("Image generation", RuntimeError("CUDA error: out of memory"))

    assert message == "Image generation failed because the GPU ran out of memory."
    assert "lower resolution" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_missing_model_files():
    message, error = friendly_failure("Image generation", RuntimeError("Repository not found: owner/model"))

    assert message == "Image generation failed because required model files were not available."
    assert "Model Manager" in error
    assert "Rust utility worker" in error
    assert "HF_TOKEN" in error


def test_worker_check_reports_inference_sidecar_capabilities(monkeypatch):
    events = []
    monkeypatch.setattr("scene_worker.runtime.emit", events.append)
    monkeypatch.setattr(
        "scene_worker.runtime.discover_gpu",
        lambda _gpu_id: {"id": "0", "name": "GPU 0", "capabilities": ["gpu"]},
    )

    run_check(SimpleNamespace(worker_id="worker-1", gpu_id="0"))

    assert events[0]["event"] == "worker_check"
    assert events[0]["jobTypes"] == [
        "image_generate",
        "image_edit",
        "video_generate",
        "video_extend",
        "video_bridge",
        "person_replace",
    ]
    assert events[0]["supportedJobTypes"] == events[0]["jobTypes"]


def test_main_check_exits_without_api_loop(monkeypatch):
    calls = []
    monkeypatch.setattr("scene_worker.runtime.run_check", lambda settings: calls.append(settings.worker_id))

    main(["--check"])

    assert calls == ["worker-local-0"]


def test_heartbeat_loaded_models_are_not_sent_as_current_job():
    class Api:
        def __init__(self):
            self.path = None
            self.payload = None

        def post(self, path, payload):
            self.path = path
            self.payload = payload
            return {}

    class Settings:
        worker_id = "worker-1"

    api = Api()
    heartbeat(api, Settings(), "idle", loaded_models=["model-a"])

    assert api.path == "/api/v1/workers/worker-1/heartbeat"
    assert api.payload == {"status": "idle", "currentJobId": None, "loadedModels": ["model-a"]}


def test_keepalive_heartbeat_reports_current_loaded_models_each_tick(monkeypatch):
    calls = []
    models_by_tick = [["model-a"], ["model-b"]]
    loaded_model_calls = []

    class StopAfterTwoHeartbeats:
        def __init__(self):
            self.calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls > 2

    class Settings:
        worker_id = "worker-1"
        heartbeat_seconds = 1

    def capture_heartbeat(api, settings, status, current_job_id=None, loaded_models=None):
        calls.append(
            {
                "api": api,
                "settings": settings,
                "status": status,
                "current_job_id": current_job_id,
                "loaded_models": loaded_models,
            }
        )

    monkeypatch.setattr("scene_worker.runtime.heartbeat", capture_heartbeat)

    def loaded_models():
        loaded_model_calls.append("tick")
        return models_by_tick[len(loaded_model_calls) - 1]

    keep_job_alive(
        api=object(),
        settings=Settings(),
        job_id="job-1",
        status="busy",
        stop_event=StopAfterTwoHeartbeats(),
        loaded_models=loaded_models,
    )

    assert [call["status"] for call in calls] == ["busy", "busy"]
    assert [call["current_job_id"] for call in calls] == ["job-1", "job-1"]
    assert [call["loaded_models"] for call in calls] == [["model-a"], ["model-b"]]
    assert len(loaded_model_calls) == 2


def test_keepalive_heartbeat_reports_empty_models_when_source_is_none(monkeypatch):
    calls = []

    class StopAfterOneHeartbeat:
        def wait(self, _interval):
            return bool(calls)

    class Settings:
        worker_id = "worker-1"
        heartbeat_seconds = 1

    def capture_heartbeat(_api, _settings, _status, _current_job_id=None, loaded_models=None):
        calls.append(loaded_models)

    monkeypatch.setattr("scene_worker.runtime.heartbeat", capture_heartbeat)

    keep_job_alive(
        api=object(),
        settings=Settings(),
        job_id="job-1",
        status="busy",
        stop_event=StopAfterOneHeartbeat(),
        loaded_models=None,
    )

    assert calls == [[]]


def test_loaded_model_resolution_failure_keeps_heartbeat_alive(monkeypatch):
    events = []

    def failing_loaded_models():
        raise RuntimeError("cache is mid-load")

    monkeypatch.setattr("scene_worker.runtime.emit", events.append)

    assert resolve_loaded_models(failing_loaded_models, job_id="job-1") == []
    assert events == [
        {
            "event": "loaded_models_failed",
            "error": "cache is mid-load",
            "jobId": "job-1",
            "reportedAt": events[0]["reportedAt"],
        }
    ]


def test_video_job_reports_dynamic_loaded_models_on_progress_and_keepalive(monkeypatch):
    heartbeat_models = []
    blocking_models = []

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                heartbeat_models.append(payload["loadedModels"])
                return {}
            if path.endswith("/progress"):
                return {"status": payload["status"], "stage": payload["stage"]}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def __init__(self):
            self.models = []

        def loaded_models(self):
            return list(self.models)

        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            self.models = ["video-model-loaded"]

        def estimate_requirements(self, _request):
            return {"previewFrames": 1}

        def run(self, *, settings, job, request, progress, cancel_requested):
            self.models = ["video-model-running"]
            progress("running", "generating", 0.5, "Rendering.")
            return {"assetId": "asset-video-1"}

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            raise AssertionError("cleanup should not be called")

    def run_immediately(_api, _settings, _job_id, _status, callback, *, loaded_models):
        blocking_models.append(loaded_models())
        result = callback()
        blocking_models.append(loaded_models())
        return result

    monkeypatch.setattr("scene_worker.runtime.ProceduralVideoAdapter", VideoAdapter)
    monkeypatch.setattr("scene_worker.runtime.run_blocking_job_step", run_immediately)

    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1"),
        {"id": "job-1", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert heartbeat_models == [
        [],
        [],
        ["video-model-loaded"],
        ["video-model-running"],
        ["video-model-running"],
    ]
    assert blocking_models == [["video-model-loaded"], ["video-model-running"]]


def test_random_batch_seeds_are_used_per_image():
    assert resolve_seed(None, "city at night", 2, [101, 202, 303, 404]) == 303


def test_explicit_seed_uses_reproducible_ladder():
    assert resolve_seed(1234, "city at night", 2, [101, 202, 303, 404]) == 1236


def test_character_image_recipe_marks_conditioning_inactive():
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "character_image",
            "prompt": "Mira portrait",
            "model": "z_image_turbo",
            "characterId": "character-1",
            "characterLookId": "look-1",
            "advanced": {},
        },
    }
    request = image_request_from_job(job)

    asset = build_asset_sidecar(
        asset_id="asset-1",
        project_id="project-1",
        generation_set_id="genset-1",
        request=request,
        job_id="job-1",
        media_rel="assets/images/mira.png",
        created_at="2026-05-17T00:00:00Z",
        seed=101,
        index=0,
        model_target=MODEL_TARGETS["z_image_turbo"],
        adapter_id="procedural_preview",
        raw_settings={},
    )

    normalized = asset["recipe"]["normalizedSettings"]
    assert normalized["characterId"] == "character-1"
    assert normalized["characterLookId"] == "look-1"
    assert normalized["characterConditioningActive"] is False


def test_replace_person_video_sidecar_preserves_lineage():
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "replace_person",
            "prompt": "Replace the hero",
            "model": "wan_2_2",
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "characterId": "character-1",
            "characterLookId": "look-1",
            "replacementMode": "full_person_keep_outfit",
            "advanced": {},
        },
    }
    request = video_request_from_job(job)

    asset = build_video_asset_sidecar(
        asset_id="asset-output",
        project_id="project-1",
        generation_set_id="genset-1",
        request=request,
        job_id="job-1",
        media_rel="assets/videos/replacement.webp",
        created_at="2026-05-17T00:00:00Z",
        seed=44,
        target=VIDEO_MODEL_TARGETS["wan_2_2"],
        raw_settings={},
    )

    assert asset["recipe"]["mode"] == "replace_person"
    assert asset["recipe"]["normalizedSettings"]["personTrackId"] == "track-1"
    assert asset["recipe"]["normalizedSettings"]["replacementMode"] == "full_person_keep_outfit"
    assert asset["recipe"]["normalizedSettings"]["personDetectionActive"] is False
    assert asset["recipe"]["normalizedSettings"]["personTrackingActive"] is False
    assert asset["recipe"]["normalizedSettings"]["replacementActive"] is False
    assert asset["lineage"]["sourceClipAssetId"] == "asset-video"
    assert asset["lineage"]["characterId"] == "character-1"
