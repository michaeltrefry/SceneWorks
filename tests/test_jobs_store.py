from __future__ import annotations

from sceneworks_api.jobs_store import JobsStore


def test_job_lifecycle_create_claim_complete(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name="GPU 0",
        capabilities=["image_generate"],
        loaded_models=[],
    )

    created = store.create_job(
        job_type="image_generate",
        project_id="project-1",
        project_name="Project 1",
        payload={"prompt": "mist over hills"},
        requested_gpu="auto",
    )
    claimed = store.claim_next_job("worker-1")

    assert claimed is not None
    assert claimed["id"] == created["id"]
    assert claimed["status"] == "preparing"
    assert claimed["assignedGpu"] == "gpu-0"

    completed = store.update_job_progress(
        claimed["id"],
        status="completed",
        stage="completed",
        progress=1,
        message="Done",
        result={"assetIds": ["asset-1"]},
    )
    worker = store.get_worker("worker-1")

    assert completed["status"] == "completed"
    assert completed["result"] == {"assetIds": ["asset-1"]}
    assert worker["status"] == "idle"
    assert worker["currentJobId"] is None


def test_non_gpu_jobs_can_claim_while_gpu_is_busy(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name=None,
        capabilities=["image_generate", "model_download"],
        loaded_models=[],
    )

    gpu_job = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={},
        requested_gpu="auto",
    )
    download_job = store.create_job(
        job_type="model_download",
        project_id=None,
        project_name=None,
        payload={"repo": "owner/model"},
        requested_gpu="auto",
    )

    assert store.claim_next_job("worker-1")["id"] == gpu_job["id"]
    assert store.claim_next_job("worker-1")["id"] == download_job["id"]


def test_claim_skips_jobs_not_supported_by_worker_capabilities(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name=None,
        capabilities=["model_download"],
        loaded_models=[],
    )
    store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={},
        requested_gpu="auto",
    )
    download_job = store.create_job(
        job_type="model_download",
        project_id=None,
        project_name=None,
        payload={"repo": "owner/model"},
        requested_gpu="auto",
    )

    assert store.claim_next_job("worker-1")["id"] == download_job["id"]


def test_auto_claim_prefers_job_matching_loaded_model(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name=None,
        capabilities=["image_generate"],
        loaded_models=["z_image_turbo", "Tongyi-MAI/Z-Image-Turbo"],
    )
    other_model_job = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "qwen_image_edit"},
        requested_gpu="auto",
    )
    warm_model_job = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "z_image_turbo"},
        requested_gpu="auto",
    )

    claimed = store.claim_next_job("worker-1")

    assert claimed["id"] == warm_model_job["id"]
    assert store.get_job(other_model_job["id"])["status"] == "queued"


def test_loaded_model_preference_does_not_skip_explicit_gpu_job(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name=None,
        capabilities=["image_generate"],
        loaded_models=["z_image_turbo"],
    )
    explicit_job = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "qwen_image_edit"},
        requested_gpu="gpu-0",
    )
    store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "z_image_turbo"},
        requested_gpu="auto",
    )

    assert store.claim_next_job("worker-1")["id"] == explicit_job["id"]


def test_explicit_gpu_job_beats_younger_warm_auto_match(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name=None,
        capabilities=["image_generate"],
        loaded_models=["model-x"],
    )
    auto_other = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "model-y"},
        requested_gpu="auto",
    )
    explicit_job = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "model-y"},
        requested_gpu="gpu-0",
    )
    warm_auto = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={"model": "model-x"},
        requested_gpu="auto",
    )

    claimed = store.claim_next_job("worker-1")

    assert claimed["id"] == explicit_job["id"]
    assert store.get_job(auto_other["id"])["status"] == "queued"
    assert store.get_job(warm_auto["id"])["status"] == "queued"


def test_idle_heartbeat_interrupts_previous_active_job(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    store.register_worker(
        worker_id="worker-1",
        gpu_id="gpu-0",
        gpu_name=None,
        capabilities=["image_generate"],
        loaded_models=[],
    )
    created = store.create_job(
        job_type="image_generate",
        project_id=None,
        project_name=None,
        payload={},
        requested_gpu="auto",
    )
    claimed = store.claim_next_job("worker-1")

    assert claimed["id"] == created["id"]

    worker = store.heartbeat_worker(
        worker_id="worker-1",
        status="idle",
        current_job_id=None,
        loaded_models=[],
    )
    job = store.get_job(created["id"])

    assert worker["status"] == "idle"
    assert worker["currentJobId"] is None
    assert job["status"] == "interrupted"
    assert job["workerId"] is None


def test_retry_job_is_capped(tmp_path):
    store = JobsStore(tmp_path / "jobs.db")
    store.initialize()
    job = store.create_job(
        job_type="placeholder",
        project_id=None,
        project_name=None,
        payload={},
        requested_gpu="auto",
        attempts=5,
    )

    import pytest

    with pytest.raises(ValueError):
        store.retry_job(job["id"])
