use std::fs;
use std::path::PathBuf;

use rusqlite::{params, Connection};
use sceneworks_core::contracts::{
    JobStatus, JobType, ProgressStage, WorkerCapability, WorkerStatus,
};
use sceneworks_core::jobs_store::{
    CreateJob, DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker,
    WorkerHeartbeat, MAX_JOB_ATTEMPTS,
};
use serde_json::{json, Map, Value};

fn temp_db(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("sceneworks-core-{name}-{}.db", std::process::id()));
    let _ = fs::remove_file(&path);
    path
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("test value is an object").clone()
}

fn store(name: &str) -> JobsStore {
    let store = JobsStore::new(temp_db(name));
    store.initialize().expect("store initializes");
    store
}

fn image_job(payload: Map<String, Value>) -> CreateJob {
    CreateJob {
        job_type: JobType::ImageGenerate,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload,
        requested_gpu: "auto".to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    }
}

fn register_image_worker(store: &JobsStore) {
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("GPU 0".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
        })
        .expect("worker registers");
}

#[test]
fn job_lifecycle_create_claim_complete() {
    let store = store("lifecycle");
    register_image_worker(&store);

    let created = store
        .create_job(image_job(object(json!({ "prompt": "mist over hills" }))))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);
    assert_eq!(claimed.status, JobStatus::Preparing);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("gpu-0"));

    let completed = store
        .update_job_progress(
            &claimed.id,
            ProgressUpdate {
                status: JobStatus::Completed,
                stage: ProgressStage::Completed,
                progress: 1.0,
                message: "Done".to_owned(),
                error: None,
                result: Some(object(json!({ "assetIds": ["asset-1"] }))),
                eta_seconds: None,
            },
        )
        .expect("progress updates");
    let worker = store.get_worker("worker-1").expect("worker loads");

    assert_eq!(completed.status, JobStatus::Completed);
    assert_eq!(completed.result, object(json!({ "assetIds": ["asset-1"] })));
    assert_eq!(worker.status, WorkerStatus::Idle);
    assert_eq!(worker.current_job_id, None);
}

#[test]
fn non_gpu_jobs_can_claim_while_gpu_is_busy() {
    let store = store("non-gpu-claim");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![
                WorkerCapability::ImageGenerate,
                WorkerCapability::ModelDownload,
            ],
            loaded_models: Vec::new(),
        })
        .expect("worker registers");

    let gpu_job = store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");
    let download_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelDownload,
            project_id: None,
            project_name: None,
            payload: object(json!({ "repo": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
        .expect("download job creates");

    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("first claim succeeds")
            .expect("first job")
            .id,
        gpu_job.id
    );
    let second = store
        .claim_next_job("worker-1")
        .expect("second claim succeeds")
        .expect("second job");
    assert_eq!(second.id, download_job.id);
    assert_eq!(second.assigned_gpu.as_deref(), Some("cpu"));
}

#[test]
fn claim_skips_jobs_not_supported_by_worker_capabilities() {
    let store = store("capabilities");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ModelDownload],
            loaded_models: Vec::new(),
        })
        .expect("worker registers");
    store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");
    let download_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelDownload,
            project_id: None,
            project_name: None,
            payload: object(json!({ "repo": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
        .expect("download job creates");

    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        download_job.id
    );
}

#[test]
fn auto_claim_prefers_job_matching_loaded_model() {
    let store = store("loaded-model-preference");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: vec![
                "z_image_turbo".to_owned(),
                "Tongyi-MAI/Z-Image-Turbo".to_owned(),
            ],
        })
        .expect("worker registers");
    let other_model_job = store
        .create_job(image_job(object(json!({ "model": "qwen_image_edit" }))))
        .expect("other model job creates");
    let warm_model_job = store
        .create_job(image_job(object(json!({ "model": "z_image_turbo" }))))
        .expect("warm model job creates");

    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, warm_model_job.id);
    assert_eq!(
        store
            .get_job(&other_model_job.id)
            .expect("other model job loads")
            .status,
        JobStatus::Queued
    );
}

#[test]
fn loaded_model_preference_does_not_skip_explicit_gpu_job() {
    let store = store("loaded-model-explicit-gpu");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: vec!["z_image_turbo".to_owned()],
        })
        .expect("worker registers");
    let explicit_job = store
        .create_job(CreateJob {
            requested_gpu: "gpu-0".to_owned(),
            ..image_job(object(json!({ "model": "qwen_image_edit" })))
        })
        .expect("explicit job creates");
    store
        .create_job(image_job(object(json!({ "model": "z_image_turbo" }))))
        .expect("warm model job creates");

    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        explicit_job.id
    );
}

#[test]
fn explicit_gpu_job_beats_younger_warm_auto_match() {
    let store = store("explicit-gpu-before-warm-auto");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: vec!["model-x".to_owned()],
        })
        .expect("worker registers");
    let auto_other = store
        .create_job(image_job(object(json!({ "model": "model-y" }))))
        .expect("auto other job creates");
    let explicit_job = store
        .create_job(CreateJob {
            requested_gpu: "gpu-0".to_owned(),
            ..image_job(object(json!({ "model": "model-y" })))
        })
        .expect("explicit job creates");
    let warm_auto = store
        .create_job(image_job(object(json!({ "model": "model-x" }))))
        .expect("warm auto job creates");

    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, explicit_job.id);
    assert_eq!(
        store
            .get_job(&auto_other.id)
            .expect("auto other job loads")
            .status,
        JobStatus::Queued
    );
    assert_eq!(
        store
            .get_job(&warm_auto.id)
            .expect("warm auto job loads")
            .status,
        JobStatus::Queued
    );
}

#[test]
fn cpu_utility_worker_does_not_claim_gpu_generation_job() {
    let store = store("cpu-utility-no-gpu-jobs");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-cpu".to_owned(),
            gpu_id: "cpu".to_owned(),
            gpu_name: Some("Rust CPU utility worker".to_owned()),
            capabilities: vec![
                WorkerCapability::Cpu,
                WorkerCapability::ModelDownload,
                WorkerCapability::LoraImport,
                WorkerCapability::FrameExtract,
                WorkerCapability::TimelineExport,
            ],
            loaded_models: Vec::new(),
        })
        .expect("worker registers");
    store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");

    assert!(store
        .claim_next_job("worker-cpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn idle_heartbeat_interrupts_previous_active_job() {
    let store = store("idle-heartbeat");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let claimed = store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);

    let worker = store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            loaded_models: Vec::new(),
        })
        .expect("heartbeat succeeds");
    let job = store.get_job(&created.id).expect("job loads");

    assert_eq!(worker.status, WorkerStatus::Idle);
    assert_eq!(worker.current_job_id, None);
    assert_eq!(job.status, JobStatus::Interrupted);
    assert_eq!(job.worker_id, None);
}

#[test]
fn retry_job_is_capped() {
    let store = store("retry-cap");
    let job = store
        .create_job(CreateJob {
            attempts: MAX_JOB_ATTEMPTS,
            ..image_job(Map::new())
        })
        .expect("job creates");

    assert!(matches!(
        store.retry_job(&job.id),
        Err(JobsStoreError::RetryLimit {
            max_attempts: MAX_JOB_ATTEMPTS
        })
    ));
}

#[test]
fn cancel_retry_and_duplicate_preserve_python_metadata_shapes() {
    let store = store("cancel-retry-duplicate");
    let original = store
        .create_job(image_job(object(json!({ "prompt": "mist over hills" }))))
        .expect("job creates");

    let canceled = store.cancel_job(&original.id).expect("job cancels");
    assert_eq!(canceled.status, JobStatus::Canceled);
    assert_eq!(canceled.stage, ProgressStage::Canceled);
    assert!(canceled.cancel_requested);
    assert_eq!(canceled.progress.as_f64(), Some(1.0));
    assert!(canceled.completed_at.is_some());
    assert!(canceled.canceled_at.is_some());

    let retry = store.retry_job(&canceled.id).expect("job retries");
    assert_eq!(retry.source_job_id.as_deref(), Some(canceled.id.as_str()));
    assert_eq!(retry.attempts, canceled.attempts + 1);
    assert_eq!(retry.duplicate_of_job_id, None);
    assert_eq!(retry.payload, canceled.payload);

    let duplicate = store
        .duplicate_job(
            &canceled.id,
            DuplicateJob {
                payload_changes: object(json!({ "prompt": "clear morning", "seed": 42 })),
                requested_gpu: Some("gpu-1".to_owned()),
            },
        )
        .expect("job duplicates");
    assert_eq!(
        duplicate.duplicate_of_job_id.as_deref(),
        Some(canceled.id.as_str())
    );
    assert_eq!(duplicate.source_job_id, None);
    assert_eq!(duplicate.requested_gpu, "gpu-1");
    assert_eq!(duplicate.payload["prompt"], json!("clear morning"));
    assert_eq!(duplicate.payload["seed"], json!(42));
}

#[test]
fn stale_sweep_marks_worker_offline_and_job_interrupted() {
    let store = store("stale-sweep");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    store
        .claim_next_job("worker-1")
        .expect("claim succeeds")
        .expect("job claimed");

    let connection = Connection::open(store.db_path()).expect("db opens");
    connection
        .execute(
            "update workers set last_seen_at = '2000-01-01T00:00:00Z' where id = ?1",
            params!["worker-1"],
        )
        .expect("worker timestamp updates");
    connection
        .execute(
            "update jobs set last_heartbeat_at = '2000-01-01T00:00:00Z' where id = ?1",
            params![created.id],
        )
        .expect("job timestamp updates");

    let sweep = store
        .mark_stale_workers_interrupted(1)
        .expect("sweep succeeds");

    assert_eq!(sweep.workers[0].status, WorkerStatus::Offline);
    assert_eq!(sweep.workers[0].current_job_id, None);
    assert_eq!(sweep.jobs[0].status, JobStatus::Interrupted);
    assert_eq!(sweep.jobs[0].worker_id, None);
}

#[test]
fn json_columns_use_python_compatible_sorted_key_order() {
    let store = store("json-order");
    let job = store
        .create_job(image_job(object(
            json!({ "z": 1, "a": { "b": 2, "a": 1 } }),
        )))
        .expect("job creates");

    let connection = Connection::open(store.db_path()).expect("db opens");
    let payload_json: String = connection
        .query_row(
            "select payload_json from jobs where id = ?1",
            params![job.id],
            |row| row.get(0),
        )
        .expect("payload json loads");

    assert_eq!(payload_json, r#"{"a":{"a":1,"b":2},"z":1}"#);
}

#[test]
fn invalid_progress_numbers_are_rejected() {
    let store = store("invalid-progress");
    let job = store
        .create_job(image_job(Map::new()))
        .expect("job creates");

    assert!(matches!(
        store.update_job_progress(
            &job.id,
            ProgressUpdate {
                status: JobStatus::Running,
                stage: ProgressStage::Running,
                progress: f64::NAN,
                message: "bad progress".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
            },
        ),
        Err(JobsStoreError::InvalidNumber(field)) if field == "progress"
    ));
}

#[test]
fn elapsed_seconds_accepts_fractional_rfc3339_timestamps() {
    let store = store("fractional-time");
    let job = store
        .create_job(image_job(Map::new()))
        .expect("job creates");
    let connection = Connection::open(store.db_path()).expect("db opens");
    connection
        .execute(
            r#"
            update jobs
               set started_at = '2026-05-17T13:00:04.521Z',
                   completed_at = '2026-05-17T13:00:09.999Z'
             where id = ?1
            "#,
            params![job.id.clone()],
        )
        .expect("timestamps update");

    let loaded = store.get_job(&job.id).expect("job loads");

    assert_eq!(
        loaded.elapsed_seconds.and_then(|value| value.as_i64()),
        Some(5)
    );
}
