use std::fs;
use std::path::PathBuf;

use rusqlite::{params, Connection};
use sceneworks_core::contracts::{
    JobStatus, JobType, ProgressStage, WorkerCapability, WorkerStatus, WorkerUtilizationSnapshot,
};
use sceneworks_core::jobs_store::{
    CreateJob, DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker, RetryJob,
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
            utilization: None,
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
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
            },
        )
        .expect("progress updates");
    let worker = store.get_worker("worker-1").expect("worker loads");

    assert_eq!(completed.status, JobStatus::Completed);
    assert_eq!(completed.result, object(json!({ "assetIds": ["asset-1"] })));
    assert_eq!(worker.status, WorkerStatus::Idle);
    assert_eq!(worker.current_job_id, None);
}

/// sc-2086 — successive progress reports must ratchet the per-job peak GPU
/// stats up only, so a stale low sample later in the run can't clobber the
/// max. Also covers clamp-to-100 and the None-passthrough case for status-only
/// updates.
#[test]
fn progress_keeps_running_max_for_peak_gpu_meters() {
    let store = store("peak-gpu-meters");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(object(json!({ "prompt": "p" }))))
        .expect("job creates");
    store.claim_next_job("worker-1").expect("claim ok");

    fn progress(memory: Option<f64>, load: Option<f64>) -> ProgressUpdate {
        ProgressUpdate {
            status: JobStatus::Running,
            stage: ProgressStage::Running,
            progress: 0.5,
            message: "running".to_owned(),
            error: None,
            result: None,
            eta_seconds: None,
            peak_gpu_memory_pct: memory,
            peak_gpu_load_pct: load,
            backend: None,
        }
    }

    let job = store
        .update_job_progress(&created.id, progress(Some(40.0), Some(60.0)))
        .expect("first sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(40.0)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(60.0)
    );

    // Higher samples ratchet up.
    let job = store
        .update_job_progress(&created.id, progress(Some(72.5), Some(85.0)))
        .expect("higher sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(72.5)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(85.0)
    );

    // Lower samples are ignored — peak stays at the previous max.
    let job = store
        .update_job_progress(&created.id, progress(Some(20.0), Some(10.0)))
        .expect("lower sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(72.5)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(85.0)
    );

    // None passes through (status-only update) and leaves peaks untouched.
    let job = store
        .update_job_progress(&created.id, progress(None, None))
        .expect("status-only update");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(72.5)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(85.0)
    );

    // Over-100 samples (rare but possible from buggy backends) clamp.
    let job = store
        .update_job_progress(&created.id, progress(Some(120.0), Some(150.0)))
        .expect("clamped sample");
    assert_eq!(
        job.peak_gpu_memory_pct.as_ref().and_then(|n| n.as_f64()),
        Some(100.0)
    );
    assert_eq!(
        job.peak_gpu_load_pct.as_ref().and_then(|n| n.as_f64()),
        Some(100.0)
    );
}

/// A job whose every progress update omits the peak fields (e.g. a CPU-only
/// utility worker, or a path where gpu_utilization() returned nothing) must
/// keep peak_gpu_memory_pct / peak_gpu_load_pct NULL across the whole
/// lifecycle — otherwise the snapshot diverges from a job that ran on a
/// peerless backend, breaking parity (sc-2086 fix-forward).
#[test]
fn progress_leaves_peaks_null_when_no_samples_arrive() {
    let store = store("peak-null-no-samples");
    register_image_worker(&store);
    let created = store
        .create_job(image_job(object(json!({ "prompt": "p" }))))
        .expect("job creates");
    store.claim_next_job("worker-1").expect("claim ok");

    let progress_no_peaks = ProgressUpdate {
        status: JobStatus::Running,
        stage: ProgressStage::Running,
        progress: 0.5,
        message: "running".to_owned(),
        error: None,
        result: None,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: None,
    };
    for _ in 0..3 {
        store
            .update_job_progress(&created.id, progress_no_peaks.clone())
            .expect("progress update");
    }
    let final_job = store.get_job(&created.id).expect("loads");
    assert!(final_job.peak_gpu_memory_pct.is_none());
    assert!(final_job.peak_gpu_load_pct.is_none());
}

/// sc-2087 — server-side job-title derivation populates the JobSnapshot.title
/// field per the design spec table. Front-end falls back to its own derivation
/// only when this is None, so the queue never displays a raw job id.
#[test]
fn job_snapshot_title_is_derived_from_payload() {
    let store = store("title-derivation");
    register_image_worker(&store);

    fn create(store: &JobsStore, job_type: JobType, payload: Value) -> String {
        store
            .create_job(CreateJob {
                job_type,
                project_id: Some("p".to_owned()),
                project_name: Some("P".to_owned()),
                payload: object(payload),
                requested_gpu: "auto".to_owned(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
            })
            .expect("job creates")
            .id
    }

    let image_id = create(
        &store,
        JobType::ImageGenerate,
        json!({ "prompt": "a sunset over the mountains" }),
    );
    let lora_train_id = create(
        &store,
        JobType::LoraTrain,
        json!({ "loraName": "kelsie-v3" }),
    );
    let caption_id = create(
        &store,
        JobType::TrainingCaption,
        json!({ "datasetName": "kelsie-set" }),
    );
    let video_id = create(
        &store,
        JobType::VideoGenerate,
        json!({ "prompt": "slow push-in on a foggy lighthouse" }),
    );
    let character_id_job = create(
        &store,
        JobType::ImageGenerate,
        json!({ "prompt": "ignored", "characterId": "char-1", "characterName": "Aria" }),
    );
    let lora_import_id = create(
        &store,
        JobType::LoraImport,
        json!({ "loraName": "detail_lora" }),
    );
    let model_download_id = create(
        &store,
        JobType::ModelDownload,
        json!({ "modelName": "Z-Image Turbo" }),
    );
    let prompt_refine_id = create(
        &store,
        JobType::PromptRefine,
        json!({ "prompt": "make it better please" }),
    );
    let unnamed_lora_id = create(&store, JobType::LoraTrain, json!({}));
    let person_detect_id = create(&store, JobType::PersonDetect, json!({}));

    let title = |id: &str| store.get_job(id).expect("loads").title.clone();
    assert_eq!(
        title(&image_id).as_deref(),
        Some("Generate Image — a sunset over the mountains"),
    );
    assert_eq!(
        title(&lora_train_id).as_deref(),
        Some("Training Run — kelsie-v3"),
    );
    assert_eq!(
        title(&caption_id).as_deref(),
        Some("Dataset Captioning — kelsie-set"),
    );
    assert_eq!(
        title(&video_id).as_deref(),
        Some("Generate Video — slow push-in on a foggy lighthouse"),
    );
    assert_eq!(
        title(&character_id_job).as_deref(),
        Some("Character Turnaround — Aria"),
    );
    assert_eq!(
        title(&lora_import_id).as_deref(),
        Some("LoRA Import — detail_lora"),
    );
    assert_eq!(
        title(&model_download_id).as_deref(),
        Some("Model Import — Z-Image Turbo"),
    );
    assert_eq!(
        title(&prompt_refine_id).as_deref(),
        Some("Prompt Refine — make it better please"),
    );
    assert_eq!(
        title(&unnamed_lora_id).as_deref(),
        Some("Training Run — (unnamed LoRA)"),
    );
    // person_detect (and other types without a meaningful subject) intentionally
    // return None so the frontend can fall back to its own derivation.
    assert_eq!(title(&person_detect_id), None);
}

/// Long image-generation prompts are truncated on a word boundary with an
/// ellipsis so the title doesn't blow out the queue row.
#[test]
fn job_snapshot_title_truncates_long_prompts() {
    let store = store("title-truncation");
    register_image_worker(&store);
    // 100 chars of "a " repeating, well over the 80-char cap.
    let long_prompt = "a ".repeat(60);
    let id = store
        .create_job(CreateJob {
            job_type: JobType::ImageGenerate,
            project_id: Some("p".to_owned()),
            project_name: Some("P".to_owned()),
            payload: object(json!({ "prompt": long_prompt })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
        .expect("job creates")
        .id;
    let title = store.get_job(&id).expect("loads").title.unwrap();
    assert!(title.starts_with("Generate Image — "));
    assert!(
        title.ends_with("…"),
        "title should end with ellipsis: {title}"
    );
    assert!(title.len() < 110, "title should be short: {title}");
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
            utilization: None,
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
fn model_convert_can_claim_while_gpu_is_busy() {
    // sc-1629: model_convert is declared non-GPU (NON_GPU_JOB_TYPES) and the
    // worker/UI treat it that way, but the dispatch SQL used to omit it from its
    // non-GPU lists — so a queued model_convert would be gated behind GPU work.
    // It must claim on the CPU lane even while a GPU job is active on the worker.
    let store = store("model-convert-claim");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-1".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![
                WorkerCapability::ImageGenerate,
                WorkerCapability::ModelConvert,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");

    let gpu_job = store
        .create_job(image_job(Map::new()))
        .expect("gpu job creates");
    let convert_job = store
        .create_job(CreateJob {
            job_type: JobType::ModelConvert,
            project_id: None,
            project_name: None,
            payload: object(json!({ "model": "owner/model" })),
            requested_gpu: "auto".to_owned(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
        .expect("convert job creates");

    // First claim takes the GPU job; it is now active on gpu-0.
    assert_eq!(
        store
            .claim_next_job("worker-1")
            .expect("first claim succeeds")
            .expect("first job")
            .id,
        gpu_job.id
    );
    // With a GPU job active only non-GPU work is claimable; model_convert must
    // still claim and land on the CPU lane.
    let second = store
        .claim_next_job("worker-1")
        .expect("second claim succeeds")
        .expect("second job");
    assert_eq!(second.id, convert_job.id);
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
            utilization: None,
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
fn claim_finds_compatible_job_behind_large_incompatible_prefix() {
    // sc-1630: a worker must still claim a compatible job even when far more than the
    // old 50-row query cap of incompatible jobs precede it in the queue — otherwise a
    // specialized/utility worker sits idle behind a long incompatible prefix.
    let store = store("starvation");
    store
        .register_worker(RegisterWorker {
            worker_id: "downloader".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::ModelDownload],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");

    // 60 image jobs the worker cannot run (no ImageGenerate capability), enqueued first
    // so they fill the front of the created_at ordering (well past the old limit 50).
    for index in 0..60 {
        let prompt = format!("incompatible {index}");
        store
            .create_job(image_job(object(json!({ "prompt": prompt }))))
            .expect("incompatible job creates");
    }
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

    let claimed = store
        .claim_next_job("downloader")
        .expect("claim succeeds")
        .expect("compatible job claimed despite the incompatible prefix");
    assert_eq!(claimed.id, download_job.id);
}

#[test]
fn real_lora_train_requires_execute_capability() {
    let store = store("lora-train-execute-routing");
    // A GPU worker that can validate dry-run plans but lacks the inference backend
    // advertises lora_train but not lora_train_execute.
    store
        .register_worker(RegisterWorker {
            worker_id: "dry-only".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    let real = store
        .create_job(lora_train_job("auto", false))
        .expect("real training job creates");

    // The dry-run-only worker must not claim the real job; it stays queued for a
    // backend-capable worker instead of being claimed and failed.
    assert!(store
        .claim_next_job("dry-only")
        .expect("claim succeeds")
        .is_none());

    store
        .register_worker(RegisterWorker {
            worker_id: "trainer".to_owned(),
            gpu_id: "gpu-1".to_owned(),
            gpu_name: None,
            capabilities: vec![
                WorkerCapability::Gpu,
                WorkerCapability::LoraTrain,
                WorkerCapability::LoraTrainExecute,
            ],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    assert_eq!(
        store
            .claim_next_job("trainer")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        real.id
    );
}

#[test]
fn dry_run_lora_train_does_not_require_execute_capability() {
    let store = store("lora-train-dry-routing");
    store
        .register_worker(RegisterWorker {
            worker_id: "dry-only".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: None,
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    let dry = store
        .create_job(lora_train_job("auto", true))
        .expect("dry-run training job creates");

    assert_eq!(
        store
            .claim_next_job("dry-only")
            .expect("claim succeeds")
            .expect("job claimed")
            .id,
        dry.id
    );
}

#[test]
fn training_progress_stages_persist_under_running_and_reject_unknown_status() {
    let store = store("training-progress-stages");
    let job = store
        .create_job(lora_train_job("auto", false))
        .expect("training job creates");

    // The trainer reports caching/training/checkpointing stages under the running
    // status; all must be accepted and persisted, not rejected as invalid.
    for (stage, label) in [
        (ProgressStage::CachingLatents, "caching_latents"),
        (ProgressStage::Training, "training"),
        (ProgressStage::Checkpointing, "checkpointing"),
    ] {
        let updated = store
            .update_job_progress(
                &job.id,
                ProgressUpdate {
                    status: JobStatus::Running,
                    stage,
                    progress: 0.5,
                    message: "training".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                },
            )
            .expect("running status with a training stage is accepted");
        assert_eq!(updated.status, JobStatus::Running);
        assert_eq!(updated.stage.as_str(), label);
    }

    // A non-contract status like "caching" (an earlier kernel bug) must be rejected
    // rather than silently persisted.
    let error = store
        .update_job_progress(
            &job.id,
            ProgressUpdate {
                status: JobStatus::Unknown("caching".to_owned()),
                stage: ProgressStage::CachingLatents,
                progress: 0.5,
                message: "caching".to_owned(),
                error: None,
                result: None,
                eta_seconds: None,
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
            },
        )
        .expect_err("an unknown status is rejected");
    assert!(matches!(error, JobsStoreError::InvalidStatus(_)));
}

#[test]
fn gpu_generation_jobs_reject_cpu_requested_gpu() {
    let store = store("gpu-jobs-reject-cpu");

    let error = store
        .create_job(CreateJob {
            requested_gpu: " CPU ".to_owned(),
            ..image_job(Map::new())
        })
        .expect_err("cpu requestedGpu should be rejected");

    assert!(matches!(error, JobsStoreError::InvalidRequestedGpu(_)));
    assert!(error.to_string().contains("cannot target CPU workers"));
}

#[test]
fn cpu_worker_cannot_claim_auto_gpu_generation_job_even_with_capability() {
    let store = store("cpu-cannot-claim-auto-gpu-job");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-cpu".to_owned(),
            gpu_id: "CPU".to_owned(),
            gpu_name: Some("CPU inference worker".to_owned()),
            capabilities: vec![WorkerCapability::Cpu, WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
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
            utilization: None,
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
            utilization: None,
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
            utilization: None,
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
fn auto_gpu_claim_defers_to_less_loaded_compatible_worker() {
    let store = store("auto-gpu-utilization-preference");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-loaded".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("Loaded GPU".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: Some(WorkerUtilizationSnapshot {
                memory_total_mb: Some(24_000),
                memory_used_mb: Some(22_000),
                memory_free_mb: Some(2_000),
                gpu_load_percent: Some(92.0),
            }),
        })
        .expect("loaded worker registers");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-idle".to_owned(),
            gpu_id: "gpu-1".to_owned(),
            gpu_name: Some("Idle GPU".to_owned()),
            capabilities: vec![WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: Some(WorkerUtilizationSnapshot {
                memory_total_mb: Some(24_000),
                memory_used_mb: Some(4_000),
                memory_free_mb: Some(20_000),
                gpu_load_percent: Some(8.0),
            }),
        })
        .expect("idle worker registers");
    let job = store
        .create_job(image_job(object(json!({ "prompt": "mist" }))))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-loaded")
        .expect("loaded claim succeeds")
        .is_none());
    let claimed = store
        .claim_next_job("worker-idle")
        .expect("idle claim succeeds")
        .expect("idle worker claims");

    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("gpu-1"));
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
            utilization: None,
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
fn idle_heartbeat_interrupts_previous_heartbeated_job() {
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

    // The owning worker reports at least one heartbeat for the job (records
    // last_heartbeat_at), so a later idle heartbeat is a genuine restart and
    // must reclaim the now-orphaned active job.
    store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Busy,
            current_job_id: Some(created.id.clone()),
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("busy heartbeat succeeds");

    let worker = store
        .heartbeat_worker(WorkerHeartbeat {
            worker_id: "worker-1".to_owned(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("heartbeat succeeds");
    let job = store.get_job(&created.id).expect("job loads");

    assert_eq!(worker.status, WorkerStatus::Idle);
    assert_eq!(worker.current_job_id, None);
    assert_eq!(job.status, JobStatus::Interrupted);
    assert_eq!(job.worker_id, None);
}

#[test]
fn idle_heartbeat_does_not_interrupt_just_claimed_job() {
    // A job claimed by one worker incarnation must survive an idle heartbeat
    // (currentJobId=null) that races the claim — e.g. from another process
    // sharing the same worker_id, or a restart firing its first idle heartbeat
    // before any progress is reported. Without a recorded heartbeat there is no
    // evidence the job was abandoned, so it must not be interrupted here.
    let store = store("idle-heartbeat-race");
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
            utilization: None,
        })
        .expect("heartbeat succeeds");
    let job = store.get_job(&created.id).expect("job loads");

    assert_eq!(worker.status, WorkerStatus::Idle);
    assert!(
        matches!(job.status, JobStatus::Preparing),
        "just-claimed job should stay active, got {:?}",
        job.status
    );
    assert_eq!(job.worker_id.as_deref(), Some("worker-1"));
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
        store.retry_job(
            &job.id,
            RetryJob {
                payload_changes: Map::new(),
            },
        ),
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

    let retry = store
        .retry_job(
            &canceled.id,
            RetryJob {
                payload_changes: Map::new(),
            },
        )
        .expect("job retries");
    assert_eq!(retry.source_job_id.as_deref(), Some(canceled.id.as_str()));
    assert_eq!(retry.attempts, canceled.attempts + 1);
    assert_eq!(retry.duplicate_of_job_id, None);
    assert_eq!(retry.payload, canceled.payload);

    let resume_retry = store
        .retry_job(
            &canceled.id,
            RetryJob {
                payload_changes: object(json!({ "downloadAction": "resume" })),
            },
        )
        .expect("resume retry creates");
    assert_eq!(
        resume_retry.source_job_id.as_deref(),
        Some(canceled.id.as_str())
    );
    assert_eq!(resume_retry.payload["prompt"], json!("mist over hills"));
    assert_eq!(resume_retry.payload["downloadAction"], json!("resume"));

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
                peak_gpu_memory_pct: None,
                peak_gpu_load_pct: None,
                backend: None,
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

fn lora_train_job(requested_gpu: &str, dry_run: bool) -> CreateJob {
    CreateJob {
        job_type: JobType::LoraTrain,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(json!({ "dryRun": dry_run, "plan": { "planVersion": 1 } })),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    }
}

#[test]
fn lora_train_rejects_cpu_requested_gpu() {
    let store = store("lora-train-rejects-cpu");

    let error = store
        .create_job(lora_train_job("cpu", true))
        .expect_err("cpu requestedGpu should be rejected for lora_train");

    assert!(matches!(error, JobsStoreError::InvalidRequestedGpu(_)));
    assert!(error.to_string().contains("cannot target CPU workers"));
}

#[test]
fn cpu_worker_cannot_claim_lora_train_even_with_capability() {
    let store = store("cpu-cannot-claim-lora-train");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-cpu".to_owned(),
            gpu_id: "cpu".to_owned(),
            gpu_name: Some("CPU inference worker".to_owned()),
            capabilities: vec![WorkerCapability::Cpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(lora_train_job("auto", true))
        .expect("lora_train job creates");

    assert!(store
        .claim_next_job("worker-cpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn gpu_worker_with_capability_claims_lora_train() {
    let store = store("gpu-claims-lora-train");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-gpu".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("GPU 0".to_owned()),
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::LoraTrain],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    let created = store
        .create_job(lora_train_job("auto", true))
        .expect("lora_train job creates");

    let claimed = store
        .claim_next_job("worker-gpu")
        .expect("claim succeeds")
        .expect("job claimed");

    assert_eq!(claimed.id, created.id);
    assert_eq!(claimed.job_type, JobType::LoraTrain);
    assert_eq!(claimed.status, JobStatus::Preparing);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("gpu-0"));
}

#[test]
fn gpu_worker_without_training_capability_skips_lora_train() {
    let store = store("gpu-without-training-cap");
    store
        .register_worker(RegisterWorker {
            worker_id: "worker-gpu".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            gpu_name: Some("GPU 0".to_owned()),
            capabilities: vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate],
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
    store
        .create_job(lora_train_job("auto", true))
        .expect("lora_train job creates");

    assert!(store
        .claim_next_job("worker-gpu")
        .expect("claim succeeds")
        .is_none());
}

#[test]
fn create_job_with_id_uses_supplied_id() {
    let store = store("create-job-with-id");

    let job = store
        .create_job_with_id(
            "job_lora_train_fixture".to_owned(),
            lora_train_job("auto", true),
        )
        .expect("job creates with supplied id");

    assert_eq!(job.id, "job_lora_train_fixture");
    assert_eq!(
        store
            .get_job("job_lora_train_fixture")
            .expect("job loads")
            .job_type,
        JobType::LoraTrain
    );
}

// --- Epic 3018: MLX-vs-torch image-job routing (sc-3021) ---

fn register_gpu_worker(
    store: &JobsStore,
    worker_id: &str,
    gpu_id: &str,
    capabilities: Vec<WorkerCapability>,
) {
    store
        .register_worker(RegisterWorker {
            worker_id: worker_id.to_owned(),
            gpu_id: gpu_id.to_owned(),
            gpu_name: None,
            capabilities,
            loaded_models: Vec::new(),
            utilization: None,
        })
        .expect("worker registers");
}

fn image_caps() -> Vec<WorkerCapability> {
    vec![WorkerCapability::Gpu, WorkerCapability::ImageGenerate]
}

fn image_job_with(payload: Value, requested_gpu: &str) -> CreateJob {
    CreateJob {
        job_type: JobType::ImageGenerate,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    }
}

#[test]
fn mlx_eligible_image_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-routing-defer");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-eligible job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    // The mlx worker claims it and runs it in-process.
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_worker_excluded_from_torch_only_image_job() {
    let store = store("mlx-routing-exclude");
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // edit_image on a Z-Image model is not a txt2img request → torch path only.
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "z_image_turbo",
                "mode": "edit_image",
                "referenceAssetId": "asset_1"
            }),
            "auto",
        ))
        .expect("job creates");

    // The mlx worker must not claim a torch-only image job.
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    // A torch worker is the home for it.
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn mlx_eligible_image_job_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-routing-fallback");
    // No mlx worker registered (Windows/Linux, or the mlx worker is down).
    register_gpu_worker(&store, "worker-torch", "cuda:0", image_caps());

    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    // With no idle mlx worker, nothing defers — the torch worker is the fallback.
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

#[test]
fn explicit_gpu_image_job_is_not_deferred_to_mlx_worker() {
    let store = store("mlx-routing-explicit-gpu");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // The user explicitly pinned this MLX-eligible job to the torch GPU; honour it.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "z_image_turbo", "prompt": "p" }),
            "mps",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the explicit-gpu job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

// --- Training routing (epic 3039, sc-3043/3049) ---

fn training_caps() -> Vec<WorkerCapability> {
    vec![
        WorkerCapability::Gpu,
        WorkerCapability::LoraTrain,
        WorkerCapability::LoraTrainExecute,
    ]
}

fn mlx_training_job(
    kernel: &str,
    base_model: &str,
    network_type: &str,
    dry_run: bool,
    requested_gpu: &str,
) -> CreateJob {
    CreateJob {
        job_type: JobType::LoraTrain,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(json!({
            "dryRun": dry_run,
            "plan": {
                "planVersion": 1,
                "target": { "kernel": kernel, "baseModel": base_model },
                "config": { "advanced": { "networkType": network_type } }
            }
        })),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    }
}

#[test]
fn mlx_eligible_training_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-training-defer");
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    let job = store
        .create_job(mlx_training_job(
            "z_image_lora",
            "z_image_turbo",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-native training job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    // The mlx worker claims it and trains in-process via mlx-gen.
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_worker_excluded_from_kolors_training_job() {
    let store = store("mlx-training-kolors");
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    // Kolors has no mlx-gen trainer crate → torch path only.
    let job = store
        .create_job(mlx_training_job(
            "kolors_lora",
            "kolors",
            "lora",
            false,
            "auto",
        ))
        .expect("job creates");

    // The mlx worker must not claim a torch-only training job.
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    // A torch worker is the home for it.
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the kolors job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn mlx_worker_excluded_from_lokr_wan_training_job() {
    let store = store("mlx-training-lokr-wan");
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    // LoKr-on-Wan has no Kronecker merge in the mlx Wan path → torch only.
    let job = store
        .create_job(mlx_training_job(
            "wan_moe_lora",
            "wan_2_2_t2v_14b",
            "lokr",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "cuda:0", training_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the LoKr-on-Wan job");
    assert_eq!(claimed.id, job.id);
}

#[test]
fn lokr_z_image_training_stays_mlx_eligible() {
    let store = store("mlx-training-lokr-zimage");
    register_gpu_worker(&store, "worker-torch", "mps", training_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", training_caps());

    // LoKr on Z-Image/SDXL/LTX is fine — the Rust engine applies it natively.
    let job = store
        .create_job(mlx_training_job(
            "z_image_lora",
            "z_image_turbo",
            "lokr",
            false,
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the LoKr Z-Image job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_eligible_training_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-training-fallback");
    // No mlx worker (Windows/Linux, or it's down) — torch is the only path.
    register_gpu_worker(&store, "worker-torch", "cuda:0", training_caps());

    let job = store
        .create_job(mlx_training_job("sdxl_lora", "sdxl", "lora", false, "auto"))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the training job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

// --- Video routing (epic 3018, sc-3036) ---

fn video_caps() -> Vec<WorkerCapability> {
    vec![WorkerCapability::Gpu, WorkerCapability::VideoGenerate]
}

fn video_job_with(payload: Value, requested_gpu: &str) -> CreateJob {
    CreateJob {
        job_type: JobType::VideoGenerate,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: requested_gpu.to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    }
}

#[test]
fn mlx_eligible_video_job_defers_from_torch_worker_to_idle_mlx_worker() {
    let store = store("mlx-video-routing-defer");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2", "mode": "text_to_video", "prompt": "a misty fjord" }),
            "auto",
        ))
        .expect("job creates");

    // The torch worker defers the MLX-eligible video job to the idle mlx worker.
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn mlx_worker_excluded_from_advanced_mode_video_job() {
    let store = store("mlx-video-routing-exclude");
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    // first_last_frame is an advanced mode — torch-only even on a Wan model (MLX
    // covers only text_to_video / image_to_video).
    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2", "mode": "first_last_frame" }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn mlx_eligible_video_job_falls_back_to_torch_when_no_mlx_worker() {
    let store = store("mlx-video-routing-fallback");
    register_gpu_worker(&store, "worker-torch", "cuda:0", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "ltx_2_3", "mode": "text_to_video", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("cuda:0"));
}

#[test]
fn explicit_gpu_video_job_is_not_deferred_to_mlx_worker() {
    let store = store("mlx-video-routing-explicit-gpu");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    let job = store
        .create_job(video_job_with(
            json!({ "model": "wan_2_2", "mode": "text_to_video", "prompt": "p" }),
            "mps",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the explicit-gpu job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn lokr_on_wan_video_stays_on_torch() {
    let store = store("mlx-video-lokr-wan");
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    // LoKr-on-Wan → torch: the diffusers-Wan path applies LoKr via PEFT; the
    // mlx-video path can't (mirrors create_video_adapter).
    let job = store
        .create_job(video_job_with(
            json!({
                "model": "wan_2_2_t2v_14b",
                "mode": "text_to_video",
                "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the LoKr-on-Wan job");
    assert_eq!(claimed.id, job.id);
}

#[test]
fn lokr_on_ltx_video_routes_to_mlx_worker() {
    let store = store("mlx-video-lokr-ltx");
    register_gpu_worker(&store, "worker-torch", "mps", video_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", video_caps());

    // LoKr-on-LTX stays MLX: the torch LTX path has no LoKr loader; the Rust engine
    // applies it natively.
    let job = store
        .create_job(video_job_with(
            json!({
                "model": "ltx_2_3",
                "mode": "text_to_video",
                "loras": [{ "path": "a.safetensors", "networkType": "lokr" }]
            }),
            "auto",
        ))
        .expect("job creates");

    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims the LoKr-on-LTX job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn flux_schnell_txt2img_routes_to_mlx_worker() {
    let store = store("mlx-routing-flux");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // FLUX.1 txt2img (sc-3023) is MLX-eligible → defers to the idle mlx worker.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "flux_schnell", "prompt": "a red fox" }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims flux txt2img");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn flux_reference_job_stays_on_torch() {
    let store = store("mlx-routing-flux-reference");
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // FLUX.1 reference/IP-Adapter stays on the Python torch path → mlx refuses it.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "flux_dev", "prompt": "p", "referenceAssetId": "asset_1" }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());

    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims flux reference job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn qwen_txt2img_routes_to_mlx_but_pose_stays_on_torch() {
    let store = store("mlx-routing-qwen");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // Plain qwen txt2img → MLX worker.
    let txt2img = store
        .create_job(image_job_with(
            json!({ "model": "qwen_image", "prompt": "a red fox" }),
            "auto",
        ))
        .expect("txt2img job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims qwen txt2img");
    assert_eq!(claimed.id, txt2img.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));

    // A strict-pose qwen job stays on the Python torch ControlNet path (sc-2291): the
    // mlx worker refuses it, the torch worker claims it without deferral.
    let pose = store
        .create_job(image_job_with(
            json!({
                "model": "qwen_image",
                "prompt": "a red fox",
                "advanced": { "poses": [{ "id": "p1" }] }
            }),
            "auto",
        ))
        .expect("pose job creates");
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims qwen pose job");
    assert_eq!(claimed.id, pose.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn flux2_klein_variants_route_to_mlx_worker() {
    let store = store("mlx-routing-flux2");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // All three FLUX.2-klein txt2img variants (MLX-only family) route to the mlx worker.
    for model in [
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        let job = store
            .create_job(image_job_with(
                json!({ "model": model, "prompt": "a red fox" }),
                "auto",
            ))
            .unwrap_or_else(|_| panic!("{model} job creates"));
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model} should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .unwrap_or_else(|| panic!("mlx claims {model}"));
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        // Completing the job returns the mlx worker to idle (the deferral only fires
        // toward an *idle* mlx worker), so the next variant defers to it too.
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                },
            )
            .expect("complete job");
    }
}

#[test]
fn flux2_edit_reference_job_routes_to_mlx_worker() {
    let store = store("mlx-routing-flux2-edit");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // FLUX.2 is MLX-only, so an edit/reference job (sc-3029) routes to the mlx worker
    // (sc-3025 kept these on Python; the edit path now exists on Rust).
    let job = store
        .create_job(image_job_with(
            json!({
                "model": "flux2_klein_9b_kv",
                "mode": "edit_image",
                "prompt": "make it golden hour",
                "sourceAssetId": "asset_1"
            }),
            "auto",
        ))
        .expect("job creates");
    assert!(store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .expect("mlx claims flux2 edit job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
}

#[test]
fn sdxl_and_realvisxl_route_to_mlx_worker() {
    let store = store("mlx-routing-sdxl");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    for model in ["sdxl", "realvisxl"] {
        let job = store
            .create_job(image_job_with(
                json!({ "model": model, "prompt": "a red fox" }),
                "auto",
            ))
            .unwrap_or_else(|_| panic!("{model} job creates"));
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model} should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .unwrap_or_else(|| panic!("mlx claims {model}"));
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                },
            )
            .expect("complete job");
    }

    // sc-3060: SDXL reference/IP-Adapter + edit_image (inpaint/outpaint) now run on the Rust
    // engine, so they route to the mlx worker (the torch worker defers).
    for payload in [
        json!({ "model": "sdxl", "prompt": "p", "referenceAssetId": "asset_1" }),
        json!({ "model": "sdxl", "prompt": "p", "mode": "edit_image", "sourceAssetId": "src_1" }),
        json!({ "model": "sdxl", "prompt": "p", "mode": "edit_image",
                "sourceAssetId": "src_1", "maskAssetId": "mask_1" }),
    ] {
        let job = store
            .create_job(image_job_with(payload, "auto"))
            .expect("advanced job creates");
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "sdxl advanced should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .expect("mlx claims sdxl advanced job");
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                },
            )
            .expect("complete job");
    }

    // A third-party LyCORIS LoRA still keeps SDXL on the Python torch path.
    let lycoris = store
        .create_job(image_job_with(
            json!({ "model": "sdxl", "prompt": "p", "loras": [{ "networkType": "lycoris" }] }),
            "auto",
        ))
        .expect("lycoris job creates");
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims sdxl lycoris job");
    assert_eq!(claimed.id, lycoris.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn image_detail_routes_to_mlx_worker() {
    // sc-3060: the tile-ControlNet detail refine (`image_detail`) now runs on the Rust
    // engine for SDXL-family backbones, so it routes to the `mlx` worker (the torch worker
    // defers); a third-party LyCORIS LoRA keeps it on torch.
    let store = store("mlx-routing-detail");
    let caps = vec![
        WorkerCapability::Gpu,
        WorkerCapability::ImageGenerate,
        WorkerCapability::ImageDetail,
    ];
    register_gpu_worker(&store, "worker-torch", "mps", caps.clone());
    register_gpu_worker(&store, "worker-mlx", "mlx", caps);

    let detail_job = |payload: Value| CreateJob {
        job_type: JobType::ImageDetail,
        project_id: Some("project-1".to_owned()),
        project_name: Some("Project 1".to_owned()),
        payload: object(payload),
        requested_gpu: "auto".to_owned(),
        source_job_id: None,
        duplicate_of_job_id: None,
        attempts: 1,
    };

    for model in ["sdxl", "realvisxl"] {
        let job = store
            .create_job(detail_job(
                json!({ "model": model, "sourceAssetId": "asset_src" }),
            ))
            .unwrap_or_else(|_| panic!("{model} detail job creates"));
        assert!(
            store
                .claim_next_job("worker-torch")
                .expect("torch claim ok")
                .is_none(),
            "{model} detail should defer off the torch worker"
        );
        let claimed = store
            .claim_next_job("worker-mlx")
            .expect("mlx claim ok")
            .unwrap_or_else(|| panic!("mlx claims {model} detail"));
        assert_eq!(claimed.id, job.id);
        assert_eq!(claimed.assigned_gpu.as_deref(), Some("mlx"));
        store
            .update_job_progress(
                &claimed.id,
                ProgressUpdate {
                    status: JobStatus::Completed,
                    stage: ProgressStage::Completed,
                    progress: 1.0,
                    message: "done".to_owned(),
                    error: None,
                    result: None,
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: None,
                },
            )
            .expect("complete detail job");
    }

    // LyCORIS detail job stays on the Python torch path.
    let lycoris = store
        .create_job(detail_job(json!({
            "model": "realvisxl",
            "sourceAssetId": "asset_src",
            "loras": [{ "networkType": "lycoris" }]
        })))
        .expect("lycoris detail job creates");
    assert!(store
        .claim_next_job("worker-mlx")
        .expect("mlx claim ok")
        .is_none());
    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims lycoris detail job");
    assert_eq!(claimed.id, lycoris.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}

#[test]
fn non_mlx_model_image_job_is_not_routed_to_mlx_worker() {
    let store = store("mlx-routing-non-mlx-model");
    register_gpu_worker(&store, "worker-torch", "mps", image_caps());
    register_gpu_worker(&store, "worker-mlx", "mlx", image_caps());

    // A torch-only image model with no mlx-gen engine (e.g. kolors — InstantID/Kolors/
    // PuLID/SenseNova have no MLX crate) stays on the Python path: the torch worker
    // claims it without deferral, and the mlx worker would refuse it.
    let job = store
        .create_job(image_job_with(
            json!({ "model": "kolors", "prompt": "p" }),
            "auto",
        ))
        .expect("job creates");

    let claimed = store
        .claim_next_job("worker-torch")
        .expect("torch claim ok")
        .expect("torch claims the non-MLX-model job");
    assert_eq!(claimed.id, job.id);
    assert_eq!(claimed.assigned_gpu.as_deref(), Some("mps"));
}
