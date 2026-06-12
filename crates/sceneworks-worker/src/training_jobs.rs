//! Native MLX LoRA/LoKr training jobs (epic 3039) — the training analog of
//! [`image_jobs`](crate::image_jobs)/[`video_jobs`](crate::video_jobs).
//!
//! Parses a `lora_train` job into the Rust-resolved [`TrainingPlan`], then either
//! validates it (dry run) or maps it onto a [`gen_core::TrainingRequest`] and drives
//! `gen_core::load_trainer(id, &LoadSpec).train(req, on_progress)` — exactly as the
//! image path maps `ImageRequest` → `GenerationRequest` and calls `Generator::generate`.
//! The engine writes the adapter to the plan's `output.outputDir`; the API registers
//! it from the staged `manifestEntry` + the files on disk (apps/rust-api jobs.rs
//! `register_trained_lora`), so the streamed `result` here is informational/UI only.
//!
//! Routing (sc-3049): the API only sends MLX-native families
//! (`z_image_lora`/`sdxl_lora`/`kolors_lora`/`wan_lora`/`wan_moe_lora`/`ltx_mlx_lora`)
//! here (`jobs_store::training_job_is_mlx_eligible`). `kolors_lora` joined the native
//! trainers in sc-4732 (engine trainer sc-4568). `lens` and LoKr-on-Wan stay on the
//! Python torch worker, which also remains the Windows/Linux path + the Mac fallback
//! (the torch Kolors trainer is kept for those paths too). The dry-run validator is
//! cross-platform; the real run is macOS-only
//! (mlx-gen builds Apple MLX) and unreachable elsewhere (the capability is never
//! advertised off macOS).

use super::*;
use sceneworks_core::training::{TrainingPlan, TRAINING_PLAN_VERSION};

// epic 3720 (sc-3724): the backend-neutral training contract types come from `gen_core`.
// Force each trainer-provider crate to link so its `inventory::submit!` trainer
// registration survives linker GC and `load_trainer` can find it. The same crates
// are referenced by image_jobs/video_jobs for generation; re-stating the training
// dependency here keeps it explicit and independent of those modules. `cfg(target_os)`
// decides which backend crates register, not which contract types this module names.
#[cfg(target_os = "macos")]
use gen_core::{
    CancelFlag, LoadSpec, LrSchedule, NetworkType, TrainingConfig, TrainingItem, TrainingOutput,
    TrainingProgress, TrainingRequest, WeightsSource,
};
#[cfg(target_os = "macos")]
use mlx_gen_kolors as _;
#[cfg(target_os = "macos")]
use mlx_gen_ltx as _;
#[cfg(target_os = "macos")]
use mlx_gen_sdxl as _;
#[cfg(target_os = "macos")]
use mlx_gen_wan as _;
#[cfg(target_os = "macos")]
use mlx_gen_z_image as _;

/// Run a `lora_train` job: parse the resolved plan, then either validate it
/// (dry run, the default) or execute real training. Mirrors the Python
/// `run_lora_train_job` split (apps/worker/scene_worker/runtime.py).
pub(crate) async fn run_lora_train_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let plan = parse_plan(&job.payload)?;
    let dry_run = job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    if dry_run {
        run_training_dry_run(api, settings, job, &plan).await
    } else {
        run_training_execution(api, settings, job, &plan).await
    }
}

/// Deserialize the Rust-resolved plan stamped into the job payload at submit time
/// (apps/rust-api training.rs). The plan round-trips through `TrainingPlan`, so a
/// payload missing/garbling it is a hard error (never a silent no-op).
fn parse_plan(payload: &JsonObject) -> WorkerResult<TrainingPlan> {
    let plan = payload.get("plan").ok_or_else(|| {
        WorkerError::InvalidPayload("Training job payload is missing a resolved plan.".to_owned())
    })?;
    serde_json::from_value(plan.clone())
        .map_err(|error| WorkerError::InvalidPayload(format!("Invalid training plan: {error}")))
}

/// Validate a resolved plan the way the Python `validate_training_plan` does:
/// reject an unknown plan version, an empty dataset, or missing dataset images.
/// Shared by the dry-run validator and the real run so both reject the same inputs.
fn validate_training_plan(settings: &Settings, plan: &TrainingPlan) -> WorkerResult<()> {
    if plan.plan_version != TRAINING_PLAN_VERSION {
        return Err(WorkerError::InvalidPayload(format!(
            "Unsupported training plan version {}; this worker understands version {}.",
            plan.plan_version, TRAINING_PLAN_VERSION
        )));
    }
    if plan.dataset.items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Training plan dataset has no items to train on.".to_owned(),
        ));
    }
    normalize_app_managed_model_path(
        settings,
        &plan.target.base_model_path,
        "Training baseModelPath",
    )?;
    resolve_training_output_dir(settings, &plan.output.output_dir, "Training outputDir")?;
    let mut missing = Vec::new();
    for item in &plan.dataset.items {
        let image_path = resolve_dataset_item_path(
            settings,
            &plan.dataset.root_path,
            &item.image_path,
            "Training dataset imagePath",
        )?;
        if !image_path.exists() {
            missing.push(image_path.display().to_string());
        }
    }
    if !missing.is_empty() {
        let preview = missing
            .iter()
            .take(3)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(WorkerError::InvalidPayload(format!(
            "{} dataset image(s) are missing on the worker, e.g. {preview}.",
            missing.len()
        )));
    }
    Ok(())
}

/// Dry-run: validate the plan and report what a real run would produce, with no
/// model load or training (so a GPU worker without the engine still validates).
/// Cross-platform — the validator and summary touch only the plan + the dataset
/// images on disk. Mirrors the Python `_run_lora_train_dry_run`.
async fn run_training_dry_run(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
) -> WorkerResult<()> {
    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Validating training plan.",
            None,
            backend,
        ),
    )
    .await?;
    validate_training_plan(settings, plan)?;
    let item_count = plan.dataset.items.len();
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            &format!("Checked {item_count} dataset item(s)."),
            None,
            backend,
        ),
    )
    .await?;
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Dry run validated {item_count} dataset item(s); training plan is ready."),
            Some(dry_run_summary(settings, plan)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// The dry-run completion summary (keys mirror the Python `dry_run_training_summary`
/// so the Training Studio reads an identical shape regardless of which worker runs it).
fn dry_run_summary(settings: &Settings, plan: &TrainingPlan) -> JsonObject {
    let base_model_installed = normalize_app_managed_model_path(
        settings,
        &plan.target.base_model_path,
        "Training baseModelPath",
    )
    .is_ok_and(|path| path.exists());
    let mut summary = JsonObject::new();
    summary.insert("mode".to_owned(), json!("dry_run"));
    summary.insert("validated".to_owned(), json!(true));
    summary.insert("dryRun".to_owned(), json!(true));
    summary.insert(
        "datasetItemCount".to_owned(),
        json!(plan.dataset.items.len()),
    );
    summary.insert("datasetId".to_owned(), json!(plan.dataset.dataset_id));
    summary.insert(
        "datasetVersion".to_owned(),
        json!(plan.dataset.dataset_version),
    );
    summary.insert("targetId".to_owned(), json!(plan.target.target_id));
    summary.insert("kernel".to_owned(), json!(plan.target.kernel));
    summary.insert("loraId".to_owned(), json!(plan.output.lora_id));
    summary.insert("outputDir".to_owned(), json!(plan.output.output_dir));
    summary.insert("fileName".to_owned(), json!(plan.output.file_name));
    summary.insert("baseModel".to_owned(), json!(plan.target.base_model));
    summary.insert(
        "baseModelRepo".to_owned(),
        json!(plan.target.base_model_repo),
    );
    summary.insert(
        "baseModelPath".to_owned(),
        json!(plan.target.base_model_path),
    );
    summary.insert("baseModelInstalled".to_owned(), json!(base_model_installed));
    summary.insert("planVersion".to_owned(), json!(plan.plan_version));
    summary
}

/// A `lora_train` progress update with the worker's backend label (mirrors
/// `image_jobs::image_progress`). LoRA training keeps `status: running` across the
/// caching/training/checkpointing/saving stages; only the final update is `completed`.
fn training_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

// --------------------------------------------------------------------------- #
// Real training — macOS / Apple-Silicon only (mlx-gen builds Apple MLX). The
// capability is never advertised off macOS, so the non-macOS arm is unreachable.
// --------------------------------------------------------------------------- #

/// Map a resolved plan's `(kernel, baseModel)` onto the mlx-gen trainer registry id
/// (the trainer id matches the generator id of the same base model). Wan splits by
/// the base model variant: the dense TI2V-5B (`wan_lora`) vs the two A14B MoE
/// variants (`wan_moe_lora` + the T2V/I2V base model). `None` for a family with no
/// mlx-gen trainer — those never route here, but the mapping fails loudly if one does.
#[cfg(target_os = "macos")]
fn engine_trainer_id(plan: &TrainingPlan) -> Option<&'static str> {
    match plan.target.kernel.as_str() {
        "z_image_lora" => Some("z_image_turbo"),
        "sdxl_lora" => Some("sdxl"),
        // Kolors is an SDXL U-Net under a ChatGLM3-6B encoder; the engine registers its
        // LoRA/LoKr trainer under the same id as its generator (`"kolors"`), sc-4568.
        "kolors_lora" => Some("kolors"),
        "ltx_mlx_lora" => Some("ltx_2_3"),
        // Dense Wan2.2-TI2V-5B.
        "wan_lora" => Some("wan2_2_ti2v_5b"),
        // A14B dual-expert MoE; the T2V/I2V base model picks the trainer.
        "wan_moe_lora" => match plan.target.base_model.as_str() {
            "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
            "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
            _ => None,
        },
        _ => None,
    }
}

/// Read an `advanced` field as a string, trimmed and non-empty, else `default`.
#[cfg(target_os = "macos")]
fn advanced_str(advanced: &JsonObject, key: &str, default: &str) -> String {
    advanced
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

/// Read an `advanced` field as an f32, else `default`.
#[cfg(target_os = "macos")]
fn advanced_f32(advanced: &JsonObject, key: &str, default: f32) -> f32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

/// Read an `advanced` field as a u32, else `default`.
#[cfg(target_os = "macos")]
fn advanced_u32(advanced: &JsonObject, key: &str, default: u32) -> u32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as u32)
        .unwrap_or(default)
}

/// Read an `advanced` field as a bool (accepting a JSON bool or a `"true"`/`"false"`
/// string), else `default`.
#[cfg(target_os = "macos")]
fn advanced_bool(advanced: &JsonObject, key: &str, default: bool) -> bool {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_bool()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .unwrap_or(default)
}

/// Normalize the advanced `mixedPrecision` string onto the engine's `train_dtype`
/// domain, which is exactly `{"bf16", "f32"}` (sc-4887). Only an explicit `"bf16"`
/// (case-insensitive) selects bf16; every other value — `"fp16"`, `"no"`, empty,
/// anything unrecognized — falls back to full-precision `"f32"`, matching the
/// engine's own "unrecognized ⇒ f32" rule. The *absent*-key default is applied by
/// the caller (`"bf16"`), so a plan that omits the key keeps the OOM fix on.
#[cfg(target_os = "macos")]
fn normalize_train_dtype(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "bf16" => "bf16".to_owned(),
        _ => "f32".to_owned(),
    }
}

/// Map the SceneWorks `TrainingConfig` (plan `config` + its free-form `advanced`
/// bag) onto the engine's typed [`gen_core::TrainingConfig`]. The optimizer string is
/// passed verbatim — the engine normalizes aliases (`adamw8bit`→`adamw`,
/// `prodigyopt`→`prodigy`). An empty `loraTargetModules` lets the family trainer use
/// its default target set.
#[cfg(target_os = "macos")]
fn map_training_config(config: &sceneworks_core::training::TrainingConfig) -> TrainingConfig {
    let advanced = &config.advanced;
    let lora_target_modules = advanced
        .get("loraTargetModules")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    TrainingConfig {
        rank: config.rank,
        alpha: config.alpha as f32,
        learning_rate: config.learning_rate.as_f64().unwrap_or(1e-4) as f32,
        steps: config.steps,
        batch_size: config.batch_size,
        gradient_accumulation: config.gradient_accumulation,
        resolution: config.resolution,
        save_every: config.save_every,
        seed: config.seed.max(0) as u64,
        optimizer: config.optimizer.clone(),
        weight_decay: advanced_f32(advanced, "weightDecay", 0.0),
        lr_scheduler: LrSchedule::parse(&advanced_str(advanced, "lrScheduler", "constant")),
        lr_warmup_steps: advanced_u32(advanced, "lrWarmupSteps", 0),
        network_type: NetworkType::parse(&advanced_str(advanced, "networkType", "lora")),
        decompose_factor: advanced
            .get("decomposeFactor")
            .and_then(Value::as_i64)
            .map(|value| value as i32)
            .unwrap_or(-1),
        lora_target_modules,
        timestep_type: advanced_str(advanced, "timestepType", "sigmoid"),
        timestep_bias: advanced_str(advanced, "timestepBias", "balanced"),
        loss_type: advanced_str(advanced, "lossType", "mse"),
        // Training compute dtype — the primary OOM fix (sc-4887). bf16 halves the
        // activation working set and drops the 1024² z-image first-step peak 135 → ~44 GB,
        // so the run survives. This mapping builds the engine config field-by-field (no
        // `..Default::default()`), so the engine's "bf16" default never reaches here — it
        // MUST be set explicitly. Sourced from the advanced `mixedPrecision` key (presets
        // already carry "bf16"); the engine only supports bf16/f32, so anything else
        // (incl. "fp16", "no", empty) normalizes to "f32". Absent → "bf16" (keep the fix on).
        train_dtype: normalize_train_dtype(&advanced_str(advanced, "mixedPrecision", "bf16")),
        // Honor the "Gradient Checkpointing" UI checkbox on the Rust path (sc-4881) — an
        // extra lever for smaller machines / higher resolution on top of the bf16 fix.
        // Previously dropped here, so the engine always ran at its `false` default. Absent
        // (legacy payloads) preserves that default.
        gradient_checkpointing: advanced_bool(advanced, "gradientCheckpointing", false),
        trigger_word: config.trigger_word.clone(),
    }
}

/// One progress event streamed from the blocking training thread to the async side.
#[cfg(target_os = "macos")]
enum TrainEvent {
    Progress(TrainingProgress),
    Done(TrainingOutput),
}

/// Execute a real training run on the in-process mlx-gen engine. Loads the (frozen)
/// base model via a [`LoadSpec`] (exactly as inference's `load_engine`), runs the
/// family trainer on a blocking thread, streams staged progress, honors cancellation
/// via the engine's [`CancelFlag`], and reports the produced adapter. The adapter is
/// written by the engine into the plan's `output.outputDir`; the API registers it
/// from the staged manifest entry.
#[cfg(target_os = "macos")]
async fn run_training_execution(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
) -> WorkerResult<()> {
    let backend = backend_label(&settings.gpu_id);
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        training_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing LoRA training.",
            None,
            backend,
        ),
    )
    .await?;

    validate_training_plan(settings, plan)?;

    let engine_id = engine_trainer_id(plan).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "No MLX trainer for kernel '{}' (base model '{}').",
            plan.target.kernel, plan.target.base_model
        ))
    })?;

    let weights_dir = resolve_app_managed_model_dir(
        settings,
        &plan.target.base_model_path,
        "Training baseModelPath",
    )?;

    let output_dir =
        resolve_training_output_dir(settings, &plan.output.output_dir, "Training outputDir")?;
    tokio::fs::create_dir_all(&output_dir).await?;

    let items: Vec<TrainingItem> = plan
        .dataset
        .items
        .iter()
        .map(|item| {
            Ok(TrainingItem {
                image_path: resolve_dataset_item_path(
                    settings,
                    &plan.dataset.root_path,
                    &item.image_path,
                    "Training dataset imagePath",
                )?,
                caption: item.caption.clone(),
            })
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    let config = map_training_config(&plan.config);
    let total_steps = config.steps;
    let file_name = plan.output.file_name.clone();
    let trigger_words = plan.output.trigger_words.clone();

    check_cancel(api, &job.id, "LoRA training canceled before it started.").await?;

    let cancel = CancelFlag::new();
    let (tx, rx) = tokio::sync::mpsc::channel::<TrainEvent>(64);

    let blocking = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> WorkerResult<()> {
            let mut trainer =
                gen_core::load_trainer(engine_id, &LoadSpec::new(WeightsSource::Dir(weights_dir)))
                    .map_err(|error| {
                        WorkerError::Engine(format!("{engine_id} trainer load failed: {error}"))
                    })?;
            let request = TrainingRequest {
                items,
                config,
                output_dir,
                file_name,
                trigger_words,
                cancel,
            };
            trainer.validate(&request).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "{engine_id} trainer rejected the plan: {error}"
                ))
            })?;
            let mut on_progress = |progress: TrainingProgress| {
                let _ = tx.blocking_send(TrainEvent::Progress(progress));
            };
            let output = trainer
                .train(&request, &mut on_progress)
                .map_err(|error| WorkerError::Engine(format!("training failed: {error}")))?;
            let _ = tx.blocking_send(TrainEvent::Done(output));
            Ok(())
        })
    };

    consume_training_events(
        api,
        settings,
        job,
        plan,
        backend,
        total_steps,
        rx,
        cancel,
        blocking,
    )
    .await
}

/// Consume training events from the blocking thread: stream staged progress, poll
/// cancel ~every 2s (draining after a cancel so the blocking sender never blocks),
/// and on the final `Done` event report completion with the result the UI shows.
/// Mirrors `image_jobs::consume_gen_events`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn consume_training_events(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &TrainingPlan,
    backend: &str,
    total_steps: u32,
    mut rx: tokio::sync::mpsc::Receiver<TrainEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
) -> WorkerResult<()> {
    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    let mut checkpoints: Vec<Value> = Vec::new();
    while let Some(event) = rx.recv().await {
        if canceled {
            continue; // drain remaining events so the blocking sender never blocks.
        }
        match event {
            TrainEvent::Progress(progress) => {
                // Poll cancel on the long training band only (cheap stages fly by).
                if matches!(progress, TrainingProgress::Training { .. })
                    && last_cancel_check.elapsed() >= Duration::from_secs(2)
                {
                    last_cancel_check = Instant::now();
                    if cancel_requested(api, &job.id, "LoRA training canceled by user.").await {
                        cancel.cancel();
                        canceled = true;
                        continue;
                    }
                }
                if let TrainingProgress::Checkpoint { step } = progress {
                    checkpoints.push(json!({ "step": step }));
                }
                let (status, stage, fraction, message) =
                    map_training_progress(progress, total_steps);
                update_job(
                    api,
                    &job.id,
                    training_progress(status, stage, fraction, &message, None, backend),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
            TrainEvent::Done(output) => {
                let result = training_result(plan, &output, &checkpoints);
                update_job(
                    api,
                    &job.id,
                    training_progress(
                        JobStatus::Completed,
                        ProgressStage::Completed,
                        1.0,
                        &format!("Trained LoRA saved as {}.", plan.output.file_name),
                        Some(result),
                        backend,
                    ),
                )
                .await?;
            }
        }
    }

    let task_result = blocking
        .await
        .map_err(|error| task_join_error("training task join", error))?;
    if canceled {
        // check_cancel already posted the Canceled update; treat the engine's
        // early return as the clean cancel.
        return Err(WorkerError::Canceled(
            "LoRA training canceled by user.".to_owned(),
        ));
    }
    task_result
}

/// Map an engine [`TrainingProgress`] event onto a job `(status, stage, fraction,
/// message)`. The fractions follow the kernel's bands (prepare 0–.08, load .08–.18,
/// cache .18–.32, train/checkpoint .32–.92, save .92–1.0) so the UI's existing
/// `caching_latents`/`training`/`checkpointing`/`saving` stages light up unchanged.
#[cfg(target_os = "macos")]
fn map_training_progress(
    progress: TrainingProgress,
    total_steps: u32,
) -> (JobStatus, ProgressStage, f64, String) {
    match progress {
        TrainingProgress::Preparing => (
            JobStatus::Running,
            ProgressStage::Preparing,
            0.06,
            "Preparing dataset.".to_owned(),
        ),
        TrainingProgress::LoadingModel => (
            JobStatus::Running,
            ProgressStage::LoadingModel,
            0.12,
            "Loading base model.".to_owned(),
        ),
        TrainingProgress::Caching { current, total } => {
            let span = if total == 0 {
                0.0
            } else {
                0.14 * (current as f64 / total as f64)
            };
            (
                JobStatus::Running,
                ProgressStage::CachingLatents,
                0.18 + span,
                format!("Caching dataset latents ({current}/{total})."),
            )
        }
        TrainingProgress::Training { step, total, loss } => (
            JobStatus::Running,
            ProgressStage::Training,
            train_fraction(step, total),
            format!("Training step {step} of {total} (loss {loss:.4})."),
        ),
        TrainingProgress::Checkpoint { step } => (
            JobStatus::Running,
            ProgressStage::Checkpointing,
            train_fraction(step, total_steps.max(step)),
            format!("Saved checkpoint at step {step}."),
        ),
        TrainingProgress::Saving => (
            JobStatus::Running,
            ProgressStage::Saving,
            0.94,
            "Saving adapter.".to_owned(),
        ),
    }
}

/// Scale a training micro-step into the 0.32–0.92 training band.
#[cfg(target_os = "macos")]
fn train_fraction(step: u32, total: u32) -> f64 {
    if total == 0 {
        return 0.32;
    }
    0.32 + 0.60 * (step as f64 / total as f64).clamp(0.0, 1.0)
}

/// Build the completion `result` the Training Studio reads (keys mirror the Python
/// trainer's `_result`). LoRA registration is driven by the staged `manifestEntry` +
/// the on-disk adapter (apps/rust-api `register_trained_lora`), not this result, so
/// this is informational/UI metadata.
#[cfg(target_os = "macos")]
fn training_result(
    plan: &TrainingPlan,
    output: &TrainingOutput,
    checkpoints: &[Value],
) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("mode".to_owned(), json!("train"));
    result.insert("kernel".to_owned(), json!(plan.target.kernel));
    result.insert("loraId".to_owned(), json!(plan.output.lora_id));
    result.insert("outputDir".to_owned(), json!(plan.output.output_dir));
    result.insert("fileName".to_owned(), json!(plan.output.file_name));
    result.insert(
        "outputPath".to_owned(),
        json!(output.adapter_path.display().to_string()),
    );
    result.insert("format".to_owned(), json!(plan.output.format));
    result.insert("datasetId".to_owned(), json!(plan.dataset.dataset_id));
    result.insert(
        "datasetVersion".to_owned(),
        json!(plan.dataset.dataset_version),
    );
    result.insert(
        "datasetItemCount".to_owned(),
        json!(plan.dataset.items.len()),
    );
    result.insert("targetId".to_owned(), json!(plan.target.target_id));
    result.insert("baseModel".to_owned(), json!(plan.target.base_model));
    result.insert("steps".to_owned(), json!(plan.config.steps));
    result.insert("stepsCompleted".to_owned(), json!(output.steps));
    result.insert("finalLoss".to_owned(), json!(output.final_loss));
    result.insert("checkpoints".to_owned(), json!(checkpoints));
    result.insert("rank".to_owned(), json!(plan.config.rank));
    result.insert("alpha".to_owned(), json!(plan.config.alpha));
    result.insert("resolution".to_owned(), json!(plan.config.resolution));
    result.insert("triggerWords".to_owned(), json!(plan.output.trigger_words));
    result.insert("planVersion".to_owned(), json!(plan.plan_version));
    result.insert("backend".to_owned(), json!("mlx"));
    result
}

/// Off macOS the mlx-gen engine is not linked; the `lora_train_execute` capability is
/// never advertised, so a real run can never be claimed here. Fail loudly if one is.
#[cfg(not(target_os = "macos"))]
async fn run_training_execution(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
    _plan: &TrainingPlan,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Native MLX LoRA training requires macOS (mlx-gen); this worker cannot execute it."
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes every test that mutates the process-global `HF_HUB_CACHE` env
    /// var. Must be a SINGLE module-level mutex: per-function `static`s are
    /// distinct locks and do not serialize against each other, so under parallel
    /// test execution one test's `set_var`/`remove_var` races another's
    /// validation read (the intermittent "must be inside an app-managed
    /// directory" flake on the macOS nax-worker CI lane).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn test_settings(data_dir: &Path) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir: data_dir.to_path_buf(),
            config_dir: data_dir.join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
        }
    }

    /// A complete resolved plan as the API serializes it, parameterized by the
    /// fields the worker glue reads. `baseModelPath` is a path that does not exist,
    /// so `baseModelInstalled` is false unless a test overrides it.
    fn plan_json(
        data_dir: &Path,
        kernel: &str,
        base_model: &str,
        network_type: &str,
        image_paths: &[&str],
    ) -> Value {
        let dataset_root = data_dir.join("datasets").join("ds-1");
        let items: Vec<Value> = image_paths
            .iter()
            .map(|path| json!({ "imagePath": path, "caption": "a photo of mychar" }))
            .collect();
        json!({
            "schemaVersion": 1,
            "planVersion": 1,
            "jobId": "job-1",
            "target": {
                "targetId": format!("{kernel}_target"),
                "kernel": kernel,
                "family": "test",
                "modality": "image",
                "outputKind": "lora",
                "baseModel": base_model,
                "baseModelPath": data_dir.join("models").join("base-missing").display().to_string()
            },
            "dataset": {
                "datasetId": "ds-1",
                "datasetVersion": 1,
                "rootPath": dataset_root.display().to_string(),
                "items": items
            },
            "config": {
                "rank": 16,
                "alpha": 32,
                "learningRate": 0.0001,
                "steps": 1000,
                "batchSize": 1,
                "gradientAccumulation": 2,
                "resolution": 1024,
                "saveEvery": 250,
                "seed": 42,
                "optimizer": "adamw8bit",
                "advanced": {
                    "networkType": network_type,
                    "lrScheduler": "cosine",
                    "lrWarmupSteps": 50,
                    "weightDecay": 0.01,
                    "decomposeFactor": 8,
                    "loraTargetModules": ["to_q", "to_k"],
                    "timestepType": "sigmoid",
                    "timestepBias": "high_noise",
                    "lossType": "mse"
                }
            },
            "output": {
                "loraId": "lora-1",
                "outputDir": data_dir.join("loras").join("lora-1").display().to_string(),
                "fileName": "lora.safetensors",
                "format": "safetensors",
                "triggerWords": ["mychar"]
            },
            "provenance": {
                "datasetId": "ds-1",
                "datasetVersion": 1,
                "targetId": format!("{kernel}_target"),
                "baseModel": base_model,
                "configSnapshot": {},
                "outputLoraId": "lora-1",
                "sourceJobId": "job-1",
                "createdAt": "2026-06-06T00:00:00Z"
            }
        })
    }

    fn parse(value: Value) -> TrainingPlan {
        serde_json::from_value(value).expect("plan deserializes")
    }

    /// sc-4887: only an explicit bf16 selects bf16; every other value (incl. the
    /// engine-unsupported fp16, "no", empty) falls back to full-precision f32.
    #[cfg(target_os = "macos")]
    #[test]
    fn normalize_train_dtype_only_bf16_selects_bf16() {
        assert_eq!(normalize_train_dtype("bf16"), "bf16");
        assert_eq!(normalize_train_dtype("BF16"), "bf16");
        assert_eq!(normalize_train_dtype("  bf16 "), "bf16");
        assert_eq!(normalize_train_dtype("fp16"), "f32");
        assert_eq!(normalize_train_dtype("no"), "f32");
        assert_eq!(normalize_train_dtype(""), "f32");
    }

    /// sc-4881 / sc-4887: the two OOM-fix levers must reach the engine config. The
    /// mapping builds it field-by-field (no `..Default::default()`), so a dropped
    /// field silently reverts to the wrong value — exactly the bug this story fixes.
    #[cfg(target_os = "macos")]
    #[test]
    fn map_training_config_wires_train_dtype_and_gradient_checkpointing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let image = image.display().to_string();

        // Absent keys: the OOM fix stays on (bf16) and checkpointing stays off, so a
        // legacy plan that omits both still trains under the safe default.
        let default_plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image],
        ));
        let mapped = map_training_config(&default_plan.config);
        assert_eq!(mapped.train_dtype, "bf16");
        assert!(!mapped.gradient_checkpointing);

        // Explicit values flow through; a non-bf16 precision normalizes to f32.
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image],
        );
        value["config"]["advanced"]["mixedPrecision"] = json!("fp16");
        value["config"]["advanced"]["gradientCheckpointing"] = json!(true);
        let mapped = map_training_config(&parse(value).config);
        assert_eq!(mapped.train_dtype, "f32");
        assert!(mapped.gradient_checkpointing);
    }

    #[test]
    fn validate_rejects_unknown_plan_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["planVersion"] = json!(999);
        let error = validate_training_plan(&settings, &parse(value))
            .expect_err("version mismatch rejected");
        assert!(error
            .to_string()
            .contains("Unsupported training plan version"));
    }

    #[test]
    fn validate_rejects_empty_dataset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[],
        ));
        let error = validate_training_plan(&settings, &plan).expect_err("empty dataset rejected");
        assert!(error.to_string().contains("no items"));
    }

    #[test]
    fn validate_rejects_missing_image() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("missing.png");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        ));
        let error = validate_training_plan(&settings, &plan).expect_err("missing image rejected");
        assert!(error.to_string().contains("missing on the worker"));
    }

    #[test]
    fn validate_accepts_present_images() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        ));
        assert!(validate_training_plan(&settings, &plan).is_ok());
    }

    #[test]
    fn validate_rejects_image_outside_dataset_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("other").join("image.png");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        ));
        let error = validate_training_plan(&settings, &plan).expect_err("outside image rejected");
        assert!(error.to_string().contains("Training dataset imagePath"));
    }

    #[test]
    fn validate_rejects_base_model_outside_data_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("image.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["target"]["baseModelPath"] = json!("/tmp/sceneworks-outside-base");
        let error = validate_training_plan(&settings, &parse(value))
            .expect_err("outside base model rejected");
        assert!(error.to_string().contains("Training baseModelPath"));
    }

    /// The base model is a read-only weights source the rust-api resolves from the
    /// shared Hugging Face hub cache, which the desktop points outside `data_dir`
    /// via `HF_HOME`. Such a path must be accepted even though it is not under the
    /// app data dir (regression for the z_image_turbo "must be inside an
    /// app-managed directory" training failure). Serialized so the `HF_HUB_CACHE`
    /// mutation can't race other env-reading tests.
    #[test]
    fn validate_accepts_base_model_in_hf_cache_outside_data_dir() {
        // Recover from poisoning: a panic in another env-mutating test must not
        // cascade into a spurious failure here — we only need the mutual
        // exclusion, not the guarded data.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let dir = tempfile::tempdir().expect("tempdir");
        let hf_cache = tempfile::tempdir().expect("hf cache tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");

        // The base model lives under the HF hub cache (outside data_dir), exactly
        // as `resolve_base_model_path` returns for an HF-cache-resident model.
        let base_model = hf_cache
            .path()
            .join("models--Tongyi-MAI--Z-Image-Turbo")
            .join("snapshots")
            .join("abc123");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["target"]["baseModelPath"] = json!(base_model.display().to_string());

        let prior = std::env::var("HF_HUB_CACHE").ok();
        std::env::set_var("HF_HUB_CACHE", hf_cache.path());
        let result = validate_training_plan(&settings, &parse(value));
        match prior {
            Some(value) => std::env::set_var("HF_HUB_CACHE", value),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }

        assert!(
            result.is_ok(),
            "HF-cache base model should validate: {result:?}"
        );
    }

    /// The REAL training run (`run_training_execution`) resolves the base model
    /// weights via `resolve_app_managed_model_dir`, a path separate from the
    /// dry-run validator. It must also accept an HF-cache-resident model dir, or
    /// the dry run passes while the real run fails with "must be inside an
    /// app-managed directory" (the z_image_turbo regression: dry run completed,
    /// real run rejected the same `~/.cache/huggingface` snapshot).
    #[test]
    fn resolve_app_managed_model_dir_accepts_hf_cache_snapshot() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let dir = tempfile::tempdir().expect("tempdir");
        let hf_cache = tempfile::tempdir().expect("hf cache tempdir");
        let settings = test_settings(dir.path());

        // A real, existing snapshot dir under the HF hub cache (outside data_dir),
        // exactly what `resolve_base_model_path` hands the worker.
        let snapshot = hf_cache
            .path()
            .join("models--Tongyi-MAI--Z-Image-Turbo")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snapshot).expect("snapshot dir");

        let prior = std::env::var("HF_HUB_CACHE").ok();
        std::env::set_var("HF_HUB_CACHE", hf_cache.path());
        let resolved = resolve_app_managed_model_dir(
            &settings,
            &snapshot.display().to_string(),
            "Training baseModelPath",
        );
        match prior {
            Some(value) => std::env::set_var("HF_HUB_CACHE", value),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }

        assert!(
            resolved.is_ok(),
            "HF-cache model dir should resolve for the real run: {resolved:?}"
        );
    }

    #[test]
    fn validate_rejects_output_dir_outside_app_lora_or_model_roots() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image = dir.path().join("datasets").join("ds-1").join("image.png");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["output"]["outputDir"] =
            json!(dir.path().join("tmp").join("lora-1").display().to_string());
        let error = validate_training_plan(&settings, &parse(value))
            .expect_err("outside output dir rejected");
        assert!(error.to_string().contains("Training outputDir"));
    }

    /// Project-scoped training (the default) writes to the owning project's tree,
    /// `<data>/projects/<slug>.sceneworks/loras/<lora_id>`, which is under the app
    /// data dir but not under `<data>/loras` or `<data>/models`. The worker must
    /// accept it (regression for the "Training outputDir must be inside an
    /// app-managed directory" failure on project-scoped runs).
    #[test]
    fn validate_accepts_project_scoped_output_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        std::fs::create_dir_all(&dataset_root).expect("dataset root");
        let image = dataset_root.join("image.png");
        std::fs::write(&image, b"png").expect("image");
        let mut value = plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lora",
            &[&image.display().to_string()],
        );
        value["output"]["outputDir"] = json!(dir
            .path()
            .join("projects")
            .join("my-character.sceneworks")
            .join("loras")
            .join("lora-1")
            .display()
            .to_string());
        assert!(
            validate_training_plan(&settings, &parse(value)).is_ok(),
            "project-scoped output dir should validate"
        );
    }

    #[test]
    fn dry_run_summary_carries_plan_facts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let image_a = dir.path().join("datasets").join("ds-1").join("a.png");
        let image_b = dir.path().join("datasets").join("ds-1").join("b.png");
        let plan = parse(plan_json(
            dir.path(),
            "sdxl_lora",
            "sdxl",
            "lora",
            &[
                &image_a.display().to_string(),
                &image_b.display().to_string(),
            ],
        ));
        let summary = dry_run_summary(&settings, &plan);
        assert_eq!(summary.get("mode").unwrap(), "dry_run");
        assert_eq!(summary.get("kernel").unwrap(), "sdxl_lora");
        assert_eq!(summary.get("datasetItemCount").unwrap(), 2);
        assert_eq!(summary.get("loraId").unwrap(), "lora-1");
        assert_eq!(summary.get("fileName").unwrap(), "lora.safetensors");
        // The placeholder base path does not exist.
        assert_eq!(summary.get("baseModelInstalled").unwrap(), false);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn engine_trainer_id_maps_mlx_native_families_and_rejects_the_rest() {
        let cases: &[(&str, &str, Option<&str>)] = &[
            ("z_image_lora", "z_image_turbo", Some("z_image_turbo")),
            ("sdxl_lora", "sdxl", Some("sdxl")),
            ("ltx_mlx_lora", "ltx_2_3", Some("ltx_2_3")),
            ("wan_lora", "wan_2_2", Some("wan2_2_ti2v_5b")),
            ("wan_moe_lora", "wan_2_2_t2v_14b", Some("wan2_2_t2v_14b")),
            ("wan_moe_lora", "wan_2_2_i2v_14b", Some("wan2_2_i2v_14b")),
            // Kolors gained a native mlx-gen trainer (sc-4568) and now routes here (sc-4732);
            // the trainer registers under the generator id `"kolors"`.
            ("kolors_lora", "kolors", Some("kolors")),
            // Lens (sidecar) has no mlx-gen trainer crate — never routes here, maps to None.
            ("lens_lora", "lens", None),
            // Unknown A14B base model variant.
            ("wan_moe_lora", "wan_2_2_mystery", None),
        ];
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        for (kernel, base_model, expected) in cases {
            let plan = parse(plan_json(
                dir.path(),
                kernel,
                base_model,
                "lora",
                &[&image.display().to_string()],
            ));
            assert_eq!(
                engine_trainer_id(&plan),
                *expected,
                "kernel={kernel} base_model={base_model}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn map_training_config_reads_advanced_and_passes_optimizer_verbatim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let plan = parse(plan_json(
            dir.path(),
            "z_image_lora",
            "z_image_turbo",
            "lokr",
            &[&image.display().to_string()],
        ));
        let cfg = map_training_config(&plan.config);
        assert_eq!(cfg.rank, 16);
        assert_eq!(cfg.alpha as u32, 32);
        assert_eq!(cfg.steps, 1000);
        assert_eq!(cfg.gradient_accumulation, 2);
        assert_eq!(cfg.seed, 42);
        // The optimizer alias is passed verbatim; the engine normalizes it.
        assert_eq!(cfg.optimizer, "adamw8bit");
        assert!((cfg.weight_decay - 0.01).abs() < 1e-6);
        assert!((cfg.learning_rate - 0.0001).abs() < 1e-6);
        assert_eq!(cfg.lr_warmup_steps, 50);
        assert_eq!(cfg.decompose_factor, 8);
        assert!(matches!(cfg.network_type, NetworkType::Lokr));
        assert!(matches!(cfg.lr_scheduler, LrSchedule::Cosine));
        assert_eq!(
            cfg.lora_target_modules,
            vec!["to_q".to_owned(), "to_k".to_owned()]
        );
        assert_eq!(cfg.timestep_bias, "high_noise");
        assert_eq!(cfg.trigger_word, None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn map_training_config_defaults_when_advanced_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("datasets").join("ds-1").join("x.png");
        let mut value = plan_json(
            dir.path(),
            "sdxl_lora",
            "sdxl",
            "lora",
            &[&image.display().to_string()],
        );
        value["config"]["advanced"] = json!({});
        let plan = parse(value);
        let cfg = map_training_config(&plan.config);
        assert!(matches!(cfg.network_type, NetworkType::Lora));
        assert!(matches!(cfg.lr_scheduler, LrSchedule::Constant));
        assert_eq!(cfg.lr_warmup_steps, 0);
        assert_eq!(cfg.decompose_factor, -1);
        assert!(cfg.lora_target_modules.is_empty());
        assert_eq!(cfg.timestep_type, "sigmoid");
        assert_eq!(cfg.loss_type, "mse");
    }

    /// Real-weights smoke (sc-4732 + sc-4764): load the Kolors trainer from the installed
    /// `Kwai-Kolors/Kolors-diffusers` snapshot and run two LoRA micro-steps on a one-image dataset.
    /// Proves the worker links `mlx-gen-kolors` (the `load_trainer("kolors", …)` registration), the
    /// snapshot's overlaid `tokenizer/tokenizer.json` (sc-4764) lets the trainer construct, and a
    /// real step runs (finite loss) + writes an adapter on the Mac GPU. The trainer loads the base
    /// at **f32** (engine choice for clean autograd), so this is memory-heavy. Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored kolors_real_weights_trains --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Kolors weights (+ tokenizer.json overlay) + Metal device; f32 base is heavy"]
    fn kolors_real_weights_trains_a_lora_step() {
        let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME set"));
        let snapshot = std::fs::read_dir(
            home.join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots"),
        )
        .expect("kolors snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a kolors snapshot dir");
        assert!(
            snapshot.join("tokenizer").join("tokenizer.json").exists(),
            "kolors snapshot is missing the overlaid tokenizer.json (sc-4764)"
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(512, 512, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 2,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: false,
            train_dtype: "bf16".to_owned(),
            resolution: 512,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".to_owned(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::parse("constant"),
            lr_warmup_steps: 0,
            network_type: NetworkType::parse("lora"),
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_owned(),
            timestep_bias: "balanced".to_owned(),
            loss_type: "mse".to_owned(),
            trigger_word: None,
        };
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: "kolors_smoke.safetensors".to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer =
            gen_core::load_trainer("kolors", &LoadSpec::new(WeightsSource::Dir(snapshot)))
                .expect("kolors trainer loads (tokenizer.json present)");
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");
        let mut last_loss = f32::NAN;
        let output = trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Training { loss, .. } = progress {
                    last_loss = loss;
                }
            })
            .expect("training runs a step");

        eprintln!(
            "[kolors-train-smoke] steps={} final_loss={} last_step_loss={} adapter={}",
            output.steps,
            output.final_loss,
            last_loss,
            output.adapter_path.display()
        );
        assert!(output.steps >= 1, "expected at least one micro-step");
        assert!(output.final_loss.is_finite(), "final loss must be finite");
        assert!(last_loss.is_finite(), "a training-step loss was observed");
        assert!(
            output_dir.join("kolors_smoke.safetensors").exists(),
            "trained adapter was written"
        );
    }

    /// Real-weights production-scale smoke (sc-4881 / sc-4874+4886+4887, Part A4): load the
    /// z-image trainer from the installed `Tongyi-MAI/Z-Image-Turbo` snapshot and run two LoRA
    /// micro-steps **at resolution 1024 with `train_dtype="bf16"`** — the exact configuration
    /// that SIGKILL-OOM'd the worker before this fix (the 1024² first step materialized ~135 GB,
    /// over the 128 GB unified-memory budget). The image *count* doesn't change the first-step peak (batch 1;
    /// the peak is the per-step forward graph), so a one-image dataset faithfully reproduces the
    /// memory profile of the 221-image production run. Passing step 1 with a finite loss proves
    /// bf16 brings the peak under budget through the **full worker path** (`map_training_config`
    /// → `load_trainer` → `train`), not just the engine isolation tests. `gradient_checkpointing`
    /// stays off here to prove bf16 alone is sufficient. Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored z_image_1024_bf16 --nocapture`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real Z-Image-Turbo weights + Metal device; loads the full model and peaks ~44 GB at 1024"]
    fn z_image_1024_bf16_trains_past_the_first_step() {
        let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME set"));
        let snapshot = std::fs::read_dir(
            home.join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots"),
        )
        .expect("z-image snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a z-image snapshot dir");

        let tmp = tempfile::tempdir().expect("tempdir");
        let image_path = tmp.path().join("swatch.png");
        image::RgbImage::from_fn(1024, 1024, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .save(&image_path)
        .expect("write test image");
        let output_dir = tmp.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 2,
            batch_size: 1,
            gradient_accumulation: 1,
            // The fix under test: bf16 forward, checkpointing OFF — bf16 alone must suffice.
            gradient_checkpointing: false,
            train_dtype: "bf16".to_owned(),
            resolution: 1024,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".to_owned(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::parse("constant"),
            lr_warmup_steps: 0,
            network_type: NetworkType::parse("lora"),
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_owned(),
            timestep_bias: "balanced".to_owned(),
            loss_type: "mse".to_owned(),
            trigger_word: None,
        };
        let request = TrainingRequest {
            items: vec![TrainingItem {
                image_path,
                caption: "a colorful test swatch".to_owned(),
            }],
            config,
            output_dir: output_dir.clone(),
            file_name: "z_image_1024_smoke.safetensors".to_owned(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };

        let mut trainer = gen_core::load_trainer(
            "z_image_turbo",
            &LoadSpec::new(WeightsSource::Dir(snapshot)),
        )
        .expect("z-image trainer loads");
        trainer
            .validate(&request)
            .expect("trainer accepts the plan");
        let mut last_loss = f32::NAN;
        let output = trainer
            .train(&request, &mut |progress| {
                if let TrainingProgress::Training { loss, .. } = progress {
                    last_loss = loss;
                }
            })
            .expect("training survives the 1024 first step (no OOM)");

        eprintln!(
            "[z-image-1024-bf16-smoke] steps={} final_loss={} last_step_loss={} adapter={}",
            output.steps,
            output.final_loss,
            last_loss,
            output.adapter_path.display()
        );
        assert!(
            output.steps >= 1,
            "expected at least one micro-step past step 1"
        );
        assert!(output.final_loss.is_finite(), "final loss must be finite");
        assert!(last_loss.is_finite(), "a training-step loss was observed");
        assert!(
            output_dir.join("z_image_1024_smoke.safetensors").exists(),
            "trained adapter was written"
        );
    }
}
