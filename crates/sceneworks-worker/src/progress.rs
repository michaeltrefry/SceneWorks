//! Job status/progress plumbing: building [`ProgressRequest`]s and posting terminal/cancel states.
use super::*;

pub(crate) async fn fail_job(
    api: &ApiClient,
    job_id: &str,
    message: &str,
    error: Option<String>,
) -> WorkerResult<()> {
    update_job(
        api,
        job_id,
        progress_payload(
            JobStatus::Failed,
            ProgressStage::Failed,
            1.0,
            message,
            error,
            None,
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn check_cancel(api: &ApiClient, job_id: &str, message: &str) -> WorkerResult<()> {
    let job: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{job_id}")).await?;
    if job.cancel_requested {
        mark_job_canceled(api, job_id, message).await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    Ok(())
}

pub(crate) async fn mark_job_canceled(
    api: &ApiClient,
    job_id: &str,
    message: &str,
) -> WorkerResult<()> {
    update_job(
        api,
        job_id,
        progress_payload(
            JobStatus::Canceled,
            ProgressStage::Canceled,
            1.0,
            message,
            None,
            None,
            None,
        ),
    )
    .await?;
    Ok(())
}

/// Run a long, self-contained blocking `task` while keeping the worker's heartbeat alive (and,
/// when given a `cancel` flag, honoring a user cancel). This is the SHARED keepalive every
/// long-inline compute path should use: the Rust worker sends no periodic heartbeat on its own
/// during a job, and posting job *progress* does NOT refresh the worker's `last_seen` (only a
/// terminal status does), so without a `Busy` ping every `progress_report_interval` (5–15s) a job
/// that runs silently past the API's worker-timeout (default 90s) gets swept to `interrupted`
/// mid-flight — then the worker's next post is 409'd and the job looks "stuck" (sc-8200, sc-8390).
///
/// Before this existed the same `select!` was open-coded per handler, so new paths (LoRA training,
/// VQA, pose/kps/person-detect) were missed — exactly the gap that hung a Krea2 LoRA run at a slow
/// step-500 checkpoint save. Streaming consumers that already own an event loop
/// (`training_jobs::consume_training_events`, caption/model/prompt/media/video) inline the same
/// interval arm instead; this helper is for the single-blocking-task shape.
///
/// `task` must own all its work — it cannot report progress between ticks. The helper pings
/// `WorkerStatus::Busy` every interval without posting any intermediate job status. When `cancel`
/// is `Some`, it also polls the API for a user cancel and trips the flag; un-interruptible compute
/// finishes its current op, then we post the terminal `Canceled` (`cancel_message`) and return
/// `WorkerError::Canceled`. Pass `None` for paths with no cancelable work (heartbeat only).
///
/// Every consumer is a job handler gated behind `any(target_os = "macos", feature =
/// "backend-candle")`, so on the plain-Linux parity build (neither) this is unused — allow
/// dead_code there only, keeping real dead-code detection on the configs that do call it.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) async fn run_blocking_with_heartbeat<R>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    cancel: Option<gen_core::CancelFlag>,
    cancel_message: &str,
    task_label: &'static str,
    mut task: tokio::task::JoinHandle<WorkerResult<R>>,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let mut canceled = false;
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut task => {
                let value = result.map_err(|error| task_join_error(task_label, error))??;
                if canceled {
                    mark_job_canceled(api, job_id, cancel_message).await?;
                    return Err(WorkerError::Canceled(cancel_message.to_owned()));
                }
                return Ok(value);
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(job_id)).await?;
                if let Some(flag) = &cancel {
                    if !canceled && cancel_requested_peek(api, job_id).await {
                        flag.cancel();
                        canceled = true;
                    }
                }
            }
        }
    }
}

/// Check-only cancel poll (sc-5515): returns `true` when the user requested
/// cancellation, WITHOUT posting any status. Unlike [`check_cancel`] this never
/// writes the terminal `Canceled`. In-loop generation/training pollers that sit in
/// front of a long, un-interruptible compute use this so the job stays non-terminal
/// ("Cancelling…") until the in-flight work actually stops; they post the terminal
/// `Canceled` themselves only once it does (sc-5515 image, sc-5516 video/training/detail).
/// Posting terminal at acknowledgement time frees the worker row
/// (`jobs_store::update_job_progress`) while the worker process is still busy, so
/// the next queued job is told a worker is free that isn't — deferring the
/// terminal write to actual-stop keeps the two in sync. Transient GET failures are
/// tolerated (read as "not canceled", retried on the next poll) so an API hiccup
/// never aborts a multi-minute run by being misread as a user cancel (sc-4174).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) async fn cancel_requested_peek(api: &ApiClient, job_id: &str) -> bool {
    let outcome: WorkerResult<JobSnapshot> = api.get_json(&format!("/api/v1/jobs/{job_id}")).await;
    match outcome {
        Ok(job) => job.cancel_requested,
        Err(error) => {
            tracing::warn!(
                event = "cancel_poll_failed",
                jobId = %job_id,
                error = %error,
                "cancel poll failed; retrying on the next poll"
            );
            false
        }
    }
}

pub(crate) async fn update_job(
    api: &ApiClient,
    job_id: &str,
    mut payload: ProgressRequest,
) -> WorkerResult<JobSnapshot> {
    // Stamp the reporting worker so the server can reject the write if this
    // worker no longer owns the job (swept stale / canceled / reclaimed). The
    // resulting 409 propagates as WorkerError::Api and aborts the local job
    // handling — i.e. the worker abandons the job (sc-4172).
    payload.worker_id = Some(api.worker_id.clone());
    api.post_json(&format!("/api/v1/jobs/{job_id}/progress"), &payload)
        .await
}

pub(crate) fn progress_payload(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    error: Option<String>,
    result: Option<JsonObject>,
    eta_seconds: Option<ContractNumber>,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error,
        result,
        eta_seconds,
        // The Rust utility worker doesn't run GPU work, so it never reports
        // per-job peak GPU stats. The Python GPU worker (scene_worker) sets
        // these (sc-2086). Same for `backend` — utility jobs run on the CPU
        // worker which never advertises a GPU runtime.
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some("cpu".to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

pub(crate) fn number_from_f64(value: f64) -> ContractNumber {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}
