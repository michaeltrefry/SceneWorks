use super::*;

pub(crate) async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobsQuery>,
) -> Result<Json<Vec<JobSnapshot>>, ApiError> {
    if let Some(status) = &query.status {
        if !JOB_STATUSES.contains(&status.as_str()) {
            return Err(ApiError::bad_request("Unsupported job status"));
        }
    }
    Ok(Json(
        store_call(state, move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            store.list_jobs(
                query.project_id.as_deref(),
                query.status.as_deref(),
                query.limit.unwrap_or(100),
            )
        })
        .await?,
    ))
}

pub(crate) async fn create_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<JobCreateRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: payload.job_type,
            project_id: payload.project_id,
            project_name: payload.project_name,
            payload: payload.payload,
            requested_gpu: payload.requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn claim_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ClaimRequest>,
) -> Result<Json<ClaimResponse>, ApiError> {
    let mlx_required = state.settings.mlx_required;
    let enforce_unsupported = state.settings.mlx_enforce_unsupported;
    let (response, decision, stranded, unsupported) =
        store_call(state.clone(), move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            // macOS MLX-required (sc-3483): before claiming, fail any MLX-eligible job left
            // stranded because no live `mlx` worker took it within the grace window — reusing
            // the worker timeout as that window. No-op when the flag is off.
            let stranded = store.fail_stranded_mlx_jobs(mlx_required, timeout)?;
            // macOS MLX-required + enforce (sc-3484): fail any queued job the Rust/MLX flow
            // can't run. No-op in warn mode (the default) — the gap is logged at claim instead.
            let unsupported = store.fail_unsupported_mlx_jobs(mlx_required, enforce_unsupported)?;
            let (job, decision) = store.claim_next_job_routed(&payload.worker_id, mlx_required)?;
            Ok((job, decision, stranded, unsupported))
        })
        .await?;
    for job in &stranded {
        emit_mlx_unavailable(job);
        publish(&state, "job.updated", job);
    }
    for (job, reason) in &unsupported {
        emit_mlx_unsupported(job, reason, "enforce");
        publish(&state, "job.updated", job);
    }
    if let Some(decision) = &decision {
        emit_route_decision(decision);
    }
    if let Some(job) = &response {
        // Warn-only (sc-3484): an unsupported job the torch worker just claimed on a Mac —
        // log the gap once so the inventory materializes while the job still runs on torch.
        // In enforce mode such a job was already failed above and never reaches here.
        if mlx_required && !enforce_unsupported {
            if let Err(reason) = mac_rust_supported(job) {
                emit_mlx_unsupported(job, &reason, "warn");
            }
        }
        publish(&state, "job.updated", job);
    }
    if response.is_some() || !stranded.is_empty() || !unsupported.is_empty() {
        publish_queue(&state).await?;
    }
    Ok(Json(ClaimResponse {
        job: response,
        extra: Default::default(),
    }))
}

/// Emit the macOS `mlx_unsupported` gap event (epic 3482 / sc-3484) as a structured JSON line
/// for the desktop stdout capture + headless `GET /api/v1/logs` buffer (sc-3447/3451/3453).
/// `mode` is `"enforce"` (the job was failed terminal) or `"warn"` (logged but still run on
/// torch). The body is the feature-precise [`UnsupportedReason`] — model/feature/detail/
/// suggestedEpic — so the Logs surface and the gap inventory name the exact port-or-drop work.
fn emit_mlx_unsupported(job: &JobSnapshot, reason: &UnsupportedReason, mode: &str) {
    let mut value = serde_json::to_value(reason).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "event".to_owned(),
            Value::String("mlx_unsupported".to_owned()),
        );
        object.insert("mode".to_owned(), Value::String(mode.to_owned()));
        object.insert("jobId".to_owned(), Value::String(job.id.clone()));
        object.insert(
            "jobType".to_owned(),
            Value::String(job.job_type.as_str().to_owned()),
        );
    }
    // Through the tracing backbone: the stdout JSON layer feeds the desktop capture
    // and the API's own ring buffer (GET /api/v1/logs) via the session-log layer.
    sceneworks_core::observability::emit_event(tracing::Level::INFO, value);
}

/// Emit the macOS `mlx_unavailable` terminal-routing event as a structured JSON line for
/// the desktop's stdout capture + the headless `GET /api/v1/logs` buffer (sc-3447/3451/3453).
/// Mirrors [`emit_route_decision`]: this is the System → Logs surface that turns "no MLX
/// worker took the job" into a named, actionable line instead of a job silently stuck or
/// run on MPS (sc-3483). `reason` carries the full actionable error set on the job.
fn emit_mlx_unavailable(job: &JobSnapshot) {
    let model = job.payload.get("model").and_then(Value::as_str);
    sceneworks_core::observability::emit_event(
        tracing::Level::INFO,
        json!({
            "event": "mlx_unavailable",
            "jobId": job.id,
            "jobType": job.job_type.as_str(),
            "model": model,
            "reason": job.error,
        }),
    );
}

/// Emit the MLX↔torch routing decision as a structured JSON line on the API's stdout
/// (sc-3449). The desktop wrapper captures this into `api.log` + the in-app Logs buffer,
/// so an MLX-eligible job that lands on torch is explained at claim time rather than
/// inferred from archaeology. Shape mirrors the worker's `emit_worker_event` events
/// (`event` + `reportedAt` + payload).
fn emit_route_decision(decision: &RouteDecision) {
    let mut value = serde_json::to_value(decision).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "event".to_owned(),
            Value::String("mlx_route_decision".to_owned()),
        );
    }
    // Emitted through the tracing backbone: the stdout JSON layer reaches the desktop
    // wrapper's capture (sc-3451) + api.log, and the session-log layer records it into
    // the API's own buffer for the headless `GET /api/v1/logs` (sc-3453).
    sceneworks_core::observability::emit_event(tracing::Level::INFO, value);
}

pub(crate) async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    Ok(Json(
        store_call(state, move |store, _timeout| store.get_job(&job_id)).await?,
    ))
}

pub(crate) async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.cancel_job(&job_id)
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

pub(crate) async fn retry_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let payload = retry_job_request_from_body(request).await?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.retry_job(
            &job_id,
            RetryJob {
                payload_changes: payload.payload_changes,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn retry_job_request_from_body(request: AxumRequest) -> Result<RetryJobRequest, ApiError> {
    let bytes = to_bytes(request.into_body(), 1024 * 1024)
        .await
        .map_err(|error| {
            ApiError::bad_request(format!("Unable to read retry request body: {error}"))
        })?;
    if bytes.iter().all(|byte| byte.is_ascii_whitespace()) {
        return Ok(RetryJobRequest::default());
    }
    serde_json::from_slice::<RetryJobRequest>(&bytes)
        .map_err(|error| ApiError::bad_request(format!("Invalid retry request body: {error}")))
}

pub(crate) async fn duplicate_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(payload): ApiJson<DuplicateJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.duplicate_job(
            &job_id,
            DuplicateJob {
                payload_changes: payload.payload_changes,
                requested_gpu: payload.requested_gpu,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn update_job_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(payload): ApiJson<ProgressRequest>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let progress = number_to_f64(&payload.progress, "progress")?;
    let eta_seconds = optional_number_to_f64(payload.eta_seconds.as_ref(), "etaSeconds")?;
    let peak_gpu_memory_pct =
        optional_number_to_f64(payload.peak_gpu_memory_pct.as_ref(), "peakGpuMemoryPct")?;
    let peak_gpu_load_pct =
        optional_number_to_f64(payload.peak_gpu_load_pct.as_ref(), "peakGpuLoadPct")?;
    let mut result = payload.result;
    // On a completing real training run, register the produced adapter as a
    // SceneWorks LoRA *before* recording completion, and fold the outcome into
    // the job result (story 1418). Doing it here keeps the result write atomic
    // and makes a registration failure visible in the job record rather than
    // silently dropping the trained output.
    if matches!(payload.status, JobStatus::Completed) {
        if let Some(status) = register_completed_training_lora(&state, &job_id).await {
            result.get_or_insert_with(JsonObject::new).extend(status);
        }
    }
    // Persist any generated assets the worker reported as `assetWrites` facts and
    // re-inject the built sidecars into the result so the UI keeps streaming them
    // (story 1656 — Rust is the single project-store writer).
    if let Some(result_obj) = result.as_mut() {
        persist_reported_assets(&state, &job_id, result_obj).await?;
    }
    let (job, status_changed) = store_call(state.clone(), move |store, _timeout| {
        // Read the prior status in the same blocking round-trip so we can tell a
        // pure progress tick (status unchanged) from a real queue transition.
        let previous_status = store.get_job(&job_id).map(|job| job.status).ok();
        let job = store.update_job_progress(
            &job_id,
            ProgressUpdate {
                status: payload.status,
                stage: payload.stage,
                progress,
                message: payload.message,
                error: payload.error,
                result,
                eta_seconds,
                peak_gpu_memory_pct,
                peak_gpu_load_pct,
                backend: payload.backend,
                worker_id: payload.worker_id,
            },
        )?;
        let status_changed = previous_status.as_ref() != Some(&job.status);
        Ok::<(JobSnapshot, bool), JobsStoreError>((job, status_changed))
    })
    .await?;
    publish(&state, "job.updated", &job);
    // sc-4203 (F-API-5): workers POST progress per inference step. The queue summary
    // is a full SQLite aggregation plus a stale-worker sweep, serialized and
    // broadcast to every SSE subscriber — but the queue composition only changes when
    // a job's status transitions (queued/running/terminal), not on a percentage tick.
    // Skip the refresh on pure ticks; the stale sweep still runs on worker heartbeats
    // and on every status transition.
    if status_changed {
        publish_queue(&state).await?;
    }
    Ok(Json(job))
}

/// Persist the generated assets a worker reports as `assetWrites` facts in its
/// progress result, then re-inject the built sidecars into `result.assets` /
/// `result.assetIds` so ImageStudio's live preview and the library refresh keep
/// streaming (story 1656). Idempotent: re-applied progress updates upsert the
/// same rows/files. No-op when there are no `assetWrites` (status-only updates,
/// or job types that still write their own assets).
pub(crate) async fn persist_reported_assets(
    state: &AppState,
    job_id: &str,
    result: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(asset_writes) = result.get("assetWrites").and_then(Value::as_array) else {
        return Ok(());
    };
    if asset_writes.is_empty() {
        return Ok(());
    }
    let asset_writes = asset_writes.clone();
    let generation_set_id = result
        .get("generationSetId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let generation_set = result.get("generationSet").cloned();
    // The job row is authoritative for the project id (never the worker payload).
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await?;
    let Some(project_id) = job.project_id.clone() else {
        return Ok(());
    };
    let job_id_owned = job_id.to_owned();
    let built = project_call(state.clone(), move |store| {
        if let Some(generation_set) = generation_set.as_ref() {
            store.write_generation_set(
                &project_id,
                &job_id_owned,
                generation_set,
                asset_writes.first(),
            )?;
        }
        let mut built = Vec::with_capacity(asset_writes.len());
        for fact in &asset_writes {
            built.push(store.persist_generated_asset(
                &project_id,
                &job_id_owned,
                &generation_set_id,
                fact,
            )?);
        }
        Ok(built)
    })
    .await?;
    let asset_ids: Vec<Value> = built
        .iter()
        .filter_map(|asset| asset.get("id").cloned())
        .collect();
    result.insert("assets".to_owned(), Value::Array(built));
    result.insert("assetIds".to_owned(), Value::Array(asset_ids));
    result.remove("assetWrites");
    result.remove("generationSet");
    Ok(())
}

/// Attempts LoRA registration for a job reporting completion, returning result
/// fields that describe the outcome — or `None` when the job is not a real
/// training run with a staged output. Never errors the progress update: a
/// registration failure is logged and surfaced via `loraRegistered: false` +
/// `loraRegistrationError` so the trained output is not silently lost.
pub(crate) async fn register_completed_training_lora(
    state: &AppState,
    job_id: &str,
) -> Option<JsonObject> {
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await
    .ok()?;
    if !matches!(job.job_type, JobType::LoraTrain) {
        return None;
    }
    match register_trained_lora(state, &job).await {
        Ok(None) => None,
        Ok(Some((lora_id, manifest_path))) => {
            let mut status = JsonObject::new();
            status.insert("loraRegistered".to_owned(), Value::Bool(true));
            status.insert("loraId".to_owned(), Value::String(lora_id));
            status.insert(
                "loraManifestPath".to_owned(),
                Value::String(manifest_path.display().to_string()),
            );
            Some(status)
        }
        Err(error) => {
            tracing::error!(
                event = "lora_registration_failed",
                jobId = %job.id,
                detail = %error.detail,
                "failed to register trained LoRA"
            );
            let mut status = JsonObject::new();
            status.insert("loraRegistered".to_owned(), Value::Bool(false));
            status.insert(
                "loraRegistrationError".to_owned(),
                Value::String(error.detail),
            );
            Some(status)
        }
    }
}

/// Registers a completed real training run's output as a normal SceneWorks LoRA,
/// returning the registered `(lora_id, manifest_path)` or `None` when there is
/// nothing to register (a dry run, or a job without a staged entry).
///
/// Security: the manifest path and output directory are recomputed from the
/// run's scope, owning project, and a validated LoRA id — never from the
/// (mutable) job payload — so a crafted or duplicated `lora_train` job cannot
/// redirect the manifest write outside the two canonical LoRA manifests
/// (`config_dir/manifests/user.loras.jsonc` or `<project>/loras/manifest.jsonc`).
/// A run whose adapter is missing under the recomputed output dir registers
/// nothing, so a failed/canceled/unwritten job never leaves a broken entry. The
/// entry shows up in `/api/v1/loras` and is selectable in the Studio (Image or
/// Video Studio, by LoRA family).
pub(crate) async fn register_trained_lora(
    state: &AppState,
    job: &JobSnapshot,
) -> Result<Option<(String, PathBuf)>, ApiError> {
    if job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        return Ok(None);
    }
    let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    else {
        return Ok(None);
    };
    // Derive the security-sensitive fields from the entry but trust nothing: the
    // scope is validated by `resolve_training_output_location`, and the id must be
    // a safe single path component before it can name an output dir / manifest.
    let scope = manifest_entry
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("project")
        .to_owned();
    let lora_id = manifest_entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("Training manifest entry requires an id"))?
        .to_owned();
    validate_lora_id_component(&lora_id)?;

    // Recompute the output dir and manifest path from trusted inputs; the job
    // payload's own manifest/output paths are deliberately ignored.
    let (output_dir, manifest_path) =
        resolve_training_output_location(state, &scope, job.project_id.as_deref(), &lora_id)
            .await?;
    // Register the adapter file(s) the plan declared, validated as plain in-tree
    // files that exist under the recomputed output dir. Using the declared final
    // name (not the first `.safetensors` on disk) means a step checkpoint sharing
    // the directory is never registered in place of the final adapter, while the
    // validation still rejects any `..`-traversing name a crafted payload injects.
    let Some(files) = trusted_adapter_files(manifest_entry.get("files"), &output_dir) else {
        return Err(ApiError::internal(format!(
            "No declared trained adapter found under {}; skipping LoRA registration",
            output_dir.display()
        )));
    };

    // Overwrite the security-sensitive fields with the trusted values, keeping
    // the descriptive metadata (name, family, triggerWords, baseModel,
    // provenance) the submit step captured. `source.path` stays relative so
    // `normalize_lora_entry` resolves it under the scope root.
    let mut entry = manifest_entry;
    entry.insert("id".to_owned(), Value::String(lora_id.clone()));
    entry.insert("scope".to_owned(), Value::String(scope));
    entry.insert(
        "source".to_owned(),
        json!({ "provider": "training", "path": format!("loras/{lora_id}") }),
    );
    entry.insert(
        "files".to_owned(),
        Value::Array(files.into_iter().map(Value::String).collect()),
    );
    entry.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));

    let upsert_id = lora_id.clone();
    mutate_manifest_entries(state, &manifest_path, "loras", move |entries| {
        // Replace any prior entry with this id (re-run) so provenance refreshes
        // without duplicating, preserving the original createdAt.
        let created_at = entries
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(upsert_id.as_str()))
            .and_then(|item| item.get("createdAt").cloned());
        let mut entries = entries
            .into_iter()
            .filter(|item| item.get("id").and_then(Value::as_str) != Some(upsert_id.as_str()))
            .collect::<Vec<_>>();
        let mut entry = entry;
        if let Some(created_at) = created_at {
            entry.insert("createdAt".to_owned(), created_at);
        }
        entries.push(Value::Object(entry));
        Ok((entries, ()))
    })
    .await?;
    Ok(Some((lora_id, manifest_path)))
}
