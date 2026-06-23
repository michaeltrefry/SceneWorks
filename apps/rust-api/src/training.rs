use super::*;

pub(crate) async fn list_training_targets() -> Json<TrainingTargetRegistry> {
    Json(builtin_training_targets())
}

pub(crate) async fn list_training_presets(
) -> Json<sceneworks_core::training::TrainingPresetRegistry> {
    Json(builtin_training_presets())
}

pub(crate) async fn list_training_datasets(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<TrainingDatasetSummary>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.list_training_datasets(&project_id)
        })
        .await?,
    ))
}

pub(crate) async fn create_training_dataset(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<TrainingDatasetCreateInput>,
) -> Result<(StatusCode, Json<TrainingDataset>), ApiError> {
    let dataset = project_call(state, move |store| {
        store.create_training_dataset(&project_id, payload)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(dataset)))
}

pub(crate) async fn upload_training_dataset_item(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        if field.name() != Some("file") {
            continue;
        }
        let filename = field.file_name().unwrap_or("dataset-upload").to_owned();
        let content_type = field.content_type().map(str::to_owned);
        let temp_path = write_upload_field_to_temp_file(&state, field).await?;
        let source_path = temp_path.clone();
        let upload = sceneworks_core::project_store::TrainingDatasetUpload {
            filename,
            content_type,
            source_path,
        };
        let item = project_call(state, move |store| {
            store.upload_training_dataset_item(&project_id, upload)
        })
        .await
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&temp_path);
        })?;
        return Ok((StatusCode::CREATED, Json(item)));
    }
    Err(ApiError::bad_request("Upload file field is required"))
}

pub(crate) async fn get_training_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
) -> Result<Json<TrainingDataset>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_training_dataset(&project_id, &dataset_id)
        })
        .await?,
    ))
}

/// Tier-0 readiness report for a dataset (sc-6533). Computed server-side and returned as one
/// structured payload the training screens render; replaces the client-only `datasetHealth`.
pub(crate) async fn get_training_dataset_readiness(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    Query(query): Query<ReadinessQuery>,
) -> Result<Json<sceneworks_core::dataset_quality::DatasetReadinessReport>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            dataset_readiness_report(&store, &project_id, &dataset_id, &query)
        })
        .await?,
    ))
}

/// Persist (or clear) a per-image quality override (sc-6534). The body lists the checks the user
/// dismissed for the image; the store strips the non-acknowledgeable ones (`decode`, `count`) and
/// keys the ack by the item's current content hash, so a later image swap voids it. Returns the
/// stored ack, or `null` when the override was cleared.
pub(crate) async fn set_training_dataset_item_quality_ack(
    State(state): State<AppState>,
    Path((project_id, dataset_id, item_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<QualityAckBody>,
) -> Result<Json<Option<sceneworks_core::dataset_quality::QualityAck>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.set_dataset_item_quality_ack(&project_id, &dataset_id, &item_id, &payload.checks)
        })
        .await?,
    ))
}

/// Read the dataset, reuse any still-valid cached Tier-0 scalars, decode the rest via
/// `sceneworks-image-quality`, roll the readiness report up, and persist the freshly-extracted
/// scalars as the content-hash + bucket-keyed cache. Runs synchronously inside `project_call`'s
/// blocking task (the image decode belongs off the async runtime).
fn dataset_readiness_report(
    store: &ProjectStore,
    project_id: &str,
    dataset_id: &str,
    query: &ReadinessQuery,
) -> Result<sceneworks_core::dataset_quality::DatasetReadinessReport, ProjectStoreError> {
    use sceneworks_core::dataset_quality::{
        evaluate_aesthetic, evaluate_tier1, readiness_context, AestheticThresholds,
        CachedTier0Scalars, DatasetKind, ItemEmbedding, Tier1Thresholds,
    };
    use sceneworks_image_quality::{aesthetic_predictor, compute_readiness, ReadinessItem};

    let (dataset, root, _project_stem) = store.training_dataset_for_plan(project_id, dataset_id)?;

    let recommended_for: Vec<String> = query
        .recommended_for
        .as_deref()
        .map(|tags| {
            tags.split(',')
                .map(|tag| tag.trim().to_owned())
                .filter(|tag| !tag.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let context = readiness_context(
        query.target_resolution,
        &recommended_for,
        query.character_type.as_deref(),
        query.min_items,
    );

    let items: Vec<ReadinessItem> = dataset
        .items
        .iter()
        .map(|item| {
            let cached_scalars = item
                .tier0_scalars
                .as_ref()
                .filter(|cache| cache.valid_for(item.content_hash.as_deref(), context.bucket_edge))
                .map(|cache| cache.scalars.clone());
            // Resolve the user's dismissed findings (sc-6534) against the current bytes — a stale
            // ack (image replaced since) yields no effective checks.
            let acknowledged = item
                .quality_ack
                .as_ref()
                .map(|ack| ack.effective_checks(item.content_hash.as_deref()))
                .unwrap_or_default();
            ReadinessItem {
                item_id: item.id.clone(),
                width: item.width,
                height: item.height,
                content_hash: item.content_hash.clone(),
                image_path: Some(root.join(&item.path)),
                cached_scalars,
                acknowledged,
            }
        })
        .collect();

    // Tier-1 (sc-6535): if the analysis worker has persisted an embedding sidecar, fold its findings
    // (embedding near-duplicates + low set diversity) into the report. Each item's dismissed checks
    // (sc-6534) carry through, so an acknowledged near-dup drops from the rollups. No sidecar (the
    // job hasn't run) → `None` → the report stays Tier-0 only, exactly as before.
    let embeddings: Vec<ItemEmbedding> = store
        .read_dataset_embeddings(project_id, dataset_id)?
        .map(|sidecar| {
            dataset
                .items
                .iter()
                .filter_map(|item| {
                    let content_hash = item.content_hash.as_deref()?;
                    let embedding = sidecar.embeddings.get(content_hash)?.clone();
                    let acknowledged = item
                        .quality_ack
                        .as_ref()
                        .map(|ack| ack.effective_checks(item.content_hash.as_deref()))
                        .unwrap_or_default();
                    Some(ItemEmbedding {
                        item_id: item.id.clone(),
                        embedding,
                        acknowledged,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    // A sidecar with no current item matching it (every image changed since the analysis) is not
    // Tier-1 data — fall back to Tier-0 only rather than a misleading 1.0 diversity.
    let tier1 = (!embeddings.is_empty())
        .then(|| evaluate_tier1(&embeddings, &Tier1Thresholds::for_kind(&context.kind)));
    // Aesthetic (sc-6537): STYLE datasets only — score the same embeddings with the bundled LAION
    // predictor. Person/object never get an aesthetic sub-score or flag (documented bias).
    let aesthetic = if context.kind == DatasetKind::Style {
        evaluate_aesthetic(
            &embeddings,
            aesthetic_predictor(),
            &context.kind,
            &AestheticThresholds::default(),
        )
    } else {
        None
    };

    let (report, extracted) = compute_readiness(
        &items,
        context.bucket_edge,
        context.min_items,
        &context.thresholds,
        tier1.as_ref(),
        aesthetic.as_ref(),
    );

    if !extracted.is_empty() {
        let updates: Vec<(String, CachedTier0Scalars)> = extracted
            .into_iter()
            .filter_map(|(item_id, scalars)| {
                // Key the cache by the item's current content hash; skip items without one (they
                // can't be validated on reuse, so they're recomputed each time).
                let content_hash = dataset
                    .items
                    .iter()
                    .find(|item| item.id == item_id)?
                    .content_hash
                    .clone()?;
                Some((
                    item_id,
                    CachedTier0Scalars {
                        content_hash,
                        bucket_edge: context.bucket_edge,
                        scalars,
                    },
                ))
            })
            .collect();
        store.cache_dataset_tier0_scalars(project_id, dataset_id, &updates)?;
    }

    Ok(report)
}

pub(crate) async fn update_training_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingDatasetUpdateInput>,
) -> Result<Json<TrainingDataset>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_training_dataset(&project_id, &dataset_id, payload)
        })
        .await?,
    ))
}

pub(crate) async fn batch_rename_training_dataset_items(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingDatasetBatchRenameInput>,
) -> Result<Json<TrainingDataset>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.batch_rename_training_dataset_items(&project_id, &dataset_id, payload)
        })
        .await?,
    ))
}

pub(crate) async fn write_training_dataset_caption_sidecars(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingDatasetCaptionSidecarsInput>,
) -> Result<Json<TrainingCaptionSidecarsResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.write_training_dataset_caption_sidecars(&project_id, &dataset_id, payload)
        })
        .await?,
    ))
}

pub(crate) async fn create_training_dataset_caption_job(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingCaptionJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_training_caption_job_request(&payload)?;
    let (dataset, dataset_root, project_name) = project_call(state.clone(), {
        let project_id = project_id.clone();
        let dataset_id = dataset_id.clone();
        move |store| store.training_dataset_for_plan(&project_id, &dataset_id)
    })
    .await?;
    if dataset.items.is_empty() {
        return Err(ApiError::bad_request(
            "Training dataset has no items to caption.",
        ));
    }
    // sc-2025: a per-image Re-Caption targets specific item ids and always
    // recaptions them; otherwise the dataset-wide recaption/missing rule applies.
    let target_ids: Option<std::collections::HashSet<&str>> = payload
        .item_ids
        .as_ref()
        .map(|ids| ids.iter().map(String::as_str).collect());
    let items = dataset
        .items
        .iter()
        .filter(|item| match &target_ids {
            Some(ids) => ids.contains(item.id.as_str()),
            None => payload.recaption || item.caption.text.trim().is_empty(),
        })
        .map(|item| {
            json!({
                "itemId": item.id.clone(),
                "imagePath": dataset_root.join(&item.path).display().to_string(),
                "existingCaption": item.caption.text.clone(),
                "triggerWords": item.caption.trigger_words.clone(),
            })
        })
        .collect::<Vec<_>>();
    if items.is_empty() {
        return Err(ApiError::bad_request(
            "No dataset items need captions. Enable recaption to overwrite existing captions.",
        ));
    }
    let options = serde_json::to_value(payload.options)
        .map_err(|error| ApiError::internal(format!("caption options serialize: {error}")))?;
    let captioner = payload.captioner;
    let model_name_or_path = payload.model_name_or_path;
    let requested_gpu = payload.requested_gpu;
    let job_payload = match json!({
        "provider": "training",
        "kind": "training_caption",
        "captioner": captioner,
        "modelNameOrPath": model_name_or_path,
        "projectId": project_id.clone(),
        "datasetId": dataset.id,
        "datasetVersion": dataset.version,
        "datasetRoot": dataset_root.display().to_string(),
        "recaption": payload.recaption,
        "options": options,
        "items": items,
    }) {
        Value::Object(map) => map,
        _ => return Err(ApiError::internal("caption job payload must be an object")),
    };
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: JobType::TrainingCaption,
            project_id: Some(project_id),
            project_name: Some(project_name),
            payload: job_payload,
            requested_gpu,
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

/// Enqueue a Dataset Doctor CLIP-embedding analysis job over a training dataset (sc-6535).
/// Mirrors [`create_training_dataset_caption_job`]: build a per-item work list (item id + absolute
/// image path + content hash) and create a GPU-routed `dataset_analysis` job. The Rust/MLX worker
/// runs the `clip_vit_l14` embedder once it advertises the capability (after the cross-repo re-pin);
/// until then the job stays queued.
pub(crate) async fn create_training_dataset_analysis_job(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<DatasetAnalysisJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_dataset_analysis_job_request(&payload)?;
    let (dataset, dataset_root, project_name) = project_call(state.clone(), {
        let project_id = project_id.clone();
        let dataset_id = dataset_id.clone();
        move |store| store.training_dataset_for_plan(&project_id, &dataset_id)
    })
    .await?;
    if dataset.items.is_empty() {
        return Err(ApiError::bad_request(
            "Training dataset has no items to analyze.",
        ));
    }
    // No item_ids → analyze the whole dataset; otherwise just the named items.
    let target_ids: Option<std::collections::HashSet<&str>> = payload
        .item_ids
        .as_ref()
        .map(|ids| ids.iter().map(String::as_str).collect());
    let items = dataset
        .items
        .iter()
        .filter(|item| match &target_ids {
            Some(ids) => ids.contains(item.id.as_str()),
            None => true,
        })
        .map(|item| {
            json!({
                "itemId": item.id.clone(),
                "imagePath": dataset_root.join(&item.path).display().to_string(),
                "contentHash": item.content_hash.clone(),
            })
        })
        .collect::<Vec<_>>();
    if items.is_empty() {
        return Err(ApiError::bad_request(
            "No matching dataset items to analyze.",
        ));
    }
    let embedder = payload.embedder;
    let model_name_or_path = payload.model_name_or_path;
    let requested_gpu = payload.requested_gpu;
    let job_payload = match json!({
        "provider": "training",
        "kind": "dataset_analysis",
        "embedder": embedder,
        "modelNameOrPath": model_name_or_path,
        "projectId": project_id.clone(),
        "datasetId": dataset.id,
        "datasetVersion": dataset.version,
        "datasetRoot": dataset_root.display().to_string(),
        "items": items,
    }) {
        Value::Object(map) => map,
        _ => {
            return Err(ApiError::internal(
                "dataset analysis job payload must be an object",
            ))
        }
    };
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: JobType::DatasetAnalysis,
            project_id: Some(project_id),
            project_name: Some(project_name),
            payload: job_payload,
            requested_gpu,
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

pub(crate) fn validate_dataset_analysis_job_request(
    payload: &DatasetAnalysisJobRequest,
) -> Result<(), ApiError> {
    if payload.embedder.trim() != "clip_vit_l14" {
        return Err(ApiError::bad_request(
            "Unsupported dataset analysis embedder. Use clip_vit_l14.",
        ));
    }
    Ok(())
}

/// Persist the analysis worker's computed CLIP embeddings to the dataset's content-hash-keyed
/// sidecar (sc-6535) — the embedding-side analog of `write_training_dataset_caption_sidecars`. A
/// metadata write: it does not bump the dataset version. Returns the count stored.
pub(crate) async fn write_training_dataset_analysis_embeddings(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<DatasetEmbeddingsBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let embeddings = sceneworks_core::dataset_quality::DatasetEmbeddings {
        space: payload.space,
        embeddings: payload
            .items
            .into_iter()
            .map(|record| (record.content_hash, record.embedding))
            .collect(),
    };
    let stored = embeddings.embeddings.len();
    project_call(state, move |store| {
        store.write_dataset_embeddings(&project_id, &dataset_id, &embeddings)
    })
    .await?;
    Ok(Json(json!({ "stored": stored })))
}

pub(crate) fn validate_training_caption_job_request(
    payload: &TrainingCaptionJobRequest,
) -> Result<(), ApiError> {
    if payload.captioner.trim() != "joy_caption" {
        return Err(ApiError::bad_request(
            "Unsupported training captioner. Use joy_caption.",
        ));
    }
    if payload.model_name_or_path.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Joy Caption model name or path is required.",
        ));
    }
    let options = &payload.options;
    if !(0.0..=2.0).contains(&options.temperature) {
        return Err(ApiError::bad_request(
            "Caption temperature must be between 0 and 2.",
        ));
    }
    if !(0.0..=1.0).contains(&options.top_p) {
        return Err(ApiError::bad_request(
            "Caption topP must be between 0 and 1.",
        ));
    }
    if options.max_new_tokens == 0 || options.max_new_tokens > 1024 {
        return Err(ApiError::bad_request(
            "Caption maxNewTokens must be between 1 and 1024.",
        ));
    }
    if options.caption_prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "Caption prompt must be at most 4000 characters.",
        ));
    }
    if options.name_input.chars().count() > 120 {
        return Err(ApiError::bad_request(
            "Caption name must be at most 120 characters.",
        ));
    }
    if options.extra_options.len() > 16 {
        return Err(ApiError::bad_request(
            "Choose at most 16 caption extra options.",
        ));
    }
    if options
        .extra_options
        .iter()
        .any(|option| option.chars().count() > 500)
    {
        return Err(ApiError::bad_request(
            "Caption extra options must be at most 500 characters.",
        ));
    }
    Ok(())
}

pub(crate) async fn delete_training_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
) -> Result<Json<TrainingDatasetMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_training_dataset(&project_id, &dataset_id)
        })
        .await?,
    ))
}

pub(crate) async fn create_training_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<LoraTrainingRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    // Both dry-run plan validation and real execution exist (story 1417). A
    // dry-run resolves and validates the plan without producing weights; a real
    // run hands the same plan to the worker's Z-Image LoRA kernel.
    //
    // Targets come from the Rust-owned registry; the request only names one.
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == payload.target_id)
        .ok_or_else(|| {
            ApiError::bad_request(format!("Unknown training target: {}", payload.target_id))
        })?;
    let preset_metadata = match payload.preset_id.as_deref() {
        Some(preset_id) => {
            let preset_registry = builtin_training_presets();
            let preset = preset_registry
                .presets
                .iter()
                .find(|preset| preset.id == preset_id)
                .ok_or_else(|| {
                    ApiError::bad_request(format!("Unknown training preset: {preset_id}"))
                })?;
            if preset.target_id != target.id {
                return Err(ApiError::bad_request(format!(
                    "Training preset '{}' targets '{}', but the request targets '{}'.",
                    preset.id, preset.target_id, target.id
                )));
            }
            if let Some(requested_version) = payload.preset_version {
                if requested_version != preset.version {
                    return Err(ApiError::bad_request(format!(
                        "Training preset '{}' is version {}, but the request pinned version {requested_version}.",
                        preset.id, preset.version
                    )));
                }
            }
            let preset_config_snapshot = serde_json::to_value(&preset.config)
                .ok()
                .and_then(|value| match value {
                    Value::Object(map) => Some(map),
                    _ => None,
                })
                .unwrap_or_default();
            Some(TrainingPresetProvenance {
                preset_id: preset.id.clone(),
                preset_version: preset.version,
                preset_name: preset.name.clone(),
                preset_config_snapshot,
            })
        }
        None => {
            if payload.preset_version.is_some() {
                return Err(ApiError::bad_request(
                    "presetVersion requires a matching presetId.",
                ));
            }
            None
        }
    };

    let output_name = payload.output_name.trim().to_owned();
    if output_name.is_empty() {
        return Err(ApiError::bad_request("Training output name is required."));
    }

    // Load the dataset, its absolute root, and the project name for the queue.
    let dataset_id = payload.dataset_id.clone();
    let (dataset, dataset_root, project_name) = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.training_dataset_for_plan(&project_id, &dataset_id)
    })
    .await?;

    // We persist only the dataset's current version, so an older pin is unrunnable.
    if let Some(requested_version) = payload.dataset_version {
        if requested_version != dataset.version {
            return Err(ApiError::bad_request(format!(
                "Dataset {} is at version {}, but the request pinned version {requested_version}.",
                dataset.id, dataset.version
            )));
        }
    }

    // Resolve absolute on-host paths and ids the kernel will consume. The job id
    // is pre-allocated so the plan can embed its own `jobId`/`sourceJobId`.
    let data_dir = state.settings.data_dir.clone();
    let base_model_path = resolve_base_model_path(target, &data_dir);
    let lora_id = format!("lora_{}", Uuid::new_v4().simple());
    let file_name = format!("{}.safetensors", slugify_lora_id(&output_name));
    let job_id = format!("job_{}", Uuid::new_v4().simple());
    let requested_gpu = training_requested_gpu(&payload.config.advanced);
    // The adapter's network parameterization (epic 2193). Recorded on the trained
    // LoRA so generation can route LoKr off the MLX backend (which is LoRA-only)
    // without opening the file — mirrors how `baseModel` gates Wan 5B/14B.
    let network_type = payload
        .config
        .advanced
        .get("networkType")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "lora".to_owned());

    // Where the produced adapter is written. The target's default `outputScope`
    // (project) lives in the config's `advanced` bag: project outputs land in the
    // project's LoRA store, global outputs in the shared data dir. The manifest
    // `source.path` stays relative to the scope's root so `normalize_lora_entry`
    // resolves the installed path on either side. The matching manifest path is
    // recomputed from the same trusted inputs at registration time
    // (`register_trained_lora`), never read back from the job payload.
    let output_scope = training_output_scope(&payload.config.advanced)?;
    let (output_dir, _manifest_path) =
        resolve_training_output_location(&state, &output_scope, Some(&project_id), &lora_id)
            .await?;
    let source_relpath = format!("loras/{lora_id}");

    // Operational guardrails (story 1419): fail fast with actionable errors for
    // the common setup problems, before a job is queued.
    //
    // The produced LoRA's family must be one an installed model accepts, or the
    // output would never be selectable in the Studio. When no model manifests
    // are present the set is empty and this is a no-op. Families come straight
    // from the manifests (not `model_catalog`) so this guardrail — which runs on
    // every submit, including the offline dry-run path — makes no network calls.
    let normalized_family = normalize_lora_family(&target.family);
    let known_families = known_lora_families_from_manifests(&state).await?;
    if !known_families.is_empty()
        && !known_families
            .iter()
            .any(|family| family == &normalized_family)
    {
        return Err(ApiError::bad_request(format!(
            "Training target '{}' produces LoRA family '{}', which no installed model accepts ({}). Install a compatible base model first.",
            target.id,
            target.family,
            known_families.join(", ")
        )));
    }

    // A real run loads the base model and writes weights, so it must have the
    // model installed and room on disk. A dry run only resolves the plan, so it
    // is exempt — that is how you preview a plan before installing the model.
    if !payload.dry_run {
        if !training_base_model_installed(&data_dir, target) {
            return Err(ApiError::bad_request(format!(
                "Base model '{}' is not installed. Install it from the model catalog before starting a real training run (dry runs work without it).",
                target.base_model
            )));
        }
        if let Some(message) = training_disk_space_error(&output_dir) {
            return Err(ApiError::bad_request(message));
        }
    }

    // Rust resolves and validates the normalized plan before any job is queued.
    let plan = build_training_plan(BuildTrainingPlan {
        job_id: &job_id,
        target,
        dataset: &dataset,
        config: payload.config,
        preset: preset_metadata,
        lora_id: &lora_id,
        base_model_path,
        dataset_root: &dataset_root,
        output_dir: &output_dir,
        file_name,
        created_at: now_rfc3339(),
    })
    .map_err(|error| ApiError::bad_request(error.to_string()))?;

    // Pre-build the LoRA registry entry the completed job will register, mirroring
    // the `lora_import` pattern: Rust owns LoRA registration. The descriptive
    // metadata is captured here; the manifest path and the security-sensitive
    // fields (id, scope, source path) are recomputed from trusted inputs when the
    // entry is upserted on completion (story 1418), so this stays purely
    // informational. Dry runs never register.
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": lora_id.clone(),
        "name": output_name.clone(),
        "scope": output_scope,
        "family": target.family.clone(),
        "baseModel": target.base_model.clone(),
        "networkType": network_type,
        "triggerWords": plan.output.trigger_words.clone(),
        "source": {
            "provider": "training",
            "path": source_relpath,
        },
        "files": [plan.output.file_name.clone()],
        "provenance": {
            "kind": "training",
            "trainingJobId": job_id.clone(),
            "targetId": plan.provenance.target_id.clone(),
            "datasetId": plan.provenance.dataset_id.clone(),
            "datasetVersion": plan.provenance.dataset_version,
            "baseModel": plan.provenance.base_model.clone(),
            "presetId": plan.provenance.preset_id.clone(),
            "presetVersion": plan.provenance.preset_version,
            "presetName": plan.provenance.preset_name.clone(),
            "presetConfigSnapshot": plan.provenance.preset_config_snapshot.clone(),
            "configSnapshot": plan.provenance.config_snapshot.clone(),
            "createdAt": timestamp.clone(),
        },
        "createdAt": timestamp.clone(),
        "updatedAt": timestamp,
    });
    if let Some(provenance) = manifest_entry
        .get_mut("provenance")
        .and_then(Value::as_object_mut)
    {
        provenance.retain(|_, value| !value.is_null());
    }

    let plan_value =
        serde_json::to_value(&plan).map_err(|error| ApiError::internal(error.to_string()))?;
    let mut job_payload = JsonObject::new();
    job_payload.insert("dryRun".to_owned(), Value::Bool(payload.dry_run));
    job_payload.insert("outputName".to_owned(), Value::String(output_name));
    job_payload.insert("plan".to_owned(), plan_value);
    job_payload.insert("manifestEntry".to_owned(), manifest_entry);

    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job_with_id(
            job_id,
            CreateJob {
                job_type: JobType::LoraTrain,
                project_id: Some(project_id),
                project_name: Some(project_name),
                payload: job_payload,
                requested_gpu,
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Absolute path to the target's base model weights on the worker host. Prefers a
/// locally-converted MLX model dir (`requiresConversion` models like Wan), then the
/// Hugging Face hub snapshot for the target's repo, falling back to the local models
/// directory. The path need not exist yet — model installation is a separate job; the
/// dry-run plan only records where the kernel will read from.
pub(crate) fn resolve_base_model_path(target: &TrainingTarget, data_dir: &FsPath) -> String {
    // Locally-converted MLX model: `requiresConversion` models (Wan TI2V-5B / A14B) keep
    // their usable weights in the app-managed `<data>/models/mlx/<id>` tree, NOT the HF cache
    // — which holds only the native *source* checkpoint the converter consumes. Inference reads
    // the converted dir via `video_jobs::local_mlx_dir` (gated on `config.json`); training must
    // read the same tree, so prefer it whenever it is populated. The converted dir is keyed by
    // the manifest model id, which equals `target.base_model` for every conversion target.
    let converted = data_dir.join("models").join("mlx").join(&target.base_model);
    if converted.join("config.json").is_file() {
        return converted.display().to_string();
    }
    if let Some(repo) = target
        .base_model_repo
        .as_deref()
        .map(str::trim)
        .filter(|repo| !repo.is_empty())
    {
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, repo) {
            // The diffusers component tree (tokenizer/ text_encoder/ transformer/ vae/) the
            // trainer and pipeline read lives inside the resolved snapshot dir
            // (snapshots/<rev>/), not at the repo cache root — which holds only blobs/ refs/
            // snapshots/. Descend into the main snapshot so callers get a usable component
            // root, exactly as inference does. Fall back to the cache root when no snapshot is
            // materialized yet (the path need not exist at dry-run time).
            if let Some(snapshot) = huggingface_snapshot_dirs(&cache_path).into_iter().next() {
                return snapshot.display().to_string();
            }
            return cache_path.display().to_string();
        }
    }
    data_dir
        .join("models")
        .join(safe_download_dir(&target.base_model))
        .display()
        .to_string()
}

/// GPU selection for a training job, read from the config's advanced bag (the
/// request has no top-level field). Defaults to `auto`; `lora_train` is
/// GPU-required, so a `cpu` value is rejected downstream when the job is created.
pub(crate) fn training_requested_gpu(advanced: &JsonObject) -> String {
    let raw = advanced
        .get("requestedGpu")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    requested_gpu_or_auto(raw)
}

/// Resolves the output scope for a training run from the config's `advanced` bag.
/// Defaults to `project` (the target default) and rejects anything other than the
/// two scopes the LoRA registry understands.
pub(crate) fn training_output_scope(advanced: &JsonObject) -> Result<String, ApiError> {
    let scope = advanced
        .get("outputScope")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("project");
    match scope {
        "project" | "global" => Ok(scope.to_owned()),
        other => Err(ApiError::bad_request(format!(
            "Unsupported training outputScope: {other}. Use project or global."
        ))),
    }
}

/// Single source of truth for where a training run's adapter is written and its
/// LoRA registered, derived only from trusted inputs: the scope, the owning
/// project, and the pre-allocated LoRA id. `create_training_job` uses it to place
/// the plan's output dir at submit time, and `register_trained_lora` recomputes
/// it at completion so a (mutable) job payload can never redirect a manifest
/// write outside the two canonical LoRA manifests. Returns
/// `(output_dir, manifest_path)`.
pub(crate) async fn resolve_training_output_location(
    state: &AppState,
    scope: &str,
    project_id: Option<&str>,
    lora_id: &str,
) -> Result<(PathBuf, PathBuf), ApiError> {
    match scope {
        "project" => {
            let project_id = project_id.ok_or_else(|| {
                ApiError::bad_request("Project-scoped training requires a project id")
            })?;
            let loras_dir = project_path_for_id(state.clone(), project_id)
                .await?
                .join("loras");
            Ok((loras_dir.join(lora_id), loras_dir.join("manifest.jsonc")))
        }
        "global" => {
            let loras_dir = state.settings.data_dir.join("loras");
            Ok((
                loras_dir.join(lora_id),
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
            ))
        }
        other => Err(ApiError::bad_request(format!(
            "Unsupported training outputScope: {other}. Use project or global."
        ))),
    }
}

/// Rejects a LoRA id that is not a single safe path component, so a crafted job
/// payload cannot escape the LoRA output/manifest tree via `..` or path
/// separators. Server-generated ids (`lora_<hex>`) always pass.
pub(crate) fn validate_lora_id_component(lora_id: &str) -> Result<(), ApiError> {
    let invalid = lora_id.is_empty()
        || lora_id == "."
        || lora_id == ".."
        || lora_id.contains("..")
        || lora_id.chars().any(|character| {
            !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.'))
        });
    if invalid {
        return Err(ApiError::bad_request(format!(
            "Invalid LoRA id for training output: {lora_id}"
        )));
    }
    Ok(())
}

/// Whether the training target's base model weights are present on disk, using
/// the same resolution `model_catalog` reports: the shared Hugging Face hub
/// cache for the target's repo, or a SceneWorks-managed `data/models/<id>`
/// install marker. A real run requires this; a dry run does not.
pub(crate) fn training_base_model_installed(data_dir: &FsPath, target: &TrainingTarget) -> bool {
    if let Some(repo) = target
        .base_model_repo
        .as_deref()
        .map(str::trim)
        .filter(|repo| !repo.is_empty())
    {
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, repo) {
            if models::huggingface_cache_health(&cache_path, &[]).installed {
                return true;
            }
        }
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        if model_is_installed(&managed) {
            return true;
        }
    }
    let managed = data_dir
        .join("models")
        .join(safe_download_dir(&target.base_model));
    model_is_installed(&managed)
}

/// Minimum free space we require at the output location before queuing a real
/// training run: enough headroom for periodic checkpoints plus the final
/// adapter. Conservative so it only trips when a disk is genuinely low.
const MIN_FREE_TRAINING_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Returns an actionable error message when the output volume is too low on
/// space for a real run, or `None` when there is room (or free space cannot be
/// determined — we do not block on an unknowable answer).
pub(crate) fn training_disk_space_error(output_dir: &FsPath) -> Option<String> {
    // `output_dir` itself rarely exists yet; probe the nearest existing parent.
    let probe = nearest_existing_ancestor(output_dir)?;
    let available = fs2::available_space(&probe).ok()?;
    insufficient_disk_space(available, MIN_FREE_TRAINING_BYTES).then(|| {
        format!(
            "Not enough free disk space to train: {} available on the volume holding {}, but at least {} is recommended. Free up space and try again.",
            human_gib(available),
            probe.display(),
            human_gib(MIN_FREE_TRAINING_BYTES)
        )
    })
}

/// Pure decision split out so the threshold logic is unit-testable without
/// touching a real filesystem.
pub(crate) fn insufficient_disk_space(available: u64, required: u64) -> bool {
    available < required
}

/// Nearest ancestor of `path` (including itself) that exists on disk.
pub(crate) fn nearest_existing_ancestor(path: &FsPath) -> Option<PathBuf> {
    let mut current = Some(path);
    while let Some(candidate) = current {
        if candidate.exists() {
            return Some(candidate.to_path_buf());
        }
        current = candidate.parent();
    }
    None
}

pub(crate) fn human_gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// The trusted `files` list for a trained LoRA: the adapter file names the plan
/// declared (staged into `manifestEntry.files` at submit), each validated as a
/// plain in-tree file and confirmed to exist under the recomputed output dir.
/// Returns `None` when none qualify.
///
/// Trusting the declared name rather than the first `.safetensors` on disk
/// matters: the trainer leaves step checkpoints (`<stem>-stepNNN.safetensors`)
/// in the same directory as the final `<stem>.safetensors`, and an arbitrary
/// pick could register an under-trained checkpoint. Requiring plain components
/// also keeps a crafted `..`-traversing `files` value from pointing generation
/// at a safetensors outside `installedPath`.
pub(crate) fn trusted_adapter_files(
    declared: Option<&Value>,
    output_dir: &FsPath,
) -> Option<Vec<String>> {
    let declared = declared?.as_array()?;
    let files = declared
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| is_plain_relative_file(name) && output_dir.join(name).is_file())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

/// Whether `name` is a single relative file path made only of normal components
/// (no `..`, root, drive prefix, or `.`), so joining it to an output dir cannot
/// escape that dir.
pub(crate) fn is_plain_relative_file(name: &str) -> bool {
    let path = FsPath::new(name);
    !name.is_empty()
        && path.file_name().is_some()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}
