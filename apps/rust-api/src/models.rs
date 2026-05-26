use super::*;

use sceneworks_core::credentials::normalize_host;

const ALLOWED_MODEL_TYPES: &[&str] = &["image", "video", "utility"];
const MODEL_SIZE_CACHE_LIMIT: usize = 64;

#[derive(Debug, Default)]
pub(crate) struct ModelSizeCache {
    entries: HashMap<ModelSizeCacheKey, u64>,
    order: VecDeque<ModelSizeCacheKey>,
}

type ModelSizeCacheKey = (String, Vec<String>);

impl ModelSizeCache {
    pub(crate) fn get(&mut self, key: &ModelSizeCacheKey) -> Option<u64> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get(key).copied()
    }

    pub(crate) fn insert(&mut self, key: ModelSizeCacheKey, value: u64) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
        while self.order.len() > MODEL_SIZE_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &ModelSizeCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DownloadContext {
    repo: String,
    files: Vec<String>,
    fallback_size_bytes: Option<u64>,
}

pub(crate) async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(model_catalog(&state).await?))
}

pub(crate) async fn create_model_download_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let download = model_download(&model)
        .ok_or_else(|| ApiError::bad_request("Model does not define a Hugging Face download"))?;
    let repo = required_string_field(&download, "repo")?.to_owned();
    let files = download
        .get("files")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.clone()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&model_id)
                .to_owned(),
        ),
    );
    job_payload.insert(
        "provider".to_owned(),
        Value::String(required_string_field(&download, "provider")?.to_owned()),
    );
    job_payload.insert("repo".to_owned(), Value::String(repo.clone()));
    job_payload.insert("files".to_owned(), json!(files));
    // Forward the catalog-declared family so the worker can re-verify the downloaded
    // weights match it (parity with model import). The catalog is project-curated, but
    // a mis-declared family would otherwise silently mismatch downstream adapter
    // selection; the worker reconciles and fails on a confident conflict (sc-1663).
    if let Some(family) = model.get("family").and_then(Value::as_str) {
        if !family.trim().is_empty() {
            job_payload.insert("family".to_owned(), Value::String(family.to_owned()));
        }
    }
    job_payload.insert(
        "targetDir".to_owned(),
        Value::String(
            state
                .settings
                .data_dir
                .join("models")
                .join(safe_download_dir(&repo))
                .display()
                .to_string(),
        ),
    );

    let job = create_generation_job(
        state,
        JobType::ModelDownload,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Convert a model's native checkpoint into the local MLX format (macOS/Apple
/// Silicon). Only valid for models whose manifest declares `mlx.requiresConversion`
/// (Wan TI2V-5B, Wan I2V-A14B); turnkey MLX models need no conversion. The native
/// source checkpoint must already be downloaded; the Rust utility worker shells out
/// to the Python/MLX `mlx_video.convert_wan` tool.
pub(crate) async fn create_model_convert_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelConvertRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let mlx = model
        .get("mlx")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("Model has no MLX variant to convert"))?;
    let requires_conversion = mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let quantize = payload.quantize_bits.is_some();
    // Two sources: models that require conversion read the native diffusers
    // checkpoint (convertSourceRepo); turnkey MLX models (a pre-converted bf16
    // `repo`) have nothing to convert, but can still be quantized in place from
    // that repo (`--quantize-only`), so a quant request on a turnkey model is valid.
    let (source_repo, quantize_only) = if requires_conversion {
        let repo = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("MLX conversion source repo is not configured"))?;
        (repo.to_owned(), false)
    } else if quantize {
        let repo = mlx
            .get("repo")
            .and_then(Value::as_str)
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("Model has no MLX repo to quantize"))?;
        (repo.to_owned(), true)
    } else {
        return Err(ApiError::bad_request(
            "Model does not require MLX conversion",
        ));
    };
    let output_dir = state
        .settings
        .data_dir
        .join("models")
        .join("mlx")
        .join(&model_id);
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.clone()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&model_id)
                .to_owned(),
        ),
    );
    job_payload.insert("sourceRepo".to_owned(), Value::String(source_repo));
    job_payload.insert(
        "outputDir".to_owned(),
        Value::String(output_dir.display().to_string()),
    );
    job_payload.insert("dtype".to_owned(), Value::String("bfloat16".to_owned()));
    if quantize_only {
        job_payload.insert("quantizeOnly".to_owned(), Value::Bool(true));
    }
    if let Some(bits) = payload.quantize_bits {
        job_payload.insert("quantizeBits".to_owned(), Value::from(bits));
    }
    if let Some(group_size) = payload.quantize_group_size {
        job_payload.insert("quantizeGroupSize".to_owned(), Value::from(group_size));
    }

    let job = create_generation_job(
        state,
        JobType::ModelConvert,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let catalog = model_catalog(&state).await?;
    let model = catalog
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    let removed_entry =
        remove_catalog_manifest_entry(&state, &manifest_path, "models", &model_id).await?;
    let cleanup_source = removed_entry.as_ref().unwrap_or(&model);
    let mut removed_paths = Vec::new();
    let mut retained_paths = Vec::new();
    let allowed_roots = vec![
        state.settings.data_dir.join("models"),
        huggingface_hub_cache_dir(&state.settings.data_dir),
    ];
    for path in model_artifact_paths(cleanup_source, &state.settings.data_dir) {
        remove_owned_artifact_path(
            path,
            &allowed_roots,
            &mut removed_paths,
            &mut retained_paths,
        )
        .await?;
    }
    if removed_entry.is_none() && removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in model catalog entries are read-only unless local files are installed",
        ));
    }
    let warnings = catalog_delete_warnings(&state, "model", &model_id, None).await?;
    let policy = if removed_entry.is_some() {
        "Removed the model registry entry and SceneWorks-owned local model files."
    } else {
        "Built-in model catalog entries are retained; SceneWorks-owned local model files were removed."
    };
    Ok(Json(json!({
        "id": model_id,
        "kind": "model",
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removed_paths.is_empty(),
        "removedPaths": removed_paths,
        "retainedPaths": retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

pub(crate) async fn create_model_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if is_multipart {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()).into_response())?;
        let (payload, staged_path) = model_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_model_import_job(state, payload).await;
        if result.is_err() {
            cleanup_staged_model_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<ModelImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_model_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

pub(crate) async fn queue_model_import_job(
    state: AppState,
    mut payload: ModelImportRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, or source path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    let model_type = match payload.model_type.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => {
            let normalized = value.to_ascii_lowercase();
            if !ALLOWED_MODEL_TYPES.contains(&normalized.as_str()) {
                return Err(ApiError::bad_request(format!(
                    "Model type must be one of {}",
                    ALLOWED_MODEL_TYPES.join(", ")
                )));
            }
            normalized
        }
        _ => "image".to_owned(),
    };
    payload.model_type = Some(model_type.clone());
    if let Some(family) = payload.family.take() {
        let models = model_catalog(&state).await?;
        payload.family = Some(validate_lora_family(&models, &family)?);
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            payload
                .source_url
                .as_deref()
                .and_then(|value| lora_source_url_file_stem(value).ok())
        })
        .or_else(|| {
            payload.source_path.as_deref().and_then(|path| {
                FsPath::new(path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "Imported Model".to_owned());
    let model_id = payload
        .model_id
        .clone()
        .unwrap_or_else(|| slugify_lora_id(&name));
    let existing_ids = model_catalog(&state)
        .await?
        .into_iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    if existing_ids.contains(&model_id) {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' already exists. Pick a different id or delete the existing model first."
        )));
    }
    let target_name = safe_download_dir(&model_id);
    let target_dir = state
        .settings
        .data_dir
        .join("models")
        .join("imports")
        .join(&target_name);
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    let source_path_rel = format!("models/imports/{target_name}");
    let allowed_source_roots = vec![state.settings.data_dir.join("models")];
    if let Some(source_path) = payload.source_path.as_deref() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("model-uploads")]
        } else {
            allowed_source_roots
        };
        validate_lora_import_source_path(source_path, &allowed_source_roots)?;
        let detected =
            detect_model_family(FsPath::new(source_path)).map_err(model_family_inspection_error)?;
        payload.family = reconcile_model_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": model_id,
        "name": name,
        "type": model_type,
        "source": {
            "provider": model_import_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path_rel,
        },
        "files": payload.files.clone(),
        "paths": {
            "model": target_dir.display().to_string(),
        },
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
    if let Some(source_url) = payload.source_url.clone() {
        if let Some(source) = manifest_entry
            .get_mut("source")
            .and_then(Value::as_object_mut)
        {
            source.insert("url".to_owned(), Value::String(source_url));
        }
    }
    if let Some(family) = payload.family.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("family".to_owned(), Value::String(family));
        }
    }
    if let Some(object) = manifest_entry.as_object_mut() {
        apply_model_manifest_defaults(object, &model_type, payload.family.as_deref());
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("modelId".to_owned(), manifest_entry["id"].clone());
    payload.insert("modelName".to_owned(), manifest_entry["name"].clone());
    payload.insert(
        "targetDir".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    let job = create_generation_job(
        state,
        JobType::ModelImport,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn model_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(ModelImportRequest, PathBuf), ApiError> {
    let mut payload = ModelImportRequest {
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        family: None,
        uploaded_source_path: false,
    };
    let mut staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one model file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("model.safetensors"));
                let path =
                    write_model_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }

            let value = field
                .text()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match field_name.as_str() {
                "modelId" => payload.model_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "type" => payload.model_type = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "repo" => payload.repo = Some(value.to_owned()),
                "sourceUrl" => payload.source_url = Some(value.to_owned()),
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = parse_result {
        if let Some(path) = staged_path.as_deref() {
            cleanup_staged_model_upload(path).await;
        }
        return Err(error);
    }

    let Some(staged_path) = staged_path else {
        return Err(ApiError::bad_request("Upload file field is required"));
    };
    Ok((payload, staged_path))
}

pub(crate) async fn write_model_upload_field_to_staged_file(
    state: &AppState,
    mut field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("model-uploads")
        .join(format!("upload-{}", Uuid::new_v4().simple()));
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(filename);
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut uploaded_bytes = 0usize;
    let write_result = async {
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            uploaded_bytes = uploaded_bytes.saturating_add(chunk.len());
            if uploaded_bytes > max_model_upload_bytes() {
                return Err(ApiError::payload_too_large(format!(
                    "Uploaded model file exceeds the {} limit",
                    format_bytes(max_model_upload_bytes() as u64)
                )));
            }
            file.write_all(&chunk)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
        }
        file.flush()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        Ok(())
    }
    .await;
    if let Err(error) = write_result {
        drop(file);
        cleanup_staged_model_upload(&temp_path).await;
        return Err(error);
    }
    Ok(temp_path)
}

pub(crate) async fn cleanup_staged_model_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

pub(crate) fn model_import_source_provider(payload: &ModelImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

pub(crate) fn model_family_inspection_error(error: SafetensorsHeaderError) -> ApiError {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect model file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("Model file has an invalid safetensors header".to_owned())
        }
    }
}

/// Applies the import-time policy for base models: confident detection rejects
/// a mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection accepts the supplied family
/// unchanged (and leaves things unset if none was supplied).
pub(crate) fn reconcile_model_family(
    supplied: Option<String>,
    detected: Option<String>,
    _context: &str,
) -> Result<Option<String>, ApiError> {
    reconcile_detected_family(supplied, detected).map_err(|mismatch| {
        ApiError::bad_request(format!(
            "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
            mismatch.detected, mismatch.supplied, mismatch.detected
        ))
    })
}

pub(crate) fn max_model_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_MODEL_UPLOAD_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        if limit > 0 {
            return limit;
        }
    }
    MAX_MODEL_UPLOAD_BYTES
}

pub(crate) async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    let mut models = merge_entries_by_id(builtin, user);
    let download_contexts = models
        .iter()
        .map(model_download_context)
        .collect::<Result<Vec<_>, _>>()?;
    let download_size_bytes = join_all(download_contexts.iter().map(|context| async move {
        match context {
            Some(context) => {
                estimate_huggingface_download_size(state, &context.repo, &context.files).await
            }
            None => None,
        }
    }))
    .await;

    for (model, (download_context, download_size_bytes)) in models
        .iter_mut()
        .zip(download_contexts.into_iter().zip(download_size_bytes))
    {
        let fallback_size_bytes = download_context
            .as_ref()
            .and_then(|context| context.fallback_size_bytes);
        let effective_download_size_bytes = download_size_bytes.or(fallback_size_bytes);
        let download_size_estimated =
            download_size_bytes.is_none() && fallback_size_bytes.is_some();
        let (downloadable, installed_path, installed) =
            if let Some(download_context) = download_context {
                let managed_path = state
                    .settings
                    .data_dir
                    .join("models")
                    .join(safe_download_dir(&download_context.repo));
                let cache_path =
                    huggingface_repo_cache_path(&state.settings.data_dir, &download_context.repo);
                let cache_installed = cache_path
                    .as_ref()
                    .is_some_and(|path| huggingface_repo_cache_exists(path));
                let managed_installed = model_is_installed(&managed_path);
                let installed_path = if cache_installed {
                    cache_path
                } else {
                    Some(managed_path)
                };
                (
                    true,
                    installed_path.map(|path| path.display().to_string()),
                    managed_installed || cache_installed,
                )
            } else if let Some(installed_path) =
                model_manifest_installed_path(model, &state.settings.data_dir)
            {
                let installed = model_is_installed(&installed_path);
                (false, Some(installed_path.display().to_string()), installed)
            } else {
                (false, None, false)
            };
        let object = model
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
        let model_id = object.get("id").and_then(Value::as_str).unwrap_or_default();
        let user_managed = user_model_ids.contains(model_id);
        object.insert(
            "catalogScope".to_owned(),
            Value::String(if user_managed { "user" } else { "builtin" }.to_owned()),
        );
        object.insert("downloadable".to_owned(), Value::Bool(downloadable));
        object.insert(
            "downloadSizeBytes".to_owned(),
            effective_download_size_bytes
                .map(|value| json!(value))
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeLabel".to_owned(),
            effective_download_size_bytes
                .map(format_bytes)
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeEstimated".to_owned(),
            Value::Bool(download_size_estimated),
        );
        object.insert(
            "installState".to_owned(),
            Value::String(if installed { "installed" } else { "missing" }.to_owned()),
        );
        object.insert(
            "installedPath".to_owned(),
            installed_path.map(Value::String).unwrap_or(Value::Null),
        );
        object.insert(
            "removable".to_owned(),
            Value::Bool(user_managed || installed),
        );
        // Gated-model signal (sc-1898): a machine-readable `gated` flag plus the
        // credential host the download requires, so the Models screen can route the
        // user to the credential screen before a download will succeed. The host
        // honors an explicit manifest `credentialHost` and otherwise derives from
        // the download provider/source URL; `licenseUrl` passes through untouched.
        let gated = object
            .get("gated")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        object.insert("gated".to_owned(), Value::Bool(gated));
        if gated {
            let credential_host = object
                .get("credentialHost")
                .and_then(Value::as_str)
                .map(normalize_host)
                .filter(|host| !host.is_empty())
                .or_else(|| derive_credential_host(object));
            object.insert(
                "credentialHost".to_owned(),
                credential_host.map(Value::String).unwrap_or(Value::Null),
            );
        }
        // macOS Model Manager: MLX availability + conversion status for models that
        // declare an `mlx` variant. Additive fields the web/Docker build ignores; the
        // probes are cheap and portable, so a const `cfg!` check gates them rather
        // than per-OS compilation. minMemoryGb passes through from the raw manifest.
        let mlx_status = if cfg!(target_os = "macos") {
            mlx_catalog_status(object, &state.settings.data_dir)
        } else {
            None
        };
        if let Some(status) = mlx_status {
            object.insert(
                "mlxInstallState".to_owned(),
                Value::String(status.install_state.to_owned()),
            );
            object.insert(
                "mlxConversionState".to_owned(),
                Value::String(status.conversion_state.to_owned()),
            );
            object.insert(
                "mlxConvertedPath".to_owned(),
                status
                    .converted_path
                    .map(|path| Value::String(path.display().to_string()))
                    .unwrap_or(Value::Null),
            );
        }
    }
    models.sort_by(|left, right| {
        let left_key = (
            left.get("type").and_then(Value::as_str).unwrap_or_default(),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(models)
}

/// Resolve the merged model manifest entry for `model_id` so the GPU worker no
/// longer re-parses `builtin.models.jsonc`/`user.models.jsonc` itself — Rust is
/// the single owner of manifest parsing/merging (story 1653). The merged entry
/// is injected into video job payloads as `modelManifestEntry`. Returns `{}`
/// when the model is absent from both manifests, which the worker treats the
/// same as before (fall back to the model's default repo).
pub(crate) async fn resolve_model_manifest_entry(
    state: &AppState,
    model_id: &str,
) -> Result<Value, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let find = |entries: &[Value]| -> Option<Value> {
        entries
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
    };
    Ok(merge_model_manifest_entry(find(&builtin), find(&user)))
}

/// One-level-deep merge of the builtin and user manifest entries for a single
/// model id. Mirrors the worker's former `ltx_model_manifest_entry` exactly so
/// this migration is behavior-preserving: user top-level keys override builtin
/// (shallow), and the nested config blocks the adapters read are merged
/// key-by-key rather than replaced wholesale. (This is intentionally deeper than
/// `merge_entries_by_id`, which the model catalog uses for display.)
pub(crate) fn merge_model_manifest_entry(builtin: Option<Value>, user: Option<Value>) -> Value {
    const NESTED_KEYS: [&str; 6] = [
        "paths",
        "resources",
        "defaults",
        "limits",
        "loraCompatibility",
        "ui",
    ];
    match (builtin, user) {
        (builtin, None) => builtin.unwrap_or_else(|| Value::Object(JsonObject::new())),
        (None, Some(user)) => user,
        (Some(builtin), Some(user)) => {
            let mut merged = builtin.clone();
            merge_object(&mut merged, user.clone());
            for key in NESTED_KEYS {
                let builtin_nested = builtin.get(key).and_then(Value::as_object);
                let user_nested = user.get(key).and_then(Value::as_object);
                if builtin_nested.is_none() && user_nested.is_none() {
                    continue;
                }
                let mut nested = builtin_nested.cloned().unwrap_or_default();
                if let Some(user_nested) = user_nested {
                    for (nested_key, value) in user_nested {
                        nested.insert(nested_key.clone(), value.clone());
                    }
                }
                if let Some(object) = merged.as_object_mut() {
                    object.insert(key.to_owned(), Value::Object(nested));
                }
            }
            merged
        }
    }
}

pub(crate) fn model_download(model: &Value) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let mut fallback = None;
    for download in downloads {
        if !is_supported_model_download(download) {
            continue;
        }
        fallback.get_or_insert(download);
        if download.get("default").and_then(Value::as_bool) == Some(true) {
            return Some(download.clone());
        }
    }
    fallback.cloned()
}

/// Best-effort credential host for a gated model when the manifest entry doesn't
/// set `credentialHost` explicitly: an explicit per-download `credentialHost`,
/// else the well-known host for the provider (`huggingface` ⇒ `huggingface.co`),
/// else the host of a `sourceUrl`. Normalized (scheme/path stripped, lower-cased)
/// to match how credentials are keyed in the store.
fn derive_credential_host(model: &serde_json::Map<String, Value>) -> Option<String> {
    let downloads = model.get("downloads")?.as_array()?;
    for download in downloads {
        if let Some(host) = download
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
        if download.get("provider").and_then(Value::as_str) == Some("huggingface") {
            return Some("huggingface.co".to_owned());
        }
        if let Some(host) = download
            .get("sourceUrl")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
    }
    None
}

pub(crate) fn is_supported_model_download(download: &Value) -> bool {
    download.get("provider").and_then(Value::as_str) == Some("huggingface")
        && download
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| !repo.is_empty())
}

pub(crate) fn model_download_context(model: &Value) -> Result<Option<DownloadContext>, ApiError> {
    let Some(download) = model_download(model) else {
        return Ok(None);
    };
    Ok(Some(DownloadContext {
        repo: required_string_field(&download, "repo")?.to_owned(),
        files: string_array_field(&download, "files"),
        fallback_size_bytes: manifest_download_size_bytes(model, &download),
    }))
}

pub(crate) fn manifest_download_size_bytes(model: &Value, download: &Value) -> Option<u64> {
    // Prefer the selected download entry, then fall back to legacy model-level metadata.
    ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
        .iter()
        .find_map(|field| download.get(*field).and_then(json_size_to_u64))
        .or_else(|| {
            ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
                .iter()
                .find_map(|field| model.get(*field).and_then(json_size_to_u64))
        })
}

pub(crate) async fn estimate_huggingface_download_size(
    state: &AppState,
    repo: &str,
    files: &[String],
) -> Option<u64> {
    if matches!(
        std::env::var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    ) {
        return None;
    }
    let cache_key = (repo.to_owned(), files.to_vec());
    if let Some(cached) = state.model_size_cache.lock().get(&cache_key) {
        return Some(cached);
    }
    let url = format!(
        "https://huggingface.co/api/models/{}?blobs=true",
        quote_huggingface_repo(repo)
    );
    let estimate =
        estimate_huggingface_download_size_uncached(&state.http_client, &url, files).await;
    if let Some(estimate) = estimate {
        state.model_size_cache.lock().insert(cache_key, estimate);
    }
    estimate
}

pub(crate) async fn estimate_huggingface_download_size_uncached(
    client: &reqwest::Client,
    url: &str,
    files: &[String],
) -> Option<u64> {
    let payload = tokio::time::timeout(Duration::from_secs(8), async {
        client
            .get(url.to_owned())
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json::<Value>()
            .await
            .ok()
    })
    .await
    .ok()??;
    let siblings = payload.get("siblings")?.as_array()?;
    download_size_from_siblings(siblings, files)
}

pub(crate) fn download_size_from_siblings(siblings: &[Value], files: &[String]) -> Option<u64> {
    let mut total = 0_u64;
    let mut found_size = false;
    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !allow_pattern_matches(filename, files) {
            continue;
        }
        let Some(size) = sibling.get("size").and_then(json_size_to_u64) else {
            continue;
        };
        found_size = true;
        total = total.saturating_add(size);
    }
    found_size.then_some(total)
}

pub(crate) fn model_is_installed(path: &FsPath) -> bool {
    path.is_dir() && path.join(".sceneworks-download-complete.json").is_file()
}

pub(crate) struct MlxCatalogStatus {
    pub(crate) install_state: &'static str,
    pub(crate) conversion_state: &'static str,
    pub(crate) converted_path: Option<PathBuf>,
}

/// macOS Model Manager status for a model's `mlx` variant. Returns `None` when the
/// model declares no `mlx` block.
///
/// `conversion_state`:
/// - `ready`            turnkey MLX repo (no conversion needed)
/// - `converted`        requiresConversion and the local MLX dir exists
/// - `needs_conversion` source checkpoint present, MLX dir absent
/// - `needs_source`     source checkpoint not downloaded yet
///
/// `install_state` is `installed` when the usable MLX artifact exists.
pub(crate) fn mlx_catalog_status(
    model: &serde_json::Map<String, Value>,
    data_dir: &FsPath,
) -> Option<MlxCatalogStatus> {
    let mlx = model.get("mlx").and_then(Value::as_object)?;
    let repo_cached = |repo: &str| {
        huggingface_repo_cache_path(data_dir, repo)
            .as_deref()
            .is_some_and(huggingface_repo_cache_exists)
    };
    if mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let converted_dir = data_dir.join("models").join("mlx").join(model_id);
        if converted_dir.join("config.json").is_file() {
            return Some(MlxCatalogStatus {
                install_state: "installed",
                conversion_state: "converted",
                converted_path: Some(converted_dir),
            });
        }
        let source_present = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: "missing",
            conversion_state: if source_present {
                "needs_conversion"
            } else {
                "needs_source"
            },
            converted_path: None,
        })
    } else {
        let installed = mlx
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: if installed { "installed" } else { "missing" },
            conversion_state: "ready",
            converted_path: None,
        })
    }
}

pub(crate) fn model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = model_manifest_installed_path(model, data_dir) {
        paths.push(path);
    }
    if let Some(repo) = model_download(model).and_then(|download| {
        download
            .get("repo")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }) {
        paths.push(data_dir.join("models").join(safe_download_dir(&repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, &repo) {
            paths.push(cache_path);
        }
    }
    if let Some(source_path) = model
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        paths.push(if path.is_absolute() {
            path
        } else {
            data_dir.join(path)
        });
    }
    unique_paths(paths)
}

pub(crate) fn model_manifest_installed_path(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let raw_path = model
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if raw_path.contains("${") {
        return None;
    }
    let path = PathBuf::from(raw_path);
    Some(if path.is_absolute() {
        path
    } else {
        data_dir.join(path)
    })
}

#[cfg(test)]
mod gated_credential_tests {
    use super::*;
    use serde_json::json;

    fn map(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn derives_huggingface_host_from_provider() {
        let model = map(json!({
            "downloads": [{ "provider": "huggingface", "repo": "black-forest-labs/FLUX.1-dev", "files": [] }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("huggingface.co")
        );
    }

    #[test]
    fn prefers_explicit_download_credential_host() {
        let model = map(json!({
            "downloads": [{ "provider": "civitai", "credentialHost": "https://Civitai.com/", "sourceUrl": "https://civitai.com/api/x" }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("civitai.com")
        );
    }

    #[test]
    fn falls_back_to_source_url_host() {
        let model = map(json!({
            "downloads": [{ "provider": "url", "sourceUrl": "https://models.example.com/path/file.safetensors" }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("models.example.com")
        );
    }

    #[test]
    fn no_downloads_yields_none() {
        assert_eq!(derive_credential_host(&map(json!({}))), None);
    }
}
