use super::*;

pub(crate) async fn list_loras(
    State(state): State<AppState>,
    Query(query): Query<LorasQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let mut items = lora_catalog(&state, query.project_id.as_deref()).await?;
    if let Some(model_family) = query.model_family {
        items.retain(|item| {
            lora_families(item)
                .iter()
                .any(|family| family == &model_family)
        });
    }
    Ok(Json(items))
}

pub(crate) async fn delete_lora(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
    Query(query): Query<CatalogDeleteQuery>,
) -> Result<Json<Value>, ApiError> {
    let catalog = lora_catalog(&state, query.project_id.as_deref()).await?;
    let lora = catalog
        .into_iter()
        .find(|item| {
            item.get("id").and_then(Value::as_str) == Some(lora_id.as_str())
                && query.scope.as_deref().map_or(true, |scope| {
                    item.get("scope").and_then(Value::as_str) == Some(scope)
                })
        })
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "LoRA not found".to_owned(),
        })?;
    let scope = query
        .scope
        .as_deref()
        .or_else(|| lora.get("scope").and_then(Value::as_str))
        .unwrap_or("global");
    let (manifest_path, allowed_roots, default_root) = match scope {
        "global" => (
            Some(
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
            ),
            vec![state.settings.data_dir.join("loras")],
            state.settings.data_dir.clone(),
        ),
        "project" => {
            let Some(project_id) = query.project_id.as_deref() else {
                return Err(ApiError::bad_request(
                    "Project LoRA deletion requires projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), project_id).await?;
            (
                Some(project_path.join("loras").join("manifest.jsonc")),
                vec![
                    state.settings.data_dir.join("loras"),
                    project_path.join("loras"),
                ],
                project_path,
            )
        }
        "builtin" => (
            None,
            vec![state.settings.data_dir.join("loras")],
            state.settings.data_dir.clone(),
        ),
        _ => return Err(ApiError::bad_request("Unsupported LoRA scope")),
    };
    let removed_entry = if let Some(manifest_path) = manifest_path.as_deref() {
        remove_catalog_manifest_entry(&state, manifest_path, "loras", &lora_id).await?
    } else {
        None
    };
    let cleanup_source = removed_entry.as_ref().unwrap_or(&lora);
    let mut removed_paths = Vec::new();
    let mut retained_paths = Vec::new();
    for path in lora_artifact_paths(cleanup_source, &default_root) {
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
            "Built-in LoRA catalog entries are read-only unless local files are installed",
        ));
    }
    let warnings =
        catalog_delete_warnings(&state, "lora", &lora_id, query.project_id.as_deref()).await?;
    let policy = if removed_entry.is_some() {
        "Removed the LoRA registry entry and SceneWorks-owned local LoRA files."
    } else {
        "Built-in LoRA catalog entries are retained; SceneWorks-owned local LoRA files were removed."
    };
    Ok(Json(json!({
        "id": lora_id,
        "kind": "lora",
        "scope": scope,
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removed_paths.is_empty(),
        "removedPaths": removed_paths,
        "retainedPaths": retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

pub(crate) async fn lora_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.loras.jsonc"), "loras").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.loras.jsonc"), "loras").await?;
    let data_dir = state.settings.data_dir.clone();
    let builtin_manifest = manifest_dir.join("builtin.loras.jsonc");
    let user_manifest = manifest_dir.join("user.loras.jsonc");
    // sc-4202 (F-API-3): normalize_lora_entry probes the filesystem for installed
    // artifact paths; run the builtin+user normalize off the async executor.
    let mut loras = {
        let data_dir = data_dir.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
            let mut loras = Vec::new();
            for lora in builtin {
                loras.push(normalize_lora_entry(
                    lora,
                    "builtin",
                    &builtin_manifest,
                    &data_dir,
                    &data_dir,
                )?);
            }
            let user = user
                .into_iter()
                .map(|lora| {
                    normalize_lora_entry(lora, "global", &user_manifest, &data_dir, &data_dir)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(merge_entries_by_id(loras, user))
        })
        .await
        .map_err(|err| ApiError::internal(format!("LoRA catalog normalize task failed: {err}")))??
    };
    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("loras").join("manifest.jsonc");
        let entries = load_manifest_entries(state, &project_manifest, "loras").await?;
        let data_dir = data_dir.clone();
        let project_loras = tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
            entries
                .into_iter()
                .map(|lora| {
                    normalize_lora_entry(
                        lora,
                        "project",
                        &project_manifest,
                        &project_path,
                        &data_dir,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|err| {
            ApiError::internal(format!("LoRA catalog project normalize task failed: {err}"))
        })??;
        loras = merge_entries_by_id(loras, project_loras);
    }
    for lora in &mut loras {
        let object = lora
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("LoRA manifest entry must be an object"))?;
        let scope = object
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("builtin");
        let installed = object
            .get("installState")
            .and_then(Value::as_str)
            .is_some_and(|state| state == "installed");
        object.insert(
            "removable".to_owned(),
            Value::Bool(scope != "builtin" || installed),
        );
    }
    loras.sort_by(|left, right| {
        let left_key = (
            left.get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("family")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("family")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(loras)
}

pub(crate) fn normalize_lora_entry(
    mut lora: Value,
    scope: &str,
    manifest_path: &FsPath,
    default_root: &FsPath,
    data_dir: &FsPath,
) -> Result<Value, ApiError> {
    let object = lora
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("LoRA manifest entry must be an object"))?;
    object
        .entry("scope".to_owned())
        .or_insert_with(|| Value::String(scope.to_owned()));
    let source_path = object
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .or_else(|| object.get("path").and_then(Value::as_str));
    let local_path = source_path.map(|source_path| {
        let path = PathBuf::from(source_path);
        if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        }
    });
    let lora_snapshot = Value::Object(object.clone());
    let installed_path = match local_path.as_ref() {
        Some(path) if lora_is_installed(path) => Some(path.clone()),
        _ => match lora_huggingface_cached_file(&lora_snapshot, data_dir) {
            Some(path) if lora_is_installed(&path) => Some(path),
            _ => local_path.clone(),
        },
    };
    let install_state = match installed_path.as_ref() {
        Some(path) if lora_is_installed(path) => "installed",
        _ => "missing",
    };
    object.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    object.insert(
        "installedPath".to_owned(),
        installed_path
            .map(|path| Value::String(path.display().to_string()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "installState".to_owned(),
        Value::String(install_state.to_owned()),
    );
    Ok(lora)
}

pub(crate) async fn create_lora_import_job(
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
        let (payload, staged_paths) = lora_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_lora_import_job(state, payload).await;
        if result.is_err() {
            for path in &staged_paths {
                cleanup_staged_lora_upload(path).await;
            }
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<LoraImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_lora_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

pub(crate) async fn queue_lora_import_job(
    state: AppState,
    mut payload: LoraImportRequest,
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
    if !matches!(payload.scope.as_str(), "global" | "project") {
        return Err(ApiError::bad_request(
            "LoRA scope must be global or project",
        ));
    }
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
        .unwrap_or_else(|| "Imported LoRA".to_owned());
    let lora_id = payload
        .lora_id
        .clone()
        .unwrap_or_else(|| slugify_lora_id(&name));
    let target_name = safe_download_dir(&lora_id);
    let (target_dir, manifest_path, source_path, project_id, project_name, allowed_source_roots) =
        if payload.scope == "project" {
            let Some(project_id) = payload.project_id.clone() else {
                return Err(ApiError::bad_request(
                    "Project LoRA imports require projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), &project_id).await?;
            (
                project_path
                    .join("loras")
                    .join("imports")
                    .join(&target_name),
                project_path.join("loras").join("manifest.jsonc"),
                format!("loras/imports/{target_name}"),
                Some(project_id),
                None,
                vec![
                    state.settings.data_dir.join("loras"),
                    project_path.join("loras"),
                ],
            )
        } else {
            (
                state.settings.data_dir.join("loras").join(&target_name),
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
                format!("loras/{target_name}"),
                None,
                None,
                vec![state.settings.data_dir.join("loras")],
            )
        };
    if let Some(source_path) = payload.source_path.as_deref() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("lora-uploads")]
        } else {
            allowed_source_roots
        };
        validate_lora_import_source_path(source_path, &allowed_source_roots)?;
        // A paired Wan A14B MoE upload (sc-1991) carries a second low-noise file.
        // Validate it against the same upload root and record both halves under the
        // high/low_noise convention so the worker resolves the high half as primary
        // (transformer) and the low half as the transformer_2 sibling.
        if let Some(secondary_source_path) = payload.secondary_source_path.as_deref() {
            validate_lora_import_source_path(secondary_source_path, &allowed_source_roots)?;
            let (high_name, low_name) = wan_moe_pair_filenames(&target_name);
            payload.files = vec![high_name, low_name];
        }
        let detected = detect_family_from_local_path(source_path)?;
        payload.family = reconcile_lora_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": lora_id,
        "name": name,
        "scope": payload.scope.clone(),
        "source": {
            "provider": lora_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path,
        },
        "files": payload.files.clone(),
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
    if let Some(base_model) = payload.base_model.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("baseModel".to_owned(), Value::String(base_model));
        }
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("loraId".to_owned(), manifest_entry["id"].clone());
    payload.insert("name".to_owned(), manifest_entry["name"].clone());
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
        JobType::LoraImport,
        project_id,
        project_name,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn lora_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(LoraImportRequest, Vec<PathBuf>), ApiError> {
    let mut payload = LoraImportRequest {
        lora_id: None,
        name: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        family: None,
        base_model: None,
        scope: default_lora_scope(),
        project_id: None,
        uploaded_source_path: false,
        secondary_source_path: None,
    };
    let mut staged_path = None;
    // Wan A14B MoE imports (sc-1991) carry a second `secondaryFile` part for the
    // low-noise expert half. Staged separately so a failed queue cleans up both.
    let mut secondary_staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one LoRA file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("lora.safetensors"));
                let path =
                    write_lora_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }
            if field_name == "secondaryFile" {
                if secondary_staged_path.is_some() {
                    return Err(ApiError::bad_request(
                        "Only one low-noise expert file can be uploaded",
                    ));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("low_noise.safetensors"));
                let path =
                    write_lora_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.secondary_source_path = Some(path.display().to_string());
                secondary_staged_path = Some(path);
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
                "loraId" => payload.lora_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "baseModel" => payload.base_model = Some(value.to_owned()),
                "scope" => payload.scope = value.to_owned(),
                "projectId" => payload.project_id = Some(value.to_owned()),
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    let staged_paths: Vec<PathBuf> = staged_path
        .iter()
        .chain(secondary_staged_path.iter())
        .cloned()
        .collect();
    if let Err(error) = parse_result {
        for path in &staged_paths {
            cleanup_staged_lora_upload(path).await;
        }
        return Err(error);
    }

    if staged_path.is_none() {
        for path in &staged_paths {
            cleanup_staged_lora_upload(path).await;
        }
        return Err(ApiError::bad_request("Upload file field is required"));
    }
    Ok((payload, staged_paths))
}

pub(crate) async fn write_lora_upload_field_to_staged_file(
    state: &AppState,
    mut field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("lora-uploads")
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
            if uploaded_bytes > max_lora_upload_bytes() {
                return Err(ApiError::payload_too_large(
                    "Uploaded LoRA file exceeds the 2GB limit",
                ));
            }
            file.write_all(&chunk)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
        }
        file.flush()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))
    }
    .await;
    if let Err(error) = write_result {
        cleanup_staged_lora_upload(&temp_path).await;
        return Err(error);
    }
    Ok(temp_path)
}

pub(crate) async fn cleanup_staged_lora_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

pub(crate) fn max_lora_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_LORA_UPLOAD_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        if limit > 0 {
            return limit;
        }
    }
    MAX_UPLOAD_BYTES
}

pub(crate) async fn validate_job_lora_compatibility(
    state: &AppState,
    project_id: Option<&str>,
    job_payload: &mut JsonObject,
    allow_inline_loras: bool,
) -> Result<(), ApiError> {
    let Some(loras) = job_payload
        .get("loras")
        .and_then(Value::as_array)
        .filter(|loras| !loras.is_empty())
        .cloned()
    else {
        return Ok(());
    };
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Model is required for LoRA compatibility"))?;
    let model_id = model_id.to_owned();
    let models = model_catalog(state).await?;
    let catalog_loras = lora_catalog(state, project_id).await?;
    // sc-4202 (F-API-3): validate_lora_specs_for_model reads safetensors headers off
    // disk (validate_lora_safetensors_header) inline. Run it on the blocking pool so a
    // slow/network volume can't stall a tokio worker thread on the job-creation path.
    let normalized = tokio::task::spawn_blocking(move || {
        validate_lora_specs_for_model(
            &models,
            &catalog_loras,
            &model_id,
            &loras,
            allow_inline_loras,
            "LoRA",
        )
    })
    .await
    .map_err(|err| {
        ApiError::internal(format!("LoRA compatibility validation task failed: {err}"))
    })??;
    job_payload.insert("loras".to_owned(), Value::Array(normalized));
    Ok(())
}

pub(crate) fn validate_lora_specs_for_model(
    models: &[Value],
    catalog_loras: &[Value],
    model_id: &str,
    attached_loras: &[Value],
    allow_inline_loras: bool,
    lora_label: &str,
) -> Result<Vec<Value>, ApiError> {
    if attached_loras.is_empty() {
        return Ok(Vec::new());
    }
    let Some(model) = models
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
    else {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} not found; cannot verify LoRA compatibility"
        )));
    };
    let model_families = model_lora_families(model);
    if model_families.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} has no declared LoRA families"
        )));
    }
    let mut normalized_loras = Vec::new();
    for attached_lora in attached_loras {
        let Some((lora_id, lora, normalized_lora, catalog_backed)) =
            hydrate_lora_spec(catalog_loras, attached_lora, allow_inline_loras, lora_label)?
        else {
            continue;
        };
        let install_state = lora.get("installState").and_then(Value::as_str);
        if install_state.is_some_and(|state| state != "installed")
            || (catalog_backed && install_state.is_none())
        {
            return Err(ApiError::bad_request(format!(
                "{lora_label} is not installed: {lora_id}"
            )));
        }
        let header = validate_lora_safetensors_header(lora_id, lora)?;
        if let Some(detected_family) = header.as_ref().and_then(detect_lora_family) {
            if !model_families
                .iter()
                .any(|model_family| model_family == &detected_family)
            {
                let model_family_list = model_families.join(", ");
                return Err(ApiError::bad_request(format!(
                    "LoRA {lora_id} appears to be a {detected_family} LoRA, which is not compatible with model {model_id} ({model_family_list})"
                )));
            }
        }
        let families = lora_families(lora);
        if families.is_empty() {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} has no declared family; cannot verify compatibility with model {model_id}"
            )));
        }
        if !families.iter().any(|family| {
            model_families
                .iter()
                .any(|model_family| model_family == family)
        }) {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} is not compatible with model {model_id}"
            )));
        }
        // Base-model gating: for families where a matching family is insufficient
        // (Wan 5B vs 14B both declare `wan-video` but have incompatible
        // architectures — 48 vs 16 latent channels), a LoRA that records its
        // trained base model only loads on that exact model. LoRAs without a
        // recorded base model fall back to family gating (legacy/imported), so this
        // never tightens behavior for existing LoRAs.
        if families.iter().any(|family| family == "wan-video") {
            if let Some(base_model) = lora_base_model(lora) {
                if base_model != model_id {
                    return Err(ApiError::bad_request(format!(
                        "LoRA {lora_id} was trained for base model {base_model}, not {model_id}; \
                         Wan 5B and 14B LoRAs are not interchangeable"
                    )));
                }
            }
        }
        normalized_loras.push(normalized_lora);
    }
    Ok(normalized_loras)
}

pub(crate) fn hydrate_lora_spec<'a>(
    catalog_loras: &'a [Value],
    attached_lora: &'a Value,
    allow_inline_loras: bool,
    lora_label: &str,
) -> Result<Option<(&'a str, &'a Value, Value, bool)>, ApiError> {
    let Some(lora_id) = job_lora_id(attached_lora) else {
        return Ok(None);
    };
    let catalog_lora = if allow_inline_loras {
        None
    } else {
        catalog_loras
            .iter()
            .find(|lora| lora.get("id").and_then(Value::as_str) == Some(lora_id))
    };
    if catalog_lora.is_none() && !allow_inline_loras {
        return Err(ApiError::bad_request(format!(
            "{lora_label} not found: {lora_id}"
        )));
    }
    let source_lora = catalog_lora.unwrap_or(attached_lora);
    let normalized_lora = match catalog_lora {
        Some(catalog_lora) => serialize_job_lora(catalog_lora, attached_lora, lora_id),
        None => normalize_inline_job_lora(attached_lora, lora_id),
    };
    Ok(Some((
        lora_id,
        source_lora,
        normalized_lora,
        catalog_lora.is_some(),
    )))
}

/// Returns the parsed safetensors header for `lora` when one is available
/// on disk. Returns `Ok(None)` when the manifest entry has no installed
/// path or no `.safetensors` file is present under it (the same "skip"
/// semantics this helper has always had). Returns an error if the file
/// exists but the header is malformed.
pub(crate) fn validate_lora_safetensors_header(
    lora_id: &str,
    lora: &Value,
) -> Result<Option<Value>, ApiError> {
    let Some(path) = lora
        .get("installedPath")
        .or_else(|| lora.get("sourcePath"))
        .or_else(|| lora.get("path"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let Some(safetensors_path) = first_safetensors_path(&path) else {
        return Ok(None);
    };
    read_safetensors_header_for_api(lora_id, &safetensors_path).map(Some)
}

pub(crate) fn read_safetensors_header_for_api(
    lora_id: &str,
    path: &FsPath,
) -> Result<Value, ApiError> {
    read_safetensors_header(path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect LoRA {lora_id}: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request(format!("LoRA {lora_id} has an invalid safetensors header"))
        }
    })
}

pub(crate) fn sweep_stale_lora_uploads(data_dir: &FsPath) -> std::io::Result<usize> {
    sweep_stale_lora_uploads_before(
        data_dir,
        SystemTime::now() - Duration::from_secs(STALE_LORA_UPLOAD_SECONDS),
    )
}

pub(crate) fn sweep_stale_lora_uploads_before(
    data_dir: &FsPath,
    cutoff: SystemTime,
) -> std::io::Result<usize> {
    let upload_root = data_dir.join("cache").join("lora-uploads");
    let entries = match std::fs::read_dir(upload_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0usize;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() {
            continue;
        }
        let filename = entry.file_name();
        let filename = filename.to_string_lossy();
        if !filename.starts_with("upload-") {
            continue;
        }
        let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
        if modified <= cutoff {
            std::fs::remove_dir_all(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

pub(crate) fn lora_source_provider(payload: &LoraImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

/// The `<stem>.high_noise.safetensors` / `<stem>.low_noise.safetensors` filenames
/// for a paired Wan A14B MoE LoRA stored under one record (sc-1991). The high-noise
/// file sorts first, so it resolves as the primary (transformer) and the low-noise
/// file as the `transformer_2` sibling. Must match the worker's identical
/// convention so the manifest `files` agree with the on-disk layout.
pub(crate) fn wan_moe_pair_filenames(stem: &str) -> (String, String) {
    (
        format!("{stem}.high_noise.safetensors"),
        format!("{stem}.low_noise.safetensors"),
    )
}

pub(crate) fn lora_url_error_message(error: LoraUrlError) -> &'static str {
    error.message()
}

/// Parses the safetensors header at `source_path` (or the first
/// `.safetensors` file under it) and runs the architecture detector.
/// Returns `Ok(None)` when no header is available or the signature is
/// inconclusive. Returns `Err` only when the file exists but its header
/// is malformed — that mirrors the pre-existing validation behaviour and
/// gives the user a clear "the file is broken" message instead of a
/// silent acceptance.
pub(crate) fn detect_family_from_local_path(source_path: &str) -> Result<Option<String>, ApiError> {
    let path = FsPath::new(source_path);
    let Some(safetensors_path) = first_safetensors_path(path) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect LoRA file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("LoRA file has an invalid safetensors header".to_owned())
        }
    })?;
    Ok(detect_lora_family(&header))
}

/// Applies the import-time family policy: confident detection rejects a
/// mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection logs a warning and accepts the
/// supplied family unchanged.
pub(crate) fn reconcile_lora_family(
    supplied: Option<String>,
    detected: Option<String>,
    context: &str,
) -> Result<Option<String>, ApiError> {
    match (supplied, detected) {
        (Some(supplied), Some(detected)) => {
            if supplied == detected {
                Ok(Some(supplied))
            } else {
                Err(ApiError::bad_request(format!(
                    "LoRA file appears to be a {detected} model, but family was declared as {supplied}. Re-import with family {detected} or pick a different file."
                )))
            }
        }
        (None, Some(detected)) => Ok(Some(detected)),
        (Some(supplied), None) => {
            println!(
                "LoRA import {context}: architecture detection inconclusive; accepting supplied family {supplied}"
            );
            Ok(Some(supplied))
        }
        (None, None) => Ok(None),
    }
}

pub(crate) fn lora_is_installed(path: &FsPath) -> bool {
    first_safetensors_path(path).is_some()
}

pub(crate) fn lora_artifact_paths(lora: &Value, default_root: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let is_huggingface_source = lora
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)
        == Some("huggingface");
    if !is_huggingface_source {
        if let Some(installed_path) = lora
            .get("installedPath")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.contains("${"))
        {
            paths.push(PathBuf::from(installed_path));
        }
    }
    if let Some(source_path) = lora
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .or_else(|| lora.get("path").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        paths.push(if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        });
    }
    unique_paths(paths)
}

pub(crate) fn lora_huggingface_cached_file(lora: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let source = lora.get("source").and_then(Value::as_object);
    let provider = source
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)?;
    if provider != "huggingface" {
        return None;
    }
    let repo = source
        .and_then(|source| source.get("repo"))
        .or_else(|| lora.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    if !repo_root.exists() {
        return None;
    }
    let file_name = source
        .and_then(|source| source.get("file"))
        .or_else(|| lora.get("file"))
        .and_then(Value::as_str)
        .or_else(|| {
            source
                .and_then(|source| source.get("files"))
                .or_else(|| lora.get("files"))
                .and_then(Value::as_array)
                .and_then(|files| files.first())
                .and_then(Value::as_str)
        });
    if let Some(file_name) = file_name {
        for snapshot in huggingface_snapshot_dirs(&repo_root) {
            let candidate = snapshot.join(file_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    huggingface_main_snapshot_dir(&repo_root)
        .and_then(|snapshot| first_safetensors_path(&snapshot))
        .or_else(|| first_safetensors_path(&repo_root))
}

pub(crate) fn lora_families(lora: &Value) -> Vec<String> {
    families_from_value_chain(
        lora,
        &["families", "compatibleFamilies", "modelFamilies"],
        Some("compatibility"),
    )
}

/// The specific base model a LoRA records it was trained for (e.g. `wan_2_2`,
/// `wan_2_2_t2v_14b`), or None. Used to gate families where a matching family is
/// not sufficient (Wan 5B and 14B both declare `wan-video` but have incompatible
/// architectures). Not normalized like families — model ids are exact strings.
pub(crate) fn lora_base_model(lora: &Value) -> Option<String> {
    for key in ["baseModel", "base_model"] {
        if let Some(value) = lora.get(key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod base_model_gating_tests {
    use super::*;

    fn wan_models() -> Vec<Value> {
        vec![
            json!({ "id": "wan_2_2", "loraCompatibility": { "families": ["wan-video"] } }),
            json!({ "id": "wan_2_2_t2v_14b", "loraCompatibility": { "families": ["wan-video"] } }),
        ]
    }

    #[test]
    fn rejects_wan_5b_lora_on_14b_model() {
        let models = wan_models();
        let lora = json!({ "id": "char", "families": ["wan-video"], "baseModel": "wan_2_2" });
        let err =
            validate_lora_specs_for_model(&models, &[], "wan_2_2_t2v_14b", &[lora], true, "LoRA")
                .expect_err("5B LoRA must be rejected on the 14B model");
        assert!(
            format!("{err:?}").contains("not interchangeable"),
            "got: {err:?}"
        );
    }

    #[test]
    fn accepts_wan_lora_on_matching_base_model() {
        let models = wan_models();
        let lora = json!({ "id": "char", "families": ["wan-video"], "baseModel": "wan_2_2" });
        validate_lora_specs_for_model(&models, &[], "wan_2_2", &[lora], true, "LoRA")
            .expect("exact base-model match must pass");
    }

    #[test]
    fn lora_without_base_model_falls_back_to_family_gating() {
        let models = wan_models();
        // No recorded baseModel (legacy/imported) -> family gating only, no rejection.
        let lora = json!({ "id": "legacy", "families": ["wan-video"] });
        validate_lora_specs_for_model(&models, &[], "wan_2_2_t2v_14b", &[lora], true, "LoRA")
            .expect("family-only LoRA must still pass");
    }
}
