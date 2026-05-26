use super::*;

pub(crate) async fn create_image_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ImageJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_image_job(&payload)?;
    let job_type = if payload.mode == "edit_image" {
        JobType::ImageEdit
    } else {
        JobType::ImageGenerate
    };
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    if payload.recipe_preset_id.is_none() {
        job_payload.remove("recipePresetId");
    }
    apply_recipe_preset_to_image_payload(&state, &payload, &mut job_payload).await?;
    let model_manifest_entry = resolve_model_manifest_entry(&state, &payload.model).await?;
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    validate_job_lora_compatibility(&state, Some(&payload.project_id), &mut job_payload, false)
        .await?;
    if payload.seed.is_none() {
        let count = job_payload
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(payload.count);
        job_payload.insert("seeds".to_owned(), random_image_seeds(count));
    }
    let job = create_generation_job(
        state,
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn create_vqa_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VqaJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_vqa_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        JobType::ImageVqa,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) fn validate_vqa_job(payload: &VqaJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.source_asset_id.trim().is_empty() {
        return Err(ApiError::bad_request("sourceAssetId is required"));
    }
    let question = payload.question.trim();
    if question.is_empty() || question.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "question must be between 1 and 4000 characters",
        ));
    }
    if !(16..=2048).contains(&payload.max_new_tokens) {
        return Err(ApiError::bad_request(
            "maxNewTokens must be between 16 and 2048",
        ));
    }
    Ok(())
}

pub(crate) async fn create_interleave_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<InterleaveJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_interleave_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        JobType::ImageInterleave,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) fn validate_interleave_job(payload: &InterleaveJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.trim().is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    // Upstream interleave_gen caps the run at 10 generated images.
    if !(1..=10).contains(&payload.max_images) {
        return Err(ApiError::bad_request("maxImages must be between 1 and 10"));
    }
    if payload
        .source_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "sourceAssetIds must not contain blank ids",
        ));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
    Ok(())
}

pub(crate) async fn apply_recipe_preset_to_image_payload(
    state: &AppState,
    payload: &ImageJobRequest,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(preset_id) = payload.recipe_preset_id.as_deref() else {
        return Ok(());
    };
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    let presets = recipe_preset_catalog(state, Some(&payload.project_id)).await?;
    let preset = presets
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(preset_id))
        .ok_or_else(|| ApiError::bad_request("Recipe preset not found"))?;

    let expanded_prompt = preset_prompt(&payload.prompt, preset);
    job_payload.insert("prompt".to_owned(), Value::String(expanded_prompt));
    if payload.model == default_image_model() {
        if let Some(model) = preset
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
    }
    apply_recipe_preset_defaults(preset, job_payload)?;
    job_payload.insert(
        "stylePreset".to_owned(),
        Value::String(preset_id.to_owned()),
    );
    merge_preset_loras_into_payload(state, &payload.project_id, preset_id, preset, job_payload)
        .await
}

/// Prepend a preset's declared LoRAs to whatever LoRAs the client already sent,
/// skipping ids that are already present. Records ids the catalog can't resolve
/// under advanced.presetMissingLoras and stamps advanced.recipePresetId. Shared
/// by the image and video job paths so preset-LoRA semantics stay identical.
pub(crate) async fn merge_preset_loras_into_payload(
    state: &AppState,
    project_id: &str,
    preset_id: &str,
    preset: &Value,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let loras = lora_catalog(state, Some(project_id)).await?;
    let existing_lora_ids = job_payload
        .get("loras")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut seen_lora_ids = existing_lora_ids;
    let mut preset_loras = Vec::new();
    let mut missing_lora_ids = Vec::new();
    for preset_lora in recipe_preset_loras(preset) {
        let Some(lora_id) = preset_lora_id(&preset_lora) else {
            continue;
        };
        let Some(lora) = loras
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(lora_id))
        else {
            missing_lora_ids.push(Value::String(lora_id.to_owned()));
            continue;
        };
        if seen_lora_ids.iter().any(|seen_id| seen_id == lora_id) {
            continue;
        }
        preset_loras.push(serialize_preset_lora(lora, &preset_lora, lora_id));
        seen_lora_ids.push(lora_id.to_owned());
    }
    let advanced = job_payload
        .entry("advanced".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if !advanced.is_object() {
        *advanced = Value::Object(JsonObject::new());
    }
    let advanced = advanced
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("advanced payload must be an object"))?;
    advanced.insert(
        "recipePresetId".to_owned(),
        Value::String(preset_id.to_owned()),
    );
    advanced.remove("recipePresetName");
    advanced.remove("recipePresetPrompt");
    if missing_lora_ids.is_empty() {
        advanced.remove("presetMissingLoras");
    } else {
        advanced.insert(
            "presetMissingLoras".to_owned(),
            Value::Array(missing_lora_ids),
        );
    }

    let user_loras = job_payload
        .remove("loras")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    preset_loras.extend(user_loras);
    job_payload.insert("loras".to_owned(), Value::Array(preset_loras));
    Ok(())
}

pub(crate) fn apply_recipe_preset_defaults(
    preset: &Value,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(defaults) = preset.get("defaults").and_then(Value::as_object) else {
        return Ok(());
    };
    if let Some(count) = defaults.get("count").and_then(Value::as_u64) {
        let count = u32::try_from(count)
            .map_err(|_| ApiError::bad_request("Recipe preset count is out of range"))?;
        if !(1..=8).contains(&count) {
            return Err(ApiError::bad_request(
                "Recipe preset count must be between 1 and 8",
            ));
        }
        job_payload.insert("count".to_owned(), json!(count));
    }
    if let Some(resolution) = defaults.get("resolution").and_then(Value::as_str) {
        let (width, height) = parse_recipe_preset_resolution(resolution)?;
        validate_dimension(width, "width", MAX_IMAGE_DIMENSION)?;
        validate_dimension(height, "height", MAX_IMAGE_DIMENSION)?;
        job_payload.insert("width".to_owned(), json!(width));
        job_payload.insert("height".to_owned(), json!(height));
        let advanced = job_payload
            .entry("advanced".to_owned())
            .or_insert_with(|| Value::Object(JsonObject::new()));
        if !advanced.is_object() {
            *advanced = Value::Object(JsonObject::new());
        }
        advanced
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("advanced payload must be an object"))?
            .insert(
                "resolution".to_owned(),
                Value::String(resolution.to_owned()),
            );
    }
    if let Some(negative_prompt) = defaults.get("negativePrompt").and_then(Value::as_str) {
        job_payload.insert(
            "negativePrompt".to_owned(),
            Value::String(negative_prompt.to_owned()),
        );
    }
    Ok(())
}

pub(crate) fn parse_recipe_preset_resolution(value: &str) -> Result<(u32, u32), ApiError> {
    let Some((width, height)) = value.split_once('x') else {
        return Err(ApiError::bad_request(
            "Recipe preset resolution must use WIDTHxHEIGHT",
        ));
    };
    let width = width
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("Recipe preset width must be a number"))?;
    let height = height
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("Recipe preset height must be a number"))?;
    Ok((width, height))
}

/// Server-side expansion of a video job's recipe preset, mirroring
/// apply_recipe_preset_to_image_payload: the client sends the raw prompt plus
/// recipePresetId and the server folds in the preset's prompt prefix/suffix,
/// model, render defaults, and LoRAs. Keeps preset semantics identical across
/// the image and video studios.
pub(crate) async fn apply_recipe_preset_to_video_payload(
    state: &AppState,
    payload: &VideoJobRequest,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(preset_id) = payload.recipe_preset_id.as_deref() else {
        return Ok(());
    };
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    let presets = recipe_preset_catalog(state, Some(&payload.project_id)).await?;
    let preset = presets
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(preset_id))
        .ok_or_else(|| ApiError::bad_request("Recipe preset not found"))?;

    let expanded_prompt = preset_prompt(&payload.prompt, preset);
    job_payload.insert("prompt".to_owned(), Value::String(expanded_prompt));
    if payload.model == default_video_model() {
        if let Some(model) = preset
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
    }
    apply_recipe_preset_video_defaults(preset, job_payload)?;
    merge_preset_loras_into_payload(state, &payload.project_id, preset_id, preset, job_payload)
        .await
}

pub(crate) fn apply_recipe_preset_video_defaults(
    preset: &Value,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(defaults) = preset.get("defaults").and_then(Value::as_object) else {
        return Ok(());
    };
    if let Some(duration) = defaults.get("duration") {
        if !duration
            .as_f64()
            .is_some_and(|value| value.is_finite() && (1.0..=30.0).contains(&value))
        {
            return Err(ApiError::bad_request(
                "Recipe preset duration must be between 1 and 30",
            ));
        }
        job_payload.insert("duration".to_owned(), duration.clone());
    }
    if let Some(fps) = defaults.get("fps") {
        if !fps.as_u64().is_some_and(|value| (1..=60).contains(&value)) {
            return Err(ApiError::bad_request(
                "Recipe preset fps must be between 1 and 60",
            ));
        }
        job_payload.insert("fps".to_owned(), fps.clone());
    }
    if let Some(quality) = defaults.get("quality").and_then(Value::as_str) {
        job_payload.insert("quality".to_owned(), Value::String(quality.to_owned()));
    }
    if let Some(resolution) = defaults.get("resolution").and_then(Value::as_str) {
        let (width, height) = parse_recipe_preset_resolution(resolution)?;
        validate_dimension(width, "width", MAX_VIDEO_DIMENSION)?;
        validate_dimension(height, "height", MAX_VIDEO_DIMENSION)?;
        job_payload.insert("width".to_owned(), json!(width));
        job_payload.insert("height".to_owned(), json!(height));
    }
    if let Some(negative_prompt) = defaults.get("negativePrompt").and_then(Value::as_str) {
        job_payload.insert(
            "negativePrompt".to_owned(),
            Value::String(negative_prompt.to_owned()),
        );
    }
    Ok(())
}

pub(crate) async fn create_video_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VideoJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_video_job(&payload)?;
    let job_type = match payload.mode.as_str() {
        "extend_clip" => JobType::VideoExtend,
        "video_bridge" => JobType::VideoBridge,
        "replace_person" => JobType::PersonReplace,
        _ => JobType::VideoGenerate,
    };
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    if payload.recipe_preset_id.is_none() {
        job_payload.remove("recipePresetId");
    }
    apply_recipe_preset_to_video_payload(&state, &payload, &mut job_payload).await?;
    // Resolve the model manifest entry here so the GPU worker never re-parses
    // builtin/user.models.jsonc itself — Rust owns manifest parsing/merging
    // (story 1653). An unknown model resolves to {}, matching the worker's
    // existing fallback to the model's default repo.
    let model_manifest_entry = resolve_model_manifest_entry(&state, &payload.model).await?;
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
    validate_job_lora_compatibility(&state, Some(&payload.project_id), &mut job_payload, false)
        .await?;
    let job = create_generation_job(
        state,
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}
