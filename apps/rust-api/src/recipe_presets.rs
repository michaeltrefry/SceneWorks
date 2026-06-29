use super::*;

pub(crate) async fn list_recipe_presets(
    State(state): State<AppState>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let mut presets = recipe_preset_catalog(&state, query.project_id.as_deref()).await?;
    if !query.include_archived.unwrap_or(false) {
        presets.retain(|preset| !recipe_preset_archived(preset));
    }
    if let Some(model) = query.model.as_deref() {
        presets.retain(|preset| preset.get("model").and_then(Value::as_str) == Some(model));
    }
    if let Some(workflow) = query.workflow.as_deref() {
        presets.retain(|preset| preset.get("workflow").and_then(Value::as_str) == Some(workflow));
    }
    if let Some(scope) = query.scope.as_deref() {
        presets.retain(|preset| preset.get("scope").and_then(Value::as_str) == Some(scope));
    }
    Ok(Json(presets))
}

pub(crate) async fn get_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let preset = recipe_preset_catalog(&state, query.project_id.as_deref())
        .await?
        .into_iter()
        .find(|preset| preset.get("id").and_then(Value::as_str) == Some(preset_id.as_str()))
        .filter(|preset| {
            query.scope.as_deref().map_or(true, |scope| {
                preset.get("scope").and_then(Value::as_str) == Some(scope)
            })
        })
        .filter(|preset| query.include_archived.unwrap_or(false) || !recipe_preset_archived(preset))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Recipe preset not found".to_owned(),
        })?;
    Ok(Json(preset))
}

pub(crate) async fn create_recipe_preset(
    State(state): State<AppState>,
    Query(query): Query<RecipePresetsQuery>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    validate_recipe_preset_query(&query)?;
    let mut preset = recipe_preset_from_payload(payload)?;
    let scope = recipe_preset_write_scope(query.scope.as_deref(), recipe_preset_scope(&preset))?;
    let project_id = recipe_preset_context_project_id(&query, &mut preset);
    let manifest_path =
        recipe_preset_write_manifest_path(&state, &scope, project_id.as_deref()).await?;
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Recipe preset must be an object"))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            object
                .get("name")
                .and_then(Value::as_str)
                .map(slugify_preset_id)
        })
        .ok_or_else(|| ApiError::bad_request("Recipe preset name is required"))?;
    object.insert("id".to_owned(), Value::String(id.clone()));
    let timestamp = now_rfc3339();
    object
        .entry("createdAt".to_owned())
        .or_insert_with(|| Value::String(timestamp.clone()));
    object.insert("updatedAt".to_owned(), Value::String(timestamp));
    let models = model_catalog(&state).await?;
    let loras = lora_catalog(&state, project_id.as_deref()).await?;
    let preset = mutate_manifest_entries(&state, &manifest_path, "presets", |mut entries| {
        let preset = normalize_recipe_preset_for_write(preset, &scope, true)?;
        validate_recipe_preset_model_workflow(&models, &preset)?;
        validate_recipe_preset_lora_compatibility(&models, &loras, &preset)?;
        if entries
            .iter()
            .any(|entry| entry.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            return Err(ApiError::bad_request("Recipe preset already exists"));
        }
        entries.push(preset.clone());
        Ok((entries, preset))
    })
    .await?;
    Ok((StatusCode::CREATED, Json(finalized_recipe_preset(preset)?)))
}

pub(crate) async fn update_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<Json<Value>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let mut patch = recipe_preset_from_payload(payload)?;
    let project_id = recipe_preset_context_project_id(&query, &mut patch);
    strip_recipe_preset_write_context(&mut patch);
    let location = find_recipe_preset_write_location(
        &state,
        &preset_id,
        project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let models = model_catalog(&state).await?;
    let loras = lora_catalog(&state, project_id.as_deref()).await?;
    let preset =
        mutate_manifest_entries(&state, &location.manifest_path, "presets", |mut entries| {
            let Some(index) = entries.iter().position(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(preset_id.as_str())
            }) else {
                return Err(recipe_preset_not_found());
            };
            let mut preset = entries[index].clone();
            merge_object(&mut preset, patch);
            if let Some(object) = preset.as_object_mut() {
                object.insert("id".to_owned(), Value::String(preset_id.clone()));
                object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
            }
            let preset = normalize_recipe_preset_for_write(preset, &location.scope, false)?;
            validate_recipe_preset_model_workflow(&models, &preset)?;
            validate_recipe_preset_lora_compatibility(&models, &loras, &preset)?;
            entries[index] = preset.clone();
            Ok((entries, preset))
        })
        .await?;
    Ok(Json(finalized_recipe_preset(preset)?))
}

pub(crate) async fn delete_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let location = find_recipe_preset_write_location(
        &state,
        &preset_id,
        query.project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let preset =
        mutate_manifest_entries(&state, &location.manifest_path, "presets", |mut entries| {
            let Some(index) = entries.iter().position(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(preset_id.as_str())
            }) else {
                return Err(recipe_preset_not_found());
            };
            let mut preset = entries[index].clone();
            if let Some(object) = preset.as_object_mut() {
                object.insert("archived".to_owned(), Value::Bool(true));
                object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
            }
            let preset = normalize_recipe_preset_for_write(preset, &location.scope, false)?;
            entries[index] = preset.clone();
            Ok((entries, preset))
        })
        .await?;
    Ok(Json(finalized_recipe_preset(preset)?))
}

pub(crate) async fn duplicate_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    validate_recipe_preset_query(&query)?;
    let location = find_recipe_preset_write_location(
        &state,
        &preset_id,
        query.project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let models = model_catalog(&state).await?;
    let loras = lora_catalog(&state, query.project_id.as_deref()).await?;
    let preset =
        mutate_manifest_entries(&state, &location.manifest_path, "presets", |mut entries| {
            let Some(source) = entries
                .iter()
                .find(|entry| entry.get("id").and_then(Value::as_str) == Some(preset_id.as_str()))
                .cloned()
            else {
                return Err(recipe_preset_not_found());
            };
            let mut duplicate = source;
            strip_recipe_preset_runtime_fields(&mut duplicate);
            let base_id = duplicate
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(preset_id.as_str());
            let duplicate_id = next_duplicate_preset_id(&entries, base_id);
            let duplicate_name = next_duplicate_preset_name(
                &entries,
                duplicate
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(base_id),
            );
            let timestamp = now_rfc3339();
            if let Some(object) = duplicate.as_object_mut() {
                object.insert("id".to_owned(), Value::String(duplicate_id));
                object.insert("name".to_owned(), Value::String(duplicate_name));
                object.insert("scope".to_owned(), Value::String(location.scope.clone()));
                object.insert("archived".to_owned(), Value::Bool(false));
                object.insert("createdAt".to_owned(), Value::String(timestamp.clone()));
                object.insert("updatedAt".to_owned(), Value::String(timestamp));
            }
            let duplicate = normalize_recipe_preset_for_write(duplicate, &location.scope, true)?;
            validate_recipe_preset_model_workflow(&models, &duplicate)?;
            validate_recipe_preset_lora_compatibility(&models, &loras, &duplicate)?;
            entries.push(duplicate.clone());
            Ok((entries, duplicate))
        })
        .await?;
    Ok((StatusCode::CREATED, Json(finalized_recipe_preset(preset)?)))
}

pub(crate) async fn recipe_preset_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin_manifest = manifest_dir.join("builtin.recipe-presets.jsonc");
    let user_manifest = manifest_dir.join("user.recipe-presets.jsonc");
    let builtin = load_manifest_entries(state, &builtin_manifest, "presets")
        .await?
        .into_iter()
        .map(|preset| normalize_recipe_preset_entry(preset, "builtin", &builtin_manifest))
        .collect::<Result<Vec<_>, _>>()?;
    let user = load_manifest_entries(state, &user_manifest, "presets")
        .await?
        .into_iter()
        .map(|preset| normalize_recipe_preset_entry(preset, "global", &user_manifest))
        .collect::<Result<Vec<_>, _>>()?;
    let models = model_catalog(state).await?;
    let mut presets = merge_entries_by_id(builtin, user);
    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("recipes").join("presets.jsonc");
        let project_presets = load_manifest_entries(state, &project_manifest, "presets")
            .await?
            .into_iter()
            .map(|preset| normalize_recipe_preset_entry(preset, "project", &project_manifest))
            .collect::<Result<Vec<_>, _>>()?;
        presets = merge_entries_by_id(presets, project_presets);
    }
    for preset in presets.iter_mut() {
        finalize_recipe_preset_entry(preset, &models)?;
    }
    presets.sort_by(|left, right| {
        let left_key = (
            recipe_preset_scope_order(left.get("scope").and_then(Value::as_str)),
            left.get("order").and_then(Value::as_i64).unwrap_or(10_000),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            recipe_preset_scope_order(right.get("scope").and_then(Value::as_str)),
            right.get("order").and_then(Value::as_i64).unwrap_or(10_000),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(presets)
}

pub(crate) fn recipe_preset_scope_order(scope: Option<&str>) -> u8 {
    match scope {
        Some("builtin") => 0,
        Some("global") => 1,
        Some("project") => 2,
        _ => 3,
    }
}

pub(crate) fn normalize_recipe_preset_entry(
    mut preset: Value,
    scope: &str,
    manifest_path: &FsPath,
) -> Result<Value, ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Recipe preset manifest entry must be an object"))?;
    object
        .entry("scope".to_owned())
        .or_insert_with(|| Value::String(scope.to_owned()));
    object.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    Ok(preset)
}

pub(crate) fn finalize_recipe_preset_entry(
    preset: &mut Value,
    models: &[Value],
) -> Result<(), ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Recipe preset manifest entry must be an object"))?;
    let mut migration_notes = Vec::new();
    if !object.contains_key("workflow") {
        if let Some(workflow) = inferred_recipe_preset_workflow(object) {
            object.insert("workflow".to_owned(), Value::String(workflow.to_owned()));
            migration_notes.push(Value::String(format!(
                "workflow inferred from legacy modes as {workflow}"
            )));
        }
    }
    if !object.contains_key("model") {
        if let Some(model) = object
            .get("workflow")
            .and_then(Value::as_str)
            .and_then(|workflow| default_recipe_preset_model_for_workflow(models, workflow))
        {
            object.insert("model".to_owned(), Value::String(model.clone()));
            migration_notes.push(Value::String(format!(
                "model defaulted to {model} for legacy preset"
            )));
        }
    }
    if !object.contains_key("modes") {
        if let Some(workflow) = object.get("workflow").and_then(Value::as_str) {
            object.insert(
                "modes".to_owned(),
                Value::Array(
                    default_recipe_preset_modes_for_workflow(workflow)
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
    }
    if !object.contains_key("loras") {
        if let Some(loras) = object.get("builtInLoras").cloned() {
            let migrated_count = loras.as_array().map(Vec::len).unwrap_or_default();
            object.insert("loras".to_owned(), loras);
            if migrated_count > 0 {
                migration_notes.push(Value::String("builtInLoras migrated to loras".to_owned()));
            }
        }
    }
    let loras = object
        .get("loras")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    object.entry("builtInLoras".to_owned()).or_insert(loras);
    object
        .entry("defaults".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    object
        .entry("prompt".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if !migration_notes.is_empty() {
        object.insert(
            "appliedDefaults".to_owned(),
            json!({
                "notes": migration_notes
            }),
        );
    }
    Ok(())
}

pub(crate) fn default_recipe_preset_model_for_workflow(
    models: &[Value],
    workflow: &str,
) -> Option<String> {
    models
        .iter()
        .find(|model| {
            model_supports_recipe_workflow(model, workflow)
                && model.get("installState").and_then(Value::as_str) == Some("installed")
        })
        .and_then(|model| model.get("id").and_then(Value::as_str))
        .map(str::to_owned)
}

pub(crate) fn model_supports_recipe_workflow(model: &Value, workflow: &str) -> bool {
    model
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|capabilities| {
            capabilities
                .iter()
                .filter_map(Value::as_str)
                .any(|capability| capability == workflow)
        })
}

pub(crate) fn default_recipe_preset_modes_for_workflow(workflow: &str) -> Vec<String> {
    match workflow {
        "text_to_image" => vec!["text_to_image", "character_image", "style_variations"],
        "edit_image" => vec!["edit_image"],
        "image_to_video" => vec!["image_to_video"],
        "text_to_video" => vec!["text_to_video"],
        "first_last_frame" => vec!["first_last_frame"],
        _ => vec![workflow],
    }
    .into_iter()
    .map(str::to_owned)
    .collect()
}

pub(crate) fn inferred_recipe_preset_workflow(object: &JsonObject) -> Option<&'static str> {
    object
        .get("modes")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_str)
        .find_map(|mode| match mode {
            "text_to_image" => Some("text_to_image"),
            "edit_image" => Some("edit_image"),
            "image_to_video" => Some("image_to_video"),
            "text_to_video" => Some("text_to_video"),
            "first_last_frame" => Some("first_last_frame"),
            _ => None,
        })
}

pub(crate) fn recipe_preset_loras(preset: &Value) -> Vec<Value> {
    preset
        .get("loras")
        .or_else(|| preset.get("builtInLoras"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn recipe_preset_archived(preset: &Value) -> bool {
    preset
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn finalized_recipe_preset(mut preset: Value) -> Result<Value, ApiError> {
    // Write paths require an explicit model before this point, so single-preset
    // response finalization does not need the read-side model catalog fallback.
    finalize_recipe_preset_entry(&mut preset, &[])?;
    Ok(preset)
}

pub(crate) fn recipe_preset_from_payload(payload: Value) -> Result<Value, ApiError> {
    match payload {
        Value::Null => Ok(Value::Object(JsonObject::new())),
        Value::Object(_) => Ok(payload),
        _ => Err(ApiError::bad_request(
            "Recipe preset payload must be an object",
        )),
    }
}

pub(crate) fn take_string_field(payload: &mut Value, field: &str) -> Option<String> {
    payload
        .as_object_mut()
        .and_then(|object| object.remove(field))
        .and_then(|value| value.as_str().map(str::to_owned))
}

pub(crate) fn recipe_preset_scope(preset: &Value) -> Option<&str> {
    preset.get("scope").and_then(Value::as_str)
}

pub(crate) fn recipe_preset_context_project_id(
    query: &RecipePresetsQuery,
    payload: &mut Value,
) -> Option<String> {
    query
        .project_id
        .clone()
        .or_else(|| take_string_field(payload, "projectId"))
}

pub(crate) fn strip_recipe_preset_write_context(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("projectId");
        object.remove("scope");
        object.remove("manifestPath");
        object.remove("builtInLoras");
        object.remove("appliedDefaults");
    }
}

pub(crate) fn strip_recipe_preset_runtime_fields(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("manifestPath");
        object.remove("builtInLoras");
        object.remove("appliedDefaults");
    }
}

pub(crate) fn recipe_preset_write_scope(
    query_scope: Option<&str>,
    payload_scope: Option<&str>,
) -> Result<String, ApiError> {
    let scope = query_scope.or(payload_scope).unwrap_or("global").trim();
    match scope {
        "global" | "project" => Ok(scope.to_owned()),
        "builtin" => Err(ApiError::bad_request(
            "Built-in recipe presets are read-only",
        )),
        _ => Err(ApiError::bad_request(
            "Recipe preset scope must be global or project",
        )),
    }
}

pub(crate) fn validate_recipe_preset_query(query: &RecipePresetsQuery) -> Result<(), ApiError> {
    if let Some(workflow) = query.workflow.as_deref() {
        validate_recipe_preset_workflow(Some(workflow), false)?;
    }
    if let Some(scope) = query.scope.as_deref() {
        match scope {
            "builtin" | "global" | "project" => {}
            _ => return Err(ApiError::bad_request("Unsupported recipe preset scope")),
        }
    }
    Ok(())
}

pub(crate) async fn recipe_preset_write_manifest_path(
    state: &AppState,
    scope: &str,
    project_id: Option<&str>,
) -> Result<PathBuf, ApiError> {
    match scope {
        "global" => Ok(state
            .settings
            .config_dir
            .join("manifests")
            .join("user.recipe-presets.jsonc")),
        "project" => {
            let Some(project_id) = project_id else {
                return Err(ApiError::bad_request(
                    "Project recipe presets require projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), project_id).await?;
            Ok(project_path.join("recipes").join("presets.jsonc"))
        }
        _ => Err(ApiError::bad_request(
            "Recipe preset scope must be global or project",
        )),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RecipePresetWriteLocation {
    scope: String,
    manifest_path: PathBuf,
}

pub(crate) fn recipe_preset_not_found() -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        detail: "Recipe preset not found".to_owned(),
    }
}

pub(crate) async fn find_recipe_preset_write_location(
    state: &AppState,
    preset_id: &str,
    project_id: Option<&str>,
    scope: Option<&str>,
) -> Result<RecipePresetWriteLocation, ApiError> {
    match scope {
        Some("builtin") => {
            return recipe_preset_readonly_or_not_found(state, preset_id, project_id).await
        }
        Some("global") => {
            return recipe_preset_location_if_present(state, preset_id, "global", project_id).await;
        }
        Some("project") => {
            return recipe_preset_location_if_present(state, preset_id, "project", project_id)
                .await;
        }
        Some(_) => return Err(ApiError::bad_request("Unsupported recipe preset scope")),
        None => {}
    }

    if project_id.is_some() {
        match recipe_preset_location_if_present(state, preset_id, "project", project_id).await {
            Ok(location) => return Ok(location),
            Err(error) if error.status == StatusCode::NOT_FOUND => {}
            Err(error) => return Err(error),
        }
    }
    match recipe_preset_location_if_present(state, preset_id, "global", project_id).await {
        Ok(location) => Ok(location),
        Err(error) if error.status == StatusCode::NOT_FOUND => {
            recipe_preset_readonly_or_not_found(state, preset_id, project_id).await
        }
        Err(error) => Err(error),
    }
}

pub(crate) async fn recipe_preset_location_if_present(
    state: &AppState,
    preset_id: &str,
    scope: &str,
    project_id: Option<&str>,
) -> Result<RecipePresetWriteLocation, ApiError> {
    let manifest_path = recipe_preset_write_manifest_path(state, scope, project_id).await?;
    let entries = load_manifest_entries(state, &manifest_path, "presets").await?;
    if entries
        .iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(preset_id))
    {
        Ok(RecipePresetWriteLocation {
            scope: scope.to_owned(),
            manifest_path,
        })
    } else {
        Err(recipe_preset_not_found())
    }
}

pub(crate) async fn recipe_preset_readonly_or_not_found(
    state: &AppState,
    preset_id: &str,
    project_id: Option<&str>,
) -> Result<RecipePresetWriteLocation, ApiError> {
    let catalog = recipe_preset_catalog(state, project_id).await?;
    if catalog.iter().any(|preset| {
        preset.get("id").and_then(Value::as_str) == Some(preset_id)
            && preset.get("scope").and_then(Value::as_str) == Some("builtin")
    }) {
        Err(ApiError::bad_request(
            "Built-in recipe presets are read-only",
        ))
    } else {
        Err(recipe_preset_not_found())
    }
}

pub(crate) fn normalize_recipe_preset_for_write(
    mut preset: Value,
    scope: &str,
    require_all: bool,
) -> Result<Value, ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Recipe preset must be an object"))?;
    object.insert("scope".to_owned(), Value::String(scope.to_owned()));
    validate_recipe_preset_id(object.get("id").and_then(Value::as_str))?;
    validate_required_string_field(
        object,
        "name",
        require_all,
        "Recipe preset name is required",
    )?;
    validate_required_string_field(
        object,
        "model",
        require_all,
        "Recipe preset model is required",
    )?;
    validate_recipe_preset_workflow(object.get("workflow").and_then(Value::as_str), require_all)?;
    validate_recipe_preset_order(object.get("order"))?;
    validate_recipe_preset_defaults(object.get("defaults"))?;
    validate_recipe_preset_prompt(object.get("prompt"))?;
    normalize_recipe_preset_loras(object)?;
    Ok(preset)
}

pub(crate) fn validate_recipe_preset_id(value: Option<&str>) -> Result<(), ApiError> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Err(ApiError::bad_request("Recipe preset id is required"));
    };
    let valid = value.chars().enumerate().all(|(index, character)| {
        character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || (index > 0 && matches!(character, '_' | '-'))
    });
    if !valid {
        return Err(ApiError::bad_request(
            "Recipe preset id must use lowercase letters, numbers, dashes, or underscores",
        ));
    }
    Ok(())
}

pub(crate) fn validate_required_string_field(
    object: &JsonObject,
    field: &str,
    require: bool,
    message: &'static str,
) -> Result<(), ApiError> {
    match object.get(field).and_then(Value::as_str).map(str::trim) {
        Some(value) if !value.is_empty() => Ok(()),
        _ if require => Err(ApiError::bad_request(message)),
        _ => Ok(()),
    }
}

pub(crate) fn validate_recipe_preset_workflow(
    value: Option<&str>,
    require: bool,
) -> Result<(), ApiError> {
    match value {
        Some(
            "text_to_image" | "edit_image" | "image_to_video" | "text_to_video"
            | "first_last_frame",
        ) => Ok(()),
        Some(_) => Err(ApiError::bad_request("Unsupported recipe preset workflow")),
        None if require => Err(ApiError::bad_request("Recipe preset workflow is required")),
        None => Ok(()),
    }
}

pub(crate) fn validate_recipe_preset_order(value: Option<&Value>) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_i64()) {
        return Err(ApiError::bad_request(
            "Recipe preset order must be an integer",
        ));
    }
    Ok(())
}

pub(crate) fn validate_recipe_preset_defaults(value: Option<&Value>) -> Result<(), ApiError> {
    let Some(defaults) = value else {
        return Ok(());
    };
    let object = defaults
        .as_object()
        .ok_or_else(|| ApiError::bad_request("Recipe preset defaults must be an object"))?;
    if let Some(resolution) = object.get("resolution").and_then(Value::as_str) {
        let (width, height) = parse_recipe_preset_resolution(resolution)?;
        validate_dimension(width, "width", MAX_IMAGE_DIMENSION)?;
        validate_dimension(height, "height", MAX_IMAGE_DIMENSION)?;
    }
    if let Some(count) = object.get("count").and_then(Value::as_u64) {
        if !(1..=8).contains(&count) {
            return Err(ApiError::bad_request(
                "Recipe preset count must be between 1 and 8",
            ));
        }
    }
    // Studio "Save as Preset" snapshots carry generation knobs in defaults. Each
    // is range-checked only when present and numeric; non-numeric values are left
    // to forward-compat (never panics), and non-finite or out-of-range values are
    // rejected so a malformed payload can't be persisted.
    if let Some(steps) = recipe_preset_default_number(object, "steps") {
        if steps.fract() != 0.0 || !(1.0..=200.0).contains(&steps) {
            return Err(ApiError::bad_request(
                "Recipe preset steps must be a whole number between 1 and 200",
            ));
        }
    }
    for (key, min, max) in RECIPE_PRESET_DEFAULT_RANGES {
        if let Some(number) = recipe_preset_default_number(object, key) {
            if number < *min || number > *max {
                return Err(ApiError::bad_request(format!(
                    "Recipe preset {key} must be between {min} and {max}"
                )));
            }
        }
    }
    Ok(())
}

// Numeric generation knobs a studio snapshot may store in `defaults`, with the
// inclusive range each must fall in. Ranges are deliberately generous so valid
// model-specific values are never rejected — they only catch clearly bad input.
const RECIPE_PRESET_DEFAULT_RANGES: &[(&str, f64, f64)] = &[
    ("guidanceScale", 0.0, 60.0),
    ("schedulerShift", 0.0, 20.0),
    ("trueCfgScale", 0.0, 30.0),
    ("ipAdapterScale", 0.0, 2.0),
    ("controlnetScale", 0.0, 4.0),
    ("upscaleFactor", 1.0, 8.0),
    ("duration", 1.0, 120.0),
    ("fps", 1.0, 240.0),
    ("videoCfgGuidanceScale", 0.0, 60.0),
    ("videoStgGuidanceScale", 0.0, 60.0),
    ("videoRescaleScale", 0.0, 10.0),
];

// Read a defaults knob as a finite f64 whether stored as a JSON number or a
// numeric string (older studio snapshots stringified text inputs). Returns None
// for absent, non-numeric, or non-finite values so callers simply skip them.
fn recipe_preset_default_number(object: &JsonObject, key: &str) -> Option<f64> {
    let value = object.get(key)?;
    let number = value.as_f64().or_else(|| {
        value
            .as_str()
            .and_then(|text| text.trim().parse::<f64>().ok())
    })?;
    number.is_finite().then_some(number)
}

pub(crate) fn validate_recipe_preset_model_workflow(
    models: &[Value],
    preset: &Value,
) -> Result<(), ApiError> {
    let Some(model_id) = preset.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(workflow) = preset.get("workflow").and_then(Value::as_str) else {
        return Ok(());
    };
    let model = models
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .ok_or_else(|| {
            ApiError::bad_request(format!("Recipe preset model not found: {model_id}"))
        })?;
    let capabilities = model
        .get("capabilities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if capabilities
        .iter()
        .filter_map(Value::as_str)
        .any(|capability| capability == workflow)
    {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "Model {model_id} does not support workflow {workflow}"
        )))
    }
}

pub(crate) fn validate_recipe_preset_lora_compatibility(
    models: &[Value],
    loras: &[Value],
    preset: &Value,
) -> Result<(), ApiError> {
    let Some(model_id) = preset.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    validate_lora_specs_for_model(
        models,
        loras,
        model_id,
        &recipe_preset_loras(preset),
        false,
        "Recipe preset LoRA",
    )?;
    Ok(())
}

pub(crate) fn validate_recipe_preset_prompt(value: Option<&Value>) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_object()) {
        return Err(ApiError::bad_request(
            "Recipe preset prompt must be an object",
        ));
    }
    Ok(())
}

pub(crate) fn normalize_recipe_preset_loras(object: &mut JsonObject) -> Result<(), ApiError> {
    if !object.contains_key("loras") {
        if let Some(loras) = object.remove("builtInLoras") {
            object.insert("loras".to_owned(), loras);
        }
    } else {
        object.remove("builtInLoras");
    }
    let Some(loras) = object.get_mut("loras") else {
        return Ok(());
    };
    let items = loras
        .as_array_mut()
        .ok_or_else(|| ApiError::bad_request("Recipe preset loras must be an array"))?;
    if items.len() > 5 {
        return Err(ApiError::bad_request(
            "Recipe presets can include at most 5 LoRAs",
        ));
    }
    for item in items {
        if let Some(id) = item.as_str().map(str::to_owned) {
            *item = json!({ "id": id });
        }
        let object = item
            .as_object()
            .ok_or_else(|| ApiError::bad_request("Recipe preset LoRA must be an object"))?;
        validate_recipe_preset_id(object.get("id").and_then(Value::as_str))?;
        if let Some(lora_id) = object.get("loraId").and_then(Value::as_str) {
            validate_recipe_preset_id(Some(lora_id))?;
        }
        if object
            .get("compatibility")
            .is_some_and(|value| !value.is_object())
        {
            return Err(ApiError::bad_request(
                "Recipe preset LoRA compatibility must be an object",
            ));
        }
        if let Some(weight) = object.get("weight").and_then(Value::as_f64) {
            if !(-2.0..=2.0).contains(&weight) {
                return Err(ApiError::bad_request(
                    "Recipe preset LoRA weight must be between -2 and 2",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn preset_prompt(prompt: &str, preset: &Value) -> String {
    let fragments = preset.get("prompt").and_then(Value::as_object);
    [
        fragments
            .and_then(|value| value.get("prefix"))
            .and_then(Value::as_str),
        Some(prompt),
        fragments
            .and_then(|value| value.get("suffix"))
            .and_then(Value::as_str),
    ]
    .into_iter()
    .flatten()
    .map(str::trim)
    .filter(|value| !value.is_empty())
    .collect::<Vec<_>>()
    .join(", ")
}

pub(crate) fn preset_lora_id(preset_lora: &Value) -> Option<&str> {
    preset_lora
        .as_str()
        .or_else(|| preset_lora.get("id").and_then(Value::as_str))
}

pub(crate) fn preset_lora_weight(lora: &Value, preset_lora: &Value) -> f64 {
    preset_lora
        .get("weight")
        .and_then(Value::as_f64)
        .or_else(|| lora.get("defaultWeight").and_then(Value::as_f64))
        .or_else(|| lora.get("weight").and_then(Value::as_f64))
        .unwrap_or(0.8)
}

pub(crate) fn serialize_preset_lora(lora: &Value, preset_lora: &Value, lora_id: &str) -> Value {
    json!({
        "id": lora_id,
        "name": lora.get("name").and_then(Value::as_str).unwrap_or(lora_id),
        "scope": lora.get("scope").and_then(Value::as_str).unwrap_or("builtin"),
        "weight": preset_lora_weight(lora, preset_lora),
        "family": lora.get("family").cloned().unwrap_or(Value::Null),
        "families": lora.get("families").cloned().unwrap_or(Value::Null),
        "compatibleFamilies": lora.get("compatibleFamilies").cloned().unwrap_or(Value::Null),
        "modelFamilies": lora.get("modelFamilies").cloned().unwrap_or(Value::Null),
        "triggerWords": lora.get("triggerWords").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        "compatibility": lora.get("compatibility").cloned().unwrap_or_else(|| Value::Object(JsonObject::new())),
        "icLora": lora.get("icLora").cloned().unwrap_or(Value::Bool(false)),
        "conditioningRole": lora.get("conditioningRole").cloned().unwrap_or(Value::Null),
        "installedPath": lora.get("installedPath").cloned().unwrap_or(Value::Null),
        "source": lora.get("source").cloned().unwrap_or(Value::Null),
        "presetManaged": true
    })
}

pub(crate) fn slugify_preset_id(value: &str) -> String {
    let id = slugify_lora_id(value);
    if id == "lora" {
        "preset".to_owned()
    } else {
        id
    }
}

pub(crate) fn next_duplicate_preset_id(entries: &[Value], base_id: &str) -> String {
    let base_id = base_id.trim().trim_end_matches("_copy");
    let first = format!("{base_id}_copy");
    if !preset_id_exists(entries, &first) {
        return first;
    }
    for index in 2.. {
        let candidate = format!("{base_id}_copy_{index}");
        if !preset_id_exists(entries, &candidate) {
            return candidate;
        }
    }
    unreachable!("infinite iterator should return a duplicate preset id")
}

pub(crate) fn preset_id_exists(entries: &[Value], id: &str) -> bool {
    entries
        .iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(id))
}

pub(crate) fn next_duplicate_preset_name(entries: &[Value], base_name: &str) -> String {
    let first = format!("{base_name} Copy");
    if !preset_name_exists(entries, &first) {
        return first;
    }
    for index in 2.. {
        let candidate = format!("{base_name} Copy {index}");
        if !preset_name_exists(entries, &candidate) {
            return candidate;
        }
    }
    unreachable!("infinite iterator should return a duplicate preset name")
}

pub(crate) fn preset_name_exists(entries: &[Value], name: &str) -> bool {
    entries
        .iter()
        .any(|entry| entry.get("name").and_then(Value::as_str) == Some(name))
}

// Typed contract helpers for Phase 2 conversion
// These functions use sceneworks-core contracts instead of Value walking.

use sceneworks_core::contracts::RecipePresetManifestEntry;

/// Deserialize a Value to a typed RecipePresetManifestEntry.
/// This replaces the Value-walking validation with serde's compile-time type checking.
#[allow(dead_code)]
pub(crate) fn value_to_recipe_preset_entry(
    value: &Value,
) -> Result<RecipePresetManifestEntry, ApiError> {
    serde_json::from_value(value.clone())
        .map_err(|e| ApiError::bad_request(format!("Invalid recipe preset: {}", e)))
}

/// Serialize a typed RecipePresetManifestEntry back to Value.
/// Preserves extra fields via flatten.
#[allow(dead_code)]
pub(crate) fn recipe_preset_entry_to_value(
    entry: &RecipePresetManifestEntry,
) -> Result<Value, ApiError> {
    serde_json::to_value(entry)
        .map_err(|e| ApiError::internal(format!("Serialization failed: {}", e)))
}

/// Validate typed recipe preset entry against models and loras.
/// Replaces validate_recipe_preset_model_workflow and validate_recipe_preset_lora_compatibility.
#[allow(dead_code)]
pub(crate) fn validate_typed_recipe_preset_entry(
    entry: &RecipePresetManifestEntry,
    models: &[Value],
    loras: &[Value],
) -> Result<(), ApiError> {
    // Model must exist and support the workflow
    let model = models
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(&entry.model))
        .ok_or_else(|| {
            ApiError::bad_request(format!("Recipe preset model not found: {}", entry.model))
        })?;

    let capabilities = model
        .get("capabilities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let workflow_str = match &entry.workflow {
        sceneworks_core::contracts::RecipePresetWorkflow::TextToImage => "text_to_image",
        sceneworks_core::contracts::RecipePresetWorkflow::ImageEdit => "edit_image",
        sceneworks_core::contracts::RecipePresetWorkflow::ImageToVideo => "image_to_video",
        sceneworks_core::contracts::RecipePresetWorkflow::TextToVideo => "text_to_video",
        sceneworks_core::contracts::RecipePresetWorkflow::FirstLastFrame => "first_last_frame",
        _ => "unknown",
    };

    if !capabilities
        .iter()
        .any(|cap| cap.as_str() == Some(workflow_str))
    {
        return Err(ApiError::bad_request(format!(
            "Model {} does not support workflow {}",
            entry.model, workflow_str
        )));
    }

    // Validate loras for compatibility
    let preset_loras: Vec<Value> = entry
        .loras
        .iter()
        .map(|lora| serde_json::to_value(lora).unwrap_or(Value::Null))
        .collect();

    validate_lora_specs_for_model(
        models,
        loras,
        &entry.model,
        &preset_loras,
        false,
        "Recipe preset LoRA",
    )?;

    Ok(())
}
