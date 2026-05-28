use super::*;

pub(crate) async fn list_characters(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<CharactersQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.list_characters(&project_id, query.include_archived.unwrap_or(false))
        })
        .await?,
    ))
}

pub(crate) async fn create_character(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<CharacterCreateRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.create_character(
            &project_id,
            CharacterCreateInput {
                name: payload.name,
                character_type: payload.character_type,
                description: payload.description,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

pub(crate) async fn get_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_character(&project_id, &character_id)
        })
        .await?,
    ))
}

pub(crate) async fn update_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character(
                &project_id,
                &character_id,
                CharacterUpdateInput {
                    name: payload.name,
                    character_type: payload.character_type,
                    description: payload.description,
                    archived: payload.archived,
                },
            )
        })
        .await?,
    ))
}

pub(crate) async fn archive_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::CharacterMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.archive_character(&project_id, &character_id)
        })
        .await?,
    ))
}

pub(crate) async fn archive_character_explicit(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::CharacterMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.archive_character(&project_id, &character_id)
        })
        .await?,
    ))
}

pub(crate) async fn purge_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::CharacterMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.purge_character(&project_id, &character_id)
        })
        .await?,
    ))
}

pub(crate) async fn add_character_reference(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterReferenceRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.add_character_reference(
            &project_id,
            &character_id,
            CharacterReferenceInput {
                asset_id: payload.asset_id,
                approved: payload.approved,
                role: payload.role,
                notes: payload.notes,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

pub(crate) async fn update_character_reference(
    State(state): State<AppState>,
    Path((project_id, character_id, asset_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<CharacterReferenceUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character_reference(
                &project_id,
                &character_id,
                &asset_id,
                CharacterReferenceUpdateInput {
                    approved: payload.approved,
                    role: payload.role,
                    notes: payload.notes,
                },
            )
        })
        .await?,
    ))
}

pub(crate) async fn remove_character_reference(
    State(state): State<AppState>,
    Path((project_id, character_id, asset_id)): Path<(String, String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.remove_character_reference(&project_id, &character_id, &asset_id)
        })
        .await?,
    ))
}

pub(crate) async fn create_character_look(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterLookRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.create_character_look(
            &project_id,
            &character_id,
            CharacterLookInput {
                name: payload.name,
                description: payload.description,
                approved_reference_ids: payload.approved_reference_ids,
                recipe_settings: payload.recipe_settings,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

pub(crate) async fn update_character_look(
    State(state): State<AppState>,
    Path((project_id, character_id, look_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<CharacterLookUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character_look(
                &project_id,
                &character_id,
                &look_id,
                CharacterLookUpdateInput {
                    name: payload.name,
                    description: payload.description,
                    approved_reference_ids: payload.approved_reference_ids,
                    recipe_settings: payload.recipe_settings,
                },
            )
        })
        .await?,
    ))
}

pub(crate) async fn delete_character_look(
    State(state): State<AppState>,
    Path((project_id, character_id, look_id)): Path<(String, String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_character_look(&project_id, &character_id, &look_id)
        })
        .await?,
    ))
}

pub(crate) async fn attach_character_lora(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterLoraRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.attach_character_lora(
            &project_id,
            &character_id,
            CharacterLoraInput {
                lora_id: payload.lora_id,
                name: payload.name,
                source_path: payload.source_path,
                trigger_words: payload.trigger_words,
                default_weight: payload.default_weight,
                compatibility: payload.compatibility,
                scope: payload.scope,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

pub(crate) async fn update_character_lora(
    State(state): State<AppState>,
    Path((project_id, character_id, link_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<CharacterLoraUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character_lora(
                &project_id,
                &character_id,
                &link_id,
                CharacterLoraUpdateInput {
                    name: payload.name,
                    trigger_words: payload.trigger_words,
                    default_weight: payload.default_weight,
                    compatibility: payload.compatibility,
                    scope: payload.scope,
                },
            )
        })
        .await?,
    ))
}

pub(crate) async fn detach_character_lora(
    State(state): State<AppState>,
    Path((project_id, character_id, link_id)): Path<(String, String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.detach_character_lora(&project_id, &character_id, &link_id)
        })
        .await?,
    ))
}

pub(crate) async fn create_character_test_job(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterTestRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_character_test_job(&payload)?;
    let character = project_call(state.clone(), {
        let project_id = project_id.clone();
        let character_id = character_id.clone();
        move |store| store.get_character(&project_id, &character_id)
    })
    .await?;
    let look = payload.look_id.as_deref().and_then(|look_id| {
        character
            .get("looks")
            .and_then(Value::as_array)
            .and_then(|looks| {
                looks
                    .iter()
                    .find(|look| look.get("id").and_then(Value::as_str) == Some(look_id))
                    .cloned()
            })
    });
    let approved_reference_ids = character
        .get("references")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|reference| {
            reference
                .get("approved")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|reference| reference.get("assetId").and_then(Value::as_str))
        .map(|asset_id| Value::String(asset_id.to_owned()))
        .collect::<Vec<_>>();
    let mut advanced = JsonObject::new();
    advanced.insert(
        "characterName".to_owned(),
        character.get("name").cloned().unwrap_or(Value::Null),
    );
    advanced.insert(
        "characterType".to_owned(),
        character.get("type").cloned().unwrap_or(Value::Null),
    );
    advanced.insert(
        "approvedReferenceIds".to_owned(),
        Value::Array(approved_reference_ids),
    );
    advanced.insert("look".to_owned(), look.unwrap_or(Value::Null));

    let mut job_payload = JsonObject::new();
    job_payload.insert(
        "mode".to_owned(),
        Value::String("character_image".to_owned()),
    );
    job_payload.insert("prompt".to_owned(), Value::String(payload.prompt));
    job_payload.insert("negativePrompt".to_owned(), Value::String(String::new()));
    job_payload.insert("model".to_owned(), Value::String(payload.model));
    job_payload.insert("count".to_owned(), json!(payload.count));
    job_payload.insert("seed".to_owned(), Value::Null);
    job_payload.insert("width".to_owned(), json!(payload.width));
    job_payload.insert("height".to_owned(), json!(payload.height));
    job_payload.insert(
        "stylePreset".to_owned(),
        Value::String("character-test".to_owned()),
    );
    job_payload.insert("sourceAssetId".to_owned(), Value::Null);
    job_payload.insert(
        "loras".to_owned(),
        character.get("loras").cloned().unwrap_or_else(|| json!([])),
    );
    job_payload.insert("characterId".to_owned(), Value::String(character_id));
    job_payload.insert(
        "characterLookId".to_owned(),
        payload.look_id.map(Value::String).unwrap_or(Value::Null),
    );
    job_payload.insert("advanced".to_owned(), Value::Object(advanced));
    // The worker's image_request_from_job requires payload.projectId; create_generation_job
    // only stores it as the job column, so inject it here (as person/timeline/training jobs do).
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    validate_job_lora_compatibility(&state, Some(&project_id), &mut job_payload, true).await?;
    let job = create_generation_job(
        state,
        JobType::ImageGenerate,
        Some(project_id),
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}
