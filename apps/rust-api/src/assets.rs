use super::*;

pub(crate) async fn list_assets(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<AssetsQuery>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.list_assets(
                &project_id,
                query.include_rejected.unwrap_or(false),
                query.include_trashed.unwrap_or(false),
            )
        })
        .await?,
    ))
}

pub(crate) async fn get_asset(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.get_asset(&project_id, &asset_id)).await?,
    ))
}

pub(crate) async fn import_asset(
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
        let filename = field.file_name().unwrap_or("upload").to_owned();
        let content_type = field.content_type().map(str::to_owned);
        let temp_path = write_upload_field_to_temp_file(&state, field).await?;
        let source_path = temp_path.clone();
        let asset = project_call(state, move |store| {
            store.import_asset(
                &project_id,
                UploadAsset {
                    filename,
                    content_type,
                    source_path,
                },
            )
        })
        .await
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&temp_path);
        })?;
        return Ok((StatusCode::CREATED, Json(asset)));
    }
    Err(ApiError::bad_request("Upload file field is required"))
}

pub(crate) async fn write_upload_field_to_temp_file(
    state: &AppState,
    mut field: axum::extract::multipart::Field<'_>,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state.settings.data_dir.join("cache").join("uploads");
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(format!("upload-{}.tmp", Uuid::new_v4().simple()));
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut uploaded_bytes = 0usize;
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        uploaded_bytes = uploaded_bytes.saturating_add(chunk.len());
        if uploaded_bytes > MAX_UPLOAD_BYTES {
            let _ = tokio::fs::remove_file(&temp_path).await;
            return Err(ApiError::payload_too_large("Uploaded file is too large"));
        }
        file.write_all(&chunk)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    file.flush()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(temp_path)
}

pub(crate) async fn update_asset_status(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<AssetStatusPatch>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_asset_status(&project_id, &asset_id, payload)
        })
        .await?,
    ))
}

pub(crate) async fn delete_asset(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::AssetMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_asset(&project_id, &asset_id)
        })
        .await?,
    ))
}

pub(crate) async fn purge_asset(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::AssetMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.purge_asset(&project_id, &asset_id)
        })
        .await?,
    ))
}
