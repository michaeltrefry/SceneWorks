use super::*;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CreatePosesRequest {
    /// One curated pose candidate per item. Each carries `jobId` plus
    /// `skeletonFile` (the worker's cached `pose_detect` preview to copy), the
    /// chosen `category`, free `tags`, and the `pose` keypoint metadata from the
    /// job result. Shape is validated by `ProjectStore::create_pose_asset`
    /// (epic 2282, sc-2287).
    pub(crate) poses: Vec<serde_json::Value>,
}

/// Persist curated DWPose skeletons into the reserved global pose library. Each
/// candidate's skeleton PNG already lives in the worker's pose-detect cache, so
/// this is a server-side copy + sidecar write (no upload body). Returns the
/// created assets so the Pose Library can refresh without a round-trip.
pub(crate) async fn create_poses(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<CreatePosesRequest>,
) -> Result<(StatusCode, Json<Vec<serde_json::Value>>), ApiError> {
    if payload.poses.is_empty() {
        return Err(ApiError::bad_request("No poses supplied"));
    }
    let created = project_call(state, move |store| {
        let mut created = Vec::with_capacity(payload.poses.len());
        for spec in &payload.poses {
            created.push(store.create_pose_asset(spec)?);
        }
        Ok(created)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(created)))
}

/// Stage File-Upload images for pose detection as TEMPORARY files, NOT workspace
/// assets (epic 2282). The Create tab uploads photos here, passes the returned
/// absolute paths to a `pose_detect` job, and the worker deletes them after
/// detection (`pose-uploads` cache dir); a startup sweep is the backstop. This is
/// why File-Upload sources don't pollute the asset library. Returns
/// `{ "sources": [{ "path", "displayName" }] }` in field order.
pub(crate) async fn create_pose_sources(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let mut sources = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        if field.name() != Some("file") {
            continue;
        }
        let display_name = field.file_name().unwrap_or("image").to_owned();
        let path = write_upload_field_to_dir(&state, field, "pose-uploads").await?;
        sources.push(serde_json::json!({
            "path": path.to_string_lossy(),
            "displayName": display_name,
        }));
    }
    if sources.is_empty() {
        return Err(ApiError::bad_request("At least one image file is required"));
    }
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "sources": sources })),
    ))
}

/// Remove stale pose-source temp uploads (`<data_dir>/cache/pose-uploads/upload-*`)
/// at startup — the backstop for detect jobs that never ran (the worker deletes its
/// own sources after a successful detect). Mirrors `sweep_stale_lora_uploads`.
pub(crate) fn sweep_stale_pose_uploads(data_dir: &FsPath) -> std::io::Result<usize> {
    let cutoff = SystemTime::now() - Duration::from_secs(STALE_LORA_UPLOAD_SECONDS);
    let upload_root = data_dir.join("cache").join("pose-uploads");
    let entries = match std::fs::read_dir(upload_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0usize;
    for entry in entries {
        let entry = entry?;
        let filename = entry.file_name();
        if !filename.to_string_lossy().starts_with("upload-") {
            continue;
        }
        let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
        if modified <= cutoff {
            let path = entry.path();
            let _ = if entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            removed += 1;
        }
    }
    Ok(removed)
}

/// Stream a worker pose-detect skeleton preview from the cache so the Create tab
/// can show each candidate before it's saved (the PNG lives server-side, not in
/// the browser). Same auth class as project files: loadable via a plain `<img>`
/// URL. The path is validated/canonicalized under the pose-detect cache root.
pub(crate) async fn get_pose_preview(
    State(state): State<AppState>,
    Path((job_id, file_name)): Path<(String, String)>,
) -> Result<axum::response::Response, ApiError> {
    let path = project_call(state, move |store| {
        store.pose_preview_path(&job_id, &file_name)
    })
    .await?;
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "image/png".to_owned())],
        bytes,
    )
        .into_response())
}
