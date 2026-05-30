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
