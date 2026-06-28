use super::*;

/// Enqueue a `face_likeness_compare` job (epic 4406, sc-4415): score a CANDIDATE asset against a
/// SOURCE identity reference asset through the shared SCRFD+ArcFace face-likeness scorer in the worker.
/// GPU-routed (`auto`) like `dataset_face_analysis` — the native face stack lives on the GPU worker.
/// The client reads the result (`{ score, detected, method, sourceRef, reason? }`) from the completed
/// job. NOT a generation post-pass: it compares two already-existing assets on demand.
///
/// The handler validates the request and confirms both assets resolve on disk inside the project
/// (mirroring the prompt-refine vision-task resolution) so a garbled id fails fast as a 400 rather than
/// dispatching a doomed job; the worker independently re-confines each path before opening it
/// (defence-in-depth, epic 4484).
pub(crate) async fn create_face_likeness_compare_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<FaceLikenessCompareRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let project_id = payload.project_id.trim();
    let source_asset_id = payload.source_asset_id.trim();
    let candidate_asset_id = payload.candidate_asset_id.trim();
    if project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if source_asset_id.is_empty() {
        return Err(ApiError::bad_request("sourceAssetId is required"));
    }
    if candidate_asset_id.is_empty() {
        return Err(ApiError::bad_request("candidateAssetId is required"));
    }

    // Resolve both assets to confined on-disk paths up front so an invalid id is a clean 400 (the
    // worker re-resolves + re-confines independently; this is fail-fast UX, not the sole guard).
    let project_path = project_path_for_id(state.clone(), project_id).await?;
    resolve_project_asset_path(state.clone(), project_id, source_asset_id, &project_path).await?;
    resolve_project_asset_path(state.clone(), project_id, candidate_asset_id, &project_path)
        .await?;

    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.to_owned()));
    job_payload.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    job_payload.insert(
        "candidateAssetId".to_owned(),
        Value::String(candidate_asset_id.to_owned()),
    );

    let job = create_generation_job(
        state,
        JobType::FaceLikenessCompare,
        Some(project_id.to_owned()),
        None,
        job_payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Resolve a project asset id to an absolute, project-confined on-disk path. Mirrors the worker's
/// `load_reference_image` / the prompt-refine `resolve_image_caption_path`: read the asset record's
/// relative `file.path`, join it under the project directory using only `Normal` components (rejecting
/// `..`/absolute traversal), and confirm the file exists. Returns a 400 for a missing/garbled record or
/// a path that escapes the project root.
async fn resolve_project_asset_path(
    state: AppState,
    project_id: &str,
    asset_id: &str,
    project_path: &std::path::Path,
) -> Result<(), ApiError> {
    let project_id_owned = project_id.to_owned();
    let asset_id_owned = asset_id.to_owned();
    let asset = project_call(state, move |store| {
        store.get_asset(&project_id_owned, &asset_id_owned)
    })
    .await?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Asset has no file path"))?;
    let mut path = project_path.to_path_buf();
    for component in std::path::Path::new(rel).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => {
                return Err(ApiError::bad_request(
                    "Asset path must stay inside the project directory",
                ))
            }
        }
    }
    if !path.exists() {
        return Err(ApiError::bad_request("Asset image not found on disk"));
    }
    Ok(())
}
