use super::*;

pub(crate) async fn list_projects(
    State(state): State<AppState>,
) -> Result<Json<Vec<sceneworks_core::project_store::ProjectSummary>>, ApiError> {
    Ok(Json(
        project_call(state, |store| store.list_projects()).await?,
    ))
}

pub(crate) async fn create_project(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ProjectCreateRequest>,
) -> Result<
    (
        StatusCode,
        Json<sceneworks_core::project_store::ProjectSummary>,
    ),
    ApiError,
> {
    let project = project_call(state, move |store| store.create_project(&payload.name)).await?;
    Ok((StatusCode::CREATED, Json(project)))
}

pub(crate) async fn get_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<sceneworks_core::project_store::ProjectSummary>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.get_project(&project_id)).await?,
    ))
}

pub(crate) async fn reindex_project_endpoint(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<sceneworks_core::project_store::ReindexResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.reindex_project(&project_id)).await?,
    ))
}
