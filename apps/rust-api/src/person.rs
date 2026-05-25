use super::*;

pub(crate) async fn list_person_tracks(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.list_person_tracks(&project_id)).await?,
    ))
}

pub(crate) async fn get_person_track(
    State(state): State<AppState>,
    Path((project_id, track_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_person_track(&project_id, &track_id)
        })
        .await?,
    ))
}

pub(crate) async fn save_person_track_corrections(
    State(state): State<AppState>,
    Path((project_id, track_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<PersonTrackCorrectionsRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.set_person_track_corrections(&project_id, &track_id, payload.corrections)
        })
        .await?,
    ))
}

pub(crate) async fn create_person_detection_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<PersonDetectionJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_person_detection_job(&payload)?;
    let project_name = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.project_stem(&project_id)
    })
    .await?;
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert(
        "sourceAssetId".to_owned(),
        Value::String(payload.source_asset_id),
    );
    job_payload.insert(
        "sourceTimestamp".to_owned(),
        payload.source_timestamp.map_or(Value::Null, Value::from),
    );
    if payload.preview {
        job_payload.insert("preview".to_owned(), Value::Bool(true));
    }
    let job = create_generation_job(
        state,
        JobType::PersonDetect,
        Some(project_id),
        Some(project_name),
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn create_person_track_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<PersonTrackJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_person_track_job(&payload)?;
    let project_name = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.project_stem(&project_id)
    })
    .await?;
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert(
        "sourceAssetId".to_owned(),
        Value::String(payload.source_asset_id),
    );
    job_payload.insert(
        "representativeFrameAssetId".to_owned(),
        Value::String(payload.representative_frame_asset_id),
    );
    job_payload.insert("detection".to_owned(), Value::Object(payload.detection));
    job_payload.insert("trackName".to_owned(), Value::String(payload.track_name));
    if payload.preview {
        job_payload.insert("preview".to_owned(), Value::Bool(true));
    }
    let job = create_generation_job(
        state,
        JobType::PersonTrack,
        Some(project_id),
        Some(project_name),
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}
