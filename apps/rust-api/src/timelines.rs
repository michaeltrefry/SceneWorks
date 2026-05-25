use super::*;

pub(crate) async fn list_timelines(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<sceneworks_core::project_store::TimelineSummary>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.list_timelines(&project_id)).await?,
    ))
}

pub(crate) async fn create_timeline(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<TimelineCreateRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let timeline = project_call(state, move |store| {
        store.create_timeline(
            &project_id,
            &payload.name,
            &payload.aspect_ratio,
            payload.fps,
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(timeline)))
}

pub(crate) async fn get_timeline(
    State(state): State<AppState>,
    Path((project_id, timeline_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_timeline(&project_id, &timeline_id)
        })
        .await?,
    ))
}

pub(crate) async fn update_timeline(
    State(state): State<AppState>,
    Path((project_id, timeline_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TimelineSaveRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.save_existing_timeline(&project_id, &timeline_id, payload.timeline)
        })
        .await?,
    ))
}

pub(crate) async fn create_timeline_export(
    State(state): State<AppState>,
    Path((project_id, timeline_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TimelineExportRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_timeline_export(&payload)?;
    let timeline_result = project_call(state.clone(), {
        let project_id = project_id.clone();
        let timeline_id = timeline_id.clone();
        move |store| store.timeline_file_and_document(&project_id, &timeline_id)
    })
    .await?;
    let timeline_name = timeline_result
        .document
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("Timeline")
        .to_owned();
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert("timelineId".to_owned(), Value::String(timeline_id));
    job_payload.insert("timelineName".to_owned(), Value::String(timeline_name));
    job_payload.insert(
        "timelinePath".to_owned(),
        Value::String(timeline_result.file.relative_path),
    );
    job_payload.insert("resolution".to_owned(), json!(payload.resolution));
    job_payload.insert("fps".to_owned(), json!(payload.fps));
    let job = create_generation_job(
        state,
        JobType::TimelineExport,
        Some(project_id),
        None,
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn extract_timeline_frame(
    State(state): State<AppState>,
    Path((project_id, timeline_id, item_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<FrameExtractRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_frame_extract(&payload)?;
    let timeline_result = project_call(state.clone(), {
        let project_id = project_id.clone();
        let timeline_id = timeline_id.clone();
        move |store| store.timeline_file_and_document(&project_id, &timeline_id)
    })
    .await?;
    let item = find_timeline_item(&timeline_result.document, &item_id)?;
    let source_asset_id = required_string_field(item, "assetId")?.to_owned();
    let timestamp = source_timestamp_for_item(item, payload.playhead_seconds)?;
    let timeline_name = timeline_result
        .document
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("Timeline")
        .to_owned();
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert("timelineId".to_owned(), Value::String(timeline_id));
    job_payload.insert("timelineName".to_owned(), Value::String(timeline_name));
    job_payload.insert(
        "timelinePath".to_owned(),
        Value::String(timeline_result.file.relative_path),
    );
    job_payload.insert("timelineItemId".to_owned(), Value::String(item_id));
    job_payload.insert("sourceAssetId".to_owned(), Value::String(source_asset_id));
    job_payload.insert("sourceTimestamp".to_owned(), json!(timestamp));
    job_payload.insert(
        "playheadSeconds".to_owned(),
        json!(payload.playhead_seconds),
    );
    job_payload.insert(
        "intendedUse".to_owned(),
        Value::String(payload.intended_use),
    );
    let job = create_generation_job(
        state,
        JobType::FrameExtract,
        Some(project_id),
        None,
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}
