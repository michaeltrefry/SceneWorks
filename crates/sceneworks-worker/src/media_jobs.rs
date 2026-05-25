use super::*;

#[derive(Debug, Clone)]
pub(crate) struct TimelineExportRequest {
    project_id: String,
    timeline_id: String,
    timeline_name: String,
    timeline_path: String,
    resolution: u32,
    fps: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct FfmpegContext<'a> {
    api: &'a ApiClient,
    settings: &'a Settings,
    job_id: &'a str,
    cancel_message: &'a str,
}

pub(crate) async fn run_frame_extract_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing frame extraction.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Frame extraction canceled before reading media.",
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Extracting,
            0.25,
            "Extracting timeline frame.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_frame_extract(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Timeline frame saved as an asset.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_frame_extract(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let timestamp = payload_f64(&job.payload, "sourceTimestamp", 0.0).clamp(0.0, 3600.0);
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_media_rel = required_value_str(
        source_asset.get("file").ok_or_else(|| {
            WorkerError::InvalidPayload("Source asset file is missing.".to_owned())
        })?,
        "path",
    )?;
    let source_media_path = safe_project_path(&project_path, source_media_rel)?;
    if !source_media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Source media not found: {}",
            source_media_path.display()
        )));
    }

    let frames_dir = project_path.join("assets").join("frames");
    tokio::fs::create_dir_all(&frames_dir).await?;
    tokio::fs::create_dir_all(project_path.join("recipes")).await?;
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let filename = format!(
        "{}_frame_{}.png",
        &created_at[..10],
        asset_suffix(&asset_id)
    );
    let media_rel = format!("assets/frames/{filename}");
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");

    let ffmpeg_context = FfmpegContext {
        api,
        settings,
        job_id: &job.id,
        cancel_message: "Frame extraction canceled by user.",
    };
    render_frame_png(
        "ffmpeg",
        &source_media_path,
        &temp_path,
        timestamp,
        1920,
        1080,
        Some(ffmpeg_context),
    )
    .await?;
    tokio::fs::rename(&temp_path, &media_path).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.85,
            "Saving extracted frame asset.",
            None,
            None,
            None,
        ),
    )
    .await?;
    if let Err(error) = check_cancel(
        api,
        &job.id,
        "Frame extraction canceled before asset promotion.",
    )
    .await
    {
        let _ = tokio::fs::remove_file(&media_path).await;
        return Err(error);
    }

    let timeline_id = job
        .payload
        .get("timelineId")
        .cloned()
        .unwrap_or(Value::Null);
    let timeline_item_id = job
        .payload
        .get("timelineItemId")
        .cloned()
        .unwrap_or(Value::Null);
    let playhead_seconds = job
        .payload
        .get("playheadSeconds")
        .cloned()
        .unwrap_or(Value::Null);
    let intended_use = optional_payload_string(&job.payload, "intendedUse").unwrap_or("reuse");
    let source_display_name = source_asset
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("clip");
    let source_rel = relative_path(&project_path, &source_media_path)?;
    let asset = json!({
        "schemaVersion": 1,
        "id": asset_id.clone(),
        "projectId": project_id,
        "generationSetId": Value::Null,
        "type": "frame",
        "displayName": format!("Frame {timestamp:.2}s from {source_display_name}"),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": 1920,
            "height": 1080,
            "duration": Value::Null,
            "fps": Value::Null
        },
        "status": {
            "favorite": false,
            "rating": 0,
            "rejected": false,
            "trashed": false
        },
        "recipe": {
            "mode": "frame_extract",
            "model": "timeline-frame-extract",
            "adapter": "ffmpeg-frame-extract",
            "prompt": format!("Extract frame at {timestamp:.2}s"),
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "timelineId": timeline_id,
                "timelineItemId": timeline_item_id,
                "playheadSeconds": playhead_seconds,
                "sourceTimestamp": timestamp,
                "intendedUse": intended_use
            },
            "rawAdapterSettings": { "sourcePath": source_rel }
        },
        "lineage": {
            "parents": [source_asset_id],
            "sourceAssetId": source_asset_id,
            "sourceTimestamp": timestamp,
            "timelineId": job.payload.get("timelineId").cloned().unwrap_or(Value::Null),
            "timelineItemId": job.payload.get("timelineItemId").cloned().unwrap_or(Value::Null),
            "intendedUse": intended_use,
            "jobId": job.id
        }
    });
    let sidecar_path = media_path.with_extension("sceneworks.json");
    write_json_value(&sidecar_path, &asset).await?;
    write_json_value(
        &project_path
            .join("recipes")
            .join(format!("{asset_id}.recipe.json")),
        &asset["recipe"],
    )
    .await?;
    store.index_asset_sidecar(project_id, &sidecar_path)?;

    let mut result = JsonObject::new();
    result.insert("assetIds".to_owned(), json!([asset_id]));
    result.insert("assets".to_owned(), json!([asset]));
    result.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    result.insert("sourceTimestamp".to_owned(), json!(timestamp));
    result.insert(
        "timelineId".to_owned(),
        job.payload
            .get("timelineId")
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert(
        "timelineItemId".to_owned(),
        job.payload
            .get("timelineItemId")
            .cloned()
            .unwrap_or(Value::Null),
    );
    Ok(result)
}

pub(crate) async fn run_person_detect_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing representative frame analysis.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Person detection canceled before frame extraction.",
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Extracting,
            0.25,
            "Extracting representative frame.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_person_detect(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Person candidates detected.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_person_detect(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_file = source_asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset file is missing.".to_owned()))?;
    let source_media_rel = required_value_str(source_file, "path")?;
    let source_media_path = safe_project_path(&project_path, source_media_rel)?;
    if !source_media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Source media not found: {}",
            source_media_path.display()
        )));
    }

    let duration = source_file
        .get("duration")
        .map_or(6.0, |value| value_f64(value, 6.0))
        .clamp(0.0, 3600.0);
    let timestamp = payload_f64(
        &job.payload,
        "sourceTimestamp",
        if duration > 0.0 { duration * 0.25 } else { 0.0 },
    )
    .clamp(0.0, duration.max(3600.0));

    let frames_dir = project_path.join("assets").join("frames");
    tokio::fs::create_dir_all(&frames_dir).await?;
    tokio::fs::create_dir_all(project_path.join("recipes")).await?;
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let filename = format!(
        "{}_person-frame_{}.png",
        &created_at[..10],
        asset_suffix(&asset_id)
    );
    let media_rel = format!("assets/frames/{filename}");
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");

    let ffmpeg_context = FfmpegContext {
        api,
        settings,
        job_id: &job.id,
        cancel_message: "Person detection canceled by user.",
    };
    render_frame_png(
        "ffmpeg",
        &source_media_path,
        &temp_path,
        timestamp,
        1280,
        720,
        Some(ffmpeg_context),
    )
    .await?;
    tokio::fs::rename(&temp_path, &media_path).await?;

    let detections = candidate_people(1280, 720, source_asset_id, timestamp);
    let source_display_name = source_asset
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("clip");
    let source_rel = relative_path(&project_path, &source_media_path)?;
    let asset = json!({
        "schemaVersion": 1,
        "id": asset_id.clone(),
        "projectId": project_id,
        "generationSetId": Value::Null,
        "type": "frame",
        "displayName": format!("Person selection frame from {source_display_name}"),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": 1280,
            "height": 720,
            "duration": Value::Null,
            "fps": Value::Null
        },
        "status": {
            "favorite": false,
            "rating": 0,
            "rejected": false,
            "trashed": false
        },
        "recipe": {
            "mode": "person_detect",
            "model": "procedural-person-detector",
            "adapter": "procedural_person_tracking",
            "prompt": "Detect selectable people in representative frame",
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sourceTimestamp": timestamp,
                "detectionCount": detections.len(),
                "personDetectionActive": false
            },
            "rawAdapterSettings": { "sourcePath": source_rel }
        },
        "lineage": {
            "parents": [source_asset_id],
            "sourceAssetId": source_asset_id,
            "sourceTimestamp": timestamp,
            "jobId": job.id
        }
    });

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.78,
            "Saving representative frame and candidate boxes.",
            None,
            None,
            None,
        ),
    )
    .await?;
    if let Err(error) = check_cancel(
        api,
        &job.id,
        "Person detection canceled before asset promotion.",
    )
    .await
    {
        let _ = tokio::fs::remove_file(&media_path).await;
        return Err(error);
    }
    let sidecar_path = media_path.with_extension("sceneworks.json");
    write_json_value(&sidecar_path, &asset).await?;
    write_json_value(
        &project_path
            .join("recipes")
            .join(format!("{asset_id}.recipe.json")),
        &asset["recipe"],
    )
    .await?;
    store.index_asset_sidecar(project_id, &sidecar_path)?;

    let mut result = JsonObject::new();
    result.insert("frameAssetId".to_owned(), Value::String(asset_id));
    result.insert("frameAsset".to_owned(), asset);
    result.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    result.insert("sourceTimestamp".to_owned(), json!(timestamp));
    result.insert("detections".to_owned(), Value::Array(detections));
    result.insert(
        "limits".to_owned(),
        json!({
            "maskStorage": "deferred",
            "correction": "single selected box corrections can be added to the track sidecar later"
        }),
    );
    Ok(result)
}

pub(crate) async fn run_person_track_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing selected-person tracking.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Person tracking canceled before saving.").await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Tracking,
            0.35,
            "Tracking selected person through sampled frames.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_person_track(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Reusable person track saved.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_person_track(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let detection = job
        .payload
        .get("detection")
        .cloned()
        .filter(Value::is_object)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Selected detection metadata is required".to_owned())
        })?;
    if detection
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .is_none()
    {
        return Err(WorkerError::InvalidPayload(
            "Selected detection metadata is required".to_owned(),
        ));
    }
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_file = source_asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset file is missing.".to_owned()))?;
    let duration = source_file
        .get("duration")
        .map_or(6.0, |value| value_f64(value, 6.0))
        .clamp(1.0, 3600.0);
    let frames = track_frames_from_detection(&detection, duration);
    let average_confidence = frames
        .iter()
        .map(|frame| {
            frame
                .get("confidence")
                .map_or(0.0, |value| value_f64(value, 0.0))
        })
        .sum::<f64>()
        / (frames.len().max(1) as f64);
    let track_id = format!("track_{}", Uuid::new_v4().simple());
    let track_name =
        optional_payload_string(&job.payload, "trackName").unwrap_or("Selected person");
    let representative_frame_asset_id = job
        .payload
        .get("representativeFrameAssetId")
        .cloned()
        .unwrap_or(Value::Null);
    let raw_selected_detection = detection.clone();
    let created_at = now_rfc3339();
    let source_display_name = source_asset
        .get("displayName")
        .cloned()
        .unwrap_or(Value::Null);
    let track = json!({
        "schemaVersion": 1,
        "id": track_id.clone(),
        "projectId": project_id,
        "name": track_name,
        "createdAt": created_at,
        "sourceAssetId": source_asset_id,
        "sourceDisplayName": source_display_name,
        "representativeFrameAssetId": representative_frame_asset_id,
        "selectedDetection": detection,
        "frames": frames,
        "corrections": [],
        "status": {
            "sampleRateFps": PERSON_TRACK_SAMPLE_RATE_FPS,
            "maskState": "deferred",
            "averageConfidence": round_to(average_confidence, 3),
            "correctionState": "ready_for_box_corrections",
            "personTrackingActive": false
        },
        "recipe": {
            "mode": "person_track",
            "model": "procedural-person-tracker",
            "adapter": "procedural_person_tracking",
            "prompt": format!("Track {track_name}"),
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sampleRateFps": PERSON_TRACK_SAMPLE_RATE_FPS,
                "personDetectionActive": false,
                "personTrackingActive": false
            },
            "rawAdapterSettings": { "selectedDetection": raw_selected_detection }
        },
        "lineage": {
            "jobId": job.id,
            "parents": [source_asset_id, job.payload.get("representativeFrameAssetId").cloned().unwrap_or(Value::Null)]
        }
    });

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.82,
            "Saving reusable person track metadata.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Person tracking canceled before sidecar write.",
    )
    .await?;
    let track_path = project_path
        .join("person-tracks")
        .join(format!("{track_id}.sceneworks.person-track.json"));
    write_json_value(&track_path, &track).await?;
    let relative = relative_path(&project_path, &track_path)?;
    let mut result = JsonObject::new();
    result.insert("trackId".to_owned(), Value::String(track_id));
    result.insert("track".to_owned(), track);
    result.insert("path".to_owned(), Value::String(relative));
    Ok(result)
}

pub(crate) async fn run_timeline_export_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.06,
            "Preparing timeline export.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Timeline export canceled before rendering.").await?;
    let request = export_request_from_job(job)?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let timeline_path = safe_project_path(&project_path, &request.timeline_path)?;
    let timeline = read_json_value(&timeline_path).await?;
    let (width, height) = output_dimensions(
        timeline
            .get("aspectRatio")
            .and_then(Value::as_str)
            .unwrap_or("16:9"),
        request.resolution,
    );
    let mut items = main_track_items(&timeline);
    items.sort_by(|left, right| {
        item_f64(left, "timelineStart", 0.0).total_cmp(&item_f64(right, "timelineStart", 0.0))
    });
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no main video items to export.".to_owned(),
        ));
    }
    let (plan, duration) = plan_segments(&items)?;

    let temp_dir = tempfile::Builder::new()
        .prefix(&format!(
            "sceneworks_export_{}_",
            safe_download_dir(&job.id)
        ))
        .tempdir()?;
    let tmp_path = temp_dir.path().to_path_buf();

    let spec = RenderSpec {
        width,
        height,
        fps: request.fps,
    };
    let export = TimelineExport {
        context: FfmpegContext {
            api,
            settings,
            job_id: &job.id,
            cancel_message: "Timeline export canceled by user.",
        },
        store: &store,
        request,
        project_path,
        timeline,
        spec,
    };
    let segments = export.render(&plan, &tmp_path).await?;
    export.finalize(&segments, &tmp_path, duration).await?;
    Ok(())
}

/// A timeline item resolved to its render plan: the optional black gap to emit
/// before it (when items leave a hole in the timeline) plus the transition it
/// carries in. Pure data, so `plan_segments` can build it without touching
/// ffmpeg or the store and the planning logic stays unit-testable.
#[derive(Debug)]
pub(crate) struct PlannedItem<'a> {
    pub(crate) leading_gap: Option<f64>,
    pub(crate) item: &'a Value,
    pub(crate) transition: Option<String>,
    pub(crate) transition_duration: f64,
}

/// Walk the sorted main-track items into an ordered render plan, inserting a
/// black gap wherever an item does not abut the previous one, and return the
/// plan together with the total timeline duration. No I/O: the gap, transition,
/// and span-validation logic can be exercised in isolation.
pub(crate) fn plan_segments(items: &[Value]) -> WorkerResult<(Vec<PlannedItem<'_>>, f64)> {
    let mut plan = Vec::with_capacity(items.len());
    let mut cursor = 0.0_f64;
    for item in items {
        let start = item_f64(item, "timelineStart", 0.0);
        let item_end = item_f64(item, "timelineEnd", start);
        if item_end <= start {
            return Err(WorkerError::InvalidPayload(
                "timelineEnd must be greater than timelineStart.".to_owned(),
            ));
        }
        let leading_gap = if start > cursor {
            let gap = start - cursor;
            cursor = start;
            Some(gap)
        } else {
            None
        };
        let transition_in = item.get("transitionIn").unwrap_or(&Value::Null);
        plan.push(PlannedItem {
            leading_gap,
            item,
            transition: transition_in
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_owned),
            transition_duration: value_f64(
                transition_in.get("duration").unwrap_or(&Value::Null),
                DEFAULT_TRANSITION_DURATION_SECONDS,
            ),
        });
        cursor = cursor.max(item_end);
    }
    Ok((plan, cursor))
}

/// Per-job state for a timeline MP4 export, resolved once so the rendering and
/// finalization steps can share it without threading a long argument list.
struct TimelineExport<'a> {
    context: FfmpegContext<'a>,
    store: &'a ProjectStore,
    request: TimelineExportRequest,
    project_path: PathBuf,
    timeline: Value,
    spec: RenderSpec,
}

impl TimelineExport<'_> {
    /// Render each planned segment (gaps then source items) into `tmp_path`,
    /// reporting progress and honoring cancellation between items.
    async fn render(
        &self,
        plan: &[PlannedItem<'_>],
        tmp_path: &Path,
    ) -> WorkerResult<Vec<TimelineSegment>> {
        let mut segments = Vec::new();
        let total = plan.len().max(1);
        for (index, planned) in plan.iter().enumerate() {
            check_cancel(
                self.context.api,
                self.context.job_id,
                "Timeline export canceled by user.",
            )
            .await?;
            if let Some(gap_duration) = planned.leading_gap {
                let gap_path = tmp_path.join(format!("segment_{:04}_gap.mp4", segments.len()));
                render_black_segment(
                    "ffmpeg",
                    &gap_path,
                    gap_duration,
                    self.spec,
                    Some(self.context),
                )
                .await?;
                segments.push(TimelineSegment {
                    path: gap_path,
                    duration: gap_duration,
                    transition: None,
                    transition_duration: 0.0,
                });
            }

            let asset_id = required_value_str(planned.item, "assetId")?;
            let asset = self.store.get_asset(&self.request.project_id, asset_id)?;
            let display_name = planned
                .item
                .get("displayName")
                .and_then(Value::as_str)
                .unwrap_or("item");
            let segment_path = tmp_path.join(format!(
                "segment_{:04}_{}.mp4",
                segments.len(),
                slugify(display_name, "timeline-export", Some(48))
            ));
            let duration = render_item_segment(
                "ffmpeg",
                &self.project_path,
                planned.item,
                &asset,
                &segment_path,
                self.spec,
                Some(self.context),
            )
            .await?;
            segments.push(TimelineSegment {
                path: segment_path,
                duration,
                transition: planned.transition.clone(),
                transition_duration: planned.transition_duration,
            });
            update_job(
                self.context.api,
                self.context.job_id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Rendering,
                    0.12 + (((index + 1) as f64 / total as f64) * 0.58),
                    "Rendering timeline segments.",
                    None,
                    None,
                    None,
                ),
            )
            .await?;
        }
        Ok(segments)
    }

    /// Mux the rendered segments into the project's render directory, write the
    /// asset sidecar and recipe, index the asset, and report completion.
    async fn finalize(
        &self,
        segments: &[TimelineSegment],
        tmp_path: &Path,
        duration: f64,
    ) -> WorkerResult<()> {
        let output_rel = format!(
            "assets/renders/{}_{}_{}.mp4",
            &now_rfc3339()[..10],
            slugify(&self.request.timeline_name, "timeline-export", Some(48)),
            asset_suffix(self.context.job_id)
        );
        let output_path = self.project_path.join(&output_rel);
        tokio::fs::create_dir_all(output_path.parent().ok_or_else(|| {
            WorkerError::InvalidPayload("Render output has no parent directory.".to_owned())
        })?)
        .await?;
        update_job(
            self.context.api,
            self.context.job_id,
            progress_payload(
                JobStatus::Saving,
                ProgressStage::Muxing,
                0.78,
                "Muxing MP4 export.",
                None,
                None,
                None,
            ),
        )
        .await?;
        mux_segments(
            "ffmpeg",
            segments,
            tmp_path,
            &output_path,
            Some(self.context),
        )
        .await?;

        let asset = build_render_asset(
            &self.request,
            &self.timeline,
            self.context.job_id,
            &output_rel,
            self.spec.width,
            self.spec.height,
            duration,
        );
        let sidecar_path = output_path.with_extension("sceneworks.json");
        write_json_value(&sidecar_path, &asset).await?;
        tokio::fs::create_dir_all(self.project_path.join("recipes")).await?;
        let asset_id = required_value_str(&asset, "id")?.to_owned();
        write_json_value(
            &self
                .project_path
                .join("recipes")
                .join(format!("{asset_id}.recipe.json")),
            &asset["recipe"],
        )
        .await?;
        self.store
            .index_asset_sidecar(&self.request.project_id, &sidecar_path)?;

        let mut result = JsonObject::new();
        result.insert("assetIds".to_owned(), json!([asset_id]));
        result.insert("assets".to_owned(), json!([asset]));
        result.insert(
            "timelineId".to_owned(),
            Value::String(self.request.timeline_id.clone()),
        );
        result.insert("renderPath".to_owned(), Value::String(output_rel));
        result.insert(
            "adapter".to_owned(),
            Value::String("ffmpeg_timeline".to_owned()),
        );
        update_job(
            self.context.api,
            self.context.job_id,
            progress_payload(
                JobStatus::Completed,
                ProgressStage::Completed,
                1.0,
                "Timeline MP4 export saved.",
                None,
                Some(result),
                None,
            ),
        )
        .await?;
        Ok(())
    }
}

pub(crate) fn candidate_people(
    width: u32,
    height: u32,
    source_asset_id: &str,
    timestamp: f64,
) -> Vec<Value> {
    let seed = format!("{source_asset_id}:{timestamp:.3}:{width}x{height}");
    let digest = Sha256::digest(seed.as_bytes());
    let templates = [
        (0.34, 0.16, 0.24, 0.68, 0.91),
        (0.58, 0.20, 0.20, 0.58, 0.78),
        (0.14, 0.26, 0.17, 0.50, 0.66),
    ];
    templates
        .iter()
        .enumerate()
        .map(|(index, (x, y, box_width, box_height, confidence))| {
            let jitter = ((digest[index] % 13) as f64 - 6.0) / 1000.0;
            json!({
                "id": format!("person_{}", index + 1),
                "label": format!("Person {}", index + 1),
                "confidence": round_to(*confidence - index as f64 * 0.04, 2),
                "box": {
                    "x": (*x + jitter).clamp(0.02, 0.92),
                    "y": *y,
                    "width": *box_width,
                    "height": *box_height
                },
                "maskState": "deferred",
                "frameWidth": width,
                "frameHeight": height
            })
        })
        .collect()
}

pub(crate) fn track_frames_from_detection(detection: &Value, duration: f64) -> Vec<Value> {
    let sample_count = ((duration.max(1.0) * PERSON_TRACK_SAMPLE_RATE_FPS).round() as usize)
        .clamp(3, PERSON_TRACK_MAX_SAMPLES);
    let base_confidence =
        value_f64(detection.get("confidence").unwrap_or(&Value::Null), 0.82).clamp(0.0, 1.0);
    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count.saturating_sub(1).max(1) as f64);
            json!({
                "timestamp": round_to(t * duration.max(0.0), 3),
                "box": {
                    "x": round_to(detection_box_f64(detection, "x", 0.35, 0.0, 1.0) + (t - 0.5) * PERSON_TRACK_X_DRIFT, 4),
                    "y": round_to(detection_box_f64(detection, "y", 0.16, 0.0, 1.0), 4),
                    "width": round_to(detection_box_f64(detection, "width", 0.24, 0.01, 1.0), 4),
                    "height": round_to(detection_box_f64(detection, "height", 0.68, 0.01, 1.0), 4)
                },
                "confidence": 0.5_f64.max(round_to(base_confidence - index as f64 * 0.006, 3)),
                "mask": Value::Null
            })
        })
        .collect()
}

pub(crate) fn detection_box_f64(
    detection: &Value,
    field: &str,
    default: f64,
    min_value: f64,
    max_value: f64,
) -> f64 {
    detection
        .get("box")
        .and_then(|value| value.get(field))
        .map_or(default, |value| value_f64(value, default))
        .clamp(min_value, max_value)
}

pub(crate) fn round_to(value: f64, places: u32) -> f64 {
    let factor = 10_f64.powi(i32::try_from(places).unwrap_or(0));
    (value * factor).round() / factor
}

pub(crate) fn export_request_from_job(job: &JobSnapshot) -> WorkerResult<TimelineExportRequest> {
    Ok(TimelineExportRequest {
        project_id: required_payload_string(&job.payload, "projectId")?.to_owned(),
        timeline_id: required_payload_string(&job.payload, "timelineId")?.to_owned(),
        timeline_name: optional_payload_string(&job.payload, "timelineName")
            .unwrap_or("Timeline")
            .to_owned(),
        timeline_path: required_payload_string(&job.payload, "timelinePath")?.to_owned(),
        resolution: payload_u32(&job.payload, "resolution", 720).clamp(240, 2160),
        fps: payload_u32(&job.payload, "fps", 30).clamp(1, 60),
    })
}

pub(crate) async fn render_frame_png(
    ffmpeg: &str,
    source_path: &Path,
    output_path: &Path,
    timestamp: f64,
    width: u32,
    height: u32,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x12110f,format=rgb24"
    );
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-ss".to_owned(),
            format!("{:.3}", timestamp.max(0.0)),
            "-i".to_owned(),
            source_path.display().to_string(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-vf".to_owned(),
            filters,
            "-f".to_owned(),
            "image2".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await?;
    if !tokio::fs::try_exists(output_path).await? {
        return Err(WorkerError::InvalidPayload(format!(
            "FFmpeg did not produce frame output: {}",
            output_path.display()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct TimelineSegment {
    path: PathBuf,
    duration: f64,
    transition: Option<String>,
    transition_duration: f64,
}

pub(crate) fn main_track_items(timeline: &Value) -> Vec<Value> {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .and_then(|tracks| {
            tracks
                .iter()
                .find(|track| {
                    track.get("id").and_then(Value::as_str) == Some("track_main")
                        || track.get("kind").and_then(Value::as_str) == Some("video")
                })
                .and_then(|track| track.get("items").and_then(Value::as_array))
        })
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn output_dimensions(aspect_ratio: &str, resolution: u32) -> (u32, u32) {
    let resolution = resolution.max(2);
    let (width, height) = match aspect_ratio {
        "9:16" => (resolution, ((resolution as f64) * 16.0 / 9.0).ceil() as u32),
        "1:1" => (resolution, resolution),
        _ => (((resolution as f64) * 16.0 / 9.0).ceil() as u32, resolution),
    };
    (even(width), even(height))
}

pub(crate) fn even(value: u32) -> u32 {
    if value % 2 == 0 {
        value
    } else {
        value + 1
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderSpec {
    width: u32,
    height: u32,
    fps: u32,
}

pub(crate) async fn render_black_segment(
    ffmpeg: &str,
    output_path: &Path,
    duration: f64,
    spec: RenderSpec,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "lavfi".to_owned(),
            "-i".to_owned(),
            format!(
                "color=c=black:s={}x{}:r={}",
                spec.width, spec.height, spec.fps
            ),
            "-t".to_owned(),
            format!("{duration:.3}"),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await
}

pub(crate) async fn render_item_segment(
    ffmpeg: &str,
    project_path: &Path,
    item: &Value,
    asset: &Value,
    output_path: &Path,
    spec: RenderSpec,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<f64> {
    let file = asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Timeline asset file is missing.".to_owned()))?;
    let media_rel = required_value_str(file, "path")?;
    let media_path = safe_project_path(project_path, media_rel)?;
    if !media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Timeline source file is missing: {}",
            media_path.display()
        )));
    }

    let source_in = item_f64(item, "sourceIn", 0.0);
    let source_out = item_f64(item, "sourceOut", item_f64(item, "timelineEnd", 4.0));
    let timeline_duration =
        item_f64(item, "timelineEnd", 4.0) - item_f64(item, "timelineStart", 0.0);
    let source_duration = (source_out - source_in).max(0.1);
    let speed = item_f64(item, "speed", 1.0).max(0.1);
    let duration = if timeline_duration > 0.0 {
        timeline_duration.max(0.1)
    } else {
        (source_duration / speed).max(0.1)
    };
    let mut vf = vec![
        format!(
            "scale={}:{}:force_original_aspect_ratio=decrease",
            spec.width, spec.height
        ),
        format!(
            "pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=black",
            spec.width, spec.height
        ),
        format!("fps={}", spec.fps),
        "format=yuv420p".to_owned(),
    ];
    let transition_in = item.get("transitionIn").unwrap_or(&Value::Null);
    let transition_out = item.get("transitionOut").unwrap_or(&Value::Null);
    if transition_in.get("type").and_then(Value::as_str) == Some("fade_from_black") {
        let fade_duration = duration.min(value_f64(
            transition_in.get("duration").unwrap_or(&Value::Null),
            0.5,
        ));
        vf.push(format!("fade=t=in:st=0:d={fade_duration:.3}"));
    }
    if transition_out.get("type").and_then(Value::as_str) == Some("fade_to_black") {
        let fade_duration = duration.min(value_f64(
            transition_out.get("duration").unwrap_or(&Value::Null),
            0.5,
        ));
        vf.push(format!(
            "fade=t=out:st={:.3}:d={fade_duration:.3}",
            (duration - fade_duration).max(0.0)
        ));
    }

    let media_type = asset.get("type").and_then(Value::as_str);
    let mime_type = file
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let is_image_source = media_type != Some("video")
        && (media_type == Some("image") || mime_type.starts_with("image/"));
    if is_image_source {
        run_ffmpeg(
            vec![
                ffmpeg.to_owned(),
                "-y".to_owned(),
                "-loop".to_owned(),
                "1".to_owned(),
                "-framerate".to_owned(),
                spec.fps.to_string(),
                "-i".to_owned(),
                media_path.display().to_string(),
                "-t".to_owned(),
                format!("{duration:.3}"),
                "-vf".to_owned(),
                vf.join(","),
                "-an".to_owned(),
                output_path.display().to_string(),
            ],
            context,
        )
        .await?;
        return Ok(duration);
    }

    let setpts = format!("setpts={:.6}*PTS", 1.0 / speed);
    let filters = std::iter::once(setpts)
        .chain(vf)
        .collect::<Vec<_>>()
        .join(",");
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-ss".to_owned(),
            format!("{source_in:.3}"),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-t".to_owned(),
            format!("{source_duration:.3}"),
            "-vf".to_owned(),
            filters,
            "-an".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await?;
    Ok(duration)
}

pub(crate) async fn mux_segments(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    tmp_path: &Path,
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    if segments
        .iter()
        .skip(1)
        .any(|segment| segment.transition.as_deref() == Some("crossfade"))
    {
        return mux_with_crossfades(ffmpeg, segments, tmp_path, output_path, context).await;
    }
    let list_path = tmp_path.join("concat.txt");
    tokio::fs::write(
        &list_path,
        concat_file_contents(segments.iter().map(|segment| &segment.path)),
    )
    .await?;
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "concat".to_owned(),
            "-safe".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            list_path.display().to_string(),
            "-c".to_owned(),
            "copy".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await
}

pub(crate) async fn mux_with_crossfades(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    tmp_path: &Path,
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let Some(first) = segments.first() else {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no rendered segments to mux.".to_owned(),
        ));
    };
    let mut current = first.path.clone();
    let mut current_duration = first.duration;
    for (index, segment) in segments.iter().enumerate().skip(1) {
        let merged = tmp_path.join(format!("xfade_{index:04}.mp4"));
        if segment.transition.as_deref() == Some("crossfade") {
            let duration = crossfade_duration(segment.transition_duration);
            let offset = (current_duration - duration).max(0.0);
            run_ffmpeg(
                vec![
                    ffmpeg.to_owned(),
                    "-y".to_owned(),
                    "-i".to_owned(),
                    current.display().to_string(),
                    "-i".to_owned(),
                    segment.path.display().to_string(),
                    "-filter_complex".to_owned(),
                    format!(
                    "[0:v][1:v]xfade=transition=fade:duration={duration:.3}:offset={offset:.3},format=yuv420p[v]"
                ),
                    "-map".to_owned(),
                    "[v]".to_owned(),
                    merged.display().to_string(),
                ],
                context,
            )
            .await?;
            current_duration += segment.duration - duration;
        } else {
            let list_path = tmp_path.join(format!("concat_{index:04}.txt"));
            tokio::fs::write(
                &list_path,
                concat_file_contents([&current, &segment.path].into_iter()),
            )
            .await?;
            run_ffmpeg(
                vec![
                    ffmpeg.to_owned(),
                    "-y".to_owned(),
                    "-f".to_owned(),
                    "concat".to_owned(),
                    "-safe".to_owned(),
                    "0".to_owned(),
                    "-i".to_owned(),
                    list_path.display().to_string(),
                    "-c".to_owned(),
                    "copy".to_owned(),
                    merged.display().to_string(),
                ],
                context,
            )
            .await?;
            current_duration += segment.duration;
        }
        current = merged;
    }
    tokio::fs::rename(current, output_path).await?;
    Ok(())
}

pub(crate) fn crossfade_duration(duration: f64) -> f64 {
    duration.clamp(0.1, 1.5)
}

pub(crate) fn concat_file_contents<'a>(paths: impl Iterator<Item = &'a PathBuf>) -> String {
    paths
        .map(|path| {
            let path = path
                .display()
                .to_string()
                .replace('\\', "/")
                .replace('\'', "'\\''");
            format!("file '{path}'\n")
        })
        .collect()
}

pub(crate) fn build_render_asset(
    request: &TimelineExportRequest,
    timeline: &Value,
    job_id: &str,
    media_rel: &str,
    width: u32,
    height: u32,
    duration: f64,
) -> Value {
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let source_asset_ids = timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .filter_map(|item| item.get("assetId").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let aspect_ratio = timeline
        .get("aspectRatio")
        .and_then(Value::as_str)
        .unwrap_or("16:9");
    json!({
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": request.project_id,
        "generationSetId": Value::Null,
        "type": "render",
        "displayName": format!("{} export", request.timeline_name),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "video/mp4",
            "width": width,
            "height": height,
            "duration": (duration * 1000.0).round() / 1000.0,
            "fps": request.fps
        },
        "status": {
            "favorite": false,
            "rating": 0,
            "rejected": false,
            "trashed": false
        },
        "recipe": {
            "mode": "timeline_export",
            "model": "ffmpeg",
            "adapter": "ffmpeg_timeline",
            "prompt": request.timeline_name,
            "negativePrompt": "",
            "seed": Value::Null,
            "loras": [],
            "normalizedSettings": {
                "timelineId": request.timeline_id,
                "resolution": request.resolution,
                "width": width,
                "height": height,
                "fps": request.fps,
                "aspectRatio": aspect_ratio
            },
            "rawAdapterSettings": {
                "timelinePath": request.timeline_path,
                "renderer": "ffmpeg segment concat"
            }
        },
        "lineage": {
            "parents": source_asset_ids,
            "sourceAssetId": request.timeline_id,
            "sourceTimestamp": Value::Null,
            "jobId": job_id
        }
    })
}

pub(crate) async fn run_ffmpeg(
    args: Vec<String>,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let Some((program, arguments)) = args.split_first() else {
        return Err(WorkerError::InvalidPayload(
            "FFmpeg command is empty.".to_owned(),
        ));
    };
    // Let the host override the default ffmpeg binary via SCENEWORKS_FFMPEG. The
    // desktop app sets this to the venv's bundled imageio-ffmpeg (it ships no
    // system ffmpeg); the server stack / Docker leave it unset and use the
    // caller's "ffmpeg" on PATH.
    let resolved_program = match std::env::var("SCENEWORKS_FFMPEG") {
        Ok(path) if program.as_str() == "ffmpeg" && !path.trim().is_empty() => path,
        _ => program.clone(),
    };
    let mut child = Command::new(&resolved_program)
        .args(arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Failed to start FFmpeg. Ensure ffmpeg is installed and on PATH: {error}"
            ))
        })?;

    let mut stderr = child.stderr.take();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        if let Some(stderr) = stderr.as_mut() {
            let _ = stderr.read_to_end(&mut bytes).await;
        }
        bytes
    });

    let status = if let Some(context) = context {
        let mut interval = tokio::time::interval(progress_report_interval(context.settings));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                status = child.wait() => break status?,
                _ = interval.tick() => {
                    heartbeat(context.api, context.settings, WorkerStatus::Busy, Some(context.job_id)).await?;
                    if let Err(error) = check_cancel(context.api, context.job_id, context.cancel_message).await {
                        let _ = child.kill().await;
                        let _ = child.wait().await;
                        return Err(error);
                    }
                }
            }
        }
    } else {
        child.wait().await?
    };

    let stderr = stderr_task.await.unwrap_or_default();
    if status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&stderr);
    let bounded = bounded_tail(&stderr, 10, 2000);
    if bounded.trim().is_empty() {
        Err(WorkerError::InvalidPayload(
            "FFmpeg command failed without stderr output.".to_owned(),
        ))
    } else {
        Err(WorkerError::InvalidPayload(bounded))
    }
}
