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

impl<'a> FfmpegContext<'a> {
    /// Build a context so callers in sibling modules (e.g. video generation,
    /// `video_jobs`) can drive [`run_ffmpeg`] with the same periodic-heartbeat +
    /// cooperative-cancel loop the in-module callers use.
    pub(crate) fn new(
        api: &'a ApiClient,
        settings: &'a Settings,
        job_id: &'a str,
        cancel_message: &'a str,
    ) -> Self {
        Self {
            api,
            settings,
            job_id,
            cancel_message,
        }
    }
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
    http_client: &reqwest::Client,
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
    let result = run_person_detect(api, settings, http_client, job).await?;
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
    http_client: &reqwest::Client,
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

    // Preview jobs (`preview: true`, claimed via the CPU worker's
    // person_detect_preview capability) keep the procedural placeholder. Real
    // jobs run the YOLO11 onnx detector (epic 3482, sc-3633) — model-backed,
    // `personDetectionActive: true`, and erroring honestly when the detector
    // can't run rather than silently degrading to boxes.
    let is_preview = job
        .payload
        .get("preview")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let confidence = job
        .payload
        .get("advanced")
        .and_then(|advanced| advanced.get("confidence"))
        .or_else(|| job.payload.get("confidence"))
        .map_or(0.25, |value| value_f64(value, 0.25))
        .clamp(0.01, 1.0);
    let (detections, detector_model, detector_adapter, detection_active, detector_meta) =
        if is_preview {
            (
                candidate_people(1280, 720, source_asset_id, timestamp),
                "procedural-person-detector".to_owned(),
                "procedural_person_tracking",
                false,
                Value::Null,
            )
        } else {
            let (boxes, device) = run_yolo11_person_detect(
                api,
                settings,
                http_client,
                job,
                media_path.clone(),
                confidence,
            )
            .await?;
            (
                boxes,
                "yolo11m".to_owned(),
                "yolo11_mlx",
                true,
                json!({ "backend": "mlx", "device": device, "model": "yolo11m" }),
            )
        };
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
            "model": detector_model,
            "adapter": detector_adapter,
            "prompt": "Detect selectable people in representative frame",
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sourceTimestamp": timestamp,
                "detectionCount": detections.len(),
                "confidence": confidence,
                "personDetectionActive": detection_active,
                "detector": detector_meta
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
        "personDetectionActive".to_owned(),
        Value::Bool(detection_active),
    );
    result.insert(
        "limits".to_owned(),
        json!({
            "maskStorage": "deferred",
            "correction": "single selected box corrections can be added to the track sidecar later"
        }),
    );
    Ok(result)
}

/// Run the native MLX YOLO11 person detector on a rendered frame, returning the
/// normalized detection array (Python `run_person_detect` shape) + the device
/// the model ran on. macOS-only: MLX (mlx-rs) is the Apple-Silicon backend
/// (epic 3482, sc-3633); the Python Ultralytics path serves Windows/Linux.
#[cfg(target_os = "macos")]
async fn run_yolo11_person_detect(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    frame_path: PathBuf,
    confidence: f64,
) -> WorkerResult<(Vec<Value>, &'static str)> {
    let download_context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Person detection canceled while fetching detector weights.",
        fresh_download: false,
    };
    let weights = crate::person_jobs::ensure_detector_weights(settings, &download_context).await?;
    let conf = confidence as f32;
    let result = tokio::task::spawn_blocking(move || {
        crate::person_jobs::detect_people_blocking(weights, frame_path, conf)
    })
    .await
    .map_err(|error| task_join_error("person detect task", error))??;
    let boxes =
        crate::person_jobs::detections_to_json(&result.detections, result.width, result.height);
    Ok((boxes, result.device))
}

#[cfg(not(target_os = "macos"))]
async fn run_yolo11_person_detect(
    _api: &ApiClient,
    _settings: &Settings,
    _http_client: &reqwest::Client,
    _job: &JobSnapshot,
    _frame_path: PathBuf,
    _confidence: f64,
) -> WorkerResult<(Vec<Value>, &'static str)> {
    Err(WorkerError::InvalidPayload(
        "Real person detection runs on the Python worker on this platform.".to_owned(),
    ))
}

/// Seconds the final frame-extraction seek is held inside the clip. `sample_timestamps`
/// is inclusive of both ends, so its last sample is exactly `duration` — but a video has
/// no frame at `duration` (the last decodable frame sits ~`1/fps` before it), and an
/// `ffmpeg -ss duration` accurate seek then yields no output and fails the whole track.
/// 0.2 s clears one frame for any clip ≥ 5 fps without meaningfully moving the sample.
/// macOS-only, like its sole caller `assemble_real_person_track` (the Python Win/Linux
/// path samples/extracts separately).
#[cfg(target_os = "macos")]
const FRAME_SEEK_GUARD_SECONDS: f64 = 0.2;

/// Clamp a sample timestamp to a frame-extraction seek that always lands on a real frame:
/// never past `duration - FRAME_SEEK_GUARD_SECONDS`. Only the final inclusive-end sample is
/// affected; every interior sample passes through unchanged. The tracker still records the
/// logical sample time — only the seek used to pull pixels is clamped.
#[cfg(target_os = "macos")]
pub(crate) fn frame_seek_timestamp(timestamp: f64, duration: f64) -> f64 {
    timestamp.min((duration - FRAME_SEEK_GUARD_SECONDS).max(0.0))
}

/// The native-MLX person segmenter used by the macOS person-track masking pass.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PersonSegmenter {
    /// SAM3 text-concept (PCS) — the box-prompt-free default (sc-4926).
    Sam3,
    /// Legacy SAM2 box-prompt video predictor (epic 3704) — kept for A/B parity + fallback.
    Sam2,
}

#[cfg(target_os = "macos")]
impl PersonSegmenter {
    /// The sidecar `tracker_meta.segmenter` id for this backend.
    fn meta_label(self) -> &'static str {
        match self {
            PersonSegmenter::Sam3 => "sam3",
            PersonSegmenter::Sam2 => "sam2.1_hiera_large",
        }
    }
}

/// Select the person segmenter: SAM3 PCS by default, the legacy SAM2 box-prompt path under
/// `SCENEWORKS_PERSON_SEGMENTER=sam2` (A/B parity validation + cutover fallback, sc-4926).
#[cfg(target_os = "macos")]
fn person_segmenter_kind() -> PersonSegmenter {
    match std::env::var("SCENEWORKS_PERSON_SEGMENTER").ok().as_deref() {
        Some("sam2") => PersonSegmenter::Sam2,
        _ => PersonSegmenter::Sam3,
    }
}

/// The real (model-backed) outcome of `assemble_real_person_track`: the resampled track frames
/// plus the metadata `run_person_track` folds into the sidecar.
struct RealPersonTrack {
    frames: Vec<Value>,
    average_confidence: f64,
    /// Sidecar `maskState` from the SAM2 segmentation pass (sc-3709): active / generated /
    /// degraded (segmenter unavailable → box-mask fallback) / missing (segmentation off).
    mask_state: &'static str,
    quality: Value,
    tracker_meta: Value,
}

/// Track the selected person through real source content: sample frames at the 2-FPS cadence, run
/// the native-MLX YOLO11 detector (sc-3633) on each, associate the boxes into track identities with
/// the SORT/ByteTrack tracker, and resample the chosen identity onto the sample cadence (sc-3634).
/// macOS-only (MLX detector); the Python Ultralytics path serves Windows/Linux.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    project_path: &std::path::Path,
    source_media_path: &std::path::Path,
    detection: &Value,
    track_id: &str,
    selected_timestamp: f64,
    duration: f64,
    confidence: f64,
    segment_enabled: bool,
) -> WorkerResult<RealPersonTrack> {
    use crate::person_track as pt;

    let selected_box = pt::NormalizedBox::from_json(detection.get("box").unwrap_or(&Value::Null));
    let download_context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Person tracking canceled while fetching detector weights.",
        fresh_download: false,
    };
    let weights = crate::person_jobs::ensure_detector_weights(settings, &download_context).await?;
    let conf = confidence as f32;
    let timestamps = pt::sample_timestamps(duration);

    let work_dir = std::env::temp_dir().join(format!("sw-person-track-{}", job.id));
    tokio::fs::create_dir_all(&work_dir).await?;

    let mut device = "mlx";
    // Keep each rendered frame (don't delete in-loop): the segmentation pass re-reads
    // the detected target frames by the same sample index. They share the cadence, so
    // assembly frame `i` ↔ `timestamps[i]` ↔ `frame_paths[i]`.
    let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(timestamps.len());
    let mut per_frame: Vec<(f64, Vec<(pt::NormalizedBox, f64)>)> =
        Vec::with_capacity(timestamps.len());
    for (index, &timestamp) in timestamps.iter().enumerate() {
        check_cancel(api, &job.id, "Person tracking canceled during sampling.").await?;
        let frame_path = work_dir.join(format!("frame_{index:04}.png"));
        let ffmpeg_context = FfmpegContext {
            api,
            settings,
            job_id: &job.id,
            cancel_message: "Person tracking canceled by user.",
        };
        render_frame_png(
            "ffmpeg",
            source_media_path,
            &frame_path,
            frame_seek_timestamp(timestamp, duration),
            1280,
            720,
            Some(ffmpeg_context),
        )
        .await?;
        let weights_for_frame = weights.clone();
        let frame_for_task = frame_path.clone();
        let result = tokio::task::spawn_blocking(move || {
            crate::person_jobs::detect_people_blocking(weights_for_frame, frame_for_task, conf)
        })
        .await
        .map_err(|error| task_join_error("person track detect task", error))??;
        device = result.device;
        let boxes = result
            .detections
            .iter()
            .map(|d| {
                (
                    pt::xyxy_to_normalized(
                        d.x1 as f64,
                        d.y1 as f64,
                        d.x2 as f64,
                        d.y2 as f64,
                        result.width,
                        result.height,
                    ),
                    d.score as f64,
                )
            })
            .collect::<Vec<_>>();
        per_frame.push((timestamp, boxes));
        frame_paths.push(frame_path);
    }

    let observations = pt::observe(per_frame);
    let assembly = pt::assemble_track(&observations, selected_box, selected_timestamp, &timestamps);
    if assembly.target_track_id.is_none() || assembly.detected_frames == 0 {
        let _ = tokio::fs::remove_dir_all(&work_dir).await;
        return Err(WorkerError::InvalidPayload(
            "Selected person was not found in the source video. Re-run detection or adjust the selection."
                .to_owned(),
        ));
    }

    // SAM2 segmentation pass (sc-3709): write a per-frame mask for each detected target
    // frame and fold the result into `maskState`. Any segmenter unavailability degrades
    // gracefully to box-derived masks (handled by the replacement loader), never failing
    // a track that already located the person.
    let mut frames_json = pt::frames_to_json(&assembly.frames);
    let mask_state = segment_assembly_frames(
        api,
        settings,
        http_client,
        job,
        project_path,
        track_id,
        &assembly.frames,
        &frame_paths,
        &mut frames_json,
        segment_enabled,
    )
    .await?;
    let _ = tokio::fs::remove_dir_all(&work_dir).await;

    Ok(RealPersonTrack {
        frames: frames_json,
        average_confidence: pt::average_confidence(&assembly.frames),
        mask_state,
        quality: assembly.quality,
        tracker_meta: json!({
            "backend": "mlx",
            "device": device,
            "model": "yolo11m",
            "tracker": "sort_bytetrack",
            "segmenter": if segment_enabled { person_segmenter_kind().meta_label() } else { "disabled" },
        }),
    })
}

/// Render dimensions of the sampled track frames (`render_frame_png` above), and therefore the
/// size of the masks the SAM2 video predictor emits.
#[cfg(target_os = "macos")]
const TRACK_FRAME_SIZE: (u32, u32) = (1280, 720);

/// Generate the selected person's track masks with the native-MLX SAM2 **video predictor**
/// (sc-3715): prompt once on the first detected frame and propagate temporally-consistent masks
/// across the `first..=last` detected span via the memory bank, so non-detected gap frames inside
/// the span still get a mask (the "survives weak-detection frames" win). Masks are written under
/// `person-tracks/{track_id}/masks/frame_{index:06}.png`, the frame's `mask` is set, and the
/// outcome rolls up into a `maskState` (Python `segment_track`): `missing` (disabled / no detected
/// frame), `degraded` (weights unavailable / propagation failed → box-mask fallback at replacement
/// time), `generated` (some detected frames masked), `active` (all detected frames masked). The
/// `generated`/`detected_total` rollup counts only detected frames, keeping the contract identical
/// to the per-frame path; gap-frame masks are additive coverage.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn segment_assembly_frames(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    project_path: &std::path::Path,
    track_id: &str,
    frames: &[crate::person_track::TrackFrame],
    frame_paths: &[PathBuf],
    frames_json: &mut [Value],
    segment_enabled: bool,
) -> WorkerResult<&'static str> {
    let detected_total = frames.iter().filter(|frame| frame.detected).count();
    if detected_total == 0 || !segment_enabled {
        return Ok("missing");
    }

    // Resolve/download the segmenter weights once; any failure degrades to box masks.
    let download_context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Person segmentation canceled while fetching segmenter weights.",
        fresh_download: false,
    };
    let masks_dir = project_path
        .join("person-tracks")
        .join(track_id)
        .join("masks");
    if tokio::fs::create_dir_all(&masks_dir).await.is_err() {
        return Ok("degraded");
    }

    // The track spans first..=last detected frame. Propagate across that contiguous clip; frames
    // outside it (before the person appears / after they leave) are left mask-less as before.
    let Some(first) = frames.iter().position(|f| f.detected) else {
        return Ok("missing");
    };
    let last = frames.iter().rposition(|f| f.detected).unwrap_or(first);

    // Clip frame paths + per-frame ByteTrack box anchors (None on the gap frames the predictor
    // fills from memory). The shared sample index ties frame `i` ↔ `frame_paths[i]` ↔ mask `i + 1`.
    if frame_paths.len() <= last {
        return Ok("degraded");
    }
    let mut clip_paths = Vec::with_capacity(last - first + 1);
    let mut anchors = Vec::with_capacity(last - first + 1);
    for (frame, path) in frames[first..=last].iter().zip(&frame_paths[first..=last]) {
        clip_paths.push(path.clone());
        anchors.push(frame.detected.then_some((
            frame.box_.x,
            frame.box_.y,
            frame.box_.width,
            frame.box_.height,
        )));
    }

    check_cancel(
        api,
        &job.id,
        "Person tracking canceled during segmentation.",
    )
    .await?;

    // Dispatch to the active segmenter (sc-4926): SAM3 text-concept PCS by default, the legacy
    // SAM2 box-prompt path under `SCENEWORKS_PERSON_SEGMENTER=sam2` (kept for A/B parity +
    // fallback during the cutover). Both return one binary mask per clip frame; either path's
    // failure degrades to box masks (handled by the replacement loader) — a track that already
    // located the person is never failed by the mask pass.
    let masks = match person_segmenter_kind() {
        PersonSegmenter::Sam3 => {
            let (model, tokenizer) = match crate::person_segment_sam3::ensure_segmenter_weights(
                settings,
                &download_context,
            )
            .await
            {
                Ok(pair) => pair,
                Err(WorkerError::Canceled(message)) => return Err(WorkerError::Canceled(message)),
                Err(_) => return Ok("degraded"),
            };
            match tokio::task::spawn_blocking(move || {
                crate::person_segment_sam3::segment_track_blocking(
                    model, tokenizer, clip_paths, anchors,
                )
            })
            .await
            {
                Ok(Ok(masks)) => masks,
                _ => return Ok("degraded"),
            }
        }
        PersonSegmenter::Sam2 => {
            let weights =
                match crate::person_segment::ensure_segmenter_weights(settings, &download_context)
                    .await
                {
                    Ok(path) => path,
                    Err(WorkerError::Canceled(message)) => {
                        return Err(WorkerError::Canceled(message))
                    }
                    Err(_) => return Ok("degraded"),
                };
            match tokio::task::spawn_blocking(move || {
                crate::person_segment::propagate_track_blocking(weights, clip_paths, anchors)
            })
            .await
            {
                Ok(Ok(masks)) => masks,
                _ => return Ok("degraded"),
            }
        }
    };

    // Write every clip frame's non-empty mask (detected + gap) and set its sidecar `mask`. All the
    // PNG encoding is blocking, so it runs in one `spawn_blocking`.
    let pending: Vec<(usize, String, PathBuf, Vec<u8>)> = masks
        .into_iter()
        .enumerate()
        .filter(|(_, pixels)| pixels.iter().any(|&p| p > 127))
        .map(|(clip_idx, pixels)| {
            let assembly_idx = first + clip_idx;
            let rel = format!(
                "person-tracks/{track_id}/masks/frame_{:06}.png",
                assembly_idx + 1
            );
            let out_path = project_path.join(&rel);
            (assembly_idx, rel, out_path, pixels)
        })
        .collect();
    let (width, height) = TRACK_FRAME_SIZE;
    let written =
        match tokio::task::spawn_blocking(move || write_track_mask_pngs(width, height, pending))
            .await
        {
            Ok(written) => written,
            Err(_) => return Ok("degraded"),
        };

    let mut generated = 0usize;
    for (assembly_idx, rel) in written {
        if let Some(entry) = frames_json.get_mut(assembly_idx) {
            entry["mask"] = Value::String(rel);
        }
        if frames
            .get(assembly_idx)
            .map(|f| f.detected)
            .unwrap_or(false)
        {
            generated += 1;
        }
    }

    Ok(crate::person_segment::rollup_mask_state(
        generated,
        detected_total,
    ))
}

/// Encode each `(assembly_idx, rel, out_path, pixels)` mask as an `L` (8-bit grayscale) PNG,
/// returning the `(assembly_idx, rel)` of the frames that were written. A single frame's failure is
/// non-fatal (matches Python's per-frame `except: continue`); it keeps `mask: null` and falls back
/// to a box mask at replacement time.
#[cfg(target_os = "macos")]
fn write_track_mask_pngs(
    width: u32,
    height: u32,
    pending: Vec<(usize, String, PathBuf, Vec<u8>)>,
) -> Vec<(usize, String)> {
    let mut written = Vec::with_capacity(pending.len());
    for (assembly_idx, rel, out_path, pixels) in pending {
        let Some(gray) = image::GrayImage::from_raw(width, height, pixels) else {
            continue;
        };
        if gray.save(&out_path).is_ok() {
            written.push((assembly_idx, rel));
        }
    }
    written
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
async fn assemble_real_person_track(
    _api: &ApiClient,
    _settings: &Settings,
    _http_client: &reqwest::Client,
    _job: &JobSnapshot,
    _project_path: &std::path::Path,
    _source_media_path: &std::path::Path,
    _detection: &Value,
    _track_id: &str,
    _selected_timestamp: f64,
    _duration: f64,
    _confidence: f64,
    _segment_enabled: bool,
) -> WorkerResult<RealPersonTrack> {
    Err(WorkerError::InvalidPayload(
        "Real person tracking runs on the Python worker on this platform.".to_owned(),
    ))
}

pub(crate) async fn run_person_track_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
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
    let result = run_person_track(api, settings, http_client, job).await?;
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
    http_client: &reqwest::Client,
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
    let confidence = job
        .payload
        .get("advanced")
        .and_then(|advanced| advanced.get("confidence"))
        .or_else(|| job.payload.get("confidence"))
        .map_or(0.25, |value| value_f64(value, 0.25))
        .clamp(0.01, 1.0);
    let is_preview = job
        .payload
        .get("preview")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // The selection frame's source timestamp (recorded in the representative frame's lineage).
    let selected_timestamp = job
        .payload
        .get("representativeFrameAssetId")
        .and_then(Value::as_str)
        .and_then(|asset_id| store.get_asset(project_id, asset_id).ok())
        .and_then(|asset| {
            asset
                .get("lineage")
                .and_then(|lineage| lineage.get("sourceTimestamp"))
                .map(|value| value_f64(value, 0.0))
        })
        .unwrap_or(0.0);

    // Segmentation is on by default (Python `advanced.segment`); a per-frame SAM2 mask is
    // written for each detected target frame. `segment: false` skips it (maskState=missing).
    let segment_enabled = job
        .payload
        .get("advanced")
        .and_then(|advanced| advanced.get("segment"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    // The track id is generated up front so the SAM2 segmentation pass can write masks under
    // `person-tracks/{track_id}/masks/` before the sidecar is assembled.
    let track_id = format!("track_{}", Uuid::new_v4().simple());

    // Preview jobs (CPU worker) keep the procedural placeholder. Real jobs run the native-MLX
    // YOLO11 detector (sc-3633) per sampled frame + the SORT/ByteTrack tracker (sc-3634), then
    // segment each detected frame with the native-MLX SAM2 segmenter (sc-3709) → maskState
    // active / generated / degraded / missing.
    let (
        frames,
        average_confidence,
        mask_state,
        person_active,
        tracker_model,
        tracker_adapter,
        quality_value,
        tracker_meta,
    ): (Vec<Value>, f64, &str, bool, String, &str, Value, Value) = if is_preview {
        let frames = track_frames_from_detection(&detection, duration);
        let avg = frames
            .iter()
            .map(|frame| {
                frame
                    .get("confidence")
                    .map_or(0.0, |value| value_f64(value, 0.0))
            })
            .sum::<f64>()
            / (frames.len().max(1) as f64);
        (
            frames,
            avg,
            "deferred",
            false,
            "procedural-person-tracker".to_owned(),
            "procedural_person_tracking",
            Value::Null,
            Value::Null,
        )
    } else {
        let source_media_rel = required_value_str(source_file, "path")?;
        let source_media_path = safe_project_path(&project_path, source_media_rel)?;
        if !source_media_path.exists() {
            return Err(WorkerError::InvalidPayload(format!(
                "Source media not found: {}",
                source_media_path.display()
            )));
        }
        let real = assemble_real_person_track(
            api,
            settings,
            http_client,
            job,
            &project_path,
            &source_media_path,
            &detection,
            &track_id,
            selected_timestamp,
            duration,
            confidence,
            segment_enabled,
        )
        .await?;
        (
            real.frames,
            real.average_confidence,
            real.mask_state,
            true,
            "yolo11m".to_owned(),
            "yolo11_bytetrack",
            real.quality,
            real.tracker_meta,
        )
    };

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

    let mut status = serde_json::Map::new();
    status.insert(
        "sampleRateFps".to_owned(),
        json!(PERSON_TRACK_SAMPLE_RATE_FPS),
    );
    status.insert("maskState".to_owned(), json!(mask_state));
    status.insert(
        "averageConfidence".to_owned(),
        json!(round_to(average_confidence, 4)),
    );
    status.insert(
        "correctionState".to_owned(),
        json!("ready_for_box_corrections"),
    );
    status.insert("personTrackingActive".to_owned(), json!(person_active));
    if person_active {
        status.insert("quality".to_owned(), quality_value);
        status.insert("tracker".to_owned(), tracker_meta.clone());
    }

    let mut normalized = serde_json::Map::new();
    normalized.insert(
        "sampleRateFps".to_owned(),
        json!(PERSON_TRACK_SAMPLE_RATE_FPS),
    );
    normalized.insert("personDetectionActive".to_owned(), json!(person_active));
    normalized.insert("personTrackingActive".to_owned(), json!(person_active));
    if person_active {
        normalized.insert("maskState".to_owned(), json!(mask_state));
        normalized.insert("tracker".to_owned(), tracker_meta);
    }

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
        "status": Value::Object(status),
        "recipe": {
            "mode": "person_track",
            "model": tracker_model,
            "adapter": tracker_adapter,
            "prompt": format!("Track {track_name}"),
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": Value::Object(normalized),
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
    _tmp_path: &Path,
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg(
        mux_with_crossfades_args(ffmpeg, segments, output_path)?,
        context,
    )
    .await
}

pub(crate) fn mux_with_crossfades_args(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    output_path: &Path,
) -> WorkerResult<Vec<String>> {
    if segments.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no rendered segments to mux.".to_owned(),
        ));
    }
    let (filter, output_label) = crossfade_filter_complex(segments);
    let mut args = vec![ffmpeg.to_owned(), "-y".to_owned()];
    for segment in segments {
        args.push("-i".to_owned());
        args.push(segment.path.display().to_string());
    }
    args.extend([
        "-filter_complex".to_owned(),
        filter,
        "-map".to_owned(),
        format!("[{output_label}]"),
        output_path.display().to_string(),
    ]);
    Ok(args)
}

fn crossfade_filter_complex(segments: &[TimelineSegment]) -> (String, String) {
    let mut filters = Vec::with_capacity(segments.len() * 2);
    for (index, _) in segments.iter().enumerate() {
        filters.push(format!(
            "[{index}:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[v{index}]"
        ));
    }

    let mut current_label = "v0".to_owned();
    let mut current_duration = segments
        .first()
        .map(|segment| segment.duration)
        .unwrap_or(0.0);
    for (index, segment) in segments.iter().enumerate().skip(1) {
        let next_label = format!("mix{index}");
        if segment.transition.as_deref() == Some("crossfade") {
            let duration = crossfade_duration(segment.transition_duration);
            let offset = (current_duration - duration).max(0.0);
            filters.push(format!(
                "[{current_label}][v{index}]xfade=transition=fade:duration={duration:.3}:offset={offset:.3},format=yuv420p[{next_label}]"
            ));
            current_duration += segment.duration - duration;
        } else {
            filters.push(format!(
                "[{current_label}][v{index}]concat=n=2:v=1:a=0,format=yuv420p[{next_label}]"
            ));
            current_duration += segment.duration;
        }
        current_label = next_label;
    }
    (filters.join(";"), current_label)
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
            WorkerError::Engine(format!(
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

    let stderr = stderr_task
        .await
        .map_err(|error| task_join_error("ffmpeg stderr reader task", error))?;
    if status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&stderr);
    let bounded = bounded_tail(&stderr, 10, 2000);
    if bounded.trim().is_empty() {
        Err(WorkerError::Engine(
            "FFmpeg command failed without stderr output.".to_owned(),
        ))
    } else {
        Err(WorkerError::Engine(bounded))
    }
}

#[cfg(all(test, target_os = "macos"))]
mod frame_seek_tests {
    use super::*;
    use crate::person_track::sample_timestamps;

    #[test]
    fn frame_seek_clamps_only_the_final_inclusive_end_sample() {
        // sample_timestamps is inclusive of both ends → the last sample == duration, which
        // is not a decodable frame. The seek clamp must pull exactly that sample inside the
        // clip and leave every interior sample (and t=0) untouched.
        let duration = 8.0;
        let stamps = sample_timestamps(duration);
        let last = *stamps.last().expect("non-empty");
        assert_eq!(last, duration, "final sample sits on the inclusive end");

        for &ts in &stamps {
            let seek = frame_seek_timestamp(ts, duration);
            assert!(
                seek <= duration - FRAME_SEEK_GUARD_SECONDS + 1e-9,
                "seek {seek} must stay a frame inside the {duration}s clip"
            );
            if ts < duration - FRAME_SEEK_GUARD_SECONDS {
                assert_eq!(seek, ts, "interior samples pass through unchanged");
            }
        }
        // The final sample is the one that gets clamped.
        assert_eq!(
            frame_seek_timestamp(last, duration),
            duration - FRAME_SEEK_GUARD_SECONDS
        );
    }

    #[test]
    fn frame_seek_never_goes_negative_on_tiny_clips() {
        // A clip shorter than the guard clamps to 0 rather than a negative seek.
        assert_eq!(frame_seek_timestamp(0.1, 0.1), 0.0);
        assert_eq!(frame_seek_timestamp(0.0, 0.0), 0.0);
    }
}

/// Real-Mac end-to-end validation for the native-MLX Replace-Person pipeline
/// (epic 3704 / sc-3709): a short person video → ffmpeg frame sampling → MLX YOLO11
/// detect (sc-3633) → SORT/ByteTrack assembly (sc-3634) → MLX SAM2 segment (sc-3706/3709)
/// → per-frame masks on disk → `maskState`. This is the model-path twin of
/// [`assemble_real_person_track`] / [`segment_assembly_frames`], driven through the same
/// blocking seams but without the ApiClient/ProjectStore plumbing (that plumbing carries no
/// MLX work). It cannot run in CI — there is no GPU or weights on the runner — so it is
/// `#[ignore]` and skips cleanly unless a clip is staged. macOS-only, like the pipeline.
///
/// Run on Apple Silicon with a short clip that contains a clearly-visible person:
///
/// ```text
/// SCENEWORKS_PERSON_E2E_VIDEO=/path/to/person_clip.mp4 \
/// SCENEWORKS_PERSON_E2E_DURATION=4 \
///   cargo test -p sceneworks-worker --lib \
///   person_track_e2e -- --ignored --nocapture
/// ```
///
/// The YOLO11 and SAM2 weights download on first use from the public `SceneWorks/*` HF
/// repos (or pin them with `SCENEWORKS_PERSON_DETECTOR_WEIGHTS` / `SCENEWORKS_SAM2_WEIGHTS`).
#[cfg(all(test, target_os = "macos"))]
mod person_track_e2e_tests {
    use super::*;
    use std::path::PathBuf;

    /// The staged source clip (`SCENEWORKS_PERSON_E2E_VIDEO`), or `None` to skip.
    fn staged_video() -> Option<PathBuf> {
        let path = PathBuf::from(std::env::var("SCENEWORKS_PERSON_E2E_VIDEO").ok()?);
        path.exists().then_some(path)
    }

    /// Clip duration in seconds (`SCENEWORKS_PERSON_E2E_DURATION`, default 4.0), used to pick
    /// the 2-FPS sample cadence — the same `sample_timestamps` the real job uses.
    fn staged_duration() -> f64 {
        std::env::var("SCENEWORKS_PERSON_E2E_DURATION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(4.0)
            .clamp(1.0, 3600.0)
    }

    #[test]
    #[ignore = "real Mac E2E: set SCENEWORKS_PERSON_E2E_VIDEO to a person clip; downloads YOLO11 + SAM2 weights; Apple Silicon only"]
    fn person_track_e2e_detect_track_segment_writes_masks_and_active_state() {
        let Some(video) = staged_video() else {
            eprintln!(
                "skipping: set SCENEWORKS_PERSON_E2E_VIDEO to a short clip containing a person"
            );
            return;
        };

        // Isolated scratch: a throwaway data dir (weights cache) + a fake project root the
        // segmentation pass writes masks into, exactly as `run_person_track` would.
        let scratch = std::env::temp_dir().join("sw-person-track-e2e");
        let _ = std::fs::remove_dir_all(&scratch);
        let data_dir = scratch.join("data");
        let project_path = scratch.join("project");
        let frames_dir = scratch.join("frames");
        std::fs::create_dir_all(&project_path).expect("project dir");
        std::fs::create_dir_all(&frames_dir).expect("frames dir");
        // `ensure_*_weights` resolve their cache under `settings.data_dir`.
        std::env::set_var("SCENEWORKS_DATA_DIR", &data_dir);
        let settings = crate::Settings::from_env();

        let timestamps = crate::person_track::sample_timestamps(staged_duration());
        assert!(!timestamps.is_empty(), "sample cadence produced no frames");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let http = reqwest::Client::new();
            let api = ApiClient::new(&settings);
            let job_id = "person-track-e2e";
            let download_context = DownloadContext {
                api: &api,
                client: &http,
                settings: &settings,
                job_id,
                cancel_message: "person track e2e canceled while fetching weights",
                fresh_download: false,
            };

            // 1. Provision the real detector weights (download-on-first-use), then sample +
            //    detect each frame exactly as `assemble_real_person_track` does.
            let det_weights =
                crate::person_jobs::ensure_detector_weights(&settings, &download_context)
                    .await
                    .expect("yolo11 weights provisioned");
            let mut per_frame: Vec<(f64, Vec<(crate::person_track::NormalizedBox, f64)>)> =
                Vec::with_capacity(timestamps.len());
            let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(timestamps.len());
            for (index, &timestamp) in timestamps.iter().enumerate() {
                let frame_path = frames_dir.join(format!("frame_{index:04}.png"));
                render_frame_png(
                    "ffmpeg",
                    &video,
                    &frame_path,
                    frame_seek_timestamp(timestamp, staged_duration()),
                    1280,
                    720,
                    None,
                )
                .await
                .expect("ffmpeg renders the sample frame");
                let weights_for_frame = det_weights.clone();
                let frame_for_task = frame_path.clone();
                let result = tokio::task::spawn_blocking(move || {
                    crate::person_jobs::detect_people_blocking(
                        weights_for_frame,
                        frame_for_task,
                        0.25,
                    )
                })
                .await
                .expect("detect task joins")
                .expect("yolo11 detection runs");
                let boxes = result
                    .detections
                    .iter()
                    .map(|d| {
                        (
                            crate::person_track::xyxy_to_normalized(
                                d.x1 as f64,
                                d.y1 as f64,
                                d.x2 as f64,
                                d.y2 as f64,
                                result.width,
                                result.height,
                            ),
                            d.score as f64,
                        )
                    })
                    .collect::<Vec<_>>();
                per_frame.push((timestamp, boxes));
                frame_paths.push(frame_path);
            }

            // Per-frame detection summary, so a failed lock is legible (e.g. a person who
            // never clears the tracker's high-confidence threshold).
            for (timestamp, boxes) in &per_frame {
                let max_conf = boxes.iter().map(|b| b.1).fold(0.0_f64, f64::max);
                eprintln!(
                    "  t={timestamp:>6.3}s  detections={}  maxConf={max_conf:.3}",
                    boxes.len()
                );
            }
            let total_detections: usize = per_frame.iter().map(|(_, boxes)| boxes.len()).sum();
            assert!(
                total_detections > 0,
                "YOLO11 found no people in any sampled frame — is the clip a person video?"
            );

            // 2. Select the single highest-confidence detection across all frames — the
            //    clear, lockable person a user would click (robust to early low-confidence
            //    false positives on textured scenes like waves). The tracker only confirms a
            //    new identity from a high-confidence box, so the selection must be one.
            let (selected_timestamp, selected_box, selected_conf) = per_frame
                .iter()
                .flat_map(|(timestamp, boxes)| boxes.iter().map(move |b| (*timestamp, b.0, b.1)))
                .max_by(|a, b| {
                    a.2.partial_cmp(&b.2)
                        .expect("detection confidence is finite")
                })
                .expect("at least one detection exists");
            eprintln!(
                "selection: t={selected_timestamp:.3}s conf={selected_conf:.3} \
                 box=({:.3},{:.3},{:.3},{:.3})",
                selected_box.x, selected_box.y, selected_box.width, selected_box.height
            );
            assert!(
                selected_conf >= 0.5,
                "best detection conf {selected_conf:.3} is below the tracker's high-confidence \
                 floor (0.5) — the clip never yields a confidently-trackable person"
            );

            let observations = crate::person_track::observe(per_frame);
            let assembly = crate::person_track::assemble_track(
                &observations,
                selected_box,
                selected_timestamp,
                &timestamps,
            );
            assert!(
                assembly.target_track_id.is_some(),
                "tracker failed to lock onto the selected person"
            );
            let detected_total = assembly
                .frames
                .iter()
                .filter(|frame| frame.detected)
                .count();
            assert!(detected_total > 0, "no detected target frames to segment");

            // 3. Provision the real SAM2 weights and propagate the person's mask across the
            //    detected span with the video predictor (sc-3715), writing masks under
            //    `person-tracks/{track_id}/masks/` (the production layout).
            let seg_weights =
                crate::person_segment::ensure_segmenter_weights(&settings, &download_context)
                    .await
                    .expect("sam2 weights provisioned");

            let first = assembly
                .frames
                .iter()
                .position(|f| f.detected)
                .expect("a detected frame exists");
            let last = assembly
                .frames
                .iter()
                .rposition(|f| f.detected)
                .unwrap_or(first);
            let mut clip_paths = Vec::new();
            let mut anchors = Vec::new();
            for (frame, path) in assembly.frames[first..=last]
                .iter()
                .zip(&frame_paths[first..=last])
            {
                clip_paths.push(path.clone());
                anchors.push(frame.detected.then(|| {
                    let b = &frame.box_;
                    (b.x, b.y, b.width, b.height)
                }));
            }
            let gap_frames = anchors.iter().filter(|a| a.is_none()).count();

            let masks = tokio::task::spawn_blocking(move || {
                crate::person_segment::propagate_track_blocking(seg_weights, clip_paths, anchors)
            })
            .await
            .expect("propagate task joins")
            .expect("sam2 propagation runs");
            assert_eq!(masks.len(), last - first + 1, "one mask per clip frame");

            // Every detected frame's propagated mask is non-empty (SAM2 actually tracked the
            // person, not a blank map); count detected frames masked for the rollup.
            let mut generated = 0usize;
            for (clip_idx, pixels) in masks.iter().enumerate() {
                let assembly_idx = first + clip_idx;
                let foreground = pixels.iter().filter(|&&p| p > 127).count();
                assert_eq!(
                    pixels.len(),
                    (1280 * 720) as usize,
                    "mask must match the rendered frame size"
                );
                if assembly.frames[assembly_idx].detected {
                    assert!(
                        foreground > 0,
                        "frame {assembly_idx} mask has no foreground — propagation lost the person"
                    );
                    generated += 1;
                }
            }

            // 4. The maskState rollup must report `active` — every detected frame masked — which
            //    is the success signal `run_person_track` writes to the sidecar.
            let mask_state = crate::person_segment::rollup_mask_state(generated, detected_total);
            eprintln!(
                "person-track E2E: sampled={} detected={} span={}..={} gapFrames={} segmented={} maskState={}",
                timestamps.len(),
                detected_total,
                first,
                last,
                gap_frames,
                generated,
                mask_state
            );
            assert_eq!(
                generated, detected_total,
                "every detected frame should be masked by propagation"
            );
            assert_eq!(
                mask_state, "active",
                "all detected frames masked → maskState=active"
            );
        });

        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// SAM3 cutover E2E (sc-4926): the same detect → track → assemble flow as the SAM2 test
    /// above, then segment the selected person with the **SAM3 text-concept (PCS)** path
    /// (`person_segment_sam3::segment_track_blocking`, prompt `"person"`, **no box prompt**) and
    /// the legacy **SAM2 box-prompt** path on the identical clip + anchors, and compare. Proves
    /// the box-free pipeline produces an `active` mask track end-to-end and reports SAM3-vs-SAM2
    /// mask agreement (the "parity/quality vs the SAM2 baseline" acceptance).
    ///
    /// ```text
    /// SCENEWORKS_SAM3_WEIGHTS=<facebook/sam3 snapshot dir> \  # or omit to download SceneWorks/sam3-mlx
    /// SCENEWORKS_PERSON_E2E_VIDEO=/path/to/person_clip.mp4 \
    /// SCENEWORKS_PERSON_E2E_DURATION=4 \
    ///   cargo test -p sceneworks-worker --lib person_track_e2e_sam3 -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "real Mac E2E: SAM3 + SAM2 weights; set SCENEWORKS_PERSON_E2E_VIDEO; Apple Silicon only"]
    fn person_track_e2e_sam3_segments_and_matches_sam2_baseline() {
        let Some(video) = staged_video() else {
            eprintln!(
                "skipping: set SCENEWORKS_PERSON_E2E_VIDEO to a short clip containing a person"
            );
            return;
        };
        let scratch = std::env::temp_dir().join("sw-person-track-e2e-sam3");
        let _ = std::fs::remove_dir_all(&scratch);
        let data_dir = scratch.join("data");
        let frames_dir = scratch.join("frames");
        std::fs::create_dir_all(&frames_dir).expect("frames dir");
        std::env::set_var("SCENEWORKS_DATA_DIR", &data_dir);
        let settings = crate::Settings::from_env();
        let timestamps = crate::person_track::sample_timestamps(staged_duration());
        assert!(!timestamps.is_empty(), "sample cadence produced no frames");

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let http = reqwest::Client::new();
            let api = ApiClient::new(&settings);
            let download_context = DownloadContext {
                api: &api,
                client: &http,
                settings: &settings,
                job_id: "person-track-e2e-sam3",
                cancel_message: "person track e2e (sam3) canceled while fetching weights",
                fresh_download: false,
            };

            // detect → track → assemble (identical to the SAM2 E2E above).
            let det_weights =
                crate::person_jobs::ensure_detector_weights(&settings, &download_context)
                    .await
                    .expect("yolo11 weights provisioned");
            let mut per_frame: Vec<(f64, Vec<(crate::person_track::NormalizedBox, f64)>)> =
                Vec::with_capacity(timestamps.len());
            let mut frame_paths: Vec<PathBuf> = Vec::with_capacity(timestamps.len());
            for (index, &timestamp) in timestamps.iter().enumerate() {
                let frame_path = frames_dir.join(format!("frame_{index:04}.png"));
                render_frame_png(
                    "ffmpeg",
                    &video,
                    &frame_path,
                    frame_seek_timestamp(timestamp, staged_duration()),
                    1280,
                    720,
                    None,
                )
                .await
                .expect("ffmpeg renders the sample frame");
                let w = det_weights.clone();
                let f = frame_path.clone();
                let result =
                    tokio::task::spawn_blocking(move || {
                        crate::person_jobs::detect_people_blocking(w, f, 0.25)
                    })
                    .await
                    .expect("detect task joins")
                    .expect("yolo11 detection runs");
                let boxes = result
                    .detections
                    .iter()
                    .map(|d| {
                        (
                            crate::person_track::xyxy_to_normalized(
                                d.x1 as f64,
                                d.y1 as f64,
                                d.x2 as f64,
                                d.y2 as f64,
                                result.width,
                                result.height,
                            ),
                            d.score as f64,
                        )
                    })
                    .collect::<Vec<_>>();
                per_frame.push((timestamp, boxes));
                frame_paths.push(frame_path);
            }
            let (selected_timestamp, selected_box, _) = per_frame
                .iter()
                .flat_map(|(t, boxes)| boxes.iter().map(move |b| (*t, b.0, b.1)))
                .max_by(|a, b| a.2.partial_cmp(&b.2).expect("finite conf"))
                .expect("at least one detection");
            let observations = crate::person_track::observe(per_frame);
            let assembly = crate::person_track::assemble_track(
                &observations,
                selected_box,
                selected_timestamp,
                &timestamps,
            );
            assert!(
                assembly.target_track_id.is_some(),
                "tracker failed to lock onto the selected person"
            );
            let detected_total = assembly.frames.iter().filter(|f| f.detected).count();
            assert!(detected_total > 0, "no detected target frames to segment");
            let first = assembly
                .frames
                .iter()
                .position(|f| f.detected)
                .expect("a detected frame exists");
            let last = assembly
                .frames
                .iter()
                .rposition(|f| f.detected)
                .unwrap_or(first);
            let mut clip_paths = Vec::new();
            let mut anchors = Vec::new();
            for (frame, path) in assembly.frames[first..=last]
                .iter()
                .zip(&frame_paths[first..=last])
            {
                clip_paths.push(path.clone());
                anchors.push(frame.detected.then(|| {
                    let b = &frame.box_;
                    (b.x, b.y, b.width, b.height)
                }));
            }

            // SAM3 text-concept segmentation (no box prompt) — the path under test.
            let (sam3_model, sam3_tok) =
                crate::person_segment_sam3::ensure_segmenter_weights(&settings, &download_context)
                    .await
                    .expect("sam3 weights provisioned");
            let (cp3, an3) = (clip_paths.clone(), anchors.clone());
            let sam3 = tokio::task::spawn_blocking(move || {
                crate::person_segment_sam3::segment_track_blocking(sam3_model, sam3_tok, cp3, an3)
            })
            .await
            .expect("sam3 task joins")
            .expect("sam3 segmentation runs");
            assert_eq!(sam3.len(), last - first + 1, "one SAM3 mask per clip frame");
            let mut sam3_generated = 0usize;
            for (i, px) in sam3.iter().enumerate() {
                let idx = first + i;
                assert_eq!(px.len(), 1280 * 720, "SAM3 mask must match the frame size");
                if assembly.frames[idx].detected {
                    assert!(
                        px.iter().any(|&p| p > 127),
                        "SAM3 frame {idx} has no foreground — concept segmentation lost the person"
                    );
                    sam3_generated += 1;
                }
            }
            let sam3_state = crate::person_segment::rollup_mask_state(sam3_generated, detected_total);
            // SAM3 masking every detected frame is the hard acceptance gate for the cutover.
            assert_eq!(
                sam3_state, "active",
                "every detected frame masked by SAM3 → maskState=active"
            );

            // SAM2 box-prompt baseline on the identical clip + anchors. Provisioning the SAM2
            // video-predictor weights needs the download API (or a `SCENEWORKS_SAM2_WEIGHTS` pin);
            // when it is unavailable the comparison is skipped — the SAM3 gate above still holds.
            let sam2_weights =
                match crate::person_segment::ensure_segmenter_weights(&settings, &download_context)
                    .await
                {
                    Ok(path) => path,
                    Err(e) => {
                        eprintln!(
                            "SAM3 E2E: sam3_state={sam3_state} sam3_generated={sam3_generated}; \
                             SAM2 baseline skipped (weights unavailable: {e}). Pin \
                             SCENEWORKS_SAM2_WEIGHTS to run the parity comparison."
                        );
                        return;
                    }
                };
            let (cp2, an2) = (clip_paths.clone(), anchors.clone());
            let sam2 = tokio::task::spawn_blocking(move || {
                crate::person_segment::propagate_track_blocking(sam2_weights, cp2, an2)
            })
            .await
            .expect("sam2 task joins")
            .expect("sam2 propagation runs");

            // Per-detected-frame mask IoU between the two segmenters.
            let mut ious = Vec::new();
            for i in 0..sam3.len() {
                if !assembly.frames[first + i].detected {
                    continue;
                }
                let (a, b) = (&sam3[i], &sam2[i]);
                if a.is_empty() || b.is_empty() {
                    continue;
                }
                let (mut inter, mut union) = (0u64, 0u64);
                for k in 0..a.len() {
                    let (pa, pb) = (a[k] > 127, b[k] > 127);
                    if pa || pb {
                        union += 1;
                        if pa && pb {
                            inter += 1;
                        }
                    }
                }
                if union > 0 {
                    ious.push(inter as f64 / union as f64);
                }
            }
            let mean_iou = if ious.is_empty() {
                0.0
            } else {
                ious.iter().sum::<f64>() / ious.len() as f64
            };
            let sam3_cov = sam3.iter().map(|m| m.iter().filter(|&&p| p > 127).count()).sum::<usize>()
                as f64
                / (sam3.len() * 1280 * 720) as f64;
            eprintln!(
                "SAM3 E2E: detected={detected_total} sam3_generated={sam3_generated} \
                 sam3_state={sam3_state} meanIoU(sam3,sam2)={mean_iou:.3} sam3_coverage={sam3_cov:.3}"
            );
            assert!(
                mean_iou > 0.5,
                "SAM3 vs SAM2 mean IoU {mean_iou:.3} below the parity floor (0.5)"
            );
        });

        let _ = std::fs::remove_dir_all(&scratch);
    }
}

#[cfg(test)]
mod crossfade_mux_tests {
    use super::*;

    fn segment(
        path: &str,
        duration: f64,
        transition: Option<&str>,
        transition_duration: f64,
    ) -> TimelineSegment {
        TimelineSegment {
            path: PathBuf::from(path),
            duration,
            transition: transition.map(str::to_owned),
            transition_duration,
        }
    }

    #[test]
    fn crossfade_mux_builds_one_filter_graph_for_all_segments() {
        let segments = vec![
            segment("a.mp4", 2.0, None, 0.5),
            segment("b.mp4", 3.0, Some("crossfade"), 0.5),
            segment("c.mp4", 4.0, None, 0.5),
        ];

        let args = mux_with_crossfades_args("ffmpeg", &segments, Path::new("out.mp4")).unwrap();
        let filter_index = args
            .iter()
            .position(|arg| arg == "-filter_complex")
            .expect("filter_complex present")
            + 1;
        let filter = &args[filter_index];

        assert_eq!(args.iter().filter(|arg| arg.as_str() == "-i").count(), 3);
        assert_eq!(
            args.iter()
                .filter(|arg| arg.as_str() == "-filter_complex")
                .count(),
            1
        );
        assert!(filter.contains("[v0][v1]xfade=transition=fade:duration=0.500:offset=1.500"));
        assert!(filter.contains("[mix1][v2]concat=n=2:v=1:a=0"));
        assert!(args.windows(2).any(|pair| pair == ["-map", "[mix2]"]));
        assert!(!args.iter().any(|arg| arg.contains("xfade_")));
        assert!(!args.iter().any(|arg| arg.contains("concat_")));
    }

    #[test]
    fn crossfade_mux_rejects_empty_segments() {
        let error = mux_with_crossfades_args("ffmpeg", &[], Path::new("out.mp4"))
            .expect_err("empty timeline rejects");

        assert!(matches!(error, WorkerError::InvalidPayload(_)));
        assert!(error
            .to_string()
            .contains("Timeline has no rendered segments"));
    }
}
