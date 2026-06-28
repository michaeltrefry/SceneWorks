//! Smart-select image segmentation on the macOS Rust worker (epic 6087, sc-6105).
//!
//! The backend half of the Image Editor's "smart-select" tool (sc-3751): an `image_segment` job
//! takes a source image asset + a box prompt and returns a binary inpaint mask asset (white-on-black
//! PNG at the source dims) the editor loads into the existing sc-2436 mask layer / sc-2476 inpaint
//! seam (`maskAssetId`). It mirrors the standalone `image_upscale` / `image_detail` job shape
//! (`upscale_jobs`): resolve the `sourceAssetId` against its project, decode, run the engine under
//! `spawn_blocking`, write one child asset with lineage back to the source.
//!
//! Engine (native-MLX **SAM3**, prompt-driven, one checkpoint): a **box** prompt runs SAM3's
//! concept *detector* (`person_segment_sam3::segment_box_blocking` →
//! `Sam3ImageSegmenter::segment_with_boxes`, sc-4923/sc-6105); **fg/bg click points** run SAM3's
//! *tracker* single-frame PVS (`person_segment_sam3::segment_points_blocking` →
//! `Sam3Tracker::segment_points`, sc-6346). Both load from the same `facebook/sam3` checkpoint —
//! SAM3 does interactive point refinement via its SAM2-lineage tracker, so no second model/download.
//! Points take precedence when both are present; exactly one path runs per job and both return the
//! same `maskAssetId` shape. macOS-only, like the SAM3 dependency; the capability is advertised only
//! by the MLX worker (`gpu.rs mlx_gpu`), so a segment job never routes off-Mac. There is no
//! torch/candle SAM3 image path yet (a Windows/Linux backport is tracked separately).

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use gen_core::CancelFlag;

use crate::downloads::DownloadContext;
use crate::person_segment_sam3::{
    ensure_segmenter_weights, segment_box_blocking, segment_points_blocking,
};
use crate::{
    fresh_asset_id, heartbeat, now_rfc3339, progress_payload, run_blocking_with_heartbeat,
    update_job, ApiClient, Settings, WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::ProjectStore;

/// SAM3 post-process defaults for a single box prompt (the engine parity defaults, geometry_parity
/// test): keep an instance when `σ(logit)·σ(presence) > THRESHOLD`, binarize its mask at `σ > MASK`.
const SEGMENT_THRESHOLD: f32 = 0.5;
const SEGMENT_MASK_THRESHOLD: f32 = 0.5;
const CANCEL_MESSAGE: &str = "Smart-select canceled by user.";

/// Resolve a `sourceAssetId` to its on-disk media path + the asset's `displayName` via the project
/// sidecar (mirrors `upscale_jobs::resolve_source` / `image_adapters.find_asset_media_path`).
fn resolve_source(
    store: &ProjectStore,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> Option<(PathBuf, Option<String>)> {
    let asset = store.get_asset(project_id, asset_id).ok()?;
    let rel = asset.get("file")?.get("path")?.as_str()?;
    let mut path = project_path.to_path_buf();
    for component in Path::new(rel).components() {
        if let std::path::Component::Normal(value) = component {
            path.push(value);
        } else {
            return None;
        }
    }
    if !path.exists() {
        return None;
    }
    let display = asset
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Some((path, display))
}

/// Parse `payload.box` (the box prompt in source-image pixel coords). Accepts either the canonical
/// 4-array `[x1, y1, x2, y2]` or an `{x, y, width, height}` object (the editor's rect shape), so the
/// frontend can send whichever is convenient.
fn parse_box(payload: &JsonObject) -> WorkerResult<[f32; 4]> {
    let value = payload.get("box").ok_or_else(|| {
        WorkerError::InvalidPayload("Smart-select requires a 'box' prompt.".to_owned())
    })?;
    if let Some(arr) = value.as_array() {
        if arr.len() == 4 {
            let mut out = [0f32; 4];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = arr[i].as_f64().ok_or_else(|| {
                    WorkerError::InvalidPayload("Smart-select box must be four numbers.".to_owned())
                })? as f32;
            }
            return Ok(out);
        }
    }
    if let Some(obj) = value.as_object() {
        let f = |key: &str| obj.get(key).and_then(Value::as_f64).map(|v| v as f32);
        if let (Some(x), Some(y), Some(w), Some(h)) = (f("x"), f("y"), f("width"), f("height")) {
            return Ok([x, y, x + w, y + h]);
        }
    }
    Err(WorkerError::InvalidPayload(
        "Smart-select box must be [x1,y1,x2,y2] or {x,y,width,height}.".to_owned(),
    ))
}

/// Parse `payload.points` (fg/bg click prompts in source-image pixel coords) for the SAM3 tracker
/// point path (sc-6346). Each entry is `{x, y, label}` — `label` `1` = foreground / `0` = background
/// (accepts an integer `label`, or a boolean `fg`; defaults to foreground). Returns `None` when
/// `points` is absent or an empty array (→ the caller falls back to the box path), `Err` on a
/// malformed entry. Points and box are alternative prompt modes; when both are present, points win.
fn parse_points(payload: &JsonObject) -> WorkerResult<Option<Vec<(f32, f32, i32)>>> {
    let Some(value) = payload.get("points") else {
        return Ok(None);
    };
    let arr = value.as_array().ok_or_else(|| {
        WorkerError::InvalidPayload("Smart-select 'points' must be an array.".to_owned())
    })?;
    if arr.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let obj = entry.as_object().ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Smart-select point must be an object {x, y, label}.".to_owned(),
            )
        })?;
        let coord = |key: &str| obj.get(key).and_then(Value::as_f64).map(|v| v as f32);
        let (Some(x), Some(y)) = (coord("x"), coord("y")) else {
            return Err(WorkerError::InvalidPayload(
                "Smart-select point needs numeric 'x' and 'y'.".to_owned(),
            ));
        };
        // label: 1 = foreground (default), 0 = background. Accept an int `label` or a bool `fg`.
        let label = obj
            .get("label")
            .and_then(Value::as_i64)
            .map(|v| i32::from(v != 0))
            .or_else(|| obj.get("fg").and_then(Value::as_bool).map(i32::from))
            .unwrap_or(1);
        out.push((x, y, label));
    }
    Ok(Some(out))
}

pub(crate) async fn run_image_segment_job(
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
            0.12,
            "Loading source image.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let payload = &job.payload;
    let source_asset_id = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Smart-select requires a source image asset.".to_owned())
        })?
        .to_owned();
    // Smart-select prompt: fg/bg points (SAM3 tracker — the interactive-click path) take precedence;
    // else a box (SAM3 concept detector, sc-6105). Both use the same SAM3 checkpoint (sc-6346).
    // Exactly one prompt mode runs per job.
    let points = parse_points(payload)?;
    let box_xyxy = match &points {
        Some(_) => None,
        None => Some(parse_box(payload)?),
    };
    // The optional text concept paired with the box (SAM3 PVS is text⊕box). Empty = rely on the
    // geometric prompt — the smart-select default, since the user draws a box around any object.
    let concept = payload
        .get("concept")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let threshold = payload
        .get("threshold")
        .and_then(Value::as_f64)
        .map(|v| v.clamp(0.0, 1.0) as f32)
        .unwrap_or(SEGMENT_THRESHOLD);
    let mask_threshold = payload
        .get("maskThreshold")
        .and_then(Value::as_f64)
        .map(|v| v.clamp(0.0, 1.0) as f32)
        .unwrap_or(SEGMENT_MASK_THRESHOLD);

    // The prompt recorded on the mask asset's `rawAdapterSettings` (both modes are engine "sam3"):
    // the click points for the point path, the box + thresholds for the box path. Built now, before
    // the prompt is moved into the segmentation task below.
    let segment_settings = if let Some(points) = &points {
        json!({
            "engine": "sam3",
            "points": points
                .iter()
                .map(|&(x, y, label)| json!({ "x": x, "y": y, "label": label }))
                .collect::<Vec<_>>(),
        })
    } else {
        json!({
            "engine": "sam3",
            "box": box_xyxy.expect("box prompt present when there are no points"),
            "threshold": threshold,
            "maskThreshold": mask_threshold,
        })
    };

    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Smart-select requires a projectId.".to_owned())
        })?
        .to_owned();
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .get_project(&project_id)
        .map_err(|e| WorkerError::InvalidPayload(format!("project not found: {e}")))?;
    let project_path = PathBuf::from(project.path);
    let (source_path, source_display) =
        resolve_source(&store, &project_id, &source_asset_id, &project_path).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Source image asset not found or missing: {source_asset_id}."
            ))
        })?;

    let source_image = crate::image_decode::decode_image_any(&source_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("Source image could not be loaded: {e}")))?
        .to_rgb8();
    let (src_w, src_h) = (source_image.width(), source_image.height());

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Downloading,
            0.25,
            "Loading SAM3 weights.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Smart-select canceled while fetching SAM3 weights.",
        fresh_download: false,
    };
    // Both prompt modes load the same facebook/sam3 checkpoint — one resolve/download.
    let (model_path, tokenizer_path) = ensure_segmenter_weights(settings, &context).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.5,
            "Segmenting the selection.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let cancel = CancelFlag::new();
    // Points → SAM3 tracker single-frame PVS (interactive clicks); box → SAM3 concept detector.
    let mask = if let Some(points) = points {
        run_blocking_with_heartbeat(
            api,
            settings,
            &job.id,
            Some(cancel.clone()),
            CANCEL_MESSAGE,
            "smart-select task",
            tokio::task::spawn_blocking(move || {
                segment_points_blocking(model_path, source_image, points)
            }),
        )
        .await?
    } else {
        let box_xyxy = box_xyxy.expect("box prompt present when there are no points");
        run_blocking_with_heartbeat(
            api,
            settings,
            &job.id,
            Some(cancel.clone()),
            CANCEL_MESSAGE,
            "smart-select task",
            tokio::task::spawn_blocking(move || {
                segment_box_blocking(
                    model_path,
                    tokenizer_path,
                    source_image,
                    box_xyxy,
                    &concept,
                    threshold,
                    mask_threshold,
                )
            }),
        )
        .await?
    };

    // The mask is a row-major `src_w*src_h` 0/255 grayscale buffer (white = the selected region).
    let mask_image = image::GrayImage::from_raw(src_w, src_h, mask)
        .ok_or_else(|| WorkerError::Engine("smart-select mask buffer size mismatch".to_owned()))?;

    // Write exactly one child asset (the mask PNG) with lineage back to the source, mirroring the
    // upscale/detail asset-write shape so the API materializes it into `result.assets`.
    let created_at = now_rfc3339();
    let generation_set_id = format!("genset_{}", uuid::Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let date = &created_at[..10];
    let suffix: String = asset_id.chars().skip(6).take(8).collect();
    let filename = format!("{date}_mask_{suffix}.png");
    let media_rel = format!("assets/images/{generation_set_id}/{filename}");
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let tmp_path = media_path.with_extension("tmp.png");
    mask_image
        .save_with_format(&tmp_path, image::ImageFormat::Png)
        .map_err(|e| WorkerError::Io(std::io::Error::other(e)))?;
    tokio::fs::rename(&tmp_path, &media_path)
        .await
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_path);
        })?;

    let source_name = payload
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(source_display)
        .unwrap_or_else(|| "Image".to_owned());
    let fact = json!({
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "type": "image",
        "width": src_w,
        "height": src_h,
        "normalizedWidth": src_w,
        "normalizedHeight": src_h,
        "count": 1,
        "seed": 0,
        "displayName": format!("{source_name} (mask)"),
        "createdAt": created_at.clone(),
        "mode": "image_segment",
        "model": "sam3",
        "adapter": "sam3",
        "prompt": "",
        "negativePrompt": "",
        "loras": [],
        "stylePreset": "",
        "sourceAssetId": source_asset_id,
        "rawAdapterSettings": {
            "segment": segment_settings,
        },
        "parents": [source_asset_id],
        "extra": {
            "isMask": true,
            "maskOfAssetId": source_asset_id,
        },
    });
    let generation_set = json!({
        "id": generation_set_id,
        "mode": "image_segment",
        "model": "sam3",
        "prompt": "",
        "negativePrompt": "",
        "count": 1,
        "createdAt": created_at,
    });
    let mut result = JsonObject::new();
    result.insert(
        "generationSetId".to_owned(),
        Value::String(generation_set_id),
    );
    result.insert("expectedCount".to_owned(), json!(1));
    result.insert("adapter".to_owned(), Value::String("sam3".to_owned()));
    result.insert("model".to_owned(), Value::String("sam3".to_owned()));
    result.insert("generationSet".to_owned(), generation_set);
    result.insert("assetWrites".to_owned(), Value::Array(vec![fact]));

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Smart-select complete.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests;
