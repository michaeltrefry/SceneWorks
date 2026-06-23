//! Native dataset CLIP-embedding analysis (epic 6529 P2, sc-6535).
//!
//! The `dataset_analysis` job embeds every dataset image with the CLIP ViT-L/14 provider through the
//! backend-neutral `gen_core::load_image_embedder` seam (macOS MLX), then POSTs the embeddings to
//! rust-api to persist the content-hash-keyed sidecar — the embedding-side analog of the caption
//! job's `/caption-sidecars` write. macOS-only for now (no candle CLIP embedder yet); off-Mac the
//! job is a precise unsupported error.

use super::*;

#[cfg(target_os = "macos")]
const CLIP_EMBEDDER_ID: &str = "clip_vit_l14";
#[cfg(target_os = "macos")]
const CLIP_EMBEDDER_MODEL: &str = "openai/clip-vit-large-patch14";
#[cfg(target_os = "macos")]
const EMBEDDING_SPACE: &str = "clip-vit-l14";
#[cfg(target_os = "macos")]
const CANCEL_MESSAGE: &str = "Dataset analysis canceled by user.";

#[cfg(target_os = "macos")]
use gen_core::{CancelFlag, Image, LoadSpec, WeightsSource};
// Force-link the MLX CLIP image embedder so its `inventory::submit!` registration (`clip_vit_l14`,
// `gen_core::ImageEmbedder`, backend `mlx`) survives the linker — the embedder analog of the
// JoyCaption anchor in `caption_jobs.rs`.
#[cfg(target_os = "macos")]
use mlx_gen_clip as _;

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct AnalysisItem {
    image_path: PathBuf,
    content_hash: String,
}

#[cfg(target_os = "macos")]
pub(crate) async fn run_dataset_analysis_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let items = analysis_items(settings, &job.payload)?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Dataset analysis job has no items to embed.".to_owned(),
        ));
    }
    let model_name_or_path = job
        .payload
        .get("modelNameOrPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(CLIP_EMBEDDER_MODEL)
        .to_owned();
    let weights_dir =
        resolve_app_managed_model_dir(settings, &model_name_or_path, "CLIP embedder model path")?;
    let backend = backend_label(&settings.gpu_id);
    let total = items.len();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.04,
            "Preparing dataset analysis job.",
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.08,
            "Loading CLIP image embedder.",
            None,
            backend,
        ),
    )
    .await?;

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<usize>(64);
    let blocking_cancel = cancel.clone();
    let blocking_items = items.clone();
    let job_id = job.id.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<Vec<(String, Vec<f32>)>> {
        emit_event(
            "dataset_analysis_load_start",
            json!({ "jobId": job_id, "engine": CLIP_EMBEDDER_ID }),
        );
        let embedder = gen_core::load_image_embedder(
            CLIP_EMBEDDER_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .map_err(|error| WorkerError::Engine(format!("CLIP embedder load failed: {error}")))?;
        emit_event(
            "dataset_analysis_load_complete",
            json!({ "jobId": job_id, "engine": CLIP_EMBEDDER_ID }),
        );
        let mut out = Vec::with_capacity(blocking_items.len());
        for (index, item) in blocking_items.into_iter().enumerate() {
            if blocking_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let image = load_analysis_image(&item.image_path)?;
            let embedding = embedder
                .embed(&image)
                .map_err(|error| WorkerError::Engine(format!("CLIP embed failed: {error}")))?;
            out.push((item.content_hash, embedding));
            // Best-effort per-item progress; a dropped receiver just means we stop reporting.
            let _ = tx.blocking_send(index);
        }
        Ok(out)
    });

    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(index) => {
                        let progress = 0.12 + 0.78 * ((index + 1) as f64 / total as f64);
                        update_job(
                            api,
                            &job.id,
                            analysis_progress(
                                JobStatus::Running,
                                ProgressStage::Running,
                                progress,
                                &format!("Analyzed image {} of {}.", index + 1, total),
                                None,
                                backend,
                            ),
                        )
                        .await?;
                    }
                    None => break,
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                match check_cancel(api, &job.id, CANCEL_MESSAGE).await {
                    Ok(()) => {}
                    Err(WorkerError::Canceled(_)) => cancel.cancel(),
                    Err(error) => return Err(error),
                }
            }
        }
    }

    let embeddings = blocking
        .await
        .map_err(|error| task_join_error("dataset analysis task join", error))??;

    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.94,
            "Saving embeddings.",
            None,
            backend,
        ),
    )
    .await?;
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let dataset_id = required_payload_string(&job.payload, "datasetId")?;
    let records: Vec<Value> = embeddings
        .iter()
        .map(|(content_hash, embedding)| json!({ "contentHash": content_hash, "embedding": embedding }))
        .collect();
    let stored: Value = api
        .post_json(
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/analysis-embeddings"
            ),
            &json!({ "space": EMBEDDING_SPACE, "items": records }),
        )
        .await?;
    update_job(
        api,
        &job.id,
        analysis_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Embedded {} training item(s).", embeddings.len()),
            Some(analysis_result(dataset_id, embeddings.len(), stored)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn analysis_items(settings: &Settings, payload: &JsonObject) -> WorkerResult<Vec<AnalysisItem>> {
    let dataset_root = payload
        .get("datasetRoot")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Dataset analysis payload.datasetRoot must be an app-managed dataset path."
                    .to_owned(),
            )
        })?;
    let items = payload
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Dataset analysis payload.items must be an array.".to_owned(),
            )
        })?;
    items
        .iter()
        .map(|item| {
            let object = item.as_object().ok_or_else(|| {
                WorkerError::InvalidPayload("Dataset analysis item must be an object.".to_owned())
            })?;
            let item_id = object
                .get("itemId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "Dataset analysis item is missing itemId.".to_owned(),
                    )
                })?;
            let content_hash = object
                .get("contentHash")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Dataset analysis item {item_id} is missing contentHash."
                    ))
                })?
                .to_owned();
            let image_path = object
                .get("imagePath")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Dataset analysis item {item_id} is missing imagePath."
                    ))
                })?;
            let image_path = resolve_dataset_item_path(
                settings,
                dataset_root,
                image_path,
                &format!("Dataset analysis item {item_id} imagePath"),
            )?;
            Ok(AnalysisItem {
                image_path,
                content_hash,
            })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn load_analysis_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("analysis image {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

#[cfg(target_os = "macos")]
fn analysis_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

#[cfg(target_os = "macos")]
fn analysis_result(dataset_id: &str, embedded_count: usize, stored: Value) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("embedder".to_owned(), json!(CLIP_EMBEDDER_ID));
    result.insert("space".to_owned(), json!(EMBEDDING_SPACE));
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert("embeddedItemCount".to_owned(), json!(embedded_count));
    result.insert(
        "stored".to_owned(),
        stored.get("stored").cloned().unwrap_or(Value::Null),
    );
    result
}

#[cfg(not(target_os = "macos"))]
pub(crate) async fn run_dataset_analysis_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Dataset analysis (CLIP embedding) needs the macOS MLX backend; no candle CLIP embedder \
         exists yet."
            .to_owned(),
    ))
}
