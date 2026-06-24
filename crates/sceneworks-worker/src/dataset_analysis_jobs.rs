//! Native dataset CLIP-embedding analysis (epic 6529 P2, sc-6535).
//!
//! The `dataset_analysis` job embeds every dataset image with the CLIP ViT-L/14 provider through the
//! backend-neutral `gen_core::load_image_embedder` seam (macOS MLX), then POSTs the embeddings to
//! rust-api to persist the content-hash-keyed sidecar — the embedding-side analog of the caption
//! job's `/caption-sidecars` write. Runs on macOS (MLX) and, with `--features backend-candle`,
//! off-Mac (candle, sc-6537); on a platform with neither, the job is a precise unsupported error.

use super::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CLIP_EMBEDDER_ID: &str = "clip_vit_l14";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CLIP_TEXT_EMBEDDER_ID: &str = "clip_vit_l14_text";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CLIP_EMBEDDER_MODEL: &str = "openai/clip-vit-large-patch14";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const EMBEDDING_SPACE: &str = "clip-vit-l14";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Dataset analysis canceled by user.";

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{CancelFlag, Image, LoadSpec, WeightsSource};
// Force-link the CLIP image + text embedders so their `inventory::submit!` registrations
// (`clip_vit_l14` + `clip_vit_l14_text`) survive the linker — the embedder analog of the JoyCaption
// anchors in `caption_jobs.rs`. MLX on macOS; candle on the `backend-candle` lane (off-Mac, sc-6537).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_clip as _;
#[cfg(target_os = "macos")]
use mlx_gen_clip as _;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct AnalysisItem {
    image_path: PathBuf,
    content_hash: String,
    caption_text: String,
    caption_hash: String,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct AnalysisEmbeddingRecord {
    content_hash: String,
    image_embedding: Vec<f32>,
    caption_hash: Option<String>,
    text_embedding: Option<Vec<f32>>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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
    let blocking =
        tokio::task::spawn_blocking(move || -> WorkerResult<Vec<AnalysisEmbeddingRecord>> {
            emit_event(
                "dataset_analysis_load_start",
                json!({ "jobId": job_id, "engine": CLIP_EMBEDDER_ID }),
            );
            let embedder = gen_core::load_image_embedder(
                CLIP_EMBEDDER_ID,
                &LoadSpec::new(WeightsSource::Dir(weights_dir.clone())),
            )
            .map_err(|error| WorkerError::Engine(format!("CLIP embedder load failed: {error}")))?;
            let needs_text = blocking_items
                .iter()
                .any(|item| !item.caption_text.trim().is_empty());
            let text_embedder = if needs_text {
                Some(
                    gen_core::load_text_embedder(
                        CLIP_TEXT_EMBEDDER_ID,
                        &LoadSpec::new(WeightsSource::Dir(weights_dir)),
                    )
                    .map_err(|error| {
                        WorkerError::Engine(format!("CLIP text embedder load failed: {error}"))
                    })?,
                )
            } else {
                None
            };
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
                let (caption_hash, text_embedding) =
                    if let Some(text_embedder) = text_embedder.as_ref() {
                        if item.caption_text.trim().is_empty() {
                            (None, None)
                        } else {
                            // Embed only the subject phrase (first sentence). Full JoyCaption
                            // "Descriptive/long" captions wash the subject out of CLIP's truncated
                            // text embedding and the cosine loses all matched-vs-mismatched signal —
                            // see `caption_alignment_text` + `CaptionAlignmentThresholds` (sc-6537).
                            let alignment_text =
                                sceneworks_core::dataset_quality::caption_alignment_text(
                                    &item.caption_text,
                                );
                            let text_embedding =
                                text_embedder.embed_text(alignment_text).map_err(|error| {
                                    WorkerError::Engine(format!("CLIP text embed failed: {error}"))
                                })?;
                            (Some(item.caption_hash), Some(text_embedding))
                        }
                    } else {
                        (None, None)
                    };
                out.push(AnalysisEmbeddingRecord {
                    content_hash: item.content_hash,
                    image_embedding: embedding,
                    caption_hash,
                    text_embedding,
                });
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
    let records = analysis_embedding_records_payload(&embeddings);
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

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn analysis_embedding_records_payload(records: &[AnalysisEmbeddingRecord]) -> Vec<Value> {
    records
        .iter()
        .map(|record| {
            json!({
                "contentHash": record.content_hash,
                "embedding": record.image_embedding,
                "captionHash": record.caption_hash,
                "textEmbedding": record.text_embedding,
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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
            let caption_text = object
                .get("captionText")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let caption_hash = object
                .get("captionHash")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| sceneworks_core::dataset_quality::caption_hash(&caption_text));
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
                caption_text,
                caption_hash,
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
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

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub(crate) async fn run_dataset_analysis_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Dataset analysis (CLIP embedding) needs the macOS MLX backend or the candle backend \
         (build with --features backend-candle)."
            .to_owned(),
    ))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn test_settings(data_dir: &Path) -> Settings {
        Settings {
            api_url: "http://127.0.0.1".to_owned(),
            access_token: None,
            data_dir: data_dir.to_path_buf(),
            config_dir: data_dir.join("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "gpu-0".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: DEFAULT_HUGGINGFACE_BASE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
        }
    }

    #[test]
    fn analysis_items_parse_caption_text_and_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        let image_path = dataset_root.join("image.png");
        let caption_hash = sceneworks_core::dataset_quality::caption_hash("a red square");
        let payload = serde_json::Map::from_iter([
            (
                "datasetRoot".to_owned(),
                json!(dataset_root.display().to_string()),
            ),
            (
                "items".to_owned(),
                json!([{
                    "itemId": "item_1",
                    "imagePath": image_path.display().to_string(),
                    "contentHash": "image_hash",
                    "captionText": "a red square",
                    "captionHash": caption_hash,
                }]),
            ),
        ]);

        let items = analysis_items(&settings, &payload).expect("items parse");

        assert_eq!(items[0].content_hash, "image_hash");
        assert_eq!(items[0].caption_text, "a red square");
        assert_eq!(items[0].caption_hash, caption_hash);
    }

    #[test]
    fn analysis_embedding_records_encode_text_pairs_and_empty_captions() {
        let payload = analysis_embedding_records_payload(&[
            AnalysisEmbeddingRecord {
                content_hash: "image_a".to_owned(),
                image_embedding: vec![1.0, 0.0],
                caption_hash: Some("caption_a".to_owned()),
                text_embedding: Some(vec![0.5, 0.25]),
            },
            AnalysisEmbeddingRecord {
                content_hash: "image_b".to_owned(),
                image_embedding: vec![0.0, 1.0],
                caption_hash: None,
                text_embedding: None,
            },
        ]);

        assert_eq!(
            Value::Array(payload),
            json!([
                {
                    "contentHash": "image_a",
                    "embedding": [1.0, 0.0],
                    "captionHash": "caption_a",
                    "textEmbedding": [0.5, 0.25]
                },
                {
                    "contentHash": "image_b",
                    "embedding": [0.0, 1.0],
                    "captionHash": null,
                    "textEmbedding": null
                }
            ])
        );
    }
}

#[cfg(all(test, target_os = "macos"))]
mod real_weights_tests {
    use super::*;

    /// Real-weights worker integration (sc-6535): proves the *worker binary* force-links
    /// `mlx-gen-clip` (so `gen_core::load_image_embedder("clip_vit_l14")` resolves **and loads** —
    /// not just that the descriptor links, which the capability test already covers), and that the
    /// worker's `load_analysis_image` feeds a real image file into the real CLIP forward. `#[ignore]`
    /// per convention — the weights live outside CI; run on a Mac with the snapshot cached + Metal.
    #[test]
    #[ignore = "real-weight: needs the openai/clip-vit-large-patch14 snapshot in the HF cache + Metal"]
    fn embeds_a_real_image_and_caption_through_the_worker_seam() {
        // Locate the cached snapshot (mirrors prompt_refine_jobs.rs's HF-cache resolution).
        let home = std::env::var("HOME").expect("HOME");
        let snapshots = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots");
        let weights_dir = std::fs::read_dir(&snapshots)
            .expect("openai/clip-vit-large-patch14 snapshot dir is cached")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .expect("a snapshot subdir");

        // Write a real (non-uniform) PNG and decode it through the worker's image path.
        let dir = tempfile::tempdir().expect("tempdir");
        let image_path = dir.path().join("probe.png");
        let mut buf = image::RgbImage::new(64, 64);
        for (x, _y, px) in buf.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 4) as u8, 120, 200]);
        }
        buf.save(&image_path).expect("encode probe png");
        let image = load_analysis_image(&image_path).expect("decode via the worker image path");
        assert_eq!((image.width, image.height), (64, 64));

        // Load the embedder through the worker's gen_core seam, then run the real CLIP forward.
        let embedder = gen_core::load_image_embedder(
            CLIP_EMBEDDER_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir.clone())),
        )
        .expect("the worker binary links + loads mlx-gen-clip's clip_vit_l14");
        assert_eq!(embedder.descriptor().embedding_dim, 768);
        let text_embedder = gen_core::load_text_embedder(
            CLIP_TEXT_EMBEDDER_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .expect("the worker binary links + loads mlx-gen-clip's clip_vit_l14_text");
        assert_eq!(text_embedder.descriptor().embedding_dim, 768);

        let embedding = embedder.embed(&image).expect("real CLIP embed");
        assert_eq!(embedding.len(), 768, "CLIP ViT-L/14 embedding is 768-d");
        assert!(
            embedding.iter().all(|v| v.is_finite()) && embedding.iter().any(|&v| v != 0.0),
            "embedding is finite + non-degenerate"
        );
        let text_embedding = text_embedder
            .embed_text("a blue and green gradient probe")
            .expect("real CLIP text embed");
        assert_eq!(
            text_embedding.len(),
            768,
            "CLIP ViT-L/14 text embedding is 768-d"
        );
        assert!(
            text_embedding.iter().all(|v| v.is_finite())
                && text_embedding.iter().any(|&v| v != 0.0),
            "text embedding is finite + non-degenerate"
        );
        println!(
            "worker clip ok: image_dim={} text_dim={}",
            embedding.len(),
            text_embedding.len()
        );
    }

    /// Tier-1 threshold calibration (sc-6535 follow-up — NOT committed): embed real LoRA training
    /// sets through the production CLIP seam and dump per-dataset diversity + near-duplicate
    /// distributions so `Tier1Thresholds` (near_dup_cosine / diversity_floor) can be set from data.
    /// `CALIB_DIR` may point at a flat image dir (one dataset) OR a dir of image-bearing subdirs
    /// (each a dataset, e.g. `~/Datasets/dreambooth/dataset`). Run:
    ///   CALIB_DIR=~/Datasets/dreambooth/dataset RUST_TEST_THREADS=1 \
    ///     cargo test -p sceneworks-worker calibrate_thresholds -- --ignored --nocapture
    #[test]
    #[ignore = "calibration: needs CLIP snapshot + datasets under ~/Datasets (set CALIB_DIR)"]
    fn calibrate_thresholds() {
        let home = std::env::var("HOME").expect("HOME");
        let snapshots = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots");
        let weights_dir = std::fs::read_dir(&snapshots)
            .expect("clip snapshot cached")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .expect("a snapshot subdir");

        let root = std::path::PathBuf::from(
            std::env::var("CALIB_DIR").unwrap_or_else(|_| format!("{home}/Datasets/Basim")),
        );
        let is_img = |path: &std::path::Path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "webp" | "bmp"
                    )
                })
                .unwrap_or(false)
        };
        let direct_imgs = |dir: &std::path::Path| {
            let mut imgs: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
                .map(|rd| {
                    rd.filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.is_file() && is_img(path))
                        .collect()
                })
                .unwrap_or_default();
            imgs.sort();
            imgs
        };

        // Each image-bearing immediate subdir is a dataset; else the root itself is one dataset.
        let mut subdirs: Vec<std::path::PathBuf> = std::fs::read_dir(&root)
            .expect("root dir")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        subdirs.sort();
        let mut datasets: Vec<(String, Vec<std::path::PathBuf>)> = Vec::new();
        for dir in &subdirs {
            let imgs = direct_imgs(dir);
            if imgs.len() >= 2 {
                datasets.push((
                    dir.file_name().unwrap().to_string_lossy().into_owned(),
                    imgs,
                ));
            }
        }
        let root_imgs = direct_imgs(&root);
        if root_imgs.len() >= 2 {
            datasets.push((
                root.file_name().unwrap().to_string_lossy().into_owned(),
                root_imgs,
            ));
        }
        assert!(
            !datasets.is_empty(),
            "no image-bearing datasets under {}",
            root.display()
        );

        let embedder = gen_core::load_image_embedder(
            CLIP_EMBEDDER_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .expect("load clip");

        println!(
            "\n=== CLIP calibration over {} dataset(s) in {} ===",
            datasets.len(),
            root.display()
        );
        // Per-dataset rows (name, n, diversity, max_cos, dups>=0.95) + a pooled near-dup tally.
        let thresholds = [0.90_f32, 0.93, 0.95, 0.97, 0.99];
        let mut rows: Vec<(String, usize, f32, f32, usize)> = Vec::new();
        let mut pooled_dups = [0usize; 5];
        let mut pooled_pairs = 0usize;
        for (name, paths) in &datasets {
            let mut vecs: Vec<Vec<f32>> = Vec::new();
            for path in paths {
                let image = match load_analysis_image(path) {
                    Ok(image) => image,
                    Err(error) => {
                        println!("skip {}: {error}", path.display());
                        continue;
                    }
                };
                let mut embedding = embedder.embed(&image).expect("embed");
                // L2-normalize so dot == cosine (the embed contract returns RAW vectors, exactly
                // as evaluate_tier1's caller normalizes them before cosine clustering).
                let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut embedding {
                        *v /= norm;
                    }
                }
                vecs.push(embedding);
            }
            let n = vecs.len();
            if n < 2 {
                continue;
            }
            let mut sum = 0.0f32;
            let mut cnt = 0usize;
            let mut max_cos = f32::MIN;
            let mut dups95 = 0usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    let cos: f32 = vecs[i].iter().zip(&vecs[j]).map(|(a, b)| a * b).sum();
                    sum += cos;
                    cnt += 1;
                    pooled_pairs += 1;
                    if cos > max_cos {
                        max_cos = cos;
                    }
                    if cos >= 0.95 {
                        dups95 += 1;
                    }
                    for (k, threshold) in thresholds.iter().enumerate() {
                        if cos >= *threshold {
                            pooled_dups[k] += 1;
                        }
                    }
                }
            }
            let diversity = 1.0 - sum / cnt as f32;
            rows.push((name.clone(), n, diversity, max_cos, dups95));
        }

        rows.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        println!(
            "{:<24} {:>4} {:>9} {:>8} {:>9}",
            "dataset", "n", "diversity", "max_cos", "dup>=.95"
        );
        for (name, n, diversity, max_cos, dups) in &rows {
            println!(
                "{:<24} {:>4} {:>9.4} {:>8.4} {:>9}",
                name, n, diversity, max_cos, dups
            );
        }

        // Pooled near-duplicate behavior across every within-dataset pair in the corpus.
        println!("\npooled near-dup crossings across {pooled_pairs} pairs:");
        for (k, threshold) in thresholds.iter().enumerate() {
            println!(
                "  >= {threshold:.2}: {} ({:.1}%)",
                pooled_dups[k],
                100.0 * pooled_dups[k] as f32 / pooled_pairs.max(1) as f32
            );
        }

        // Floor-calibration distribution (only meaningful with several datasets).
        if rows.len() >= 3 {
            let mut divs: Vec<f32> = rows.iter().map(|r| r.2).collect();
            divs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f32| divs[((p / 100.0) * (divs.len() as f32 - 1.0)).round() as usize];
            println!(
                "\nper-dataset diversity: min={:.4} p10={:.4} median={:.4} p90={:.4} max={:.4}",
                divs[0],
                pct(10.0),
                pct(50.0),
                pct(90.0),
                divs[divs.len() - 1]
            );
            for floor in [0.10f32, 0.12, 0.14, 0.15, 0.18, 0.20] {
                let flagged = rows.iter().filter(|r| r.2 < floor).count();
                println!(
                    "  floor {floor:.2}: {flagged} / {} datasets flagged low_diversity",
                    rows.len()
                );
            }
        }
    }

    /// Caption↔image alignment floor calibration (sc-6537). Embeds each image under `CALIB_DIR` with
    /// its sibling `<stem>.txt` caption, then dumps the MATCHED (own-caption) vs MISMATCHED
    /// (cross-subject) cosine distributions + a floor-crossing table so
    /// `CaptionAlignmentThresholds::cosine_floor` can be set from data. Subject = the image's parent
    /// dir (`<subject>/`), so MISMATCHED stays strictly cross-subject.
    ///
    /// Prints TWO tables — `[FULL CAPTION]` (embed the whole `.txt`) vs `[FIRST SENTENCE]` (embed only
    /// `caption_alignment_text`, what the worker actually does). The production-faithful result that set
    /// the `0.15` floor: `~/Datasets/dreambooth/dataset` captioned by the **real** JoyCaption job
    /// ("Descriptive/long"). FULL CAPTION → matched & mismatched medians both ≈0.11 (no signal);
    /// FIRST SENTENCE → matched min ≈0.18 / median ≈0.25 vs mismatched max ≈0.16 (clean gap). Embedding
    /// the full descriptive paragraph washes the subject out of CLIP's 77-token-truncated text vector.
    /// Run:
    ///   CALIB_DIR=~/Datasets/dreambooth/dataset RUST_TEST_THREADS=1 \
    ///     cargo test -p sceneworks-worker sweep_caption_alignment -- --ignored --nocapture
    #[test]
    #[ignore = "calibration: caption↔image alignment over CALIB_DIR images + sibling .txt captions"]
    fn sweep_caption_alignment() {
        let home = std::env::var("HOME").expect("HOME");
        let snapshots = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots");
        let weights_dir = std::fs::read_dir(&snapshots)
            .expect("clip snapshot cached")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .expect("a snapshot subdir");
        let root = std::path::PathBuf::from(
            std::env::var("CALIB_DIR")
                .unwrap_or_else(|_| format!("{home}/Datasets/dreambooth/dataset")),
        );

        // Recursively collect (image, caption text) pairs: an image with a sibling `<stem>.txt`.
        fn collect(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, String)>) {
            let Ok(read) = std::fs::read_dir(dir) else {
                return;
            };
            let mut entries: Vec<std::path::PathBuf> =
                read.filter_map(Result::ok).map(|e| e.path()).collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    collect(&path, out);
                } else if path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
                {
                    if let Ok(caption) = std::fs::read_to_string(path.with_extension("txt")) {
                        let caption = caption.trim().to_owned();
                        if !caption.is_empty() {
                            out.push((path, caption));
                        }
                    }
                }
            }
        }
        let mut pairs = Vec::new();
        collect(&root, &mut pairs);
        assert!(
            pairs.len() >= 4,
            "need >=4 (image, .txt caption) pairs under {} — got {}",
            root.display(),
            pairs.len()
        );

        let image_embedder = gen_core::load_image_embedder(
            CLIP_EMBEDDER_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir.clone())),
        )
        .expect("image embedder");
        let text_embedder = gen_core::load_text_embedder(
            CLIP_TEXT_EMBEDDER_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .expect("text embedder");
        let norm = |mut v: Vec<f32>| {
            let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag > 0.0 {
                for x in &mut v {
                    *x /= mag;
                }
            }
            v
        };
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();

        // Subject label = parent dir name (dreambooth: dog/cat/teapot/backpack). Keeps MISMATCHED
        // strictly cross-subject — a *different photo of the same subject* is a genuine match, not a
        // mismatch, so counting it as one would bias toward a no-separation conclusion.
        let subject = |p: &std::path::Path| {
            p.parent()
                .and_then(|d| d.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_owned()
        };

        let mut imgs = Vec::with_capacity(pairs.len());
        let mut subjects = Vec::with_capacity(pairs.len());
        let mut full_caps = Vec::with_capacity(pairs.len());
        for (path, caption) in &pairs {
            let image = load_analysis_image(path).expect("decode");
            imgs.push(norm(image_embedder.embed(&image).expect("img embed")));
            subjects.push(subject(path));
            full_caps.push(caption.clone());
        }
        let n = imgs.len();

        // Subject-phrase reduction — strips the long descriptive paragraph CLIP truncates at 77 tokens
        // anyway, leaving the subject phrase ("Photograph of a Shiba Inu dog"). Delegates to the SAME
        // core fn the worker applies in production, so this calibrated floor matches what the analysis
        // job actually embeds.
        let first_sentence =
            |c: &str| sceneworks_core::dataset_quality::caption_alignment_text(c).to_owned();

        // Run matched/mismatched/floor analysis for one caption-preprocessing mode.
        let analyze = |label: &str, prep: &dyn Fn(&str) -> String| {
            let txts: Vec<Vec<f32>> = full_caps
                .iter()
                .map(|c| norm(text_embedder.embed_text(&prep(c)).expect("text embed")))
                .collect();
            let mut matched: Vec<f32> = (0..n).map(|i| cos(&imgs[i], &txts[i])).collect();
            // Every cross-subject (image_i, caption_j) pair is a genuinely wrong caption.
            let mut mismatched: Vec<f32> = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    if subjects[i] != subjects[j] {
                        mismatched.push(cos(&imgs[i], &txts[j]));
                    }
                }
            }
            matched.sort_by(|a, b| a.partial_cmp(b).unwrap());
            mismatched.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct =
                |v: &[f32], p: f32| v[((p / 100.0) * (v.len() as f32 - 1.0)).round() as usize];

            println!(
                "\n=== [{label}] caption↔image alignment over {n} pairs in {} ===",
                root.display()
            );
            println!(
                "MATCHED    (own caption):   min={:.4} p05={:.4} p25={:.4} median={:.4} p75={:.4}",
                matched[0],
                pct(&matched, 5.0),
                pct(&matched, 25.0),
                pct(&matched, 50.0),
                pct(&matched, 75.0)
            );
            println!(
                "MISMATCHED (cross-subject): median={:.4} p75={:.4} p95={:.4} max={:.4}  (n={})",
                pct(&mismatched, 50.0),
                pct(&mismatched, 75.0),
                pct(&mismatched, 95.0),
                mismatched[mismatched.len() - 1],
                mismatched.len()
            );
            println!("floor: % MATCHED flagged (false-pos — keep low) | % MISMATCHED flagged (caught — keep high):");
            for floor in [0.06f32, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.24] {
                let false_pos = matched.iter().filter(|c| **c < floor).count();
                let caught = mismatched.iter().filter(|c| **c < floor).count();
                println!(
                    "  floor {floor:.2}: matched {false_pos}/{n} ({:.0}%) | mismatched {caught}/{} ({:.0}%)",
                    100.0 * false_pos as f32 / n as f32,
                    mismatched.len(),
                    100.0 * caught as f32 / mismatched.len() as f32
                );
            }
        };

        analyze("FULL CAPTION", &|c: &str| c.to_owned());
        analyze("FIRST SENTENCE", &first_sentence);
    }
}
