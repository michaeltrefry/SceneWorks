//! Native dataset captioning (epic 3550, sc-3556 MLX; epic 5095 sc-5098 candle).
//!
//! SceneWorks keeps the existing `training_caption` job contract and result shape,
//! but the in-process Rust worker can serve `captioner=joy_caption` through the
//! backend-neutral `gen_core::load_captioner` seam: the macOS `mlx` worker via mlx-gen's
//! JoyCaption provider, and the Windows/CUDA candle worker via candle-gen-joycaption
//! (`--features backend-candle`). `cfg(target_os)` only decides which provider crate registers the
//! captioner; the job flow below is identical. The Python torch captioner remains the explicit
//! non-MLX/non-candle fallback (and the default Windows/Linux path).

use super::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const JOY_CAPTION_MODEL: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Training captioning canceled by user.";

// epic 3720 (sc-3724): the backend-neutral captioner contract types come from `gen_core`; the
// `as _;` provider link below stays mlx-gen-specific (it registers the JoyCaption captioner into
// the registry).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::{
    CancelFlag, CaptionOptions, CaptionRequest, CaptionSampling, Image, LoadSpec, Progress,
    WeightsSource,
};
#[cfg(target_os = "macos")]
use mlx_gen_joycaption as _;
// Candle JoyCaption captioner force-link anchor (sc-5098): keeps its `inventory::submit!` captioner
// registration (`fancyfeast/llama-joycaption-beta-one-hf-llava`, backend `candle`) from being dropped
// by the MSVC release linker. Mirrors the mlx anchor above.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_joycaption as _;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct CaptionItem {
    item_id: String,
    image_path: PathBuf,
    trigger_words: Vec<String>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Clone, Debug)]
struct CaptionJobOptions {
    options: CaptionOptions,
    sampling: CaptionSampling,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[derive(Debug)]
enum CaptionEvent {
    Step {
        index: usize,
        current: u32,
        total: u32,
    },
    Captioned {
        index: usize,
        item_id: String,
        text: String,
        trigger_words: Vec<String>,
    },
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_training_caption_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    if job
        .payload
        .get("captioner")
        .and_then(Value::as_str)
        .unwrap_or_default()
        != "joy_caption"
    {
        return Err(WorkerError::InvalidPayload(
            "Unsupported training captioner for the MLX worker; use joy_caption.".to_owned(),
        ));
    }

    let items = caption_items(settings, &job.payload)?;
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Training caption job has no items to caption.".to_owned(),
        ));
    }
    let options = caption_job_options(&job.payload);
    let model_name_or_path = job
        .payload
        .get("modelNameOrPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(JOY_CAPTION_MODEL)
        .to_owned();
    let weights_dir = resolve_caption_weights_dir(settings, &model_name_or_path)?;
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.04,
            "Preparing training caption job.",
            None,
            backend,
        ),
    )
    .await?;

    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.08,
            "Loading JoyCaption MLX model.",
            None,
            backend,
        ),
    )
    .await?;

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<CaptionEvent>(64);
    let blocking_cancel = cancel.clone();
    let blocking_items = items.clone();
    let blocking_options = options.clone();
    let job_id = job.id.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<()> {
        emit_event(
            "caption_pipeline_load_start",
            json!({
                "jobId": job_id,
                "engine": JOY_CAPTION_MODEL,
            }),
        );
        let captioner = gen_core::load_captioner(
            JOY_CAPTION_MODEL,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .map_err(|error| WorkerError::Engine(format!("JoyCaption MLX load failed: {error}")))?;
        emit_event(
            "caption_pipeline_load_complete",
            json!({
                "jobId": job_id,
                "engine": JOY_CAPTION_MODEL,
            }),
        );

        for (index, item) in blocking_items.into_iter().enumerate() {
            if blocking_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let image = load_caption_image(&item.image_path)?;
            let mut request = CaptionRequest {
                image,
                options: blocking_options.options.clone(),
                sampling: blocking_options.sampling,
                trigger_words: item.trigger_words.clone(),
                cancel: blocking_cancel.clone(),
                ..Default::default()
            };
            request.prompt = request.options.custom_prompt.clone();
            let mut on_progress = |progress: Progress| {
                if let Progress::Step { current, total } = progress {
                    let _ = tx.blocking_send(CaptionEvent::Step {
                        index,
                        current,
                        total,
                    });
                }
            };
            let output = captioner
                .caption(&request, &mut on_progress)
                .map_err(|error| {
                    WorkerError::Engine(format!("JoyCaption MLX generation failed: {error}"))
                })?;
            // Prepend any missing trigger words to the caption. Backend-neutral worker-local helper
            // (sc-5098) — the same logic mlx-gen + candle-gen each ship locally — so the shared path
            // names no backend-specific symbol.
            let text = apply_trigger_words(&output.text, &item.trigger_words);
            tx.blocking_send(CaptionEvent::Captioned {
                index,
                item_id: item.item_id,
                text,
                trigger_words: item.trigger_words,
            })
            .map_err(|_| WorkerError::Canceled(CANCEL_MESSAGE.to_owned()))?;
        }
        Ok(())
    });

    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut captions = Vec::with_capacity(items.len());
    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(CaptionEvent::Step { index, current, total }) => {
                        let progress = caption_step_progress(index, current, total, items.len());
                        update_job(
                            api,
                            &job.id,
                            caption_progress(
                                JobStatus::Running,
                                ProgressStage::Running,
                                progress,
                                &format!("Captioning image {} of {}.", index + 1, items.len()),
                                None,
                                backend,
                            ),
                        )
                        .await?;
                    }
                    Some(CaptionEvent::Captioned { index, item_id, text, trigger_words }) => {
                        captions.push(json!({
                            "itemId": item_id,
                            "caption": {
                                "text": text,
                                "source": "auto",
                                "triggerWords": trigger_words,
                            }
                        }));
                        let progress = 0.12 + 0.76 * ((index + 1) as f64 / items.len() as f64);
                        update_job(
                            api,
                            &job.id,
                            caption_progress(
                                JobStatus::Running,
                                ProgressStage::Running,
                                progress,
                                &format!("Captioned image {} of {}.", index + 1, items.len()),
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

    blocking
        .await
        .map_err(|error| task_join_error("caption task join", error))??;

    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.94,
            "Saving generated captions.",
            None,
            backend,
        ),
    )
    .await?;
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let dataset_id = required_payload_string(&job.payload, "datasetId")?;
    let sidecars: Value = api
        .post_json(
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-sidecars"
            ),
            &json!({ "items": captions }),
        )
        .await?;
    update_job(
        api,
        &job.id,
        caption_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Created captions for {} training item(s).", items.len()),
            Some(caption_result(
                &model_name_or_path,
                dataset_id,
                items.len(),
                sidecars,
            )),
            backend,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
pub(crate) async fn run_training_caption_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "The in-process JoyCaption worker needs the macOS MLX backend or the Windows candle backend; \
         use the Python torch captioner on this platform."
            .to_owned(),
    ))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_progress(
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
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_result(
    model_name_or_path: &str,
    dataset_id: &str,
    captioned_count: usize,
    sidecars: Value,
) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("captioner".to_owned(), json!("joy_caption"));
    result.insert("modelNameOrPath".to_owned(), json!(model_name_or_path));
    result.insert("datasetId".to_owned(), json!(dataset_id));
    result.insert(
        "datasetVersion".to_owned(),
        sidecars
            .get("dataset")
            .and_then(|dataset| dataset.get("version"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert("captionedItemCount".to_owned(), json!(captioned_count));
    result.insert(
        "sidecars".to_owned(),
        sidecars
            .get("sidecars")
            .cloned()
            .unwrap_or_else(|| json!([])),
    );
    result
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_items(settings: &Settings, payload: &JsonObject) -> WorkerResult<Vec<CaptionItem>> {
    let dataset_root = payload
        .get("datasetRoot")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Caption payload.datasetRoot must be an app-managed dataset path.".to_owned(),
            )
        })?;
    let items = payload
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Caption payload.items must be an array.".to_owned())
        })?;
    items
        .iter()
        .map(|item| {
            let object = item.as_object().ok_or_else(|| {
                WorkerError::InvalidPayload("Caption item must be an object.".to_owned())
            })?;
            let item_id = object
                .get("itemId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("Caption item is missing itemId.".to_owned())
                })?
                .to_owned();
            let image_path = object
                .get("imagePath")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "Caption item {item_id} is missing imagePath."
                    ))
                })?;
            let trigger_words = object
                .get("triggerWords")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let image_path = resolve_dataset_item_path(
                settings,
                dataset_root,
                image_path,
                &format!("Caption item {item_id} imagePath"),
            )?;
            Ok(CaptionItem {
                item_id,
                image_path,
                trigger_words,
            })
        })
        .collect()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn caption_job_options(payload: &JsonObject) -> CaptionJobOptions {
    let options = payload
        .get("options")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    CaptionJobOptions {
        options: CaptionOptions {
            caption_type: option_string(&options, "captionType", "Descriptive"),
            caption_length: option_string(&options, "captionLength", "long"),
            extra_options: options
                .get("extraOptions")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            name_input: option_string(&options, "nameInput", ""),
            custom_prompt: option_string(&options, "captionPrompt", ""),
            low_vram: options
                .get("lowVram")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        sampling: CaptionSampling {
            temperature: value_f64(options.get("temperature").unwrap_or(&Value::Null), 0.6) as f32,
            top_p: value_f64(options.get("topP").unwrap_or(&Value::Null), 0.9) as f32,
            max_new_tokens: options
                .get("maxNewTokens")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(256),
            // sc-3963 engine knob: `None` keeps the per-call fresh seed (captions vary across
            // runs, the pre-bump behavior); an explicit `options.seed` reproduces a caption.
            seed: options.get("seed").and_then(Value::as_u64),
            // gen-core d8038beb (sc-7176 pin sync) exposed the JoyCaption repetition penalty that was
            // previously an internal engine constant. `..Default::default()` fills the new
            // `repetition_penalty` (1.05) + `repetition_context` (256) with the shipped values, so
            // captions stay byte-identical to the pre-bump behavior (and any future additive sampling
            // fields keep their engine defaults until the worker wires them).
            ..CaptionSampling::default()
        },
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn option_string(options: &JsonObject, key: &str, default: &str) -> String {
    options
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_caption_weights_dir(
    settings: &Settings,
    model_name_or_path: &str,
) -> WorkerResult<PathBuf> {
    resolve_app_managed_model_dir(settings, model_name_or_path, "JoyCaption model path")
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn load_caption_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("caption image {}: {error}", path.display()))
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
fn caption_step_progress(index: usize, current: u32, total: u32, item_count: usize) -> f64 {
    let item_count = item_count.max(1) as f64;
    let within = if total > 0 {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.12 + 0.76 * ((index as f64 + within) / item_count)).min(0.9)
}

/// Prepend the trigger words that are not already present (case-insensitive substring) to the
/// captioner's text, comma-joined, with the cleaned caption last. Backend-neutral copy of the
/// identical helper both `mlx-gen` and `candle-gen` ship locally (the comment in the engines noted it
/// should be lifted out of the provider) — keeping it here means the shared caption path names no
/// backend-specific symbol (sc-5098). Matches the engine behavior 1:1, including the empty-caption
/// case (just the missing trigger words).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn apply_trigger_words(caption: &str, trigger_words: &[String]) -> String {
    let cleaned = caption.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower_caption = cleaned.to_lowercase();
    let mut parts: Vec<String> = trigger_words
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty())
        .filter(|word| !lower_caption.contains(&word.to_lowercase()))
        .map(ToOwned::to_owned)
        .collect();
    if !cleaned.is_empty() {
        parts.push(cleaned);
    }
    parts.join(", ")
}

#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
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
            gpu_memory_limit_bytes: 0,
        }
    }

    #[test]
    fn caption_job_options_preserve_training_surface() {
        let options = caption_job_options(&serde_json::Map::from_iter([(
            "options".to_owned(),
            json!({
                "captionType": "Straightforward",
                "captionLength": "40",
                "extraOptions": ["Mention lighting."],
                "nameInput": "Mira",
                "captionPrompt": "Use the custom prompt.",
                "temperature": 0.5,
                "topP": 0.8,
                "maxNewTokens": 128,
                "lowVram": true
            }),
        )]));
        assert_eq!(options.options.caption_type, "Straightforward");
        assert_eq!(options.options.caption_length, "40");
        assert_eq!(options.options.extra_options, vec!["Mention lighting."]);
        assert_eq!(options.options.name_input, "Mira");
        assert_eq!(options.options.custom_prompt, "Use the custom prompt.");
        assert!(options.options.low_vram);
        assert_eq!(options.sampling.temperature, 0.5);
        assert_eq!(options.sampling.top_p, 0.8);
        assert_eq!(options.sampling.max_new_tokens, 128);
    }

    #[test]
    fn apply_trigger_words_prepends_only_missing() {
        let triggers = vec!["mika_token".to_owned(), "hat".to_owned()];
        // "hat" already appears in the caption → dropped; "mika_token" prepended.
        assert_eq!(
            apply_trigger_words("A portrait of Mika wearing a hat.", &triggers),
            "mika_token, A portrait of Mika wearing a hat."
        );
        // Empty/whitespace caption → just the (whitespace-normalized) trigger words.
        assert_eq!(apply_trigger_words("   ", &triggers), "mika_token, hat");
        // No triggers → the cleaned caption unchanged.
        assert_eq!(apply_trigger_words("  a  cat ", &[]), "a cat");
    }

    #[test]
    fn caption_items_require_ids_and_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        let image_path = dataset_root.join("image.png");
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
                    "triggerWords": ["miraStyle", ""]
                }]),
            ),
        ]);
        let items = caption_items(&settings, &payload).expect("items parse");
        assert_eq!(items[0].item_id, "item_1");
        assert_eq!(items[0].image_path, image_path);
        assert_eq!(items[0].trigger_words, vec!["miraStyle"]);
    }

    #[test]
    fn caption_items_reject_paths_outside_dataset_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = test_settings(dir.path());
        let dataset_root = dir.path().join("datasets").join("ds-1");
        let payload = serde_json::Map::from_iter([
            (
                "datasetRoot".to_owned(),
                json!(dataset_root.display().to_string()),
            ),
            (
                "items".to_owned(),
                json!([{
                    "itemId": "item_1",
                    "imagePath": dir.path().join("other.png").display().to_string()
                }]),
            ),
        ]);
        let error = caption_items(&settings, &payload).expect_err("unsafe image path rejected");
        assert!(
            error.to_string().contains("Caption item item_1 imagePath"),
            "{error}"
        );
    }
}
