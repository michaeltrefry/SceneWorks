//! SenseNova-U1 understanding + interleave jobs (epic 3180, sc-3905 — Path B).
//!
//! These are the two SenseNova-U1 modes the mlx-gen `Generator` contract can't express
//! (`GenerationOutput` is Images/Video only), so they bypass the registry and call the public
//! [`T2iModel`](mlx_gen_sensenova::T2iModel) methods directly:
//!
//! * **VQA** (`image_vqa`): one source image + a question → a text answer. No asset write.
//! * **Interleave / Document Studio** (`image_interleave`): a prompt (+ optional source images) →
//!   ordered text + generated images, persisted as image assets plus an
//!   [`InterleavedDocument`](sceneworks_core::contracts::InterleavedDocument) `document` asset.
//!
//! Path A (sc-3900) routes the image-producing modes through the `Generator` registry and bumped
//! the `mlx-gen` pin + force-linked `mlx_gen_sensenova`; this module reuses that dependency. The
//! `Generator` crate has no public `SenseNova` constructor, so the worker assembles a concrete
//! `T2iModel` here (replicating the engine's private `load_inner`) from the public re-exports.
//!
//! Parity: VQA mirrors the Python `SenseNovaU1Adapter.answer_question`; interleave mirrors
//! `generate_interleaved` / `_write_interleaved_document` byte-for-byte (request fields, the
//! interleave resolution buckets + think/no-think system protocol, and the response/asset shapes).
//! The understanding + generation model loads dense (no distill LoRA, no quantization) exactly as
//! the torch adapter does (`_load_model(distill_lora=None)`), keeping the VQA decode bit-identical.
//!
//! Off macOS the in-process engine is unavailable and the `image_vqa` / `image_interleave`
//! capabilities are never advertised by this worker, so the handlers are unreachable stubs that
//! error loudly (the Python torch worker serves these modes on Windows/Linux).

use super::*;
// Only the macOS handlers parse an `ImageRequest`; the non-macOS stubs don't.
#[cfg(target_os = "macos")]
use sceneworks_core::image_request::ImageRequest;

#[cfg(target_os = "macos")]
use mlx_gen::image::{decoded_to_image, resize_bicubic_u8};
#[cfg(target_os = "macos")]
use mlx_gen::tokenizer::TextTokenizer;
#[cfg(target_os = "macos")]
use mlx_gen::Image;
#[cfg(target_os = "macos")]
use mlx_gen_sensenova::{
    load_raw, load_tokenizer, smart_resize, NeoChatConfig, Sampler, T2iModel, T2iOptions,
    INTERLEAVE_RESOLUTIONS, INTERLEAVE_SYSTEM_MESSAGE,
};
#[cfg(target_os = "macos")]
use mlx_rs::ops::divide;
#[cfg(target_os = "macos")]
use mlx_rs::Array;

/// The adapter id recorded on the generated assets + the interleaved document (the MLX SenseNova
/// path, matching the `adapter_label` the sc-3900 image rows use).
#[cfg(target_os = "macos")]
const SENSENOVA_ADAPTER: &str = "mlx_sensenova";

// ===========================================================================
// VQA (image_vqa)
// ===========================================================================

/// Visual question answering: a text answer about one source image. Mirrors the Python
/// `SenseNovaU1Adapter.answer_question` — same request fields, same `{answer, question,
/// sourceAssetId, model, realModelInference}` result, no asset write. The source image resolves
/// only through the project sidecar/DB (`load_reference_image`), so there is no client-supplied
/// path escape.
#[cfg(target_os = "macos")]
pub(crate) async fn run_vqa_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let model_id = if request.model.trim().is_empty() {
        "sensenova_u1_8b".to_owned()
    } else {
        request.model.clone()
    };
    let question = job
        .payload
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Visual question answering requires a question.".to_owned())
        })?
        .to_owned();
    let source_asset_id = job
        .payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Visual question answering requires a source image asset.".to_owned(),
            )
        })?
        .to_owned();

    // VQA latency ~ output tokens + input vision tokens; both default low and are tunable per
    // request (top-level payload, mirroring the Python adapter — NOT under `advanced`).
    let max_new_tokens = payload_int(&job.payload, "maxNewTokens", 256, 16, 2048) as usize;
    let max_image_pixels = payload_int(
        &job.payload,
        "maxImagePixels",
        768 * 768,
        256 * 256,
        2048 * 2048,
    );

    let weights_dir = resolve_weights_dir(&request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SenseNova-U1 weights not found".to_owned()))?;

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let backend = backend_label(&settings.gpu_id).to_owned();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing visual question.",
            None,
            &backend,
        ),
    )
    .await?;

    // Decode the source image on the async side (Send `Image` moves into the blocking task).
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &source_asset_id,
        &project_path,
    )?;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.6,
            "Analyzing image.",
            None,
            &backend,
        ),
    )
    .await?;

    let job_id = job.id.clone();
    let question_for_vqa = question.clone();
    let answer = tokio::task::spawn_blocking(move || -> WorkerResult<String> {
        emit_load_event("image_pipeline_load_start", &job_id, "sensenova_u1_8b", 0);
        let (model, tokenizer) = load_sensenova_model(&weights_dir)?;
        emit_load_event(
            "image_pipeline_load_complete",
            &job_id,
            "sensenova_u1_8b",
            0,
        );
        // ImageNet-normalized inside `vqa`; pass [3,H,W] in [0,1], 32-aligned, within the
        // understanding pixel budget (default 768², `load_image_native` min 256²).
        let pixel_values = image_to_chw01(&source, 256 * 256, max_image_pixels)?;
        let answer = model
            .vqa(
                &tokenizer,
                &question_for_vqa,
                std::slice::from_ref(&pixel_values),
                max_new_tokens,
                Sampler::Greedy,
            )
            .map_err(|error| {
                WorkerError::InvalidPayload(format!("SenseNova VQA failed: {error}"))
            })?;
        Ok(strip_reasoning(&answer))
    })
    .await
    .map_err(|error| WorkerError::InvalidPayload(format!("VQA task join: {error}")))??;

    let result = json!({
        "answer": answer,
        "question": question,
        "sourceAssetId": source_asset_id,
        "model": model_id,
        "realModelInference": true,
    })
    .as_object()
    .cloned()
    .expect("json! object literal");

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Answer ready.",
            Some(result),
            &backend,
        ),
    )
    .await?;
    Ok(())
}

/// Off macOS the in-process engine is unavailable; `image_vqa` is served by the Python torch
/// worker (the `mlx` worker — the only one advertising this capability — is macOS-only).
#[cfg(not(target_os = "macos"))]
pub(crate) async fn run_vqa_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_vqa runs on the macOS MLX worker or the Python torch worker, not this Rust worker"
            .to_owned(),
    ))
}

// ===========================================================================
// Interleave / Document Studio (image_interleave)
// ===========================================================================

/// Interleaved text-image generation: one model rollout yields ordered text + images, persisted as
/// a `document` asset whose segments reference the generated image assets in order. Mirrors the
/// Python `generate_interleaved` → `_write_interleaved_document` contract (request fields, resolution
/// buckets, think/no-think protocol, asset/result shapes). The base understanding+generation model
/// loads dense (no distill LoRA) — interleave needs the full model, never the distilled gen LoRA.
#[cfg(target_os = "macos")]
pub(crate) async fn run_interleave_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let prompt = request.prompt.trim().to_owned();
    if prompt.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Interleaved generation requires a prompt.".to_owned(),
        ));
    }
    let model_id = if request.model.trim().is_empty() {
        "sensenova_u1_8b".to_owned()
    } else {
        request.model.clone()
    };
    let advanced = &request.advanced;

    let max_images = advanced_int(advanced, "maxImages", 6, 1, 10) as usize;
    // Snap the requested W×H to the nearest interleave bucket by aspect ratio (log-space), mirroring
    // the Python `interleave_resolution_for`. Defaults 2048×1152 (16:9), clamped 256..4096.
    let req_width = payload_int(&job.payload, "width", 2048, 256, 4096);
    let req_height = payload_int(&job.payload, "height", 1152, 256, 4096);
    let (width, height) = interleave_resolution_snap(req_width, req_height);

    let source_asset_ids: Vec<String> = job
        .payload
        .get("sourceAssetIds")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    // Upstream interleave defaults (examples/interleave/inference.py @238d6cf).
    let steps = advanced_int(advanced, "numInferenceSteps", 50, 1, 100) as usize;
    let cfg_scale = advanced_float(advanced, "guidanceScale", 4.0);
    let img_cfg_scale = advanced_float(advanced, "imageGuidanceScale", 1.0);
    let timestep_shift = advanced_float(advanced, "timestepShift", 3.0);
    let max_new_tokens = advanced_int(advanced, "maxNewTokens", 2048, 64, 8192) as usize;
    // Non-Think by default: the document is the deliverable, so skip the chain-of-thought.
    let think_mode = advanced
        .get("thinkMode")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let system_message = advanced
        .get("systemMessage")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| INTERLEAVE_SYSTEM_MESSAGE.to_owned());
    let seed = resolve_seed(&request, 0);

    let weights_dir = resolve_weights_dir(&request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SenseNova-U1 weights not found".to_owned()))?;

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    tokio::fs::create_dir_all(project_path.join("assets").join("documents")).await?;
    tokio::fs::create_dir_all(project_path.join("assets").join("images")).await?;
    let backend = backend_label(&settings.gpu_id).to_owned();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing interleaved document.",
            None,
            &backend,
        ),
    )
    .await?;

    // Decode the optional source images on the async side (Send moves into the blocking task).
    let mut input_images = Vec::with_capacity(source_asset_ids.len());
    for asset_id in &source_asset_ids {
        input_images.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            asset_id,
            &project_path,
        )?);
    }

    // The engine `interleave_gen` is a single uninterruptible rollout, so check for cancellation
    // before launching it. `check_cancel` marks the job canceled + returns `Canceled` on a cancel;
    // a real API error still propagates.
    match check_cancel(api, &job.id, "Interleaved generation canceled by user.").await {
        Ok(()) => {}
        Err(WorkerError::Canceled(_)) => return Ok(()),
        Err(other) => return Err(other),
    }

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.45,
            "Composing interleaved document.",
            None,
            &backend,
        ),
    )
    .await?;
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;

    // The engine's `interleave_gen` is a single synchronous rollout with no per-segment callback,
    // so (like the Python adapter's single `interleave_gen` call) the document streams as one final
    // result rather than incrementally. The whole rollout runs on a blocking thread; the decoded
    // images come back as Send `Image`s for asset writing on the async side.
    let job_id = job.id.clone();
    let prompt_for_gen = prompt.clone();
    let (generated_text, images) =
        tokio::task::spawn_blocking(move || -> WorkerResult<(String, Vec<Image>)> {
            emit_load_event("image_pipeline_load_start", &job_id, "sensenova_u1_8b", 0);
            let (model, tokenizer) = load_sensenova_model(&weights_dir)?;
            emit_load_event(
                "image_pipeline_load_complete",
                &job_id,
                "sensenova_u1_8b",
                0,
            );

            // Source images: [3,H,W] in [0,1], 32-aligned. Bounds mirror the torch
            // `interleave_gen` (`load_image_native` min 512², max min(2048², 4096²/n)).
            let n = input_images.len().max(1) as i64;
            let max_pixels = (2048 * 2048).min((4096 * 4096) / n);
            let mut input_arrays = Vec::with_capacity(input_images.len());
            for image in &input_images {
                input_arrays.push(image_to_chw01(image, 512 * 512, max_pixels)?);
            }

            let opts = T2iOptions {
                cfg_scale,
                img_cfg_scale,
                num_steps: steps,
                timestep_shift,
                seed: seed as u64,
                think_mode,
                ..Default::default()
            };
            let out = model
                .interleave_gen(
                    &tokenizer,
                    &prompt_for_gen,
                    &input_arrays,
                    width,
                    height,
                    &opts,
                    &system_message,
                    max_new_tokens,
                    max_images,
                    None,
                )
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!("SenseNova interleave failed: {error}"))
                })?;
            // The generated images are model-space [-1,1] `[1,3,H,W]` arrays — decode each to RGB8
            // exactly as the `Generator` image path does (`decoded_to_image`).
            let mut decoded = Vec::with_capacity(out.images.len());
            for image in &out.images {
                decoded.push(decoded_to_image(image).map_err(|error| {
                    WorkerError::InvalidPayload(format!("SenseNova interleave decode: {error}"))
                })?);
            }
            Ok((out.text, decoded))
        })
        .await
        .map_err(|error| WorkerError::InvalidPayload(format!("interleave task join: {error}")))??;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving interleaved document.",
            None,
            &backend,
        ),
    )
    .await?;

    let result = write_interleaved_document(
        &request,
        job,
        &project_path,
        &prompt,
        &model_id,
        seed,
        max_images,
        width,
        height,
        steps,
        cfg_scale,
        img_cfg_scale,
        timestep_shift,
        max_new_tokens,
        think_mode,
        &generated_text,
        images,
    )?;

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Interleaved document ready.",
            Some(result),
            &backend,
        ),
    )
    .await?;
    Ok(())
}

/// Off macOS the in-process engine is unavailable; `image_interleave` is served by the Python torch
/// worker (the `mlx` worker — the only one advertising this capability — is macOS-only).
#[cfg(not(target_os = "macos"))]
pub(crate) async fn run_interleave_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "image_interleave runs on the macOS MLX worker or the Python torch worker, not this Rust worker"
            .to_owned(),
    ))
}

// ---------------------------------------------------------------------------
// Document assembly (macOS): write the generated images as ordinary image assets, split the model
// text on `<image>` markers into ordered segments, then write the `InterleavedDocument` body + the
// `document` asset fact. Mirrors the Python `_write_interleaved_document` / `_build_interleaved_segments`.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn write_interleaved_document(
    request: &ImageRequest,
    job: &JobSnapshot,
    project_path: &Path,
    prompt: &str,
    model_id: &str,
    seed: i64,
    max_images: usize,
    width: i32,
    height: i32,
    steps: usize,
    cfg_scale: f32,
    img_cfg_scale: f32,
    timestep_shift: f32,
    max_new_tokens: usize,
    think_mode: bool,
    generated_text: &str,
    images: Vec<Image>,
) -> WorkerResult<JsonObject> {
    let resolution = format!("{width}x{height}");
    // Flat telemetry mirroring the Python `raw_settings` (advanced overlay + resolved knobs).
    let mut raw_settings = request.advanced.clone();
    raw_settings.insert("realModelInference".to_owned(), Value::Bool(true));
    raw_settings.insert("repo".to_owned(), Value::String(model_repo_for(request)));
    raw_settings.insert("numInferenceSteps".to_owned(), json!(steps));
    raw_settings.insert("guidanceScale".to_owned(), json!(cfg_scale));
    raw_settings.insert("imageGuidanceScale".to_owned(), json!(img_cfg_scale));
    raw_settings.insert("timestepShift".to_owned(), json!(timestep_shift));
    raw_settings.insert("maxImages".to_owned(), json!(max_images));
    raw_settings.insert("maxNewTokens".to_owned(), json!(max_new_tokens));
    raw_settings.insert("thinkMode".to_owned(), Value::Bool(think_mode));
    raw_settings.insert("resolution".to_owned(), Value::String(resolution.clone()));

    // Generated images persist as ordinary image assets — the worker saves the PNG + reports facts,
    // and the Rust API builds + indexes their sidecars. The document references them in order.
    let plan = ImagePlan::with_count(request, images.len() as u32);
    let mut image_raw_settings = raw_settings.clone();
    image_raw_settings.insert("interleaved".to_owned(), Value::Bool(true));
    let mut image_writes: Vec<Value> = Vec::with_capacity(images.len());
    for (index, image) in images.into_iter().enumerate() {
        let fact = write_image_asset(
            &plan,
            index,
            seed,
            image.width,
            image.height,
            image.pixels,
            SENSENOVA_ADAPTER,
            image_raw_settings.clone(),
            project_path,
        )?;
        image_writes.push(Value::Object(fact));
    }
    let image_asset_ids: Vec<String> = image_writes
        .iter()
        .filter_map(|write| write.get("assetId").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();

    let segments = build_interleaved_segments(generated_text, &image_writes);

    let created_at = crate::now_rfc3339();
    let document_id = format!("doc_{}", Uuid::new_v4().simple());
    let media_rel = format!("assets/documents/{document_id}.json");
    let document_body = json!({
        "schemaVersion": 1,
        "id": document_id,
        "projectId": request.project_id,
        "jobId": job.id,
        "model": model_id,
        "prompt": prompt,
        "createdAt": created_at,
        "segments": segments,
    });
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_path = media_path.with_extension("tmp.json");
    std::fs::write(
        &temp_path,
        serde_json::to_vec_pretty(&document_body)
            .map_err(|error| WorkerError::InvalidPayload(format!("serialize document: {error}")))?,
    )?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let display_name: String = {
        let trimmed: String = prompt.chars().take(56).collect();
        if trimmed.trim().is_empty() {
            "Interleaved document".to_owned()
        } else {
            trimmed
        }
    };
    let asset_id = fresh_asset_id();
    let document_write = json!({
        "type": "document",
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "application/json",
        "displayName": display_name,
        "createdAt": created_at,
        "mode": "interleave",
        "model": model_id,
        "adapter": SENSENOVA_ADAPTER,
        "prompt": prompt,
        "negativePrompt": "",
        "seed": seed,
        "loras": [],
        "rawAdapterSettings": raw_settings,
        "maxImages": max_images,
        "resolution": resolution,
        "imageCount": image_asset_ids.len(),
        "parents": image_asset_ids,
    });

    let mut asset_writes = image_writes;
    asset_writes.push(document_write);
    let expected_count = asset_writes.len();

    Ok(json!({
        "documentId": document_id,
        "documentAssetId": asset_id,
        "imageAssetIds": image_asset_ids,
        "segments": segments,
        "model": model_id,
        "realModelInference": true,
        "generationSetId": plan.genset_id,
        "expectedCount": expected_count,
        "generationSet": plan.generation_set,
        "assetWrites": asset_writes,
    })
    .as_object()
    .cloned()
    .expect("json! object literal"))
}

/// Split the model output on its inline `<image>` markers and slot the generated image assets in
/// order: text[0], image[0], text[1], image[1], …. Mirrors the Python `_build_interleaved_segments`
/// (reads each image fact's `assetId` + `mediaPath`).
#[cfg(target_os = "macos")]
fn build_interleaved_segments(generated_text: &str, image_writes: &[Value]) -> Vec<Value> {
    let mut segments = Vec::new();
    for (index, part) in generated_text.split("<image>").enumerate() {
        let text = part.trim();
        if !text.is_empty() {
            segments.push(json!({ "type": "text", "text": text }));
        }
        if let Some(write) = image_writes.get(index) {
            segments.push(json!({
                "type": "image",
                "assetId": write.get("assetId").cloned().unwrap_or(Value::Null),
                "path": write.get("mediaPath").cloned().unwrap_or(Value::Null),
            }));
        }
    }
    segments
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Assemble the concrete unified `T2iModel` + tokenizer for a SenseNova-U1 snapshot, replicating the
/// engine's private `load_inner` from public re-exports. Loads dense bf16 with NO distill LoRA and
/// NO quantization — the understanding (VQA) + interleave paths use the full base model, exactly as
/// the torch adapter's `_load_model(distill_lora=None)`, keeping the VQA decode bit-identical.
#[cfg(target_os = "macos")]
fn load_sensenova_model(weights_dir: &Path) -> WorkerResult<(T2iModel, TextTokenizer)> {
    let cfg = NeoChatConfig::from_dir(weights_dir)
        .map_err(|error| WorkerError::InvalidPayload(format!("SenseNova-U1 config: {error}")))?;
    let weights = load_raw(weights_dir)
        .map_err(|error| WorkerError::InvalidPayload(format!("SenseNova-U1 weights: {error}")))?;
    let model = T2iModel::from_weights(&weights, &cfg).map_err(|error| {
        WorkerError::InvalidPayload(format!("SenseNova-U1 model build: {error}"))
    })?;
    let tokenizer = load_tokenizer(weights_dir)
        .map_err(|error| WorkerError::InvalidPayload(format!("SenseNova-U1 tokenizer: {error}")))?;
    Ok((model, tokenizer))
}

/// Decode an [`Image`] (RGB8 HWC) to a `[3,H,W]` f32 tensor in `[0,1]`, smart-resized to a
/// 32-aligned bucket within `[min_pixels, max_pixels]`. Replicates the engine's private
/// `image_to_chw01` (its `preprocess_image` ImageNet-normalizes internally, so this stays in
/// `[0,1]`). VQA passes the understanding budget; interleave passes the it2i source budget.
#[cfg(target_os = "macos")]
fn image_to_chw01(img: &Image, min_pixels: i64, max_pixels: i64) -> WorkerResult<Array> {
    let (in_w, in_h) = (img.width as i32, img.height as i32);
    let (out_h, out_w) = smart_resize(in_h, in_w, 32, min_pixels, max_pixels);
    let hwc = resize_bicubic_u8(
        &img.pixels,
        in_h as usize,
        in_w as usize,
        out_h as usize,
        out_w as usize,
    );
    let hwc = Array::from_slice(&hwc, &[out_h, out_w, 3]);
    let chw = hwc
        .transpose_axes(&[2, 0, 1])
        .map_err(|error| WorkerError::InvalidPayload(format!("image transpose: {error}")))?;
    divide(&chw, Array::from_f32(255.0))
        .map_err(|error| WorkerError::InvalidPayload(format!("image normalize: {error}")))
}

/// The SenseNova-U1 repo (manifest `repo` else the default), for the document telemetry.
#[cfg(target_os = "macos")]
fn model_repo_for(request: &ImageRequest) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("sensenova/SenseNova-U1-8B-MoT")
        .to_owned()
}

/// Snap a requested W×H to the nearest interleave bucket by aspect ratio in log-space (ties resolve
/// to the first bucket). Mirrors the Python `snap_to_aspect_bucket` over the same
/// `INTERLEAVE_RESOLUTIONS` table (priority order).
#[cfg(target_os = "macos")]
fn interleave_resolution_snap(width: i64, height: i64) -> (i32, i32) {
    let target = (width.max(1) as f64 / height.max(1) as f64).ln();
    let mut best = INTERLEAVE_RESOLUTIONS[0].1;
    let mut best_distance = f64::INFINITY;
    for &(_, (bucket_w, bucket_h)) in INTERLEAVE_RESOLUTIONS {
        let distance = (target - (bucket_w as f64 / bucket_h as f64).ln()).abs();
        if distance < best_distance {
            best_distance = distance;
            best = (bucket_w, bucket_h);
        }
    }
    best
}

/// Drop any `<think>…</think>` reasoning so only the answer is returned — removes complete think
/// blocks and any dangling/unclosed one (reasoning truncated by `max_new_tokens`). Mirrors the
/// Python `SenseNovaU1Adapter._strip_reasoning`. Used by the macOS VQA handler; also unit-tested
/// cross-platform (it is pure string logic), so it compiles under `test` off macOS too.
#[cfg(any(target_os = "macos", test))]
fn strip_reasoning(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            // Complete block: drop `<think>…</think>` and continue after it.
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            // Dangling/unclosed block (truncated by max_new_tokens): drop everything from
            // `<think>` on.
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_owned()
}

/// `safe_int` over a top-level payload field: parse (int / float / numeric string) else `default`,
/// then clamp to `[lo, hi]`. Used by the macOS handlers; also unit-tested cross-platform.
#[cfg(any(target_os = "macos", test))]
fn payload_int(payload: &JsonObject, key: &str, default: i64, lo: i64, hi: i64) -> i64 {
    payload
        .get(key)
        .and_then(json_to_i64)
        .unwrap_or(default)
        .clamp(lo, hi)
}

/// `safe_int` over an `advanced` field (same parse/clamp as [`payload_int`]).
#[cfg(target_os = "macos")]
fn advanced_int(advanced: &JsonObject, key: &str, default: i64, lo: i64, hi: i64) -> i64 {
    payload_int(advanced, key, default, lo, hi)
}

/// `_advanced_float`: parse an `advanced` field as f32 else `default`.
#[cfg(target_os = "macos")]
fn advanced_float(advanced: &JsonObject, key: &str, default: f32) -> f32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

#[cfg(any(target_os = "macos", test))]
fn json_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|float| float as i64))
        .or_else(|| value.as_str()?.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_reasoning_removes_think_blocks() {
        assert_eq!(strip_reasoning("<think>reasoning</think>answer"), "answer");
        assert_eq!(
            strip_reasoning("before <think>mid</think> after"),
            "before  after"
        );
        // Dangling/unclosed block (truncated by max_new_tokens) is dropped entirely.
        assert_eq!(strip_reasoning("answer<think>cut off"), "answer");
        // No think block: returned trimmed, unchanged.
        assert_eq!(strip_reasoning("  plain answer  "), "plain answer");
    }

    #[test]
    fn payload_int_parses_clamps_and_defaults() {
        let map = json!({ "a": 500, "b": "1024", "c": 3.0, "d": "bad" })
            .as_object()
            .cloned()
            .unwrap();
        assert_eq!(payload_int(&map, "a", 256, 16, 2048), 500);
        assert_eq!(payload_int(&map, "b", 256, 16, 2048), 1024);
        assert_eq!(
            payload_int(&map, "c", 256, 16, 2048),
            16,
            "3 clamps up to lo"
        );
        assert_eq!(
            payload_int(&map, "d", 256, 16, 2048),
            256,
            "unparseable → default"
        );
        assert_eq!(payload_int(&map, "missing", 768, 16, 2048), 768);
        assert_eq!(
            payload_int(&map, "a", 256, 16, 400),
            400,
            "500 clamps to hi"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn interleave_resolution_snaps_to_aspect_bucket() {
        // Exact 16:9 → its bucket.
        assert_eq!(interleave_resolution_snap(2048, 1152), (2048, 1152));
        // Square-ish → 1:1.
        assert_eq!(interleave_resolution_snap(1000, 1000), (1536, 1536));
        // Tall portrait → 9:16.
        assert_eq!(interleave_resolution_snap(1152, 2048), (1152, 2048));
        // Extreme wide → 3:1.
        assert_eq!(interleave_resolution_snap(3000, 1000), (2592, 864));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_segments_interleaves_text_and_images() {
        let writes = vec![
            json!({ "assetId": "asset_a", "mediaPath": "assets/images/g/a.png" }),
            json!({ "assetId": "asset_b", "mediaPath": "assets/images/g/b.png" }),
        ];
        let segments = build_interleaved_segments("intro<image>middle<image>end", &writes);
        assert_eq!(segments.len(), 5);
        assert_eq!(segments[0], json!({ "type": "text", "text": "intro" }));
        assert_eq!(
            segments[1],
            json!({ "type": "image", "assetId": "asset_a", "path": "assets/images/g/a.png" })
        );
        assert_eq!(segments[2], json!({ "type": "text", "text": "middle" }));
        assert_eq!(
            segments[3],
            json!({ "type": "image", "assetId": "asset_b", "path": "assets/images/g/b.png" })
        );
        assert_eq!(segments[4], json!({ "type": "text", "text": "end" }));
    }

    /// The HF-cache snapshot dir for a cached repo (test helper).
    #[cfg(target_os = "macos")]
    fn hf_snapshot(model_dir: &str) -> PathBuf {
        let home = std::env::var("HOME").expect("HOME set");
        std::fs::read_dir(
            PathBuf::from(home).join(format!(".cache/huggingface/hub/{model_dir}/snapshots")),
        )
        .expect("HF cache snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a snapshot dir")
    }

    /// A synthetic RGB8 gradient (test source image).
    #[cfg(target_os = "macos")]
    fn gradient_image(width: u32, height: u32) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push((x % 256) as u8);
                pixels.push((y % 256) as u8);
                pixels.push(((x + y) % 256) as u8);
            }
        }
        Image {
            width,
            height,
            pixels,
        }
    }

    /// Real-weights smoke: SenseNova-U1 VQA. Loads the dense base `T2iModel` (~35GB
    /// `sensenova/SenseNova-U1-8B-MoT`), preprocesses a synthetic image, and asserts the answer
    /// text is non-empty (post think-strip). Needs the HF cache + a Metal device; run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_vqa_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + Metal device"]
    fn sensenova_vqa_real_weights_answers_non_empty() {
        let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
        let (model, tokenizer) = load_sensenova_model(&snapshot).expect("load model");
        let image = gradient_image(512, 512);
        let pixel_values = image_to_chw01(&image, 256 * 256, 768 * 768).expect("preprocess");
        let answer = model
            .vqa(
                &tokenizer,
                "What colors appear in this image?",
                std::slice::from_ref(&pixel_values),
                64,
                Sampler::Greedy,
            )
            .expect("vqa");
        let answer = strip_reasoning(&answer);
        assert!(
            !answer.is_empty(),
            "VQA answer should be non-empty: {answer:?}"
        );
    }

    /// Real-weights smoke: SenseNova-U1 interleave. Loads the dense base `T2iModel`, runs a short
    /// think-mode interleave rollout (mirroring the engine's own real-weight test, which reliably
    /// emits an image), decodes the generated image(s), and asserts ≥1 image + a valid segment set
    /// with at least one image segment. Small 512² + 8 steps for speed (production buckets are
    /// 1536²+ / 50 steps). Run on demand:
    /// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_interleave_real_weights`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + Metal device"]
    fn sensenova_interleave_real_weights_produces_document() {
        let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
        let (model, tokenizer) = load_sensenova_model(&snapshot).expect("load model");
        let opts = T2iOptions {
            cfg_scale: 4.0,
            img_cfg_scale: 1.0,
            num_steps: 8,
            timestep_shift: 3.0,
            seed: 42,
            think_mode: true,
            ..Default::default()
        };
        // Budget mirrors the engine's own passing interleave real-weight test (512 new tokens,
        // generous max_images) so the think-mode rollout reliably reaches an `<img>`.
        let out = model
            .interleave_gen(
                &tokenizer,
                "Generate an illustration of a single red circle on a white background, then briefly describe it.",
                &[],
                512,
                512,
                &opts,
                INTERLEAVE_SYSTEM_MESSAGE,
                512,
                4,
                None,
            )
            .expect("interleave_gen");
        assert!(!out.images.is_empty(), "expected >= 1 generated image");
        let mut image_writes = Vec::new();
        for (index, image) in out.images.iter().enumerate() {
            let decoded = decoded_to_image(image).expect("decode");
            assert_eq!(
                decoded.pixels.len(),
                (decoded.width * decoded.height * 3) as usize
            );
            image_writes.push(json!({
                "assetId": format!("asset_{index}"),
                "mediaPath": format!("assets/images/g/{index}.png"),
            }));
        }
        let segments = build_interleaved_segments(&out.text, &image_writes);
        assert!(
            segments.iter().any(|s| s["type"] == "image"),
            "document should contain >= 1 image segment: {segments:?}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_segments_trailing_image_without_marker_is_appended() {
        // One `<image>` marker but two images generated. Mirrors the Python splitter exactly:
        // split → ["only one", ""]; index 0 slots image[0] after the text, index 1 (empty text)
        // still slots image[1] because `index < len(image_writes)`. So the extra image trails.
        let writes = vec![
            json!({ "assetId": "asset_a", "mediaPath": "a.png" }),
            json!({ "assetId": "asset_b", "mediaPath": "b.png" }),
        ];
        let segments = build_interleaved_segments("only one<image>", &writes);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], json!({ "type": "text", "text": "only one" }));
        assert_eq!(segments[1]["assetId"], "asset_a");
        assert_eq!(segments[2]["assetId"], "asset_b");
    }
}
