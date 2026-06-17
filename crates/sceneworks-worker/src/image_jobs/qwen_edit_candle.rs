// Candle (Windows/CUDA) Qwen-Image-Edit route (sc-5487, epic 5480) — reference-conditioned image
// editing on the Qwen-Image-Edit family off-Mac via `candle_gen_qwen_image::QwenEdit`. The reference
// + edit prompt go through the Qwen2.5-VL vision-language encoder, the reference is VAE-encoded into
// the transformer's dual-latent sequence, and the MMDiT denoises a reference-respecting edit. Before
// this an off-Mac `edit_image` job on a Qwen-Image-Edit model fell back to the Python torch worker.
//
// **Candle-only.** macOS keeps the MLX `qwen_image_edit` registry path (qwen.rs). The candle `QwenEdit`
// is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build (the `include!`
// in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it shares that
// module's imports (ImageRequest/Settings/WorkerResult/`load_reference_image`/`huggingface_snapshot_dir`/
// `resolve_app_managed_model_dir`/`resolve_seed`/`start_gen_stream`/`drive_gen_items`/
// `consume_gen_events`/`non_empty`/`gen_core`/… all in scope).
//
// Qwen-Image-Edit is a dual-latent reference concat (NOT strength-img2img + mask): the source is the
// reference, the prompt is the instruction. So this lane handles `edit_image` + `sourceAssetId` (no
// sub-modes / inpaint / outpaint — that masked shape is the SDXL edit lane's). The provider
// condition-resizes the reference internally (~384²), so — unlike the FLUX.2 lane — the source is NOT
// pre-fit to the render size.

/// Qwen-Image-Edit denoise steps default (the production, non-distilled variants).
const QWEN_EDIT_CANDLE_DEFAULT_STEPS: u32 = 30;
/// True-CFG guidance default.
const QWEN_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle Qwen-Image-Edit assets + telemetry.
const QWEN_EDIT_CANDLE_ENGINE: &str = "candle_qwen_edit";
/// Default Qwen-Image-Edit base repo when the manifest omits `repo`.
const QWEN_EDIT_CANDLE_DEFAULT_REPO: &str = "Qwen/Qwen-Image-Edit";

/// Qwen-Image-Edit model ids the candle edit route accepts. All non-lightning variants map to the
/// single edit engine model (the architecture is identical; `-2511` only flips `zero_cond_t`, which
/// `QwenEdit` auto-detects from `transformer/config.json`). The `-2511_lightning` distill needs the
/// 4-step LoRA — `QwenEdit` has no LoRA support — so it stays on the MLX / torch path.
fn is_qwen_edit_candle_model(model: &str) -> bool {
    matches!(
        model,
        "qwen_image_edit" | "qwen_image_edit_2509" | "qwen_image_edit_2511"
    )
}

/// True when this is a candle-eligible Qwen edit job: a Qwen-Image-Edit `edit_image` job with a source
/// image. Mirrors `jobs_store::qwen_edit_candle_eligible` so the worker and router agree on the lane.
fn qwen_edit_candle_mode(request: &ImageRequest) -> bool {
    request.mode == "edit_image" && non_empty(&request.source_asset_id)
}

/// The Qwen-Image-Edit base repo for this request: manifest `repo` else the default.
fn qwen_edit_candle_repo(request: &ImageRequest) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(QWEN_EDIT_CANDLE_DEFAULT_REPO)
        .to_owned()
}

/// Resolve the Qwen-Image-Edit base snapshot: an explicit `modelPath` dir (advanced or manifest) wins,
/// else the HF cache snapshot for the manifest `repo`. `None` means the base is not present locally, so
/// the job is not candle-runnable. Mirrors `resolve_flux2_edit_candle_base`.
fn resolve_qwen_edit_candle_base(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = request
        .advanced
        .get("modelPath")
        .or_else(|| request.model_manifest_entry.get("modelPath"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
    {
        return resolve_app_managed_model_dir(settings, &path, "Qwen edit modelPath").map(Some);
    }
    let repo = qwen_edit_candle_repo(request);
    Ok(huggingface_snapshot_dir(&settings.data_dir, &repo))
}

/// True when this is a candle-eligible Qwen edit job (a Qwen-Image-Edit `edit_image` job with a source)
/// whose base resolves locally. Mirrors `jobs_store::qwen_edit_candle_eligible` (minus the weight-
/// resolve check).
fn qwen_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_qwen_edit_candle_model(&request.model)
        && qwen_edit_candle_mode(request)
        && matches!(resolve_qwen_edit_candle_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (30).
fn qwen_edit_candle_steps(request: &ImageRequest) -> u32 {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    };
    request
        .advanced
        .get("steps")
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get("steps").and_then(parse))
        .map(|steps| steps.clamp(1, 50) as u32)
        .unwrap_or(QWEN_EDIT_CANDLE_DEFAULT_STEPS)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (4.0), clamped.
fn qwen_edit_candle_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(QWEN_EDIT_CANDLE_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Flat telemetry recorded on candle Qwen-Image-Edit assets.
fn qwen_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("referenceCount".to_owned(), json!(1));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(QWEN_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Load the Qwen edit source asset (the `sourceAssetId` is required) as an engine [`Image`].
fn load_qwen_edit_source(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
) -> WorkerResult<Image> {
    let source_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Qwen edit requires a source image".to_owned())
        })?;
    load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )
}

/// Real candle Qwen-Image-Edit generation: resolve the source + base on the async side, then load
/// `QwenEdit` once + generate each image on the blocking thread. The provider condition-resizes the
/// reference internally, so the source is passed as-is (no render-size pre-fit). `request.count` edits
/// of the same source, each its own seed. `generate` takes `&self`, so the per-item closure needs no
/// `mut`. Reuses [`consume_gen_events`].
async fn generate_candle_qwen_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let qwen_base = resolve_qwen_edit_candle_base(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen-Image-Edit base not found".to_owned()))?;
    if !qwen_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "Qwen edit requires edit_image mode + a source image".to_owned(),
        ));
    }
    let (width, height) = (request.width, request.height);
    let reference = load_qwen_edit_source(request, project_path, settings)?;

    let steps = qwen_edit_candle_steps(request);
    let guidance = qwen_edit_candle_guidance(request);
    let repo = qwen_edit_candle_repo(request);
    let raw_settings = qwen_edit_candle_raw_settings(request, &repo, steps, guidance);

    // Per-image work items: (seed, prompt) — `request.count` edits of the same source.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative = request.negative_prompt.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "qwen_edit",
        0,
        move || {
            let model = QwenEdit::load(&QwenEditPaths { root: qwen_base })
                .map_err(|error| WorkerError::Engine(format!("Qwen edit load failed: {error}")))?;
            Ok((model, reference))
        },
        move |(model, reference), tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = QwenEditRequest {
                    prompt,
                    negative: negative.clone(),
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    seed: seed as u64,
                    cancel: cancel.clone(),
                };
                let result = model.generate(&req, std::slice::from_ref(&reference), &mut *on_progress);
                let out = match result {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Qwen edit generation failed: {error}"
                        )));
                    }
                };
                Ok(Some((seed, out.width, out.height, out.pixels)))
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        QWEN_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
