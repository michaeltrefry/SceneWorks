// Candle (Windows/CUDA) Z-Image img2img / edit route (sc-6595, epic 5480) — pixel-conditioned editing
// on Z-Image-Turbo off-Mac via `candle_gen_z_image::ZImageEdit`. The candle sibling of the MLX z-image
// img2img path (the registered `z_image_turbo` generator's `Conditioning::Reference` route, driven by
// `resolve_zimage_edit_init` in zimage.rs). Both `z_image_edit` and `z_image_turbo` (mode `edit_image`)
// reach this one lane — two ids, one engine (the Turbo weights with a source-latent init).
//
// **Candle-only.** macOS keeps the MLX `z_image_turbo` registry generator (img2img via the engine's
// `Reference` conditioning); the candle `z_image_turbo` descriptor is txt2img-only, so the candle
// `ZImageEdit` is a bespoke provider — this whole file is gated to the Windows/CUDA candle build (the
// `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it
// shares that module's imports (`ImageRequest`/`Settings`/`WorkerResult`/`advanced`/`load_reference_image`/
// `fit_engine_image`/`huggingface_snapshot_dir`/`resolve_app_managed_model_dir`/`resolve_seed`/
// `start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/`gen_core`/… all in scope).

/// img2img strength default — the worker's `advanced.strength` default (`resolve_zimage_edit_init`,
/// torch `ZImageImg2ImgPipeline` 0.6). With Z-Image's `init_time_step`, higher ⇒ closer to the source
/// (the structure-preservation convention; the inverse of the SDXL knob — see `candle_gen_z_image::edit`).
const ZIMAGE_EDIT_CANDLE_DEFAULT_STRENGTH: f32 = 0.6;
/// Denoise-steps default — the distilled 4-step Turbo schedule (the txt2img / MLX z-image default).
const ZIMAGE_EDIT_CANDLE_DEFAULT_STEPS: u32 = 4;
/// The Z-Image base diffusers repo when the manifest omits `repo`.
const ZIMAGE_EDIT_CANDLE_DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";
/// The adapter/engine id recorded on candle Z-Image edit assets + telemetry (distinct from the txt2img
/// `candle_z_image` and the `candle_zimage_control` lanes).
const ZIMAGE_EDIT_CANDLE_ENGINE: &str = "candle_zimage_edit";

/// Model ids the candle Z-Image edit route accepts: the txt2img `z_image_turbo` (in `edit_image` mode)
/// and the dedicated `z_image_edit` id — both drive the Turbo weights' img2img path.
fn is_zimage_edit_candle_model(model: &str) -> bool {
    matches!(model, "z_image_turbo" | "z_image_edit")
}

/// Resolve the Z-Image base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the
/// HF cache snapshot for the manifest `repo` (default `Tongyi-MAI/Z-Image-Turbo`). `None` ⇒ not present
/// locally (the job is not candle-runnable, falls through to torch). Mirrors `resolve_zimage_control_base`.
fn resolve_zimage_edit_candle_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Z-Image edit modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(ZIMAGE_EDIT_CANDLE_DEFAULT_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible Z-Image edit job: a z-image-family `edit_image` job with a source
/// image whose base resolves locally. Mirrors `jobs_store::zimage_edit_candle_eligible` so the worker and
/// router agree.
fn zimage_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_zimage_edit_candle_model(&request.model)
        && request.mode == "edit_image"
        && non_empty(&request.source_asset_id)
        && matches!(
            resolve_zimage_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (4, distilled).
fn zimage_edit_candle_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or(ZIMAGE_EDIT_CANDLE_DEFAULT_STEPS)
}

/// Load the source asset (required for an edit) — mirrors the MLX `resolve_zimage_edit_init` source load.
fn load_zimage_edit_source(
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
            WorkerError::InvalidPayload("Z-Image edit requires a source image".to_owned())
        })?;
    load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )
}

/// Flat telemetry recorded on candle Z-Image edit assets. No guidance — Z-Image-Turbo is
/// guidance-distilled.
fn zimage_edit_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    strength: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("strength".to_owned(), json!(strength));
    raw.insert("mode".to_owned(), Value::String("edit_image".to_owned()));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(ZIMAGE_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Z-Image img2img generation: resolve the source + base on the async side, fit the source to
/// the render size (honoring `fit_mode`), then load `ZImageEdit` once + generate each image on the blocking
/// thread. `request.count` images, each its own seed, all editing the same source. Z-Image-Turbo is
/// distilled (no CFG / negative prompt), so the request carries no guidance. `ZImageEdit::generate` takes
/// `&self`, so the per-item closure needs no `mut`. Reuses [`consume_gen_events`].
async fn generate_candle_zimage_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let base = resolve_zimage_edit_candle_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Z-Image base (Z-Image-Turbo) weights not found".to_owned())
    })?;

    let (width, height) = (request.width, request.height);
    // Load + fit the source to the render geometry honoring `fit_mode` (crop/pad/stretch — the provider
    // also resizes internally, but pre-fitting here is what avoids stretching an off-aspect source).
    let source = load_zimage_edit_source(request, project_path, settings)?;
    let source = fit_engine_image(source, width, height, &request.fit_mode)?;

    let steps = zimage_edit_candle_steps(request);
    let strength = advanced::f32_clamped(
        &request.advanced,
        "strength",
        ZIMAGE_EDIT_CANDLE_DEFAULT_STRENGTH,
        0.05..=1.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(ZIMAGE_EDIT_CANDLE_DEFAULT_REPO)
        .to_owned();
    let raw_settings = zimage_edit_candle_raw_settings(request, &repo, steps, strength);

    // Per-image work items: (seed, prompt) — `request.count` edits of the same source.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "zimage_edit",
        0,
        move || {
            let model = ZImageEdit::load(&ZImageEditPaths { base }).map_err(|error| {
                WorkerError::Engine(format!("Z-Image edit load failed: {error}"))
            })?;
            Ok((model, source))
        },
        move |(model, source), tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = ZImageEditRequest {
                    prompt,
                    width,
                    height,
                    steps: steps as usize,
                    strength,
                    seed: seed as u64,
                    cancel: cancel.clone(),
                };
                let out = match model.generate(&req, &source, &mut *on_progress) {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Z-Image edit generation failed: {error}"
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
        ZIMAGE_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
