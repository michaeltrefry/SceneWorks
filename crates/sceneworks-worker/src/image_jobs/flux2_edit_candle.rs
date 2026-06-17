// Candle (Windows/CUDA) FLUX.2-klein image-edit route (sc-5487, epic 5480) — Kontext-style reference-
// conditioned editing on FLUX.2-klein off-Mac via `candle_gen_flux2::Flux2Edit`. FLUX.2-klein has no
// torch path (it is diffusers/MLX-only), so before this an off-Mac `edit_image` job on klein had no
// real lane; this routes it to candle instead of the torch fallback.
//
// **Candle-only.** macOS keeps the MLX FLUX.2 edit path (flux2.rs `generate_flux2_edit_stream`); the
// candle `Flux2Edit` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build
// (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so
// it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`resolve_app_managed_model_dir`/`resolve_seed`/`start_gen_stream`/
// `drive_gen_items`/`consume_gen_events`/`non_empty`/`gen_core`/… all in scope).
//
// FLUX.2 edit is a single-reference (or multi-reference) token concat, NOT a strength-based img2img +
// mask: the source is the reference, the prompt is the instruction. So this lane has no sub-modes /
// inpaint / outpaint (unlike the SDXL edit lane) — it handles `edit_image` + `sourceAssetId`.

/// FLUX.2-klein denoise steps default (distilled klein generates in 4).
const FLUX2_EDIT_CANDLE_DEFAULT_STEPS: u32 = 4;
/// Guidance default — distilled klein runs CFG-free at 1.0 (a single forward; >1.0 adds a negative pass).
const FLUX2_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 1.0;
/// The adapter/engine id recorded on candle FLUX.2 edit assets + telemetry (distinct from the txt2img
/// `candle_flux2` lane).
const FLUX2_EDIT_CANDLE_ENGINE: &str = "candle_flux2_edit";
/// Default FLUX.2-klein base repo when the manifest omits `repo`.
const FLUX2_EDIT_CANDLE_DEFAULT_REPO: &str = "black-forest-labs/FLUX.2-klein-9B";

/// FLUX.2-klein model ids the candle edit route accepts (the base 9b + true_v2 share the edit variant;
/// the -kv distill needs the reference-K/V cache port and stays on the MLX/torch path for now, as does
/// `flux2_dev` — no klein-only candle crate covers them yet).
fn is_flux2_edit_candle_model(model: &str) -> bool {
    matches!(model, "flux2_klein_9b" | "flux2_klein_9b_true_v2")
}

/// True when this is a candle-eligible FLUX.2 edit job: a klein-family `edit_image` job with a source
/// image. Mirrors `jobs_store::flux2_edit_candle_eligible` so the worker and router agree on the lane
/// boundary.
fn flux2_edit_candle_mode(request: &ImageRequest) -> bool {
    request.mode == "edit_image" && non_empty(&request.source_asset_id)
}

/// Resolve the FLUX.2-klein base snapshot: an explicit `modelPath` dir (advanced or manifest) wins, else
/// the HF cache snapshot for the manifest `repo` (default `FLUX2_EDIT_CANDLE_DEFAULT_REPO`). `None` means
/// the base is not present locally, so the job is not candle-runnable. Mirrors
/// `resolve_sdxl_edit_candle_base`.
fn resolve_flux2_edit_candle_base(
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
        return resolve_app_managed_model_dir(settings, &path, "FLUX.2 edit modelPath").map(Some);
    }
    let repo = flux2_edit_candle_repo(request);
    Ok(huggingface_snapshot_dir(&settings.data_dir, &repo))
}

/// The FLUX.2-klein base repo for this request: manifest `repo` else the klein default.
fn flux2_edit_candle_repo(request: &ImageRequest) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(FLUX2_EDIT_CANDLE_DEFAULT_REPO)
        .to_owned()
}

/// True when this is a candle-eligible FLUX.2 edit job (a klein-family `edit_image` job with a source)
/// whose base resolves locally. Mirrors `jobs_store::flux2_edit_candle_eligible` (minus the weight-
/// resolve check).
fn flux2_edit_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux2_edit_candle_model(&request.model)
        && flux2_edit_candle_mode(request)
        && matches!(
            resolve_flux2_edit_candle_base(request, settings),
            Ok(Some(_))
        )
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (4).
fn flux2_edit_candle_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or(FLUX2_EDIT_CANDLE_DEFAULT_STEPS)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (1.0), clamped.
fn flux2_edit_candle_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(FLUX2_EDIT_CANDLE_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Resize an RGB image to exactly `width`×`height` honoring `mode` without distorting it (the candle
/// FLUX.2 twin of the macOS `fit_rgb`): `crop` covers + center-crops, `pad`/`outpaint` contain +
/// letterbox on black, `stretch` is the legacy non-aspect resize. The Image-Edit source is pre-fitted so
/// an off-aspect edit doesn't stretch; the provider re-resizes to the render size internally.
fn flux2_edit_fit_rgb(
    source: &image::RgbImage,
    width: u32,
    height: u32,
    mode: &str,
) -> image::RgbImage {
    use image::imageops::FilterType::Lanczos3;
    let width = width.max(1);
    let height = height.max(1);
    let (src_w, src_h) = (source.width(), source.height());
    match mode {
        "stretch" => image::imageops::resize(source, width, height, Lanczos3),
        "crop" => {
            let ratio = (width as f32 / src_w as f32).max(height as f32 / src_h as f32);
            let new_w = width.max((src_w as f32 * ratio).ceil() as u32);
            let new_h = height.max((src_h as f32 * ratio).ceil() as u32);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let left = (new_w - width) / 2;
            let top = (new_h - height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
        // "pad" / "outpaint": contain + center on a black canvas (letterbox).
        _ => {
            let (new_w, new_h, left, top) =
                gen_core::imageops::contain_box(src_w, src_h, width, height);
            let resized = image::imageops::resize(source, new_w.max(1), new_h.max(1), Lanczos3);
            let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
            canvas
        }
    }
}

/// Fit an engine [`Image`] (RGB8) to `width`×`height` by `mode` via [`flux2_edit_fit_rgb`].
fn flux2_edit_fit_image(source: Image, width: u32, height: u32, mode: &str) -> WorkerResult<Image> {
    let rgb =
        image::RgbImage::from_raw(source.width, source.height, source.pixels).ok_or_else(|| {
            WorkerError::InvalidPayload("FLUX.2 edit image buffer size mismatch".to_owned())
        })?;
    let fitted = flux2_edit_fit_rgb(&rgb, width, height, mode);
    Ok(Image {
        width: fitted.width(),
        height: fitted.height(),
        pixels: fitted.into_raw(),
    })
}

/// Flat telemetry recorded on candle FLUX.2 edit assets (parity with the macOS FLUX.2 edit recipe keys).
fn flux2_edit_candle_raw_settings(
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
        Value::String(FLUX2_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Load the FLUX.2 edit source asset (the `sourceAssetId` is required) as an engine [`Image`].
fn load_flux2_edit_source(
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
            WorkerError::InvalidPayload("FLUX.2 edit requires a source image".to_owned())
        })?;
    load_reference_image(
        &settings.data_dir,
        &request.project_id,
        source_id,
        project_path,
    )
}

/// Real candle FLUX.2-klein edit generation: resolve the source + base on the async side, pre-fit the
/// source to the render geometry, then load `Flux2Edit` once + generate each image on the blocking
/// thread. `request.count` edits of the same source, each its own seed. `generate` takes `&self`, so the
/// per-item closure needs no `mut`. Reuses [`consume_gen_events`].
async fn generate_candle_flux2_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let flux2_base = resolve_flux2_edit_candle_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("FLUX.2-klein base not found".to_owned())
    })?;
    if !flux2_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires edit_image mode + a source image".to_owned(),
        ));
    }
    let (width, height) = (request.width, request.height);
    let source = load_flux2_edit_source(request, project_path, settings)?;
    // Pre-fit the Image-Edit source to W×H (crop / pad / outpaint→pad) so an off-aspect edit doesn't
    // stretch; `stretch` keeps the legacy non-aspect resize. The provider re-resizes internally.
    let reference = if request.fit_mode == "stretch" {
        source
    } else {
        flux2_edit_fit_image(source, width, height, &request.fit_mode)?
    };

    let steps = flux2_edit_candle_steps(request);
    let guidance = flux2_edit_candle_guidance(request);
    let repo = flux2_edit_candle_repo(request);
    let raw_settings = flux2_edit_candle_raw_settings(request, &repo, steps, guidance);

    // Per-image work items: (seed, prompt) — `request.count` edits of the same source.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative = request.negative_prompt.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "flux2_edit",
        0,
        move || {
            let model = Flux2Edit::load(&Flux2EditPaths { root: flux2_base })
                .map_err(|error| WorkerError::Engine(format!("FLUX.2 edit load failed: {error}")))?;
            Ok((model, reference))
        },
        move |(model, reference), tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = Flux2EditRequest {
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
                            "FLUX.2 edit generation failed: {error}"
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
        FLUX2_EDIT_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
