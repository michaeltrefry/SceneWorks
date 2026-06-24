// Candle (Windows/CUDA) FLUX.2 image-edit route (sc-5487 klein, epic 5480; sc-7736 dev, epic 6564) —
// Kontext-style reference-conditioned editing off-Mac via `candle_gen_flux2::Flux2Edit`. FLUX.2-klein has
// no torch path (it is diffusers/MLX-only), so before this an off-Mac `edit_image` job on klein had no
// real lane; this routes it to candle instead of the torch fallback. **sc-7736** generalizes the lane to
// the 32B **dev** flagship (`flux2_dev`): the same DiT token-concat edit, loaded via the Q4 CPU-stage →
// quantize-onto-GPU path (`Flux2Edit::load_dev`) with embedded distilled guidance (no negative pass).
//
// **Candle-only.** macOS keeps the MLX FLUX.2 edit path (flux2.rs `generate_flux2_edit_stream`); the
// candle `Flux2Edit` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build
// (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so
// it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`resolve_app_managed_model_dir`/`resolve_quant`/`resolve_seed`/
// `start_gen_stream`/`drive_gen_items`/`consume_gen_events`/`non_empty`/`gen_core`/… all in scope).
//
// FLUX.2 edit is a single-reference (or multi-reference) token concat, NOT a strength-based img2img +
// mask: the source is the reference, the prompt is the instruction. So this lane has no sub-modes /
// inpaint / outpaint (unlike the SDXL edit lane) — it handles `edit_image` + `sourceAssetId` (plus the
// multi-image picker's plural `referenceAssetIds`, sc-6211 parity).

/// FLUX.2-klein denoise steps default (distilled klein generates in 4).
const FLUX2_EDIT_CANDLE_DEFAULT_STEPS: u32 = 4;
/// Guidance default — distilled klein runs CFG-free at 1.0 (a single forward; >1.0 adds a negative pass).
const FLUX2_EDIT_CANDLE_DEFAULT_GUIDANCE: f32 = 1.0;
/// FLUX.2-dev denoise steps default (the guidance-distilled 32B; FLUX.1-dev pattern, ~28 steps).
const FLUX2_EDIT_CANDLE_DEV_STEPS: u32 = 28;
/// FLUX.2-dev embedded-guidance default (distilled scalar, NOT true-CFG — no negative pass).
const FLUX2_EDIT_CANDLE_DEV_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle FLUX.2 edit assets + telemetry (distinct from the txt2img
/// `candle_flux2` lane). Shared by klein + dev edit (the dev variant is the same edit surface).
const FLUX2_EDIT_CANDLE_ENGINE: &str = "candle_flux2_edit";
/// Default FLUX.2-klein base repo when the manifest omits `repo`.
const FLUX2_EDIT_CANDLE_DEFAULT_REPO: &str = "black-forest-labs/FLUX.2-klein-9B";
/// Default FLUX.2-dev base repo (the 32B flagship; sc-7460/7736). The candle lane loads the dense
/// diffusers snapshot and Q4-quantizes it at load (no install-time packed convert off-Mac).
const FLUX2_EDIT_CANDLE_DEV_REPO: &str = "black-forest-labs/FLUX.2-dev";
/// Cap on references fed to a single FLUX.2 edit (the multi-image picker, sc-6211): the dev edit is
/// activation-bound, so cap at the engine's validated native fan-out (parity with the MLX
/// `MAX_EDIT_REFERENCES`).
const FLUX2_EDIT_CANDLE_MAX_REFERENCES: usize = 5;

/// True when this is the FLUX.2 **dev** edit variant (`flux2_dev`): the 32B flagship that loads via the
/// Q4 quantize-onto-GPU path with embedded distilled guidance, vs the dense klein family.
fn is_flux2_edit_candle_dev(model: &str) -> bool {
    model == "flux2_dev"
}

/// FLUX.2 model ids the candle edit route accepts: the klein base 9b + true_v2 (which share the edit
/// variant) and the dev 32B flagship (sc-7736). The klein `-kv` distill needs the reference-K/V cache
/// port and stays on the MLX/torch path for now.
fn is_flux2_edit_candle_model(model: &str) -> bool {
    matches!(
        model,
        "flux2_klein_9b" | "flux2_klein_9b_true_v2" | "flux2_dev"
    )
}

/// True when this is a candle-eligible FLUX.2 edit job: a klein/dev `edit_image` job with a source
/// image. Mirrors `jobs_store::flux2_edit_candle_eligible` so the worker and router agree on the lane
/// boundary.
fn flux2_edit_candle_mode(request: &ImageRequest) -> bool {
    request.mode == "edit_image" && non_empty(&request.source_asset_id)
}

/// Resolve the FLUX.2 base snapshot: an explicit `modelPath` dir (advanced or manifest) wins, else the
/// HF cache snapshot for the manifest `repo` (default per family — klein vs dev). `None` means the base
/// is not present locally, so the job is not candle-runnable. Mirrors `resolve_sdxl_edit_candle_base`.
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

/// The FLUX.2 base repo for this request: manifest `repo` else the per-family default (dev vs klein).
fn flux2_edit_candle_repo(request: &ImageRequest) -> String {
    let default = if is_flux2_edit_candle_dev(&request.model) {
        FLUX2_EDIT_CANDLE_DEV_REPO
    } else {
        FLUX2_EDIT_CANDLE_DEFAULT_REPO
    };
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

/// True when this is a candle-eligible FLUX.2 edit job (a klein/dev `edit_image` job with a source)
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

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → the family default
/// (klein 4 / dev 28).
fn flux2_edit_candle_steps(request: &ImageRequest, default: u32) -> u32 {
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
        .unwrap_or(default)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the family default
/// (klein 1.0 / dev 4.0), clamped.
fn flux2_edit_candle_guidance(request: &ImageRequest, default: f32) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Reference asset ids for a FLUX.2 edit, in order. The multi-image picker (sc-6211) sends the plural
/// `referenceAssetIds` — take all of them, capped at [`FLUX2_EDIT_CANDLE_MAX_REFERENCES`]; with no plural
/// list it falls back to the single Image-Edit `sourceAssetId` (`edit_image` mode). Mirrors the MLX
/// `flux2_edit_reference_ids` (multi) + `boogu_edit_reference_ids`.
fn flux2_edit_candle_reference_ids(request: &ImageRequest) -> Vec<String> {
    if !request.reference_asset_ids.is_empty() {
        // The parsed list is already trimmed + non-empty (sceneworks-core `string_list`).
        return request
            .reference_asset_ids
            .iter()
            .take(FLUX2_EDIT_CANDLE_MAX_REFERENCES)
            .cloned()
            .collect();
    }
    if let Some(id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    Vec::new()
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
    quant_bits: Option<i64>,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw.insert(
        "editEngine".to_owned(),
        Value::String(FLUX2_EDIT_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Load the FLUX.2 edit reference set: the `N ∈ [1, 5]` reference images (plural `referenceAssetIds`,
/// else the single `sourceAssetId` — [`flux2_edit_candle_reference_ids`]), each pre-fit to the render
/// W×H (crop / pad / outpaint→pad; `stretch` keeps the legacy non-aspect resize). The provider
/// re-resizes internally, but pre-fitting keeps an off-aspect edit from stretching. Errors if no source.
fn load_flux2_edit_references(
    request: &ImageRequest,
    project_path: &Path,
    settings: &Settings,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<Image>> {
    let ids = flux2_edit_candle_reference_ids(request);
    if ids.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires a source image".to_owned(),
        ));
    }
    let mut references = Vec::with_capacity(ids.len());
    for id in &ids {
        let source =
            load_reference_image(&settings.data_dir, &request.project_id, id, project_path)?;
        let fitted = if request.fit_mode == "stretch" {
            source
        } else {
            flux2_edit_fit_image(source, width, height, &request.fit_mode)?
        };
        references.push(fitted);
    }
    Ok(references)
}

/// Real candle FLUX.2 edit generation: resolve the source(s) + base on the async side, pre-fit each to
/// the render geometry, then load `Flux2Edit` once + generate each image on the blocking thread.
/// `request.count` edits of the same reference set, each its own seed. dev (`flux2_dev`) loads Q4 via
/// `load_dev` with embedded distilled guidance; klein loads dense (CFG-free at guidance 1.0; >1 adds a
/// negative pass). `generate` takes `&self`, so the per-item closure needs no `mut`. Reuses
/// [`consume_gen_events`].
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
    let flux2_base = resolve_flux2_edit_candle_base(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2 base not found".to_owned()))?;
    if !flux2_edit_candle_mode(request) {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires edit_image mode + a source image".to_owned(),
        ));
    }
    let is_dev = is_flux2_edit_candle_dev(&request.model);
    let (width, height) = (request.width, request.height);
    let references = load_flux2_edit_references(request, project_path, settings, width, height)?;

    // dev (32B) loads Q4 (manifest `mlx.quantize: 4` → `resolve_quant`); klein loads dense. The dev
    // edit is activation-bound — multi-reference adds latent tokens to the DiT stream — but the candle
    // engine query-row-chunks its joint attention (sc-6217/sc-7523), so a device OOM surfaces as a load/
    // generate error rather than silently corrupting; no Mac-style unified-memory pre-guard applies here.
    let (quant, quant_bits) = if is_dev {
        resolve_quant(request)
    } else {
        (None, None)
    };
    let steps = flux2_edit_candle_steps(
        request,
        if is_dev {
            FLUX2_EDIT_CANDLE_DEV_STEPS
        } else {
            FLUX2_EDIT_CANDLE_DEFAULT_STEPS
        },
    );
    let guidance = flux2_edit_candle_guidance(
        request,
        if is_dev {
            FLUX2_EDIT_CANDLE_DEV_GUIDANCE
        } else {
            FLUX2_EDIT_CANDLE_DEFAULT_GUIDANCE
        },
    );
    let repo = flux2_edit_candle_repo(request);
    let raw_settings =
        flux2_edit_candle_raw_settings(request, &repo, steps, guidance, quant_bits, references.len());

    // Per-image work items: (seed, prompt) — `request.count` edits of the same reference set.
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
            let paths = Flux2EditPaths { root: flux2_base };
            let model = if is_dev {
                Flux2Edit::load_dev(&paths, quant)
            } else {
                Flux2Edit::load(&paths)
            }
            .map_err(|error| WorkerError::Engine(format!("FLUX.2 edit load failed: {error}")))?;
            Ok((model, references))
        },
        move |(model, references), tx, cancel| {
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
                let result = model.generate(&req, &references, &mut *on_progress);
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
