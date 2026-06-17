/// How a FLUX.2 edit job batches its iterations.
enum Flux2Grouping {
    /// `count` independent images (per-image seeds), the plain reference/edit path.
    Plain,
    /// The 11-angle Character-Studio set: shared seed, per-angle prompt augment.
    Angles,
    /// The best-effort pose tier: `n` poses, shared seed, `[skeleton, reference]` sets.
    Poses(usize),
}

/// Decide the grouping for a FLUX.2 edit job (parity with the `MlxFlux2Adapter`
/// decision: pose set > angle set > plain, all gated to `character_image` mode — an
/// `edit_image` job is never grouped). The caller only reaches this with a reference
/// present, so `is_character_image` reduces to the mode check.
fn flux2_grouping(request: &ImageRequest) -> Flux2Grouping {
    if request.mode != "character_image" {
        return Flux2Grouping::Plain;
    }
    let poses = pose_entries(request).len();
    if poses > 0 {
        return Flux2Grouping::Poses(poses);
    }
    if advanced::flag(&request.advanced, "angleSet") {
        return Flux2Grouping::Angles;
    }
    Flux2Grouping::Plain
}

/// True when the FLUX.2 Image-Edit source should be pre-fitted to W×H (parity with the
/// `MlxFlux2Adapter` fit gate): `edit_image` mode, a source asset, no character
/// `referenceAssetId`, and a non-`stretch` fit mode. The Character-Studio reference
/// path stays at native resolution.
fn should_fit_edit_source(request: &ImageRequest) -> bool {
    let has_source = request
        .source_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    // No character referenceAssetId (absent or empty).
    let no_reference = !request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    request.mode == "edit_image" && has_source && no_reference && request.fit_mode != "stretch"
}

/// Where a `src_w`×`src_h` image lands when contained (long edge fits) and centered in
/// a `width`×`height` box: `(new_w, new_h, left, top)`. Parity with Python `_contain_box`
/// (shared by the pad fit so the kept region lines up). Integer-divides the offsets.
fn contain_box(src_w: u32, src_h: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
    let ratio = (width as f32 / src_w as f32).min(height as f32 / src_h as f32);
    let new_w = ((src_w as f32 * ratio).round() as u32).max(1);
    let new_h = ((src_h as f32 * ratio).round() as u32).max(1);
    (new_w, new_h, (width - new_w) / 2, (height - new_h) / 2)
}

/// Resize an RGB image to exactly `width`×`height` honoring `mode` without distorting it
/// (parity with Python `fit_image`, RGB path only — no inpaint mask exists on the MLX
/// FLUX.2 edit path, so `outpaint` degrades to `pad` geometry):
///   - `crop`:    scale to COVER (short edge fits), center-crop the overflow.
///   - `pad`/`outpaint`: scale to CONTAIN (long edge fits), center on a black canvas.
///   - `stretch`: legacy non-aspect-preserving resize.
fn fit_rgb(source: &image::RgbImage, width: u32, height: u32, mode: &str) -> image::RgbImage {
    use image::imageops::FilterType::Lanczos3;
    let width = width.max(1);
    let height = height.max(1);
    let (src_w, src_h) = (source.width(), source.height());
    match mode {
        "stretch" => image::imageops::resize(source, width, height, Lanczos3),
        "crop" => {
            let ratio = (width as f32 / src_w as f32).max(height as f32 / src_h as f32);
            // Ceil so the scaled image always fully covers the target before cropping.
            let new_w = width.max((src_w as f32 * ratio).ceil() as u32);
            let new_h = height.max((src_h as f32 * ratio).ceil() as u32);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let left = (new_w - width) / 2;
            let top = (new_h - height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
        // "pad" / "outpaint": contain + center on a black canvas (letterbox).
        _ => {
            let (new_w, new_h, left, top) = contain_box(src_w, src_h, width, height);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
            canvas
        }
    }
}

/// Fit an engine [`Image`] (RGB8) to `width`×`height` by `mode` via [`fit_rgb`].
fn fit_engine_image(source: Image, width: u32, height: u32, mode: &str) -> WorkerResult<Image> {
    let rgb =
        image::RgbImage::from_raw(source.width, source.height, source.pixels).ok_or_else(|| {
            WorkerError::InvalidPayload("edit source buffer size mismatch".to_owned())
        })?;
    let fitted = fit_rgb(&rgb, width, height, mode);
    Ok(Image {
        width: fitted.width(),
        height: fitted.height(),
        pixels: fitted.into_raw(),
    })
}

// ---------------------------------------------------------------------------
// FLUX.2-klein edit / reference (macOS, sc-3029): the `flux2_klein_9b_edit` and
// `flux2_klein_9b_kv_edit` variants. FLUX.2-klein is MLX-only (no torch), so this
// is where its edit/reference jobs run. One output per requested count, each
// conditioned on the shared reference image(s); the -kv variant auto-engages the
// reference-K/V cache (~2.4× edit speedup).
// ---------------------------------------------------------------------------

/// The engine edit-variant id for a FLUX.2 SceneWorks model, or `None` if the model
/// has no edit variant. The base 9b + true_v2 share `flux2_klein_9b_edit`; the -kv
/// distill uses `flux2_klein_9b_kv_edit` (reference-K/V cache); dev uses the
/// `flux2_dev_edit` variant (sc-5919) — the same dev snapshot, edit conditioning via
/// the DiT token concat (Reference / MultiReference), embedded guidance, no -kv cache.
fn flux2_edit_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "flux2_klein_9b" | "flux2_klein_9b_true_v2" => Some("flux2_klein_9b_edit"),
        "flux2_klein_9b_kv" => Some("flux2_klein_9b_kv_edit"),
        "flux2_dev" => Some("flux2_dev_edit"),
        _ => None,
    }
}

/// Reference asset ids for a FLUX.2 edit: the character-flow `referenceAssetId`, else
/// the Image-Edit `sourceAssetId` (edit_image mode). Mirrors the Python
/// `ref_id = referenceAssetId or (sourceAssetId if edit_image)`.
fn flux2_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
    if let Some(id) = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        return vec![id.to_owned()];
    }
    if request.mode == "edit_image" {
        if let Some(id) = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
        {
            return vec![id.to_owned()];
        }
    }
    Vec::new()
}

/// True when this is a FLUX.2 edit job (a flux2 edit-capable model + ≥1 reference)
/// whose base weights resolve — routed to the edit variant rather than txt2img.
fn flux2_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    flux2_edit_engine_id(&request.model).is_some()
        && !flux2_edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// One `Reference` (single) or one `MultiReference` (N) edit conditioning from the
/// resolved reference images (cloned per output).
fn build_edit_conditioning(references: &[Image]) -> Vec<Conditioning> {
    if references.len() == 1 {
        vec![Conditioning::Reference {
            image: references[0].clone(),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference {
            images: references.to_vec(),
        }]
    }
}

/// Estimated peak unified-memory footprint (GB) of a FLUX.2-dev edit at `width`×`height` with
/// `reference_count` reference images — the input to the sc-6124 multi-reference memory guard. The
/// dev edit is activation-bound: the DiT attends over the target latent plus every reference latent,
/// each ≈⌈W/16⌉·⌈H/16⌉ tokens (VAE ×8, patch ×2), so the peak scales with the total sequence length.
/// Anchored on the sc-5923 worker-layer measurements (Q4 packed, `/usr/bin/time -l` peak on a 128 GB
/// Mac): single-reference 1024² ~81 GB, two-reference 1024² ~104 GB — a linear-in-tokens fit over
/// those two edit points (`BASE + PER_TOKEN·(1 + refs)·tokens_per_image`). Only used on the
/// multi-reference branch (`reference_count >= 2`); txt2img and single-reference are covered directly
/// by the declared `minMemoryGb`, and the cheaper txt2img per-token slope is intentionally out of
/// scope here (the fit is calibrated to the heavier edit sequence the guard actually gates).
fn flux2_dev_edit_peak_gb(reference_count: usize, width: u32, height: u32) -> f64 {
    const BASE_GB: f64 = 35.0;
    const PER_TOKEN_GB: f64 = 0.005_615; // (104 − 81) GB / (12288 − 8192) tokens, sc-5923.
    let tokens_per_image = (f64::from(width) / 16.0).ceil() * (f64::from(height) / 16.0).ceil();
    let total_tokens = (1.0 + reference_count as f64) * tokens_per_image;
    BASE_GB + PER_TOKEN_GB * total_tokens
}

/// Prevent a silent OOM on a FLUX.2-dev **multi-reference** edit (sc-6124). The default reachable
/// surface (txt2img + single-reference edit) fits the declared `minMemoryGb` (96), but a
/// multi-reference edit adds each reference's latent tokens to the DiT sequence and is
/// activation-bound — two references at 1024² peak ~104 GB (sc-5923), above the floor. It runs on a
/// 128 GB Mac but would be SIGKILL'd mid-render on a 96–104 GB machine. When the estimated peak plus
/// a fixed runtime/OS headroom exceeds the machine's unified memory, reject with an actionable
/// message instead. `reference_count < 2` and a failed RAM probe (`available_gb == None`)
/// short-circuit to `Ok`, so the guard is inert for every path reachable today (the worker assembles
/// at most one user reference; this fires only once a multi-image edit picker feeds two or more) and
/// never blocks a machine that can actually fit the edit.
fn flux2_dev_edit_memory_guard(
    reference_count: usize,
    width: u32,
    height: u32,
    available_gb: Option<f64>,
) -> WorkerResult<()> {
    if reference_count < 2 {
        return Ok(());
    }
    let Some(available_gb) = available_gb else {
        return Ok(());
    };
    // The measured 2-reference/1024² config (~104 GB) completed on a 128 GB Mac, so require the
    // estimated peak plus a fixed headroom for the OS + MLX Metal transient allocations.
    const HEADROOM_GB: f64 = 16.0;
    let needed_gb = flux2_dev_edit_peak_gb(reference_count, width, height);
    if available_gb + f64::EPSILON < needed_gb + HEADROOM_GB {
        return Err(WorkerError::InvalidPayload(format!(
            "FLUX.2-dev multi-reference edit at {width}×{height} with {reference_count} reference \
             images needs ~{needed} GB of unified memory (with headroom) but this machine has \
             ~{available} GB. Lower the output resolution, use a single reference image, or run on \
             a Mac with more memory.",
            needed = needed_gb.round() as i64,
            available = available_gb.round() as i64,
        )));
    }
    Ok(())
}

/// Generate one FLUX.2 edit image conditioned on `conditioning` (the reference set).
/// Distilled klein: guidance 1.0, no negative prompt.
#[allow(clippy::too_many_arguments)]
fn flux2_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("edit generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("edit generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "edit generator returned non-image output".to_owned(),
        )),
    }
}

fn flux2_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    engine_id: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("editEngine".to_owned(), Value::String(engine_id.to_owned()));
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Real FLUX.2 edit generation: load the edit variant once, then `count` outputs each
/// conditioned on the shared reference set. Mirrors [`generate_stream`]'s blocking-
/// thread + streamed-events shape and reuses [`consume_gen_events`].
async fn generate_flux2_edit_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let engine_id = flux2_edit_engine_id(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not a FLUX.2 edit model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2 weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    let adapter_label = model.adapter_label();

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = flux2_edit_reference_ids(request);
    let mut references = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        references.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "FLUX.2 edit requires a reference image".to_owned(),
        ));
    }
    // sc-3030 fit_image: pre-fit the Image-Edit source to the output W×H (crop / pad /
    // outpaint→pad) so an off-aspect edit doesn't stretch. Character-Studio references
    // stay native (the `should_fit_edit_source` gate excludes them).
    if should_fit_edit_source(request) {
        references = references
            .into_iter()
            .map(|reference| {
                fit_engine_image(reference, request.width, request.height, &request.fit_mode)
            })
            .collect::<WorkerResult<Vec<_>>>()?;
    }

    // sc-6124: guard the activation-bound FLUX.2-dev *multi-reference* edit against a silent OOM on
    // machines below its real requirement. The reachable surface (txt2img + single-reference edit)
    // fits the declared `minMemoryGb` (96); a second reference adds ~4096 latent tokens to the DiT
    // stream and pushes the 1024² peak to ~104 GB (sc-5923), over the floor. Single-reference and
    // txt2img short-circuit, so this is inert until a multi-image edit picker feeds ≥2 references.
    if engine_id == "flux2_dev_edit" {
        flux2_dev_edit_memory_guard(
            references.len(),
            request.width,
            request.height,
            crate::gpu::total_unified_memory_gb().await,
        )?;
    }

    // sc-3030 per-iteration grouping: a Character-Studio angle set (11 shared-seed,
    // per-angle prompt) / best-effort pose tier (one per pose, shared seed, each a
    // `[skeleton, reference]` set) / else the plain per-image reference path.
    let grouping = flux2_grouping(request);
    let set_seed = resolve_seed(request, 0);
    let (seeds, prompts, pose_keypoints): (
        Vec<i64>,
        Vec<String>,
        Option<Vec<Vec<crate::openpose_skeleton::Keypoint>>>,
    ) = match &grouping {
        Flux2Grouping::Poses(count) => {
            // Shared seed so only the pose changes across the set (Python parity).
            let keypoints = parse_poses(request)
                .into_iter()
                .map(|pose| pose.keypoints)
                .collect();
            let prompts = vec![augment_prompt_for_pose(&request.prompt); *count];
            (vec![set_seed; *count], prompts, Some(keypoints))
        }
        Flux2Grouping::Angles => {
            // Shared seed so noise-derived attributes (hair, lighting) stay constant
            // across angles — only the head pose changes (sc-2050 InstantID strategy).
            let prompts = CHARACTER_ANGLE_SET_ORDER
                .iter()
                .map(|angle| augment_prompt_for_angle(&request.prompt, angle))
                .collect();
            (
                vec![set_seed; CHARACTER_ANGLE_SET_ORDER.len()],
                prompts,
                None,
            )
        }
        Flux2Grouping::Plain => {
            let count = request.count as usize;
            let seeds = (0..count)
                .map(|index| resolve_seed(request, index))
                .collect();
            (seeds, vec![request.prompt.clone(); count], None)
        }
    };
    let total = seeds.len();

    let mut raw_settings = flux2_edit_raw_settings(
        request,
        &repo,
        engine_id,
        steps,
        quant_bits,
        guidance,
        references.len(),
    );
    match grouping {
        Flux2Grouping::Angles => {
            raw_settings.insert("angleSet".to_owned(), Value::Bool(true));
        }
        Flux2Grouping::Poses(_) => {
            raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        }
        Flux2Grouping::Plain => {}
    }

    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = load_spec(weights_dir, quant, adapters, None);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(
                tx,
                seeds.into_iter().zip(prompts),
                move |index, (seed, prompt), on_progress| {
                    // Pose tier: pair this pose's body-only skeleton (DWPose body, no
                    // hands/face — Python `draw_bodypose`) with the reference as a
                    // `[skeleton, reference]` multi-image set; else the plain reference set.
                    let conditioning = match &pose_keypoints {
                        Some(keypoints) => {
                            let skeleton = crate::openpose_skeleton::draw_wholebody(
                                width,
                                height,
                                &keypoints[index],
                                None,
                                None,
                                stickwidth,
                            );
                            vec![Conditioning::MultiReference {
                                images: vec![
                                    Image {
                                        width,
                                        height,
                                        pixels: skeleton.into_raw(),
                                    },
                                    references[0].clone(),
                                ],
                            }]
                        }
                        None => build_edit_conditioning(&references),
                    };
                    let (out_w, out_h, pixels) = flux2_edit_generate_one(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        conditioning,
                        &cancel,
                        on_progress,
                    )?;
                    Ok(Some((seed, out_w, out_h, pixels)))
                },
            )
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// FLUX.2-dev strict-pose Fun-Controlnet-Union (macOS, sc-6055 / engine sc-2292):
// the `flux2_dev_control` registry generator — a VACE ControlNet on the dev base.
// One image per library pose, each conditioned on a DWPose skeleton fed to the
// `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` branch (TRUE pose lock, not the
// best-effort `[skeleton, reference]` tier above). FLUX.2 is MLX-only, so this is
// the only strict-pose path for dev (no candle sibling). Mirrors the Z-Image MLX
// control path (`generate_zimage_control_stream`).
// ---------------------------------------------------------------------------

/// The engine registry id for the FLUX.2-dev Fun-Controlnet-Union variant (sc-2292).
const FLUX2_DEV_CONTROL_ENGINE_ID: &str = "flux2_dev_control";
/// Default Fun-Controlnet-Union control-weights repo + the `-2602` CFG-distilled variant (the
/// recommended one — the previous version lost CFG distillation after control training).
const FLUX2_CONTROL_REPO: &str = "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union";
const FLUX2_CONTROL_FILE: &str = "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors";
/// The asset `adapter` id recorded on FLUX.2-dev strict-pose assets (the dev base MLX label).
const FLUX2_CONTROL_ADAPTER_LABEL: &str = "mlx_flux2";

/// True when this is a FLUX.2-dev strict-pose job (`flux2_dev` + ≥1 pose, not edit mode) whose base
/// weights resolve — routed to the Fun-Controlnet-Union control path rather than the best-effort edit
/// pose tier or plain txt2img. Gated to `flux2_dev` (klein has no control checkpoint). Control-weights
/// presence is NOT part of the gate: they are fetched on first use in the stream (a missing checkpoint
/// downloads, then errors loudly only on a real failure — never silently drops the poses).
fn flux2_dev_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "flux2_dev"
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// The (repo, filename) of the FLUX.2-dev control weights — `advanced.controlWeights.{repo,filename}`
/// overrides, else the `-2602` Fun-Controlnet-Union default (parity with the Z-Image resolver).
fn flux2_control_repo_file(request: &ImageRequest) -> (String, String) {
    let cw = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    (
        pick("repo", FLUX2_CONTROL_REPO),
        pick("filename", FLUX2_CONTROL_FILE),
    )
}

/// Resolve the Fun-Controlnet-Union checkpoint the engine loads, downloading on first use. Order: an
/// env-pinned file (`SCENEWORKS_CONTROLNET_FLUX2`) → a whole-repo HF cache snapshot → download into the
/// app cache. Mirrors the candle `ensure_zimage_control_weights` / `ensure_kolors_control_weights`. The
/// 8.2 GB control checkpoint is lazy-fetched only on the first pose job (vs bloating the base download).
async fn ensure_flux2_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = flux2_control_repo_file(request);
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_FLUX2") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, &repo) {
        let f = snapshot.join(&file);
        if f.is_file() {
            return Ok(f);
        }
    }
    let client = reqwest::Client::new();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "FLUX.2-dev strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-flux2")
        .join(&file);
    crate::downloads::ensure_hf_cached_file(&context, &repo, "main", &file, &dst).await
}

/// Pose ControlNet lock strength for FLUX.2-dev: `advanced.controlScale` (default 0.75, clamp [0,2]).
/// The Fun-Controlnet-Union README recommends 0.65–0.80 for the dev branch, so the default sits at the
/// mid-point (Z-Image's strict-pose default is 0.9; the dev branch over-locks above ~0.8).
fn flux2_control_scale(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(&request.advanced, "controlScale", 0.75, 0.0..=2.0)
}

/// Generate one FLUX.2-dev strict-pose image: the `control` skeleton drives the Fun-Controlnet-Union
/// pose branch at `control_scale`. dev is guidance-distilled (embedded scalar) — `guidance` rides the
/// transformer's guidance embedder (no true-CFG). `reference` is the optional identity img2img-init
/// shared across the pose set (opt-in, off by default).
#[allow(clippy::too_many_arguments)]
fn flux2_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    control: Image,
    control_scale: f32,
    reference: Option<&(Image, f32)>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut conditioning = vec![Conditioning::Control {
        image: control,
        kind: ControlKind::Pose,
        scale: control_scale,
    }];
    if let Some((image, strength)) = reference {
        conditioning.push(Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        });
    }
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("FLUX.2-dev control generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("FLUX.2-dev control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "FLUX.2-dev control generator returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn flux2_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(FLUX2_DEV_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

/// The clamped identity img2img-init strength for the FLUX.2-dev strict-pose set, or `None` for the
/// pose-only tier (mirrors `zimage_identity_strength`). `Some` iff `advanced.referenceStrength > 0`
/// AND a non-empty `referenceAssetId` — the dev control engine accepts an optional `Reference` init
/// next to the required `Control`. Off by default (pose-only is the validated path).
fn flux2_identity_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// Resolve the optional identity img2img-init for the FLUX.2-dev strict-pose set: `Some((image,
/// strength))` when [`flux2_identity_strength`] engages (decoding `referenceAssetId`), else `None`
/// (the default pose-only tier). The reference is shared across the whole pose set.
fn resolve_flux2_identity_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(strength) = flux2_identity_strength(request) else {
        return Ok(None);
    };
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("flux2_identity_strength guarantees a non-empty referenceAssetId");
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(Some((image, strength)))
}

/// Build the FLUX.2-dev control LoadSpec: the base dev snapshot + the Fun-Controlnet-Union overlay
/// (+ quant + adapters). The dev base loads manifest-aware (a pre-quantized Q4 snapshot loads packed);
/// the bf16 control overlay loads dense and quantizes in place under `with_quant`.
fn flux2_control_spec(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

#[cfg(all(target_os = "macos", test))]
fn flux2_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = flux2_control_spec(weights_dir, control_weights, quant, adapters);
    gen_core::load(FLUX2_DEV_CONTROL_ENGINE_ID, &spec)
        .map_err(|error| WorkerError::Engine(format!("FLUX.2-dev control load failed: {error}")))
}

/// Real FLUX.2-dev strict-pose generation: one image per pose, each conditioned on a DWPose skeleton
/// locked by the Fun-Controlnet-Union branch (sc-6055; engine sc-2292). Mirrors
/// [`generate_zimage_control_stream`] — the control checkpoint is fetched on first use, then the dev
/// control engine loads once on the blocking thread and renders one image per pose (shared seed so
/// only the pose changes across the set). dev keeps its embedded guidance (no CFG).
async fn generate_flux2_dev_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Optional identity img2img-init (opt-in, off by default — `referenceStrength`-gated), shared
    // across the pose set. `None` → the pose-only tier (the validated sc-2292 default).
    let identity_init = resolve_flux2_identity_init(request, settings, project_path)?;

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("FLUX.2-dev weights not found".to_owned()))?;
    let control_weights = ensure_flux2_control_weights(api, settings, job, request).await?;
    let (quant, quant_bits) = resolve_quant(request);
    let model = mlx_model("flux2_dev")
        .ok_or_else(|| WorkerError::InvalidPayload("flux2_dev model row missing".to_owned()))?;
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let control_scale = flux2_control_scale(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &model);
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = flux2_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        count,
    );
    // Strict pose shares one seed across the set so noise-derived attributes (hair, wardrobe,
    // lighting) stay constant while only the pose changes (Z-Image parity).
    let seed = resolve_seed(request, 0);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = flux2_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        FLUX2_DEV_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "FLUX.2-dev control load failed".to_owned(),
        move |generator, tx, cancel| {
            let identity_init = identity_init.as_ref();
            drive_gen_items(tx, poses, move |_index, pose, on_progress| {
                let skeleton = crate::openpose_skeleton::draw_wholebody(
                    width,
                    height,
                    &pose.keypoints,
                    pose.hands.as_deref(),
                    pose.face.as_deref(),
                    stickwidth,
                );
                let control = Image {
                    width,
                    height,
                    pixels: skeleton.into_raw(),
                };
                let (out_w, out_h, pixels) = flux2_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    control,
                    control_scale,
                    identity_init,
                    &cancel,
                    on_progress,
                )?;
                Ok(Some((seed, out_w, out_h, pixels)))
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
        FLUX2_CONTROL_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// Qwen-Image strict-pose ControlNet (macOS, epic 3401 / sc-3575): the InstantX
// `Qwen-Image-ControlNet-Union` variant registered in mlx-gen as `qwen_image_control`.
// One image per library pose, shared seed, true CFG + character LoRA on the base Qwen model.
// ---------------------------------------------------------------------------
