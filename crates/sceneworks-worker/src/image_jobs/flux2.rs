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
// Qwen-Image strict-pose ControlNet (macOS, epic 3401 / sc-3575): the InstantX
// `Qwen-Image-ControlNet-Union` variant registered in mlx-gen as `qwen_image_control`.
// One image per library pose, shared seed, true CFG + character LoRA on the base Qwen model.
// ---------------------------------------------------------------------------
