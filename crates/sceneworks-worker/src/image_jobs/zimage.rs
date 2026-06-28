/// The engine registry id for the Z-Image Fun-Controlnet-Union variant.
const ZIMAGE_CONTROL_ENGINE_ID: &str = "z_image_turbo_control";
/// Default Fun-Controlnet-Union control-weights repo + file (sc-2257 parity).
const ZIMAGE_CONTROL_REPO: &str = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1";
const ZIMAGE_CONTROL_FILE: &str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors";

/// The engine registry id for the **base** (non-distilled, full-CFG) Z-Image Fun-Controlnet-Union
/// variant (sc-8251). Same VACE Fun-Union control branch as the Turbo variant, but assembled from a
/// base `Tongyi-MAI/Z-Image` snapshot + the base control checkpoint, and driven with REAL CFG.
const ZIMAGE_BASE_CONTROL_ENGINE_ID: &str = "z_image_control";
/// Default base Fun-Controlnet-Union control-weights repo + file (sc-8251).
const ZIMAGE_BASE_CONTROL_REPO: &str = "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1";
const ZIMAGE_BASE_CONTROL_FILE: &str = "Z-Image-Fun-Controlnet-Union-2.1.safetensors";

// `pose_entries` / `parse_poses` / `PoseInput` moved to `base.rs` (shared by the candle InstantID
// lane, sc-5491); still in scope here via the shared `image_jobs` module.

/// True when this is a Z-Image strict-pose job (z-image + ≥1 pose) whose base weights
/// resolve — routed to the Fun-Controlnet-Union control path rather than plain txt2img.
/// Control-weights presence is checked in the stream so a missing checkpoint errors
/// loudly instead of silently dropping the poses to the txt2img path.
fn zimage_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "z_image_turbo"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// True when this is a **base** Z-Image strict-control job (`z_image` + ≥1 pose) whose base weights
/// resolve — routed to the base Fun-Controlnet-Union path (`z_image_control`) rather than plain
/// txt2img (sc-8251). The base mirror of [`zimage_control_available`]; keyed on the base model id so
/// the Turbo control path is untouched.
fn zimage_base_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "z_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the Fun-Controlnet-Union checkpoint (`advanced.controlWeights.{repo,filename}`
/// else defaults) to a single `.safetensors` in the HF cache. `None` when absent (the
/// model-download flow fetches it ahead of generation, like base weights).
fn resolve_control_weights_for(
    request: &ImageRequest,
    settings: &Settings,
    default_repo: &'static str,
    default_file: &'static str,
) -> Option<PathBuf> {
    let control = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let str_field = |key: &str, default: &'static str| -> String {
        control
            .and_then(|control| control.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    let repo = str_field("repo", default_repo);
    let filename = str_field("filename", default_file);
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, &repo)?;
    let path = snapshot.join(filename);
    path.exists().then_some(path)
}

/// Resolve the Z-Image Fun-Controlnet-Union checkpoint. The default repo comes from the shared
/// strict-control table (single source of truth — `STRICT_CONTROL_ENGINES`); the file default stays
/// engine-specific.
fn resolve_control_weights(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    resolve_control_weights_for(
        request,
        settings,
        strict_control_default_repo(ZIMAGE_CONTROL_ENGINE_ID),
        ZIMAGE_CONTROL_FILE,
    )
}

/// Resolve the **base** Z-Image Fun-Controlnet-Union checkpoint (sc-8251). The default repo comes from
/// the shared strict-control table (single source of truth — `STRICT_CONTROL_ENGINES`); the file
/// default stays engine-specific.
fn resolve_base_control_weights(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    resolve_control_weights_for(
        request,
        settings,
        strict_control_default_repo(ZIMAGE_BASE_CONTROL_ENGINE_ID),
        ZIMAGE_BASE_CONTROL_FILE,
    )
}

/// Pose ControlNet lock strength: `advanced.controlScale` (default 0.9, clamp [0,2]).
fn resolve_control_scale(request: &ImageRequest) -> f32 {
    request
        .advanced
        .get("controlScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(0.9)
        .clamp(0.0, 2.0)
}

// `PoseInput` + `parse_poses` moved to `base.rs` (shared by the candle InstantID lane, sc-5491).

/// Load the Z-Image Fun-Controlnet-Union generator (base snapshot + control overlay).
fn zimage_control_spec(
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
fn zimage_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = zimage_control_spec(weights_dir, control_weights, quant, adapters);
    gen_core::load(ZIMAGE_CONTROL_ENGINE_ID, &spec)
        .map_err(|error| WorkerError::Engine(format!("Z-Image control load failed: {error}")))
}

#[cfg(all(target_os = "macos", test))]
fn zimage_base_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    // Shares the Turbo control's `LoadSpec` shape (base dir + control overlay); only the engine id differs.
    let spec = zimage_control_spec(weights_dir, control_weights, quant, adapters);
    gen_core::load(ZIMAGE_BASE_CONTROL_ENGINE_ID, &spec)
        .map_err(|error| WorkerError::Engine(format!("Z-Image base control load failed: {error}")))
}

/// Generate one strict-pose image: the pre-built `conditioning` (the required `Control` plus an optional
/// identity `Reference`, assembled by the shared [`build_control_conditioning`] driver) drives the
/// Fun-Controlnet-Union branch. Z-Image-Turbo is guidance-distilled (no CFG / negative).
#[allow(clippy::too_many_arguments)]
fn zimage_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
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
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("control generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "control generator returned non-image output".to_owned(),
        )),
    }
}

fn zimage_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Z-Image-Turbo is guidance-distilled — no CFG.
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw
}

/// Real Z-Image strict-pose generation: one image per pose, each conditioned on a DWPose
/// skeleton rendered from the pose keypoints + locked by the Fun-Controlnet-Union branch.
/// Mirrors [`generate_stream`]'s blocking-thread + streamed-events shape (the MLX
/// generator is `!Send` + single-thread), reusing [`consume_gen_events`].
async fn generate_zimage_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Identity img2img-init (sc-2328 / sc-3146) — OPT-IN escape hatch, off by default. The
    // Fun-Controlnet-Union pose head denoises the pose FROM NOISE, so seeding from a reference
    // init fights the pose lock on few-step Turbo (validated marginal on 8-step Turbo; no single
    // strength holds BOTH identity and pose). It engages only when advanced.referenceStrength > 0
    // AND a referenceAssetId is present — parity with `MlxZImageAdapter._identity_init_requested`.
    // The reference is shared across the whole pose set (identity is constant; only the per-pose
    // skeleton changes). None → the pose-only tier (the validated sc-2257 default).
    let identity_init = resolve_zimage_identity_init(request, settings, project_path)?;

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Z-Image weights not found".to_owned()))?;
    let control_weights = resolve_control_weights(request, settings).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Z-Image strict-pose control weights not found (download {ZIMAGE_CONTROL_REPO})."
        ))
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let zimage = mlx_model("z_image_turbo")
        .ok_or_else(|| WorkerError::InvalidPayload("z-image model row missing".to_owned()))?;
    let steps = resolve_steps(request, &zimage);
    let control_scale = resolve_control_scale(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &zimage);
    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds (z_image_turbo_control = {Pose, Canny, Depth}) and resolve an optional user-supplied
    // control-map passthrough. The current (pose-only) job sets no `controlMode`, so `kind == Pose` and the
    // skeleton preprocessor below runs exactly as before.
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(ZIMAGE_CONTROL_ENGINE_ID, &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    // sc-8249 source threading: for canny/depth WITHOUT a user-supplied control map, the control map is
    // auto-derived from the input image (canny edges / Depth-Anything-V2). The pose tier never needs a
    // source (the skeleton is synthetic).
    let control_source = resolve_control_source(request, settings, project_path)?;
    // Auto depth-estimator weights: provisioned only for a depth job WITHOUT a user-supplied depth map
    // (passthrough short-circuits estimation). Shared across the set; fetched once on first depth job
    // (sc-8242).
    let depth_weights_dir = if control_kind == ControlKind::Depth && user_control.is_none() {
        Some(ensure_depth_estimator_dir(api, settings, job).await?)
    } else {
        None
    };
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings =
        zimage_control_raw_settings(request, &repo, steps, quant_bits, control_scale, count);
    // Strict pose shares one seed across the set so noise-derived attributes (hair,
    // wardrobe, lighting) stay constant while only the pose changes (Python parity).
    let seed = resolve_seed(request, 0);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = zimage_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        ZIMAGE_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "Z-Image control load failed".to_owned(),
        move |generator, tx, cancel| {
            let identity_init = identity_init.as_ref();
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            let depth_weights_dir = depth_weights_dir.as_deref();
            drive_gen_items(tx, poses, move |_index, pose, on_progress| {
                let control = preprocess_control_entry(
                    &control_kind,
                    user_control,
                    Some(&pose),
                    control_source,
                    width,
                    height,
                    stickwidth,
                    depth_weights_dir,
                )?;
                let conditioning = build_control_conditioning(
                    control,
                    control_kind.clone(),
                    control_scale,
                    identity_init,
                );
                let (out_w, out_h, pixels) = zimage_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    conditioning,
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
        ZIMAGE_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Generate one **base** Z-Image strict-control image (sc-8251): like [`zimage_control_generate_one`]
/// but the base is the non-distilled full-CFG foundation model, so it forwards a real CFG `guidance`
/// scale + an optional negative prompt (the base `z_image_control` descriptor `supports_guidance` +
/// `supports_negative_prompt`).
#[allow(clippy::too_many_arguments)]
fn zimage_base_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance: Some(guidance),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("base Z-Image control generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("base Z-Image control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "base Z-Image control generator returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn zimage_base_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // The base is the full-CFG foundation model — record the real guidance scale.
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(ZIMAGE_BASE_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

/// Real **base** Z-Image strict-control generation (sc-8251): one image per pose, each conditioned on
/// a DWPose skeleton (or — when `advanced.controlMode` is canny/depth — an auto-derived control map
/// over the threaded input image) + locked by the base Fun-Controlnet-Union branch. The base mirror of
/// [`generate_zimage_control_stream`], differing only in the engine id (`z_image_control`), the base
/// control checkpoint, and REAL CFG (`guidance` + negative prompt) — the base is undistilled. Pose
/// rendering + source threading reuse the SAME shared strict-control driver, so the pose tier is
/// byte-identical to the Turbo path.
async fn generate_zimage_base_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Identity img2img-init (opt-in escape hatch, off by default) — same gate as the Turbo path
    // (`advanced.referenceStrength > 0` AND a `referenceAssetId`). Shared across the whole pose set.
    let identity_init = resolve_zimage_identity_init(request, settings, project_path)?;

    let zimage = mlx_model("z_image")
        .ok_or_else(|| WorkerError::InvalidPayload("z-image base model row missing".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Z-Image base weights not found".to_owned()))?;
    let control_weights = resolve_base_control_weights(request, settings).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Z-Image base strict-control weights not found (download {ZIMAGE_BASE_CONTROL_REPO})."
        ))
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &zimage);
    let guidance = resolve_guidance(request, &zimage).unwrap_or(zimage.default_guidance());
    let negative_prompt = resolve_negative_prompt(request, &zimage);
    let control_scale = resolve_control_scale(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &zimage);
    // Shared strict-control driver: validate the requested ControlKind against the engine's
    // supported_kinds (z_image_control = {Pose, Canny, Depth}) + resolve an optional user-supplied
    // control-map passthrough. A pose job sets no `controlMode`, so `kind == Pose` (byte-identical render).
    let control_kind = requested_control_kind(request)?;
    validate_control_kind(ZIMAGE_BASE_CONTROL_ENGINE_ID, &control_kind)?;
    let user_control = resolve_user_control_map(request, settings, project_path)?;
    // sc-8251 source threading: for canny/depth WITHOUT a user-supplied control map, the control map is
    // auto-derived from the input image (canny edges / Depth-Anything-V2). The pose tier never needs a
    // source (the skeleton is synthetic).
    let control_source = resolve_control_source(request, settings, project_path)?;
    let depth_weights_dir = if control_kind == ControlKind::Depth && user_control.is_none() {
        Some(ensure_depth_estimator_dir(api, settings, job).await?)
    } else {
        None
    };
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = zimage_base_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        count,
    );
    // Strict control shares one seed across the set so noise-derived attributes stay constant
    // while only the per-pose skeleton changes (Python parity).
    let seed = resolve_seed(request, 0);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = zimage_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        ZIMAGE_BASE_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "Z-Image base control load failed".to_owned(),
        move |generator, tx, cancel| {
            let identity_init = identity_init.as_ref();
            let user_control = user_control.as_ref();
            let control_source = control_source.as_ref();
            let depth_weights_dir = depth_weights_dir.as_deref();
            let negative_prompt = negative_prompt.clone();
            drive_gen_items(tx, poses, move |_index, pose, on_progress| {
                let control = preprocess_control_entry(
                    &control_kind,
                    user_control,
                    Some(&pose),
                    control_source,
                    width,
                    height,
                    stickwidth,
                    depth_weights_dir,
                )?;
                let conditioning = build_control_conditioning(
                    control,
                    control_kind.clone(),
                    control_scale,
                    identity_init,
                );
                let (out_w, out_h, pixels) = zimage_base_control_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
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
        ZIMAGE_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// The clamped identity img2img-init strength for the Z-Image strict-pose set, or `None` for the
/// pose-only tier (sc-3146). `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty
/// `referenceAssetId` is present — parity with `MlxZImageAdapter._identity_init_requested`. The
/// strict-pose stream always carries poses (`zimage_control_available`), so the
/// bare-reference-without-poses rejection is handled upstream; here a `referenceStrength` set
/// without an asset simply falls back to pose-only, matching the Python gate rather than erroring.
///
/// `strength` is the user value clamped to `[0.05, 1.0]` and carries the mflux `image_strength`
/// convention **verbatim** (no numeric inversion): the mlx-gen Z-Image control engine and mflux
/// agree — higher strength → later denoise start (`init_time_step`) → output stays closer to the
/// init. Mirrors `MlxZImageAdapter._reference_strength` + the sidecar's verbatim forward. Pure
/// (request only) so the parity-sensitive gate + clamp are unit-testable without asset I/O.
fn zimage_identity_strength(request: &ImageRequest) -> Option<f32> {
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

/// Resolve the optional identity img2img-init for the Z-Image strict-pose set (sc-3146):
/// `Some((image, strength))` when [`zimage_identity_strength`] engages, decoding `referenceAssetId`
/// via [`load_reference_image`]; `None` for the default pose-only tier. The reference is shared
/// across the whole pose set (identity is constant; only the per-pose skeleton changes).
fn resolve_zimage_identity_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(strength) = zimage_identity_strength(request) else {
        return Ok(None);
    };
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("zimage_identity_strength guarantees a non-empty referenceAssetId");
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(Some((image, strength)))
}

/// Resolve the Z-Image Image-Edit img2img init for `mode == "edit_image"` (epic 3529):
/// `Some((source, strength))` decoding `sourceAssetId` and pre-fitting it to the output W×H
/// (crop/pad/outpaint via [`should_fit_edit_source`]/[`fit_engine_image`] — never stretch an
/// off-aspect source); `None` when not an edit job or no source asset (the caller then falls
/// back to the identity-init reference path / plain txt2img). `strength` is the torch
/// `ZImageImg2ImgPipeline` knob (`advanced.strength`, default 0.6) forwarded verbatim to the
/// engine — its `init_time_step(steps, strength)` matches the diffusers img2img start step.
/// Both `z_image_edit` and `z_image_turbo` (mode `edit_image`) drive this one path (same
/// Turbo-weights img2img call in torch).
fn resolve_zimage_edit_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    if request.mode != "edit_image" {
        return Ok(None);
    }
    let Some(asset_id) = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let source = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    let image = if should_fit_edit_source(request) {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    } else {
        source
    };
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0);
    Ok(Some((image, strength)))
}

/// The asset `adapter` id for Z-Image (strict-pose shares the base z-image label).
const ZIMAGE_ADAPTER_LABEL: &str = "mlx_z_image";

// ---------------------------------------------------------------------------
// Character-Studio angle set + best-effort pose tier + fit_image (macOS, sc-3030):
// the per-iteration batch orchestration on top of FLUX.2-klein edit. An angle set
// loops the 11 canonical head angles (shared seed, per-angle prompt augment); the
// best-effort pose tier pairs each pose's body skeleton with the reference as a
// `[skeleton, reference]` multi-image set; fit_image pre-fits an Image-Edit source
// to the output W×H (crop/pad/outpaint) so off-aspect edits don't stretch. Faithful
// ports of `character_studio_angles.py` + the `MlxFlux2Adapter` / `fit_image` paths.
// ---------------------------------------------------------------------------

/// The 11 canonical Character-Studio angles, in order. Re-exported from the canonical
/// [`sceneworks_core::angle_kps`] source of truth (the same table the Key Point Library serves
/// as its built-in presets — sc-4434) so the worker and the library can never drift.
const CHARACTER_ANGLE_SET_ORDER: [&str; 11] = sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER;

// `angle_prompt_augment` / `strip_base_prompt` / `augment_prompt_for_angle` moved to `base.rs`
// (shared by the candle InstantID angle-set lane, sc-5491); still in scope here (same module) for
// `augment_prompt_for_pose` below + the Z-Image angle routing.

/// The pose-skeleton instruction appended to the prompt for the best-effort pose tier
/// (parity with `character_studio_angles.POSE_SKELETON_PROMPT`).
const POSE_SKELETON_PROMPT: &str =
    "matching the exact body pose shown in the OpenPose skeleton reference image";

/// Append the pose-skeleton cue to the user's base prompt (parity with
/// `augment_prompt_for_pose`).
fn augment_prompt_for_pose(base: &str) -> String {
    let base = strip_base_prompt(base);
    if base.is_empty() {
        POSE_SKELETON_PROMPT.to_owned()
    } else {
        format!("{base}, {POSE_SKELETON_PROMPT}")
    }
}
