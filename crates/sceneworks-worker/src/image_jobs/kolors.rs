// ---------------------------------------------------------------------------
// Kolors advanced conditioning (macOS, epic 3090): img2img (sc-4765) + the
// IP-Adapter-Plus reference (sc-4767). Both ride the base `generate_stream`
// `Reference` conditioning — the engine's single `kolors` model handles img2img
// (a `Reference` with no IP-Adapter loaded) and the IP image prompt (a `Reference`
// once `with_ip_adapter` installs the decoupled-attn weights), so this module only
// resolves the per-mode init image + the IP snapshot dir. Kolors is a guidance/CFG
// family (negative prompt + guidance 5.0), so the CFG flows through `guidance` (not
// `true_cfg`) on the base path. The strict-pose tier (sc-4766) is a combined
// ControlNet + IP-Adapter + img2img pass that the pinned engine does not yet
// support (sc-5012) and lands as a dedicated stream once that engine feature does.
// ---------------------------------------------------------------------------

/// The Kolors IP-Adapter-Plus snapshot repo: a CLIP-ViT-L/14-336 image encoder under
/// `image_encoder/` + the `ip_adapter_plus_general.safetensors` weights (the `image_proj`
/// Resampler + decoupled-attn K/V pairs). The torch `KolorsDiffusersAdapter` downloads it at
/// `refs/pr/4`; the MLX path reuses the same HF-cache snapshot (no new weight to ship).
const KOLORS_IP_ADAPTER_REPO: &str = "Kwai-Kolors/Kolors-IP-Adapter-Plus";
/// IP-Adapter scale when the request omits `ipAdapterScale` — the torch
/// `KolorsDiffusersAdapter._ip_adapter_scale` default 0.6.
const KOLORS_IP_SCALE: f32 = 0.6;
/// img2img strength for a Kolors edit when the request omits `strength` — the torch
/// `KolorsImg2ImgPipeline` default 0.6, forwarded verbatim to the engine's
/// `init_time_step(steps, strength)`.
const KOLORS_EDIT_STRENGTH: f32 = 0.6;

/// Resolve the Kolors img2img init for `mode == "edit_image"` (sc-4765): `Some((source, strength))`
/// decoding `sourceAssetId` and pre-fitting it to the output W×H (crop/pad/outpaint via
/// [`should_fit_edit_source`]/[`fit_engine_image`] — never stretch an off-aspect source, epic 2551);
/// `None` when not an edit job or no source asset (the base path then falls back to plain txt2img).
/// `strength` is the torch `advanced.strength` (default 0.6) forwarded verbatim to the engine's
/// `Reference` (img2img init) — its `init_time_step(steps, strength)` matches the diffusers
/// `KolorsImg2ImgPipeline` start step.
fn resolve_kolors_edit_init(
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
    let strength =
        advanced::f32_clamped(&request.advanced, "strength", KOLORS_EDIT_STRENGTH, 0.05..=1.0);
    Ok(Some((image, strength)))
}

/// Resolve the Kolors IP-Adapter-Plus reference image prompt (sc-4767): `(image, scale)` decoding
/// `referenceAssetId` + the IP scale (`advanced.ipAdapterScale`, default 0.6). Once the IP-Adapter
/// is installed via [`resolve_kolors_ip_adapter_dir`] + `LoadSpec::with_ip_adapter`, the engine
/// treats this `Reference` as the image prompt (decoupled cross-attn) rather than an img2img init.
fn resolve_kolors_ip_reference(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<(Image, f32)> {
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Kolors IP-Adapter requires a reference image".to_owned())
        })?;
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let scale =
        advanced::f32_clamped(&request.advanced, "ipAdapterScale", KOLORS_IP_SCALE, 0.0..=1.0);
    Ok((image, scale))
}

/// Resolve the Kolors IP-Adapter-Plus snapshot directory the engine loads (`image_encoder/`
/// CLIP-ViT-L/14-336 + `ip_adapter_plus_general.safetensors`). Errors loudly if the snapshot or
/// either required file is absent — the model-download flow fetches the repo (`refs/pr/4`) ahead of
/// generation, like the base weights; mirrors the SDXL/FLUX IP-Adapter dir resolvers.
fn resolve_kolors_ip_adapter_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    let missing = || {
        WorkerError::InvalidPayload(format!(
            "Kolors IP-Adapter weights not found (download {KOLORS_IP_ADAPTER_REPO})."
        ))
    };
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, KOLORS_IP_ADAPTER_REPO)
        .ok_or_else(missing)?;
    if !snapshot.join("ip_adapter_plus_general.safetensors").exists()
        || !snapshot.join("image_encoder").join("model.safetensors").exists()
    {
        return Err(missing());
    }
    Ok(snapshot)
}

// ---------------------------------------------------------------------------
// Kolors strict-pose tier (macOS, sc-4766 / engine sc-5012): the Character-Studio
// pose-locked character variations. One image per library pose, each driven by a
// DWPose skeleton (the pose ControlNet, `Kwai-Kolors/Kolors-ControlNet-Pose`) + the
// IP-Adapter-Plus identity reference + an img2img init from that same reference — all
// in ONE combined engine pass (`Kolors::denoise_controlnet_ip_latents`). Faithful port
// of the torch `KolorsDiffusersAdapter._run_pose` / `_generate_pose_set`. Mirrors
// `generate_zimage_control_stream` (one image per pose, shared seed, streamed events),
// but loads base + ControlNet + IP-Adapter in one `LoadSpec` and passes Control +
// Reference together (the combined tier the engine gained in sc-5012).
// ---------------------------------------------------------------------------

/// The official Kolors pose ControlNet checkpoint (a diffusers `ControlNetModel` snapshot dir with
/// `diffusion_pytorch_model.safetensors`). Loaded via `LoadSpec::with_control(Dir(..))` — the engine
/// expects a directory, not a single file.
const KOLORS_CONTROLNET_REPO: &str = "Kwai-Kolors/Kolors-ControlNet-Pose";
/// Pose ControlNet conditioning scale when the request omits `openPoseScale` — the torch
/// `KolorsDiffusersAdapter._openpose_scale` default 0.7 (clamp [0, 2]).
const KOLORS_POSE_CONTROL_SCALE: f32 = 0.7;
/// img2img init strength for the pose tier — the torch `_run_pose` default 1.0 (at full strength the
/// reference init only seeds latent dimensions; identity rides the IP-Adapter, structure the pose).
const KOLORS_POSE_IMG2IMG_STRENGTH: f32 = 1.0;

/// True when this is a Kolors strict-pose job: `kolors` + ≥1 `advanced.poses` + a `referenceAssetId`
/// (the identity) whose base weights resolve. Routed to the combined control stream rather than the
/// base path. Mirrors the torch `_pose_entries` gate (poses ONLY with a reference). The ControlNet /
/// IP-Adapter checkpoints are validated in the stream so a missing one errors loudly.
fn kolors_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "kolors"
        && !pose_entries(request).is_empty()
        && request
            .reference_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the Kolors pose ControlNet snapshot directory (`diffusion_pytorch_model.safetensors`
/// inside). Errors loudly if absent — the model-download flow fetches it ahead of generation.
fn resolve_kolors_controlnet_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    let missing = || {
        WorkerError::InvalidPayload(format!(
            "Kolors pose ControlNet weights not found (download {KOLORS_CONTROLNET_REPO})."
        ))
    };
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, KOLORS_CONTROLNET_REPO)
        .ok_or_else(missing)?;
    if !snapshot.join("diffusion_pytorch_model.safetensors").exists() {
        return Err(missing());
    }
    Ok(snapshot)
}

/// Pose ControlNet lock strength: `advanced.openPoseScale` (default 0.7, clamp [0, 2]) — the torch
/// `_openpose_scale` knob.
fn kolors_pose_control_scale(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(
        &request.advanced,
        "openPoseScale",
        KOLORS_POSE_CONTROL_SCALE,
        0.0..=2.0,
    )
}

/// The combined Kolors `LoadSpec`: base snapshot + the pose ControlNet (`with_control`) + the
/// IP-Adapter-Plus (`with_ip_adapter`) + quant + LoRA. The engine's `kolors` model routes a request
/// carrying both `Control` and `Reference` to its combined pose pass (sc-5012).
fn kolors_control_spec(
    weights_dir: PathBuf,
    control_dir: PathBuf,
    ip_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::Dir(control_dir))
        .with_ip_adapter(WeightsSource::Dir(ip_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

/// Generate one Kolors pose image: the `control` skeleton locks the pose, the `reference` drives
/// identity via the IP-Adapter (scale `ip_scale`) AND seeds the img2img init (strength
/// `img2img_strength`). Kolors is true-CFG (negative prompt + guidance). The reference is shared
/// across the pose set (identity is constant; only the per-pose skeleton changes).
#[allow(clippy::too_many_arguments)]
fn kolors_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    control: Image,
    control_scale: f32,
    reference: &Image,
    ip_scale: f32,
    img2img_strength: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let conditioning = vec![
        Conditioning::Control {
            image: control,
            kind: ControlKind::Pose,
            scale: control_scale,
        },
        Conditioning::Reference {
            image: reference.clone(),
            strength: Some(ip_scale),
        },
    ];
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        strength: Some(img2img_strength),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("kolors pose generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("kolors pose generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "kolors pose generator returned non-image output".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn kolors_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    control_scale: f32,
    ip_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = mlx_raw_settings(request, repo, steps, quant_bits, guidance);
    raw.insert("controlNetPose".to_owned(), json!(KOLORS_CONTROLNET_REPO));
    raw.insert("openPoseScale".to_owned(), json!(control_scale));
    raw.insert("ipAdapterScale".to_owned(), json!(ip_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw
}

/// Real Kolors strict-pose generation: one image per pose, each conditioned on a DWPose skeleton
/// (pose ControlNet) + the shared IP-Adapter identity reference + an img2img init from that
/// reference, in one combined engine pass (sc-5012). Mirrors [`generate_zimage_control_stream`]'s
/// blocking-thread + streamed-events shape, reusing [`consume_gen_events`].
async fn generate_kolors_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let kolors = mlx_model("kolors")
        .ok_or_else(|| WorkerError::InvalidPayload("kolors model row missing".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Kolors weights not found".to_owned()))?;
    let control_dir = resolve_kolors_controlnet_dir(settings)?;
    let ip_dir = resolve_kolors_ip_adapter_dir(settings)?;

    // The identity reference: the IP-Adapter image prompt AND the img2img init, shared across the
    // pose set (`kolors_control_available` guarantees a non-empty referenceAssetId).
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Kolors pose tier requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;

    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &kolors);
    let guidance = resolve_guidance(request, &kolors);
    let negative_prompt = resolve_negative_prompt(request, &kolors);
    let control_scale = kolors_pose_control_scale(request);
    let ip_scale =
        advanced::f32_clamped(&request.advanced, "ipAdapterScale", KOLORS_IP_SCALE, 0.0..=1.0);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &kolors);
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = kolors_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        ip_scale,
        count,
    );
    // The pose set shares one seed so noise-derived attributes stay constant while only the pose
    // changes (torch `_generate_pose_set` shares `set_seed`).
    let seed = resolve_seed(request, 0);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = kolors_control_spec(weights_dir, control_dir, ip_dir, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        "kolors",
        adapter_count,
        spec,
        "Kolors pose ControlNet load failed".to_owned(),
        move |generator, tx, cancel| {
            let reference = &reference;
            let negative_prompt = negative_prompt.clone();
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
                let (out_w, out_h, pixels) = kolors_control_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    control,
                    control_scale,
                    reference,
                    ip_scale,
                    KOLORS_POSE_IMG2IMG_STRENGTH,
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
        kolors.adapter_label(),
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
