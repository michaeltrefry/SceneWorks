/// The engine registry id for the Qwen-Image ControlNet-Union variant.
const QWEN_CONTROL_ENGINE_ID: &str = "qwen_image_control";
/// Default InstantX Qwen-Image-ControlNet-Union weights (Apache-2.0, DWPose-trained).
const QWEN_CONTROL_REPO: &str = "InstantX/Qwen-Image-ControlNet-Union";
const QWEN_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";

/// True when this is the base-Qwen strict-pose tier: `qwen_image` + non-empty object
/// `advanced.poses`, not edit mode, and base weights available. A `referenceAssetId`, if present,
/// is ignored for parity with the Python torch `QwenImageControlNetPipeline` path; identity comes
/// from character LoRA adapters on the base transformer.
fn qwen_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "qwen_image"
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

fn resolve_qwen_control_weights(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    resolve_control_weights_for(request, settings, QWEN_CONTROL_REPO, QWEN_CONTROL_FILE)
}

/// Load the Qwen-Image ControlNet-Union generator (base snapshot + InstantX control overlay).
fn qwen_control_spec(
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
fn qwen_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = qwen_control_spec(weights_dir, control_weights, quant, adapters);
    gen_core::load(QWEN_CONTROL_ENGINE_ID, &spec).map_err(|error| {
        WorkerError::Engine(format!("Qwen strict-pose control load failed: {error}"))
    })
}

/// Generate one Qwen strict-pose image: the pose skeleton drives the InstantX control branch at
/// `control_scale`; prompt, true CFG, negative prompt, quant, and LoRA/LoKr mirror base Qwen.
#[allow(clippy::too_many_arguments)]
fn qwen_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    control: Image,
    control_scale: f32,
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
        conditioning: vec![Conditioning::Control {
            image: control,
            kind: ControlKind::Pose,
            scale: control_scale,
        }],
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("Qwen strict-pose generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Qwen strict-pose generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Qwen strict-pose generator returned non-image output".to_owned(),
        )),
    }
}

fn qwen_control_raw_settings(
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
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(QWEN_CONTROL_ENGINE_ID.to_owned()),
    );
    raw
}

/// Real Qwen strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton. Mirrors the Python `_generate_pose_set` path: shared seed, full body/hands/face
/// skeleton, `advanced.controlScale`, true CFG, and character LoRA identity.
async fn generate_qwen_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let qwen = mlx_model("qwen_image")
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen model row missing".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen-Image weights not found".to_owned()))?;
    let control_weights = resolve_qwen_control_weights(request, settings).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Qwen strict-pose control weights not found (download {QWEN_CONTROL_REPO})."
        ))
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &qwen);
    let guidance = resolve_guidance(request, &qwen).unwrap_or(qwen.default_guidance());
    let negative_prompt = resolve_negative_prompt(request, &qwen);
    let control_scale = resolve_control_scale(request);
    let adapters = resolve_adapters(request, settings)?;
    let repo = model_repo(request, &qwen);
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings = qwen_control_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        control_scale,
        count,
    );
    let seed = resolve_seed(request, 0);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = qwen_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        QWEN_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "Qwen strict-pose control load failed".to_owned(),
        move |generator, tx, cancel| {
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
                let (out_w, out_h, pixels) = qwen_control_generate_one(
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
        "mlx_qwen",
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
// Qwen-Image-Edit (macOS, sc-3397): the `qwen_image_edit` / `_2509` / `_2511` ids, all
// served by the engine's single `qwen_image_edit` model (Reference/MultiReference
// dual-latent). This is where Qwen edit/reference jobs run — `edit_image`, the
// Character-Studio reference flow (subject variation), the 11-angle set, and the
// best-effort pose tier (`[reference, skeleton]` multi-image). True CFG (negative prompt
// + guidance from `trueCfgScale`), LoRA/LoKr, Q4/Q8, fit_image. Base-Qwen strict-pose
// ControlNet is handled by the `qwen_image_control` path above; the `_2511_lightning` distill
// (sampler + lightx2v LoRA) is sc-3398.
// ---------------------------------------------------------------------------

/// The engine edit-model id for a Qwen SceneWorks model, or `None` if it has no edit
/// variant. `qwen_image_edit` / `_2509` / `_2511` / `_2511_lightning` all map to the single
/// `qwen_image_edit` engine model; the lightning id differs only in its sampler + distill
/// LoRA (see [`qwen_edit_lightning`]), not in the engine model.
fn qwen_edit_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => Some("qwen_image_edit"),
        _ => None,
    }
}

/// The Lightning few-step distill for a Qwen edit variant, or `None` for the production
/// (multi-step true-CFG) path. The engine's `lightning` sampler (static-shift schedule +
/// CFG-off single forward, mlx-gen-qwen-image `model_edit.rs`, sc-2909) only produces a
/// clean image when the matching lightx2v distill LoRA is stacked at load time — so a
/// lightning variant carries both. Worker-local: the distill LoRA is fetched lazily into
/// the HF cache on first use (`ensure_distill_lora_cached`), mirroring the Python path's
/// `load_lora_weights` → `fuse_lora` (it is not a manifest install artifact). sc-3398.
struct LightningDistill {
    /// The engine sampler id passed via `GenerationRequest.sampler`.
    sampler: &'static str,
    /// HuggingFace repo holding the distill LoRA.
    repo: &'static str,
    /// The distill LoRA filename within the repo (the 4-step bf16 variant matches the
    /// 4-step `default_steps` of the lightning model row).
    file: &'static str,
}

/// The Lightning distill config for a SceneWorks model id, or `None` for every
/// production variant. Only `qwen_image_edit_2511_lightning` is a distilled variant today.
fn qwen_edit_lightning(model: &str) -> Option<LightningDistill> {
    match model {
        "qwen_image_edit_2511_lightning" => Some(LightningDistill {
            sampler: "lightning",
            repo: "lightx2v/Qwen-Image-Edit-2511-Lightning",
            file: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        }),
        _ => None,
    }
}

/// Ensure a single distill-LoRA `file` from HuggingFace `repo` is materialized in the
/// shared HF hub cache, returning its absolute path. Fast-paths when the file is already
/// cached; otherwise fetches just that file into the standard `models--<org>--<name>`
/// layout (deduping with the Python loader and other tools, sc-1904) — the Rust generate
/// path assumes weights are cached, so the lightning path fetches its distill LoRA lazily
/// here, mirroring the Python `load_lora_weights` HF download (sc-3398, decision (b):
/// worker-local, not a cross-cutting model-install artifact).
async fn ensure_distill_lora_cached(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    file: &str,
) -> WorkerResult<PathBuf> {
    // Fast path: already materialized in the hub cache (the common case after first use).
    if let Some(snapshot_dir) =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
    {
        let candidate = snapshot_dir.join(file);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let repo_dir = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    let revision = "main";
    let client = reqwest::Client::new();
    let snapshot =
        HuggingFaceSnapshot::resolve(&client, settings, repo, revision, &[file.to_owned()]).await?;
    if snapshot.files.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "Distill LoRA {file} not found in Hugging Face repo {repo}."
        )));
    }
    let mut progress = DownloadProgress::new(
        repo,
        directory_size(&repo_dir.join("blobs")).await,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    download_snapshot_into_cache(
        &DownloadContext {
            api,
            client: &client,
            settings,
            job_id: &job.id,
            cancel_message: "Generation canceled while fetching the Lightning distill LoRA.",
            fresh_download: false,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let snapshot_dir = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Hugging Face snapshot for {repo} missing after download."
            ))
        })?;
    let path = snapshot_dir.join(file);
    if !path.is_file() {
        return Err(WorkerError::InvalidPayload(format!(
            "Distill LoRA {file} missing from the {repo} snapshot after download."
        )));
    }
    Ok(path)
}

/// Reference asset ids for a Qwen edit: the character-flow `referenceAssetId`, else the
/// Image-Edit `sourceAssetId` (edit_image mode). Mirrors the Python
/// `ref = referenceAssetId or (sourceAssetId if edit_image)` and the FLUX.2 edit path.
fn qwen_edit_reference_ids(request: &ImageRequest) -> Vec<String> {
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

/// True when this is a Qwen edit job (a qwen edit-capable model + ≥1 reference) whose
/// weights resolve — routed to the `qwen_image_edit` engine model rather than txt2img.
fn qwen_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    qwen_edit_engine_id(&request.model).is_some()
        && !qwen_edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the Qwen edit true-CFG guidance. The engine's `guidance` IS the true CFG
/// (diffusers `true_cfg_scale`), so this reads `advanced.trueCfgScale` (NOT
/// `guidanceScale`, the inert embedded-guidance knob that the Python edit path pins at
/// 1.0) else the family default (4.0). The Character-Studio reference path clamps to
/// [1, 10] (Python `_reference_true_cfg_scale`); `edit_image` floors at 1.0 (the engine
/// needs CFG > 1 to engage).
fn resolve_qwen_edit_guidance(request: &ImageRequest, model: &ResolvedModel) -> f32 {
    let raw = request
        .advanced
        .get("trueCfgScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    if request.mode == "character_image" {
        raw.clamp(1.0, 10.0)
    } else {
        raw.max(1.0)
    }
}

/// Flat telemetry for a Qwen edit generation (parity with `flux2_edit_raw_settings`).
fn qwen_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // The engine guidance is the true CFG — record it under the key the Python path uses.
    raw.insert("trueCfgScale".to_owned(), json!(guidance));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert(
        "editEngine".to_owned(),
        Value::String("qwen_image_edit".to_owned()),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Generate one Qwen edit image conditioned on `conditioning` (the reference set). True
/// CFG: passes the negative prompt + `guidance`. Mirrors [`flux2_edit_generate_one`].
/// `sampler` selects the engine recipe — `Some("lightning")` runs the few-step distilled
/// path (CFG-off single forward), `None` the production multi-step true-CFG path (sc-3398).
#[allow(clippy::too_many_arguments)]
fn qwen_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: f32,
    sampler: Option<&str>,
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
        sampler: sampler.map(str::to_owned),
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

/// Real Qwen-Image-Edit generation: load the `qwen_image_edit` engine model once, then
/// one output per grouped iteration each conditioned on the shared reference set. Mirrors
/// [`generate_flux2_edit_stream`]'s blocking-thread + streamed-events shape and reuses the
/// shared grouping ([`flux2_grouping`]) and [`consume_gen_events`]; differs in true-CFG
/// guidance (`trueCfgScale`) + the negative prompt, the `[reference, skeleton]` pose order
/// (reference first drives the VL identity prompt), and the body-only pose skeleton.
async fn generate_qwen_edit_stream(
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
    let engine_id = qwen_edit_engine_id(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not a Qwen edit model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Qwen-Image-Edit weights not found".to_owned())
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &model);
    let guidance = resolve_qwen_edit_guidance(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    // Lightning few-step distill (sc-3398): the `lightning` sampler engages the engine's
    // CFG-off static-shift recipe, and the matching lightx2v distill LoRA is stacked AHEAD
    // of any user LoRAs (the user's own `loras` still occupy their slots — mirrors the
    // Python fuse-distill-then-stack path). The distill LoRA is fetched lazily into the HF
    // cache on first use. `None`/`Vec::new()` for the production multi-step variants.
    let lightning = qwen_edit_lightning(&request.model);
    let sampler = lightning.as_ref().map(|distill| distill.sampler);
    let mut adapters: Vec<AdapterSpec> = Vec::new();
    if let Some(distill) = &lightning {
        let path =
            ensure_distill_lora_cached(api, settings, job, distill.repo, distill.file).await?;
        adapters.push(AdapterSpec::new(path, 1.0, AdapterKind::Lora));
    }
    adapters.extend(resolve_adapters(request, settings)?);
    let repo = model_repo(request, &model);
    let adapter_label = model.adapter_label();

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = qwen_edit_reference_ids(request);
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
            "Qwen-Image-Edit requires a reference image".to_owned(),
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

    // Per-iteration grouping (shared with the FLUX.2 edit path): a Character-Studio angle
    // set (11 shared-seed, per-angle prompt) / best-effort pose tier (one per pose, shared
    // seed, each a `[reference, skeleton]` set) / else the plain per-image reference path.
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
            // across angles — only the head pose changes.
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

    let mut raw_settings = qwen_edit_raw_settings(
        request,
        &repo,
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
    // Record the Lightning recipe for telemetry/A-B parity (matches the Python `distillLora`
    // key format `repo/file`); absent on the production multi-step variants.
    if let Some(distill) = &lightning {
        raw_settings.insert(
            "sampler".to_owned(),
            Value::String(distill.sampler.to_owned()),
        );
        raw_settings.insert(
            "distillLora".to_owned(),
            Value::String(format!("{}/{}", distill.repo, distill.file)),
        );
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
                    // Pose tier: pair the reference with this pose's body-only skeleton (DWPose
                    // body, no hands/face — Python `draw_bodypose`) as a `[reference, skeleton]`
                    // multi-image set. Reference FIRST: the engine VL-encodes references[0] for
                    // the prompt embeds (identity), the skeleton is added dual-latent geometry.
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
                                    references[0].clone(),
                                    Image {
                                        width,
                                        height,
                                        pixels: skeleton.into_raw(),
                                    },
                                ],
                            }]
                        }
                        None => build_edit_conditioning(&references),
                    };
                    let (out_w, out_h, pixels) = qwen_edit_generate_one(
                        generator,
                        &prompt,
                        negative_prompt.clone(),
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        sampler,
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
// SenseNova-U1 it2i (macOS, sc-3900 / epic 3180): instruction edit + Character Studio on the
// unified `sensenova_u1_8b` / `sensenova_u1_8b_fast` ids. The same model does T2I (the base
// `generate_stream` path) and it2i here; a `Conditioning::Reference` (single, edit_image) or
// `Conditioning::MultiReference` (N, character_image incl. the angle set) drives the
// understanding-path vision encoder. SenseNova uses BOTH CFG knobs: the text CFG via `guidance`
// (`advanced.guidanceScale`, default 4.0 base / 1.0 fast) AND the image-guidance via `true_cfg`
// (`advanced.imageGuidanceScale` → engine `img_cfg_scale`, default 1.0 edit / 1.5 character) — so
// it is NOT a `uses_true_cfg` family. No negative prompt (`supports_negative_prompt=false`). The
// fast variant merges its 8-step distill LoRA internally at load (`load_fast`) — the worker only
// selects the engine id; there is no user-LoRA slot (`supports_lora=false`). SenseNova has no
// ControlNet, so strict pose is excluded by `sensenova_mlx_eligible`. Mirrors
// `generate_qwen_edit_stream`'s blocking-thread + streamed-events shape.
// ---------------------------------------------------------------------------
