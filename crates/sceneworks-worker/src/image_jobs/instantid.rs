/// The SceneWorks model id for native InstantID (production = InstantID on RealVisXL_V5.0).
const INSTANTID_MODEL: &str = "instantid_realvisxl";
/// SDXL base for InstantID when the manifest omits `repo` (the photoreal production base).
const INSTANTID_SDXL_REPO: &str = "SG161222/RealVisXL_V5.0";
/// Stock InstantID checkpoint repo — the IdentityNet `ControlNetModel/` lives here.
const INSTANTID_CONTROLNET_REPO: &str = "InstantX/InstantID";
/// Converted-weights bundle (download-on-first-use): the MLX `ip-adapter.safetensors`
/// (`tools/convert_instantid.py`) + the native face stack `scrfd_10g.safetensors`
/// (`convert_scrfd.py`) + `arcface_iresnet100.safetensors` (`convert_glintr100.py`). Public
/// repo, mirroring the YOLO11 / SAM2 `SceneWorks/*-mlx` uploads (sc-3633 / sc-3707).
const INSTANTID_MLX_REPO: &str = "SceneWorks/instantid-mlx";
const INSTANTID_IP_ADAPTER_FILE: &str = "ip-adapter.safetensors";
const INSTANTID_SCRFD_FILE: &str = "scrfd_10g.safetensors";
const INSTANTID_ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";
/// The IdentityNet weight file inside `ControlNetModel/` (a stock diffusers SDXL ControlNet).
const INSTANTID_CONTROLNET_FILES: [&str; 2] =
    ["config.json", "diffusion_pytorch_model.safetensors"];
/// Torch-parity defaults (the `instantid_realvisxl` MODEL_TARGETS): RealVisXL is tuned for a
/// low CFG; the engine's own `InstantIdRequest::default` guidance (5.0) is for base SDXL.
const INSTANTID_DEFAULT_STEPS: u32 = 30;
const INSTANTID_DEFAULT_GUIDANCE: f32 = 3.0;
const INSTANTID_IP_SCALE: f32 = 0.8;
const INSTANTID_CONTROLNET_SCALE: f32 = 0.8;
/// xinsir OpenPose-SDXL ControlNet (the pose-mode second branch, sc-3117). Loads via the stock
/// `load_controlnet` (no conversion) — `image_adapters.py:615-617` parity.
const INSTANTID_OPENPOSE_REPO: &str = "xinsir/controlnet-openpose-sdxl-1.0";
/// Torch-parity default OpenPose lock (`instantid_adapter.py::_openpose_scale`, default 0.7).
const INSTANTID_OPENPOSE_SCALE: f32 = 0.7;
/// The face-restore re-render side (the engine's production crop size, sc-3380).
const INSTANTID_FACE_RESTORE_SIDE: u32 = 1024;
/// The adapter/engine id recorded on InstantID assets + telemetry, selected by backend: the native
/// MLX provider on macOS, the candle (Windows/CUDA) provider off-Mac (sc-5491). Distinguishes the two
/// lanes in the asset sidecar + the `instantIdEngine` raw-settings key.
#[cfg(target_os = "macos")]
const INSTANTID_ENGINE: &str = "mlx_instantid";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const INSTANTID_ENGINE: &str = "candle_instantid";

/// How an InstantID character job batches its iterations (torch-parity precedence: a pose set
/// wins over an angle set, which wins over plain identity — `instantid_adapter.py:655`).
enum InstantIdMode {
    /// `count` images at the reference's natural head pose (engine `generate`, W×H letterboxed).
    Identity,
    /// The 11-view Character-Studio set, shared seed (engine `generate_with_kps` from the
    /// worker-owned [`INSTANTID_ANGLE_KPS`] presets, square).
    AngleSet,
    /// `n` pose-library poses, shared seed — MultiControlNet IdentityNet + OpenPose (engine
    /// `generate_pose`, square).
    PoseSet(usize),
}

/// The 11-view Character-Studio angle set flag.
fn instantid_angle_set(request: &ImageRequest) -> bool {
    advanced::flag(&request.advanced, "angleSet")
}

/// Classify the InstantID iteration mode (pose set > angle set > plain identity).
fn instantid_mode(request: &ImageRequest) -> InstantIdMode {
    let poses = pose_entries(request).len();
    if poses > 0 {
        InstantIdMode::PoseSet(poses)
    } else if instantid_angle_set(request) {
        InstantIdMode::AngleSet
    } else {
        InstantIdMode::Identity
    }
}

/// Per-image InstantID action (the engine entry point this iteration calls). `Send` (it is moved
/// into the blocking task): `BodyPoint = Option<(f64, f64)>`, `&'static str`, and the unit variant
/// are all `Send`.
enum InstantIdAction {
    /// `generate` — the reference's natural head pose, W×H letterboxed.
    Identity,
    /// `generate_with_kps` — a Character-Studio view from worker-owned landmark presets (square).
    /// Carries the normalized 5-point kps directly (sc-4424) rather than an angle name, so the
    /// worker owns the framing presets and arbitrary/user-defined kps flow through the same path.
    Angle([(f32, f32); 5]),
    /// `generate_pose` — MultiControlNet IdentityNet + OpenPose on these COCO-18 keypoints (square).
    Pose(Vec<BodyPoint>),
}

/// Bridge the worker's gallery-normalized keypoints (`openpose_skeleton::Keypoint = Option<(f32,
/// f32)>`) to the engine's `BodyPoint = Option<(f64, f64)>`. `parse_poses` already applied the
/// COCO-18 normalize + conf<=0 drop, so this is just the f32→f64 widening.
fn pose_to_body_points(keypoints: &[crate::openpose_skeleton::Keypoint]) -> Vec<BodyPoint> {
    keypoints
        .iter()
        .map(|point| point.map(|(x, y)| (x as f64, y as f64)))
        .collect()
}

/// Resolve the RealVisXL (SDXL) base snapshot for InstantID: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (default
/// RealVisXL_V5.0). The big base is staged by the normal model-download flow; `None` here
/// means it is not present, so the job is not MLX-runnable (falls through to torch).
fn resolve_instantid_sdxl_base(
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
        return resolve_app_managed_model_dir(settings, &path, "InstantID SDXL modelPath")
            .map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(INSTANTID_SDXL_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a native-MLX-eligible InstantID job: the production model in
/// `character_image` mode with a reference face whose SDXL base resolves locally. ALL InstantID
/// modes are now native (sc-3345 identity + angle set; sc-3381 pose mode + face-restore via the
/// #193 engine). Mirrors `jobs_store::instantid_mlx_eligible` so the worker and the router agree.
fn instantid_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == INSTANTID_MODEL
        && request.mode == "character_image"
        && non_empty(&request.reference_asset_id)
        && matches!(resolve_instantid_sdxl_base(request, settings), Ok(Some(_)))
}

/// The number of images an InstantID job produces: `n` for a pose set, the active angle
/// collection's length for an angle set (sc-4450 — variable N, not fixed 11), else `request.count`.
fn instantid_image_count(request: &ImageRequest, settings: &Settings) -> u32 {
    match instantid_mode(request) {
        InstantIdMode::PoseSet(count) => count as u32,
        InstantIdMode::AngleSet => active_angle_collection(request, settings).1.len() as u32,
        InstantIdMode::Identity => request.count,
    }
}

/// Resolve the active angle-set collection for this job (sc-4450): the per-generation override
/// (`advanced.keypointCollectionId`) → the user default → the built-in 11. Built-in fallback on
/// any store error so angle generation never hard-fails on a Key Point Library hiccup.
fn active_angle_collection(
    request: &ImageRequest,
    settings: &Settings,
) -> (
    String,
    Vec<sceneworks_core::project_store::ResolvedAnglePreset>,
) {
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let override_id = advanced::str(&request.advanced, "keypointCollectionId", "");
    let override_id = override_id.trim();
    let override_id = (!override_id.is_empty()).then_some(override_id);
    store
        .resolve_angle_collection(override_id)
        .unwrap_or_else(|_| {
            (
                sceneworks_core::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID.to_owned(),
                builtin_angle_presets(),
            )
        })
}

/// The built-in 11 as resolved angle presets (the worker-side fallback when the store is
/// unreachable, sc-4450).
fn builtin_angle_presets() -> Vec<sceneworks_core::project_store::ResolvedAnglePreset> {
    use sceneworks_core::{angle_kps, project_store::ResolvedAnglePreset};
    angle_kps::BUILTIN_ANGLE_KPS
        .iter()
        .map(|(angle, kps)| ResolvedAnglePreset {
            preset_id: angle_kps::builtin_preset_id(angle),
            name: angle_kps::builtin_angle_display_name(angle),
            angle: Some((*angle).to_owned()),
            kps: *kps,
        })
        .collect()
}

/// Resolve InstantID denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` →
/// the torch-parity default (30).
fn instantid_steps(request: &ImageRequest) -> u32 {
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
        .map(|steps| steps.clamp(1, 80) as u32)
        .unwrap_or(INSTANTID_DEFAULT_STEPS)
}

/// Resolve InstantID guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the
/// RealVisXL-tuned default (3.0). Clamped to a sane CFG range.
fn instantid_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(INSTANTID_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Resolve InstantID quantization. **fp16 (dense) is the default** — the validated identity
/// envelope (ArcFace-cosine 0.82 @1024²); Q8/Q4 only on an explicit `advanced.mlxQuantize` /
/// manifest opt-in (identity drops to ~0.64 @512² and full-res quant is unvalidated). Returns
/// the engine `bits` (`Some(4)`/`Some(8)`/`None`) + the recipe bit count.
fn instantid_quant(request: &ImageRequest) -> (Option<i32>, Option<i64>) {
    let raw = request
        .advanced
        .get("mlxQuantize")
        .and_then(quant_int)
        .or_else(|| {
            request
                .model_manifest_entry
                .get("mlx")
                .and_then(|mlx| mlx.get("quantize"))
                .and_then(quant_int)
        });
    match raw {
        Some(bits) if bits > 0 && bits <= 4 => (Some(4), Some(4)),
        Some(bits) if bits > 4 => (Some(8), Some(8)),
        // None / 0 / negative → fp16 (the default + the validated InstantID envelope).
        _ => (None, None),
    }
}

/// Flat telemetry recorded on InstantID assets (parity with `mlx_raw_settings` + the torch
/// `InstantIDAdapter` recipe keys).
#[allow(clippy::too_many_arguments)]
fn instantid_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: f32,
    ip_scale: f32,
    controlnet_scale: f32,
    angle_set: bool,
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
    raw.insert("ipAdapterScale".to_owned(), json!(ip_scale));
    raw.insert(
        "controlnetConditioningScale".to_owned(),
        json!(controlnet_scale),
    );
    raw.insert(
        "instantIdEngine".to_owned(),
        Value::String(INSTANTID_ENGINE.to_owned()),
    );
    if angle_set {
        raw.insert("angleSet".to_owned(), Value::Bool(true));
    }
    raw
}

/// Resolve a single InstantID weight file: return it if already present in `dir`, else
/// download `url` into `dir` (atomic `.tmp` + rename, so a partial download is never mistaken
/// for a complete one — same shape as `person_segment::ensure_segmenter_weights`).
async fn ensure_instantid_file(
    context: &DownloadContext<'_>,
    repo: &str,
    dir: &Path,
    name: &str,
) -> WorkerResult<PathBuf> {
    ensure_hf_cached_file(context, repo, "main", name, &dir.join(name)).await
}

/// Resolve only the SCRFD detector weights (`scrfd_10g.safetensors`) from the same converted
/// bundle InstantID uses — for the standalone kps-extraction capability (sc-4433), which needs
/// face detection but neither ArcFace nor the SDXL/IdentityNet stack. Shares the env override
/// (`SCENEWORKS_INSTANTID_WEIGHTS`) + app cache + download-on-first-use with
/// [`ensure_instantid_weights`], so a prior InstantID run leaves it already cached.
// The standalone kps-extraction capability is a macOS path; the candle lane only loads SCRFD via the
// InstantID face stack (`with_face`), so this helper is unused off-Mac — allow it dead there.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) async fn ensure_scrfd_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "KPS extraction canceled while fetching SCRFD weights.",
        fresh_download: false,
    };
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_SCRFD_FILE,
    )
    .await
}

/// Resolve the candle face-stack DIRECTORY (`scrfd_10g.safetensors` + `arcface_iresnet100.safetensors`)
/// for the off-Mac kps-extraction capability (sc-5497, epic 5482). Unlike the Mac path — which loads
/// SCRFD alone via [`ensure_scrfd_weights`] — the candle `candle_gen_face::load` loads the SCRFD
/// detector AND the ArcFace recognizer from one directory by their canonical names, so BOTH files must
/// be staged. Shares the same env override (`SCENEWORKS_INSTANTID_WEIGHTS`) + app cache +
/// download-on-first-use with [`ensure_instantid_weights`], so a prior InstantID / PuLID / extraction
/// run leaves it already cached. Returns the bundle dir (which IS the candle face stack's load dir,
/// exactly the `face_dir` the candle InstantID path resolves).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn ensure_face_stack_dir(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "KPS extraction canceled while fetching face-stack weights.",
        fresh_download: false,
    };
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    ensure_instantid_file(&context, INSTANTID_MLX_REPO, &bundle_dir, INSTANTID_SCRFD_FILE).await?;
    ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_ARCFACE_FILE,
    )
    .await?;
    Ok(bundle_dir)
}

/// Resolve all InstantID weight inputs, downloading the small converted bundle + the stock
/// IdentityNet on first use. Returns `(identitynet_dir, ip_adapter, scrfd, arcface)` — all
/// `Send` paths; the `!Send` MLX load happens on the blocking thread. Resolution order favours
/// an env override / the HF cache before any network fetch.
async fn ensure_instantid_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(WeightsSource, PathBuf, PathBuf, PathBuf)> {
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "InstantID generation canceled while fetching weights.",
        fresh_download: false,
    };

    // Converted bundle (ip-adapter + face stack): an env-pinned dir (pre-staged for local
    // validation) wins, else the app cache (download missing files from SceneWorks/instantid-mlx).
    let bundle_dir = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| settings.data_dir.join("cache").join("instantid-mlx"));
    let ip_adapter = ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_IP_ADAPTER_FILE,
    )
    .await?;
    let scrfd = ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_SCRFD_FILE,
    )
    .await?;
    let arcface = ensure_instantid_file(
        &context,
        INSTANTID_MLX_REPO,
        &bundle_dir,
        INSTANTID_ARCFACE_FILE,
    )
    .await?;

    // IdentityNet (stock InstantX ControlNetModel): env override → HF cache snapshot →
    // download the two files into the app cache.
    if let Ok(dir) = std::env::var("SCENEWORKS_INSTANTID_CONTROLNET") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok((WeightsSource::Dir(dir), ip_adapter, scrfd, arcface));
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, INSTANTID_CONTROLNET_REPO)
    {
        let controlnet = snapshot.join("ControlNetModel");
        if controlnet
            .join("diffusion_pytorch_model.safetensors")
            .exists()
        {
            return Ok((WeightsSource::Dir(controlnet), ip_adapter, scrfd, arcface));
        }
    }
    let controlnet_dir = settings.data_dir.join("cache").join("instantid-controlnet");
    for file in INSTANTID_CONTROLNET_FILES {
        let source = format!("ControlNetModel/{file}");
        ensure_hf_cached_file(
            &context,
            INSTANTID_CONTROLNET_REPO,
            "main",
            &source,
            &controlnet_dir.join(file),
        )
        .await?;
    }
    Ok((
        WeightsSource::Dir(controlnet_dir),
        ip_adapter,
        scrfd,
        arcface,
    ))
}

/// Resolve the xinsir OpenPose-SDXL ControlNet dir for pose mode: env override
/// (`SCENEWORKS_INSTANTID_OPENPOSE`) → HF cache snapshot → download the two files on first use. A
/// stock diffusers SDXL ControlNet (loads via `with_openpose`/`load_controlnet`, no conversion).
async fn ensure_instantid_openpose(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<WeightsSource> {
    if let Ok(dir) = std::env::var("SCENEWORKS_INSTANTID_OPENPOSE") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok(WeightsSource::Dir(dir));
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, INSTANTID_OPENPOSE_REPO) {
        if snapshot
            .join("diffusion_pytorch_model.safetensors")
            .exists()
        {
            return Ok(WeightsSource::Dir(snapshot));
        }
    }
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "InstantID generation canceled while fetching OpenPose weights.",
        fresh_download: false,
    };
    let dir = settings.data_dir.join("cache").join("instantid-openpose");
    for file in INSTANTID_CONTROLNET_FILES {
        ensure_instantid_file(&context, INSTANTID_OPENPOSE_REPO, &dir, file).await?;
    }
    Ok(WeightsSource::Dir(dir))
}

/// Real InstantID generation: resolve the reference + weights on the async side, then load the
/// bespoke `InstantId` provider once + generate each image on the blocking thread (the MLX
/// model is `!Send`). Three modes (torch parity): single identity (`generate`), the 11-view angle
/// set (`generate_with_kps` from the worker-owned [`INSTANTID_ANGLE_KPS`] presets — sc-4424), and
/// the pose-library set (`generate_pose`, MultiControlNet IdentityNet with xinsir OpenPose —
/// sc-3117). `advanced.faceRestore` adds the ADetailer-style re-render pass (`restore_face`,
/// sc-3380) on each output. The engine `generate*` take the per-job `CancelFlag` (via
/// `InstantIdRequest.cancel`) and a `Progress` callback (sc-4380/sc-4382), so streaming is
/// per-step (`Step`/`Decoding` events) and cancellation is honoured mid-denoise — same contract
/// as the registry families. Reuses [`consume_gen_events`] for the asset writes.
async fn generate_instantid_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sdxl_base = resolve_instantid_sdxl_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("InstantID base (RealVisXL) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("InstantID requires a reference face image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;

    let (controlnet, ip_adapter, scrfd_path, arcface_path) =
        ensure_instantid_weights(api, settings, job).await?;

    // User style/character LoRAs (sc-6038). InstantID is a stock SDXL (RealVisXL) UNet, so SDXL
    // adapters apply on top of IdentityNet + the identity IP-Adapter — and the manifest advertises
    // `families:[sdxl]`, so the picker offers them. Resolved + path-confined exactly like every other
    // SDXL-family path (base.rs/sdxl.rs); merged onto the UNet by the engine `InstantIdPaths.adapters`
    // seam. Shared across all three modes (identity / angle set / pose) since they share the one load.
    let adapters = resolve_adapters(request, settings)?;
    let adapter_count = adapters.len();

    let steps = instantid_steps(request);
    let guidance = instantid_guidance(request);
    let (quant_bits, recipe_bits) = instantid_quant(request);
    // The candle InstantID stack runs dense f16 — there is no quantized path. Ignore the MLX quant
    // knob entirely on this lane: don't apply it (the `quantize` step in the load closure is macOS-
    // only) and don't record it as applied (`recipe_bits` -> None). `let _` consumes the otherwise-
    // unused `quant_bits`.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let recipe_bits: Option<i64> = {
        let _ = (quant_bits, recipe_bits);
        None
    };
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        INSTANTID_IP_SCALE,
        0.0..=1.0,
    );
    let controlnet_scale = advanced::f32_clamped(
        &request.advanced,
        "controlnetConditioningScale",
        INSTANTID_CONTROLNET_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(INSTANTID_SDXL_REPO)
        .to_owned();
    let mode = instantid_mode(request);
    let angle_set = matches!(mode, InstantIdMode::AngleSet);
    let pose_set = matches!(mode, InstantIdMode::PoseSet(_));
    // The active Key Point Library collection drives the angle set (sc-4450): per-generation
    // override > user default > built-in 11. Resolved once (and only for angle jobs).
    let angle_collection = angle_set.then(|| active_angle_collection(request, settings));
    let openpose_scale = advanced::f32_clamped(
        &request.advanced,
        "openPoseScale",
        INSTANTID_OPENPOSE_SCALE,
        0.0..=2.0,
    );
    let face_restore = advanced::flag(&request.advanced, "faceRestore");
    // Load the xinsir OpenPose ControlNet only for pose mode (it is the MultiControlNet second
    // branch; identity/angle modes don't need it).
    let openpose = if pose_set {
        Some(ensure_instantid_openpose(api, settings, job).await?)
    } else {
        None
    };

    let mut raw_settings = instantid_raw_settings(
        request,
        &repo,
        steps,
        recipe_bits,
        guidance,
        ip_scale,
        controlnet_scale,
        angle_set,
    );
    if pose_set {
        raw_settings.insert("poseLibrary".to_owned(), Value::Bool(true));
        raw_settings.insert("openPoseScale".to_owned(), json!(openpose_scale));
    }
    if face_restore {
        raw_settings.insert("faceRestore".to_owned(), Value::Bool(true));
    }
    // Record how many user LoRAs were merged onto the SDXL UNet (sc-6038) so the asset sidecar shows
    // the adapters were applied (they previously rode the request but were silently dropped).
    if adapter_count > 0 {
        raw_settings.insert("appliedLoraCount".to_owned(), json!(adapter_count));
    }
    // Record which collection + ordered presets produced the set, so each asset (by index) maps
    // back to the preset that rendered it (sc-4450).
    if let Some((collection_id, presets)) = &angle_collection {
        raw_settings.insert("keypointCollectionId".to_owned(), json!(collection_id));
        raw_settings.insert(
            "anglePresetIds".to_owned(),
            json!(presets
                .iter()
                .map(|preset| preset.preset_id.clone())
                .collect::<Vec<_>>()),
        );
    }

    // Per-image work items: (seed, prompt, action). Pose + angle sets share one seed (only the
    // pose changes across the set — noise-derived attributes stay constant); single identity is
    // per-seed at the reference's natural pose.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String, InstantIdAction)> = match &mode {
        InstantIdMode::PoseSet(_) => {
            let set_seed = resolve_seed(request, 0);
            parse_poses(request)
                .into_iter()
                .map(|pose| {
                    (
                        set_seed,
                        request.prompt.clone(),
                        InstantIdAction::Pose(pose_to_body_points(&pose.keypoints)),
                    )
                })
                .collect()
        }
        InstantIdMode::AngleSet => {
            let set_seed = resolve_seed(request, 0);
            // One image per preset in the active collection's order (sc-4450). Built-in presets
            // carry their canonical angle so the prompt still gets the per-angle clause; custom
            // presets render to their kps with the base prompt.
            let presets = angle_collection
                .as_ref()
                .map(|(_, presets)| presets.clone())
                .unwrap_or_else(builtin_angle_presets);
            presets
                .into_iter()
                .map(|preset| {
                    let prompt = match &preset.angle {
                        Some(angle) => augment_prompt_for_angle(&request.prompt, angle),
                        None => request.prompt.clone(),
                    };
                    (set_seed, prompt, InstantIdAction::Angle(preset.kps))
                })
                .collect()
        }
        InstantIdMode::Identity => (0..request.count as usize)
            .map(|index| {
                (
                    resolve_seed(request, index),
                    request.prompt.clone(),
                    InstantIdAction::Identity,
                )
            })
            .collect(),
    };
    let total = work.len();

    // Curated unified-sampler selection (epic 7114, sc-7432). InstantID builds its bespoke request
    // OUTSIDE base.rs's generic plumbing, so read the per-generation knob here and N3-normalize it
    // against the shared curated menu both engines honor — mlx #538 / candle #130 route a curated
    // solver/scheduler through the additive `denoise_curated` path; an unknown name drops back to the
    // engine default + emits an event rather than hard-failing `validate_request`. N1: with neither set
    // the request carries `None` ⇒ the bespoke ancestral default loop runs byte-for-byte unchanged.
    let (curated_samplers, curated_schedulers) = curated_image_menu();
    let (sampler, scheduler, _shift) = read_advanced_sampling_knobs(&request.advanced);
    let sampler = normalize_sampling_knob(
        sampler,
        &curated_samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &curated_schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );

    let negative_prompt = request.negative_prompt.clone();
    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "instantid",
        adapter_count,
        move || {
            let paths = InstantIdPaths {
                sdxl_base,
                identitynet: controlnet,
                ip_adapter,
                // User LoRA/LoKr adapters (sc-6038), resolved above and merged onto the SDXL UNet by
                // both engine lanes (mlx-gen #477 / candle-gen #86 both carry the field; worker mlx
                // pin now 19d5522, candle pin c98609f). Populated for BOTH backends — superseding the
                // earlier candle-only `Vec::new()` stopgap from #730.
                adapters,
            };
            let model = InstantId::load(&paths)
                .map_err(|error| WorkerError::Engine(format!("InstantID load failed: {error}")))?;
            // Attach OpenPose (pose mode) BEFORE quantize so it quantizes with the stack; quantize
            // before with_face (the engine's documented order). `with_openpose` is backend-neutral
            // (both engines take `&WeightsSource` and consume+return `self`).
            let model = match &openpose {
                Some(source) => model.with_openpose(source).map_err(|error| {
                    WorkerError::Engine(format!("InstantID OpenPose load failed: {error}"))
                })?,
                None => model,
            };
            // Quantization is an MLX-only knob — the candle InstantID stack runs dense f16 and has no
            // `quantize` method (the candle lane already forced `quant_bits` out, above).
            #[cfg(target_os = "macos")]
            let model = match quant_bits {
                Some(bits) => model.quantize(bits).map_err(|error| {
                    WorkerError::Engine(format!("InstantID quantize failed: {error}"))
                })?,
                None => model,
            };
            // Attach the SCRFD + ArcFace face stack. The MLX engine loads the two weight files
            // explicitly; the candle FaceEmbedder (sc-5490) loads the pair from THEIR DIRECTORY by the
            // canonical `scrfd_10g.safetensors` + `arcface_iresnet100.safetensors` names (exactly what
            // `ensure_instantid_weights` stages), so it takes the dir, not the two paths.
            #[cfg(target_os = "macos")]
            let model = {
                let scrfd = Weights::from_file(&scrfd_path).map_err(|error| {
                    WorkerError::Engine(format!("InstantID SCRFD weights {scrfd_path:?}: {error}"))
                })?;
                let arcface = Weights::from_file(&arcface_path).map_err(|error| {
                    WorkerError::Engine(format!(
                        "InstantID ArcFace weights {arcface_path:?}: {error}"
                    ))
                })?;
                model.with_face(&scrfd, &arcface).map_err(|error| {
                    WorkerError::Engine(format!("InstantID face stack: {error}"))
                })?
            };
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            let model = {
                let face_dir = scrfd_path.parent().unwrap_or(scrfd_path.as_path());
                // `arcface_path` is staged in the same dir; `with_face(dir)` resolves it by name.
                let _ = &arcface_path;
                model.with_face(face_dir).map_err(|error| {
                    WorkerError::Engine(format!("InstantID face stack: {error}"))
                })?
            };
            // Face-restore needs the reference identity embedding (imposed on the re-rendered crop).
            // Detect it once on the raw reference. The candle `largest_face` takes the neutral
            // `gen_core::Image`; the MLX engine takes raw RGB bytes + dims.
            let restore_embedding = if face_restore {
                #[cfg(target_os = "macos")]
                let embedding = model
                    .largest_face(
                        &reference.pixels,
                        reference.height as usize,
                        reference.width as usize,
                    )
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "InstantID face-restore reference: {error}"
                        ))
                    })?
                    .embedding;
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                let embedding = model
                    .largest_face(&reference)
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "InstantID face-restore reference: {error}"
                        ))
                    })?
                    .embedding;
                Some(embedding)
            } else {
                None
            };
            Ok((model, reference, restore_embedding))
        },
        move |(model, reference, restore_embedding), tx, cancel| {
            // The candle `generate*` / `restore_face` take `&mut self` (each call sets the face IP
            // tokens on the UNet before the denoise), so the per-item closure mutates `model`; the MLX
            // engine's are `&self`. Bind `mut` for the candle lane and allow the unused-mut on macOS.
            #[allow(unused_mut)]
            let mut model = model;
            drive_gen_items(
                tx,
                work,
                move |_index, (seed, prompt, action), on_progress| {
                    if cancel.is_cancelled() {
                        return Ok(None);
                    }
                    // Per-step progress → GenEvent::Step, so `consume_gen_events` streams step
                    // updates, fires `image_inference_start`, and polls the cancel API (sc-4382 —
                    // without Step events an InstantID job could never be cancelled).
                    // Angle + pose sets use a square canvas (the engine forces `req.height =
                    // req.width` for the canonical landmark/skeleton — the sc-2009 kps-aspect rule);
                    // single identity keeps the requested W×H (the engine letterboxes the reference).
                    let req = InstantIdRequest {
                        prompt,
                        negative: negative_prompt.clone(),
                        width,
                        height,
                        steps: steps as usize,
                        guidance,
                        ip_adapter_scale: ip_scale,
                        controlnet_scale,
                        openpose_scale,
                        seed: seed as u64,
                        sampler: sampler.clone(),
                        scheduler: scheduler.clone(),
                        cancel: cancel.clone(),
                    };
                    let result = match &action {
                        InstantIdAction::Identity => {
                            model.generate(&req, &reference, &mut *on_progress)
                        }
                        InstantIdAction::Angle(kps) => {
                            model.generate_with_kps(&req, &reference, kps, &mut *on_progress)
                        }
                        InstantIdAction::Pose(keypoints) => {
                            model.generate_pose(&req, &reference, keypoints, &mut *on_progress)
                        }
                    };
                    let mut out = match result {
                        Ok(out) => out,
                        // A cancel tripped mid-denoise surfaces as the engine's cancelled error —
                        // stop cleanly (consume_gen_events posts the Canceled update).
                        Err(_) if cancel.is_cancelled() => return Ok(None),
                        Err(error) => {
                            return Err(WorkerError::Engine(format!(
                                "InstantID generation failed: {error}"
                            )));
                        }
                    };
                    // Optional ADetailer-style face-restore re-render (sc-3380), imposing the
                    // reference identity on the cropped face with the gender-neutral restore prompt.
                    if let Some(embedding) = &restore_embedding {
                        let restore_req = InstantIdRequest {
                            prompt: FACE_RESTORE_PROMPT.to_owned(),
                            negative: negative_prompt.clone(),
                            width: INSTANTID_FACE_RESTORE_SIDE,
                            height: INSTANTID_FACE_RESTORE_SIDE,
                            steps: steps as usize,
                            guidance,
                            ip_adapter_scale: ip_scale,
                            controlnet_scale,
                            openpose_scale,
                            seed: seed as u64,
                            sampler: sampler.clone(),
                            scheduler: scheduler.clone(),
                            cancel: cancel.clone(),
                        };
                        out = match model.restore_face(
                            &restore_req,
                            &out,
                            embedding,
                            &mut *on_progress,
                        ) {
                            Ok(out) => out,
                            Err(_) if cancel.is_cancelled() => return Ok(None),
                            Err(error) => {
                                return Err(WorkerError::InvalidPayload(format!(
                                    "InstantID face-restore failed: {error}"
                                )))
                            }
                        };
                    }
                    Ok(Some((seed, out.width, out.height, out.pixels)))
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
        INSTANTID_ENGINE,
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
// Tile-ControlNet detail refine (macOS, epic 3041 / sc-3060): the standalone
// `image_detail` job (Image Editor, epic 2427). Faithful port of the Python
// `run_image_detail` + `_refine_tiled_detail` (image_adapters.py) onto the engine's
// SDXL tile-ControlNet path: each tile is img2img-refined with itself as both the
// init (Reference) and the ControlNet conditioning (Control, control=same), then
// recomposed with a raised-cosine feather over the overlap. Unlike the diffusers
// pipeline, the engine requires width/height ∈ [512, 2048] and multiples of 8, so a
// tile is run at the nearest valid size and the result resized back before blending.
// ---------------------------------------------------------------------------
