// Candle (Windows/CUDA) Qwen-Image 2512-Fun-Controlnet-Union (strict control) route (sc-5489 origin /
// sc-8350 repoint, epic 8236) — `qwen_image` + `advanced.poses` off-Mac via
// `candle_gen_qwen_image::QwenFunControl`. The candle sibling of the MLX Qwen 2512-Fun strict-control path
// (qwen.rs `generate_qwen_control_stream`): one image per pose (or, with `advanced.controlMode =
// canny|depth` + a source, an auto-derived canny / Depth-Anything-V2 map), each fed to the VACE-style
// `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` branch overlaid on the Qwen-Image-2512 base.
//
// **sc-8350 source swap.** This lane previously loaded the InstantX `Qwen-Image-ControlNet-Union`
// checkpoint (`QwenControl`, a residual-ControlNet on the `Qwen/Qwen-Image` base). It now rides the
// 2512-Fun-Union VACE engine (`QwenFunControl`) on the `Qwen/Qwen-Image-2512` base — input-agnostic
// (pose/canny/depth, no mode index), matching the `STRICT_CONTROL_ENGINES` `qwen_image_control` row. The
// candle-gen InstantX `control.rs` engine (`QwenControl`) stays in the crate but is no longer used by the
// worker.
//
// **Candle-only.** macOS keeps the MLX `qwen_image_control` registry generator; the candle `QwenFunControl`
// is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build (the `include!` in
// image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it shares that
// module's imports (`parse_poses`/`Settings`/`WorkerResult`/`huggingface_snapshot_dir`/
// `ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// Default 2512-Fun-Controlnet-Union weights (Apache-2.0, input-agnostic VACE control) — same repo/file
/// the MLX path uses (`qwen.rs` 2512-Fun constants); the candle lane keeps its own constant (sc-8350).
const QWEN_CONTROL_REPO: &str = "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union";
const QWEN_CONTROL_FILE: &str = "diffusion_pytorch_model.safetensors";
/// The Qwen-Image-2512 base diffusers repo when the manifest omits `repo` (the 2512-Fun base, sc-8350).
const QWEN_CONTROL_DEFAULT_REPO: &str = "Qwen/Qwen-Image-2512";
/// ControlNet conditioning-scale default (the strict-pose tier).
const QWEN_CONTROL_DEFAULT_SCALE: f32 = 1.0;
/// Denoise-steps default (Qwen-Image production).
const QWEN_CONTROL_DEFAULT_STEPS: u32 = 30;
/// CFG default.
const QWEN_CONTROL_DEFAULT_GUIDANCE: f32 = 4.0;
/// The adapter/engine id recorded on candle Qwen control assets (distinct from the txt2img
/// `candle_qwen` lane).
const QWEN_CONTROL_ENGINE: &str = "candle_qwen_control";
/// The [`STRICT_CONTROL_ENGINES`] catalog id this candle lane validates `advanced.controlMode` against
/// (the `qwen_image_control` row — `{Pose, Canny, Depth}`). As of sc-8350 the candle lane loads the
/// 2512-Fun-Controlnet-Union checkpoint on the Qwen-Image-2512 base (`QwenFunControl`), matching the
/// table's `qwen_image_control` repo (`alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`) exactly.
const QWEN_CONTROL_ENGINE_ID: &str = "qwen_image_control";

/// Model ids the candle Qwen ControlNet route accepts.
fn is_qwen_control_model(model: &str) -> bool {
    model == "qwen_image"
}

/// Resolve the Qwen-Image-2512 base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) →
/// the HF cache snapshot for the manifest `repo` (default `Qwen/Qwen-Image-2512`, sc-8350). `None` ⇒ not
/// present locally (the job is not candle-runnable, falls through to torch). Mirrors
/// `resolve_kolors_ipadapter_base`.
fn resolve_qwen_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Qwen control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(QWEN_CONTROL_DEFAULT_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible Qwen strict-pose job: `qwen_image` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::qwen_control_candle_eligible` so the worker and router agree.
fn qwen_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_qwen_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_qwen_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=100) → manifest `steps` → default (30).
fn qwen_control_steps(request: &ImageRequest) -> u32 {
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
        .map(|steps| steps.clamp(1, 100) as u32)
        .unwrap_or(QWEN_CONTROL_DEFAULT_STEPS)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (4.0), clamped.
fn qwen_control_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(QWEN_CONTROL_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// The (repo, filename) of the ControlNet weights — `advanced.controlWeights.{repo,filename}` overrides,
/// else the InstantX Union default (parity with the MLX `resolve_control_weights_for`).
fn qwen_control_repo_file(request: &ImageRequest) -> (String, String) {
    let cw = request.advanced.get("controlWeights").and_then(Value::as_object);
    let pick = |key: &str, default: &str| {
        cw.and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    (
        pick("repo", QWEN_CONTROL_REPO),
        pick("filename", QWEN_CONTROL_FILE),
    )
}

/// Resolve the 2512-Fun-Controlnet-Union weight **file** the `QwenFunControl` provider loads (sc-8350),
/// downloading on first use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_QWEN`) → a whole-repo HF
/// cache snapshot → download into the app cache.
async fn ensure_qwen_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = qwen_control_repo_file(request);
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_QWEN") {
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
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Qwen strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-qwen")
        .join(&file);
    ensure_hf_cached_file(&context, &repo, "main", &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Qwen control assets.
fn qwen_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(QWEN_CONTROL_ENGINE.to_owned()),
    );
    raw
}

/// The per-lane half of the candle Qwen 2512-Fun strict-control [`CandleStrictControl`] driver (sc-8304 /
/// sc-8350): the resolved base + 2512-Fun-Union control weight paths + the request numerics. Qwen runs
/// true CFG, so it carries a negative prompt + guidance. Moved onto the blocking thread, loaded once,
/// drives every pose.
struct QwenStrictControl {
    qwen_base: PathBuf,
    controlnet: PathBuf,
    prompt: String,
    negative: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    control_scale: f32,
}

impl CandleStrictControl for QwenStrictControl {
    type Model = QwenFunControl;

    fn engine_id(&self) -> &'static str {
        QWEN_CONTROL_ENGINE_ID
    }

    fn engine_label(&self) -> &'static str {
        QWEN_CONTROL_ENGINE
    }

    fn stream_tag(&self) -> &'static str {
        "qwen_control"
    }

    fn load(&self) -> WorkerResult<Self::Model> {
        let paths = QwenFunControlPaths {
            qwen_base: self.qwen_base.clone(),
            controlnet: self.controlnet.clone(),
        };
        QwenFunControl::load(&paths).map_err(|error| {
            WorkerError::Engine(format!("Qwen 2512-Fun strict-control load failed: {error}"))
        })
    }

    fn generate_one(
        &self,
        model: &Self::Model,
        control: &Image,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> WorkerResult<Image> {
        let req = QwenFunControlRequest {
            prompt: self.prompt.clone(),
            negative: self.negative.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps as usize,
            guidance: self.guidance,
            control_scale: self.control_scale,
            seed,
            cancel: cancel.clone(),
        };
        model.generate(&req, control, on_progress).map_err(|error| {
            WorkerError::Engine(format!("Qwen 2512-Fun strict-control generation failed: {error}"))
        })
    }
}

/// Real candle Qwen strict-pose generation: one image per pose, each conditioned on a full DWPose skeleton
/// (`controlMode` unset) or a canny/depth control map. Resolves the base + InstantX control weights, then
/// hands a [`QwenStrictControl`] to the shared [`run_candle_strict_control`] driver (validation against
/// `qwen_image_control`'s `supported_kinds`, per-pose preprocessing, scoring). `generate` takes the
/// per-job `CancelFlag` + a `Progress` callback (per-step streaming + mid-denoise cancel). The pose path
/// is byte-preserved.
async fn generate_candle_qwen_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let qwen_base = resolve_qwen_control_base(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Qwen-Image base weights not found".to_owned()))?;
    let controlnet = ensure_qwen_control_weights(api, settings, job, request).await?;

    let steps = qwen_control_steps(request);
    let guidance = qwen_control_guidance(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        QWEN_CONTROL_DEFAULT_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(QWEN_CONTROL_DEFAULT_REPO)
        .to_owned();

    let pose_count = pose_entries(request).len();
    let raw_settings =
        qwen_control_raw_settings(request, &repo, steps, guidance, control_scale, pose_count);

    let provider = QwenStrictControl {
        qwen_base,
        controlnet,
        prompt: request.prompt.clone(),
        negative: request.negative_prompt.clone(),
        width: request.width,
        height: request.height,
        steps,
        guidance,
        control_scale,
    };

    run_candle_strict_control(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        provider,
        raw_settings,
        asset_writes,
    )
    .await
}
