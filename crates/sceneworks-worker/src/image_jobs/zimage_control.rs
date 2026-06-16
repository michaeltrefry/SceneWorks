// Candle (Windows/CUDA) Z-Image Fun-ControlNet (strict pose) route (sc-5489, epic 5480) —
// `z_image_turbo` + `advanced.poses` off-Mac via `candle_gen_z_image::ZImageControl`. The LAST family
// of the sc-5489 3-family ControlNet port (after Qwen + Kolors). The candle sibling of the MLX Z-Image
// strict-pose path (zimage.rs `generate_zimage_control_stream`): one image per pose, each conditioned
// on a full DWPose skeleton (rendered cross-platform by `openpose_skeleton::draw_wholebody`) fed to the
// VACE-style `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` branch.
//
// **Candle-only.** macOS keeps the MLX `z_image_turbo_control` registry generator (zimage.rs); the
// candle `ZImageControl` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle
// build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs`
// module, so it shares that module's imports (`parse_poses`/`pose_entries`/`Settings`/`WorkerResult`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// Default Fun-Controlnet-Union weights — the **8-step** variant the MLX path uses (zimage.rs
/// `ZIMAGE_CONTROL_FILE`); the candle `ZImageControl::generate` runs the matching 8-step schedule.
const ZIMAGE_CTRL_REPO: &str = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1";
const ZIMAGE_CTRL_FILE: &str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors";
/// The Z-Image base diffusers repo when the manifest omits `repo`.
const ZIMAGE_CTRL_DEFAULT_REPO: &str = "Tongyi-MAI/Z-Image-Turbo";
/// ControlNet conditioning-scale default (the strict-pose tier).
const ZIMAGE_CTRL_DEFAULT_SCALE: f32 = 1.0;
/// Denoise-steps default — the 8-step Fun-ControlNet variant (vs the 4-step distilled txt2img).
const ZIMAGE_CTRL_DEFAULT_STEPS: u32 = 8;
/// The adapter/engine id recorded on candle Z-Image control assets (distinct from the txt2img
/// `candle_z_image`/`candle_zimage` lane).
const ZIMAGE_CTRL_ENGINE: &str = "candle_zimage_control";

/// Model ids the candle Z-Image ControlNet route accepts.
fn is_zimage_control_model(model: &str) -> bool {
    model == "z_image_turbo"
}

/// Resolve the Z-Image base (diffusers) snapshot: an explicit `modelPath` (advanced or manifest) → the
/// HF cache snapshot for the manifest `repo` (default `Tongyi-MAI/Z-Image-Turbo`). `None` ⇒ not present
/// locally (the job is not candle-runnable, falls through to torch). Mirrors `resolve_kolors_control_base`.
fn resolve_zimage_control_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Z-Image control modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(ZIMAGE_CTRL_DEFAULT_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible Z-Image strict-pose job: `z_image_turbo` with a non-empty
/// `advanced.poses`, not edit mode, whose base resolves locally. Mirrors
/// `jobs_store::zimage_control_candle_eligible` so the worker and router agree.
fn zimage_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_zimage_control_model(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && matches!(resolve_zimage_control_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=50) → manifest `steps` → default (8).
fn zimage_control_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or(ZIMAGE_CTRL_DEFAULT_STEPS)
}

/// The (repo, filename) of the ControlNet weights — `advanced.controlWeights.{repo,filename}` overrides,
/// else the Fun-Controlnet-Union 8-step default (parity with the MLX `resolve_control_weights`).
fn zimage_control_repo_file(request: &ImageRequest) -> (String, String) {
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
        pick("repo", ZIMAGE_CTRL_REPO),
        pick("filename", ZIMAGE_CTRL_FILE),
    )
}

/// Resolve the Fun-Controlnet-Union weight **file** the `ZImageControl` provider loads, downloading on
/// first use. Order: an env-pinned file (`SCENEWORKS_CONTROLNET_ZIMAGE`) → a whole-repo HF cache
/// snapshot → download into the app cache. Mirrors `ensure_kolors_control_weights`.
async fn ensure_zimage_control_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<PathBuf> {
    let (repo, file) = zimage_control_repo_file(request);
    if let Ok(p) = std::env::var("SCENEWORKS_CONTROLNET_ZIMAGE") {
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
        cancel_message: "Z-Image strict-pose generation canceled while fetching control weights.",
        fresh_download: false,
    };
    let dst = settings
        .data_dir
        .join("cache")
        .join("controlnet-zimage")
        .join(&file);
    ensure_hf_cached_file(&context, &repo, "main", &file, &dst).await?;
    Ok(dst)
}

/// Flat telemetry recorded on candle Z-Image control assets. No guidance — Z-Image-Turbo is
/// guidance-distilled.
fn zimage_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw.insert(
        "controlEngine".to_owned(),
        Value::String(ZIMAGE_CTRL_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Z-Image strict-pose generation: one image per pose, each conditioned on a full DWPose
/// skeleton. The provider loads once on the blocking thread; each pose renders its skeleton + generates.
/// Z-Image-Turbo is distilled (no CFG / negative prompt), so the request carries no guidance.
/// `ZImageControl::generate` takes `&self`, so the per-item closure does not need `&mut`.
async fn generate_candle_zimage_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let base = resolve_zimage_control_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Z-Image base (Z-Image-Turbo) weights not found".to_owned())
    })?;
    let controlnet = ensure_zimage_control_weights(api, settings, job, request).await?;

    let steps = zimage_control_steps(request);
    let control_scale = advanced::f32_clamped(
        &request.advanced,
        "controlScale",
        ZIMAGE_CTRL_DEFAULT_SCALE,
        0.0..=2.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(ZIMAGE_CTRL_DEFAULT_REPO)
        .to_owned();

    let poses = parse_poses(request);
    let pose_count = poses.len();
    let raw_settings = zimage_control_raw_settings(request, &repo, steps, control_scale, pose_count);

    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    // One shared seed across the pose set (the MLX `_generate_pose_set` convention).
    let seed = resolve_seed(request, 0);
    let prompt = request.prompt.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "zimage_control",
        0,
        move || {
            let paths = ZImageControlPaths { base, control: controlnet };
            let model = ZImageControl::load(&paths).map_err(|error| {
                WorkerError::Engine(format!("Z-Image strict-pose control load failed: {error}"))
            })?;
            Ok((model, poses))
        },
        move |(model, poses), tx, cancel| {
            drive_gen_items(tx, poses, move |_index, pose, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
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
                let req = ZImageControlRequest {
                    prompt: prompt.clone(),
                    width,
                    height,
                    steps: steps as usize,
                    control_scale,
                    seed: seed as u64,
                    cancel: cancel.clone(),
                };
                let out = match model.generate(&req, &control, &mut *on_progress) {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "Z-Image strict-pose generation failed: {error}"
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
        ZIMAGE_CTRL_ENGINE,
        &raw_settings,
        pose_count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
