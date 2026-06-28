// Candle (Windows/CUDA) SDXL IP-Adapter-Plus reference route (sc-5488, epic 5480) — reference-image
// (identity) conditioning on SDXL/RealVisXL off-Mac via `candle_gen_sdxl::IpAdapterSdxl`. The
// reference-conditioning sibling of the candle InstantID lane (instantid.rs), but plain SDXL: no face
// stack, no IdentityNet/OpenPose ControlNet — just the CLIP ViT-H image tokens → pure-IP denoise.
//
// **Candle-only.** macOS keeps the MLX SDXL IP path (the registry `SdxlSubMode::Ip` in sdxl.rs); there
// is no MLX `IpAdapterSdxl`, so this whole file is gated to the Windows/CUDA candle build (the
// `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it
// shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// h94 IP-Adapter repo (the ViT-H encoder + the plus/plus-face SDXL weights), matching the MLX SDXL IP
/// path's `SDXL_IP_ADAPTER_REPO`.
const SDXL_IPADAPTER_REPO: &str = "h94/IP-Adapter";
/// The IP-Adapter-Plus (ViT-H) bundle inside the repo (`image_proj` Resampler + `ip_adapter.*` K/V).
const SDXL_IPADAPTER_BUNDLE_SRC: &str = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors";
/// The CLIP ViT-H image-encoder files inside the repo (config + weights).
const SDXL_IPADAPTER_ENCODER_SRC: [&str; 2] = [
    "models/image_encoder/config.json",
    "models/image_encoder/model.safetensors",
];
/// IP-Adapter scale default — torch plus parity (matches the MLX SDXL path's `SDXL_IP_SCALE`).
const SDXL_IPADAPTER_IP_SCALE: f32 = 0.7;
/// Denoise steps default (SDXL production).
const SDXL_IPADAPTER_DEFAULT_STEPS: u32 = 30;
/// CFG default — the reference-conditioned envelope validated on GPU (sc-5488); base SDXL uses ~7.
const SDXL_IPADAPTER_DEFAULT_GUIDANCE: f32 = 5.0;
/// The adapter/engine id recorded on candle SDXL IP-Adapter assets + telemetry (distinct from the
/// txt2img `candle_sdxl` and the `candle_instantid` lanes).
const SDXL_IPADAPTER_ENGINE: &str = "candle_sdxl_ipadapter";

/// SDXL model ids the candle IP-Adapter route accepts (the txt2img-eligible SDXL family).
fn is_sdxl_ipadapter_model(model: &str) -> bool {
    matches!(model, "sdxl" | "realvisxl")
}

/// Default SDXL base repo for a model id when the manifest omits `repo`.
fn sdxl_ipadapter_default_repo(model: &str) -> &'static str {
    match model {
        "realvisxl" => "SG161222/RealVisXL_V5.0",
        _ => "stabilityai/stable-diffusion-xl-base-1.0",
    }
}

/// Resolve the SDXL base snapshot for the IP-Adapter route: an explicit `modelPath` dir (advanced or
/// manifest) wins, else the HF cache snapshot for the manifest `repo` (default by model id). `None`
/// means the base is not present locally, so the job is not candle-runnable (falls through to torch).
fn resolve_sdxl_ipadapter_base(
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
        return resolve_app_managed_model_dir(settings, &path, "SDXL IP-Adapter modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| sdxl_ipadapter_default_repo(&request.model));
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible SDXL IP-Adapter job: an sdxl-family model with a reference image
/// (and NOT an img2img/inpaint/edit shape — those advanced SDXL modes are sc-5487) whose base resolves
/// locally. Mirrors `jobs_store::sdxl_ipadapter_candle_eligible` so the worker and router agree.
fn sdxl_ipadapter_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sdxl_ipadapter_model(&request.model)
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !non_empty(&request.source_asset_id)
        && !non_empty(&request.mask_asset_id)
        && matches!(resolve_sdxl_ipadapter_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → default (30).
fn sdxl_ipadapter_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or(SDXL_IPADAPTER_DEFAULT_STEPS)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → the reference-tuned default
/// (5.0), clamped to a sane CFG range.
fn sdxl_ipadapter_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(SDXL_IPADAPTER_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Resolve the IP-Adapter bundle file + the CLIP ViT-H image-encoder dir, downloading from
/// `h94/IP-Adapter` on first use. Resolution order: an env-pinned root (pre-staged, the validation
/// path) → a whole-repo HF cache snapshot → download the bundle + encoder into the app cache. Returns
/// `(bundle_file, image_encoder_dir)` — what [`IpAdapterSdxlPaths`] wants.
async fn ensure_sdxl_ipadapter_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(PathBuf, PathBuf)> {
    // Env override: a directory laid out like the h94 repo (or its HF snapshot), pre-staged for local
    // validation (`SCENEWORKS_IPADAPTER_SDXL`).
    if let Ok(root) = std::env::var("SCENEWORKS_IPADAPTER_SDXL") {
        let root = PathBuf::from(root);
        let bundle = root.join("sdxl_models").join("ip-adapter-plus_sdxl_vit-h.safetensors");
        let encoder = root.join("models").join("image_encoder");
        if bundle.is_file() && encoder.is_dir() {
            return Ok((bundle, encoder));
        }
    }
    // Whole-repo HF cache snapshot already present (the model-download flow staged it).
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, SDXL_IPADAPTER_REPO) {
        let bundle = snapshot
            .join("sdxl_models")
            .join("ip-adapter-plus_sdxl_vit-h.safetensors");
        let encoder = snapshot.join("models").join("image_encoder");
        if bundle.is_file() && encoder.join("model.safetensors").is_file() {
            return Ok((bundle, encoder));
        }
    }
    // Download-on-first-use into the app cache (flat dest, nested source — the InstantID bundle pattern).
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "SDXL IP-Adapter generation canceled while fetching weights.",
        fresh_download: false,
    };
    let cache = settings.data_dir.join("cache").join("ipadapter-sdxl");
    let bundle = ensure_hf_cached_file(
        &context,
        SDXL_IPADAPTER_REPO,
        "main",
        SDXL_IPADAPTER_BUNDLE_SRC,
        &cache.join("ip-adapter-plus_sdxl_vit-h.safetensors"),
    )
    .await?;
    let encoder = cache.join("image_encoder");
    for source in SDXL_IPADAPTER_ENCODER_SRC {
        let name = source.rsplit('/').next().unwrap_or(source);
        ensure_hf_cached_file(
            &context,
            SDXL_IPADAPTER_REPO,
            "main",
            source,
            &encoder.join(name),
        )
        .await?;
    }
    Ok((bundle, encoder))
}

/// Flat telemetry recorded on candle SDXL IP-Adapter assets (parity with the InstantID/`mlx_raw_settings`
/// recipe-key shape).
fn sdxl_ipadapter_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    ip_scale: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("ipAdapterScale".to_owned(), json!(ip_scale));
    raw.insert(
        "ipAdapterEngine".to_owned(),
        Value::String(SDXL_IPADAPTER_ENGINE.to_owned()),
    );
    raw
}

/// Real candle SDXL IP-Adapter generation: resolve the reference + weights on the async side, then load
/// the `IpAdapterSdxl` provider once + generate each image on the blocking thread. `request.count`
/// images, each its own seed; the engine `generate` takes the per-job `CancelFlag` + a `Progress`
/// callback, so streaming is per-step and cancellation is honoured mid-denoise — same contract as the
/// registry families + the InstantID lane. Reuses [`consume_gen_events`] for the asset writes.
async fn generate_candle_sdxl_ipadapter_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let sdxl_base = resolve_sdxl_ipadapter_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("SDXL IP-Adapter base (SDXL/RealVisXL) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("SDXL IP-Adapter requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let (ip_bundle, image_encoder) = ensure_sdxl_ipadapter_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle SDXL IP-Adapter
    // lane is the With-Character route for an SDXL-family model — score every output against the
    // reference face through the SHARED generator-agnostic seam, but only for an Image Studio "With
    // Character" (`character_image`) generation (a non-character reference job records no identity score).
    // Stage the antelopev2 SCRFD + ArcFace bundle (the scorer's candle leg loads it); the `!Send` scorer
    // is built ONCE inside the load closure and reused across the N outputs (source embedded once — the
    // caching AC). The source is the CURRENT job's `referenceAssetId`. Staging is non-fatal (failure → no
    // scorer → scores omitted, generation still renders).
    let score_likeness = request.mode == "character_image";
    let face_stack_dir = if score_likeness {
        match ensure_face_stack_dir(api, settings, job).await {
            Ok(dir) => Some(dir),
            Err(error) => {
                tracing::warn!(error = %error, "character_image face-stack staging failed; likeness scores omitted");
                None
            }
        }
    } else {
        None
    };
    let likeness_source = face_stack_dir.as_ref().map(|_| reference.clone());
    let likeness_source_ref = reference_id.to_owned();

    let steps = sdxl_ipadapter_steps(request);
    let guidance = sdxl_ipadapter_guidance(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        SDXL_IPADAPTER_IP_SCALE,
        0.0..=1.0,
    );
    // Curated unified-sampler selection (epic 7114, sc-7432): the candle `IpAdapterSdxl` provider honors
    // a curated solver/scheduler via the shared `denoise_curated` primitive (#130). Read + N3-normalize
    // against the shared curated menu (an unknown name drops to the engine default + emits an event). N1:
    // unset ⇒ `None` ⇒ the native ancestral default loop runs byte-exact. (`sdxl`/`realvisxl` are
    // MODEL_TABLE rows already advertising the curated menu — guarded by the existing drift guard.)
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
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| sdxl_ipadapter_default_repo(&request.model))
        .to_owned();
    let raw_settings = sdxl_ipadapter_raw_settings(request, &repo, steps, guidance, ip_scale);

    // Per-image work items: (seed, prompt) — `request.count` images at the reference identity.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative_prompt = request.negative_prompt.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "sdxl_ipadapter",
        0,
        move || {
            let paths = IpAdapterSdxlPaths {
                sdxl_base,
                ip_adapter: ip_bundle,
                image_encoder,
            };
            let model = IpAdapterSdxl::load(&paths).map_err(|error| {
                WorkerError::Engine(format!("SDXL IP-Adapter load failed: {error}"))
            })?;
            // Per-job identity-likeness scorer built ONCE here (on the blocking thread where the `!Send`
            // face stack is allowed); source embedded once, reused across every output (sc-4411 caching
            // AC). `None` ⇒ non-fatal staging / construction failure ⇒ scores omitted.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            Ok((model, reference, scorer))
        },
        move |(model, reference, scorer), tx, cancel| {
            // `IpAdapterSdxl::generate` takes `&mut self` (it sets the IP image tokens on the UNet before
            // the denoise), so the per-item closure mutates `model`.
            let mut model = model;
            drive_gen_items_scored(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = IpAdapterSdxlRequest {
                    prompt,
                    negative: negative_prompt.clone(),
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    ip_adapter_scale: ip_scale,
                    seed: seed as u64,
                    sampler: sampler.clone(),
                    scheduler: scheduler.clone(),
                    cancel: cancel.clone(),
                };
                let out = match model.generate(&req, &reference, &mut *on_progress) {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "SDXL IP-Adapter generation failed: {error}"
                        )));
                    }
                };
                // Score this finished image against the cached source embedding (sc-4411). Image build +
                // pixel clone paid ONLY when a scorer exists; non-frontal → honest detected:false N/A;
                // `None` scorer ⇒ field omitted.
                let face_likeness = scorer.as_ref().and_then(|scorer| {
                    crate::face_likeness::score_generated_image(
                        Some(scorer),
                        &Image {
                            width: out.width,
                            height: out.height,
                            pixels: out.pixels.clone(),
                        },
                        Some(likeness_source_ref.as_str()),
                    )
                });
                Ok(Some((seed, out.width, out.height, out.pixels, face_likeness)))
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
        SDXL_IPADAPTER_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
