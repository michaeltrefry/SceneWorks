// Candle (Windows/CUDA) Kolors IP-Adapter-Plus reference route (sc-5488, epic 5480) — reference-image
// (identity) conditioning on Kolors off-Mac via `candle_gen_kolors::IpAdapterKolors`. The Kolors sibling
// of the candle SDXL IP-Adapter lane (sdxl_ipadapter.rs): CLIP ViT-L/14-336 image tokens injected into
// the vendored SDXL UNet alongside the encoder_hid_proj-projected ChatGLM3 text path, denoised with the
// Kolors leading-Euler sampler.
//
// **Candle-only.** macOS keeps the MLX Kolors IP path (the `Reference` conditioning the registry
// `kolors` generator handles once `with_ip_adapter` installs the K/V — kolors.rs, sc-4767); the candle
// `IpAdapterKolors` is a bespoke provider, so this whole file is gated to the Windows/CUDA candle build
// (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so
// it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// The Kolors IP-Adapter-Plus repo (CLIP ViT-L/14-336 `image_encoder/` + the `image_proj` Resampler +
/// `ip_adapter.*` K/V). Same repo the MLX path uses (`kolors.rs` `KOLORS_IP_ADAPTER_REPO`).
const KOLORS_IPADAPTER_REPO: &str = "Kwai-Kolors/Kolors-IP-Adapter-Plus";
/// The repo revision carrying the **safetensors** (`refs/pr/4`): the repo's `main` ships only `.bin`
/// (a torch pickle candle can't read); PR-4 adds `ip_adapter_plus_general.safetensors` +
/// `image_encoder/model.safetensors`. The torch/MLX download flow uses this same rev.
const KOLORS_IPADAPTER_REVISION: &str = "refs/pr/4";
/// The IP-Adapter-Plus bundle file (root of the snapshot).
const KOLORS_IPADAPTER_BUNDLE: &str = "ip_adapter_plus_general.safetensors";
/// The CLIP ViT-L/14-336 image-encoder files inside the repo (config + weights).
const KOLORS_IPADAPTER_ENCODER_SRC: [&str; 2] = [
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
];
/// IP-Adapter scale default — the torch `KolorsDiffusersAdapter._ip_adapter_scale` 0.6 (matches the MLX
/// path's `KOLORS_IP_SCALE`, and the candle `IpAdapterKolors::DEFAULT_IP_ADAPTER_SCALE`).
const KOLORS_IPADAPTER_IP_SCALE: f32 = 0.6;
/// Denoise steps default (Kolors production — diffusers `KolorsPipeline`).
const KOLORS_IPADAPTER_DEFAULT_STEPS: u32 = 50;
/// CFG default (Kolors production guidance).
const KOLORS_IPADAPTER_DEFAULT_GUIDANCE: f32 = 5.0;
/// The Kolors base diffusers repo when the manifest omits `repo`.
const KOLORS_IPADAPTER_DEFAULT_REPO: &str = "Kwai-Kolors/Kolors-diffusers";
/// The adapter/engine id recorded on candle Kolors IP-Adapter assets + telemetry (distinct from the
/// txt2img `candle_kolors` lane).
const KOLORS_IPADAPTER_ENGINE: &str = "candle_kolors_ipadapter";

/// Model ids the candle Kolors IP-Adapter route accepts.
fn is_kolors_ipadapter_model(model: &str) -> bool {
    model == "kolors"
}

/// Resolve the Kolors base (diffusers) snapshot for the IP-Adapter route: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (default
/// `Kwai-Kolors/Kolors-diffusers`). `None` means the base is not present locally, so the job is not
/// candle-runnable (falls through to torch). Mirrors `resolve_sdxl_ipadapter_base`.
fn resolve_kolors_ipadapter_base(
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
        return resolve_app_managed_model_dir(settings, &path, "Kolors IP-Adapter modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(KOLORS_IPADAPTER_DEFAULT_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible Kolors IP-Adapter job: the `kolors` model with a reference image
/// (and NOT an img2img/inpaint/edit shape — those are sc-5487) whose base resolves locally. Mirrors
/// `jobs_store::kolors_ipadapter_candle_eligible` so the worker and router agree.
fn kolors_ipadapter_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_kolors_ipadapter_model(&request.model)
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !non_empty(&request.source_asset_id)
        && !non_empty(&request.mask_asset_id)
        && matches!(resolve_kolors_ipadapter_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → default (50).
fn kolors_ipadapter_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or(KOLORS_IPADAPTER_DEFAULT_STEPS)
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → default (5.0), clamped.
fn kolors_ipadapter_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(KOLORS_IPADAPTER_DEFAULT_GUIDANCE);
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Resolve the Kolors IP-Adapter-Plus snapshot **directory** (`image_encoder/` + the bundle file) the
/// `IpAdapterKolors` provider loads, downloading from `Kwai-Kolors/Kolors-IP-Adapter-Plus` **@
/// `refs/pr/4`** on first use. Resolution order: an env-pinned root (pre-staged, the validation path) →
/// a whole-repo HF cache snapshot → download the bundle + encoder into the app cache. Returns the dir
/// (what [`IpAdapterKolorsPaths::ip_adapter`] wants).
async fn ensure_kolors_ipadapter_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    let has_layout = |dir: &Path| {
        dir.join(KOLORS_IPADAPTER_BUNDLE).is_file()
            && dir.join("image_encoder").join("model.safetensors").is_file()
    };
    // Env override: a directory laid out like the snapshot, pre-staged for local validation.
    if let Ok(root) = std::env::var("SCENEWORKS_IPADAPTER_KOLORS") {
        let root = PathBuf::from(root);
        if has_layout(&root) {
            return Ok(root);
        }
    }
    // Whole-repo HF cache snapshot already present (the model-download flow staged it at refs/pr/4).
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, KOLORS_IPADAPTER_REPO) {
        if has_layout(&snapshot) {
            return Ok(snapshot);
        }
    }
    // Download-on-first-use into the app cache (the snapshot layout: bundle at root + image_encoder/).
    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Kolors IP-Adapter generation canceled while fetching weights.",
        fresh_download: false,
    };
    let cache = settings.data_dir.join("cache").join("ipadapter-kolors");
    ensure_hf_cached_file(
        &context,
        KOLORS_IPADAPTER_REPO,
        KOLORS_IPADAPTER_REVISION,
        KOLORS_IPADAPTER_BUNDLE,
        &cache.join(KOLORS_IPADAPTER_BUNDLE),
    )
    .await?;
    for source in KOLORS_IPADAPTER_ENCODER_SRC {
        let name = source.rsplit('/').next().unwrap_or(source);
        ensure_hf_cached_file(
            &context,
            KOLORS_IPADAPTER_REPO,
            KOLORS_IPADAPTER_REVISION,
            source,
            &cache.join("image_encoder").join(name),
        )
        .await?;
    }
    Ok(cache)
}

/// Flat telemetry recorded on candle Kolors IP-Adapter assets (parity with the SDXL/InstantID recipe-key
/// shape).
fn kolors_ipadapter_raw_settings(
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
        Value::String(KOLORS_IPADAPTER_ENGINE.to_owned()),
    );
    raw
}

/// Real candle Kolors IP-Adapter generation: resolve the reference + weights on the async side, then
/// load the `IpAdapterKolors` provider once + generate each image on the blocking thread. `request.count`
/// images, each its own seed; `generate` takes the per-job `CancelFlag` + a `Progress` callback, so
/// streaming is per-step and cancellation is honoured mid-denoise — same contract as the SDXL IP lane.
async fn generate_candle_kolors_ipadapter_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let kolors_base = resolve_kolors_ipadapter_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("Kolors IP-Adapter base (Kolors-diffusers) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Kolors IP-Adapter requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let ip_adapter = ensure_kolors_ipadapter_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle Kolors IP-Adapter
    // lane is the With-Character route for a Kolors model — score every output against the reference face
    // through the SHARED generator-agnostic seam. Eligibility goes through `resolve_character_image_
    // likeness_source` (the SAME gate the macOS lanes use), so the angle/pose/edit exclusion is explicit
    // and self-contained here — NOT dependent on dispatch order (an angle/pose job is excluded by the
    // helper even if it ever reached this lane, so it can never be double-scored). The helper's decode is
    // ignored: the already-decoded `reference` (this lane's generation input, the current job's
    // `referenceAssetId`) is the scorer source, so there is no second decode. Stage the antelopev2 SCRFD +
    // ArcFace bundle; the `!Send` scorer is built ONCE inside the load closure and reused across the N
    // outputs (source embedded once). Staging is non-fatal (failure → scores omitted).
    let score_likeness =
        resolve_character_image_likeness_source(request, settings, project_path).is_some();
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

    let steps = kolors_ipadapter_steps(request);
    let guidance = kolors_ipadapter_guidance(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        KOLORS_IPADAPTER_IP_SCALE,
        0.0..=1.0,
    );
    // Curated unified-sampler selection (epic 7114, sc-7432): the candle `IpAdapterKolors` provider
    // honors a curated solver/scheduler via the shared `denoise_curated` primitive (#130). Read +
    // N3-normalize against the shared curated menu. N1: unset ⇒ `None` ⇒ the native default, byte-exact.
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
        .unwrap_or(KOLORS_IPADAPTER_DEFAULT_REPO)
        .to_owned();
    let raw_settings = kolors_ipadapter_raw_settings(request, &repo, steps, guidance, ip_scale);

    // Per-image work items: (seed, prompt) — `request.count` images at the reference identity.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let negative_prompt = request.negative_prompt.clone();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "kolors_ipadapter",
        0,
        move || {
            let paths = IpAdapterKolorsPaths {
                kolors_base,
                ip_adapter,
            };
            let model = IpAdapterKolors::load(&paths).map_err(|error| {
                WorkerError::Engine(format!("Kolors IP-Adapter load failed: {error}"))
            })?;
            // Per-job identity-likeness scorer built ONCE here (`!Send` face stack on the blocking
            // thread); source embedded once, reused across every output (sc-4411). `None` ⇒ non-fatal
            // staging / construction failure ⇒ scores omitted.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            Ok((model, reference, scorer))
        },
        move |(model, reference, scorer), tx, cancel| {
            // `IpAdapterKolors::generate` takes `&mut self` (it sets the IP image tokens on the UNet
            // before the denoise), so the per-item closure mutates `model`.
            let mut model = model;
            drive_gen_items_scored(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = IpAdapterKolorsRequest {
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
                            "Kolors IP-Adapter generation failed: {error}"
                        )));
                    }
                };
                // Score this finished image against the cached source embedding (sc-4411). Clone paid
                // ONLY when a scorer exists; non-frontal → honest detected:false N/A; `None` ⇒ omitted.
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
        KOLORS_IPADAPTER_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
