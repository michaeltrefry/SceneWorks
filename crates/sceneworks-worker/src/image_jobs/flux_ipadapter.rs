// Candle (Windows/CUDA) FLUX XLabs IP-Adapter reference route (sc-5872, epic 5480) — reference-image
// (identity) conditioning on FLUX.1 [dev]/[schnell] off-Mac via `candle_gen_flux::IpAdapterFlux`. The
// FLUX sibling of the candle SDXL/Kolors IP-Adapter lanes (sdxl_ipadapter.rs / kolors_ipadapter.rs):
// the pooled CLIP-ViT-L image embedding is projected (XLabs `ImageProjModel`) into image-prompt tokens
// and injected as a decoupled cross-attention into the forked FLUX DiT's double blocks, denoised with
// the FLUX flow-match schedule (a single distilled forward per step).
//
// **Candle-only.** macOS keeps the MLX FLUX XLabs IP path (epic 3621, the registry `Reference` route
// the `flux1_*` generators handle via `with_ip_adapter`); the candle `IpAdapterFlux` is a bespoke
// provider, so this whole file is gated to the Windows/CUDA candle build (the `include!` in
// image_jobs.rs carries the cfg). It is `include!`d into the `image_jobs` module, so it shares that
// module's imports (ImageRequest/Settings/WorkerResult/`advanced`/`load_reference_image`/
// `huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in scope unqualified).

/// The XLabs FLUX IP-Adapter repo + the single bundle file (`ip_adapter_proj_model` + the 19
/// per-double-block K/V projectors). Same repo the MLX path uses (`base.rs` `FLUX_IP_ADAPTER_REPO`).
const FLUX_IPADAPTER_ADAPTER_REPO: &str = "XLabs-AI/flux-ip-adapter";
const FLUX_IPADAPTER_ADAPTER_FILE: &str = "ip_adapter.safetensors";
/// The CLIP ViT-L/14 image encoder repo (the XLabs adapter conditions on `openai/clip-vit-large-patch14`
/// pooled image embeds) + the weight file the provider's `resolve_image_encoder` looks for in the dir.
const FLUX_IPADAPTER_ENCODER_REPO: &str = "openai/clip-vit-large-patch14";
const FLUX_IPADAPTER_ENCODER_FILE: &str = "model.safetensors";
/// Both repos ship the safetensors on `main` (unlike the Kolors `refs/pr/4`).
const FLUX_IPADAPTER_REVISION: &str = "main";
/// IP-Adapter scale default — the XLabs resemblance tier 0.7 (matches `base.rs` `FLUX_IP_SCALE`, the MLX
/// path, and the candle `IpAdapterFlux::DEFAULT_IP_SCALE`).
const FLUX_IPADAPTER_IP_SCALE: f32 = 0.7;
/// The adapter/engine id recorded on candle FLUX IP-Adapter assets + telemetry (distinct from the
/// txt2img `candle_flux` lane).
const FLUX_IPADAPTER_ENGINE: &str = "candle_flux_ipadapter";

/// Model ids the candle FLUX IP-Adapter route accepts (both variants — the forked DiT injects the same
/// XLabs adapter for each; dev embeds the guidance scalar, schnell ignores it).
fn is_flux_ipadapter_model(model: &str) -> bool {
    matches!(model, "flux_dev" | "flux_schnell")
}

/// The black-forest-labs base repo when the manifest omits `repo` (variant-keyed).
fn flux_ipadapter_default_repo(model: &str) -> &'static str {
    match model {
        "flux_schnell" => "black-forest-labs/FLUX.1-schnell",
        _ => "black-forest-labs/FLUX.1-dev",
    }
}

/// Distilled default step count (FLUX parity): schnell 4, dev 25.
fn flux_ipadapter_default_steps(model: &str) -> u32 {
    match model {
        "flux_schnell" => 4,
        _ => 25,
    }
}

/// Default guidance: dev embeds 3.5; schnell is timestep-distilled (no guidance).
fn flux_ipadapter_default_guidance(model: &str) -> f32 {
    match model {
        "flux_schnell" => 0.0,
        _ => 3.5,
    }
}

/// Resolve the FLUX base (BFL snapshot) for the IP-Adapter route: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (default
/// `black-forest-labs/FLUX.1-{dev,schnell}` by variant). `None` means the base is not present locally,
/// so the job is not candle-runnable (falls through to torch). Mirrors `resolve_kolors_ipadapter_base`.
fn resolve_flux_ipadapter_base(
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
        return resolve_app_managed_model_dir(settings, &path, "FLUX IP-Adapter modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| flux_ipadapter_default_repo(&request.model));
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible FLUX IP-Adapter job: a `flux_dev`/`flux_schnell` model with a
/// reference image (and NOT an img2img/inpaint/edit shape — those are sc-5487) whose base resolves
/// locally. Mirrors `jobs_store::flux_ipadapter_candle_eligible` so the worker and router agree.
fn flux_ipadapter_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_flux_ipadapter_model(&request.model)
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !non_empty(&request.source_asset_id)
        && !non_empty(&request.mask_asset_id)
        && matches!(resolve_flux_ipadapter_base(request, settings), Ok(Some(_)))
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=100) → manifest `steps` → variant default.
fn flux_ipadapter_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or_else(|| flux_ipadapter_default_steps(&request.model))
}

/// Resolve guidance: `advanced.guidanceScale` → manifest `guidanceScale` → variant default, clamped.
fn flux_ipadapter_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or_else(|| flux_ipadapter_default_guidance(&request.model));
    advanced::f32_clamped(
        &request.advanced,
        "guidanceScale",
        manifest_default,
        0.0..=30.0,
    )
}

/// Resolve the XLabs adapter file + the CLIP-ViT-L encoder dir the `IpAdapterFlux` provider loads,
/// downloading on first use. Resolution order for each: an env-pinned root (`SCENEWORKS_IPADAPTER_FLUX`,
/// the validation-staging layout `<root>/ip_adapter.safetensors` + `<root>/image_encoder/`) → a
/// whole-repo HF cache snapshot → download into the app cache. Returns `(adapter_file, encoder_dir)`
/// (the latter is what [`IpAdapterFluxPaths::image_encoder`] wants — `resolve_image_encoder` finds
/// `model.safetensors` inside).
async fn ensure_flux_ipadapter_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(PathBuf, PathBuf)> {
    // Env override: a staged dir laid out like the validation harness.
    if let Ok(root) = std::env::var("SCENEWORKS_IPADAPTER_FLUX") {
        let root = PathBuf::from(root);
        let adapter = root.join(FLUX_IPADAPTER_ADAPTER_FILE);
        let encoder = root.join("image_encoder");
        if adapter.is_file() && encoder.join(FLUX_IPADAPTER_ENCODER_FILE).is_file() {
            return Ok((adapter, encoder));
        }
    }

    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "FLUX IP-Adapter generation canceled while fetching weights.",
        fresh_download: false,
    };
    let cache = settings.data_dir.join("cache").join("ipadapter-flux");

    // XLabs adapter file: a whole-repo HF cache snapshot already carrying it, else download-on-first-use.
    let adapter = match huggingface_snapshot_dir(&settings.data_dir, FLUX_IPADAPTER_ADAPTER_REPO)
        .map(|snapshot| snapshot.join(FLUX_IPADAPTER_ADAPTER_FILE))
        .filter(|file| file.is_file())
    {
        Some(file) => file,
        None => {
            let dst = cache.join(FLUX_IPADAPTER_ADAPTER_FILE);
            ensure_hf_cached_file(
                &context,
                FLUX_IPADAPTER_ADAPTER_REPO,
                FLUX_IPADAPTER_REVISION,
                FLUX_IPADAPTER_ADAPTER_FILE,
                &dst,
            )
            .await?;
            dst
        }
    };

    // CLIP-ViT-L encoder dir (must contain `model.safetensors`): the whole-repo snapshot, else download.
    let encoder = match huggingface_snapshot_dir(&settings.data_dir, FLUX_IPADAPTER_ENCODER_REPO)
        .filter(|snapshot| snapshot.join(FLUX_IPADAPTER_ENCODER_FILE).is_file())
    {
        Some(snapshot) => snapshot,
        None => {
            let dir = cache.join("image_encoder");
            ensure_hf_cached_file(
                &context,
                FLUX_IPADAPTER_ENCODER_REPO,
                FLUX_IPADAPTER_REVISION,
                FLUX_IPADAPTER_ENCODER_FILE,
                &dir.join(FLUX_IPADAPTER_ENCODER_FILE),
            )
            .await?;
            dir
        }
    };

    Ok((adapter, encoder))
}

/// Flat telemetry recorded on candle FLUX IP-Adapter assets (parity with the SDXL/Kolors recipe-key
/// shape).
fn flux_ipadapter_raw_settings(
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
        Value::String(FLUX_IPADAPTER_ENGINE.to_owned()),
    );
    raw
}

/// Real candle FLUX IP-Adapter generation: resolve the reference + weights on the async side, then load
/// the `IpAdapterFlux` provider once + generate each image on the blocking thread. `request.count`
/// images, each its own seed; `generate` takes the per-job `CancelFlag` + a `Progress` callback, so
/// streaming is per-step and cancellation is honoured mid-denoise — same contract as the SDXL/Kolors IP
/// lanes.
async fn generate_candle_flux_ipadapter_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let flux_base = resolve_flux_ipadapter_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("FLUX IP-Adapter base (FLUX.1 snapshot) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("FLUX IP-Adapter requires a reference image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let (adapter_file, encoder_dir) = ensure_flux_ipadapter_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle FLUX XLabs
    // IP-Adapter lane is the With-Character route for a FLUX.1 model — score every output against the
    // reference face through the SHARED generator-agnostic seam. Eligibility goes through
    // `resolve_character_image_likeness_source` (the SAME gate the macOS lanes use), so the angle/pose/
    // edit exclusion is explicit and self-contained here — NOT dependent on dispatch order (an angle/pose
    // job is excluded by the helper even if it ever reached this lane, so it can never be double-scored).
    // The helper's decode is ignored: the already-decoded `reference` (this lane's generation input, the
    // current job's `referenceAssetId`) is the scorer source, so there is no second decode. Stage the
    // antelopev2 SCRFD + ArcFace bundle; the `!Send` scorer is built ONCE inside the load closure and
    // reused across the N outputs (source embedded once). Staging is non-fatal (failure → scores omitted).
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

    let steps = flux_ipadapter_steps(request);
    let guidance = flux_ipadapter_guidance(request);
    let ip_scale = advanced::f32_clamped(
        &request.advanced,
        "ipAdapterScale",
        FLUX_IPADAPTER_IP_SCALE,
        0.0..=1.0,
    );
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| flux_ipadapter_default_repo(&request.model))
        .to_owned();
    let raw_settings = flux_ipadapter_raw_settings(request, &repo, steps, guidance, ip_scale);

    // Per-image work items: (seed, prompt) — `request.count` images at the reference identity.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "flux_ipadapter",
        0,
        move || {
            let paths = IpAdapterFluxPaths {
                flux_base,
                ip_adapter: adapter_file,
                image_encoder: encoder_dir,
            };
            let model = IpAdapterFlux::load(&paths).map_err(|error| {
                WorkerError::Engine(format!("FLUX IP-Adapter load failed: {error}"))
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
            // `IpAdapterFlux::generate` takes `&self` (the IP tokens live in a per-call injector, not on
            // the DiT), so — unlike the SDXL/Kolors lanes — the per-item closure needs no `mut model`.
            drive_gen_items_scored(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = IpAdapterFluxRequest {
                    prompt,
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    ip_adapter_scale: ip_scale,
                    seed: seed as u64,
                    cancel: cancel.clone(),
                };
                let out = match model.generate(&req, &reference, &mut *on_progress) {
                    Ok(out) => out,
                    Err(_) if cancel.is_cancelled() => return Ok(None),
                    Err(error) => {
                        return Err(WorkerError::Engine(format!(
                            "FLUX IP-Adapter generation failed: {error}"
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
        FLUX_IPADAPTER_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
