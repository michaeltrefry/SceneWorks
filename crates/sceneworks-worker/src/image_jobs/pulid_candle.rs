// Candle (Windows/CUDA) PuLID-FLUX face-identity route (sc-5492, epic 5480) — identity-preserving
// `character_image` generation on FLUX.1-dev off-Mac via `candle_gen_pulid::PulidFlux`. The candle
// sibling of the macOS `pulid_flux` registry route (image_jobs/pulid.rs): the reference face →
// SCRFD/ArcFace + BiSeNet `face_features_image` → EVA02-CLIP tower → IDFormer → 20 PerceiverAttentionCA
// modules injected into the FLUX DiT, denoised with the FLUX flow-match schedule (a single distilled
// forward per step).
//
// **Candle-only.** macOS keeps the inventory-registered `pulid_flux` MLX generator (it rides the
// shared `start_cached_gen_stream` registry path with an env-var weight seam); the candle `PulidFlux`
// is a bespoke provider taking explicit weight paths, so this whole file is gated to the Windows/CUDA
// candle build (the `include!` in image_jobs.rs carries the cfg). It is `include!`d into the
// `image_jobs` module, so it shares that module's imports (ImageRequest/Settings/WorkerResult/`advanced`/
// `load_reference_image`/`huggingface_snapshot_dir`/`ensure_hf_cached_file`/`start_gen_stream`/… all in
// scope unqualified). Distinct model id `pulid_flux_dev` (not `flux_dev`) cleanly disambiguates this
// from the FLUX XLabs IP-Adapter lane (flux_ipadapter.rs).

/// SceneWorks model id for native PuLID-FLUX (FLUX.1-dev backbone + PuLID injection).
const PULID_CANDLE_MODEL: &str = "pulid_flux_dev";
/// FLUX.1-dev backbone repo when the manifest omits `repo` (the same default the MLX route uses).
const PULID_CANDLE_FLUX_REPO: &str = "black-forest-labs/FLUX.1-dev";
/// The PuLID-FLUX adapter (IDFormer + the 20 PerceiverAttention CA blocks). Public repo.
const PULID_CANDLE_ADAPTER_REPO: &str = "guozinan/PuLID";
const PULID_CANDLE_ADAPTER_FILE: &str = "pulid_flux_v0.9.1.safetensors";
/// The converted-weights bundle (the EVA02-CLIP-L-336 tower + the BiSeNet face-parsing net) — the SAME
/// `SceneWorks/pulid-flux-mlx` repo the MLX route downloads (the candle EVA/BiSeNet loaders read the
/// identical mlx-converted layout: OHWI conv, bare keys).
const PULID_CANDLE_MLX_REPO: &str = "SceneWorks/pulid-flux-mlx";
const PULID_CANDLE_EVA_FILE: &str = "eva02_clip_l_336.safetensors";
const PULID_CANDLE_BISENET_FILE: &str = "bisenet_parsing.safetensors";
/// The SCRFD detector + ArcFace embedder: the SAME converted files InstantID ships (reused from
/// `SceneWorks/instantid-mlx`); only EVA + BiSeNet + the PuLID adapter are PuLID-specific.
const PULID_CANDLE_FACE_REPO: &str = "SceneWorks/instantid-mlx";
const PULID_CANDLE_SCRFD_FILE: &str = "scrfd_10g.safetensors";
const PULID_CANDLE_ARCFACE_FILE: &str = "arcface_iresnet100.safetensors";
/// Both bundle repos ship the safetensors on `main`.
const PULID_CANDLE_REVISION: &str = "main";

/// Torch/MLX-parity defaults (the `pulid_flux_dev` "photoreal" preset): 30 steps at guidance 4.0,
/// id_weight 1.0.
const PULID_CANDLE_DEFAULT_STEPS: u32 = 30;
const PULID_CANDLE_DEFAULT_GUIDANCE: f32 = 4.0;
const PULID_CANDLE_DEFAULT_ID_WEIGHT: f32 = 1.0;
/// The adapter/engine id recorded on candle PuLID-FLUX assets + telemetry (distinct from the macOS
/// `mlx_pulid_flux` label and the txt2img `candle_flux` lane).
const PULID_CANDLE_ENGINE: &str = "candle_pulid_flux";

/// Resolve the FLUX.1-dev backbone snapshot for candle PuLID-FLUX: an explicit `modelPath` dir
/// (advanced or manifest) wins, else the HF cache snapshot for the manifest `repo` (default
/// FLUX.1-dev). `None` means the base is not present locally, so the job is not candle-runnable (falls
/// through to torch). Mirrors `resolve_flux_ipadapter_base` / the MLX `resolve_pulid_flux_base`.
fn resolve_pulid_candle_base(
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
        return resolve_app_managed_model_dir(settings, &path, "PuLID-FLUX modelPath").map(Some);
    }
    let repo = request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(PULID_CANDLE_FLUX_REPO);
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// True when this is a candle-eligible PuLID-FLUX job: the `pulid_flux_dev` model in `character_image`
/// mode with a reference face whose FLUX.1-dev base resolves locally. Mirrors
/// `jobs_store::pulid_flux_candle_eligible` so the router and worker agree (and the MLX
/// `pulid_flux_available`). PuLID-FLUX is text-to-image-with-a-face only — no `edit_image`.
fn pulid_candle_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == PULID_CANDLE_MODEL
        && request.mode == "character_image"
        && non_empty(&request.reference_asset_id)
        && matches!(resolve_pulid_candle_base(request, settings), Ok(Some(_)))
}

/// Resolve PuLID denoise steps: `advanced.steps` (clamped 1..=80) → manifest `steps` → 30.
fn pulid_candle_steps(request: &ImageRequest) -> u32 {
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
        .unwrap_or(PULID_CANDLE_DEFAULT_STEPS)
}

/// Resolve PuLID guidance: `advanced.guidanceScale` → manifest `guidanceScale` → 4.0 (the FLUX.1-dev
/// guidance-distilled CFG the distilled single forward consumes).
fn pulid_candle_guidance(request: &ImageRequest) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(PULID_CANDLE_DEFAULT_GUIDANCE);
    advanced::f32_clamped(&request.advanced, "guidanceScale", manifest_default, 0.0..=30.0)
}

/// The PuLID identity-strength knob → the provider's `id_weight` (0.0 = the no-id ablation = plain
/// FLUX). Torch/MLX clamp band 0.0–3.0 (the upstream gradio slider), default 1.0.
fn pulid_candle_id_weight(request: &ImageRequest) -> f32 {
    advanced::f32_clamped(
        &request.advanced,
        "idWeight",
        PULID_CANDLE_DEFAULT_ID_WEIGHT,
        0.0..=3.0,
    )
}

/// Resolve the PuLID adapter file + the EVA tower file + the native face-stack dir the `PulidFlux`
/// provider loads, downloading on first use into ONE bundle dir (so it doubles as the provider's
/// `face_dir`: `candle_gen_face::load_with_parser_on` reads `scrfd_10g` + `arcface_iresnet100` +
/// `bisenet_parsing` from it by name, ignoring the adapter/EVA files alongside). Resolution: a
/// `SCENEWORKS_PULID_WEIGHTS` pre-staged bundle dir (all five files present) → a whole-repo HF cache
/// snapshot per file → download-on-first-use into the app cache. Returns `(adapter, eva, face_dir)`.
async fn ensure_pulid_candle_weights(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<(PathBuf, PathBuf, PathBuf)> {
    // Env override: a pre-staged bundle dir holding all five loose files (the validation layout).
    if let Ok(root) = std::env::var("SCENEWORKS_PULID_WEIGHTS") {
        let root = PathBuf::from(root);
        let adapter = root.join(PULID_CANDLE_ADAPTER_FILE);
        let eva = root.join(PULID_CANDLE_EVA_FILE);
        if adapter.is_file()
            && eva.is_file()
            && root.join(PULID_CANDLE_SCRFD_FILE).is_file()
            && root.join(PULID_CANDLE_ARCFACE_FILE).is_file()
            && root.join(PULID_CANDLE_BISENET_FILE).is_file()
        {
            return Ok((adapter, eva, root));
        }
    }

    let client = reqwest::Client::new();
    let context = DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "PuLID-FLUX generation canceled while fetching weights.",
        fresh_download: false,
    };
    // One bundle dir holds every loose file (it IS the provider's face_dir).
    let bundle = settings.data_dir.join("cache").join("pulid-flux");

    // Resolve one bundle file: a whole-repo HF cache snapshot already carrying it, else download into
    // `bundle` on first use.
    async fn resolve_one(
        context: &DownloadContext<'_>,
        settings: &Settings,
        repo: &str,
        file: &str,
        bundle: &Path,
    ) -> WorkerResult<PathBuf> {
        if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, repo)
            .map(|snapshot| snapshot.join(file))
            .filter(|path| path.is_file())
        {
            return Ok(snapshot);
        }
        let dst = bundle.join(file);
        ensure_hf_cached_file(context, repo, PULID_CANDLE_REVISION, file, &dst).await?;
        Ok(dst)
    }

    // PuLID adapter (public guozinan/PuLID) + EVA (SceneWorks bundle). These may resolve to a separate
    // whole-repo snapshot, so keep their resolved paths (NOT assumed to be in `bundle`).
    let adapter = resolve_one(
        &context,
        settings,
        PULID_CANDLE_ADAPTER_REPO,
        PULID_CANDLE_ADAPTER_FILE,
        &bundle,
    )
    .await?;
    let eva = resolve_one(
        &context,
        settings,
        PULID_CANDLE_MLX_REPO,
        PULID_CANDLE_EVA_FILE,
        &bundle,
    )
    .await?;
    // The three face-stack files MUST land together in `bundle` (the provider's face_dir), so download
    // each straight into it rather than accepting a scattered whole-repo snapshot.
    for (repo, file) in [
        (PULID_CANDLE_FACE_REPO, PULID_CANDLE_SCRFD_FILE),
        (PULID_CANDLE_FACE_REPO, PULID_CANDLE_ARCFACE_FILE),
        (PULID_CANDLE_MLX_REPO, PULID_CANDLE_BISENET_FILE),
    ] {
        ensure_hf_cached_file(&context, repo, PULID_CANDLE_REVISION, file, &bundle.join(file)).await?;
    }

    Ok((adapter, eva, bundle))
}

/// Flat telemetry recorded on candle PuLID-FLUX assets (parity with the MLX `pulidFlux` recipe keys).
fn pulid_candle_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    guidance: f32,
    id_weight: f32,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("pulidFlux".to_owned(), Value::Bool(true));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert("guidanceScale".to_owned(), json!(guidance));
    raw.insert("idWeight".to_owned(), json!(id_weight));
    raw.insert(
        "pulidFluxEngine".to_owned(),
        Value::String(PULID_CANDLE_ENGINE.to_owned()),
    );
    raw
}

/// Real candle PuLID-FLUX generation: resolve the reference + weights on the async side, then load the
/// `PulidFlux` provider once + generate each image on the blocking thread. `request.count` images, each
/// its own seed; `idWeight` rides the provider's `id_weight`. `generate` takes `&self` (the id_embedding
/// and CA injector are built per call), so — like the FLUX IP lane — the per-item closure needs no
/// `mut`. Reuses the shared streaming seam (`consume_gen_events`) so step/cancel/asset behavior matches
/// every other candle family.
async fn generate_candle_pulid_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let flux_base = resolve_pulid_candle_base(request, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload("PuLID-FLUX base (FLUX.1-dev) not found".to_owned())
    })?;
    let reference_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("PuLID-FLUX requires a reference face image".to_owned())
        })?;
    let reference = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        reference_id,
        project_path,
    )?;
    let (adapter, eva, face_dir) = ensure_pulid_candle_weights(api, settings, job).await?;

    // Identity-likeness scoring (epic 4406, sc-4411 plain With-Character): the candle PuLID-FLUX lane
    // serves the single-identity `character_image` path (one identity image per seed) and has no
    // angle/pose tier — score every output against the reference face through the SHARED generator-
    // agnostic seam. Eligibility goes through `resolve_character_image_likeness_source` (the SAME gate the
    // macOS lanes use), so the angle/pose/edit exclusion is explicit and self-contained here — NOT
    // dependent on dispatch order. The helper's decode is ignored: the already-decoded `reference` (this
    // lane's generation input, the current job's `referenceAssetId`) is the scorer source, so there is no
    // second decode. Stage the antelopev2 SCRFD + ArcFace bundle (the same one the scorer's candle leg
    // loads; distinct from PuLID's own BiSeNet `face_dir`); the `!Send` scorer is built ONCE inside the
    // load closure and reused across the N outputs (source embedded once — the caching AC). Staging is
    // non-fatal (failure → no scorer → scores omitted, generation still renders).
    let score_likeness =
        resolve_character_image_likeness_source(request, settings, project_path).is_some();
    let face_stack_dir = if score_likeness {
        match ensure_face_stack_dir(api, settings, job).await {
            Ok(dir) => Some(dir),
            Err(error) => {
                tracing::warn!(error = %error, "PuLID-FLUX face-stack staging failed; likeness scores omitted");
                None
            }
        }
    } else {
        None
    };
    let likeness_source = face_stack_dir.as_ref().map(|_| reference.clone());
    let likeness_source_ref = reference_id.to_owned();

    let steps = pulid_candle_steps(request);
    let guidance = pulid_candle_guidance(request);
    let id_weight = pulid_candle_id_weight(request);
    // Curated unified-sampler selection (epic 7114, sc-7432): the candle `PulidFlux` provider was made
    // sampler-pluggable in #130 (the `scheduler` axis re-strides FLUX's native schedule over the dev
    // time-shift `mu`). Read + N3-normalize against the shared curated menu. N1: unset ⇒ `None` ⇒ the
    // native euler-over-native-schedule default runs byte-exact.
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
        .unwrap_or(PULID_CANDLE_FLUX_REPO)
        .to_owned();
    let raw_settings = pulid_candle_raw_settings(request, &repo, steps, guidance, id_weight);

    // Per-image work items: (seed, prompt) — `request.count` images at the reference identity.
    let (width, height) = (request.width, request.height);
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();

    let (cancel, rx, blocking) = start_gen_stream(
        job.id.clone(),
        "pulid_flux",
        0,
        move || {
            let paths = PulidFluxPaths {
                flux_base,
                pulid_weights: adapter,
                eva_weights: eva,
                face_dir,
            };
            let model = PulidFlux::load(&paths).map_err(|error| {
                WorkerError::Engine(format!("PuLID-FLUX load failed: {error}"))
            })?;
            // Build the per-job identity-likeness scorer ONCE here (on the blocking thread where the
            // `!Send` face stack is allowed), embedding the source identity face a single time and
            // reusing it across every output (sc-4411 caching AC). `None` ⇒ non-fatal staging /
            // construction failure ⇒ scores omitted; the generation still renders.
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_face_likeness_scorer(dir, source)
                }
                _ => None,
            };
            Ok((model, reference, scorer))
        },
        move |(model, reference, scorer), tx, cancel| {
            drive_gen_items_scored(tx, work, move |_index, (seed, prompt), on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let req = PulidFluxRequest {
                    prompt,
                    width,
                    height,
                    steps: steps as usize,
                    guidance,
                    id_weight,
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
                            "PuLID-FLUX generation failed: {error}"
                        )));
                    }
                };
                // Score this finished image against the cached source embedding (sc-4411). The Image
                // build + pixel clone is paid ONLY when a scorer exists; a non-frontal / no-face result
                // records an honest detected:false N/A, `None` scorer ⇒ field omitted.
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
        PULID_CANDLE_ENGINE,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
