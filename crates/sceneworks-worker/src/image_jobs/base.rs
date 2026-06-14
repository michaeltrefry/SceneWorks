#[cfg(target_os = "macos")]
fn mlx_available(request: &ImageRequest, settings: &Settings) -> bool {
    mlx_model(&request.model).is_some()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageRoute {
    ZImageControl,
    QwenControl,
    KolorsControl,
    Flux2Edit,
    QwenEdit,
    InstantId,
    PulidFlux,
    SdxlAdvanced,
    SensenovaEdit,
    Bernini,
    Mlx,
}

#[cfg(target_os = "macos")]
fn resolve_image_route(request: &ImageRequest, settings: &Settings) -> Option<ImageRoute> {
    if zimage_control_available(request, settings) {
        Some(ImageRoute::ZImageControl)
    } else if qwen_control_available(request, settings) {
        Some(ImageRoute::QwenControl)
    } else if kolors_control_available(request, settings) {
        Some(ImageRoute::KolorsControl)
    } else if flux2_edit_available(request, settings) {
        Some(ImageRoute::Flux2Edit)
    } else if qwen_edit_available(request, settings) {
        Some(ImageRoute::QwenEdit)
    } else if instantid_available(request, settings) {
        Some(ImageRoute::InstantId)
    } else if pulid_flux_available(request, settings) {
        Some(ImageRoute::PulidFlux)
    } else if sdxl_advanced_available(request, settings) {
        Some(ImageRoute::SdxlAdvanced)
    } else if sensenova_edit_available(request, settings) {
        Some(ImageRoute::SensenovaEdit)
    } else if bernini_image_available(request, settings) {
        // Bernini still-image companion (sc-5424): t2i / i2i on the `bernini_image` id. Must win
        // over the generic `mlx_available` arm below — `bernini_image` is in MODEL_TABLE (so
        // `mlx_available` would match it), but the generic `generate_stream` leaves `frames`/
        // `video_mode` unset, which the engine treats as a multi-frame video request.
        Some(ImageRoute::Bernini)
    } else if mlx_available(request, settings) {
        Some(ImageRoute::Mlx)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
impl ImageRoute {
    fn image_count(self, request: &ImageRequest, settings: &Settings) -> u32 {
        match self {
            ImageRoute::ZImageControl
            | ImageRoute::QwenControl
            | ImageRoute::KolorsControl => pose_entries(request).len() as u32,
            ImageRoute::Flux2Edit | ImageRoute::QwenEdit => grouped_edit_image_count(request),
            ImageRoute::InstantId => instantid_image_count(request, settings),
            ImageRoute::SensenovaEdit => match flux2_grouping(request) {
                Flux2Grouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
                // SenseNova has no strict-pose (ControlNet) path; pose jobs are excluded
                // upstream, so any residual grouping preserves the requested image count.
                Flux2Grouping::Poses(_) | Flux2Grouping::Plain => request.count,
            },
            // PuLID-FLUX is one identity image per seed (no angle/pose grouping) — like the base
            // MLX + SDXL-advanced + Bernini paths, the effective count is the requested count.
            ImageRoute::PulidFlux
            | ImageRoute::SdxlAdvanced
            | ImageRoute::Bernini
            | ImageRoute::Mlx => request.count,
        }
    }
}

#[cfg(target_os = "macos")]
fn grouped_edit_image_count(request: &ImageRequest) -> u32 {
    match flux2_grouping(request) {
        Flux2Grouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
        Flux2Grouping::Poses(count) => count as u32,
        Flux2Grouping::Plain => request.count,
    }
}

/// The HuggingFace repo for the model: the manifest entry's `repo` wins, else the
/// family default. Shared by the MLX path and the candle lane (sc-5096).
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn model_repo(request: &ImageRequest, model: &ResolvedModel) -> String {
    request
        .model_manifest_entry
        .get("repo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(model.default_repo())
        .to_owned()
}

/// Resolve the weights snapshot directory: an explicit `modelPath` dir wins, else the
/// HuggingFace cache snapshot for the model repo. `None` when the model is not a known
/// engine family or its snapshot is absent.
#[cfg(target_os = "macos")]
pub(crate) fn resolve_weights_dir(
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
        return resolve_app_managed_model_dir(settings, &path, "Image modelPath").map(Some);
    }
    let Some(model) = mlx_model(&request.model) else {
        return Ok(None);
    };
    Ok(huggingface_snapshot_dir(
        &settings.data_dir,
        &model_repo(request, &model),
    ))
}

#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn quant_int(value: &Value) -> Option<i64> {
    if value.is_boolean() {
        return None;
    }
    value
        .as_i64()
        .or_else(|| value.as_str()?.trim().parse().ok())
}

/// Resolve quantization: `advanced.mlxQuantize` → `manifest.mlx.quantize` → Q8
/// default. The engine supports Q4/Q8; map (<=0 → dense, <=4 → Q4, else Q8). Returns the
/// engine quant + the effective bit count for the recipe (None = dense bf16).
///
/// Shared by the MLX path and the candle lane (sc-5126). On the candle lane it is called ONLY for a
/// family whose descriptor advertises `supported_quants` (i.e. Lens — see `generate_candle_stream`'s
/// `model.supports_quant()` gate), so the Q8 default applies to Lens exactly like the MLX families;
/// the sc-3675/sc-5096 candle families advertise no quant and never reach this resolver (stay dense).
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn resolve_quant(request: &ImageRequest) -> (Option<Quant>, Option<i64>) {
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
        None => (Some(Quant::Q8), Some(8)),
        Some(bits) if bits <= 0 => (None, None),
        Some(bits) if bits <= 4 => (Some(Quant::Q4), Some(4)),
        Some(_) => (Some(Quant::Q8), Some(8)),
    }
}

/// Resolve denoise steps: `advanced.steps` (clamped 1..=80) else the family default.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn resolve_steps(request: &ImageRequest, model: &ResolvedModel) -> u32 {
    request
        .advanced
        .get("steps")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|steps| (steps as u32).clamp(1, 80))
        .unwrap_or(model.default_steps())
}

/// Resolve the guidance scale. Distilled variants (z-image-turbo, flux schnell) take
/// no guidance — the engine rejects `Some(_)` on them — so this returns `None`. For a
/// guided variant (flux dev) it is `advanced.guidanceScale` else the family default.
/// Shared by the MLX path and the candle lane (sc-5096); the descriptor's `supports_guidance` is the
/// candle descriptor on the Windows lane, so a distilled candle family (z-image, flux schnell) still
/// gets `None` and a guided one (flux dev, flux2, qwen, sdxl) gets the scale.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn resolve_guidance(request: &ImageRequest, model: &ResolvedModel) -> Option<f32> {
    if !model.supports_guidance() {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    Some(scale)
}

/// True for a TRUE-CFG family whose engine reads the CFG scale from `true_cfg` (with a real
/// negative prompt) and **rejects** the distilled `guidance` scalar — i.e. Chroma (epic 3531),
/// uniquely identified by `supports_guidance=false` + `supports_negative_prompt=true`. The
/// guidance-distilled families (`z_image_turbo`, `flux_schnell`) are `false`/`false` (no CFG at
/// all), and the `guidance`-scalar families (qwen / sdxl / flux2 …) are `true`/*. For a true-CFG
/// family the worker forwards `advanced.guidanceScale` as `true_cfg`, not `guidance`.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn uses_true_cfg(model: &ResolvedModel) -> bool {
    !model.supports_guidance() && model.supports_negative_prompt()
}

/// Resolve the true-CFG scale for a true-CFG family (Chroma). `None` for every other family
/// (their CFG, if any, flows through [`resolve_guidance`]). The scale is `advanced.guidanceScale`
/// (the same user knob) else the family default — forwarded to the engine as `GenerationRequest.true_cfg`.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn resolve_true_cfg(request: &ImageRequest, model: &ResolvedModel) -> Option<f32> {
    if !uses_true_cfg(model) {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    Some(scale)
}

/// The negative prompt to pass to the engine. `None` for variants without true CFG
/// (the engine rejects `negative_prompt` on the distilled families) and for an empty
/// prompt (the true-CFG engines fall back to their own neutral negative).
/// Shared by the MLX path and the candle lane (sc-5096); on the Windows lane `supports_negative_prompt`
/// is the candle descriptor, so distilled candle families (z-image, flux schnell) get `None`.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn resolve_negative_prompt(request: &ImageRequest, model: &ResolvedModel) -> Option<String> {
    if !model.supports_negative_prompt() {
        return None;
    }
    let trimmed = request.negative_prompt.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// First non-empty of installedPath/sourcePath/path/source.path on a LoRA spec.
/// Shared by the MLX path and the candle Lens lane (sc-5126).
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
pub(crate) fn lora_path(lora: &Value) -> Option<PathBuf> {
    for key in ["installedPath", "sourcePath", "path"] {
        if let Some(value) = lora
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(PathBuf::from(value));
        }
    }
    lora.get("source")
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Classify a LoRA file into the mlx-gen adapter `kind`. SceneWorks peft-LoKr (stamped
/// `networkType: lokr`) → `Lokr` (the engine's metadata-gated `apply_lokr` peft path). Everything
/// else → `Lora`, INCLUDING third-party LyCORIS (LoHa / kohya non-peft LoKr): since epic 3641
/// (sc-3642/3643/3671) the engine's `apply_adapter_specs_autoprefix` detects `lokr_*` / `hada_*`
/// keys by sniff and routes them to its third-party reconstruction regardless of the declared kind,
/// so `Lora` is the correct hint and the worker no longer rejects them. (A LyCORIS algo the engine
/// doesn't implement — e.g. (IA)³/OFT — has no `lokr_*`/`hada_*` keys, so the engine's LoRA loader
/// finds nothing and surfaces a loud "matched nothing" error rather than mis-applying.)
///
/// Shared by the MLX path and the candle Lens lane (sc-5126): candle-gen-lens's `merge_adapters`
/// dispatches on this `kind` (a `lokr`-metadata file declared `Lora` would find no lora_A/B keys and
/// it surfaces the mismatch loudly), so the same `networkType: lokr` classification feeds both lanes.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
pub(crate) fn classify_adapter(file: &Path) -> WorkerResult<AdapterKind> {
    let header = read_safetensors_header(file)
        .map_err(|error| WorkerError::InvalidPayload(format!("LoRA header: {error}")))?;
    let network_type = header
        .get("__metadata__")
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    if network_type.as_deref() == Some("lokr") {
        return Ok(AdapterKind::Lokr);
    }
    Ok(AdapterKind::Lora)
}

/// Resolve up to 3 request LoRAs into engine adapter specs (path + scale + kind).
/// Shared by the MLX path and the candle Lens lane (sc-5126).
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
fn resolve_adapters(request: &ImageRequest) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let path = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        let file = if path.is_dir() {
            first_safetensors_path(&path).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "LoRA has no .safetensors under {}",
                    path.display()
                ))
            })?
        } else {
            path
        };
        if !file.exists() {
            return Err(WorkerError::InvalidPayload(format!(
                "LoRA file is missing: {}",
                file.display()
            )));
        }
        let kind = classify_adapter(&file)?;
        let scale = lora
            .get("weight")
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            .unwrap_or(0.8) as f32;
        specs.push(AdapterSpec::new(file, scale, kind));
    }
    Ok(specs)
}

fn mlx_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Distilled variants run without CFG (guidance == None → null in the recipe).
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw
}

fn load_spec(
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    if let Some(dir) = ip_adapter_dir {
        spec = spec.with_ip_adapter(WeightsSource::Dir(dir));
    }
    spec
}

/// Registry-only generator load (epic 3720, sc-3724): resolve `engine_id` through the
/// backend-neutral `gen_core::load` seam and return a `Box<dyn gen_core::Generator>`. Optionally
/// installs an IP-Adapter from `ip_adapter_dir` (`LoadSpec::with_ip_adapter`) — the FLUX.1 XLabs
/// IP-Adapter reference path (epic 3621), after which the engine treats a `Conditioning::Reference`
/// as the image prompt. `cfg(target_os)` decides which provider crate registered the engine, not
/// this call.
#[cfg(all(target_os = "macos", test))]
fn load_engine(
    engine_id: &str,
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = load_spec(weights_dir, quant, adapters, ip_adapter_dir);
    gen_core::load(engine_id, &spec)
        .map_err(|error| WorkerError::Engine(format!("{engine_id} load failed: {error}")))
}

/// XLabs FLUX IP-Adapter repos (epic 3621). The torch `flux_dev` path already declares +
/// downloads these (the `ipAdapter` block in `image_adapters`); the MLX path reuses the same
/// HF-cache snapshots — there is no new weight to ship.
#[cfg(target_os = "macos")]
const FLUX_IP_ADAPTER_REPO: &str = "XLabs-AI/flux-ip-adapter";
#[cfg(target_os = "macos")]
const FLUX_IP_IMAGE_ENCODER_REPO: &str = "openai/clip-vit-large-patch14";
/// IP-Adapter scale when the request omits `ipAdapterScale` (XLabs resemblance tier 0.7, matching
/// the torch `FluxDiffusersAdapter`).
#[cfg(target_os = "macos")]
const FLUX_IP_SCALE: f32 = 0.7;
/// `trueCfgScale` default for the FLUX.1-dev IP-Adapter path (real CFG; torch default ~4.0).
#[cfg(target_os = "macos")]
const FLUX_IP_TRUE_CFG: f32 = 4.0;

/// The FLUX.1 engine families that carry the XLabs IP-Adapter (both variants — the Rust engine has
/// no diffusers `load_ip_adapter` schnell limitation).
#[cfg(target_os = "macos")]
fn is_flux_model(model: &str) -> bool {
    matches!(model, "flux_schnell" | "flux_dev")
}

/// The SenseNova-U1 SceneWorks ids (base + 8-step distill), both served by the unified
/// `mlx-gen-sensenova` engine (sc-3900).
#[cfg(target_os = "macos")]
fn is_sensenova_model(model: &str) -> bool {
    matches!(model, "sensenova_u1_8b" | "sensenova_u1_8b_fast")
}

/// Stage the engine's IP-Adapter dir contract from the two cached HF snapshots:
/// `<staged>/ip_adapter.safetensors` (XLabs) + `<staged>/image_encoder/model.safetensors`
/// (openai CLIP-ViT-L). Errors loudly if either snapshot is missing — mirrors the SDXL IP path
/// (`resolve_ip_adapter_dir`); the repos reach the cache via the model-download flow / the torch
/// `flux_dev` path, not a new provisioning step.
#[cfg(target_os = "macos")]
fn resolve_flux_ip_adapter_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    let missing = || {
        WorkerError::InvalidPayload(format!(
            "FLUX IP-Adapter weights not found (download {FLUX_IP_ADAPTER_REPO} + {FLUX_IP_IMAGE_ENCODER_REPO})."
        ))
    };
    let adapter_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_ADAPTER_REPO)
            .ok_or_else(missing)?;
    let clip_snap =
        crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, FLUX_IP_IMAGE_ENCODER_REPO)
            .ok_or_else(missing)?;
    let ip_file = adapter_snap.join("ip_adapter.safetensors");
    let clip_file = clip_snap.join("model.safetensors");
    if !ip_file.exists() || !clip_file.exists() {
        return Err(missing());
    }
    let staged = settings.data_dir.join("staged").join("flux-ip-adapter");
    let encoder_dir = staged.join("image_encoder");
    std::fs::create_dir_all(&encoder_dir)
        .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter dir: {e}")))?;
    // Re-link each call: the HF-cache targets are immutable, so a stable staged dir is reusable.
    let link = |src: &Path, dst: PathBuf| -> WorkerResult<()> {
        let _ = std::fs::remove_file(&dst);
        std::os::unix::fs::symlink(src, &dst)
            .map_err(|e| WorkerError::InvalidPayload(format!("stage flux ip-adapter link: {e}")))
    };
    link(&ip_file, staged.join("ip_adapter.safetensors"))?;
    link(&clip_file, encoder_dir.join("model.safetensors"))?;
    Ok(staged)
}

/// Emit an `image_pipeline_load_{start,complete}` event from inside a blocking
/// generation closure (sc-3450), parity with the Python worker's pipeline-load
/// events. On the backend path `gen_core::load` is a single atomic call that also fuses
/// any distill LoRA and applies user LoRAs (`spec.with_adapters`), so there is no
/// separable fuse/apply step to bracket: the adapter total (`adapter_count` =
/// distill + user) is reported here instead of via the torch worker's separate
/// `image_distill_lora_fuse_*` / `image_lora_apply_*` sub-phase events. A `start`
/// with no matching `complete` means the load failed (the error propagates via `?`).
pub(crate) fn emit_load_event(event: &str, job_id: &str, engine: &str, adapter_count: usize) {
    emit_event(
        event,
        json!({
            "jobId": job_id,
            "engine": engine,
            "adapterCount": adapter_count,
        }),
    );
}

/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
///
/// `reference` is the optional identity img2img-init (sc-3619): `(image, strength)` adds a
/// `Reference` conditioning that seeds the denoise from the reference latents — the plain
/// (no-ControlNet) Z-Image reference-without-pose path, reusing the same engine img2img the
/// strict-pose tier already drives. `None` → plain txt2img.
#[allow(clippy::too_many_arguments)]
fn generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    reference: Option<&(Image, f32)>,
    true_cfg: Option<f32>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let conditioning = match reference {
        Some((image, strength)) => vec![Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        }],
        None => Vec::new(),
    };
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images
                .pop()
                .ok_or_else(|| WorkerError::Engine("generator produced no image".to_owned()))?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "generator returned non-image output".to_owned(),
        )),
    }
}

/// Within-image step fraction mapped into the 0.10..0.95 generation band.
fn step_fraction(index: usize, current: u32, total: u32, count: u32) -> f64 {
    let per = 0.85 / count.max(1) as f64;
    let within = if total > 0 {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.1 + per * (index as f64 + within)).min(0.95)
}

/// Resolve a reference/source asset id to an in-memory RGB8 image (the engine VAE-encodes + resizes
/// it). Uses the indexed `ProjectStore::get_asset` → `file.path`. Shared by the MLX image/video
/// conditioning paths and the candle video i2v conditioning (sc-5175), so it lives here (both lanes)
/// rather than in a macOS-only include.
#[cfg(any(
    target_os = "macos",
    all(target_os = "windows", feature = "backend-candle")
))]
pub(crate) fn load_reference_image(
    data_dir: &Path,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, asset_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id}: {error}"))
        })?;
    let rel = asset
        .get("file")
        .and_then(|file| file.get("path"))
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("reference asset {asset_id} has no media path"))
        })?;
    // The asset's file.path comes from an on-disk sidecar the user can edit, so
    // route it through safe_project_path (rejects `..`/absolute components) rather
    // than a bare join — matching the media-jobs reads and keeping a poisoned
    // sidecar from reading an arbitrary file as the reference (sc-4278 / F-MLXW-14).
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = image::open(&path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("reference image {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// Real MLX generation: load once on a blocking thread, generate each image, and
/// stream step/decode/image events back to the async worker (which saves PNGs, emits
/// `assetWrites`, and polls cancel). MLX runs entirely on the blocking thread (the
/// `Box<dyn Generator>` is `!Send` and the MLX device is single-thread).
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
async fn generate_stream(
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
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("model weights not found".to_owned()))?;
    // sc-3723: surface the descriptor-derived backend ("mlx" for every linked family today; a
    // future candle row would self-describe) over the gpu-id-derived label. Falls back to the
    // passed-in label only if a descriptor ever advertised an empty backend (never today).
    let backend = if model.backend().is_empty() {
        backend
    } else {
        model.backend()
    };
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    // True-CFG families (Chroma) carry the CFG scale in `true_cfg`, not `guidance` (which their
    // engine rejects); `None` for every other family. The recipe records the effective CFG knob.
    let model_true_cfg = resolve_true_cfg(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, &model);
    let raw_settings = mlx_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance.or(model_true_cfg),
    );
    let engine_id = model.engine_id();
    let adapter_label = model.adapter_label();
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();
    // Reference conditioning for the base MLX path, resolved once (constant across the set):
    //  • Z-Image reference-identity img2img-init (sc-3619),
    //  • FLUX.1 XLabs IP-Adapter (epic 3621 — both schnell + dev; `strength = ipAdapterScale`, plus
    //    real CFG via `trueCfgScale` on dev), and
    //  • Kolors img2img (sc-4765, `edit_image` + `sourceAssetId`) + the IP-Adapter-Plus reference
    //    (sc-4767, `referenceAssetId` → image prompt at `ipAdapterScale`). Qwen/SDXL reference
    //    divert to their own advanced branches before reaching here.
    let has_reference = request
        .reference_asset_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    let (identity_init, flux_ip_dir, flux_true_cfg): (
        Option<(Image, f32)>,
        Option<PathBuf>,
        Option<f32>,
    ) = if matches!(request.model.as_str(), "z_image_turbo" | "z_image_edit") {
        // Z-Image base path: `edit_image` → img2img-edit (sourceAssetId + strength, epic 3529);
        // otherwise the identity-init reference (referenceAssetId + referenceStrength, sc-3619).
        // Both feed the engine's single `Reference` conditioning; only the source + strength
        // keying differs. The strict-pose ControlNet tier diverts earlier (zimage_control_available).
        let init = if request.mode == "edit_image" {
            resolve_zimage_edit_init(request, settings, project_path)?
        } else {
            resolve_zimage_identity_init(request, settings, project_path)?
        };
        (init, None, None)
    } else if is_flux_model(&request.model) && has_reference && request.mode != "edit_image" {
        let reference_id = request
            .reference_asset_id
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &reference_id,
            project_path,
        )?;
        let scale = advanced::f32_clamped(
            &request.advanced,
            "ipAdapterScale",
            FLUX_IP_SCALE,
            0.0..=1.0,
        );
        let ip_dir = resolve_flux_ip_adapter_dir(settings)?;
        // Real CFG only on dev (schnell is distilled — no CFG).
        let true_cfg = (request.model == "flux_dev").then(|| {
            advanced::f32_clamped(
                &request.advanced,
                "trueCfgScale",
                FLUX_IP_TRUE_CFG,
                1.0..=10.0,
            )
        });
        (Some((image, scale)), Some(ip_dir), true_cfg)
    } else if request.model == "kolors" && request.mode == "edit_image" {
        // Kolors img2img (sc-4765): `sourceAssetId` + `strength` → the engine's `Reference`
        // (img2img init, no IP-Adapter loaded). Kolors carries CFG through `guidance` + negative
        // prompt (resolved above), not `true_cfg`.
        let init = resolve_kolors_edit_init(request, settings, project_path)?;
        (init, None, None)
    } else if request.model == "kolors" && has_reference {
        // Kolors IP-Adapter-Plus reference (sc-4767): `referenceAssetId` → the IP image prompt at
        // `ipAdapterScale`. `with_ip_adapter` makes the engine treat the `Reference` as the image
        // prompt (decoupled cross-attn) rather than an img2img init.
        let (image, scale) = resolve_kolors_ip_reference(request, settings, project_path)?;
        let ip_dir = resolve_kolors_ip_adapter_dir(settings)?;
        (Some((image, scale)), Some(ip_dir), None)
    } else {
        (None, None, None)
    };
    // The CFG scale passed to the engine as `true_cfg`: the FLUX.1-dev reference path's scale if
    // present, otherwise the true-CFG family scale (Chroma). `None` for the guidance-scalar and
    // distilled families, which carry CFG (if any) through `guidance` instead.
    let true_cfg = flux_true_cfg.or(model_true_cfg);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    let spec = load_spec(weights_dir, quant, adapters, flux_ip_dir);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let (out_w, out_h, pixels) = generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    negative_prompt.clone(),
                    identity_init.as_ref(),
                    true_cfg,
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
        adapter_label,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Whether `model` is served by the candle (Windows/CUDA) backend's **txt2img** lane. SDXL/RealVisXL
/// (sc-3675) plus the four image families wired in sc-5096 — z-image, flux schnell/dev, flux2-klein,
/// qwen-image — plus Lens / Lens-Turbo (sc-5126, the first candle family with quant + LoRA/LoKr).
/// `realvisxl` shares the candle `"sdxl"` engine via a weights swap; every other id maps 1:1 to its
/// `MODEL_TABLE` engine id. Edit/control/reference shapes and the non-base weight variants stay on the
/// Python torch worker (the router's `image_request_candle_eligible` enforces the same boundary), so
/// this gate is intentionally the base-id set only. Lens is pure T2I (no conditioning), so it joins
/// the lane with no new dispatch shape — only quant + adapters, which `generate_candle_stream`
/// resolves from the descriptor.
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
fn is_candle_engine(model: &str) -> bool {
    matches!(
        model,
        "sdxl"
            | "realvisxl"
            | "z_image_turbo"
            | "flux_schnell"
            | "flux_dev"
            | "flux2_klein_9b"
            | "qwen_image"
            | "lens"
            | "lens_turbo"
    )
}

/// The per-asset `adapter` id recorded for a candle image engine (`candle_<family>`), the candle
/// sibling of the `MODEL_TABLE` `mlx_<family>` labels. Used both per-asset (`generate_candle_stream`)
/// and at the generation-set level (`adapter_id`) so the sidecar + result agree on the backend.
/// (sc-5099 extends this same labeling to the video + caption engines.)
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
fn candle_adapter_label(model: &str) -> &'static str {
    match model {
        "z_image_turbo" => "candle_z_image",
        "flux_schnell" | "flux_dev" => "candle_flux",
        "flux2_klein_9b" => "candle_flux2",
        "qwen_image" => "candle_qwen",
        "lens" | "lens_turbo" => "candle_lens",
        // sdxl / realvisxl share the candle "sdxl" engine.
        _ => CANDLE_ADAPTER,
    }
}

/// Windows/CUDA candle execution path (sc-3675 SDXL, generalized in sc-5096). The macOS dispatch is
/// MLX-bound; candle is a narrow **txt2img-only** lane, so this is a trimmed sibling of
/// [`generate_stream`] that drives the SAME neutral streaming harness (`start_cached_gen_stream` →
/// `generate_one` → `consume_gen_events`) against the registry-resolved candle generator.
///
/// Backend-neutral resolution (sc-5096): the per-engine repo / steps / guidance / negative prompt all
/// come from the shared [`mlx_model`] join (`MODEL_TABLE` row + the linked candle descriptor), exactly
/// like the MLX path — so adding a family needs no new dispatch logic, just its provider crate linked.
/// Quant + LoRA/LoKr are **descriptor-gated** (sc-5126): resolved (via the same `resolve_quant` /
/// `resolve_adapters` the MLX path uses) only when the linked candle descriptor advertises them — i.e.
/// for Lens (Q4/Q8 + LoRA/LoKr); the sc-3675/sc-5096 families advertise neither, so they stay dense +
/// adapter-free exactly as before. No reference/img2img/control — those shapes fall back to the Python
/// worker upstream (`image_request_candle_eligible`). Reached only when `backend_candle_enabled`
/// (default off → production routing unchanged until parity).
#[cfg(all(target_os = "windows", feature = "backend-candle"))]
async fn generate_candle_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    _device_backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let adapter_label = candle_adapter_label(&request.model);
    // Join the MODEL_TABLE row with the linked candle descriptor (same resolver the MLX path uses).
    // `None` means the candle provider crate for this id wasn't linked/registered — fail loud rather
    // than silently stubbing.
    let model = mlx_model(&request.model).ok_or_else(|| {
        WorkerError::Engine(format!(
            "candle backend not linked for model {} (no registered generator)",
            request.model
        ))
    })?;
    let engine_id = model.engine_id();
    // Report the descriptor's tensor backend ("candle"), not the gpu-id device label
    // (`_device_backend`), on the streamed progress + inference events (sc-3678) — parity with the
    // macOS path's `model.backend()` override, so the worker log + the UI architecture pill clearly
    // attribute the run to Candle.
    let backend = if model.backend().is_empty() {
        "candle"
    } else {
        model.backend()
    };
    let repo = model_repo(request, &model);
    let weights_dir = huggingface_snapshot_dir(&settings.data_dir, &repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("candle weights snapshot not found for {repo}"))
    })?;

    // Descriptor-derived denoise/guidance surface (distilled families → no guidance/negative; guided
    // families → the scale + negative prompt). Identical to the MLX path; quant + LoRA are omitted.
    let steps = resolve_steps(request, &model);
    let guidance = resolve_guidance(request, &model);
    let true_cfg = resolve_true_cfg(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);

    // Per-payload flash/accel-attention (sc-3674): the UI Advanced toggle sends `advanced.flashAttn`
    // (default on). Process-global toggle, set before the generator loads (the candle pipeline reads
    // it at load) — race-free because the worker runs image jobs sequentially. The providers expose
    // the runtime knob under different names (SDXL `set_flash_attn`, Z-Image `set_accel_attn`); the
    // diffusion-transformer families (flux/flux2/qwen) bake it via the build feature with no runtime
    // toggle. No effect unless the crate was built with its flash/accel feature.
    let flash_attn = request
        .advanced
        .get("flashAttn")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    match request.model.as_str() {
        "sdxl" | "realvisxl" => candle_gen_sdxl::set_flash_attn(flash_attn),
        "z_image_turbo" => candle_gen_z_image::set_accel_attn(flash_attn),
        _ => {}
    }

    // Descriptor-gated quant + adapters (sc-5126). Lens advertises Q4/Q8 (Q8 default) + LoRA/LoKr, so
    // it resolves them like the MLX path; the sc-3675/sc-5096 families advertise neither and skip both
    // (dense bf16/fp16, no adapters) — preserving their shipped behavior. The router only lets a quant
    // request / LoRA reach this worker for a family that supports it (`image_request_candle_eligible`).
    let (quant, quant_bits) = if model.supports_quant() {
        resolve_quant(request)
    } else {
        (None, None)
    };
    let adapters = if model.supports_adapters() {
        resolve_adapters(request)?
    } else {
        Vec::new()
    };
    let adapter_count = adapters.len();

    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    // Record the effective CFG knob (guidance for guided families, else true_cfg) + quant bits in the
    // recipe, so a Lens asset's sidecar reflects the Q4/Q8 it ran at (parity with the MLX path).
    let raw_settings = mlx_raw_settings(request, &repo, steps, quant_bits, guidance.or(true_cfg));
    let spec = load_spec(weights_dir, quant, adapters, None);

    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("candle {engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let (out_w, out_h, pixels) = generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    negative_prompt.clone(),
                    None,
                    true_cfg,
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
        adapter_label,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// Consume the streamed generation events (step / decoding / image) from the blocking
/// thread: write each finished image as an asset fact, stream progress, and poll cancel
/// ~every 2s (draining the channel after a cancel so the blocking sender never blocks).
/// Shared by the base txt2img path ([`generate_stream`]) and the Z-Image strict-pose
/// control path ([`generate_zimage_control_stream`]). `total` is the number of images
/// the job produces (the request count, or the pose count).
#[allow(clippy::too_many_arguments)]
async fn consume_gen_events(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    adapter_label: &str,
    raw_settings: &JsonObject,
    total: usize,
    mut rx: tokio::sync::mpsc::Receiver<GenEvent>,
    cancel: CancelFlag,
    blocking: tokio::task::JoinHandle<WorkerResult<()>>,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let total_u32 = total as u32;
    let mut canceled = false;
    let mut last_cancel_check = Instant::now();
    // Per-image inference lifecycle events (sc-3450), parity with the Python worker's
    // `image_inference_start`/`image_inference_complete`. The first event for an index
    // marks its start; `GenEvent::Image` marks completion. This is the single shared
    // streaming seam, so every MLX image family reports the same phases on mlx-worker.log
    // + the in-app Logs screen.
    let mut started: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut mark_started = |index: usize| {
        if started.insert(index) {
            emit_event(
                "image_inference_start",
                json!({
                    "jobId": job.id,
                    "imageIndex": index,
                    "imageCount": total,
                    "backend": backend,
                }),
            );
        }
    };
    // Heartbeat + cancel-poll on a fixed interval, not only when the blocking
    // thread emits an event. The cold model-load phase (multi-GB load + quantize)
    // emits nothing, so without an interval arm the worker reports no Busy
    // heartbeat and honors no cancel until the first denoise step — long enough
    // for the API's staleness check to think it died (sc-4276 / F-MLXW-12;
    // mirrors the caption-job select!-with-interval).
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                if canceled {
                    continue; // drain remaining events so the blocking sender never blocks.
                }
                match event {
            GenEvent::Step {
                index,
                current,
                total: step_total,
            } => {
                mark_started(index);
                if last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    if cancel_requested_peek(api, &job.id).await {
                        // Trip the flag + show "Cancelling…", but stay non-terminal until the
                        // in-flight image actually stops (terminal Canceled posted after the
                        // blocking run returns) — sc-5515.
                        begin_image_cancel(api, &job.id, &cancel, plan, asset_writes, backend).await;
                        canceled = true;
                        continue;
                    }
                }
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, current, step_total, total_u32),
                        &format!("Image {}/{total} — step {current}/{step_total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Decoding { index } => {
                mark_started(index);
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        step_fraction(index, 1, 1, total_u32),
                        &format!("Image {}/{total} — decoding.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
            }
            GenEvent::Image {
                index,
                seed,
                width,
                height,
                pixels,
            } => {
                let fact = write_image_asset(
                    plan,
                    index,
                    seed,
                    width,
                    height,
                    pixels,
                    adapter_label,
                    raw_settings.clone(),
                    project_path,
                )?;
                asset_writes.push(Value::Object(fact));
                emit_event(
                    "image_inference_complete",
                    json!({
                        "jobId": job.id,
                        "imageIndex": index,
                        "backend": backend,
                    }),
                );
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.1 + 0.85 * ((index + 1) as f64 / total as f64),
                        &format!("Generated image {}/{total}.", index + 1),
                        Some(streaming_result(plan, asset_writes)),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                if !canceled && last_cancel_check.elapsed() >= Duration::from_secs(2) {
                    last_cancel_check = Instant::now();
                    if cancel_requested_peek(api, &job.id).await {
                        begin_image_cancel(api, &job.id, &cancel, plan, asset_writes, backend).await;
                        canceled = true;
                    }
                }
            }
        }
    }

    let task_result = blocking
        .await
        .map_err(|error| task_join_error("generation task join", error))?;
    if canceled {
        // The generation has now actually stopped, so post the TERMINAL Canceled here
        // (not at the earlier cancel poll, which only tripped the flag + showed
        // "Cancelling…"). This terminal write is what frees the worker row
        // (`jobs_store::update_job_progress`), so it lands exactly as the worker process
        // returns to its claim loop — the next queued job waits only until the GPU is
        // genuinely free, and the UI shows "Cancelling…" until completion (sc-5515).
        // result=None lets `coalesce` keep any partial images already streamed.
        let message = "Image generation canceled by user.";
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    task_result
}

// ---------------------------------------------------------------------------
// Z-Image strict-pose ControlNet (macOS, sc-3028): the Fun-Controlnet-Union
// `z_image_turbo_control` variant. One image per pose, each driven by a DWPose
// skeleton rendered from the pose's keypoints (see `openpose_skeleton`).
// ---------------------------------------------------------------------------

// Candle image lane labeling + engine-gate unit tests (sc-5099). Windows/candle-gated (the functions
// only exist on that build); pure string maps, no GPU.
#[cfg(all(test, target_os = "windows", feature = "backend-candle"))]
mod candle_label_tests {
    use super::*;

    #[test]
    fn candle_image_adapter_labels_are_per_family() {
        assert_eq!(candle_adapter_label("z_image_turbo"), "candle_z_image");
        assert_eq!(candle_adapter_label("flux_schnell"), "candle_flux");
        assert_eq!(candle_adapter_label("flux_dev"), "candle_flux");
        assert_eq!(candle_adapter_label("flux2_klein_9b"), "candle_flux2");
        assert_eq!(candle_adapter_label("qwen_image"), "candle_qwen");
        assert_eq!(candle_adapter_label("lens"), "candle_lens");
        assert_eq!(candle_adapter_label("lens_turbo"), "candle_lens");
        assert_eq!(candle_adapter_label("sdxl"), "candle_sdxl");
        assert_eq!(candle_adapter_label("realvisxl"), "candle_sdxl");
        // Every wired engine carries a `candle_`-prefixed label, distinct from the `mlx_` labels.
        for model in [
            "z_image_turbo",
            "flux_schnell",
            "flux_dev",
            "flux2_klein_9b",
            "qwen_image",
            "lens",
            "lens_turbo",
            "sdxl",
            "realvisxl",
        ] {
            assert!(candle_adapter_label(model).starts_with("candle_"));
        }
    }

    #[test]
    fn is_candle_engine_covers_only_the_wired_txt2img_families() {
        for model in [
            "sdxl",
            "realvisxl",
            "z_image_turbo",
            "flux_schnell",
            "flux_dev",
            "flux2_klein_9b",
            "qwen_image",
            "lens",
            "lens_turbo",
        ] {
            assert!(is_candle_engine(model), "{model} should be a candle engine");
        }
        // Non-candle families + non-base variants (edit ids, the kv distill) are not in the lane.
        for model in [
            "chroma1_hd",
            "kolors",
            "z_image_edit",
            "qwen_image_edit",
            "flux2_klein_9b_kv",
            "wan_2_2",
        ] {
            assert!(!is_candle_engine(model), "{model} must not be a candle engine");
        }
    }
}
