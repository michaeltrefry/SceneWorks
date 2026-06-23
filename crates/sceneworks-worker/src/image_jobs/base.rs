// Image fit/crop/pad geometry shared by the MLX edit handlers (flux2/qwen/sdxl/kolors/sensenova/
// zimage), the candle edit handlers (*_edit_candle.rs), and the video I2V resolve paths
// (video_jobs.rs). Kept in base.rs — included on macOS AND the `backend-candle` lane (and nowhere
// else) — so `crate::image_jobs::fit_engine_image` resolves on exactly the lanes that call it. Moved
// here from the macOS-only flux2.rs (sc-6231; the sc-6139 fit-mode refactor left it macOS-gated, which
// broke the candle build because video_jobs.rs / the candle edit handlers call it). No `#[cfg]` here:
// availability follows base.rs's own include cfg, which matches the callers'.

/// Where a `src_w`×`src_h` image lands when contained (long edge fits) and centered in
/// a `width`×`height` box: `(new_w, new_h, left, top)`. Parity with Python `_contain_box`
/// (shared by the pad fit so the kept region lines up). Integer-divides the offsets.
fn contain_box(src_w: u32, src_h: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
    let ratio = (width as f32 / src_w as f32).min(height as f32 / src_h as f32);
    let new_w = ((src_w as f32 * ratio).round() as u32).max(1);
    let new_h = ((src_h as f32 * ratio).round() as u32).max(1);
    (new_w, new_h, (width - new_w) / 2, (height - new_h) / 2)
}

/// Resize an RGB image to exactly `width`×`height` honoring `mode` without distorting it
/// (parity with Python `fit_image`, RGB path only — no inpaint mask exists on the MLX
/// FLUX.2 edit path, so `outpaint` degrades to `pad` geometry):
///   - `crop`:    scale to COVER (short edge fits), center-crop the overflow.
///   - `pad`/`outpaint`: scale to CONTAIN (long edge fits), center on a black canvas.
///   - `stretch`: legacy non-aspect-preserving resize.
fn fit_rgb(source: &image::RgbImage, width: u32, height: u32, mode: &str) -> image::RgbImage {
    use image::imageops::FilterType::Lanczos3;
    let width = width.max(1);
    let height = height.max(1);
    let (src_w, src_h) = (source.width(), source.height());
    match mode {
        "stretch" => image::imageops::resize(source, width, height, Lanczos3),
        "crop" => {
            let ratio = (width as f32 / src_w as f32).max(height as f32 / src_h as f32);
            // Ceil so the scaled image always fully covers the target before cropping.
            let new_w = width.max((src_w as f32 * ratio).ceil() as u32);
            let new_h = height.max((src_h as f32 * ratio).ceil() as u32);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let left = (new_w - width) / 2;
            let top = (new_h - height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
        // "pad" / "outpaint": contain + center on a black canvas (letterbox).
        _ => {
            let (new_w, new_h, left, top) = contain_box(src_w, src_h, width, height);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
            canvas
        }
    }
}

/// Fit an engine [`Image`] (RGB8) to `width`×`height` by `mode` via [`fit_rgb`].
/// `pub(crate)` so the video I2V resolve paths (`video_jobs.rs`, sc-6139) can pre-fit a
/// starting image to the output dims with the same crop/pad geometry as the image-edit lane.
pub(crate) fn fit_engine_image(
    source: Image,
    width: u32,
    height: u32,
    mode: &str,
) -> WorkerResult<Image> {
    let rgb =
        image::RgbImage::from_raw(source.width, source.height, source.pixels).ok_or_else(|| {
            WorkerError::InvalidPayload("edit source buffer size mismatch".to_owned())
        })?;
    let fitted = fit_rgb(&rgb, width, height, mode);
    Ok(Image {
        width: fitted.width(),
        height: fitted.height(),
        pixels: fitted.into_raw(),
    })
}

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
    Flux2DevControl,
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
    } else if flux2_dev_control_available(request, settings) {
        // FLUX.2-dev strict pose (advanced.poses) → Fun-Controlnet-Union. Wins over the edit/
        // best-effort pose tier below (`flux2_edit_available` needs a reference; a flux2_dev pose
        // job is the real ControlNet path, with the reference an opt-in img2img-init).
        Some(ImageRoute::Flux2DevControl)
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
            | ImageRoute::KolorsControl
            | ImageRoute::Flux2DevControl => pose_entries(request).len() as u32,
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
/// engine family or its snapshot is absent. Available on the candle lane too (sc-5501): the
/// off-Mac SenseNova-U1 VQA / interleave handlers resolve their snapshot through it.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
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
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, &model_repo(request, &model));
    // Ideogram 4 ships a turnkey with packed `q4/` (default) + `q8/` self-contained subdirs; point
    // the engine at the chosen quant's subdir rather than the repo root (epic 4725 / sc-5992),
    // mirroring the LTX bundle pattern. The packed weights auto-detect their quant on load. The
    // turbo variant (mlx-gen #488) shares the same turnkey — each subdir also carries the bundled
    // `turbo_lora.safetensors` the `ideogram_4_turbo` engine installs at load.
    if request.model == "ideogram_4" || request.model == "ideogram_4_turbo" {
        return Ok(snapshot.map(|root| ideogram_model_subdir(&root, request)));
    }
    // Boogu (epic 6387) ships a turnkey with pre-packed Q8 `base/ turbo/ edit/` subfolders (default) +
    // full-precision `*-bf16/`; point the engine at the variant's subfolder rather than the repo root
    // (the packed weights auto-detect their quant on load).
    if matches!(
        request.model.as_str(),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit"
    ) {
        return Ok(snapshot.map(|root| boogu_model_subdir(&root, request)));
    }
    Ok(snapshot)
}

/// Pick the engine-complete packed subdir of an Ideogram 4 turnkey `root`: `q8/` when the request
/// opts into Q8 (`advanced.mlxQuantize: 8`) AND it is downloaded, else the default `q4/`. Falls back
/// to `root` if neither subdir is present (a partially-downloaded bundle surfaces as a load error
/// rather than a silent half-load). On-demand `q8/` download is a follow-up; `q4/` is the manifest
/// default.
fn ideogram_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let wants_q8 = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .is_some_and(|bits| bits > 4);
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        dir.join("transformer/model.safetensors")
            .is_file()
            .then_some(dir)
    };
    if wants_q8 {
        if let Some(dir) = present("q8") {
            return dir;
        }
    }
    present("q4")
        .or_else(|| present("q8"))
        .unwrap_or_else(|| root.to_path_buf())
}

/// Pick the engine-complete subfolder of a Boogu turnkey `root` for the requested variant. Each
/// catalog id maps to a variant folder: `boogu_image`→`base`, `boogu_image_turbo`→`turbo`,
/// `boogu_image_edit`→`edit`. **Q8 is the shipped default** (the pre-packed `<variant>/` folder); an
/// explicit advanced `mlxQuantize <= 4` selects the full-precision `<variant>-bf16/` build instead —
/// the source the engine quantizes at load (no packed Q4 is shipped) or runs dense. Falls back to
/// whichever subfolder is present, then `root`, so a partially-downloaded bundle surfaces as a load
/// error rather than a silent half-load. (The `*-bf16/` files are an on-demand download fetched by
/// [`ensure_boogu_bf16_present`] before this resolves, sc-6568; if that fetch was skipped/failed the
/// bf16 request falls back to the Q8 folder.)
fn boogu_model_subdir(root: &Path, request: &ImageRequest) -> PathBuf {
    let variant = match request.model.as_str() {
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => "base",
    };
    let bf16 = format!("{variant}-bf16");
    let wants_bf16 = request
        .advanced
        .get("mlxQuantize")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .is_some_and(|bits| bits <= 4);
    let present = |name: &str| -> Option<PathBuf> {
        let dir = root.join(name);
        dir.join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
            .then_some(dir)
    };
    if wants_bf16 {
        if let Some(dir) = present(&bf16) {
            return dir;
        }
    }
    present(variant)
        .or_else(|| present(&bf16))
        .unwrap_or_else(|| root.to_path_buf())
}

/// On-demand fetch of the full-precision `<variant>-bf16/` Boogu subfolder (sc-6568). The S1 catalog
/// download pulls only the packed Q8 `<variant>/` subfolder, so when a job opts into the bf16 build
/// (advanced `mlxQuantize <= 4` — the same gate [`boogu_model_subdir`] uses to select it) and that
/// subfolder isn't present yet, pull just its files into the HF cache so `boogu_model_subdir` resolves
/// it. No-op when bf16 isn't requested, the model isn't Boogu, the turnkey snapshot isn't downloaded
/// yet (`boogu_model_subdir` then falls back to Q8 / surfaces the load error), or the bf16 subfolder
/// is already complete. Fails loud on a real download error — fast, before any compute; a missing `hf`
/// CLI leaves the subfolder absent so the request gracefully falls back to Q8. Mirrors
/// [`crate::video_jobs::ensure_ltx_q8_present`].
#[cfg(target_os = "macos")]
async fn ensure_boogu_bf16_present(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &ImageRequest,
) -> WorkerResult<()> {
    let variant = match request.model.as_str() {
        "boogu_image" => "base",
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => return Ok(()),
    };
    let wants_bf16 = request
        .advanced
        .get("mlxQuantize")
        .and_then(quant_int)
        .is_some_and(|bits| bits <= 4);
    if !wants_bf16 {
        return Ok(());
    }
    let Some(model) = mlx_model(&request.model) else {
        return Ok(());
    };
    let Some(root) = huggingface_snapshot_dir(&settings.data_dir, &model_repo(request, &model))
    else {
        // Turnkey not downloaded at all → leave it to the load path's "weights not found" error.
        return Ok(());
    };
    let bf16 = format!("{variant}-bf16");
    if root
        .join(&bf16)
        .join("transformer/diffusion_pytorch_model.safetensors")
        .is_file()
    {
        return Ok(());
    }
    let scratch = settings
        .data_dir
        .join("cache")
        .join(format!(".boogu-bf16-fetch-{}", job.id));
    tokio::fs::create_dir_all(&scratch).await?;
    // The bf16 subfolder nests transformer/mllm/vae (leaf-dir globs, like the catalog Q8 entry).
    let files = vec![
        format!("{bf16}/transformer/*"),
        format!("{bf16}/mllm/*"),
        format!("{bf16}/vae/*"),
    ];
    let result = crate::model_jobs::download_model_with_hf_cli(
        api,
        settings,
        job,
        &model_repo(request, &model),
        "main",
        &files,
        &scratch,
    )
    .await;
    let _ = tokio::fs::remove_dir_all(&scratch).await;
    result.map(|_| ())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_adapters(request: &ImageRequest, settings: &Settings) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let raw = lora_path(lora).ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA is missing a usable path.".to_owned())
        })?;
        // The path is attacker-controllable payload; confine it to an app-managed
        // root before any on-disk use (sc-5723 / WKA-002).
        let path = crate::normalize_app_managed_lora_path(settings, &raw)?;
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

/// N3 (epic 7114): a per-generation `sampler` / `scheduler` knob that names something the engine does
/// NOT advertise must never hard-fail the generation. `gen_core::Capabilities::validate_request` (and
/// each engine's own `validate`) rejects an unadvertised name with an `Err`, so the worker pre-filters
/// the knob here against the linked descriptor's advertised surface (`Capabilities.samplers` /
/// `.schedulers`): an advertised name passes through untouched; an unknown one — a stale recipe, a
/// per-backend capability gap (candle advertises a narrower set than mlx until P4), or manifest drift —
/// is dropped back to the engine default (`None`) and a `sampling_knob_unsupported` worker event is
/// emitted for observability. `None` and the `"default"` sentinel are already stripped at the read site,
/// so this only fires on a real, unsupported name. Shared by the MLX (`generate_stream`) + candle
/// (`generate_candle_stream`) image lanes and the video lane (`run_loaded_video_generation`).
pub(crate) fn normalize_sampling_knob(
    requested: Option<String>,
    advertised: &[&str],
    knob: &str,
    model_id: &str,
    job_id: &str,
    engine: &str,
) -> Option<String> {
    let name = requested?;
    if advertised.contains(&name.as_str()) {
        return Some(name);
    }
    tracing::warn!(
        "{engine}: requested {knob} {name:?} is not advertised (supported: {advertised:?}); \
         falling back to the engine default"
    );
    emit_event(
        "sampling_knob_unsupported",
        json!({
            "jobId": job_id,
            "engine": engine,
            "model": model_id,
            "knob": knob,
            "requested": name,
            "supported": advertised,
        }),
    );
    None
}

/// Read the raw per-generation sampler / scheduler / schedule-shift knobs from a job's `advanced`
/// block (the 1753 front-half carrier). `sampler` / `scheduler` strip the `"default"` sentinel + blanks
/// to `None`, so the engine default — N1's guaranteed no-op — is the ABSENCE of a name, not a magic
/// string; `scheduler_shift` accepts the `schedulerShift` (or legacy `timestepShift`) key as a number or
/// numeric string. Shared by the MLX (`generate_stream`) + candle (`generate_candle_stream`) image lanes
/// — the result is then realvisxl-forced (the lightning checkpoint) and N3-guarded via
/// [`normalize_sampling_knob`]. Returns `(sampler, scheduler, scheduler_shift)`.
pub(crate) fn read_advanced_sampling_knobs(
    advanced: &JsonObject,
) -> (Option<String>, Option<String>, Option<f32>) {
    let name = |key: &str| {
        advanced
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != "default")
            .map(str::to_owned)
    };
    let scheduler_shift = advanced
        .get("schedulerShift")
        .or_else(|| advanced.get("timestepShift"))
        .and_then(|value| value.as_f64().or_else(|| value.as_str()?.trim().parse().ok()))
        .map(|value| value as f32);
    (name("sampler"), name("scheduler"), scheduler_shift)
}

/// The curated sampler/scheduler menu (epic 7114 decision 2) the **bespoke** conditioned image paths
/// honor — the shared `gen_core` solver/scheduler vocabulary the unified-sampler engines gate on
/// (`Solver::from_name` / the additive `denoise_curated` path; mlx #537/#538/#539, candle #130). The
/// bespoke per-family paths (InstantID, Kolors-conditioned, PuLID — sc-7432) build CUSTOM request
/// structs OUTSIDE `generate_stream`'s generic plumbing, so they N3-normalize the per-request knob
/// against THIS menu instead of a `Capabilities` list: every engine's advertised set is a superset of
/// it (their native default is the only extra, and `"default"`/`None` already strip to the engine
/// default), so a name that survives [`normalize_sampling_knob`] here also passes the engine's own
/// `validate_request`. This is also the single source of truth the manifest⊆engine drift guard
/// (`engines.rs`) checks these out-of-`MODEL_TABLE` models against, so the runtime and the guard never
/// disagree. Derived from `gen_core` (the engines' own vocab), so it tracks the framework on BOTH
/// backends rather than hard-coding names. Returns `(samplers, schedulers)`.
pub(crate) fn curated_image_menu() -> (Vec<&'static str>, Vec<&'static str>) {
    (
        gen_core::sampling::Solver::ALL
            .iter()
            .map(|solver| solver.name())
            .collect(),
        gen_core::sampling::Scheduler::ALL
            .iter()
            .map(|scheduler| scheduler.name())
            .collect(),
    )
}

#[cfg(test)]
mod sampling_knob_tests {
    use super::*;

    #[test]
    fn advertised_name_passes_through() {
        let advertised = ["euler", "dpmpp_2m", "uni_pc"];
        assert_eq!(
            normalize_sampling_knob(
                Some("dpmpp_2m".to_owned()),
                &advertised,
                "sampler",
                "qwen_image",
                "job-1",
                "mlx",
            ),
            Some("dpmpp_2m".to_owned())
        );
    }

    #[test]
    fn unadvertised_name_falls_back_to_default() {
        // N3: a name the engine doesn't advertise (a legacy `dpmpp`/`unipc` recipe, or a candle
        // per-backend gap) is dropped to the engine default (`None`) instead of hard-failing the
        // generation in `validate_request`.
        let advertised = ["lightning"];
        assert_eq!(
            normalize_sampling_knob(
                Some("dpmpp".to_owned()),
                &advertised,
                "sampler",
                "qwen_image",
                "job-1",
                "mlx",
            ),
            None
        );
    }

    #[test]
    fn unset_knob_stays_unset() {
        let advertised = ["euler"];
        assert_eq!(
            normalize_sampling_knob(None, &advertised, "scheduler", "m", "j", "mlx"),
            None
        );
    }

    fn advanced(value: serde_json::Value) -> JsonObject {
        value.as_object().expect("object").clone()
    }

    // N1 (epic 7114): the guaranteed no-op default. A job with no sampling knobs — or the explicit
    // `"default"` sentinel the UI sends for "Model default" — must resolve to ALL `None`, i.e. the engine
    // runs its existing native path byte-for-byte. This guards the worker read against a future change
    // that silently injects a non-default sampler onto the default path.
    #[test]
    fn n1_default_advanced_is_a_no_op() {
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({}))),
            (None, None, None)
        );
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({
                "sampler": "default",
                "scheduler": "default",
                "steps": 30
            }))),
            (None, None, None)
        );
        // Blank / whitespace-only names are also treated as the default (no name).
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({"sampler": "  ", "scheduler": ""}))),
            (None, None, None)
        );
    }

    #[test]
    fn read_passes_real_names_and_shift_through() {
        assert_eq!(
            read_advanced_sampling_knobs(&advanced(serde_json::json!({
                "sampler": "dpmpp_2m",
                "scheduler": "sgm_uniform",
                "schedulerShift": 2.5
            }))),
            (
                Some("dpmpp_2m".to_owned()),
                Some("sgm_uniform".to_owned()),
                Some(2.5)
            )
        );
        // schedulerShift accepts a numeric string and the legacy `timestepShift` key.
        let (_, _, shift) = read_advanced_sampling_knobs(&advanced(serde_json::json!({
            "timestepShift": "1.5"
        })));
        assert_eq!(shift, Some(1.5));
    }
}

/// Optional prompt-enhancement settings resolved from a job request's `advanced` block and threaded
/// into a [`GenerationRequest`] (sc-6135). Mirrors the LTX-2.3 video path (`advanced.enhancePrompt` /
/// `enhanceTemperature` / `enhanceMaxTokens`). Only FLUX.2-dev / FLUX.2-dev-edit act on it — the
/// Mistral3 caption upsampler (sc-6030), text-only for txt2img and image-conditioned on the
/// reference image(s) for edit; every other engine ignores the fields, and the dev Image-Studio
/// toggle (manifest `ui.promptEnhance`) is the only surface that sets `enhancePrompt`, so this is a
/// no-op for all other models.
#[derive(Clone, Default)]
pub(crate) struct PromptEnhance {
    enabled: bool,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl PromptEnhance {
    /// Resolve from a job request's `advanced` settings (same keys as the LTX-2.3 video path).
    pub(crate) fn from_advanced(advanced: &JsonObject) -> Self {
        PromptEnhance {
            enabled: advanced::bool(advanced, "enhancePrompt"),
            temperature: advanced
                .get("enhanceTemperature")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            max_tokens: advanced
                .get("enhanceMaxTokens")
                .and_then(Value::as_u64)
                .map(|value| value as u32),
        }
    }

    /// Write the resolved enhancement settings onto a `GenerationRequest`.
    fn apply(&self, request: &mut GenerationRequest) {
        request.enhance_prompt = self.enabled;
        request.enhance_temperature = self.temperature;
        request.enhance_max_tokens = self.max_tokens;
    }
}

/// Generate one image (RGB8) at the given seed; `on_progress` streams denoise steps.
/// `guidance` is `None` for distilled variants (the engine rejects it on them).
///
/// `reference` is the optional identity img2img-init (sc-3619): `(image, strength)` adds a
/// `Reference` conditioning that seeds the denoise from the reference latents — the plain
/// (no-ControlNet) Z-Image reference-without-pose path, reusing the same engine img2img the
/// strict-pose tier already drives. `None` → plain txt2img. `enhance` carries the optional
/// caption-upsampling settings (sc-6135; only FLUX.2-dev acts on them).
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
    edit_mask: Option<&Image>,
    true_cfg: Option<f32>,
    sampler: Option<&str>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    enhance: &PromptEnhance,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut conditioning = match reference {
        Some((image, strength)) => vec![Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        }],
        None => Vec::new(),
    };
    // Inpaint / outpaint mask (Ideogram 4 edit, sc-6303): a `Conditioning::Mask` (white = repaint)
    // alongside the source `Reference`. Only the Ideogram edit path supplies one today; every other
    // base-path family passes `None`.
    if let Some(mask) = edit_mask {
        conditioning.push(Conditioning::Mask {
            image: mask.clone(),
        });
    }
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg,
        sampler: sampler.map(str::to_owned),
        scheduler: scheduler.map(str::to_owned),
        scheduler_shift,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    enhance.apply(&mut request);
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
    all(not(target_os = "macos"), feature = "backend-candle")
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
    let decoded = crate::image_decode::decode_image_any(&path)
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

/// img2img (Remix) strength for a plain Ideogram 4 edit with no mask — mirrors the sdxl/z-image 0.6
/// edit default and the engine's `DEFAULT_IMG2IMG_STRENGTH`. Shared by the macOS MLX edit path and the
/// candle in-lane edit (sc-6598), so it compiles off-Mac under `backend-candle` too.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IDEOGRAM_EDIT_STRENGTH: f32 = 0.6;
/// Heavier img2img strength for masked inpaint / outpaint (regenerate the painted region) — mirrors
/// the sdxl 0.85 inpaint default and the engine's `DEFAULT_INPAINT_STRENGTH`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IDEOGRAM_INPAINT_STRENGTH: f32 = 0.85;

/// Resolve the Boogu instruction-edit source: the `sourceAssetId` image, fit to the output W×H (so
/// it satisfies the engine's multiple-of-16 guard and aligns to the target aspect). Returns
/// `(source, strength)`; `None` when not an edit / no source. The `strength` is inert for Boogu (the
/// edit is structural — the engine ignores `Conditioning::Reference.strength`), so a full-strength
/// 1.0 is returned for the contract. No mask / outpaint path (the descriptor accepts only `Reference`).
///
/// Shared by the macOS MLX `generate_stream` and the off-Mac candle `generate_candle_stream` (sc-7524):
/// Boogu is the same engine family for T2I and edit on both backends (the registered `boogu_image_edit`
/// resolves the source `Reference` in-lane, like Ideogram), so both lanes resolve the edit source the
/// same way. Its deps (`load_reference_image`, `fit_engine_image`) already compile off-Mac under
/// `backend-candle` (the Ideogram edit path uses them too).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_boogu_edit(
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
    let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
    Ok(Some((source, 1.0)))
}

/// Resolve the Ideogram 4 `edit_image` conditioning (sc-6303) into the base MLX path's
/// `(source, strength, optional-mask)` shape (→ the engine's `Conditioning::Reference` +
/// `Conditioning::Mask`). Three sub-shapes, mirroring the sdxl edit classification:
///   * **img2img / Remix** — `sourceAssetId`, no mask: pre-fit the source to the output W×H
///     (crop/pad, never stretch) → `(source, 0.6, None)`.
///   * **masked inpaint** — `+ maskAssetId`: the mask fit with the same geometry → `(source, 0.85,
///     Some(mask))` (white = repaint).
///   * **outpaint** — `fit_mode == "outpaint"`: contain-pad the source onto the canvas and generate
///     the border via [`gen_core::imageops::outpaint_border_mask`] (using the ORIGINAL source dims so
///     it lines up), unioning any user mask (white wins).
///
/// `None` when not an edit job or no source asset (the caller falls back to plain txt2img).
///
/// Shared by the macOS MLX `generate_stream` and the off-Mac candle `generate_candle_stream`
/// (sc-6598): Ideogram is the same engine for T2I and edit on both backends, so both lanes resolve the
/// edit conditioning the same way. Its deps (`load_reference_image`, `fit_engine_image`, `non_empty`,
/// the `gen_core::imageops` mask helpers) are all already compiled off-Mac under `backend-candle`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_ideogram_edit(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32, Option<Image>)>> {
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
    let is_outpaint = request.fit_mode == "outpaint";
    let has_user_mask = non_empty(&request.mask_asset_id);
    let strength = advanced::f32_clamped(
        &request.advanced,
        "strength",
        if is_outpaint || has_user_mask {
            IDEOGRAM_INPAINT_STRENGTH
        } else {
            IDEOGRAM_EDIT_STRENGTH
        },
        0.05..=1.0,
    );

    if is_outpaint {
        // Pad the source onto the target canvas (contain) and regenerate the border. The border mask
        // uses the ORIGINAL source dims so it lines up with the padded canvas (same contain geometry
        // as `fit_engine_image`'s "outpaint"/pad). Any user mask unions into the border (white wins).
        let (src_w, src_h) = (source.width, source.height);
        let canvas = fit_engine_image(source, request.width, request.height, "outpaint")?;
        let mut mask =
            gen_core::imageops::outpaint_border_mask(src_w, src_h, request.width, request.height);
        if has_user_mask {
            let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
            let user_mask = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                mask_id,
                project_path,
            )?;
            let user_mask = fit_engine_image(user_mask, request.width, request.height, "pad")?;
            mask = gen_core::imageops::union_masks(&mask, &user_mask).map_err(|error| {
                WorkerError::Engine(format!("ideogram outpaint mask union failed: {error}"))
            })?;
        }
        return Ok(Some((canvas, strength, Some(mask))));
    }

    // img2img / inpaint: pre-fit the source to the output W×H so an off-aspect edit doesn't stretch.
    let source = fit_engine_image(source, request.width, request.height, &request.fit_mode)?;
    let mask = if has_user_mask {
        let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
        let user_mask = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            mask_id,
            project_path,
        )?;
        // Align the mask to the source with the SAME fit geometry.
        Some(fit_engine_image(
            user_mask,
            request.width,
            request.height,
            &request.fit_mode,
        )?)
    } else {
        None
    };
    Ok(Some((source, strength, mask)))
}

// ---------------------------------------------------------------------------
// Shared pose + angle-prompt helpers. Used by the macOS Z-Image strict-pose control path
// (`zimage.rs`) AND the InstantID lane (`instantid.rs`) on BOTH backends — the candle InstantID
// provider (sc-5491) needs them off-Mac, so they live here in the shared include rather than in the
// macOS-only `zimage.rs` (same reason `load_reference_image` does). All `include!`d image-job files
// share one module, so moving these here keeps them visible to `zimage.rs` on macOS unchanged.
// ---------------------------------------------------------------------------

/// True for a present, non-blank optional asset id (the conditioning-asset presence test shared by
/// the SDXL advanced sub-mode, PuLID, and InstantID gates). Moved here from the macOS-only `sdxl.rs`
/// so the candle InstantID lane (sc-5491) can use it off-Mac.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn non_empty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|id| !id.trim().is_empty())
}

/// The object-shaped `advanced.poses` entries (the strict-pose tier; empty otherwise).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn pose_entries(request: &ImageRequest) -> Vec<&Value> {
    request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .map(|poses| poses.iter().filter(|pose| pose.is_object()).collect())
        .unwrap_or_default()
}

/// A pose's parsed keypoints, ready for [`crate::openpose_skeleton::draw_wholebody`].
// The candle InstantID pose lane reads only `keypoints` (→ OpenPose body skeleton); `hands`/`face` are
// the Z-Image whole-body strict-pose path's (macOS), so allow them dead off-Mac.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct PoseInput {
    keypoints: Vec<crate::openpose_skeleton::Keypoint>,
    hands: Option<Vec<crate::openpose_skeleton::Hand>>,
    face: Option<Vec<crate::openpose_skeleton::Keypoint>>,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn parse_poses(request: &ImageRequest) -> Vec<PoseInput> {
    use crate::openpose_skeleton::{normalize_face, normalize_hands, normalize_keypoints};
    pose_entries(request)
        .into_iter()
        .map(|entry| PoseInput {
            keypoints: entry
                .get("keypoints")
                .map(normalize_keypoints)
                .unwrap_or_else(|| vec![None; 18]),
            hands: entry.get("hands").and_then(normalize_hands),
            face: entry.get("face").and_then(normalize_face),
        })
        .collect()
}

/// The per-angle continuation clause appended to the user's prompt (parity with
/// `character_studio_angles.ANGLE_PROMPT_AUGMENTS`). Unknown angle → empty.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn angle_prompt_augment(angle: &str) -> &'static str {
    match angle {
        "front" => {
            "frontal portrait, looking directly at the camera, head and shoulders, neutral expression"
        }
        "three_quarter_left" => {
            "three-quarter left profile, head turned slightly to the left, three-quarter view"
        }
        "three_quarter_right" => {
            "three-quarter right profile, head turned slightly to the right, three-quarter view"
        }
        "left_profile" => {
            "full left profile, head turned 90 degrees to the left, side view of the head"
        }
        "right_profile" => {
            "full right profile, head turned 90 degrees to the right, side view of the head"
        }
        "up" => "looking up, head tilted slightly upward toward the sky",
        "down" => "looking down, head tilted slightly downward toward the floor",
        "up_left" => {
            "looking up and to the left, head tilted slightly upward and turned slightly to the left"
        }
        "up_right" => {
            "looking up and to the right, head tilted slightly upward and turned slightly to the right"
        }
        "down_left" => {
            "looking down and to the left, head tilted slightly downward and turned slightly to the left"
        }
        "down_right" => {
            "looking down and to the right, head tilted slightly downward and turned slightly to the right"
        }
        _ => "",
    }
}

/// Strip the user's base prompt for augmentation: trim whitespace, then trailing
/// `,`/`.`/`;` — exactly Python's `(base or "").strip().rstrip(",.;")` (which can
/// leave a trailing space, e.g. `"a . "` → `"a "`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn strip_base_prompt(base: &str) -> &str {
    base.trim().trim_end_matches([',', '.', ';'])
}

/// Append the per-angle clause to the user's base prompt (parity with
/// `augment_prompt_for_angle`). Empty base + unknown angle → empty string.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn augment_prompt_for_angle(base: &str, angle: &str) -> String {
    let augment = angle_prompt_augment(angle);
    let base = strip_base_prompt(base);
    if !base.is_empty() && !augment.is_empty() {
        format!("{base}, {augment}")
    } else if !augment.is_empty() {
        augment.to_owned()
    } else {
        base.to_owned()
    }
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
    // sc-6568: a bf16 opt-in for Boogu fetches the full-precision `<variant>-bf16/` subfolder on
    // demand (the catalog ships only the Q8 default) before snapshot resolution. No-op for every
    // other model / the default Q8 path.
    ensure_boogu_bf16_present(api, settings, job, request).await?;
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
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    // RealVisXL Lightning (sc-6075): a standalone few-step *distilled checkpoint* (the
    // SDXL-Lightning distillation is baked into the weights — no acceleration LoRA). It must run
    // on the engine's `lightning` (Euler-trailing) few-step schedule, not the 30-step
    // `euler_ancestral` default, so the schedule matches the checkpoint regardless of the UI
    // payload — mirrors the qwen `*_lightning` sampler forcing. The engine then applies the
    // CFG-off, few-step recipe (steps/guidance come from the manifest defaults via the model row).
    let sampler = if request.model == "realvisxl_lightning" {
        Some("lightning".to_owned())
    } else {
        sampler
    };
    // N3 (epic 7114): drop a sampler/scheduler the linked engine descriptor doesn't advertise back to
    // the engine default + emit an event, instead of letting `validate_request` hard-fail the whole
    // generation over a sampling knob (a stale recipe, manifest drift, or a per-backend gap). The forced
    // `realvisxl_lightning` sampler above is always in that family's advertised set, so it passes through.
    let caps = &model.descriptor.capabilities;
    let sampler = normalize_sampling_knob(
        sampler,
        &caps.samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &caps.schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );
    // True-CFG families (Chroma) carry the CFG scale in `true_cfg`, not `guidance` (which their
    // engine rejects); `None` for every other family. The recipe records the effective CFG knob.
    let model_true_cfg = resolve_true_cfg(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let adapters = resolve_adapters(request, settings)?;
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
    // Ideogram 4 inpaint/outpaint mask (sc-6303), set by the ideogram edit branch below and threaded
    // to `generate_one` as a `Conditioning::Mask`. `None` for every other family / plain img2img.
    let mut ideogram_edit_mask: Option<Image> = None;
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
    } else if matches!(request.model.as_str(), "ideogram_4" | "ideogram_4_turbo")
        && request.mode == "edit_image"
    {
        // Ideogram 4 img2img (Remix) + mask inpaint / outpaint (Edit), sc-6303: `sourceAssetId` →
        // the engine's `Reference` (img2img init); a `maskAssetId` (inpaint) or `fit_mode ==
        // "outpaint"` adds a `Conditioning::Mask` (white = repaint), threaded via `ideogram_edit_mask`.
        // Works in both quality (`ideogram_4`) and turbo (same base + TurboTime LoRA). No IP-Adapter.
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => {
                ideogram_edit_mask = mask;
                (Some((source, strength)), None, None)
            }
            None => (None, None, None),
        }
    } else if request.model == "boogu_image_edit" && request.mode == "edit_image" {
        // Boogu instruction edit (epic 6387): `sourceAssetId` → the engine's `Reference` (the Qwen3-VL
        // vision tower reads it + it VAE-encodes into the DiT spatial latent); the prompt is the edit
        // instruction. No mask / IP-Adapter (the `boogu_image_edit` descriptor accepts only `Reference`).
        match resolve_boogu_edit(request, settings, project_path)? {
            Some((source, strength)) => (Some((source, strength)), None, None),
            None => (None, None, None),
        }
    } else {
        (None, None, None)
    };
    // The CFG scale passed to the engine as `true_cfg`: the FLUX.1-dev reference path's scale if
    // present, otherwise the true-CFG family scale (Chroma). `None` for the guidance-scalar and
    // distilled families, which carry CFG (if any) through `guidance` instead.
    let true_cfg = flux_true_cfg.or(model_true_cfg);

    // Ideogram 4 (epic 4725, sc-6501) is JSON-caption-only: a raw plain-text prompt is
    // out-of-distribution and stochastically renders the "Image blocked by safety filter"
    // placeholder (sc-6307, reference-confirmed faithful). The web Image Studio auto-expands plain
    // prompts into rich captions; this is the worker-side HARD GUARANTEE that raw plain text never
    // tokenizes — it wraps a non-caption prompt into a minimal valid caption (covers the API path
    // and any UI bypass). A prompt that is already a caption passes through unchanged. No-op for
    // every other family.
    let is_ideogram = crate::ideogram_caption::is_ideogram_model(&request.model);
    let prompt = if is_ideogram {
        crate::ideogram_caption::ensure_caption_prompt(&request.prompt)
    } else {
        request.prompt.clone()
    };
    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    // sc-6135: caption upsampling (FLUX.2-dev only; every other engine ignores it). Resolved from
    // the request's advanced `enhancePrompt` toggle, gated to dev by the manifest `ui.promptEnhance`.
    let enhance = PromptEnhance::from_advanced(&request.advanced);
    let spec = load_spec(weights_dir, quant, adapters, flux_ip_dir);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                    generate_one(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        negative_prompt.clone(),
                        identity_init.as_ref(),
                        ideogram_edit_mask.as_ref(),
                        true_cfg,
                        sampler.as_deref(),
                        scheduler.as_deref(),
                        scheduler_shift,
                        &enhance,
                        &cancel,
                        on_progress,
                    )
                };
                let (mut out_w, mut out_h, mut pixels) = render(seed, on_progress)?;
                let mut final_seed = seed;
                // Detect-and-recover safety net (sc-6501): the caption guard makes the placeholder
                // rare, but a residual one can still occur even with a caption. Detect it via the
                // baked-text heuristic (NOT a std/flatness check — the text lifts std to ~10) and
                // reseed transparently, keeping the first clean render. Gated to Ideogram 4; a no-op
                // elsewhere (and on turbo, which is CFG-free and cannot produce the placeholder).
                if is_ideogram
                    && crate::ideogram_caption::looks_like_placeholder(&pixels, out_w, out_h)
                {
                    let retries = crate::ideogram_caption::placeholder_recovery_retries();
                    for attempt in 0..retries {
                        if cancel.is_cancelled() {
                            break;
                        }
                        let retry_seed = crate::ideogram_caption::recovery_seed(seed, attempt);
                        tracing::warn!(
                            "ideogram 4 placeholder detected (seed {seed}); reseeding {retry_seed} \
                             (attempt {}/{retries})",
                            attempt + 1,
                        );
                        let (rw, rh, rpixels) = render(retry_seed, on_progress)?;
                        let recovered =
                            !crate::ideogram_caption::looks_like_placeholder(&rpixels, rw, rh);
                        out_w = rw;
                        out_h = rh;
                        pixels = rpixels;
                        final_seed = retry_seed;
                        if recovered {
                            break;
                        }
                    }
                }
                Ok(Some((final_seed, out_w, out_h, pixels)))
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

/// Whether `model` is served by the candle (Windows/CUDA) backend's generic lane (txt2img, plus the
/// Ideogram 4 in-lane edit, below). SDXL/RealVisXL (sc-3675) plus the four image families wired in
/// sc-5096 — z-image, flux schnell/dev, flux2-klein, qwen-image — plus Lens / Lens-Turbo (sc-5126, the
/// first candle family with quant + LoRA/LoKr) plus Ideogram 4 + Turbo (sc-6597/sc-6598, epic 6561).
/// `realvisxl` shares the candle `"sdxl"` engine via a weights swap; every other id maps 1:1 to its
/// `MODEL_TABLE` engine id. For the OTHER families, edit/control/reference shapes route to their bespoke
/// candle lanes (checked before this gate in the dispatch) or to the Python torch worker; Ideogram is
/// the exception — its img2img/mask edit is the SAME engine as its T2I, so `generate_candle_stream`
/// resolves the edit conditioning in-lane (mirroring the MLX `generate_stream`), no separate stream.
/// Lens is pure T2I; only quant + adapters, which `generate_candle_stream` resolves from the descriptor.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_candle_engine(model: &str) -> bool {
    matches!(
        model,
        "sdxl"
            | "realvisxl"
            // RealVisXL Lightning (sc-7176): shares the candle `sdxl` engine via a weights swap; the
            // few-step `lightning` sampler is forced in `generate_candle_stream`. txt2img-only (the
            // router defers its conditioning shapes to torch), so it rides the base candle txt2img lane.
            | "realvisxl_lightning"
            | "z_image_turbo"
            | "flux_schnell"
            | "flux_dev"
            | "flux2_klein_9b"
            // FLUX.2-dev (sc-7458): the 32B flagship rides the generic candle txt2img lane like klein.
            // `generate_candle_stream` resolves Q4 (manifest `mlx.quantize: 4` + the dev descriptor's
            // `supported_quants`) so the dense snapshot is staged in CPU RAM and quantized onto the GPU
            // at load. Edit/control/reference shapes route to their bespoke lanes or torch (story 4).
            | "flux2_dev"
            | "qwen_image"
            | "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "lens"
            | "lens_turbo"
            | "kolors"
            | "sensenova_u1_8b"
            | "sensenova_u1_8b_fast"
            | "ideogram_4"
            | "ideogram_4_turbo"
            // Boogu-Image-0.1 (sc-7524, epic 6831): Base + Turbo (txt2img) and the Edit checkpoint, all
            // on the generic candle lane. Like Ideogram, `boogu_image_edit`'s instruction edit is in-lane
            // (the engine resolves the source `Reference`), not a separate bespoke stream.
            | "boogu_image"
            | "boogu_image_turbo"
            | "boogu_image_edit"
    )
}

/// The per-asset `adapter` id recorded for a candle image engine (`candle_<family>`), the candle
/// sibling of the `MODEL_TABLE` `mlx_<family>` labels. Used both per-asset (`generate_candle_stream`)
/// and at the generation-set level (`adapter_id`) so the sidecar + result agree on the backend.
/// (sc-5099 extends this same labeling to the video + caption engines.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_adapter_label(model: &str) -> &'static str {
    match model {
        "z_image_turbo" => "candle_z_image",
        "flux_schnell" | "flux_dev" => "candle_flux",
        "flux2_klein_9b" | "flux2_dev" => "candle_flux2",
        "qwen_image" => "candle_qwen",
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => "candle_chroma",
        "lens" | "lens_turbo" => "candle_lens",
        "kolors" => "candle_kolors",
        "sensenova_u1_8b" | "sensenova_u1_8b_fast" => "candle_sensenova",
        "ideogram_4" | "ideogram_4_turbo" => "candle_ideogram",
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => "candle_boogu",
        // sdxl / realvisxl share the candle "sdxl" engine.
        _ => CANDLE_ADAPTER,
    }
}

/// The candle Ideogram 4 weights repo (bf16). Ideogram is the lone candle image family whose upstream
/// isn't candle-readable — the published `SceneWorks/ideogram-4-mlx` turnkey (the MODEL_TABLE default +
/// the macOS MLX repo) is MLX-quantized — so the candle lane loads bf16 from a separate repo (sc-6859),
/// the image sibling of the video `CANDLE_WAN_5B_REPO`. The bf16 weights live under the repo's `bf16/`
/// subdir so quant variants (`q4/`/`q8/`) can slot in later without a rename (cf. the MLX turnkey).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_IDEOGRAM_REPO: &str = "SceneWorks/ideogram-4";

/// Resolve the candle Ideogram weights repo: the off-Mac (`std::env::consts::OS`) download entry's
/// `repo` from the manifest (the bf16 repo) — the single source of truth, also driving the downloader —
/// else the [`CANDLE_IDEOGRAM_REPO`] default. Deliberately NOT the entry-level `repo`, which is the
/// macOS MLX turnkey. Mirrors how `candle_video_repo` overrides the MLX repo with the diffusers one.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_ideogram_repo(request: &ImageRequest) -> String {
    let os = std::env::consts::OS;
    request
        .model_manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|downloads| {
            downloads.iter().find_map(|entry| {
                let matches_os = entry
                    .get("platforms")
                    .and_then(Value::as_array)
                    .is_some_and(|platforms| platforms.iter().any(|p| p.as_str() == Some(os)));
                if !matches_os {
                    return None;
                }
                entry
                    .get("repo")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|repo| !repo.is_empty())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| CANDLE_IDEOGRAM_REPO.to_owned())
}

/// The precision subdir within the candle Ideogram repo snapshot. Candle is bf16-only today (the
/// provider rejects on-the-fly quant), so this resolves the `bf16/` subdir (mirroring the MLX turnkey's
/// `q4/`/`q8/`); a future candle quant variant slots in alongside it. Falls back to the snapshot root so
/// a flat (subdir-less) layout still loads.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_ideogram_subdir(root: &Path) -> PathBuf {
    let bf16 = root.join("bf16");
    if bf16.join("transformer").join("model.safetensors").is_file() {
        bf16
    } else {
        root.to_path_buf()
    }
}

/// The candle Boogu-Image-0.1 weights repo (bf16). Like Ideogram, Boogu's published turnkey
/// (`SceneWorks/boogu-image-mlx`, the `MODEL_TABLE` default + the macOS MLX repo) is MLX-quantized and
/// not candle-readable, so off-Mac the candle lane loads bf16 from a separate sibling repo (sc-7524) with
/// per-variant `base/ turbo/ edit/` subfolders (each a complete `transformer/ mllm/ vae/` snapshot the
/// provider's `pipeline::load_components` reads). The image sibling of the video `CANDLE_WAN_5B_REPO`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const CANDLE_BOOGU_REPO: &str = "SceneWorks/boogu-image";

/// Resolve the candle Boogu weights repo: the off-Mac (`std::env::consts::OS`) download entry's `repo`
/// from the manifest (the bf16 repo) — the single source of truth, also driving the downloader — else the
/// [`CANDLE_BOOGU_REPO`] default. Deliberately NOT the entry-level `repo`, which is the macOS MLX turnkey.
/// Mirrors `candle_ideogram_repo`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_boogu_repo(request: &ImageRequest) -> String {
    let os = std::env::consts::OS;
    request
        .model_manifest_entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|downloads| {
            downloads.iter().find_map(|entry| {
                let matches_os = entry
                    .get("platforms")
                    .and_then(Value::as_array)
                    .is_some_and(|platforms| platforms.iter().any(|p| p.as_str() == Some(os)));
                if !matches_os {
                    return None;
                }
                entry
                    .get("repo")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|repo| !repo.is_empty())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| CANDLE_BOOGU_REPO.to_owned())
}

/// Pick the engine-complete `base/ turbo/ edit/` subfolder of a candle Boogu snapshot `root` for the
/// requested variant (`boogu_image`→`base`, `boogu_image_turbo`→`turbo`, `boogu_image_edit`→`edit`) —
/// each a complete `transformer/ mllm/ vae/` snapshot. Candle is bf16-only (the provider rejects
/// on-the-fly quant), so there is no quant subdir to choose. Falls back to the snapshot root so a flat
/// (subdir-less) layout still loads. Mirrors the MLX `boogu_model_subdir` variant mapping.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_boogu_subdir(root: &Path, model: &str) -> PathBuf {
    let variant = match model {
        "boogu_image_turbo" => "turbo",
        "boogu_image_edit" => "edit",
        _ => "base",
    };
    let dir = root.join(variant);
    if dir.join("transformer").is_dir() {
        dir
    } else {
        root.to_path_buf()
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
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
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
    let is_ideogram = crate::ideogram_caption::is_ideogram_model(&request.model);
    // Boogu (sc-7524) is the second candle image family whose upstream turnkey isn't candle-readable: its
    // `SceneWorks/boogu-image-mlx` turnkey is MLX-quantized, so the candle lane loads bf16 from the sibling
    // `SceneWorks/boogu-image` (per-variant `base/ turbo/ edit/` subfolders), exactly as Ideogram below.
    let is_boogu = matches!(
        request.model.as_str(),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit"
    );
    // Ideogram + Boogu are the candle image families whose upstream isn't candle-readable: the published
    // turnkeys (`SceneWorks/ideogram-4-mlx` / `SceneWorks/boogu-image-mlx`, the MODEL_TABLE defaults + the
    // macOS MLX repos) are MLX-quantized, so the candle lane loads bf16 from a separate repo
    // (`SceneWorks/ideogram-4`'s `bf16/` subdir / `SceneWorks/boogu-image`'s per-variant subfolder)
    // instead — the image sibling of the candle video `candle_video_repo` override (sc-6859). macOS keeps
    // the MLX turnkeys. Every other candle family shares its upstream diffusers repo via `model_repo`.
    let repo = if is_ideogram {
        candle_ideogram_repo(request)
    } else if is_boogu {
        candle_boogu_repo(request)
    } else {
        model_repo(request, &model)
    };
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, &repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("candle weights snapshot not found for {repo}"))
    })?;
    let weights_dir = if is_ideogram {
        candle_ideogram_subdir(&snapshot)
    } else if is_boogu {
        candle_boogu_subdir(&snapshot, &request.model)
    } else {
        snapshot
    };

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
        // realvisxl_lightning shares the candle `sdxl` engine (sc-7176), so the SDXL flash toggle applies.
        "sdxl" | "realvisxl" | "realvisxl_lightning" => candle_gen_sdxl::set_flash_attn(flash_attn),
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
        resolve_adapters(request, settings)?
    } else {
        Vec::new()
    };
    let adapter_count = adapters.len();

    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    // Ideogram 4 (epic 4725, sc-6501) is JSON-caption-only: a raw plain-text prompt is out-of-
    // distribution and stochastically renders the "Image blocked by safety filter" placeholder. Wrap a
    // non-caption prompt into a minimal valid caption — the same worker-side guarantee the macOS path
    // applies via `ideogram_caption::ensure_caption_prompt`. No-op (a clone) for every other family.
    // (`is_ideogram` was resolved above with the weights repo.)
    let prompt = if is_ideogram {
        crate::ideogram_caption::ensure_caption_prompt(&request.prompt)
    } else {
        request.prompt.clone()
    };
    // In-lane edit conditioning (sc-6598 Ideogram / sc-7524 Boogu): resolve the source `Reference`
    // (+ optional `Mask` for Ideogram) + strength once, seed-independent — the candle sibling of the MLX
    // `generate_stream` edit path. Both families edit on the SAME engine as their T2I (no separate bespoke
    // stream), so the generic lane resolves the source here. `resolve_ideogram_edit` / `resolve_boogu_edit`
    // return `None` for a non-edit (T2I) job, and each is gated to its family so a stray job reaching this
    // generic lane is untouched. Boogu has no mask (the `boogu_image_edit` descriptor accepts only
    // `Reference` — the Qwen3-VL vision tower reads it + it VAE-encodes into the DiT reference latent).
    // Other candle edit families (sdxl/flux2/qwen/z-image) have their own bespoke streams (checked before
    // this dispatch).
    let (edit_reference, edit_mask) = if is_ideogram {
        match resolve_ideogram_edit(request, settings, project_path)? {
            Some((source, strength, mask)) => (Some((source, strength)), mask),
            None => (None, None),
        }
    } else if request.model == "boogu_image_edit" {
        match resolve_boogu_edit(request, settings, project_path)? {
            Some((source, strength)) => (Some((source, strength)), None),
            None => (None, None),
        }
    } else {
        (None, None)
    };
    let (width, height) = (request.width, request.height);
    // Per-payload sampler / scheduler / schedule-shift, mirroring the MLX `generate_stream` lane (the
    // 1753 front-half advanced carrier — epic 7114 P5, sc-7127). RealVisXL Lightning (sc-7176) forces the
    // few-step `lightning` id regardless of the payload: candle-gen-sdxl advertises `["ddim", "lightning"]`,
    // so it survives the N3 guard below. Every value is then run through `normalize_sampling_knob` against
    // this family's advertised surface — a name candle doesn't honor (candle adopts the unified framework in
    // P4, so most families advertise only their family default today) is dropped back to the engine default
    // + a `sampling_knob_unsupported` event, never a hard-fail. The curated knobs light up per-family with
    // zero worker change as the candle engines are adopted.
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    let sampler = if request.model == "realvisxl_lightning" {
        Some("lightning".to_owned())
    } else {
        sampler
    };
    let caps = &model.descriptor.capabilities;
    let sampler = normalize_sampling_knob(
        sampler,
        &caps.samplers,
        "sampler",
        &request.model,
        &job.id,
        backend,
    );
    let scheduler = normalize_sampling_knob(
        scheduler,
        &caps.schedulers,
        "scheduler",
        &request.model,
        &job.id,
        backend,
    );
    // sc-6135 / sc-7458: caption upsampling is FLUX.2-dev-only. On candle (off-Mac) dev now runs here,
    // but the Mistral3/Pixtral caption-upsampler vision tower is NOT ported (deferred to epic 6564
    // story 4), so `enhance` degrades to **passthrough**: it is carried onto the `GenerationRequest`
    // for uniformity, but the candle `Flux2Generator` ignores `enhance_prompt`, so the raw prompt is
    // used verbatim. Critically this is a no-op, NOT a fall-back to the Python torch worker — the dev
    // T2I job stays on candle (a future candle enhancer lights up here with no router change). Every
    // other candle family ignores the fields too.
    let enhance = PromptEnhance::from_advanced(&request.advanced);
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
                let render = |seed: i64, on_progress: &mut dyn FnMut(Progress)| {
                    generate_one(
                        generator,
                        &prompt,
                        width,
                        height,
                        seed,
                        steps,
                        guidance,
                        negative_prompt.clone(),
                        edit_reference.as_ref(),
                        edit_mask.as_ref(),
                        true_cfg,
                        // Per-payload sampler / scheduler / schedule-shift (sc-7127), already N3-guarded
                        // against this family's advertised surface above. RealVisXL Lightning forces
                        // `lightning`; most candle families advertise only their default until P4, so an
                        // unsupported request was dropped to `None` (the engine default) before reaching here.
                        sampler.as_deref(),
                        scheduler.as_deref(),
                        scheduler_shift,
                        &enhance,
                        &cancel,
                        on_progress,
                    )
                };
                let (mut out_w, mut out_h, mut pixels) = render(seed, on_progress)?;
                let mut final_seed = seed;
                // Ideogram 4 placeholder detect-and-reseed (sc-6858, parity with the macOS
                // `generate_stream` net, sc-6501): the caption guard above makes it rare, but a residual
                // "Image blocked by safety filter" placeholder can still occur even with a caption.
                // Detect via the baked-text heuristic and reseed transparently, keeping the first clean
                // render. Gated to Ideogram 4; a no-op for every other candle family, for turbo (CFG-free,
                // cannot produce it), and for an edit (the output is anchored to a real source latent, so
                // `looks_like_placeholder` returns false).
                if is_ideogram
                    && crate::ideogram_caption::looks_like_placeholder(&pixels, out_w, out_h)
                {
                    let retries = crate::ideogram_caption::placeholder_recovery_retries();
                    for attempt in 0..retries {
                        if cancel.is_cancelled() {
                            break;
                        }
                        let retry_seed = crate::ideogram_caption::recovery_seed(seed, attempt);
                        tracing::warn!(
                            "ideogram 4 placeholder detected (seed {seed}); reseeding {retry_seed} \
                             (attempt {}/{retries})",
                            attempt + 1,
                        );
                        let (rw, rh, rpixels) = render(retry_seed, on_progress)?;
                        let recovered =
                            !crate::ideogram_caption::looks_like_placeholder(&rpixels, rw, rh);
                        out_w = rw;
                        out_h = rh;
                        pixels = rpixels;
                        final_seed = retry_seed;
                        if recovered {
                            break;
                        }
                    }
                }
                Ok(Some((final_seed, out_w, out_h, pixels)))
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
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_label_tests {
    use super::*;

    #[test]
    fn candle_image_adapter_labels_are_per_family() {
        assert_eq!(candle_adapter_label("z_image_turbo"), "candle_z_image");
        assert_eq!(candle_adapter_label("flux_schnell"), "candle_flux");
        assert_eq!(candle_adapter_label("flux_dev"), "candle_flux");
        assert_eq!(candle_adapter_label("flux2_klein_9b"), "candle_flux2");
        assert_eq!(candle_adapter_label("flux2_dev"), "candle_flux2");
        assert_eq!(candle_adapter_label("qwen_image"), "candle_qwen");
        assert_eq!(candle_adapter_label("chroma1_hd"), "candle_chroma");
        assert_eq!(candle_adapter_label("chroma1_base"), "candle_chroma");
        assert_eq!(candle_adapter_label("chroma1_flash"), "candle_chroma");
        assert_eq!(candle_adapter_label("lens"), "candle_lens");
        assert_eq!(candle_adapter_label("lens_turbo"), "candle_lens");
        assert_eq!(candle_adapter_label("kolors"), "candle_kolors");
        assert_eq!(candle_adapter_label("sensenova_u1_8b"), "candle_sensenova");
        assert_eq!(
            candle_adapter_label("sensenova_u1_8b_fast"),
            "candle_sensenova"
        );
        assert_eq!(candle_adapter_label("ideogram_4"), "candle_ideogram");
        assert_eq!(candle_adapter_label("ideogram_4_turbo"), "candle_ideogram");
        // Boogu (sc-7524): all three variants share the `candle_boogu` asset stamp.
        assert_eq!(candle_adapter_label("boogu_image"), "candle_boogu");
        assert_eq!(candle_adapter_label("boogu_image_turbo"), "candle_boogu");
        assert_eq!(candle_adapter_label("boogu_image_edit"), "candle_boogu");
        assert_eq!(candle_adapter_label("sdxl"), "candle_sdxl");
        assert_eq!(candle_adapter_label("realvisxl"), "candle_sdxl");
        // Every wired engine carries a `candle_`-prefixed label, distinct from the `mlx_` labels.
        for model in [
            "z_image_turbo",
            "flux_schnell",
            "flux_dev",
            "flux2_klein_9b",
            "flux2_dev",
            "qwen_image",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "lens",
            "lens_turbo",
            "kolors",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "ideogram_4",
            "ideogram_4_turbo",
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
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
            "realvisxl_lightning",
            "z_image_turbo",
            "flux_schnell",
            "flux_dev",
            "flux2_klein_9b",
            "flux2_dev",
            "qwen_image",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "lens",
            "lens_turbo",
            "kolors",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "ideogram_4",
            "ideogram_4_turbo",
            // Boogu (sc-7524): Base + Turbo (txt2img) AND `boogu_image_edit` — unlike z_image_edit /
            // qwen_image_edit (bespoke streams), Boogu's instruction edit is in-lane on the generic
            // candle stream (like Ideogram), so `boogu_image_edit` IS a candle engine.
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
        ] {
            assert!(is_candle_engine(model), "{model} should be a candle engine");
        }
        // Non-candle families + non-base variants (the bespoke-stream edit ids, the kv distill) are not
        // in the generic lane. (kolors / sensenova ARE candle engines now — sc-5576 — for their base
        // txt2img shape; `boogu_image_edit` IS — sc-7524 — because its edit is in-lane, not bespoke.)
        for model in [
            "bernini_image",
            "z_image_edit",
            "qwen_image_edit",
            "flux2_klein_9b_kv",
            "wan_2_2",
        ] {
            assert!(!is_candle_engine(model), "{model} must not be a candle engine");
        }
    }
}
