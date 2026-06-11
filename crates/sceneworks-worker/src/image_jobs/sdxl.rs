/// Default h94 IP-Adapter snapshot repo (ViT-H encoder + plus/plus-face SDXL weights).
const SDXL_IP_ADAPTER_REPO: &str = "h94/IP-Adapter";
/// img2img strength for a plain SDXL edit (torch `SdxlDiffusersAdapter` default 0.6).
const SDXL_EDIT_STRENGTH: f32 = 0.6;
/// img2img strength for masked inpaint / outpaint (torch default 0.85).
const SDXL_INPAINT_STRENGTH: f32 = 0.85;
/// IP-Adapter scale when the request omits it — matches the torch plus-face default 0.7
/// (`SdxlDiffusersAdapter._ip_adapter_scale`); the engine's own fallback is 0.6.
const SDXL_IP_SCALE: f32 = 0.7;

/// Which advanced SDXL path a request maps onto (or `None` for plain txt2img, which stays
/// on [`generate_mlx_stream`]). Outpaint wins over a plain mask when `fit_mode == outpaint`
/// (the torch path checks outpaint first, then unions any user mask into the border).
enum SdxlSubMode {
    /// Reference image-prompt via IP-Adapter (txt2img + decoupled cross-attn).
    Ip,
    /// Plain img2img edit (Reference init only).
    Edit,
    /// Masked inpaint (Reference init + Mask).
    Inpaint,
    /// Outpaint = inpaint with a generated border mask (+ optional user-mask union).
    Outpaint,
}

fn non_empty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|id| !id.trim().is_empty())
}

/// The engine-backed SDXL family row for a model id (`sdxl` / `realvisxl`), if any.
fn sdxl_engine_model(model: &str) -> Option<&'static MlxModel> {
    mlx_model(model).filter(|entry| entry.engine_id == "sdxl")
}

/// Classify an SDXL job into an advanced sub-mode. `None` = plain txt2img (no reference,
/// not an edit) → handled by the base MLX path.
fn sdxl_sub_mode(request: &ImageRequest) -> Option<SdxlSubMode> {
    if request.mode == "edit_image" {
        if !non_empty(&request.source_asset_id) {
            return None;
        }
        if request.fit_mode == "outpaint" {
            return Some(SdxlSubMode::Outpaint);
        }
        if non_empty(&request.mask_asset_id) {
            return Some(SdxlSubMode::Inpaint);
        }
        return Some(SdxlSubMode::Edit);
    }
    if non_empty(&request.reference_asset_id) {
        return Some(SdxlSubMode::Ip);
    }
    None
}

/// True when this is an SDXL advanced job (sdxl-family model + an advanced sub-mode) whose
/// base weights resolve — routed here rather than to plain txt2img.
fn sdxl_advanced_available(request: &ImageRequest, settings: &Settings) -> bool {
    sdxl_engine_model(&request.model).is_some()
        && sdxl_sub_mode(request).is_some()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the IP-Adapter snapshot directory (`advanced.ipAdapterRepo` override, else the
/// h94 default). The engine loader finds the ViT-H encoder + plus/plus-face weights inside.
fn resolve_ip_adapter_dir(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    let repo = request
        .advanced
        .get("ipAdapterRepo")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(SDXL_IP_ADAPTER_REPO);
    huggingface_snapshot_dir(&settings.data_dir, repo)
}

/// Resolve a mask asset id to an RGB8 [`Image`] (the engine luma-converts + binarizes it).
fn load_mask_asset_image(
    settings: &Settings,
    project_id: &str,
    mask_asset_id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    load_reference_image(&settings.data_dir, project_id, mask_asset_id, project_path)
}

/// Composite `source` contained (long edge fits) + centered on a black `width`×`height`
/// canvas, using the **engine's** `contain_box` so the padded source lines up pixel-for-pixel
/// with [`mlx_gen::image::outpaint_border_mask`] (both derive the same kept rect).
fn sdxl_outpaint_canvas(source: &image::RgbImage, width: u32, height: u32) -> Image {
    use image::imageops::FilterType::Lanczos3;
    let (new_w, new_h, left, top) =
        mlx_gen::image::contain_box(source.width(), source.height(), width, height);
    let resized = image::imageops::resize(source, new_w.max(1), new_h.max(1), Lanczos3);
    let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
    image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
    Image {
        width,
        height,
        pixels: canvas.into_raw(),
    }
}

/// An [`Image`] (RGB8) as an `image::RgbImage` for host-side compositing.
fn engine_image_to_rgb(image: Image) -> WorkerResult<image::RgbImage> {
    image::RgbImage::from_raw(image.width, image.height, image.pixels)
        .ok_or_else(|| WorkerError::InvalidPayload("image buffer size mismatch".to_owned()))
}

/// Build the SDXL generator spec for an advanced job. `ip_adapter_dir` (Some only in IP mode)
/// adds the decoupled cross-attn weights at load — the engine then treats a `Reference` as the
/// image prompt rather than an img2img init.
fn sdxl_advanced_spec(
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if let Some(ip) = ip_adapter_dir {
        spec = spec.with_ip_adapter(WeightsSource::Dir(ip));
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

/// Generate one SDXL image conditioned on `conditioning` (Reference[/Mask]). SDXL is true-CFG
/// (negative prompt + guidance honoured). The img2img strength / IP scale ride the Reference
/// `strength` field, so no separate `req.strength` is needed.
#[allow(clippy::too_many_arguments)]
fn sdxl_advanced_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
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
        guidance,
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("sdxl advanced generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images
                .pop()
                .ok_or_else(|| WorkerError::Engine("sdxl advanced produced no image".to_owned()))?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "sdxl advanced returned non-image output".to_owned(),
        )),
    }
}

/// Recipe facts recorded on the assets (the sub-mode + strengths/IP scale that drove it).
#[allow(clippy::too_many_arguments)]
fn sdxl_advanced_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    mode_tag: &str,
    strength: f32,
    ip_scale: Option<f32>,
) -> JsonObject {
    let mut raw = mlx_raw_settings(request, repo, steps, quant_bits, guidance);
    raw.insert("sdxlMode".to_owned(), Value::String(mode_tag.to_owned()));
    raw.insert("strength".to_owned(), json!(strength));
    if let Some(scale) = ip_scale {
        raw.insert("ipAdapterScale".to_owned(), json!(scale));
    }
    raw
}

/// Real SDXL advanced generation: resolve the conditioning images on the async side, then load
/// once + generate `count` images on the blocking thread (the MLX generator is `!Send`). Reuses
/// [`consume_gen_events`] for streaming + asset writes.
async fn generate_sdxl_advanced_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    let model = sdxl_engine_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an SDXL engine model".to_owned()))?;
    let sub_mode = sdxl_sub_mode(request)
        .ok_or_else(|| WorkerError::InvalidPayload("not an SDXL advanced job".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SDXL weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, model);
    let guidance = resolve_guidance(request, model);
    let negative_prompt = resolve_negative_prompt(request, model);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, model);
    let adapter_label = model.adapter_label;
    let (width, height) = (request.width, request.height);

    // Build the (seed-independent) conditioning + decide whether IP weights load. Images are
    // decoded here on the async side and moved into the blocking task (each cloned per seed).
    let (conditioning, ip_adapter_dir, mode_tag, strength, ip_scale): (
        Vec<Conditioning>,
        Option<PathBuf>,
        &str,
        f32,
        Option<f32>,
    ) = match sub_mode {
        SdxlSubMode::Ip => {
            let reference_id = request
                .reference_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("IP-Adapter requires a reference image".to_owned())
                })?;
            let reference = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                reference_id,
                project_path,
            )?;
            let ip_dir = resolve_ip_adapter_dir(request, settings).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "SDXL IP-Adapter weights not found (download {SDXL_IP_ADAPTER_REPO})."
                ))
            })?;
            let scale = advanced::f32_clamped(
                &request.advanced,
                "ipAdapterScale",
                SDXL_IP_SCALE,
                0.0..=1.0,
            );
            (
                vec![Conditioning::Reference {
                    image: reference,
                    strength: Some(scale),
                }],
                Some(ip_dir),
                "ip_adapter",
                scale,
                Some(scale),
            )
        }
        SdxlSubMode::Edit | SdxlSubMode::Inpaint => {
            let source_id = request
                .source_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("SDXL edit requires a source image".to_owned())
                })?;
            let source = load_reference_image(
                &settings.data_dir,
                &request.project_id,
                source_id,
                project_path,
            )?;
            // Pre-fit the source to the output W×H (crop/pad) so an off-aspect edit doesn't
            // stretch — torch parity with `load_source_image` + `fit_image`.
            let source = fit_engine_image(source, width, height, &request.fit_mode)?;
            let is_inpaint = matches!(sub_mode, SdxlSubMode::Inpaint);
            let strength = advanced::f32_clamped(
                &request.advanced,
                "strength",
                if is_inpaint {
                    SDXL_INPAINT_STRENGTH
                } else {
                    SDXL_EDIT_STRENGTH
                },
                0.0..=1.0,
            );
            let mut conditioning = vec![Conditioning::Reference {
                image: source,
                strength: Some(strength),
            }];
            if is_inpaint {
                let mask_id = request
                    .mask_asset_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload("inpaint requires a mask image".to_owned())
                    })?;
                let mask =
                    load_mask_asset_image(settings, &request.project_id, mask_id, project_path)?;
                // Align the mask to the source with the SAME fit geometry.
                let mask = fit_engine_image(mask, width, height, &request.fit_mode)?;
                conditioning.push(Conditioning::Mask { image: mask });
            }
            (
                conditioning,
                None,
                if is_inpaint { "inpaint" } else { "edit" },
                strength,
                None,
            )
        }
        SdxlSubMode::Outpaint => {
            let source_id = request
                .source_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("outpaint requires a source image".to_owned())
                })?;
            let source = engine_image_to_rgb(load_reference_image(
                &settings.data_dir,
                &request.project_id,
                source_id,
                project_path,
            )?)?;
            let (src_w, src_h) = (source.width(), source.height());
            let canvas = sdxl_outpaint_canvas(&source, width, height);
            // White = generate (the padded border), black = keep (the centered source).
            let mut mask = mlx_gen::image::outpaint_border_mask(src_w, src_h, width, height);
            if non_empty(&request.mask_asset_id) {
                // Union the user edit region with the border (white wins) — pad-fit the user
                // mask onto the same contained geometry first.
                let mask_id = request.mask_asset_id.as_deref().unwrap().trim();
                let user_mask =
                    load_mask_asset_image(settings, &request.project_id, mask_id, project_path)?;
                let user_mask = fit_engine_image(user_mask, width, height, "pad")?;
                mask = mlx_gen::image::union_masks(&mask, &user_mask).map_err(|error| {
                    WorkerError::Engine(format!("outpaint mask union failed: {error}"))
                })?;
            }
            let strength = advanced::f32_clamped(
                &request.advanced,
                "strength",
                SDXL_INPAINT_STRENGTH,
                0.0..=1.0,
            );
            (
                vec![
                    Conditioning::Reference {
                        image: canvas,
                        strength: Some(strength),
                    },
                    Conditioning::Mask { image: mask },
                ],
                None,
                "outpaint",
                strength,
                None,
            )
        }
    };

    let raw_settings = sdxl_advanced_raw_settings(
        request, &repo, steps, quant_bits, guidance, mode_tag, strength, ip_scale,
    );
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count)
        .map(|index| resolve_seed(request, index))
        .collect();
    let total = seeds.len();

    let prompt = request.prompt.clone();
    let negative_prompt = negative_prompt.clone();
    let adapter_count = adapters.len();
    let spec = sdxl_advanced_spec(weights_dir, quant, adapters, ip_adapter_dir);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        "sdxl",
        adapter_count,
        spec,
        "sdxl advanced load failed".to_owned(),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let (out_w, out_h, pixels) = sdxl_advanced_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    conditioning.clone(),
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
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

// ---------------------------------------------------------------------------
// InstantID identity-preserving character image (macOS, epic 3109 engine / sc-3345
// integration): the production `instantid_realvisxl` model — InstantID on RealVisXL +
// the stock SDXL IdentityNet ControlNet + the native MLX face stack (SCRFD + ArcFace),
// all in-process with zero Python. Two modes only (torch parity): a single-identity
// `character_image` (the reference's natural head pose) and the 11-view Character-Studio
// angle set. Pose-library mode (`advanced.poses`) + face-restore (`advanced.faceRestore`)
// are NOT handled here — they stay on the torch `InstantIDAdapter` (engine sc-3117 /
// sc-3380 not yet ported), gated out by `instantid_available` so the torch worker claims
// them. fp16 only for now (the validated envelope); Q8/Q4 ride explicit `mlxQuantize`
// (unvalidated at 1024², gated by sc-3329 follow-up). The provider is the bespoke
// `mlx_gen_instantid::InstantId` (not an inventory `Generator`), so this is a dedicated
// stream parallel to `generate_sdxl_advanced_stream`, not an MLX_MODELS row.
// ---------------------------------------------------------------------------
