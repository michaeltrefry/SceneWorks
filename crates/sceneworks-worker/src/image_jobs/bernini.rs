// ---------------------------------------------------------------------------
// Bernini still-image companion (macOS, epic 4699 / sc-5424): the full Qwen2.5-VL semantic planner +
// Wan2.2-A14B renderer producing a SINGLE image. The Bernini engine descriptor is `Modality::Both`,
// so the image-typed `bernini_image` catalog id maps to the SAME `engine_id:"bernini"` the video
// `bernini` id uses (mirroring `z_image_edit → z_image_turbo` — two ids, one engine), and the still
// surface is served here through Image Studio + `image_jobs` instead of Video Studio.
//
// Two tasks: t2i (text→image, no conditioning) and i2i (`edit_image` — the source image, resolved
// from `sourceAssetId`, is handed to the engine as a `Conditioning::Reference`; the planner ViT/VAE-
// encodes it at its own native resolution, so the worker does NOT pre-fit it to the output W×H). The
// worker forces `frames:1` + `video_mode:"t2i"|"i2i"` so the engine returns `GenerationOutput::Images`
// (a single still) rather than a clip. No LoRA (the descriptor reports `supports_lora:false`); steps +
// guidance flow through the standard resolvers (the descriptor advertises `supports_guidance:true` +
// `supports_negative_prompt:true`). Q4 default / Q8 opt-in at load. The turnkey `SceneWorks/bernini-mlx`
// snapshot is shared with the video id (resolved via [`crate::video_jobs::resolve_bernini_model_dir`]).
//
// Scope note: the engine's i2i is a planner-guided structural re-render (the source feeds the ViT/VAE
// conditioning), NOT a denoise-strength img2img — the `Conditioning::Reference` strength is ignored by
// the engine, so the worker passes `None` and does not surface a strength knob that would do nothing.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Bernini image asset (matches the `bernini_image` MODEL_TABLE
/// row's `adapter_label`, so the per-asset `adapter` + the generation-set `adapter` agree).
#[cfg(target_os = "macos")]
const BERNINI_IMAGE_ADAPTER: &str = "mlx_bernini";

/// True when this is a Bernini still-image job whose weights resolve: the `bernini_image` id + a
/// resolvable snapshot dir (env override → app-managed → turnkey download). Both t2i and i2i route
/// here (i2i adds the source conditioning); plain t2i is NOT served by the generic `mlx_available`
/// path because that path leaves `frames`/`video_mode` unset, which the engine would treat as a
/// (multi-frame) video request — so Bernini stills must use this dedicated path.
#[cfg(target_os = "macos")]
fn bernini_image_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "bernini_image"
        && crate::video_jobs::resolve_bernini_model_dir(settings).is_ok()
}

/// The Bernini engine task string for a SceneWorks image mode: `edit_image` → `i2i` (source-image
/// edit), everything else → `t2i` (text→image). Selects the engine guidance/conditioning path
/// (`resolve_vit_mode`/`task_to_vit_mode`); both still tasks resolve to `vae_txt_vit_wapg`.
#[cfg(target_os = "macos")]
fn bernini_image_engine_task(mode: &str) -> &'static str {
    if mode == "edit_image" {
        "i2i"
    } else {
        "t2i"
    }
}

/// MLX quantization for a Bernini image load: Q4 default (the validated 64 GB-fitting tier, sc-4709
/// ~44 GB peak), Q8 opt-in via the advanced `mlxQuantize:8` control, explicit `<= 0` ⇒ bf16 (power
/// users with ample RAM). Mirrors the video path's [`crate::video_jobs::resolve_bernini_quant`] (Q4
/// default, not the generic image `resolve_quant`'s Q8 default — the snapshot is ~93 GB at bf16).
#[cfg(target_os = "macos")]
fn resolve_bernini_image_quant(request: &ImageRequest) -> (Option<Quant>, Option<i64>) {
    match request.advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    }) {
        Some(bits) if bits <= 0 => (None, None),
        Some(bits) if bits <= 4 => (Some(Quant::Q4), Some(4)),
        Some(_) => (Some(Quant::Q8), Some(8)),
        None => (Some(Quant::Q4), Some(4)),
    }
}

/// Flat telemetry for a real MLX Bernini image generation (parity with the other edit handlers +
/// `bernini_raw_settings`). Records the engine task so the lineage shows whether the planner ran t2i
/// or i2i, plus the standard repo/steps/guidance/quant knobs.
#[cfg(target_os = "macos")]
fn bernini_image_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    task: &str,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("berniniTask".to_owned(), Value::String(task.to_owned()));
    raw.insert("editEngine".to_owned(), Value::String("bernini".to_owned()));
    raw
}

/// Generate one Bernini still (RGB8) at `seed`. Builds the engine request with `frames:Some(1)` +
/// `video_mode:Some(task)` so the engine returns a single image, and the (optional) i2i source as the
/// shared `conditioning`. Standard guidance family (`guidance` carries the CFG scale, negative prompt
/// forwarded); no LoRA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
fn bernini_image_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    task: &'static str,
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
        // A single still: `frames == 1` makes the engine return `GenerationOutput::Images`.
        frames: Some(1),
        video_mode: Some(task.to_owned()),
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("Bernini image generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images
                .pop()
                .ok_or_else(|| WorkerError::Engine("Bernini image produced no image".to_owned()))?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Bernini image returned non-image output".to_owned(),
        )),
    }
}

/// Real MLX Bernini still-image generation (epic 4699 / sc-5424): load the full planner+renderer once
/// (Q4 default), then one image per seed — t2i from the prompt alone, or i2i conditioned on the
/// `sourceAssetId` source. Mirrors [`generate_sensenova_edit_stream`]'s blocking-thread + streamed-
/// events shape; differs in forcing `frames:1` + the engine task string, no negative-prompt/CFG
/// special-casing (standard guidance family), and no reference-fit (the engine resizes the source
/// internally).
#[cfg(target_os = "macos")]
async fn generate_bernini_image_stream(
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
    let engine_id = model.engine_id();
    let weights_dir = crate::video_jobs::resolve_bernini_model_dir(settings)?;
    let backend = if model.backend().is_empty() {
        backend
    } else {
        model.backend()
    };
    let (quant, quant_bits) = resolve_bernini_image_quant(request);
    let steps = resolve_steps(request, &model);
    // Standard guidance family: `guidance` carries the CFG scale (engine `omega_txt`); the negative
    // prompt is forwarded (descriptor advertises both). No true-CFG.
    let guidance = resolve_guidance(request, &model);
    let negative_prompt = resolve_negative_prompt(request, &model);
    let repo = model_repo(request, &model);
    let task = bernini_image_engine_task(&request.mode);

    // i2i (`edit_image`): resolve the source image into the engine's `Conditioning::Reference`. The
    // engine ViT/VAE-encodes it at native resolution (no worker-side fit), and ignores the reference
    // strength (planner-guided structural re-render, not a denoise-strength img2img). t2i has no
    // conditioning. The routing gate (`bernini_image_mlx_eligible`) already requires a `sourceAssetId`
    // for `edit_image`, so this is defense in depth.
    let conditioning: Vec<Conditioning> = if task == "i2i" {
        let source_id = request
            .source_asset_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "bernini_image edit_image requires a source image (sourceAssetId).".to_owned(),
                )
            })?;
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            source_id,
            project_path,
        )?;
        vec![Conditioning::Reference {
            image,
            strength: None,
        }]
    } else {
        Vec::new()
    };

    let raw_settings =
        bernini_image_raw_settings(request, &repo, steps, quant_bits, guidance, task);
    let count = request.count as usize;
    let seeds: Vec<i64> = (0..count).map(|index| resolve_seed(request, index)).collect();
    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);

    let spec = load_spec(weights_dir, quant, Vec::new(), None);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, seeds, move |_index, seed, on_progress| {
                let (out_w, out_h, pixels) = bernini_image_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    steps,
                    guidance,
                    task,
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
        BERNINI_IMAGE_ADAPTER,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
