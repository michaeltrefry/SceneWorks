/// True when this is a SenseNova it2i job: a SenseNova model + ≥1 reference (the character
/// `referenceAssetId`, or the Image-Edit `sourceAssetId` in `edit_image` mode) whose weights
/// resolve. Plain T2I (no reference) is NOT routed here — it rides the base `mlx_available` path.
/// Reuses [`qwen_edit_reference_ids`] (the generic `ref = referenceAssetId or sourceAssetId-if-edit`
/// rule, not Qwen-specific).
fn sensenova_edit_available(request: &ImageRequest, settings: &Settings) -> bool {
    is_sensenova_model(&request.model)
        && !qwen_edit_reference_ids(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Snap a dimension to SenseNova's 32-pixel cell (the engine rejects off-cell sizes), clamped to
/// the descriptor's [256, 2048] range. SenseNova's trained buckets are already 32-aligned; this
/// guards a hand-set advanced width/height.
fn sensenova_dim(value: u32) -> u32 {
    let snapped = value.div_ceil(32) * 32;
    snapped.clamp(256, 2048)
}

/// The SenseNova image-conditioning guidance (`true_cfg` → engine `img_cfg_scale`):
/// `advanced.imageGuidanceScale` else the per-mode default — 1.5 for Character Studio
/// (`character_image`, pulls harder toward the reference subject, sc-2015) / 1.0 for instruction
/// edit (the upstream it2i default). Floored at 1.0. Mirrors the Python `_image_guidance_scale`.
fn resolve_sensenova_img_cfg(request: &ImageRequest) -> f32 {
    let default = if request.mode == "character_image" {
        1.5
    } else {
        1.0
    };
    request
        .advanced
        .get("imageGuidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
        .max(1.0)
}

/// The SenseNova flow-match timestep shift (`scheduler_shift` → engine `timestep_shift`):
/// `advanced.schedulerShift` (or the legacy `timestepShift`) else 3.0; a non-positive value falls
/// back to 3.0. The only sampling knob SenseNova exposes (mirrors the Python adapter).
fn resolve_sensenova_timestep_shift(request: &ImageRequest) -> f32 {
    let raw = request
        .advanced
        .get("schedulerShift")
        .or_else(|| request.advanced.get("timestepShift"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(3.0);
    if raw > 0.0 {
        raw
    } else {
        3.0
    }
}

/// Flat telemetry for a SenseNova it2i generation (parity with `qwen_edit_raw_settings`).
#[allow(clippy::too_many_arguments)]
fn sensenova_edit_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
    img_cfg: f32,
    timestep_shift: f32,
    reference_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    if let Some(scale) = guidance {
        raw.insert("guidanceScale".to_owned(), json!(scale));
    }
    raw.insert("imageGuidanceScale".to_owned(), json!(img_cfg));
    raw.insert("schedulerShift".to_owned(), json!(timestep_shift));
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert(
        "editEngine".to_owned(),
        Value::String("sensenova_u1".to_owned()),
    );
    raw.insert("referenceCount".to_owned(), json!(reference_count));
    raw
}

/// Generate one SenseNova it2i image conditioned on `conditioning` (the reference set). Dual CFG:
/// `guidance` carries the text CFG, `true_cfg` the image guidance; `scheduler_shift` the
/// flow-match timestep shift. No negative prompt.
#[allow(clippy::too_many_arguments)]
fn sensenova_edit_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    guidance: Option<f32>,
    img_cfg: f32,
    timestep_shift: f32,
    conditioning: Vec<Conditioning>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        guidance,
        true_cfg: Some(img_cfg),
        scheduler_shift: Some(timestep_shift),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator.generate(&request, on_progress).map_err(|error| {
        WorkerError::Engine(format!("SenseNova edit generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("SenseNova edit produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "SenseNova edit returned non-image output".to_owned(),
        )),
    }
}

/// Real SenseNova-U1 it2i generation: load the unified model once (base or distilled `_fast`),
/// then one output per grouped iteration each conditioned on the shared reference set. Mirrors
/// [`generate_qwen_edit_stream`]'s blocking-thread + streamed-events shape; differs in the dual
/// CFG (`guidance` text + `true_cfg` image), no negative prompt, no pose tier (SenseNova has no
/// ControlNet), and no Lightning fetch (the `_fast` distill LoRA is merged inside the engine load).
async fn generate_sensenova_edit_stream(
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
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("SenseNova-U1 weights not found".to_owned()))?;
    let (quant, quant_bits) = resolve_quant(request);
    let steps = resolve_steps(request, &model);
    // Dual CFG: the text CFG flows through `guidance` (Some — SenseNova `supports_guidance`); the
    // image-conditioning guidance through `true_cfg`.
    let guidance = resolve_guidance(request, &model);
    let img_cfg = resolve_sensenova_img_cfg(request);
    let timestep_shift = resolve_sensenova_timestep_shift(request);
    let repo = model_repo(request, &model);
    let adapter_label = model.adapter_label();
    let (out_w, out_h) = (sensenova_dim(request.width), sensenova_dim(request.height));

    // Resolve the reference image(s) on the async side (decode → Send Image moved in).
    let reference_ids = qwen_edit_reference_ids(request);
    let mut references = Vec::with_capacity(reference_ids.len());
    for id in &reference_ids {
        references.push(load_reference_image(
            &settings.data_dir,
            &request.project_id,
            id,
            project_path,
        )?);
    }
    if references.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "SenseNova-U1 it2i requires a reference image".to_owned(),
        ));
    }
    // sc-3030 fit_image: pre-fit an off-aspect Image-Edit source to the output W×H (crop / pad /
    // outpaint→pad). Character-Studio references stay native (`should_fit_edit_source` excludes them).
    if should_fit_edit_source(request) {
        references = references
            .into_iter()
            .map(|reference| fit_engine_image(reference, out_w, out_h, &request.fit_mode))
            .collect::<WorkerResult<Vec<_>>>()?;
    }
    let conditioning = build_edit_conditioning(&references);

    // Per-iteration grouping: a Character-Studio angle set (11 shared-seed, per-angle prompt) or the
    // plain per-image reference path. SenseNova has no pose tier (excluded by `sensenova_mlx_eligible`).
    let grouping = flux2_grouping(request);
    let set_seed = resolve_seed(request, 0);
    let (seeds, prompts): (Vec<i64>, Vec<String>) = match &grouping {
        Flux2Grouping::Angles => {
            // Shared seed so noise-derived attributes stay constant across angles.
            let prompts = CHARACTER_ANGLE_SET_ORDER
                .iter()
                .map(|angle| augment_prompt_for_angle(&request.prompt, angle))
                .collect();
            (vec![set_seed; CHARACTER_ANGLE_SET_ORDER.len()], prompts)
        }
        Flux2Grouping::Plain => {
            let count = request.count as usize;
            let seeds = (0..count)
                .map(|index| resolve_seed(request, index))
                .collect();
            (seeds, vec![request.prompt.clone(); count])
        }
        Flux2Grouping::Poses(_) => {
            // Unreachable: strict pose is excluded by `sensenova_mlx_eligible` (no ControlNet).
            return Err(WorkerError::InvalidPayload(
                "SenseNova-U1 has no strict-pose (ControlNet) path".to_owned(),
            ));
        }
    };
    let total = seeds.len();

    let mut raw_settings = sensenova_edit_raw_settings(
        request,
        &repo,
        steps,
        quant_bits,
        guidance,
        img_cfg,
        timestep_shift,
        references.len(),
    );
    if matches!(grouping, Flux2Grouping::Angles) {
        raw_settings.insert("angleSet".to_owned(), Value::Bool(true));
    }

    // Angle-set identity-likeness scoring (epic 4406, sc-4409): generator-agnostic — a Character-
    // Studio angle set on SenseNova-U1 is scored through the same shared seam as InstantID / FLUX.2 /
    // Qwen. Stage the antelopev2 face stack (shared bundle, no-op if cached) and capture the source
    // identity reference + asset id; the `!Send` scorer is built ONCE in the closure and reused across
    // angles. Angle-set only; staging is non-fatal (failure → no scorer → scores omitted, set renders).
    let angle_set = matches!(grouping, Flux2Grouping::Angles);
    let face_stack_dir = if angle_set {
        match ensure_face_stack_dir(api, settings, job).await {
            Ok(dir) => Some(dir),
            Err(error) => {
                tracing::warn!(error = %error, "angle-set face-stack staging failed; likeness scores omitted");
                None
            }
        }
    } else {
        None
    };
    let likeness_source = (angle_set && face_stack_dir.is_some()).then(|| references[0].clone());
    let likeness_source_ref = reference_ids.first().cloned();

    // No user adapters by design (sc-6038): SenseNova-U1 is an 8B MoT autoregressive model with no
    // diffusion-LoRA merge path, and its manifest declares `loraCompatibility.families:
    // ["sensenova-u1"]` (its own family — no LoRA declares it), so the picker offers none. The empty
    // adapter list is intentional, not a dropped wiring (contrast `instantid.rs`, which DOES apply
    // user SDXL LoRAs).
    let spec = load_spec(weights_dir, quant, Vec::new(), None);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        0,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            // Per-job identity-likeness scorer built ONCE on the generator-worker thread (the `!Send`
            // face stack lives here); source embedded once, reused across every angle (sc-4409).
            let scorer = match (&face_stack_dir, &likeness_source) {
                (Some(dir), Some(source)) => {
                    crate::face_likeness::build_angle_set_scorer(dir, source)
                }
                _ => None,
            };
            drive_gen_items_scored(
                tx,
                seeds.into_iter().zip(prompts),
                move |_index, (seed, prompt), on_progress| {
                    let (w, h, pixels) = sensenova_edit_generate_one(
                        generator,
                        &prompt,
                        out_w,
                        out_h,
                        seed,
                        steps,
                        guidance,
                        img_cfg,
                        timestep_shift,
                        conditioning.clone(),
                        &cancel,
                        on_progress,
                    )?;
                    // Score this finished angle against the cached source embedding (sc-4409). The
                    // Image build + pixel clone is paid ONLY when a scorer exists (an angle set) — the
                    // common plain-edit path has no scorer, so this is a no-op with no clone. Profile/
                    // up/down → honest detected:false N/A; `None` scorer ⇒ field omitted.
                    let face_likeness = scorer.as_ref().and_then(|scorer| {
                        crate::face_likeness::score_angle_image(
                            Some(scorer),
                            &Image {
                                width: w,
                                height: h,
                                pixels: pixels.clone(),
                            },
                            likeness_source_ref.as_deref(),
                        )
                    });
                    Ok(Some((seed, w, h, pixels, face_likeness)))
                },
            )
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
// SDXL advanced conditioning (macOS, epic 3041 / sc-3060): reference (IP-Adapter),
// img2img edit, masked inpaint, and outpaint on the `sdxl` engine model. The plain
// txt2img + LoRA path stays on `generate_stream`; this branch handles every SDXL
// shape that used to fall through to the Python torch `SdxlDiffusersAdapter`. The
// engine selects the path from the loaded weights (`ip_adapter`) + conditioning combo
// (mlx-gen-sdxl PRs #137/#138); we just build the right `LoadSpec` + `Conditioning`.
// ---------------------------------------------------------------------------
