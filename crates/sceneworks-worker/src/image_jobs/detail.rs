/// The xinsir tile ControlNet repo (parity with Python `TILE_CONTROLNET_REPO`).
const TILE_CONTROLNET_REPO: &str = "xinsir/controlnet-tile-sdxl-1.0";
const DETAIL_DEFAULT_PROMPT: &str = "ultra detailed, sharp focus, fine texture, high quality";
const DETAIL_DEFAULT_NEGATIVE: &str = "blurry, soft, lowres, smooth, plastic";

/// The locked detail recipe (sc-2437 round-2 spike defaults), resolved from `advanced`.
#[derive(Clone)]
struct DetailParams {
    strength: f32,
    cn_scale: f32,
    steps: u32,
    guidance: f32,
    tile: u32,
    overlap: u32,
    prompt: String,
    negative: String,
    seed: i64,
}

fn resolve_detail_params(request: &ImageRequest) -> DetailParams {
    DetailParams {
        strength: advanced::f32_clamped(&request.advanced, "strength", 0.55, 0.2..=1.0),
        cn_scale: advanced::f32_clamped(&request.advanced, "cnScale", 0.7, 0.1..=1.5),
        steps: advanced::u32_clamped(&request.advanced, "steps", 24, 1..=60),
        guidance: advanced::f32_clamped(&request.advanced, "guidanceScale", 5.0, 1.0..=15.0),
        tile: advanced::u32_clamped(&request.advanced, "tile", 1024, 512..=1536),
        overlap: advanced::u32_clamped(&request.advanced, "overlap", 128, 0..=512),
        prompt: advanced::str(&request.advanced, "prompt", DETAIL_DEFAULT_PROMPT),
        negative: advanced::str(&request.advanced, "negativePrompt", DETAIL_DEFAULT_NEGATIVE),
        // Python defaults the detail seed to 7 when the payload omits one.
        seed: request.seed.unwrap_or(7),
    }
}

/// Round a tile dimension up to the nearest multiple of 8 and clamp to the engine's
/// `[512, 2048]` SDXL bounds, so an arbitrary-sized crop can be run through the engine.
fn engine_dim(value: u32) -> u32 {
    value.div_ceil(8).saturating_mul(8).clamp(512, 2048)
}

/// Raised-cosine alpha ramp over the `overlap` borders so tiles blend seamlessly
/// (parity with Python `_detail_feather`). Row-major `tile_h`×`tile_w` weights.
fn detail_feather(tile_w: u32, tile_h: u32, overlap: u32) -> Vec<f32> {
    fn ramp(n: u32, overlap: u32) -> Vec<f32> {
        let mut weights = vec![1.0f32; n as usize];
        if overlap > 0 && n > overlap {
            for index in 0..overlap as usize {
                let edge = 0.5
                    - 0.5 * (std::f32::consts::PI * (index as f32 + 0.5) / overlap as f32).cos();
                weights[index] = edge;
                weights[n as usize - 1 - index] = edge;
            }
        }
        weights
    }
    let wx = ramp(tile_w, overlap);
    let wy = ramp(tile_h, overlap);
    let mut out = Vec::with_capacity((tile_w * tile_h) as usize);
    for &vy in &wy {
        for &vx in &wx {
            out.push(vy * vx);
        }
    }
    out
}

/// Build the SDXL generator spec with the tile ControlNet overlay.
fn detail_spec(weights_dir: PathBuf, control_file: PathBuf, quant: Option<Quant>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_file));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    spec
}

/// Refine one tile (already sized to engine-valid `eng_w`×`eng_h`): img2img on the tile
/// with the tile as the ControlNet image (control=same). Returns the refined RGB8 buffer.
#[allow(clippy::too_many_arguments)]
fn detail_refine_tile(
    generator: &dyn Generator,
    tile: Image,
    eng_w: u32,
    eng_h: u32,
    params: &DetailParams,
    seed: i64,
    cancel: &CancelFlag,
) -> WorkerResult<Vec<u8>> {
    let mut noop = |_progress: Progress| {};
    let request = GenerationRequest {
        prompt: params.prompt.clone(),
        negative_prompt: Some(params.negative.clone()),
        width: eng_w,
        height: eng_h,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(params.steps),
        guidance: Some(params.guidance),
        conditioning: vec![
            Conditioning::Reference {
                image: tile.clone(),
                strength: Some(params.strength),
            },
            Conditioning::Control {
                image: tile,
                kind: ControlKind::Other("tile".to_owned()),
                scale: params.cn_scale,
            },
        ],
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut noop)
        .map_err(|error| WorkerError::Engine(format!("detail tile failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => Ok(images
            .pop()
            .ok_or_else(|| WorkerError::Engine("detail tile produced no image".to_owned()))?
            .pixels),
        _ => Err(WorkerError::Engine(
            "detail tile returned non-image output".to_owned(),
        )),
    }
}

/// Tiled feathered detail refine (parity with Python `_refine_tiled_detail`). Returns the
/// recomposed image + the tile count. Runs on the blocking thread (the generator is `!Send`).
fn refine_tiled_detail(
    generator: &dyn Generator,
    source: &image::RgbImage,
    params: &DetailParams,
    cancel: &CancelFlag,
    on_tile: &mut dyn FnMut(usize, usize),
) -> WorkerResult<(image::RgbImage, usize)> {
    use image::imageops::FilterType::Lanczos3;
    let (width, height) = (source.width(), source.height());
    let step = params.tile.saturating_sub(params.overlap).max(1);
    let xs: Vec<u32> = (0..width.saturating_sub(params.overlap).max(1))
        .step_by(step as usize)
        .collect();
    let ys: Vec<u32> = (0..height.saturating_sub(params.overlap).max(1))
        .step_by(step as usize)
        .collect();
    let total = xs.len() * ys.len();
    let mut acc = vec![0.0f32; (width * height * 3) as usize];
    let mut wsum = vec![0.0f32; (width * height) as usize];
    let mut done = 0usize;
    for &y in &ys {
        for &x in &xs {
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(
                    "Detail enhancement canceled by user.".to_owned(),
                ));
            }
            let x0 = x.min(width.saturating_sub(params.tile));
            let y0 = y.min(height.saturating_sub(params.tile));
            let tile_w = params.tile.min(width - x0);
            let tile_h = params.tile.min(height - y0);
            let crop = image::imageops::crop_imm(source, x0, y0, tile_w, tile_h).to_image();
            // Run at an engine-valid size (mult-8, ≥512), then resize the refined tile back.
            let (eng_w, eng_h) = (engine_dim(tile_w), engine_dim(tile_h));
            let eng_crop = if (eng_w, eng_h) == (tile_w, tile_h) {
                crop
            } else {
                image::imageops::resize(&crop, eng_w, eng_h, Lanczos3)
            };
            let tile_img = Image {
                width: eng_w,
                height: eng_h,
                pixels: eng_crop.into_raw(),
            };
            let refined_px = detail_refine_tile(
                generator,
                tile_img,
                eng_w,
                eng_h,
                params,
                params.seed + done as i64,
                cancel,
            )?;
            let refined = image::RgbImage::from_raw(eng_w, eng_h, refined_px).ok_or_else(|| {
                WorkerError::InvalidPayload("detail refined tile size mismatch".to_owned())
            })?;
            let refined = if (eng_w, eng_h) == (tile_w, tile_h) {
                refined
            } else {
                image::imageops::resize(&refined, tile_w, tile_h, Lanczos3)
            };
            let feather = detail_feather(tile_w, tile_h, params.overlap);
            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    let f = feather[(ty * tile_w + tx) as usize];
                    let src = refined.get_pixel(tx, ty).0;
                    let gx = x0 + tx;
                    let gy = y0 + ty;
                    let acc_base = ((gy * width + gx) * 3) as usize;
                    acc[acc_base] += src[0] as f32 * f;
                    acc[acc_base + 1] += src[1] as f32 * f;
                    acc[acc_base + 2] += src[2] as f32 * f;
                    wsum[(gy * width + gx) as usize] += f;
                }
            }
            done += 1;
            on_tile(done, total);
        }
    }
    Ok((compose_feathered(&acc, &wsum, width, height), total))
}

/// Normalize the feather-weighted accumulator back to an RGB8 image.
///
/// Each pixel is the weighted mean `acc / wsum` of every tile that covered it. The divisor
/// MUST be the true accumulated feather weight: a pixel on the IMAGE boundary is covered by a
/// single edge tile whose raised-cosine feather ramps toward ~0 over the `overlap`-wide border
/// (there is no neighboring tile to sum the partition-of-unity back to 1). A previous
/// `.max(1.0)` guard divided those border pixels by 1.0 while `acc = src * f` (f→0), stamping a
/// dark rounded-corner vignette — most of the frame in the common single-tile case. Guard only
/// against a literal divide-by-zero; every pixel is covered by ≥1 tile because the tile origins
/// are clamped to the boundary, so `wsum` is strictly positive in practice (sc-8229).
fn compose_feathered(acc: &[f32], wsum: &[f32], width: u32, height: u32) -> image::RgbImage {
    let mut out = image::RgbImage::new(width, height);
    for gy in 0..height {
        for gx in 0..width {
            let w = wsum[(gy * width + gx) as usize].max(f32::EPSILON);
            let base = ((gy * width + gx) * 3) as usize;
            out.put_pixel(
                gx,
                gy,
                image::Rgb([
                    (acc[base] / w).clamp(0.0, 255.0) as u8,
                    (acc[base + 1] / w).clamp(0.0, 255.0) as u8,
                    (acc[base + 2] / w).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    out
}

/// Build the detail child-asset fact (lineage to the source) + generation set, matching the
/// Python `run_image_detail` result shape so `persist_reported_assets` indexes it identically.
#[allow(clippy::too_many_arguments)]
fn detail_result(
    request: &ImageRequest,
    genset_id: &str,
    created_at: &str,
    asset_id: &str,
    media_rel: &str,
    model: &str,
    params: &DetailParams,
    tiles: usize,
    width: u32,
    height: u32,
) -> JsonObject {
    let source_asset_id = request.source_asset_id.clone().unwrap_or_default();
    let detail_settings = json!({
        "enabled": true,
        "backbone": model,
        "controlNet": TILE_CONTROLNET_REPO,
        "strength": params.strength,
        "cnScale": params.cn_scale,
        "steps": params.steps,
        "guidanceScale": params.guidance,
        "tile": params.tile,
        "overlap": params.overlap,
        "tiles": tiles,
        "width": width,
        "height": height,
    });
    let fact = json!({
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "type": "image",
        "width": width,
        "height": height,
        "normalizedWidth": width,
        "normalizedHeight": height,
        "count": 1,
        "seed": params.seed,
        "displayName": "Detail enhanced",
        "createdAt": created_at,
        "mode": "image_detail",
        "model": model,
        "adapter": "mlx_sdxl",
        "prompt": params.prompt,
        "negativePrompt": params.negative,
        "loras": [],
        "stylePreset": "",
        "sourceAssetId": source_asset_id,
        "rawAdapterSettings": { "detail": detail_settings, "realModelInference": true },
        "parents": [source_asset_id],
        "extra": {
            "isDetailEnhanced": true,
            "detailFromAssetId": source_asset_id,
            "backbone": model,
            "strength": params.strength,
            "cnScale": params.cn_scale,
        },
    });
    let generation_set = json!({
        "id": genset_id,
        "mode": "image_detail",
        "model": model,
        "prompt": params.prompt,
        "negativePrompt": params.negative,
        "count": 1,
        "createdAt": created_at,
    });
    json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": "mlx_sdxl",
        "model": model,
        "generationSet": generation_set,
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Native MLX tile-ControlNet detail refine (`JobType::ImageDetail`) on the macOS engine.
pub(crate) async fn run_image_detail_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let model = if request.model.trim().is_empty() {
        "realvisxl".to_owned()
    } else {
        request.model.clone()
    };
    let engine_model = sdxl_engine_model(&model).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{model} does not support detail enhancement."))
    })?;
    let source_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Detail-enhance jobs require a source image asset.".to_owned(),
            )
        })?
        .to_owned();

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    tokio::fs::create_dir_all(project_path.join("assets").join("images").join(&genset_id)).await?;
    let backend = backend_label(&settings.gpu_id);

    let params = resolve_detail_params(&request);
    let (quant, _) = resolve_quant(&request);
    // Reuse the model's manifest/modelPath/cache resolution; engine_model gives the default repo.
    let weights_dir = resolve_weights_dir(&request, settings)?
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, engine_model.default_repo()));
    let weights_dir = weights_dir
        .ok_or_else(|| WorkerError::InvalidPayload("SDXL detail weights not found".to_owned()))?;
    let control_repo = advanced::str(
        &request.advanced,
        "tileControlNetRepo",
        TILE_CONTROLNET_REPO,
    );
    let control_dir =
        huggingface_snapshot_dir(&settings.data_dir, &control_repo).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "tile ControlNet weights not found (download {control_repo})."
            ))
        })?;
    let control_file = first_safetensors_path(&control_dir).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "no .safetensors under the tile ControlNet snapshot {}",
            control_dir.display()
        ))
    })?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Loading source image.",
            None,
            backend,
        ),
    )
    .await?;

    let source = engine_image_to_rgb(load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &source_id,
        &project_path,
    )?)?;

    let created_at = now_rfc3339();
    let asset_id = fresh_asset_id();
    let filename = format!("{}_detail_{}.png", &created_at[..10], &asset_id[6..14]);
    let media_rel = format!("assets/images/{genset_id}/{filename}");

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, usize)>(16);
    let blocking = {
        let params_ref = params.clone();
        let cancel = cancel.clone();
        let spec = detail_spec(weights_dir, control_file, quant);
        tokio::spawn(async move {
            crate::generator_cache::with_cached_generator(
                "sdxl",
                spec,
                "sdxl detail load failed",
                move |generator| {
                    let mut on_tile = |done: usize, total: usize| {
                        let _ = tx.blocking_send((done, total));
                    };
                    refine_tiled_detail(generator, &source, &params_ref, &cancel, &mut on_tile)
                },
            )
            .await
        })
    };

    let mut last_cancel_check = Instant::now();
    let mut canceled = false;
    while let Some((done, total)) = rx.recv().await {
        if canceled {
            continue; // drain so the blocking sender never blocks; terminal posts after stop.
        }
        if last_cancel_check.elapsed() >= Duration::from_secs(2) {
            last_cancel_check = Instant::now();
            if cancel_requested_peek(api, &job.id).await {
                // Trip the engine flag and show a NON-terminal "Cancelling…" (indeterminate bar);
                // the terminal Canceled is deferred to after the blocking refinement actually
                // stops so the worker row isn't freed while it's still grinding the current tile
                // (sc-5516; mirrors the image path sc-5515). Best-effort update.
                cancel.cancel();
                let _ = update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.0,
                        "Cancelling — finishing the current tile…",
                        None,
                        backend,
                    ),
                )
                .await;
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
                0.45 + 0.5 * (done as f64 / total.max(1) as f64),
                &format!("Refining detail tile {done}/{total}."),
                None,
                backend,
            ),
        )
        .await?;
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }

    let join = blocking
        .await
        .map_err(|error| task_join_error("detail task join", error))?;
    if canceled {
        // Refinement has actually stopped now — post the TERMINAL Canceled here. This terminal
        // write frees the worker row (`jobs_store::update_job_progress`) exactly as the worker
        // returns to its claim loop, so the next queued job waits only until the GPU is genuinely
        // free (sc-5516). The engine's own early return (`join`) is discarded as the clean cancel.
        let message = "Detail enhancement canceled by user.";
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
    let (refined, tiles) = join?;
    let (out_w, out_h) = (refined.width(), refined.height());
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");
    refined
        .save_with_format(&temp_path, image::ImageFormat::Png)
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let result = detail_result(
        &request,
        &genset_id,
        &created_at,
        &asset_id,
        &media_rel,
        &model,
        &params,
        tiles,
        out_w,
        out_h,
    );
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Detail enhancement complete.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}
