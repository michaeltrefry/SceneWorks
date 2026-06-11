/// The engine registry id for the Z-Image Fun-Controlnet-Union variant.
const ZIMAGE_CONTROL_ENGINE_ID: &str = "z_image_turbo_control";
/// Default Fun-Controlnet-Union control-weights repo + file (sc-2257 parity).
const ZIMAGE_CONTROL_REPO: &str = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1";
const ZIMAGE_CONTROL_FILE: &str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors";

/// The object-shaped `advanced.poses` entries (the strict-pose tier; empty otherwise).
fn pose_entries(request: &ImageRequest) -> Vec<&Value> {
    request
        .advanced
        .get("poses")
        .and_then(Value::as_array)
        .map(|poses| poses.iter().filter(|pose| pose.is_object()).collect())
        .unwrap_or_default()
}

/// True when this is a Z-Image strict-pose job (z-image + ≥1 pose) whose base weights
/// resolve — routed to the Fun-Controlnet-Union control path rather than plain txt2img.
/// Control-weights presence is checked in the stream so a missing checkpoint errors
/// loudly instead of silently dropping the poses to the txt2img path.
fn zimage_control_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == "z_image_turbo"
        && !pose_entries(request).is_empty()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Resolve the Fun-Controlnet-Union checkpoint (`advanced.controlWeights.{repo,filename}`
/// else defaults) to a single `.safetensors` in the HF cache. `None` when absent (the
/// model-download flow fetches it ahead of generation, like base weights).
fn resolve_control_weights_for(
    request: &ImageRequest,
    settings: &Settings,
    default_repo: &'static str,
    default_file: &'static str,
) -> Option<PathBuf> {
    let control = request
        .advanced
        .get("controlWeights")
        .and_then(Value::as_object);
    let str_field = |key: &str, default: &'static str| -> String {
        control
            .and_then(|control| control.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(default)
            .to_owned()
    };
    let repo = str_field("repo", default_repo);
    let filename = str_field("filename", default_file);
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, &repo)?;
    let path = snapshot.join(filename);
    path.exists().then_some(path)
}

/// Resolve the Z-Image Fun-Controlnet-Union checkpoint.
fn resolve_control_weights(request: &ImageRequest, settings: &Settings) -> Option<PathBuf> {
    resolve_control_weights_for(request, settings, ZIMAGE_CONTROL_REPO, ZIMAGE_CONTROL_FILE)
}

/// Pose ControlNet lock strength: `advanced.controlScale` (default 0.9, clamp [0,2]).
fn resolve_control_scale(request: &ImageRequest) -> f32 {
    request
        .advanced
        .get("controlScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(0.9)
        .clamp(0.0, 2.0)
}

/// A pose's parsed keypoints, ready for [`crate::openpose_skeleton::draw_wholebody`].
struct PoseInput {
    keypoints: Vec<crate::openpose_skeleton::Keypoint>,
    hands: Option<Vec<crate::openpose_skeleton::Hand>>,
    face: Option<Vec<crate::openpose_skeleton::Keypoint>>,
}

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

/// Load the Z-Image Fun-Controlnet-Union generator (base snapshot + control overlay).
fn zimage_control_spec(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_weights));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

#[cfg(all(target_os = "macos", test))]
fn zimage_control_load(
    weights_dir: PathBuf,
    control_weights: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = zimage_control_spec(weights_dir, control_weights, quant, adapters);
    mlx_gen::load(ZIMAGE_CONTROL_ENGINE_ID, &spec)
        .map_err(|error| WorkerError::Engine(format!("Z-Image control load failed: {error}")))
}

/// Generate one strict-pose image: the `control` skeleton drives the Fun-Controlnet-Union
/// pose branch at `control_scale`. Z-Image-Turbo is guidance-distilled (no CFG / negative).
///
/// `reference` is the optional identity img2img-init shared across the pose set (sc-3146):
/// `(image, strength)` adds a `Reference` conditioning next to the required `Control`, seeding
/// the denoise from the reference latents. `strength` is the engine's img2img strength (mflux
/// `image_strength` convention: higher = more init kept). `None` → the pose-only tier.
#[allow(clippy::too_many_arguments)]
fn zimage_control_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    width: u32,
    height: u32,
    seed: i64,
    steps: u32,
    control: Image,
    control_scale: f32,
    reference: Option<&(Image, f32)>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut conditioning = vec![Conditioning::Control {
        image: control,
        kind: ControlKind::Pose,
        scale: control_scale,
    }];
    if let Some((image, strength)) = reference {
        conditioning.push(Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        });
    }
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(steps),
        conditioning,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, on_progress)
        .map_err(|error| WorkerError::Engine(format!("control generation failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("control generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "control generator returned non-image output".to_owned(),
        )),
    }
}

fn zimage_control_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    control_scale: f32,
    pose_count: usize,
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Z-Image-Turbo is guidance-distilled — no CFG.
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert("controlScale".to_owned(), json!(control_scale));
    raw.insert("poseCount".to_owned(), json!(pose_count));
    raw
}

/// Real Z-Image strict-pose generation: one image per pose, each conditioned on a DWPose
/// skeleton rendered from the pose keypoints + locked by the Fun-Controlnet-Union branch.
/// Mirrors [`generate_mlx_stream`]'s blocking-thread + streamed-events shape (the MLX
/// generator is `!Send` + single-thread), reusing [`consume_gen_events`].
async fn generate_zimage_control_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;
    // Identity img2img-init (sc-2328 / sc-3146) — OPT-IN escape hatch, off by default. The
    // Fun-Controlnet-Union pose head denoises the pose FROM NOISE, so seeding from a reference
    // init fights the pose lock on few-step Turbo (validated marginal on 8-step Turbo; no single
    // strength holds BOTH identity and pose). It engages only when advanced.referenceStrength > 0
    // AND a referenceAssetId is present — parity with `MlxZImageAdapter._identity_init_requested`.
    // The reference is shared across the whole pose set (identity is constant; only the per-pose
    // skeleton changes). None → the pose-only tier (the validated sc-2257 default).
    let identity_init = resolve_zimage_identity_init(request, settings, project_path)?;

    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Z-Image weights not found".to_owned()))?;
    let control_weights = resolve_control_weights(request, settings).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Z-Image strict-pose control weights not found (download {ZIMAGE_CONTROL_REPO})."
        ))
    })?;
    let (quant, quant_bits) = resolve_quant(request);
    let zimage = mlx_model("z_image_turbo")
        .ok_or_else(|| WorkerError::InvalidPayload("z-image model row missing".to_owned()))?;
    let steps = resolve_steps(request, zimage);
    let control_scale = resolve_control_scale(request);
    let adapters = resolve_adapters(request)?;
    let repo = model_repo(request, zimage);
    let poses = parse_poses(request);
    let count = poses.len();
    let raw_settings =
        zimage_control_raw_settings(request, &repo, steps, quant_bits, control_scale, count);
    // Strict pose shares one seed across the set so noise-derived attributes (hair,
    // wardrobe, lighting) stay constant while only the pose changes (Python parity).
    let seed = resolve_seed(request, 0);

    let prompt = request.prompt.clone();
    let (width, height) = (request.width, request.height);
    let stickwidth = crate::openpose_skeleton::body_stickwidth(width, height);
    let adapter_count = adapters.len();
    let spec = zimage_control_spec(weights_dir, control_weights, quant, adapters);
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        ZIMAGE_CONTROL_ENGINE_ID,
        adapter_count,
        spec,
        "Z-Image control load failed".to_owned(),
        move |generator, tx, cancel| {
            let identity_init = identity_init.as_ref();
            drive_gen_items(tx, poses, move |_index, pose, on_progress| {
                let skeleton = crate::openpose_skeleton::draw_wholebody(
                    width,
                    height,
                    &pose.keypoints,
                    pose.hands.as_deref(),
                    pose.face.as_deref(),
                    stickwidth,
                );
                let control = Image {
                    width,
                    height,
                    pixels: skeleton.into_raw(),
                };
                let (out_w, out_h, pixels) = zimage_control_generate_one(
                    generator,
                    &prompt,
                    width,
                    height,
                    seed,
                    steps,
                    control,
                    control_scale,
                    identity_init,
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
        ZIMAGE_ADAPTER_LABEL,
        &raw_settings,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

/// The clamped identity img2img-init strength for the Z-Image strict-pose set, or `None` for the
/// pose-only tier (sc-3146). `Some(strength)` iff `advanced.referenceStrength > 0` AND a non-empty
/// `referenceAssetId` is present — parity with `MlxZImageAdapter._identity_init_requested`. The
/// strict-pose stream always carries poses (`zimage_control_available`), so the
/// bare-reference-without-poses rejection is handled upstream; here a `referenceStrength` set
/// without an asset simply falls back to pose-only, matching the Python gate rather than erroring.
///
/// `strength` is the user value clamped to `[0.05, 1.0]` and carries the mflux `image_strength`
/// convention **verbatim** (no numeric inversion): the mlx-gen Z-Image control engine and mflux
/// agree — higher strength → later denoise start (`init_time_step`) → output stays closer to the
/// init. Mirrors `MlxZImageAdapter._reference_strength` + the sidecar's verbatim forward. Pure
/// (request only) so the parity-sensitive gate + clamp are unit-testable without asset I/O.
fn zimage_identity_strength(request: &ImageRequest) -> Option<f32> {
    let strength = request
        .advanced
        .get("referenceStrength")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .filter(|strength| *strength > 0.0)?;
    let has_asset = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|id| !id.is_empty());
    has_asset.then(|| (strength as f32).clamp(0.05, 1.0))
}

/// Resolve the optional identity img2img-init for the Z-Image strict-pose set (sc-3146):
/// `Some((image, strength))` when [`zimage_identity_strength`] engages, decoding `referenceAssetId`
/// via [`load_reference_image`]; `None` for the default pose-only tier. The reference is shared
/// across the whole pose set (identity is constant; only the per-pose skeleton changes).
fn resolve_zimage_identity_init(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<(Image, f32)>> {
    let Some(strength) = zimage_identity_strength(request) else {
        return Ok(None);
    };
    let asset_id = request
        .reference_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .expect("zimage_identity_strength guarantees a non-empty referenceAssetId");
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(Some((image, strength)))
}

/// Resolve the Z-Image Image-Edit img2img init for `mode == "edit_image"` (epic 3529):
/// `Some((source, strength))` decoding `sourceAssetId` and pre-fitting it to the output W×H
/// (crop/pad/outpaint via [`should_fit_edit_source`]/[`fit_engine_image`] — never stretch an
/// off-aspect source); `None` when not an edit job or no source asset (the caller then falls
/// back to the identity-init reference path / plain txt2img). `strength` is the torch
/// `ZImageImg2ImgPipeline` knob (`advanced.strength`, default 0.6) forwarded verbatim to the
/// engine — its `init_time_step(steps, strength)` matches the diffusers img2img start step.
/// Both `z_image_edit` and `z_image_turbo` (mode `edit_image`) drive this one path (same
/// Turbo-weights img2img call in torch).
fn resolve_zimage_edit_init(
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
    let image = if should_fit_edit_source(request) {
        fit_engine_image(source, request.width, request.height, &request.fit_mode)?
    } else {
        source
    };
    let strength = advanced::f32_clamped(&request.advanced, "strength", 0.6, 0.05..=1.0);
    Ok(Some((image, strength)))
}

/// The asset `adapter` id for Z-Image (strict-pose shares the base z-image label).
const ZIMAGE_ADAPTER_LABEL: &str = "mlx_z_image";

// ---------------------------------------------------------------------------
// Character-Studio angle set + best-effort pose tier + fit_image (macOS, sc-3030):
// the per-iteration batch orchestration on top of FLUX.2-klein edit. An angle set
// loops the 11 canonical head angles (shared seed, per-angle prompt augment); the
// best-effort pose tier pairs each pose's body skeleton with the reference as a
// `[skeleton, reference]` multi-image set; fit_image pre-fits an Image-Edit source
// to the output W×H (crop/pad/outpaint) so off-aspect edits don't stretch. Faithful
// ports of `character_studio_angles.py` + the `MlxFlux2Adapter` / `fit_image` paths.
// ---------------------------------------------------------------------------

/// The 11 canonical Character-Studio angles, in order. Re-exported from the canonical
/// [`sceneworks_core::angle_kps`] source of truth (the same table the Key Point Library serves
/// as its built-in presets — sc-4434) so the worker and the library can never drift.
const CHARACTER_ANGLE_SET_ORDER: [&str; 11] = sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER;

/// The per-angle continuation clause appended to the user's prompt (parity with
/// `character_studio_angles.ANGLE_PROMPT_AUGMENTS`). Unknown angle → empty.
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
fn strip_base_prompt(base: &str) -> &str {
    base.trim().trim_end_matches([',', '.', ';'])
}

/// Append the per-angle clause to the user's base prompt (parity with
/// `augment_prompt_for_angle`). Empty base + unknown angle → empty string.
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

/// The pose-skeleton instruction appended to the prompt for the best-effort pose tier
/// (parity with `character_studio_angles.POSE_SKELETON_PROMPT`).
const POSE_SKELETON_PROMPT: &str =
    "matching the exact body pose shown in the OpenPose skeleton reference image";

/// Append the pose-skeleton cue to the user's base prompt (parity with
/// `augment_prompt_for_pose`).
fn augment_prompt_for_pose(base: &str) -> String {
    let base = strip_base_prompt(base);
    if base.is_empty() {
        POSE_SKELETON_PROMPT.to_owned()
    } else {
        format!("{base}, {POSE_SKELETON_PROMPT}")
    }
}
