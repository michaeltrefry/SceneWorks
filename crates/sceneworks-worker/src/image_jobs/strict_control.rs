// Shared worker strict-control driver (epic 8236, sc-8243). The single source of truth for the
// Fun-Union family of strict-control engines and the one preprocess → conditioning core all of them
// share. Collapses the per-engine duplication that previously lived in `zimage.rs` / `flux2.rs` /
// `qwen.rs` (the three MLX registry-backed strict-control paths): every one of them rendered a DWPose
// skeleton, wrapped it in `Conditioning::Control { kind: Pose, scale }`, and (flux2 / z-image) appended
// an optional identity `Reference`. That core is now ONE function here; the per-engine streams keep only
// their genuinely-divergent numerics (guidance mode, negative prompt, PiD, weight resolution, raw_settings
// keys, control_scale defaults).
//
// **Single source of truth for `supported_kinds`.** [`STRICT_CONTROL_ENGINES`] is the authority sc-8244
// (manifest) and sc-8245 (web picker) consume — and the gate the worker uses to reject an unsupported
// `ControlKind` for a given engine. Today every reachable job is pose-only (`controlMode` unset →
// [`ControlKind::Pose`]); the canny / depth / user-supplied-passthrough branches are structurally present
// (so the driver is capable per S0) but only reached once the per-model exposure stories (sc-8248
// flux2 +canny/+depth, sc-8249 z-image +canny/+depth, sc-8250 qwen +canny/+depth) flip the request to set
// `advanced.controlMode`. Pose stays the proven path and is byte-preserved.
//
// macOS-only: the registry strict-control engines this drives are MLX (`gen_core::load(engine_id, …)`);
// the candle siblings (`qwen_control.rs` / `zimage_control.rs` / `flux2_control_candle.rs`) are bespoke
// non-registry providers and are NOT collapsed here (see the sc-8243 PR / follow-up).

/// One Fun-Union strict-control engine: its registry id, default control-weights repo, and the set of
/// [`ControlKind`]s it accepts. THE authority for `supported_kinds` (sc-8244 manifest / sc-8245 web
/// picker consume the same notion). `repo` is the *default* Fun-Union control repo for the engine; the
/// per-engine stream still honors an `advanced.controlWeights.{repo,filename}` override when resolving
/// the actual checkpoint (this row is the catalog default, not a hard pin).
#[derive(Clone, Copy, Debug)]
struct StrictControlEngine {
    /// The mlx-gen registry id (`gen_core::load(engine_id, spec)`).
    engine_id: &'static str,
    /// The default Fun-Union control-weights HF repo for this engine.
    repo: &'static str,
    /// The `ControlKind`s this engine accepts. Pose is the proven tier on every engine; canny / depth
    /// are unlocked per-model by sc-8248 / sc-8249 / sc-8250 (the driver supports them generically here).
    supported_kinds: &'static [ControlKind],
}

/// The Fun-Union strict-control catalog (S0 table). SINGLE SOURCE OF TRUTH for `(engine_id, control_repo,
/// supported_kinds)`:
/// - `flux1_dev_control` — `{Pose, Canny, Depth}` (Shakker Union-Pro-2.0; E2 sc-8239 / wiring sc-8244)
/// - `flux2_dev_control` — `{Pose, Canny, Depth}`
/// - `z_image_turbo_control` — `{Pose, Canny, Depth}`
/// - `z_image_control` — `{Pose, Canny, Depth}` (base, full-CFG; alibaba-pai Z-Image-Fun-Union; sc-8251)
/// - `qwen_image_control` — `{Pose, Canny, Depth}` (alibaba-pai 2512-Fun-Union; sc-8267 source swap / sc-8250 exposure)
///
/// The SDXL tile detail-upscale path (`ControlKind::Other("tile")`, `image_jobs/detail.rs`) is OUTSIDE
/// this family and is deliberately NOT listed.
const STRICT_CONTROL_ENGINES: &[StrictControlEngine] = &[
    StrictControlEngine {
        engine_id: "flux1_dev_control",
        repo: "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        supported_kinds: &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth],
    },
    StrictControlEngine {
        engine_id: "flux2_dev_control",
        repo: "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
        supported_kinds: &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth],
    },
    StrictControlEngine {
        engine_id: "z_image_turbo_control",
        repo: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        supported_kinds: &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth],
    },
    StrictControlEngine {
        // Base (non-distilled, full-CFG) Z-Image Fun-Controlnet-Union (sc-8251). Same VACE
        // Fun-Union control branch as the Turbo variant, but assembled from a base
        // `Tongyi-MAI/Z-Image` snapshot + the base control checkpoint.
        engine_id: "z_image_control",
        repo: "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
        supported_kinds: &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth],
    },
    StrictControlEngine {
        engine_id: "qwen_image_control",
        repo: "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union",
        supported_kinds: &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth],
    },
];

/// The catalog row for a registry strict-control engine id, or `None` if it is not a Fun-Union
/// strict-control engine (e.g. the SDXL tile detail path, which must never route through this driver).
fn strict_control_engine(engine_id: &str) -> Option<&'static StrictControlEngine> {
    STRICT_CONTROL_ENGINES
        .iter()
        .find(|entry| entry.engine_id == engine_id)
}

/// The catalog DEFAULT control-weights repo for a Fun-Union strict-control engine — the single source of
/// truth each engine's `controlWeights.repo`-override resolver falls back to. Panics on a non-Fun-Union
/// engine id (a programming error: only the three registry strict-control streams call this with their
/// own engine id).
fn strict_control_default_repo(engine_id: &str) -> &'static str {
    strict_control_engine(engine_id)
        .unwrap_or_else(|| panic!("{engine_id} is not a Fun-Union strict-control engine"))
        .repo
}

/// Validate a requested [`ControlKind`] against an engine's `supported_kinds` (the [`STRICT_CONTROL_ENGINES`]
/// authority). `Ok(())` when supported; a clear, actionable `InvalidPayload` otherwise. An unknown engine
/// id is itself an error — only the Fun-Union catalog engines route here.
fn validate_control_kind(engine_id: &str, kind: &ControlKind) -> WorkerResult<()> {
    let Some(entry) = strict_control_engine(engine_id) else {
        return Err(WorkerError::InvalidPayload(format!(
            "{engine_id} is not a Fun-Union strict-control engine"
        )));
    };
    if entry.supported_kinds.contains(kind) {
        return Ok(());
    }
    let supported = entry
        .supported_kinds
        .iter()
        .map(control_kind_label)
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkerError::InvalidPayload(format!(
        "{engine_id} does not support {} control (supported: {supported})",
        control_kind_label(kind),
    )))
}

/// A stable lowercase label for a [`ControlKind`] (telemetry / error messages / the `controlMode` request
/// field). `Other(name)` carries its bespoke name verbatim.
fn control_kind_label(kind: &ControlKind) -> String {
    match kind {
        ControlKind::Pose => "pose".to_owned(),
        ControlKind::Canny => "canny".to_owned(),
        ControlKind::Depth => "depth".to_owned(),
        ControlKind::Other(name) => name.clone(),
    }
}

/// Parse the requested control kind from the job. The default is [`ControlKind::Pose`] (the proven tier;
/// every current job omits `controlMode`, so existing pose jobs are byte-preserved). `advanced.controlMode`
/// — when a future per-model exposure story sets it — selects `canny` / `depth` / `pose`. An unknown value
/// is rejected loudly rather than silently falling back.
fn requested_control_kind(request: &ImageRequest) -> WorkerResult<ControlKind> {
    let Some(mode) = request
        .advanced
        .get("controlMode")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(ControlKind::Pose);
    };
    match mode.to_ascii_lowercase().as_str() {
        "pose" => Ok(ControlKind::Pose),
        "canny" => Ok(ControlKind::Canny),
        "depth" => Ok(ControlKind::Depth),
        other => Err(WorkerError::InvalidPayload(format!(
            "unknown controlMode '{other}' (expected pose, canny, or depth)"
        ))),
    }
}

/// A decoded user-supplied control map (`advanced.controlImage` asset id), already at native resolution.
/// Present iff the job carries an explicit control map to use verbatim instead of a preprocessor.
fn resolve_user_control_map(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<Image>> {
    let Some(asset_id) = request
        .advanced
        .get("controlImage")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
    else {
        return Ok(None);
    };
    let image = load_reference_image(&settings.data_dir, &request.project_id, asset_id, project_path)?;
    Ok(Some(image))
}

/// The asset id canny / depth auto-derive their control map FROM (sc-8244 source threading), or `None`
/// when the job carries no input image. Precedence: the Image-Edit / control `sourceAssetId` (the
/// canonical "input image" the picker sends), else the character `referenceAssetId` (so a control job
/// that only carried a reference still has something to derive from). Pure (request only) so the
/// precedence is unit-testable without asset I/O.
fn control_source_asset_id(request: &ImageRequest) -> Option<&str> {
    request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .or_else(|| {
            request
                .reference_asset_id
                .as_deref()
                .map(str::trim)
                .filter(|id| !id.is_empty())
        })
}

/// The decoded input image canny / depth auto-derive their control map FROM (sc-8244 source threading).
///
/// This is the raw photo the preprocessor runs over for `controlMode = canny | depth` when no explicit
/// `controlImage` passthrough is given — distinct from [`resolve_user_control_map`], which is a *pre-made*
/// control map used verbatim. The asset id is picked by [`control_source_asset_id`]; `None` when neither
/// a source nor a reference is present — the pose tier never needs a source (its skeleton is synthetic),
/// and canny/depth then surface the shared driver's clear "requires a source image" error rather than
/// silently producing nothing.
fn resolve_control_source(
    request: &ImageRequest,
    settings: &Settings,
    project_path: &Path,
) -> WorkerResult<Option<Image>> {
    let Some(asset_id) = control_source_asset_id(request) else {
        return Ok(None);
    };
    let image = load_reference_image(&settings.data_dir, &request.project_id, asset_id, project_path)?;
    Ok(Some(image))
}

/// Estimate a depth control map from an arbitrary input image (sc-8242).
///
/// This is the depth analogue of [`crate::canny::canny_control_image_default`]: given a `source` RGB
/// image, it runs the native-MLX Depth Anything V2 estimator ([`crate::depth::depth_control_image`])
/// and returns a normalized single-channel depth-control [`Image`] at the source's dimensions.
///
/// `source` is the raw image to estimate FROM (NOT a pre-made depth map — that flows through the
/// user-supplied-passthrough branch in [`preprocess_control_entry`] before this is ever reached).
/// `depth_weights_dir` is the resolved Depth-Anything-V2-Small snapshot dir; it must be present for
/// auto-estimation (the engine driver provisions it before the gen stream).
///
/// Errors when no `source` is available (auto depth has nothing to estimate from — the caller must
/// supply either a source image or a user-supplied depth map) or when the estimator weights are
/// unavailable. macOS-only (MLX inference); off-Mac there is no registry strict-control path.
#[cfg(target_os = "macos")]
fn depth_control_image(
    source: Option<&Image>,
    depth_weights_dir: Option<&Path>,
) -> WorkerResult<Image> {
    let source = source.ok_or_else(|| {
        WorkerError::InvalidPayload(
            "depth control requires either a source image to estimate from or a user-supplied \
             depth map (advanced.controlImage)"
                .to_owned(),
        )
    })?;
    let weights_dir = depth_weights_dir.ok_or_else(|| {
        WorkerError::Engine(
            "depth estimator weights are unavailable (Depth-Anything-V2-Small snapshot not provisioned)"
                .to_owned(),
        )
    })?;
    let rgb = image::RgbImage::from_raw(source.width, source.height, source.pixels.clone())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("depth source buffer size mismatch".to_owned())
        })?;
    let depth = crate::depth::depth_control_image(&rgb, weights_dir)?;
    Ok(Image {
        width: depth.width(),
        height: depth.height(),
        pixels: depth.into_raw(),
    })
}

/// Off-Mac stub: there is no registry strict-control path on the candle lane, so depth auto-estimation
/// is never reached — but the shared driver must still compile. Mirrors the macOS signature.
#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn depth_control_image(
    _source: Option<&Image>,
    _depth_weights_dir: Option<&Path>,
) -> WorkerResult<Image> {
    Err(WorkerError::InvalidPayload(
        "automatic depth estimation is macOS-only".to_owned(),
    ))
}

/// Provision the Depth Anything V2 (Small) estimator snapshot and return the directory holding
/// `model.safetensors` (what [`crate::depth::depth_control_image`] loads via `from_dir`).
///
/// Resolution order mirrors [`ensure_flux2_control_weights`]: an explicit `SCENEWORKS_DEPTH_ANYTHING_V2`
/// dir override → an existing HF cache snapshot → a lazy fetch of the single weight file into the app
/// cache on first use. The ~100 MB Small checkpoint is fetched only on the first depth job (it never
/// bloats a base download). Shared by every strict-control engine that admits `ControlKind::Depth`.
#[cfg(target_os = "macos")]
async fn ensure_depth_estimator_dir(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<PathBuf> {
    use crate::depth::{DEPTH_ANYTHING_V2_FILE, DEPTH_ANYTHING_V2_SMALL_REPO};

    if let Ok(p) = std::env::var("SCENEWORKS_DEPTH_ANYTHING_V2") {
        let p = PathBuf::from(p);
        if p.join(DEPTH_ANYTHING_V2_FILE).is_file() {
            return Ok(p);
        }
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, DEPTH_ANYTHING_V2_SMALL_REPO)
    {
        if snapshot.join(DEPTH_ANYTHING_V2_FILE).is_file() {
            return Ok(snapshot);
        }
    }
    let dir = settings.data_dir.join("cache").join("depth-anything-v2");
    let client = reqwest::Client::new();
    let context = crate::downloads::DownloadContext {
        api,
        client: &client,
        settings,
        job_id: &job.id,
        cancel_message: "Canceled while fetching depth-estimator weights.",
        fresh_download: false,
    };
    crate::downloads::ensure_hf_cached_file(
        &context,
        DEPTH_ANYTHING_V2_SMALL_REPO,
        "main",
        DEPTH_ANYTHING_V2_FILE,
        &dir.join(DEPTH_ANYTHING_V2_FILE),
    )
    .await?;
    Ok(dir)
}

/// Preprocess one control entry into the control [`Image`] the engine consumes.
///
/// Dispatch by [`ControlKind`]:
/// - **user-supplied passthrough** — if `user_control` is `Some`, it is used verbatim for ANY kind (the
///   caller already validated the kind), skipping preprocessing. This is the only path for `Other(_)`.
/// - **pose** — render the DWPose whole-body skeleton from `pose` via
///   [`crate::openpose_skeleton::draw_wholebody`] (body + hands 21×2 + face 68 when carried). Byte-identical
///   to the old per-engine pose preprocessing.
/// - **canny** — [`crate::canny::canny_control_image_default`] over `source` (a user-supplied source
///   image). Requires a source — canny has no synthetic input.
/// - **depth** — [`depth_control_image`] over `source`: the native-MLX Depth Anything V2 estimator
///   (sc-8242), using the provisioned `depth_weights_dir`. Like canny, requires a `source` (auto depth
///   has nothing to estimate from otherwise); a user-supplied depth map still short-circuits above.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn preprocess_control_entry(
    kind: &ControlKind,
    user_control: Option<&Image>,
    pose: Option<&PoseInput>,
    source: Option<&Image>,
    width: u32,
    height: u32,
    stickwidth: u32,
    depth_weights_dir: Option<&Path>,
) -> WorkerResult<Image> {
    if let Some(image) = user_control {
        return Ok(image.clone());
    }
    match kind {
        ControlKind::Pose => {
            let pose = pose.ok_or_else(|| {
                WorkerError::InvalidPayload("pose control requires a pose entry".to_owned())
            })?;
            let skeleton = crate::openpose_skeleton::draw_wholebody(
                width,
                height,
                &pose.keypoints,
                pose.hands.as_deref(),
                pose.face.as_deref(),
                stickwidth,
            );
            Ok(Image {
                width,
                height,
                pixels: skeleton.into_raw(),
            })
        }
        ControlKind::Canny => {
            let source = source.ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "canny control requires a source image (advanced.controlImage)".to_owned(),
                )
            })?;
            let rgb = image::RgbImage::from_raw(source.width, source.height, source.pixels.clone())
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("canny source buffer size mismatch".to_owned())
                })?;
            let edges = crate::canny::canny_control_image_default(&rgb);
            Ok(Image {
                width: edges.width(),
                height: edges.height(),
                pixels: edges.into_raw(),
            })
        }
        ControlKind::Depth => depth_control_image(source, depth_weights_dir),
        ControlKind::Other(name) => Err(WorkerError::InvalidPayload(format!(
            "{name} control requires a user-supplied control map (advanced.controlImage)"
        ))),
    }
}

/// Build the strict-control conditioning for one entry: the required control plus an optional shared
/// identity img2img-init `Reference` (flux2 / z-image opt-in tier). [`ControlKind::Depth`] uses the
/// dedicated `Conditioning::Depth` variant; every other kind uses `Conditioning::Control { kind, scale }`.
/// Byte-identical to the old per-engine construction for the pose tier.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn build_control_conditioning(
    control: Image,
    kind: ControlKind,
    scale: f32,
    identity_init: Option<&(Image, f32)>,
) -> Vec<Conditioning> {
    let mut conditioning = match kind {
        ControlKind::Depth => vec![Conditioning::Depth { image: control }],
        kind => vec![Conditioning::Control {
            image: control,
            kind,
            scale,
        }],
    };
    if let Some((image, strength)) = identity_init {
        conditioning.push(Conditioning::Reference {
            image: image.clone(),
            strength: Some(*strength),
        });
    }
    conditioning
}
