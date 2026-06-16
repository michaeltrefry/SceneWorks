//! Real-ESRGAN image upscaling on the Rust worker (epic 3482, sc-3489).
//!
//! Ports the Python `scene_worker/upscalers.py` + `image_adapters.run_image_upscale`
//! `image_upscale` job to Rust so the Image Editor upscale tool (epic 2427) keeps
//! working on a Python-free Mac. The upscaler is Real-ESRGAN (RRDBNet x2/x4) run via
//! `ort` (onnxruntime) with the CoreML execution provider — the same `ort` stack +
//! bundled `libonnxruntime.dylib` sc-3487 ships for DWPose. The path-selection spike
//! (`docs/sc-3489/spike-findings.md`) validated pixel parity vs the shipped torch
//! upscaler (CoreML PSNR ~72–74 dB, visually identical) at ~9× the CPU-EP speed.
//!
//! macOS-only: the CoreML EP only means anything on Apple Silicon, and the Python
//! torch Real-ESRGAN / AuraSR path stays the Windows/Linux backend. The tiling math
//! (`tile_slices`, crop/place) is pure and unit-tested without the onnx weights; only
//! the onnxruntime inference is gated.
//!
//! Engine scope: `engine=real-esrgan` (the default, the `ort`/CoreML path above) and
//! `engine=seedvr2` (epic 4811 / sc-4815 — the native-MLX one-step diffusion super-resolution
//! upscaler) are served here. SeedVR2 runs in-process via the `mlx-gen-seedvr2` registry generator,
//! driven through the shared `with_cached_generator` seam (single-resident engine cache — it evicts
//! any cached image-gen engine, bounding peak memory, which matters because the SeedVR2 image path
//! has no spatial tiling yet). It takes a target resolution (factor → `round_to_16(src × factor)`)
//! and an optional `--softness` pre-blur; `mac_only` (the Windows/Linux SeedVR2 backend is the
//! separate Candle port, sc-5157). `aura-sr` (a 617M-param torch-only GigaGAN) was dropped on Mac
//! after the sc-3668 port-or-drop spike — it is refused by the routing oracle
//! (`upscale_job_is_mlx_eligible`) and hidden in the Mac UI, so it only runs on the Python worker on
//! Windows/Linux.
//!
//! Tiling parity (matched to `upscalers.py:_run_tiled`):
//!  - tile grid `tile_slices(w,h,512)`; per-tile crop padded by `tile_pad=16`
//!    (clamped to image bounds); inner (unpadded) region copied back at factor scale.
//!  - input RGB f32 CHW in `[0,1]`; output clamp `[0,1]` → round → u8.

// `HashMap`/`Mutex`/`OnceLock` back the Real-ESRGAN per-factor session cache (Mac-only `ort` path).
#[cfg(target_os = "macos")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};

use crate::downloads::{ensure_hf_cached_file, DownloadContext};
use crate::generator_cache::with_cached_generator;
use gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image as GenImage, LoadSpec,
    WeightsSource,
};
use image::RgbImage;
// Real-ESRGAN runs via `ort` + the CoreML execution provider — Mac-only (the candle Windows lane
// serves SeedVR2, not Real-ESRGAN; Real-ESRGAN/AuraSR stay on the Python torch worker off-Mac).
#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(target_os = "macos")]
use ort::session::Session;
#[cfg(target_os = "macos")]
use ort::value::Tensor;
use serde_json::{json, Value};

use crate::{
    cancel_requested_peek, fresh_asset_id, heartbeat, mark_job_canceled, now_rfc3339,
    progress_payload, progress_report_interval, task_join_error, update_job, ApiClient, Settings,
    WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::ProjectStore;

// Real-ESRGAN tiling (Mac-only `ort`/CoreML path).
#[cfg(target_os = "macos")]
const TILE_SIZE: usize = 512;
#[cfg(target_os = "macos")]
const TILE_PAD: usize = 16;
const MAX_UPSCALE_TARGET_DIMENSION: u32 = 8192;
const MAX_UPSCALE_TARGET_PIXELS: u64 =
    MAX_UPSCALE_TARGET_DIMENSION as u64 * MAX_UPSCALE_TARGET_DIMENSION as u64;
const CANCEL_MESSAGE: &str = "Image upscale canceled by user.";

/// SceneWorks-owned HuggingFace repo hosting the pre-exported ONNX (reproducible from
/// `scripts/spikes/sc3489_export_reference.py`). Public; downloaded on first use,
/// parity with sc-3487's rtmlib weights. Overridable via the manifest `onnx` resource
/// or the env pin `SCENEWORKS_REALESRGAN_X{2,4}_ONNX`.
#[cfg(target_os = "macos")]
const ONNX_REPO: &str = "SceneWorks/real-esrgan-onnx";

#[cfg(target_os = "macos")]
fn onnx_file(factor: u8) -> String {
    format!("real_esrgan_x{factor}.onnx")
}

// ---------------------------------------------------------------------------
// pure tiling math (ported from upscalers.py; unit-tested without weights) — Real-ESRGAN, Mac-only
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// `upscalers.py:tile_slices` — single tile if the image fits, else a row-major grid.
#[cfg(target_os = "macos")]
fn tile_slices(w: usize, h: usize, tile: usize) -> Vec<Tile> {
    if tile == 0 || tile >= w.max(h) {
        return vec![Tile {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        }];
    }
    let mut out = Vec::new();
    let mut y0 = 0;
    while y0 < h {
        let mut x0 = 0;
        while x0 < w {
            out.push(Tile {
                x0,
                y0,
                x1: (x0 + tile).min(w),
                y1: (y0 + tile).min(h),
            });
            x0 += tile;
        }
        y0 += tile;
    }
    out
}

/// CHW f32 `[0,1]` for the crop region `[x0,x1) × [y0,y1)` of an RGB image.
#[cfg(target_os = "macos")]
fn crop_to_chw(
    img: &RgbImage,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> (Vec<f32>, usize, usize) {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut data = vec![0.0f32; 3 * ch * cw];
    for c in 0..3 {
        let plane = c * ch * cw;
        for yy in 0..ch {
            for xx in 0..cw {
                let p = img.get_pixel((x0 + xx) as u32, (y0 + yy) as u32);
                data[plane + yy * cw + xx] = p[c] as f32 / 255.0;
            }
        }
    }
    (data, cw, ch)
}

fn validate_upscale_target_dimensions(width: u32, height: u32) -> WorkerResult<()> {
    let pixels = u64::from(width) * u64::from(height);
    if width > MAX_UPSCALE_TARGET_DIMENSION
        || height > MAX_UPSCALE_TARGET_DIMENSION
        || pixels > MAX_UPSCALE_TARGET_PIXELS
    {
        return Err(WorkerError::InvalidPayload(format!(
            "Upscale target {width}x{height} exceeds the {MAX_UPSCALE_TARGET_DIMENSION}px side / {MAX_UPSCALE_TARGET_PIXELS} pixel limit."
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// onnxruntime upscaler (cached per-factor process-wide, like pose_jobs::DETECTOR) — Real-ESRGAN,
// Mac-only (`ort`/CoreML). The candle Windows lane serves SeedVR2 (below), not Real-ESRGAN.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
struct Upscaler {
    session: Session,
    #[allow(dead_code)]
    device: &'static str,
}

#[cfg(target_os = "macos")]
static UPSCALERS: OnceLock<Mutex<HashMap<u8, Upscaler>>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn ort_err<R>(e: ort::Error<R>) -> WorkerError {
    WorkerError::Engine(format!("onnxruntime: {e}"))
}

#[cfg(target_os = "macos")]
fn build_session(path: &Path, coreml: bool) -> WorkerResult<Session> {
    let mut b = Session::builder().map_err(ort_err)?;
    if coreml {
        b = b
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(ort_err)?;
    }
    b.commit_from_file(path).map_err(ort_err)
}

#[cfg(target_os = "macos")]
impl Upscaler {
    /// Load the session, preferring CoreML and falling back to CPU if the provider
    /// can't initialise (mirrors `pose_jobs::Detector::load`).
    fn load(path: &Path) -> WorkerResult<Self> {
        match build_session(path, true) {
            Ok(session) => Ok(Self {
                session,
                device: "coreml",
            }),
            Err(_) => Ok(Self {
                session: build_session(path, false)?,
                device: "cpu",
            }),
        }
    }

    /// Tiled x`factor` upscale of one RGB image → an upscaled RGB image. Tiling +
    /// crop/place is a verbatim port of `upscalers.py:_run_tiled`.
    fn upscale(
        &mut self,
        img: &RgbImage,
        factor: usize,
        cancel: &CancelFlag,
    ) -> WorkerResult<RgbImage> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let ow = w
            .checked_mul(factor)
            .ok_or_else(|| WorkerError::InvalidPayload("upscale width overflow".to_owned()))?;
        let oh = h
            .checked_mul(factor)
            .ok_or_else(|| WorkerError::InvalidPayload("upscale height overflow".to_owned()))?;
        let ow_u32 = u32::try_from(ow)
            .map_err(|_| WorkerError::InvalidPayload("upscale width overflow".to_owned()))?;
        let oh_u32 = u32::try_from(oh)
            .map_err(|_| WorkerError::InvalidPayload("upscale height overflow".to_owned()))?;
        validate_upscale_target_dimensions(ow_u32, oh_u32)?;
        let mut output = vec![0u8; ow * oh * 3];
        for tl in tile_slices(w, h, TILE_SIZE) {
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            let cx0 = tl.x0.saturating_sub(TILE_PAD);
            let cy0 = tl.y0.saturating_sub(TILE_PAD);
            let cx1 = (tl.x1 + TILE_PAD).min(w);
            let cy1 = (tl.y1 + TILE_PAD).min(h);
            let (data, cw, ch) = crop_to_chw(img, cx0, cy0, cx1, cy1);

            let tensor =
                Tensor::from_array((vec![1i64, 3, ch as i64, cw as i64], data)).map_err(ort_err)?;
            let outputs = self.session.run(ort::inputs![tensor]).map_err(ort_err)?;
            let (oshape, odata) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
            let (och, ocw) = (oshape[2] as usize, oshape[3] as usize);

            // inner (unpadded) region within the upscaled crop → destination
            let ix0 = (tl.x0 - cx0) * factor;
            let iy0 = (tl.y0 - cy0) * factor;
            let iw = (tl.x1 - tl.x0) * factor;
            let ih = (tl.y1 - tl.y0) * factor;
            let (dst_x0, dst_y0) = (tl.x0 * factor, tl.y0 * factor);
            for yy in 0..ih {
                for xx in 0..iw {
                    let (sy, sx) = (iy0 + yy, ix0 + xx);
                    let di = ((dst_y0 + yy) * ow + dst_x0 + xx) * 3;
                    for c in 0..3 {
                        let v = odata[c * och * ocw + sy * ocw + sx];
                        output[di + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
        RgbImage::from_raw(ow as u32, oh as u32, output)
            .ok_or_else(|| WorkerError::InvalidPayload("upscale buffer size mismatch".to_owned()))
    }
}

/// Blocking upscale: load+cache the factor's session (amortising the CoreML graph
/// compile across a batch), run it. All `ort` objects live inside this closure and
/// never cross an await (mirrors `pose_jobs::detect_batch`).
#[cfg(target_os = "macos")]
fn upscale_blocking(
    onnx_path: PathBuf,
    factor: u8,
    img: RgbImage,
    cancel: CancelFlag,
) -> WorkerResult<RgbImage> {
    use std::collections::hash_map::Entry;
    let cell = UPSCALERS.get_or_init(|| Mutex::new(HashMap::new()));
    // Recover from a poisoned lock rather than panicking every subsequent job: if a
    // prior upscale panicked mid-run holding this lock, take the inner guard and
    // clear the possibly-corrupt cached sessions so the match below reloads a fresh
    // one for this factor (sc-4277 / F-MLXW-13).
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        guard.clear();
        guard
    });
    let upscaler = match guard.entry(factor) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => e.insert(Upscaler::load(&onnx_path)?),
    };
    upscaler.upscale(&img, factor as usize, &cancel)
}

// ---------------------------------------------------------------------------
// ONNX weight provisioning (download-on-first-use, mirrors Python resolution order)
// ---------------------------------------------------------------------------

/// Resolve the ONNX for `factor`. Order: explicit env pin
/// (`SCENEWORKS_REALESRGAN_X{factor}_ONNX`, then `SCENEWORKS_REALESRGAN_ONNX`), then the
/// app cache `<data_dir>/cache/upscale/`, then a manifest `onnx` resource if the job
/// carried one, else download from the default HF repo.
#[cfg(target_os = "macos")]
async fn ensure_onnx(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    factor: u8,
    manifest_entry: &Value,
) -> WorkerResult<PathBuf> {
    for key in [
        format!("SCENEWORKS_REALESRGAN_X{factor}_ONNX"),
        "SCENEWORKS_REALESRGAN_ONNX".to_owned(),
    ] {
        if let Ok(pinned) = std::env::var(&key) {
            let path = PathBuf::from(pinned);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    let cache = settings.data_dir.join("cache").join("upscale");
    let target = cache.join(onnx_file(factor));
    if target.exists() {
        return Ok(target);
    }

    // manifest resource: resources.imageUpscalers.real-esrgan.x{factor}.onnx -> {repo,file}
    let (repo, file) = manifest_onnx_resource(manifest_entry, factor)
        .unwrap_or_else(|| (ONNX_REPO.to_owned(), onnx_file(factor)));

    tokio::fs::create_dir_all(&cache).await?;
    let context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Image upscale canceled while fetching Real-ESRGAN weights.",
        fresh_download: false,
    };
    ensure_hf_cached_file(&context, &repo, "main", &file, &target)
        .await
        .map_err(|error| match error {
            WorkerError::InvalidPayload(detail) => WorkerError::InvalidPayload(format!(
                "Real-ESRGAN ONNX download failed ({repo}/{file}): {detail}. Set SCENEWORKS_REALESRGAN_X{factor}_ONNX to a local export, or populate the {ONNX_REPO} HF repo."
            )),
            other => WorkerError::Engine(format!(
                "Real-ESRGAN ONNX download failed ({repo}/{file}): {other}. Set SCENEWORKS_REALESRGAN_X{factor}_ONNX to a local export, or populate the {ONNX_REPO} HF repo."
            )),
        })
}

/// Pull a `{repo,file}` ONNX resource out of a job's `modelManifestEntry` if present:
/// `resources.imageUpscalers.real-esrgan.x{factor}.onnx`.
#[cfg(target_os = "macos")]
fn manifest_onnx_resource(manifest_entry: &Value, factor: u8) -> Option<(String, String)> {
    let onnx = manifest_entry
        .get("resources")?
        .get("imageUpscalers")
        .or_else(|| manifest_entry.get("resources")?.get("upscalers"))?
        .get("real-esrgan")?
        .get(format!("x{factor}"))?
        .get("onnx")?;
    let repo = onnx.get("repo")?.as_str()?.to_owned();
    let file = onnx
        .get("file")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| onnx_file(factor));
    Some((repo, file))
}

// ---------------------------------------------------------------------------
// SeedVR2 native-MLX upscaler (epic 4811 / sc-4815)
// ---------------------------------------------------------------------------

/// Upstream HuggingFace repo holding the raw SeedVR2 ComfyUI checkpoint. The `mlx-gen-seedvr2`
/// registry loads it directly (converts to MLX layout in-memory, no Python). Public; downloaded on
/// first use. Overridable via the manifest `seedvr2` resource or `SCENEWORKS_SEEDVR2_CHECKPOINT`.
const SEEDVR2_REPO: &str = "numz/SeedVR2_comfyUI";
/// The exact filenames `Seedvr2Pipeline::load` expects in the checkpoint dir (3B fp16 DiT + VAE).
const SEEDVR2_DIT_FILE: &str = "seedvr2_ema_3b_fp16.safetensors";
const SEEDVR2_VAE_FILE: &str = "ema_vae_fp16.safetensors";

/// Round up to the nearest multiple of 16 (the SeedVR2 VAE /8 · patch /2 constraint the registry
/// validates), floored at 16. `factor × src` is usually already a multiple of 16; odd sizes round.
fn round_to_16(v: u32) -> u32 {
    (v.div_ceil(16)).max(1) * 16
}

/// Pull a `{repo, ditFile, vaeFile}` override out of a job's `modelManifestEntry` if present:
/// `resources.imageUpscalers.seedvr2` (or the legacy `resources.upscalers.seedvr2`).
fn manifest_seedvr2_resource(manifest_entry: &Value) -> Option<(String, String, String)> {
    let node = manifest_entry
        .get("resources")?
        .get("imageUpscalers")
        .or_else(|| manifest_entry.get("resources")?.get("upscalers"))?
        .get("seedvr2")?;
    let repo = node.get("repo")?.as_str()?.to_owned();
    let dit = node
        .get("ditFile")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| SEEDVR2_DIT_FILE.to_owned());
    let vae = node
        .get("vaeFile")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| SEEDVR2_VAE_FILE.to_owned());
    Some((repo, dit, vae))
}

/// Resolve the raw SeedVR2 checkpoint dir (containing the canonical DiT + VAE filenames the engine
/// loads). Order mirrors `ensure_onnx`: env pin (`SCENEWORKS_SEEDVR2_CHECKPOINT`, a dir holding both
/// files) → the app cache `<data_dir>/cache/upscale/seedvr2/` → download from the manifest/default HF
/// repo on first use. The source filenames may be overridden by the manifest, but they are always
/// stored under the canonical names so `Seedvr2Pipeline::load` finds them.
async fn ensure_seedvr2_checkpoint(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
    manifest_entry: &Value,
) -> WorkerResult<PathBuf> {
    if let Ok(pinned) = std::env::var("SCENEWORKS_SEEDVR2_CHECKPOINT") {
        let dir = PathBuf::from(pinned);
        if dir.join(SEEDVR2_DIT_FILE).exists() && dir.join(SEEDVR2_VAE_FILE).exists() {
            return Ok(dir);
        }
    }

    let dir = settings
        .data_dir
        .join("cache")
        .join("upscale")
        .join("seedvr2");
    tokio::fs::create_dir_all(&dir).await?;

    let (repo, dit_src, vae_src) = manifest_seedvr2_resource(manifest_entry).unwrap_or_else(|| {
        (
            SEEDVR2_REPO.to_owned(),
            SEEDVR2_DIT_FILE.to_owned(),
            SEEDVR2_VAE_FILE.to_owned(),
        )
    });

    let context = DownloadContext {
        api,
        client: http_client,
        settings,
        job_id: &job.id,
        cancel_message: "Image upscale canceled while fetching SeedVR2 weights.",
        fresh_download: false,
    };
    for (src_file, canonical) in [
        (dit_src.as_str(), SEEDVR2_DIT_FILE),
        (vae_src.as_str(), SEEDVR2_VAE_FILE),
    ] {
        let target = dir.join(canonical);
        if target.exists() {
            continue;
        }
        ensure_hf_cached_file(&context, &repo, "main", src_file, &target)
            .await
            .map_err(|error| {
                let detail = match &error {
                    WorkerError::InvalidPayload(d) => d.clone(),
                    other => other.to_string(),
                };
                WorkerError::Engine(format!(
                    "SeedVR2 weight download failed ({repo}/{src_file}): {detail}. Set \
                     SCENEWORKS_SEEDVR2_CHECKPOINT to a local checkpoint dir, or populate {SEEDVR2_REPO}."
                ))
            })?;
    }
    Ok(dir)
}

/// Run a SeedVR2 image upscale: the LR `source` (native resolution) → a `round_to_16(factor×)`
/// super-resolved RGB image. Goes through the shared single-resident generator cache
/// (`with_cached_generator`) so loading SeedVR2 evicts any cached image-gen engine (the engine's
/// image path has no spatial tiling — keeping one resident model bounds peak memory). `softness`
/// (0..1) is the optional `--softness` pre-blur; `seed` makes the generative result reproducible.
async fn run_seedvr2_upscale(
    dir: PathBuf,
    source: RgbImage,
    factor: u8,
    softness: f32,
    seed: u64,
    cancel: CancelFlag,
) -> WorkerResult<RgbImage> {
    let (src_w, src_h) = (source.width(), source.height());
    let target_w = round_to_16(src_w.saturating_mul(u32::from(factor)));
    let target_h = round_to_16(src_h.saturating_mul(u32::from(factor)));
    validate_upscale_target_dimensions(target_w, target_h)?;
    let image = GenImage {
        width: src_w,
        height: src_h,
        pixels: source.as_raw().clone(),
    };

    with_cached_generator(
        "seedvr2",
        LoadSpec::new(WeightsSource::Dir(dir)),
        "SeedVR2 engine load",
        move |generator| {
            let request = GenerationRequest {
                width: target_w,
                height: target_h,
                count: 1,
                seed: Some(seed),
                softness: Some(softness),
                conditioning: vec![Conditioning::Reference {
                    image,
                    strength: None,
                }],
                cancel: cancel.clone(),
                ..Default::default()
            };
            let output = generator
                .generate(&request, &mut |_progress| {})
                .map_err(|error| match error {
                    gen_core::Error::Canceled => WorkerError::Canceled(error.to_string()),
                    other => WorkerError::Engine(format!("SeedVR2 upscale failed: {other}")),
                })?;
            let image = match output {
                GenerationOutput::Images(mut images) if !images.is_empty() => images.remove(0),
                GenerationOutput::Images(_) => {
                    return Err(WorkerError::Engine("SeedVR2 produced no image".to_owned()))
                }
                other => {
                    return Err(WorkerError::Engine(format!(
                        "SeedVR2 returned non-image output: {other:?}"
                    )))
                }
            };
            RgbImage::from_raw(image.width, image.height, image.pixels)
                .ok_or_else(|| WorkerError::Engine("SeedVR2 image buffer size mismatch".to_owned()))
        },
    )
    .await
}

async fn run_upscale_with_heartbeat<R>(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    cancel: CancelFlag,
    mut task: tokio::task::JoinHandle<WorkerResult<R>>,
) -> WorkerResult<R>
where
    R: Send + 'static,
{
    let mut canceled = false;
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut task => {
                let value = result.map_err(|error| task_join_error("upscale task", error))??;
                if canceled {
                    mark_job_canceled(api, job_id, CANCEL_MESSAGE).await?;
                    return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
                }
                return Ok(value);
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(job_id)).await?;
                if !canceled && cancel_requested_peek(api, job_id).await {
                    cancel.cancel();
                    canceled = true;
                    update_job(
                        api,
                        job_id,
                        progress_payload(
                            JobStatus::Running,
                            ProgressStage::Running,
                            0.45,
                            "Canceling image upscale.",
                            None,
                            None,
                            None,
                        ),
                    )
                    .await?;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// source resolution (mirrors image_adapters.find_asset_media_path / _source_display_name)
// ---------------------------------------------------------------------------

/// Resolve a `sourceAssetId` to its on-disk media path (native resolution) + the
/// asset's `displayName`, via the project sidecar — mirrors `find_asset_media_path`.
fn resolve_source(
    store: &ProjectStore,
    project_id: &str,
    asset_id: &str,
    project_path: &Path,
) -> Option<(PathBuf, Option<String>)> {
    let asset = store.get_asset(project_id, asset_id).ok()?;
    let rel = asset.get("file")?.get("path")?.as_str()?;
    let mut path = project_path.to_path_buf();
    for component in Path::new(rel).components() {
        if let std::path::Component::Normal(value) = component {
            path.push(value);
        } else {
            return None;
        }
    }
    if !path.exists() {
        return None;
    }
    let display = asset
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Some((path, display))
}

// ---------------------------------------------------------------------------
// job handler
// ---------------------------------------------------------------------------

pub(crate) async fn run_image_upscale_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.12,
            "Loading source image.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let payload = &job.payload;
    let source_asset_id = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Upscale jobs require a source image asset.".to_owned())
        })?
        .to_owned();
    let factor: u8 = match payload.get("factor").and_then(Value::as_u64).unwrap_or(2) {
        4 => 4,
        _ => 2,
    };
    let engine = payload
        .get("engine")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("real-esrgan")
        .to_lowercase();
    // Canonical engine id. The mlx worker serves Real-ESRGAN (`ort`/CoreML) and SeedVR2 (native
    // MLX); aura-sr was dropped on Mac (sc-3668). The routing oracle (`upscale_job_is_mlx_eligible`)
    // already refuses anything else for the mlx worker, so this match is a defensive guard.
    let engine_id = match engine.as_str() {
        "real-esrgan" | "realesrgan" | "real_esrgan" => "real-esrgan",
        "seedvr2" => "seedvr2",
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "Rust upscaler supports engine=real-esrgan or engine=seedvr2 (got {other}); aura-sr is dropped on Mac, available on Windows/Linux (sc-3668)."
            )));
        }
    };
    // SeedVR2-only knobs (ignored by Real-ESRGAN): the `--softness` pre-blur (0..1) and a seed for
    // reproducible generative output. Both default to 0 and are read leniently from the payload.
    let softness = payload
        .get("softness")
        .and_then(Value::as_f64)
        .map(|v| v.clamp(0.0, 1.0) as f32)
        .unwrap_or(0.0);
    let seed = payload.get("seed").and_then(Value::as_u64).unwrap_or(0);
    let manifest_entry = payload
        .get("modelManifestEntry")
        .cloned()
        .unwrap_or(Value::Null);

    // resolve the source asset against its originating project
    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref())
        .ok_or_else(|| WorkerError::InvalidPayload("Upscale jobs require a projectId.".to_owned()))?
        .to_owned();
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .get_project(&project_id)
        .map_err(|e| WorkerError::InvalidPayload(format!("project not found: {e}")))?;
    let project_path = PathBuf::from(project.path);
    let (source_path, source_display) =
        resolve_source(&store, &project_id, &source_asset_id, &project_path).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Source image asset not found or missing: {source_asset_id}."
            ))
        })?;

    let source_image = image::open(&source_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("Source image could not be loaded: {e}")))?
        .to_rgb8();
    let (src_w, src_h) = (source_image.width(), source_image.height());

    let upscaled = if engine_id == "seedvr2" {
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Downloading,
                0.25,
                "Loading SeedVR2 weights.",
                None,
                None,
                None,
            ),
        )
        .await?;
        let dir =
            ensure_seedvr2_checkpoint(api, settings, http_client, job, &manifest_entry).await?;

        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Running,
                ProgressStage::Running,
                0.45,
                &format!("Upscaling {factor}x with SeedVR2."),
                None,
                None,
                None,
            ),
        )
        .await?;
        let cancel = CancelFlag::new();
        let seed_source = source_image.clone();
        run_upscale_with_heartbeat(
            api,
            settings,
            &job.id,
            cancel.clone(),
            tokio::spawn(async move {
                run_seedvr2_upscale(dir, seed_source, factor, softness, seed, cancel).await
            }),
        )
        .await?
    } else {
        // Real-ESRGAN runs via `ort`/CoreML — Mac-only. Off-Mac the candle worker serves only
        // `engine=seedvr2` (the routing oracle refuses Real-ESRGAN here, sending it to the Python
        // torch worker), so this branch is unreachable on the candle lane; keep it a clear error so
        // the function compiles and a misroute fails loudly rather than silently.
        #[cfg(target_os = "macos")]
        {
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Downloading,
                    0.25,
                    "Loading Real-ESRGAN weights.",
                    None,
                    None,
                    None,
                ),
            )
            .await?;
            let onnx_path =
                ensure_onnx(api, settings, http_client, job, factor, &manifest_entry).await?;

            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Running,
                    0.45,
                    &format!("Upscaling {factor}x with Real-ESRGAN."),
                    None,
                    None,
                    None,
                ),
            )
            .await?;
            let cancel = CancelFlag::new();
            run_upscale_with_heartbeat(
                api,
                settings,
                &job.id,
                cancel.clone(),
                tokio::task::spawn_blocking(move || {
                    upscale_blocking(onnx_path, factor, source_image, cancel)
                }),
            )
            .await?
        }
        #[cfg(not(target_os = "macos"))]
        {
            return Err(WorkerError::InvalidPayload(format!(
                "engine={engine_id} is not served by the candle worker (it serves engine=seedvr2); \
                 Real-ESRGAN / AuraSR run on the Python torch worker off-Mac"
            )));
        }
    };
    let (out_w, out_h) = (upscaled.width(), upscaled.height());

    // write exactly one child asset with lineage back to the source (mirrors
    // image_adapters.run_image_upscale).
    let created_at = now_rfc3339();
    let generation_set_id = format!("genset_{}", uuid::Uuid::new_v4().simple());
    let asset_id = fresh_asset_id();
    let date = &created_at[..10];
    // filename suffix = asset_id[6:14] (the 8 hex chars after "asset_")
    let suffix: String = asset_id.chars().skip(6).take(8).collect();
    let filename = format!("{date}_upscaled_x{factor}_{suffix}.png");
    let media_rel = format!("assets/images/{generation_set_id}/{filename}");
    let media_path = project_path.join(&media_rel);
    if let Some(parent) = media_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let tmp_path = media_path.with_extension("tmp.png");
    upscaled
        .save_with_format(&tmp_path, image::ImageFormat::Png)
        .map_err(|e| WorkerError::Io(std::io::Error::other(e)))?;
    tokio::fs::rename(&tmp_path, &media_path)
        .await
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp_path);
        })?;

    let source_name = payload
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or(source_display)
        .unwrap_or_else(|| "Image".to_owned());
    let mut upscale_settings = json!({
        "enabled": true,
        "engine": engine_id,
        "factor": factor,
        "sourceWidth": src_w,
        "sourceHeight": src_h,
        "width": out_w,
        "height": out_h,
    });
    if engine_id == "seedvr2" {
        // SeedVR2 is a generative one-step upscaler: record the detail/softness knob and the seed so
        // the result is reproducible + the UI can surface what produced it.
        upscale_settings["softness"] = json!(softness);
        upscale_settings["seed"] = json!(seed);
    }
    let fact = json!({
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "type": "image",
        "width": out_w,
        "height": out_h,
        "normalizedWidth": out_w,
        "normalizedHeight": out_h,
        "count": 1,
        "seed": seed,
        "displayName": format!("{source_name} ({factor}x upscaled)"),
        "createdAt": created_at.clone(),
        "mode": "image_upscale",
        "model": engine_id,
        "adapter": engine_id,
        "prompt": "",
        "negativePrompt": "",
        "loras": [],
        "stylePreset": "",
        "sourceAssetId": source_asset_id,
        "rawAdapterSettings": { "upscale": upscale_settings },
        "parents": [source_asset_id],
        "extra": {
            "isUpscaled": true,
            "upscaledFromAssetId": source_asset_id,
            "factor": factor,
            "engine": engine_id,
        },
    });
    let generation_set = json!({
        "id": generation_set_id,
        "mode": "image_upscale",
        "model": engine_id,
        "prompt": "",
        "negativePrompt": "",
        "count": 1,
        "createdAt": created_at,
    });
    let mut result = JsonObject::new();
    result.insert(
        "generationSetId".to_owned(),
        Value::String(generation_set_id),
    );
    result.insert("expectedCount".to_owned(), json!(1));
    result.insert("adapter".to_owned(), Value::String(engine_id.to_owned()));
    result.insert("model".to_owned(), Value::String(engine_id.to_owned()));
    result.insert("generationSet".to_owned(), generation_set);
    result.insert("assetWrites".to_owned(), Value::Array(vec![fact]));

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Image upscale complete.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests;
