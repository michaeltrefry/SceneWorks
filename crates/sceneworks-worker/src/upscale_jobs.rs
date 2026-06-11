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
//! Engine scope: only `engine=real-esrgan` (the default) is served here. `aura-sr`
//! (a separate GAN upscaler, no clean ONNX export) stays a tracked Mac gap and is
//! refused by the routing oracle (`upscale_job_is_mlx_eligible`) so it keeps falling
//! to the Python worker.
//!
//! Tiling parity (matched to `upscalers.py:_run_tiled`):
//!  - tile grid `tile_slices(w,h,512)`; per-tile crop padded by `tile_pad=16`
//!    (clamped to image bounds); inner (unpadded) region copied back at factor scale.
//!  - input RGB f32 CHW in `[0,1]`; output clamp `[0,1]` → round → u8.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use image::RgbImage;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Value};

use crate::{
    fresh_asset_id, heartbeat, now_rfc3339, progress_payload, update_job, ApiClient, Settings,
    WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::ProjectStore;

const TILE_SIZE: usize = 512;
const TILE_PAD: usize = 16;

/// SceneWorks-owned HuggingFace repo hosting the pre-exported ONNX (reproducible from
/// `scripts/spikes/sc3489_export_reference.py`). Public; downloaded on first use,
/// parity with sc-3487's rtmlib weights. Overridable via the manifest `onnx` resource
/// or the env pin `SCENEWORKS_REALESRGAN_X{2,4}_ONNX`.
const ONNX_REPO: &str = "SceneWorks/real-esrgan-onnx";

fn onnx_file(factor: u8) -> String {
    format!("real_esrgan_x{factor}.onnx")
}

// ---------------------------------------------------------------------------
// pure tiling math (ported from upscalers.py; unit-tested without weights)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// `upscalers.py:tile_slices` — single tile if the image fits, else a row-major grid.
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

// ---------------------------------------------------------------------------
// onnxruntime upscaler (cached per-factor process-wide, like pose_jobs::DETECTOR)
// ---------------------------------------------------------------------------

struct Upscaler {
    session: Session,
    #[allow(dead_code)]
    device: &'static str,
}

static UPSCALERS: OnceLock<Mutex<HashMap<u8, Upscaler>>> = OnceLock::new();

fn ort_err<R>(e: ort::Error<R>) -> WorkerError {
    WorkerError::InvalidPayload(format!("onnxruntime: {e}"))
}

fn build_session(path: &Path, coreml: bool) -> WorkerResult<Session> {
    let mut b = Session::builder().map_err(ort_err)?;
    if coreml {
        b = b
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(ort_err)?;
    }
    b.commit_from_file(path).map_err(ort_err)
}

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
    fn upscale(&mut self, img: &RgbImage, factor: usize) -> WorkerResult<RgbImage> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (ow, oh) = (w * factor, h * factor);
        let mut output = vec![0u8; ow * oh * 3];
        for tl in tile_slices(w, h, TILE_SIZE) {
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
fn upscale_blocking(onnx_path: PathBuf, factor: u8, img: RgbImage) -> WorkerResult<RgbImage> {
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
    upscaler.upscale(&img, factor as usize)
}

// ---------------------------------------------------------------------------
// ONNX weight provisioning (download-on-first-use, mirrors Python resolution order)
// ---------------------------------------------------------------------------

/// Resolve the ONNX for `factor`. Order: explicit env pin
/// (`SCENEWORKS_REALESRGAN_X{factor}_ONNX`, then `SCENEWORKS_REALESRGAN_ONNX`), then the
/// app cache `<data_dir>/cache/upscale/`, then a manifest `onnx` resource if the job
/// carried one, else download from the default HF repo.
async fn ensure_onnx(
    settings: &Settings,
    http_client: &reqwest::Client,
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
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");

    tokio::fs::create_dir_all(&cache).await?;
    let bytes = http_client
        .get(&url)
        .send()
        .await?
        .error_for_status()
        .map_err(|e| {
            WorkerError::InvalidPayload(format!(
                "Real-ESRGAN ONNX download failed ({url}): {e}. Set SCENEWORKS_REALESRGAN_X{factor}_ONNX to a local export, or populate the {ONNX_REPO} HF repo."
            ))
        })?
        .bytes()
        .await?;
    let tmp = target.with_extension("onnx.tmp");
    tokio::fs::write(&tmp, &bytes).await?;
    tokio::fs::rename(&tmp, &target).await?;
    Ok(target)
}

/// Pull a `{repo,file}` ONNX resource out of a job's `modelManifestEntry` if present:
/// `resources.imageUpscalers.real-esrgan.x{factor}.onnx`.
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
    if !matches!(
        engine.as_str(),
        "real-esrgan" | "realesrgan" | "real_esrgan"
    ) {
        // aura-sr (and any future engine) stays a torch/Mac gap — the routing oracle
        // refuses it for the mlx worker, so this is a defensive guard only.
        return Err(WorkerError::InvalidPayload(format!(
            "Rust upscaler supports only engine=real-esrgan (got {engine}); aura-sr stays on the Python path (sc-3489 follow-up)."
        )));
    }
    let engine_id = "real-esrgan"; // matches image_adapters RealESRGANUpscaler.id
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
    let onnx_path = ensure_onnx(settings, http_client, factor, &manifest_entry).await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.45,
            &format!("Upscaling {factor}x with {engine_id}."),
            None,
            None,
            None,
        ),
    )
    .await?;
    let upscaled =
        tokio::task::spawn_blocking(move || upscale_blocking(onnx_path, factor, source_image))
            .await
            .map_err(|e| WorkerError::InvalidPayload(format!("upscale task: {e}")))??;
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
    let upscale_settings = json!({
        "enabled": true,
        "engine": engine_id,
        "factor": factor,
        "sourceWidth": src_w,
        "sourceHeight": src_h,
        "width": out_w,
        "height": out_h,
    });
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
        "seed": 0,
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
