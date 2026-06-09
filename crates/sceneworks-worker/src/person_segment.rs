//! Native-MLX SAM2 person segmentation on the Rust worker (epic 3704, sc-3709;
//! Slice 3 of sc-3488 / epic 3482).
//!
//! Ports the Python `scene_worker/person_adapters.py` `segment_track` /
//! `_Sam2Segmenter` to Rust so the Replace-Person mask-generation step runs on a
//! Python-free Mac. Each detected track frame is segmented with the native-MLX
//! `mlx-gen-sam2` box-prompt segmenter (Hiera encoder + FPN neck + prompt encoder +
//! two-way mask decoder, sc-3705/3706, GO at sc-3708) and the binary `L` mask is
//! written under `person-tracks/{track_id}/masks/`.
//!
//! macOS-only, like `person_jobs`: `mlx-gen-sam2` builds Apple MLX from source and is
//! meaningless off Apple Silicon. The Python SAM2 path stays the Windows/Linux backend.
//!
//! The orchestration (which frames to segment, the maskState rollup, the sidecar
//! write) lives in `media_jobs::assemble_real_person_track`; this module owns only the
//! weight resolution/download and the single-frame segment call.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use image::GrayImage;

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ModelSize, Sam2Segmenter};

use crate::{Settings, WorkerError, WorkerResult};

/// The production SAM2 size (matches the spike + the `SceneWorks/sam2-mlx` upload).
const SEG_FILE: &str = "sam2.1_hiera_large.safetensors";

/// Download-on-first-use source: the converted MLX SAM2 weights owned by SceneWorks
/// (sc-3707, uploaded from the official Meta `.pt` via `tools/convert_sam2_to_mlx.py`;
/// bit-identical to the avbiswas reference). Public repo, so no credentials are needed —
/// same shape as the YOLO11 detector weights (`person_jobs::DET_URL`).
const SEG_URL: &str =
    "https://huggingface.co/SceneWorks/sam2-mlx/resolve/main/sam2.1_hiera_large.safetensors";

/// The SAM2 segmenter is loaded once and cached process-wide (like the YOLO detector
/// and Python's lazy model load). Holds `None` until the first frame is segmented.
static SEGMENTER: OnceLock<Mutex<Option<Sam2Segmenter>>> = OnceLock::new();

/// Resolve already-present SAM2 weights: an explicit env pin
/// (`SCENEWORKS_SAM2_WEIGHTS`), then the app cache `<data_dir>/cache/person-segment/`,
/// then the model dir `<data_dir>/models/person-segment/`. Returns `None` when nothing
/// is staged (then `ensure_segmenter_weights` downloads it).
pub(crate) fn resolve_segmenter_weights(settings: &Settings) -> Option<PathBuf> {
    if let Ok(pinned) = std::env::var("SCENEWORKS_SAM2_WEIGHTS") {
        let path = PathBuf::from(pinned);
        if path.exists() {
            return Some(path);
        }
    }
    for sub in ["cache/person-segment", "models/person-segment"] {
        let candidate = settings.data_dir.join(sub).join(SEG_FILE);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve the SAM2 weights, downloading them from HuggingFace on first use (into the
/// app cache). Mirrors `person_jobs::ensure_detector_weights` — atomic `.tmp` + rename
/// so a partial download is never mistaken for a complete one.
pub(crate) async fn ensure_segmenter_weights(
    settings: &Settings,
    http_client: &reqwest::Client,
) -> WorkerResult<PathBuf> {
    if let Some(path) = resolve_segmenter_weights(settings) {
        return Ok(path);
    }
    let cache = settings.data_dir.join("cache").join("person-segment");
    tokio::fs::create_dir_all(&cache).await?;
    let target = cache.join(SEG_FILE);
    let bytes = http_client
        .get(SEG_URL)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let tmp = target.with_extension("safetensors.tmp");
    tokio::fs::write(&tmp, &bytes).await?;
    tokio::fs::rename(&tmp, &target).await?;
    Ok(target)
}

/// Scale a normalized `(x, y, width, height)` box to a pixel-space `[x1, y1, x2, y2]`
/// corner box on a `width`×`height` frame, clamped to the frame. SAM2 takes the box as a
/// corner-point prompt in original pixel space.
pub(crate) fn box_norm_to_pixels(
    box_norm: (f64, f64, f64, f64),
    width: u32,
    height: u32,
) -> [f32; 4] {
    let (bx, by, bw, bh) = box_norm;
    let (w, h) = (width as f32, height as f32);
    [
        (bx as f32 * w).clamp(0.0, w),
        (by as f32 * h).clamp(0.0, h),
        ((bx + bw) as f32 * w).clamp(0.0, w),
        ((by + bh) as f32 * h).clamp(0.0, h),
    ]
}

/// Roll the per-frame segmentation outcome into the sidecar `maskState` (Python
/// `segment_track`): `generated` masks out of `detected_total` detected target frames →
/// `degraded` (none written → box-mask fallback at replacement time), `active` (every
/// detected frame segmented), or `generated` (a partial subset).
pub(crate) fn rollup_mask_state(generated: usize, detected_total: usize) -> &'static str {
    if generated == 0 {
        "degraded"
    } else if generated >= detected_total {
        "active"
    } else {
        "generated"
    }
}

/// Segment one rendered frame to a binary person mask and write it to `out_path` as an
/// `L` (8-bit grayscale) PNG (0 / 255). `box_norm` is the tracked box in normalized
/// `(x, y, width, height)` space (0..1); it is scaled to the frame's pixel dimensions
/// and passed to SAM2 as a corner-point box prompt. The MLX model is loaded once and
/// cached process-wide (like the YOLO detector and Python's lazy model load); invoke via
/// `spawn_blocking`. The image decode + GPU segment + PNG encode are all blocking work,
/// so doing them together here keeps the mask off the async path entirely.
pub(crate) fn segment_person_blocking(
    weights_path: PathBuf,
    image_path: PathBuf,
    box_norm: (f64, f64, f64, f64),
    out_path: PathBuf,
) -> WorkerResult<()> {
    let img = image::open(&image_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
        .to_rgb8();
    let (width, height) = (img.width(), img.height());
    let box_xyxy = box_norm_to_pixels(box_norm, width, height);

    let cell = SEGMENTER.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().expect("person segmenter mutex poisoned");
    if guard.is_none() {
        let weights = Weights::from_file(&weights_path)
            .map_err(|e| WorkerError::InvalidPayload(format!("sam2 weights load: {e}")))?;
        let segmenter = Sam2Segmenter::from_weights_for_size(&weights, Sam2ModelSize::Large)
            .map_err(|e| WorkerError::InvalidPayload(format!("sam2 segmenter build: {e}")))?;
        *guard = Some(segmenter);
    }
    let segmenter = guard.as_ref().expect("segmenter loaded");
    let mask = segmenter
        .segment(img.as_raw(), height, width, box_xyxy)
        .map_err(|e| WorkerError::InvalidPayload(format!("sam2 segment: {e}")))?;
    let pixels = mask.as_slice::<u8>().to_vec();
    let gray = GrayImage::from_raw(width, height, pixels).ok_or_else(|| {
        WorkerError::InvalidPayload("sam2 mask dimensions did not match the frame".to_owned())
    })?;
    gray.save(&out_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("sam2 mask save: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_norm_scales_and_clamps_to_the_frame() {
        // A centered half-size box on a 1280×720 frame → [320,180,960,540].
        let px = box_norm_to_pixels((0.25, 0.25, 0.5, 0.5), 1280, 720);
        assert!((px[0] - 320.0).abs() < 1e-3);
        assert!((px[1] - 180.0).abs() < 1e-3);
        assert!((px[2] - 960.0).abs() < 1e-3);
        assert!((px[3] - 540.0).abs() < 1e-3);
        // A box that overflows the right/bottom edge clamps to the frame extent.
        let clamped = box_norm_to_pixels((0.9, 0.9, 0.5, 0.5), 100, 100);
        assert_eq!(clamped[2], 100.0);
        assert_eq!(clamped[3], 100.0);
    }

    #[test]
    fn mask_state_rollup_matches_python_segment_track() {
        // No masks written → degraded (box-mask fallback).
        assert_eq!(rollup_mask_state(0, 4), "degraded");
        // Every detected frame segmented → active.
        assert_eq!(rollup_mask_state(4, 4), "active");
        // A partial subset → generated.
        assert_eq!(rollup_mask_state(2, 4), "generated");
        // A single detected frame, segmented → active (not generated).
        assert_eq!(rollup_mask_state(1, 1), "active");
    }
}
