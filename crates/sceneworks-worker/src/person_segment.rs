//! Native-MLX SAM2 person segmentation on the Rust worker (epic 3704, sc-3709 → sc-3715;
//! Slice 3 of sc-3488 / epic 3482).
//!
//! Ports the Python `scene_worker/person_adapters.py` `segment_track` to Rust so the
//! Replace-Person mask-generation step runs on a Python-free Mac. **sc-3715** upgrades the
//! per-frame box-prompt segmenter (sc-3709) to the native-MLX SAM2 **video predictor**
//! (`mlx-gen-sam2` `Sam2VideoPredictor`, sc-3714): prompt SAM2 once on the selected track's
//! first detected frame, then propagate temporally-consistent masks across the clip via the
//! memory bank — so frames where the detector dropped out (occlusion / motion blur) still get
//! a mask, and a detected frame that drifts is corrected back with its ByteTrack box. The
//! binary `L` masks are written under `person-tracks/{track_id}/masks/`.
//!
//! macOS-only, like `person_jobs`: `mlx-gen-sam2` builds Apple MLX from source and is
//! meaningless off Apple Silicon. The Python SAM2 path stays the Windows/Linux backend (its
//! per-frame quality gap is tracked in epic 3792 — backport video-SAM2 via candle-gen).
//!
//! The orchestration (which span to propagate, the maskState rollup, the sidecar write) lives
//! in `media_jobs::assemble_real_person_track`; this module owns the weight
//! resolution/download and the clip-level propagate call.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::downloads::ensure_cached_file;
use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ModelSize, Sam2VideoPredictor};

use crate::{Settings, WorkerError, WorkerResult};

/// The production SAM2 size (matches the spike + the `SceneWorks/sam2-mlx` upload).
const SEG_FILE: &str = "sam2.1_hiera_large.safetensors";

/// Download-on-first-use source: the converted MLX SAM2 weights owned by SceneWorks
/// (sc-3707, uploaded from the official Meta `.pt` via `tools/convert_sam2_to_mlx.py`;
/// bit-identical to the avbiswas reference). Public repo, so no credentials are needed —
/// same shape as the YOLO11 detector weights (`person_jobs::DET_URL`).
const SEG_URL: &str =
    "https://huggingface.co/SceneWorks/sam2-mlx/resolve/main/sam2.1_hiera_large.safetensors";

/// The SAM2 video predictor is loaded once and cached process-wide (weights load is the
/// expensive part; the per-clip tracking state is built fresh each call). Holds `None` until
/// the first track is propagated. Like the YOLO detector cache and Python's lazy model load.
static PREDICTOR: OnceLock<Mutex<Option<Sam2VideoPredictor>>> = OnceLock::new();

/// Minimum fraction of a propagated mask's foreground that must fall inside the frame's
/// ByteTrack box for a detected frame to count as "on the person". Below this the propagation
/// has drifted off the tracked person, so that frame is re-prompted with its box (2nd pass).
const COVERAGE_MIN: f64 = 0.5;

/// A normalized `(x, y, width, height)` box (0..1), the per-frame ByteTrack anchor.
pub(crate) type BoxNorm = (f64, f64, f64, f64);

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
    let target = settings
        .data_dir
        .join("cache")
        .join("person-segment")
        .join(SEG_FILE);
    ensure_cached_file(http_client, SEG_URL, &target).await
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

/// Fraction of a binary mask's foreground (`> 127`) that falls inside `box_norm` on a
/// `width`×`height` frame. `0.0` when the mask is empty. Used to detect a propagated frame
/// whose mask has drifted off the tracked person (low coverage → re-prompt with the box).
pub(crate) fn mask_box_coverage(pixels: &[u8], box_norm: BoxNorm, width: u32, height: u32) -> f64 {
    let [x1, y1, x2, y2] = box_norm_to_pixels(box_norm, width, height);
    let (w, h) = (width as usize, height as usize);
    let (mut fg, mut inside) = (0u64, 0u64);
    for y in 0..h {
        for x in 0..w {
            if pixels[y * w + x] > 127 {
                fg += 1;
                if (x as f32) >= x1 && (x as f32) < x2 && (y as f32) >= y1 && (y as f32) < y2 {
                    inside += 1;
                }
            }
        }
    }
    if fg == 0 {
        0.0
    } else {
        inside as f64 / fg as f64
    }
}

/// Propagate the selected person's mask across a clip with the native-MLX SAM2 **video
/// predictor** (sc-3714). `clip_frame_paths` is the contiguous span the track spans (clip-local
/// frame `0` = the first detected frame); `anchors[i]` is the frame's ByteTrack box in
/// normalized `(x, y, width, height)` space when frame `i` was detected, else `None`. `anchors[0]`
/// must be `Some` — it is the one-shot prompt.
///
/// Pass 1 prompts the first detected frame and propagates forward (the memory bank carries the
/// person through the non-detected gap frames). Pass 2 (only if needed) re-prompts any **detected**
/// frame whose mask drifted off its box (`mask_box_coverage < COVERAGE_MIN`) or came back empty,
/// re-seeding those frames as box anchors so the track is corrected — this guards against a drifted
/// propagated mask regressing vs the old per-frame box prompt.
///
/// Returns one binary mask (row-major `width*height`, `0`/`255`) per clip frame, in clip order.
/// The model loads once and is cached process-wide; run under `spawn_blocking` (image decode + GPU
/// propagation are blocking).
pub(crate) fn propagate_track_blocking(
    weights_path: PathBuf,
    clip_frame_paths: Vec<PathBuf>,
    anchors: Vec<Option<BoxNorm>>,
) -> WorkerResult<Vec<Vec<u8>>> {
    assert_eq!(
        clip_frame_paths.len(),
        anchors.len(),
        "frames/anchors mismatch"
    );
    let prompt =
        anchors.first().copied().flatten().ok_or_else(|| {
            WorkerError::InvalidPayload("propagate clip needs a prompt frame".into())
        })?;

    // Decode every clip frame to RGB8; they share the rendered frame size.
    let mut rgb: Vec<Vec<u8>> = Vec::with_capacity(clip_frame_paths.len());
    let (mut width, mut height) = (0u32, 0u32);
    for path in &clip_frame_paths {
        let img = image::open(path)
            .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
            .to_rgb8();
        if width == 0 {
            (width, height) = (img.width(), img.height());
        } else if img.width() != width || img.height() != height {
            return Err(WorkerError::InvalidPayload(
                "person clip frames are not all the same size".into(),
            ));
        }
        rgb.push(img.into_raw());
    }
    let frame_refs: Vec<&[u8]> = rgb.iter().map(|f| f.as_slice()).collect();

    let cell = PREDICTOR.get_or_init(|| Mutex::new(None));
    // Recover from a poisoned lock rather than panicking every subsequent job: a
    // prior propagation that panicked mid-run leaves the lock poisoned, so take the
    // inner guard and drop the possibly-corrupt cached predictor to force a clean
    // reload below (sc-4277 / F-MLXW-13).
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(&weights_path)
            .map_err(|e| WorkerError::InvalidPayload(format!("sam2 weights load: {e}")))?;
        let predictor =
            Sam2VideoPredictor::from_weights_for_size(&weights, Sam2ModelSize::Large)
                .map_err(|e| WorkerError::InvalidPayload(format!("sam2 predictor build: {e}")))?;
        *guard = Some(predictor);
    }
    let predictor = guard.as_ref().expect("predictor loaded");

    let run = |seeds: &[(usize, BoxNorm)]| -> WorkerResult<Vec<Vec<u8>>> {
        let mut state = predictor
            .init_state_from_frames(&frame_refs, height, width)
            .map_err(|e| WorkerError::InvalidPayload(format!("sam2 init_state: {e}")))?;
        for &(idx, box_norm) in seeds {
            predictor
                .add_new_box(
                    &mut state,
                    idx as i32,
                    box_norm_to_pixels(box_norm, width, height),
                )
                .map_err(|e| WorkerError::InvalidPayload(format!("sam2 add_box: {e}")))?;
        }
        let masks = predictor
            .propagate(&mut state)
            .map_err(|e| WorkerError::InvalidPayload(format!("sam2 propagate: {e}")))?;
        // `propagate` yields the prompt frame onward in order; build a dense per-clip-frame vec.
        let mut out = vec![Vec::new(); clip_frame_paths.len()];
        for (frame_idx, low) in &masks {
            let mask = predictor
                .mask_to_video_res(&state, low)
                .map_err(|e| WorkerError::InvalidPayload(format!("sam2 mask resize: {e}")))?;
            out[*frame_idx as usize] = mask.as_slice::<u8>().to_vec();
        }
        Ok(out)
    };

    // Pass 1: prompt once on the first detected frame.
    let pass1 = run(&[(0, prompt)])?;

    // Find detected frames whose propagated mask drifted off (or missed) the person.
    let weak: Vec<(usize, BoxNorm)> = anchors
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(i, anchor)| anchor.map(|b| (i, b)))
        .filter(|&(i, b)| {
            pass1.get(i).map(|m| !m.is_empty()).unwrap_or(false)
                && mask_box_coverage(&pass1[i], b, width, height) < COVERAGE_MIN
        })
        .collect();
    if weak.is_empty() {
        return Ok(pass1);
    }

    // Pass 2: re-seed the prompt + every weak frame as box anchors and re-propagate.
    let mut seeds = vec![(0usize, prompt)];
    seeds.extend(weak);
    run(&seeds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// sc-4277 / F-MLXW-13: the model-cache locks recover from poison instead of
    /// `.expect()`-panicking. A prior job panicking while holding the lock must
    /// leave a recoverable lock whose cached model is reset (None) so the next job
    /// reloads cleanly rather than panicking forever. This pins the recovery idiom
    /// used at the three call sites.
    #[test]
    fn poisoned_model_cache_lock_recovers_and_resets() {
        let cache: Mutex<Option<i32>> = Mutex::new(Some(7));
        // Poison the lock: panic in another thread while holding it.
        let poisoner = &cache;
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let _guard = poisoner.lock().unwrap();
                panic!("simulated mid-job panic while holding the model lock");
            });
            assert!(handle.join().is_err(), "the poisoning thread panicked");
        });
        assert!(cache.lock().is_err(), "precondition: lock is now poisoned");

        // The recovery idiom used at the call sites: take the inner guard and reset
        // the cached model so the reload path runs.
        let mut guard = cache.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        });
        assert_eq!(*guard, None, "cached model is dropped on poison recovery");
        *guard = Some(42); // reload
        assert_eq!(*guard, Some(42));
    }

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
    fn mask_box_coverage_measures_foreground_inside_box() {
        // 10×10 frame, a 4×4 foreground block at (2,2)..(6,6).
        let (w, h) = (10u32, 10u32);
        let mut pixels = vec![0u8; (w * h) as usize];
        for y in 2..6 {
            for x in 2..6 {
                pixels[y * w as usize + x] = 255;
            }
        }
        // A box covering the whole block → coverage 1.0.
        let full = mask_box_coverage(&pixels, (0.2, 0.2, 0.4, 0.4), w, h);
        assert!((full - 1.0).abs() < 1e-9, "full coverage was {full}");
        // A box over the empty top-left corner → coverage 0.0 (none of the fg inside).
        let none = mask_box_coverage(&pixels, (0.0, 0.0, 0.1, 0.1), w, h);
        assert!(none.abs() < 1e-9, "disjoint coverage was {none}");
        // An empty mask → 0.0, never a divide-by-zero.
        assert_eq!(
            mask_box_coverage(&[0u8; 100], (0.0, 0.0, 1.0, 1.0), w, h),
            0.0
        );
        // A box over the left half of the block → ~half the foreground inside.
        let half = mask_box_coverage(&pixels, (0.0, 0.0, 0.4, 1.0), w, h);
        assert!((half - 0.5).abs() < 1e-9, "half coverage was {half}");
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
