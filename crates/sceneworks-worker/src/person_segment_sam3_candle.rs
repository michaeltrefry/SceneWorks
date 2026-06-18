//! Off-Mac (Windows/CUDA) **SAM3** text-concept person segmentation on the Rust worker — the candle
//! sibling of [`crate::person_segment_sam3`] (epic 5482, sc-6247 under sc-5062).
//!
//! Drives `candle-gen-sam3`'s `Sam3VideoModel::propagate("person")` exactly as the macOS path drives
//! the MLX twin: SAM3 detects + tracks *every* person across the clip with its own memory bank, and
//! we pick the object whose masks best fall inside the selected ByteTrack track's boxes (the per-frame
//! box is an *association hint*, never a segmenter prompt) and emit that object's per-frame binary
//! mask. The downstream contract is identical to the Mac SAM3 / SAM2 paths — one `L` mask per clip
//! frame, written under `person-tracks/{track_id}/masks/` by `media_jobs::segment_assembly_frames` —
//! so this **replaces the off-Mac SAM2 box-prompt stub** (`maskState = "missing"`) with real masks.
//!
//! Off-Mac + `backend-candle` only (`candle-gen-sam3` builds candle/CUDA). It loads the **stock
//! `facebook/sam3` checkpoint directly** (`model.safetensors` + `tokenizer.json`; no conversion).
//! Affine quant (`Sam3VideoModel::quantize`, sc-6246) is available via `SCENEWORKS_SAM3_QUANT`
//! (`q8`/`q4`), but the off-Mac default is **dense** — quantizing SAM3's PE vision ViT backbone NaNs
//! (its massive activations overflow GGUF's f16 quant scale, sc-6361; NOT a candle/Blackwell bug —
//! candle GGUF quant is correct on sm_120), and dense is bit-exact and fits the box. The pure
//! association/mask helpers are shared line-for-line with the MLX module; only the seam is candle.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::default_device;
use candle_gen::gen_core::Quant;
use candle_gen_sam3::{Sam3TextConfig, Sam3Tokenizer, Sam3VideoModel, VideoFrameOutput, Weights};

use crate::downloads::{ensure_hf_cached_file, DownloadContext};
use crate::{Settings, WorkerError, WorkerResult};

/// SAM3 checkpoint files (loaded stock from `facebook/sam3`, no conversion).
const MODEL_FILE: &str = "model.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Download-on-first-use repo: the stock `facebook/sam3` checkpoint mirror owned by SceneWorks (the
/// same `model.safetensors` + `tokenizer.json` the MLX path uses — no MLX-specific conversion, despite
/// the `-mlx` name). Until the public mirror is signed off, point `SCENEWORKS_SAM3_WEIGHTS` at a local
/// `facebook/sam3` snapshot dir.
const SEG_REPO: &str = "SceneWorks/sam3-mlx";

/// SAM3 model input is a square 1008×1008 (the processor resizes to a fixed square, not
/// aspect-preserving — `Sam3VideoProcessor` `size={1008,1008}`).
const INPUT_SIZE: u32 = 1008;

/// SAM3 mask logits come back on a 288×288 low-res grid (`Sam3VideoModel` `LOW_RES`).
const MASK_GRID: usize = 288;

/// The text concept driving PCS — the whole point of SAM3 (no manual box).
const CONCEPT_PROMPT: &str = "person";

/// A normalized `(x, y, width, height)` box (0..1), the per-frame ByteTrack anchor — used here only to
/// *associate* a SAM3 object id with the selected track, never as a segmenter prompt.
pub(crate) type BoxNorm = (f64, f64, f64, f64);

/// Affine-quantization level for the segmenter, from `SCENEWORKS_SAM3_QUANT`: **dense by default**
/// off-Mac (quantizing SAM3's PE vision backbone NaNs — its massive activations overflow GGUF's f16
/// quant scale, sc-6361, not a candle/Blackwell bug); `q8`/`q4` opt back in. `None` = dense F32.
fn quant_level() -> Option<Quant> {
    parse_quant(&std::env::var("SCENEWORKS_SAM3_QUANT").unwrap_or_default())
}

/// Parse the `SCENEWORKS_SAM3_QUANT` value (split out so the mapping is unit-testable). The default
/// off-Mac is **dense** (`None`): quantizing SAM3's PE vision ViT backbone produces NaN masks — its
/// massive activations overflow GGUF's f16 q8_1 block scale (sc-6361), hardware-agnostic and NOT a
/// candle/Blackwell bug (candle GGUF quant is correct on sm_120; the heads quantize fine, only the
/// backbone breaks). Dense is bit-exact and ~3.4 GB fits the GPU-worker box, so quant buys ~nothing
/// for SAM3. `q8`/`q4` remain opt-in; `off`/`f32`/`none`/unset/unrecognized all stay dense.
fn parse_quant(value: &str) -> Option<Quant> {
    match value.trim().to_ascii_lowercase().as_str() {
        "q4" | "4" => Some(Quant::Q4),
        "q8" | "8" => Some(Quant::Q8),
        _ => None,
    }
}

/// The parsed SAM3 checkpoint is cached process-wide (the multi-GB safetensors parse is the expensive
/// part). A **fresh** `Sam3VideoModel` is assembled from it per clip: the model carries per-session
/// tracking state (obj ids, memory banks) with no reset, so reusing one across clips would leak
/// identities. Mirrors the MLX module's cache + poison-recovery idiom.
static WEIGHTS: OnceLock<Mutex<Option<Weights>>> = OnceLock::new();

/// Resolve already-present SAM3 weights: an explicit env pin (`SCENEWORKS_SAM3_WEIGHTS`, a dir or the
/// `model.safetensors` inside it), then the app cache `<data_dir>/cache/person-segment-sam3/`, then the
/// model dir `<data_dir>/models/person-segment-sam3/`. Both files must be present. Returns
/// `(model_path, tokenizer_path)` or `None` (then [`ensure_segmenter_weights`] downloads them).
pub(crate) fn resolve_segmenter_weights(settings: &Settings) -> Option<(PathBuf, PathBuf)> {
    let pair_in = |dir: &Path| -> Option<(PathBuf, PathBuf)> {
        let model = dir.join(MODEL_FILE);
        let tokenizer = dir.join(TOKENIZER_FILE);
        (model.exists() && tokenizer.exists()).then_some((model, tokenizer))
    };
    if let Ok(pinned) = std::env::var("SCENEWORKS_SAM3_WEIGHTS") {
        let p = PathBuf::from(pinned);
        let dir = if p.is_file() {
            p.parent().map(Path::to_path_buf).unwrap_or(p)
        } else {
            p
        };
        if let Some(pair) = pair_in(&dir) {
            return Some(pair);
        }
    }
    for sub in ["cache/person-segment-sam3", "models/person-segment-sam3"] {
        if let Some(pair) = pair_in(&settings.data_dir.join(sub)) {
            return Some(pair);
        }
    }
    None
}

/// Resolve the SAM3 weights, downloading `model.safetensors` + `tokenizer.json` from HuggingFace on
/// first use (into the app cache) with streaming progress/cancel and size-aware resume.
pub(crate) async fn ensure_segmenter_weights(
    settings: &Settings,
    context: &DownloadContext<'_>,
) -> WorkerResult<(PathBuf, PathBuf)> {
    if let Some(pair) = resolve_segmenter_weights(settings) {
        return Ok(pair);
    }
    let dir = settings.data_dir.join("cache").join("person-segment-sam3");
    let model =
        ensure_hf_cached_file(context, SEG_REPO, "main", MODEL_FILE, &dir.join(MODEL_FILE)).await?;
    let tokenizer = ensure_hf_cached_file(
        context,
        SEG_REPO,
        "main",
        TOKENIZER_FILE,
        &dir.join(TOKENIZER_FILE),
    )
    .await?;
    Ok((model, tokenizer))
}

/// Roll the per-clip mask outcome into the sidecar `maskState`: `degraded` (no frame masked → box
/// fallback), `active` (every detected frame masked), else `generated`. Mirrors
/// `person_segment::rollup_mask_state` (Mac-only) so the off-Mac contract is identical.
pub(crate) fn rollup_mask_state(generated: usize, detected_total: usize) -> &'static str {
    if generated == 0 {
        "degraded"
    } else if generated >= detected_total {
        "active"
    } else {
        "generated"
    }
}

/// Preprocess an RGB frame to the SAM3 input tensor: resize to a 1008×1008 square (bilinear, fixed-
/// square — *not* aspect-preserving), normalize by mean/std `0.5` to `[-1,1]`, packed NCHW
/// `[1,3,1008,1008]` f32 on `device`.
fn input_tensor(img: &image::RgbImage, device: &Device) -> WorkerResult<Tensor> {
    let resized = image::imageops::resize(
        img,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let chw = normalize_chw(resized.as_raw(), INPUT_SIZE as usize);
    let n = INPUT_SIZE as usize;
    Tensor::from_vec(chw, (1, 3, n, n), device)
        .map_err(|e| WorkerError::Engine(format!("sam3 input tensor: {e}")))
}

/// Pack a `size×size` interleaved-RGB `u8` buffer into a channel-major `[3·size·size]` f32 vector with
/// the SAM3 normalization `x/127.5 − 1` (mean=std=0.5 → range `[-1,1]`). Split out so the
/// normalization is unit-testable without an image decode. Shared verbatim with the MLX module.
fn normalize_chw(rgb: &[u8], size: usize) -> Vec<f32> {
    let plane = size * size;
    let mut out = vec![0f32; 3 * plane];
    for (p, px) in rgb.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + p] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    out
}

/// Fraction of a SAM3 object's foreground (mask logit `> 0`) on the `grid×grid` low-res mask whose
/// pixel center falls inside the normalized `box_norm`. `0.0` when the mask is empty. The association
/// score: how much of a candidate object sits under the selected person's box.
fn mask_box_containment(mask_logits: &[f32], grid: usize, box_norm: BoxNorm) -> f64 {
    let (bx, by, bw, bh) = box_norm;
    let (x1, y1, x2, y2) = (bx, by, bx + bw, by + bh);
    let (mut fg, mut inside) = (0u64, 0u64);
    for my in 0..grid {
        for mx in 0..grid {
            if mask_logits[my * grid + mx] > 0.0 {
                fg += 1;
                let (cx, cy) = (
                    (mx as f64 + 0.5) / grid as f64,
                    (my as f64 + 0.5) / grid as f64,
                );
                if cx >= x1 && cx < x2 && cy >= y1 && cy < y2 {
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

/// Pick the SAM3 object id that best matches the selected track: accumulate each present object's
/// `mask_box_containment` over every anchored clip frame, then take the id with the greatest total.
/// `None` when no object ever overlaps an anchor (→ degraded to box masks). Shared with the MLX module.
fn select_object(outputs: &[VideoFrameOutput], anchors: &[Option<BoxNorm>]) -> Option<i32> {
    use std::collections::HashMap;
    let mut score: HashMap<i32, f64> = HashMap::new();
    for (frame, anchor) in outputs.iter().zip(anchors) {
        let Some(box_norm) = anchor else { continue };
        for (oid, mask) in frame.obj_ids.iter().zip(&frame.masks) {
            let c = mask_box_containment(mask, MASK_GRID, *box_norm);
            if c > 0.0 {
                *score.entry(*oid).or_insert(0.0) += c;
            }
        }
    }
    // Deterministic tie-break on the object id so repeated runs agree.
    score
        .into_iter()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(oid, _)| oid)
}

/// Binarize a `grid×grid` SAM3 mask (logit `> 0`) to 0/255, resize it to `width×height` (bilinear) and
/// re-threshold to a clean binary `L` mask — the per-clip-frame output the orchestrator writes.
fn mask_to_frame(mask_logits: &[f32], grid: usize, width: u32, height: u32) -> Vec<u8> {
    let bin: Vec<u8> = mask_logits
        .iter()
        .map(|&v| if v > 0.0 { 255 } else { 0 })
        .collect();
    let Some(small) = image::GrayImage::from_raw(grid as u32, grid as u32, bin) else {
        return Vec::new();
    };
    let resized =
        image::imageops::resize(&small, width, height, image::imageops::FilterType::Triangle);
    resized
        .into_raw()
        .into_iter()
        .map(|v| if v > 127 { 255 } else { 0 })
        .collect()
}

/// Segment the selected person across a clip with the off-Mac candle SAM3 **text-concept (PCS) video
/// pipeline** (sc-6247). `clip_frame_paths` is the contiguous detected span (clip-local frame `0` =
/// first detected frame); `anchors[i]` is the frame's ByteTrack box in normalized `(x, y, width,
/// height)` when frame `i` was detected, else `None`. At least one anchor must be `Some` — it is the
/// association hint, not a prompt.
///
/// Returns one binary mask (row-major `width*height`, `0`/`255`) per clip frame, in clip order; an
/// empty vec for a frame where the selected object was absent (orchestrator skips empties → box
/// fallback). The checkpoint parses once and is cached process-wide; run under `spawn_blocking` (image
/// decode + GPU inference are blocking).
pub(crate) fn segment_track_blocking(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    clip_frame_paths: Vec<PathBuf>,
    anchors: Vec<Option<BoxNorm>>,
) -> WorkerResult<Vec<Vec<u8>>> {
    assert_eq!(
        clip_frame_paths.len(),
        anchors.len(),
        "frames/anchors mismatch"
    );
    if !anchors.iter().any(Option::is_some) {
        return Err(WorkerError::InvalidPayload(
            "person segmentation clip needs at least one detected frame to associate".into(),
        ));
    }

    let device = default_device().map_err(|e| WorkerError::Engine(format!("sam3 device: {e}")))?;

    // Decode every clip frame to RGB8 (shared rendered size) and build the SAM3 input tensors.
    let mut frames: Vec<Tensor> = Vec::with_capacity(clip_frame_paths.len());
    let (mut width, mut height) = (0u32, 0u32);
    for path in &clip_frame_paths {
        let img = crate::image_decode::decode_image_any(path)
            .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
            .to_rgb8();
        if width == 0 {
            (width, height) = (img.width(), img.height());
        } else if img.width() != width || img.height() != height {
            return Err(WorkerError::InvalidPayload(
                "person clip frames are not all the same size".into(),
            ));
        }
        frames.push(input_tensor(&img, &device)?);
    }

    // Cached checkpoint; recover from a poisoned lock by dropping the cached weights and reloading.
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(&model_path, &device)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    // Fresh model per clip (clean tracking state) + tokenize the concept once.
    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    // Optional quant (opt-in via `SCENEWORKS_SAM3_QUANT`); off-Mac defaults to dense (sc-6361), so
    // unset leaves the F32 result unchanged.
    if let Some(quant) = quant_level() {
        model
            .quantize(quant)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(&tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT, &device)
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;

    let outputs = model
        .propagate(&frames, &input_ids, &text_mask)
        .map_err(|e| WorkerError::Engine(format!("sam3 propagate: {e}")))?;

    // Associate SAM3's identities to the selected track, then emit that object's per-frame mask.
    let Some(selected) = select_object(&outputs, &anchors) else {
        // SAM3 found no "person" overlapping any anchor → no masks (degrade to box fallback).
        return Ok(vec![Vec::new(); clip_frame_paths.len()]);
    };
    let masks = outputs
        .iter()
        .map(|frame| {
            frame
                .obj_ids
                .iter()
                .position(|&o| o == selected)
                .map(|i| mask_to_frame(&frame.masks[i], MASK_GRID, width, height))
                .unwrap_or_default()
        })
        .collect();
    Ok(masks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_defaults_to_dense_and_honors_opt_in() {
        assert_eq!(
            parse_quant(""),
            None,
            "unset → dense (SAM3 backbone quant overflows GGUF f16 scale, sc-6361)"
        );
        assert_eq!(parse_quant("q8"), Some(Quant::Q8), "explicit opt-in");
        assert_eq!(parse_quant("8"), Some(Quant::Q8));
        assert_eq!(
            parse_quant(" Q4 "),
            Some(Quant::Q4),
            "trimmed + case-insensitive"
        );
        assert_eq!(parse_quant("4"), Some(Quant::Q4));
        assert_eq!(parse_quant("off"), None);
        assert_eq!(parse_quant("F32"), None);
        assert_eq!(parse_quant("none"), None);
        assert_eq!(parse_quant("garbage"), None, "unrecognized → dense");
    }

    #[test]
    fn rollup_maps_generated_counts() {
        assert_eq!(rollup_mask_state(0, 5), "degraded");
        assert_eq!(rollup_mask_state(3, 5), "generated");
        assert_eq!(rollup_mask_state(5, 5), "active");
        assert_eq!(rollup_mask_state(6, 5), "active");
    }

    #[test]
    fn normalize_maps_to_signed_unit_range_channel_major() {
        // 2×2 RGB: black, white, mid-gray, red. mean=std=0.5 → x/127.5 − 1.
        let rgb = [0, 0, 0, 255, 255, 255, 128, 128, 128, 255, 0, 0];
        let chw = normalize_chw(&rgb, 2);
        let plane = 4;
        assert!((chw[0] - (-1.0)).abs() < 1e-6); // R of black
        assert!((chw[1] - 1.0).abs() < 1e-6); // R of white
        assert!((chw[plane] - (-1.0)).abs() < 1e-6); // G of black
        assert!((chw[2 * plane + 3] - (-1.0)).abs() < 1e-6); // B of red = 0 → -1
        assert!((chw[3] - 1.0).abs() < 1e-6); // R of red = 255 → 1
    }

    #[test]
    fn containment_measures_foreground_inside_box() {
        let grid = 10;
        let mut logits = vec![-1.0f32; grid * grid];
        for my in 2..6 {
            for mx in 2..6 {
                logits[my * grid + mx] = 1.0;
            }
        }
        let full = mask_box_containment(&logits, grid, (0.2, 0.2, 0.4, 0.4));
        assert!((full - 1.0).abs() < 1e-9, "full was {full}");
        let none = mask_box_containment(&logits, grid, (0.0, 0.0, 0.1, 0.1));
        assert!(none.abs() < 1e-9, "disjoint was {none}");
        let half = mask_box_containment(&logits, grid, (0.0, 0.0, 0.4, 1.0));
        assert!((half - 0.5).abs() < 1e-9, "half was {half}");
    }

    /// Two-object clip: object 7 sits under the anchor every frame, object 9 elsewhere. The selector
    /// must return 7, aggregating containment across the span.
    #[test]
    fn select_object_picks_the_id_overlapping_the_anchor() {
        let grid = MASK_GRID;
        let block = |r: std::ops::Range<usize>, c: std::ops::Range<usize>| -> Vec<f32> {
            let mut m = vec![-1.0f32; grid * grid];
            for y in r.clone() {
                for x in c.clone() {
                    m[y * grid + x] = 1.0;
                }
            }
            m
        };
        let left = block(0..grid / 2, 0..grid / 2);
        let right = block(grid / 2..grid, grid / 2..grid);
        let outputs = vec![
            VideoFrameOutput {
                obj_ids: vec![7, 9],
                masks: vec![left.clone(), right.clone()],
            },
            VideoFrameOutput {
                obj_ids: vec![7, 9],
                masks: vec![left, right],
            },
        ];
        let anchors = vec![Some((0.0, 0.0, 0.5, 0.5)), Some((0.0, 0.0, 0.5, 0.5))];
        assert_eq!(select_object(&outputs, &anchors), Some(7));
        assert_eq!(select_object(&outputs, &[None, None]), None);
    }

    #[test]
    fn mask_to_frame_binarizes_and_resizes() {
        let grid = 4;
        let mut logits = vec![-1.0f32; grid * grid];
        for y in 0..2 {
            for x in 0..2 {
                logits[y * grid + x] = 1.0;
            }
        }
        let out = mask_to_frame(&logits, grid, 8, 8);
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&v| v == 0 || v == 255), "output is binary");
        assert_eq!(out[0], 255, "top-left corner is foreground");
        assert_eq!(out[63], 0, "bottom-right corner is background");
    }
}
