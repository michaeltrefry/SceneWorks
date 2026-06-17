//! Native-MLX **SAM3** text-concept person segmentation on the Rust worker (epic 4910, sc-4926).
//!
//! The box-prompt-free upgrade of `person_segment` (SAM2). Instead of prompting the segmenter
//! with the selected person's ByteTrack box, this drives the SAM3 **Promptable Concept
//! Segmentation (PCS)** video pipeline (`mlx-gen-sam3` `Sam3VideoModel::propagate`) from the text
//! concept `"person"`: SAM3 detects *every* person on every frame and tracks them across the clip
//! with its own memory bank + identity bookkeeping, returning per-frame `obj_id → mask`.
//!
//! Replace-Person still needs *one* selected person's mask per frame, so the per-frame ByteTrack
//! box stops being a *prompt* and becomes an *association hint*: we pick the SAM3 object whose
//! masks best fall inside the selected track's boxes across the span, then emit that object's
//! per-frame mask. The downstream contract is identical to the SAM2 path — one binary `L` mask
//! per clip frame, written under `person-tracks/{track_id}/masks/` by the orchestrator in
//! `media_jobs::segment_assembly_frames` — so the replacement loader and Wan-VACE are unchanged.
//!
//! macOS-only, like `person_segment` / `person_jobs`: `mlx-gen-sam3` builds Apple MLX from source
//! and is meaningless off Apple Silicon. **Cross-platform divergence (surfaced, not silent — cf.
//! epic 3792):** the Python/torch SAM2 *box-prompt* path stays the Windows/Linux backend until a
//! parallel SAM3 backport; only the macOS MLX worker gets the text-concept upgrade today.
//!
//! Unlike SAM2 (converted `.pt` → MLX), SAM3 loads the **stock `facebook/sam3` checkpoint
//! directly** (`model.safetensors` + `tokenizer.json`); no conversion step. The model is
//! affine-quantized after load (`Sam3VideoModel::quantize`, sc-4925) — **Q8 by default**
//! (~0.9 GB, near-lossless), tunable via `SCENEWORKS_SAM3_QUANT` (`q8`/`q4`/`off`).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::downloads::{ensure_hf_cached_file, DownloadContext};
use mlx_gen::weights::Weights;
use mlx_gen_sam3::{Sam3TextConfig, Sam3Tokenizer, Sam3VideoModel, VideoFrameOutput};
use mlx_rs::Array;

use crate::{Settings, WorkerError, WorkerResult};

/// SAM3 checkpoint files (loaded stock from `facebook/sam3`, no conversion).
const MODEL_FILE: &str = "model.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Download-on-first-use repo: the SAM3 weights mirror owned by SceneWorks. Publishing this
/// mirror (3.2 GB, Meta SAM License → must ship a LICENSE copy with the Materials) is gated on
/// sign-off; until then point `SCENEWORKS_SAM3_WEIGHTS` at a local `facebook/sam3` snapshot dir.
const SEG_REPO: &str = "SceneWorks/sam3-mlx";

/// SAM3 model input is a square 1008×1008 (the processor resizes to a fixed square, not
/// aspect-preserving — `Sam3VideoProcessor` `size={1008,1008}`, `default_to_square`).
const INPUT_SIZE: u32 = 1008;

/// SAM3 mask logits come back on a 288×288 low-res grid (`Sam3VideoModel` `LOW_RES`).
const MASK_GRID: usize = 288;

/// The text concept driving PCS — the whole point of the SAM3 upgrade (no manual box).
const CONCEPT_PROMPT: &str = "person";

/// A normalized `(x, y, width, height)` box (0..1), the per-frame ByteTrack anchor — used here
/// only to *associate* a SAM3 object id with the selected track, never as a segmenter prompt.
pub(crate) type BoxNorm = (f64, f64, f64, f64);

/// Affine-quantization bits for the segmenter, from `SCENEWORKS_SAM3_QUANT`: **Q8 by default**
/// (`8` — near-lossless, engine image Q8 IoU 0.9988, ~0.9 GB resident vs F32 ~3.2 GB), `q4` for
/// the smaller/lossier Q4, or `off`/`f32` to keep dense F32. `None` = no quantization.
fn quant_bits() -> Option<i32> {
    parse_quant_bits(&std::env::var("SCENEWORKS_SAM3_QUANT").unwrap_or_default())
}

/// Parse the `SCENEWORKS_SAM3_QUANT` value (split out so the mapping is unit-testable). Unset or
/// unrecognized → the safe Q8 default.
fn parse_quant_bits(value: &str) -> Option<i32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "f32" | "none" | "0" => None,
        "q4" | "4" => Some(4),
        _ => Some(8),
    }
}

/// The parsed SAM3 checkpoint is cached process-wide (the 3.2 GB safetensors parse is the
/// expensive part). A **fresh** `Sam3VideoModel` is assembled from it per clip: the model carries
/// per-session tracking state (obj ids, memory banks) and exposes no reset, so reusing one across
/// clips would leak identities. Building from cached weights is cheap (layer assembly over
/// already-resident arrays). Mirrors the SAM2 predictor cache + poison-recovery idiom.
static WEIGHTS: OnceLock<Mutex<Option<Weights>>> = OnceLock::new();

/// Resolve already-present SAM3 weights: an explicit env pin (`SCENEWORKS_SAM3_WEIGHTS`, a dir or
/// the `model.safetensors` inside it), then the app cache `<data_dir>/cache/person-segment-sam3/`,
/// then the model dir `<data_dir>/models/person-segment-sam3/`. Both `model.safetensors` and
/// `tokenizer.json` must be present. Returns `(model_path, tokenizer_path)` or `None` (then
/// `ensure_segmenter_weights` downloads them).
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

/// Resolve the SAM3 weights, downloading `model.safetensors` + `tokenizer.json` from HuggingFace
/// on first use (into the app cache) with streaming progress/cancel and size-aware resume.
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

/// Preprocess an RGB frame to the SAM3 input tensor: resize to a 1008×1008 square (bilinear,
/// matching the processor's fixed-square resize — *not* aspect-preserving), rescale to `[0,1]`,
/// normalize by mean/std `0.5` to `[-1,1]`, packed NCHW `[1,3,1008,1008]` f32.
fn input_tensor(img: &image::RgbImage) -> Array {
    let resized = image::imageops::resize(
        img,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let chw = normalize_chw(resized.as_raw(), INPUT_SIZE as usize);
    Array::from_slice(&chw, &[1, 3, INPUT_SIZE as i32, INPUT_SIZE as i32])
}

/// Pack a `size×size` interleaved-RGB `u8` buffer into a channel-major `[3·size·size]` f32 vector
/// with the SAM3 normalization `(x/255 − 0.5) / 0.5` = `x/127.5 − 1` (mean=std=0.5 → range
/// `[-1,1]`). Split out so the normalization is unit-testable without an image decode.
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

/// Fraction of a SAM3 object's foreground (mask logit `> 0` ⇔ σ `> 0.5`) on the `grid×grid`
/// low-res mask whose pixel center falls inside the normalized `box_norm`. `0.0` when the mask is
/// empty. SAM3 masks live in the squashed 1008² space, which maps to the full normalized frame, so
/// a mask pixel `(mx,my)` has normalized center `((mx+0.5)/grid, (my+0.5)/grid)` — directly
/// comparable to the ByteTrack box. This is the association score: how much of a candidate object
/// sits under the selected person's box.
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

/// Pick the SAM3 object id that best matches the selected track: for every clip frame with an
/// anchor box, accumulate each present object's `mask_box_containment`, then take the id with the
/// greatest total. `None` when no object ever overlaps an anchor (→ degraded to box masks). The
/// per-frame sum (not a single best frame) rewards an object that stays under the box across the
/// span, which disambiguates the selected person from nearby people SAM3 also segmented.
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
    // Deterministic tie-break on the object id (BTree-like) so repeated runs agree.
    score
        .into_iter()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(oid, _)| oid)
}

/// Binarize a `grid×grid` SAM3 mask (logit `> 0`) to a 0/255 buffer, then resize it to
/// `width×height` (bilinear) and re-threshold to a clean binary `L` mask — the per-clip-frame
/// output the orchestrator writes. Inverts SAM3's uniform 1008² squash back to the frame aspect.
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

/// Segment the selected person across a clip with the native-MLX SAM3 **text-concept (PCS) video
/// pipeline** (sc-4926). `clip_frame_paths` is the contiguous detected span (clip-local frame `0`
/// = first detected frame); `anchors[i]` is the frame's ByteTrack box in normalized
/// `(x, y, width, height)` when frame `i` was detected, else `None`. At least one anchor must be
/// `Some` — it is the association hint, not a prompt.
///
/// SAM3 runs once over the whole span (`propagate("person")`), segmenting and tracking *all* people
/// with its own identities; we then [`select_object`] the id that best overlaps the anchors and
/// emit that object's per-frame mask (gap frames the detector missed are still covered when SAM3
/// tracked the person through them — the same "survives weak-detection frames" win the SAM2 video
/// predictor gave us, now without any box prompt).
///
/// Returns one binary mask (row-major `width*height`, `0`/`255`) per clip frame, in clip order; an
/// empty vec for a frame where the selected object was absent (orchestrator skips empties → box
/// fallback). The checkpoint parses once and is cached process-wide; run under `spawn_blocking`
/// (image decode + GPU inference are blocking).
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

    // Decode every clip frame to RGB8 (shared rendered size) and build the SAM3 input tensors.
    let mut frames: Vec<Array> = Vec::with_capacity(clip_frame_paths.len());
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
        frames.push(input_tensor(&img));
    }

    // Cached checkpoint; recover from a poisoned lock by dropping the cached weights and reloading
    // (mirrors person_segment / sc-4277 F-MLXW-13).
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(&model_path)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    // Fresh model per clip (clean tracking state) + tokenize the concept once.
    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    // Quantize (Q8 default) for a ~0.9 GB footprint vs F32 ~3.2 GB (sc-4925). The dense path is
    // parity-preserving, so the F32 (`SCENEWORKS_SAM3_QUANT=off`) result is unchanged.
    if let Some(bits) = quant_bits() {
        model
            .quantize(bits)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize q{bits}: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(&tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT)
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

/// Every tracked person's per-frame mask + a stable left-to-right paint order — the input to the
/// SCAIL-2 color-mask painter (sc-5448). Unlike [`segment_track_blocking`] (which selects ONE person
/// via ByteTrack anchors for replace_person), this keeps EVERY SAM3 object so each person can be
/// painted a distinct palette color.
pub(crate) struct AllPersonMasks {
    /// SAM3 object ids in left-to-right paint order (ascending centroid-x in the frame where each
    /// object first appears); person index 0 = leftmost = the SCAIL-2 palette's first color.
    pub order: Vec<i32>,
    /// `per_frame[f]` = `(obj_id, binary mask row-major width*height, 0/255)` for every object
    /// present on frame `f` (empty masks dropped). Object ids index into [`AllPersonMasks::order`].
    pub per_frame: Vec<Vec<(i32, Vec<u8>)>>,
    pub width: u32,
    pub height: u32,
}

/// Normalized centroid-x (0..1) of a SAM3 low-res mask's foreground (logit `> 0`); `None` if the
/// mask is empty. Used to sort tracked people left-to-right for deterministic palette assignment.
fn mask_centroid_x(mask_logits: &[f32], grid: usize) -> Option<f64> {
    let (mut sum_x, mut n) = (0f64, 0u64);
    for my in 0..grid {
        for mx in 0..grid {
            if mask_logits[my * grid + mx] > 0.0 {
                sum_x += (mx as f64 + 0.5) / grid as f64;
                n += 1;
            }
        }
    }
    (n > 0).then(|| sum_x / n as f64)
}

/// Segment + track every "person" across already-decoded RGB `frames` with the SAM3 text-concept
/// (PCS) video pipeline, returning all objects' per-frame masks + the left-to-right paint order.
/// In-memory sibling of [`segment_track_blocking`] (no temp frame files): the SCAIL-2 reference /
/// driving frames are already decoded `Image`s, so the temp-PNG round-trip is skipped. `frames` must
/// be non-empty and uniform-sized. The checkpoint parses once and is cached process-wide; run under
/// `spawn_blocking` (GPU inference is blocking).
pub(crate) fn segment_all_persons_in_memory(
    model_path: &Path,
    tokenizer_path: &Path,
    frames: &[image::RgbImage],
) -> WorkerResult<AllPersonMasks> {
    let first = frames.first().ok_or_else(|| {
        WorkerError::InvalidPayload("scail2 segmentation: no frames to segment".into())
    })?;
    let (width, height) = (first.width(), first.height());
    if frames
        .iter()
        .any(|f| f.width() != width || f.height() != height)
    {
        return Err(WorkerError::InvalidPayload(
            "scail2 segmentation: frames are not all the same size".into(),
        ));
    }
    let tensors: Vec<Array> = frames.iter().map(input_tensor).collect();

    // Cached checkpoint; recover from a poisoned lock by dropping + reloading (mirrors
    // `segment_track_blocking` / sc-4277 F-MLXW-13).
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(model_path)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    if let Some(bits) = quant_bits() {
        model
            .quantize(bits)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize q{bits}: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT)
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;

    let outputs = model
        .propagate(&tensors, &input_ids, &text_mask)
        .map_err(|e| WorkerError::Engine(format!("sam3 propagate: {e}")))?;

    // Paint order: each object's centroid-x in the FIRST frame it appears, ascending (tie-break on
    // first-seen frame, then object id, so repeated runs agree).
    use std::collections::BTreeMap;
    let mut first_seen: BTreeMap<i32, (usize, f64)> = BTreeMap::new();
    for (f, frame) in outputs.iter().enumerate() {
        for (oid, logits) in frame.obj_ids.iter().zip(&frame.masks) {
            if first_seen.contains_key(oid) {
                continue;
            }
            if let Some(cx) = mask_centroid_x(logits, MASK_GRID) {
                first_seen.insert(*oid, (f, cx));
            }
        }
    }
    let mut order: Vec<i32> = first_seen.keys().copied().collect();
    order.sort_by(|a, b| {
        let (fa, xa) = first_seen[a];
        let (fb, xb) = first_seen[b];
        xa.partial_cmp(&xb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(fa.cmp(&fb))
            .then(a.cmp(b))
    });

    let per_frame = outputs
        .iter()
        .map(|frame| {
            frame
                .obj_ids
                .iter()
                .zip(&frame.masks)
                .map(|(oid, logits)| (*oid, mask_to_frame(logits, MASK_GRID, width, height)))
                .filter(|(_, mask)| !mask.is_empty())
                .collect()
        })
        .collect();

    Ok(AllPersonMasks {
        order,
        per_frame,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_bits_defaults_to_q8_and_honors_overrides() {
        assert_eq!(parse_quant_bits(""), Some(8), "unset → Q8 default");
        assert_eq!(parse_quant_bits("q8"), Some(8));
        assert_eq!(parse_quant_bits("8"), Some(8));
        assert_eq!(
            parse_quant_bits(" Q4 "),
            Some(4),
            "trimmed + case-insensitive"
        );
        assert_eq!(parse_quant_bits("4"), Some(4));
        assert_eq!(parse_quant_bits("off"), None);
        assert_eq!(parse_quant_bits("F32"), None);
        assert_eq!(parse_quant_bits("none"), None);
        assert_eq!(
            parse_quant_bits("garbage"),
            Some(8),
            "unrecognized → safe Q8"
        );
    }

    #[test]
    fn normalize_maps_to_signed_unit_range_channel_major() {
        // 2×2 RGB: black, white, and a mid-gray pixel. mean=std=0.5 → x/127.5 − 1.
        let rgb = [
            0, 0, 0, // (0,0) black
            255, 255, 255, // (0,1) white
            128, 128, 128, // (1,0) ~mid
            255, 0, 0, // (1,1) red
        ];
        let chw = normalize_chw(&rgb, 2);
        let plane = 4;
        // Channel-major: R plane first.
        assert!((chw[0] - (-1.0)).abs() < 1e-6); // R of black
        assert!((chw[1] - 1.0).abs() < 1e-6); // R of white
        assert!((chw[plane] - (-1.0)).abs() < 1e-6); // G of black
        assert!((chw[2 * plane + 3] - (-1.0)).abs() < 1e-6); // B of red pixel = 0 → -1
        assert!((chw[3] - 1.0).abs() < 1e-6); // R of red pixel = 255 → 1
    }

    #[test]
    fn containment_measures_foreground_inside_box() {
        // 10×10 grid, a 4×4 foreground block at rows/cols 2..6 (logit 1.0 inside, -1.0 outside).
        let grid = 10;
        let mut logits = vec![-1.0f32; grid * grid];
        for my in 2..6 {
            for mx in 2..6 {
                logits[my * grid + mx] = 1.0;
            }
        }
        // Box covering the whole block (normalized 0.2..0.6) → containment 1.0.
        let full = mask_box_containment(&logits, grid, (0.2, 0.2, 0.4, 0.4));
        assert!((full - 1.0).abs() < 1e-9, "full was {full}");
        // Box over the empty top-left corner → 0.0.
        let none = mask_box_containment(&logits, grid, (0.0, 0.0, 0.1, 0.1));
        assert!(none.abs() < 1e-9, "disjoint was {none}");
        // Box over the left half of the block → ~half the foreground inside.
        let half = mask_box_containment(&logits, grid, (0.0, 0.0, 0.4, 1.0));
        assert!((half - 0.5).abs() < 1e-9, "half was {half}");
        // Empty mask → 0.0, never divide-by-zero.
        assert_eq!(
            mask_box_containment(&vec![-1.0; grid * grid], grid, (0.0, 0.0, 1.0, 1.0)),
            0.0
        );
    }

    /// A small synthetic two-object clip: object 7 sits under the anchor box every frame, object 9
    /// sits elsewhere. The selector must return 7, aggregating containment across the span.
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
        // obj 7 in the top-left quadrant; obj 9 in the bottom-right.
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
        // Anchor over the top-left quadrant on both frames.
        let anchors = vec![Some((0.0, 0.0, 0.5, 0.5)), Some((0.0, 0.0, 0.5, 0.5))];
        assert_eq!(select_object(&outputs, &anchors), Some(7));
        // No anchors → no selection.
        assert_eq!(select_object(&outputs, &[None, None]), None);
    }

    #[test]
    fn mask_to_frame_binarizes_and_resizes() {
        // 4×4 grid, top-left 2×2 foreground → resized to 8×8 should keep a top-left fg region.
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

    /// Real-weights integration smoke (sc-4926, H4): the whole SceneWorks-side pipe — preprocess →
    /// `Sam3VideoModel::propagate("person")` → associate to the anchor → emit a per-frame mask —
    /// against the stock `facebook/sam3` checkpoint. Proves the cutover actually segments a person,
    /// not just that the pure helpers work. `#[ignore]`d (needs the 3.2 GB weights + GPU); run with:
    ///   SCENEWORKS_SAM3_WEIGHTS=<facebook/sam3 snapshot dir> \
    ///   SCENEWORKS_SAM3_SMOKE_IMAGE=<person jpg/png, e.g. ultralytics zidane.jpg 1280×720> \
    ///   cargo test -p sceneworks-worker --release sam3_real_weights_person_smoke -- --ignored --nocapture
    #[test]
    #[ignore = "real SAM3 weights + GPU; set SCENEWORKS_SAM3_WEIGHTS + SCENEWORKS_SAM3_SMOKE_IMAGE"]
    fn sam3_real_weights_person_smoke() {
        let snap = std::env::var("SCENEWORKS_SAM3_WEIGHTS")
            .expect("set SCENEWORKS_SAM3_WEIGHTS to a facebook/sam3 snapshot dir");
        let dir = {
            let p = PathBuf::from(&snap);
            if p.is_file() {
                p.parent().unwrap().to_path_buf()
            } else {
                p
            }
        };
        let model = dir.join(MODEL_FILE);
        let tokenizer = dir.join(TOKENIZER_FILE);
        let image = PathBuf::from(
            std::env::var("SCENEWORKS_SAM3_SMOKE_IMAGE")
                .expect("set SCENEWORKS_SAM3_SMOKE_IMAGE to a person image"),
        );

        // Two-frame clip from the same still; an anchor over the prominent foreground person
        // (zidane.jpg: the central/right player). The mask must mostly fall inside the anchor.
        let anchor: BoxNorm = (0.40, 0.10, 0.55, 0.88);
        let masks = segment_track_blocking(
            model,
            tokenizer,
            vec![image.clone(), image],
            vec![Some(anchor), Some(anchor)],
        )
        .expect("segment_track_blocking");

        assert_eq!(masks.len(), 2, "one mask per clip frame");
        let (w, h) = (1280usize, 720usize);
        for (i, mask) in masks.iter().enumerate() {
            assert_eq!(mask.len(), w * h, "frame {i} mask size");
            let fg = mask.iter().filter(|&&v| v > 127).count();
            let frac = fg as f64 / (w * h) as f64;
            // A real person mask covers a non-trivial, non-whole region of the frame.
            assert!(
                (0.02..0.80).contains(&frac),
                "frame {i} foreground fraction {frac:.3} is implausible (empty or whole-frame)"
            );
            // Most of the emitted foreground sits inside the anchor box.
            let (bx, by, bw, bh) = anchor;
            let (x1, y1, x2, y2) = (bx, by, bx + bw, by + bh);
            let inside = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .filter(|&(x, y)| mask[y * w + x] > 127)
                .filter(|&(x, y)| {
                    let (cx, cy) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                    cx >= x1 && cx < x2 && cy >= y1 && cy < y2
                })
                .count();
            let containment = inside as f64 / fg.max(1) as f64;
            assert!(
                containment > 0.5,
                "frame {i} mask containment in anchor {containment:.3} too low (wrong object?)"
            );
            eprintln!("frame {i}: fg_frac={frac:.3} containment={containment:.3}");
        }
    }
}
