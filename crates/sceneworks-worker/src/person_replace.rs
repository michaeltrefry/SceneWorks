//! Replace-person mask pipeline (epic 3040, sc-3521).
//!
//! The native-Rust port of the Python worker's replace_person mask preparation
//! (`person_adapters.py` / `video_adapters.py`): turn a saved person track + its
//! stored SAM2 segmentation masks into the per-frame binary masks the Wan-VACE
//! engine consumes (`Conditioning::ControlClip { mask, .. }`; white = the person
//! region to regenerate). Person detect/track/segment stays upstream (native MLX
//! YOLO11 + SORT/ByteTrack + SAM2, sc-3633/3634/3709); only the rasterization /
//! resample / stored-mask-load logic ports here.
//!
//! This module is **cross-platform** (no MLX / engine types) so it unit-tests on
//! the Linux CI lane — the mask-port-vs-Python correctness gate. The engine call
//! that consumes the masks lives in [`crate::video_jobs`] (macOS-only).
//!
//! Faithful ports (same arithmetic so the rasterized masks match the Python output
//! for a fixture track):
//!   * [`apply_track_corrections`] — `person_adapters.apply_track_corrections`
//!   * [`resample_indices`] — `person_adapters._resample_indices`
//!   * [`box_mask`] — `person_adapters._box_mask` (3% pad, PIL-inclusive rectangle)
//!   * [`load_track_masks`] — `person_adapters.load_track_masks` (stored seg mask /
//!     box fallback / `segmentation`|`degraded_box`|`mixed` mode)
//!   * [`person_track_masks`] — `video_adapters.person_track_masks` (the
//!     selected-detection single-frame fallback when a track has no frames)

use std::path::Path;

use image::RgbImage;
use serde_json::{json, Value};

use crate::{WorkerError, WorkerResult};

/// The chosen mask source across the resampled frames (Python `load_track_masks` mode).
pub(crate) const MODE_SEGMENTATION: &str = "segmentation";
pub(crate) const MODE_DEGRADED_BOX: &str = "degraded_box";
pub(crate) const MODE_MIXED: &str = "mixed";

/// Clamp to `[0, 1]` (Python `safe_float(.., 0.0, 0.0, 1.0)`).
fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Round half-to-even (banker's rounding), matching Python's built-in `round` so the
/// resampled frame indices match the Python worker exactly.
fn python_round(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    if diff < 0.5 {
        floor as i64
    } else if diff > 0.5 {
        floor as i64 + 1
    } else {
        // Exactly .5 → round to the even neighbour.
        let f = floor as i64;
        if f % 2 == 0 {
            f
        } else {
            f + 1
        }
    }
}

/// Resampled frame indices mapping `count` output frames onto `total` track frames
/// (Python `_resample_indices`): evenly spaced inclusive of both ends, clamped to the
/// last index. `count <= 1` or `total <= 1` → all-zero (the single frame repeated).
pub(crate) fn resample_indices(total: usize, count: usize) -> Vec<usize> {
    if count <= 1 || total <= 1 {
        return vec![0; count.max(1)];
    }
    (0..count)
        .map(|index| {
            let position = (total - 1) as f64 * index as f64 / (count - 1) as f64;
            (python_round(position) as usize).min(total - 1)
        })
        .collect()
}

/// Apply persisted box/reject corrections to a track's frames (Python
/// `apply_track_corrections`, sc-1485). Returns cloned frame objects with corrections
/// applied — never mutating the sidecar:
///   * a corrected box overrides the tracked box and clears that frame's stored mask
///     (the old mask no longer matches the new box → the loader regenerates a box mask);
///   * a rejected frame borrows the nearest accepted frame's box (mask cleared), so the
///     mask never neutralizes a region the user flagged as wrong.
///
/// Corrections are keyed by `frameIndex`; out-of-range / malformed entries are ignored.
pub(crate) fn apply_track_corrections(track: &Value) -> Vec<Value> {
    let mut frames: Vec<Value> = track
        .get("frames")
        .and_then(Value::as_array)
        .map(|frames| frames.iter().filter(|f| f.is_object()).cloned().collect())
        .unwrap_or_default();
    if frames.is_empty() {
        return frames;
    }

    // Corrections keyed by frameIndex (bool is excluded — Python rejects `isinstance bool`).
    let mut rejected = vec![false; frames.len()];
    if let Some(corrections) = track.get("corrections").and_then(Value::as_array) {
        for correction in corrections {
            let Some(correction) = correction.as_object() else {
                continue;
            };
            let index = match correction.get("frameIndex") {
                Some(Value::Number(number)) => match number.as_i64() {
                    Some(index) if (0..frames.len() as i64).contains(&index) => index as usize,
                    _ => continue,
                },
                _ => continue,
            };
            if let Some(frame) = frames[index].as_object_mut() {
                if let Some(box_value) = correction.get("box").filter(|b| b.is_object()) {
                    frame.insert("box".to_owned(), box_value.clone());
                    frame.insert("mask".to_owned(), Value::Null);
                    frame.insert("corrected".to_owned(), Value::Bool(true));
                }
                if correction
                    .get("rejected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    rejected[index] = true;
                    frame.insert("rejected".to_owned(), Value::Bool(true));
                }
            }
        }
    }

    let accepted: Vec<usize> = (0..frames.len())
        .filter(|index| !rejected[*index])
        .collect();
    if !accepted.is_empty() && accepted.len() < frames.len() {
        let accepted_boxes: Vec<(usize, Value)> = accepted
            .iter()
            .map(|&index| {
                (
                    index,
                    frames[index].get("box").cloned().unwrap_or(Value::Null),
                )
            })
            .collect();
        for index in 0..frames.len() {
            if !rejected[index] {
                continue;
            }
            // Nearest accepted frame, tie-broken toward the lower index (Python
            // `min(accepted, key=lambda c: (abs(c - index), c))`).
            let nearest = accepted_boxes
                .iter()
                .min_by_key(|(candidate, _)| (candidate.abs_diff(index), *candidate))
                .map(|(_, box_value)| box_value.clone())
                .unwrap_or(Value::Null);
            if let Some(frame) = frames[index].as_object_mut() {
                frame.insert("box".to_owned(), nearest);
                frame.insert("mask".to_owned(), Value::Null);
            }
        }
    }
    frames
}

/// Rasterize a normalized `{x,y,width,height}` box to a binary `width × height` RGB mask
/// (white = the person region to regenerate; black elsewhere) with a 3% pad on each side
/// (Python `_box_mask`). A missing/empty box object → an all-black mask. The rectangle is
/// PIL-inclusive of its far edge (matching `ImageDraw.rectangle`), clipped to the image.
pub(crate) fn box_mask(box_value: Option<&Value>, width: u32, height: u32) -> RgbImage {
    let mut mask = RgbImage::new(width, height);
    let Some(object) = box_value.and_then(Value::as_object) else {
        return mask;
    };
    if object.is_empty() {
        return mask;
    }
    let get = |key: &str| clamp01(object.get(key).and_then(Value::as_f64).unwrap_or(0.0));
    let (x, y, w, h) = (get("x"), get("y"), get("width"), get("height"));
    let (wf, hf) = (width as f64, height as f64);
    // int(..) truncates toward zero; x/y/w/h are in [0,1] so all products are non-negative.
    let pad_x = (wf * 0.03) as i64;
    let pad_y = (hf * 0.03) as i64;
    let left = ((x * wf) as i64 - pad_x).max(0);
    let top = ((y * hf) as i64 - pad_y).max(0);
    let right = (((x + w) * wf) as i64 + pad_x).min(width as i64);
    let bottom = (((y + h) * hf) as i64 + pad_y).min(height as i64);
    // PIL rectangle fills both corners inclusively; clip the far edge to the last pixel.
    let x1 = right.min(width as i64 - 1);
    let y1 = bottom.min(height as i64 - 1);
    for py in top..=y1 {
        for px in left..=x1 {
            mask.put_pixel(px as u32, py as u32, image::Rgb([255, 255, 255]));
        }
    }
    mask
}

/// Load per-frame replacement masks for a track, resampled to `count` frames (Python
/// `load_track_masks`). Corrections are applied first; then per frame a stored
/// segmentation mask is used when its file still exists **and** the box was not corrected,
/// otherwise a box mask is rasterized from the (possibly corrected) box. Returns the masks
/// plus the source mode: [`MODE_SEGMENTATION`] (all stored), [`MODE_DEGRADED_BOX`] (all
/// box-derived), or [`MODE_MIXED`].
pub(crate) fn load_track_masks(
    project_path: &Path,
    track: &Value,
    width: u32,
    height: u32,
    count: usize,
) -> WorkerResult<(Vec<RgbImage>, &'static str)> {
    let frames = apply_track_corrections(track);
    if frames.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Person track has no frames; cannot build replacement masks.".to_owned(),
        ));
    }
    let indices = resample_indices(frames.len(), count);
    let mut masks = Vec::with_capacity(indices.len());
    let mut segmentation = 0usize;
    for index in &indices {
        let frame = &frames[*index];
        let stored = frame
            .get("mask")
            .and_then(Value::as_str)
            .map(|rel| project_path.join(rel))
            .filter(|path| path.exists());
        match stored {
            Some(path) => {
                let loaded = crate::image_decode::decode_image_any(&path)
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "replacement mask {}: {error}",
                            path.display()
                        ))
                    })?
                    .to_luma8();
                // Match Pillow `convert("L").resize((w, h))` (default bilinear), then fan the
                // single luma channel out to RGB (white = regenerate).
                let resized = image::imageops::resize(
                    &loaded,
                    width,
                    height,
                    image::imageops::FilterType::Triangle,
                );
                let mut rgb = RgbImage::new(width, height);
                for (dst, src) in rgb.pixels_mut().zip(resized.pixels()) {
                    let value = src.0[0];
                    *dst = image::Rgb([value, value, value]);
                }
                masks.push(rgb);
                segmentation += 1;
            }
            None => masks.push(box_mask(frame.get("box"), width, height)),
        }
    }
    let mode = if segmentation == indices.len() {
        MODE_SEGMENTATION
    } else if segmentation == 0 {
        MODE_DEGRADED_BOX
    } else {
        MODE_MIXED
    };
    Ok((masks, mode))
}

/// Per-frame replacement masks for a track, with the selected-detection single-frame
/// fallback (Python `video_adapters.person_track_masks`): a track with no frames but a
/// `selectedDetection.box` is treated as a one-frame track from that box.
pub(crate) fn person_track_masks(
    project_path: &Path,
    track: &Value,
    width: u32,
    height: u32,
    count: usize,
) -> WorkerResult<(Vec<RgbImage>, &'static str)> {
    let has_frames = track
        .get("frames")
        .and_then(Value::as_array)
        .is_some_and(|frames| !frames.is_empty());
    if has_frames {
        return load_track_masks(project_path, track, width, height, count);
    }
    let selected_box = track
        .get("selectedDetection")
        .and_then(|detection| detection.get("box"))
        .filter(|box_value| box_value.is_object())
        .cloned()
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Person track has no usable boxes.".to_owned())
        })?;
    let synthetic = json!({ "frames": [{ "box": selected_box, "mask": Value::Null }] });
    load_track_masks(project_path, &synthetic, width, height, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_pixels(mask: &RgbImage) -> usize {
        mask.pixels().filter(|p| p.0[0] > 0).count()
    }

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(python_round(0.5), 0);
        assert_eq!(python_round(1.5), 2);
        assert_eq!(python_round(2.5), 2);
        assert_eq!(python_round(2.4), 2);
        assert_eq!(python_round(2.6), 3);
    }

    #[test]
    fn resample_indices_matches_python_formula() {
        // count <= 1 or total <= 1 → repeated zero.
        assert_eq!(resample_indices(10, 1), vec![0]);
        assert_eq!(resample_indices(1, 4), vec![0, 0, 0, 0]);
        // Evenly spaced inclusive of both ends.
        assert_eq!(resample_indices(5, 5), vec![0, 1, 2, 3, 4]);
        // Upsample: 3 track frames onto 5 outputs (round half-to-even at the .5 steps).
        // positions: 0, 0.5, 1.0, 1.5, 2.0 → 0, 0, 1, 2, 2.
        assert_eq!(resample_indices(3, 5), vec![0, 0, 1, 2, 2]);
        // Downsample: 9 track frames onto 5 outputs → 0, 2, 4, 6, 8.
        assert_eq!(resample_indices(9, 5), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn box_mask_pads_three_percent_and_fills_inclusive() {
        // A centered half-size box at 100×100: pad = int(100*0.03) = 3 each side.
        let box_value = json!({ "x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5 });
        let mask = box_mask(Some(&box_value), 100, 100);
        // left=22, top=22, right=78, bottom=78 → inclusive 22..=78 = 57 px each axis.
        assert_eq!(white_pixels(&mask), 57 * 57);
        assert_eq!(mask.get_pixel(22, 22).0, [255, 255, 255]);
        assert_eq!(mask.get_pixel(78, 78).0, [255, 255, 255]);
        assert_eq!(mask.get_pixel(21, 21).0, [0, 0, 0]);
        assert_eq!(mask.get_pixel(79, 79).0, [0, 0, 0]);
    }

    #[test]
    fn box_mask_clips_to_image_bounds() {
        // A full-frame box pads past the edges → the whole image is white, no panic.
        let box_value = json!({ "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0 });
        let mask = box_mask(Some(&box_value), 40, 24);
        assert_eq!(white_pixels(&mask), 40 * 24);
    }

    #[test]
    fn box_mask_missing_or_empty_is_black() {
        assert_eq!(white_pixels(&box_mask(None, 16, 16)), 0);
        let empty = json!({});
        assert_eq!(white_pixels(&box_mask(Some(&empty), 16, 16)), 0);
    }

    #[test]
    fn corrections_override_box_and_clear_mask() {
        let track = json!({
            "frames": [
                { "box": { "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2 }, "mask": "m0.png" },
                { "box": { "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2 }, "mask": "m1.png" }
            ],
            "corrections": [
                { "frameIndex": 0, "box": { "x": 0.3, "y": 0.3, "width": 0.1, "height": 0.1 } }
            ]
        });
        let frames = apply_track_corrections(&track);
        assert_eq!(frames[0]["box"]["x"], json!(0.3));
        assert_eq!(frames[0]["mask"], Value::Null);
        assert_eq!(frames[0]["corrected"], json!(true));
        // The untouched frame keeps its stored mask.
        assert_eq!(frames[1]["mask"], json!("m1.png"));
    }

    #[test]
    fn rejected_frame_borrows_nearest_accepted_box() {
        let track = json!({
            "frames": [
                { "box": { "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2 }, "mask": "m0.png" },
                { "box": { "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2 }, "mask": "m1.png" },
                { "box": { "x": 0.8, "y": 0.8, "width": 0.1, "height": 0.1 }, "mask": "m2.png" }
            ],
            "corrections": [ { "frameIndex": 1, "rejected": true } ]
        });
        let frames = apply_track_corrections(&track);
        // Frame 1 rejected → nearest accepted (0 and 2 are equidistant; tie → lower index 0).
        assert_eq!(frames[1]["box"]["x"], json!(0.1));
        assert_eq!(frames[1]["mask"], Value::Null);
        assert_eq!(frames[1]["rejected"], json!(true));
    }

    #[test]
    fn load_track_masks_degrades_to_box_when_no_stored_mask() {
        let dir = std::env::temp_dir().join(format!("sw_masks_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let track = json!({
            "frames": [
                { "box": { "x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5 }, "mask": Value::Null }
            ]
        });
        let (masks, mode) = load_track_masks(&dir, &track, 64, 64, 3).unwrap();
        assert_eq!(masks.len(), 3);
        assert_eq!(mode, MODE_DEGRADED_BOX);
        assert!(white_pixels(&masks[0]) > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_track_masks_uses_stored_segmentation_when_present() {
        let dir = std::env::temp_dir().join(format!("sw_masks_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        // A real stored mask file: a 10×10 all-white luma PNG.
        let stored = image::GrayImage::from_pixel(10, 10, image::Luma([255]));
        stored.save(dir.join("seg0.png")).unwrap();
        let track = json!({
            "frames": [ { "box": { "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1 }, "mask": "seg0.png" } ]
        });
        let (masks, mode) = load_track_masks(&dir, &track, 32, 32, 1).unwrap();
        assert_eq!(mode, MODE_SEGMENTATION);
        // The stored all-white mask resizes to a fully-white 32×32 (not the small box).
        assert_eq!(white_pixels(&masks[0]), 32 * 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn person_track_masks_falls_back_to_selected_detection() {
        let dir = std::env::temp_dir().join(format!("sw_masks_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let track = json!({
            "frames": [],
            "selectedDetection": { "box": { "x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5 } }
        });
        let (masks, mode) = person_track_masks(&dir, &track, 64, 64, 2).unwrap();
        assert_eq!(masks.len(), 2);
        assert_eq!(mode, MODE_DEGRADED_BOX);
        assert!(white_pixels(&masks[0]) > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn person_track_masks_errors_without_frames_or_selection() {
        let dir = std::env::temp_dir();
        let track = json!({ "frames": [] });
        assert!(person_track_masks(&dir, &track, 32, 32, 1).is_err());
    }
}
