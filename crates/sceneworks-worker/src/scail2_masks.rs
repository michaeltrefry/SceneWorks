//! SCAIL-2 color-coded segmentation-mask painting (epic 5439, sc-5448).
//!
//! SCAIL-2 conditions on **color-coded** segmentation masks, not binary ones: each tracked person is
//! painted a distinct solid color, on a solid background, and the engine's
//! `extract_and_compress_mask_to_latent` reads those colors as one of seven exclusive classes
//! (threshold ≥ 225 per channel → white = R&G&B, red = R, green = G, blue = B, yellow = R&G,
//! magenta = R&B, cyan = G&B; all-off black = the "hidden" background class). This module turns the
//! native-SAM3 per-person masks ([`AllPersonMasks`]) into those painted RGB masks.
//!
//! Palette + conventions mirror upstream **SCAIL-Pose** (`TrackSam3/track.py`): one color per WHOLE
//! person (never per-body-part), persons ordered left-to-right, and the background encodes
//! whose-world-to-keep — animation keeps the reference's world (driving-mask bg **black**, ref-mask bg
//! **white**); cross-identity replacement keeps the driving's world (driving bg **white**, ref bg
//! **black**, used by sc-5452).

use gen_core::Image;

use crate::person_segment_sam3::AllPersonMasks;
use crate::{WorkerError, WorkerResult};

/// Person palette in RGB, person 0 = leftmost. Maps onto the engine's six chromatic color classes
/// (the seventh, white, is the "visible background" class). Upstream `DEFAULT_PALETTE_BGR` is written
/// BGR→RGB so the engine sees: 0 = blue, 1 = red, 2 = green, 3 = magenta, 4 = cyan, 5 = yellow.
pub(crate) const PALETTE: [[u8; 3]; 6] = [
    [0, 0, 255],   // person 0 — blue
    [255, 0, 0],   // person 1 — red
    [0, 255, 0],   // person 2 — green
    [255, 0, 255], // person 3 — magenta
    [0, 255, 255], // person 4 — cyan
    [255, 255, 0], // person 5 — yellow
];

/// Solid backgrounds: black = the "hidden" class (no channel on); white = the "visible" class.
pub(crate) const BG_BLACK: [u8; 3] = [0, 0, 0];
pub(crate) const BG_WHITE: [u8; 3] = [255, 255, 255];

/// The palette color for an object id, or `None` if the object is past the 6-color cap (SCAIL-2's
/// scheme has six person classes; extra people get no color and read as background).
fn color_for(order: &[i32], oid: i32) -> Option<[u8; 3]> {
    order
        .iter()
        .position(|&o| o == oid)
        .filter(|&i| i < PALETTE.len())
        .map(|i| PALETTE[i])
}

/// Fill an RGB pixel buffer with a solid background color.
fn filled(width: usize, height: usize, bg: [u8; 3]) -> Vec<u8> {
    let mut px = vec![0u8; width * height * 3];
    for chunk in px.chunks_exact_mut(3) {
        chunk.copy_from_slice(&bg);
    }
    px
}

/// Paint one person's binary mask into the RGB buffer with `color`.
fn paint(px: &mut [u8], mask: &[u8], color: [u8; 3]) {
    for (p, &m) in mask.iter().enumerate() {
        if m > 0 {
            let o = p * 3;
            if o + 3 <= px.len() {
                px[o..o + 3].copy_from_slice(&color);
            }
        }
    }
}

/// Paint the per-frame **driving** color masks: a solid `bg`, each tracked person filled its palette
/// color. People are painted in left-to-right order so a right-neighbor overlap wins (the upstream
/// paint order), keeping assignments deterministic. Returns one RGB mask per driving frame.
pub(crate) fn paint_driving_masks(masks: &AllPersonMasks, bg: [u8; 3]) -> Vec<Image> {
    let (w, h) = (masks.width as usize, masks.height as usize);
    masks
        .per_frame
        .iter()
        .map(|frame| {
            let mut px = filled(w, h, bg);
            for &oid in &masks.order {
                let Some(color) = color_for(&masks.order, oid) else {
                    continue;
                };
                if let Some((_, mask)) = frame.iter().find(|(o, _)| *o == oid) {
                    paint(&mut px, mask, color);
                }
            }
            Image {
                width: masks.width,
                height: masks.height,
                pixels: px,
            }
        })
        .collect()
}

/// Paint the **reference** character mask: the primary (largest-area) detected person filled blue
/// (the person-0 palette color, so it pairs with the driving clip's person 0) on a solid `bg`. The
/// reference is a single image → a single-frame [`AllPersonMasks`]. Errors if SAM3 found no person
/// (the reference must be a clear, unobstructed photo of the character).
pub(crate) fn paint_reference_mask(masks: &AllPersonMasks, bg: [u8; 3]) -> WorkerResult<Image> {
    let (w, h) = (masks.width as usize, masks.height as usize);
    let frame = masks.per_frame.first().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "scail2 reference: SAM3 returned no frame for the reference image".into(),
        )
    })?;
    let primary = frame
        .iter()
        .max_by_key(|(_, mask)| mask.iter().filter(|&&v| v > 0).count())
        .filter(|(_, mask)| mask.iter().any(|&v| v > 0))
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "scail2 reference: no person detected in the reference image — use a clear, \
                 unobstructed photo of the character"
                    .into(),
            )
        })?;
    let mut px = filled(w, h, bg);
    paint(&mut px, &primary.1, PALETTE[0]);
    Ok(Image {
        width: masks.width,
        height: masks.height,
        pixels: px,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×1 frame: pixel 0 owned by object 7, pixel 1 by object 3.
    fn two_person_masks() -> AllPersonMasks {
        // Object 3 is leftmost (paint order 0 → blue); object 7 is to its right (order 1 → red).
        AllPersonMasks {
            order: vec![3, 7],
            per_frame: vec![vec![(7, vec![0, 255]), (3, vec![255, 0])]],
            width: 2,
            height: 1,
        }
    }

    #[test]
    fn driving_paints_palette_by_left_to_right_order_on_bg() {
        let out = paint_driving_masks(&two_person_masks(), BG_BLACK);
        assert_eq!(out.len(), 1);
        // pixel 0 = object 3 = order[0] = blue; pixel 1 = object 7 = order[1] = red.
        assert_eq!(&out[0].pixels[0..3], &PALETTE[0], "leftmost person → blue");
        assert_eq!(&out[0].pixels[3..6], &PALETTE[1], "next person → red");
    }

    #[test]
    fn driving_background_is_solid_where_no_person() {
        let masks = AllPersonMasks {
            order: vec![1],
            per_frame: vec![vec![(1, vec![255, 0])]],
            width: 2,
            height: 1,
        };
        let out = paint_driving_masks(&masks, BG_WHITE);
        assert_eq!(&out[0].pixels[0..3], &PALETTE[0], "person → blue");
        assert_eq!(&out[0].pixels[3..6], &BG_WHITE, "empty pixel → white bg");
    }

    #[test]
    fn objects_past_six_get_no_color() {
        // Seven objects; the seventh (order index 6) is past the palette cap → stays background.
        let order: Vec<i32> = (0..7).collect();
        let per_frame = vec![(0..7)
            .map(|i: i32| (i, vec![if i == 6 { 255u8 } else { 0 }]))
            .collect()];
        let masks = AllPersonMasks {
            order,
            per_frame,
            width: 1,
            height: 1,
        };
        let out = paint_driving_masks(&masks, BG_BLACK);
        assert_eq!(
            &out[0].pixels[0..3],
            &BG_BLACK,
            "7th person → no color, stays bg"
        );
    }

    #[test]
    fn reference_paints_largest_person_blue() {
        // Object 9 has 1 pixel, object 4 has 3 pixels → object 4 is the primary character.
        let masks = AllPersonMasks {
            order: vec![4, 9],
            per_frame: vec![vec![(9, vec![255, 0, 0, 0]), (4, vec![0, 255, 255, 255])]],
            width: 4,
            height: 1,
        };
        let out = paint_reference_mask(&masks, BG_WHITE).expect("a person is present");
        assert_eq!(
            &out.pixels[0..3],
            &BG_WHITE,
            "object-9 pixel is not the primary → bg"
        );
        assert_eq!(
            &out.pixels[3..6],
            &PALETTE[0],
            "largest person painted blue"
        );
    }

    #[test]
    fn reference_errors_when_no_person() {
        let masks = AllPersonMasks {
            order: vec![],
            per_frame: vec![vec![]],
            width: 2,
            height: 1,
        };
        assert!(paint_reference_mask(&masks, BG_WHITE).is_err());
    }
}
