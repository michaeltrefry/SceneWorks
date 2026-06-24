//! One-tap image fixes that rewrite an image's pixels (sc-6539): smart-crop and EXIF-strip. Both
//! decode through [`load_oriented`] so the EXIF orientation tag is baked into the pixels first, then
//! re-encode as PNG — which carries no metadata, so the result is upright + clean. The orchestration
//! (resolve items → transform → re-point) lives in the API; these are the pure pixel ops.

use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use sceneworks_core::dataset_quality::plan_smart_crop;

/// Decode an image and bake in its EXIF orientation. `image` 0.25 does NOT auto-apply orientation on
/// decode, and supported uploads are stored byte-for-byte (a phone JPEG keeps its orientation tag),
/// so every decode in the pipeline currently ignores it. Reading the tag here and applying it makes
/// the *pixels* upright — so a subsequent re-encode that drops the tag (PNG carries none) corrects the
/// image rather than silently rotating it. A format with no orientation (PNG) yields a no-op.
pub fn load_oriented(path: &Path) -> image::ImageResult<DynamicImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// EXIF-strip fix: write an upright, metadata-free PNG copy of `src` to `out` (drops EXIF/GPS/ICC).
pub fn write_metadata_stripped(src: &Path, out: &Path) -> image::ImageResult<()> {
    load_oriented(src)?.save_with_format(out, ImageFormat::Png)
}

/// Smart-crop fix: trim `src` toward a trainable aspect (crop-loss ≤ `target_crop_loss`) and write the
/// result as PNG to `out`. The crop is planned on the *oriented* dimensions (so a rotated source crops
/// the correct axis) — [`plan_smart_crop`] returns `None` when no crop is needed, in which case the
/// oriented image is written unchanged (still upright + metadata-stripped). Returns whether a crop was
/// applied. `bias` is an optional normalized focal point (`None` centers — the saliency-free default).
pub fn write_smart_cropped(
    src: &Path,
    target_crop_loss: f64,
    bias: Option<(f64, f64)>,
    out: &Path,
) -> image::ImageResult<bool> {
    let image = load_oriented(src)?;
    match plan_smart_crop(image.width(), image.height(), target_crop_loss, bias) {
        Some(rect) => {
            image
                .crop_imm(rect.x, rect.y, rect.width, rect.height)
                .save_with_format(out, ImageFormat::Png)?;
            Ok(true)
        }
        None => {
            image.save_with_format(out, ImageFormat::Png)?;
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageReader, RgbImage};

    fn write_png(dir: &std::path::Path, name: &str, w: u32, h: u32) -> std::path::PathBuf {
        let mut img = RgbImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
        }
        let path = dir.join(name);
        img.save_with_format(&path, ImageFormat::Png).unwrap();
        path
    }

    fn dims(path: &std::path::Path) -> (u32, u32) {
        ImageReader::open(path)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap()
    }

    #[test]
    fn smart_crop_trims_an_extreme_aspect_to_png() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "wide.png", 200, 64);
        let out = dir.path().join("wide-cropped.png");
        let cropped = write_smart_cropped(&src, 0.25, None, &out).unwrap();
        assert!(cropped, "200x64 (crop-loss 0.68) needs a crop");
        let (w, h) = dims(&out);
        assert_eq!(h, 64, "short edge kept");
        assert!((64..200).contains(&w));
        assert!(
            f64::from(w - h) / f64::from(w) < 0.35,
            "clears the crop-loss flag"
        );
    }

    #[test]
    fn smart_crop_leaves_a_square_uncropped_but_re_encodes() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "square.png", 100, 100);
        let out = dir.path().join("square-out.png");
        let cropped = write_smart_cropped(&src, 0.25, None, &out).unwrap();
        assert!(!cropped, "square needs no crop");
        assert_eq!(
            dims(&out),
            (100, 100),
            "written unchanged (oriented + stripped)"
        );
    }

    #[test]
    fn strip_metadata_writes_a_valid_same_size_png() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "meta.png", 48, 32);
        let out = dir.path().join("meta-stripped.png");
        write_metadata_stripped(&src, &out).unwrap();
        assert_eq!(dims(&out), (48, 32));
        // re-decodes cleanly (a valid PNG).
        assert!(load_oriented(&out).is_ok());
    }

    #[test]
    fn load_oriented_is_a_no_op_for_an_unoriented_png() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "plain.png", 70, 50);
        let img = load_oriented(&src).unwrap();
        assert_eq!((img.width(), img.height()), (70, 50));
    }

    #[test]
    fn baking_a_90_degree_orientation_swaps_dims() {
        // The behavior load_oriented relies on for rotated sources: a 90°/270° tag swaps width and
        // height, which is exactly why the crop must be planned on the *oriented* dimensions.
        let mut img = image::DynamicImage::ImageRgb8(RgbImage::new(10, 20));
        img.apply_orientation(image::metadata::Orientation::Rotate90);
        assert_eq!((img.width(), img.height()), (20, 10));
    }
}
