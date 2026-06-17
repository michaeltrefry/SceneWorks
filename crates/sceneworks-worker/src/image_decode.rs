//! Image-decode backstop (sc-6143).
//!
//! The worker's `image` crate is built `png`/`jpeg`/`webp`-only, so a valid AVIF/HEIC/HEIF/TIFF/
//! BMP/GIF asset fails to decode (`The image format Avif is not supported`). Import-time
//! normalization ([`sceneworks_core::project_store`]) converts new uploads to PNG, but assets that
//! predate that change — or arrive by a path that skips import normalization — would still fail at
//! the ~dozen worker decode sites.
//!
//! [`decode_image_any`] is a drop-in replacement for `image::open`: same signature, same error type.
//! It first tries the fast native decode (and a content-sniffed in-memory decode, which also handles
//! a supported image whose extension is wrong or absent), then, only for a recognized
//! unsupported-but-valid format, transcodes to a temp PNG via the shared
//! [`sceneworks_core::media_convert`] routine and decodes that. So a job degrades to
//! "transcode + run" instead of failing, and the format whack-a-mole lives in one place.

use std::io;
use std::path::Path;

use image::{DynamicImage, ImageError, ImageResult};
use sceneworks_core::media_convert::{sniff_image_kind, transcode_to_png};

/// Decode an image file, transcoding a valid-but-unsupported format to PNG on the fly. Drop-in for
/// `image::open`.
// Used on every always-compiled decode path (video/caption); the conditional allow only silences
// the bare (non-macOS, non-candle) lib build where the remaining callers are cfg'd out.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn decode_image_any(path: impl AsRef<Path>) -> ImageResult<DynamicImage> {
    let path = path.as_ref();
    match image::open(path) {
        Ok(image) => Ok(image),
        Err(open_error) => {
            // Read once; reuse for the in-memory decode and the format sniff.
            let bytes = std::fs::read(path).map_err(ImageError::IoError)?;
            // A supported image with a wrong/absent extension (the no-extension temp-upload case)
            // decodes here without shelling out — `image::open` keys off the extension.
            if let Ok(image) = image::load_from_memory(&bytes) {
                return Ok(image);
            }
            match transcode_then_decode(path, &bytes) {
                Some(Ok(image)) => Ok(image),
                // Transcode attempted but failed: surface the original decoder error (keeps the
                // familiar message) — the warning was already logged.
                Some(Err(_)) | None => Err(open_error),
            }
        }
    }
}

/// Decode image bytes, transcoding a valid-but-unsupported format to PNG on the fly. Drop-in for
/// `image::load_from_memory` (used where the worker already holds the bytes — e.g. a no-extension
/// temp upload sniffed from memory).
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn decode_image_bytes_any(bytes: &[u8]) -> ImageResult<DynamicImage> {
    match image::load_from_memory(bytes) {
        Ok(image) => Ok(image),
        Err(memory_error) => match transcode_then_decode_bytes(bytes) {
            Some(Ok(image)) => Ok(image),
            Some(Err(_)) | None => Err(memory_error),
        },
    }
}

/// `Some` when `bytes` are a recognized unsupported-but-valid format we attempted to transcode (the
/// inner `Result` is the decode outcome); `None` when the format is not one we transcode.
fn transcode_then_decode(src: &Path, bytes: &[u8]) -> Option<ImageResult<DynamicImage>> {
    let kind = sniff_image_kind(bytes)?;
    if kind.is_natively_supported() {
        return None; // would have decoded above already
    }
    Some(run_transcode_decode(|out| transcode_to_png(src, out)))
}

fn transcode_then_decode_bytes(bytes: &[u8]) -> Option<ImageResult<DynamicImage>> {
    let kind = sniff_image_kind(bytes)?;
    if kind.is_natively_supported() {
        return None;
    }
    Some(run_transcode_decode(|out| {
        let dir = out.parent().unwrap_or_else(|| Path::new("."));
        let src = dir.join("source.bin");
        std::fs::write(&src, bytes)
            .map_err(|error| sceneworks_core::media_convert::TranscodeError(error.to_string()))?;
        transcode_to_png(&src, out)
    }))
}

/// Run a transcode (writing a PNG into a temp dir) and decode the result, logging and downgrading a
/// transcode failure so the caller can fall back to the original decoder error.
fn run_transcode_decode<F>(transcode: F) -> ImageResult<DynamicImage>
where
    F: FnOnce(&Path) -> Result<(), sceneworks_core::media_convert::TranscodeError>,
{
    let dir = tempfile::tempdir().map_err(ImageError::IoError)?;
    let out = dir.path().join("decoded.png");
    if let Err(error) = transcode(&out) {
        eprintln!("image_decode_transcode_failed: {error}");
        return Err(ImageError::IoError(io::Error::other(error.to_string())));
    }
    image::open(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A supported image whose path has no extension still decodes (the `image::open`-fails →
    /// in-memory-decode fallback), without any transcode.
    #[test]
    fn decodes_supported_image_with_no_extension() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("no_extension_here");
        let png = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        png.save_with_format(&path, image::ImageFormat::Png)
            .expect("write png without a .png extension");

        let decoded = decode_image_any(&path).expect("decodes by content");
        assert_eq!(decoded.to_rgb8().dimensions(), (2, 2));
        assert_eq!(
            decode_image_bytes_any(&std::fs::read(&path).unwrap())
                .expect("bytes decode")
                .to_rgb8()
                .dimensions(),
            (2, 2)
        );
    }

    /// A genuinely unreadable/garbage file surfaces an error (not a panic, not a transcode hang).
    #[test]
    fn unrecognized_bytes_error_out() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("junk.png");
        std::fs::write(&path, b"this is not an image").expect("write junk");
        assert!(decode_image_any(&path).is_err());
        assert!(decode_image_bytes_any(b"this is not an image").is_err());
    }

    /// End-to-end backstop: a BMP (valid, but not in the worker's `image` build) decodes via the
    /// transcode path. macOS-only because it relies on `sips`; the ffmpeg path is identical.
    #[cfg(target_os = "macos")]
    #[test]
    fn decodes_unsupported_bmp_via_transcode() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("pixel.bmp");
        std::fs::write(&path, super::tests_support::one_pixel_bmp()).expect("write bmp");

        // The worker's image build can't decode BMP directly...
        assert!(image::open(&path).is_err());
        // ...but the backstop transcodes it.
        let decoded = decode_image_any(&path).expect("BMP decodes via transcode");
        assert_eq!(decoded.to_rgb8().dimensions(), (1, 1));
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests_support {
    /// A valid 1×1 24-bit BMP.
    pub(super) fn one_pixel_bmp() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&58u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&54u32.to_le_bytes());
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&2835i32.to_le_bytes());
        bytes.extend_from_slice(&2835i32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&[0x20, 0x40, 0x80, 0x00]);
        bytes
    }
}
