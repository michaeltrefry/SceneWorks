//! Depth-map preprocessor for the Fun-Controlnet-Union depth head (epic 8236, sc-8242 mac /
//! sc-8413 candle). A model-agnostic preprocessor — sibling of [`crate::canny`] /
//! `openpose_skeleton`: an arbitrary input image → a depth control image for
//! `ControlKind::Depth`.
//!
//! Unlike canny/pose (pure raster, cross-platform), depth needs real neural inference, so it runs
//! a from-scratch Depth Anything V2 port (DINOv2 ViT-S/14 + DPT) on whichever backend the build
//! selects:
//! - **macOS** — the native-MLX [`mlx_gen_depth`] port (sc-8242);
//! - **off-Mac + `backend-candle`** — the candle/CUDA `candle_gen_depth` sibling (sc-8413), so
//!   `ControlKind::Depth` auto-estimates on Windows/Linux instead of erroring "macOS-only".
//!
//! Both load the **Small** variant (`depth-anything/Depth-Anything-V2-Small-hf`, apache-2.0,
//! ungated) — favoring speed/size for a preprocessing tier — from a snapshot dir holding
//! `model.safetensors`, and expose the same [`depth_control_image`] entry point.
//!
//! Output contract: an [`image::RgbImage`], the same type the pose preprocessor's
//! `draw_wholebody` and the canny preprocessor return and the `ControlKind::Depth` path
//! consumes — a single-channel depth map min/max-normalized to `[0,255]` and broadcast
//! across the three RGB channels (near = bright, the standard ControlNet depth
//! convention), at the input image's dimensions.
//!
//! The control-lane routing that drives [`depth_control_image`] off-Mac (the shared candle
//! strict-control driver) is the separate sc-8304; this module only provides the estimation entry
//! point so that PR can call it.

/// The Hugging Face repo for the default depth estimator: Depth Anything V2 **Small**
/// (apache-2.0, ungated — ships standard `model.safetensors`; no re-host needed). The
/// `-hf` (transformers) mirror is the safetensors-bearing one (the base
/// `depth-anything/Depth-Anything-V2-Small` ships only a `.pth`).
///
/// Always available (the strict-control weight-resolution + smoke tests reference it on every
/// platform); only the estimation function is backend-gated.
pub const DEPTH_ANYTHING_V2_SMALL_REPO: &str = "depth-anything/Depth-Anything-V2-Small-hf";

/// The single weight file the estimator loads from its snapshot dir. (Both the MLX and candle
/// loaders read every `*.safetensors` in the dir; the published Small checkpoint ships exactly
/// this one file.)
pub const DEPTH_ANYTHING_V2_FILE: &str = "model.safetensors";

/// Estimate a depth control image from an arbitrary RGB input, loading the Depth Anything V2
/// estimator from `weights_dir` (a directory containing `model.safetensors`).
///
/// `img` is the source RGB image; the returned [`image::RgbImage`] is the normalized
/// grayscale-broadcast depth map at the SAME dimensions, drop-in for the `ControlKind::Depth`
/// path (the sibling of `canny::canny_control_image_default`). macOS path: MLX inference.
#[cfg(target_os = "macos")]
pub fn depth_control_image(
    img: &image::RgbImage,
    weights_dir: &std::path::Path,
) -> crate::WorkerResult<image::RgbImage> {
    use crate::WorkerError;

    let model = mlx_gen_depth::DepthAnythingV2::from_dir(weights_dir)
        .map_err(|error| WorkerError::Engine(format!("depth estimator load: {error}")))?;
    let (w, h) = (img.width(), img.height());
    let control = model
        .estimate_control_rgb8(img.as_raw(), w, h)
        .map_err(|error| WorkerError::Engine(format!("depth estimate: {error}")))?;
    image::RgbImage::from_raw(w, h, control).ok_or_else(|| {
        WorkerError::Engine("depth estimator returned a mis-sized control buffer".to_owned())
    })
}

/// Off-Mac (Windows/Linux) candle/CUDA depth auto-estimation — the sibling of the macOS MLX path
/// above, backed by the from-scratch `candle_gen_depth` Depth Anything V2 port (sc-8413, the
/// Windows/CUDA twin of `mlx-gen-depth`).
///
/// Identical signature + contract to the macOS function: `img` is the source RGB image,
/// `weights_dir` is the Depth-Anything-V2-Small snapshot dir holding `model.safetensors`, and the
/// returned [`image::RgbImage`] is the normalized grayscale-broadcast depth control map at the SAME
/// dimensions. `DepthAnythingV2::from_dir` runs on the build's default candle device (CUDA when
/// built+available, else CPU); `estimate_control_rgb8` returns the `w·h·3` RGB control buffer.
///
/// Replaces the prior off-Mac "automatic depth estimation is macOS-only" error so the candle
/// strict-control driver (sc-8304) can auto-estimate `ControlKind::Depth`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub fn depth_control_image(
    img: &image::RgbImage,
    weights_dir: &std::path::Path,
) -> crate::WorkerResult<image::RgbImage> {
    use crate::WorkerError;

    let model = candle_gen_depth::DepthAnythingV2::from_dir(weights_dir)
        .map_err(|error| WorkerError::Engine(format!("depth estimator load: {error}")))?;
    let (w, h) = (img.width(), img.height());
    let control = model
        .estimate_control_rgb8(img.as_raw(), w, h)
        .map_err(|error| WorkerError::Engine(format!("depth estimate: {error}")))?;
    image::RgbImage::from_raw(w, h, control).ok_or_else(|| {
        WorkerError::Engine("depth estimator returned a mis-sized control buffer".to_owned())
    })
}

#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
mod candle_tests {
    //! Off-Mac candle depth-path tests (sc-8413).
    //!
    //! - [`constants_point_at_ungated_small`] always runs (no weights / no GPU): it pins the weight
    //!   source the strict-control resolver + this estimator agree on.
    //! - [`estimates_depth_control_at_input_dims`] is a real-weight smoke gated on
    //!   `SCENEWORKS_DEPTH_ANYTHING_V2` (the same env var the strict-control depth smokes use); it
    //!   skips when unset/missing so the candle-worker CI lane (which BUILDS the cuda feature but has
    //!   no GPU and no cached weights) stays green. `DepthAnythingV2::from_dir` falls back to the CPU
    //!   device when no GPU is present, so it CAN run in CI once the ~100 MB Small checkpoint is
    //!   cached and the env var is set; otherwise the coordinator's local CUDA build runs it.
    use super::*;

    #[test]
    fn constants_point_at_ungated_small() {
        assert_eq!(
            DEPTH_ANYTHING_V2_SMALL_REPO,
            "depth-anything/Depth-Anything-V2-Small-hf"
        );
        assert_eq!(DEPTH_ANYTHING_V2_FILE, "model.safetensors");
    }

    #[test]
    fn estimates_depth_control_at_input_dims() {
        let Ok(dir) = std::env::var("SCENEWORKS_DEPTH_ANYTHING_V2") else {
            eprintln!("skipping: set SCENEWORKS_DEPTH_ANYTHING_V2 to the Small snapshot dir");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        if !dir.join(DEPTH_ANYTHING_V2_FILE).is_file() {
            eprintln!(
                "skipping: {DEPTH_ANYTHING_V2_FILE} missing in {}",
                dir.display()
            );
            return;
        }
        // A small non-uniform RGB gradient input.
        let (w, h) = (64u32, 48u32);
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x * 4 % 256) as u8, (y * 5 % 256) as u8, 128])
        });
        let control = depth_control_image(&img, &dir).expect("candle depth estimate");
        assert_eq!(
            control.dimensions(),
            (w, h),
            "depth map must match input dims"
        );
        // Grayscale broadcast: every pixel's three channels are equal.
        assert!(
            control.pixels().all(|p| p[0] == p[1] && p[1] == p[2]),
            "depth control must be grayscale-broadcast across RGB"
        );
    }
}
