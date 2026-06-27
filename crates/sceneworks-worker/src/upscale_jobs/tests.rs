//! Pure-math unit tests for the Real-ESRGAN tiling port (sc-3489). No onnx weights
//! needed — these lock the tiling/crop/manifest logic against `upscalers.py`. They run on
//! macOS and on the off-Mac candle lane (sc-5499), since the tiling path is now shared.

use super::*;
// `Rgb`/`RgbImage` back the Real-ESRGAN tiling/crop tests + the off-Mac ort smoke (the `ort` path,
// macOS + the candle lane) AND the SeedVR2 real-weight smoke (Mac MLX + the candle lane).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use image::{Rgb, RgbImage};
use serde_json::json;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn tile_slices_single_when_image_fits() {
    // tile >= max(w,h) → one tile covering the whole image (upscalers.py).
    let t = tile_slices(400, 300, 512);
    assert_eq!(
        t,
        vec![Tile {
            x0: 0,
            y0: 0,
            x1: 400,
            y1: 300
        }]
    );
    // exact-fit edge: tile == max dim still single (>= guard).
    assert_eq!(tile_slices(512, 512, 512).len(), 1);
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn tile_slices_grid_row_major_clamped() {
    // 768x768 @ tile 512 → 2x2 grid, edge tiles clamped to bounds.
    let t = tile_slices(768, 768, 512);
    assert_eq!(t.len(), 4);
    assert_eq!(
        t[0],
        Tile {
            x0: 0,
            y0: 0,
            x1: 512,
            y1: 512
        }
    );
    assert_eq!(
        t[1],
        Tile {
            x0: 512,
            y0: 0,
            x1: 768,
            y1: 512
        }
    );
    assert_eq!(
        t[2],
        Tile {
            x0: 0,
            y0: 512,
            x1: 512,
            y1: 768
        }
    );
    assert_eq!(
        t[3],
        Tile {
            x0: 512,
            y0: 512,
            x1: 768,
            y1: 768
        }
    );
    // full coverage, no gaps/overlaps in the (unpadded) inner grid
    let covered: usize = t.iter().map(|s| (s.x1 - s.x0) * (s.y1 - s.y0)).sum();
    assert_eq!(covered, 768 * 768);
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn tile_slices_zero_tile_is_single() {
    assert_eq!(tile_slices(100, 50, 0).len(), 1);
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn crop_to_chw_layout_and_normalization() {
    let mut img = RgbImage::new(3, 2);
    img.put_pixel(0, 0, Rgb([255, 0, 0]));
    img.put_pixel(1, 0, Rgb([0, 255, 0]));
    img.put_pixel(2, 0, Rgb([0, 0, 255]));
    img.put_pixel(0, 1, Rgb([0, 0, 0]));
    img.put_pixel(1, 1, Rgb([128, 128, 128]));
    img.put_pixel(2, 1, Rgb([255, 255, 255]));

    let (data, cw, ch) = crop_to_chw(&img, 0, 0, 3, 2);
    assert_eq!((cw, ch), (3, 2));
    assert_eq!(data.len(), 3 * 3 * 2);
    // CHW: R plane first. R[0,0]=1.0, G[0,0]=0.0, B[2,0]=1.0
    assert!((data[0] - 1.0).abs() < 1e-6); // R(0,0)
    let g_plane = cw * ch;
    assert!((data[g_plane + 1] - 1.0).abs() < 1e-6); // G(1,0)
    let b_plane = 2 * cw * ch;
    assert!((data[b_plane + 2] - 1.0).abs() < 1e-6); // B(2,0)
                                                     // mid-gray (1,1).G normalizes to 128/255
    assert!((data[g_plane + 4] - 128.0 / 255.0).abs() < 1e-6);
    assert!((data[g_plane - 1] - 1.0).abs() < 1e-6); // R(2,1)=255 last in R plane
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn crop_to_chw_subregion() {
    let mut img = RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, Rgb([(x * 10) as u8, (y * 10) as u8, 0]));
        }
    }
    let (data, cw, ch) = crop_to_chw(&img, 1, 1, 3, 3);
    assert_eq!((cw, ch), (2, 2));
    // R plane top-left = pixel (1,1).R = 10/255
    assert!((data[0] - 10.0 / 255.0).abs() < 1e-6);
    // G plane bottom-right = pixel (2,2).G = 20/255
    let g = cw * ch;
    assert!((data[g + 3] - 20.0 / 255.0).abs() < 1e-6);
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn pad_chw_even_replicates_to_even_dims_for_pixel_unshuffle() {
    // odd width (the 323px-wide regression): 3x2 CHW → padded to 4x2, the new column replicates
    // the last (no dark seam to bleed into the readable region via the x2 unshuffle conv).
    let (data, cw, ch) = crop_chw_const_gradient(3, 2);
    let (padded, pw, ph) = pad_chw_even(data.clone(), cw, ch);
    assert_eq!((pw, ph), (4, 2));
    assert_eq!(padded.len(), 3 * 4 * 2);
    for c in 0..3 {
        for y in 0..2 {
            // the appended column (x=3) equals the original last column (x=2)
            let last = data[c * ch * cw + y * cw + 2];
            let appended = padded[c * ph * pw + y * pw + 3];
            assert!((appended - last).abs() < 1e-6);
            // existing columns are preserved
            for x in 0..3 {
                assert!(
                    (padded[c * ph * pw + y * pw + x] - data[c * ch * cw + y * cw + x]).abs()
                        < 1e-6
                );
            }
        }
    }

    // odd height too → row replicated; even×even is returned untouched (no copy).
    let (d2, _, _) = crop_chw_const_gradient(2, 3);
    let (_, pw2, ph2) = pad_chw_even(d2, 2, 3);
    assert_eq!((pw2, ph2), (2, 4));
    let (d3, _, _) = crop_chw_const_gradient(4, 2);
    let (out3, pw3, ph3) = pad_chw_even(d3.clone(), 4, 2);
    assert_eq!((pw3, ph3), (4, 2));
    assert_eq!(out3, d3, "already-even dims are returned unchanged");
}

/// Tiny deterministic CHW buffer (each channel a distinct gradient) for the padding test.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn crop_chw_const_gradient(cw: usize, ch: usize) -> (Vec<f32>, usize, usize) {
    let mut data = vec![0.0f32; 3 * cw * ch];
    for c in 0..3 {
        for y in 0..ch {
            for x in 0..cw {
                data[c * ch * cw + y * cw + x] = (c * 100 + y * 10 + x) as f32 / 255.0;
            }
        }
    }
    (data, cw, ch)
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn onnx_filename_per_factor() {
    assert_eq!(onnx_file(2), "real_esrgan_x2.onnx");
    assert_eq!(onnx_file(4), "real_esrgan_x4.onnx");
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn manifest_onnx_resource_extracts_repo_file() {
    let entry = json!({
        "resources": {
            "imageUpscalers": {
                "real-esrgan": {
                    "x4": { "onnx": { "repo": "acme/esrgan-onnx", "file": "x4.onnx" } }
                }
            }
        }
    });
    assert_eq!(
        manifest_onnx_resource(&entry, 4),
        Some(("acme/esrgan-onnx".to_owned(), "x4.onnx".to_owned()))
    );
    // missing factor → None (falls back to default repo)
    assert_eq!(manifest_onnx_resource(&entry, 2), None);
    // file defaults to the conventional name when absent
    let no_file = json!({
        "resources": { "imageUpscalers": { "real-esrgan": { "x2": { "onnx": { "repo": "acme/e" } } } } }
    });
    assert_eq!(
        manifest_onnx_resource(&no_file, 2),
        Some(("acme/e".to_owned(), "real_esrgan_x2.onnx".to_owned()))
    );
    assert_eq!(manifest_onnx_resource(&Value::Null, 4), None);
}

// ---------------------------------------------------------------------------
// SeedVR2 (sc-4815): pure helpers + a gated real-weight integration smoke
// ---------------------------------------------------------------------------

#[test]
fn round_to_16_rounds_up_floored_at_16() {
    assert_eq!(round_to_16(96), 96); // already a multiple
    assert_eq!(round_to_16(64), 64);
    assert_eq!(round_to_16(100), 112); // rounds up to the next multiple of 16
    assert_eq!(round_to_16(17), 32);
    assert_eq!(round_to_16(1), 16); // floored at 16
    assert_eq!(round_to_16(0), 16);
}

#[test]
fn upscale_target_dimensions_are_bounded_before_allocation() {
    validate_upscale_target_dimensions(8192, 8192).expect("8k square accepted");
    assert!(matches!(
        validate_upscale_target_dimensions(8193, 1024),
        Err(WorkerError::InvalidPayload(_))
    ));
    assert!(matches!(
        validate_upscale_target_dimensions(8192, 8193),
        Err(WorkerError::InvalidPayload(_))
    ));
}

#[test]
fn manifest_seedvr2_resource_extracts_overrides_and_defaults() {
    let entry = json!({
        "resources": {
            "imageUpscalers": {
                "seedvr2": { "repo": "acme/seedvr2", "ditFile": "dit.safetensors", "vaeFile": "vae.safetensors" }
            }
        }
    });
    assert_eq!(
        manifest_seedvr2_resource(&entry),
        Some((
            "acme/seedvr2".to_owned(),
            "dit.safetensors".to_owned(),
            "vae.safetensors".to_owned()
        ))
    );
    // only `repo` → the DiT/VAE filenames default to the canonical names the engine loads.
    let repo_only = json!({
        "resources": { "imageUpscalers": { "seedvr2": { "repo": "acme/s" } } }
    });
    assert_eq!(
        manifest_seedvr2_resource(&repo_only),
        Some((
            "acme/s".to_owned(),
            SEEDVR2_DIT_FILE.to_owned(),
            SEEDVR2_VAE_FILE.to_owned()
        ))
    );
    assert_eq!(manifest_seedvr2_resource(&Value::Null), None);
}

/// Resolve the locally-cached `numz/SeedVR2_comfyUI` checkpoint dir (env override or the HF cache),
/// so the smoke below can run on real weights without a download. `None` ⇒ skip.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn cached_seedvr2_checkpoint() -> Option<std::path::PathBuf> {
    if let Ok(pinned) = std::env::var("SCENEWORKS_SEEDVR2_CHECKPOINT") {
        let dir = std::path::PathBuf::from(pinned);
        if dir.join(SEEDVR2_DIT_FILE).exists() && dir.join(SEEDVR2_VAE_FILE).exists() {
            return Some(dir);
        }
    }
    let base = std::path::Path::new(&std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    (snap.join(SEEDVR2_DIT_FILE).exists() && snap.join(SEEDVR2_VAE_FILE).exists()).then_some(snap)
}

/// Real-weight smoke for the SceneWorks SeedVR2 integration (sc-4815 Mac / sc-5928 candle): drives the
/// exact worker dispatch path — `with_cached_generator("seedvr2", …)` → registry → `generate` — on the
/// cached 3B checkpoint, asserting (a) the factor→`round_to_16` target dims and (b) that the `softness`
/// request field actually reaches the engine (a softened run differs from a faithful one). On Mac this
/// resolves to the MLX provider; on the Windows/CUDA candle build it resolves to `candle-gen-seedvr2`
/// (the sc-5928 worker-path validation that the force-link + routing reach the candle engine end-to-
/// end). Gated on the checkpoint being present (skips in CI, which has no weights), mirroring the
/// family worker E2E smokes. Set `SCENEWORKS_SEEDVR2_CHECKPOINT` to the ckpt dir and run with
/// `cargo test -p sceneworks-worker --features backend-candle -- --ignored seedvr2_upscale_real_weight_smoke`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[tokio::test]
#[ignore = "real-weight: needs the cached numz/SeedVR2_comfyUI checkpoint (~7 GB) + the seedvr2 backend (MLX on Mac / candle on Windows)"]
async fn seedvr2_upscale_real_weight_smoke() {
    let Some(dir) = cached_seedvr2_checkpoint() else {
        eprintln!("SKIP: SeedVR2 checkpoint not cached (numz/SeedVR2_comfyUI)");
        return;
    };
    // 48x32 deterministic gradient → factor 2 → 96x64 (both already multiples of 16).
    let mut img = RgbImage::new(48, 32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = Rgb([(x * 5) as u8, (y * 7) as u8, ((x + y) * 3 % 256) as u8]);
    }
    let faithful = run_seedvr2_upscale(dir.clone(), img.clone(), 2, 0.0, 7, CancelFlag::new())
        .await
        .expect("seedvr2 upscale (softness 0)");
    assert_eq!((faithful.width(), faithful.height()), (96, 64));

    // The softness request field must reach the engine: a heavily-softened run changes the result.
    let softened = run_seedvr2_upscale(dir, img, 2, 0.8, 7, CancelFlag::new())
        .await
        .expect("seedvr2 upscale (softness 0.8)");
    assert_eq!((softened.width(), softened.height()), (96, 64));
    assert_ne!(
        faithful.as_raw(),
        softened.as_raw(),
        "softness must change the output (the request field is wired to the engine)"
    );
}

// ---------------------------------------------------------------------------
// Real-ESRGAN off-Mac (sc-5499): a gated real-weight smoke on the candle lane's `ort`/CUDA path
// ---------------------------------------------------------------------------

/// Real-weight smoke for the off-Mac Real-ESRGAN `ort` path (sc-5499): drives the exact worker seam
/// `upscale_blocking` → `Upscaler::load` (CUDA EP, CPU fallback) → tiled `Upscaler::upscale` on a real
/// exported `real_esrgan_x{2,4}.onnx`, asserting the output is exactly `factor×` the input and is a
/// real super-resolution (not an all-zero / identity buffer). Gated on the ONNX being present (skips in
/// CI, which has no weights), like the SeedVR2 smoke. Point `SCENEWORKS_REALESRGAN_X{factor}_ONNX` at a
/// local export (optionally `SCENEWORKS_TEST_UPSCALE_IMAGE` at a source image; else a synthetic gradient
/// is used) and run with `cargo test -p sceneworks-worker --features backend-candle --lib -- --ignored
/// real_esrgan_candle_real_weights_upscales`. `ORT_DYLIB_PATH` must point at an onnxruntime build.
#[cfg(not(target_os = "macos"))]
#[test]
#[ignore = "real-weight: needs an exported real_esrgan ONNX (SCENEWORKS_REALESRGAN_X{factor}_ONNX) + an onnxruntime (ORT_DYLIB_PATH)"]
fn real_esrgan_candle_real_weights_upscales() {
    let factor: u8 = match std::env::var("SCENEWORKS_TEST_UPSCALE_FACTOR")
        .ok()
        .as_deref()
    {
        Some("4") => 4,
        _ => 2,
    };
    let Some(onnx) = ["SCENEWORKS_REALESRGAN_X".to_owned() + &factor.to_string() + "_ONNX"]
        .into_iter()
        .chain(["SCENEWORKS_REALESRGAN_ONNX".to_owned()])
        .find_map(|key| std::env::var(&key).ok().map(std::path::PathBuf::from))
        .filter(|p| p.exists())
    else {
        eprintln!("SKIP: no Real-ESRGAN ONNX (set SCENEWORKS_REALESRGAN_X{factor}_ONNX)");
        return;
    };

    // A real photo if provided, else a deterministic gradient (Real-ESRGAN is a deterministic conv,
    // so a synthetic input still exercises the full graph + tiling end-to-end).
    let src = match std::env::var("SCENEWORKS_TEST_UPSCALE_IMAGE").ok() {
        Some(path) => image::open(&path)
            .unwrap_or_else(|e| panic!("load SCENEWORKS_TEST_UPSCALE_IMAGE {path}: {e}"))
            .to_rgb8(),
        None => {
            let mut img = RgbImage::new(64, 48);
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                *pixel = Rgb([(x * 3) as u8, (y * 5) as u8, ((x + y) * 2 % 256) as u8]);
            }
            img
        }
    };
    let (sw, sh) = (src.width(), src.height());

    let out = upscale_blocking(onnx, factor, src, CancelFlag::new()).expect("Real-ESRGAN upscale");
    assert_eq!(
        (out.width(), out.height()),
        (sw * u32::from(factor), sh * u32::from(factor)),
        "output must be exactly factor× the source"
    );
    // not an all-zero buffer (the graph actually ran + wrote pixels).
    assert!(
        out.as_raw().iter().any(|&b| b != 0),
        "upscaled image must not be all-black"
    );
    eprintln!(
        "Real-ESRGAN x{factor}: {sw}x{sh} -> {}x{}",
        out.width(),
        out.height()
    );
}

// --- Dataset Doctor one-tap upscale plumbing (sc-6539), pure + weight-free ---

#[test]
fn parse_dataset_upscale_items_reads_valid_entries_and_skips_malformed() {
    let payload = json!({
        "items": [
            { "itemId": "a", "imagePath": "/data/a.png", "assetId": "asset_a" },
            { "itemId": "", "imagePath": "/data/blank.png" }, // empty id → skipped
            { "itemId": "c" },                                // no imagePath → skipped
            { "itemId": "d", "imagePath": "/data/d.jpg" },
        ],
    });
    let items = parse_dataset_upscale_items(payload.as_object().unwrap());
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].item_id, "a");
    assert_eq!(items[0].image_path, std::path::PathBuf::from("/data/a.png"));
    assert_eq!(items[1].item_id, "d");
}

#[test]
fn parse_dataset_upscale_items_is_empty_without_an_items_array() {
    assert!(parse_dataset_upscale_items(json!({}).as_object().unwrap()).is_empty());
}

#[test]
fn dataset_repoint_body_maps_records_with_a_null_asset_id() {
    let body = dataset_repoint_body(&[
        (
            "a".to_owned(),
            "training/datasets/ds/upscaled/a.png".to_owned(),
        ),
        (
            "b".to_owned(),
            "training/datasets/ds/upscaled/b.png".to_owned(),
        ),
    ]);
    let items = body
        .get("items")
        .and_then(serde_json::Value::as_array)
        .expect("items array");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["itemId"], "a");
    assert_eq!(
        items[0]["sourcePath"],
        "training/datasets/ds/upscaled/a.png"
    );
    assert!(
        items[0]["assetId"].is_null(),
        "dataset fix re-points to bytes, not a minted child asset"
    );
}
