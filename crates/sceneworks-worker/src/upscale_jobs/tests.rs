//! Pure-math unit tests for the Real-ESRGAN tiling port (sc-3489). No onnx weights
//! needed — these lock the tiling/crop/manifest logic against `upscalers.py`.

use super::*;
use image::{Rgb, RgbImage};
use serde_json::json;

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

#[test]
fn tile_slices_zero_tile_is_single() {
    assert_eq!(tile_slices(100, 50, 0).len(), 1);
}

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

#[test]
fn onnx_filename_per_factor() {
    assert_eq!(onnx_file(2), "real_esrgan_x2.onnx");
    assert_eq!(onnx_file(4), "real_esrgan_x4.onnx");
}

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
