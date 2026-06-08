//! sc-3489 — Rust Real-ESRGAN (RRDBNet) upscaler via `ort` (onnxruntime), faithful
//! port of apps/worker/scene_worker/upscalers.py tiling + inference. Loads the ONNX
//! exported by sc3489_export_reference.py (the SAME nateraw weights the torch path
//! ships) and renders a tiled x2/x4 upscale, then reports pixel parity vs the torch
//! reference PNG. onnxruntime runs on CoreML or CPU.
//!
//! Tiling parity (matched to upscalers.py:_run_tiled):
//!  - tile grid: tile_slices(w,h,512); per-tile crop padded by tile_pad=16 (clamped
//!    to image bounds); inner region (unpadded) copied back at factor scale.
//!  - input: RGB f32 CHW in [0,1]; output: clamp[0,1] -> round -> u8.

use anyhow::{anyhow, Result};
use image::{RgbImage};
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::time::Instant;

const TILE_SIZE: usize = 512;
const TILE_PAD: usize = 16;

struct Tile {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

fn tile_slices(w: usize, h: usize, tile: usize) -> Vec<Tile> {
    if tile == 0 || tile >= w.max(h) {
        return vec![Tile { x0: 0, y0: 0, x1: w, y1: h }];
    }
    let mut out = Vec::new();
    let mut y0 = 0;
    while y0 < h {
        let mut x0 = 0;
        while x0 < w {
            out.push(Tile {
                x0,
                y0,
                x1: (x0 + tile).min(w),
                y1: (y0 + tile).min(h),
            });
            x0 += tile;
        }
        y0 += tile;
    }
    out
}

fn build_session(path: &Path, coreml: bool) -> Result<Session> {
    let mut b = Session::builder().map_err(|e| anyhow!(e.to_string()))?;
    if coreml {
        b = b
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| anyhow!(e.to_string()))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!(e.to_string()))
}

/// CHW f32 [0,1] from an RGB sub-image (crop region [x0,x1) x [y0,y1)).
fn crop_to_chw(img: &RgbImage, x0: usize, y0: usize, x1: usize, y1: usize) -> (Vec<f32>, usize, usize) {
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut data = vec![0.0f32; 3 * ch * cw];
    for c in 0..3 {
        let plane = c * ch * cw;
        for yy in 0..ch {
            for xx in 0..cw {
                let p = img.get_pixel((x0 + xx) as u32, (y0 + yy) as u32);
                data[plane + yy * cw + xx] = p[c] as f32 / 255.0;
            }
        }
    }
    (data, cw, ch)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut input = String::new();
    let mut out = PathBuf::from("/tmp/sc3489/rust_out.png");
    let mut reference = String::new();
    let mut factor = 4usize;
    let mut coreml = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = args[i + 1].clone(); i += 2; }
            "--input" => { input = args[i + 1].clone(); i += 2; }
            "--out" => { out = PathBuf::from(&args[i + 1]); i += 2; }
            "--reference" => { reference = args[i + 1].clone(); i += 2; }
            "--factor" => { factor = args[i + 1].parse().unwrap(); i += 2; }
            "--device" => { coreml = args[i + 1] == "coreml"; i += 2; }
            _ => i += 1,
        }
    }
    let device = if coreml { "coreml" } else { "cpu" };

    let t0 = Instant::now();
    let mut sess = build_session(Path::new(&model), coreml)?;
    let load_s = t0.elapsed().as_secs_f64();

    let img = image::open(&input)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (ow, oh) = (w * factor, h * factor);
    // output RGB buffer
    let mut output = vec![0u8; ow * oh * 3];

    let tiles = tile_slices(w, h, TILE_SIZE);
    let t1 = Instant::now();
    let mut tile_ms_total = 0.0f64;
    for (ti, tl) in tiles.iter().enumerate() {
        let cx0 = tl.x0.saturating_sub(TILE_PAD);
        let cy0 = tl.y0.saturating_sub(TILE_PAD);
        let cx1 = (tl.x1 + TILE_PAD).min(w);
        let cy1 = (tl.y1 + TILE_PAD).min(h);
        let (data, cw, ch) = crop_to_chw(&img, cx0, cy0, cx1, cy1);

        let tt = Instant::now();
        let tensor = Tensor::from_array((vec![1i64, 3, ch as i64, cw as i64], data))
            .map_err(|e| anyhow!(e.to_string()))?;
        let outputs = sess.run(ort::inputs![tensor]).map_err(|e| anyhow!(e.to_string()))?;
        let (oshape, odata) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow!(e.to_string()))?;
        let dt = tt.elapsed().as_secs_f64() * 1000.0;
        tile_ms_total += dt;

        let och = oshape[2] as usize; // ch*factor
        let ocw = oshape[3] as usize; // cw*factor
        // inner (unpadded) region within the tile's upscaled crop
        let ix0 = (tl.x0 - cx0) * factor;
        let iy0 = (tl.y0 - cy0) * factor;
        let iw = (tl.x1 - tl.x0) * factor;
        let ih = (tl.y1 - tl.y0) * factor;
        let dst_x0 = tl.x0 * factor;
        let dst_y0 = tl.y0 * factor;
        for yy in 0..ih {
            for xx in 0..iw {
                let sy = iy0 + yy;
                let sx = ix0 + xx;
                let dy = dst_y0 + yy;
                let dx = dst_x0 + xx;
                let di = (dy * ow + dx) * 3;
                for c in 0..3 {
                    let v = odata[c * och * ocw + sy * ocw + sx];
                    output[di + c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        if ti == 0 {
            eprintln!("[rust] tile0 {cw}x{ch} -> {ocw}x{och} ({dt:.0}ms, first incl. graph compile)");
        }
    }
    let infer_s = t1.elapsed().as_secs_f64();

    let out_img = RgbImage::from_raw(ow as u32, oh as u32, output).unwrap();
    out_img.save(&out)?;
    eprintln!(
        "[rust] device={device} load={load_s:.1}s {} tiles infer={infer_s:.2}s (sum tile {tile_ms_total:.0}ms) -> {ow}x{oh} {}",
        tiles.len(),
        out.display()
    );

    if !reference.is_empty() {
        let refimg = image::open(&reference)?.to_rgb8();
        assert_eq!((refimg.width(), refimg.height()), (ow as u32, oh as u32), "ref size mismatch");
        let a = out_img.as_raw();
        let b = refimg.as_raw();
        let mut maxd = 0i32;
        let mut sumsq = 0.0f64;
        let mut sumabs = 0.0f64;
        for k in 0..a.len() {
            let d = (a[k] as i32 - b[k] as i32).abs();
            if d > maxd { maxd = d; }
            sumsq += (d * d) as f64;
            sumabs += d as f64;
        }
        let n = a.len() as f64;
        let mse = sumsq / n;
        let psnr = if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
        eprintln!(
            "[parity rust.{device}-vs-torch] max|Δ|={maxd} mean|Δ|={:.5} PSNR={psnr:.2}dB",
            sumabs / n
        );
    }
    Ok(())
}
