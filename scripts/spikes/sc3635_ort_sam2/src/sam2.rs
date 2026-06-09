//! sc-3635 — Rust SAM2.1 box-prompt person segmenter via `ort` (onnxruntime).
//! Loads the two-graph image-predictor split exported by onnx-community
//! (vision_encoder.onnx + prompt_encoder_mask_decoder.onnx, inlined to single files
//! by sc3635_reference.py) and reproduces the production `_Sam2Segmenter` contract
//! (apps/worker/scene_worker/person_adapters.py): box prompt -> single best mask
//! (argmax of IoU scores) -> binary "L" mask. Reports per-frame latency, the EP used,
//! and mask IoU vs the Python onnxruntime reference mask.
//!
//! Preprocessing (matches Sam2ImageProcessor / sc3635_reference.preprocess):
//!  - resize to 1024x1024 (square stretch, bilinear), /255, ImageNet normalize.
//! Box prompt: orig px -> 1024 space (independent x/y scale). One dummy padding point
//! (label -10, ignored by the prompt encoder) satisfies the graph's required point
//! inputs alongside the dedicated input_boxes path.

use anyhow::{anyhow, Result};
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::time::Instant;

const SIZE: usize = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn build_session(path: &Path, coreml: bool) -> Result<Session> {
    let mut b = Session::builder().map_err(|e| anyhow!(e.to_string()))?;
    if coreml {
        b = b
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| anyhow!(e.to_string()))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!(e.to_string()))
}

/// SAM2 preprocessing -> NCHW f32 [1,3,1024,1024].
fn preprocess(img: &image::RgbImage) -> Vec<f32> {
    let resized = image::imageops::resize(img, SIZE as u32, SIZE as u32, image::imageops::FilterType::Triangle);
    let mut data = vec![0f32; 3 * SIZE * SIZE];
    for c in 0..3 {
        let plane = c * SIZE * SIZE;
        for y in 0..SIZE {
            for x in 0..SIZE {
                let p = resized.get_pixel(x as u32, y as u32);
                data[plane + y * SIZE + x] = ((p[c] as f32 / 255.0) - MEAN[c]) / STD[c];
            }
        }
    }
    data
}

/// Bilinear upsample a [hm x wm] logit map to (out_w,out_h), threshold >0 -> {0,255}.
fn upsample_threshold(low: &[f32], wm: usize, hm: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_w * out_h];
    // align_corners=false mapping (matches torch F.interpolate default).
    let sx = wm as f32 / out_w as f32;
    let sy = hm as f32 / out_h as f32;
    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(hm - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(wm - 1);
            let wx = fx - x0 as f32;
            let v00 = low[y0 * wm + x0];
            let v01 = low[y0 * wm + x1];
            let v10 = low[y1 * wm + x0];
            let v11 = low[y1 * wm + x1];
            let top = v00 + (v01 - v00) * wx;
            let bot = v10 + (v11 - v10) * wx;
            let v = top + (bot - top) * wy;
            out[oy * out_w + ox] = if v > 0.0 { 255 } else { 0 };
        }
    }
    out
}

fn extract_f32(v: &ort::value::DynValue) -> Result<(Vec<i64>, Vec<f32>)> {
    let (shape, data) = v.try_extract_tensor::<f32>().map_err(|e| anyhow!(e.to_string()))?;
    Ok((shape.to_vec(), data.to_vec()))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut enc = String::from("/tmp/sc3635/onnx_inlined/vision_encoder.onnx");
    let mut dec = String::from("/tmp/sc3635/onnx_inlined/prompt_encoder_mask_decoder.onnx");
    let mut input = String::from("/tmp/sc3635/zidane/input.png");
    let mut reference = String::from("/tmp/sc3635/zidane/mask_ortcpu.png");
    let mut out = String::from("/tmp/sc3635/zidane/mask_rust.png");
    let mut box_xyxy = [710f32, 40.0, 1150.0, 715.0];
    let mut coreml = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--enc" => { enc = args[i + 1].clone(); i += 2; }
            "--dec" => { dec = args[i + 1].clone(); i += 2; }
            "--input" => { input = args[i + 1].clone(); i += 2; }
            "--reference" => { reference = args[i + 1].clone(); i += 2; }
            "--out" => { out = args[i + 1].clone(); i += 2; }
            "--device" => { coreml = args[i + 1] == "coreml"; i += 2; }
            "--box" => {
                let parts: Vec<f32> = args[i + 1].split(',').map(|s| s.parse().unwrap()).collect();
                box_xyxy = [parts[0], parts[1], parts[2], parts[3]];
                i += 2;
            }
            _ => i += 1,
        }
    }
    let device = if coreml { "coreml" } else { "cpu" };

    let img = image::open(&input)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let sx = SIZE as f32 / w as f32;
    let sy = SIZE as f32 / h as f32;
    let box1024 = [box_xyxy[0] * sx, box_xyxy[1] * sy, box_xyxy[2] * sx, box_xyxy[3] * sy];

    let t0 = Instant::now();
    let mut enc_sess = build_session(Path::new(&enc), coreml)?;
    let mut dec_sess = build_session(Path::new(&dec), false)?; // decoder always CPU (tiny, CoreML-hostile)
    let load_s = t0.elapsed().as_secs_f64();

    let pix = preprocess(&img);

    // ---- encoder ----
    let t1 = Instant::now();
    let pix_t = Tensor::from_array((vec![1i64, 3, SIZE as i64, SIZE as i64], pix))
        .map_err(|e| anyhow!(e.to_string()))?;
    let enc_out = enc_sess.run(ort::inputs![pix_t]).map_err(|e| anyhow!(e.to_string()))?;
    let (s0, e0) = extract_f32(&enc_out[0])?;
    let (s1, e1) = extract_f32(&enc_out[1])?;
    let (s2, e2) = extract_f32(&enc_out[2])?;
    let enc_s = t1.elapsed().as_secs_f64();

    // ---- decoder (positional order: points, labels, boxes, emb0, emb1, emb2) ----
    let t2 = Instant::now();
    let pts = Tensor::from_array((vec![1i64, 1, 1, 2], vec![0f32, 0.0])).map_err(|e| anyhow!(e.to_string()))?;
    let lbl = Tensor::from_array((vec![1i64, 1, 1], vec![-10i64])).map_err(|e| anyhow!(e.to_string()))?;
    let bx = Tensor::from_array((vec![1i64, 1, 4], box1024.to_vec())).map_err(|e| anyhow!(e.to_string()))?;
    let emb0 = Tensor::from_array((s0, e0)).map_err(|e| anyhow!(e.to_string()))?;
    let emb1 = Tensor::from_array((s1, e1)).map_err(|e| anyhow!(e.to_string()))?;
    let emb2 = Tensor::from_array((s2, e2)).map_err(|e| anyhow!(e.to_string()))?;
    let dec_out = dec_sess
        .run(ort::inputs![pts, lbl, bx, emb0, emb1, emb2])
        .map_err(|e| anyhow!(e.to_string()))?;
    let (iou_shape, iou_scores) = extract_f32(&dec_out[0])?; // [1,1,3]
    let (mask_shape, pred_masks) = extract_f32(&dec_out[1])?; // [1,1,3,Hm,Wm]
    let dec_s = t2.elapsed().as_secs_f64();

    let num_masks = iou_shape[iou_shape.len() - 1] as usize;
    let best = (0..num_masks)
        .max_by(|&a, &b| iou_scores[a].partial_cmp(&iou_scores[b]).unwrap())
        .unwrap();
    let hm = mask_shape[3] as usize;
    let wm = mask_shape[4] as usize;
    let plane = hm * wm;
    let low = &pred_masks[best * plane..(best + 1) * plane];

    let mask = upsample_threshold(low, wm, hm, w, h);
    let cov = 100.0 * mask.iter().filter(|&&v| v > 0).count() as f32 / mask.len() as f32;
    image::GrayImage::from_raw(w as u32, h as u32, mask.clone()).unwrap().save(&out)?;

    eprintln!(
        "[rust] device={device}(enc)/cpu(dec) load={load_s:.1}s enc={:.0}ms dec={:.0}ms total={:.0}ms \
         scores={:?} best={best} mask={wm}x{hm}->{w}x{h} cov={cov:.1}% -> {out}",
        enc_s * 1000.0, dec_s * 1000.0, (enc_s + dec_s) * 1000.0,
        iou_scores.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
    );

    if !reference.is_empty() && Path::new(&reference).exists() {
        let refimg = image::open(&reference)?.to_luma8();
        let rb = refimg.as_raw();
        let (mut inter, mut uni) = (0u64, 0u64);
        for k in 0..mask.len() {
            let a = mask[k] > 127;
            let b = rb[k] > 127;
            if a && b { inter += 1; }
            if a || b { uni += 1; }
        }
        let iou = if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
        eprintln!("[parity rust.{device}-vs-python.ortcpu] IoU={iou:.4}");
    }
    Ok(())
}
