//! Local real-weight GPU smoke for the candle RealVisXL Lightning worker lane (sc-7176, the worker
//! half of sc-6128). `#[ignore]`d — run by hand on the RTX PRO 6000. It drives the real candle SDXL
//! engine via `gen_core::load("sdxl")` — the same runtime seam `generate_candle_stream` uses, minus
//! the API/job plumbing — with the **forced** few-step `lightning` (Euler-trailing, CFG-off) sampler
//! the worker pins for `realvisxl_lightning`, against the distilled RealVisXL_V5.0_Lightning weights.
//! This is the end-to-end worker-lane validation that backs the macOnly drop.
//!
//! Setup (PowerShell; the diffusers fp16 components must be in the HF cache — download via the Model
//! Manager / the manifest `realvisxl_lightning` entry):
//! ```text
//! # the snapshot dir holding model_index.json + unet/ text_encoder/ ... *.fp16.safetensors
//! $env:REALVISXL_LIGHTNING_DIR="C:\Users\Michael\.cache\huggingface\hub\models--SG161222--RealVisXL_V5.0_Lightning\snapshots\<hash>"
//! $env:RVXL_OUT_DIR="D:\sceneworks-sampler-validate\rvxl-lightning"
//! # optional: RVXL_STEPS=5 RVXL_W=1024 RVXL_H=1024 RVXL_PROMPT="..." RVXL_CONTRAST=1 (also render ddim@steps)
//! cargo test -p sceneworks-worker --features backend-candle --release realvisxl_lightning_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};

fn env_path(key: &str) -> PathBuf {
    // Trim: a cmd `set VAR=value && ...` keeps the trailing space before `&&`.
    PathBuf::from(
        std::env::var(key)
            .unwrap_or_else(|_| panic!("set ${key}"))
            .trim(),
    )
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|_| default.to_string())
}

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check (an
/// all-black / NaN-clamped decode collapses toward 0; a noisy wrong-schedule render is HIGH, so this
/// only guards the degenerate floor — the real lightning-vs-ddim quality call is the saved-PNG eyeball).
fn image_std(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt()
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("rgb buffer")
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

fn render(
    weights_dir: &Path,
    prompt: &str,
    w: u32,
    h: u32,
    steps: u32,
    guidance: f32,
    sampler: Option<&str>,
) -> Image {
    // Same seam as `generate_candle_stream`: registry load of the candle `sdxl` engine + a dense
    // (no-quant, no-adapter) LoadSpec pointed at the RealVisXL Lightning diffusers components.
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.to_path_buf()));
    let generator = gen_core::load("sdxl", &spec).expect("load candle sdxl provider");
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(guidance),
        sampler: sampler.map(str::to_owned),
        ..Default::default()
    };
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress {}] {s}", sampler.unwrap_or("default"));
                last = s;
            }
        })
        .expect("sdxl generate");
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

#[test]
#[ignore = "real-weight GPU smoke; needs the candle RealVisXL_V5.0_Lightning diffusers snapshot"]
fn realvisxl_lightning_candle_gpu_smoke() {
    let weights_dir = env_path("REALVISXL_LIGHTNING_DIR");
    assert!(
        weights_dir.join("model_index.json").is_file(),
        "REALVISXL_LIGHTNING_DIR must point at the diffusers snapshot (model_index.json missing): {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("RVXL_OUT_DIR", "rvxl-lightning-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("RVXL_STEPS", "5").parse().expect("RVXL_STEPS");
    let w: u32 = env_or("RVXL_W", "1024").parse().expect("RVXL_W");
    let h: u32 = env_or("RVXL_H", "1024").parse().expect("RVXL_H");
    let prompt = env_or(
        "RVXL_PROMPT",
        "a photorealistic portrait of a red fox in a snowy forest, golden hour, sharp focus",
    );

    // The shipped behavior: realvisxl_lightning forces the few-step `lightning` sampler, CFG-off
    // (guidance 1.0 — the distilled checkpoint is trained CFG-free).
    println!("[smoke] rendering lightning @ {steps} steps ({w}x{h}) ...");
    let lightning = render(&weights_dir, &prompt, w, h, steps, 1.0, Some("lightning"));
    let lightning_std = image_std(&lightning);
    save_png(
        &lightning,
        &out_dir.join(format!("lightning_{steps}step.png")),
    );
    println!(
        "[smoke] lightning {}x{} std {:.2} -> {}",
        lightning.width,
        lightning.height,
        lightning_std,
        out_dir.join(format!("lightning_{steps}step.png")).display()
    );

    // Optional contrast: the SAME checkpoint on the default `ddim` schedule (real CFG) at the same low
    // step count is visibly under-denoised — the wrong-schedule result the forced lightning sampler
    // avoids. ddim needs real CFG (guidance > 1; the lightning path is CFG-off, ddim+1.0 is invalid),
    // so render it at the base SDXL default 7.5. Saved for the eyeball only.
    if env_or("RVXL_CONTRAST", "0") == "1" {
        println!("[smoke] rendering ddim @ {steps} steps (contrast) ...");
        let ddim = render(&weights_dir, &prompt, w, h, steps, 7.5, Some("ddim"));
        save_png(&ddim, &out_dir.join(format!("ddim_{steps}step.png")));
        println!(
            "[smoke] ddim contrast std {:.2} -> {}",
            image_std(&ddim),
            out_dir.join(format!("ddim_{steps}step.png")).display()
        );
    }

    assert!(
        lightning_std > 5.0,
        "lightning render looks degenerate (std {lightning_std:.2}) — possible NaN / all-black decode"
    );
    println!("[smoke] DONE: lightning render coherent at {steps} steps");
}
