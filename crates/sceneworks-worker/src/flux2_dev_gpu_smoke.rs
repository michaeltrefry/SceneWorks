//! Local real-weight GPU smoke for the candle FLUX.2-dev worker lane (epic 6564 sc-7458, the worker
//! half of the sc-7457 provider). `#[ignore]`d — run by hand on the RTX PRO 6000. It drives the real
//! candle FLUX.2-dev engine via `gen_core::load("flux2_dev")` with a **Q4** `LoadSpec` — the exact
//! runtime seam `generate_candle_stream` uses (minus the API/job plumbing) once the router routes
//! `flux2_dev` to candle off-Mac. The 32B doesn't fit the GPU dense, so the dev quant path stages the
//! dense diffusers snapshot in system RAM and quantizes each projection onto the GPU at load
//! (candle-gen-flux2 sc-7457) — this is the end-to-end worker-lane validation backing the routing wire.
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120): the cap=80 PTX baseline JIT-no-ops
//! candle's CUDA quantized matmul on sm_120 (sc-7457 black-image root cause → sc-7544 packaging).
//!
//! Setup (PowerShell; point at the dense FLUX.2-dev diffusers snapshot — `transformer/ text_encoder/
//! vae/ tokenizer/ model_index.json`):
//! ```text
//! $env:FLUX2_DEV_DIR="D:\models\FLUX.2-dev"
//! $env:FLUX2_DEV_OUT_DIR="D:\sceneworks-sampler-validate\flux2-dev"
//! # optional: FLUX2_DEV_STEPS=28 FLUX2_DEV_W=1024 FLUX2_DEV_H=1024 FLUX2_DEV_GUIDANCE=4.0
//! #           FLUX2_DEV_QUANT=q4|q8  FLUX2_DEV_PROMPT="..."
//! cargo test -p sceneworks-worker --features backend-candle --release flux2_dev_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};

/// A synthetic RGB test image (a smooth diagonal gradient with a centered block) — enough to exercise
/// the VAE encode + reference/control token path on real weights without shipping a fixture. The dev
/// edit / control GPU validation behind the *quality* call lives in `candle-gen-flux2`'s `flux2-edit` /
/// `flux2-control` examples (sc-7460, real refs + a synthetic OpenPose skeleton); these worker smokes
/// prove the worker links + drives the bespoke `Flux2Edit::load_dev` / `Flux2Control::load` providers.
fn synthetic_rgb(w: u32, h: u32) -> Image {
    let (wu, hu) = (w as usize, h as usize);
    let mut pixels = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        for x in 0..wu {
            let i = (y * wu + x) * 3;
            pixels[i] = ((x * 255) / wu.max(1)) as u8;
            pixels[i + 1] = ((y * 255) / hu.max(1)) as u8;
            let centered = x > wu / 3 && x < 2 * wu / 3 && y > hu / 3 && y < 2 * hu / 3;
            pixels[i + 2] = if centered { 220 } else { 40 };
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

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

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check. The
/// sc-7457 dev-quant bug (CUDA-Q4 matmul no-op at cap=80) produced an all-black decode whose std
/// collapses toward 0; this guards that degenerate floor (the real quality call is the saved-PNG eyeball).
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

#[test]
#[ignore = "real-weight GPU smoke; needs the dense FLUX.2-dev diffusers snapshot + a CUDA device (cap=120)"]
fn flux2_dev_candle_gpu_smoke() {
    let weights_dir = env_path("FLUX2_DEV_DIR");
    assert!(
        weights_dir.join("model_index.json").is_file(),
        "FLUX2_DEV_DIR must point at the dense FLUX.2-dev diffusers snapshot (model_index.json missing): {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("FLUX2_DEV_OUT_DIR", "flux2-dev-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("FLUX2_DEV_STEPS", "28")
        .parse()
        .expect("FLUX2_DEV_STEPS");
    let w: u32 = env_or("FLUX2_DEV_W", "1024").parse().expect("FLUX2_DEV_W");
    let h: u32 = env_or("FLUX2_DEV_H", "1024").parse().expect("FLUX2_DEV_H");
    let guidance: f32 = env_or("FLUX2_DEV_GUIDANCE", "4.0")
        .parse()
        .expect("FLUX2_DEV_GUIDANCE");
    // The shipped worker forces Q4 (manifest `mlx.quantize: 4` → `resolve_quant`); allow Q8 for an
    // optional contrast run. dev advertises only Q4/Q8 — dense doesn't fit the GPU.
    let quant = match env_or("FLUX2_DEV_QUANT", "q4").as_str() {
        "q8" | "Q8" => Quant::Q8,
        _ => Quant::Q4,
    };
    let prompt = env_or(
        "FLUX2_DEV_PROMPT",
        "a rusty robot holding a lit candle in a dark workshop, cinematic, sharp focus",
    );

    // Same seam as `generate_candle_stream`: a registry load of the candle `flux2_dev` engine + a
    // Q4 `LoadSpec` pointed at the dense diffusers snapshot. The dev quant path CPU-stages the dense
    // weights and quantizes each projection onto the GPU — the 32B never lands on the GPU dense.
    println!(
        "[smoke] loading flux2_dev ({quant:?}) from {} ...",
        weights_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.clone())).with_quant(quant);
    let generator = gen_core::load("flux2_dev", &spec).expect("load candle flux2_dev provider");

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        // dev is guidance-distilled (embedded scalar, single forward — no negative pass).
        guidance: Some(guidance),
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps, guidance {guidance} ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("flux2_dev generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let std = image_std(&image);
    let png = out_dir.join(format!(
        "flux2_dev_{}_{steps}step.png",
        env_or("FLUX2_DEV_QUANT", "q4")
    ));
    save_png(&image, &png);
    println!(
        "[smoke] flux2_dev {}x{} std {:.2} -> {}",
        image.width,
        image.height,
        std,
        png.display()
    );
    assert_eq!((image.width, image.height), (w, h));
    assert!(
        std > 5.0,
        "flux2_dev render looks degenerate (std {std:.2}) — possible NaN / all-black decode \
         (check CUDA_COMPUTE_CAP=120: cap=80 JIT-no-ops the quantized matmul on sm_120, sc-7457)"
    );
    println!("[smoke] DONE: flux2_dev {quant:?} render coherent at {steps} steps");
}

/// Real-weight GPU smoke for the candle FLUX.2-dev **edit** worker lane (sc-7736) — drives the bespoke
/// `candle_gen_flux2::Flux2Edit::load_dev` provider (Q4 CPU-stage → quantize-onto-GPU) the worker's
/// `generate_candle_flux2_edit_stream` loads, with a reference (env `FLUX2_DEV_REF`, else a synthetic
/// image). Embedded distilled guidance, single forward (no negative pass). Proves the worker links + runs
/// the dev edit provider end-to-end; the quality A/B is the engine's `flux2-edit --variant dev` example.
///
/// ```text
/// $env:FLUX2_DEV_DIR="D:\models\FLUX.2-dev"; $env:FLUX2_DEV_REF="D:\models\FLUX.2-dev\teaser_generation.png"
/// cargo test -p sceneworks-worker --features backend-candle --release flux2_dev_edit_candle_gpu_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "real-weight GPU smoke; needs the dense FLUX.2-dev diffusers snapshot + a CUDA device (cap=120)"]
fn flux2_dev_edit_candle_gpu_smoke() {
    use candle_gen_flux2::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};

    let weights_dir = env_path("FLUX2_DEV_DIR");
    assert!(
        weights_dir.join("model_index.json").is_file(),
        "FLUX2_DEV_DIR must point at the dense FLUX.2-dev diffusers snapshot: {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("FLUX2_DEV_OUT_DIR", "flux2-dev-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let (w, h) = (
        env_or("FLUX2_DEV_W", "512").parse().expect("FLUX2_DEV_W"),
        env_or("FLUX2_DEV_H", "512").parse().expect("FLUX2_DEV_H"),
    );
    let steps: usize = env_or("FLUX2_DEV_STEPS", "8")
        .parse()
        .expect("FLUX2_DEV_STEPS");

    let reference = match std::env::var("FLUX2_DEV_REF") {
        Ok(p) if !p.trim().is_empty() => {
            let img = image::open(p.trim()).expect("open FLUX2_DEV_REF").to_rgb8();
            Image {
                width: img.width(),
                height: img.height(),
                pixels: img.into_raw(),
            }
        }
        _ => synthetic_rgb(w, h),
    };

    println!("[smoke] loading flux2_dev EDIT (Q4) from {} ...", weights_dir.display());
    let model = Flux2Edit::load_dev(
        &Flux2EditPaths {
            root: weights_dir.clone(),
        },
        Some(Quant::Q4),
    )
    .expect("load candle Flux2Edit dev");

    let req = Flux2EditRequest {
        prompt: env_or("FLUX2_DEV_PROMPT", "make the person wear a bright red wizard hat"),
        negative: String::new(),
        width: w,
        height: h,
        steps,
        guidance: 4.0,
        seed: 42,
        cancel: gen_core::runtime::CancelFlag::new(),
    };
    println!("[smoke] dev edit {w}x{h} @ {steps} steps (single ref) ...");
    let image = model
        .generate(&req, std::slice::from_ref(&reference), &mut |_| {})
        .expect("flux2_dev edit generate");
    let std = image_std(&image);
    let png = out_dir.join("flux2_dev_edit_candle.png");
    save_png(&image, &png);
    println!("[smoke] dev edit {}x{} std {:.2} -> {}", image.width, image.height, std, png.display());
    assert_eq!((image.width, image.height), (w, h));
    assert!(std > 5.0, "dev edit render degenerate (std {std:.2}) — check CUDA_COMPUTE_CAP=120");
    println!("[smoke] DONE: flux2_dev edit (candle) coherent");
}

/// Real-weight GPU smoke for the candle FLUX.2-dev **strict-pose control** worker lane (sc-7736) — drives
/// the bespoke `candle_gen_flux2::Flux2Control::load` provider the worker's
/// `generate_candle_flux2_control_stream` loads, with a synthetic control image and the Fun-Controlnet-
/// Union checkpoint (env `FLUX2_CONTROL`). Proves the worker links + runs the dev control provider; the
/// pose-conditioning A/B (scale 0 vs 0.75) is the engine's `flux2-control` example (sc-7460).
///
/// ```text
/// $env:FLUX2_DEV_DIR="D:\models\FLUX.2-dev"
/// $env:FLUX2_CONTROL="D:\models\FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"
/// cargo test -p sceneworks-worker --features backend-candle --release flux2_dev_control_candle_gpu_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "real-weight GPU smoke; needs the dense FLUX.2-dev snapshot + the Fun-Controlnet-Union ckpt + a CUDA device (cap=120)"]
fn flux2_dev_control_candle_gpu_smoke() {
    use candle_gen_flux2::{Flux2Control, Flux2ControlPaths, Flux2ControlRequest};

    let weights_dir = env_path("FLUX2_DEV_DIR");
    let control = env_path("FLUX2_CONTROL");
    assert!(
        weights_dir.join("model_index.json").is_file(),
        "FLUX2_DEV_DIR must point at the dense FLUX.2-dev diffusers snapshot: {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("FLUX2_DEV_OUT_DIR", "flux2-dev-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let (w, h) = (
        env_or("FLUX2_DEV_W", "768").parse().expect("FLUX2_DEV_W"),
        env_or("FLUX2_DEV_H", "768").parse().expect("FLUX2_DEV_H"),
    );
    let steps: usize = env_or("FLUX2_DEV_STEPS", "28")
        .parse()
        .expect("FLUX2_DEV_STEPS");
    let pose = synthetic_rgb(w, h);

    println!("[smoke] loading flux2_dev CONTROL (Q4) from {} + {} ...", weights_dir.display(), control.display());
    let model = Flux2Control::load(
        &Flux2ControlPaths {
            root: weights_dir.clone(),
            control: control.clone(),
        },
        Some(Quant::Q4),
    )
    .expect("load candle Flux2Control dev");

    let req = Flux2ControlRequest {
        prompt: env_or(
            "FLUX2_DEV_PROMPT",
            "a knight in ornate steel armor, dramatic cinematic lighting",
        ),
        width: w,
        height: h,
        steps,
        guidance: 4.0,
        control_scale: 0.75,
        seed: 42,
        cancel: gen_core::runtime::CancelFlag::new(),
    };
    println!("[smoke] dev control {w}x{h} @ {steps} steps (scale 0.75) ...");
    let image = model
        .generate(&req, &pose, &mut |_| {})
        .expect("flux2_dev control generate");
    let std = image_std(&image);
    let png = out_dir.join("flux2_dev_control_candle.png");
    save_png(&image, &png);
    println!("[smoke] dev control {}x{} std {:.2} -> {}", image.width, image.height, std, png.display());
    assert_eq!((image.width, image.height), (w, h));
    assert!(std > 5.0, "dev control render degenerate (std {std:.2}) — check CUDA_COMPUTE_CAP=120");
    println!("[smoke] DONE: flux2_dev control (candle) coherent");
}
