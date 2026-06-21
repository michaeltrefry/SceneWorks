//! Local real-weight GPU smoke for the candle SCAIL-2 worker lane (sc-6837 / sc-6836). `#[ignore]`d —
//! run by hand on the RTX PRO 6000. It drives the **shipped** worker conditioning code
//! ([`crate::person_segment_sam3_candle::segment_all_persons_in_memory`] + the [`crate::scail2_masks`]
//! painters) and then the real candle SCAIL-2 engine via `gen_core::load("scail2_14b")` — the same
//! runtime seam the worker uses, minus the API/job plumbing. This is the engine + worker-conditioning
//! validation that backs both stories' GPU merge gate.
//!
//! Setup (PowerShell, after extracting driving frames + converting the reference to PNG):
//! ```text
//! $env:SCENEWORKS_CANDLE_SCAIL2_DIR="D:\sceneworks-scail2-validate\snapshot"
//! $env:SCAIL2_SAM3_DIR="D:\sam3-weights"
//! $env:SCAIL2_REF="D:\sceneworks-scail2-validate\joel.png"
//! $env:SCAIL2_DRIVING_DIR="D:\sceneworks-scail2-validate\driving"
//! $env:SCAIL2_OUT_DIR="D:\sceneworks-scail2-validate\out"
//! # optional: SCAIL2_FRAMES=21 SCAIL2_STEPS=8 SCAIL2_W=640 SCAIL2_H=352 SCAIL2_MODE=animation|replacement
//! cargo test -p sceneworks-worker --features backend-candle --release scail2_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, ReplacementMode,
    WeightsSource,
};

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

fn load_rgb(path: &Path) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .to_rgb8();
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

fn to_rgb_image(img: &Image) -> image::RgbImage {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone()).expect("rgb buffer")
}

/// Per-frame pixel std-dev, averaged — a cheap "is the clip non-degenerate" check (a NaN-clamped /
/// all-black engine output collapses toward 0).
fn avg_frame_std(frames: &[Image]) -> f64 {
    let mut total = 0f64;
    for f in frames {
        let n = f.pixels.len() as f64;
        let mean = f.pixels.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = f
            .pixels
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        total += var.sqrt();
    }
    total / frames.len().max(1) as f64
}

#[test]
#[ignore = "real-weight GPU smoke; needs the candle SCAIL-2 snapshot + SAM3 weights + driving assets"]
fn scail2_candle_gpu_smoke() {
    // Anchor the provider's inventory registration into the test binary (mirrors video_jobs.rs).
    use candle_gen_scail2 as _;

    let snapshot = env_path("SCENEWORKS_CANDLE_SCAIL2_DIR");
    let sam3_dir = env_path("SCAIL2_SAM3_DIR");
    let sam3_model = sam3_dir.join("model.safetensors");
    let sam3_tok = sam3_dir.join("tokenizer.json");
    let ref_path = env_path("SCAIL2_REF");
    let driving_dir = env_path("SCAIL2_DRIVING_DIR");
    let out_dir = env_path("SCAIL2_OUT_DIR");
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let frames_n: usize = env_or("SCAIL2_FRAMES", "21")
        .parse()
        .expect("SCAIL2_FRAMES");
    let steps: u32 = env_or("SCAIL2_STEPS", "8").parse().expect("SCAIL2_STEPS");
    let width: u32 = env_or("SCAIL2_W", "640").parse().expect("SCAIL2_W");
    let height: u32 = env_or("SCAIL2_H", "352").parse().expect("SCAIL2_H");
    let mode = env_or("SCAIL2_MODE", "animation"); // "animation" | "replacement"
    let replacement = mode == "replacement";
    // Optional CFG override: unset → engine DEFAULT_GUIDANCE; "1.0" disables CFG (pure conditional).
    let guidance: Option<f32> = std::env::var("SCAIL2_GUIDANCE")
        .ok()
        .and_then(|v| v.trim().parse().ok());
    // Optional explicit prompt / negative prompt (defaults below).
    let prompt = env_or("SCAIL2_PROMPT", "a person performing the driving motion");
    let neg = std::env::var("SCAIL2_NEG")
        .ok()
        .map(|v| v.trim().to_string());

    // --- reference character + its color-coded mask (SAM3 -> the primary person painted blue) ---
    // Animation keeps the reference's world (white bg); replacement discards it (black bg).
    let reference = load_rgb(&ref_path);
    let ref_rgb = to_rgb_image(&reference);
    let ref_masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
        &sam3_model,
        &sam3_tok,
        std::slice::from_ref(&ref_rgb),
    )
    .expect("sam3 reference segmentation");
    let ref_bg = if replacement {
        crate::scail2_masks::BG_BLACK
    } else {
        crate::scail2_masks::BG_WHITE
    };
    let ref_mask = crate::scail2_masks::paint_reference_mask(&ref_masks, ref_bg).expect("ref mask");

    // --- driving frames + per-frame color masks (SAM3 -> every person painted its palette color) ---
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&driving_dir)
        .expect("read driving dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    paths.sort();
    paths.truncate(frames_n);
    assert!(
        !paths.is_empty(),
        "no driving .png frames in {}",
        driving_dir.display()
    );
    let driving: Vec<Image> = paths.iter().map(|p| load_rgb(p)).collect();
    let driving_rgb: Vec<image::RgbImage> = driving.iter().map(to_rgb_image).collect();
    let drive_masks = crate::person_segment_sam3_candle::segment_all_persons_in_memory(
        &sam3_model,
        &sam3_tok,
        &driving_rgb,
    )
    .expect("sam3 driving segmentation");
    let drive_bg = if replacement {
        crate::scail2_masks::BG_WHITE
    } else {
        crate::scail2_masks::BG_BLACK
    };
    let driving_masks = crate::scail2_masks::paint_driving_masks(&drive_masks, drive_bg);
    assert_eq!(
        driving_masks.len(),
        driving.len(),
        "one mask per driving frame"
    );

    println!(
        "[smoke] ref {}x{} | {} driving frames @ {}x{} | mode={mode} steps={steps}",
        reference.width,
        reference.height,
        driving.len(),
        width,
        height
    );

    // --- build the request + run the engine via the worker's runtime seam (gen_core::load) ---
    let conditioning = vec![
        Conditioning::Reference {
            image: reference,
            strength: None,
        },
        Conditioning::Mask { image: ref_mask },
        Conditioning::ControlClip {
            frames: driving,
            mask: driving_masks,
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        },
    ];
    let req = GenerationRequest {
        prompt,
        negative_prompt: neg,
        width,
        height,
        frames: Some(frames_n as u32),
        fps: Some(16),
        steps: Some(steps),
        guidance,
        seed: Some(42),
        conditioning,
        video_mode: Some(mode.clone()),
        ..Default::default()
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot));
    let generator = gen_core::load("scail2_14b", &spec).expect("load scail2_14b candle provider");

    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("scail2 generate");

    let out_frames = match output {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("expected Video output, got {other:?}"),
    };
    assert!(!out_frames.is_empty(), "engine returned no frames");

    for (i, f) in out_frames.iter().enumerate() {
        let p = out_dir.join(format!("out_{i:05}.png"));
        image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
            .expect("out frame buffer")
            .save(&p)
            .unwrap_or_else(|e| panic!("save {}: {e}", p.display()));
    }
    let avg_std = avg_frame_std(&out_frames);
    println!(
        "[smoke] DONE: wrote {} frames to {} (avg per-frame std {:.2})",
        out_frames.len(),
        out_dir.display(),
        avg_std
    );
    assert!(
        avg_std > 5.0,
        "output frames look degenerate (avg std {avg_std:.2}) — possible NaN / all-black decode"
    );
}
