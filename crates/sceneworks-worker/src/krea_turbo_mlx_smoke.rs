//! Local real-weight MLX smoke for the Krea 2 Turbo worker lane (epic 7565 sc-7575, the P2 worker-path
//! validation). `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX
//! `krea_2_turbo` engine via `gen_core::load("krea_2_turbo")` with a **Q8** `LoadSpec` pointed at the
//! packed `q8/` turnkey subdir — the exact runtime seam `generate_stream` uses (minus the API/job
//! plumbing) when the router routes `krea_2_turbo` to the in-process MLX generator. The point is to prove
//! the **worker crate links + drives the engine end-to-end** (the `use mlx_gen_krea as _;` force-link
//! anchor in `image_jobs.rs` keeps `inventory::submit!` from being GC'd, so `gen_core::load` resolves the
//! registered generator), not just the engine crate in isolation (that's `mlx-gen-krea`'s own real-weight
//! tests). Turbo is TDM-distilled few-step (8) + CFG-free (no negative pass, guidance inert).
//!
//! Setup — point at the published turnkey `SceneWorks/krea-2-turbo-mlx` (the worker default). With the
//! manifest download already in the HF cache, no env is needed: the smoke auto-resolves the cached
//! snapshot's `q8/` subdir (the same `<root>/q8` selection `image_jobs::base::krea_model_subdir` makes).
//! Override `KREA_TURBO_DIR` to point at a snapshot root or a `q8/`-bearing dir directly.
//! ```text
//! # optional: KREA_TURBO_DIR=/path/to/krea-2-turbo-mlx  (root containing q8/, or the q8/ dir itself)
//! # optional: KREA_STEPS=8 KREA_W=1024 KREA_H=1024 KREA_PROMPT="..." KREA_OUT_DIR=/tmp/krea_turbo_smoke
//! cargo test -p sceneworks-worker --release krea_turbo_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// The engine-complete packed subdir to load: mirror `image_jobs::base::krea_model_subdir`'s default —
/// prefer `<root>/q8` (the shipped default; carries `transformer/diffusion_pytorch_model.safetensors`),
/// else `root` itself if it already *is* a q8 root. Errors loud if neither resolves so a half-download
/// surfaces as a clear failure rather than a confusing engine load error.
fn resolve_q8_dir(root: &Path) -> PathBuf {
    let is_engine_root = |d: &Path| {
        d.join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
    };
    let q8 = root.join("q8");
    if is_engine_root(&q8) {
        return q8;
    }
    assert!(
        is_engine_root(root),
        "KREA_TURBO_DIR must point at the turnkey root (containing q8/) or a q8/ dir with a packed \
         transformer/diffusion_pytorch_model.safetensors; neither found under {}",
        root.display()
    );
    root.to_path_buf()
}

/// Auto-discover the cached `SceneWorks/krea-2-turbo-mlx` turnkey snapshot in the HF hub cache, returning
/// the snapshot whose `q8/` subdir carries the packed transformer. `None` if the manifest download hasn't
/// been pulled (the smoke then panics with the `KREA_TURBO_DIR` hint).
fn cached_turnkey_root() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--krea-2-turbo-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let dir = e.path();
            dir.join("q8/transformer/diffusion_pytorch_model.safetensors")
                .is_file()
                .then_some(dir)
        })
}

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check. A NaN /
/// all-black / flat decode collapses the std toward 0; this guards that degenerate floor. The real
/// quality call is the saved-PNG eyeball (the `recommended`-flag verdict on sc-7574).
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
#[ignore = "real-weight MLX smoke; needs the SceneWorks/krea-2-turbo-mlx q8 turnkey cached + an Apple-Silicon Mac"]
fn krea_turbo_mlx_gpu_smoke() {
    let root = match std::env::var("KREA_TURBO_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_turnkey_root().unwrap_or_else(|| {
            panic!(
                "no cached SceneWorks/krea-2-turbo-mlx q8 turnkey found; download it via the manifest \
                 or set KREA_TURBO_DIR to the turnkey root (containing q8/)"
            )
        }),
    };
    let q8_dir = resolve_q8_dir(&root);

    let out_dir = PathBuf::from(env_or("KREA_OUT_DIR", "/tmp/krea_turbo_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("KREA_STEPS", "8").parse().expect("KREA_STEPS");
    let w: u32 = env_or("KREA_W", "1024").parse().expect("KREA_W");
    let h: u32 = env_or("KREA_H", "1024").parse().expect("KREA_H");
    let prompt = env_or(
        "KREA_PROMPT",
        "a photorealistic portrait of a red fox sitting in a sunlit autumn forest, sharp focus, \
         shallow depth of field",
    );

    // Same seam as the worker's MLX image path: a registry load of the `krea_2_turbo` generator (the
    // worker-crate force-link anchor keeps it registered) + a Q8 `LoadSpec` pointed at the packed q8
    // turnkey subdir. The packed weights auto-detect their quant, so `with_quant(Q8)` is a no-op match
    // to the manifest's `mlx.quantize: 8`.
    println!(
        "[smoke] loading krea_2_turbo (Q8) from {} ...",
        q8_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(q8_dir.clone())).with_quant(Quant::Q8);
    let generator = gen_core::load("krea_2_turbo", &spec).expect("load mlx krea_2_turbo generator");

    // Turbo: few-step + CFG-free. `guidance: None` mirrors the worker's `resolve_guidance` (the
    // descriptor advertises supports_guidance=false, so no value is forwarded and there is no negative
    // pass) — one DiT forward per step.
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: None,
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps (CFG-free) ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("krea_2_turbo generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let std = image_std(&image);
    let png = out_dir.join(format!("krea_turbo_q8_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[smoke] krea_2_turbo {}x{} std {:.2} -> {}",
        image.width,
        image.height,
        std,
        png.display()
    );
    assert_eq!(
        (image.width, image.height),
        (w, h),
        "engine returned the wrong dimensions"
    );
    assert!(
        std > 5.0,
        "krea_2_turbo render looks degenerate (std {std:.2}) — possible NaN / all-black / flat decode"
    );
    println!(
        "[smoke] DONE: krea_2_turbo Q8 render coherent at {steps} steps through the worker lane"
    );
}
