//! Local real-weight MLX smoke for the Chroma1-Base **Q4** worker lane (sc-8777, epic 8506 Group-B).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX `chroma1_base`
//! engine via `gen_core::load("chroma1_base")` with a **Q4** `LoadSpec` pointed at the packed `q4/`
//! turnkey subdir — the exact runtime seam `generate_stream` uses (minus the API/job plumbing) when
//! the router routes a `chroma1_base` MLX job (`standard_tier_subdir` → the `q4/` subdir; `resolve_quant`
//! → `Quant::Q4`).
//!
//! Purpose: on-device evidence that the SceneWorks/chroma1-base-mlx pre-quantized q4 turnkey loads
//! through the worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and
//! renders a non-degenerate image. Chroma packs ONLY the transformer DiT per-tier; the T5-XXL text
//! encoder + VAE stay dense (chroma never quantizes its T5 — it is NOT a dense-TE model in the manifest
//! sense, it simply has no packed-TE tier), so the q4 load-quant packs only the already-packed
//! transformer (a harmless no-op match) and loads the dense T5 unchanged. The other two variants
//! (chroma1_hd / chroma1_flash) share this crate + layout, so this single q4 proof covers the family.
//!
//! Setup — point at the published turnkey `SceneWorks/chroma1-base-mlx` (the worker default). With the q4
//! tier already in the HF cache, no env is needed: the smoke auto-resolves the cached snapshot's `q4/`
//! subdir (the same selection `image_jobs::base::standard_tier_subdir` makes for `mlxQuantize: 4`).
//! Override `CHROMA_Q4_DIR` to point at a snapshot root or a `q4/`-bearing dir directly.
//! ```text
//! # optional: CHROMA_Q4_DIR=/path/to/chroma1-base-mlx  (root containing q4/, or the q4/ dir itself)
//! # optional: CHROMA_STEPS=40 CHROMA_W=1024 CHROMA_H=1024 CHROMA_PROMPT="..." CHROMA_OUT_DIR=/tmp/chroma1_base_q4_smoke
//! cargo test -p sceneworks-worker --release chroma1_base_q4_mlx_gpu_smoke -- --ignored --nocapture
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

/// The engine-complete packed subdir to load: mirror `image_jobs::base::standard_tier_subdir`'s q4
/// selection — prefer `<root>/q4` (chroma turnkeys pack the backbone under `transformer/`), else `root`
/// itself if it already *is* a q4 root. Errors loud if neither resolves so a half-download surfaces as
/// a clear failure rather than a confusing engine load error.
fn resolve_q4_dir(root: &Path) -> PathBuf {
    let is_engine_root = |d: &Path| {
        d.join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
    };
    let q4 = root.join("q4");
    if is_engine_root(&q4) {
        return q4;
    }
    assert!(
        is_engine_root(root),
        "CHROMA_Q4_DIR must point at the turnkey root (containing q4/) or a q4/ dir with a packed \
         transformer/diffusion_pytorch_model.safetensors; neither found under {}",
        root.display()
    );
    root.to_path_buf()
}

/// Auto-discover the cached `SceneWorks/chroma1-base-mlx` turnkey snapshot in the HF hub cache, returning
/// the snapshot whose `q4/` subdir carries the packed transformer. `None` if the q4 tier hasn't been
/// pulled (the smoke then panics with the `CHROMA_Q4_DIR` hint).
fn cached_turnkey_root() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--chroma1-base-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let dir = e.path();
            dir.join("q4/transformer/diffusion_pytorch_model.safetensors")
                .is_file()
                .then_some(dir)
        })
}

/// Per-pixel mean over the RGB buffer — the "is it black?" floor check, reported for the record.
fn image_mean(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n
}

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check. A
/// NaN / all-black / flat decode collapses the std toward 0; this guards that degenerate floor. The real
/// quality call is the saved-PNG eyeball.
fn image_std(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = image_mean(img);
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt()
}

/// Whether EVERY pixel byte is exactly 0 — the precise degenerate signature of a broken decode.
fn is_all_zero(img: &Image) -> bool {
    !img.pixels.is_empty() && img.pixels.iter().all(|&p| p == 0)
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("rgb buffer")
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/chroma1-base-mlx q4 turnkey cached + an Apple-Silicon Mac"]
fn chroma1_base_q4_mlx_gpu_smoke() {
    let root = match std::env::var("CHROMA_Q4_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_turnkey_root().unwrap_or_else(|| {
            panic!(
                "no cached SceneWorks/chroma1-base-mlx q4 turnkey found; download it via the manifest \
                 (`hf download SceneWorks/chroma1-base-mlx --include 'q4/*'`) or set CHROMA_Q4_DIR to \
                 the turnkey root (containing q4/)"
            )
        }),
    };
    let q4_dir = resolve_q4_dir(&root);

    let out_dir = PathBuf::from(env_or("CHROMA_OUT_DIR", "/tmp/chroma1_base_q4_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("CHROMA_STEPS", "40").parse().expect("CHROMA_STEPS");
    let w: u32 = env_or("CHROMA_W", "1024").parse().expect("CHROMA_W");
    let h: u32 = env_or("CHROMA_H", "1024").parse().expect("CHROMA_H");
    let prompt = env_or(
        "CHROMA_PROMPT",
        "a photorealistic portrait of a red fox sitting in a sunlit autumn forest, sharp focus, \
         shallow depth of field",
    );

    // Same seam as the worker's MLX image path: a registry load of the `chroma1_base` generator (the
    // worker-crate force-link anchor `use mlx_gen_chroma as _;` keeps it registered) + a Q4 `LoadSpec`
    // pointed at the packed q4 turnkey subdir. The packed transformer auto-detects its quant, so
    // `with_quant(Q4)` matches the manifest's `mlx.quantize: 4` tier; the dense T5 loads unchanged.
    println!(
        "[smoke] loading chroma1_base (Q4) from {} ...",
        q4_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(q4_dir.clone())).with_quant(Quant::Q4);
    let generator = gen_core::load("chroma1_base", &spec).expect("load mlx chroma1_base generator");

    // Chroma1-Base is a TRUE-CFG family: the engine advertises `supports_guidance=false` +
    // `supports_negative_prompt=true`, so the worker forwards the CFG scale as `true_cfg` (NOT the
    // distilled `guidance` scalar, which the engine rejects) — mirroring `resolve_true_cfg` /
    // `resolve_negative_prompt` in image_jobs::base (manifest defaults: 40 steps, guidance 3.0).
    let req = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt: Some("blurry, low quality, distorted".to_string()),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        true_cfg: Some(3.0),
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("chroma1_base generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let all_zero = is_all_zero(&image);
    let png = out_dir.join(format!("chroma1_base_q4_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[smoke] chroma1_base Q4 {}x{} mean {:.2} std {:.2} all_zero={} -> {}",
        image.width,
        image.height,
        mean,
        std,
        all_zero,
        png.display()
    );
    assert_eq!(
        (image.width, image.height),
        (w, h),
        "engine returned the wrong dimensions"
    );
    assert!(
        !all_zero,
        "chroma1_base Q4 decode is ALL-ZERO — a broken packed load/decode"
    );
    assert!(
        std > 20.0,
        "chroma1_base Q4 render looks degenerate (std {std:.2}) — possible NaN / all-black / flat decode"
    );
    println!(
        "[smoke] DONE: chroma1_base Q4 render coherent (mean {mean:.2}, std {std:.2}, NOT all-zero) \
         at {steps} steps through the worker packed lane"
    );
}
