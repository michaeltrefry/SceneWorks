//! On-device per-tier memory-footprint measurement harness (sc-8516, epic 8506).
//!
//! Purpose: produce REAL steady-state resident + peak GPU-memory numbers per (model × quant tier)
//! so the sc-8509 RAM→tier suggestion (apps/web/src/tierSuggestion.js) can be calibrated from
//! measured data instead of the disk×multiplier estimate, and so the sc-8508 manifest footprint
//! fields (`footprint.residentMemoryBytes` / `peakMemoryBytes`) can be populated.
//!
//! Each test drives the EXACT worker runtime seam a real job uses — a registry `gen_core::load(id)`
//! of a packed tier subdir (the same one `image_jobs::base::standard_tier_subdir` resolves) + ONE
//! generation — while sampling the MLX process-global memory counters that generator_cache.rs already
//! publishes to telemetry:
//!   * `mlx_rs::memory::get_active_memory()` — bytes currently live on the GPU allocator. RESIDENT is
//!     sampled from this AFTER the generation AND AFTER a `clear_cache()` that releases the gen's
//!     transient working buffers, so it is the steady-state weight footprint ONLY, NOT the transient.
//!     (MLX is lazy: directly after `load` NOTHING is materialized — `get_active_memory()` reads ~0 —
//!     so resident is only observable once a gen has forced the weights resident. The credibility fix
//!     is the `clear_cache()` before the sample: the old harness read active post-gen WITHOUT it, which
//!     folded the freeable transient INTO resident — inflating resident, understating peak−resident.)
//!   * `mlx_rs::memory::get_peak_memory()` — high-water mark since process start / last reset. Reset
//!     BEFORE load and read AFTER the generation, so the reported peak covers load + the single
//!     generation (the true install-time ceiling the RAM suggestion must budget for).
//!
//! RUN ONE TIER PER PROCESS. The MLX counters + allocator peak high-water mark are process-global and
//! persist across tests in the same binary invocation, so each tier MUST be measured in its OWN
//! `cargo test … footprint_<x>` invocation for a clean allocator + peak counter — otherwise a heavier
//! earlier tier's peak leaks into a lighter later one. The manifest numbers were each captured
//! fresh-process this way.
//!
//! These are `#[ignore]`d real-weight smokes — run by hand on an Apple-Silicon Mac with the tier's
//! turnkey cached (same convention as the *_mlx_smoke tests). Each prints a single machine-parseable
//! `[[FOOTPRINT]] {json}` line (model, tier, diskSizeBytes, residentMemoryBytes, peakMemoryBytes) so
//! a run over the measurable set can be scraped straight into builtin.models.jsonc.
//!
//! ```text
//! # One tier per invocation (fresh MLX allocator + peak counter each time):
//! cargo test -p sceneworks-worker --release footprint_sdxl_q8         -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release footprint_z_image_turbo_q4 -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release footprint_z_image_q4       -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release footprint_lens_turbo_q4    -- --ignored --nocapture
//! ```
//!
//! CALIBRATION SET (prefer already-on-disk tiers; do NOT trigger a 30GB+ download sweep — sc-8516
//! records un-measured tiers for a backfill story and the estimate keeps their suggestion accurate):
//!   * sdxl            q8  (SceneWorks/sdxl-base-mlx      q8/)
//!   * z_image_turbo   q4  (SceneWorks/z-image-turbo-mlx  q4/)
//!   * z_image         q4  (SceneWorks/z-image-mlx        q4/)
//!   * lens_turbo      q4  (SceneWorks/lens-turbo-mlx     q4/)
//!
//! Extra tiers (other quants / models) auto-measure if their subdir is cached; otherwise the test
//! panics with a download hint and is simply not part of the run.

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Locate a cached `SceneWorks/<repo>` turnkey snapshot whose `<tier>/` subdir carries the packed
/// backbone file `sentinel` (e.g. `unet/diffusion_pytorch_model.safetensors`). Returns the `<tier>/`
/// dir itself, ready to hand to `WeightsSource::Dir`. `None` if the tier hasn't been pulled.
fn cached_tier_dir(repo: &str, tier: &str, sentinel: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let repo_cache = repo.replace('/', "--");
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{repo_cache}"))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let tier_dir = e.path().join(tier);
            tier_dir.join(sentinel).is_file().then_some(tier_dir)
        })
}

/// Resolve the tier dir from an explicit override env (a turnkey root OR the tier dir itself) or the
/// HF cache, panicking with a download hint if neither resolves so a half/no download surfaces clearly.
fn resolve_tier_dir(env_key: &str, repo: &str, tier: &str, sentinel: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_key) {
        let p = p.trim();
        if !p.is_empty() {
            let root = PathBuf::from(p);
            let sub = root.join(tier);
            if sub.join(sentinel).is_file() {
                return sub;
            }
            if root.join(sentinel).is_file() {
                return root;
            }
            panic!(
                "{env_key}={} has no packed {tier}/{sentinel} (nor is it itself a {tier} root)",
                root.display()
            );
        }
    }
    cached_tier_dir(repo, tier, sentinel).unwrap_or_else(|| {
        panic!(
            "no cached {repo} {tier} turnkey found; download it \
             (`hf download {repo} --include '{tier}/*'`) or set {env_key} to the turnkey root"
        )
    })
}

/// The load-quant for a tier, or `None` for the dense `bf16` tier (loaded without `.with_quant`, the
/// same as the worker's dense path — the packed q4/q8 subdirs auto-detect their quant, so `with_quant`
/// just names the tier).
fn quant_for(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

/// Mean per-pixel std across the RGB buffer — the cheap "is the render non-degenerate?" floor so a
/// broken decode doesn't silently produce a bogus footprint number.
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

/// Run one real load + generation for `(model, tier)` at `dir`, sampling the MLX memory counters
/// around it, and emit the machine-parseable `[[FOOTPRINT]]` line. Returns (residentBytes, peakBytes).
///
/// Sequence (mirrors the worker's per-job memory lifecycle in generator_cache.rs, and separates the
/// steady-state weight footprint from the generation transient — the sc-8516 credibility fix):
///   1. `clear_cache()` then `reset_peak_memory()` — start the peak high-water mark from a clean base
///      so it captures load + this one generation, not leftovers from a prior test in the same process.
///   2. record `baseline = get_active_memory()` — anything already live (should be ~0 in a
///      one-tier-per-process run; subtracted so each line is the marginal footprint of THIS tier).
///   3. load the packed tier + generate once. PEAK = `get_peak_memory()` read right after gen — the
///      load+gen high-water ceiling (peak was reset in step 1 and never since, so it spans the window).
///   4. `clear_cache()` to RELEASE the generation's transient working buffers (VAE-decode scratch,
///      attention activations, latents) back to the OS, THEN sample
///      `residentBytes = get_active_memory() - baseline` — the STEADY-STATE resident WEIGHTS only.
///
/// Why resident is sampled POST-gen-and-clear rather than pre-gen: MLX evaluates lazily, so directly
/// after `gen_core::load` NOTHING is materialized on the GPU allocator (`get_active_memory()` reads
/// ~0 — verified) — the weights are only realized when the first forward pass touches them. So a true
/// steady-state resident is only observable AFTER a generation has forced materialization. The fix
/// versus the original harness is the `clear_cache()` on line below: the old code sampled
/// `get_active_memory()` post-gen WITHOUT releasing the cache, folding the gen's freeable transient
/// INTO resident (inflating resident, understating peak−resident). Dropping the cache first leaves
/// only the live weight arrays the generator still holds.
fn measure_footprint(
    model: &str,
    tier: &str,
    engine_id: &str,
    dir: &Path,
    req: GenerationRequest,
) -> (u64, u64) {
    println!(
        "[footprint] loading {model} ({tier}) from {} ...",
        dir.display()
    );

    // 1. Clean baseline: drop the allocator's free cache and zero the peak high-water mark so the
    //    numbers below are this tier's own load+gen, not a prior test's residue.
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let baseline = mlx_rs::memory::get_active_memory() as u64;

    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
    if let Some(q) = quant_for(tier) {
        spec = spec.with_quant(q);
    }
    let generator = gen_core::load(engine_id, &spec)
        .unwrap_or_else(|e| panic!("load {engine_id} ({tier}): {e:?}"));

    // 3. One generation, then the load+gen peak high-water mark (reset in step 1, never since).
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .unwrap_or_else(|e| panic!("{model} {tier} generate: {e:?}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };
    let std = image_std(&image);
    assert!(
        std > 5.0,
        "{model} {tier} render looks degenerate (std {std:.2}) — measured footprint would be bogus"
    );
    let peak = mlx_rs::memory::get_peak_memory() as u64;

    // 4. STEADY-STATE RESIDENT: release the gen's transient working buffers, THEN sample. This is the
    //    credibility fix — the old harness sampled active WITHOUT this clear_cache, so resident carried
    //    the freeable transient and peak−resident was an artifact, not the real transient.
    mlx_rs::memory::clear_cache();
    let active_resident = mlx_rs::memory::get_active_memory() as u64;
    let resident = active_resident.saturating_sub(baseline);
    let transient = peak.saturating_sub(active_resident);

    // Machine-parseable line — scrape `[[FOOTPRINT]]` to backfill builtin.models.jsonc.
    println!(
        "[[FOOTPRINT]] {{\"model\":\"{model}\",\"tier\":\"{tier}\",\"residentMemoryBytes\":{resident},\"peakMemoryBytes\":{peak}}}"
    );
    println!(
        "[footprint] {model} {tier}: resident {:.2} GiB (active {:.2}, baseline {:.2}) | peak {:.2} GiB | transient (peak−resident) {:.2} GiB | render std {:.1}",
        resident as f64 / 1024.0 / 1024.0 / 1024.0,
        active_resident as f64 / 1024.0 / 1024.0 / 1024.0,
        baseline as f64 / 1024.0 / 1024.0 / 1024.0,
        peak as f64 / 1024.0 / 1024.0 / 1024.0,
        transient as f64 / 1024.0 / 1024.0 / 1024.0,
        std,
    );
    (resident, peak)
}

/// SDXL-family default request: 1024², real CFG. Kept modest on steps to keep the harness quick; the
/// footprint is dominated by resident weights + a single denoise's activations, not step count.
fn sdxl_request() -> GenerationRequest {
    GenerationRequest {
        prompt: env_or(
            "FP_PROMPT",
            "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
        ),
        width: env_or("FP_W", "1024").parse().expect("FP_W"),
        height: env_or("FP_H", "1024").parse().expect("FP_H"),
        count: 1,
        seed: Some(42),
        steps: Some(env_or("FP_STEPS", "12").parse().expect("FP_STEPS")),
        guidance: Some(7.0),
        ..Default::default()
    }
}

/// Distilled/turbo request: few steps. `guidance`: `Some(1.0)` for lens (CFG-scale 1 accepted), or
/// `None` for guidance-DISTILLED engines like z_image_turbo that reject any `guidance` value.
fn turbo_request(steps: u32, guidance: Option<f32>) -> GenerationRequest {
    GenerationRequest {
        prompt: env_or(
            "FP_PROMPT",
            "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
        ),
        width: env_or("FP_W", "1024").parse().expect("FP_W"),
        height: env_or("FP_H", "1024").parse().expect("FP_H"),
        count: 1,
        seed: Some(42),
        steps: Some(
            env_or("FP_STEPS", &steps.to_string())
                .parse()
                .expect("FP_STEPS"),
        ),
        guidance,
        ..Default::default()
    }
}

const SDXL_SENTINEL: &str = "unet/diffusion_pytorch_model.safetensors";
// Lens turnkeys pack the DiT under transformer/diffusion_pytorch_model.safetensors …
const LENS_SENTINEL: &str = "transformer/diffusion_pytorch_model.safetensors";
// … while Z-Image turnkeys pack it under transformer/model.safetensors.
const ZIMAGE_SENTINEL: &str = "transformer/model.safetensors";

#[test]
#[ignore = "footprint measurement; needs SceneWorks/sdxl-base-mlx q8 cached + an Apple-Silicon Mac"]
fn footprint_sdxl_q8() {
    let dir = resolve_tier_dir(
        "FP_SDXL_Q8_DIR",
        "SceneWorks/sdxl-base-mlx",
        "q8",
        SDXL_SENTINEL,
    );
    measure_footprint("sdxl", "q8", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/sdxl-base-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_sdxl_q4() {
    let dir = resolve_tier_dir(
        "FP_SDXL_Q4_DIR",
        "SceneWorks/sdxl-base-mlx",
        "q4",
        SDXL_SENTINEL,
    );
    measure_footprint("sdxl", "q4", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/sdxl-base-mlx bf16 cached + an Apple-Silicon Mac"]
fn footprint_sdxl_bf16() {
    let dir = resolve_tier_dir(
        "FP_SDXL_BF16_DIR",
        "SceneWorks/sdxl-base-mlx",
        "bf16",
        SDXL_SENTINEL,
    );
    measure_footprint("sdxl", "bf16", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/z-image-turbo-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_z_image_turbo_q4() {
    let dir = resolve_tier_dir(
        "FP_ZIMAGE_TURBO_Q4_DIR",
        "SceneWorks/z-image-turbo-mlx",
        "q4",
        ZIMAGE_SENTINEL,
    );
    measure_footprint(
        "z_image_turbo",
        "q4",
        "z_image_turbo",
        &dir,
        turbo_request(8, None),
    );
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/z-image-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_z_image_q4() {
    let dir = resolve_tier_dir(
        "FP_ZIMAGE_Q4_DIR",
        "SceneWorks/z-image-mlx",
        "q4",
        ZIMAGE_SENTINEL,
    );
    // Non-turbo Z-Image runs real CFG at more steps; still a single denoise for the footprint.
    measure_footprint("z_image", "q4", "z_image", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/lens-turbo-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_lens_turbo_q4() {
    let dir = resolve_tier_dir(
        "FP_LENS_Q4_DIR",
        "SceneWorks/lens-turbo-mlx",
        "q4",
        LENS_SENTINEL,
    );
    measure_footprint(
        "lens_turbo",
        "q4",
        "lens_turbo",
        &dir,
        turbo_request(4, Some(1.0)),
    );
}
