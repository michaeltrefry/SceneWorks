//! sc-6541 — native-Rust LoRA train→generate driver (research instrument, test-only).
//!
//! The training + generation halves of the closed-loop study
//! (`docs/sc-6541/closed-loop-protocol.md`). Entirely native MLX via the `gen_core` contract —
//! no Python. Two `#[ignore]` tests, each parameterized by env, that produce artifacts the
//! `lora_eval_harness` then scores:
//!
//! - `train_zimage_lora` — `gen_core::load_trainer("z_image_turbo", Dir(base)).train(req)` over
//!   `TRAIN_DIR` → a LoRA `.safetensors` in `OUTPUT_DIR`.
//! - `generate_zimage_grid` — `gen_core::load("z_image_turbo", Dir(base)[+adapter]).generate` over
//!   the fixed prompt grid → PNGs in `GEN_DIR` (+ `prompts.json`). With `ADAPTER_PATH` set it uses
//!   the LoRA; unset = the no-LoRA baseline floor (protocol §2.1).
//!
//! The full loop on this Mac (M5 Max, 64 GB; MLX is `!Send` → `RUST_TEST_THREADS=1`):
//! ```sh
//! . "$HOME/.cargo/env"
//! BASE=~/Datasets/z-image-base
//! # 1. no-LoRA baseline grid
//! GEN_DIR=~/Datasets/lora-eval/gen/baseline TRIGGER="sks man" \
//!   RUST_TEST_THREADS=1 cargo test -p sceneworks-worker --lib generate_zimage_grid -- --ignored --nocapture
//! # 2. train a clean calibration LoRA
//! TRAIN_DIR=~/Datasets/lora-eval/basim-train-cal OUTPUT_DIR=~/Datasets/lora-eval/adapters \
//!   ADAPTER_NAME=basim-clean.safetensors TRIGGER="sks man" Z_RESOLUTION=512 Z_STEPS=400 \
//!   RUST_TEST_THREADS=1 cargo test -p sceneworks-worker --lib train_zimage_lora -- --ignored --nocapture
//! # 3. LoRA grid, then score both vs the reference pool with the eval harness
//! GEN_DIR=~/Datasets/lora-eval/gen/basim-clean ADAPTER_PATH=~/Datasets/lora-eval/adapters/basim-clean.safetensors \
//!   TRIGGER="sks man" RUST_TEST_THREADS=1 cargo test -p sceneworks-worker --lib generate_zimage_grid -- --ignored --nocapture
//! ```

#[cfg(test)]
mod driver {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use gen_core::{
        AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Image, LoadSpec,
        TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
    };
    use mlx_gen::weights::Weights;
    use mlx_gen_face::FaceAnalysis;

    use crate::image_jobs::{INSTANTID_ARCFACE_FILE, INSTANTID_SCRFD_FILE};

    /// Z-Image-Turbo trainer + generator share this registry id (the SceneWorks `z_image_lora`
    /// kernel name maps to it). See `mlx-gen-z-image/src/model.rs` `MODEL_ID`.
    const MODEL_ID: &str = "z_image_turbo";

    /// The fixed person prompt grid (protocol §7). `{trigger}` is substituted at runtime so the
    /// training caption and the generation prompts share the same trigger token. Spans a plain
    /// identity portrait (identity floor), varied settings/lighting/wardrobe (prompt adherence +
    /// spread), and compositions the training set will not contain (generalization).
    const PROMPT_GRID: &[(&str, &str)] = &[
        ("p01_portrait", "a photo of {trigger}, head and shoulders portrait, neutral grey background, sharp focus"),
        ("p02_outdoor", "a photo of {trigger} standing in a green park, bright daylight, full body"),
        ("p03_office", "a photo of {trigger} sitting at a desk in a modern office, indoor lighting"),
        ("p04_night", "a photo of {trigger} walking on a city street at night, neon signs, bokeh"),
        ("p05_smiling", "a candid photo of {trigger} laughing, warm golden-hour light"),
        ("p06_suit", "a professional studio portrait of {trigger} wearing a dark suit and tie"),
        ("p07_beach", "a photo of {trigger} at a sunny beach, ocean behind, bright sunlight"),
        ("p08_closeup", "an extreme close-up photo of {trigger}'s face, detailed skin, sharp focus"),
    ];

    fn home() -> PathBuf {
        PathBuf::from(std::env::var("HOME").expect("HOME"))
    }

    fn base_dir() -> PathBuf {
        std::env::var("ZIMAGE_BASE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home().join("Datasets/z-image-base"))
    }

    fn env_str(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }

    fn env_u32(key: &str, default: u32) -> u32 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn env_f32(key: &str, default: f32) -> f32 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn require_env(key: &str) -> PathBuf {
        PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("set {key}")))
    }

    /// Image files in a directory (non-recursive), sorted for determinism.
    fn image_files(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("read dir {}: {e}", dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|x| x.to_str())
                    .map(|x| {
                        matches!(
                            x.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp"
                        )
                    })
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        files
    }

    fn write_png(img: &Image, path: &Path) {
        let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .expect("RGB buffer dims match");
        buf.save(path)
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    /// Resolve the SCRFD+ArcFace bundle dir (env override, else the app-managed cache).
    fn face_bundle_dir() -> PathBuf {
        std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                home().join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
            })
    }

    /// Preprocess a directory into face-centered square crops (protocol §2.1 / the center-crop
    /// confound): SCRFD-detect the largest face, crop a `CROP_PAD`× padded square centered on it
    /// (clamped to the frame), resize to `CROP_SIZE`. This recovers faces that the trainer's own
    /// `center_crop_square` would drop on tall full-body shots, and makes the face dominant so
    /// identity learns strongly and degradations actually hit it. No-face images fall back to a
    /// plain center square (and are reported). `SRC_DIR` → `DST_DIR`.
    #[test]
    #[ignore = "research preprocessing: SCRFD face-centered crops; needs the SCRFD/ArcFace bundle + Metal. Set SRC_DIR/DST_DIR"]
    fn face_center_crop_dir() {
        let src = require_env("SRC_DIR");
        let dst = require_env("DST_DIR");
        std::fs::create_dir_all(&dst).unwrap();
        let pad = env_f32("CROP_PAD", 2.2);
        let out_size = env_u32("CROP_SIZE", 768);

        let bundle = face_bundle_dir();
        let scrfd = Weights::from_file(bundle.join(INSTANTID_SCRFD_FILE)).expect("SCRFD weights");
        let arcface =
            Weights::from_file(bundle.join(INSTANTID_ARCFACE_FILE)).expect("ArcFace weights");
        let fa = FaceAnalysis::load(&scrfd, &arcface).expect("FaceAnalysis");

        let mut done = 0usize;
        let mut no_face = 0usize;
        for path in image_files(&src) {
            let dyn_img = crate::image_decode::decode_image_any(&path).expect("decode");
            let rgb = dyn_img.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());
            let dets = fa
                .detect(rgb.as_raw(), h as usize, w as usize)
                .expect("detect");

            // Square crop geometry: centered on the largest face (bbox = [x1,y1,x2,y2]), side =
            // pad × the longer face edge; fall back to a plain center square if no face.
            let (cx, cy, mut side) = match dets.first() {
                Some(d) => {
                    let [x1, y1, x2, y2] = d.bbox;
                    let face = (x2 - x1).max(y2 - y1).max(1.0);
                    ((x1 + x2) / 2.0, (y1 + y2) / 2.0, face * pad)
                }
                None => {
                    no_face += 1;
                    (w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32)
                }
            };
            side = side.min(w as f32).min(h as f32);
            let x = (cx - side / 2.0).clamp(0.0, (w as f32 - side).max(0.0));
            let y = (cy - side / 2.0).clamp(0.0, (h as f32 - side).max(0.0));

            let cropped = dyn_img
                .crop_imm(x as u32, y as u32, side as u32, side as u32)
                .resize_exact(out_size, out_size, image::imageops::FilterType::Lanczos3);
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
            cropped
                .save(dst.join(format!("{stem}.png")))
                .unwrap_or_else(|e| panic!("save {stem}: {e}"));
            done += 1;
        }
        println!(
            "face-cropped {done} images ({no_face} no-face fallbacks) pad {pad} size {out_size} → {}",
            dst.display()
        );
        assert!(done > 0, "no images in {}", src.display());
    }

    /// Construct a degraded variant of `SRC_DIR` → `DST_DIR` (protocol §2 / §10). Count and
    /// framing are held fixed (degradation changes pixels/selection only):
    ///   * `MODE=blur` — Gaussian-blur every image (`SIGMA`, default 4.0). Exercises the blur floor.
    ///   * `MODE=neardup` — keep `N_BASE` (default 4) distinct images, replicate them round-robin
    ///     to the original count with a small per-copy brightness jitter. Near-dups (high CLIP
    ///     cosine / pHash Hamming ≈0) but NOT byte-identical, so they trip near-dup + low-diversity,
    ///     not the SHA exact-dup path. Exercises the diversity / near-dup signal.
    #[test]
    #[ignore = "research preprocessing: build a degraded dataset variant. Set SRC_DIR/DST_DIR/MODE"]
    fn degrade_dir() {
        let src = require_env("SRC_DIR");
        let dst = require_env("DST_DIR");
        std::fs::create_dir_all(&dst).unwrap();
        let mode = env_str("MODE", "blur");
        let files = image_files(&src);
        assert!(!files.is_empty(), "no images in {}", src.display());

        match mode.as_str() {
            "blur" => {
                let sigma = env_f32("SIGMA", 4.0);
                for p in &files {
                    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
                    image::open(p)
                        .expect("open")
                        .blur(sigma)
                        .save(dst.join(format!("{stem}.png")))
                        .unwrap();
                }
                println!(
                    "blurred {} images (sigma {sigma}) → {}",
                    files.len(),
                    dst.display()
                );
            }
            "neardup" => {
                let n_base = (env_u32("N_BASE", 4) as usize).min(files.len());
                let target = files.len();
                let bases: Vec<&PathBuf> = files.iter().take(n_base).collect();
                let mut written = 0usize;
                while written < target {
                    let b = bases[written % n_base];
                    // Small, always-nonzero brightness jitter: near-dup, never byte-identical.
                    let delta = (written % 5 + 1) as i32 * 2;
                    image::open(b)
                        .expect("open")
                        .brighten(delta)
                        .save(dst.join(format!("nd_{written:02}.png")))
                        .unwrap();
                    written += 1;
                }
                println!(
                    "near-dup variant: {target} images from {n_base} base images → {}",
                    dst.display()
                );
            }
            other => panic!("MODE must be blur|neardup, got {other:?}"),
        }
    }

    /// Train one LoRA over `TRAIN_DIR` and write it to `OUTPUT_DIR/ADAPTER_NAME`.
    /// Config (held fixed across the study's variants): `Z_RESOLUTION` / `Z_STEPS` / `Z_LR` /
    /// `Z_RANK`, `TRIGGER`. bf16 (the 512² peak is well under 64 GB; 1024² bf16 peaks ~44 GB).
    #[test]
    #[ignore = "research: trains a real Z-Image LoRA; needs the base snapshot + Metal. Set TRAIN_DIR/OUTPUT_DIR"]
    fn train_zimage_lora() {
        let train_dir = require_env("TRAIN_DIR");
        let output_dir = require_env("OUTPUT_DIR");
        std::fs::create_dir_all(&output_dir).unwrap();
        let trigger = env_str("TRIGGER", "sks man");
        let caption = format!("a photo of {trigger}");
        let file_name = env_str("ADAPTER_NAME", "lora.safetensors");

        let items: Vec<TrainingItem> = image_files(&train_dir)
            .into_iter()
            .map(|image_path| TrainingItem {
                image_path,
                caption: caption.clone(),
            })
            .collect();
        assert!(
            !items.is_empty(),
            "no training images in {}",
            train_dir.display()
        );
        println!(
            "training on {} images, trigger {trigger:?}, caption {caption:?}",
            items.len()
        );

        let rank = env_u32("Z_RANK", 16);
        let config = TrainingConfig {
            rank,
            alpha: rank as f32,
            learning_rate: env_f32("Z_LR", 1e-4),
            steps: env_u32("Z_STEPS", 400),
            resolution: env_u32("Z_RESOLUTION", 512),
            save_every: 0,
            seed: env_u32("Z_SEED", 7) as u64,
            train_dtype: "bf16".to_string(),
            ..Default::default()
        };
        println!(
            "config: rank {} steps {} res {} lr {} dtype {}",
            config.rank, config.steps, config.resolution, config.learning_rate, config.train_dtype
        );

        let req = TrainingRequest {
            items,
            config,
            output_dir: output_dir.clone(),
            file_name: file_name.clone(),
            trigger_words: vec![trigger],
            cancel: CancelFlag::new(),
        };

        let mut trainer =
            gen_core::load_trainer(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(base_dir())))
                .expect("load z_image_turbo trainer");
        trainer.validate(&req).expect("validate training request");

        let mut last_loss = f32::NAN;
        let out = trainer
            .train(&req, &mut |p| match p {
                TrainingProgress::Caching { current, total } => {
                    if current == 1 || current == total {
                        println!("caching {current}/{total}");
                    }
                }
                TrainingProgress::Training { step, total, loss } => {
                    last_loss = loss;
                    if step == 1 || step % 25 == 0 || step == total {
                        println!("step {step}/{total} loss {loss:.4}");
                    }
                }
                TrainingProgress::Checkpoint { step } => println!("checkpoint @ {step}"),
                _ => {}
            })
            .expect("train");

        println!(
            "DONE adapter={} steps={} final_loss={} (last_seen={last_loss:.4})",
            out.adapter_path.display(),
            out.steps,
            out.final_loss
        );
        assert!(out.adapter_path.exists(), "adapter file written");
        assert!(out.final_loss.is_finite(), "final loss finite");
    }

    /// Generate the fixed prompt grid into `GEN_DIR` (+ a `prompts.json` map the eval harness
    /// reads for prompt adherence). `ADAPTER_PATH` set → use that LoRA at `ADAPTER_SCALE`;
    /// unset → the no-LoRA baseline. `GEN_SEEDS` is a comma list (default `42,1234`).
    #[test]
    #[ignore = "research: real Z-Image generation; needs the base snapshot + Metal. Set GEN_DIR (+ optional ADAPTER_PATH)"]
    fn generate_zimage_grid() {
        let gen_dir = require_env("GEN_DIR");
        std::fs::create_dir_all(&gen_dir).unwrap();
        let trigger = env_str("TRIGGER", "sks man");
        let res = env_u32("GEN_RES", 1024);
        let steps = std::env::var("GEN_STEPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        let seeds: Vec<u64> = env_str("GEN_SEEDS", "42,1234")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        assert!(!seeds.is_empty(), "GEN_SEEDS parsed empty");

        let spec =
            match std::env::var("ADAPTER_PATH") {
                Ok(p) => {
                    let scale = env_f32("ADAPTER_SCALE", 0.8);
                    println!("generating WITH adapter {p} scale {scale}");
                    LoadSpec::new(WeightsSource::Dir(base_dir())).with_adapters(vec![
                        AdapterSpec::new(PathBuf::from(p), scale, AdapterKind::Lora),
                    ])
                }
                Err(_) => {
                    println!("generating NO-LoRA baseline");
                    LoadSpec::new(WeightsSource::Dir(base_dir()))
                }
            };

        // Direct provider constructor, NOT the gen_core registry: the worker's test build also
        // links a stub `z_image_turbo` registration (engines.rs derivation test), so the registry
        // `load` panics on the duplicate id. `mlx_gen_z_image::load` is the same T2I generator,
        // bypassing the registry. (The trainer registry has no such stub — `load_trainer` is fine.)
        let generator = mlx_gen_z_image::load(&spec).expect("load z_image_turbo generator");

        let mut prompts_json: BTreeMap<String, String> = BTreeMap::new();
        let mut written = 0usize;
        for (id, template) in PROMPT_GRID {
            let prompt = template.replace("{trigger}", &trigger);
            prompts_json.insert((*id).to_string(), prompt.clone());
            for seed in &seeds {
                let req = GenerationRequest {
                    prompt: prompt.clone(),
                    width: res,
                    height: res,
                    seed: Some(*seed),
                    steps,
                    ..Default::default()
                };
                let out = generator.generate(&req, &mut |_p| {}).expect("generate");
                let img = match out {
                    GenerationOutput::Images(mut v) => v.swap_remove(0),
                    GenerationOutput::Video { .. } => panic!("expected an image"),
                };
                write_png(&img, &gen_dir.join(format!("{id}_{seed}.png")));
                written += 1;
            }
            println!("{id}: {} seeds", seeds.len());
        }

        let prompts_path = gen_dir.join("prompts.json");
        std::fs::write(
            &prompts_path,
            serde_json::to_string_pretty(&prompts_json).unwrap(),
        )
        .unwrap();
        println!(
            "DONE wrote {written} images + {} to {}",
            prompts_path.display(),
            gen_dir.display()
        );
        assert_eq!(written, PROMPT_GRID.len() * seeds.len());
    }
}
