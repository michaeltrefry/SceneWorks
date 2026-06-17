use super::*;
use serde_json::json;

fn request(value: Value) -> ImageRequest {
    ImageRequest::from_payload(&value.as_object().cloned().unwrap())
}

#[test]
fn render_and_save_writes_png_and_contract_fact() {
    let dir = tempfile::tempdir().unwrap();
    let project_path = dir.path();
    std::fs::create_dir_all(project_path.join("assets").join("images")).unwrap();
    // Distinct dimensions (>= the 256 min, so they survive clamping) also catch a
    // width/height transpose in the encoder.
    let req = request(json!({
        "projectId": "p", "model": "z_image_turbo", "prompt": "Mist over hills",
        "count": 2, "width": 320, "height": 256, "seed": 101,
        "stylePreset": "cinematic", "modelManifestEntry": { "family": "z-image" }
    }));
    let plan = ImagePlan::new(&req);

    let seed = resolve_seed(&req, 0);
    let pixels = stub_rgb8(req.width, req.height, seed);
    let fact = write_image_asset(
        &plan,
        0,
        seed,
        req.width,
        req.height,
        pixels,
        STUB_ADAPTER,
        stub_raw_settings(&req),
        project_path,
    )
    .unwrap();

    let media_rel = fact.get("mediaPath").and_then(Value::as_str).unwrap();
    assert!(media_rel.starts_with(&format!("assets/images/{}/", plan.genset_id)));
    assert!(media_rel.ends_with("_0001.png"));
    let decoded = image::open(project_path.join(media_rel)).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (320, 256));

    for key in [
        "assetId",
        "mediaPath",
        "mimeType",
        "width",
        "height",
        "normalizedWidth",
        "normalizedHeight",
        "count",
        "family",
        "seed",
        "displayName",
        "createdAt",
        "mode",
        "model",
        "adapter",
        "prompt",
        "negativePrompt",
        "loras",
        "stylePreset",
        "characterId",
        "characterLookId",
        "sourceAssetId",
        "rawAdapterSettings",
    ] {
        assert!(fact.contains_key(key), "fact missing key {key}");
    }
    assert_eq!(fact["adapter"], json!("procedural_preview"));
    assert_eq!(fact["family"], json!("z-image"));
    assert_eq!(fact["seed"], json!(101));
    assert_eq!(fact["width"], json!(320));
    assert_eq!(fact["displayName"], json!("Mist over hills #1"));
    assert_eq!(
        fact["rawAdapterSettings"]["realModelInference"],
        json!(false)
    );
}

#[test]
fn resolve_seed_matches_python_precedence() {
    // base seed wins (seed + index), even over an explicit seeds list.
    let base = request(json!({ "projectId": "p", "seed": 100, "seeds": [7, 8] }));
    assert_eq!(resolve_seed(&base, 0), 100);
    assert_eq!(resolve_seed(&base, 2), 102);
    // explicit per-image seeds when no base seed.
    let listed = request(json!({ "projectId": "p", "seeds": [7, 8] }));
    assert_eq!(resolve_seed(&listed, 1), 8);
    // deterministic hash fallback (same prompt+index -> same seed).
    let none = request(json!({ "projectId": "p", "prompt": "hello" }));
    assert_eq!(resolve_seed(&none, 0), resolve_seed(&none, 0));
    assert_ne!(resolve_seed(&none, 0), resolve_seed(&none, 1));
}

#[test]
fn distinct_seeds_produce_distinct_pixels() {
    let a = stub_rgb8(8, 8, 1);
    let b = stub_rgb8(8, 8, 5000);
    assert_eq!(a.len(), 8 * 8 * 3);
    assert_ne!(a, b);
}

#[test]
fn streaming_result_carries_facts_for_api_persistence() {
    let plan = ImagePlan::new(&request(
        json!({ "projectId": "p", "prompt": "x", "count": 1 }),
    ));
    let writes = vec![json!({ "assetId": "a1" })];
    let result = streaming_result(&plan, &writes);
    assert_eq!(result["generationSetId"], json!(plan.genset_id));
    assert_eq!(result["assetWrites"].as_array().map(Vec::len), Some(1));
    assert!(result.contains_key("generationSet"));
}

#[test]
fn backend_label_defaults_empty_to_cpu() {
    assert_eq!(backend_label("mlx"), "mlx");
    assert_eq!(backend_label(""), "cpu");
}

#[cfg(target_os = "macos")]
#[test]
fn quant_mapping_defaults_to_q8_and_maps_bits() {
    use gen_core::Quant;
    let default = request(json!({ "projectId": "p" }));
    assert!(matches!(
        resolve_quant(&default),
        (Some(Quant::Q8), Some(8))
    ));
    let q4 = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 4 } }));
    assert!(matches!(resolve_quant(&q4), (Some(Quant::Q4), Some(4))));
    let dense = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 0 } }));
    assert!(matches!(resolve_quant(&dense), (None, None)));
    let six = request(json!({ "projectId": "p", "advanced": { "mlxQuantize": 6 } }));
    assert!(matches!(resolve_quant(&six), (Some(Quant::Q8), Some(8))));
}

#[cfg(target_os = "macos")]
#[test]
fn steps_default_is_family_default_and_clamps() {
    let zimage = mlx_model("z_image_turbo").unwrap();
    let schnell = mlx_model("flux_schnell").unwrap();
    let dev = mlx_model("flux_dev").unwrap();
    // Family defaults (Python MODEL_TARGETS parity): z-image 8, schnell 4, dev 28.
    assert_eq!(
        resolve_steps(&request(json!({ "projectId": "p" })), &zimage),
        8
    );
    assert_eq!(
        resolve_steps(&request(json!({ "projectId": "p" })), &schnell),
        4
    );
    assert_eq!(
        resolve_steps(&request(json!({ "projectId": "p" })), &dev),
        28
    );
    // advanced.steps overrides, clamped to 1..=80.
    assert_eq!(
        resolve_steps(
            &request(json!({ "projectId": "p", "advanced": { "steps": 200 } })),
            &dev
        ),
        80
    );
    assert_eq!(
        resolve_steps(
            &request(json!({ "projectId": "p", "advanced": { "steps": 12 } })),
            &schnell
        ),
        12
    );
}

#[cfg(target_os = "macos")]
#[test]
fn mlx_model_table_maps_known_families() {
    assert_eq!(
        mlx_model("z_image_turbo").unwrap().engine_id(),
        "z_image_turbo"
    );
    assert_eq!(
        mlx_model("flux_schnell").unwrap().engine_id(),
        "flux1_schnell"
    );
    assert_eq!(mlx_model("flux_dev").unwrap().engine_id(), "flux1_dev");
    assert_eq!(mlx_model("flux_dev").unwrap().adapter_label(), "mlx_flux");
    let qwen = mlx_model("qwen_image").unwrap();
    assert_eq!(qwen.engine_id(), "qwen_image");
    assert_eq!(qwen.adapter_label(), "mlx_qwen");
    assert_eq!(qwen.default_steps(), 20);
    assert!(qwen.supports_guidance() && qwen.supports_negative_prompt());
    // All three FLUX.2-klein variants share the engine's single txt2img model.
    for id in [
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
    ] {
        let m = mlx_model(id).unwrap();
        assert_eq!(m.engine_id(), "flux2_klein_9b");
        assert_eq!(m.adapter_label(), "mlx_flux2");
        assert!(m.supports_guidance() && !m.supports_negative_prompt());
    }
    // Distilled variants are 4-step; the undistilled true_v2 is 24-step.
    assert_eq!(mlx_model("flux2_klein_9b").unwrap().default_steps(), 4);
    assert_eq!(mlx_model("flux2_klein_9b_kv").unwrap().default_steps(), 4);
    assert_eq!(
        mlx_model("flux2_klein_9b_true_v2").unwrap().default_steps(),
        24
    );
    // FLUX.2-dev (epic 5914): its OWN engine model (not a klein weight variant), embedded
    // distilled guidance (guidance scalar, no negative prompt — like klein but ~28 steps /
    // guidance 4.0). Shares the `mlx_flux2` adapter.
    let dev = mlx_model("flux2_dev").unwrap();
    assert_eq!(dev.engine_id(), "flux2_dev");
    assert_eq!(dev.adapter_label(), "mlx_flux2");
    assert_eq!(dev.default_repo(), "black-forest-labs/FLUX.2-dev");
    assert_eq!(dev.default_steps(), 28);
    assert_eq!(dev.default_guidance(), 4.0);
    assert!(dev.supports_guidance() && !dev.supports_negative_prompt());
    // SDXL + the realvisxl finetune share the single `sdxl` engine model (real CFG).
    for id in ["sdxl", "realvisxl"] {
        let m = mlx_model(id).unwrap();
        assert_eq!(m.engine_id(), "sdxl");
        assert_eq!(m.adapter_label(), "mlx_sdxl");
        assert_eq!(m.default_steps(), 30);
        assert!(m.supports_guidance() && m.supports_negative_prompt());
    }
    assert!(mlx_model("instantid_sdxl").is_none());

    // SenseNova-U1 (sc-3900): base + 8-step distill `_fast`, each its own engine id. Uses
    // text guidance (4.0 base / 1.0 fast) AND image guidance (true_cfg) but advertises NO
    // negative prompt — so it is NOT a `uses_true_cfg` family (see `uses_true_cfg`).
    let base = mlx_model("sensenova_u1_8b").unwrap();
    assert_eq!(base.engine_id(), "sensenova_u1_8b");
    assert_eq!(base.default_repo(), "sensenova/SenseNova-U1-8B-MoT");
    assert_eq!(base.default_steps(), 50);
    assert_eq!(base.default_guidance(), 4.0);
    assert_eq!(base.adapter_label(), "mlx_sensenova");
    assert!(base.supports_guidance() && !base.supports_negative_prompt());
    assert!(
        !uses_true_cfg(&base),
        "dual-CFG, not a true-CFG-only family"
    );
    let fast = mlx_model("sensenova_u1_8b_fast").unwrap();
    assert_eq!(fast.engine_id(), "sensenova_u1_8b_fast");
    assert_eq!(fast.default_steps(), 8);
    assert_eq!(fast.default_guidance(), 1.0);
    assert_eq!(fast.adapter_label(), "mlx_sensenova");
    assert!(fast.supports_guidance() && !fast.supports_negative_prompt());

    // Bernini still-image companion (sc-5424): the image-typed `bernini_image` id maps to the SAME
    // `bernini` engine the video id uses (two ids, one engine — like z_image_edit → z_image_turbo).
    // Engine defaults 40 steps / guidance 4.0 (`FullDefaults`); standard guidance family (the
    // descriptor advertises both guidance and a negative prompt).
    let bernini_image = mlx_model("bernini_image").unwrap();
    assert_eq!(bernini_image.engine_id(), "bernini");
    assert_eq!(bernini_image.adapter_label(), "mlx_bernini");
    assert_eq!(bernini_image.default_steps(), 40);
    assert_eq!(bernini_image.default_guidance(), 4.0);
    assert!(bernini_image.supports_guidance() && bernini_image.supports_negative_prompt());
}

#[cfg(target_os = "macos")]
#[test]
fn sensenova_dual_cfg_and_shift_resolve_per_mode() {
    // Image guidance (true_cfg): edit default 1.0, character default 1.5, override via
    // imageGuidanceScale, floored at 1.0.
    assert_eq!(
        resolve_sensenova_img_cfg(&request(json!({ "projectId": "p", "mode": "edit_image" }))),
        1.0
    );
    assert_eq!(
        resolve_sensenova_img_cfg(&request(
            json!({ "projectId": "p", "mode": "character_image" })
        )),
        1.5
    );
    assert_eq!(
        resolve_sensenova_img_cfg(&request(json!({
            "projectId": "p", "mode": "character_image",
            "advanced": { "imageGuidanceScale": 2.5 }
        }))),
        2.5
    );
    assert_eq!(
        resolve_sensenova_img_cfg(&request(json!({
            "projectId": "p", "mode": "edit_image",
            "advanced": { "imageGuidanceScale": 0.2 }
        }))),
        1.0,
        "img cfg is floored at 1.0"
    );
    // Timestep shift: default 3.0, schedulerShift (or legacy timestepShift) overrides,
    // non-positive falls back to 3.0.
    assert_eq!(
        resolve_sensenova_timestep_shift(&request(json!({ "projectId": "p" }))),
        3.0
    );
    assert_eq!(
        resolve_sensenova_timestep_shift(&request(json!({
            "projectId": "p", "advanced": { "schedulerShift": 4.5 }
        }))),
        4.5
    );
    assert_eq!(
        resolve_sensenova_timestep_shift(&request(json!({
            "projectId": "p", "advanced": { "timestepShift": 2.0 }
        }))),
        2.0
    );
    assert_eq!(
        resolve_sensenova_timestep_shift(&request(json!({
            "projectId": "p", "advanced": { "schedulerShift": 0.0 }
        }))),
        3.0
    );
    // 32-cell snap, clamped to [256, 2048].
    assert_eq!(sensenova_dim(1536), 1536); // already 32-aligned
    assert_eq!(sensenova_dim(1000), 1024); // rounds up to the next multiple of 32
    assert_eq!(sensenova_dim(100), 256); // clamps to the minimum
    assert_eq!(sensenova_dim(5000), 2048); // clamps to the maximum
}

#[cfg(target_os = "macos")]
#[test]
fn sensenova_edit_available_needs_a_reference() {
    // Plain T2I (no reference) is NOT the edit path — it rides the base mlx path.
    assert!(!sensenova_edit_available(
        &request(json!({ "projectId": "p", "model": "sensenova_u1_8b", "prompt": "a fox" })),
        &Settings::from_env()
    ));
    // edit_image needs a source; character_image needs a reference. (Weights may be
    // absent in CI, so only assert the negative/structural cases here.)
    assert!(qwen_edit_reference_ids(&request(json!({
        "projectId": "p", "model": "sensenova_u1_8b", "mode": "edit_image", "sourceAssetId": "s"
    })))
    .contains(&"s".to_owned()));
    assert!(qwen_edit_reference_ids(&request(json!({
        "projectId": "p", "model": "sensenova_u1_8b", "mode": "character_image",
        "referenceAssetId": "r"
    })))
    .contains(&"r".to_owned()));
}

/// Real-weights smoke: SenseNova-U1 it2i. Loads `sensenova_u1_8b` (the ~35GB
/// `sensenova/SenseNova-U1-8B-MoT` snapshot) and generates one image conditioned on a synthetic
/// reference via the worker's dual-CFG it2i path (text `guidance` + image `true_cfg` +
/// `scheduler_shift`). The worker-level entry for the sc-3900 parity gate (component +
/// early-step + coherence, not pixel bit-parity — the port runs f32 vs the bf16 reference).
/// Needs the HF cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_it2i_real_weights`.
/// Uses 8 steps + 512² for speed (the production base default is 50 steps).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + Metal device"]
fn sensenova_it2i_real_weights_generates_one_image() {
    let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
    let generator = load_engine(
        mlx_model("sensenova_u1_8b").unwrap().engine_id(),
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let reference = gen_core::Image {
        width: 512,
        height: 512,
        pixels: stub_rgb8(512, 512, 7),
    };
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = sensenova_edit_generate_one(
        generator.as_ref(),
        "make it a watercolor painting",
        512,
        512,
        42,
        8,
        Some(4.0), // text CFG
        1.0,       // image CFG (edit default)
        3.0,       // timestep shift
        build_edit_conditioning(std::slice::from_ref(&reference)),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights regression for sc-5567: a SenseNova-U1 **8B Fast** count=2 batch was
/// OOM-killed (SIGKILL) on image 1 because nothing released MLX's freed-buffer cache
/// between batch images, so image 0's retained working set stacked on top of the dense
/// weights and crossed the unified-memory ceiling. This loads the distilled `_fast`
/// engine once (production shape: load-once + per-image), generates two images with the
/// production [`release_gen_cache_between_items`] call between them, and asserts the seam
/// actually frees the cache without evicting the live weights — and that image 1 then
/// completes. The OOM itself is a process-level SIGKILL a test can't catch, so we prove
/// the *mechanism* (cache released, weights retained) plus the peak readout. Run under
/// `/usr/bin/time -l` on a Mac (peak is wired Metal memory, not in `ps` RSS) to compare
/// count=1 vs count=2 footprint:
/// `cargo test -p sceneworks-worker --lib -- --ignored sensenova_fast_batch_releases_cache --nocapture`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real SenseNova-U1-8B-MoT weights (~35GB) + the _fast distill LoRA + a Metal device"]
fn sensenova_fast_batch_releases_cache_between_images() {
    let snapshot = hf_snapshot("models--sensenova--SenseNova-U1-8B-MoT");
    let generator = load_engine(
        mlx_model("sensenova_u1_8b_fast").unwrap().engine_id(),
        snapshot,
        Some(gen_core::Quant::Q8), // production default (resolve_quant)
        Vec::new(),
        None,
    )
    .unwrap();
    // The committed smoke runs at 512² for speed; set `SC5567_DIM` (e.g. 2048) to
    // reproduce the production-resolution footprint the user OOM'd at — the retained
    // per-image cache scales with the activation working set, so the release matters
    // far more there.
    let dim: u32 = std::env::var("SC5567_DIM")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(512);
    let reference = gen_core::Image {
        width: dim,
        height: dim,
        pixels: stub_rgb8(dim, dim, 7),
    };
    let conditioning = build_edit_conditioning(std::slice::from_ref(&reference));
    let cancel = gen_core::CancelFlag::new();

    let gib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gen_one = |seed: i64| {
        sensenova_edit_generate_one(
            generator.as_ref(),
            "make it a watercolor painting",
            dim,
            dim,
            seed,
            8,         // _fast: 8-step distill
            Some(1.0), // _fast text CFG
            1.0,       // image CFG
            3.0,       // timestep shift
            conditioning.clone(),
            &cancel,
            &mut |_| {},
        )
        .unwrap()
    };

    mlx_rs::memory::reset_peak_memory();

    // Image 0 — succeeds today (the bug was image 1).
    let (w0, h0, px0) = gen_one(42);
    assert_eq!((w0, h0), (dim, dim));
    assert_eq!(px0.len() as u32, dim * dim * 3);

    // After image 0 returns, its transient arrays are dropped → they sit in MLX's
    // freed-buffer cache; the only live arrays are the model weights.
    let cache_before = mlx_rs::memory::get_cache_memory();
    let active_before = mlx_rs::memory::get_active_memory();

    // The production between-images release (the fix).
    release_gen_cache_between_items();

    let cache_after = mlx_rs::memory::get_cache_memory();
    let active_after = mlx_rs::memory::get_active_memory();
    eprintln!(
        "sc-5567 release: cache {:.2}->{:.2} GiB, active {:.2}->{:.2} GiB, peak {:.2} GiB",
        gib(cache_before),
        gib(cache_after),
        gib(active_before),
        gib(active_after),
        gib(mlx_rs::memory::get_peak_memory()),
    );

    // The seam returns retained buffers to the OS...
    assert!(
        cache_after < cache_before || cache_before == 0,
        "clear_cache should shrink the freed-buffer cache ({cache_before} -> {cache_after} bytes)"
    );
    // ...without evicting the live model weights (the whole point — image 1 must reuse them).
    assert!(
        active_after as f64 >= active_before as f64 * 0.8,
        "live weights must survive the cache release ({active_before} -> {active_after} bytes)"
    );

    // Image 1 — the image that OOM-killed the worker pre-fix. Completing proves the
    // count=2 batch path is sound once the cache is released between images.
    let (w1, h1, px1) = gen_one(43);
    assert_eq!((w1, h1), (dim, dim));
    assert_eq!(px1.len() as u32, dim * dim * 3);
    assert!(px1.windows(2).any(|w| w[0] != w[1]));
    eprintln!(
        "sc-5567 count=2 OK: peak footprint {:.2} GiB",
        gib(mlx_rs::memory::get_peak_memory())
    );
}

/// Bernini still-image companion (epic 4699 / sc-5424) pure mapping: the SceneWorks image mode →
/// engine task string and the Q4-default quant resolver. Runs in CI on Mac (no weights).
#[cfg(target_os = "macos")]
#[test]
fn bernini_image_task_and_quant_mapping() {
    // `edit_image` → i2i; everything else (text_to_image / empty / anything) → t2i.
    assert_eq!(bernini_image_engine_task("edit_image"), "i2i");
    assert_eq!(bernini_image_engine_task("text_to_image"), "t2i");
    assert_eq!(bernini_image_engine_task(""), "t2i");
    assert_eq!(bernini_image_engine_task("character_image"), "t2i");
    // Q4 default (NOT the generic image Q8 default); `mlxQuantize` selects Q8 / bf16.
    let quant = |bits: Option<i64>| {
        let advanced = match bits {
            Some(b) => json!({ "mlxQuantize": b }),
            None => json!({}),
        };
        resolve_bernini_image_quant(&request(json!({
            "projectId": "p", "model": "bernini_image", "prompt": "p", "advanced": advanced,
        })))
    };
    let (q, bits) = quant(None);
    assert!(
        matches!(q, Some(gen_core::Quant::Q4)) && bits == Some(4),
        "default → Q4"
    );
    let (q, bits) = quant(Some(4));
    assert!(matches!(q, Some(gen_core::Quant::Q4)) && bits == Some(4));
    let (q, bits) = quant(Some(8));
    assert!(matches!(q, Some(gen_core::Quant::Q8)) && bits == Some(8));
    let (q, bits) = quant(Some(0));
    assert!(q.is_none() && bits.is_none(), "<=0 → bf16 dense");
    // The dedicated route claims only the `bernini_image` id (the model check short-circuits before
    // any weight resolution), so a different model never diverts here.
    assert!(!bernini_image_available(
        &request(json!({ "projectId": "p", "model": "z_image_turbo", "prompt": "p" })),
        &Settings::from_env()
    ));
}

/// Resolve the Bernini MLX snapshot dir for the real-weight smokes: env override → the local
/// mlx-gen-models conversion caches / app-managed dir → the turnkey `SceneWorks/bernini-mlx` HF-cache
/// snapshot. Mirrors the video test's `bernini_dir()` candidate list. `None` ⇒ skip (weights live
/// outside CI).
#[cfg(target_os = "macos")]
fn bernini_image_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_BERNINI_DIR") {
        let path = PathBuf::from(dir.trim());
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let home = dirs_home();
    for rel in [
        ".cache/mlx-gen-models/bernini-mlx-upload",
        ".cache/mlx-gen-models/bernini_full_mlx_bf16",
        "Library/Application Support/SceneWorks/data/models/mlx/bernini",
    ] {
        let path = home.join(rel);
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let snaps = home.join(".cache/huggingface/hub/models--SceneWorks--bernini-mlx/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").is_file())
}

/// Real-weights smoke: Bernini t2i (sc-5424 / sc-4709). Loads `bernini` through the SAME
/// `load_engine` → `gen_core::load("bernini")` seam the worker image path uses (proving the
/// `mlx_gen_bernini` force-link survived in the worker binary), then drives the dedicated
/// `bernini_image_generate_one` with `frames:1` + `video_mode:"t2i"`, asserting it returns a single
/// RGB8 still with denoise progress. 8 steps + 512² for speed (~44 GB Q4 peak). Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored bernini_image_t2i_real_weights`.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
#[test]
fn bernini_image_t2i_real_weights_generates_one_image() {
    let Some(dir) = bernini_image_dir() else {
        eprintln!("skipping bernini_image_t2i_real_weights: no Bernini MLX snapshot found");
        return;
    };
    let generator =
        load_engine("bernini", dir, Some(gen_core::Quant::Q4), Vec::new(), None).unwrap();
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = bernini_image_generate_one(
        generator.as_ref(),
        "a weathered lighthouse on a rocky cliff at golden hour, photorealistic, cinematic",
        None,
        512,
        512,
        42,
        8,
        Some(4.0),
        "t2i",
        Vec::new(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!(pixels.len(), (w * h * 3) as usize, "RGB8-sized buffer");
    assert!(w >= 256 && h >= 256, "plausible still dimensions");
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(
        pixels.windows(2).any(|x| x[0] != x[1]),
        "non-constant image"
    );
}

/// Real-weights smoke: Bernini i2i (sc-5424). Same load seam, but `video_mode:"i2i"` + a synthetic
/// source as the engine's `Conditioning::Reference` (the planner ViT/VAE-encodes it; the worker does
/// no pre-fit). Asserts the edit returns a single RGB8 still. Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored bernini_image_i2i_real_weights`.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Bernini snapshot; run manually on a Mac with SceneWorks/bernini-mlx present"]
#[test]
fn bernini_image_i2i_real_weights_generates_one_image() {
    let Some(dir) = bernini_image_dir() else {
        eprintln!("skipping bernini_image_i2i_real_weights: no Bernini MLX snapshot found");
        return;
    };
    let generator =
        load_engine("bernini", dir, Some(gen_core::Quant::Q4), Vec::new(), None).unwrap();
    let source = gen_core::Image {
        width: 512,
        height: 512,
        pixels: stub_rgb8(512, 512, 7),
    };
    let conditioning = vec![gen_core::Conditioning::Reference {
        image: source,
        strength: None,
    }];
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = bernini_image_generate_one(
        generator.as_ref(),
        "make it a watercolor painting, soft pastel palette",
        None,
        512,
        512,
        42,
        8,
        Some(4.0),
        "i2i",
        conditioning,
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!(pixels.len(), (w * h * 3) as usize, "RGB8-sized buffer");
    assert!(w >= 256 && h >= 256, "plausible still dimensions");
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(
        pixels.windows(2).any(|x| x[0] != x[1]),
        "non-constant image"
    );
}

/// sc-3344 parity gate (worker path): drive the native `pulid_flux` registry generator through the
/// SAME load seam the worker uses (`load_engine` → `gen_core::load("pulid_flux")` with the engine's
/// env-var weight resolution filled from local caches) and confirm it produces an identity-preserving
/// render. Validates the actual cutover integration — the env-var seam + `Conditioning::Reference {
/// strength = idWeight }` mapping — not just the engine. Asserts ArcFace cosine softly (the engine
/// envelope is ≈0.68 @512²/20-step, scaling to the torch sc-2012 baseline ≈0.80 @1024²/30-step).
///
/// Weights: FLUX.1-dev + guozinan/PuLID in the HF cache; the converted EVA + face stack
/// (`eva02_clip_l_336`/`scrfd_10g`/`arcface_iresnet100`/`bisenet_parsing`) in a bundle dir
/// (`SCENEWORKS_PULID_WEIGHTS` overrides, default the mlx-gen `tools/golden`); a reference face
/// (`SCENEWORKS_TEST_FACE` overrides). Metal device. On demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored pulid_flux_real_weights --nocapture`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs FLUX.1-dev + PuLID + converted EVA/face weights + Metal device"]
fn pulid_flux_real_weights_holds_identity() {
    let flux_base = hf_snapshot("models--black-forest-labs--FLUX.1-dev");
    let pulid_adapter = hf_snapshot("models--guozinan--PuLID").join(PULID_ADAPTER_FILE);
    let bundle = std::env::var("SCENEWORKS_PULID_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join("Repos/mlx-gen/tools/golden"));
    let eva = bundle.join(PULID_EVA_FILE);
    let scrfd = bundle.join(INSTANTID_SCRFD_FILE);
    let arcface = bundle.join(INSTANTID_ARCFACE_FILE);
    let bisenet = bundle.join(PULID_BISENET_FILE);
    for p in [&pulid_adapter, &eva, &scrfd, &arcface, &bisenet] {
        assert!(p.exists(), "missing PuLID weight: {}", p.display());
    }
    // Fill the engine's env-var weight seam from the local caches (exactly what
    // `generate_pulid_flux_stream` does before the cached load).
    std::env::set_var("PULID_FLUX_WEIGHTS", &pulid_adapter);
    std::env::set_var("PULID_EVA_WEIGHTS", &eva);
    std::env::set_var("PULID_FACE_WEIGHTS_DIR", &bundle);

    // Reference face: a portrait PNG (`SCENEWORKS_TEST_FACE`) if present, else the real face image
    // embedded in the face-align golden (`<bundle>/face_align_goldens.safetensors`, the same
    // reference the engine e2e uses — always present alongside the converted weights).
    let reference = match std::env::var("SCENEWORKS_TEST_FACE")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
    {
        Some(face_path) => {
            let decoded = image::open(&face_path)
                .unwrap_or_else(|e| panic!("reference face {face_path}: {e}"))
                .to_rgb8();
            gen_core::Image {
                width: decoded.width(),
                height: decoded.height(),
                pixels: decoded.into_raw(),
            }
        }
        None => {
            let g = Weights::from_file(bundle.join("face_align_goldens.safetensors"))
                .expect("face_align_goldens.safetensors in the bundle (reference face)");
            let a = g.require("image").unwrap();
            let sh = a.shape();
            let pixels: Vec<u8> = a
                .try_as_slice::<i32>()
                .unwrap()
                .iter()
                .map(|&v| v as u8)
                .collect();
            gen_core::Image {
                width: sh[1] as u32,
                height: sh[0] as u32,
                pixels,
            }
        }
    };

    // Reference ArcFace embedding (native face stack — detection + embedder only, no parser).
    let face = mlx_gen_face::FaceAnalysis::load(
        &Weights::from_file(&scrfd).unwrap(),
        &Weights::from_file(&arcface).unwrap(),
    )
    .unwrap();
    let ref_faces = face
        .analyze(
            &reference.pixels,
            reference.height as usize,
            reference.width as usize,
        )
        .unwrap();
    let ref_emb = ref_faces
        .first()
        .expect("a face in the reference image")
        .embedding
        .clone();

    // Load via the worker's registry seam at the production default quant (Q8 — near-lossless for
    // PuLID identity per engine sc-3076, the manifest `mlx.quantize`) + generate with the worker's
    // request mapping. The PuLID conditioning (EVA/IDFormer/CA) stays f32 regardless of quant.
    let generator = load_engine(
        "pulid_flux",
        flux_base,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let cancel = gen_core::CancelFlag::new();
    let req = gen_core::GenerationRequest {
        prompt: "a portrait photo of a person, headshot, looking at the camera".to_owned(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(20),
        guidance: Some(4.0),
        true_cfg: None,
        timestep_to_start_cfg: Some(4),
        conditioning: vec![gen_core::Conditioning::Reference {
            image: reference.clone(),
            strength: Some(1.0),
        }],
        cancel: cancel.clone(),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let image = match out {
        gen_core::GenerationOutput::Images(mut v) => v.remove(0),
        other => panic!("expected an image, got {other:?}"),
    };
    assert_eq!((image.width, image.height), (512, 512));

    let gen_faces = face
        .analyze(&image.pixels, image.height as usize, image.width as usize)
        .unwrap();
    let gf = gen_faces.first().expect("a face in the generated image");
    let cos = cosine(&gf.embedding, &ref_emb);
    println!(
        "PuLID-FLUX worker-path ArcFace cosine(generated, reference) = {cos:.4} \
         (engine ≈0.68 @512²/20-step → torch ≈0.80 @1024²/30-step)"
    );
    assert!(cos > 0.3, "identity not transferred (cosine {cos:.4})");
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_negative_prompt_only_for_true_cfg_families() {
    let qwen = mlx_model("qwen_image").unwrap();
    let flux = mlx_model("flux_dev").unwrap();
    // qwen (true CFG) passes a non-empty negative prompt; empty → None (fallback).
    assert_eq!(
        resolve_negative_prompt(
            &request(json!({ "projectId": "p", "negativePrompt": "blurry" })),
            &qwen
        ),
        Some("blurry".to_owned())
    );
    assert_eq!(
        resolve_negative_prompt(
            &request(json!({ "projectId": "p", "negativePrompt": "  " })),
            &qwen
        ),
        None
    );
    // Non-true-CFG families never pass a negative prompt (the engine rejects it).
    assert_eq!(
        resolve_negative_prompt(
            &request(json!({ "projectId": "p", "negativePrompt": "blurry" })),
            &flux
        ),
        None
    );
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_guidance_none_for_distilled_set_for_dev() {
    let schnell = mlx_model("flux_schnell").unwrap();
    let dev = mlx_model("flux_dev").unwrap();
    let zimage = mlx_model("z_image_turbo").unwrap();
    // Distilled variants take no guidance (the engine rejects Some on them).
    assert_eq!(
        resolve_guidance(&request(json!({ "projectId": "p" })), &schnell),
        None
    );
    assert_eq!(
        resolve_guidance(&request(json!({ "projectId": "p" })), &zimage),
        None
    );
    // flux dev defaults to 3.5, overridable via advanced.guidanceScale.
    assert_eq!(
        resolve_guidance(&request(json!({ "projectId": "p" })), &dev),
        Some(3.5)
    );
    assert_eq!(
        resolve_guidance(
            &request(json!({ "projectId": "p", "advanced": { "guidanceScale": 2.0 } })),
            &dev
        ),
        Some(2.0)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn adapter_id_reports_per_family_mlx_label() {
    assert_eq!(
        adapter_id(&request(json!({ "model": "z_image_turbo" }))),
        "mlx_z_image"
    );
    assert_eq!(
        adapter_id(&request(json!({ "model": "flux_schnell" }))),
        "mlx_flux"
    );
    assert_eq!(
        adapter_id(&request(json!({ "model": "flux_dev" }))),
        "mlx_flux"
    );
    assert_eq!(adapter_id(&request(json!({ "model": "sdxl" }))), "mlx_sdxl");
    // Kolors base T2I is MLX-routed now (sc-3875) → records the mlx_kolors adapter label.
    assert_eq!(
        adapter_id(&request(json!({ "model": "kolors" }))),
        "mlx_kolors"
    );
    // Lens / Lens-Turbo are MLX-routed (sc-5105) via the MODEL_TABLE registry families → both record
    // the shared mlx_lens adapter label.
    assert_eq!(adapter_id(&request(json!({ "model": "lens" }))), "mlx_lens");
    assert_eq!(
        adapter_id(&request(json!({ "model": "lens_turbo" }))),
        "mlx_lens"
    );
    // Bernini still-image companion (sc-5424) IS a MODEL_TABLE row (engine `bernini`), so unlike the
    // bespoke pulid/instantid routes `adapter_id` resolves its real per-set label.
    assert_eq!(
        adapter_id(&request(json!({ "model": "bernini_image" }))),
        "mlx_bernini"
    );
    // PuLID-FLUX (sc-3344) is MLX-routed but via a BESPOKE route (not the MODEL_TABLE registry
    // families), so `adapter_id` — which only resolves MODEL_TABLE rows — reports the stub label;
    // the real per-asset label (`mlx_pulid_flux`) is applied in `generate_pulid_flux_stream` via
    // `consume_gen_events`. Same shape as the InstantID bespoke route.
    assert_eq!(
        adapter_id(&request(json!({ "model": "pulid_flux_dev" }))),
        "procedural_preview"
    );
}

/// The Z-Image + FLUX.1 + Qwen-Image providers linked into the worker
/// self-registered via inventory.
#[cfg(target_os = "macos")]
#[test]
fn mlx_engine_registry_links_image_families() {
    let ids: Vec<&str> = gen_core::registry::generators()
        .map(|reg| (reg.descriptor)().id)
        .collect();
    for id in [
        "z_image_turbo",
        "flux1_schnell",
        "flux1_dev",
        "qwen_image",
        "qwen_image_control",
        "qwen_image_edit",
        "flux2_klein_9b",
        "sdxl",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
        "kolors",
        "lens",
        "lens_turbo",
        // Bernini still-image companion (sc-5424): the `Modality::Both` `bernini` engine must be
        // registry-linked from the IMAGE path too — proves `use mlx_gen_bernini as _;` in
        // image_jobs.rs keeps the `ModelRegistration` (the "no generator registered" trap).
        "bernini",
    ] {
        assert!(ids.contains(&id), "registry missing {id}");
    }
}

/// Resolve a HuggingFace cache snapshot dir for `models--<dir>` (test helper).
#[cfg(target_os = "macos")]
fn hf_snapshot(model_dir: &str) -> std::path::PathBuf {
    std::fs::read_dir(dirs_home().join(format!(".cache/huggingface/hub/{model_dir}/snapshots")))
        .expect("HF cache snapshots dir")
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a snapshot dir")
}

/// Load + generate one small image through the public mlx-gen path (test helper).
/// Keyed by SceneWorks model id — the engine id + step default come from the table,
/// so several SceneWorks ids can share one engine id (e.g. the FLUX.2 variants).
#[cfg(target_os = "macos")]
fn smoke_generate_one(
    sceneworks_id: &str,
    snapshot: std::path::PathBuf,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
) {
    let model = mlx_model(sceneworks_id).unwrap();
    let generator = load_engine(
        model.engine_id(),
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let steps = model.default_steps();
    let (w, h, pixels) = generate_one(
        generator.as_ref(),
        "a serene mountain lake at dawn",
        512,
        512,
        42,
        steps,
        guidance,
        negative_prompt,
        None,
        None,
        None,
        None,
        None,
        &PromptEnhance::default(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    // Not a flat image.
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights smoke: load + generate one small Z-Image image. Needs the HF cache
/// (`Tongyi-MAI/Z-Image-Turbo`) + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored zimage_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Z-Image weights + Metal device"]
fn zimage_real_weights_generates_one_image() {
    smoke_generate_one(
        "z_image_turbo",
        hf_snapshot("models--Tongyi-MAI--Z-Image-Turbo"),
        None,
        None,
    );
}

/// Real-weights smoke: load + generate one small FLUX.1-schnell image (4-step,
/// guidance-distilled). Needs the HF cache (`black-forest-labs/FLUX.1-schnell`) +
/// a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux_schnell_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real FLUX.1-schnell weights + Metal device"]
fn flux_schnell_real_weights_generates_one_image() {
    smoke_generate_one(
        "flux_schnell",
        hf_snapshot("models--black-forest-labs--FLUX.1-schnell"),
        None,
        None,
    );
}

/// Real-weights smoke: load + generate one small FLUX.1-dev image (guided, 28-step).
/// Needs the HF cache (`black-forest-labs/FLUX.1-dev`) + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux_dev_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real FLUX.1-dev weights + Metal device"]
fn flux_dev_real_weights_generates_one_image() {
    smoke_generate_one(
        "flux_dev",
        hf_snapshot("models--black-forest-labs--FLUX.1-dev"),
        Some(3.5),
        None,
    );
}

/// Real-weights smoke: load + generate one small Qwen-Image image (true CFG,
/// guidance 4.0 + a negative prompt). Needs the HF cache (`Qwen/Qwen-Image`) + a
/// Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored qwen_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Qwen-Image weights + Metal device"]
fn qwen_real_weights_generates_one_image() {
    smoke_generate_one(
        "qwen_image",
        hf_snapshot("models--Qwen--Qwen-Image"),
        Some(4.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke: FLUX.2-klein-9b (4-step distilled, guidance 1.0, no negative).
/// Needs the HF cache (`black-forest-labs/FLUX.2-klein-9B`) + a Metal device; run on
/// demand: `cargo test -p sceneworks-worker --lib -- --ignored flux2_klein_9b_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + Metal device"]
fn flux2_klein_9b_real_weights_generates_one_image() {
    smoke_generate_one(
        "flux2_klein_9b",
        hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b"),
        Some(1.0),
        None,
    );
}

/// Real-weights smoke: FLUX.2-klein-9b-kv txt2img (the separately-distilled checkpoint
/// loaded through the base txt2img loader). Needs the HF cache
/// (`black-forest-labs/FLUX.2-klein-9b-kv`) + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real FLUX.2-klein-9b-kv weights + Metal device"]
fn flux2_klein_9b_kv_real_weights_generates_one_image() {
    smoke_generate_one(
        "flux2_klein_9b_kv",
        hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b-kv"),
        Some(1.0),
        None,
    );
}

/// Real-weights smoke: Microsoft Lens base (epic 3164 / sc-5105) — 20-step, standard guidance 5.0 +
/// a negative prompt (the `mlx-gen-lens` descriptor is `supports_guidance` + `supports_negative_prompt`,
/// NOT true-CFG). Loads the `microsoft/Lens` snapshot dir (tokenizer/ text_encoder/ transformer/ vae/)
/// at the Q8 default (encoder MoE + DiT). Needs the HF cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored lens_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real microsoft/Lens weights + Metal device"]
fn lens_real_weights_generates_one_image() {
    smoke_generate_one(
        "lens",
        hf_snapshot("models--microsoft--Lens"),
        Some(5.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke: Lens-Turbo (the distilled 4-step / guidance 1.0 variant, ≈ no CFG) — same
/// architecture/weights tree as base Lens, different defaults. Loads the `microsoft/Lens-Turbo`
/// snapshot at the Q8 default. Needs the HF cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored lens_turbo_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real microsoft/Lens-Turbo weights + Metal device"]
fn lens_turbo_real_weights_generates_one_image() {
    smoke_generate_one(
        "lens_turbo",
        hf_snapshot("models--microsoft--Lens-Turbo"),
        Some(1.0),
        None,
    );
}

/// Real-weights smoke: Lens-Turbo at a **1024 resolution bucket** (sc-5105 acceptance — the Lens port
/// supports 1024/1440 × 9 ARs; the worker forwards `width`/`height` straight to the engine, which
/// validates ÷16). Confirms the larger latent + the 20B Q8 encoder run at a real bucket size on Mac,
/// not just the 512² basic smoke. Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored lens_turbo_real_weights_bucket`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real microsoft/Lens-Turbo weights + Metal device"]
fn lens_turbo_real_weights_bucket_resolution() {
    let model = mlx_model("lens_turbo").unwrap();
    let generator = load_engine(
        model.engine_id(),
        hf_snapshot("models--microsoft--Lens-Turbo"),
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let cancel = gen_core::CancelFlag::new();
    let (w, h, pixels) = generate_one(
        generator.as_ref(),
        "a serene mountain lake at dawn",
        1024,
        1024,
        7,
        model.default_steps(),
        Some(1.0),
        None,
        None,
        None,
        None,
        None,
        None,
        &PromptEnhance::default(),
        &cancel,
        &mut |_p| {},
    )
    .unwrap();
    assert_eq!((w, h), (1024, 1024));
    assert_eq!(pixels.len(), 1024 * 1024 * 3);
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights smoke: FLUX.2-klein-9b-true_v2 (wikeeyang undistilled fine-tune,
/// 24-step). Loads the locally-assembled converted diffusers dir under the SceneWorks
/// data dir (`models/mlx/flux2_klein_9b_true_v2`) via the modelPath seam — verifying
/// the converted-dir layout passthrough on the base `flux2_klein_9b` loader. Needs a
/// previously-converted dir + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a converted true_v2 dir + Metal device"]
fn flux2_klein_9b_true_v2_real_weights_generates_one_image() {
    let dir = dirs_home()
        .join("Library/Application Support/SceneWorks/data/models/mlx/flux2_klein_9b_true_v2");
    smoke_generate_one("flux2_klein_9b_true_v2", dir, Some(1.0), None);
}

/// The install-time pre-quantized Q4 dev snapshot dir (`models/mlx/flux2_dev`, assembled by the
/// `flux2_dev_quant` convert job); `SCENEWORKS_FLUX2_DEV_DIR` overrides it.
#[cfg(target_os = "macos")]
fn flux2_dev_dir() -> std::path::PathBuf {
    std::env::var("SCENEWORKS_FLUX2_DEV_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join("Library/Application Support/SceneWorks/data/models/mlx/flux2_dev")
        })
}

/// Worker-layer dev real-weight e2e (sc-5923): load `engine_id` from the converted Q4 dev dir and
/// generate one RGB8 image at an env-configurable size/steps (`SCENEWORKS_FLUX2_DEV_SIZE` /
/// `SCENEWORKS_FLUX2_DEV_STEPS`, default 512 / the dev default 28). For the footprint reconciliation,
/// set `SCENEWORKS_FLUX2_DEV_SIZE=1024` (the production default) and wrap the test binary in
/// `/usr/bin/time -l` to read the steady-state "peak memory footprint" vs the manifest `minMemoryGb`
/// (peak is activation-bound, reached within a couple of steps, so a low `STEPS` measures the same
/// ceiling faster). `conditioning` adds edit reference(s); empty = T2I. Proves the worker's
/// `gen_core::load` force-link reaches the dev loader, the packed snapshot loads + generates (the Q4
/// LoadSpec hint is a no-op on the already-packed weights), and the requested conditioning is
/// consumed (a real, non-flat RGB8 frame). Returns the decoded image.
#[cfg(target_os = "macos")]
fn dev_worker_generate(
    engine_id: &str,
    base: std::path::PathBuf,
    conditioning: Vec<gen_core::Conditioning>,
) -> Image {
    let size: u32 = std::env::var("SCENEWORKS_FLUX2_DEV_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let steps: u32 = std::env::var("SCENEWORKS_FLUX2_DEV_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let generator =
        load_engine(engine_id, base, Some(gen_core::Quant::Q4), Vec::new(), None).unwrap();
    let mut steps_seen = 0u32;
    let request = GenerationRequest {
        prompt: "a serene mountain lake at dawn".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(4.0),
        conditioning,
        ..Default::default()
    };
    let out = generator
        .generate(&request, &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        })
        .unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.remove(0),
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (size, size), "output dimensions");
    assert_eq!(
        img.pixels.len(),
        (size * size * 3) as usize,
        "RGB8 pixel count"
    );
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(
        img.pixels.windows(2).any(|w| w[0] != w[1]),
        "render is flat (degenerate)"
    );
    img
}

/// Real-weights smoke (sc-5921 / sc-5923 boundary): FLUX.2-dev T2I (embedded guidance, ~28-step)
/// through the worker `flux2_dev` engine path — proves the `mlx_gen_flux2` force-link reaches the dev
/// loader and the packed snapshot loads + generates. Env-sizable (default 512²; set
/// `SCENEWORKS_FLUX2_DEV_SIZE=1024` for the production-default footprint run, sc-5923). Needs a
/// previously-converted Q4 dir + a 128 GB Metal device. Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux2_dev_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a converted FLUX.2-dev Q4 dir + 128 GB Metal device"]
fn flux2_dev_real_weights_generates_one_image() {
    dev_worker_generate("flux2_dev", flux2_dev_dir(), Vec::new());
}

/// Real-weights smoke (sc-5923): FLUX.2-dev EDIT through the worker `flux2_dev_edit` engine path
/// (the `flux2_edit_engine_id("flux2_dev")` variant, sc-5919/5922). Loads the SAME converted Q4 dev
/// snapshot, then renders with a single `Conditioning::Reference` AND with a 2-image
/// `Conditioning::MultiReference` — the two reference shapes the worker `generate_flux2_edit_stream`
/// builds (grouped / pose-library `[skeleton, reference]`). Asserts each produces a real, non-flat
/// RGB8 frame and that the two differ (the references are actually consumed, not dropped). Env-sizable
/// like the T2I smoke; `SCENEWORKS_FLUX2_DEV_EDIT_REFS=1|2` isolates one pass for a clean per-pass
/// footprint measurement (sc-5923 — multi-reference adds the most sequence tokens, so it drives the
/// peak). Needs the converted Q4 dir + a 128 GB Metal device. Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux2_dev_edit_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a converted FLUX.2-dev Q4 dir + 128 GB Metal device"]
fn flux2_dev_edit_real_weights_generates_one_image() {
    let dir = flux2_dev_dir();
    let only: Option<u32> = std::env::var("SCENEWORKS_FLUX2_DEV_EDIT_REFS")
        .ok()
        .and_then(|s| s.parse().ok());

    let single = (only != Some(2)).then(|| {
        dev_worker_generate(
            "flux2_dev_edit",
            dir.clone(),
            vec![gen_core::Conditioning::Reference {
                image: synthetic_rgb(512, 512, |x, y| {
                    [((x * 255) / 512) as u8, ((y * 255) / 512) as u8, 96]
                }),
                strength: None,
            }],
        )
    });
    let multi = (only != Some(1)).then(|| {
        dev_worker_generate(
            "flux2_dev_edit",
            dir,
            vec![gen_core::Conditioning::MultiReference {
                images: vec![
                    synthetic_rgb(512, 512, |x, _| [((x * 255) / 512) as u8, 60, 200]),
                    synthetic_rgb(512, 512, |_, y| [40, ((y * 255) / 512) as u8, 180]),
                ],
            }],
        )
    });
    // When both passes ran (the default), the two reference shapes must drive different output.
    if let (Some(single), Some(multi)) = (single, multi) {
        assert_ne!(
            single.pixels, multi.pixels,
            "single- and multi-reference edits should differ (references consumed)"
        );
    }
}

/// Real-weights smoke: SDXL base (real CFG, guidance 7.0 + a negative prompt,
/// 30-step, Q8). Verifies the engine's SDXL quant default works (the Python
/// vendored path had no quant). Needs the HF cache
/// (`stabilityai/stable-diffusion-xl-base-1.0`) + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored sdxl_real_weights`.
/// SDXL native is 1024²; min is 512 — this smoke uses 512² for speed.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real SDXL weights + Metal device"]
fn sdxl_real_weights_generates_one_image() {
    smoke_generate_one(
        "sdxl",
        hf_snapshot("models--stabilityai--stable-diffusion-xl-base-1.0"),
        Some(7.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke: the RealVisXL finetune through the same `sdxl` engine model.
/// Needs the HF cache (`SG161222/RealVisXL_V5.0`) + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real RealVisXL weights + Metal device"]
fn realvisxl_real_weights_generates_one_image() {
    smoke_generate_one(
        "realvisxl",
        hf_snapshot("models--SG161222--RealVisXL_V5.0"),
        Some(7.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke (sc-3875 + sc-4764): load + generate one Kolors T2I image through the
/// worker's own `kolors` engine path. Exercises the ChatGLM3-6B encoder + SDXL-family U-Net +
/// SDXL VAE load AND, critically, proves the snapshot's overlaid `tokenizer/tokenizer.json`
/// (sc-4764) lets `KolorsTokenizer::from_dir` construct — the engine errors here without it.
/// Needs the HF cache (`Kwai-Kolors/Kolors-diffusers`, with the tokenizer overlay) + a Metal
/// device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored kolors_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Kolors weights (+ tokenizer.json overlay) + Metal device"]
fn kolors_real_weights_generates_one_image() {
    let snapshot = hf_snapshot("models--Kwai-Kolors--Kolors-diffusers");
    assert!(
        snapshot.join("tokenizer").join("tokenizer.json").exists(),
        "kolors snapshot is missing the overlaid tokenizer.json (sc-4764)"
    );
    smoke_generate_one(
        "kolors",
        snapshot,
        Some(5.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke (sc-4765): Kolors img2img through the base path — a `Reference` conditioning
/// (img2img init, no IP-Adapter loaded) on the `kolors` engine, the same `load_engine` + `generate_one`
/// seam `generate_stream` drives. Needs the Kolors snapshot (+ tokenizer overlay) + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Kolors weights (+ tokenizer.json overlay) + Metal device"]
fn kolors_real_weights_img2img_generates_one_image() {
    let base = hf_snapshot("models--Kwai-Kolors--Kolors-diffusers");
    let generator =
        load_engine("kolors", base, Some(gen_core::Quant::Q8), Vec::new(), None).unwrap();
    let source = synthetic_rgb(512, 512, |x, y| {
        [((x * 255) / 512) as u8, ((y * 255) / 512) as u8, 96]
    });
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = generate_one(
        generator.as_ref(),
        "an oil painting of a landscape",
        512,
        512,
        42,
        8,
        Some(5.0),
        Some("blurry, low quality".to_owned()),
        Some(&(source, 0.6)),
        None,
        None,
        None,
        None,
        &PromptEnhance::default(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.iter().any(|&p| p > 16) && pixels.iter().any(|&p| p < 239));
}

/// Real-weights smoke (sc-4767): Kolors IP-Adapter-Plus reference through the base path — a `Reference`
/// conditioning with the IP-Adapter loaded (`with_ip_adapter`), so the engine treats it as the image
/// prompt. Needs the Kolors + IP-Adapter-Plus snapshots + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Kolors + IP-Adapter-Plus weights + Metal device"]
fn kolors_real_weights_ip_adapter_generates_one_image() {
    let base = hf_snapshot("models--Kwai-Kolors--Kolors-diffusers");
    let ip = hf_snapshot("models--Kwai-Kolors--Kolors-IP-Adapter-Plus");
    let generator = load_engine(
        "kolors",
        base,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        Some(ip),
    )
    .unwrap();
    let reference = synthetic_rgb(512, 512, |x, y| {
        [((x * 255) / 512) as u8, 128, ((y * 255) / 512) as u8]
    });
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = generate_one(
        generator.as_ref(),
        "a portrait of a person",
        512,
        512,
        42,
        8,
        Some(5.0),
        Some("blurry, low quality".to_owned()),
        Some(&(reference, 0.6)),
        None,
        None,
        None,
        None,
        &PromptEnhance::default(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.iter().any(|&p| p > 16) && pixels.iter().any(|&p| p < 239));
}

/// A deterministic synthetic RGB [`Image`] from a per-pixel `(x, y) -> [r, g, b]` closure.
#[cfg(target_os = "macos")]
fn synthetic_rgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Image {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            pixels[i..i + 3].copy_from_slice(&f(x, y));
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Real-weights smoke (sc-4766 / engine sc-5012): load the combined Kolors pose spec (base + pose
/// ControlNet + IP-Adapter-Plus) and generate one image through the **dispatchable registry** path
/// (`gen_core::load("kolors", spec).generate(req)`) with BOTH a `Control` (skeleton) and a `Reference`
/// (identity) conditioning — the combined strict-pose tier. This is the worker-integration complement
/// to the engine's mlx-gen#403 test (which exercised the direct `Kolors` API, NOT the registry): it
/// proves the relaxed `validate_impl` admits Control+Reference when an IP-Adapter is loaded and the
/// combined `generate_impl` arm runs end-to-end through the same `gen_core::load` seam the worker's
/// `generate_kolors_control_stream` uses. Needs the three HF snapshots (Kolors-diffusers + the
/// tokenizer overlay, Kolors-ControlNet-Pose, Kolors-IP-Adapter-Plus) + a Metal device. On demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored kolors_real_weights_pose --nocapture`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Kolors + ControlNet-Pose + IP-Adapter-Plus weights + Metal device"]
fn kolors_real_weights_pose_generates_one_image() {
    let base = hf_snapshot("models--Kwai-Kolors--Kolors-diffusers");
    assert!(
        base.join("tokenizer").join("tokenizer.json").exists(),
        "kolors snapshot is missing the overlaid tokenizer.json (sc-4764)"
    );
    let controlnet = hf_snapshot("models--Kwai-Kolors--Kolors-ControlNet-Pose");
    let ip = hf_snapshot("models--Kwai-Kolors--Kolors-IP-Adapter-Plus");

    // The combined LoadSpec the worker's `kolors_control_spec` builds: base + ControlNet (Dir) +
    // IP-Adapter (Dir) + Q8.
    let spec = LoadSpec::new(WeightsSource::Dir(base))
        .with_control(WeightsSource::Dir(controlnet))
        .with_ip_adapter(WeightsSource::Dir(ip))
        .with_quant(gen_core::Quant::Q8);
    let generator = gen_core::load("kolors", &spec).expect("load combined kolors pose spec");

    let (w, h) = (512u32, 512u32);
    // Reference = the IP identity + img2img init; skeleton = the pose ControlNet. Synthetic content
    // (the combined denoise is numerically validated in mlx-gen#403; this asserts a coherent render).
    let reference = synthetic_rgb(w, h, |x, y| {
        [((x * 255) / w) as u8, ((y * 255) / h) as u8, 128]
    });
    let skeleton = synthetic_rgb(w, h, |x, y| {
        if (h / 5..h / 5 + 12).contains(&y) && (w / 6..5 * w / 6).contains(&x) {
            [255, 255, 255]
        } else {
            [0, 0, 0]
        }
    });

    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let request = GenerationRequest {
        prompt: "a portrait of a person, studio lighting".to_owned(),
        negative_prompt: Some("blurry, low quality".to_owned()),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(8),
        guidance: Some(5.0),
        strength: Some(1.0),
        conditioning: vec![
            Conditioning::Control {
                image: skeleton,
                kind: ControlKind::Pose,
                scale: 0.7,
            },
            Conditioning::Reference {
                image: reference,
                strength: Some(0.6),
            },
        ],
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        })
        .expect("combined kolors pose generation");
    let GenerationOutput::Images(images) = output else {
        panic!("expected image output");
    };
    let image = images.into_iter().next().expect("one image");
    assert_eq!((image.width, image.height), (w, h));
    assert_eq!(image.pixels.len() as u32, w * h * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(
        image.pixels.iter().any(|&p| p > 16) && image.pixels.iter().any(|&p| p < 239),
        "degenerate kolors pose render"
    );
}

/// L2-normalized cosine similarity between two ArcFace embeddings (test helper).
#[cfg(target_os = "macos")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Real-weights validation for sc-4424/sc-4427: render the worker-owned InstantID angle
/// presets through the engine `generate_with_kps` pass-in path and assert, per view, that
/// (a) **the engine honours the pass-in landmarks** — the detected face lands where the
/// preset's nose keypoint says it should, proving the worker (not the engine's retired
/// hardcoded table) now owns the framing — and (b) **identity holds** — ArcFace cosine vs the
/// reference stays well above floor on the measurable frontal-ish views. Profiles are ArcFace
/// identity-N/A (frontal-only metric, see the likeness-score memo) so they assert placement
/// only, and a missing detection on a profile is tolerated (recorded, not failed).
///
/// This is the framing-fill goal from epic 4422: the presets pull head-and-shoulders up into
/// the frame instead of the old lower-half framing, so LoRA training gets more character pixels.
///
/// Needs: RealVisXL (`SG161222/RealVisXL_V5.0`) + InstantID IdentityNet (`InstantX/InstantID`)
/// in the HF cache, the converted bundle (`scrfd_10g`/`arcface_iresnet100`/`ip-adapter`) in the
/// app cache (`SCENEWORKS_INSTANTID_WEIGHTS` overrides), and a reference face
/// (`SCENEWORKS_TEST_FACE` overrides). Metal device. On demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored instantid_angle_kps_real_weights --nocapture`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real InstantID weights + Metal device"]
fn instantid_angle_kps_real_weights_fills_frame_and_holds_identity() {
    // --- resolve weights (HF cache + app bundle) ---
    let sdxl_base = hf_snapshot("models--SG161222--RealVisXL_V5.0");
    let identitynet = hf_snapshot("models--InstantX--InstantID").join("ControlNetModel");
    let bundle = std::env::var("SCENEWORKS_INSTANTID_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join("Library/Application Support/SceneWorks/data/cache/instantid-mlx")
        });
    let scrfd_path = bundle.join(INSTANTID_SCRFD_FILE);
    let arcface_path = bundle.join(INSTANTID_ARCFACE_FILE);
    let ip_adapter = bundle.join(INSTANTID_IP_ADAPTER_FILE);
    for p in [
        &sdxl_base,
        &identitynet,
        &scrfd_path,
        &arcface_path,
        &ip_adapter,
    ] {
        assert!(p.exists(), "missing InstantID weight: {}", p.display());
    }
    let face_path = std::env::var("SCENEWORKS_TEST_FACE").unwrap_or_else(|_| {
        "/Users/michael/Library/Application Support/SceneWorks/data/projects/ab.sceneworks/assets/images/genset_e6b07eb5b5374627af1bf47083bac305/2026-06-10_qwen_image_edit_2511_lightning_22-year-old-woman-with-fair-complexion-a-p_0001.png".to_owned()
    });
    let decoded = image::open(&face_path)
        .unwrap_or_else(|e| panic!("reference face {face_path}: {e}"))
        .to_rgb8();
    let reference = Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    };

    // --- load model + native face stack (production load order) ---
    let paths = InstantIdPaths {
        sdxl_base,
        identitynet: WeightsSource::Dir(identitynet),
        ip_adapter,
        adapters: Vec::new(),
    };
    let model = InstantId::load(&paths).expect("InstantID load");
    let scrfd = Weights::from_file(&scrfd_path).expect("SCRFD weights");
    let arcface = Weights::from_file(&arcface_path).expect("ArcFace weights");
    let model = model.with_face(&scrfd, &arcface).expect("face stack");

    // Reference identity embedding (frontal source).
    let ref_face = model
        .largest_face(
            &reference.pixels,
            reference.height as usize,
            reference.width as usize,
        )
        .expect("reference face detected");

    // Square canvas (the engine forces square for kps; sc-2009 aspect rule).
    let side: u32 = 1024;
    // Views where ArcFace identity is meaningful (frontal-ish; profiles are N/A — the metric is
    // frontal-only, see the likeness-score memo).
    let identity_views = ["front", "three_quarter_left", "three_quarter_right"];
    // Strict profiles occlude the far eye, so SCRFD's landmark regression there is unreliable —
    // placement is recorded but not hard-asserted (identity is N/A too).
    let profile_views = ["left_profile", "right_profile"];
    // Identity floor — well below the ~0.83-0.87 seen in sc-3345/sc-3365 validation, so the
    // assertion catches a regression without being brittle to seed/quant jitter.
    const IDENTITY_FLOOR: f32 = 0.50;
    // Mean per-landmark distance (square-fraction) between the GENERATED face's detected kps and
    // the preset kps the engine was told to draw — the direct "did `generate_with_kps` honour the
    // pass-in landmarks" check. IdentityNet conditions on these points, so the realized face's
    // landmarks track them tightly; the tolerance absorbs SCRFD/seed jitter.
    const PLACEMENT_TOL: f32 = 0.10;

    let mut failures: Vec<String> = Vec::new();
    println!("\n  view                 kps-dist   area%   id-cos   verdict");
    for &angle in CHARACTER_ANGLE_SET_ORDER.iter() {
        let kps = sceneworks_core::angle_kps::angle_kps(angle).expect("built-in angle kps");
        let req = InstantIdRequest {
            prompt: augment_prompt_for_angle("a portrait photo of a woman", angle),
            negative: "blurry, low quality, deformed".to_owned(),
            width: side,
            height: side,
            steps: INSTANTID_DEFAULT_STEPS as usize,
            guidance: INSTANTID_DEFAULT_GUIDANCE,
            ip_adapter_scale: INSTANTID_IP_SCALE,
            controlnet_scale: INSTANTID_CONTROLNET_SCALE,
            seed: 12345,
            ..InstantIdRequest::default()
        };
        let out = model
            .generate_with_kps(&req, &reference, &kps, &mut |_| {})
            .unwrap_or_else(|e| panic!("{angle}: generate_with_kps failed: {e}"));
        assert_eq!((out.width, out.height), (side, side), "{angle}: canvas");

        let is_identity_view = identity_views.contains(&angle);
        let is_profile = profile_views.contains(&angle);
        match model.largest_face(&out.pixels, out.height as usize, out.width as usize) {
            Ok(face) => {
                // Mean distance between the generated face's detected landmarks and the preset.
                let kps_dist: f32 = (0..5)
                    .map(|i| {
                        let dx = face.kps[i][0] / side as f32 - kps[i].0;
                        let dy = face.kps[i][1] / side as f32 - kps[i].1;
                        (dx * dx + dy * dy).sqrt()
                    })
                    .sum::<f32>()
                    / 5.0;
                let area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                    / (side as f32 * side as f32);
                let id = cosine(&ref_face.embedding, &face.embedding);
                // Placement is hard-asserted on non-profile views (reliable landmarks).
                let placed = is_profile || kps_dist <= PLACEMENT_TOL;
                let id_ok = !is_identity_view || id >= IDENTITY_FLOOR;
                let verdict = if placed && id_ok { "ok" } else { "FAIL" };
                println!(
                    "  {angle:<20} {kps_dist:>6.3}   {:>5.1}  {id:>6.3}  {verdict}",
                    area * 100.0,
                );
                if !is_profile && kps_dist > PLACEMENT_TOL {
                    failures.push(format!(
                        "{angle}: realized landmarks {kps_dist:.3} > {PLACEMENT_TOL} from preset"
                    ));
                }
                if is_identity_view && id < IDENTITY_FLOOR {
                    failures.push(format!(
                        "{angle}: identity {id:.3} < floor {IDENTITY_FLOOR}"
                    ));
                }
            }
            Err(e) => {
                println!("  {angle:<20} no-detect ({e})");
                // A frontal-ish view that fails to detect is a real failure; a profile is N/A.
                if is_identity_view {
                    failures.push(format!("{angle}: no face detected (frontal view)"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "preset validation failures:\n  {}",
        failures.join("\n  ")
    );
}

/// Load + generate one small image for a TRUE-CFG family (Chroma): the CFG scale rides
/// `true_cfg` (not the distilled `guidance` scalar the engine rejects), mirroring
/// [`generate_stream`]'s wiring. Sibling of [`smoke_generate_one`].
#[cfg(target_os = "macos")]
fn smoke_generate_one_true_cfg(
    sceneworks_id: &str,
    snapshot: std::path::PathBuf,
    true_cfg: Option<f32>,
    negative_prompt: Option<String>,
) {
    let model = mlx_model(sceneworks_id).unwrap();
    let generator = load_engine(
        model.engine_id(),
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let steps = model.default_steps();
    let (w, h, pixels) = generate_one(
        generator.as_ref(),
        "a serene mountain lake at dawn",
        512,
        512,
        42,
        steps,
        None,
        negative_prompt,
        None,
        true_cfg,
        None,
        None,
        None,
        &PromptEnhance::default(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights smoke: Chroma1-HD full true-CFG (sc-3843). Exercises the true-CFG worker
/// path (`true_cfg` carries the scale; the engine rejects `guidance`). Needs the HF cache
/// (`lodestones/Chroma1-HD`) + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Chroma1-HD weights + Metal device"]
fn chroma_hd_real_weights_generates_one_image() {
    smoke_generate_one_true_cfg(
        "chroma1_hd",
        hf_snapshot("models--lodestones--Chroma1-HD"),
        Some(3.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke: Chroma1-Base (beta-sigma schedule, same true-CFG path as HD).
/// Needs the HF cache (`lodestones/Chroma1-Base`) + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Chroma1-Base weights + Metal device"]
fn chroma_base_real_weights_generates_one_image() {
    smoke_generate_one_true_cfg(
        "chroma1_base",
        hf_snapshot("models--lodestones--Chroma1-Base"),
        Some(3.0),
        Some("blurry, low quality".to_owned()),
    );
}

/// Real-weights smoke: Chroma1-Flash few-step distilled (true_cfg≈1, negative inert).
/// Needs the HF cache (`lodestones/Chroma1-Flash`) + a Metal device.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Chroma1-Flash weights + Metal device"]
fn chroma_flash_real_weights_generates_one_image() {
    smoke_generate_one_true_cfg(
        "chroma1_flash",
        hf_snapshot("models--lodestones--Chroma1-Flash"),
        Some(1.0),
        None,
    );
}

// --- Z-Image strict-pose control path (sc-3028) ---

#[cfg(target_os = "macos")]
#[test]
fn resolve_control_scale_defaults_and_clamps() {
    assert_eq!(
        resolve_control_scale(&request(json!({ "projectId": "p" }))),
        0.9
    );
    assert_eq!(
        resolve_control_scale(&request(
            json!({ "projectId": "p", "advanced": { "controlScale": 0.65 } })
        )),
        0.65
    );
    // Clamp to [0, 2].
    assert_eq!(
        resolve_control_scale(&request(
            json!({ "projectId": "p", "advanced": { "controlScale": 5.0 } })
        )),
        2.0
    );
    assert_eq!(
        resolve_control_scale(&request(
            json!({ "projectId": "p", "advanced": { "controlScale": -1.0 } })
        )),
        0.0
    );
}

#[cfg(target_os = "macos")]
#[test]
fn pose_entries_filters_to_objects() {
    let req = request(json!({
        "projectId": "p",
        "advanced": { "poses": [{ "id": "a" }, "not-an-object", { "id": "b" }] }
    }));
    assert_eq!(pose_entries(&req).len(), 2);
    // No poses → empty (not a strict-pose job).
    assert!(pose_entries(&request(json!({ "projectId": "p" }))).is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn parse_poses_extracts_keypoints_hands_face() {
    let req = request(json!({
        "projectId": "p",
        "advanced": { "poses": [{
            "id": "a",
            "keypoints": [[0.5, 0.2], [0.5, 0.35]],
            "hands": [[[0.1, 0.1]], [[0.2, 0.2]]],
            "face": [[0.5, 0.18]]
        }] }
    }));
    let poses = parse_poses(&req);
    assert_eq!(poses.len(), 1);
    assert_eq!(poses[0].keypoints.len(), 18); // padded
    assert_eq!(poses[0].keypoints[0], Some((0.5, 0.2)));
    assert!(poses[0].hands.is_some());
    assert!(poses[0].face.is_some());
}

#[cfg(target_os = "macos")]
#[test]
fn qwen_control_raw_settings_records_control_recipe() {
    let req = request(json!({
        "projectId": "p",
        "model": "qwen_image",
        "advanced": {
            "poses": [{ "id": "pose_1" }],
            "controlScale": 0.5
        }
    }));
    let raw = qwen_control_raw_settings(&req, "Qwen/Qwen-Image", 20, Some(4), 4.0, 0.5, 1);
    assert_eq!(
        raw.get("controlEngine").and_then(Value::as_str),
        Some(QWEN_CONTROL_ENGINE_ID)
    );
    assert_eq!(raw.get("controlScale"), Some(&json!(0.5)));
    assert_eq!(raw.get("poseCount"), Some(&json!(1)));
    assert_eq!(raw.get("guidanceScale"), Some(&json!(4.0)));
    assert_eq!(raw.get("mlxQuantize"), Some(&json!(4)));
}

/// sc-3031 A/B dump (NOT a CI test): generate ONE image through the **real new-adapter
/// path** — the production resolvers (`model_repo` / `resolve_steps` / `resolve_guidance` /
/// `resolve_negative_prompt` / `resolve_quant` / `resolve_adapters` / `resolve_weights_dir` /
/// `resolve_seed`) + the `load_engine` + `generate_one` core that `generate_stream`
/// drives — and write it to `$SC3031_OUT` for head-to-head comparison against the Python
/// `Mlx*Adapter` output. Covers the txt2img families (z-image / flux / qwen / flux2 / sdxl).
/// Env: `SC3031_PAYLOAD` (job-payload JSON object), `SC3031_OUT` (.png path); set
/// `SCENEWORKS_DATA_DIR` + `HF_HOME` so weights resolve. Run:
/// `SC3031_PAYLOAD=… SC3031_OUT=… cargo test -p sceneworks-worker --lib -- --ignored --exact \
///   image_jobs::tests::sc3031_ab_dump_txt2img`
#[cfg(target_os = "macos")]
#[ignore = "sc-3031 A/B dump harness: drive via SC3031_PAYLOAD + SC3031_OUT"]
#[test]
fn sc3031_ab_dump_txt2img() {
    let payload: Value =
        serde_json::from_str(&std::env::var("SC3031_PAYLOAD").expect("SC3031_PAYLOAD"))
            .expect("SC3031_PAYLOAD is JSON");
    let out = std::env::var("SC3031_OUT").expect("SC3031_OUT");
    let req = request(payload);
    let settings = Settings::from_env(); // honors SCENEWORKS_DATA_DIR + HF_HOME

    let model = mlx_model(&req.model).expect("an MLX txt2img model id");
    let _repo = model_repo(&req, &model);
    let steps = resolve_steps(&req, &model);
    let guidance = resolve_guidance(&req, &model);
    let negative = resolve_negative_prompt(&req, &model);
    let (quant, _bits) = resolve_quant(&req);
    let weights = resolve_weights_dir(&req, &settings)
        .expect("weights resolve")
        .expect("weights in HF cache");
    let adapters = resolve_adapters(&req, &settings).expect("adapters");
    let seed = resolve_seed(&req, 0);
    let generator = load_engine(model.engine_id(), weights, quant, adapters, None).expect("load");

    let cancel = CancelFlag::new();
    let (w, h, pixels) = generate_one(
        generator.as_ref(),
        &req.prompt,
        req.width,
        req.height,
        seed,
        steps,
        guidance,
        negative,
        None,
        None,
        None,
        None,
        None,
        &PromptEnhance::default(),
        &cancel,
        &mut |_| {},
    )
    .expect("generate");
    image::RgbImage::from_raw(w, h, pixels)
        .expect("rgb buffer")
        .save(&out)
        .expect("save png");
    eprintln!(
        "sc3031 rust dump: model={} {w}x{h} seed={seed} steps={steps} guidance={guidance:?} -> {out}",
        req.model
    );
}

/// sc-3031 A/B dump (pose): generate ONE strict-pose image through the **real new-adapter
/// path** — the production `resolve_control_weights` / `resolve_control_scale` / `parse_poses`,
/// the `draw_wholebody` skeleton render, and `zimage_control_load` / `zimage_control_generate_one`
/// (the core that `generate_zimage_control_stream` drives) — and write it to `$SC3031_OUT` for
/// head-to-head comparison against the Python `MlxZImageAdapter` strict-pose tier (drive the
/// Python side with the SAME `advanced.poses` payload). Env: `SC3031_PAYLOAD` (must carry
/// `advanced.poses`), `SC3031_OUT`; set `SCENEWORKS_DATA_DIR` + `HF_HOME`.
#[cfg(target_os = "macos")]
#[ignore = "sc-3031 A/B dump harness (pose): drive via SC3031_PAYLOAD + SC3031_OUT"]
#[test]
fn sc3031_ab_dump_pose() {
    let payload: Value =
        serde_json::from_str(&std::env::var("SC3031_PAYLOAD").expect("SC3031_PAYLOAD"))
            .expect("SC3031_PAYLOAD is JSON");
    let out = std::env::var("SC3031_OUT").expect("SC3031_OUT");
    let req = request(payload);
    let settings = Settings::from_env();

    let weights = resolve_weights_dir(&req, &settings)
        .expect("z-image weights resolve")
        .expect("z-image weights in HF cache");
    let control_weights =
        resolve_control_weights(&req, &settings).expect("Fun-Controlnet-Union weights");
    let (quant, _bits) = resolve_quant(&req);
    let zimage = mlx_model("z_image_turbo").expect("z-image model row");
    let steps = resolve_steps(&req, &zimage);
    let control_scale = resolve_control_scale(&req);
    let adapters = resolve_adapters(&req, &settings).expect("adapters");
    let seed = resolve_seed(&req, 0);

    let pose = parse_poses(&req)
        .into_iter()
        .next()
        .expect("advanced.poses");
    let (w, h) = (req.width, req.height);
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        crate::openpose_skeleton::body_stickwidth(w, h),
    );
    let control = Image {
        width: w,
        height: h,
        pixels: skeleton.into_raw(),
    };

    let generator = zimage_control_load(weights, control_weights, quant, adapters).expect("load");
    let cancel = CancelFlag::new();
    let (ow, oh, pixels) = zimage_control_generate_one(
        generator.as_ref(),
        &req.prompt,
        w,
        h,
        seed,
        steps,
        control,
        control_scale,
        None,
        &cancel,
        &mut |_| {},
    )
    .expect("generate");
    image::RgbImage::from_raw(ow, oh, pixels)
        .expect("rgb buffer")
        .save(&out)
        .expect("save png");
    eprintln!(
        "sc3031 rust pose dump: {ow}x{oh} seed={seed} steps={steps} control_scale={control_scale} -> {out}"
    );
}

/// sc-3031 A/B (pose, no model): render JUST the control skeleton from
/// `$SC3031_PAYLOAD`'s `advanced.poses` via the production `parse_poses` + `draw_wholebody`
/// and write it to `$SC3031_OUT`, to compare the Rust skeleton renderer against the Python
/// one for the same keypoints (separates skeleton-render parity from the engine/schedule).
/// CPU-only, instant (no weights / no Metal). macOS-gated because it uses `parse_poses`
/// (part of the macOS strict-pose path); the Linux workspace-check lane configures that out.
#[cfg(target_os = "macos")]
#[ignore = "sc-3031 skeleton-render dump: drive via SC3031_PAYLOAD + SC3031_OUT"]
#[test]
fn sc3031_dump_skeleton() {
    let payload: Value =
        serde_json::from_str(&std::env::var("SC3031_PAYLOAD").expect("SC3031_PAYLOAD"))
            .expect("SC3031_PAYLOAD is JSON");
    let out = std::env::var("SC3031_OUT").expect("SC3031_OUT");
    let req = request(payload);
    let pose = parse_poses(&req)
        .into_iter()
        .next()
        .expect("advanced.poses");
    let (w, h) = (req.width, req.height);
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        crate::openpose_skeleton::body_stickwidth(w, h),
    );
    image::RgbImage::from_raw(w, h, skeleton.into_raw())
        .expect("rgb buffer")
        .save(&out)
        .expect("save png");
    eprintln!("sc3031 rust skeleton dump: {w}x{h} -> {out}");
}

/// Identity img2img-init gate + clamp (sc-3146): the parity-sensitive decision the
/// strict-pose stream makes before loading the reference image. Mirrors the Python
/// `MlxZImageAdapter._identity_init_requested` + `_reference_strength` semantics.
#[cfg(target_os = "macos")]
#[test]
fn zimage_identity_strength_gate_and_clamp() {
    let with = |adv: Value, asset: Value| {
        let mut payload = json!({
            "projectId": "p", "model": "z_image_turbo", "prompt": "a knight"
        });
        let obj = payload.as_object_mut().unwrap();
        obj.insert("advanced".to_owned(), adv);
        if !asset.is_null() {
            obj.insert("referenceAssetId".to_owned(), asset);
        }
        zimage_identity_strength(&request(payload))
    };
    let approx = |got: Option<f32>, want: f32| match got {
        Some(value) => assert!((value - want).abs() < 1e-6, "got {value}, want {want}"),
        None => panic!("expected Some({want}), got None"),
    };

    // Pose-only tiers → None: no referenceStrength; referenceStrength == 0 (parity:
    // the Python gate requires > 0); referenceStrength > 0 but no/blank asset (a bare
    // reference has no MLX home, so it falls back to pose-only rather than erroring).
    assert_eq!(with(json!({}), json!("ref_1")), None);
    assert_eq!(
        with(json!({ "referenceStrength": 0.0 }), json!("ref_1")),
        None
    );
    assert_eq!(with(json!({ "referenceStrength": 0.6 }), Value::Null), None);
    assert_eq!(
        with(json!({ "referenceStrength": 0.6 }), json!("   ")),
        None
    );

    // Engaged: strength forwarded verbatim (no inversion) and clamped to [0.05, 1.0].
    approx(
        with(json!({ "referenceStrength": 0.6 }), json!("ref_1")),
        0.6,
    );
    approx(
        with(json!({ "referenceStrength": "0.45" }), json!("ref_1")),
        0.45,
    );
    assert_eq!(
        with(json!({ "referenceStrength": 1.8 }), json!("ref_1")),
        Some(1.0)
    );
    assert_eq!(
        with(json!({ "referenceStrength": 0.01 }), json!("ref_1")),
        Some(0.05)
    );
}

/// Real-weights smoke: Z-Image strict-pose ControlNet. Loads the base
/// `Tongyi-MAI/Z-Image-Turbo` snapshot + the cached Fun-Controlnet-Union checkpoint,
/// renders a DWPose skeleton, and generates one pose image. Needs both in the HF
/// cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored zimage_control_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Z-Image + Fun-Controlnet-Union weights + Metal device"]
fn zimage_control_real_weights_generates_one_pose() {
    let base = hf_snapshot("models--Tongyi-MAI--Z-Image-Turbo");
    let control = std::fs::read_dir(dirs_home().join(
        ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1/snapshots",
    ))
    .expect("control snapshots dir")
    .flatten()
    .map(|entry| entry.path())
    .find(|path| path.is_dir())
    .map(|dir| dir.join(super::ZIMAGE_CONTROL_FILE))
    .filter(|path| path.exists())
    .expect("control weights file");

    let generator =
        zimage_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();

    // A minimal standing skeleton at 512².
    let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
        [0.5, 0.2],
        [0.5, 0.35],
        [0.42, 0.35],
        [0.40, 0.5],
        [0.40, 0.65],
        [0.58, 0.35],
        [0.60, 0.5],
        [0.60, 0.65],
        [0.45, 0.6],
        [0.45, 0.8],
        [0.45, 0.95],
        [0.55, 0.6],
        [0.55, 0.8],
        [0.55, 0.95],
        [0.48, 0.18],
        [0.52, 0.18],
        [0.46, 0.2],
        [0.54, 0.2]
    ]));
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        512,
        512,
        &kp,
        None,
        None,
        crate::openpose_skeleton::body_stickwidth(512, 512),
    );
    let control = gen_core::Image {
        width: 512,
        height: 512,
        pixels: skeleton.into_raw(),
    };

    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = zimage_control_generate_one(
        generator.as_ref(),
        "a person standing in a meadow",
        512,
        512,
        42,
        8,
        control,
        0.9,
        None,
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights smoke: Qwen-Image strict-pose ControlNet. Loads the base
/// `Qwen/Qwen-Image` snapshot + the cached InstantX ControlNet-Union checkpoint,
/// renders one DWPose skeleton, and generates one image through `qwen_image_control`.
/// Needs both in the HF cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored qwen_control_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Qwen-Image + InstantX ControlNet weights + Metal device"]
fn qwen_control_real_weights_generates_one_pose() {
    let base = hf_snapshot("models--Qwen--Qwen-Image");
    let control =
        hf_snapshot("models--InstantX--Qwen-Image-ControlNet-Union").join(super::QWEN_CONTROL_FILE);
    assert!(
        control.exists(),
        "Qwen control weights missing: {control:?}"
    );

    let generator =
        qwen_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();

    let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
        [0.5, 0.2],
        [0.5, 0.35],
        [0.42, 0.35],
        [0.40, 0.5],
        [0.40, 0.65],
        [0.58, 0.35],
        [0.60, 0.5],
        [0.60, 0.65],
        [0.45, 0.6],
        [0.45, 0.8],
        [0.45, 0.95],
        [0.55, 0.6],
        [0.55, 0.8],
        [0.55, 0.95],
        [0.48, 0.18],
        [0.52, 0.18],
        [0.46, 0.2],
        [0.54, 0.2]
    ]));
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        512,
        512,
        &kp,
        None,
        None,
        crate::openpose_skeleton::body_stickwidth(512, 512),
    );
    let control = gen_core::Image {
        width: 512,
        height: 512,
        pixels: skeleton.into_raw(),
    };

    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = qwen_control_generate_one(
        generator.as_ref(),
        "a person standing in a meadow",
        Some("blurry, low quality".to_owned()),
        512,
        512,
        42,
        4,
        4.0,
        control,
        0.9,
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

// --- FLUX.2 edit path (sc-3029) ---

#[cfg(target_os = "macos")]
#[test]
fn flux2_edit_engine_id_maps_variants() {
    assert_eq!(
        flux2_edit_engine_id("flux2_klein_9b"),
        Some("flux2_klein_9b_edit")
    );
    assert_eq!(
        flux2_edit_engine_id("flux2_klein_9b_true_v2"),
        Some("flux2_klein_9b_edit")
    );
    assert_eq!(
        flux2_edit_engine_id("flux2_klein_9b_kv"),
        Some("flux2_klein_9b_kv_edit")
    );
    // FLUX.2-dev edit (sc-5922): the dev model routes to the `flux2_dev_edit` variant.
    assert_eq!(flux2_edit_engine_id("flux2_dev"), Some("flux2_dev_edit"));
    assert_eq!(flux2_edit_engine_id("z_image_turbo"), None);
    assert_eq!(flux2_edit_engine_id("sdxl"), None);
}

// ---- sc-6124: FLUX.2-dev multi-reference edit memory guard -----------------------------------

#[cfg(target_os = "macos")]
#[test]
fn flux2_dev_edit_peak_gb_tracks_sc5923_measurements() {
    // sc-5923 worker-layer peaks (Q4, 1024², /usr/bin/time -l): single-ref ~81, 2-ref ~104 GB.
    let single = flux2_dev_edit_peak_gb(1, 1024, 1024);
    let double = flux2_dev_edit_peak_gb(2, 1024, 1024);
    assert!(
        (single - 81.0).abs() < 2.0,
        "single-reference estimate {single} GB ≉ measured ~81"
    );
    assert!(
        (double - 104.0).abs() < 2.0,
        "two-reference estimate {double} GB ≉ measured ~104"
    );
    // Monotonic in both reference count and resolution.
    assert!(double > single);
    assert!(flux2_dev_edit_peak_gb(2, 768, 768) < double);
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_dev_edit_memory_guard_gates_multiref_on_small_machines() {
    // Single reference / txt2img always pass — covered by the declared minMemoryGb — even on a
    // small machine, and regardless of the RAM probe.
    assert!(flux2_dev_edit_memory_guard(1, 1024, 1024, Some(64.0)).is_ok());
    assert!(flux2_dev_edit_memory_guard(0, 1024, 1024, Some(64.0)).is_ok());
    // Two references at 1024² (~104 GB peak): fits a 128 GB Mac, rejected on a 96 GB one.
    assert!(flux2_dev_edit_memory_guard(2, 1024, 1024, Some(128.0)).is_ok());
    let err = flux2_dev_edit_memory_guard(2, 1024, 1024, Some(96.0)).unwrap_err();
    assert!(
        matches!(&err, WorkerError::InvalidPayload(msg) if msg.contains("multi-reference")),
        "expected an actionable multi-reference rejection, got {err:?}"
    );
    // Dropping to 768² brings the two-reference edit (~74 GB) back under a 96 GB budget.
    assert!(flux2_dev_edit_memory_guard(2, 768, 768, Some(96.0)).is_ok());
    // A failed RAM probe is lenient (don't block a possibly-fine job).
    assert!(flux2_dev_edit_memory_guard(2, 1024, 1024, None).is_ok());
}

// ---- sc-6135: FLUX.2-dev caption-upsampling (enhance_prompt) threading ------------------------

#[cfg(target_os = "macos")]
#[test]
fn prompt_enhance_reads_advanced_settings() {
    // Absent → disabled with no overrides (the default for every model/job).
    let off = PromptEnhance::from_advanced(
        &request(json!({
            "projectId": "p", "model": "flux2_dev", "prompt": "a fox"
        }))
        .advanced,
    );
    assert!(!off.enabled);
    assert_eq!(off.temperature, None);
    assert_eq!(off.max_tokens, None);

    // The dev Image-Studio toggle sets `enhancePrompt`; optional temperature / max-tokens follow
    // (same keys as the LTX-2.3 video path).
    let on = PromptEnhance::from_advanced(&request(json!({
        "projectId": "p", "model": "flux2_dev", "prompt": "a fox",
        "advanced": { "enhancePrompt": true, "enhanceTemperature": 0.2, "enhanceMaxTokens": 256 }
    }))
    .advanced);
    assert!(on.enabled);
    assert_eq!(on.temperature, Some(0.2));
    assert_eq!(on.max_tokens, Some(256));
}

// ---- sc-6055: FLUX.2-dev strict-pose (flux2_dev_control) -------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn flux2_control_scale_defaults_and_clamps() {
    // README-recommended dev default (0.65–0.80 mid-point), distinct from Z-Image's 0.9.
    assert_eq!(
        flux2_control_scale(&request(json!({ "projectId": "p" }))),
        0.75
    );
    assert_eq!(
        flux2_control_scale(&request(
            json!({ "projectId": "p", "advanced": { "controlScale": 0.7 } })
        )),
        0.7
    );
    // Clamp to [0, 2].
    assert_eq!(
        flux2_control_scale(&request(
            json!({ "projectId": "p", "advanced": { "controlScale": 5.0 } })
        )),
        2.0
    );
    assert_eq!(
        flux2_control_scale(&request(
            json!({ "projectId": "p", "advanced": { "controlScale": -1.0 } })
        )),
        0.0
    );
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_control_repo_file_defaults_and_overrides() {
    let (repo, file) = flux2_control_repo_file(&request(json!({ "projectId": "p" })));
    assert_eq!(repo, FLUX2_CONTROL_REPO);
    assert_eq!(file, FLUX2_CONTROL_FILE);
    // `advanced.controlWeights` overrides repo + filename (parity with the Z-Image resolver).
    let (repo, file) = flux2_control_repo_file(&request(json!({
        "projectId": "p",
        "advanced": { "controlWeights": { "repo": "me/custom", "filename": "x.safetensors" } }
    })));
    assert_eq!(repo, "me/custom");
    assert_eq!(file, "x.safetensors");
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_control_raw_settings_records_control_recipe() {
    let req = request(json!({
        "projectId": "p", "model": "flux2_dev",
        "advanced": { "poses": [{ "id": "pose_1" }], "controlScale": 0.5 }
    }));
    // 0.5 is exactly representable in f32 (the `control_scale` arg is f32), so json! round-trips it.
    let raw = flux2_control_raw_settings(
        &req,
        "black-forest-labs/FLUX.2-dev",
        28,
        Some(4),
        Some(4.0),
        0.5,
        1,
    );
    assert_eq!(
        raw.get("controlEngine").and_then(Value::as_str),
        Some(FLUX2_DEV_CONTROL_ENGINE_ID)
    );
    assert_eq!(raw.get("controlScale"), Some(&json!(0.5)));
    assert_eq!(raw.get("poseCount"), Some(&json!(1)));
    // dev keeps its embedded guidance (NOT distilled like Z-Image, which nulls guidance).
    assert_eq!(raw.get("guidanceScale"), Some(&json!(4.0)));
    assert_eq!(raw.get("mlxQuantize"), Some(&json!(4)));
    assert_eq!(raw.get("realModelInference"), Some(&json!(true)));
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_identity_strength_gates_on_strength_and_asset() {
    // Off by default (no referenceStrength) — the pose-only tier.
    assert_eq!(
        flux2_identity_strength(&request(
            json!({ "projectId": "p", "referenceAssetId": "r" })
        )),
        None
    );
    // referenceStrength set but no asset → None.
    assert_eq!(
        flux2_identity_strength(&request(
            json!({ "projectId": "p", "advanced": { "referenceStrength": 0.5 } })
        )),
        None
    );
    // Both present → clamped strength (the opt-in img2img-init).
    assert_eq!(
        flux2_identity_strength(&request(json!({
            "projectId": "p", "referenceAssetId": "r", "advanced": { "referenceStrength": 0.5 }
        }))),
        Some(0.5)
    );
    // Clamp to [0.05, 1.0].
    assert_eq!(
        flux2_identity_strength(&request(json!({
            "projectId": "p", "referenceAssetId": "r", "advanced": { "referenceStrength": 2.0 }
        }))),
        Some(1.0)
    );
}

/// Real-weights smoke: FLUX.2-dev strict-pose Fun-Controlnet-Union (sc-6055; engine sc-2292). Loads
/// the converted Q4 dev snapshot (`models/mlx/flux2_dev`, assembled by the `flux2_dev_quant` convert
/// job) + the cached Fun-Controlnet-Union `-2602` checkpoint, renders a DWPose skeleton, and generates
/// one pose image through `flux2_dev_control`. Needs both + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux2_dev_control_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs converted FLUX.2-dev + Fun-Controlnet-Union weights + Metal device"]
fn flux2_dev_control_real_weights_generates_one_pose() {
    let base = std::env::var("SCENEWORKS_FLUX2_DEV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join("Library/Application Support/SceneWorks/data/models/mlx/flux2_dev")
        });
    let control = std::fs::read_dir(dirs_home().join(
        ".cache/huggingface/hub/models--alibaba-pai--FLUX.2-dev-Fun-Controlnet-Union/snapshots",
    ))
    .expect("control snapshots dir")
    .flatten()
    .map(|entry| entry.path())
    .find(|path| path.is_dir())
    .map(|dir| dir.join(FLUX2_CONTROL_FILE))
    .filter(|path| path.exists())
    .expect("control weights file");

    let generator =
        flux2_control_load(base, control, Some(gen_core::Quant::Q4), Vec::new()).unwrap();

    // A minimal standing skeleton at 512².
    let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
        [0.5, 0.2],
        [0.5, 0.35],
        [0.42, 0.35],
        [0.40, 0.5],
        [0.40, 0.65],
        [0.58, 0.35],
        [0.60, 0.5],
        [0.60, 0.65],
        [0.45, 0.6],
        [0.45, 0.8],
        [0.45, 0.95],
        [0.55, 0.6],
        [0.55, 0.8],
        [0.55, 0.95],
        [0.48, 0.18],
        [0.52, 0.18],
        [0.46, 0.2],
        [0.54, 0.2]
    ]));
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        512,
        512,
        &kp,
        None,
        None,
        crate::openpose_skeleton::body_stickwidth(512, 512),
    );
    let control = gen_core::Image {
        width: 512,
        height: 512,
        pixels: skeleton.into_raw(),
    };

    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = flux2_control_generate_one(
        generator.as_ref(),
        "a person standing in a meadow, photorealistic",
        512,
        512,
        42,
        8,
        Some(4.0), // dev embedded guidance
        control,
        0.75,
        None,
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_edit_reference_ids_prefers_reference_then_source() {
    // referenceAssetId (character flow) wins.
    assert_eq!(
        flux2_edit_reference_ids(&request(json!({
            "projectId": "p", "referenceAssetId": "ref_1", "sourceAssetId": "src_1"
        }))),
        vec!["ref_1".to_owned()]
    );
    // sourceAssetId only in edit_image mode.
    assert_eq!(
        flux2_edit_reference_ids(&request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src_1"
        }))),
        vec!["src_1".to_owned()]
    );
    // sourceAssetId without edit_image mode is ignored (it's the txt2img path).
    assert!(flux2_edit_reference_ids(&request(json!({
        "projectId": "p", "sourceAssetId": "src_1"
    })))
    .is_empty());
    assert!(flux2_edit_reference_ids(&request(json!({ "projectId": "p" }))).is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn build_edit_conditioning_single_vs_multi() {
    let img = |seed| gen_core::Image {
        width: 8,
        height: 8,
        pixels: stub_rgb8(8, 8, seed),
    };
    match build_edit_conditioning(std::slice::from_ref(&img(1))).as_slice() {
        [gen_core::Conditioning::Reference { .. }] => {}
        other => panic!("expected one Reference, got {other:?}"),
    }
    match build_edit_conditioning(&[img(1), img(2)]).as_slice() {
        [gen_core::Conditioning::MultiReference { images }] => assert_eq!(images.len(), 2),
        other => panic!("expected MultiReference, got {other:?}"),
    }
}

/// Real-weights smoke: FLUX.2-klein edit. Loads `flux2_klein_9b_edit` (base 9B
/// snapshot) and generates one image conditioned on a synthetic reference. Needs
/// the HF cache (`black-forest-labs/FLUX.2-klein-9B`) + a Metal device; run on
/// demand: `cargo test -p sceneworks-worker --lib -- --ignored flux2_edit_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + Metal device"]
fn flux2_edit_real_weights_generates_one_image() {
    let snapshot = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b");
    let generator = load_engine(
        "flux2_klein_9b_edit",
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let reference = gen_core::Image {
        width: 512,
        height: 512,
        pixels: stub_rgb8(512, 512, 7),
    };
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = flux2_edit_generate_one(
        generator.as_ref(),
        "make it a watercolor painting",
        512,
        512,
        42,
        4,
        Some(1.0),
        build_edit_conditioning(std::slice::from_ref(&reference)),
        &PromptEnhance::default(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

// --- Angle set / pose tier / fit_image (sc-3030) ---

#[cfg(target_os = "macos")]
#[test]
fn character_angle_set_is_eleven_ordered_angles() {
    assert_eq!(CHARACTER_ANGLE_SET_ORDER.len(), 11);
    assert_eq!(CHARACTER_ANGLE_SET_ORDER[0], "front");
    // Every angle has a non-empty augment clause.
    for angle in CHARACTER_ANGLE_SET_ORDER {
        assert!(
            !angle_prompt_augment(angle).is_empty(),
            "no augment for {angle}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn augment_prompt_for_angle_appends_clause_and_strips_punctuation() {
    assert_eq!(
        augment_prompt_for_angle("a knight", "front"),
        "a knight, frontal portrait, looking directly at the camera, head and shoulders, neutral expression"
    );
    // Trailing punctuation on the base is stripped before the comma join.
    assert_eq!(
        augment_prompt_for_angle("a knight.", "left_profile"),
        "a knight, full left profile, head turned 90 degrees to the left, side view of the head"
    );
    // Empty base → the augment clause alone.
    assert_eq!(
        augment_prompt_for_angle("", "down"),
        "looking down, head tilted slightly downward toward the floor"
    );
    // Unknown angle (no clause) → the base prompt unchanged.
    assert_eq!(augment_prompt_for_angle("a knight", "sideways"), "a knight");
}

#[cfg(target_os = "macos")]
#[test]
fn augment_prompt_for_pose_appends_cue() {
    assert_eq!(
        augment_prompt_for_pose("a hero"),
        "a hero, matching the exact body pose shown in the OpenPose skeleton reference image"
    );
    assert_eq!(augment_prompt_for_pose("  "), POSE_SKELETON_PROMPT);
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_grouping_poses_over_angles_over_plain() {
    // Pose set wins even when angleSet is also set.
    let poses = request(json!({
        "projectId": "p", "mode": "character_image", "referenceAssetId": "ref",
        "advanced": { "angleSet": true, "poses": [{ "id": "a" }, { "id": "b" }] }
    }));
    assert!(matches!(flux2_grouping(&poses), Flux2Grouping::Poses(2)));
    // angleSet without poses → the 11-angle set.
    let angles = request(json!({
        "projectId": "p", "mode": "character_image", "referenceAssetId": "ref",
        "advanced": { "angleSet": true }
    }));
    assert!(matches!(flux2_grouping(&angles), Flux2Grouping::Angles));
    // character_image with neither → plain.
    let plain = request(json!({
        "projectId": "p", "mode": "character_image", "referenceAssetId": "ref"
    }));
    assert!(matches!(flux2_grouping(&plain), Flux2Grouping::Plain));
    // edit_image never groups, even with angleSet (mode gate).
    let edit = request(json!({
        "projectId": "p", "mode": "edit_image", "sourceAssetId": "src",
        "advanced": { "angleSet": true }
    }));
    assert!(matches!(flux2_grouping(&edit), Flux2Grouping::Plain));
}

/// Minimal valid safetensors (8-byte LE header length + JSON header). No `networkType`, so
/// `classify_adapter` reports `Lora`; the resolver only reads the header, so empty tensors are fine.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_min_lora(path: &std::path::Path) {
    let header = json!({ "__metadata__": { "format": "pt" } });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buffer.extend_from_slice(&header_bytes);
    std::fs::write(path, buffer).unwrap();
}

/// A valid safetensors header that declares tensor data the file doesn't actually
/// contain — i.e. a truncated/interrupted download (sc-6072).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn write_truncated_lora(path: &std::path::Path) {
    let header = json!({
        "__metadata__": { "format": "pt" },
        "lora.down.weight": { "dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
    buffer.extend_from_slice(&header_bytes);
    // Declared 512 bytes of tensor data, but write none — the file is short.
    std::fs::write(path, buffer).unwrap();
}

/// sc-6072: a truncated LoRA is rejected at adapter-classification time (the
/// generation path's last gate before the engine) instead of reaching the MLX
/// loader and surfacing the opaque "invalid data offsets" error.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn classify_adapter_rejects_truncated_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    let lora_file = dir.path().join("truncated.safetensors");
    write_truncated_lora(&lora_file);

    let error = classify_adapter(&lora_file).expect_err("truncated LoRA must be rejected");
    let message = error.to_string().to_ascii_lowercase();
    assert!(
        message.contains("incomplete") || message.contains("truncated"),
        "error should name the incompleteness, got: {error}"
    );
}

/// sc-6038: InstantID is a stock SDXL (RealVisXL) UNet, so a user-selected SDXL LoRA must resolve
/// into a non-empty adapter set — the worker now feeds these into `InstantIdPaths.adapters` (they
/// were previously dropped: `instantid.rs` never called `resolve_adapters`). Guards the worker half
/// of the fix; the engine merge is covered by mlx-gen #477 / candle-gen #86 + the real-weight smoke.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn instantid_resolves_user_loras_into_adapters() {
    let dir = tempfile::tempdir().unwrap();
    let mut settings = Settings::from_env();
    settings.data_dir = dir.path().to_path_buf();

    // A real (tiny, valid) safetensors LoRA under the app-managed data dir — both the path
    // containment guard and the header classification in `resolve_adapters` run against it.
    let lora_file = dir.path().join("style.safetensors");
    write_min_lora(&lora_file);

    let req = request(json!({
        "projectId": "p", "model": "instantid_realvisxl", "mode": "character_image",
        "prompt": "portrait", "referenceAssetId": "ref-1",
        "loras": [{ "path": lora_file.to_string_lossy(), "weight": 0.65 }],
        "modelManifestEntry": { "family": "instantid" }
    }));

    let adapters = resolve_adapters(&req, &settings).expect("resolve adapters");
    assert_eq!(
        adapters.len(),
        1,
        "the selected SDXL LoRA must resolve to one InstantID adapter (not silently dropped)"
    );
    assert_eq!(
        adapters[0].scale, 0.65,
        "the per-LoRA weight carries through"
    );
    assert!(matches!(adapters[0].kind, AdapterKind::Lora));
    assert_eq!(
        adapters[0].path.file_name().and_then(|n| n.to_str()),
        Some("style.safetensors"),
        "the confined LoRA path resolves to the on-disk file"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn image_route_count_follows_dispatch_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut settings = Settings::from_env();
    settings.data_dir = dir.path().to_path_buf();
    let model_path = dir.path().to_string_lossy().to_string();

    let zimage_pose = request(json!({
        "projectId": "p", "model": "z_image_turbo", "count": 7,
        "advanced": { "modelPath": model_path.clone(), "poses": [{ "id": "a" }, { "id": "b" }] }
    }));
    let route = resolve_image_route(&zimage_pose, &settings).unwrap();
    assert_eq!(route, ImageRoute::ZImageControl);
    assert_eq!(route.image_count(&zimage_pose, &settings), 2);

    let qwen_pose = request(json!({
        "projectId": "p", "model": "qwen_image", "mode": "character_image", "count": 5,
        "advanced": { "modelPath": model_path.clone(), "poses": [{ "id": "a" }, { "id": "b" }, { "id": "c" }] }
    }));
    let route = resolve_image_route(&qwen_pose, &settings).unwrap();
    assert_eq!(route, ImageRoute::QwenControl);
    assert_eq!(route.image_count(&qwen_pose, &settings), 3);

    let flux2_angle = request(json!({
        "projectId": "p", "model": "flux2_klein_9b", "mode": "character_image",
        "referenceAssetId": "ref", "count": 2,
        "advanced": { "modelPath": model_path.clone(), "angleSet": true }
    }));
    let route = resolve_image_route(&flux2_angle, &settings).unwrap();
    assert_eq!(route, ImageRoute::Flux2Edit);
    assert_eq!(
        route.image_count(&flux2_angle, &settings),
        CHARACTER_ANGLE_SET_ORDER.len() as u32
    );

    let qwen_edit_pose = request(json!({
        "projectId": "p", "model": "qwen_image_edit_2511", "mode": "character_image",
        "referenceAssetId": "ref", "count": 2,
        "advanced": { "modelPath": model_path.clone(), "poses": [{ "id": "a" }, { "id": "b" }] }
    }));
    let route = resolve_image_route(&qwen_edit_pose, &settings).unwrap();
    assert_eq!(route, ImageRoute::QwenEdit);
    assert_eq!(route.image_count(&qwen_edit_pose, &settings), 2);

    let instantid_angle = request(json!({
        "projectId": "p", "model": "instantid_realvisxl", "mode": "character_image",
        "referenceAssetId": "ref", "count": 2,
        "advanced": { "modelPath": model_path.clone(), "angleSet": true }
    }));
    let route = resolve_image_route(&instantid_angle, &settings).unwrap();
    assert_eq!(route, ImageRoute::InstantId);
    assert_eq!(route.image_count(&instantid_angle, &settings), 11);

    // PuLID-FLUX (sc-3344): character_image + reference on the FLUX.1-dev base → one identity image
    // per requested count (no angle/pose grouping).
    let pulid = request(json!({
        "projectId": "p", "model": "pulid_flux_dev", "mode": "character_image",
        "referenceAssetId": "ref", "count": 3,
        "advanced": { "modelPath": model_path.clone() }
    }));
    let route = resolve_image_route(&pulid, &settings).unwrap();
    assert_eq!(route, ImageRoute::PulidFlux);
    assert_eq!(route.image_count(&pulid, &settings), 3);

    let sdxl_ip = request(json!({
        "projectId": "p", "model": "sdxl", "referenceAssetId": "ref", "count": 4,
        "advanced": { "modelPath": model_path.clone() }
    }));
    let route = resolve_image_route(&sdxl_ip, &settings).unwrap();
    assert_eq!(route, ImageRoute::SdxlAdvanced);
    assert_eq!(route.image_count(&sdxl_ip, &settings), 4);

    let sensenova_angle = request(json!({
        "projectId": "p", "model": "sensenova_u1_8b", "mode": "character_image",
        "referenceAssetId": "ref", "count": 2,
        "advanced": { "modelPath": model_path.clone(), "angleSet": true }
    }));
    let route = resolve_image_route(&sensenova_angle, &settings).unwrap();
    assert_eq!(route, ImageRoute::SensenovaEdit);
    assert_eq!(
        route.image_count(&sensenova_angle, &settings),
        CHARACTER_ANGLE_SET_ORDER.len() as u32
    );

    let plain_mlx = request(json!({
        "projectId": "p", "model": "z_image_turbo", "count": 6,
        "advanced": { "modelPath": model_path.clone() }
    }));
    let route = resolve_image_route(&plain_mlx, &settings).unwrap();
    assert_eq!(route, ImageRoute::Mlx);
    assert_eq!(route.image_count(&plain_mlx, &settings), 6);
}

#[cfg(target_os = "macos")]
#[test]
fn should_fit_edit_source_only_for_off_aspect_edit_image() {
    // edit_image + source + no reference + non-stretch → fit.
    assert!(should_fit_edit_source(&request(json!({
        "projectId": "p", "mode": "edit_image", "sourceAssetId": "src", "fitMode": "crop"
    }))));
    // A character reference present → the reference path stays native.
    assert!(!should_fit_edit_source(&request(json!({
        "projectId": "p", "mode": "edit_image", "sourceAssetId": "src",
        "referenceAssetId": "ref", "fitMode": "crop"
    }))));
    // stretch keeps the legacy naive resize.
    assert!(!should_fit_edit_source(&request(json!({
        "projectId": "p", "mode": "edit_image", "sourceAssetId": "src", "fitMode": "stretch"
    }))));
    // character_image is never the edit-source fit path.
    assert!(!should_fit_edit_source(&request(json!({
        "projectId": "p", "mode": "character_image", "referenceAssetId": "ref"
    }))));
}

#[cfg(target_os = "macos")]
#[test]
fn contain_box_centers_the_contained_rect() {
    // Wide source contained in a square: full width, centered vertically.
    assert_eq!(contain_box(100, 50, 50, 50), (50, 25, 0, 12));
    // Tall source: full height, centered horizontally.
    assert_eq!(contain_box(50, 100, 50, 50), (25, 50, 12, 0));
}

#[cfg(target_os = "macos")]
#[test]
fn fit_rgb_crop_pad_stretch_produce_exact_dims_and_geometry() {
    // 100×50 solid white source.
    let source = image::RgbImage::from_pixel(100, 50, image::Rgb([255, 255, 255]));

    // crop → cover + center-crop, exact target dims, no black bars (all white).
    let cropped = fit_rgb(&source, 50, 50, "crop");
    assert_eq!((cropped.width(), cropped.height()), (50, 50));
    assert_eq!(cropped.get_pixel(0, 0), &image::Rgb([255, 255, 255]));
    assert_eq!(cropped.get_pixel(25, 25), &image::Rgb([255, 255, 255]));

    // pad → contain + letterbox: black top/bottom bars, white band in the middle.
    let padded = fit_rgb(&source, 50, 50, "pad");
    assert_eq!((padded.width(), padded.height()), (50, 50));
    assert_eq!(padded.get_pixel(0, 0), &image::Rgb([0, 0, 0])); // top bar
    assert_eq!(padded.get_pixel(25, 24), &image::Rgb([255, 255, 255])); // content band

    // outpaint degrades to pad geometry (same letterbox).
    assert_eq!(
        fit_rgb(&source, 50, 50, "outpaint").into_raw(),
        padded.into_raw()
    );

    // stretch → exact target dims (aspect not preserved).
    let stretched = fit_rgb(&source, 40, 30, "stretch");
    assert_eq!((stretched.width(), stretched.height()), (40, 30));
}

// --- Qwen-Image-Edit path (sc-3397) ---

#[cfg(target_os = "macos")]
#[test]
fn qwen_edit_model_table_rows() {
    for id in [
        "qwen_image_edit",
        "qwen_image_edit_2509",
        "qwen_image_edit_2511",
    ] {
        let m = mlx_model(id).unwrap();
        assert_eq!(m.engine_id(), "qwen_image_edit");
        assert_eq!(m.default_repo(), "Qwen/Qwen-Image-Edit-2511");
        assert_eq!(m.default_steps(), 40);
        assert_eq!(m.default_guidance(), 4.0);
        assert_eq!(m.adapter_label(), "mlx_qwen");
        assert!(m.supports_guidance() && m.supports_negative_prompt());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn qwen_edit_lightning_model_row_is_cfg_off_4step_distill() {
    // sc-3398: shares the engine model + base weights with the production edit rows
    // but runs the 4-step CFG-off recipe + the lightx2v distill.
    let m = mlx_model("qwen_image_edit_2511_lightning").unwrap();
    assert_eq!(m.engine_id(), "qwen_image_edit");
    assert_eq!(m.default_repo(), "Qwen/Qwen-Image-Edit-2511");
    assert_eq!(m.default_steps(), 4);
    assert_eq!(m.default_guidance(), 1.0);
    assert_eq!(m.adapter_label(), "mlx_qwen");
    // sc-3723: `supports_negative_prompt` is now read from the shared `qwen_image_edit` engine
    // descriptor (true — the model CAN do true CFG), NOT the old per-variant row flag. The
    // lightning CFG-off behavior is enforced by the ENGINE under the `lightning` sampler
    // (mlx-gen `model_edit.rs`: `neg = None` when `is_lightning`, regardless of any negative
    // prompt the worker passes), so descriptor-derivation is behavior-equivalent — the
    // lightning recipe identity below (sampler + distill LoRA) is the real CFG-off invariant.
    assert!(m.supports_negative_prompt());

    // The lightning lookup carries the engine sampler + the lightx2v 4-step distill LoRA;
    // the production edit ids carry none.
    let distill = qwen_edit_lightning("qwen_image_edit_2511_lightning").unwrap();
    assert_eq!(distill.sampler, "lightning");
    assert_eq!(distill.repo, "lightx2v/Qwen-Image-Edit-2511-Lightning");
    assert_eq!(
        distill.file,
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    );
    assert!(qwen_edit_lightning("qwen_image_edit_2511").is_none());
    assert!(qwen_edit_lightning("qwen_image_edit").is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn qwen_edit_engine_id_maps_variants() {
    for id in [
        "qwen_image_edit",
        "qwen_image_edit_2509",
        "qwen_image_edit_2511",
        // The Lightning distill maps to the same engine model (sc-3398); only its
        // sampler + distill LoRA differ.
        "qwen_image_edit_2511_lightning",
    ] {
        assert_eq!(qwen_edit_engine_id(id), Some("qwen_image_edit"));
    }
    // Base txt2img Qwen and other families have no edit variant.
    assert_eq!(qwen_edit_engine_id("qwen_image"), None);
    assert_eq!(qwen_edit_engine_id("flux2_klein_9b"), None);
}

#[cfg(target_os = "macos")]
#[test]
fn qwen_edit_reference_ids_prefers_reference_then_source() {
    // referenceAssetId (character flow) wins over a source.
    assert_eq!(
        qwen_edit_reference_ids(&request(json!({
            "projectId": "p", "referenceAssetId": "ref_1", "sourceAssetId": "src_1"
        }))),
        vec!["ref_1".to_owned()]
    );
    // sourceAssetId only in edit_image mode.
    assert_eq!(
        qwen_edit_reference_ids(&request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src_1"
        }))),
        vec!["src_1".to_owned()]
    );
    // sourceAssetId without edit_image mode is ignored (the txt2img path).
    assert!(qwen_edit_reference_ids(&request(json!({
        "projectId": "p", "sourceAssetId": "src_1"
    })))
    .is_empty());
    assert!(qwen_edit_reference_ids(&request(json!({ "projectId": "p" }))).is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_qwen_edit_guidance_reads_true_cfg_scale_not_guidance_scale() {
    let model = mlx_model("qwen_image_edit_2511").unwrap();
    // Default is the family true-CFG default (4.0).
    assert_eq!(
        resolve_qwen_edit_guidance(
            &request(json!({ "projectId": "p", "mode": "edit_image" })),
            &model
        ),
        4.0
    );
    // guidanceScale (the inert embedded-guidance knob the Python edit path pins at 1.0)
    // is IGNORED — only trueCfgScale drives the engine's true CFG.
    assert_eq!(
        resolve_qwen_edit_guidance(
            &request(json!({
                "projectId": "p", "mode": "edit_image",
                "advanced": { "guidanceScale": 1.0 }
            })),
            &model
        ),
        4.0
    );
    // trueCfgScale overrides.
    assert_eq!(
        resolve_qwen_edit_guidance(
            &request(json!({
                "projectId": "p", "mode": "edit_image",
                "advanced": { "trueCfgScale": 6.0 }
            })),
            &model
        ),
        6.0
    );
    // The character reference path clamps to [1, 10].
    assert_eq!(
        resolve_qwen_edit_guidance(
            &request(json!({
                "projectId": "p", "mode": "character_image",
                "advanced": { "trueCfgScale": 50.0 }
            })),
            &model
        ),
        10.0
    );
    assert_eq!(
        resolve_qwen_edit_guidance(
            &request(json!({
                "projectId": "p", "mode": "character_image",
                "advanced": { "trueCfgScale": 0.5 }
            })),
            &model
        ),
        1.0
    );
    // edit_image floors at 1.0 (the engine needs CFG > 1 to engage).
    assert_eq!(
        resolve_qwen_edit_guidance(
            &request(json!({
                "projectId": "p", "mode": "edit_image",
                "advanced": { "trueCfgScale": 0.5 }
            })),
            &model
        ),
        1.0
    );
}

/// Real-weights smoke: Qwen-Image-Edit. Loads `qwen_image_edit` (Qwen-Image-Edit-2511
/// snapshot) and generates one image conditioned on a synthetic reference — true CFG
/// (guidance 4.0 + a negative prompt). Needs the HF cache (`Qwen/Qwen-Image-Edit-2511`)
/// and a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored qwen_edit_real_weights`.
/// Uses 4 steps + 512² for speed (the production default is 40 steps).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Qwen-Image-Edit-2511 weights + Metal device"]
fn qwen_edit_real_weights_generates_one_image() {
    let snapshot = hf_snapshot("models--Qwen--Qwen-Image-Edit-2511");
    let generator = load_engine(
        "qwen_image_edit",
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    let reference = gen_core::Image {
        width: 512,
        height: 512,
        pixels: stub_rgb8(512, 512, 7),
    };
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = qwen_edit_generate_one(
        generator.as_ref(),
        "make it a watercolor painting",
        Some("blurry, low quality".to_owned()),
        512,
        512,
        42,
        4,
        4.0,
        None,
        build_edit_conditioning(std::slice::from_ref(&reference)),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights smoke: Qwen-Image-Edit **Lightning** (sc-3398). Loads `qwen_image_edit`
/// (Qwen-Image-Edit-2511 snapshot) with the lightx2v 4-step distill LoRA stacked on, then
/// generates one image via the `lightning` sampler (CFG-off single forward, no negative
/// prompt) at 4 steps. Needs the HF cache for BOTH `Qwen/Qwen-Image-Edit-2511` and the
/// distill LoRA `lightx2v/Qwen-Image-Edit-2511-Lightning` + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored qwen_edit_lightning_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Qwen-Image-Edit-2511 + lightx2v distill LoRA weights + Metal device"]
fn qwen_edit_lightning_real_weights_generates_one_image() {
    let distill = qwen_edit_lightning("qwen_image_edit_2511_lightning").unwrap();
    // The distill LoRA lives in the HF cache (download it once via the lightning
    // generate path, or `huggingface-cli download <repo>`); resolve its snapshot file.
    let snapshot_dir =
        crate::model_jobs::huggingface_snapshot_dir(&Settings::from_env().data_dir, distill.repo)
            .expect("lightx2v distill LoRA must be cached for this smoke");
    let lora_path = snapshot_dir.join(distill.file);
    assert!(
        lora_path.is_file(),
        "distill LoRA missing in cache: {}",
        lora_path.display()
    );

    let snapshot = hf_snapshot("models--Qwen--Qwen-Image-Edit-2511");
    let generator = load_engine(
        "qwen_image_edit",
        snapshot,
        Some(gen_core::Quant::Q8),
        vec![AdapterSpec::new(lora_path, 1.0, AdapterKind::Lora)],
        None,
    )
    .unwrap();
    let reference = gen_core::Image {
        width: 512,
        height: 512,
        pixels: stub_rgb8(512, 512, 7),
    };
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = qwen_edit_generate_one(
        generator.as_ref(),
        "make it a watercolor painting",
        // Lightning runs CFG-off: no negative prompt, guidance 1.0 (engine ignores it).
        None,
        512,
        512,
        42,
        4,
        1.0,
        Some("lightning"),
        build_edit_conditioning(std::slice::from_ref(&reference)),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

/// Real-weights smoke: the best-effort pose tier — a `[skeleton, reference]`
/// `MultiReference` edit through `flux2_klein_9b_edit`. Verifies the engine accepts
/// the multi-image pose conditioning on real weights (the single-reference smoke
/// above does not). Needs the HF cache (`black-forest-labs/FLUX.2-klein-9B`) + a
/// Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux2_pose_tier_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + Metal device"]
fn flux2_pose_tier_real_weights_generates_one_image() {
    let snapshot = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b");
    let generator = load_engine(
        "flux2_klein_9b_edit",
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();
    // A minimal standing skeleton (body only — the best-effort tier uses no
    // hands/face) + a synthetic reference, paired as the pose multi-image set.
    let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
        [0.5, 0.2],
        [0.5, 0.35],
        [0.42, 0.35],
        [0.40, 0.5],
        [0.40, 0.65],
        [0.58, 0.35],
        [0.60, 0.5],
        [0.60, 0.65],
        [0.45, 0.6],
        [0.45, 0.8],
        [0.45, 0.95],
        [0.55, 0.6],
        [0.55, 0.8],
        [0.55, 0.95],
        [0.48, 0.18],
        [0.52, 0.18],
        [0.46, 0.2],
        [0.54, 0.2]
    ]));
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        512,
        512,
        &kp,
        None,
        None,
        crate::openpose_skeleton::body_stickwidth(512, 512),
    );
    let skeleton_img = gen_core::Image {
        width: 512,
        height: 512,
        pixels: skeleton.into_raw(),
    };
    let reference = gen_core::Image {
        width: 512,
        height: 512,
        pixels: stub_rgb8(512, 512, 7),
    };
    let conditioning = vec![gen_core::Conditioning::MultiReference {
        images: vec![skeleton_img, reference],
    }];
    let cancel = gen_core::CancelFlag::new();
    let mut steps_seen = 0u32;
    let (w, h, pixels) = flux2_edit_generate_one(
        generator.as_ref(),
        &augment_prompt_for_pose("a knight standing in a courtyard"),
        512,
        512,
        42,
        4,
        Some(1.0),
        conditioning,
        &PromptEnhance::default(),
        &cancel,
        &mut |p| {
            if let gen_core::Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        },
    )
    .unwrap();
    assert_eq!((w, h), (512, 512));
    assert_eq!(pixels.len(), 512 * 512 * 3);
    assert!(steps_seen >= 1, "expected denoise step progress");
    assert!(pixels.windows(2).any(|w| w[0] != w[1]));
}

#[cfg(target_os = "macos")]
fn dirs_home() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
}

#[cfg(target_os = "macos")]
#[test]
fn sdxl_sub_mode_classifies_advanced_shapes() {
    // Plain txt2img (no reference, not an edit) → not an advanced job.
    assert!(sdxl_sub_mode(&request(json!({ "model": "sdxl", "prompt": "a fox" }))).is_none());
    // Reference (not edit) → IP-Adapter.
    assert!(matches!(
        sdxl_sub_mode(&request(
            json!({ "model": "sdxl", "referenceAssetId": "ref_1" })
        )),
        Some(SdxlSubMode::Ip)
    ));
    // edit_image + source → plain img2img edit.
    assert!(matches!(
        sdxl_sub_mode(&request(
            json!({ "model": "sdxl", "mode": "edit_image", "sourceAssetId": "src_1" })
        )),
        Some(SdxlSubMode::Edit)
    ));
    // edit_image + source + mask → inpaint.
    assert!(matches!(
        sdxl_sub_mode(&request(json!({
            "model": "sdxl", "mode": "edit_image",
            "sourceAssetId": "src_1", "maskAssetId": "mask_1"
        }))),
        Some(SdxlSubMode::Inpaint)
    ));
    // fit_mode outpaint wins over a user mask (the torch path checks outpaint first,
    // then unions the user mask into the generated border).
    assert!(matches!(
        sdxl_sub_mode(&request(json!({
            "model": "sdxl", "mode": "edit_image", "sourceAssetId": "src_1",
            "fitMode": "outpaint", "maskAssetId": "mask_1"
        }))),
        Some(SdxlSubMode::Outpaint)
    ));
    // edit_image without a source → nothing to do (falls through, not advanced).
    assert!(sdxl_sub_mode(&request(json!({ "model": "sdxl", "mode": "edit_image" }))).is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn engine_dim_rounds_up_to_mult8_and_clamps() {
    assert_eq!(engine_dim(1024), 1024); // already valid
    assert_eq!(engine_dim(1000), 1000); // already a multiple of 8
    assert_eq!(engine_dim(1001), 1008); // rounds up to the next multiple of 8
    assert_eq!(engine_dim(500), 512); // clamps to the engine minimum
    assert_eq!(engine_dim(3000), 2048); // clamps to the engine maximum
}

#[cfg(target_os = "macos")]
#[test]
fn detail_feather_ramps_over_overlap() {
    // No overlap → a flat field of ones (every pixel contributes fully).
    let flat = detail_feather(8, 8, 0);
    assert_eq!(flat.len(), 64);
    assert!(flat.iter().all(|&w| (w - 1.0).abs() < 1e-6));

    // With overlap, the borders ramp down (raised cosine) while the center stays 1.0.
    let f = detail_feather(16, 16, 4);
    assert_eq!(f.len(), 256);
    let at = |x: usize, y: usize| f[y * 16 + x];
    assert!((at(8, 8) - 1.0).abs() < 1e-6, "center is full weight");
    assert!(at(0, 0) < at(8, 8), "corner is feathered below center");
    // Symmetric across the tile.
    assert!((at(0, 8) - at(15, 8)).abs() < 1e-6);
    assert!((at(8, 0) - at(8, 15)).abs() < 1e-6);
}

/// sc-3625 real-Mac E2E (epic 3621): drive the WORKER's FLUX.1 XLabs IP-Adapter reference path
/// end to end on real weights — `resolve_flux_ip_adapter_dir` staging from the real HF cache +
/// `load_engine` + a real `Conditioning::Reference` dev `true_cfg` render against the
/// pinned mlx-gen engine. Guards the worker plumbing the engine-side A/B can't: the staged-dir
/// contract + the dev reference render NOT regressing to the pre-#173 saturation (which
/// collapsed `true_cfg=4` to a near-uniform white frame). Run (needs FLUX.1-dev +
/// `XLabs-AI/flux-ip-adapter` + `openai/clip-vit-large-patch14` in the HF cache):
/// ```text
/// HF_HUB_CACHE=$HOME/.cache/huggingface/hub \
///   cargo test -p sceneworks-worker --release flux_ip_reference_worker_e2e -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-Mac E2E: loads FLUX.1-dev + XLabs IP-Adapter + CLIP-ViT-L from the HF cache"]
fn flux_ip_reference_worker_e2e() {
    fn hf_cache() -> String {
        std::env::var("HF_HUB_CACHE").unwrap_or_else(|_| {
            format!("{}/.cache/huggingface/hub", std::env::var("HOME").unwrap())
        })
    }
    fn hf_snapshot(repo: &str, needs: &str) -> PathBuf {
        let safe = repo.replace('/', "--");
        let snaps = PathBuf::from(hf_cache())
            .join(format!("models--{safe}"))
            .join("snapshots");
        std::fs::read_dir(&snaps)
            .unwrap_or_else(|_| panic!("HF snapshot for {repo}"))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.join(needs).exists())
            .unwrap_or_else(|| panic!("a complete {repo} snapshot with {needs}"))
    }

    // Point the worker's HF-cache resolver + Settings at the real cache + a temp data dir.
    std::env::set_var("HF_HUB_CACHE", hf_cache());
    let data = tempfile::tempdir().unwrap();
    std::env::set_var("SCENEWORKS_DATA_DIR", data.path());
    let settings = Settings::from_env();

    // (1) The worker's OWN staging fn, against the real cache (the net-new sc-3625 fs logic).
    let staged = resolve_flux_ip_adapter_dir(&settings).expect("stage flux ip-adapter dir");
    assert!(
        staged.join("ip_adapter.safetensors").exists(),
        "staged ip_adapter.safetensors"
    );
    assert!(
        staged.join("image_encoder/model.safetensors").exists(),
        "staged image_encoder/model.safetensors"
    );

    // (2) Load FLUX.1-dev through the worker loader with the staged IP dir, resolving the
    // engine id from the engines::MODEL_TABLE exactly as the real dispatch does (model "flux_dev"
    // → engine "flux1_dev").
    let engine_id = mlx_model("flux_dev")
        .expect("flux_dev in MODEL_TABLE")
        .engine_id();
    let flux_dev = hf_snapshot("black-forest-labs/FLUX.1-dev", "transformer");
    let generator = load_engine(engine_id, flux_dev, None, vec![], Some(staged))
        .unwrap_or_else(|e| panic!("load_engine {engine_id} + ip: {e}"));

    // (3) Reference render through the dev `true_cfg` path (white-dot garbage pre-#173).
    let reference = {
        let p = "/tmp/flux_ab/reference.png";
        if std::path::Path::new(p).exists() {
            let img = image::open(p).unwrap().to_rgb8();
            Image {
                width: img.width(),
                height: img.height(),
                pixels: img.into_raw(),
            }
        } else {
            // Synthetic fallback: a solid orange field (still drives the IP branch).
            Image {
                width: 64,
                height: 64,
                pixels: [255u8, 140, 0]
                    .iter()
                    .cycle()
                    .take(64 * 64 * 3)
                    .copied()
                    .collect(),
            }
        }
    };
    let req = |conditioning, true_cfg| GenerationRequest {
        prompt: "an oil painting in the bold swirling brushstroke style of Van Gogh".into(),
        width: 512,
        height: 512,
        seed: Some(2),
        steps: Some(16),
        true_cfg,
        conditioning,
        ..Default::default()
    };
    let run = |r: &GenerationRequest| match generator.generate(r, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => unreachable!(),
    };

    let ref_out = run(&req(
        vec![Conditioning::Reference {
            image: reference,
            strength: Some(0.7),
        }],
        Some(4.0),
    ));
    let plain = run(&req(vec![], None));

    // Non-degenerate: the pre-#173 saturation collapsed dev true_cfg=4 to a near-uniform white.
    let n = ref_out.pixels.len() as f64;
    let mean = ref_out.pixels.iter().map(|&b| b as f64).sum::<f64>() / n;
    let var = ref_out
        .pixels
        .iter()
        .map(|&b| (b as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    assert!(
        var > 200.0,
        "reference render near-uniform (var={var:.1}) — true_cfg saturation regression"
    );
    assert!(
        mean < 245.0,
        "reference render near-white (mean={mean:.1}) — true_cfg saturation regression"
    );

    // The reference actually changed the image vs plain txt2img (IP branch is applied).
    let diff = ref_out
        .pixels
        .iter()
        .zip(&plain.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diff > ref_out.pixels.len() / 10,
        "reference barely changed the render ({diff} px)"
    );

    if let Ok(p) = std::env::var("FLUX_IP_WORKER_OUT") {
        image::RgbImage::from_raw(ref_out.width, ref_out.height, ref_out.pixels.clone())
            .unwrap()
            .save(&p)
            .unwrap();
        println!("[worker-e2e] wrote {p}");
    }
    println!("[worker-e2e] OK: staged dir contract + flux_dev load + dev true_cfg=4 reference render — var={var:.1} mean={mean:.1} diff-vs-txt2img={diff}px");
}

// ─────────────────────────────────────────────────────────────────────────────
// Ideogram 4 (epic 4725, sc-6000): worker-path validation. CI-safe mapping/lineage
// unit tests + a real-weight #[ignore] smoke through the registry load seam for
// both a structured JSON caption and a plain-text prompt.
// ─────────────────────────────────────────────────────────────────────────────

/// Ideogram 4 resolves to its own engine + the V4_QUALITY_48 defaults, and quant resolution
/// returns Q4 — the manifest declares `mlx.quantize: 4` to match the packed `q4/` default subdir
/// the model runs (sc-6237), so the recorded recipe quant is honest. Runs in CI on Mac (no weights).
#[cfg(target_os = "macos")]
#[test]
fn ideogram_engine_defaults_and_quant_resolution() {
    let model = mlx_model("ideogram_4").expect("ideogram_4 in MODEL_TABLE");
    assert_eq!(model.engine_id(), "ideogram_4");
    assert_eq!(model.adapter_label(), "mlx_ideogram");
    assert_eq!(model.default_steps(), 48); // V4_QUALITY_48 preset

    // The catalog entry carries `mlx.quantize: 4` so the resolved quant matches the packed q4/
    // default the model actually runs (sc-6237) — without it, resolve_quant would record the
    // generic Q8 in the recipe.
    let req = |advanced: Value| {
        request(json!({
            "projectId": "p", "model": "ideogram_4", "prompt": "p", "advanced": advanced,
            "modelManifestEntry": { "mlx": { "quantize": 4 } },
        }))
    };
    // Asymmetric-CFG guidance 7.0 from the model row.
    assert_eq!(resolve_guidance(&req(json!({})), &model), Some(7.0));
    // Default → Q4 (manifest packed default); advanced.mlxQuantize overrides to Q8 / bf16-dense.
    assert!(matches!(
        resolve_quant(&req(json!({}))),
        (Some(Quant::Q4), Some(4))
    ));
    assert!(matches!(
        resolve_quant(&req(json!({ "mlxQuantize": 8 }))),
        (Some(Quant::Q8), Some(8))
    ));
    assert!(matches!(
        resolve_quant(&req(json!({ "mlxQuantize": 4 }))),
        (Some(Quant::Q4), Some(4))
    ));
    assert!(matches!(
        resolve_quant(&req(json!({ "mlxQuantize": 0 }))),
        (None, None)
    ));
}

/// `ideogram_model_subdir` picks the packed `q4/` subdir by default and `q8/` only when the request
/// opts in (`mlxQuantize > 4`) AND q8 is downloaded, falling back to q4 (then the root, so a
/// half-downloaded bundle surfaces as a load error rather than a silent half-load). This is the
/// *effective* quant selection — the q4 default is why resolve_quant's generic Q8 is inert.
#[cfg(target_os = "macos")]
#[test]
fn ideogram_subdir_prefers_q4_and_opts_into_q8() {
    let root = tempfile::tempdir().unwrap();
    let touch = |sub: &str| {
        let dir = root.path().join(sub).join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"stub").unwrap();
    };
    let req = |advanced: Value| {
        request(
            json!({ "projectId": "p", "model": "ideogram_4", "prompt": "p", "advanced": advanced }),
        )
    };
    // Neither subdir present → root.
    assert_eq!(
        ideogram_model_subdir(root.path(), &req(json!({}))),
        root.path()
    );
    // q4 only → q4, even when q8 is requested but absent.
    touch("q4");
    assert_eq!(
        ideogram_model_subdir(root.path(), &req(json!({}))),
        root.path().join("q4")
    );
    assert_eq!(
        ideogram_model_subdir(root.path(), &req(json!({ "mlxQuantize": 8 }))),
        root.path().join("q4"),
        "q8 opt-in falls back to q4 when q8 absent",
    );
    // Both present → q4 by default, q8 only on opt-in.
    touch("q8");
    assert_eq!(
        ideogram_model_subdir(root.path(), &req(json!({}))),
        root.path().join("q4")
    );
    assert_eq!(
        ideogram_model_subdir(root.path(), &req(json!({ "mlxQuantize": 8 }))),
        root.path().join("q8"),
    );
}

/// Raw-settings lineage: the Ideogram recipe records the real-inference flag, repo, resolved
/// steps/guidance/quant, AND the sc-6147 structured-prompt blob passes through verbatim
/// (`mlx_raw_settings` clones `advanced`, so the asset recipe can rehydrate the builder).
#[cfg(target_os = "macos")]
#[test]
fn ideogram_raw_settings_records_recipe_and_structured_prompt() {
    let structured = json!({
        "version": 1,
        "intent": "a red fox in the snow",
        "caption": { "compositional_deconstruction": { "background": "snow", "elements": [] } },
        "magicPromptBackend": "prompt_refine",
        "edited": false,
        "runtimePrompt": "{}",
    });
    let req = request(json!({
        "projectId": "p", "model": "ideogram_4", "prompt": "{}",
        "advanced": { "structuredPrompt": structured.clone() },
        "modelManifestEntry": { "mlx": { "quantize": 4 } },
    }));
    // Resolve the quant the real path would (manifest packed default → Q4, sc-6237), then record it.
    let (_quant, quant_bits) = resolve_quant(&req);
    let raw = mlx_raw_settings(&req, "SceneWorks/ideogram-4-mlx", 48, quant_bits, Some(7.0));
    assert_eq!(raw.get("realModelInference"), Some(&json!(true)));
    assert_eq!(raw.get("repo"), Some(&json!("SceneWorks/ideogram-4-mlx")));
    assert_eq!(raw.get("numInferenceSteps"), Some(&json!(48)));
    assert_eq!(raw.get("guidanceScale").and_then(Value::as_f64), Some(7.0));
    // The recipe records the *effective* q4 — not the old generic Q8 (sc-6237).
    assert_eq!(raw.get("mlxQuantize"), Some(&json!(4)));
    // sc-6147: the structured-prompt blob survives into rawAdapterSettings unchanged, so
    // "Use this recipe" can rehydrate the structured builder.
    assert_eq!(raw.get("structuredPrompt"), Some(&structured));
}

/// Resolve the Ideogram 4 turnkey's packed `q4/` subdir for the real-weight smoke: env override →
/// the cached public-but-gated `SceneWorks/ideogram-4-mlx` snapshot. `None` ⇒ skip (weights live
/// outside CI).
#[cfg(target_os = "macos")]
fn ideogram_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let has_transformer = |dir: &Path| dir.join("transformer/model.safetensors").is_file();
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_IDEOGRAM_DIR") {
        let path = PathBuf::from(dir.trim());
        if has_transformer(&path) {
            return Some(path);
        }
        let q4 = path.join("q4");
        if has_transformer(&q4) {
            return Some(q4);
        }
    }
    let snaps =
        dirs_home().join(".cache/huggingface/hub/models--SceneWorks--ideogram-4-mlx/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|entry| entry.path().join("q4"))
        .find(|q4| has_transformer(q4))
}

/// Real-weights smoke (sc-6000): drive Ideogram 4 through the SAME `load_engine` →
/// `gen_core::load("ideogram_4")` seam the worker uses (proving the `mlx_gen_ideogram` force-link
/// survives in the worker binary), with the resolve_quant Q8 spec over the packed q4/ weights —
/// exactly the production load. Generates from BOTH a structured JSON caption and a plain-text
/// prompt on one load, asserting each returns a non-constant RGB8 image with denoise progress.
/// Steps default low (env `IDEOGRAM4_SMOKE_STEPS` / `IDEOGRAM4_SMOKE_RES`): this checks mechanics,
/// not quality (Ideogram undercooks below ~50 steps — see epic notes). Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored ideogram_4_real_weights --nocapture`.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Ideogram 4 snapshot; run manually on a Mac with SceneWorks/ideogram-4-mlx cached"]
#[test]
fn ideogram_4_real_weights_generates_caption_and_plain_images() {
    let Some(dir) = ideogram_dir() else {
        eprintln!(
            "skipping ideogram_4_real_weights: no SceneWorks/ideogram-4-mlx q4 snapshot found"
        );
        return;
    };
    let env_u32 = |key: &str, default: u32| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let steps = env_u32("IDEOGRAM4_SMOKE_STEPS", 8);
    let res = env_u32("IDEOGRAM4_SMOKE_RES", 512);

    let model = mlx_model("ideogram_4").unwrap();
    let req = request(json!({
        "projectId": "p", "model": "ideogram_4", "prompt": "p", "advanced": {},
        "modelManifestEntry": { "mlx": { "quantize": 4 } },
    }));
    // Q4 spec matching the packed q4 weights — the exact production path (sc-6237).
    let (quant, _bits) = resolve_quant(&req);
    let guidance = resolve_guidance(&req, &model); // 7.0

    let generator = load_engine("ideogram_4", dir, quant, Vec::new(), None).unwrap();
    let cancel = gen_core::CancelFlag::new();
    // No caption-upsampling for the smoke (sc-6135 is FLUX.2-dev-only; inert for Ideogram).
    let enhance = PromptEnhance::default();

    // A valid Ideogram caption (required compositional_deconstruction + a high-level description),
    // and the plain-text fallback the same scene reads as.
    const CAPTION_JSON: &str = "{\"high_level_description\": \"A photograph of a red fox sitting in a snowy forest at golden hour.\", \"compositional_deconstruction\": {\"background\": \"A snowy pine forest at golden hour.\", \"elements\": [{\"type\": \"obj\", \"bbox\": [250, 320, 950, 760], \"desc\": \"A red fox sitting upright in the snow, facing the camera.\"}]}}";
    const PLAIN_TEXT: &str = "a red fox sitting in a snowy forest at golden hour";

    for (label, prompt) in [("json_caption", CAPTION_JSON), ("plain_text", PLAIN_TEXT)] {
        let mut steps_seen = 0u32;
        let (w, h, pixels) = generate_one(
            generator.as_ref(),
            prompt,
            res,
            res,
            42,
            steps,
            guidance,
            None,
            None,
            None,
            None,
            None,
            None,
            &enhance,
            &cancel,
            &mut |p| {
                if let gen_core::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap_or_else(|error| panic!("ideogram {label} generation failed: {error}"));
        assert_eq!(
            pixels.len(),
            (w * h * 3) as usize,
            "{label}: RGB8-sized buffer"
        );
        assert!(
            w == res && h == res,
            "{label}: output matches requested {res}²"
        );
        assert!(steps_seen >= 1, "{label}: expected denoise step progress");
        assert!(
            pixels.windows(2).any(|x| x[0] != x[1]),
            "{label}: non-constant image"
        );
        eprintln!("ideogram {label}: {w}x{h} RGB8, {steps_seen} steps observed");
    }
}
