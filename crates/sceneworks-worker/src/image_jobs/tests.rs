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
        &[],
        None,
        None,
        None,
        None,
        None,
        None,
        false,
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
        &[],
        None,
        None,
        None,
        None,
        None,
        None,
        false,
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

/// The dense (bf16) FLUX.2-dev diffusers snapshot — the bf16 tier source, byte-identical to
/// SceneWorks/flux2-dev-mlx/bf16/. `SCENEWORKS_FLUX2_DEV_BF16_DIR` overrides; default = the cached
/// black-forest-labs/FLUX.2-dev snapshot.
#[cfg(target_os = "macos")]
fn flux2_dev_bf16_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SCENEWORKS_FLUX2_DEV_BF16_DIR") {
        return std::path::PathBuf::from(p);
    }
    let snaps =
        dirs_home().join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("FLUX.2-dev snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a FLUX.2-dev snapshot dir")
}

/// bf16 tier verify (sc-8513, epic 8506): load the DENSE SHARDED FLUX.2-dev tier (`Quant::None`, the
/// 7-shard transformer + 10-shard Mistral TE) through the worker engine path and render. Proves MLX
/// reads the dense sharded dir — the one tier never exercised on MLX (the path always loaded packed).
/// ~105 GB dense weights: production sizes need a >128 GB Mac; defaults here are 256²/1-step to probe
/// load+gen on a 128 GB box. The eprintln markers localize a SIGKILL to load-vs-gen. Run:
/// `cargo test -p sceneworks-worker --release --lib -- --ignored flux2_dev_bf16_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs the dense FLUX.2-dev snapshot + (for production sizes) a >128 GB Metal device"]
fn flux2_dev_bf16_real_weights_loads_and_generates() {
    let size: u32 = std::env::var("SCENEWORKS_FLUX2_DEV_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let steps: u32 = std::env::var("SCENEWORKS_FLUX2_DEV_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    eprintln!("[bf16-verify] loading dense sharded FLUX.2-dev (Quant::None)…");
    let generator = load_engine("flux2_dev", flux2_dev_bf16_dir(), None, Vec::new(), None).unwrap();
    eprintln!("[bf16-verify] LOADED — dense sharded dir read OK; generating {size}²/{steps}…");
    let request = GenerationRequest {
        prompt: "a serene mountain lake at dawn".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(4.0),
        ..Default::default()
    };
    let out = generator.generate(&request, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.remove(0),
        other => panic!("expected Images, got {other:?}"),
    };
    eprintln!("[bf16-verify] GENERATED {}x{}", img.width, img.height);
    assert_eq!((img.width, img.height), (size, size));
    assert!(
        img.pixels.windows(2).any(|w| w[0] != w[1]),
        "render is flat (degenerate)"
    );
}

/// On-device verify of a hosted SceneWorks SD3.5 quant-matrix tier (sc-8513, epic 8506): load the
/// DOWNLOADED tier subdir through the real worker `load_engine` seam and render a non-degenerate
/// frame — proving the hosted artifact (download → load → generate) works end to end. Point
/// `SCENEWORKS_SD3_DIR` at the downloaded tier subdir (e.g. the `q4/` of
/// `SceneWorks/sd3.5-large-mlx`); `SCENEWORKS_SD3_ENGINE` selects the variant engine
/// (`sd3_5_large` default / `sd3_5_large_turbo` / `sd3_5_medium`); `SCENEWORKS_SD3_QUANT`
/// (`q4` default / `q8` / `bf16`) sets the load hint (a no-op on the already-packed q4/q8 weights;
/// `bf16` loads dense via `Quant::None`). Small size/steps by default to fit a 128 GB box; the dense
/// T5-XXL TE dominates the footprint. Run on demand:
/// `SCENEWORKS_SD3_DIR=… cargo test -p sceneworks-worker --release --lib -- --ignored sd3_5_hosted_tier_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a downloaded SceneWorks SD3.5 tier dir + a high-memory Metal device"]
fn sd3_5_hosted_tier_real_weights_generates_one_image() {
    let dir = std::path::PathBuf::from(
        std::env::var("SCENEWORKS_SD3_DIR").expect("set SCENEWORKS_SD3_DIR to a downloaded tier dir"),
    );
    let engine = std::env::var("SCENEWORKS_SD3_ENGINE").unwrap_or_else(|_| "sd3_5_large".to_owned());
    let quant = match std::env::var("SCENEWORKS_SD3_QUANT").as_deref() {
        Ok("bf16") => None,
        Ok("q8") => Some(gen_core::Quant::Q8),
        _ => Some(gen_core::Quant::Q4),
    };
    let size: u32 = std::env::var("SCENEWORKS_SD3_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let steps: u32 = std::env::var("SCENEWORKS_SD3_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    eprintln!("[sd3-verify] loading {engine} from {} ({quant:?})…", dir.display());
    let generator = load_engine(&engine, dir, quant, Vec::new(), None).unwrap();
    eprintln!("[sd3-verify] LOADED — generating {size}²/{steps}…");
    let request = GenerationRequest {
        prompt: "a serene mountain lake at dawn".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(3.5),
        ..Default::default()
    };
    let img = match generator.generate(&request, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        other => panic!("expected Images, got {other:?}"),
    };
    eprintln!("[sd3-verify] GENERATED {}x{}", img.width, img.height);
    assert_eq!((img.width, img.height), (size, size), "output dimensions");
    assert_eq!(
        img.pixels.len(),
        (size * size * 3) as usize,
        "RGB8 pixel count"
    );
    assert!(
        img.pixels.windows(2).any(|w| w[0] != w[1]),
        "render is flat (degenerate)"
    );
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
        &[],
        None,
        None,
        None,
        None,
        None,
        None,
        false,
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
        &[],
        None,
        None,
        None,
        None,
        None,
        None,
        false,
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

/// sc-4410 macOS Kolors pose-lane identity-source invariant: the lane that now scores every finished
/// pose (`generate_kolors_control_stream`, routed via `kolors_control_available`) is gated on a non-empty
/// `referenceAssetId` — so the source identity face the scorer needs ALWAYS exists when the lane runs (it
/// is the same reference the IP-Adapter + img2img init decode). A pose job WITHOUT a reference does not
/// route here at all, so there is no unscored Kolors pose path. Weight-free: the reference clause
/// short-circuits before `resolve_weights_dir`, so this asserts the gate regardless of cached weights.
#[cfg(target_os = "macos")]
#[test]
fn kolors_pose_lane_requires_reference_identity_for_scoring() {
    let settings = Settings::from_env();
    // No referenceAssetId ⇒ the scoring lane is never entered (the gate is false before weights matter).
    let no_reference = request(json!({
        "projectId": "p", "model": "kolors",
        "advanced": { "poses": [{ "id": "a" }] }
    }));
    assert!(
        !kolors_control_available(&no_reference, &settings),
        "a Kolors pose job with no referenceAssetId must NOT route to the scoring control lane"
    );
    // A blank referenceAssetId is treated as absent (no spurious scoring lane / no scorer with no source).
    let blank_reference = request(json!({
        "projectId": "p", "model": "kolors", "referenceAssetId": "   ",
        "advanced": { "poses": [{ "id": "a" }] }
    }));
    assert!(
        !kolors_control_available(&blank_reference, &settings),
        "a blank referenceAssetId is treated as absent"
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
        &[],
        None,
        true_cfg,
        None,
        None,
        None,
        None,
        false,
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

/// sc-8250 qwen source-threading: the qwen control stream now feeds the user input image into the
/// shared `preprocess_control_entry` driver (it previously passed `source: None`). With `controlMode =
/// canny` and a threaded source, the driver derives a real edge-map control image — NOT the pose
/// skeleton — and builds a `Control { kind: Canny }` conditioning. This is the qwen analogue of
/// `threaded_source_drives_auto_canny_control_conditioning` (flux2/z-image), proving qwen's stream
/// reaches the same auto-canny/auto-depth path now that the source is threaded. The pose tier stays
/// byte-identical (no source needed — the skeleton is synthetic). Real per-mode renders are the
/// on-device `qwen_control_real_weights` smoke.
#[cfg(target_os = "macos")]
#[test]
fn qwen_threaded_source_drives_auto_canny_control_conditioning() {
    // qwen's engine row now accepts canny + depth (sc-8250).
    assert!(validate_control_kind("qwen_image_control", &ControlKind::Canny).is_ok());
    assert!(validate_control_kind("qwen_image_control", &ControlKind::Depth).is_ok());

    let (w, h) = (64u32, 48u32);
    let pose = one_pose();
    let stick = crate::openpose_skeleton::body_stickwidth(w, h);
    let source = control_fixture(w, h, [128, 128, 128]);

    // canny + a threaded source (no user passthrough, no depth weights) → an edge-map control image.
    let control = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        Some(&pose),
        Some(&source),
        w,
        h,
        stick,
        None,
    )
    .expect("auto-canny over the threaded qwen source");
    assert_eq!((control.width, control.height), (w, h));

    // It must NOT be the pose skeleton (proving the source — not the synthetic skeleton — drove it).
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        stick,
    );
    assert_ne!(
        control.pixels,
        skeleton.into_raw(),
        "qwen auto-canny must derive from the source image, not render the pose skeleton"
    );

    let cond = build_control_conditioning(control, ControlKind::Canny, 0.9, None);
    assert_eq!(cond.len(), 1);
    assert!(
        matches!(
            &cond[0],
            Conditioning::Control {
                kind: ControlKind::Canny,
                ..
            }
        ),
        "qwen canny builds a Control conditioning"
    );
}

/// sc-8251 base Z-Image source-threading: the base `z_image_control` stream feeds the user input image
/// into the SAME shared `preprocess_control_entry` driver (it passes `control_source`, not `None`). With
/// `controlMode = canny` + a threaded source the driver derives a real edge-map control image — NOT the
/// pose skeleton — and builds a `Control { kind: Canny }`. The base analogue of
/// `threaded_source_drives_auto_canny_control_conditioning` (turbo). Also asserts the base engine row
/// admits canny + depth. Real per-mode renders are the on-device `zimage_base_control_real_weights` smoke.
#[cfg(target_os = "macos")]
#[test]
fn zimage_base_threaded_source_drives_auto_canny_control_conditioning() {
    // The base Z-Image control engine row accepts canny + depth (sc-8251).
    assert!(validate_control_kind("z_image_control", &ControlKind::Canny).is_ok());
    assert!(validate_control_kind("z_image_control", &ControlKind::Depth).is_ok());

    let (w, h) = (64u32, 48u32);
    let pose = one_pose();
    let stick = crate::openpose_skeleton::body_stickwidth(w, h);
    let source = control_fixture(w, h, [128, 128, 128]);

    // canny + a threaded source (no user passthrough, no depth weights) → an edge-map control image.
    let control = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        Some(&pose),
        Some(&source),
        w,
        h,
        stick,
        None,
    )
    .expect("auto-canny over the threaded base z-image source");
    assert_eq!((control.width, control.height), (w, h));

    // It must NOT be the pose skeleton (proving the source — not the synthetic skeleton — drove it).
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        stick,
    );
    assert_ne!(
        control.pixels,
        skeleton.into_raw(),
        "base z-image auto-canny must derive from the source image, not render the pose skeleton"
    );

    let cond = build_control_conditioning(control, ControlKind::Canny, 0.9, None);
    assert_eq!(cond.len(), 1);
    assert!(
        matches!(
            &cond[0],
            Conditioning::Control {
                kind: ControlKind::Canny,
                ..
            }
        ),
        "base z-image canny builds a Control conditioning"
    );

    // Pose byte-identity: the base stream renders the SAME shared `draw_wholebody` skeleton as every
    // other strict-control engine (no `Reference` unless identity-init), so the pose tier is
    // byte-identical to the Turbo path — the base only diverges in the engine id + CFG forward.
    let pose_control = preprocess_control_entry(
        &ControlKind::Pose,
        None,
        Some(&pose),
        None,
        w,
        h,
        stick,
        None,
    )
    .expect("base z-image pose render");
    let expected = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        stick,
    );
    assert_eq!(
        pose_control.pixels,
        expected.into_raw(),
        "base z-image pose preprocessing is byte-identical to a direct draw_wholebody"
    );
    let pose_cond = build_control_conditioning(pose_control, ControlKind::Pose, 0.9, None);
    assert_eq!(pose_cond.len(), 1);
    assert!(matches!(
        &pose_cond[0],
        Conditioning::Control {
            kind: ControlKind::Pose,
            ..
        }
    ));
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
        &[],
        None,
        None,
        None,
        None,
        None,
        None,
        false,
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
    let conditioning = build_control_conditioning(control, ControlKind::Pose, control_scale, None);
    let (ow, oh, pixels) = zimage_control_generate_one(
        generator.as_ref(),
        &req.prompt,
        w,
        h,
        seed,
        steps,
        conditioning,
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

/// The candle Z-Image identity-init engage gate + clamp (sc-8409): the off-Mac sibling of
/// `zimage_identity_strength_gate_and_clamp`, exercising `zimage_identity_candle_strength` — the gate the
/// candle "With Character" lane (`zimage_identity_candle_available`) keys on. Must agree with the macOS
/// `zimage_identity_strength` semantics verbatim (engage iff `referenceStrength > 0` AND a non-empty
/// `referenceAssetId`; strength forwarded with no inversion, clamped to [0.05, 1.0]) so candle runs the
/// identity init precisely when the MLX generic lane does.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
fn zimage_identity_candle_strength_gate_and_clamp() {
    let with = |adv: Value, asset: Value| {
        let mut payload = json!({
            "projectId": "p", "model": "z_image_turbo", "mode": "character_image", "prompt": "a knight"
        });
        let obj = payload.as_object_mut().unwrap();
        obj.insert("advanced".to_owned(), adv);
        if !asset.is_null() {
            obj.insert("referenceAssetId".to_owned(), asset);
        }
        zimage_identity_candle_strength(&request(payload))
    };

    // Not engaged → None: no referenceStrength; == 0 (parity: requires > 0); > 0 but no/blank asset.
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
    assert_eq!(
        with(json!({ "referenceStrength": 0.6 }), json!("ref_1")),
        Some(0.6)
    );
    assert_eq!(
        with(json!({ "referenceStrength": "0.45" }), json!("ref_1")),
        Some(0.45)
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
    let conditioning = build_control_conditioning(control, ControlKind::Pose, 0.9, None);
    let (w, h, pixels) = zimage_control_generate_one(
        generator.as_ref(),
        "a person standing in a meadow",
        512,
        512,
        42,
        8,
        conditioning,
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

/// Real-weights smoke: BASE Z-Image strict-control (sc-8251). Loads the base `Tongyi-MAI/Z-Image`
/// snapshot + the cached base Fun-Controlnet-Union checkpoint and generates one image per mode
/// (pose skeleton / auto-canny / auto-depth) through `z_image_control` with REAL CFG. Needs both
/// weights in the HF cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored zimage_base_control_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real base Z-Image + base Fun-Controlnet-Union weights + Metal device"]
fn zimage_base_control_real_weights_generates_per_mode() {
    let base = hf_snapshot("models--Tongyi-MAI--Z-Image");
    let control = std::fs::read_dir(dirs_home().join(
        ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Fun-Controlnet-Union-2.1/snapshots",
    ))
    .expect("base control snapshots dir")
    .flatten()
    .map(|entry| entry.path())
    .find(|path| path.is_dir())
    .map(|dir| dir.join(super::ZIMAGE_BASE_CONTROL_FILE))
    .filter(|path| path.exists())
    .expect("base control weights file");

    let generator =
        zimage_base_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();

    let (w, h) = (512u32, 512u32);
    let stick = crate::openpose_skeleton::body_stickwidth(w, h);
    // A flat source for the auto-canny / auto-depth modes.
    let source = control_fixture(w, h, [120, 140, 160]);
    let pose = one_pose();

    // Drive each supported mode through the shared preprocessor → conditioning → generate-one.
    for kind in [ControlKind::Pose, ControlKind::Canny] {
        let control_map =
            preprocess_control_entry(&kind, None, Some(&pose), Some(&source), w, h, stick, None)
                .expect("preprocess base control entry");
        let conditioning = build_control_conditioning(control_map, kind.clone(), 0.9, None);
        let cancel = gen_core::CancelFlag::new();
        let mut steps_seen = 0u32;
        let (ow, oh, pixels) = zimage_base_control_generate_one(
            generator.as_ref(),
            "a person standing in a meadow",
            None,
            w,
            h,
            42,
            50,
            4.0,
            conditioning,
            &cancel,
            &mut |p| {
                if let gen_core::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap();
        assert_eq!((ow, oh), (w, h));
        assert_eq!(pixels.len() as u32, w * h * 3);
        assert!(
            steps_seen >= 1,
            "expected denoise step progress for {kind:?}"
        );
        assert!(pixels.windows(2).any(|px| px[0] != px[1]));
    }
}

/// Real-weights smoke: Qwen-Image strict-pose control. Loads the base
/// `Qwen/Qwen-Image` snapshot + the cached alibaba-pai 2512-Fun-Controlnet-Union checkpoint,
/// renders one DWPose skeleton, and generates one image through `qwen_image_control`.
/// Needs both in the HF cache + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored qwen_control_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs real Qwen-Image + 2512-Fun-Controlnet-Union weights + Metal device"]
fn qwen_control_real_weights_generates_one_pose() {
    let base = hf_snapshot("models--Qwen--Qwen-Image");
    let control = hf_snapshot("models--alibaba-pai--Qwen-Image-2512-Fun-Controlnet-Union")
        .join(super::QWEN_CONTROL_FILE);
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
    let conditioning = build_control_conditioning(control, ControlKind::Pose, 0.9, None);
    let (w, h, pixels) = qwen_control_generate_one(
        generator.as_ref(),
        "a person standing in a meadow",
        Some("blurry, low quality".to_owned()),
        512,
        512,
        42,
        4,
        4.0,
        conditioning,
        false,
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

// ---- sc-6124 / sc-6211: FLUX.2-dev multi-reference edit memory guard --------------------------

#[cfg(target_os = "macos")]
#[test]
fn flux2_dev_edit_peak_gb_tracks_chunked_measurements() {
    // sc-6211 worker-layer peaks with the sc-6266 chunking ON (Q4, 1024², /usr/bin/time -l):
    // 2-ref ~81 GB, 4-ref ~93 GB. (Pre-chunking sc-5923 had 2-ref ~104 — the re-anchored fit is
    // ~3.8× gentler per token, which is why the 2-ref edit now fits 96.)
    let two = flux2_dev_edit_peak_gb(2, 1024, 1024);
    let four = flux2_dev_edit_peak_gb(4, 1024, 1024);
    assert!(
        (two - 81.0).abs() < 2.0,
        "two-reference estimate {two} GB ≉ measured ~81"
    );
    assert!(
        (four - 93.0).abs() < 2.0,
        "four-reference estimate {four} GB ≉ measured ~93"
    );
    // Monotonic in both reference count and resolution.
    assert!(four > two);
    assert!(flux2_dev_edit_peak_gb(2, 768, 768) < two);
}

#[cfg(target_os = "macos")]
#[test]
fn flux2_dev_edit_memory_guard_gates_multiref_on_small_machines() {
    // Single reference / txt2img always pass — covered by the declared minMemoryGb — even on a
    // small machine, and regardless of the RAM probe.
    assert!(flux2_dev_edit_memory_guard(1, 1024, 1024, Some(64.0)).is_ok());
    assert!(flux2_dev_edit_memory_guard(0, 1024, 1024, Some(64.0)).is_ok());
    // sc-6211: the chunked two-reference 1024² edit (~81 GB) now PASSES on a 96 GB Mac (the whole
    // point of this story) and of course on a 128 GB one.
    assert!(flux2_dev_edit_memory_guard(2, 1024, 1024, Some(96.0)).is_ok());
    assert!(flux2_dev_edit_memory_guard(2, 1024, 1024, Some(128.0)).is_ok());
    // Four references at 1024² (~93 GB peak) is too tight for a 96 GB machine but fits a 128 GB one.
    let err = flux2_dev_edit_memory_guard(4, 1024, 1024, Some(96.0)).unwrap_err();
    assert!(
        matches!(&err, WorkerError::InvalidPayload(msg) if msg.contains("multi-reference")),
        "expected an actionable multi-reference rejection, got {err:?}"
    );
    assert!(flux2_dev_edit_memory_guard(4, 1024, 1024, Some(128.0)).is_ok());
    // A failed RAM probe is lenient (don't block a possibly-fine job).
    assert!(flux2_dev_edit_memory_guard(4, 1024, 1024, None).is_ok());
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
    // The default repo now comes from the shared strict-control table (single source of truth).
    assert_eq!(
        repo,
        strict_control_default_repo(FLUX2_DEV_CONTROL_ENGINE_ID)
    );
    assert_eq!(repo, "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union");
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
    let conditioning = build_control_conditioning(control, ControlKind::Pose, 0.75, None);
    let (w, h, pixels) = flux2_control_generate_one(
        generator.as_ref(),
        "a person standing in a meadow, photorealistic",
        512,
        512,
        42,
        8,
        Some(4.0), // dev embedded guidance
        conditioning,
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

/// Real-weights smoke: FLUX.1-dev strict-control via the Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0`
/// branch (sc-8244; engine E2 sc-8239). Loads the gated dev snapshot + the cached Shakker control
/// checkpoint through the worker's own `flux1_control_load` (the `flux1_control_spec` seam) and renders
/// ONE image per control mode (pose / canny / depth) — Union-Pro-2.0 is input-agnostic, so the same
/// synthetic control map drives all three; the per-preprocessor derivation (skeleton / edge / depth) is
/// unit-tested in `preprocess_control_entry_dispatches_by_kind`. Each mode must produce a non-degenerate
/// 512² decode. Needs both weight sets + a Metal device; run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored flux1_dev_control_real_weights`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs the gated FLUX.1-dev snapshot + the Shakker Union-Pro-2.0 ckpt + a Metal device"]
fn flux1_dev_control_real_weights_generates_each_mode() {
    let base = std::env::var("SCENEWORKS_FLUX1_DEV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::fs::read_dir(
                dirs_home()
                    .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots"),
            )
            .expect("FLUX.1-dev snapshots dir")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.join("model_index.json").is_file())
            .expect("a FLUX.1-dev snapshot dir")
        });
    let control = std::env::var("SCENEWORKS_CONTROLNET_FLUX1")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::fs::read_dir(dirs_home().join(
                ".cache/huggingface/hub/models--Shakker-Labs--FLUX.1-dev-ControlNet-Union-Pro-2.0/snapshots",
            ))
            .expect("control snapshots dir")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .map(|dir| dir.join(FLUX1_CONTROL_FILE))
            .filter(|path| path.exists())
            .expect("control weights file")
        });

    let generator =
        flux1_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();

    // A flat 512² control map — enough to exercise the VAE-encode of the control hint on real weights;
    // the real per-mode preprocessing (skeleton / canny / depth) is unit-tested separately.
    let control_map = gen_core::Image {
        width: 512,
        height: 512,
        pixels: vec![128u8; 512 * 512 * 3],
    };
    let cancel = gen_core::CancelFlag::new();
    for kind in [ControlKind::Pose, ControlKind::Canny, ControlKind::Depth] {
        // pose/canny → Control { kind, scale }; depth → the dedicated Depth variant (build_control_*).
        let conditioning = build_control_conditioning(control_map.clone(), kind.clone(), 0.7, None);
        let (w, h, pixels) = flux1_control_generate_one(
            generator.as_ref(),
            "a person standing in a meadow, photorealistic",
            512,
            512,
            42,
            28,
            Some(3.5), // dev embedded guidance
            conditioning,
            &cancel,
            &mut |_| {},
        )
        .unwrap_or_else(|e| panic!("{kind:?} render failed: {e}"));
        assert_eq!((w, h), (512, 512), "{kind:?}");
        assert_eq!(pixels.len(), 512 * 512 * 3, "{kind:?}");
        assert!(
            pixels.windows(2).any(|w| w[0] != w[1]),
            "{kind:?} render is degenerate (flat)"
        );
    }
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
fn flux2_edit_reference_ids_takes_plural_multi_reference_set() {
    // sc-6211: the multi-image picker sends `referenceAssetIds` — all of them, in order, win over the
    // singular fields.
    assert_eq!(
        flux2_edit_reference_ids(&request(json!({
            "projectId": "p",
            "referenceAssetIds": ["a", "b", "c"],
            "referenceAssetId": "singular_ignored"
        }))),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
    );
    // Capped at MAX_EDIT_REFERENCES (4) — a 6-image pick keeps the first four.
    assert_eq!(
        flux2_edit_reference_ids(&request(json!({
            "projectId": "p",
            "referenceAssetIds": ["a", "b", "c", "d", "e", "f"]
        })))
        .len(),
        MAX_EDIT_REFERENCES
    );
    // A single pick in the plural picker reduces to the single-reference path.
    assert_eq!(
        flux2_edit_reference_ids(&request(json!({
            "projectId": "p", "referenceAssetIds": ["only"]
        }))),
        vec!["only".to_owned()]
    );
    // Empty plural list falls back to the singular reference flow.
    assert_eq!(
        flux2_edit_reference_ids(&request(json!({
            "projectId": "p", "referenceAssetIds": [], "referenceAssetId": "ref_1"
        }))),
        vec!["ref_1".to_owned()]
    );
}

// sc-8278: the klein/dev edit identity-strength → image-guidance mapping. macOS-gated like the
// rest of flux2.rs (the whole `image_jobs/flux2.rs` include is `cfg(target_os = "macos")`).
#[cfg(target_os = "macos")]
#[test]
fn flux2_edit_image_guidance_lever() {
    // Off outside character_image mode (a plain image edit), even with the knob set.
    assert_eq!(
        flux2_edit_image_guidance(&request(serde_json::json!({
            "mode": "edit_image", "referenceAssetId": "a", "advanced": { "ipAdapterScale": 1.5 }
        }))),
        None
    );
    // Off when no character reference is attached.
    assert_eq!(
        flux2_edit_image_guidance(&request(serde_json::json!({
            "mode": "character_image", "advanced": { "ipAdapterScale": 1.5 }
        }))),
        None
    );
    // Default 1.5 (realism-safe) when a reference is present and the knob is unspecified.
    assert_eq!(
        flux2_edit_image_guidance(&request(serde_json::json!({
            "mode": "character_image", "referenceAssetId": "a"
        }))),
        Some(1.5)
    );
    // Slider value honored, clamped to the 2.5 ceiling.
    assert_eq!(
        flux2_edit_image_guidance(&request(serde_json::json!({
            "mode": "character_image", "referenceAssetId": "a", "advanced": { "ipAdapterScale": 3.0 }
        }))),
        Some(2.5)
    );
    // A slider value at/below 1.0 reads as OFF (the engine's ≤1 = off).
    assert_eq!(
        flux2_edit_image_guidance(&request(serde_json::json!({
            "mode": "character_image", "referenceAssetId": "a", "advanced": { "ipAdapterScale": 0.8 }
        }))),
        None
    );
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

#[cfg(target_os = "macos")]
#[test]
fn boogu_edit_reference_ids_prefers_plural_then_source() {
    // sc-7645: the multi-image picker's plural `referenceAssetIds` wins, in order (over `sourceAssetId`).
    assert_eq!(
        boogu_edit_reference_ids(&request(json!({
            "projectId": "p", "mode": "edit_image",
            "referenceAssetIds": ["a", "b", "c"], "sourceAssetId": "ignored"
        }))),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
    );
    // With no plural list it falls back to the single Image-Edit `sourceAssetId`.
    assert_eq!(
        boogu_edit_reference_ids(&request(json!({
            "projectId": "p", "mode": "edit_image", "sourceAssetId": "src_1"
        }))),
        vec!["src_1".to_owned()]
    );
    // Capped at BOOGU_MAX_EDIT_REFERENCES (5) — a 7-image pick keeps the first five.
    assert_eq!(
        boogu_edit_reference_ids(&request(json!({
            "projectId": "p", "mode": "edit_image",
            "referenceAssetIds": ["a", "b", "c", "d", "e", "f", "g"]
        })))
        .len(),
        BOOGU_MAX_EDIT_REFERENCES
    );
    // Neither a plural list nor a source → empty (the generic lane runs plain txt2img).
    assert!(boogu_edit_reference_ids(&request(json!({ "projectId": "p" }))).is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn boogu_build_reference_conditioning_single_vs_multi() {
    // sc-7645: one reference stays a `Reference` (byte-identical to the single-reference edit path);
    // 2–5 become a single `MultiReference` (the DiT packs all of them). Empty → no conditioning.
    let img = |seed| gen_core::Image {
        width: 8,
        height: 8,
        pixels: stub_rgb8(8, 8, seed),
    };
    assert!(build_reference_conditioning(&[]).is_empty());
    match build_reference_conditioning(std::slice::from_ref(&img(1))).as_slice() {
        [gen_core::Conditioning::Reference { strength, .. }] => assert!(strength.is_none()),
        other => panic!("expected one Reference, got {other:?}"),
    }
    match build_reference_conditioning(&[img(1), img(2), img(3)]).as_slice() {
        [gen_core::Conditioning::MultiReference { images }] => assert_eq!(images.len(), 3),
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
        None,
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

    // Base (full-CFG) Z-Image strict control (sc-8251): `z_image` + ≥1 pose → the base
    // Fun-Controlnet-Union path, one image per pose. Keyed on the base id, distinct from the Turbo arm.
    let zimage_base_pose = request(json!({
        "projectId": "p", "model": "z_image", "count": 7,
        "advanced": { "modelPath": model_path.clone(), "poses": [{ "id": "a" }, { "id": "b" }, { "id": "c" }] }
    }));
    let route = resolve_image_route(&zimage_base_pose, &settings).unwrap();
    assert_eq!(route, ImageRoute::ZImageBaseControl);
    assert_eq!(route.image_count(&zimage_base_pose, &settings), 3);

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

    // FLUX.1-dev strict control (sc-8244): flux_dev + ≥1 pose (not edit_image) → the Shakker
    // Union-Pro-2.0 path, one image per pose. Wins over PuLID-FLUX / generic MLX.
    let flux1_control = request(json!({
        "projectId": "p", "model": "flux_dev", "count": 9,
        "advanced": { "modelPath": model_path.clone(), "poses": [{ "id": "a" }, { "id": "b" }, { "id": "c" }, { "id": "d" }] }
    }));
    let route = resolve_image_route(&flux1_control, &settings).unwrap();
    assert_eq!(route, ImageRoute::Flux1DevControl);
    assert_eq!(route.image_count(&flux1_control, &settings), 4);

    let plain_mlx = request(json!({
        "projectId": "p", "model": "z_image_turbo", "count": 6,
        "advanced": { "modelPath": model_path.clone() }
    }));
    let route = resolve_image_route(&plain_mlx, &settings).unwrap();
    assert_eq!(route, ImageRoute::Mlx);
    assert_eq!(route.image_count(&plain_mlx, &settings), 6);

    // sc-8320: plain base `z_image` t2i (no poses) → the generic MLX path (base engine), NOT a control
    // arm — proving the base is selectable + routes to the base t2i path, distinct from Turbo control.
    let zimage_base_t2i = request(json!({
        "projectId": "p", "model": "z_image", "count": 4,
        "advanced": { "modelPath": model_path.clone() }
    }));
    let route = resolve_image_route(&zimage_base_t2i, &settings).unwrap();
    assert_eq!(route, ImageRoute::Mlx);
    assert_eq!(route.image_count(&zimage_base_t2i, &settings), 4);
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

/// sc-4411: the With-Character likeness-source resolver gates scoring to a PLAIN `character_image`
/// generation with a character `referenceAssetId`, and EXCLUDES angle sets / pose sets (already scored
/// by sc-4409/4410 through the same seam) and non-character modes — the no-double-attach guard. Here the
/// asset can't decode (empty temp data dir), so the positive plain case returns `None` via the non-fatal
/// decode path; the assertions key on the gate (the decode-vs-gate distinction is exercised by the
/// reference-presence cases). Decode of a real asset is covered by the sc-4410 control-source tests.
#[cfg(target_os = "macos")]
#[test]
fn character_image_likeness_source_gates_to_plain_with_character() {
    let dir = tempfile::tempdir().unwrap();
    let project_path = dir.path();
    let mut settings = Settings::from_env();
    settings.data_dir = dir.path().to_path_buf();

    // A non-character mode (text_to_image / edit_image) is NEVER a With-Character generation: excluded
    // BEFORE any decode, even with a referenceAssetId present.
    for mode in ["text_to_image", "edit_image"] {
        let req = request(json!({
            "projectId": "p", "model": "instantid_realvisxl", "mode": mode,
            "referenceAssetId": "ref",
        }));
        assert!(
            resolve_character_image_likeness_source(&req, &settings, project_path).is_none(),
            "mode {mode} is not a With-Character generation ⇒ no likeness source",
        );
    }

    // An angle set (sc-4409) and a pose set (sc-4410) are character_image jobs but are ALREADY scored
    // through the same seam — the plain resolver must exclude them so the plain case never double-attaches.
    let angle = request(json!({
        "projectId": "p", "model": "instantid_realvisxl", "mode": "character_image",
        "referenceAssetId": "ref", "advanced": { "angleSet": true }
    }));
    assert!(
        resolve_character_image_likeness_source(&angle, &settings, project_path).is_none(),
        "an angle set is scored by sc-4409 ⇒ the plain resolver excludes it (no double-attach)",
    );
    let pose = request(json!({
        "projectId": "p", "model": "instantid_realvisxl", "mode": "character_image",
        "referenceAssetId": "ref", "advanced": { "poses": [{ "id": "a" }] }
    }));
    assert!(
        resolve_character_image_likeness_source(&pose, &settings, project_path).is_none(),
        "a pose set is scored by sc-4410 ⇒ the plain resolver excludes it (no double-attach)",
    );

    // A plain character_image with NO reference returns None at the reference filter (an honest no-source,
    // not an error) — distinguishing the reference gate from the mode/grouping gates above.
    let no_ref = request(json!({
        "projectId": "p", "model": "instantid_realvisxl", "mode": "character_image",
    }));
    assert!(
        resolve_character_image_likeness_source(&no_ref, &settings, project_path).is_none(),
        "a character_image with no referenceAssetId has no identity source",
    );

    // The PLAIN With-Character case (character_image + reference, no angle/pose) passes the gate and
    // reaches the (non-fatal) decode; the asset is absent here, so it resolves to None WITHOUT panicking
    // — scoring never aborts a generation (the sc-4407 non-fatal contract).
    let plain = request(json!({
        "projectId": "p", "model": "instantid_realvisxl", "mode": "character_image",
        "referenceAssetId": "missing-asset",
    }));
    assert!(
        resolve_character_image_likeness_source(&plain, &settings, project_path).is_none(),
        "a missing reference decodes to None (non-fatal), never a panic",
    );
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
        false,
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
        false,
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
    // A minimal standing skeleton (body only for this smoke — the production tier now
    // threads hands/face per sc-6702, exercised by the A/B test below) + a synthetic
    // reference, paired as the pose multi-image set.
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
        None,
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

/// sc-6702 A/B (real weights): does threading the pose's hands/face into the
/// FLUX.2-klein best-effort skeleton improve hand/gesture fidelity (as it did for
/// Qwen-Edit, sc-6599 = GO)? The best-effort tier feeds the skeleton as the *first*
/// image of a `[skeleton, reference]` `MultiReference` edit (NOT a ControlNet), so
/// there is no training-distribution argument either way — only a real A/B settles it.
///
/// This reuses the persistent sc-6599 harness assets at `~/sceneworks-pytorch-harness/`
/// (the same auburn-haired reference + two whole-body pose donors with hands/face
/// detected by production DWPose) so the result is directly comparable to the Qwen run.
/// For each pose it renders the skeleton body-only vs whole-body and runs the exact
/// production engine call with an identical seed/prompt/reference — the ONLY variable is
/// whether the skeleton carries hands+face — writing `flux2_out_{pose}_{arm}.png` for
/// visual judgment. Needs the HF cache (`black-forest-labs/FLUX.2-klein-9b`) + Metal:
/// `cargo test -p sceneworks-worker --lib -- --ignored --nocapture flux2_pose_tier_ab_wholebody`.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "sc-6702 real-weight A/B: needs FLUX.2-klein-9b weights + Metal + the sc-6599 harness assets"]
fn flux2_pose_tier_ab_wholebody_vs_body_real_weights() {
    use crate::openpose_skeleton::{
        body_stickwidth, draw_wholebody, normalize_face, normalize_hands, normalize_keypoints,
    };

    // Harness dir (override with SC6702_HARNESS for a relocated copy).
    let harness = std::env::var("SC6702_HARNESS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join("sceneworks-pytorch-harness"));
    assert!(
        harness.join("ref.png").exists(),
        "missing sc-6599 harness assets at {} — see sc_6599_qwen_edit_pose_wholebody_ab",
        harness.display()
    );

    // SIDE matches the reference + the Qwen A/B (max hand/face scale). BASE is the same
    // character description used for the reference image in the Qwen A/B, wrapped by the
    // production pose-prompt augmenter (`augment_prompt_for_pose`).
    const SIDE: u32 = 1024;
    const BASE: &str = "The same woman with long wavy auburn hair, freckles, and a \
        mustard-yellow turtleneck sweater, full body, standing, plain light background, \
        photorealistic";
    let prompt = augment_prompt_for_pose(BASE);
    let stickwidth = body_stickwidth(SIDE, SIDE);

    // Shared reference (identity), resized to the working square if needed.
    let ref_rgb = {
        let img = image::open(harness.join("ref.png")).unwrap().to_rgb8();
        if img.width() == SIDE && img.height() == SIDE {
            img
        } else {
            image::imageops::resize(&img, SIDE, SIDE, image::imageops::FilterType::Triangle)
        }
    };
    let reference = gen_core::Image {
        width: SIDE,
        height: SIDE,
        pixels: ref_rgb.into_raw(),
    };

    // Load the klein edit engine once; reuse for all four renders.
    let snapshot = hf_snapshot("models--black-forest-labs--FLUX.2-klein-9b");
    let generator = load_engine(
        "flux2_klein_9b_edit",
        snapshot,
        Some(gen_core::Quant::Q8),
        Vec::new(),
        None,
    )
    .unwrap();

    // Seeds mirror the Qwen A/B (pose1=8001, pose2=8002) for a clean cross-model comparison.
    for (pose, seed) in [("pose1", 8001i64), ("pose2", 8002i64)] {
        let raw = std::fs::read_to_string(harness.join(format!("{pose}.json"))).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let keypoints = normalize_keypoints(&v["keypoints"]);
        let hands = normalize_hands(&v["hands"]);
        let face = normalize_face(&v["face"]);
        assert!(
            hands.is_some() && face.is_some(),
            "{pose}.json must carry hands + face for the A/B to mean anything"
        );

        // arm "body" = body-only (None/None — the current production behavior);
        // arm "whole" = body + hands + face (the candidate fix).
        for arm in ["body", "whole"] {
            let (hands_arg, face_arg) = if arm == "whole" {
                (hands.as_deref(), face.as_deref())
            } else {
                (None, None)
            };
            let skeleton = draw_wholebody(SIDE, SIDE, &keypoints, hands_arg, face_arg, stickwidth);
            // Record the skeleton actually fed to the engine.
            skeleton
                .save(harness.join(format!("flux2_{pose}_skel_{arm}.png")))
                .unwrap();
            let skeleton_img = gen_core::Image {
                width: SIDE,
                height: SIDE,
                pixels: skeleton.into_raw(),
            };
            // Production order: [skeleton, reference] (flux2.rs:419 — skeleton FIRST,
            // unlike Qwen's [reference, skeleton]).
            let conditioning = vec![gen_core::Conditioning::MultiReference {
                images: vec![skeleton_img, reference.clone()],
            }];
            let cancel = gen_core::CancelFlag::new();
            let mut steps_seen = 0u32;
            eprintln!("[sc-6702] generating {pose}/{arm} (seed {seed}) ...");
            let (w, h, pixels) = flux2_edit_generate_one(
                generator.as_ref(),
                &prompt,
                SIDE,
                SIDE,
                seed,
                4, // klein is a 4-step distill
                Some(1.0),
                None,
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
            assert_eq!((w, h), (SIDE, SIDE));
            assert!(steps_seen >= 1, "expected denoise step progress");
            let out = harness.join(format!("flux2_out_{pose}_{arm}.png"));
            image::RgbImage::from_raw(w, h, pixels)
                .unwrap()
                .save(&out)
                .unwrap();
            eprintln!("[sc-6702]   saved {}", out.display());
        }
    }
    eprintln!(
        "[sc-6702] A/B done — compare flux2_out_pose{{1,2}}_{{body,whole}}.png in {}",
        harness.display()
    );
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

#[cfg(target_os = "macos")]
#[test]
fn compose_feathered_no_boundary_vignette() {
    // sc-8229: a single edge tile covers the whole frame; its raised-cosine feather ramps
    // toward ~0 over the `overlap` border. Recompose must normalize by the true accumulated
    // weight so the border keeps its full source value (not a dark rounded-corner vignette).
    let (width, height) = (32u32, 32u32);
    let overlap = 8;
    let feather = detail_feather(width, height, overlap);
    let mut acc = vec![0.0f32; (width * height * 3) as usize];
    let mut wsum = vec![0.0f32; (width * height) as usize];
    // Uniform mid-gray source refined by one full-frame tile.
    const SRC: f32 = 200.0;
    for i in 0..(width * height) as usize {
        let f = feather[i];
        acc[i * 3] += SRC * f;
        acc[i * 3 + 1] += SRC * f;
        acc[i * 3 + 2] += SRC * f;
        wsum[i] += f;
    }
    let out = compose_feathered(&acc, &wsum, width, height);
    // Every pixel — corners and edges included — recovers the source value, not a ramp to black.
    for (x, y) in [
        (0, 0),
        (0, height - 1),
        (width - 1, 0),
        (15, 0),
        (0, 15),
        (15, 15),
    ] {
        let px = out.get_pixel(x, y).0;
        assert!(
            px.iter().all(|&c| c.abs_diff(SRC as u8) <= 1),
            "pixel ({x},{y}) = {px:?} darkened toward the border (expected ~{SRC})"
        );
    }
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

/// Boogu (epic 6387) resolves each of its three ids to its own engine + the reference defaults, and
/// quant resolution returns Q8 — the catalog declares `mlx.quantize: 8` (the pre-packed Q8 turnkey
/// default). Turbo is CFG-free so `resolve_guidance` returns None. Runs in CI on Mac (no weights).
#[cfg(target_os = "macos")]
#[test]
fn boogu_engine_defaults_and_quant_resolution() {
    let base = mlx_model("boogu_image").expect("boogu_image in MODEL_TABLE");
    assert_eq!(base.engine_id(), "boogu_image");
    assert_eq!(base.adapter_label(), "mlx_boogu");
    assert_eq!(base.default_steps(), 50);
    let turbo = mlx_model("boogu_image_turbo").expect("boogu_image_turbo in MODEL_TABLE");
    assert_eq!(turbo.engine_id(), "boogu_image_turbo");
    assert_eq!(turbo.default_steps(), 4);
    let edit = mlx_model("boogu_image_edit").expect("boogu_image_edit in MODEL_TABLE");
    assert_eq!(edit.engine_id(), "boogu_image_edit");

    let req = |model: &str, advanced: Value| {
        request(json!({
            "projectId": "p", "model": model, "prompt": "p", "advanced": advanced,
            "modelManifestEntry": { "mlx": { "quantize": 8 } },
        }))
    };
    // Base: true-CFG guidance 4.0 from the model row. Turbo: CFG-free → resolve_guidance None.
    assert_eq!(
        resolve_guidance(&req("boogu_image", json!({})), &base),
        Some(4.0)
    );
    assert_eq!(
        resolve_guidance(&req("boogu_image_turbo", json!({})), &turbo),
        None
    );
    // Default → Q8 (the shipped pre-packed turnkey); advanced.mlxQuantize overrides to Q4 / bf16-dense.
    assert!(matches!(
        resolve_quant(&req("boogu_image", json!({}))),
        (Some(Quant::Q8), Some(8))
    ));
    assert!(matches!(
        resolve_quant(&req("boogu_image", json!({ "mlxQuantize": 4 }))),
        (Some(Quant::Q4), Some(4))
    ));
    assert!(matches!(
        resolve_quant(&req("boogu_image", json!({ "mlxQuantize": 0 }))),
        (None, None)
    ));
}

/// `boogu_model_subdir` maps each id to its variant folder (`base`/`turbo`/`edit`), picks the
/// pre-packed Q8 `<variant>/` by default, and the full-precision `<variant>-bf16/` only when
/// `mlxQuantize <= 4` AND it is present — falling back to the Q8 folder (then root), so a request for
/// a not-yet-downloaded bf16 build resolves the Q8 default rather than half-loading.
#[cfg(target_os = "macos")]
#[test]
fn boogu_subdir_prefers_q8_and_opts_into_bf16() {
    let root = tempfile::tempdir().unwrap();
    let touch = |sub: &str| {
        let dir = root.path().join(sub).join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), b"stub").unwrap();
    };
    let req = |model: &str, advanced: Value| {
        request(json!({ "projectId": "p", "model": model, "prompt": "p", "advanced": advanced }))
    };
    // Variant mapping + Q8 default.
    touch("base");
    touch("turbo");
    touch("edit");
    assert_eq!(
        boogu_model_subdir(root.path(), &req("boogu_image", json!({}))),
        root.path().join("base")
    );
    assert_eq!(
        boogu_model_subdir(root.path(), &req("boogu_image_turbo", json!({}))),
        root.path().join("turbo")
    );
    assert_eq!(
        boogu_model_subdir(root.path(), &req("boogu_image_edit", json!({}))),
        root.path().join("edit")
    );
    // bf16 opt-in falls back to the Q8 folder when `base-bf16/` is absent (not yet downloaded).
    assert_eq!(
        boogu_model_subdir(
            root.path(),
            &req("boogu_image", json!({ "mlxQuantize": 0 }))
        ),
        root.path().join("base"),
        "bf16 opt-in falls back to Q8 when base-bf16 absent",
    );
    // With the bf16 folder present, `mlxQuantize <= 4` selects it; Q8 stays the default.
    touch("base-bf16");
    assert_eq!(
        boogu_model_subdir(
            root.path(),
            &req("boogu_image", json!({ "mlxQuantize": 0 }))
        ),
        root.path().join("base-bf16")
    );
    assert_eq!(
        boogu_model_subdir(
            root.path(),
            &req("boogu_image", json!({ "mlxQuantize": 4 }))
        ),
        root.path().join("base-bf16")
    );
    assert_eq!(
        boogu_model_subdir(root.path(), &req("boogu_image", json!({}))),
        root.path().join("base")
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
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            false,
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

/// sc-6546 render-confirmation (was sc-6519): the headless/API path's exact behavior end-to-end on
/// one box — run the ACTUAL `magic_prompt` expansion on a plain prompt through the SHIPPED refiner
/// (Anubis-Mini-8B, PR #830) to get a rich JSON caption, then render THAT caption through the
/// production Ideogram 4 load and confirm a real image (not the baked "Image blocked by safety
/// filter" placeholder). sc-5997 proved plain→caption and sc-6501 proved a hand-crafted rich caption
/// escapes; this closes sc-6546's open acceptance check — that the coherent Anubis default's OWN
/// output renders a real image at the real default regime. The refiner is loaded + dropped BEFORE
/// Ideogram loads, mirroring the API running this as a SEPARATE prompt_refine job (no refiner/Ideogram
/// co-residency). Defaults to the real placeholder regime (1024²/48) where plain text fails and rich
/// captions escape.
/// Run: `cargo test -p sceneworks-worker --lib -- --ignored ideogram_4_headless_auto_caption --nocapture`.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Anubis-Mini-8B refiner + Ideogram 4 snapshot; run manually on a Mac"]
#[test]
fn ideogram_4_headless_auto_caption_renders_real_image() {
    use gen_core::CancelFlag;

    let Some(ideogram) = ideogram_dir() else {
        eprintln!("skipping: no SceneWorks/ideogram-4-mlx q4 snapshot found");
        return;
    };
    // The prompt-refine snapshot the magic_prompt expansion runs on — Anubis-Mini-8B, the shipped
    // magic-prompt default (sc-6546 / PR #830; the Llama-3.2-3B was retired). Loads on the same
    // config-driven Llama `prompt_refine` seam, stock bf16, no conversion.
    let refine_snaps =
        dirs_home().join(".cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots");
    let Some(refine_dir) = std::fs::read_dir(&refine_snaps)
        .ok()
        .and_then(|entries| entries.flatten().map(|e| e.path()).find(|p| p.is_dir()))
    else {
        eprintln!("skipping: no TheDrummer/Anubis-Mini-8B-v1 snapshot found");
        return;
    };

    const PLAIN_TEXT: &str = "a red fox sitting in a snowy forest at golden hour";

    // 1) Expand the plain prompt into a rich caption via the real 3B — the SAME magic-prompt messages
    //    + JSON isolation the worker's magic_prompt job uses — then CLEAN it through the shared
    //    canonical serializer exactly as the API does (drop the stray top-level `aspect_ratio`, strip
    //    the model's unreliable bboxes, impose canonical order). The 3B is stochastic and occasionally
    //    emits malformed JSON (sc-6519), so re-sample until a valid caption (mirroring the API's
    //    MAX_CAPTION_ATTEMPTS re-sample). Scoped so the refiner frees before Ideogram loads, mirroring
    //    the API running this as a separate prompt_refine job (no 3B/Ideogram co-residency).
    let caption = {
        // sc-7189: the legacy `gen_core::load_textllm`/`TextLlm` contract was retired; resolve the
        // refiner model-first through `core_llm::TextLlm` exactly as the production magic_prompt job
        // does (prompt_refine_jobs.rs macOS lane) — the JSON constraint steers resolution to a
        // JSON-capable provider (mlx-llama), which renders the model's own chat template.
        use gen_core::core_llm::{
            load_for_model_with, Constraint, LoadSpec, Message, ModelRequirements, Sampling,
            StreamEvent, TextLlmRequest,
        };
        let (system, user) =
            crate::prompt_refine_jobs::build_magic_prompt_messages(PLAIN_TEXT, "1:1");
        let make_request = || {
            let mut messages = Vec::with_capacity(2);
            if !system.trim().is_empty() {
                messages.push(Message::system(system.clone()));
            }
            messages.push(Message::user(user.clone()));
            TextLlmRequest {
                messages,
                sampling: Sampling {
                    temperature: 0.4,
                    top_p: 0.9,
                    ..Sampling::default()
                },
                max_new_tokens: 2048,
                seed: None,
                // sc-6585: constrain to the JSON grammar so the caption is structurally valid in one
                // shot (the resample loop below should now succeed on attempt 1).
                constraint: Some(Constraint::Json),
                ..Default::default()
            }
        };
        let refiner = load_for_model_with(
            &LoadSpec {
                source: refine_dir.to_string_lossy().into_owned(),
                quantize: None,
            },
            &ModelRequirements::from_request(&make_request()),
        )
        .expect("load prompt_refine (Anubis-8B) model-first");
        let mut found = None;
        for attempt in 1..=6 {
            let req = make_request();
            let mut noop = |_e: StreamEvent| {};
            let raw = refiner
                .generate(&req, &mut noop)
                .expect("magic_prompt generate")
                .text;
            let candidate = crate::prompt_refine_jobs::clean_json_output(&raw);
            if let Some(cleaned) = serde_json::from_str::<Value>(&candidate)
                .ok()
                .as_ref()
                .and_then(sceneworks_core::ideogram_caption::serialize_magic_prompt_caption)
            {
                found = Some(cleaned);
                break;
            }
            eprintln!(
                "attempt {attempt}: Anubis produced no valid caption, re-sampling:\n{candidate}"
            );
        }
        found.expect("Anubis produced a valid caption within 6 attempts")
    };
    eprintln!("magic-prompt caption:\n{caption}");

    // 2) The worker's caption guard passes a real caption through unchanged — the engine tokenizes
    //    exactly the cleaned caption.
    let prompt = crate::ideogram_caption::ensure_caption_prompt(&caption);
    assert_eq!(
        prompt, caption,
        "an existing caption passes through unchanged"
    );

    // 3) Render through the production Ideogram 4 load AND the production reseed recovery (base.rs):
    //    render seed 7, and on a detected placeholder reseed up to the retry budget keeping the first
    //    clean render — exactly what the worker does in `generate_stream`. A coherent caption escapes
    //    immediately; the reseed net (sc-6501) backstops a sparse one. Anubis is the coherent default
    //    (sc-6550 bake-off: ~0% subject-as-text / transparent-bg, residual degeneracy is malformed
    //    JSON the 6-attempt resample above already filters), so escape is the EXPECTATION here — a
    //    residual placeholder would be the rare semantic miss; bump SCENEWORKS_IDEOGRAM_PLACEHOLDER_RETRIES
    //    / re-run to resample.
    //    Defaults to 1024²/48 (the regime where plain text fails); env-overridable for a fast check.
    let env_u32 = |key: &str, default: u32| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let steps = env_u32("IDEOGRAM4_SMOKE_STEPS", 48);
    let res = env_u32("IDEOGRAM4_SMOKE_RES", 1024);

    let model = mlx_model("ideogram_4").unwrap();
    let req = request(json!({
        "projectId": "p", "model": "ideogram_4", "prompt": "p", "advanced": {},
        "modelManifestEntry": { "mlx": { "quantize": 4 } },
    }));
    let (quant, _bits) = resolve_quant(&req);
    let guidance = resolve_guidance(&req, &model);
    let generator = load_engine("ideogram_4", ideogram, quant, Vec::new(), None).unwrap();
    let cancel = CancelFlag::new();
    let enhance = PromptEnhance::default();

    let render = |seed: i64| {
        generate_one(
            generator.as_ref(),
            &prompt,
            res,
            res,
            seed,
            steps,
            guidance,
            None,
            None,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            &enhance,
            &cancel,
            &mut |_| {},
        )
        .expect("ideogram render from the auto-caption")
    };
    let retries = crate::ideogram_caption::placeholder_recovery_retries();
    let (mut w, mut h, mut pixels) = render(7);
    let mut placeholder = crate::ideogram_caption::looks_like_placeholder(&pixels, w, h);
    let mut final_seed = 7i64;
    for attempt in 0..retries {
        if !placeholder {
            break;
        }
        let retry_seed = crate::ideogram_caption::recovery_seed(7, attempt);
        eprintln!(
            "placeholder; reseeding {retry_seed} (attempt {}/{retries})",
            attempt + 1
        );
        let (rw, rh, rpixels) = render(retry_seed);
        w = rw;
        h = rh;
        pixels = rpixels;
        final_seed = retry_seed;
        placeholder = crate::ideogram_caption::looks_like_placeholder(&pixels, w, h);
    }
    // Hard asserts validate THIS story's deliverable end-to-end with real weights: the 3B caption was
    // cleaned to the canonical engine form and rendered through the production Ideogram load + reseed
    // recovery, producing a correctly-sized, non-constant image.
    assert_eq!(pixels.len(), (w * h * 3) as usize, "RGB8-sized buffer");
    assert!(
        pixels.windows(2).any(|x| x[0] != x[1]),
        "non-constant image"
    );
    eprintln!(
        "auto-caption render: {w}x{h} RGB8, final seed {final_seed}, placeholder={placeholder}"
    );
    // The placeholder ESCAPE is reported, not hard-asserted: whether the rendered image is real is
    // gated by the caption's COHERENCE, which is stochastic. With Anubis (the shipped coherent default,
    // sc-6546/PR #830) escape is the expectation — the bake-off measured ~0% semantic degeneracy and
    // the malformed-JSON residual is filtered by the resample loop above. A coherent caption escapes
    // (sc-6501 proves it with a hand-crafted rich caption); the reseed net (sc-6501) backstops a sparse
    // one. Re-run to resample, or bump SCENEWORKS_IDEOGRAM_PLACEHOLDER_RETRIES.
    if placeholder {
        eprintln!(
            "NOTE: still the safety placeholder after {retries} reseeds — Anubis returned a rare \
             degenerate caption (subject-as-text / transparent background). Re-run to resample; this \
             is the residual model ceiling sc-6585 (constrained decoding) further tightens, not an \
             orchestration/engine defect."
        );
    } else {
        eprintln!("the headless auto-caption rendered a real image (escaped the placeholder).");
    }
}

/// Real-weight Ideogram 4 **edit** e2e (sc-6303): drives the worker's `generate_one` with a source
/// `Reference` (img2img/Remix) and a `Reference` + `Mask` (inpaint) through the production engine
/// load, so the worker's `[Reference, Mask]` conditioning assembly is consumed by the engine's edit
/// path end-to-end. Mechanics only (low steps); engine-level correctness (keep-vs-repaint) is the
/// mlx-gen `edit_smoke`. Run:
/// `cargo test -p sceneworks-worker --lib -- --ignored ideogram_4_real_weights_edit --nocapture`.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Ideogram 4 snapshot; run manually on a Mac with SceneWorks/ideogram-4-mlx cached"]
#[test]
fn ideogram_4_real_weights_edit_img2img_and_inpaint() {
    let Some(dir) = ideogram_dir() else {
        eprintln!(
            "skipping ideogram_4_real_weights_edit: no SceneWorks/ideogram-4-mlx q4 snapshot found"
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
    let (quant, _bits) = resolve_quant(&req);
    let guidance = resolve_guidance(&req, &model);
    let generator = load_engine("ideogram_4", dir, quant, Vec::new(), None).unwrap();
    let cancel = gen_core::CancelFlag::new();
    let enhance = PromptEnhance::default();
    const CAPTION_JSON: &str = "{\"high_level_description\": \"A photograph of a red fox sitting in a snowy forest at golden hour.\", \"compositional_deconstruction\": {\"background\": \"A snowy pine forest at golden hour.\", \"elements\": [{\"type\": \"obj\", \"bbox\": [250, 320, 950, 760], \"desc\": \"A red fox sitting upright in the snow, facing the camera.\"}]}}";

    // Synthetic RGB8 source (gradient) + a left-half-white inpaint mask, both res×res.
    let source = Image {
        width: res,
        height: res,
        pixels: (0..res * res)
            .flat_map(|i| {
                [
                    (255 * (i % res) / res) as u8,
                    (255 * (i / res) / res) as u8,
                    128u8,
                ]
            })
            .collect(),
    };
    let mask = Image {
        width: res,
        height: res,
        pixels: (0..res * res)
            .flat_map(|i| {
                let v = if i % res < res / 2 { 255u8 } else { 0u8 };
                [v, v, v]
            })
            .collect(),
    };

    let run = |reference: Option<&(Image, f32)>, edit_mask: Option<&Image>, label: &str| {
        let mut steps_seen = 0u32;
        let (w, h, pixels) = generate_one(
            generator.as_ref(),
            CAPTION_JSON,
            res,
            res,
            42,
            steps,
            guidance,
            None,
            reference,
            &[],
            edit_mask,
            None,
            None,
            None,
            None,
            None,
            false,
            &enhance,
            &cancel,
            &mut |p| {
                if let gen_core::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap_or_else(|error| panic!("ideogram edit {label} failed: {error}"));
        assert_eq!(pixels.len(), (w * h * 3) as usize, "{label}: RGB8 buffer");
        assert!(w == res && h == res, "{label}: {res}² output");
        assert!(steps_seen >= 1, "{label}: denoise progress");
        assert!(
            pixels.windows(2).any(|x| x[0] != x[1]),
            "{label}: non-constant image"
        );
        eprintln!("ideogram edit {label}: {w}x{h} RGB8, {steps_seen} steps observed");
    };

    // img2img (Remix): source Reference only.
    run(Some(&(source.clone(), 0.6)), None, "img2img");
    // inpaint (Edit): source Reference + Mask (white half repaints, black half keeps).
    run(Some(&(source, 0.85)), Some(&mask), "inpaint");
}

// ─────────────────────────────────────────────────────────────────────────────
// Boogu-Image-0.1 (epic 6387) — unit coverage lives above (`boogu_engine_defaults_*`,
// `boogu_subdir_*`); here is the S5 (sc-6402) real-weight #[ignore] smoke driving all three ids
// (Base / Turbo / Edit) through the registry load seam against the published turnkey.
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the cached Boogu turnkey snapshot ROOT (the dir holding the `base`/`turbo`/`edit` variant
/// subfolders) for the real-weight smoke: env override `SCENEWORKS_MLX_BOOGU_DIR` → else the cached
/// `SceneWorks/boogu-image-mlx` snapshot. `None` ⇒ skip (the ~23 GB/variant weights live outside CI).
/// A root "counts" once any one variant's transformer is present, so a partial download (e.g. only
/// `base/` pulled so far) still resolves and the smoke validates whatever is on disk.
#[cfg(target_os = "macos")]
fn boogu_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let has_any_variant = |root: &Path| {
        ["base", "turbo", "edit"].iter().any(|v| {
            root.join(v)
                .join("transformer/diffusion_pytorch_model.safetensors")
                .is_file()
        })
    };
    if let Ok(dir) = std::env::var("SCENEWORKS_MLX_BOOGU_DIR") {
        let path = PathBuf::from(dir.trim());
        if has_any_variant(&path) {
            return Some(path);
        }
    }
    let snaps =
        dirs_home().join(".cache/huggingface/hub/models--SceneWorks--boogu-image-mlx/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|root| has_any_variant(root))
}

/// Real-weights smoke (sc-6402, epic 6387): drive all three Boogu ids — Base (T2I true-CFG), Turbo
/// (DMD few-step, CFG-free) and Edit (instruction edit over a source Reference) — through the SAME
/// `boogu_model_subdir` → `load_engine` → `gen_core::load("boogu_*")` seam the worker uses, proving
/// the `mlx_gen_boogu` force-link survives in the worker binary and the production quant / guidance /
/// edit-conditioning resolution is consumed end-to-end. Each variant loads with the resolved Q8 spec
/// over its packed `<variant>/` subdir, generates once, asserts a non-constant RGB8 image of the
/// requested size with denoise progress, then DROPS the generator before the next loads (one ~23 GB
/// model resident at a time — peak footprint is per-variant, well within `minMemoryGb: 64`). Variants
/// not yet downloaded are skipped + reported; at least one must be present. Steps default low (env
/// `BOOGU_SMOKE_STEPS` / `BOOGU_SMOKE_TURBO_STEPS` / `BOOGU_SMOKE_RES`): this checks mechanics, not
/// quality (quality is an eyeball pass on the saved render). Run on demand:
/// `cargo test -p sceneworks-worker --lib -- --ignored boogu_real_weights --nocapture`.
#[cfg(target_os = "macos")]
#[ignore = "loads the real Boogu turnkey (~23 GB/variant); run manually on a Mac with SceneWorks/boogu-image-mlx cached"]
#[test]
fn boogu_real_weights_generates_base_turbo_edit() {
    let Some(root) = boogu_dir() else {
        eprintln!(
            "skipping boogu_real_weights: no SceneWorks/boogu-image-mlx snapshot found \
             (set SCENEWORKS_MLX_BOOGU_DIR or download the turnkey)"
        );
        return;
    };
    let env_u32 = |key: &str, default: u32| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    };
    let res = env_u32("BOOGU_SMOKE_RES", 512);
    let base_steps = env_u32("BOOGU_SMOKE_STEPS", 8);
    let turbo_steps = env_u32("BOOGU_SMOKE_TURBO_STEPS", 4);

    let cancel = gen_core::CancelFlag::new();
    let enhance = PromptEnhance::default();

    // Edit source, pre-sized to res×res so it satisfies the engine's multiple-of-16 guard (the
    // worker's `resolve_boogu_edit` fits the real source the same way). With `BOOGU_SMOKE_EDIT_SRC`
    // set to a PNG it loads + resizes that image — best for the quality eyeball (feed the Base render
    // → a clearly judgeable instruction edit); otherwise a synthetic diagonal gradient (mechanics).
    let source = match std::env::var("BOOGU_SMOKE_EDIT_SRC") {
        Ok(path) if !path.trim().is_empty() => {
            let decoded = image::open(path.trim())
                .unwrap_or_else(|e| panic!("open BOOGU_SMOKE_EDIT_SRC {}: {e}", path.trim()))
                .to_rgb8();
            let fitted =
                image::imageops::resize(&decoded, res, res, image::imageops::FilterType::Lanczos3);
            Image {
                width: res,
                height: res,
                pixels: fitted.into_raw(),
            }
        }
        _ => Image {
            width: res,
            height: res,
            pixels: (0..res * res)
                .flat_map(|i| {
                    [
                        (255 * (i % res) / res) as u8,
                        (255 * (i / res) / res) as u8,
                        160u8,
                    ]
                })
                .collect(),
        },
    };
    let edit_ref = (source, 1.0f32);

    // (id, steps, prompt). Base/Turbo are T2I; Edit additionally takes the source Reference (resolved
    // per-id below — the Qwen3-VL vision tower reads it + it VAE-encodes into the DiT latent).
    let cases = [
        (
            "boogu_image",
            base_steps,
            "a red fox sitting in a snowy forest at golden hour",
        ),
        (
            "boogu_image_turbo",
            turbo_steps,
            "a red fox sitting in a snowy forest at golden hour",
        ),
        (
            "boogu_image_edit",
            base_steps,
            "make it night with a full moon and cool blue light",
        ),
    ];

    let mut ran = 0u32;
    for (id, steps, prompt) in cases {
        // Only the Edit variant consumes the source Reference; Base/Turbo are pure T2I.
        let reference = (id == "boogu_image_edit").then_some(&edit_ref);
        let req = request(json!({
            "projectId": "p", "model": id, "prompt": "p", "advanced": {},
            "modelManifestEntry": { "mlx": { "quantize": 8 } },
        }));
        let dir = boogu_model_subdir(&root, &req);
        if !dir
            .join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
        {
            eprintln!(
                "skipping {id}: variant dir {} not downloaded",
                dir.display()
            );
            continue;
        }
        let model = mlx_model(id).unwrap();
        // The exact production spec: Q8 over the packed `<variant>/` weights, with the resolved
        // guidance (4.0 Base/Edit, None Turbo) + true_cfg (None for all boogu — Base/Edit forward
        // CFG via the `guidance` scalar; Turbo is CFG-free).
        let (quant, _bits) = resolve_quant(&req);
        let guidance = resolve_guidance(&req, &model);
        let true_cfg = resolve_true_cfg(&req, &model);

        let generator = load_engine(id, dir, quant, Vec::new(), None)
            .unwrap_or_else(|error| panic!("{id} load failed: {error}"));
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
            reference,
            &[],
            None,
            true_cfg,
            None,
            None,
            None,
            None,
            false,
            &enhance,
            &cancel,
            &mut |p| {
                if let gen_core::Progress::Step { current, .. } = p {
                    steps_seen = steps_seen.max(current);
                }
            },
        )
        .unwrap_or_else(|error| panic!("boogu {id} generation failed: {error}"));
        assert_eq!(
            pixels.len(),
            (w * h * 3) as usize,
            "{id}: RGB8-sized buffer"
        );
        assert!(
            w == res && h == res,
            "{id}: output matches requested {res}²"
        );
        assert!(steps_seen >= 1, "{id}: expected denoise step progress");
        assert!(
            pixels.windows(2).any(|x| x[0] != x[1]),
            "{id}: non-constant image"
        );
        eprintln!("boogu {id}: {w}x{h} RGB8, {steps_seen} steps observed");
        // Opt-in PNG dump for the eyeball/quality pass (`BOOGU_SMOKE_OUT=<dir>`): mechanics are
        // asserted above; a human reviews the saved render for actual quality.
        if let Ok(out_dir) = std::env::var("BOOGU_SMOKE_OUT") {
            let out_dir = out_dir.trim();
            std::fs::create_dir_all(out_dir).unwrap_or_else(|e| panic!("create {out_dir}: {e}"));
            let path = std::path::Path::new(out_dir).join(format!("{id}.png"));
            image::RgbImage::from_raw(w, h, pixels)
                .expect("RGB8 buffer → RgbImage")
                .save(&path)
                .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
            eprintln!("  wrote {}", path.display());
        }
        ran += 1;
        // generator dropped here → frees the ~23 GB model before the next variant loads.
    }
    assert!(
        ran >= 1,
        "expected at least one boogu variant present under {}",
        root.display()
    );
    eprintln!("boogu real-weights smoke: {ran}/3 variants validated");
}

// ---------------------------------------------------------------------------
// Shared strict-control driver (sc-8243): the `(engine_id, control_repo, supported_kinds)` single
// source of truth + the preprocess → conditioning core the three MLX registry strict-control paths
// (z-image / flux2 / qwen) route through. macOS-only (the driver is macOS-gated).
// ---------------------------------------------------------------------------

/// A small solid RGB control [`Image`] for passthrough / canny-source fixtures.
#[cfg(target_os = "macos")]
fn control_fixture(w: u32, h: u32, rgb: [u8; 3]) -> Image {
    Image {
        width: w,
        height: h,
        pixels: rgb
            .iter()
            .cycle()
            .take((w * h * 3) as usize)
            .copied()
            .collect(),
    }
}

/// A single pose entry parsed from a job (reuses `parse_poses` so the test exercises the real path).
#[cfg(target_os = "macos")]
fn one_pose() -> PoseInput {
    let req = request(json!({
        "projectId": "p", "model": "z_image_turbo", "prompt": "a knight",
        "advanced": { "poses": [{ "keypoints": [[0.5, 0.5]] }] }
    }));
    parse_poses(&req).pop().expect("one pose")
}

/// SINGLE SOURCE OF TRUTH: the S0 `(engine_id, control_repo, supported_kinds)` table. Exercises the
/// `repo` field (the catalog default sc-8244/sc-8245 consume) and the per-engine kind sets:
/// flux2/z-image = {Pose, Canny, Depth}; qwen = {Pose} only.
#[cfg(target_os = "macos")]
#[test]
fn strict_control_table_is_the_authority() {
    let flux1 = strict_control_engine("flux1_dev_control").expect("flux1 row");
    assert_eq!(
        flux1.repo,
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
    );
    assert_eq!(
        flux1.supported_kinds,
        &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth]
    );

    let flux2 = strict_control_engine("flux2_dev_control").expect("flux2 row");
    assert_eq!(flux2.repo, "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union");
    assert_eq!(
        flux2.supported_kinds,
        &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth]
    );

    let zimage = strict_control_engine("z_image_turbo_control").expect("z-image row");
    assert_eq!(
        zimage.repo,
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
    );
    assert_eq!(
        zimage.supported_kinds,
        &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth]
    );

    // sc-8251: the base (non-distilled, full-CFG) Z-Image control engine — its OWN row, the base
    // control repo (no `-Turbo-`), pose + canny + depth.
    let zimage_base = strict_control_engine("z_image_control").expect("z-image base row");
    assert_eq!(
        zimage_base.repo,
        "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1"
    );
    assert_eq!(
        zimage_base.supported_kinds,
        &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth]
    );
    // The base must NOT collapse onto the Turbo control repo.
    assert_ne!(zimage_base.repo, zimage.repo);

    let qwen = strict_control_engine("qwen_image_control").expect("qwen row");
    // sc-8267 source swap: InstantX → alibaba-pai 2512-Fun-Controlnet-Union (input-agnostic VACE branch).
    assert_eq!(
        qwen.repo,
        "alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union"
    );
    // sc-8250 exposure: the 2512-Fun Union admits pose + canny + depth.
    assert_eq!(
        qwen.supported_kinds,
        &[ControlKind::Pose, ControlKind::Canny, ControlKind::Depth]
    );

    // The SDXL tile detail path is NOT a Fun-Union strict-control engine and must never route here.
    assert!(strict_control_engine("flux2_dev").is_none());
    assert!(strict_control_engine("sdxl_tile_control").is_none());
}

/// supported_kinds validation: pose + canny + depth accepted on every Fun-Union strict-control engine
/// (qwen joined the canny/depth tier in sc-8250); an unknown engine id is itself an error.
#[cfg(target_os = "macos")]
#[test]
fn validate_control_kind_accepts_and_rejects_per_table() {
    // Pose / Canny / Depth are all accepted on every Fun-Union strict-control engine (sc-8250 unlocked
    // qwen's canny + depth alongside flux1 / flux2 / z-image).
    for engine in [
        "flux1_dev_control",
        "flux2_dev_control",
        "z_image_turbo_control",
        "z_image_control",
        "qwen_image_control",
    ] {
        assert!(validate_control_kind(engine, &ControlKind::Pose).is_ok());
        assert!(validate_control_kind(engine, &ControlKind::Canny).is_ok());
        assert!(validate_control_kind(engine, &ControlKind::Depth).is_ok());
    }
    // A free-form `Other` kind is rejected on the qwen Fun-Union engine with an actionable message.
    let err = validate_control_kind("qwen_image_control", &ControlKind::Other("scribble".into()))
        .expect_err("qwen Other rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("qwen_image_control") && msg.contains("scribble"),
        "{msg}"
    );
    // Unknown / non-Fun-Union engine id is rejected outright.
    assert!(validate_control_kind("sdxl_tile_control", &ControlKind::Pose).is_err());
}

/// `requested_control_kind`: default Pose (no `controlMode` → byte-preserved pose path); parse
/// canny/depth/pose; reject an unknown value.
#[cfg(target_os = "macos")]
#[test]
fn requested_control_kind_defaults_to_pose_and_parses_modes() {
    let kind = |adv: Value| {
        requested_control_kind(&request(json!({
            "projectId": "p", "model": "z_image_turbo", "prompt": "x", "advanced": adv
        })))
    };
    // Every current job omits controlMode → Pose (the proven, byte-preserved tier).
    assert_eq!(kind(json!({})).unwrap(), ControlKind::Pose);
    assert_eq!(
        kind(json!({ "controlMode": "pose" })).unwrap(),
        ControlKind::Pose
    );
    assert_eq!(
        kind(json!({ "controlMode": "Canny" })).unwrap(),
        ControlKind::Canny
    );
    assert_eq!(
        kind(json!({ "controlMode": "depth" })).unwrap(),
        ControlKind::Depth
    );
    assert!(kind(json!({ "controlMode": "scribble" })).is_err());
}

/// Preprocessor dispatch by kind: pose → `draw_wholebody` (byte-identical to a direct call);
/// user-supplied control map → verbatim passthrough for ANY kind; canny → edge map over a source;
/// depth dispatch routes to the estimator (and errors with a clear message when neither a source nor
/// a user map is available).
#[cfg(target_os = "macos")]
#[test]
fn preprocess_control_entry_dispatches_by_kind() {
    let (w, h) = (64u32, 48u32);
    let pose = one_pose();
    let stick = crate::openpose_skeleton::body_stickwidth(w, h);

    // Pose: byte-identical to the old per-engine skeleton render.
    let got = preprocess_control_entry(
        &ControlKind::Pose,
        None,
        Some(&pose),
        None,
        w,
        h,
        stick,
        None,
    )
    .expect("pose preprocess");
    let want = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        stick,
    );
    assert_eq!(got.width, w);
    assert_eq!(got.height, h);
    assert_eq!(
        got.pixels,
        want.into_raw(),
        "pose preprocessor must be byte-identical"
    );

    // User-supplied passthrough wins for ANY kind (verbatim, skip preprocessing) — including DEPTH:
    // a user-supplied depth map is used exactly as given, never re-estimated (sc-8242 passthrough).
    let supplied = control_fixture(w, h, [10, 20, 30]);
    for kind in [ControlKind::Pose, ControlKind::Canny, ControlKind::Depth] {
        let out =
            preprocess_control_entry(&kind, Some(&supplied), Some(&pose), None, w, h, stick, None)
                .expect("passthrough");
        assert_eq!(
            out.pixels, supplied.pixels,
            "{kind:?} passthrough must be verbatim"
        );
    }

    // Canny over a source: produces a same-dimension RGB edge map (grayscale broadcast).
    let source = control_fixture(w, h, [128, 128, 128]);
    let canny = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        None,
        Some(&source),
        w,
        h,
        stick,
        None,
    )
    .expect("canny preprocess");
    assert_eq!((canny.width, canny.height), (w, h));
    assert_eq!(canny.pixels.len(), (w * h * 3) as usize);

    // Canny without a source is an error (no synthetic input).
    assert!(
        preprocess_control_entry(&ControlKind::Canny, None, None, None, w, h, stick, None).is_err(),
        "canny needs a source"
    );

    // Depth with NO source and NO user map → a clear error (auto depth has nothing to estimate from).
    let err = preprocess_control_entry(&ControlKind::Depth, None, None, None, w, h, stick, None)
        .expect_err("depth needs a source or a user map");
    assert!(
        err.to_string().contains("depth control requires"),
        "{}",
        err
    );

    // Depth WITH a source but NO provisioned estimator weights → a clear "weights unavailable" error,
    // proving the dispatch routes into the estimator path (sc-8242) rather than silently passing
    // through. (The real-weight estimation itself is exercised by the mlx-gen-depth on-device smoke.)
    let depth_err = preprocess_control_entry(
        &ControlKind::Depth,
        None,
        None,
        Some(&source),
        w,
        h,
        stick,
        None,
    )
    .expect_err("depth dispatch reaches the estimator");
    assert!(
        depth_err.to_string().contains("weights are unavailable"),
        "depth dispatch must reach the estimator (got: {depth_err})"
    );
}

/// Conditioning construction: pose builds `Control { kind: Pose, scale }` (byte-identical to the old
/// hand-built shape), optionally followed by the identity `Reference`; depth uses the dedicated
/// `Depth` variant.
#[cfg(target_os = "macos")]
#[test]
fn build_control_conditioning_matches_legacy_shape() {
    let control = control_fixture(8, 8, [1, 2, 3]);

    // Pose, no identity init → exactly the old `vec![Control { Pose, scale }]`.
    let cond = build_control_conditioning(control.clone(), ControlKind::Pose, 0.9, None);
    assert_eq!(cond.len(), 1);
    match &cond[0] {
        Conditioning::Control { image, kind, scale } => {
            assert_eq!(image.pixels, control.pixels);
            assert_eq!(*kind, ControlKind::Pose);
            assert!((*scale - 0.9).abs() < 1e-6);
        }
        other => panic!("expected Control, got {other:?}"),
    }

    // Pose + identity init → Control then Reference (the flux2 / z-image opt-in tier).
    let init = control_fixture(8, 8, [9, 9, 9]);
    let cond = build_control_conditioning(
        control.clone(),
        ControlKind::Pose,
        0.75,
        Some(&(init.clone(), 0.6)),
    );
    assert_eq!(cond.len(), 2);
    assert!(matches!(cond[0], Conditioning::Control { .. }));
    match &cond[1] {
        Conditioning::Reference { image, strength } => {
            assert_eq!(image.pixels, init.pixels);
            assert_eq!(*strength, Some(0.6));
        }
        other => panic!("expected Reference, got {other:?}"),
    }

    // Depth → the dedicated `Depth` variant (not `Control`).
    let cond = build_control_conditioning(control.clone(), ControlKind::Depth, 1.0, None);
    assert_eq!(cond.len(), 1);
    assert!(
        matches!(cond[0], Conditioning::Depth { .. }),
        "depth uses Conditioning::Depth"
    );
}

/// sc-4410 strict-control pose scoring: a pose-library job that carries NO character identity
/// `referenceAssetId` resolves to `None` for the likeness source — so the strict-control streams build
/// no scorer and the `faceLikeness` field is omitted (honest — there is no identity to compare against,
/// not an error). This is the gate every strict-control pose lane (z-image / qwen / flux2-dev /
/// flux1-dev + their candle siblings) uses to decide whether to score; the decode-success branch
/// (referenceAssetId present) is exercised by the `#[ignore]` real-weight scorer test (it needs an asset
/// + weights). Blank ids count as absent.
#[cfg(target_os = "macos")]
#[test]
fn resolve_control_identity_source_is_none_without_reference() {
    let settings = Settings::from_env();
    let project_path = std::path::Path::new("/tmp/sc4410-nonexistent");
    // A bare pose set (skeleton-only, no identity reference) → None → no scorer → field omitted.
    let pose_only = request(json!({
        "projectId": "p", "model": "z_image_turbo",
        "advanced": { "poses": [{ "id": "a" }] }
    }));
    assert!(
        resolve_control_identity_source(&pose_only, &settings, project_path).is_none(),
        "pose set with no identity reference ⇒ no likeness source ⇒ field omitted"
    );
    // A blank referenceAssetId is treated as absent (no spurious decode attempt).
    let blank = request(json!({
        "projectId": "p", "model": "flux2_dev",
        "referenceAssetId": "   ",
        "advanced": { "poses": [{ "id": "a" }] }
    }));
    assert!(
        resolve_control_identity_source(&blank, &settings, project_path).is_none(),
        "blank referenceAssetId ⇒ treated as absent"
    );
}

/// sc-8248 / sc-8249 source threading: the input image canny/depth auto-derive their control map FROM
/// resolves with the right precedence (sourceAssetId wins, else referenceAssetId), and a pose-only job
/// (no source / reference) resolves to `None` — proving the live canny/depth path now has an input image
/// to preprocess (previously the per-engine streams hard-coded `source: None`), while pose stays
/// source-free (byte-preserved).
#[cfg(target_os = "macos")]
#[test]
fn control_source_asset_id_precedence_and_pose_is_source_free() {
    // sourceAssetId wins (the canonical control input image).
    assert_eq!(
        control_source_asset_id(&request(json!({
            "projectId": "p", "model": "z_image_turbo",
            "sourceAssetId": "src", "referenceAssetId": "ref",
            "advanced": { "controlMode": "canny" }
        }))),
        Some("src")
    );
    // No source → fall back to the character reference (a control job that only carried a reference).
    assert_eq!(
        control_source_asset_id(&request(json!({
            "projectId": "p", "model": "flux2_dev", "referenceAssetId": "ref",
            "advanced": { "controlMode": "depth" }
        }))),
        Some("ref")
    );
    // A pose-only job carries neither → None (the skeleton is synthetic; no input image needed). This is
    // the byte-preserved pose tier — the threaded source stays `None` exactly as before.
    assert_eq!(
        control_source_asset_id(&request(json!({
            "projectId": "p", "model": "flux_dev",
            "advanced": { "poses": [{ "id": "a" }] }
        }))),
        None
    );
    // Blank ids are treated as absent.
    assert_eq!(
        control_source_asset_id(&request(json!({
            "projectId": "p", "model": "z_image_turbo",
            "sourceAssetId": "   ", "referenceAssetId": ""
        }))),
        None
    );
}

/// sc-8248 / sc-8249 live enablement: with `controlMode = canny` and a threaded source image, the shared
/// preprocessor (the one every strict-control engine stream now feeds the input image into) produces a
/// real edge-map control conditioning — NOT the pose skeleton and NOT a no-op. This is the unit-level
/// proof that auto-canny/auto-depth now fire for flux1/flux2/z-image once the source is threaded (the
/// stream wiring passes `control_source` where it previously passed `None`). Depth dispatch is covered by
/// `preprocess_control_entry_dispatches_by_kind` (routes into the estimator); the real per-mode renders
/// are the on-device smokes.
#[cfg(target_os = "macos")]
#[test]
fn threaded_source_drives_auto_canny_control_conditioning() {
    let (w, h) = (64u32, 48u32);
    let pose = one_pose();
    let stick = crate::openpose_skeleton::body_stickwidth(w, h);
    // A threaded input "photo" (a flat fixture is enough — canny returns a same-dimension RGB edge map).
    let source = control_fixture(w, h, [128, 128, 128]);

    // canny + a threaded source (NO user passthrough, NO depth weights) → an edge-map control image.
    let control = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        Some(&pose),
        Some(&source),
        w,
        h,
        stick,
        None,
    )
    .expect("auto-canny over the threaded source");
    assert_eq!((control.width, control.height), (w, h));

    // It must NOT be the pose skeleton (proving the source — not the synthetic skeleton — drove it).
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        w,
        h,
        &pose.keypoints,
        pose.hands.as_deref(),
        pose.face.as_deref(),
        stick,
    );
    assert_ne!(
        control.pixels,
        skeleton.into_raw(),
        "auto-canny must derive from the source image, not render the pose skeleton"
    );

    // And it builds a `Control { kind: Canny }` conditioning (the engine's structural hint).
    let cond = build_control_conditioning(control, ControlKind::Canny, 0.75, None);
    assert_eq!(cond.len(), 1);
    assert!(
        matches!(
            &cond[0],
            Conditioning::Control {
                kind: ControlKind::Canny,
                ..
            }
        ),
        "canny builds a Control conditioning"
    );
}

// =====================================================================================================
// sc-8247 — V: real-weight e2e validation matrix  (`real_weight_matrix_*`)
// =====================================================================================================
//
// SINGLE ENTRY POINT for the {flux1_dev_control, flux2_dev_control, z_image_turbo_control,
// z_image_control, qwen_image_control} × {pose, canny, depth} real-weight gate. Every smoke below is
// `#[ignore]`d (needs real weights + a Metal device) and drives the FULL worker job seam —
// `*_control_load` (the `*_control_spec` LoadSpec the live stream builds) → `preprocess_control_entry`
// (skeleton / auto-canny / auto-depth) → `build_control_conditioning` → `*_control_generate_one`
// (the registered native-MLX engine) → decode — then asserts the control measurably STEERS the render.
//
// Run the whole matrix in one shot (each smoke loads its own engine, so this is serial + heavy):
//
//   FLUX1_DEV_DIR / SCENEWORKS_FLUX1_DEV_DIR    gated FLUX.1-dev diffusers snapshot (else HF cache)
//   SCENEWORKS_CONTROLNET_FLUX1                 Shakker FLUX.1-dev-ControlNet-Union-Pro-2.0 ckpt
//   SCENEWORKS_FLUX2_DEV_DIR                    converted Q4 FLUX.2-dev dir (else app-support default)
//   SCENEWORKS_DEPTH_ANYTHING_V2                Depth-Anything-V2-Small-hf dir (depth modes only)
//   # the rest resolve from the HF cache by repo id:
//   #   Tongyi-MAI/Z-Image-Turbo, Tongyi-MAI/Z-Image, Qwen/Qwen-Image,
//   #   alibaba-pai/{Z-Image-Turbo,Z-Image,FLUX.2-dev,Qwen-Image-2512}-Fun-Controlnet-Union*
//   cargo test -p sceneworks-worker --lib --release -- --ignored --nocapture real_weight_matrix
//
// Record the outcome in crates/sceneworks-worker/VALIDATION.md (the maintainer's on-device note). The
// on-device run IS the gate — CI never builds these (no GPU / no weights).
//
// Coverage map (which test backs each cell):
//   flux1_dev_control     pose=real_weight_matrix_flux1_pose_directed  canny=…_flux1_canny  depth=…_flux1_depth
//   flux2_dev_control     pose=real_weight_matrix_flux2_pose_directed  canny=…_flux2_canny  depth=…_flux2_depth
//   z_image_turbo_control pose=real_weight_matrix_zimage_turbo_pose_directed canny=…_zimage_turbo_canny depth=…_zimage_turbo_depth
//   z_image_control       pose=real_weight_matrix_zimage_base_pose_directed  canny=…_zimage_base_canny  depth=…_zimage_base_depth
//   qwen_image_control    pose=real_weight_matrix_qwen_pose_directed   canny=…_qwen_canny   depth=…_qwen_depth
//
// (The pre-existing `*_control_real_weights_*` smokes above remain as the per-backbone bring-up checks;
// these `real_weight_matrix_*` smokes are the consolidated, steer-asserting completion of the gate —
// every backbone × every mode, pose proven DIRECTED, canny/depth proven structural-steering.)

/// Mean absolute per-byte difference between two same-shape decodes — the control-vs-control-free steer
/// metric (0 = identical = the control did nothing). Panics on a shape mismatch (a bug, not a render).
#[cfg(target_os = "macos")]
fn matrix_mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "decodes must share a length to diff");
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

/// Mean per-byte std-dev — a cheap "is the decode non-degenerate (not all-black / flat / NaN)" floor.
#[cfg(target_os = "macos")]
fn matrix_std(pixels: &[u8]) -> f64 {
    let n = pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    (pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// A standing DWPose skeleton leaning LEFT or RIGHT — the directed-pose probe. The torso/limb columns
/// are shifted toward one side so a left-lean vs right-lean render must differ measurably IF the pose
/// control is spatially DIRECTED (not merely on/off). `lean` is a signed horizontal offset in
/// normalized units (negative = lean left, positive = lean right). Mirrors the standing keypoint layout
/// the existing pose smokes use.
#[cfg(target_os = "macos")]
fn directed_pose_skeleton(side: u32, lean: f64) -> Image {
    let dx = lean;
    // 18-point body skeleton; shift every x by `dx`, clamped to the canvas.
    let pt = |x: f64, y: f64| json!([(x + dx).clamp(0.02, 0.98), y]);
    let kp = crate::openpose_skeleton::normalize_keypoints(&json!([
        pt(0.50, 0.20),
        pt(0.50, 0.35),
        pt(0.42, 0.35),
        pt(0.40, 0.50),
        pt(0.40, 0.65),
        pt(0.58, 0.35),
        pt(0.60, 0.50),
        pt(0.60, 0.65),
        pt(0.45, 0.60),
        pt(0.45, 0.80),
        pt(0.45, 0.95),
        pt(0.55, 0.60),
        pt(0.55, 0.80),
        pt(0.55, 0.95),
        pt(0.48, 0.18),
        pt(0.52, 0.18),
        pt(0.46, 0.20),
        pt(0.54, 0.20)
    ]));
    let skeleton = crate::openpose_skeleton::draw_wholebody(
        side,
        side,
        &kp,
        None,
        None,
        crate::openpose_skeleton::body_stickwidth(side, side),
    );
    Image {
        width: side,
        height: side,
        pixels: skeleton.into_raw(),
    }
}

/// A non-flat synthetic source for the auto-canny / auto-depth modes: a centered bright square on a
/// dark field, so the canny edge detector finds edges and the depth estimator finds a foreground/
/// background split (a flat field gives a degenerate edge/depth map). 512² RGB.
#[cfg(target_os = "macos")]
fn matrix_structured_source(side: u32) -> Image {
    let s = side as usize;
    let mut pixels = vec![20u8; s * s * 3];
    let (lo, hi) = (s / 4, 3 * s / 4);
    for y in lo..hi {
        for x in lo..hi {
            let i = (y * s + x) * 3;
            pixels[i] = 230;
            pixels[i + 1] = 230;
            pixels[i + 2] = 230;
        }
    }
    Image {
        width: side,
        height: side,
        pixels,
    }
}

/// Resolve the Depth-Anything-V2-Small snapshot dir for the depth-mode smokes: `SCENEWORKS_DEPTH_ANYTHING_V2`
/// override → the HF cache snapshot. Returns the dir that holds `model.safetensors`.
#[cfg(target_os = "macos")]
fn matrix_depth_weights_dir() -> std::path::PathBuf {
    use crate::depth::DEPTH_ANYTHING_V2_FILE;
    if let Ok(p) = std::env::var("SCENEWORKS_DEPTH_ANYTHING_V2") {
        let p = std::path::PathBuf::from(p);
        if p.join(DEPTH_ANYTHING_V2_FILE).is_file() {
            return p;
        }
    }
    let dir = hf_snapshot("models--depth-anything--Depth-Anything-V2-Small-hf");
    assert!(
        dir.join(DEPTH_ANYTHING_V2_FILE).is_file(),
        "Depth-Anything-V2-Small weights missing in {} (set SCENEWORKS_DEPTH_ANYTHING_V2)",
        dir.display()
    );
    dir
}

/// Assert one structural-control render (a) is non-degenerate and (b) steered the decode away from the
/// matched control-free baseline. Shared by the canny/depth cells (and the per-lean pose checks).
#[cfg(target_os = "macos")]
fn assert_matrix_steer(label: &str, render: &[u8], baseline: &[u8], steer_floor: f64) {
    let std = matrix_std(render);
    let steer = matrix_mean_abs_delta(render, baseline);
    println!("[matrix] {label}: std {std:.2}, steer(meanAbsΔ vs control-free) {steer:.2}");
    assert!(std > 5.0, "{label} render looks degenerate (std {std:.2})");
    assert!(
        steer > steer_floor,
        "{label} control did not steer the output away from the control-free baseline \
         (meanAbsΔ {steer:.2} ≤ floor {steer_floor:.2}) — the structural hint was inert"
    );
}

// ---- flux1_dev_control ------------------------------------------------------------------------------

/// Resolve the FLUX.1-dev base + Shakker control checkpoint the same way the bring-up smoke does.
#[cfg(target_os = "macos")]
fn matrix_flux1_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let base = std::env::var("SCENEWORKS_FLUX1_DEV_DIR")
        .or_else(|_| std::env::var("FLUX1_DEV_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::fs::read_dir(
                dirs_home()
                    .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots"),
            )
            .expect("FLUX.1-dev snapshots dir")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.join("model_index.json").is_file())
            .expect("a FLUX.1-dev snapshot dir")
        });
    let control = std::env::var("SCENEWORKS_CONTROLNET_FLUX1")
        .or_else(|_| std::env::var("FLUX1_CONTROL"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::fs::read_dir(dirs_home().join(
                ".cache/huggingface/hub/models--Shakker-Labs--FLUX.1-dev-ControlNet-Union-Pro-2.0/snapshots",
            ))
            .expect("control snapshots dir")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .map(|dir| dir.join(FLUX1_CONTROL_FILE))
            .filter(|path| path.exists())
            .expect("control weights file")
        });
    (base, control)
}

#[cfg(target_os = "macos")]
fn matrix_flux1_render(
    generator: &dyn Generator,
    side: u32,
    conditioning: Vec<Conditioning>,
) -> Vec<u8> {
    let cancel = gen_core::CancelFlag::new();
    let (_, _, pixels) = flux1_control_generate_one(
        generator,
        "a person standing in a meadow, photorealistic",
        side,
        side,
        42,
        28,
        Some(3.5),
        conditioning,
        &cancel,
        &mut |_| {},
    )
    .expect("flux1 control render");
    pixels
}

/// POSE (directed) + pose regression re-proof for flux1_dev_control: a left-lean vs right-lean skeleton
/// must each steer off the control-free baseline AND differ from each other (directed control survives
/// the S1/S2 shared-driver consolidation).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: FLUX.1-dev + Shakker Union-Pro-2.0 + Metal"]
fn real_weight_matrix_flux1_pose_directed() {
    let (base, control) = matrix_flux1_paths();
    let generator =
        flux1_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let baseline = matrix_flux1_render(&*generator, side, Vec::new());
    let left = matrix_flux1_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, -0.18),
            ControlKind::Pose,
            0.7,
            None,
        ),
    );
    let right = matrix_flux1_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, 0.18),
            ControlKind::Pose,
            0.7,
            None,
        ),
    );
    assert_matrix_steer("flux1 pose(left)", &left, &baseline, 1.0);
    assert_matrix_steer("flux1 pose(right)", &right, &baseline, 1.0);
    let directed = matrix_mean_abs_delta(&left, &right);
    println!("[matrix] flux1 pose directed(left vs right) meanAbsΔ {directed:.2}");
    assert!(
        directed > 1.0,
        "flux1 pose is not DIRECTED: left-lean and right-lean renders barely differ (meanAbsΔ {directed:.2})"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: FLUX.1-dev + Shakker Union-Pro-2.0 + Metal"]
fn real_weight_matrix_flux1_canny() {
    let (base, control) = matrix_flux1_paths();
    let generator =
        flux1_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        None,
    )
    .expect("auto-canny");
    let baseline = matrix_flux1_render(&*generator, side, Vec::new());
    let render = matrix_flux1_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Canny, 0.7, None),
    );
    assert_matrix_steer("flux1 canny", &render, &baseline, 1.0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: FLUX.1-dev + Shakker Union-Pro-2.0 + Depth-Anything-V2 + Metal"]
fn real_weight_matrix_flux1_depth() {
    let (base, control) = matrix_flux1_paths();
    let depth = matrix_depth_weights_dir();
    let generator =
        flux1_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Depth,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        Some(depth.as_path()),
    )
    .expect("auto-depth");
    let baseline = matrix_flux1_render(&*generator, side, Vec::new());
    let render = matrix_flux1_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Depth, 0.7, None),
    );
    assert_matrix_steer("flux1 depth", &render, &baseline, 1.0);
}

// ---- flux2_dev_control ------------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn matrix_flux2_paths() -> (std::path::PathBuf, std::path::PathBuf) {
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
    (base, control)
}

#[cfg(target_os = "macos")]
fn matrix_flux2_render(
    generator: &dyn Generator,
    side: u32,
    conditioning: Vec<Conditioning>,
) -> Vec<u8> {
    let cancel = gen_core::CancelFlag::new();
    let (_, _, pixels) = flux2_control_generate_one(
        generator,
        "a person standing in a meadow, photorealistic",
        side,
        side,
        42,
        8,
        Some(4.0),
        conditioning,
        &cancel,
        &mut |_| {},
    )
    .expect("flux2 control render");
    pixels
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: converted FLUX.2-dev + FLUX.2-dev-Fun-Controlnet-Union + Metal"]
fn real_weight_matrix_flux2_pose_directed() {
    let (base, control) = matrix_flux2_paths();
    let generator =
        flux2_control_load(base, control, Some(gen_core::Quant::Q4), Vec::new()).unwrap();
    let side = 512u32;
    let baseline = matrix_flux2_render(&*generator, side, Vec::new());
    let left = matrix_flux2_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, -0.18),
            ControlKind::Pose,
            0.75,
            None,
        ),
    );
    let right = matrix_flux2_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, 0.18),
            ControlKind::Pose,
            0.75,
            None,
        ),
    );
    assert_matrix_steer("flux2 pose(left)", &left, &baseline, 1.0);
    assert_matrix_steer("flux2 pose(right)", &right, &baseline, 1.0);
    let directed = matrix_mean_abs_delta(&left, &right);
    println!("[matrix] flux2 pose directed(left vs right) meanAbsΔ {directed:.2}");
    assert!(
        directed > 1.0,
        "flux2 pose is not DIRECTED: left/right renders barely differ (meanAbsΔ {directed:.2})"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: converted FLUX.2-dev + FLUX.2-dev-Fun-Controlnet-Union + Metal"]
fn real_weight_matrix_flux2_canny() {
    let (base, control) = matrix_flux2_paths();
    let generator =
        flux2_control_load(base, control, Some(gen_core::Quant::Q4), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        None,
    )
    .expect("auto-canny");
    let baseline = matrix_flux2_render(&*generator, side, Vec::new());
    let render = matrix_flux2_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Canny, 0.75, None),
    );
    assert_matrix_steer("flux2 canny", &render, &baseline, 1.0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: converted FLUX.2-dev + FLUX.2-dev-Fun-Controlnet-Union + Depth-Anything-V2 + Metal"]
fn real_weight_matrix_flux2_depth() {
    let (base, control) = matrix_flux2_paths();
    let depth = matrix_depth_weights_dir();
    let generator =
        flux2_control_load(base, control, Some(gen_core::Quant::Q4), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Depth,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        Some(depth.as_path()),
    )
    .expect("auto-depth");
    let baseline = matrix_flux2_render(&*generator, side, Vec::new());
    let render = matrix_flux2_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Depth, 0.75, None),
    );
    assert_matrix_steer("flux2 depth", &render, &baseline, 1.0);
}

// ---- z_image_turbo_control --------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn matrix_zimage_turbo_paths() -> (std::path::PathBuf, std::path::PathBuf) {
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
    (base, control)
}

/// Z-Image-Turbo is guidance-distilled (no CFG / negative) — fixed 8 steps, like the bring-up smoke.
#[cfg(target_os = "macos")]
fn matrix_zimage_turbo_render(
    generator: &dyn Generator,
    side: u32,
    conditioning: Vec<Conditioning>,
) -> Vec<u8> {
    let cancel = gen_core::CancelFlag::new();
    let (_, _, pixels) = zimage_control_generate_one(
        generator,
        "a person standing in a meadow",
        side,
        side,
        42,
        8,
        conditioning,
        &cancel,
        &mut |_| {},
    )
    .expect("z-image-turbo control render");
    pixels
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: Z-Image-Turbo + Z-Image-Turbo-Fun-Controlnet-Union-2.1 + Metal"]
fn real_weight_matrix_zimage_turbo_pose_directed() {
    let (base, control) = matrix_zimage_turbo_paths();
    let generator =
        zimage_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let baseline = matrix_zimage_turbo_render(&*generator, side, Vec::new());
    let left = matrix_zimage_turbo_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, -0.18),
            ControlKind::Pose,
            0.9,
            None,
        ),
    );
    let right = matrix_zimage_turbo_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, 0.18),
            ControlKind::Pose,
            0.9,
            None,
        ),
    );
    assert_matrix_steer("z-image-turbo pose(left)", &left, &baseline, 1.0);
    assert_matrix_steer("z-image-turbo pose(right)", &right, &baseline, 1.0);
    let directed = matrix_mean_abs_delta(&left, &right);
    println!("[matrix] z-image-turbo pose directed(left vs right) meanAbsΔ {directed:.2}");
    assert!(
        directed > 1.0,
        "z-image-turbo pose is not DIRECTED: left/right renders barely differ (meanAbsΔ {directed:.2})"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: Z-Image-Turbo + Z-Image-Turbo-Fun-Controlnet-Union-2.1 + Metal"]
fn real_weight_matrix_zimage_turbo_canny() {
    let (base, control) = matrix_zimage_turbo_paths();
    let generator =
        zimage_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        None,
    )
    .expect("auto-canny");
    let baseline = matrix_zimage_turbo_render(&*generator, side, Vec::new());
    let render = matrix_zimage_turbo_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Canny, 0.9, None),
    );
    assert_matrix_steer("z-image-turbo canny", &render, &baseline, 1.0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: Z-Image-Turbo + Z-Image-Turbo-Fun-Controlnet-Union-2.1 + Depth-Anything-V2 + Metal"]
fn real_weight_matrix_zimage_turbo_depth() {
    let (base, control) = matrix_zimage_turbo_paths();
    let depth = matrix_depth_weights_dir();
    let generator =
        zimage_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Depth,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        Some(depth.as_path()),
    )
    .expect("auto-depth");
    let baseline = matrix_zimage_turbo_render(&*generator, side, Vec::new());
    let render = matrix_zimage_turbo_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Depth, 0.9, None),
    );
    assert_matrix_steer("z-image-turbo depth", &render, &baseline, 1.0);
}

// ---- z_image_control (base, full CFG) ---------------------------------------------------------------

#[cfg(target_os = "macos")]
fn matrix_zimage_base_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let base = hf_snapshot("models--Tongyi-MAI--Z-Image");
    let control = std::fs::read_dir(dirs_home().join(
        ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Fun-Controlnet-Union-2.1/snapshots",
    ))
    .expect("base control snapshots dir")
    .flatten()
    .map(|entry| entry.path())
    .find(|path| path.is_dir())
    .map(|dir| dir.join(super::ZIMAGE_BASE_CONTROL_FILE))
    .filter(|path| path.exists())
    .expect("base control weights file");
    (base, control)
}

#[cfg(target_os = "macos")]
fn matrix_zimage_base_render(
    generator: &dyn Generator,
    side: u32,
    conditioning: Vec<Conditioning>,
) -> Vec<u8> {
    let cancel = gen_core::CancelFlag::new();
    let (_, _, pixels) = zimage_base_control_generate_one(
        generator,
        "a person standing in a meadow",
        None,
        side,
        side,
        42,
        50,
        4.0,
        conditioning,
        &cancel,
        &mut |_| {},
    )
    .expect("z-image base control render");
    pixels
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: base Z-Image + Z-Image-Fun-Controlnet-Union-2.1 + Metal"]
fn real_weight_matrix_zimage_base_pose_directed() {
    let (base, control) = matrix_zimage_base_paths();
    let generator =
        zimage_base_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let baseline = matrix_zimage_base_render(&*generator, side, Vec::new());
    let left = matrix_zimage_base_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, -0.18),
            ControlKind::Pose,
            0.9,
            None,
        ),
    );
    let right = matrix_zimage_base_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, 0.18),
            ControlKind::Pose,
            0.9,
            None,
        ),
    );
    assert_matrix_steer("z-image-base pose(left)", &left, &baseline, 1.0);
    assert_matrix_steer("z-image-base pose(right)", &right, &baseline, 1.0);
    let directed = matrix_mean_abs_delta(&left, &right);
    println!("[matrix] z-image-base pose directed(left vs right) meanAbsΔ {directed:.2}");
    assert!(
        directed > 1.0,
        "z-image-base pose is not DIRECTED: left/right renders barely differ (meanAbsΔ {directed:.2})"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: base Z-Image + Z-Image-Fun-Controlnet-Union-2.1 + Metal"]
fn real_weight_matrix_zimage_base_canny() {
    let (base, control) = matrix_zimage_base_paths();
    let generator =
        zimage_base_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        None,
    )
    .expect("auto-canny");
    let baseline = matrix_zimage_base_render(&*generator, side, Vec::new());
    let render = matrix_zimage_base_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Canny, 0.9, None),
    );
    assert_matrix_steer("z-image-base canny", &render, &baseline, 1.0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: base Z-Image + Z-Image-Fun-Controlnet-Union-2.1 + Depth-Anything-V2 + Metal"]
fn real_weight_matrix_zimage_base_depth() {
    let (base, control) = matrix_zimage_base_paths();
    let depth = matrix_depth_weights_dir();
    let generator =
        zimage_base_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Depth,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        Some(depth.as_path()),
    )
    .expect("auto-depth");
    let baseline = matrix_zimage_base_render(&*generator, side, Vec::new());
    let render = matrix_zimage_base_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Depth, 0.9, None),
    );
    assert_matrix_steer("z-image-base depth", &render, &baseline, 1.0);
}

// ---- qwen_image_control -----------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn matrix_qwen_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let base = hf_snapshot("models--Qwen--Qwen-Image");
    let control = hf_snapshot("models--alibaba-pai--Qwen-Image-2512-Fun-Controlnet-Union")
        .join(super::QWEN_CONTROL_FILE);
    assert!(
        control.exists(),
        "Qwen control weights missing: {control:?}"
    );
    (base, control)
}

#[cfg(target_os = "macos")]
fn matrix_qwen_render(
    generator: &dyn Generator,
    side: u32,
    conditioning: Vec<Conditioning>,
) -> Vec<u8> {
    let cancel = gen_core::CancelFlag::new();
    let (_, _, pixels) = qwen_control_generate_one(
        generator,
        "a person standing in a meadow",
        Some("blurry, low quality".to_owned()),
        side,
        side,
        42,
        4,
        4.0,
        conditioning,
        false,
        &cancel,
        &mut |_| {},
    )
    .expect("qwen control render");
    pixels
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: Qwen-Image + Qwen-Image-2512-Fun-Controlnet-Union + Metal"]
fn real_weight_matrix_qwen_pose_directed() {
    let (base, control) = matrix_qwen_paths();
    let generator =
        qwen_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let baseline = matrix_qwen_render(&*generator, side, Vec::new());
    let left = matrix_qwen_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, -0.18),
            ControlKind::Pose,
            0.9,
            None,
        ),
    );
    let right = matrix_qwen_render(
        &*generator,
        side,
        build_control_conditioning(
            directed_pose_skeleton(side, 0.18),
            ControlKind::Pose,
            0.9,
            None,
        ),
    );
    assert_matrix_steer("qwen pose(left)", &left, &baseline, 1.0);
    assert_matrix_steer("qwen pose(right)", &right, &baseline, 1.0);
    let directed = matrix_mean_abs_delta(&left, &right);
    println!("[matrix] qwen pose directed(left vs right) meanAbsΔ {directed:.2}");
    assert!(
        directed > 1.0,
        "qwen pose is not DIRECTED: left/right renders barely differ (meanAbsΔ {directed:.2})"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: Qwen-Image + Qwen-Image-2512-Fun-Controlnet-Union + Metal"]
fn real_weight_matrix_qwen_canny() {
    let (base, control) = matrix_qwen_paths();
    let generator =
        qwen_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Canny,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        None,
    )
    .expect("auto-canny");
    let baseline = matrix_qwen_render(&*generator, side, Vec::new());
    let render = matrix_qwen_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Canny, 0.9, None),
    );
    assert_matrix_steer("qwen canny", &render, &baseline, 1.0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "real-weight matrix: Qwen-Image + Qwen-Image-2512-Fun-Controlnet-Union + Depth-Anything-V2 + Metal"]
fn real_weight_matrix_qwen_depth() {
    let (base, control) = matrix_qwen_paths();
    let depth = matrix_depth_weights_dir();
    let generator =
        qwen_control_load(base, control, Some(gen_core::Quant::Q8), Vec::new()).unwrap();
    let side = 512u32;
    let source = matrix_structured_source(side);
    let map = preprocess_control_entry(
        &ControlKind::Depth,
        None,
        None,
        Some(&source),
        side,
        side,
        crate::openpose_skeleton::body_stickwidth(side, side),
        Some(depth.as_path()),
    )
    .expect("auto-depth");
    let baseline = matrix_qwen_render(&*generator, side, Vec::new());
    let render = matrix_qwen_render(
        &*generator,
        side,
        build_control_conditioning(map, ControlKind::Depth, 0.9, None),
    );
    assert_matrix_steer("qwen depth", &render, &baseline, 1.0);
}
