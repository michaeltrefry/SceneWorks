use std::fs;
use std::path::{Path, PathBuf};

use sceneworks_core::training::{
    build_training_plan, builtin_training_targets, BuildTrainingPlan, LoraTrainingRequest,
    TrainingConfig, TrainingDataset, TrainingModality, TrainingOutputKind, TrainingPlan,
    TrainingPlanError, TrainingPresetRegistry, TrainingProvenance, TrainingTargetRegistry,
    TRAINING_CONTRACT_SCHEMA_VERSION, TRAINING_PLAN_VERSION,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

fn fixture_path(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("rust_migration_contracts")
        .join("training")
        .join(relative_path)
}

fn load_fixture(relative_path: &str) -> Value {
    let path = fixture_path(relative_path);
    let payload = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&payload)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn assert_round_trip<T>(relative_path: &str)
where
    T: DeserializeOwned + Serialize,
{
    let original = load_fixture(relative_path);
    let typed: T = serde_json::from_value(original.clone())
        .unwrap_or_else(|error| panic!("failed to deserialize {relative_path}: {error}"));
    let encoded = serde_json::to_value(typed)
        .unwrap_or_else(|error| panic!("failed to serialize {relative_path}: {error}"));

    assert_eq!(
        encoded, original,
        "{relative_path} drifted after typed round-trip"
    );
}

#[test]
fn training_dataset_round_trips() {
    assert_round_trip::<TrainingDataset>("dataset.json");
}

#[test]
fn training_config_round_trips() {
    assert_round_trip::<TrainingConfig>("training-config.json");
}

#[test]
fn lora_training_request_round_trips() {
    assert_round_trip::<LoraTrainingRequest>("lora-training-request.json");
}

#[test]
fn training_plan_round_trips() {
    assert_round_trip::<TrainingPlan>("training-plan.json");
}

#[test]
fn training_provenance_round_trips() {
    assert_round_trip::<TrainingProvenance>("training-provenance.json");
}

#[test]
fn training_target_registry_round_trips() {
    assert_round_trip::<TrainingTargetRegistry>("target-registry.json");
}

#[test]
fn training_preset_registry_round_trips() {
    assert_round_trip::<TrainingPresetRegistry>("preset-registry.json");
}

#[test]
fn builtin_registry_matches_committed_snapshot() {
    let expected = load_fixture("target-registry.json");
    let encoded =
        serde_json::to_value(builtin_training_targets()).expect("builtin registry serializes");

    assert_eq!(
        encoded, expected,
        "builtin target registry drifted from committed snapshot"
    );
}

#[test]
fn builtin_targets_gate_network_types() {
    let registry = builtin_training_targets();

    let advertises = |target: &sceneworks_core::training::TrainingTarget, value: &str| {
        target
            .limits
            .get("networkTypes")
            .and_then(Value::as_array)
            .is_some_and(|types| types.iter().any(|entry| entry.as_str() == Some(value)))
    };

    // Every target advertises `lora`; only the validated torch/PEFT backends
    // (epic 2193) advertise `lokr`: the Z-Image/SDXL image backends (v1), the
    // Kolors image backend (SDXL-architecture, epic 1929 / sc-2217), and the
    // Wan2.2 5B video backend (sc-2211). MLX-only and MoE targets stay lora-only.
    let lokr_targets: Vec<&str> = registry
        .targets
        .iter()
        .filter(|target| {
            assert!(
                advertises(target, "lora"),
                "target {} must advertise lora in limits.networkTypes",
                target.id
            );
            advertises(target, "lokr")
        })
        .map(|target| target.id.as_str())
        .collect();

    assert_eq!(
        lokr_targets,
        ["z_image_turbo_lora", "sdxl_lora", "kolors_lora", "wan_lora"]
    );
}

#[test]
#[ignore = "regen helper: run with REGEN_FIXTURE=1 to rewrite target-registry.json"]
fn regen_target_registry_fixture() {
    if std::env::var("REGEN_FIXTURE").is_err() {
        return;
    }
    let pretty = serde_json::to_string_pretty(&builtin_training_targets())
        .expect("builtin registry serializes");
    fs::write(fixture_path("target-registry.json"), pretty + "\n").expect("write fixture");
}

#[test]
#[ignore = "regen helper: run with REGEN_FIXTURE=1 to rewrite preset-registry.json"]
fn regen_preset_registry_fixture() {
    if std::env::var("REGEN_FIXTURE").is_err() {
        return;
    }
    let pretty =
        serde_json::to_string_pretty(&sceneworks_core::training::builtin_training_presets())
            .expect("builtin preset registry serializes");
    fs::write(fixture_path("preset-registry.json"), pretty + "\n").expect("write fixture");
}

#[test]
fn builtin_preset_registry_matches_committed_snapshot() {
    let expected = load_fixture("preset-registry.json");
    let encoded = serde_json::to_value(sceneworks_core::training::builtin_training_presets())
        .expect("builtin preset registry serializes");

    assert_eq!(
        encoded, expected,
        "builtin preset registry drifted from committed snapshot"
    );
}

#[test]
fn builtin_preset_registry_exposes_optimizer_sensitive_defaults() {
    let registry = sceneworks_core::training::builtin_training_presets();
    let prodigy = registry
        .presets
        .iter()
        .find(|preset| preset.id == "z_image_turbo_lora.character.prodigyopt.balanced")
        .expect("prodigy preset present");

    assert_eq!(prodigy.target_id, "z_image_turbo_lora");
    assert_eq!(prodigy.optimizer, "prodigyopt");
    assert_eq!(prodigy.config.optimizer, "prodigyopt");
    assert_eq!(prodigy.config.learning_rate.as_f64(), Some(1.0));
    assert_eq!(prodigy.config.steps, 1600);
    assert_eq!(prodigy.config.advanced["sampleEvery"], 200);
    assert_eq!(prodigy.config.advanced["sampleSteps"], 8);
    assert_eq!(prodigy.ui["experimental"], true);

    let balanced = registry
        .presets
        .iter()
        .find(|preset| preset.id == "z_image_turbo_lora.character.adamw8bit.balanced")
        .expect("balanced character preset present");
    assert_eq!(balanced.config.steps, 3000);
    assert_eq!(
        balanced.config.advanced["trainingAdapterRepo"],
        "ostris/zimage_turbo_training_adapter"
    );
    assert_eq!(balanced.config.advanced["timestepType"], "sigmoid");
    assert_eq!(balanced.config.advanced["timestepBias"], "high_noise");
    assert_eq!(balanced.config.advanced["gradientCheckpointing"], true);
}

#[test]
fn builtin_registry_exposes_z_image_turbo_target() {
    let registry = builtin_training_targets();
    assert_eq!(registry.schema_version, TRAINING_CONTRACT_SCHEMA_VERSION);

    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "z-image");
    assert_eq!(target.base_model, "z_image_turbo");
    assert_eq!(target.kernel, "z_image_lora");
    assert_eq!(target.defaults.rank, 16);
    assert_eq!(target.defaults.resolution, 1024);
    assert_eq!(target.defaults.trigger_word, None);
}

#[test]
fn builtin_registry_exposes_sdxl_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "sdxl_lora")
        .expect("sdxl_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "sdxl");
    assert_eq!(target.base_model, "sdxl");
    assert_eq!(target.kernel, "sdxl_lora");
    assert_eq!(target.defaults.rank, 16);
    assert_eq!(target.defaults.resolution, 1024);
    // Real CFG previews (positive guidance), unlike the distilled Z-Image target.
    assert_eq!(
        target.defaults.advanced.get("sampleGuidanceScale"),
        Some(&serde_json::json!(7.0))
    );
    // SDXL UNet attention modules drive the LoRA injection.
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
}

#[test]
fn builtin_registry_exposes_kolors_target() {
    // Kolors (epic 1929) is an SDXL-architecture U-Net target served by the
    // `kolors_lora` kernel; it reuses the SDXL attention modules + LoKr support
    // (epic 2193 / sc-2217) and resolves the Kolors-diffusers base.
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "kolors_lora")
        .expect("kolors_lora target present");

    assert_eq!(target.modality, TrainingModality::Image);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "kolors");
    assert_eq!(target.base_model, "kolors");
    assert_eq!(target.kernel, "kolors_lora");
    assert_eq!(
        target.base_model_repo.as_deref(),
        Some("Kwai-Kolors/Kolors-diffusers")
    );
    // SDXL-shared attention modules + LoKr advertised (sc-2217).
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
    assert_eq!(
        target.limits.get("networkTypes"),
        Some(&serde_json::json!(["lora", "lokr"]))
    );
}

#[test]
fn builtin_presets_expose_sdxl_character_default() {
    let registry = sceneworks_core::training::builtin_training_presets();
    let default_character = registry
        .presets
        .iter()
        .find(|preset| preset.id == "sdxl_lora.character.adamw8bit.balanced")
        .expect("sdxl character balanced preset present");

    assert_eq!(default_character.target_id, "sdxl_lora");
    assert_eq!(default_character.optimizer, "adamw8bit");
    assert_eq!(default_character.config.rank, 16);
    assert_eq!(
        default_character.ui.get("default"),
        Some(&serde_json::json!(true))
    );

    let style = registry
        .presets
        .iter()
        .find(|preset| preset.id == "sdxl_lora.style.adamw8bit.balanced")
        .expect("sdxl style preset present");
    assert_eq!(style.config.rank, 32);
    assert_eq!(style.config.alpha, 16);
}

#[test]
fn builtin_registry_exposes_ltx_video_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "ltx_video_lora")
        .expect("ltx_video_lora target present");

    assert_eq!(target.modality, TrainingModality::Video);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "ltx-video");
    assert_eq!(target.base_model, "ltx_2_3");
    assert_eq!(target.kernel, "ltx_mlx_lora");
    assert_eq!(target.defaults.rank, 32);
    assert_eq!(target.defaults.resolution, 768);
    assert_eq!(target.defaults.optimizer, "adamw");
    // Apple-Silicon/MLX-only marker the capability gate (story 1538) and the
    // frontend key off.
    assert_eq!(
        target.limits.get("appleSiliconOnly"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn ltx_video_target_resolves_image_dataset_into_plan() {
    // The video target consumes the image dataset fixture unchanged: build a plan
    // and confirm it carries the video target's kernel/family/modality.
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "ltx_video_lora")
        .expect("ltx_video_lora target present");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_ltx",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_ltx_new",
        base_model_path: "/models/ltx".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_ltx_new"),
        file_name: "ltx_character.safetensors".to_owned(),
        created_at: "2026-05-22T00:00:00Z".to_owned(),
    })
    .expect("ltx plan resolves");

    assert_eq!(plan.plan_version, TRAINING_PLAN_VERSION);
    assert_eq!(plan.target.kernel, "ltx_mlx_lora");
    assert_eq!(plan.target.family, "ltx-video");
    assert_eq!(plan.target.modality, TrainingModality::Video);
    assert!(!plan.dataset.items.is_empty());
}

#[test]
fn builtin_registry_exposes_wan_target() {
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "wan_lora")
        .expect("wan_lora target present");

    assert_eq!(target.modality, TrainingModality::Video);
    assert_eq!(target.output_kind, TrainingOutputKind::Lora);
    assert_eq!(target.family, "wan-video");
    assert_eq!(target.base_model, "wan_2_2");
    assert_eq!(target.kernel, "wan_lora");
    assert_eq!(target.defaults.rank, 32);
    assert_eq!(target.defaults.resolution, 512);
    // Cross-platform torch trainer: plain AdamW (adamw8bit is CUDA-only and falls
    // back), unlike the MLX LTX target which is also adamw but MPS-only.
    assert_eq!(target.defaults.optimizer, "adamw");
    // Wan transformer attention projections drive the LoRA injection.
    assert_eq!(
        target.defaults.advanced.get("loraTargetModules"),
        Some(&serde_json::json!(["to_q", "to_k", "to_v", "to_out.0"]))
    );
    // Still-image training: each item encodes to a single Wan-VAE latent frame.
    assert_eq!(
        target.defaults.advanced.get("numFrames"),
        Some(&serde_json::json!(1))
    );
    // Not MLX/Apple-Silicon-gated — runs on CUDA and MPS.
    assert_eq!(target.limits.get("appleSiliconOnly"), None);
}

#[test]
fn wan_target_resolves_image_dataset_into_plan() {
    // Like LTX, the Wan video target consumes the image dataset fixture unchanged.
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "wan_lora")
        .expect("wan_lora target present");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_wan",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_wan_new",
        base_model_path: "/models/wan".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_wan_new"),
        file_name: "wan_character.safetensors".to_owned(),
        created_at: "2026-05-26T00:00:00Z".to_owned(),
    })
    .expect("wan plan resolves");

    assert_eq!(plan.plan_version, TRAINING_PLAN_VERSION);
    assert_eq!(plan.target.kernel, "wan_lora");
    assert_eq!(plan.target.family, "wan-video");
    assert_eq!(plan.target.modality, TrainingModality::Video);
    assert!(!plan.dataset.items.is_empty());
}

#[test]
fn builtin_registry_exposes_wan_moe_targets() {
    let registry = builtin_training_targets();
    for (id, base_model) in [
        ("wan_t2v_14b_lora", "wan_2_2_t2v_14b"),
        ("wan_i2v_14b_lora", "wan_2_2_i2v_14b"),
    ] {
        let target = registry
            .targets
            .iter()
            .find(|target| target.id == id)
            .unwrap_or_else(|| panic!("{id} target present"));
        assert_eq!(target.modality, TrainingModality::Video);
        assert_eq!(target.output_kind, TrainingOutputKind::Lora);
        assert_eq!(target.family, "wan-video");
        assert_eq!(target.base_model, base_model);
        // Both A14B variants share the dual-expert MoE kernel.
        assert_eq!(target.kernel, "wan_moe_lora");
        assert_eq!(target.defaults.rank, 32);
        assert_eq!(target.defaults.resolution, 512);
    }
}

#[test]
fn unknown_training_fields_and_values_are_preserved() {
    let mut dataset = load_fixture("dataset.json");
    // Unknown enum value falls back to the string-enum `Unknown` variant.
    dataset["status"] = Value::String("curating".to_owned());
    // Unknown top-level and nested keys survive via flattened `extra` maps.
    dataset["futureField"] = Value::String("kept".to_owned());
    dataset["items"][0]["caption"]["futureCaptionField"] = Value::Bool(true);

    let typed: TrainingDataset =
        serde_json::from_value(dataset.clone()).expect("unknown dataset fields parse");
    let encoded = serde_json::to_value(typed).expect("unknown dataset fields serialize");

    assert_eq!(encoded, dataset);
}

#[test]
fn training_plan_fixture_pins_current_plan_version() {
    let plan = load_fixture("training-plan.json");
    assert_eq!(
        plan["planVersion"].as_u64(),
        Some(u64::from(TRAINING_PLAN_VERSION))
    );
}

fn dataset_fixture() -> TrainingDataset {
    serde_json::from_value(load_fixture("dataset.json")).expect("dataset fixture parses")
}

#[test]
fn build_training_plan_resolves_paths_ids_and_provenance() {
    let mut dataset = dataset_fixture();
    dataset.items[0].caption.text = "a portrait of woman, soft light".to_owned();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");
    let dataset_root = Path::new("/data/training/ds_abc123");
    let output_dir = Path::new("/data/loras/lora_new");
    let mut config = target.defaults.clone();
    config.trigger_word = Some("auroraStyle".to_owned());

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/cache/huggingface/Tongyi-MAI/Z-Image-Turbo".to_owned(),
        dataset_root,
        output_dir,
        file_name: "aurora_style.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect("plan resolves");

    assert_eq!(plan.plan_version, TRAINING_PLAN_VERSION);
    assert_eq!(plan.job_id, "job_test");
    // The plan is self-referential so the kernel never needs the job record.
    assert_eq!(plan.provenance.source_job_id, "job_test");
    assert_eq!(plan.target.target_id, target.id);
    assert_eq!(plan.target.kernel, target.kernel);
    assert_eq!(
        plan.target.base_model_path,
        "/data/cache/huggingface/Tongyi-MAI/Z-Image-Turbo"
    );
    assert_eq!(plan.dataset.dataset_id, "ds_abc123");
    assert_eq!(plan.dataset.dataset_version, 3);
    assert_eq!(plan.dataset.items.len(), 2);
    assert_eq!(
        plan.dataset.items[0].caption,
        "auroraStyle, a portrait of woman, soft light"
    );
    // Item paths resolve under the dataset root with the host separator.
    let mut expected_image = dataset_root.to_path_buf();
    for component in Path::new("images/001.png").components() {
        expected_image.push(component);
    }
    assert_eq!(
        plan.dataset.items[0].image_path,
        expected_image.display().to_string()
    );
    assert_eq!(plan.output.lora_id, "lora_new");
    assert_eq!(plan.output.file_name, "aurora_style.safetensors");
    assert_eq!(plan.output.trigger_words, vec!["auroraStyle".to_owned()]);
    assert_eq!(plan.provenance.output_lora_id, "lora_new");
    assert_eq!(plan.provenance.dataset_version, 3);
}

#[test]
fn build_training_plan_omits_trigger_words_when_unset() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");

    let plan = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_new"),
        file_name: "aurora.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect("plan resolves");

    assert!(plan.output.trigger_words.is_empty());
}

#[test]
fn build_training_plan_rejects_empty_dataset() {
    let mut dataset = dataset_fixture();
    dataset.items.clear();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");

    let error = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config: target.defaults.clone(),
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_new"),
        file_name: "aurora.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect_err("empty dataset is rejected");

    assert_eq!(error, TrainingPlanError::EmptyDataset);
}

/// Builds a plan from the Z-Image target defaults with the learning-rate
/// scheduler knobs overridden, so the validation tests exercise the same submit
/// path the API uses (`build_training_plan` → `validate_training_config`).
fn build_plan_with_lr_overrides(
    scheduler: Option<Value>,
    warmup: Option<Value>,
) -> Result<TrainingPlan, TrainingPlanError> {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == "z_image_turbo_lora")
        .expect("z_image_turbo_lora target present");
    let mut config = target.defaults.clone();
    match scheduler {
        Some(value) => {
            config.advanced.insert("lrScheduler".to_owned(), value);
        }
        None => {
            config.advanced.remove("lrScheduler");
        }
    }
    if let Some(value) = warmup {
        config.advanced.insert("lrWarmupSteps".to_owned(), value);
    }

    build_training_plan(BuildTrainingPlan {
        job_id: "job_lr",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_lr",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_lr"),
        file_name: "lr.safetensors".to_owned(),
        created_at: "2026-05-23T00:00:00Z".to_owned(),
    })
}

#[test]
fn build_training_plan_accepts_supported_lr_schedulers() {
    // Canonical names plus case/whitespace variants the validator normalizes.
    for name in ["constant", "linear", "cosine", "Cosine", " LINEAR "] {
        let plan = build_plan_with_lr_overrides(Some(Value::String(name.to_owned())), None)
            .unwrap_or_else(|error| panic!("scheduler {name:?} should be accepted: {error}"));
        // The submitted value is preserved verbatim in the resolved config; only
        // validation normalizes for the membership check.
        assert_eq!(
            plan.config.advanced["lrScheduler"],
            Value::String(name.to_owned())
        );
    }
}

#[test]
fn build_training_plan_rejects_unknown_lr_scheduler() {
    let error = build_plan_with_lr_overrides(Some(Value::String("warmup_cosine".to_owned())), None)
        .expect_err("unknown scheduler is rejected");
    match error {
        TrainingPlanError::InvalidConfig(detail) => {
            assert!(detail.contains("Unsupported lrScheduler"), "got: {detail}");
            assert!(
                detail.contains("warmup_cosine"),
                "names the bad value: {detail}"
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn build_training_plan_rejects_non_string_lr_scheduler() {
    let error = build_plan_with_lr_overrides(Some(Value::Bool(true)), None)
        .expect_err("non-string scheduler is rejected");
    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));
}

#[test]
fn build_training_plan_validates_lr_warmup_steps() {
    // A non-negative warmup shorter than the 3000-step run is accepted.
    let plan =
        build_plan_with_lr_overrides(Some(Value::String("cosine".to_owned())), Some(json!(100)))
            .expect("warmup within range is accepted");
    assert_eq!(plan.config.advanced["lrWarmupSteps"], json!(100));

    // Warmup at or beyond the total step count is rejected.
    let error =
        build_plan_with_lr_overrides(Some(Value::String("cosine".to_owned())), Some(json!(5000)))
            .expect_err("warmup beyond the run is rejected");
    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));

    // Negative (and otherwise non-integer) warmup is rejected.
    let error = build_plan_with_lr_overrides(None, Some(json!(-5)))
        .expect_err("negative warmup is rejected");
    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));
}

#[test]
fn build_training_plan_rejects_invalid_config() {
    let dataset = dataset_fixture();
    let registry = builtin_training_targets();
    let target = registry.targets.first().expect("a builtin target exists");
    let mut config = target.defaults.clone();
    config.rank = 0;

    let error = build_training_plan(BuildTrainingPlan {
        job_id: "job_test",
        target,
        dataset: &dataset,
        config,
        preset: None,
        lora_id: "lora_new",
        base_model_path: "/data/models/z_image_turbo".to_owned(),
        dataset_root: Path::new("/data/training/ds_abc123"),
        output_dir: Path::new("/data/loras/lora_new"),
        file_name: "aurora.safetensors".to_owned(),
        created_at: "2026-05-21T00:00:00Z".to_owned(),
    })
    .expect_err("zero rank is rejected");

    assert!(matches!(error, TrainingPlanError::InvalidConfig(_)));
}
