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
use serde_json::Value;

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
