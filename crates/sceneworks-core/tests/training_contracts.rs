use std::fs;
use std::path::PathBuf;

use sceneworks_core::training::{
    builtin_training_targets, LoraTrainingRequest, TrainingConfig, TrainingDataset,
    TrainingModality, TrainingOutputKind, TrainingPlan, TrainingProvenance, TrainingTargetRegistry,
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
