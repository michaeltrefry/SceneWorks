use std::fs;
use std::path::PathBuf;

use sceneworks_core::contracts::{
    Asset, Character, GenerationSet, JobProtocolFixture, LoraManifest, LoraManifestEntry,
    ModelInstallMarker, ModelManifest, ModelManifestEntry, PersonTrack, Project, QueueSummary,
    Recipe, ResourceSidecarsFixture, Timeline,
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
fn job_protocol_fixture_round_trips_without_field_drift() {
    assert_round_trip::<JobProtocolFixture>("job_protocol.json");
}

#[test]
fn resource_sidecar_index_round_trips_without_field_drift() {
    assert_round_trip::<ResourceSidecarsFixture>("resource_sidecars.json");
}

#[test]
fn persisted_sidecars_round_trip_without_field_drift() {
    assert_round_trip::<Project>("sidecars/project.json");
    assert_round_trip::<Asset>("sidecars/asset-image.sceneworks.json");
    assert_round_trip::<Asset>("sidecars/asset-video.sceneworks.json");
    assert_round_trip::<GenerationSet>("sidecars/generation-set.json");
    assert_round_trip::<Recipe>("sidecars/recipe.json");
    assert_round_trip::<Character>("sidecars/character.sceneworks.character.json");
    assert_round_trip::<Timeline>("sidecars/timeline.sceneworks.timeline.json");
    assert_round_trip::<PersonTrack>("sidecars/person-track.sceneworks.person-track.json");
    assert_round_trip::<ModelManifestEntry>("sidecars/model-manifest-entry.json");
    assert_round_trip::<LoraManifestEntry>("sidecars/lora-manifest-entry.json");
    assert_round_trip::<ModelInstallMarker>("sidecars/model-install-marker.json");
}

#[test]
fn unknown_contract_values_and_fields_are_preserved() {
    let mut asset = load_fixture("sidecars/asset-image.sceneworks.json");
    asset["type"] = Value::String("depth_map".to_owned());
    asset["futureField"] = Value::String("kept".to_owned());
    asset["file"]["codec"] = Value::String("png".to_owned());

    let typed: Asset = serde_json::from_value(asset.clone()).expect("unknown asset fields parse");
    let encoded = serde_json::to_value(typed).expect("unknown asset fields serialize");

    assert_eq!(encoded, asset);
}

#[test]
fn queue_summary_contract_round_trips_known_shapes() {
    let job_protocol = load_fixture("job_protocol.json");
    let job_snapshot = job_protocol["jobSnapshot"].clone();
    let queue = json!({
        "counts": {
            "queued": 1,
            "running": 1,
            "completed": 0,
            "future_status": 2
        },
        "activeJobs": [job_snapshot],
        "workers": [{
            "id": "worker-gpu-0",
            "gpuId": "gpu-0",
            "gpuName": "Fixture GPU",
            "status": "busy",
            "currentJobId": "job_fixture",
            "capabilities": ["placeholder", "gpu", "image_generate"],
            "loadedModels": ["Tongyi-MAI/Z-Image-Turbo"],
            "registeredAt": "2026-05-17T13:00:00Z",
            "lastSeenAt": "2026-05-17T13:00:04Z",
            "futureWorkerField": true
        }],
        "maxJobAttempts": 5,
        "futureQueueField": "kept"
    });

    let typed: QueueSummary = serde_json::from_value(queue.clone()).expect("queue summary parses");
    let encoded = serde_json::to_value(typed).expect("queue summary serializes");

    assert_eq!(encoded, queue);
}

#[test]
fn manifest_wrappers_round_trip_entries() {
    let model_entry = load_fixture("sidecars/model-manifest-entry.json");
    let lora_entry = load_fixture("sidecars/lora-manifest-entry.json");
    let models = json!({ "schemaVersion": 1, "models": [model_entry], "futureRoot": true });
    let loras = json!({ "schemaVersion": 1, "loras": [lora_entry], "futureRoot": true });

    let typed_models: ModelManifest =
        serde_json::from_value(models.clone()).expect("model manifest parses");
    let typed_loras: LoraManifest =
        serde_json::from_value(loras.clone()).expect("lora manifest parses");

    assert_eq!(
        serde_json::to_value(typed_models).expect("model manifest serializes"),
        models
    );
    assert_eq!(
        serde_json::to_value(typed_loras).expect("lora manifest serializes"),
        loras
    );
}
