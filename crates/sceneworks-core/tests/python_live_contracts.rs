#![cfg(feature = "python-live")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sceneworks_core::contracts::{
    Asset, Character, GenerationSet, JobSnapshot, LoraManifestEntry, ModelInstallMarker,
    ModelManifestEntry, PersonTrack, Project, QueueSummary, Recipe, Timeline,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn live_contract_dir() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("sceneworks-live-contracts-{}", std::process::id()));
    path
}

fn python_command() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python".to_owned())
}

fn export_live_contracts(output_dir: &Path) {
    let root = repo_root();
    if output_dir.exists() {
        fs::remove_dir_all(output_dir).expect("remove previous live contract output");
    }
    let status = Command::new(python_command())
        .current_dir(&root)
        .arg(root.join("scripts").join("export-rust-live-contracts.py"))
        .arg(output_dir)
        .status()
        .expect("run Python live contract exporter");

    assert!(status.success(), "Python live contract exporter failed");
}

fn load_live(output_dir: &Path, filename: &str) -> Value {
    let path = output_dir.join(filename);
    let payload = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&payload)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

fn assert_live_round_trip<T>(output_dir: &Path, filename: &str)
where
    T: DeserializeOwned + Serialize,
{
    let original = load_live(output_dir, filename);
    let typed: T = serde_json::from_value(original.clone())
        .unwrap_or_else(|error| panic!("failed to deserialize live {filename}: {error}"));
    let encoded = serde_json::to_value(typed)
        .unwrap_or_else(|error| panic!("failed to serialize live {filename}: {error}"));

    assert_eq!(
        encoded, original,
        "live {filename} drifted after typed round-trip"
    );
}

#[test]
fn python_live_payloads_round_trip_through_rust_contracts() {
    let output_dir = live_contract_dir();
    export_live_contracts(&output_dir);

    assert_live_round_trip::<Project>(&output_dir, "project.json");
    assert_live_round_trip::<Asset>(&output_dir, "asset-image.sceneworks.json");
    assert_live_round_trip::<Asset>(&output_dir, "asset-video.sceneworks.json");
    assert_live_round_trip::<GenerationSet>(&output_dir, "generation-set.json");
    assert_live_round_trip::<Recipe>(&output_dir, "recipe.json");
    assert_live_round_trip::<Character>(&output_dir, "character.sceneworks.character.json");
    assert_live_round_trip::<Timeline>(&output_dir, "timeline.sceneworks.timeline.json");
    assert_live_round_trip::<PersonTrack>(&output_dir, "person-track.sceneworks.person-track.json");
    assert_live_round_trip::<ModelManifestEntry>(&output_dir, "model-manifest-entry.json");
    assert_live_round_trip::<LoraManifestEntry>(&output_dir, "lora-manifest-entry.json");
    assert_live_round_trip::<ModelInstallMarker>(&output_dir, "model-install-marker.json");
    assert_live_round_trip::<JobSnapshot>(&output_dir, "job-snapshot.json");
    assert_live_round_trip::<QueueSummary>(&output_dir, "queue-summary.json");

    let person_track = load_live(&output_dir, "person-track.sceneworks.person-track.json");
    assert!(person_track["frames"][0].get("timestamp").is_some());
    assert!(person_track["frames"][0].get("time").is_none());
}
