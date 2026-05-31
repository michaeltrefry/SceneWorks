use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use sceneworks_core::character_store::write_character_sidecar;
use sceneworks_core::project_store::{
    CharacterCreateInput, CharacterLookInput, CharacterLookUpdateInput, CharacterLoraInput,
    CharacterLoraUpdateInput, CharacterReferenceInput, CharacterReferenceUpdateInput, ProjectStore,
    UploadAsset,
};
use serde_json::{json, Map, Value};

fn fixture_path(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("rust_migration_contracts")
        .join(relative_path)
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).expect("json reads")).expect("json parses")
}

#[test]
fn character_sidecar_writer_byte_matches_fixture() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let fixture_path = fixture_path("sidecars/character.sceneworks.character.json");
    let expected = fs::read_to_string(&fixture_path)
        .expect("fixture reads")
        .replace("\r\n", "\n");
    let character: Value = serde_json::from_str(&expected).expect("fixture parses");

    write_character_sidecar(temp_dir.path(), &character).expect("sidecar writes");

    let actual = fs::read_to_string(
        temp_dir
            .path()
            .join("characters/character_fixture.sceneworks.character.json"),
    )
    .expect("written sidecar reads");
    assert_eq!(actual, expected);
}

#[test]
fn character_sidecar_writer_preserves_string_newlines_as_json_escapes() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let character = json!({
        "schemaVersion": 1,
        "id": "character_multiline",
        "projectId": "project_fixture",
        "name": "Mira",
        "type": "person",
        "description": "line one\nline two",
        "createdAt": "2026-05-17T13:00:00Z",
        "updatedAt": "2026-05-17T13:00:00Z",
        "status": { "archived": false },
        "references": [],
        "looks": [],
        "loras": [],
        "trainedLoras": []
    });

    write_character_sidecar(temp_dir.path(), &character).expect("sidecar writes");

    let bytes = fs::read(
        temp_dir
            .path()
            .join("characters/character_multiline.sceneworks.character.json"),
    )
    .expect("written sidecar reads");
    assert!(!bytes.contains(&b'\r'));
    assert!(String::from_utf8(bytes)
        .expect("utf8")
        .contains(r#""description": "line one\nline two""#));
}

#[test]
fn character_crud_updates_sidecars_and_project_index() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Characters").expect("project creates");
    let project_path = PathBuf::from(&project.path);

    let character = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Mira".to_owned(),
                character_type: "person".to_owned(),
                description: "Lead".to_owned(),
            },
        )
        .expect("character creates");
    let character_id = character["id"].as_str().expect("character id").to_owned();
    let character_path = project_path.join(format!(
        "characters/{character_id}.sceneworks.character.json"
    ));
    assert!(character_path.exists());

    let connection = Connection::open(project_path.join("project.db")).expect("db opens");
    let indexed_name: String = connection
        .query_row(
            "select name from characters where id = ?1",
            params![character_id],
            |row| row.get(0),
        )
        .expect("character indexed");
    assert_eq!(indexed_name, "Mira");

    store
        .archive_character(&project.id, &character_id)
        .expect("character archives");
    assert_eq!(
        store
            .list_characters(&project.id, false)
            .expect("characters list")
            .len(),
        0
    );
    assert_eq!(
        store
            .list_characters(&project.id, true)
            .expect("archived characters list")
            .len(),
        1
    );

    store
        .purge_character(&project.id, &character_id)
        .expect("character purges");
    let remaining: i64 = connection
        .query_row("select count(*) from characters", [], |row| row.get(0))
        .expect("count reads");
    assert_eq!(remaining, 0);
    assert!(!character_path.exists());
}

#[test]
fn references_sync_asset_metadata_and_character_reference_table() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("References").expect("project creates");
    let project_path = PathBuf::from(&project.path);
    let source = temp_dir.path().join("reference.png");
    fs::write(&source, b"png-bytes").expect("source writes");
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "reference.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("asset imports");
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();
    let character = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Mira".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("character creates");
    let character_id = character["id"].as_str().expect("character id").to_owned();

    store
        .add_character_reference(
            &project.id,
            &character_id,
            CharacterReferenceInput {
                asset_id: asset_id.clone(),
                approved: false,
                role: "reference".to_owned(),
                notes: "front".to_owned(),
            },
        )
        .expect("reference adds");
    store
        .update_character_reference(
            &project.id,
            &character_id,
            &asset_id,
            CharacterReferenceUpdateInput {
                approved: Some(true),
                role: Some("hero".to_owned()),
                notes: None,
            },
        )
        .expect("reference updates");

    let sidecar_path = project_path.join(
        asset["sidecarPath"]
            .as_str()
            .expect("sidecar path")
            .replace('/', std::path::MAIN_SEPARATOR_STR),
    );
    let asset_sidecar = read_json(&sidecar_path);
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"][0]["characterId"],
        character_id
    );
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"][0]["approved"],
        true
    );

    let connection = Connection::open(project_path.join("project.db")).expect("db opens");
    let approved: i64 = connection
        .query_row(
            "select approved from character_references where character_id = ?1 and asset_id = ?2",
            params![character_id, asset_id],
            |row| row.get(0),
        )
        .expect("reference indexed");
    assert_eq!(approved, 1);

    store
        .remove_character_reference(&project.id, &character_id, &asset_id)
        .expect("reference removes");
    let asset_sidecar = read_json(&sidecar_path);
    assert_eq!(
        asset_sidecar["metadata"]["characterReferences"]
            .as_array()
            .expect("metadata links"),
        &Vec::<Value>::new()
    );
}

#[test]
fn looks_loras_and_reindex_are_persisted() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Looks").expect("project creates");
    let project_path = PathBuf::from(&project.path);
    let character = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Mira".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("character creates");
    let character_id = character["id"].as_str().expect("character id").to_owned();

    let with_look = store
        .create_character_look(
            &project.id,
            &character_id,
            CharacterLookInput {
                name: "Rain coat".to_owned(),
                description: "Wet hair".to_owned(),
                approved_reference_ids: vec!["asset_1".to_owned()],
                recipe_settings: Map::from_iter([("style".to_owned(), json!("noir"))]),
            },
        )
        .expect("look creates");
    let look_id = with_look["looks"][0]["id"]
        .as_str()
        .expect("look id")
        .to_owned();
    let updated = store
        .update_character_look(
            &project.id,
            &character_id,
            &look_id,
            CharacterLookUpdateInput {
                name: Some("Blue coat".to_owned()),
                ..CharacterLookUpdateInput::default()
            },
        )
        .expect("look updates");
    assert_eq!(updated["looks"][0]["name"], "Blue coat");
    let without_look = store
        .delete_character_look(&project.id, &character_id, &look_id)
        .expect("look deletes");
    assert_eq!(without_look["looks"].as_array().expect("looks").len(), 0);

    let lora_dir = temp_dir.path().join("data/loras");
    fs::create_dir_all(&lora_dir).expect("lora dir creates");
    let lora_source = lora_dir.join("mira.safetensors");
    fs::write(&lora_source, b"lora").expect("lora source writes");
    let with_lora = store
        .attach_character_lora(
            &project.id,
            &character_id,
            CharacterLoraInput {
                lora_id: Some("lora_fixture".to_owned()),
                name: "Mira LoRA".to_owned(),
                source_path: Some(lora_source.display().to_string()),
                trigger_words: vec!["mira".to_owned()],
                default_weight: 0.8,
                compatibility: Map::from_iter([("families".to_owned(), json!(["z-image"]))]),
                scope: "project".to_owned(),
            },
        )
        .expect("lora attaches");
    let link_id = with_lora["loras"][0]["id"]
        .as_str()
        .expect("link id")
        .to_owned();
    let project_lora_path = project_path.join(
        with_lora["loras"][0]["projectPath"]
            .as_str()
            .expect("project lora path")
            .replace('/', std::path::MAIN_SEPARATOR_STR),
    );
    assert_eq!(fs::read(project_lora_path).expect("lora copied"), b"lora");
    let with_updated_lora = store
        .update_character_lora(
            &project.id,
            &character_id,
            &link_id,
            CharacterLoraUpdateInput {
                default_weight: Some(-0.5),
                ..CharacterLoraUpdateInput::default()
            },
        )
        .expect("lora updates");
    assert_eq!(with_updated_lora["loras"][0]["defaultWeight"], -0.5);
    let without_lora = store
        .detach_character_lora(&project.id, &character_id, &link_id)
        .expect("lora detaches");
    assert_eq!(without_lora["loras"].as_array().expect("loras").len(), 0);

    store.reindex_project(&project.id).expect("reindex works");
    let connection = Connection::open(project_path.join("project.db")).expect("db opens");
    let count: i64 = connection
        .query_row("select count(*) from characters", [], |row| row.get(0))
        .expect("characters count reads");
    assert_eq!(count, 1);
}

#[test]
fn list_characters_recovers_stale_character_index_from_sidecars() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Recovery").expect("project creates");
    let project_path = PathBuf::from(&project.path);
    let character = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Original".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("character creates");
    let character_id = character["id"].as_str().expect("character id").to_owned();
    let sidecar_path = project_path.join(format!(
        "characters/{character_id}.sceneworks.character.json"
    ));
    let mut sidecar = read_json(&sidecar_path);
    sidecar["name"] = json!("Recovered");
    sidecar["status"]["archived"] = json!(true);
    write_character_sidecar(&project_path, &sidecar).expect("sidecar rewrites");

    let connection = Connection::open(project_path.join("project.db")).expect("db opens");
    connection
        .execute(
            "update characters set name = 'Stale', archived = 0 where id = ?1",
            params![character_id],
        )
        .expect("index made stale");

    assert_eq!(
        store
            .list_characters(&project.id, false)
            .expect("visible list")
            .len(),
        0
    );
    let archived = store
        .list_characters(&project.id, true)
        .expect("archived list");
    assert_eq!(archived[0]["name"], "Recovered");

    let indexed: (String, i64) = connection
        .query_row(
            "select name, archived from characters where id = ?1",
            params![character_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("index refreshed");
    assert_eq!(indexed, ("Recovered".to_owned(), 1));
}
