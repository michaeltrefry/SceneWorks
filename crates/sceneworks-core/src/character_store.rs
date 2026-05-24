use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::asset_index::{asset_sidecars, normalize_asset, row_to_asset_record, upsert_asset_row};
use crate::project_store::{apply_project_migrations, ProjectStoreError, ProjectStoreResult};
use crate::store_util::{
    optional_bool, optional_f64, optional_str, random_hex, read_json, relative_string, write_json,
};
use crate::time::utc_now;

pub const CHARACTER_SIDECAR_PATTERN: &str = ".sceneworks.character.json";

const CHARACTER_INDEX_FINGERPRINT_KEY: &str = "characterIndexFingerprint";

#[derive(Debug, Clone)]
pub struct CharacterCreateInput {
    pub name: String,
    pub character_type: String,
    pub description: String,
}

#[derive(Debug, Default, Clone)]
pub struct CharacterUpdateInput {
    pub name: Option<String>,
    pub character_type: Option<String>,
    pub description: Option<String>,
    pub archived: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct CharacterReferenceInput {
    pub asset_id: String,
    pub approved: bool,
    pub role: String,
    pub notes: String,
}

#[derive(Debug, Default, Clone)]
pub struct CharacterReferenceUpdateInput {
    pub approved: Option<bool>,
    pub role: Option<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CharacterLookInput {
    pub name: String,
    pub description: String,
    pub approved_reference_ids: Vec<String>,
    pub recipe_settings: Map<String, Value>,
}

#[derive(Debug, Default, Clone)]
pub struct CharacterLookUpdateInput {
    pub name: Option<String>,
    pub description: Option<String>,
    pub approved_reference_ids: Option<Vec<String>>,
    pub recipe_settings: Option<Map<String, Value>>,
}

#[derive(Debug, Clone)]
pub struct CharacterLoraInput {
    pub lora_id: Option<String>,
    pub name: String,
    pub source_path: Option<String>,
    pub trigger_words: Vec<String>,
    pub default_weight: f64,
    pub compatibility: Map<String, Value>,
    pub scope: String,
}

#[derive(Debug, Default, Clone)]
pub struct CharacterLoraUpdateInput {
    pub name: Option<String>,
    pub trigger_words: Option<Vec<String>>,
    pub default_weight: Option<f64>,
    pub compatibility: Option<Map<String, Value>>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharacterMutationResult {
    pub id: String,
    pub status: String,
}

#[derive(Debug)]
pub struct CharacterStore<'a> {
    data_dir: &'a Path,
    project_path: PathBuf,
}

impl<'a> CharacterStore<'a> {
    pub fn new(data_dir: &'a Path, project_path: impl Into<PathBuf>) -> Self {
        Self {
            data_dir,
            project_path: project_path.into(),
        }
    }

    pub fn list_characters(
        &self,
        project_id: &str,
        include_archived: bool,
    ) -> ProjectStoreResult<Vec<Value>> {
        ensure_character_index(&self.project_path)?;
        let connection = connect_project_db(&self.project_path)?;
        let mut statement = connection.prepare("select sidecar_path from characters")?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;

        let mut characters = Vec::new();
        for sidecar_rel in rows {
            let sidecar_path = self.project_path.join(sidecar_rel);
            if !sidecar_path.exists() {
                continue;
            }
            let Ok(character) = read_json(&sidecar_path) else {
                continue;
            };
            if character
                .pointer("/status/archived")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                && !include_archived
            {
                continue;
            }
            characters.push(hydrate_character(
                project_id,
                &self.project_path,
                character,
            )?);
        }
        characters.sort_by(|left, right| {
            right
                .get("updatedAt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    left.get("updatedAt")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        });
        Ok(characters)
    }

    pub fn get_character(&self, project_id: &str, character_id: &str) -> ProjectStoreResult<Value> {
        let character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn create_character(
        &self,
        project_id: &str,
        input: CharacterCreateInput,
    ) -> ProjectStoreResult<Value> {
        validate_character_name(&input.name)?;
        validate_character_type(&input.character_type)?;
        validate_text_length(&input.description, "description", 2000)?;

        let now = utc_now();
        let character_id = format!("character_{}", random_hex(16)?);
        let mut character = json!({
            "schemaVersion": 1,
            "id": character_id,
            "projectId": project_id,
            "name": input.name.trim(),
            "type": input.character_type,
            "description": input.description.trim(),
            "createdAt": now,
            "updatedAt": now,
            "status": { "archived": false },
            "references": [],
            "looks": [],
            "loras": [],
            "trainedLoras": []
        });
        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, false)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn update_character(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let object = value_object_mut(&mut character, "Character sidecar")?;
        if let Some(name) = input.name {
            validate_character_name(&name)?;
            object.insert("name".to_owned(), Value::String(name.trim().to_owned()));
        }
        if let Some(character_type) = input.character_type {
            validate_character_type(&character_type)?;
            object.insert("type".to_owned(), Value::String(character_type));
        }
        if let Some(description) = input.description {
            validate_text_length(&description, "description", 2000)?;
            object.insert(
                "description".to_owned(),
                Value::String(description.trim().to_owned()),
            );
        }
        if let Some(archived) = input.archived {
            object
                .entry("status".to_owned())
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| {
                    ProjectStoreError::BadRequest("Character status must be an object".to_owned())
                })?
                .insert("archived".to_owned(), Value::Bool(archived));
        }

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn archive_character(
        &self,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        value_object_mut(&mut character, "Character sidecar")?
            .entry("status".to_owned())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Character status must be an object".to_owned())
            })?
            .insert("archived".to_owned(), Value::Bool(true));

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        Ok(CharacterMutationResult {
            id: character_id.to_owned(),
            status: "archived".to_owned(),
        })
    }

    pub fn purge_character(
        &self,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let path = find_character_file(&self.project_path, character_id)?;
        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        purge_character_on_connection(&transaction, character_id)?;
        fs::remove_file(path)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        Ok(CharacterMutationResult {
            id: character_id.to_owned(),
            status: "purged".to_owned(),
        })
    }

    pub fn add_reference(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterReferenceInput,
    ) -> ProjectStoreResult<Value> {
        validate_required_text(&input.asset_id, "assetId")?;
        validate_text_length(&input.role, "role", 80)?;
        validate_text_length(&input.notes, "notes", 1000)?;
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let now = utc_now();
        let reference = json!({
            "assetId": input.asset_id,
            "approved": input.approved,
            "role": if input.role.trim().is_empty() { "reference" } else { input.role.trim() },
            "notes": input.notes,
            "addedAt": now,
            "approvedAt": if input.approved { Value::String(now) } else { Value::Null }
        });

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        update_asset_character_link(
            &transaction,
            &self.project_path,
            character_id,
            &reference,
            false,
        )?;
        let object = value_object_mut(&mut character, "Character sidecar")?;
        let mut references = object
            .remove("references")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter(|item| {
                item.get("assetId")
                    .and_then(Value::as_str)
                    .is_some_and(|asset_id| asset_id != input.asset_id)
            })
            .collect::<Vec<_>>();
        references.insert(0, reference);
        object.insert("references".to_owned(), Value::Array(references));
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn update_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
        input: CharacterReferenceUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let references = character
            .get_mut("references")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Character references must be an array".to_owned())
            })?;
        let reference = references
            .iter_mut()
            .find(|item| item.get("assetId").and_then(Value::as_str) == Some(asset_id))
            .ok_or_else(|| ProjectStoreError::NotFound("Reference not found".to_owned()))?;
        if let Some(approved) = input.approved {
            value_object_mut(reference, "Character reference")?
                .insert("approved".to_owned(), Value::Bool(approved));
            value_object_mut(reference, "Character reference")?.insert(
                "approvedAt".to_owned(),
                if approved {
                    Value::String(utc_now())
                } else {
                    Value::Null
                },
            );
        }
        if let Some(role) = input.role {
            validate_text_length(&role, "role", 80)?;
            value_object_mut(reference, "Character reference")?.insert(
                "role".to_owned(),
                Value::String(if role.trim().is_empty() {
                    "reference".to_owned()
                } else {
                    role
                }),
            );
        }
        if let Some(notes) = input.notes {
            validate_text_length(&notes, "notes", 1000)?;
            value_object_mut(reference, "Character reference")?
                .insert("notes".to_owned(), Value::String(notes));
        }

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        update_asset_character_link(
            &transaction,
            &self.project_path,
            character_id,
            reference,
            false,
        )?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn remove_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let references = character
            .get("references")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let reference = references
            .iter()
            .find(|item| item.get("assetId").and_then(Value::as_str) == Some(asset_id))
            .ok_or_else(|| ProjectStoreError::NotFound("Reference not found".to_owned()))?;

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        update_asset_character_link(
            &transaction,
            &self.project_path,
            character_id,
            reference,
            true,
        )?;
        value_object_mut(&mut character, "Character sidecar")?.insert(
            "references".to_owned(),
            Value::Array(
                references
                    .into_iter()
                    .filter(|item| item.get("assetId").and_then(Value::as_str) != Some(asset_id))
                    .collect(),
            ),
        );
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn create_look(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLookInput,
    ) -> ProjectStoreResult<Value> {
        validate_character_name(&input.name)?;
        validate_text_length(&input.description, "description", 1000)?;
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let now = utc_now();
        let look = json!({
            "id": format!("look_{}", random_hex(16)?),
            "name": input.name.trim(),
            "description": input.description.trim(),
            "approvedReferenceIds": input.approved_reference_ids,
            "recipeSettings": input.recipe_settings,
            "createdAt": now,
            "updatedAt": now
        });
        prepend_array_field(&mut character, "looks", look)?;

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn update_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
        input: CharacterLookUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let looks = character
            .get_mut("looks")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Character looks must be an array".to_owned())
            })?;
        let look = looks
            .iter_mut()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(look_id))
            .ok_or_else(|| ProjectStoreError::NotFound("Look not found".to_owned()))?;
        let object = value_object_mut(look, "Character look")?;
        if let Some(name) = input.name {
            validate_character_name(&name)?;
            object.insert("name".to_owned(), Value::String(name.trim().to_owned()));
        }
        if let Some(description) = input.description {
            validate_text_length(&description, "description", 1000)?;
            object.insert(
                "description".to_owned(),
                Value::String(description.trim().to_owned()),
            );
        }
        if let Some(approved_reference_ids) = input.approved_reference_ids {
            object.insert(
                "approvedReferenceIds".to_owned(),
                json!(approved_reference_ids),
            );
        }
        if let Some(recipe_settings) = input.recipe_settings {
            object.insert("recipeSettings".to_owned(), Value::Object(recipe_settings));
        }
        object.insert("updatedAt".to_owned(), Value::String(utc_now()));

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn delete_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let looks = character
            .get("looks")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        value_object_mut(&mut character, "Character sidecar")?.insert(
            "looks".to_owned(),
            Value::Array(
                looks
                    .into_iter()
                    .filter(|item| item.get("id").and_then(Value::as_str) != Some(look_id))
                    .collect(),
            ),
        );

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn attach_lora(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLoraInput,
    ) -> ProjectStoreResult<Value> {
        validate_character_name(&input.name)?;
        validate_lora_weight(input.default_weight)?;
        validate_lora_scope(&input.scope, true)?;
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let (project_lora_path, copied) = copy_lora_into_project(
            &self.project_path,
            self.data_dir,
            character_id,
            input.source_path.as_deref(),
        )?;
        let now = utc_now();
        let link = json!({
            "id": format!("character_lora_{}", random_hex(16)?),
            "loraId": input.lora_id,
            "name": input.name.trim(),
            "sourcePath": input.source_path,
            "projectPath": project_lora_path,
            "copiedIntoProject": copied,
            "category": "character",
            "scope": input.scope,
            "triggerWords": input.trigger_words,
            "defaultWeight": input.default_weight,
            "compatibility": input.compatibility,
            "createdAt": now,
            "updatedAt": now
        });
        prepend_array_field(&mut character, "loras", link)?;

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn update_lora_link(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
        input: CharacterLoraUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let loras = character
            .get_mut("loras")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Character LoRAs must be an array".to_owned())
            })?;
        let lora = loras
            .iter_mut()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(link_id))
            .ok_or_else(|| ProjectStoreError::NotFound("Character LoRA not found".to_owned()))?;
        let object = value_object_mut(lora, "Character LoRA")?;
        if let Some(name) = input.name {
            validate_character_name(&name)?;
            object.insert("name".to_owned(), Value::String(name.trim().to_owned()));
        }
        if let Some(trigger_words) = input.trigger_words {
            object.insert("triggerWords".to_owned(), json!(trigger_words));
        }
        if let Some(default_weight) = input.default_weight {
            validate_lora_weight(default_weight)?;
            object.insert("defaultWeight".to_owned(), json!(default_weight));
        }
        if let Some(compatibility) = input.compatibility {
            object.insert("compatibility".to_owned(), Value::Object(compatibility));
        }
        if let Some(scope) = input.scope {
            validate_lora_scope(&scope, true)?;
            object.insert("scope".to_owned(), Value::String(scope));
        }
        object.insert("updatedAt".to_owned(), Value::String(utc_now()));

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }

    pub fn detach_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
    ) -> ProjectStoreResult<Value> {
        let mut character = read_json(&find_character_file(&self.project_path, character_id)?)?;
        let loras = character
            .get("loras")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        value_object_mut(&mut character, "Character sidecar")?.insert(
            "loras".to_owned(),
            Value::Array(
                loras
                    .into_iter()
                    .filter(|item| item.get("id").and_then(Value::as_str) != Some(link_id))
                    .collect(),
            ),
        );

        let mut connection = connect_project_db(&self.project_path)?;
        let transaction = connection.transaction()?;
        write_character(&self.project_path, &mut character, true)?;
        index_character_on_connection(&transaction, &self.project_path, &character)?;
        store_character_index_fingerprint(&transaction, &self.project_path)?;
        transaction.commit()?;

        hydrate_character(project_id, &self.project_path, character)
    }
}

pub fn apply_character_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    connection.execute_batch(
        "
        create table if not exists characters (
          id text primary key,
          project_id text not null,
          name text not null,
          type text not null,
          description text not null default '',
          sidecar_path text not null,
          created_at text not null,
          updated_at text not null,
          archived integer not null default 0
        );
        create table if not exists character_references (
          character_id text not null,
          asset_id text not null,
          approved integer not null default 0,
          role text not null default 'reference',
          notes text not null default '',
          added_at text not null,
          approved_at text,
          primary key (character_id, asset_id)
        );
        create table if not exists character_looks (
          id text primary key,
          character_id text not null,
          name text not null,
          description text not null default '',
          approved_reference_ids text not null default '[]',
          recipe_settings text not null default '{}',
          created_at text not null,
          updated_at text not null
        );
        create table if not exists character_loras (
          id text primary key,
          character_id text not null,
          lora_id text,
          name text not null,
          source_path text,
          project_path text,
          copied_into_project integer not null default 0,
          category text not null default 'character',
          scope text not null default 'project',
          trigger_words text not null default '[]',
          default_weight real not null default 1.0,
          compatibility text not null default '{}',
          created_at text not null,
          updated_at text not null
        );
        ",
    )?;
    Ok(())
}

pub fn clear_character_tables(connection: &Connection) -> ProjectStoreResult<()> {
    connection.execute("delete from character_loras", [])?;
    connection.execute("delete from character_looks", [])?;
    connection.execute("delete from character_references", [])?;
    connection.execute("delete from characters", [])?;
    Ok(())
}

pub fn reindex_characters_on_connection(
    connection: &Connection,
    project_path: &Path,
) -> ProjectStoreResult<u32> {
    let mut count = 0;
    for sidecar_path in character_sidecars(project_path)? {
        let Ok(character) = read_json(&sidecar_path) else {
            continue;
        };
        if character.get("id").is_none() {
            continue;
        }
        index_character_sidecar_on_connection(connection, project_path, &sidecar_path, &character)?;
        count += 1;
    }
    store_character_index_fingerprint(connection, project_path)?;
    Ok(count)
}

pub fn write_character_sidecar(project_path: &Path, character: &Value) -> ProjectStoreResult<()> {
    let character_id = required_str(character, "id")?.to_owned();
    write_character_json(&character_file(project_path, &character_id), character)
}

fn ensure_character_index(project_path: &Path) -> ProjectStoreResult<()> {
    let (fingerprint, sidecar_count) = character_index_fingerprint(project_path)?;
    let mut connection = connect_project_db(project_path)?;
    let indexed_count = connection.query_row("select count(*) from characters", [], |row| {
        row.get::<_, u64>(0)
    })?;
    let stored_fingerprint = connection
        .query_row(
            "select value from project_metadata where key = ?1",
            params![CHARACTER_INDEX_FINGERPRINT_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if stored_fingerprint.as_deref() == Some(fingerprint.as_str()) && indexed_count == sidecar_count
    {
        return Ok(());
    }

    // Sidecars are authoritative. If a previous commit failed after an atomic sidecar write,
    // or files changed outside the DB path, rebuild only when the sidecar fingerprint changes.
    let transaction = connection.transaction()?;
    clear_character_tables(&transaction)?;
    reindex_characters_on_connection(&transaction, project_path)?;
    transaction.commit()?;
    Ok(())
}

fn store_character_index_fingerprint(
    connection: &Connection,
    project_path: &Path,
) -> ProjectStoreResult<()> {
    let (fingerprint, _) = character_index_fingerprint(project_path)?;
    connection.execute(
        "insert or replace into project_metadata (key, value) values (?1, ?2)",
        params![CHARACTER_INDEX_FINGERPRINT_KEY, fingerprint],
    )?;
    Ok(())
}

fn character_index_fingerprint(project_path: &Path) -> ProjectStoreResult<(String, u64)> {
    let mut entries = Vec::new();
    for path in character_sidecars(project_path)? {
        let metadata = fs::metadata(&path)?;
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        entries.push(format!(
            "{}:{}:{}",
            relative_string(project_path, &path)?,
            metadata.len(),
            modified_ns
        ));
    }
    entries.sort();
    Ok((entries.join("|"), entries.len() as u64))
}

fn index_character_on_connection(
    connection: &Connection,
    project_path: &Path,
    character: &Value,
) -> ProjectStoreResult<()> {
    let sidecar_path = character_file(project_path, required_str(character, "id")?);
    index_character_sidecar_on_connection(connection, project_path, &sidecar_path, character)
}

fn index_character_sidecar_on_connection(
    connection: &Connection,
    project_path: &Path,
    sidecar_path: &Path,
    character: &Value,
) -> ProjectStoreResult<()> {
    let character_id = required_str(character, "id")?;
    purge_character_on_connection(connection, character_id)?;
    let sidecar_rel = relative_string(project_path, sidecar_path)?;
    connection.execute(
        "
        insert or replace into characters (
          id, project_id, name, type, description, sidecar_path, created_at, updated_at, archived
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ",
        params![
            character_id,
            required_str(character, "projectId")?,
            required_str(character, "name")?,
            required_str(character, "type")?,
            optional_str(character, "description").unwrap_or(""),
            sidecar_rel,
            required_str(character, "createdAt")?,
            optional_str(character, "updatedAt")
                .or_else(|| optional_str(character, "createdAt"))
                .unwrap_or(""),
            optional_bool(character.get("status").unwrap_or(&Value::Null), "archived")
                .unwrap_or(false),
        ],
    )?;

    for reference in character
        .get("references")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        connection.execute(
            "
            insert or replace into character_references (
              character_id, asset_id, approved, role, notes, added_at, approved_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            params![
                character_id,
                required_str(&reference, "assetId")?,
                optional_bool(&reference, "approved").unwrap_or(false),
                optional_str(&reference, "role").unwrap_or("reference"),
                optional_str(&reference, "notes").unwrap_or(""),
                optional_str(&reference, "addedAt").unwrap_or(""),
                optional_str(&reference, "approvedAt"),
            ],
        )?;
    }

    for look in character
        .get("looks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        connection.execute(
            "
            insert or replace into character_looks (
              id, character_id, name, description, approved_reference_ids, recipe_settings,
              created_at, updated_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                required_str(&look, "id")?,
                character_id,
                required_str(&look, "name")?,
                optional_str(&look, "description").unwrap_or(""),
                to_json_string(look.get("approvedReferenceIds").unwrap_or(&json!([])))?,
                to_json_string(look.get("recipeSettings").unwrap_or(&json!({})))?,
                optional_str(&look, "createdAt").unwrap_or(""),
                optional_str(&look, "updatedAt")
                    .or_else(|| optional_str(&look, "createdAt"))
                    .unwrap_or(""),
            ],
        )?;
    }

    for lora in character
        .get("loras")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        connection.execute(
            "
            insert or replace into character_loras (
              id, character_id, lora_id, name, source_path, project_path, copied_into_project,
              category, scope, trigger_words, default_weight, compatibility, created_at, updated_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            ",
            params![
                required_str(&lora, "id")?,
                character_id,
                optional_str(&lora, "loraId"),
                required_str(&lora, "name")?,
                optional_str(&lora, "sourcePath"),
                optional_str(&lora, "projectPath"),
                optional_bool(&lora, "copiedIntoProject").unwrap_or(false),
                optional_str(&lora, "category").unwrap_or("character"),
                optional_str(&lora, "scope").unwrap_or("project"),
                to_json_string(lora.get("triggerWords").unwrap_or(&json!([])))?,
                optional_f64(&lora, "defaultWeight").unwrap_or(1.0),
                to_json_string(lora.get("compatibility").unwrap_or(&json!({})))?,
                optional_str(&lora, "createdAt").unwrap_or(""),
                optional_str(&lora, "updatedAt")
                    .or_else(|| optional_str(&lora, "createdAt"))
                    .unwrap_or(""),
            ],
        )?;
    }
    Ok(())
}

fn purge_character_on_connection(
    connection: &Connection,
    character_id: &str,
) -> ProjectStoreResult<()> {
    connection.execute(
        "delete from character_references where character_id = ?1",
        params![character_id],
    )?;
    connection.execute(
        "delete from character_looks where character_id = ?1",
        params![character_id],
    )?;
    connection.execute(
        "delete from character_loras where character_id = ?1",
        params![character_id],
    )?;
    connection.execute(
        "delete from characters where id = ?1",
        params![character_id],
    )?;
    Ok(())
}

fn connect_project_db(project_path: &Path) -> ProjectStoreResult<Connection> {
    fs::create_dir_all(project_path)?;
    let connection = Connection::open(project_path.join("project.db"))?;
    apply_project_migrations(&connection)?;
    Ok(connection)
}

fn character_sidecars(project_path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    let character_dir = project_path.join("characters");
    let mut sidecars = Vec::new();
    if !character_dir.exists() {
        return Ok(sidecars);
    }
    for entry in fs::read_dir(character_dir)? {
        let path = entry?.path();
        if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(CHARACTER_SIDECAR_PATTERN))
        {
            sidecars.push(path);
        }
    }
    Ok(sidecars)
}

fn character_file(project_path: &Path, character_id: &str) -> PathBuf {
    project_path
        .join("characters")
        .join(format!("{character_id}.sceneworks.character.json"))
}

fn find_character_file(project_path: &Path, character_id: &str) -> ProjectStoreResult<PathBuf> {
    let path = character_file(project_path, character_id);
    if path.exists() {
        return Ok(path);
    }
    for candidate in character_sidecars(project_path)? {
        let Ok(character) = read_json(&candidate) else {
            continue;
        };
        if character.get("id").and_then(Value::as_str) == Some(character_id) {
            return Ok(candidate);
        }
    }
    Err(ProjectStoreError::NotFound(
        "Character not found".to_owned(),
    ))
}

fn write_character(
    project_path: &Path,
    character: &mut Value,
    touch_updated_at: bool,
) -> ProjectStoreResult<()> {
    if touch_updated_at {
        value_object_mut(character, "Character sidecar")?
            .insert("updatedAt".to_owned(), Value::String(utc_now()));
    }
    write_character_sidecar(project_path, character)
}

fn hydrate_character(
    project_id: &str,
    project_path: &Path,
    mut character: Value,
) -> ProjectStoreResult<Value> {
    let references = character
        .get("references")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut reference| {
            let asset = reference
                .get("assetId")
                .and_then(Value::as_str)
                .and_then(|asset_id| {
                    character_asset_summary(project_id, project_path, asset_id)
                        .ok()
                        .flatten()
                })
                .unwrap_or(Value::Null);
            if let Some(object) = reference.as_object_mut() {
                object.insert("asset".to_owned(), asset);
            }
            reference
        })
        .collect::<Vec<_>>();
    let approved_references = references
        .iter()
        .filter(|reference| {
            reference
                .get("approved")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    let object = value_object_mut(&mut character, "Character sidecar")?;
    object.insert("references".to_owned(), Value::Array(references));
    object.insert(
        "approvedReferences".to_owned(),
        Value::Array(approved_references),
    );
    Ok(character)
}

fn character_asset_summary(
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<Value>> {
    let Some(sidecar_path) = find_asset_sidecar_path(project_path, asset_id)? else {
        return Ok(None);
    };
    let asset = normalize_asset(project_id, project_path, &sidecar_path)?;
    Ok(Some(json!({
        "id": asset.get("id").cloned().unwrap_or(Value::Null),
        "type": asset.get("type").cloned().unwrap_or(Value::Null),
        "displayName": asset.get("displayName").cloned().unwrap_or(Value::Null),
        "url": asset.get("url").cloned().unwrap_or(Value::Null),
        "status": asset.get("status").cloned().unwrap_or_else(|| json!({})),
        "file": asset.get("file").cloned().unwrap_or_else(|| json!({}))
    })))
}

fn update_asset_character_link(
    connection: &Connection,
    project_path: &Path,
    character_id: &str,
    reference: &Value,
    remove: bool,
) -> ProjectStoreResult<()> {
    let asset_id = required_str(reference, "assetId")?;
    let sidecar_path =
        find_asset_sidecar_path_on_connection(connection, project_path, asset_id)?
            .ok_or_else(|| ProjectStoreError::NotFound("Reference asset not found".to_owned()))?;
    let mut asset = read_json(&sidecar_path)?;
    let metadata = value_object_mut(&mut asset, "Asset sidecar")?
        .entry("metadata".to_owned())
        .or_insert_with(|| json!({}));
    let metadata = metadata.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Asset metadata must be an object".to_owned())
    })?;
    let mut links = metadata
        .remove("characterReferences")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter(|item| item.get("characterId").and_then(Value::as_str) != Some(character_id))
        .collect::<Vec<_>>();
    if !remove {
        links.push(json!({
            "characterId": character_id,
            "source": "character-sidecar",
            "approved": reference.get("approved").and_then(Value::as_bool).unwrap_or(false),
            "role": reference.get("role").and_then(Value::as_str).unwrap_or("reference"),
            "linkedAt": reference.get("addedAt").and_then(Value::as_str).unwrap_or("").to_owned(),
            "approvedAt": reference.get("approvedAt").cloned().unwrap_or(Value::Null)
        }));
    }
    metadata.insert("characterReferences".to_owned(), Value::Array(links));
    write_json(&sidecar_path, &asset)?;
    index_asset_on_connection(connection, project_path, &asset, Some(&sidecar_path))
}

fn find_asset_sidecar_path(
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<PathBuf>> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    find_asset_sidecar_path_on_connection(&connection, project_path, asset_id)
}

fn find_asset_sidecar_path_on_connection(
    connection: &Connection,
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<PathBuf>> {
    if let Some(record) = connection
        .query_row(
            "select file_path, sidecar_path from assets where id = ?1",
            params![asset_id],
            row_to_asset_record,
        )
        .optional()?
    {
        let mut candidates = Vec::new();
        if let Some(sidecar_path) = record.sidecar_path {
            candidates.push(project_path.join(sidecar_path));
        }
        if let Some(file_path) = record.file_path {
            candidates.push(
                project_path
                    .join(file_path)
                    .with_extension("sceneworks.json"),
            );
        }
        for candidate in candidates {
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }
    }
    for sidecar_path in asset_sidecars(project_path)? {
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        if asset.get("id").and_then(Value::as_str) == Some(asset_id) {
            return Ok(Some(sidecar_path));
        }
    }
    Ok(None)
}

fn index_asset_on_connection(
    connection: &Connection,
    project_path: &Path,
    asset: &Value,
    sidecar_path: Option<&Path>,
) -> ProjectStoreResult<()> {
    let sidecar_rel = match sidecar_path {
        Some(path) => Some(relative_string(project_path, path)?),
        None => None,
    };
    upsert_asset_row(connection, asset, sidecar_rel.as_deref())
}

fn copy_lora_into_project(
    project_path: &Path,
    data_dir: &Path,
    character_id: &str,
    source_path: Option<&str>,
) -> ProjectStoreResult<(Option<String>, bool)> {
    let Some(source_path) = source_path.filter(|value| !value.trim().is_empty()) else {
        return Ok((None, false));
    };
    let source_path = PathBuf::from(source_path);
    if !source_path.exists() || !(source_path.is_file() || source_path.is_dir()) {
        return Err(ProjectStoreError::BadRequest(format!(
            "LoRA source path not found: {}",
            source_path.display()
        )));
    }
    assert_allowed_lora_source(project_path, data_dir, &source_path)?;
    let target_dir = project_path
        .join("loras")
        .join("characters")
        .join(character_id);
    fs::create_dir_all(&target_dir)?;
    let target = target_dir.join(
        source_path
            .file_name()
            .ok_or_else(|| ProjectStoreError::BadRequest("Invalid LoRA source path".to_owned()))?,
    );
    if fs::canonicalize(&source_path).ok() != fs::canonicalize(&target).ok() {
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &target)?;
        } else {
            fs::copy(&source_path, &target)?;
        }
    }
    Ok((Some(relative_string(project_path, &target)?), true))
}

fn assert_allowed_lora_source(
    project_path: &Path,
    data_dir: &Path,
    source_path: &Path,
) -> ProjectStoreResult<()> {
    let resolved = fs::canonicalize(source_path)?;
    let roots = [data_dir.join("loras"), project_path.join("loras")];
    for root in roots {
        if let Ok(root) = fs::canonicalize(root) {
            if resolved.starts_with(root) {
                return Ok(());
            }
        }
    }
    Err(ProjectStoreError::BadRequest(
        "LoRA source path must be inside the app-managed data/loras or project/loras folders"
            .to_owned(),
    ))
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> ProjectStoreResult<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn prepend_array_field(payload: &mut Value, field: &str, item: Value) -> ProjectStoreResult<()> {
    let object = value_object_mut(payload, "Character sidecar")?;
    let mut items = object
        .remove(field)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    items.insert(0, item);
    object.insert(field.to_owned(), Value::Array(items));
    Ok(())
}

fn write_character_json(path: &Path, payload: &Value) -> ProjectStoreResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = String::new();
    write_character_value(payload, 0, None, &mut output)?;
    output.push('\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    fs::write(&tmp_path, output)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn write_character_value(
    value: &Value,
    indent: usize,
    key_hint: Option<&str>,
    output: &mut String,
) -> ProjectStoreResult<()> {
    match value {
        Value::Object(object) if should_inline_object(key_hint) => {
            write_inline_value(&Value::Object(object.clone()), output)?;
        }
        Value::Object(object) => {
            if object.is_empty() {
                output.push_str("{}");
                return Ok(());
            }
            output.push_str("{\n");
            let keys = ordered_character_keys(object);
            for (index, key) in keys.iter().enumerate() {
                output.push_str(&" ".repeat(indent + 2));
                output.push_str(&serde_json::to_string(key)?);
                output.push_str(": ");
                write_character_value(&object[*key], indent + 2, Some(key), output)?;
                if index + 1 != keys.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&" ".repeat(indent));
            output.push('}');
        }
        Value::Array(items) if items.is_empty() || items.iter().all(is_inline_json_value) => {
            write_inline_value(value, output)?;
        }
        Value::Array(items) => {
            output.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                output.push_str(&" ".repeat(indent + 2));
                write_character_value(item, indent + 2, key_hint, output)?;
                if index + 1 != items.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&" ".repeat(indent));
            output.push(']');
        }
        _ => write_inline_value(value, output)?,
    }
    Ok(())
}

fn should_inline_object(key_hint: Option<&str>) -> bool {
    matches!(key_hint, Some("recipeSettings" | "compatibility"))
}

fn is_inline_json_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

fn write_inline_value(value: &Value, output: &mut String) -> ProjectStoreResult<()> {
    match value {
        Value::Object(object) => {
            output.push_str("{ ");
            let keys = ordered_character_keys(object);
            for (index, key) in keys.iter().enumerate() {
                output.push_str(&serde_json::to_string(key)?);
                output.push_str(": ");
                write_inline_value(&object[*key], output)?;
                if index + 1 != keys.len() {
                    output.push_str(", ");
                }
            }
            output.push_str(" }");
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                write_inline_value(item, output)?;
                if index + 1 != items.len() {
                    output.push_str(", ");
                }
            }
            output.push(']');
        }
        _ => output.push_str(&serde_json::to_string(value)?),
    }
    Ok(())
}

fn ordered_character_keys(object: &Map<String, Value>) -> Vec<&str> {
    let preferred: &[&str] =
        if object.contains_key("schemaVersion") && object.contains_key("trainedLoras") {
            &[
                "schemaVersion",
                "id",
                "projectId",
                "name",
                "type",
                "description",
                "createdAt",
                "updatedAt",
                "status",
                "references",
                "looks",
                "loras",
                "trainedLoras",
            ]
        } else if object.contains_key("assetId") && object.contains_key("addedAt") {
            &[
                "assetId",
                "approved",
                "role",
                "notes",
                "addedAt",
                "approvedAt",
                "asset",
            ]
        } else if object.contains_key("approvedReferenceIds") {
            &[
                "id",
                "name",
                "description",
                "approvedReferenceIds",
                "recipeSettings",
                "createdAt",
                "updatedAt",
            ]
        } else if object.contains_key("loraId") || object.contains_key("copiedIntoProject") {
            &[
                "id",
                "loraId",
                "name",
                "sourcePath",
                "projectPath",
                "copiedIntoProject",
                "category",
                "scope",
                "triggerWords",
                "defaultWeight",
                "compatibility",
                "createdAt",
                "updatedAt",
            ]
        } else if object.contains_key("archived") {
            &["archived"]
        } else {
            &[]
        };

    let mut keys = Vec::new();
    for key in preferred {
        if object.contains_key(*key) {
            keys.push(*key);
        }
    }
    for key in object.keys() {
        if !keys.iter().any(|existing| existing == key) {
            keys.push(key.as_str());
        }
    }
    keys
}

fn to_json_string(value: &Value) -> ProjectStoreResult<String> {
    serde_json::to_string(value).map_err(Into::into)
}

fn required_str<'a>(value: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    optional_str(value, key).ok_or_else(|| {
        ProjectStoreError::BadRequest(format!(
            "{} is required",
            camel_to_title(key).unwrap_or_else(|| key.to_owned())
        ))
    })
}

fn value_object_mut<'a>(
    value: &'a mut Value,
    label: &str,
) -> ProjectStoreResult<&'a mut Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{label} must be an object")))
}

fn validate_required_text(value: &str, key: &str) -> ProjectStoreResult<()> {
    if value.trim().is_empty() {
        return Err(ProjectStoreError::BadRequest(format!(
            "{} is required",
            camel_to_title(key).unwrap_or_else(|| key.to_owned())
        )));
    }
    Ok(())
}

fn validate_text_length(value: &str, key: &str, max: usize) -> ProjectStoreResult<()> {
    if value.chars().count() > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at most {max} characters"
        )));
    }
    Ok(())
}

fn validate_character_name(value: &str) -> ProjectStoreResult<()> {
    validate_required_text(value, "name")?;
    let max = 120;
    if value.chars().count() > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "name must be at most {max} characters"
        )));
    }
    Ok(())
}

fn validate_character_type(value: &str) -> ProjectStoreResult<()> {
    if matches!(value, "person" | "creature" | "object") {
        return Ok(());
    }
    Err(ProjectStoreError::BadRequest(
        "Character type must be person, creature, or object".to_owned(),
    ))
}

fn validate_lora_weight(value: f64) -> ProjectStoreResult<()> {
    if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
        return Err(ProjectStoreError::BadRequest(
            "defaultWeight must be between -2 and 2".to_owned(),
        ));
    }
    Ok(())
}

fn validate_lora_scope(value: &str, allow_empty: bool) -> ProjectStoreResult<()> {
    if allow_empty && value.trim().is_empty() {
        return Ok(());
    }
    if matches!(value, "project" | "global") {
        return Ok(());
    }
    Err(ProjectStoreError::BadRequest(
        "scope must be project or global".to_owned(),
    ))
}

fn camel_to_title(value: &str) -> Option<String> {
    if value.is_empty() {
        return None;
    }
    let mut output = String::new();
    for character in value.chars() {
        if character.is_uppercase() && !output.is_empty() {
            output.push(' ');
        }
        output.push(character);
    }
    let mut characters = output.chars();
    let first = characters.next()?.to_uppercase().collect::<String>();
    Some(format!("{first}{}", characters.as_str()))
}
