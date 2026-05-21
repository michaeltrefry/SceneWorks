use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::project_store::{ProjectStoreError, ProjectStoreResult};
use crate::training::{
    Caption, CaptionSource, TrainingDataset, TrainingDatasetItem, TrainingDatasetStatus,
    TrainingModality, TRAINING_CONTRACT_SCHEMA_VERSION,
};

const DATASET_MANIFEST_NAME: &str = "dataset.sceneworks.training-dataset.json";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetCreateInput {
    pub name: String,
    #[serde(default)]
    pub modality: Option<TrainingModality>,
    #[serde(default)]
    pub status: Option<TrainingDatasetStatus>,
    #[serde(default)]
    pub items: Vec<TrainingDatasetItemInput>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetUpdateInput {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub status: Option<TrainingDatasetStatus>,
    /// Full replacement for the dataset's ordered item set when present.
    #[serde(default)]
    pub items: Option<Vec<TrainingDatasetItemInput>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetItemInput {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub asset_id: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub caption: Option<CaptionInput>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionInput {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub source: Option<CaptionSource>,
    #[serde(default)]
    pub trigger_words: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetSummary {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub modality: TrainingModality,
    pub status: TrainingDatasetStatus,
    pub version: u32,
    pub item_count: usize,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetMutationResult {
    pub id: String,
    pub status: String,
}

#[derive(Debug)]
pub struct TrainingDatasetStore {
    project_path: PathBuf,
}

impl TrainingDatasetStore {
    pub fn new(project_path: impl Into<PathBuf>) -> Self {
        Self {
            project_path: project_path.into(),
        }
    }

    pub fn list_datasets(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<Vec<TrainingDatasetSummary>> {
        ensure_training_dataset_table(&self.project_path)?;
        list_dataset_summaries_from_index(&self.project_path, project_id)
    }

    pub fn create_dataset(
        &self,
        project_id: &str,
        input: TrainingDatasetCreateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let name = validated_dataset_name(&input.name)?;
        let modality = input.modality.unwrap_or(TrainingModality::Image);
        validate_supported_modality(&modality)?;
        let dataset_id = format!("ds_{}", random_hex(16)?);
        let dataset_root = dataset_root(&self.project_path, &dataset_id);
        let media_dir = dataset_root.join("images");
        fs::create_dir_all(&media_dir)?;
        let now = utc_now();
        let items = match materialize_items(
            &self.project_path,
            &media_dir,
            &modality,
            project_id,
            input.items,
            &now,
        ) {
            Ok(items) => items,
            Err(error) => {
                let _ = fs::remove_dir_all(&dataset_root);
                return Err(error);
            }
        };
        let dataset = TrainingDataset {
            schema_version: TRAINING_CONTRACT_SCHEMA_VERSION,
            id: dataset_id,
            version: 1,
            project_id: Some(project_id.to_owned()),
            name,
            modality,
            status: input.status.unwrap_or(TrainingDatasetStatus::Draft),
            created_at: now.clone(),
            updated_at: now,
            items,
            extra: Default::default(),
        };
        if let Err(error) = self.save_dataset(&dataset) {
            let _ = fs::remove_dir_all(&dataset_root);
            return Err(error);
        }
        Ok(dataset)
    }

    pub fn get_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDataset> {
        let dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        Ok(dataset)
    }

    pub fn update_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetUpdateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let mut dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        if let Some(name) = input.name {
            dataset.name = validated_dataset_name(&name)?;
        }
        if let Some(status) = input.status {
            dataset.status = status;
        }
        if let Some(items) = input.items {
            let dataset_root = dataset_root(&self.project_path, &dataset.id);
            let temp_media_dir = dataset_root.join(format!("images.tmp-{}", random_hex(8)?));
            fs::create_dir_all(&temp_media_dir)?;
            let now = utc_now();
            let next_items = match materialize_items(
                &self.project_path,
                &temp_media_dir,
                &dataset.modality,
                project_id,
                items,
                &now,
            ) {
                Ok(items) => items,
                Err(error) => {
                    let _ = fs::remove_dir_all(&temp_media_dir);
                    return Err(error);
                }
            };
            let backup_media_dir = replace_dataset_media_dir(&dataset_root, &temp_media_dir)?;
            dataset.items = next_items;
            dataset.version = dataset.version.saturating_add(1);
            dataset.updated_at = utc_now();
            let manifest_path = dataset_manifest_path(&self.project_path, &dataset.id);
            if let Err(error) = write_json(&manifest_path, &dataset) {
                rollback_dataset_media_dir(&dataset_root, backup_media_dir.as_deref());
                return Err(error);
            }
            remove_optional_dir(backup_media_dir)?;
            index_dataset(&self.project_path, &dataset, &manifest_path)?;
            return Ok(dataset);
        }
        dataset.updated_at = utc_now();
        self.save_dataset(&dataset)?;
        Ok(dataset)
    }

    pub fn delete_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDatasetMutationResult> {
        let dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        let root = dataset_root(&self.project_path, dataset_id);
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        remove_dataset_index(&self.project_path, dataset_id)?;
        Ok(TrainingDatasetMutationResult {
            id: dataset_id.to_owned(),
            status: "deleted".to_owned(),
        })
    }

    fn read_dataset_by_id(&self, dataset_id: &str) -> ProjectStoreResult<TrainingDataset> {
        if !is_safe_id(dataset_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid training dataset ID".to_owned(),
            ));
        }
        let manifest_path = dataset_manifest_path(&self.project_path, dataset_id);
        if !manifest_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Training dataset not found".to_owned(),
            ));
        }
        read_dataset(&manifest_path)
    }

    fn save_dataset(&self, dataset: &TrainingDataset) -> ProjectStoreResult<()> {
        let manifest_path = dataset_manifest_path(&self.project_path, &dataset.id);
        write_json(&manifest_path, dataset)?;
        index_dataset(&self.project_path, dataset, &manifest_path)
    }
}

fn materialize_items(
    project_path: &Path,
    media_dir: &Path,
    modality: &TrainingModality,
    project_id: &str,
    inputs: Vec<TrainingDatasetItemInput>,
    now: &str,
) -> ProjectStoreResult<Vec<TrainingDatasetItem>> {
    let mut item_ids = Vec::new();
    let mut items = Vec::new();
    for (index, input) in inputs.into_iter().enumerate() {
        let item_id = item_id_for_input(&input, index)?;
        if item_ids.iter().any(|existing| existing == &item_id) {
            return Err(ProjectStoreError::BadRequest(
                "Training dataset item IDs must be unique".to_owned(),
            ));
        }
        item_ids.push(item_id.clone());
        let item = materialize_item(
            project_path,
            media_dir,
            modality,
            project_id,
            input,
            item_id,
            now,
        )?;
        items.push(item);
    }
    Ok(items)
}

fn item_id_for_input(input: &TrainingDatasetItemInput, index: usize) -> ProjectStoreResult<String> {
    match input.id.as_deref() {
        Some(id) if is_safe_id(id) => Ok(id.to_owned()),
        Some(_) => Err(ProjectStoreError::BadRequest(
            "Invalid training dataset item ID".to_owned(),
        )),
        None => Ok(format!("item_{:04}", index + 1)),
    }
}

fn materialize_item(
    project_path: &Path,
    media_dir: &Path,
    modality: &TrainingModality,
    project_id: &str,
    input: TrainingDatasetItemInput,
    item_id: String,
    now: &str,
) -> ProjectStoreResult<TrainingDatasetItem> {
    validate_supported_modality(modality)?;
    let source = resolve_item_source(project_path, project_id, &input, modality)?;
    let extension = source
        .path
        .extension()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(|value| format!(".{}", value.to_ascii_lowercase()))
        .unwrap_or_else(|| ".bin".to_owned());
    let relative_path = format!("images/{item_id}{extension}");
    let target_path = media_dir.join(format!("{item_id}{extension}"));
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&source.path, &target_path)?;
    let caption = input.caption.unwrap_or_default();
    Ok(TrainingDatasetItem {
        id: item_id,
        asset_id: source.asset_id,
        path: relative_path,
        display_name: input.display_name.unwrap_or(source.display_name),
        caption: Caption {
            text: caption.text,
            source: caption.source.unwrap_or(CaptionSource::Manual),
            trigger_words: caption.trigger_words,
            updated_at: Some(now.to_owned()),
            extra: Default::default(),
        },
        width: input.width.or(source.width),
        height: input.height.or(source.height),
        added_at: now.to_owned(),
        extra: Default::default(),
    })
}

#[derive(Debug)]
struct ItemSource {
    path: PathBuf,
    asset_id: Option<String>,
    display_name: String,
    width: Option<u32>,
    height: Option<u32>,
}

fn resolve_item_source(
    project_path: &Path,
    project_id: &str,
    input: &TrainingDatasetItemInput,
    modality: &TrainingModality,
) -> ProjectStoreResult<ItemSource> {
    if let Some(asset_id) = input.asset_id.as_deref().filter(|value| !value.is_empty()) {
        return resolve_asset_source(project_path, project_id, asset_id, modality);
    }
    let relative_path = input
        .path
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("Dataset item assetId or path is required".to_owned())
        })?;
    if !is_safe_relative_path(relative_path) {
        return Err(ProjectStoreError::BadRequest(
            "Invalid dataset item path".to_owned(),
        ));
    }
    let root = fs::canonicalize(project_path)?;
    let path = fs::canonicalize(project_path.join(relative_path)).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ProjectStoreError::NotFound("Dataset item file not found".to_owned())
        } else {
            ProjectStoreError::Io(error)
        }
    })?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(ProjectStoreError::BadRequest(
            "Invalid dataset item path".to_owned(),
        ));
    }
    ensure_supported_item_mime(&path, modality)?;
    Ok(ItemSource {
        display_name: input_display_name(&path),
        path,
        asset_id: None,
        width: None,
        height: None,
    })
}

fn resolve_asset_source(
    project_path: &Path,
    project_id: &str,
    asset_id: &str,
    modality: &TrainingModality,
) -> ProjectStoreResult<ItemSource> {
    if !is_safe_id(asset_id) {
        return Err(ProjectStoreError::BadRequest("Invalid asset ID".to_owned()));
    }
    let connection = Connection::open(project_path.join("project.db"))?;
    let (file_path, sidecar_path): (String, Option<String>) = connection
        .query_row(
            "select file_path, sidecar_path from assets where id = ?1",
            params![asset_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|error| match error {
            rusqlite::Error::QueryReturnedNoRows => {
                ProjectStoreError::NotFound("Asset not found".to_owned())
            }
            other => ProjectStoreError::Sqlite(other),
        })?;
    let sidecar = sidecar_path
        .map(|path| project_path.join(path))
        .unwrap_or_else(|| {
            project_path
                .join(&file_path)
                .with_extension("sceneworks.json")
        });
    let asset: Value = read_json_value(&sidecar)?;
    if asset.get("projectId").and_then(Value::as_str) != Some(project_id) {
        return Err(ProjectStoreError::BadRequest(
            "Asset belongs to a different project".to_owned(),
        ));
    }
    let expected_type = match modality {
        TrainingModality::Image => "image",
        TrainingModality::Video | TrainingModality::Audio | TrainingModality::Unknown(_) => {
            return Err(ProjectStoreError::BadRequest(
                "Only image training datasets are supported".to_owned(),
            ))
        }
    };
    if asset.get("type").and_then(Value::as_str) != Some(expected_type) {
        return Err(ProjectStoreError::BadRequest(
            "Training dataset items must be image assets".to_owned(),
        ));
    }
    let root = fs::canonicalize(project_path)?;
    let path = fs::canonicalize(project_path.join(&file_path))?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(ProjectStoreError::BadRequest(
            "Invalid asset file path".to_owned(),
        ));
    }
    Ok(ItemSource {
        path,
        asset_id: Some(asset_id.to_owned()),
        display_name: asset
            .get("displayName")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| input_display_name(Path::new(&file_path))),
        width: optional_u32(asset.pointer("/file/width")),
        height: optional_u32(asset.pointer("/file/height")),
    })
}

fn validated_dataset_name(name: &str) -> ProjectStoreResult<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(ProjectStoreError::BadRequest(
            "Training dataset name is required".to_owned(),
        ));
    }
    if name.chars().count() > 120 {
        return Err(ProjectStoreError::BadRequest(
            "Training dataset name must be at most 120 characters".to_owned(),
        ));
    }
    Ok(name.to_owned())
}

fn validate_supported_modality(modality: &TrainingModality) -> ProjectStoreResult<()> {
    match modality {
        TrainingModality::Image => Ok(()),
        TrainingModality::Video | TrainingModality::Audio | TrainingModality::Unknown(_) => Err(
            ProjectStoreError::BadRequest("Only image training datasets are supported".to_owned()),
        ),
    }
}

fn ensure_supported_item_mime(path: &Path, modality: &TrainingModality) -> ProjectStoreResult<()> {
    validate_supported_modality(modality)?;
    let mime = mime_guess::from_path(path).first_raw().unwrap_or_default();
    if !mime.starts_with("image/") {
        return Err(ProjectStoreError::BadRequest(
            "Training dataset items must be image files".to_owned(),
        ));
    }
    Ok(())
}

fn ensure_dataset_project(project_id: &str, dataset: &TrainingDataset) -> ProjectStoreResult<()> {
    if dataset.project_id.as_deref() != Some(project_id) {
        return Err(ProjectStoreError::NotFound(
            "Training dataset not found".to_owned(),
        ));
    }
    Ok(())
}

pub fn ensure_training_dataset_table(project_path: &Path) -> ProjectStoreResult<()> {
    let connection = Connection::open(project_path.join("project.db"))?;
    apply_training_dataset_migrations(&connection)
}

pub fn apply_training_dataset_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    connection.execute_batch(
        "
        create table if not exists training_datasets (
          id text primary key,
          project_id text not null,
          name text not null,
          modality text not null,
          status text not null,
          version integer not null,
          item_count integer not null default 0,
          file_path text not null,
          created_at text not null,
          updated_at text not null
        );
        create index if not exists idx_training_datasets_project_updated
          on training_datasets(project_id, updated_at);
        ",
    )?;
    ensure_training_dataset_column(connection, "item_count", "integer not null default 0")?;
    Ok(())
}

fn list_dataset_summaries_from_index(
    project_path: &Path,
    project_id: &str,
) -> ProjectStoreResult<Vec<TrainingDatasetSummary>> {
    let connection = Connection::open(project_path.join("project.db"))?;
    let mut statement = connection.prepare(
        "
        select id, project_id, name, modality, status, version, item_count, created_at, updated_at
          from training_datasets
         where project_id = ?1
         order by updated_at desc, name asc
        ",
    )?;
    let summaries = statement
        .query_map(params![project_id], |row| {
            let item_count: i64 = row.get(6)?;
            Ok(TrainingDatasetSummary {
                id: row.get(0)?,
                project_id: row.get(1)?,
                name: row.get(2)?,
                modality: parse_string_enum(&row.get::<_, String>(3)?),
                status: parse_string_enum(&row.get::<_, String>(4)?),
                version: row.get(5)?,
                item_count: usize::try_from(item_count).unwrap_or(0),
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(summaries)
}

fn index_dataset(
    project_path: &Path,
    dataset: &TrainingDataset,
    manifest_path: &Path,
) -> ProjectStoreResult<()> {
    ensure_training_dataset_table(project_path)?;
    let rel_path = relative_string(project_path, manifest_path)?;
    let connection = Connection::open(project_path.join("project.db"))?;
    connection.execute(
        "
        insert or replace into training_datasets (
          id, project_id, name, modality, status, version, item_count, file_path, created_at, updated_at
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ",
        params![
            dataset.id,
            dataset.project_id.as_deref().unwrap_or_default(),
            dataset.name,
            dataset.modality.as_str(),
            dataset.status.as_str(),
            dataset.version,
            i64::try_from(dataset.items.len()).unwrap_or(i64::MAX),
            rel_path,
            dataset.created_at,
            dataset.updated_at,
        ],
    )?;
    Ok(())
}

fn remove_dataset_index(project_path: &Path, dataset_id: &str) -> ProjectStoreResult<()> {
    ensure_training_dataset_table(project_path)?;
    let connection = Connection::open(project_path.join("project.db"))?;
    connection.execute(
        "delete from training_datasets where id = ?1",
        params![dataset_id],
    )?;
    Ok(())
}

fn ensure_training_dataset_column(
    connection: &Connection,
    column: &str,
    definition: &str,
) -> ProjectStoreResult<()> {
    let mut statement = connection.prepare("pragma table_info(training_datasets)")?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>("name"))?
        .collect::<Result<Vec<_>, _>>()?;
    if !columns.iter().any(|existing| existing == column) {
        connection.execute(
            &format!("alter table training_datasets add column {column} {definition}"),
            [],
        )?;
    }
    Ok(())
}

fn parse_string_enum<T>(value: &str) -> T
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(Value::String(value.to_owned()))
        .expect("string enum deserialization is infallible")
}

fn dataset_root(project_path: &Path, dataset_id: &str) -> PathBuf {
    project_path
        .join("training")
        .join("datasets")
        .join(dataset_id)
}

fn dataset_manifest_path(project_path: &Path, dataset_id: &str) -> PathBuf {
    dataset_root(project_path, dataset_id).join(DATASET_MANIFEST_NAME)
}

fn replace_dataset_media_dir(
    dataset_root: &Path,
    temp_media_dir: &Path,
) -> ProjectStoreResult<Option<PathBuf>> {
    let media_dir = dataset_root.join("images");
    let backup_dir = dataset_root.join(format!("images.backup-{}", random_hex(8)?));
    if media_dir.exists() {
        fs::rename(&media_dir, &backup_dir)?;
    }
    match fs::rename(temp_media_dir, &media_dir) {
        Ok(()) => Ok(backup_dir.exists().then_some(backup_dir)),
        Err(error) => {
            if backup_dir.exists() {
                let _ = fs::rename(&backup_dir, &media_dir);
            }
            Err(ProjectStoreError::Io(error))
        }
    }
}

fn rollback_dataset_media_dir(dataset_root: &Path, backup_media_dir: Option<&Path>) {
    let media_dir = dataset_root.join("images");
    let _ = fs::remove_dir_all(&media_dir);
    if let Some(backup_media_dir) = backup_media_dir {
        let _ = fs::rename(backup_media_dir, media_dir);
    }
}

fn remove_optional_dir(path: Option<PathBuf>) -> ProjectStoreResult<()> {
    if let Some(path) = path {
        fs::remove_dir_all(path)?;
    }
    Ok(())
}

fn read_dataset(path: &Path) -> ProjectStoreResult<TrainingDataset> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

fn read_json_value(path: &Path) -> ProjectStoreResult<Value> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

fn write_json<T: Serialize>(path: &Path, payload: &T) -> ProjectStoreResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = serde_json::to_string_pretty(payload)?;
    output.push('\n');
    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, output)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn relative_string(root: &Path, path: &Path) -> ProjectStoreResult<String> {
    Ok(path
        .strip_prefix(root)
        .map_err(|_| ProjectStoreError::BadRequest("Path is outside project".to_owned()))?
        .to_string_lossy()
        .replace('\\', "/"))
}

fn is_safe_relative_path(relative_path: &str) -> bool {
    !relative_path.trim().is_empty()
        && !relative_path.contains('\\')
        && Path::new(relative_path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

fn is_safe_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn input_display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("dataset-item")
        .to_owned()
}

fn optional_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn random_hex(bytes: usize) -> ProjectStoreResult<String> {
    let connection = Connection::open_in_memory()?;
    Ok(connection.query_row(
        &format!("select lower(hex(randomblob({bytes})))"),
        [],
        |row| row.get(0),
    )?)
}

fn utc_now() -> String {
    format_unix_seconds(now_unix_seconds())
}

fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn format_unix_seconds(timestamp: i64) -> String {
    let days = timestamp.div_euclid(86_400);
    let seconds_of_day = timestamp.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let adjusted_days = days + 719_468;
    let era = adjusted_days.div_euclid(146_097);
    let day_of_era = adjusted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}
