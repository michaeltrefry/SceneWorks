use std::fs;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::asset_index::{
    asset_sidecars, normalize_asset, row_to_asset_record, upsert_asset_row, AssetRecord,
};
use crate::character_store::{
    apply_character_migrations, clear_character_tables, reindex_characters_on_connection,
    CharacterStore,
};
pub use crate::character_store::{
    CharacterCreateInput, CharacterLookInput, CharacterLookUpdateInput, CharacterLoraInput,
    CharacterLoraUpdateInput, CharacterMutationResult, CharacterReferenceInput,
    CharacterReferenceUpdateInput, CharacterUpdateInput, CHARACTER_SIDECAR_PATTERN,
};
use crate::slug::slugify;
use crate::store_util::{
    is_safe_relative_path, optional_f64, optional_str, optional_u64, random_hex, read_json,
    relative_string, write_json,
};
use crate::time::utc_now;
use crate::training::TrainingDataset;
use crate::training_store::{
    apply_training_dataset_migrations, TrainingCaptionSidecarsResult,
    TrainingDatasetBatchRenameInput, TrainingDatasetCaptionSidecarsInput,
    TrainingDatasetCreateInput, TrainingDatasetMutationResult, TrainingDatasetStore,
    TrainingDatasetSummary, TrainingDatasetUpdateInput,
};

pub const PROJECT_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "assets/documents",
    "characters",
    "generation-sets",
    "loras",
    "person-tracks",
    "recipes",
    "timelines",
    "training/datasets",
    "trash",
    "cache",
];
pub type ProjectStoreResult<T> = Result<T, ProjectStoreError>;

#[derive(Debug)]
pub enum ProjectStoreError {
    Io(std::io::Error),
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    BadRequest(String),
    NotFound(String),
}

impl std::fmt::Display for ProjectStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Sqlite(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::BadRequest(detail) => write!(formatter, "{detail}"),
            Self::NotFound(detail) => write!(formatter, "{detail}"),
        }
    }
}

impl std::error::Error for ProjectStoreError {}

impl From<std::io::Error> for ProjectStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for ProjectStoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for ProjectStoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReindexResult {
    pub project_id: String,
    pub assets: u32,
    pub generation_sets: u32,
    pub timelines: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineSummary {
    pub id: String,
    pub name: String,
    pub file_path: String,
    pub aspect_ratio: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration: f64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineFile {
    pub path: PathBuf,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimelineFileDocument {
    pub file: TimelineFile,
    pub document: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetMutationResult {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetStatusPatch {
    pub favorite: Option<bool>,
    pub rating: Option<u8>,
    pub rejected: Option<bool>,
    pub trashed: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct UploadAsset {
    pub filename: String,
    pub content_type: Option<String>,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ProjectFile {
    pub path: PathBuf,
    pub content_type: String,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct RegistryItem {
    id: Option<String>,
    name: Option<String>,
    path: Option<String>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Debug)]
pub struct ProjectStore {
    data_dir: PathBuf,
    app_version: String,
    /// Guards recent-project registry reads/writes only; project DB and project-file mutations
    /// rely on SQLite transactions and filesystem operations for their own serialization.
    lock: Mutex<()>,
}

impl ProjectStore {
    pub fn new(data_dir: impl Into<PathBuf>, app_version: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            app_version: app_version.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }

    pub fn registry_path(&self) -> PathBuf {
        self.data_dir.join("recent-projects.json")
    }

    pub fn list_projects(&self) -> ProjectStoreResult<Vec<ProjectSummary>> {
        let _guard = self.lock.lock();
        self.ensure_data_dirs()?;
        let mut projects = Vec::new();
        for item in self.load_registry()? {
            let Some(path) = item.path else {
                continue;
            };
            let project_path = PathBuf::from(path);
            if project_path.exists() {
                projects.push(read_project_summary(&project_path)?);
            }
        }
        Ok(projects)
    }

    pub fn create_project(&self, name: &str) -> ProjectStoreResult<ProjectSummary> {
        let name = name.trim();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Project name is required".to_owned(),
            ));
        }
        if name.chars().count() > 120 {
            return Err(ProjectStoreError::BadRequest(
                "Project name must be at most 120 characters".to_owned(),
            ));
        }

        let _guard = self.lock.lock();
        self.ensure_data_dirs()?;
        let project_id = format!("project_{}", random_hex(16)?);
        let slug = slugify(name, "project", None);
        let mut project_path = self.projects_dir().join(format!("{slug}.sceneworks"));
        if project_path.exists() {
            let suffix = &project_id[project_id.len().saturating_sub(8)..];
            project_path = self
                .projects_dir()
                .join(format!("{slug}-{suffix}.sceneworks"));
        }

        fs::create_dir_all(&project_path)?;
        for folder in PROJECT_FOLDERS {
            fs::create_dir_all(project_path.join(folder))?;
        }
        write_project_file(&self.app_version, &project_path, &project_id, name)?;
        apply_project_migrations(&connect_project_db(&project_path)?)?;

        let mut registry = self
            .load_registry()?
            .into_iter()
            .filter(|item| item.id.as_deref() != Some(project_id.as_str()))
            .collect::<Vec<_>>();
        registry.insert(
            0,
            RegistryItem {
                id: Some(project_id),
                name: Some(name.to_owned()),
                path: Some(project_path.display().to_string()),
                extra: Map::new(),
            },
        );
        self.save_registry(&registry)?;

        read_project_summary(&project_path)
    }

    pub fn get_project(&self, project_id: &str) -> ProjectStoreResult<ProjectSummary> {
        let project_path = self.find_project_path(project_id)?;
        read_project_summary(&project_path)
    }

    pub fn list_training_datasets(
        &self,
        project_id: &str,
    ) -> ProjectStoreResult<Vec<TrainingDatasetSummary>> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).list_datasets(project_id)
    }

    pub fn create_training_dataset(
        &self,
        project_id: &str,
        input: TrainingDatasetCreateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).create_dataset(project_id, input)
    }

    pub fn get_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDataset> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).get_dataset(project_id, dataset_id)
    }

    /// Loads a training dataset together with its absolute on-disk root and the
    /// project stem — the inputs the Rust API needs to resolve a
    /// [`crate::training::TrainingPlan`] and label the queued job. Resolves the
    /// project path once rather than re-reading the registry per value.
    pub fn training_dataset_for_plan(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<(TrainingDataset, PathBuf, String)> {
        let project_path = self.find_project_path(project_id)?;
        let dataset =
            TrainingDatasetStore::new(project_path.clone()).get_dataset(project_id, dataset_id)?;
        let root = crate::training_store::dataset_root(&project_path, dataset_id);
        let project_stem = project_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        Ok((dataset, root, project_stem))
    }

    pub fn update_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetUpdateInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).update_dataset(project_id, dataset_id, input)
    }

    pub fn batch_rename_training_dataset_items(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetBatchRenameInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path)
            .batch_rename_dataset_items(project_id, dataset_id, input)
    }

    pub fn write_training_dataset_caption_sidecars(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetCaptionSidecarsInput,
    ) -> ProjectStoreResult<TrainingCaptionSidecarsResult> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path)
            .write_caption_sidecars(project_id, dataset_id, input)
    }

    pub fn delete_training_dataset(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<TrainingDatasetMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        TrainingDatasetStore::new(project_path).delete_dataset(project_id, dataset_id)
    }

    pub fn project_stem(&self, project_id: &str) -> ProjectStoreResult<String> {
        let project_path = self.find_project_path(project_id)?;
        Ok(project_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_owned())
    }

    pub fn reindex_project(&self, project_id: &str) -> ProjectStoreResult<ReindexResult> {
        let project_path = self.find_project_path(project_id)?;
        let counts = reindex_project_path(&project_path)?;
        Ok(ReindexResult {
            project_id: project_id.to_owned(),
            assets: counts.assets,
            generation_sets: counts.generation_sets,
            timelines: counts.timelines,
        })
    }

    pub fn list_timelines(&self, project_id: &str) -> ProjectStoreResult<Vec<TimelineSummary>> {
        let project_path = self.find_project_path(project_id)?;
        ensure_project_db_ready(&project_path)?;
        let connection = connect_project_db(&project_path)?;
        let mut statement = connection.prepare(
            "
            select id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
              from timelines
             order by updated_at desc
            ",
        )?;
        let timelines = statement
            .query_map([], |row| {
                Ok(TimelineSummary {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    file_path: row.get(2)?,
                    aspect_ratio: row.get(3)?,
                    width: row.get(4)?,
                    height: row.get(5)?,
                    fps: row.get(6)?,
                    duration: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProjectStoreError::from)?;
        Ok(timelines)
    }

    pub fn create_timeline(
        &self,
        project_id: &str,
        name: &str,
        aspect_ratio: &str,
        fps: u32,
    ) -> ProjectStoreResult<Value> {
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name is required".to_owned(),
            ));
        }
        if name.chars().count() > 120 {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name must be at most 120 characters".to_owned(),
            ));
        }
        if !(1..=60).contains(&fps) {
            return Err(ProjectStoreError::BadRequest(
                "FPS must be between 1 and 60".to_owned(),
            ));
        }
        let (width, height) = timeline_dimensions(aspect_ratio)?;
        let timeline = json!({
            "schemaVersion": 1,
            "id": format!("timeline_{}", random_hex(16)?),
            "projectId": project_id,
            "name": name,
            "aspectRatio": aspect_ratio,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": 0.0,
            "tracks": default_timeline_tracks(),
            "transitions": [],
            "createdAt": Value::Null,
            "updatedAt": Value::Null,
        });
        self.save_timeline(project_id, timeline)
    }

    pub fn get_timeline(&self, project_id: &str, timeline_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let timeline_file = find_timeline_file(&project_path, timeline_id)?;
        read_json(&timeline_file.path)
    }

    pub fn save_timeline(
        &self,
        project_id: &str,
        mut timeline: Value,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let timeline_id = required_str(&timeline, "id")?.to_owned();
        let timeline_project_id = required_str(&timeline, "projectId")?;
        if timeline_project_id != project_id {
            return Err(ProjectStoreError::BadRequest(
                "Project ID mismatch".to_owned(),
            ));
        }
        let name = required_str(&timeline, "name")?.to_owned();
        if name.is_empty() {
            return Err(ProjectStoreError::BadRequest(
                "Timeline name is required".to_owned(),
            ));
        }
        let now = utc_now();
        validate_and_default_timeline(&mut timeline)?;
        let duration = compute_timeline_duration(&timeline);
        let object = timeline.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Timeline must be an object".to_owned())
        })?;
        if !object.contains_key("createdAt") || object.get("createdAt") == Some(&Value::Null) {
            object.insert("createdAt".to_owned(), Value::String(now.clone()));
        }
        object.insert("updatedAt".to_owned(), Value::String(now));
        object.insert("duration".to_owned(), json!(duration));
        if !object
            .get("tracks")
            .and_then(Value::as_array)
            .is_some_and(|tracks| !tracks.is_empty())
        {
            object.insert("tracks".to_owned(), default_timeline_tracks());
        }
        object
            .entry("transitions".to_owned())
            .or_insert_with(|| json!([]));
        normalize_timeline_items(&mut timeline)?;

        let path = timeline_file_path(&project_path, &timeline_id, &name);
        let rel_path = relative_string(&project_path, &path)?;
        write_json(&path, &timeline)?;
        index_timeline(&project_path, &timeline, &rel_path)?;
        Ok(timeline)
    }

    pub fn save_existing_timeline(
        &self,
        project_id: &str,
        timeline_id: &str,
        timeline: Value,
    ) -> ProjectStoreResult<Value> {
        if required_str(&timeline, "id")? != timeline_id {
            return Err(ProjectStoreError::BadRequest(
                "Timeline ID mismatch".to_owned(),
            ));
        }
        self.save_timeline(project_id, timeline)
    }

    pub fn timeline_file(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFile> {
        let project_path = self.find_project_path(project_id)?;
        find_timeline_file(&project_path, timeline_id)
    }

    pub fn timeline_file_and_document(
        &self,
        project_id: &str,
        timeline_id: &str,
    ) -> ProjectStoreResult<TimelineFileDocument> {
        let project_path = self.find_project_path(project_id)?;
        let file = find_timeline_file(&project_path, timeline_id)?;
        let document = read_json(&file.path)?;
        Ok(TimelineFileDocument { file, document })
    }

    pub fn list_assets(
        &self,
        project_id: &str,
        include_rejected: bool,
        include_trashed: bool,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        ensure_project_db_ready(&project_path)?;
        let total = {
            let connection = connect_project_db(&project_path)?;
            connection.query_row("select count(*) from assets", [], |row| {
                row.get::<_, i64>(0)
            })?
        };
        if total == 0 && project_has_sidecars(&project_path) {
            reindex_project_path(&project_path)?;
        }

        let connection = connect_project_db(&project_path)?;
        let mut statement = connection.prepare(
            "
            select sidecar_path, file_path
              from assets
             where (?1 or rejected = 0)
               and (?2 or trashed = 0)
             order by created_at desc
            ",
        )?;
        let rows = statement.query_map(params![include_rejected, include_trashed], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<String>>(1)?,
            ))
        })?;
        let mut seen_asset_ids = Vec::new();
        let mut assets = Vec::new();
        for row in rows {
            let (sidecar_rel, file_rel) = row?;
            let mut candidates = Vec::new();
            if let Some(sidecar_rel) = sidecar_rel {
                candidates.push(project_path.join(sidecar_rel));
            }
            if let Some(file_rel) = file_rel {
                candidates.push(
                    project_path
                        .join(file_rel)
                        .with_extension("sceneworks.json"),
                );
            }
            for sidecar_path in candidates {
                if !sidecar_path.exists() {
                    continue;
                }
                let Ok(asset) = normalize_asset(project_id, &project_path, &sidecar_path) else {
                    continue;
                };
                let asset_id = asset
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if seen_asset_ids.iter().any(|seen| seen == &asset_id) {
                    break;
                }
                seen_asset_ids.push(asset_id);
                assets.push(asset);
                break;
            }
        }
        Ok(assets)
    }

    pub fn list_characters(
        &self,
        project_id: &str,
        include_archived: bool,
    ) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path)
            .list_characters(project_id, include_archived)
    }

    pub fn create_character(
        &self,
        project_id: &str,
        input: CharacterCreateInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).create_character(project_id, input)
    }

    pub fn get_character(&self, project_id: &str, character_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).get_character(project_id, character_id)
    }

    pub fn update_character(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_character(
            project_id,
            character_id,
            input,
        )
    }

    pub fn archive_character(
        &self,
        project_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).archive_character(character_id)
    }

    pub fn purge_character(
        &self,
        project_id: &str,
        character_id: &str,
    ) -> ProjectStoreResult<CharacterMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).purge_character(character_id)
    }

    pub fn add_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterReferenceInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).add_reference(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
        input: CharacterReferenceUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_reference(
            project_id,
            character_id,
            asset_id,
            input,
        )
    }

    pub fn remove_character_reference(
        &self,
        project_id: &str,
        character_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).remove_reference(
            project_id,
            character_id,
            asset_id,
        )
    }

    pub fn create_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLookInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).create_look(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
        input: CharacterLookUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_look(
            project_id,
            character_id,
            look_id,
            input,
        )
    }

    pub fn delete_character_look(
        &self,
        project_id: &str,
        character_id: &str,
        look_id: &str,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).delete_look(
            project_id,
            character_id,
            look_id,
        )
    }

    pub fn attach_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        input: CharacterLoraInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).attach_lora(
            project_id,
            character_id,
            input,
        )
    }

    pub fn update_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
        input: CharacterLoraUpdateInput,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).update_lora_link(
            project_id,
            character_id,
            link_id,
            input,
        )
    }

    pub fn detach_character_lora(
        &self,
        project_id: &str,
        character_id: &str,
        link_id: &str,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        CharacterStore::new(&self.data_dir, project_path).detach_lora(
            project_id,
            character_id,
            link_id,
        )
    }

    pub fn list_person_tracks(&self, project_id: &str) -> ProjectStoreResult<Vec<Value>> {
        let project_path = self.find_project_path(project_id)?;
        let folder = project_path.join("person-tracks");
        if !folder.exists() {
            return Ok(Vec::new());
        }

        let mut tracks = Vec::new();
        for entry in read_dir_paths(&folder)? {
            if !entry
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".sceneworks.person-track.json"))
            {
                continue;
            }
            if let Ok(track) = normalize_person_track(&project_path, &entry) {
                tracks.push(track);
            }
        }
        tracks.sort_by(|left, right| {
            right
                .get("createdAt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    left.get("createdAt")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        });
        Ok(tracks)
    }

    pub fn get_person_track(&self, project_id: &str, track_id: &str) -> ProjectStoreResult<Value> {
        if !is_safe_track_id(track_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let tracks_dir = project_path.join("person-tracks");
        let track_path = project_path
            .join("person-tracks")
            .join(format!("{track_id}.sceneworks.person-track.json"));
        if !track_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Person track not found".to_owned(),
            ));
        }
        let tracks_root = fs::canonicalize(&tracks_dir)?;
        let resolved_path = fs::canonicalize(&track_path)?;
        if !resolved_path.starts_with(&tracks_root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        normalize_person_track(&project_path, &track_path)
    }

    /// Replace the correction set on a person track (sc-1485). Each incoming
    /// correction targets a frame by index and either adjusts its box or rejects
    /// the frame; the server stamps author/createdAt/source so the sidecar always
    /// records who corrected it and when. The corrections array is the UI's full
    /// view, so it is written wholesale (a frame omitted from the set is treated
    /// as having no correction). Returns the normalized, updated track.
    pub fn set_person_track_corrections(
        &self,
        project_id: &str,
        track_id: &str,
        corrections: Vec<Value>,
    ) -> ProjectStoreResult<Value> {
        if !is_safe_track_id(track_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let tracks_dir = project_path.join("person-tracks");
        let track_path = tracks_dir.join(format!("{track_id}.sceneworks.person-track.json"));
        if !track_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Person track not found".to_owned(),
            ));
        }
        let tracks_root = fs::canonicalize(&tracks_dir)?;
        let resolved_path = fs::canonicalize(&track_path)?;
        if !resolved_path.starts_with(&tracks_root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid person track ID".to_owned(),
            ));
        }

        let mut track = read_json(&track_path)?;
        let frame_count = track
            .get("frames")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let mut normalized = normalize_person_track_corrections(corrections, frame_count)?;
        let has_corrections = !normalized.is_empty();
        normalized.sort_by_key(|correction| {
            correction
                .get("frameIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        });

        let object = track.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Person track sidecar must be an object".to_owned())
        })?;
        object.insert("corrections".to_owned(), Value::Array(normalized));
        let status = object
            .entry("status")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Person track status must be an object".to_owned())
            })?;
        status.insert(
            "correctionState".to_owned(),
            Value::String(
                if has_corrections {
                    "box_corrections_applied"
                } else {
                    "ready_for_box_corrections"
                }
                .to_owned(),
            ),
        );

        write_json(&track_path, &track)?;
        normalize_person_track(&project_path, &track_path)
    }

    pub fn import_asset(&self, project_id: &str, upload: UploadAsset) -> ProjectStoreResult<Value> {
        if fs::metadata(&upload.source_path)?.len() == 0 {
            return Err(ProjectStoreError::BadRequest(
                "Uploaded file is empty".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let upload_dir = project_path.join("assets").join("uploads");
        fs::create_dir_all(&upload_dir)?;

        let guessed_mime = guess_mime_from_filename(&upload.filename);
        let content_type = upload
            .content_type
            .as_deref()
            .filter(|value| !value.is_empty() && *value != "application/octet-stream")
            .map(str::to_owned)
            .or(guessed_mime)
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        if !content_type.starts_with("image/") && !content_type.starts_with("video/") {
            return Err(ProjectStoreError::BadRequest(
                "Only image and video uploads are supported".to_owned(),
            ));
        }

        let asset_id = format!("asset_{}", random_hex(16)?);
        let created_at = utc_now();
        let extension = upload_extension(&upload.filename, &content_type);
        let suffix = &asset_id[asset_id.len().saturating_sub(8)..];
        let filename = format!(
            "{}-{suffix}{extension}",
            safe_filename(&upload.filename, &asset_id)
        );
        let media_path = upload_dir.join(filename);
        move_or_copy_file(&upload.source_path, &media_path)?;
        let media_rel = relative_string(&project_path, &media_path)?;
        let display_name = Path::new(&upload.filename)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                media_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("upload")
            })
            .to_owned();

        let asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": project_id,
            "generationSetId": Value::Null,
            "type": media_type_for_mime(&content_type)?,
            "displayName": display_name,
            "createdAt": created_at,
            "file": {
                "path": media_rel,
                "mimeType": content_type,
                "width": Value::Null,
                "height": Value::Null,
                "duration": Value::Null,
                "fps": Value::Null
            },
            "status": {
                "favorite": false,
                "rating": 0,
                "rejected": false,
                "trashed": false
            },
            "recipe": {
                "mode": "upload",
                "model": "manual-import",
                "adapter": "api-upload",
                "prompt": upload.filename,
                "negativePrompt": "",
                "seed": 0,
                "loras": [],
                "stylePreset": "none",
                "normalizedSettings": {},
                "rawAdapterSettings": { "contentType": content_type }
            },
            "lineage": {
                "parents": [],
                "sourceAssetId": Value::Null,
                "sourceTimestamp": Value::Null,
                "jobId": Value::Null
            }
        });
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    /// Persist a generated asset the worker reported as flat facts: build the
    /// sidecar (Rust owns this schema now — story 1656), write it next to the
    /// media the worker already saved, write the recipe, and index project.db.
    /// Idempotent — re-applied progress updates upsert the same row/files.
    /// Returns the built sidecar so the API can re-inject it into the job result.
    pub fn persist_generated_asset(
        &self,
        project_id: &str,
        job_id: &str,
        generation_set_id: &str,
        fact: &Value,
    ) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let media_rel = fact
            .get("mediaPath")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("generated asset fact missing mediaPath".to_owned())
            })?;
        let asset_id = fact.get("assetId").and_then(Value::as_str).ok_or_else(|| {
            ProjectStoreError::BadRequest("generated asset fact missing assetId".to_owned())
        })?;
        let asset = build_generated_asset_sidecar(project_id, job_id, generation_set_id, fact);
        let media_path = project_path.join(media_rel);
        let sidecar_path = media_path.with_extension("sceneworks.json");
        if let Some(parent) = sidecar_path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_json(&sidecar_path, &asset)?;
        let recipes_dir = project_path.join("recipes");
        fs::create_dir_all(&recipes_dir)?;
        write_json(
            &recipes_dir.join(format!("{asset_id}.recipe.json")),
            &asset["recipe"],
        )?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        Ok(asset)
    }

    /// Write the generation-set JSON for a job from the worker-reported facts.
    /// Idempotent — overwrites the same `<id>.json` on re-applied updates.
    pub fn write_generation_set(
        &self,
        project_id: &str,
        job_id: &str,
        generation_set: &Value,
    ) -> ProjectStoreResult<()> {
        let project_path = self.find_project_path(project_id)?;
        let id = generation_set
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("generation set fact missing id".to_owned())
            })?;
        let get = |key: &str| generation_set.get(key).cloned().unwrap_or(Value::Null);
        let record = json!({
            "schemaVersion": 1,
            "id": id,
            "projectId": project_id,
            "jobId": job_id,
            "mode": get("mode"),
            "model": get("model"),
            "prompt": get("prompt"),
            "negativePrompt": get("negativePrompt"),
            "count": get("count"),
            "createdAt": get("createdAt"),
        });
        let dir = project_path.join("generation-sets");
        fs::create_dir_all(&dir)?;
        write_json(&dir.join(format!("{id}.json")), &record)?;
        Ok(())
    }

    pub fn update_asset_status(
        &self,
        project_id: &str,
        asset_id: &str,
        patch: AssetStatusPatch,
    ) -> ProjectStoreResult<Value> {
        if patch.rating.is_some_and(|rating| rating > 5) {
            return Err(ProjectStoreError::BadRequest(
                "Rating must be between 0 and 5".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let status = asset
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
            })?
            .entry("status")
            .or_insert_with(|| json!({}));
        let status = status.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset status must be an object".to_owned())
        })?;
        if let Some(value) = patch.favorite {
            status.insert("favorite".to_owned(), Value::Bool(value));
        }
        if let Some(value) = patch.rating {
            status.insert("rating".to_owned(), json!(value));
        }
        if let Some(value) = patch.rejected {
            status.insert("rejected".to_owned(), Value::Bool(value));
        }
        if let Some(value) = patch.trashed {
            status.insert("trashed".to_owned(), Value::Bool(value));
        }
        write_json(&sidecar_path, &asset)?;
        index_asset(&project_path, &asset, Some(&sidecar_path))?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    pub fn get_asset(&self, project_id: &str, asset_id: &str) -> ProjectStoreResult<Value> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        normalize_asset(project_id, &project_path, &sidecar_path)
    }

    pub fn index_asset_sidecar(
        &self,
        project_id: &str,
        sidecar_path: &Path,
    ) -> ProjectStoreResult<()> {
        let project_path = self.find_project_path(project_id)?;
        let asset = read_json(sidecar_path)?;
        index_asset(&project_path, &asset, Some(sidecar_path))
    }

    pub fn delete_asset(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<AssetMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let mut asset = read_json(&sidecar_path)?;
        let media_rel = asset
            .pointer("/file/path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let media_path = project_path.join(&media_rel);
        let status = asset
            .as_object_mut()
            .ok_or_else(|| {
                ProjectStoreError::BadRequest("Asset sidecar must be an object".to_owned())
            })?
            .entry("status")
            .or_insert_with(|| json!({}));
        let status = status.as_object_mut().ok_or_else(|| {
            ProjectStoreError::BadRequest("Asset status must be an object".to_owned())
        })?;
        status.insert("trashed".to_owned(), Value::Bool(true));

        let trash_dir = project_path.join("trash").join(asset_id);
        fs::create_dir_all(&trash_dir)?;
        if media_path.exists() && media_path.is_file() {
            let trashed_media_path = trash_dir.join(media_path.file_name().ok_or_else(|| {
                ProjectStoreError::BadRequest("Invalid asset media path".to_owned())
            })?);
            fs::rename(&media_path, &trashed_media_path)?;
            if let Some(file) = asset.get_mut("file").and_then(Value::as_object_mut) {
                file.insert(
                    "path".to_owned(),
                    Value::String(relative_string(&project_path, &trashed_media_path)?),
                );
            }
        }
        let trashed_sidecar_path = trash_dir.join(sidecar_path.file_name().ok_or_else(|| {
            ProjectStoreError::BadRequest("Invalid asset sidecar path".to_owned())
        })?);
        write_json(&trashed_sidecar_path, &asset)?;
        if sidecar_path != trashed_sidecar_path {
            fs::remove_file(&sidecar_path).ok();
        }
        index_asset(&project_path, &asset, Some(&trashed_sidecar_path))?;
        Ok(AssetMutationResult {
            id: asset_id.to_owned(),
            status: "trashed".to_owned(),
        })
    }

    pub fn purge_asset(
        &self,
        project_id: &str,
        asset_id: &str,
    ) -> ProjectStoreResult<AssetMutationResult> {
        let project_path = self.find_project_path(project_id)?;
        let sidecar_path = self.find_asset_sidecar(&project_path, asset_id)?;
        let asset = read_json(&sidecar_path)?;
        let media_path = project_path.join(
            asset
                .pointer("/file/path")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        if media_path.exists() && media_path.is_file() {
            fs::remove_file(media_path)?;
        }
        fs::remove_file(&sidecar_path).ok();
        let trash_dir = project_path.join("trash");
        if sidecar_path.parent().is_some_and(|parent| {
            parent.file_name().and_then(|name| name.to_str()) == Some(asset_id)
                && parent.parent() == Some(trash_dir.as_path())
        }) {
            if let Some(parent) = sidecar_path.parent() {
                fs::remove_dir_all(parent).ok();
            }
        }
        purge_asset_record(&project_path, asset_id)?;
        Ok(AssetMutationResult {
            id: asset_id.to_owned(),
            status: "purged".to_owned(),
        })
    }

    pub fn project_file(
        &self,
        project_id: &str,
        relative_path: &str,
    ) -> ProjectStoreResult<ProjectFile> {
        if !is_safe_relative_path(relative_path) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid project file path".to_owned(),
            ));
        }
        let project_path = self.find_project_path(project_id)?;
        let root = fs::canonicalize(&project_path)?;
        let target = project_path.join(relative_path);
        let target = fs::canonicalize(&target).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ProjectStoreError::NotFound("File not found".to_owned())
            } else {
                ProjectStoreError::Io(error)
            }
        })?;
        if !target.starts_with(&root) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid project file path".to_owned(),
            ));
        }
        if !target.is_file() {
            return Err(ProjectStoreError::NotFound("File not found".to_owned()));
        }
        let content_type = guess_mime_from_filename(&target.display().to_string())
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        Ok(ProjectFile {
            path: target,
            content_type,
        })
    }

    fn ensure_data_dirs(&self) -> ProjectStoreResult<()> {
        fs::create_dir_all(self.projects_dir())?;
        for folder in ["models", "loras", "cache"] {
            fs::create_dir_all(self.data_dir.join(folder))?;
        }
        Ok(())
    }

    fn load_registry(&self) -> ProjectStoreResult<Vec<RegistryItem>> {
        let registry_path = self.registry_path();
        if !registry_path.exists() {
            return Ok(Vec::new());
        }
        let payload = fs::read_to_string(registry_path)?;
        Ok(serde_json::from_str(&payload)?)
    }

    fn save_registry(&self, projects: &[RegistryItem]) -> ProjectStoreResult<()> {
        fs::create_dir_all(&self.data_dir)?;
        let tmp_path = self.registry_path().with_extension("json.tmp");
        write_json(&tmp_path, &serde_json::to_value(projects)?)?;
        fs::rename(tmp_path, self.registry_path())?;
        Ok(())
    }

    fn find_project_path(&self, project_id: &str) -> ProjectStoreResult<PathBuf> {
        let _guard = self.lock.lock();
        for item in self.load_registry()? {
            if item.id.as_deref() == Some(project_id) {
                let Some(path) = item.path else {
                    break;
                };
                let project_path = PathBuf::from(path);
                if project_path.exists() {
                    return Ok(project_path);
                }
                break;
            }
        }
        Err(ProjectStoreError::NotFound("Project not found".to_owned()))
    }

    fn find_asset_sidecar(
        &self,
        project_path: &Path,
        asset_id: &str,
    ) -> ProjectStoreResult<PathBuf> {
        find_asset_sidecar_path(project_path, asset_id)?
            .ok_or_else(|| ProjectStoreError::NotFound("Asset not found".to_owned()))
    }
}

#[derive(Default)]
struct ReindexCounts {
    assets: u32,
    generation_sets: u32,
    timelines: u32,
}

/// Bump whenever the project.db schema changes (new table, column, or index).
/// `apply_project_migrations` only re-runs its DDL when `PRAGMA user_version`
/// is behind this; forgetting to bump means an existing DB never gets the change.
const PROJECT_SCHEMA_VERSION: i64 = 1;

fn project_schema_version(connection: &Connection) -> ProjectStoreResult<i64> {
    Ok(connection.query_row("pragma user_version", [], |row| row.get(0))?)
}

pub fn apply_project_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    if project_schema_version(connection)? >= PROJECT_SCHEMA_VERSION {
        return Ok(());
    }
    connection.execute_batch(
        "
        create table if not exists project_metadata (
          key text primary key,
          value text not null
        );
        insert or ignore into project_metadata (key, value) values ('schemaVersion', '1');
        create table if not exists assets (
          id text primary key,
          type text not null,
          display_name text not null,
          file_path text not null,
          generation_set_id text,
          created_at text not null,
          favorite integer not null default 0,
          rating integer not null default 0,
          rejected integer not null default 0,
          trashed integer not null default 0
        );
        create table if not exists generation_sets (
          id text primary key,
          mode text not null,
          model text not null,
          prompt text not null,
          created_at text not null,
          job_id text
        );
        create table if not exists timelines (
          id text primary key,
          name text not null,
          file_path text not null,
          aspect_ratio text not null,
          width integer not null,
          height integer not null,
          fps integer not null,
          duration real not null default 0,
          created_at text not null,
          updated_at text not null
        );
        ",
    )?;
    ensure_column(connection, "assets", "sidecar_path", "text")?;
    apply_character_migrations(connection)?;
    apply_training_dataset_migrations(connection)?;
    // Pragma assignment cannot be parameterized; the version is a trusted const.
    connection.execute_batch(&format!("pragma user_version = {PROJECT_SCHEMA_VERSION}"))?;
    Ok(())
}

pub fn ensure_project_db_ready(project_path: &Path) -> ProjectStoreResult<()> {
    apply_project_migrations(&connect_project_db(project_path)?)
}

fn reindex_project_path(project_path: &Path) -> ProjectStoreResult<ReindexCounts> {
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    transaction.execute("delete from assets", [])?;
    transaction.execute("delete from generation_sets", [])?;
    transaction.execute("delete from timelines", [])?;
    clear_character_tables(&transaction)?;

    let mut counts = ReindexCounts::default();
    for sidecar_path in asset_sidecars(project_path)? {
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        if asset.get("id").is_none() || asset.pointer("/file/path").is_none() {
            continue;
        }
        let sidecar_rel = relative_string(project_path, &sidecar_path)?;
        index_asset_on_connection(&transaction, &asset, Some(&sidecar_rel))?;
        counts.assets += 1;
    }

    let generation_set_dir = project_path.join("generation-sets");
    for entry in read_dir_paths(&generation_set_dir)? {
        if entry.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(generation_set) = read_json(&entry) else {
            continue;
        };
        if generation_set.get("id").is_none() {
            continue;
        }
        transaction.execute(
            "
            insert or replace into generation_sets (id, mode, model, prompt, created_at, job_id)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            ",
            params![
                required_str(&generation_set, "id")?,
                optional_str(&generation_set, "mode").unwrap_or("unknown"),
                optional_str(&generation_set, "model").unwrap_or("unknown"),
                optional_str(&generation_set, "prompt").unwrap_or(""),
                optional_str(&generation_set, "createdAt").unwrap_or(""),
                optional_str(&generation_set, "jobId"),
            ],
        )?;
        counts.generation_sets += 1;
    }

    let timeline_dir = project_path.join("timelines");
    for entry in read_dir_paths(&timeline_dir)? {
        if !entry
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(timeline) = read_json(&entry) else {
            continue;
        };
        if timeline.get("id").is_none() {
            continue;
        }
        let rel_path = relative_string(project_path, &entry)?;
        transaction.execute(
            "
            insert or replace into timelines (
              id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                required_str(&timeline, "id")?,
                optional_str(&timeline, "name").unwrap_or("Timeline"),
                rel_path,
                optional_str(&timeline, "aspectRatio").unwrap_or("16:9"),
                optional_u64(&timeline, "width").unwrap_or(1280),
                optional_u64(&timeline, "height").unwrap_or(720),
                optional_u64(&timeline, "fps").unwrap_or(30),
                optional_f64(&timeline, "duration").unwrap_or(0.0),
                optional_str(&timeline, "createdAt").unwrap_or(""),
                optional_str(&timeline, "updatedAt")
                    .or_else(|| optional_str(&timeline, "createdAt"))
                    .unwrap_or(""),
            ],
        )?;
        counts.timelines += 1;
    }

    reindex_characters_on_connection(&transaction, project_path)?;

    transaction.commit()?;
    Ok(counts)
}

fn timeline_dimensions(aspect_ratio: &str) -> ProjectStoreResult<(u32, u32)> {
    match aspect_ratio {
        "16:9" => Ok((1280, 720)),
        "9:16" => Ok((720, 1280)),
        "1:1" => Ok((1024, 1024)),
        _ => Err(ProjectStoreError::BadRequest(
            "Aspect ratio must be one of 16:9, 9:16, or 1:1".to_owned(),
        )),
    }
}

fn default_timeline_tracks() -> Value {
    json!([
        {"id": "track_main", "name": "Main", "kind": "video", "locked": false, "muted": false, "items": []},
        {"id": "track_overlay", "name": "Overlay", "kind": "overlay", "locked": false, "muted": false, "items": []},
        {"id": "track_audio", "name": "Audio", "kind": "audio", "locked": false, "muted": false, "items": []}
    ])
}

fn compute_timeline_duration(timeline: &Value) -> f64 {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .filter_map(|item| item.get("timelineEnd").and_then(Value::as_f64))
        .fold(0.0, f64::max)
}

fn validate_and_default_timeline(timeline: &mut Value) -> ProjectStoreResult<()> {
    let object = timeline
        .as_object_mut()
        .ok_or_else(|| ProjectStoreError::BadRequest("Timeline must be an object".to_owned()))?;
    object
        .entry("schemaVersion".to_owned())
        .or_insert_with(|| json!(1));
    object
        .entry("aspectRatio".to_owned())
        .or_insert_with(|| Value::String("16:9".to_owned()));
    object
        .entry("width".to_owned())
        .or_insert_with(|| json!(1280));
    object
        .entry("height".to_owned())
        .or_insert_with(|| json!(720));
    object.entry("fps".to_owned()).or_insert_with(|| json!(30));
    object
        .entry("duration".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("tracks".to_owned())
        .or_insert_with(|| json!([]));
    object
        .entry("transitions".to_owned())
        .or_insert_with(|| json!([]));

    validate_u64_range(timeline, "schemaVersion", 0, u32::MAX as u64)?;
    validate_enum(timeline, "aspectRatio", &["16:9", "9:16", "1:1"])?;
    validate_u64_range(timeline, "width", 256, 3840)?;
    validate_u64_range(timeline, "height", 256, 3840)?;
    validate_u64_range(timeline, "fps", 1, 60)?;
    validate_f64_range(timeline, "duration", 0.0, f64::INFINITY)?;

    let tracks = timeline
        .get_mut("tracks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("tracks must be an array".to_owned()))?;
    for track in tracks {
        validate_timeline_track(track)?;
    }

    let transitions = timeline
        .get_mut("transitions")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("transitions must be an array".to_owned()))?;
    for transition in transitions {
        validate_timeline_transition(transition)?;
    }
    Ok(())
}

fn validate_timeline_track(track: &mut Value) -> ProjectStoreResult<()> {
    required_str(track, "id")?;
    required_str(track, "name")?;
    validate_enum(track, "kind", &["video", "overlay", "audio"])?;
    let object = track.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline track must be an object".to_owned())
    })?;
    object
        .entry("locked".to_owned())
        .or_insert_with(|| Value::Bool(false));
    object
        .entry("muted".to_owned())
        .or_insert_with(|| Value::Bool(false));
    object
        .entry("items".to_owned())
        .or_insert_with(|| json!([]));
    validate_bool(track, "locked")?;
    validate_bool(track, "muted")?;
    let items = track
        .get_mut("items")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ProjectStoreError::BadRequest("track items must be an array".to_owned()))?;
    for item in items {
        validate_timeline_item(item)?;
    }
    Ok(())
}

fn validate_timeline_item(item: &mut Value) -> ProjectStoreResult<()> {
    required_str(item, "trackId")?;
    validate_required_string(item, "assetId", Some(1), None)?;
    validate_required_string(item, "displayName", Some(1), Some(160))?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    if !object.contains_key("id") || object.get("id") == Some(&Value::Null) {
        object.insert(
            "id".to_owned(),
            Value::String(format!("item_{}", random_hex(16)?)),
        );
    }
    object
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("video".to_owned()));
    object
        .entry("sourceIn".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("sourceOut".to_owned())
        .or_insert_with(|| json!(4.0));
    object
        .entry("timelineStart".to_owned())
        .or_insert_with(|| json!(0.0));
    object
        .entry("timelineEnd".to_owned())
        .or_insert_with(|| json!(4.0));
    object
        .entry("speed".to_owned())
        .or_insert_with(|| json!(1.0));
    object
        .entry("fit".to_owned())
        .or_insert_with(|| Value::String("fit".to_owned()));
    object
        .entry("volume".to_owned())
        .or_insert_with(|| json!(1.0));
    object
        .entry("versionAssetIds".to_owned())
        .or_insert_with(|| json!([]));
    object
        .entry("versionHistory".to_owned())
        .or_insert_with(|| json!([]));

    validate_required_string(item, "id", Some(1), None)?;
    validate_enum(item, "type", &["video", "image", "audio"])?;
    let source_in = validate_f64_range(item, "sourceIn", 0.0, f64::INFINITY)?;
    let source_out = validate_f64_range(item, "sourceOut", 0.0, f64::INFINITY)?;
    let timeline_start = validate_f64_range(item, "timelineStart", 0.0, f64::INFINITY)?;
    let timeline_end = validate_f64_range(item, "timelineEnd", 0.0, f64::INFINITY)?;
    let speed = validate_f64_range(item, "speed", 0.1, 8.0)?;
    validate_enum(item, "fit", &["fit", "fill", "stretch"])?;
    let volume = validate_f64_range(item, "volume", 0.0, 2.0)?;
    let object = item.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item must be an object".to_owned())
    })?;
    object.insert("sourceIn".to_owned(), json!(source_in));
    object.insert("sourceOut".to_owned(), json!(source_out));
    object.insert("timelineStart".to_owned(), json!(timeline_start));
    object.insert("timelineEnd".to_owned(), json!(timeline_end));
    object.insert("speed".to_owned(), json!(speed));
    object.insert("volume".to_owned(), json!(volume));
    object
        .entry("transitionIn".to_owned())
        .or_insert(Value::Null);
    object
        .entry("transitionOut".to_owned())
        .or_insert(Value::Null);
    if source_out <= source_in {
        return Err(ProjectStoreError::BadRequest(
            "sourceOut must be greater than sourceIn.".to_owned(),
        ));
    }
    if timeline_end <= timeline_start {
        return Err(ProjectStoreError::BadRequest(
            "timelineEnd must be greater than timelineStart.".to_owned(),
        ));
    }

    item.get("versionAssetIds")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("versionAssetIds must be an array".to_owned())
        })?;
    let versions = item
        .get_mut("versionHistory")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            ProjectStoreError::BadRequest("versionHistory must be an array".to_owned())
        })?;
    for version in versions {
        validate_timeline_item_version(version)?;
    }
    if let Some(transition) = item
        .get_mut("transitionIn")
        .filter(|value| !value.is_null())
    {
        validate_timeline_transition(transition)?;
    }
    if let Some(transition) = item
        .get_mut("transitionOut")
        .filter(|value| !value.is_null())
    {
        validate_timeline_transition(transition)?;
    }
    Ok(())
}

fn validate_timeline_item_version(version: &mut Value) -> ProjectStoreResult<()> {
    validate_required_string(version, "assetId", Some(1), None)?;
    let object = version.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline item version must be an object".to_owned())
    })?;
    object
        .entry("source".to_owned())
        .or_insert_with(|| Value::String("manual".to_owned()));
    validate_enum(
        version,
        "source",
        &[
            "original",
            "replacement",
            "extension",
            "bridge",
            "restore",
            "manual",
        ],
    )
}

fn validate_timeline_transition(transition: &mut Value) -> ProjectStoreResult<()> {
    let object = transition.as_object_mut().ok_or_else(|| {
        ProjectStoreError::BadRequest("Timeline transition must be an object".to_owned())
    })?;
    if !object.contains_key("id") || object.get("id") == Some(&Value::Null) {
        object.insert(
            "id".to_owned(),
            Value::String(format!("transition_{}", random_hex(16)?)),
        );
    }
    object
        .entry("type".to_owned())
        .or_insert_with(|| Value::String("cut".to_owned()));
    object
        .entry("duration".to_owned())
        .or_insert_with(|| json!(0.0));
    validate_required_string(transition, "id", Some(1), None)?;
    validate_enum(
        transition,
        "type",
        &["cut", "crossfade", "fade_from_black", "fade_to_black"],
    )?;
    validate_f64_range(transition, "duration", 0.0, 10.0)?;
    Ok(())
}

fn normalize_timeline_items(timeline: &mut Value) -> ProjectStoreResult<()> {
    let Some(tracks) = timeline.get_mut("tracks").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for track in tracks {
        let Some(items) = track.get_mut("items").and_then(Value::as_array_mut) else {
            continue;
        };
        for item in items {
            let Some(object) = item.as_object_mut() else {
                continue;
            };
            let Some(asset_id) = object
                .get("assetId")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            object
                .entry("currentVersionAssetId".to_owned())
                .or_insert_with(|| Value::String(asset_id.clone()));
            let version_ids = object
                .entry("versionAssetIds".to_owned())
                .or_insert_with(|| json!([]));
            if let Some(version_ids) = version_ids.as_array_mut() {
                let has_asset = version_ids
                    .iter()
                    .any(|value| value.as_str() == Some(&asset_id));
                if !has_asset {
                    version_ids.push(Value::String(asset_id.clone()));
                }
            }
            let needs_history = object
                .get("versionHistory")
                .and_then(Value::as_array)
                .map_or(true, Vec::is_empty);
            if needs_history {
                object.insert(
                    "versionHistory".to_owned(),
                    json!([{
                        "assetId": asset_id,
                        "createdAt": Value::Null,
                        "source": "original",
                        "jobId": Value::Null,
                        "note": Value::Null
                    }]),
                );
            }
        }
    }
    Ok(())
}

fn validate_required_string(
    payload: &Value,
    key: &str,
    min_length: Option<usize>,
    max_length: Option<usize>,
) -> ProjectStoreResult<()> {
    let value = required_str(payload, key)?;
    let length = value.chars().count();
    if let Some(min) = min_length.filter(|min| length < *min) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at least {min} characters"
        )));
    }
    if let Some(max) = max_length.filter(|max| length > *max) {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be at most {max} characters"
        )));
    }
    Ok(())
}

fn validate_enum(payload: &Value, key: &str, allowed: &[&str]) -> ProjectStoreResult<()> {
    let value = required_str(payload, key)?;
    if !allowed.contains(&value) {
        return Err(ProjectStoreError::BadRequest(format!(
            "Unsupported value for {key}: {value}"
        )));
    }
    Ok(())
}

fn validate_bool(payload: &Value, key: &str) -> ProjectStoreResult<()> {
    if payload.get(key).and_then(Value::as_bool).is_none() {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be a boolean"
        )));
    }
    Ok(())
}

fn validate_u64_range(payload: &Value, key: &str, min: u64, max: u64) -> ProjectStoreResult<u64> {
    let value = payload
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{key} must be an integer")))?;
    if value < min || value > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(value)
}

fn validate_f64_range(payload: &Value, key: &str, min: f64, max: f64) -> ProjectStoreResult<f64> {
    let value = payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("{key} must be a number")))?;
    if !value.is_finite() || value < min || value > max {
        return Err(ProjectStoreError::BadRequest(format!(
            "{key} must be between {min} and {max}"
        )));
    }
    Ok(value)
}

fn timeline_file_path(project_path: &Path, timeline_id: &str, name: &str) -> PathBuf {
    let suffix = timeline_id
        .chars()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    project_path.join("timelines").join(format!(
        "{}-{suffix}.sceneworks.timeline.json",
        slugify(name, "timeline", Some(48))
    ))
}

fn index_timeline(project_path: &Path, timeline: &Value, rel_path: &str) -> ProjectStoreResult<()> {
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    transaction.execute(
        "
        insert or replace into timelines (
          id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ",
        params![
            required_str(timeline, "id")?,
            optional_str(timeline, "name").unwrap_or("Timeline"),
            rel_path,
            optional_str(timeline, "aspectRatio").unwrap_or("16:9"),
            optional_u64(timeline, "width").unwrap_or(1280),
            optional_u64(timeline, "height").unwrap_or(720),
            optional_u64(timeline, "fps").unwrap_or(30),
            optional_f64(timeline, "duration").unwrap_or(0.0),
            optional_str(timeline, "createdAt").unwrap_or(""),
            optional_str(timeline, "updatedAt")
                .or_else(|| optional_str(timeline, "createdAt"))
                .unwrap_or(""),
        ],
    )?;
    transaction.commit()?;
    Ok(())
}

fn find_timeline_file(project_path: &Path, timeline_id: &str) -> ProjectStoreResult<TimelineFile> {
    ensure_project_db_ready(project_path)?;
    let connection = connect_project_db(project_path)?;
    let indexed_path = connection
        .query_row(
            "select file_path from timelines where id = ?1",
            params![timeline_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if let Some(indexed_path) = indexed_path.as_deref() {
        let path = project_path.join(indexed_path);
        if path.exists() {
            return Ok(TimelineFile {
                path,
                relative_path: indexed_path.to_owned(),
            });
        }
    }

    let timeline_dir = project_path.join("timelines");
    for candidate in read_dir_paths(&timeline_dir)? {
        if !candidate
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        {
            continue;
        }
        let Ok(timeline) = read_json(&candidate) else {
            continue;
        };
        if timeline.get("id").and_then(Value::as_str) == Some(timeline_id) {
            let rel_path = relative_string(project_path, &candidate)?;
            // Single-statement autocommit keeps the stale-index repair atomic; if this grows
            // beyond one write, wrap the whole repair in an explicit transaction.
            connection.execute(
                "update timelines set file_path = ?1 where id = ?2",
                params![rel_path, timeline_id],
            )?;
            return Ok(TimelineFile {
                path: candidate,
                relative_path: rel_path,
            });
        }
    }

    if let Some(indexed_path) = indexed_path {
        return Err(ProjectStoreError::NotFound(format!(
            "Timeline file not found at indexed path {indexed_path}; reindex required"
        )));
    }
    Err(ProjectStoreError::NotFound("Timeline not found".to_owned()))
}

fn read_project_summary(project_path: &Path) -> ProjectStoreResult<ProjectSummary> {
    let project_file = project_path.join("project.json");
    if !project_file.exists() {
        return Err(ProjectStoreError::NotFound(
            "Project file not found".to_owned(),
        ));
    }
    let payload = read_json(&project_file)?;
    Ok(ProjectSummary {
        id: required_str(&payload, "id")?.to_owned(),
        name: required_str(&payload, "name")?.to_owned(),
        path: project_path.display().to_string(),
        created_at: required_str(&payload, "createdAt")?.to_owned(),
    })
}

fn write_project_file(
    app_version: &str,
    project_path: &Path,
    project_id: &str,
    name: &str,
) -> ProjectStoreResult<()> {
    let folders = PROJECT_FOLDERS
        .iter()
        .map(|folder| {
            (
                folder.replace('/', "_"),
                Value::String((*folder).to_owned()),
            )
        })
        .collect::<Map<_, _>>();
    write_json(
        &project_path.join("project.json"),
        &json!({
            "schemaVersion": 1,
            "appVersion": app_version,
            "id": project_id,
            "name": name,
            "createdAt": utc_now(),
            "folders": folders
        }),
    )
}

fn connect_project_db(project_path: &Path) -> ProjectStoreResult<Connection> {
    fs::create_dir_all(project_path)?;
    Ok(Connection::open(project_path.join("project.db"))?)
}

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> ProjectStoreResult<()> {
    let mut statement = connection.prepare(&format!("pragma table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !columns.iter().any(|existing| existing == column) {
        connection.execute(
            &format!("alter table {table} add column {column} {definition}"),
            [],
        )?;
    }
    Ok(())
}

/// Assemble the on-disk asset sidecar from the worker-reported flat facts. Rust
/// is the single owner of this envelope schema now (story 1656): the worker
/// ships values (paths, dimensions, seed, recipe inputs) and Rust builds the
/// `file`/`status`/`recipe`/`lineage` structure, matching the shape pinned by
/// `resource_sidecars.json` and the Python `build_asset_sidecar`. `type` is
/// derived from the mime so the video slice can reuse this.
fn build_generated_asset_sidecar(
    project_id: &str,
    job_id: &str,
    generation_set_id: &str,
    fact: &Value,
) -> Value {
    let get = |key: &str| fact.get(key).cloned().unwrap_or(Value::Null);
    let mime = fact
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("image/png");
    let media_type = if mime.starts_with("video/") {
        "video"
    } else {
        "image"
    };
    let parents = match fact.get("sourceAssetId").and_then(Value::as_str) {
        Some(source) => vec![Value::String(source.to_owned())],
        None => Vec::new(),
    };
    json!({
        "schemaVersion": 1,
        "id": get("assetId"),
        "projectId": project_id,
        "generationSetId": generation_set_id,
        "type": media_type,
        "displayName": get("displayName"),
        "createdAt": get("createdAt"),
        "file": {
            "path": get("mediaPath"),
            "mimeType": mime,
            "width": get("width"),
            "height": get("height"),
            "duration": get("duration"),
            "fps": get("fps"),
        },
        "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
        "recipe": {
            "mode": get("mode"),
            "model": get("model"),
            "adapter": get("adapter"),
            "prompt": get("prompt"),
            "negativePrompt": get("negativePrompt"),
            "seed": get("seed"),
            "loras": fact.get("loras").cloned().unwrap_or_else(|| json!([])),
            "stylePreset": get("stylePreset"),
            "normalizedSettings": {
                "width": get("normalizedWidth"),
                "height": get("normalizedHeight"),
                "count": get("count"),
                "family": get("family"),
                "characterId": get("characterId"),
                "characterLookId": get("characterLookId"),
                "characterConditioningActive": false,
                "characterConditioningNote": "Character metadata is recorded, but adapter-level character conditioning is not active in this build.",
            },
            "rawAdapterSettings": fact.get("rawAdapterSettings").cloned().unwrap_or_else(|| json!({})),
        },
        "lineage": {
            "parents": parents,
            "sourceAssetId": get("sourceAssetId"),
            "sourceTimestamp": Value::Null,
            "jobId": job_id,
        },
    })
}

fn index_asset(
    project_path: &Path,
    asset: &Value,
    sidecar_path: Option<&Path>,
) -> ProjectStoreResult<()> {
    let sidecar_path = match sidecar_path {
        Some(path) => path.to_path_buf(),
        None => project_path
            .join(required_str(
                asset.get("file").ok_or_else(|| {
                    ProjectStoreError::BadRequest("Asset file is required".to_owned())
                })?,
                "path",
            )?)
            .with_extension("sceneworks.json"),
    };
    let sidecar_rel = relative_string(project_path, &sidecar_path)?;
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    index_asset_on_connection(&transaction, asset, Some(&sidecar_rel))?;
    transaction.commit()?;
    Ok(())
}

fn index_asset_on_connection(
    connection: &Connection,
    asset: &Value,
    sidecar_rel: Option<&str>,
) -> ProjectStoreResult<()> {
    upsert_asset_row(connection, asset, sidecar_rel)
}

fn find_asset_record(
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<AssetRecord>> {
    ensure_project_db_ready(project_path)?;
    let connection = connect_project_db(project_path)?;
    connection
        .query_row(
            "select file_path, sidecar_path from assets where id = ?1",
            params![asset_id],
            row_to_asset_record,
        )
        .optional()
        .map_err(Into::into)
}

fn find_asset_sidecar_path(
    project_path: &Path,
    asset_id: &str,
) -> ProjectStoreResult<Option<PathBuf>> {
    if let Some(record) = find_asset_record(project_path, asset_id)? {
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

fn purge_asset_record(project_path: &Path, asset_id: &str) -> ProjectStoreResult<()> {
    let mut connection = connect_project_db(project_path)?;
    let transaction = connection.transaction()?;
    apply_project_migrations(&transaction)?;
    transaction.execute("delete from assets where id = ?1", params![asset_id])?;
    transaction.commit()?;
    Ok(())
}

fn project_has_sidecars(project_path: &Path) -> bool {
    asset_sidecars(project_path).is_ok_and(|paths| !paths.is_empty())
}

fn normalize_person_track(project_path: &Path, track_path: &Path) -> ProjectStoreResult<Value> {
    let mut track = read_json(track_path)?;
    let track_rel = relative_string(project_path, track_path)?;
    if let Some(object) = track.as_object_mut() {
        object.insert("path".to_owned(), Value::String(track_rel));
    }
    Ok(track)
}

/// Validate and canonicalize the incoming correction set for a person track.
/// Each entry targets an in-range frame and carries a box adjustment, a
/// rejection, or both; the server stamps author/createdAt/source. Entries that
/// neither adjust a box nor reject the frame are dropped (a reset frame).
/// Duplicate frame indices keep the last occurrence so the set is one entry per
/// frame.
fn normalize_person_track_corrections(
    corrections: Vec<Value>,
    frame_count: usize,
) -> ProjectStoreResult<Vec<Value>> {
    let created_at = utc_now();
    let mut normalized: Vec<(u64, Value)> = Vec::with_capacity(corrections.len());
    for correction in corrections {
        let object = correction.as_object().ok_or_else(|| {
            ProjectStoreError::BadRequest("Each correction must be an object".to_owned())
        })?;
        let frame_index = object
            .get("frameIndex")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "Correction frameIndex must be a non-negative integer".to_owned(),
                )
            })?;
        if frame_count == 0 || frame_index as usize >= frame_count {
            return Err(ProjectStoreError::BadRequest(format!(
                "Correction frameIndex {frame_index} is outside the track's {frame_count} frames"
            )));
        }
        let rejected = object
            .get("rejected")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let box_value = match object.get("box") {
            Some(Value::Null) | None => None,
            Some(value) => Some(validate_normalized_box(value)?),
        };
        if box_value.is_none() && !rejected {
            continue;
        }
        let author = object
            .get("author")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("ui")
            .to_owned();
        let source = object
            .get("source")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("manual")
            .to_owned();
        let mut entry = serde_json::Map::new();
        entry.insert("frameIndex".to_owned(), json!(frame_index));
        if let Some(box_value) = box_value {
            entry.insert("box".to_owned(), box_value);
        }
        entry.insert("rejected".to_owned(), Value::Bool(rejected));
        entry.insert("author".to_owned(), Value::String(author));
        entry.insert("source".to_owned(), Value::String(source));
        entry.insert("createdAt".to_owned(), Value::String(created_at.clone()));
        if let Some(slot) = normalized
            .iter_mut()
            .find(|(index, _)| *index == frame_index)
        {
            slot.1 = Value::Object(entry);
        } else {
            normalized.push((frame_index, Value::Object(entry)));
        }
    }
    Ok(normalized.into_iter().map(|(_, value)| value).collect())
}

/// Validate a normalized 0..1 bounding box and return a canonical copy holding
/// only x/y/width/height as finite numbers in range with positive extent.
fn validate_normalized_box(value: &Value) -> ProjectStoreResult<Value> {
    let object = value.as_object().ok_or_else(|| {
        ProjectStoreError::BadRequest("Correction box must be an object".to_owned())
    })?;
    let mut box_out = serde_json::Map::new();
    for key in ["x", "y", "width", "height"] {
        let component = object.get(key).and_then(Value::as_f64).ok_or_else(|| {
            ProjectStoreError::BadRequest(format!("Correction box.{key} must be a number"))
        })?;
        if !component.is_finite() || !(0.0..=1.0).contains(&component) {
            return Err(ProjectStoreError::BadRequest(format!(
                "Correction box.{key} must be between 0 and 1"
            )));
        }
        box_out.insert(key.to_owned(), json!(component));
    }
    let width = box_out.get("width").and_then(Value::as_f64).unwrap_or(0.0);
    let height = box_out.get("height").and_then(Value::as_f64).unwrap_or(0.0);
    if width <= 0.0 || height <= 0.0 {
        return Err(ProjectStoreError::BadRequest(
            "Correction box must have positive width and height".to_owned(),
        ));
    }
    Ok(Value::Object(box_out))
}

fn read_dir_paths(path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Into::into))
        .collect()
}

fn is_safe_track_id(track_id: &str) -> bool {
    !track_id.trim().is_empty()
        && track_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn required_str<'a>(payload: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    optional_str(payload, key)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("Missing required field: {key}")))
}

fn safe_filename(value: &str, fallback: &str) -> String {
    let name = value.replace('\\', "/");
    let stem = Path::new(name.rsplit('/').next().unwrap_or_default())
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let slug = slugify(stem, fallback, Some(64));
    if slug.is_empty() {
        fallback.to_owned()
    } else {
        slug
    }
}

fn media_type_for_mime(mime_type: &str) -> ProjectStoreResult<&'static str> {
    if mime_type.starts_with("image/") {
        return Ok("image");
    }
    if mime_type.starts_with("video/") {
        return Ok("video");
    }
    Err(ProjectStoreError::BadRequest(
        "Only image and video uploads are supported".to_owned(),
    ))
}

fn guess_mime_from_filename(filename: &str) -> Option<String> {
    mime_guess::from_path(filename)
        .first_raw()
        .map(str::to_owned)
        .or_else(|| {
            match Path::new(filename)
                .extension()
                .and_then(|value| value.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("heic") => Some("image/heic".to_owned()),
                Some("heif") => Some("image/heif".to_owned()),
                Some("avif") => Some("image/avif".to_owned()),
                Some("tif" | "tiff") => Some("image/tiff".to_owned()),
                _ => None,
            }
        })
}

fn upload_extension(filename: &str, mime_type: &str) -> String {
    if let Some(extension) = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
    {
        return format!(".{}", extension.to_ascii_lowercase());
    }
    match mime_guess::get_mime_extensions_str(mime_type).and_then(|extensions| extensions.first()) {
        Some(extension) => format!(".{extension}"),
        None => ".bin".to_owned(),
    }
}

fn move_or_copy_file(source: &Path, destination: &Path) -> ProjectStoreResult<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, destination)?;
            fs::remove_file(source)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_generated_asset_sidecar, guess_mime_from_filename, is_safe_relative_path,
        ProjectStore, PROJECT_FOLDERS,
    };
    use serde_json::{json, Value};

    #[test]
    fn build_generated_asset_sidecar_assembles_the_contract_schema() {
        // sc-1656: Rust owns the generated-asset sidecar schema now. Pin the
        // envelope it assembles from worker facts to the shared contract
        // (resource_sidecars.json imageAsset + recipe required keys) and the
        // load-bearing field mappings.
        let fact = json!({
            "assetId": "asset_abc",
            "mediaPath": "assets/images/genset_x/2026-05-25_z_image_turbo_city_0001.png",
            "mimeType": "image/png",
            "width": 1280,
            "height": 768,
            "normalizedWidth": 1024,
            "normalizedHeight": 768,
            "count": 1,
            "family": "z-image",
            "seed": 42,
            "index": 0,
            "displayName": "city #1",
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "city",
            "negativePrompt": "",
            "loras": [],
            "stylePreset": "none",
            "characterId": Value::Null,
            "characterLookId": Value::Null,
            "sourceAssetId": Value::Null,
            "rawAdapterSettings": {"steps": 8},
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_x", &fact);

        for key in [
            "schemaVersion",
            "id",
            "projectId",
            "generationSetId",
            "type",
            "displayName",
            "createdAt",
            "file",
            "status",
            "recipe",
            "lineage",
        ] {
            assert!(asset.get(key).is_some(), "missing top-level key {key}");
        }
        for key in [
            "mode",
            "model",
            "adapter",
            "prompt",
            "negativePrompt",
            "seed",
            "loras",
            "normalizedSettings",
            "rawAdapterSettings",
        ] {
            assert!(
                asset["recipe"].get(key).is_some(),
                "missing recipe key {key}"
            );
        }
        assert_eq!(asset["type"], json!("image"));
        assert_eq!(asset["id"], json!("asset_abc"));
        assert_eq!(asset["projectId"], json!("project-1"));
        assert_eq!(asset["generationSetId"], json!("genset_x"));
        assert_eq!(asset["file"]["path"], fact["mediaPath"]);
        assert_eq!(asset["file"]["width"], json!(1280));
        assert_eq!(asset["recipe"]["adapter"], json!("z_image_diffusers"));
        assert_eq!(asset["recipe"]["seed"], json!(42));
        assert_eq!(asset["recipe"]["normalizedSettings"]["width"], json!(1024));
        assert_eq!(
            asset["recipe"]["normalizedSettings"]["family"],
            json!("z-image")
        );
        assert_eq!(asset["lineage"]["jobId"], json!("job-1"));
    }

    #[test]
    fn build_generated_asset_sidecar_derives_video_type_from_mime() {
        let fact = json!({
            "assetId": "asset_v",
            "mediaPath": "assets/videos/genset_y/clip.mp4",
            "mimeType": "video/mp4",
        });
        let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_y", &fact);
        assert_eq!(asset["type"], json!("video"));
        assert_eq!(asset["file"]["mimeType"], json!("video/mp4"));
    }

    #[test]
    fn project_create_writes_python_compatible_files_and_registry() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

        let project = store.create_project("My Project").expect("project creates");

        assert!(project.id.starts_with("project_"));
        assert!(project.path.ends_with("my-project.sceneworks"));
        for folder in PROJECT_FOLDERS {
            assert!(std::path::Path::new(&project.path).join(folder).exists());
        }
        let registry = std::fs::read_to_string(temp_dir.path().join("data/recent-projects.json"))
            .expect("registry reads");
        assert!(registry.contains(&project.id));
        assert!(std::path::Path::new(&project.path)
            .join("project.db")
            .exists());
    }

    #[test]
    fn reindex_rebuilds_python_project_tables() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let image_dir = project_path.join("assets/images");
        std::fs::write(image_dir.join("image.png"), b"image").expect("image writes");
        std::fs::write(
            image_dir.join("image.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "asset-1",
                "type": "image",
                "displayName": "Image",
                "createdAt": "2026-05-17T00:00:00Z",
                "generationSetId": "genset-1",
                "file": {"path": "assets/images/image.png"},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");
        std::fs::write(
            project_path.join("generation-sets/genset-1.json"),
            serde_json::to_string_pretty(&json!({
                "id": "genset-1",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": "test",
                "createdAt": "2026-05-17T00:00:00Z",
                "jobId": "job-1"
            }))
            .expect("json"),
        )
        .expect("genset writes");
        std::fs::write(
            project_path.join("timelines/main.sceneworks.timeline.json"),
            serde_json::to_string_pretty(&json!({
                "id": "timeline-1",
                "name": "Main",
                "aspectRatio": "16:9",
                "width": 1280,
                "height": 720,
                "fps": 30,
                "duration": 3.5,
                "createdAt": "2026-05-17T00:00:00Z",
                "updatedAt": "2026-05-17T00:00:00Z"
            }))
            .expect("json"),
        )
        .expect("timeline writes");

        let counts = store.reindex_project(&project.id).expect("reindex works");

        assert_eq!(counts.assets, 1);
        assert_eq!(counts.generation_sets, 1);
        assert_eq!(counts.timelines, 1);
    }

    #[test]
    fn document_assets_are_indexed_and_listed() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Docs").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);

        // assets/documents is created at project init (PROJECT_FOLDERS).
        assert!(project_path.join("assets/documents").exists());

        let document_dir = project_path.join("assets/documents");
        std::fs::write(
            document_dir.join("doc_1.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "doc_1",
                "projectId": project.id,
                "jobId": "job-1",
                "model": "sensenova_u1_8b",
                "prompt": "illustrated guide",
                "createdAt": "2026-05-23T00:00:00Z",
                "segments": [
                    {"type": "text", "text": "Step one."},
                    {"type": "image", "assetId": "asset-img-1", "path": "assets/images/a.png"}
                ]
            }))
            .expect("json"),
        )
        .expect("document writes");
        std::fs::write(
            document_dir.join("doc_1.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "doc_1",
                "type": "document",
                "displayName": "Illustrated guide",
                "createdAt": "2026-05-23T00:00:00Z",
                "file": {"path": "assets/documents/doc_1.json"},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");

        let counts = store.reindex_project(&project.id).expect("reindex works");
        assert_eq!(counts.assets, 1);

        let assets = store
            .list_assets(&project.id, false, false)
            .expect("assets list");
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["id"], "doc_1");
        assert_eq!(assets[0]["type"], "document");
    }

    #[test]
    fn person_tracks_list_and_detail_match_python_sidecar_shape() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        std::fs::write(
            project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "track_1",
                "projectId": project.id,
                "name": "Hero",
                "createdAt": "2026-05-17T00:00:00Z",
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "frames": [],
                "status": {}
            }))
            .expect("json"),
        )
        .expect("track sidecar writes");

        let tracks = store.list_person_tracks(&project.id).expect("tracks list");
        assert_eq!(tracks[0]["id"], "track_1");
        assert_eq!(
            tracks[0]["path"],
            "person-tracks/track_1.sceneworks.person-track.json"
        );

        let track = store
            .get_person_track(&project.id, "track_1")
            .expect("track detail");
        assert_eq!(track["name"], "Hero");
        assert!(store.get_person_track(&project.id, "../track_1").is_err());
        assert!(store.get_person_track(&project.id, "track~1").is_err());
    }

    #[test]
    fn person_track_corrections_persist_validate_and_clear() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
        let project = store.create_project("Fixture").expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        std::fs::write(
            project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "track_1",
                "projectId": project.id,
                "name": "Hero",
                "createdAt": "2026-05-17T00:00:00Z",
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "frames": [
                    {"timestamp": 0.0, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.5}},
                    {"timestamp": 0.5, "box": {"x": 0.3, "y": 0.1, "width": 0.2, "height": 0.5}}
                ],
                "corrections": [],
                "status": {"correctionState": "ready_for_box_corrections"}
            }))
            .expect("json"),
        )
        .expect("track sidecar writes");

        // A box adjustment and a rejection persist; the server stamps metadata and
        // drops the trailing no-op entry (no box, not rejected) for frame 0.
        let updated = store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![
                    json!({"frameIndex": 0, "box": {"x": 0.5, "y": 0.2, "width": 0.25, "height": 0.4}, "author": "editor"}),
                    json!({"frameIndex": 1, "rejected": true}),
                    json!({"frameIndex": 0, "rejected": false}),
                ],
            )
            .expect("corrections persist");
        let corrections = updated["corrections"]
            .as_array()
            .expect("corrections array");
        assert_eq!(corrections.len(), 2);
        assert_eq!(corrections[0]["frameIndex"], 0);
        assert_eq!(corrections[0]["box"]["x"], 0.5);
        assert_eq!(corrections[0]["author"], "editor");
        assert_eq!(corrections[0]["source"], "manual");
        assert!(corrections[0]["createdAt"].is_string());
        assert_eq!(corrections[1]["frameIndex"], 1);
        assert_eq!(corrections[1]["rejected"], true);
        assert_eq!(
            updated["status"]["correctionState"],
            "box_corrections_applied"
        );

        // Out-of-range frame indices and out-of-bounds boxes are rejected.
        assert!(store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![json!({"frameIndex": 9, "rejected": true})]
            )
            .is_err());
        assert!(store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![json!({"frameIndex": 0, "box": {"x": 1.5, "y": 0.0, "width": 0.2, "height": 0.2}})]
            )
            .is_err());

        // Clearing corrections returns the track to the ready state.
        let cleared = store
            .set_person_track_corrections(&project.id, "track_1", vec![])
            .expect("corrections clear");
        assert!(cleared["corrections"].as_array().expect("array").is_empty());
        assert_eq!(
            cleared["status"]["correctionState"],
            "ready_for_box_corrections"
        );
    }

    #[test]
    fn project_file_paths_reject_traversal_and_backslashes() {
        assert!(is_safe_relative_path("assets/images/image.png"));
        assert!(!is_safe_relative_path("../outside.txt"));
        assert!(!is_safe_relative_path("assets\\..\\outside.txt"));
        assert!(!is_safe_relative_path("/absolute/path.png"));
    }

    #[test]
    fn mime_guess_covers_modern_image_uploads() {
        assert_eq!(
            guess_mime_from_filename("reference.heic").as_deref(),
            Some("image/heic")
        );
        assert_eq!(
            guess_mime_from_filename("reference.avif").as_deref(),
            Some("image/avif")
        );
    }
}
