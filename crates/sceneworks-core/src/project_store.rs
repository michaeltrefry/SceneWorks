use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

pub const ASSET_SIDECAR_PATTERN: &str = "*.sceneworks.json";
pub const PROJECT_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "characters",
    "generation-sets",
    "loras",
    "person-tracks",
    "recipes",
    "timelines",
    "trash",
    "cache",
];
const ASSET_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "trash",
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
    pub bytes: Vec<u8>,
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

    pub fn import_asset(&self, project_id: &str, upload: UploadAsset) -> ProjectStoreResult<Value> {
        if upload.bytes.is_empty() {
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
            .or(guessed_mime)
            .unwrap_or("application/octet-stream")
            .to_owned();
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
        fs::write(&media_path, &upload.bytes)?;
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
            .unwrap_or("application/octet-stream")
            .to_owned();
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

#[derive(Debug)]
struct AssetRecord {
    file_path: Option<String>,
    sidecar_path: Option<String>,
}

pub fn apply_project_migrations(connection: &Connection) -> ProjectStoreResult<()> {
    connection.execute_batch(
        "
        create table if not exists project_metadata (
          key text primary key,
          value text not null
        );
        insert or replace into project_metadata (key, value) values ('schemaVersion', '1');
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
    Ok(())
}

pub fn ensure_project_db_ready(project_path: &Path) -> ProjectStoreResult<()> {
    apply_project_migrations(&connect_project_db(project_path)?)
}

pub fn reindex_project_path(project_path: &Path) -> ProjectStoreResult<ReindexResult> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    connection.execute("delete from assets", [])?;
    connection.execute("delete from generation_sets", [])?;
    connection.execute("delete from timelines", [])?;

    let mut counts = ReindexCounts::default();
    for sidecar_path in asset_sidecars(project_path)? {
        let Ok(asset) = read_json(&sidecar_path) else {
            continue;
        };
        if asset.get("id").is_none() || asset.pointer("/file/path").is_none() {
            continue;
        }
        let sidecar_rel = relative_string(project_path, &sidecar_path)?;
        index_asset_on_connection(&connection, &asset, Some(&sidecar_rel))?;
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
        connection.execute(
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
        connection.execute(
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

    Ok(ReindexResult {
        project_id: String::new(),
        assets: counts.assets,
        generation_sets: counts.generation_sets,
        timelines: counts.timelines,
    })
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
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    index_asset_on_connection(&connection, asset, Some(&sidecar_rel))
}

fn index_asset_on_connection(
    connection: &Connection,
    asset: &Value,
    sidecar_rel: Option<&str>,
) -> ProjectStoreResult<()> {
    let status = asset.get("status").unwrap_or(&Value::Null);
    connection.execute(
        "
        insert or replace into assets (
          id, type, display_name, file_path, generation_set_id, created_at,
          favorite, rating, rejected, trashed, sidecar_path
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        ",
        params![
            required_str(asset, "id")?,
            required_str(asset, "type")?,
            required_str(asset, "displayName")?,
            asset
                .get("file")
                .and_then(|file| optional_str(file, "path"))
                .ok_or_else(|| ProjectStoreError::BadRequest(
                    "Asset file path is required".to_owned()
                ))?,
            optional_str(asset, "generationSetId"),
            required_str(asset, "createdAt")?,
            optional_bool(status, "favorite").unwrap_or(false),
            optional_u64(status, "rating").unwrap_or(0),
            optional_bool(status, "rejected").unwrap_or(false),
            optional_bool(status, "trashed").unwrap_or(false),
            sidecar_rel,
        ],
    )?;
    Ok(())
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

fn row_to_asset_record(row: &Row<'_>) -> rusqlite::Result<AssetRecord> {
    Ok(AssetRecord {
        file_path: row.get(0)?,
        sidecar_path: row.get(1)?,
    })
}

fn purge_asset_record(project_path: &Path, asset_id: &str) -> ProjectStoreResult<()> {
    let connection = connect_project_db(project_path)?;
    apply_project_migrations(&connection)?;
    connection.execute("delete from assets where id = ?1", params![asset_id])?;
    Ok(())
}

fn project_has_sidecars(project_path: &Path) -> bool {
    asset_sidecars(project_path).is_ok_and(|paths| !paths.is_empty())
}

fn asset_sidecars(project_path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    let mut sidecars = Vec::new();
    for folder in ASSET_FOLDERS {
        collect_sidecars(&project_path.join(folder), &mut sidecars)?;
    }
    let timeline_dir = project_path.join("timelines");
    sidecars.retain(|path| !path.starts_with(&timeline_dir));
    Ok(sidecars)
}

fn collect_sidecars(path: &Path, sidecars: &mut Vec<PathBuf>) -> ProjectStoreResult<()> {
    if !path.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_sidecars(&path, sidecars)?;
        } else if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(".sceneworks.json"))
        {
            sidecars.push(path);
        }
    }
    Ok(())
}

fn normalize_asset(
    project_id: &str,
    project_path: &Path,
    sidecar_path: &Path,
) -> ProjectStoreResult<Value> {
    let mut asset = read_json(sidecar_path)?;
    if let Some(path) = asset.pointer("/file/path").and_then(Value::as_str) {
        let normalized_path = path.replace('\\', "/");
        if let Some(object) = asset.as_object_mut() {
            object.insert(
                "url".to_owned(),
                Value::String(format!(
                    "/api/v1/projects/{project_id}/files/{normalized_path}"
                )),
            );
        }
    }
    let sidecar_rel = relative_string(project_path, sidecar_path)?;
    if let Some(object) = asset.as_object_mut() {
        object.insert("sidecarPath".to_owned(), Value::String(sidecar_rel));
    }
    Ok(asset)
}

fn read_json(path: &Path) -> ProjectStoreResult<Value> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

fn write_json(path: &Path, payload: &Value) -> ProjectStoreResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = serde_json::to_string_pretty(payload)?;
    output.push('\n');
    fs::write(path, output)?;
    Ok(())
}

fn read_dir_paths(path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Into::into))
        .collect()
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
        && Path::new(relative_path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

fn required_str<'a>(payload: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    optional_str(payload, key)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("Missing required field: {key}")))
}

fn optional_str<'a>(payload: &'a Value, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(Value::as_str)
}

fn optional_bool(payload: &Value, key: &str) -> Option<bool> {
    payload.get(key).and_then(Value::as_bool)
}

fn optional_u64(payload: &Value, key: &str) -> Option<u64> {
    payload.get(key).and_then(Value::as_u64)
}

fn optional_f64(payload: &Value, key: &str) -> Option<f64> {
    payload.get(key).and_then(Value::as_f64)
}

fn slugify(value: &str, fallback: &str, max_length: Option<usize>) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash && !slug.is_empty() {
            slug.push('-');
            previous_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = fallback.to_owned();
    }
    if let Some(max_length) = max_length {
        slug.truncate(max_length);
    }
    slug
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

fn guess_mime_from_filename(filename: &str) -> Option<&'static str> {
    match Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("png") => Some("image/png"),
        Some("gif") => Some("image/gif"),
        Some("webp") => Some("image/webp"),
        Some("bmp") => Some("image/bmp"),
        Some("svg") => Some("image/svg+xml"),
        Some("mp4") => Some("video/mp4"),
        Some("mov") => Some("video/quicktime"),
        Some("webm") => Some("video/webm"),
        Some("mkv") => Some("video/x-matroska"),
        Some("avi") => Some("video/x-msvideo"),
        _ => None,
    }
}

fn upload_extension(filename: &str, mime_type: &str) -> String {
    if let Some(extension) = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
    {
        return format!(".{}", extension.to_ascii_lowercase());
    }
    match mime_type {
        "image/jpeg" => ".jpg",
        "image/png" => ".png",
        "image/gif" => ".gif",
        "image/webp" => ".webp",
        "video/mp4" => ".mp4",
        "video/quicktime" => ".mov",
        "video/webm" => ".webm",
        _ => ".bin",
    }
    .to_owned()
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

#[cfg(test)]
mod tests {
    use super::{ProjectStore, PROJECT_FOLDERS};
    use serde_json::json;

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
}
