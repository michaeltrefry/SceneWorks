use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, Row};
use serde_json::Value;

use crate::project_store::{ProjectStoreError, ProjectStoreResult};
use crate::store_util::{optional_bool, optional_str, optional_u64, read_json, relative_string};

pub(crate) const ASSET_SIDECAR_PATTERN: &str = "*.sceneworks.json";

pub(crate) const ASSET_FOLDERS: &[&str] = &[
    "assets/images",
    "assets/videos",
    "assets/uploads",
    "assets/frames",
    "assets/renders",
    "assets/documents",
    "trash",
];

#[derive(Debug)]
pub(crate) struct AssetRecord {
    pub(crate) file_path: Option<String>,
    pub(crate) sidecar_path: Option<String>,
}

pub(crate) fn row_to_asset_record(row: &Row<'_>) -> rusqlite::Result<AssetRecord> {
    Ok(AssetRecord {
        file_path: row.get(0)?,
        sidecar_path: row.get(1)?,
    })
}

pub(crate) fn asset_sidecars(project_path: &Path) -> ProjectStoreResult<Vec<PathBuf>> {
    let mut sidecars = Vec::new();
    for folder in ASSET_FOLDERS {
        collect_sidecars(&project_path.join(folder), &mut sidecars)?;
    }
    let timeline_dir = project_path.join("timelines");
    sidecars.retain(|path| !path.starts_with(&timeline_dir));
    Ok(sidecars)
}

pub(crate) fn collect_sidecars(path: &Path, sidecars: &mut Vec<PathBuf>) -> ProjectStoreResult<()> {
    if !path.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_sidecars(&path, sidecars)?;
        } else if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(ASSET_SIDECAR_PATTERN.trim_start_matches('*')))
        {
            sidecars.push(path);
        }
    }
    Ok(())
}

pub(crate) fn normalize_asset(
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

pub(crate) fn upsert_asset_row(
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

fn required_str<'a>(asset: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    optional_str(asset, key)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("Missing required field: {key}")))
}
