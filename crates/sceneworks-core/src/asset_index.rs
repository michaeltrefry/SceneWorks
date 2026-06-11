use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension, Row};
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
    "assets/poses",
    "trash",
];

#[derive(Debug)]
pub(crate) struct AssetRecord {
    pub(crate) file_path: Option<String>,
    pub(crate) sidecar_path: Option<String>,
}

/// Resolve an asset id to its on-disk sidecar path: prefer the indexed
/// `assets` row's recorded sidecar/file paths, falling back to a scan of all
/// sidecars matching by `id`. Shared by `project_store` and `character_store`,
/// which previously kept byte-identical copies that could drift (sc-4272 /
/// F-CORE-12). Takes a ready `Connection` so each store keeps its own DB-prep
/// (ensure-ready vs migrate) in its thin wrapper.
pub(crate) fn find_asset_sidecar_path_on_connection(
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

/// Upsert an asset's index row, deriving the stored relative sidecar path from
/// its absolute path. Shared by both stores (sc-4272 / F-CORE-12).
pub(crate) fn index_asset_on_connection(
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

/// The studio / feature an asset originated from. Drives Asset Library hygiene
/// (sc-2024): the Library shows only studio-generated and uploaded media, never
/// Character Studio test outputs (those live under the character). Returns an
/// explicit `origin` when the sidecar carries one, otherwise derives it from the
/// recipe mode + asset type so legacy sidecars (written before the field existed)
/// still classify correctly on read and on reindex.
pub(crate) fn asset_origin(asset: &Value) -> String {
    if let Some(origin) = asset.get("origin").and_then(Value::as_str) {
        if !origin.is_empty() {
            return origin.to_owned();
        }
    }
    let mode = asset
        .get("recipe")
        .and_then(|recipe| recipe.get("mode"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let asset_type = asset
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    derive_origin(mode, asset_type).to_owned()
}

/// Classify an asset's origin from its generation `mode` and media `type`.
/// `character_image` (the Character Studio test mode) and `upload` (manual
/// import) are mode-driven; everything else maps to the studio that produces
/// that media type.
pub(crate) fn derive_origin(mode: &str, asset_type: &str) -> &'static str {
    match mode {
        "character_image" => "character_studio",
        "upload" => "upload",
        _ => match asset_type {
            "video" => "video_studio",
            "document" => "document_studio",
            _ => "image_studio",
        },
    }
}

/// Caches generation-set JSON by id within a single batch normalize so each
/// `generation-sets/<id>.json` is read+parsed at most once instead of once per
/// asset in the set. `None` records a missing/unreadable set so it isn't probed
/// again. Sized for a `list_assets` call; single-asset GETs use a throwaway one
/// (sc-4270 / F-CORE-10).
#[derive(Default)]
pub(crate) struct GenerationSetCache {
    entries: HashMap<String, Option<Value>>,
}

pub(crate) fn normalize_asset(
    project_id: &str,
    project_path: &Path,
    sidecar_path: &Path,
) -> ProjectStoreResult<Value> {
    normalize_asset_cached(
        project_id,
        project_path,
        sidecar_path,
        &mut GenerationSetCache::default(),
    )
}

/// Like [`normalize_asset`] but resolves the embedded `generationSet` through a
/// caller-owned [`GenerationSetCache`], so a batch listing reads each
/// generation-set file once rather than once per asset (sc-4270 / F-CORE-10).
pub(crate) fn normalize_asset_cached(
    project_id: &str,
    project_path: &Path,
    sidecar_path: &Path,
    generation_sets: &mut GenerationSetCache,
) -> ProjectStoreResult<Value> {
    let mut asset = read_json(sidecar_path)?;
    // Guarantee every API response carries an `origin`, even for legacy sidecars
    // written before the field existed (sc-2024).
    let origin = asset_origin(&asset);
    if let Some(object) = asset.as_object_mut() {
        object
            .entry("origin".to_owned())
            .or_insert_with(|| Value::String(origin));
    }
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
    if let Some(generation_set_id) = asset.get("generationSetId").and_then(Value::as_str) {
        let cached = generation_sets
            .entries
            .entry(generation_set_id.to_owned())
            .or_insert_with(|| {
                let generation_set_path = project_path
                    .join("generation-sets")
                    .join(format!("{generation_set_id}.json"));
                read_json(&generation_set_path).ok()
            });
        if let Some(generation_set) = cached.clone() {
            if let Some(object) = asset.as_object_mut() {
                object.insert("generationSet".to_owned(), generation_set);
            }
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
          favorite, rating, rejected, trashed, sidecar_path, origin
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
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
            asset_origin(asset),
        ],
    )?;
    Ok(())
}

fn required_str<'a>(asset: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    optional_str(asset, key)
        .ok_or_else(|| ProjectStoreError::BadRequest(format!("Missing required field: {key}")))
}

#[cfg(test)]
mod origin_tests {
    use super::{asset_origin, derive_origin};
    use serde_json::json;

    #[test]
    fn derives_character_studio_from_character_image_mode() {
        assert_eq!(
            derive_origin("character_image", "image"),
            "character_studio"
        );
        // Mode wins over media type so a character test frame still classifies.
        assert_eq!(
            derive_origin("character_image", "video"),
            "character_studio"
        );
    }

    #[test]
    fn derives_upload_from_upload_mode() {
        assert_eq!(derive_origin("upload", "image"), "upload");
    }

    #[test]
    fn derives_studio_origin_by_media_type() {
        assert_eq!(derive_origin("text_to_image", "image"), "image_studio");
        assert_eq!(derive_origin("image_to_video", "video"), "video_studio");
        assert_eq!(derive_origin("interleave", "document"), "document_studio");
    }

    #[test]
    fn asset_origin_prefers_explicit_field() {
        let asset = json!({
            "type": "image",
            "origin": "character_studio",
            "recipe": { "mode": "text_to_image" },
        });
        assert_eq!(asset_origin(&asset), "character_studio");
    }

    #[test]
    fn asset_origin_derives_when_field_absent_or_empty() {
        // Legacy sidecar: no origin field, classified by recipe mode.
        let legacy = json!({ "type": "image", "recipe": { "mode": "character_image" } });
        assert_eq!(asset_origin(&legacy), "character_studio");
        // Empty origin falls back to derivation rather than returning "".
        let empty =
            json!({ "type": "video", "origin": "", "recipe": { "mode": "image_to_video" } });
        assert_eq!(asset_origin(&empty), "video_studio");
    }
}
