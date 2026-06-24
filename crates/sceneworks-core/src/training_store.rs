use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::dataset_quality::{
    caption_hash, CachedTier0Scalars, DatasetEmbeddings, QualityAck, QualityCheck,
};
use crate::project_store::{apply_project_migrations, ProjectStoreError, ProjectStoreResult};
use crate::store_util::{
    atomic_write, ensure_column, is_safe_id, is_safe_relative_path, parse_string_enum, random_hex,
    read_json, relative_string, write_json,
};
use crate::time::utc_now;
use crate::training::{
    Caption, CaptionSource, TrainingDataset, TrainingDatasetItem, TrainingDatasetStatus,
    TrainingModality, TRAINING_CONTRACT_SCHEMA_VERSION,
};

const DATASET_MANIFEST_NAME: &str = "dataset.sceneworks.training-dataset.json";
// Tier-1 CLIP embeddings sidecar (sc-6535) — kept out of the manifest (768×f32 per item is too big
// to inline like tier0_scalars); content-hash-keyed so it survives dataset edits.
const DATASET_EMBEDDINGS_NAME: &str = "dataset.sceneworks.embeddings.json";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetCreateInput {
    pub name: String,
    #[serde(default)]
    pub modality: Option<TrainingModality>,
    #[serde(default)]
    pub status: Option<TrainingDatasetStatus>,
    /// Optional owning character (sc-2022). `None` creates a general dataset.
    #[serde(default)]
    pub character_id: Option<String>,
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
    /// Associate the dataset with a character (sc-2022) when present. `Some`
    /// sets/changes the owning character; absent leaves it unchanged.
    #[serde(default)]
    pub character_id: Option<String>,
    /// Full replacement for the dataset's ordered item set when present.
    #[serde(default)]
    pub items: Option<Vec<TrainingDatasetItemInput>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetBatchRenameInput {
    pub items: Vec<TrainingDatasetRenameItemInput>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetRenameItemInput {
    pub item_id: String,
    #[serde(default)]
    pub new_item_id: Option<String>,
    #[serde(default)]
    pub file_stem: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetCaptionSidecarsInput {
    #[serde(default)]
    pub items: Vec<TrainingDatasetCaptionSidecarItemInput>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetCaptionSidecarItemInput {
    pub item_id: String,
    pub caption: CaptionInput,
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
    /// Project-relative path to the dataset's first item image, for a list
    /// thumbnail (sc-2025). `None` when the dataset has no items.
    #[serde(default)]
    pub cover_path: Option<String>,
    /// Owning character id (sc-2022), or `None` for a general dataset. Lets the
    /// Character Studio filter the list to one character's datasets client-side.
    #[serde(default)]
    pub character_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetMutationResult {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCaptionSidecar {
    pub item_id: String,
    pub image_path: String,
    pub caption_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCaptionSidecarsResult {
    pub dataset: TrainingDataset,
    pub sidecars: Vec<TrainingCaptionSidecar>,
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
            character_id: input.character_id,
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
        // sc-2022: a present `character_id` (sets/changes the owning character);
        // applied before either save path below so both persist it.
        if let Some(character_id) = input.character_id {
            dataset.character_id = Some(character_id);
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

    /// Persist freshly-extracted Tier-0 scalars onto matching dataset items as the content-hash +
    /// bucket-keyed cache (sc-6533). This is a derived-metadata write: it does **not** bump the
    /// dataset version or `updated_at` (the pixels didn't change, only our cache of measurements),
    /// and it silently skips ids no longer present — the set may have been edited between the
    /// readiness read and this write, and a stale entry self-invalidates via
    /// [`CachedTier0Scalars::valid_for`].
    pub fn cache_tier0_scalars(
        &self,
        project_id: &str,
        dataset_id: &str,
        updates: &[(String, CachedTier0Scalars)],
    ) -> ProjectStoreResult<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        let mut changed = false;
        for item in &mut dataset.items {
            if let Some(entry) = updates.iter().find(|(id, _)| id == &item.id) {
                item.tier0_scalars = Some(entry.1.clone());
                changed = true;
            }
        }
        if changed {
            self.save_dataset(&dataset)?;
        }
        Ok(())
    }

    /// Persist the dataset's CLIP image embeddings to the content-hash-keyed sidecar (sc-6535).
    /// Like `cache_tier0_scalars` this is a metadata write — it does NOT bump the dataset version
    /// (the pixels didn't change). The analysis job sends the full set, so the sidecar is replaced.
    pub fn write_dataset_embeddings(
        &self,
        project_id: &str,
        dataset_id: &str,
        embeddings: &DatasetEmbeddings,
    ) -> ProjectStoreResult<()> {
        let dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        write_json(
            &dataset_embeddings_path(&self.project_path, dataset_id),
            embeddings,
        )
    }

    /// Read the dataset's persisted CLIP embeddings, or `None` when the analysis job hasn't run yet.
    pub fn read_dataset_embeddings(
        &self,
        project_id: &str,
        dataset_id: &str,
    ) -> ProjectStoreResult<Option<DatasetEmbeddings>> {
        let dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        match fs::read(dataset_embeddings_path(&self.project_path, dataset_id)) {
            Ok(bytes) => Ok(Some(
                serde_json::from_slice(&bytes).map_err(ProjectStoreError::Json)?,
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(ProjectStoreError::Io(error)),
        }
    }

    /// Persist (or clear) a per-image quality override (sc-6534). `checks` are the findings the user
    /// dismissed for the image; the non-acknowledgeable ones (`Decode` is untrainable, `Count` is
    /// dataset-level) are stripped, and an empty result clears any existing ack. Like
    /// `cache_tier0_scalars` this is a metadata write — it does NOT bump the dataset version (the
    /// pixels didn't change). The ack is keyed by the item's current content hash so it self-voids
    /// via [`QualityAck::valid_for`] if the image is later replaced. Returns the stored ack (or
    /// `None` when cleared) so the caller can confirm what was actually kept.
    pub fn set_item_quality_ack(
        &self,
        project_id: &str,
        dataset_id: &str,
        item_id: &str,
        checks: &[QualityCheck],
        expected_content_hash: Option<&str>,
        expected_caption_hash: Option<&str>,
    ) -> ProjectStoreResult<Option<QualityAck>> {
        let mut dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        let item = dataset
            .items
            .iter_mut()
            .find(|item| item.id == item_id)
            .ok_or_else(|| {
                ProjectStoreError::NotFound("Training dataset item not found".to_owned())
            })?;
        let kept: Vec<QualityCheck> = checks
            .iter()
            .filter(|check| !matches!(check, QualityCheck::Decode | QualityCheck::Count))
            .cloned()
            .collect();
        let expected_content_hash = expected_content_hash
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                ProjectStoreError::BadRequest(
                    "Quality acknowledgements require expectedContentHash.".to_owned(),
                )
            })?;
        if item.content_hash.as_deref() != Some(expected_content_hash) {
            return Err(ProjectStoreError::BadRequest(
                "Quality acknowledgement is stale; refresh dataset readiness.".to_owned(),
            ));
        }
        let expected_caption_hash = expected_caption_hash
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some(expected_caption_hash) = expected_caption_hash {
            if caption_hash(&item.caption.text) != expected_caption_hash {
                return Err(ProjectStoreError::BadRequest(
                    "Caption acknowledgement is stale; refresh dataset readiness.".to_owned(),
                ));
            }
        } else if kept.contains(&QualityCheck::CaptionAlignment) {
            return Err(ProjectStoreError::BadRequest(
                "Caption alignment acknowledgements require expectedCaptionHash.".to_owned(),
            ));
        }
        let ack = if kept.is_empty() {
            None
        } else {
            Some(QualityAck {
                content_hash: expected_content_hash.to_owned(),
                caption_hash: kept
                    .contains(&QualityCheck::CaptionAlignment)
                    .then(|| caption_hash(&item.caption.text)),
                checks: kept,
            })
        };
        item.quality_ack = ack.clone();
        self.save_dataset(&dataset)?;
        Ok(ack)
    }

    pub fn batch_rename_dataset_items(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetBatchRenameInput,
    ) -> ProjectStoreResult<TrainingDataset> {
        let mut dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        let plans = rename_plans(&self.project_path, &dataset, input.items)?;
        if plans.is_empty() {
            return Ok(dataset);
        }
        let applied_renames = apply_file_renames(&plans)?;
        for plan in &plans {
            if let Some(item) = dataset
                .items
                .iter_mut()
                .find(|item| item.id == plan.original_item_id)
            {
                item.id = plan.next_item_id.clone();
                item.path = plan.next_relative_path.clone();
                item.display_name = plan.next_display_name.clone();
            }
        }
        dataset.version = dataset.version.saturating_add(1);
        dataset.updated_at = utc_now();
        if let Err(error) = self.save_dataset(&dataset) {
            rollback_file_renames(&applied_renames);
            return Err(error);
        }
        Ok(dataset)
    }

    pub fn write_caption_sidecars(
        &self,
        project_id: &str,
        dataset_id: &str,
        input: TrainingDatasetCaptionSidecarsInput,
    ) -> ProjectStoreResult<TrainingCaptionSidecarsResult> {
        let mut dataset = self.read_dataset_by_id(dataset_id)?;
        ensure_dataset_project(project_id, &dataset)?;
        if !input.items.is_empty() {
            let now = utc_now();
            apply_caption_patches(&mut dataset, input.items, &now)?;
            dataset.version = dataset.version.saturating_add(1);
            dataset.updated_at = now;
            self.save_dataset(&dataset)?;
        }
        let sidecars = write_dataset_caption_sidecars(&self.project_path, &dataset)?;
        Ok(TrainingCaptionSidecarsResult { dataset, sidecars })
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

#[derive(Debug, Clone)]
struct RenamePlan {
    original_item_id: String,
    next_item_id: String,
    next_relative_path: String,
    next_display_name: String,
    source_path: PathBuf,
    destination_path: PathBuf,
    source_caption_path: PathBuf,
    destination_caption_path: PathBuf,
}

#[derive(Debug, Clone)]
struct AppliedRename {
    source: PathBuf,
    destination: PathBuf,
    temp: PathBuf,
}

fn rename_plans(
    project_path: &Path,
    dataset: &TrainingDataset,
    inputs: Vec<TrainingDatasetRenameItemInput>,
) -> ProjectStoreResult<Vec<RenamePlan>> {
    if inputs.is_empty() {
        return Err(ProjectStoreError::BadRequest(
            "Batch rename requires at least one item".to_owned(),
        ));
    }
    let dataset_root = dataset_root(project_path, &dataset.id);
    let mut seen_inputs = Vec::new();
    let mut plans = Vec::new();
    for input in inputs {
        if !is_safe_id(&input.item_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid training dataset item ID".to_owned(),
            ));
        }
        if seen_inputs.iter().any(|item_id| item_id == &input.item_id) {
            return Err(ProjectStoreError::BadRequest(
                "Batch rename item IDs must be unique".to_owned(),
            ));
        }
        seen_inputs.push(input.item_id.clone());
        let item = dataset
            .items
            .iter()
            .find(|item| item.id == input.item_id)
            .ok_or_else(|| {
                ProjectStoreError::NotFound("Training dataset item not found".to_owned())
            })?;
        let next_item_id = match input.new_item_id.as_deref().map(str::trim) {
            Some(value) if !value.is_empty() && is_safe_id(value) => value.to_owned(),
            Some("") => item.id.clone(),
            Some(_) => {
                return Err(ProjectStoreError::BadRequest(
                    "Invalid training dataset item ID".to_owned(),
                ))
            }
            None => item.id.clone(),
        };
        let source_path = dataset_item_path(project_path, dataset, item)?;
        if !source_path.exists() {
            return Err(ProjectStoreError::NotFound(
                "Training dataset item file not found".to_owned(),
            ));
        }
        let extension = Path::new(&item.path)
            .extension()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .map(|value| format!(".{}", value.to_ascii_lowercase()))
            .unwrap_or_else(|| ".bin".to_owned());
        let current_stem = Path::new(&item.path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or(item.id.as_str());
        let next_stem = match input.file_stem.as_deref().map(str::trim) {
            Some(value) if !value.is_empty() && is_safe_file_stem(value) => value.to_owned(),
            Some("") => current_stem.to_owned(),
            Some(_) => {
                return Err(ProjectStoreError::BadRequest(
                    "Invalid training dataset file stem".to_owned(),
                ))
            }
            None => current_stem.to_owned(),
        };
        let next_relative_path = format!("images/{next_stem}{extension}");
        let destination_path = dataset_root
            .join("images")
            .join(format!("{next_stem}{extension}"));
        let next_display_name = display_name_for_rename(&input.display_name, &next_relative_path)?;
        if next_item_id != item.id
            || next_relative_path != item.path
            || next_display_name != item.display_name
        {
            plans.push(RenamePlan {
                original_item_id: item.id.clone(),
                next_item_id,
                next_relative_path,
                next_display_name,
                source_caption_path: source_path.with_extension("txt"),
                destination_caption_path: destination_path.with_extension("txt"),
                source_path,
                destination_path,
            });
        }
    }
    validate_projected_dataset_items(dataset, &plans)?;
    Ok(plans)
}

fn validate_projected_dataset_items(
    dataset: &TrainingDataset,
    plans: &[RenamePlan],
) -> ProjectStoreResult<()> {
    let mut projected_ids = Vec::new();
    let mut projected_paths = Vec::new();
    for item in &dataset.items {
        let plan = plans.iter().find(|plan| plan.original_item_id == item.id);
        let item_id = plan
            .map(|plan| plan.next_item_id.as_str())
            .unwrap_or(item.id.as_str());
        let path = plan
            .map(|plan| plan.next_relative_path.as_str())
            .unwrap_or(item.path.as_str());
        if projected_ids.iter().any(|existing| existing == &item_id) {
            return Err(ProjectStoreError::BadRequest(
                "Training dataset item IDs must be unique".to_owned(),
            ));
        }
        if projected_paths.iter().any(|existing| existing == &path) {
            return Err(ProjectStoreError::BadRequest(
                "Training dataset item paths must be unique".to_owned(),
            ));
        }
        projected_ids.push(item_id);
        projected_paths.push(path);
    }
    let source_paths = plans
        .iter()
        .map(|plan| &plan.source_path)
        .collect::<Vec<_>>();
    for plan in plans {
        if plan.destination_path.exists()
            && plan.destination_path != plan.source_path
            && !source_paths.contains(&&plan.destination_path)
        {
            return Err(ProjectStoreError::BadRequest(
                "Training dataset item file already exists".to_owned(),
            ));
        }
    }
    Ok(())
}

fn apply_file_renames(plans: &[RenamePlan]) -> ProjectStoreResult<Vec<AppliedRename>> {
    let mut applied = Vec::new();
    let token = random_hex(8)?;
    let mut moves = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        if plan.source_path != plan.destination_path {
            moves.push((
                plan.source_path.clone(),
                plan.destination_path.clone(),
                plan.source_path
                    .with_file_name(format!(".rename-{token}-{index}.tmp")),
            ));
        }
        if plan.source_caption_path.exists()
            && plan.source_caption_path != plan.destination_caption_path
        {
            moves.push((
                plan.source_caption_path.clone(),
                plan.destination_caption_path.clone(),
                plan.source_caption_path
                    .with_file_name(format!(".caption-rename-{token}-{index}.tmp")),
            ));
        }
    }
    for (source, destination, temp) in &moves {
        if let Some(parent) = temp.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::rename(source, temp) {
            Ok(()) => applied.push(AppliedRename {
                source: source.clone(),
                destination: destination.clone(),
                temp: temp.clone(),
            }),
            Err(error) => {
                rollback_file_renames(&applied);
                return Err(ProjectStoreError::Io(error));
            }
        }
    }
    for rename in &applied {
        if let Some(parent) = rename.destination.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Err(error) = fs::rename(&rename.temp, &rename.destination) {
            rollback_file_renames(&applied);
            return Err(ProjectStoreError::Io(error));
        }
    }
    Ok(applied)
}

fn rollback_file_renames(applied: &[AppliedRename]) {
    for rename in applied.iter().rev() {
        if rename.temp.exists() {
            let _ = fs::rename(&rename.temp, &rename.source);
        } else if rename.destination.exists() && !rename.source.exists() {
            let _ = fs::rename(&rename.destination, &rename.source);
        }
    }
}

fn apply_caption_patches(
    dataset: &mut TrainingDataset,
    inputs: Vec<TrainingDatasetCaptionSidecarItemInput>,
    now: &str,
) -> ProjectStoreResult<()> {
    let mut seen_inputs = Vec::new();
    for input in inputs {
        if !is_safe_id(&input.item_id) {
            return Err(ProjectStoreError::BadRequest(
                "Invalid training dataset item ID".to_owned(),
            ));
        }
        if seen_inputs.iter().any(|item_id| item_id == &input.item_id) {
            return Err(ProjectStoreError::BadRequest(
                "Caption sidecar item IDs must be unique".to_owned(),
            ));
        }
        seen_inputs.push(input.item_id.clone());
        let item = dataset
            .items
            .iter_mut()
            .find(|item| item.id == input.item_id)
            .ok_or_else(|| {
                ProjectStoreError::NotFound("Training dataset item not found".to_owned())
            })?;
        item.caption = Caption {
            text: input.caption.text,
            source: input.caption.source.unwrap_or(CaptionSource::Manual),
            trigger_words: input.caption.trigger_words,
            updated_at: Some(now.to_owned()),
            extra: Default::default(),
        };
    }
    Ok(())
}

fn write_dataset_caption_sidecars(
    project_path: &Path,
    dataset: &TrainingDataset,
) -> ProjectStoreResult<Vec<TrainingCaptionSidecar>> {
    let mut sidecars = Vec::new();
    for item in &dataset.items {
        let image_path = dataset_item_path(project_path, dataset, item)?;
        let caption_path = image_path.with_extension("txt");
        write_text(
            &caption_path,
            &format!(
                "{}\n",
                caption_text_with_trigger_words(&item.caption.text, &item.caption.trigger_words)
            ),
        )?;
        sidecars.push(TrainingCaptionSidecar {
            item_id: item.id.clone(),
            image_path: item.path.clone(),
            caption_path: relative_string(project_path, &caption_path)?,
        });
    }
    Ok(sidecars)
}

fn caption_text_with_trigger_words(caption: &str, trigger_words: &[String]) -> String {
    let cleaned = caption.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower = cleaned.to_lowercase();
    let mut parts = trigger_words
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty() && !lower.contains(&word.to_lowercase()))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !cleaned.is_empty() {
        parts.push(cleaned);
    }
    parts.join(", ")
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
    // sc-6531 (Dataset Doctor): every item must carry real pixel dimensions + a content hash —
    // the foundation Tier-0 checks (min-resolution, crop-loss, exact-dup) rely on. Prefer
    // caller-/asset-provided dimensions; otherwise read them from the stored file's header
    // (covers path uploads and library assets with no recorded size, and backfills on any
    // re-materialize). The read is header-only and the hash streams the file — neither decodes
    // the image, so core stays codec-free.
    let mut width = input.width.or(source.width);
    let mut height = input.height.or(source.height);
    if width.is_none() || height.is_none() {
        if let Some((file_w, file_h)) = crate::media_convert::image_dimensions(&target_path) {
            width = width.or(Some(file_w));
            height = height.or(Some(file_h));
        }
    }
    let content_hash = crate::media_convert::file_content_hash(&target_path).ok();
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
        width,
        height,
        content_hash,
        // Fresh file → no cached Tier-0 scalars yet; the readiness endpoint fills them lazily
        // (and re-fills when content hash or bucket edge changes). sc-6533.
        tier0_scalars: None,
        // Fresh file → no dismissed findings yet (sc-6534).
        quality_ack: None,
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
    let asset: Value = read_json(&sidecar)?;
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
    // Route through the version-gated comprehensive migration so the training
    // path stops replaying DDL on every call (the table is created either way).
    apply_project_migrations(&connection)
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
          character_id text,
          file_path text not null,
          created_at text not null,
          updated_at text not null
        );
        create index if not exists idx_training_datasets_project_updated
          on training_datasets(project_id, updated_at);
        ",
    )?;
    ensure_training_dataset_column(connection, "item_count", "integer not null default 0")?;
    ensure_training_dataset_column(connection, "character_id", "text")?;
    Ok(())
}

fn list_dataset_summaries_from_index(
    project_path: &Path,
    project_id: &str,
) -> ProjectStoreResult<Vec<TrainingDatasetSummary>> {
    let connection = Connection::open(project_path.join("project.db"))?;
    let mut statement = connection.prepare(
        "
        select id, project_id, name, modality, status, version, item_count, created_at, updated_at, file_path, character_id
          from training_datasets
         where project_id = ?1
         order by updated_at desc, name asc
        ",
    )?;
    let rows = statement
        .query_map(params![project_id], |row| {
            let item_count: i64 = row.get(6)?;
            Ok((
                TrainingDatasetSummary {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    name: row.get(2)?,
                    modality: parse_string_enum(&row.get::<_, String>(3)?),
                    status: parse_string_enum(&row.get::<_, String>(4)?),
                    version: row.get(5)?,
                    item_count: usize::try_from(item_count).unwrap_or(0),
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    cover_path: None,
                    character_id: row.get::<_, Option<String>>(10)?,
                },
                row.get::<_, Option<String>>(9)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let summaries = rows
        .into_iter()
        .map(|(mut summary, manifest_rel)| {
            summary.cover_path = manifest_rel
                .as_deref()
                .and_then(|rel| resolve_dataset_cover(project_path, &summary.id, rel));
            summary
        })
        .collect();
    Ok(summaries)
}

/// Project-relative path to a dataset's first item image, read from its on-disk
/// manifest. Powers the dataset list thumbnail (sc-2025) without storing a
/// denormalized column. `None` when the manifest is unreadable or item-less.
fn resolve_dataset_cover(
    project_path: &Path,
    dataset_id: &str,
    manifest_rel: &str,
) -> Option<String> {
    let manifest = read_json(&project_path.join(manifest_rel)).ok()?;
    let item_path = manifest
        .get("items")?
        .as_array()?
        .first()?
        .get("path")?
        .as_str()?;
    if !is_safe_relative_path(item_path) {
        return None;
    }
    relative_string(
        project_path,
        &dataset_root(project_path, dataset_id).join(item_path),
    )
    .ok()
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
          id, project_id, name, modality, status, version, item_count, character_id, file_path, created_at, updated_at
        ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        ",
        params![
            dataset.id,
            dataset.project_id.as_deref().unwrap_or_default(),
            dataset.name,
            dataset.modality.as_str(),
            dataset.status.as_str(),
            dataset.version,
            i64::try_from(dataset.items.len()).unwrap_or(i64::MAX),
            dataset.character_id,
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
    // Thin fixed-table wrapper over the shared store_util helper (sc-4271).
    ensure_column(connection, "training_datasets", column, definition)?;
    Ok(())
}

pub(crate) fn dataset_root(project_path: &Path, dataset_id: &str) -> PathBuf {
    project_path
        .join("training")
        .join("datasets")
        .join(dataset_id)
}

fn dataset_manifest_path(project_path: &Path, dataset_id: &str) -> PathBuf {
    dataset_root(project_path, dataset_id).join(DATASET_MANIFEST_NAME)
}

fn dataset_embeddings_path(project_path: &Path, dataset_id: &str) -> PathBuf {
    dataset_root(project_path, dataset_id).join(DATASET_EMBEDDINGS_NAME)
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
    let mut dataset: TrainingDataset = serde_json::from_str(&payload)?;
    // sc-6531 (Dataset Doctor): lazily backfill dimensions for items written before we measured
    // them, so *every* item carries accurate width/height (the foundation Tier-0 relies on). The
    // manifest lives at `<dataset_root>/<manifest>`, so its parent is the dataset root the item
    // paths are relative to. Header-only reads, in-memory only — a read never rewrites the
    // manifest; a later save persists the filled values opportunistically. The `is_some` guard
    // makes this zero-I/O once dimensions are present (the steady state after this ships).
    if let Some(dataset_root) = path.parent() {
        backfill_missing_dimensions(dataset_root, &mut dataset);
    }
    Ok(dataset)
}

/// Fill absent `width`/`height` on dataset items from their on-disk image headers (sc-6531).
/// Skips items that already have both dimensions or whose stored path is unsafe; a missing or
/// unreadable file leaves the item untouched (best-effort, never fails the read).
fn backfill_missing_dimensions(dataset_root: &Path, dataset: &mut TrainingDataset) {
    for item in &mut dataset.items {
        if item.width.is_some() && item.height.is_some() {
            continue;
        }
        if !is_safe_relative_path(&item.path) {
            continue;
        }
        if let Some((width, height)) =
            crate::media_convert::image_dimensions(&dataset_root.join(&item.path))
        {
            item.width = item.width.or(Some(width));
            item.height = item.height.or(Some(height));
        }
    }
}

fn write_text(path: &Path, payload: &str) -> ProjectStoreResult<()> {
    atomic_write(path, payload.as_bytes())
}

fn dataset_item_path(
    project_path: &Path,
    dataset: &TrainingDataset,
    item: &TrainingDatasetItem,
) -> ProjectStoreResult<PathBuf> {
    if !is_safe_relative_path(&item.path) {
        return Err(ProjectStoreError::BadRequest(
            "Invalid training dataset item path".to_owned(),
        ));
    }
    let path = dataset_root(project_path, &dataset.id).join(&item.path);
    let root = fs::canonicalize(project_path)?;
    let canonical = fs::canonicalize(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ProjectStoreError::NotFound("Training dataset item file not found".to_owned())
        } else {
            ProjectStoreError::Io(error)
        }
    })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(ProjectStoreError::BadRequest(
            "Invalid training dataset item path".to_owned(),
        ));
    }
    Ok(path)
}

fn is_safe_file_stem(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && value.chars().count() <= 96
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn display_name_for_rename(
    input: &Option<String>,
    relative_path: &str,
) -> ProjectStoreResult<String> {
    match input.as_deref().map(str::trim) {
        Some(value) if value.chars().count() > 160 => Err(ProjectStoreError::BadRequest(
            "Training dataset display name must be at most 160 characters".to_owned(),
        )),
        Some(value) if !value.is_empty() => Ok(value.to_owned()),
        _ => Ok(input_display_name(Path::new(relative_path))),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset_quality::Tier0Scalars;

    fn item(id: &str, content_hash: &str) -> TrainingDatasetItem {
        TrainingDatasetItem {
            id: id.to_owned(),
            asset_id: None,
            path: format!("images/{id}.png"),
            display_name: id.to_owned(),
            caption: Caption {
                text: String::new(),
                source: CaptionSource::Manual,
                trigger_words: Vec::new(),
                updated_at: None,
                extra: Default::default(),
            },
            width: Some(512),
            height: Some(512),
            content_hash: Some(content_hash.to_owned()),
            tier0_scalars: None,
            quality_ack: None,
            added_at: utc_now(),
            extra: Default::default(),
        }
    }

    fn saved_dataset(project_path: &Path) -> TrainingDatasetStore {
        let dataset = TrainingDataset {
            schema_version: TRAINING_CONTRACT_SCHEMA_VERSION,
            id: "ds_test".to_owned(),
            version: 1,
            project_id: Some("proj".to_owned()),
            character_id: None,
            name: "Test".to_owned(),
            modality: TrainingModality::Image,
            status: TrainingDatasetStatus::Draft,
            created_at: utc_now(),
            updated_at: utc_now(),
            items: vec![item("item_1", "h1"), item("item_2", "h2")],
            extra: Default::default(),
        };
        let store = TrainingDatasetStore::new(project_path.to_path_buf());
        store.save_dataset(&dataset).expect("save dataset");
        store
    }

    fn cache_entry(content_hash: &str, bucket_edge: u32) -> CachedTier0Scalars {
        CachedTier0Scalars {
            content_hash: content_hash.to_owned(),
            bucket_edge,
            scalars: Tier0Scalars {
                blur_variance: 1234.0,
                shadow_clip: 0.0,
                highlight_clip: 0.0,
                phash: vec![1, 2, 3, 4, 5, 6, 7, 8],
            },
        }
    }

    #[test]
    fn cache_tier0_scalars_persists_only_matching_items_without_bumping_version() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = saved_dataset(dir.path());

        let updates = vec![("item_1".to_owned(), cache_entry("h1", 512))];
        store
            .cache_tier0_scalars("proj", "ds_test", &updates)
            .expect("cache scalars");

        let reloaded = store.get_dataset("proj", "ds_test").expect("reload");
        let item_1 = reloaded.items.iter().find(|i| i.id == "item_1").unwrap();
        let cached = item_1.tier0_scalars.as_ref().expect("item_1 has cache");
        assert_eq!(cached.content_hash, "h1");
        assert_eq!(cached.bucket_edge, 512);
        assert_eq!(cached.scalars.blur_variance, 1234.0);

        // The unaddressed item stays uncached, and a metadata write leaves the version untouched.
        let item_2 = reloaded.items.iter().find(|i| i.id == "item_2").unwrap();
        assert!(item_2.tier0_scalars.is_none());
        assert_eq!(reloaded.version, 1);
    }

    #[test]
    fn cache_tier0_scalars_ignores_unknown_ids() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = saved_dataset(dir.path());

        store
            .cache_tier0_scalars(
                "proj",
                "ds_test",
                &[("ghost".to_owned(), cache_entry("hx", 512))],
            )
            .expect("unknown id is a no-op");

        let reloaded = store.get_dataset("proj", "ds_test").expect("reload");
        assert!(reloaded.items.iter().all(|i| i.tier0_scalars.is_none()));
    }

    #[test]
    fn set_item_quality_ack_strips_unacknowledgeable_and_keys_by_hash() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = saved_dataset(dir.path());

        // Blur is dismissable; Decode (untrainable) and the dataset-level Count are stripped.
        let stored = store
            .set_item_quality_ack(
                "proj",
                "ds_test",
                "item_1",
                &[
                    QualityCheck::Blur,
                    QualityCheck::Decode,
                    QualityCheck::Count,
                ],
                Some("h1"),
                None,
            )
            .expect("set ack")
            .expect("ack stored");
        assert_eq!(stored.checks, vec![QualityCheck::Blur]);
        assert_eq!(
            stored.content_hash, "h1",
            "keyed by the item's current hash"
        );

        let reloaded = store.get_dataset("proj", "ds_test").expect("reload");
        let item_1 = reloaded.items.iter().find(|i| i.id == "item_1").unwrap();
        assert_eq!(
            item_1.quality_ack.as_ref().map(|a| a.checks.clone()),
            Some(vec![QualityCheck::Blur])
        );
        // Metadata write: the version is untouched.
        assert_eq!(reloaded.version, 1);
    }

    #[test]
    fn set_item_quality_ack_clears_on_empty() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = saved_dataset(dir.path());

        store
            .set_item_quality_ack(
                "proj",
                "ds_test",
                "item_1",
                &[QualityCheck::Blur],
                Some("h1"),
                None,
            )
            .expect("set ack");
        // An empty set (or a set of only unacknowledgeable checks) clears the override.
        let cleared = store
            .set_item_quality_ack(
                "proj",
                "ds_test",
                "item_1",
                &[QualityCheck::Decode],
                Some("h1"),
                None,
            )
            .expect("clear ack");
        assert!(cleared.is_none());

        let reloaded = store.get_dataset("proj", "ds_test").expect("reload");
        let item_1 = reloaded.items.iter().find(|i| i.id == "item_1").unwrap();
        assert!(item_1.quality_ack.is_none());
    }

    #[test]
    fn set_item_quality_ack_rejects_stale_expected_hashes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = saved_dataset(dir.path());

        let stale_content = store.set_item_quality_ack(
            "proj",
            "ds_test",
            "item_1",
            &[QualityCheck::Blur],
            Some("old_hash"),
            None,
        );
        assert!(matches!(
            stale_content,
            Err(ProjectStoreError::BadRequest(_))
        ));

        let stale_caption = store.set_item_quality_ack(
            "proj",
            "ds_test",
            "item_1",
            &[QualityCheck::CaptionAlignment],
            Some("h1"),
            Some(&caption_hash("old caption")),
        );
        assert!(matches!(
            stale_caption,
            Err(ProjectStoreError::BadRequest(_))
        ));

        let current_caption = caption_hash("");
        let stored = store
            .set_item_quality_ack(
                "proj",
                "ds_test",
                "item_1",
                &[QualityCheck::CaptionAlignment],
                Some("h1"),
                Some(&current_caption),
            )
            .expect("set current caption ack")
            .expect("ack stored");
        assert_eq!(
            stored.caption_hash.as_deref(),
            Some(current_caption.as_str())
        );
    }
}
