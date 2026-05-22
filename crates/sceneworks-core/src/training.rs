//! Rust-owned contracts for SceneWorks native LoRA training.
//!
//! SceneWorks owns its training product surface: dataset storage, manifests,
//! validation, queue semantics, the target registry, and LoRA registration all
//! live in Rust. Python is a narrow execution kernel that consumes a fully
//! normalized [`TrainingPlan`] and produces weights — it never reads SceneWorks
//! storage, config defaults, or this registry directly.
//!
//! These contracts are intentionally generic over `modality`, `output kind`,
//! and `family` so the same shapes serve future image, video, and audio
//! targets. The first production target is an image LoRA for Z-Image-Turbo,
//! exposed by [`builtin_training_targets`].
//!
//! `ai-toolkit` is reference material only — a source of sensible defaults and
//! terminology. None of its YAML config format or runtime option set is
//! embedded here: hyperparameters use generic LoRA terms (`rank`, `alpha`,
//! `learningRate`, `steps`) and free-form `advanced`/`limits`/`ui` bags carry
//! anything engine-specific without coupling the contract to a trainer.
//!
//! All shapes follow the crate's contract conventions: `camelCase` JSON, a
//! trailing flattened [`ExtraFields`] for forward compatibility, and string
//! enums that round-trip unknown values via an `Unknown(String)` variant.

use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

use crate::contracts::{string_enum, ContractNumber, ExtraFields, JsonObject};

/// Schema version stamped on persisted training contracts.
pub const TRAINING_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Version of the normalized [`TrainingPlan`] handed to the Python kernel. The
/// kernel rejects plans whose `planVersion` it does not understand.
pub const TRAINING_PLAN_VERSION: u32 = 1;

string_enum! {
    /// Output modality of a training target. `Image` is the first production
    /// target; `Video` and `Audio` are reserved so the contract stays generic.
    pub enum TrainingModality {
        Image => "image",
        Video => "video",
        Audio => "audio",
    }
}

string_enum! {
    /// What a training run produces. Only LoRA adapters are produced today.
    pub enum TrainingOutputKind {
        Lora => "lora",
    }
}

string_enum! {
    /// Lifecycle state of a training dataset.
    pub enum TrainingDatasetStatus {
        Draft => "draft",
        Ready => "ready",
        Archived => "archived",
    }
}

string_enum! {
    /// Origin of a dataset item's caption.
    pub enum CaptionSource {
        Manual => "manual",
        Imported => "imported",
        Auto => "auto",
    }
}

/// A training dataset: an ordered collection of captioned items owned and
/// persisted by Rust (see story 1410 for the store).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDataset {
    pub schema_version: u32,
    pub id: String,
    /// Monotonic version bumped whenever items or captions change. Provenance
    /// pins an exact version so a re-train is reproducible.
    pub version: u32,
    /// Owning project, when the dataset is project-scoped. `None` means global.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    pub name: String,
    pub modality: TrainingModality,
    pub status: TrainingDatasetStatus,
    pub created_at: String,
    pub updated_at: String,
    pub items: Vec<TrainingDatasetItem>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// A single captioned training example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingDatasetItem {
    pub id: String,
    /// Source SceneWorks asset, when the item was selected from the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    /// Path relative to the dataset root (Rust owns dataset storage layout).
    pub path: String,
    pub display_name: String,
    pub caption: Caption,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub added_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Caption text and provenance for a dataset item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Caption {
    pub text: String,
    pub source: CaptionSource,
    pub trigger_words: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// A registered training target: the combination of a base model, output kind,
/// and execution kernel, plus the defaults and bounds the UI builds on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingTarget {
    /// Stable target id, e.g. `z_image_turbo_lora`.
    pub id: String,
    pub name: String,
    pub modality: TrainingModality,
    pub output_kind: TrainingOutputKind,
    /// SceneWorks LoRA/model family, e.g. `z-image`. Drives downstream
    /// generation-side compatibility of the produced LoRA.
    pub family: String,
    /// Manifest model id this target trains against.
    pub base_model: String,
    /// Optional source repository for the base model weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_repo: Option<String>,
    /// Identifier of the Python execution kernel that runs this target.
    pub kernel: String,
    /// Visible (simple-panel) config defaults for this target.
    pub defaults: TrainingConfig,
    /// Bounds and choices for advanced fields; free-form to stay generic.
    pub limits: JsonObject,
    /// Presentation hints (labels, descriptions); free-form.
    pub ui: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// The registry of available training targets. Rust owns the built-in set
/// returned by [`builtin_training_targets`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingTargetRegistry {
    pub schema_version: u32,
    pub targets: Vec<TrainingTarget>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Generic LoRA training hyperparameters.
///
/// The visible fields back the simple config panel; engine-specific knobs live
/// in the free-form `advanced` bag so the contract never couples to a specific
/// trainer's option set. This shape doubles as a target's `defaults` and as the
/// resolved config inside a [`TrainingPlan`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingConfig {
    /// LoRA network rank (dimension).
    pub rank: u32,
    /// LoRA alpha scaling factor.
    pub alpha: u32,
    pub learning_rate: ContractNumber,
    /// Total training steps.
    pub steps: u32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    /// Square training resolution edge in pixels (e.g. `1024`). Aspect-ratio
    /// bucketing details, when used, live in `advanced`.
    pub resolution: u32,
    /// Checkpoint cadence, in steps.
    pub save_every: u32,
    pub seed: i64,
    /// Optimizer name, kept a free string to stay engine-agnostic.
    pub optimizer: String,
    /// Trigger word baked into captions and surfaced on the output LoRA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_word: Option<String>,
    /// Advanced, collapsed-by-default fields. Free-form by design.
    pub advanced: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Payload contract for submitting a LoRA training job.
///
/// The matching `JobType`, queue routing, and dry-run plan builder are added in
/// story 1416; this is only the request shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoraTrainingRequest {
    pub target_id: String,
    pub dataset_id: String,
    /// Pin a dataset version; `None` means "use the dataset's current version".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_version: Option<u32>,
    pub config: TrainingConfig,
    /// Human-facing name for the resulting LoRA.
    pub output_name: String,
    /// When true, the queue resolves a [`TrainingPlan`] and stops short of
    /// running the kernel. Defaults to true: dry-run is the only mode supported
    /// today, so the API rejects `false` until real execution exists.
    #[serde(default = "default_dry_run")]
    pub dry_run: bool,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

fn default_dry_run() -> bool {
    true
}

/// The fully normalized plan Rust hands to the Python execution kernel.
///
/// Every path is absolute and every hyperparameter concrete: the kernel reads
/// only this document. `planVersion` lets the kernel reject formats it does not
/// understand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPlan {
    pub schema_version: u32,
    pub plan_version: u32,
    pub job_id: String,
    pub target: TrainingPlanTarget,
    pub dataset: TrainingPlanDataset,
    pub config: TrainingConfig,
    pub output: TrainingPlanOutput,
    pub provenance: TrainingProvenance,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Resolved target details inside a [`TrainingPlan`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPlanTarget {
    pub target_id: String,
    pub kernel: String,
    pub family: String,
    pub modality: TrainingModality,
    pub output_kind: TrainingOutputKind,
    pub base_model: String,
    /// Optional source repository for the base model weights. The kernel prefers
    /// this for `from_pretrained` (matching the generation load path, cache or
    /// download), falling back to `base_model_path` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_repo: Option<String>,
    /// Absolute, resolved path to the base model weights on the worker.
    pub base_model_path: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Resolved dataset details inside a [`TrainingPlan`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPlanDataset {
    pub dataset_id: String,
    pub dataset_version: u32,
    /// Absolute root directory the kernel reads images and captions from.
    pub root_path: String,
    pub items: Vec<TrainingPlanItem>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// A single resolved training example inside a [`TrainingPlan`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPlanItem {
    /// Absolute image path resolved by Rust.
    pub image_path: String,
    pub caption: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Where and how the kernel writes the produced adapter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPlanOutput {
    /// Pre-allocated SceneWorks LoRA id the output registers under.
    pub lora_id: String,
    /// Absolute directory the kernel writes the adapter into.
    pub output_dir: String,
    /// File name for the produced adapter, e.g. `my_style.safetensors`.
    pub file_name: String,
    /// Serialized weight format; `safetensors` today.
    pub format: String,
    pub trigger_words: Vec<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// Provenance captured for a training run, linking the output LoRA back to its
/// dataset, config, base model, and job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingProvenance {
    pub dataset_id: String,
    pub dataset_version: u32,
    pub target_id: String,
    pub base_model: String,
    /// Full config snapshot captured at submit time for reproducibility.
    pub config_snapshot: JsonObject,
    /// SceneWorks LoRA id the run produced (or will produce).
    pub output_lora_id: String,
    /// Job that produced this output.
    pub source_job_id: String,
    pub created_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// The built-in training targets Rust owns out of the box.
///
/// The first target is an image LoRA for Z-Image-Turbo. Defaults are informed
/// by common LoRA practice (and `ai-toolkit` as reference), not derived from
/// any external config format.
pub fn builtin_training_targets() -> TrainingTargetRegistry {
    TrainingTargetRegistry {
        schema_version: TRAINING_CONTRACT_SCHEMA_VERSION,
        targets: vec![z_image_turbo_lora_target()],
        extra: ExtraFields::new(),
    }
}

fn z_image_turbo_lora_target() -> TrainingTarget {
    TrainingTarget {
        id: "z_image_turbo_lora".to_owned(),
        name: "Z-Image-Turbo LoRA".to_owned(),
        modality: TrainingModality::Image,
        output_kind: TrainingOutputKind::Lora,
        family: "z-image".to_owned(),
        base_model: "z_image_turbo".to_owned(),
        base_model_repo: Some("Tongyi-MAI/Z-Image-Turbo".to_owned()),
        kernel: "z_image_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 16,
            alpha: 16,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 2000,
            batch_size: 1,
            gradient_accumulation: 1,
            resolution: 1024,
            save_every: 250,
            seed: 42,
            optimizer: "adamw8bit".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                "cacheLatents": true,
                "networkType": "lora",
                "scheduler": "constant",
                "epochs": 1,
                "repeats": 10,
                "bucketStrategy": "aspect",
                "sampleEvery": 250,
                "sampleSteps": 9,
                "sampleGuidanceScale": 0.0,
                "qualityPreset": "balanced",
                "outputScope": "project",
                "requestedGpu": "auto"
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [4, 128],
            "alpha": [1, 128],
            "steps": [200, 6000],
            "resolutions": [512, 768, 1024],
            "batchSize": [1, 4],
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Z-Image-Turbo LoRA",
            "description": "Train an image LoRA for the Z-Image-Turbo base model.",
            "recommendedFor": ["character", "style"]
        })),
        extra: ExtraFields::new(),
    }
}

/// Converts a JSON value known to be an object literal into a [`JsonObject`].
/// Non-object inputs yield an empty map; all call sites here pass object
/// literals.
fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(map) => map,
        _ => JsonObject::new(),
    }
}

/// Resolved inputs the [`build_training_plan`] resolver normalizes into a
/// [`TrainingPlan`]. The caller (the Rust API) owns I/O — loading the dataset,
/// allocating the output LoRA id, and resolving absolute on-host paths — so the
/// builder itself stays a pure, testable normalization step.
#[derive(Debug)]
pub struct BuildTrainingPlan<'a> {
    /// Id the job will be created under; embedded as the plan's `jobId` and
    /// provenance `sourceJobId` so the plan is self-describing.
    pub job_id: &'a str,
    pub target: &'a TrainingTarget,
    pub dataset: &'a TrainingDataset,
    /// Resolved config (target defaults already merged with user overrides).
    pub config: TrainingConfig,
    /// Pre-allocated SceneWorks LoRA id the output registers under.
    pub lora_id: &'a str,
    /// Absolute path to the base model weights on the worker host.
    pub base_model_path: String,
    /// Absolute dataset root the worker reads images and captions from.
    pub dataset_root: &'a Path,
    /// Absolute directory the kernel writes the adapter into.
    pub output_dir: &'a Path,
    /// Adapter file name, e.g. `aurora_style.safetensors`.
    pub file_name: String,
    pub created_at: String,
}

/// Error produced when a [`LoraTrainingRequest`] cannot be resolved into a valid
/// [`TrainingPlan`]. These map to client errors: the request is structurally
/// fine but the dataset or config cannot produce a runnable plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingPlanError {
    /// The dataset has no items to train on.
    EmptyDataset,
    /// A hyperparameter is out of range; carries a human-facing reason.
    InvalidConfig(String),
}

impl std::fmt::Display for TrainingPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDataset => {
                formatter.write_str("Training dataset has no items. Add at least one image.")
            }
            Self::InvalidConfig(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for TrainingPlanError {}

/// Normalizes resolved request inputs into the [`TrainingPlan`] the Python
/// kernel consumes. Validates the dataset is non-empty and the config is
/// runnable; absolute paths and ids are resolved by the caller.
pub fn build_training_plan(
    input: BuildTrainingPlan<'_>,
) -> Result<TrainingPlan, TrainingPlanError> {
    validate_training_config(&input.config)?;
    if input.dataset.items.is_empty() {
        return Err(TrainingPlanError::EmptyDataset);
    }

    let trigger_words = match input.config.trigger_word.as_deref().map(str::trim) {
        Some(word) if !word.is_empty() => vec![word.to_owned()],
        _ => Vec::new(),
    };

    let items = input
        .dataset
        .items
        .iter()
        .map(|item| TrainingPlanItem {
            image_path: resolve_item_path(input.dataset_root, &item.path),
            caption: caption_with_trigger_words(
                &item.caption.text,
                &combined_trigger_words(&item.caption.trigger_words, &trigger_words),
            ),
            width: item.width,
            height: item.height,
            extra: ExtraFields::new(),
        })
        .collect::<Vec<_>>();

    let config_snapshot = match serde_json::to_value(&input.config) {
        Ok(Value::Object(map)) => map,
        _ => JsonObject::new(),
    };

    Ok(TrainingPlan {
        schema_version: TRAINING_CONTRACT_SCHEMA_VERSION,
        plan_version: TRAINING_PLAN_VERSION,
        job_id: input.job_id.to_owned(),
        target: TrainingPlanTarget {
            target_id: input.target.id.clone(),
            kernel: input.target.kernel.clone(),
            family: input.target.family.clone(),
            modality: input.target.modality.clone(),
            output_kind: input.target.output_kind.clone(),
            base_model: input.target.base_model.clone(),
            base_model_repo: input.target.base_model_repo.clone(),
            base_model_path: input.base_model_path,
            extra: ExtraFields::new(),
        },
        dataset: TrainingPlanDataset {
            dataset_id: input.dataset.id.clone(),
            dataset_version: input.dataset.version,
            root_path: input.dataset_root.display().to_string(),
            items,
            extra: ExtraFields::new(),
        },
        config: input.config,
        output: TrainingPlanOutput {
            lora_id: input.lora_id.to_owned(),
            output_dir: input.output_dir.display().to_string(),
            file_name: input.file_name,
            format: "safetensors".to_owned(),
            trigger_words,
            extra: ExtraFields::new(),
        },
        provenance: TrainingProvenance {
            dataset_id: input.dataset.id.clone(),
            dataset_version: input.dataset.version,
            target_id: input.target.id.clone(),
            base_model: input.target.base_model.clone(),
            config_snapshot,
            output_lora_id: input.lora_id.to_owned(),
            source_job_id: input.job_id.to_owned(),
            created_at: input.created_at,
            extra: ExtraFields::new(),
        },
        extra: ExtraFields::new(),
    })
}

fn combined_trigger_words(item_words: &[String], output_words: &[String]) -> Vec<String> {
    let mut words = Vec::new();
    for word in output_words.iter().chain(item_words.iter()) {
        let trimmed = word.trim();
        if !trimmed.is_empty()
            && !words
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(trimmed))
        {
            words.push(trimmed.to_owned());
        }
    }
    words
}

fn caption_with_trigger_words(caption: &str, trigger_words: &[String]) -> String {
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

/// Joins a dataset item's forward-slash relative path onto the absolute root
/// using the host's path separator. Dataset item paths are stored POSIX-style
/// (the store forbids backslashes), so joining the whole string would leave
/// mixed separators on Windows; pushing each component normalizes them.
fn resolve_item_path(dataset_root: &Path, relative_path: &str) -> String {
    let mut path = dataset_root.to_path_buf();
    for component in Path::new(relative_path).components() {
        path.push(component);
    }
    path.display().to_string()
}

fn validate_training_config(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    let positive = |value: u32, field: &str| {
        if value == 0 {
            Err(TrainingPlanError::InvalidConfig(format!(
                "{field} must be at least 1."
            )))
        } else {
            Ok(())
        }
    };
    positive(config.rank, "rank")?;
    positive(config.alpha, "alpha")?;
    positive(config.steps, "steps")?;
    positive(config.resolution, "resolution")?;
    positive(config.batch_size, "batchSize")?;
    positive(config.gradient_accumulation, "gradientAccumulation")?;
    let learning_rate = config.learning_rate.as_f64().unwrap_or(f64::NAN);
    if !(learning_rate.is_finite() && learning_rate > 0.0) {
        return Err(TrainingPlanError::InvalidConfig(
            "learningRate must be a positive finite number.".to_owned(),
        ));
    }
    Ok(())
}
