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
use crate::store_util::is_safe_relative_path;

/// Schema version stamped on persisted training contracts.
pub const TRAINING_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Version of the normalized [`TrainingPlan`] handed to the Python kernel. The
/// kernel rejects plans whose `planVersion` it does not understand.
pub const TRAINING_PLAN_VERSION: u32 = 1;

/// Learning-rate schedulers both worker kernels honor. Submit-time validation
/// rejects any `advanced.lrScheduler` outside this set so a non-constant choice
/// is never accepted as no-op metadata. This is the *learning-rate* scheduler;
/// the flow-matching noise scheduler is configured separately via
/// `advanced.timestepType`/`timestepBias`.
pub const SUPPORTED_LR_SCHEDULERS: [&str; 3] = ["constant", "linear", "cosine"];

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
    /// Owning character, when the dataset is scoped to one (sc-2022). `None` for
    /// general (style / standalone) datasets. Set when images are imported from a
    /// character or the dataset is created from a character's images.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_id: Option<String>,
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

/// A named, model-aware training preset. Presets are complete configs for the
/// initial built-in set so callers always submit concrete worker values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPreset {
    /// Stable preset id, e.g. `z_image_turbo_lora.character.adamw8bit.balanced`.
    pub id: String,
    /// Monotonic preset version. Training provenance pins this value.
    pub version: u32,
    pub target_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recommended_for: Vec<String>,
    pub optimizer: String,
    pub quality_preset: String,
    pub config: TrainingConfig,
    /// Presentation hints (short descriptions, ordering, default flags).
    pub ui: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

/// The registry of available built-in training presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPresetRegistry {
    pub schema_version: u32,
    pub presets: Vec<TrainingPreset>,
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
    /// Optional built-in preset used as the starting point for this config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_version: Option<u32>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_config_snapshot: Option<JsonObject>,
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
        targets: vec![
            z_image_turbo_lora_target(),
            sdxl_lora_target(),
            kolors_lora_target(),
            lens_turbo_lora_target(),
            ltx_video_lora_target(),
            wan_lora_target(),
            wan_t2v_14b_lora_target(),
            wan_i2v_14b_lora_target(),
        ],
        extra: ExtraFields::new(),
    }
}

/// The built-in training presets Rust owns out of the box.
pub fn builtin_training_presets() -> TrainingPresetRegistry {
    let target = z_image_turbo_lora_target();
    let sdxl_target = sdxl_lora_target();
    let kolors_target = kolors_lora_target();
    let wan_target = wan_lora_target();
    let wan_t2v_14b_target = wan_t2v_14b_lora_target();
    let wan_i2v_14b_target = wan_i2v_14b_lora_target();
    TrainingPresetRegistry {
        schema_version: TRAINING_CONTRACT_SCHEMA_VERSION,
        presets: vec![
            z_image_preset(
                &target,
                "z_image_turbo_lora.character.adamw8bit.balanced",
                "Character balanced",
                &["character"],
                ("adamw8bit", "balanced"),
                |mut config| {
                    config.steps = 3000;
                    config
                },
                object(json!({
                    "description": "Research-backed first run for 12-25 clean character images.",
                    "default": true,
                    "order": 10
                })),
            ),
            z_image_preset(
                &target,
                "z_image_turbo_lora.character.adamw8bit.conservative",
                "Character conservative",
                &["character"],
                ("adamw8bit", "conservative"),
                |mut config| {
                    config.rank = 8;
                    config.alpha = 8;
                    config.learning_rate = number(0.00005);
                    config.steps = 3000;
                    config
                },
                object(json!({
                    "description": "Lower-rank, lower-LR character preset for noisy early samples or tight identity datasets.",
                    "order": 20
                })),
            ),
            z_image_preset(
                &target,
                "z_image_turbo_lora.character.adamw.balanced",
                "Character balanced (AdamW)",
                &["character"],
                ("adamw", "balanced"),
                |mut config| {
                    config.optimizer = "adamw".to_owned();
                    config.steps = 3000;
                    config
                },
                object(json!({
                    "description": "Balanced character LoRA defaults for full AdamW.",
                    "order": 30
                })),
            ),
            z_image_preset(
                &target,
                "z_image_turbo_lora.character.prodigyopt.balanced",
                "Prodigy character (experimental)",
                &["character"],
                ("prodigyopt", "balanced"),
                |mut config| {
                    config.optimizer = "prodigyopt".to_owned();
                    config.learning_rate = number(1.0);
                    config.steps = 1600;
                    config.save_every = 200;
                    config.advanced.insert("sampleEvery".to_owned(), json!(200));
                    config
                        .advanced
                        .insert("optimizerDCoef".to_owned(), json!(1.0));
                    config
                },
                object(json!({
                    "description": "Experimental Prodigy optimizer variant; prefer AdamW8bit for first runs.",
                    "experimental": true,
                    "order": 40
                })),
            ),
            z_image_preset(
                &target,
                "z_image_turbo_lora.style.adamw8bit.balanced",
                "Style balanced",
                &["style"],
                ("adamw8bit", "balanced"),
                |mut config| {
                    config.rank = 32;
                    config.alpha = 16;
                    config.steps = 3000;
                    config.advanced.insert("sampleEvery".to_owned(), json!(250));
                    config
                },
                object(json!({
                    "description": "Higher-capacity style LoRA defaults for texture and visual-language transfer.",
                    "order": 50
                })),
            ),
            z_image_preset(
                &target,
                "z_image_turbo_lora.character.adamw8bit.low_vram",
                "Low VRAM character",
                &["character"],
                ("adamw8bit", "low_vram"),
                |mut config| {
                    config.rank = 8;
                    config.alpha = 8;
                    config.resolution = 768;
                    config.steps = 2000;
                    config.save_every = 250;
                    config
                        .advanced
                        .insert("qualityPreset".to_owned(), json!("low_vram"));
                    config.advanced.insert("sampleEvery".to_owned(), json!(250));
                    config
                        .advanced
                        .insert("cacheLatents".to_owned(), json!(true));
                    config
                },
                object(json!({
                    "description": "12GB-friendly 768px preset based on public Z-Image Turbo training reports.",
                    "order": 60
                })),
            ),
            sdxl_preset(
                &sdxl_target,
                "sdxl_lora.character.adamw8bit.balanced",
                "Character balanced",
                &["character"],
                ("adamw8bit", "balanced"),
                |config| config,
                object(json!({
                    "description": "Balanced first run for 12-25 clean character images on SDXL.",
                    "default": true,
                    "order": 10
                })),
            ),
            sdxl_preset(
                &sdxl_target,
                "sdxl_lora.character.adamw8bit.conservative",
                "Character conservative",
                &["character"],
                ("adamw8bit", "conservative"),
                |mut config| {
                    config.rank = 8;
                    config.alpha = 8;
                    config.learning_rate = number(0.00005);
                    config
                },
                object(json!({
                    "description": "Lower-rank, lower-LR character preset for tight identity datasets.",
                    "order": 20
                })),
            ),
            sdxl_preset(
                &sdxl_target,
                "sdxl_lora.style.adamw8bit.balanced",
                "Style balanced",
                &["style"],
                ("adamw8bit", "balanced"),
                |mut config| {
                    config.rank = 32;
                    config.alpha = 16;
                    config
                },
                object(json!({
                    "description": "Higher-capacity style LoRA defaults for texture and look transfer.",
                    "order": 30
                })),
            ),
            sdxl_preset(
                &sdxl_target,
                "sdxl_lora.character.adamw8bit.low_vram",
                "Low VRAM character",
                &["character"],
                ("adamw8bit", "low_vram"),
                |mut config| {
                    config.rank = 8;
                    config.alpha = 8;
                    config.resolution = 768;
                    config.steps = 1200;
                    config
                },
                object(json!({
                    "description": "768px preset for tighter-VRAM SDXL training.",
                    "order": 40
                })),
            ),
            // Kolors reuses the SDXL preset recipe (same U-Net architecture); only
            // the target (pipeline + ChatGLM3 encoder) and the LoRA family differ.
            sdxl_preset(
                &kolors_target,
                "kolors_lora.character.adamw8bit.balanced",
                "Character balanced",
                &["character"],
                ("adamw8bit", "balanced"),
                |config| config,
                object(json!({
                    "description": "Balanced first run for 12-25 clean character images on Kolors.",
                    "default": true,
                    "order": 10
                })),
            ),
            sdxl_preset(
                &kolors_target,
                "kolors_lora.character.adamw8bit.conservative",
                "Character conservative",
                &["character"],
                ("adamw8bit", "conservative"),
                |mut config| {
                    config.rank = 8;
                    config.alpha = 8;
                    config.learning_rate = number(0.00005);
                    config
                },
                object(json!({
                    "description": "Lower-rank, lower-LR character preset for tight identity datasets.",
                    "order": 20
                })),
            ),
            sdxl_preset(
                &kolors_target,
                "kolors_lora.style.adamw8bit.balanced",
                "Style balanced",
                &["style"],
                ("adamw8bit", "balanced"),
                |mut config| {
                    config.rank = 32;
                    config.alpha = 16;
                    config
                },
                object(json!({
                    "description": "Higher-capacity style LoRA defaults for texture and look transfer.",
                    "order": 30
                })),
            ),
            sdxl_preset(
                &kolors_target,
                "kolors_lora.character.adamw8bit.low_vram",
                "Low VRAM character",
                &["character"],
                ("adamw8bit", "low_vram"),
                |mut config| {
                    config.rank = 8;
                    config.alpha = 8;
                    config.resolution = 768;
                    config.steps = 1200;
                    config
                },
                object(json!({
                    "description": "768px preset for tighter-VRAM Kolors training.",
                    "order": 40
                })),
            ),
            wan_preset(
                &wan_target,
                "wan_lora.character.adamw.balanced",
                "Character balanced",
                &["character"],
                ("adamw", "balanced"),
                |config| config,
                object(json!({
                    "description": "Balanced first run for 12-25 clean character stills on Wan2.2-TI2V-5B.",
                    "default": true,
                    "order": 10
                })),
            ),
            wan_preset(
                &wan_target,
                "wan_lora.character.adamw.conservative",
                "Character conservative",
                &["character"],
                ("adamw", "conservative"),
                |mut config| {
                    config.rank = 16;
                    config.alpha = 16;
                    config.learning_rate = number(0.00005);
                    config
                },
                object(json!({
                    "description": "Lower-rank, lower-LR character preset for tight identity datasets.",
                    "order": 20
                })),
            ),
            wan_preset(
                &wan_target,
                "wan_lora.style.adamw.balanced",
                "Style balanced",
                &["style"],
                ("adamw", "balanced"),
                |mut config| {
                    config.steps = 2000;
                    config
                },
                object(json!({
                    "description": "Higher-step style LoRA defaults for look and motion-free texture transfer.",
                    "order": 30
                })),
            ),
            wan_preset(
                &wan_t2v_14b_target,
                "wan_t2v_14b_lora.character.adamw.balanced",
                "Character balanced",
                &["character"],
                ("adamw", "balanced"),
                |config| config,
                object(json!({
                    "description": "Balanced first run for character stills on Wan2.2 14B T2V (both noise experts).",
                    "default": true,
                    "order": 10
                })),
            ),
            wan_preset(
                &wan_t2v_14b_target,
                "wan_t2v_14b_lora.style.adamw.balanced",
                "Style balanced",
                &["style"],
                ("adamw", "balanced"),
                |mut config| {
                    config.steps = 2000;
                    config
                },
                object(json!({
                    "description": "Higher-step style LoRA for Wan2.2 14B T2V.",
                    "order": 20
                })),
            ),
            wan_preset(
                &wan_i2v_14b_target,
                "wan_i2v_14b_lora.character.adamw.balanced",
                "Character balanced",
                &["character"],
                ("adamw", "balanced"),
                |config| config,
                object(json!({
                    "description": "Balanced first run for character stills on Wan2.2 14B I2V (both noise experts).",
                    "default": true,
                    "order": 10
                })),
            ),
            wan_preset(
                &wan_i2v_14b_target,
                "wan_i2v_14b_lora.style.adamw.balanced",
                "Style balanced",
                &["style"],
                ("adamw", "balanced"),
                |mut config| {
                    config.steps = 2000;
                    config
                },
                object(json!({
                    "description": "Higher-step style LoRA for Wan2.2 14B I2V.",
                    "order": 20
                })),
            ),
        ],
        extra: ExtraFields::new(),
    }
}

fn z_image_preset<F>(
    target: &TrainingTarget,
    id: &str,
    name: &str,
    recommended_for: &[&str],
    optimizer_quality: (&str, &str),
    mutate: F,
    ui: JsonObject,
) -> TrainingPreset
where
    F: FnOnce(TrainingConfig) -> TrainingConfig,
{
    let (optimizer, quality_preset) = optimizer_quality;
    let mut config = mutate(target.defaults.clone());
    config.optimizer = optimizer.to_owned();
    config
        .advanced
        .insert("qualityPreset".to_owned(), json!(quality_preset));
    config.advanced.insert("sampleSteps".to_owned(), json!(8));
    config
        .advanced
        .insert("sampleGuidanceScale".to_owned(), json!(1.0));
    config.advanced.insert(
        "trainingAdapterRepo".to_owned(),
        json!("ostris/zimage_turbo_training_adapter"),
    );
    config
        .advanced
        .insert("trainingAdapterVersion".to_owned(), json!("v2-default"));
    config
        .advanced
        .insert("timestepType".to_owned(), json!("sigmoid"));
    config
        .advanced
        .insert("timestepBias".to_owned(), json!("high_noise"));
    config.advanced.insert("lossType".to_owned(), json!("mse"));
    config
        .advanced
        .insert("gradientCheckpointing".to_owned(), json!(true));
    config
        .advanced
        .insert("cacheTextEmbeddings".to_owned(), json!(true));
    config
        .advanced
        .insert("weightDecay".to_owned(), json!(0.0001));
    TrainingPreset {
        id: id.to_owned(),
        version: 1,
        target_id: target.id.clone(),
        name: name.to_owned(),
        recommended_for: recommended_for
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        optimizer: optimizer.to_owned(),
        quality_preset: quality_preset.to_owned(),
        config,
        ui,
        extra: ExtraFields::new(),
    }
}

fn sdxl_preset<F>(
    target: &TrainingTarget,
    id: &str,
    name: &str,
    recommended_for: &[&str],
    optimizer_quality: (&str, &str),
    mutate: F,
    ui: JsonObject,
) -> TrainingPreset
where
    F: FnOnce(TrainingConfig) -> TrainingConfig,
{
    let (optimizer, quality_preset) = optimizer_quality;
    let mut config = mutate(target.defaults.clone());
    config.optimizer = optimizer.to_owned();
    config
        .advanced
        .insert("qualityPreset".to_owned(), json!(quality_preset));
    // SDXL uses real CFG, so previews render at a positive guidance (the
    // flow-matching Z-Image presets sample at guidance 0.0). No de-distill
    // training adapter applies (SDXL base is not step-distilled).
    config.advanced.insert("sampleSteps".to_owned(), json!(30));
    config
        .advanced
        .insert("sampleGuidanceScale".to_owned(), json!(7.0));
    config.advanced.insert("lossType".to_owned(), json!("mse"));
    config
        .advanced
        .insert("gradientCheckpointing".to_owned(), json!(true));
    config
        .advanced
        .insert("cacheTextEmbeddings".to_owned(), json!(true));
    config
        .advanced
        .insert("weightDecay".to_owned(), json!(0.0001));
    TrainingPreset {
        id: id.to_owned(),
        version: 1,
        target_id: target.id.clone(),
        name: name.to_owned(),
        recommended_for: recommended_for
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        optimizer: optimizer.to_owned(),
        quality_preset: quality_preset.to_owned(),
        config,
        ui,
        extra: ExtraFields::new(),
    }
}

/// Build a Wan2.2 video LoRA preset. The video-specific advanced knobs
/// (flow-matching sampling, caching, gradient checkpointing, numFrames) live on
/// the target defaults, so the preset only overrides the optimizer + quality
/// label and whatever the `mutate` closure tweaks (rank/steps/LR).
fn wan_preset<F>(
    target: &TrainingTarget,
    id: &str,
    name: &str,
    recommended_for: &[&str],
    optimizer_quality: (&str, &str),
    mutate: F,
    ui: JsonObject,
) -> TrainingPreset
where
    F: FnOnce(TrainingConfig) -> TrainingConfig,
{
    let (optimizer, quality_preset) = optimizer_quality;
    let mut config = mutate(target.defaults.clone());
    config.optimizer = optimizer.to_owned();
    config
        .advanced
        .insert("qualityPreset".to_owned(), json!(quality_preset));
    TrainingPreset {
        id: id.to_owned(),
        version: 1,
        target_id: target.id.clone(),
        name: name.to_owned(),
        recommended_for: recommended_for
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        optimizer: optimizer.to_owned(),
        quality_preset: quality_preset.to_owned(),
        config,
        ui,
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
            steps: 3000,
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
                "cacheTextEmbeddings": true,
                "gradientCheckpointing": true,
                "networkType": "lora",
                "timestepType": "sigmoid",
                "timestepBias": "high_noise",
                "lossType": "mse",
                "weightDecay": 0.0001,
                // Learning-rate scheduler, distinct from the flow-matching noise
                // scheduler driven by `timestepType`/`timestepBias`. `constant`
                // holds the optimizer LR fixed for the whole run (matching every
                // pre-scheduler training run); the worker also honors `linear` and
                // `cosine`, with an optional `lrWarmupSteps` ramp.
                "lrScheduler": "constant",
                "sampleEvery": 250,
                "sampleSteps": 8,
                "sampleGuidanceScale": 1.0,
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
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
            // Network parameterization for the adapter. `lora` is the universal
            // default; `lokr` (LyCORIS Kronecker) is a torch/PEFT-only option
            // advertised on the validated image backends (epic 2193).
            "networkTypes": ["lora", "lokr"],
            "lrSchedulers": ["constant", "linear", "cosine"],
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

/// Image LoRA training for Microsoft Lens, applied at inference to Lens-Turbo.
///
/// Lens-Turbo is a 4-step *distillation* of `microsoft/Lens`; training a LoRA
/// directly on the distilled velocity field drifts (the same trap that produced
/// off-identity LoRAs on Z-Image-Turbo). So this target trains against the
/// non-distilled `microsoft/Lens` (20-step, CFG 5.0) — the direct weight-parent
/// of Turbo, so the adapter transfers cleanly — and the output registers as a
/// `lens` family LoRA the Lens adapter loads onto Lens-Turbo at generation time.
/// `base_model_repo` points the plan's `baseModelPath` at the Lens HF cache,
/// independent of the served `lens_turbo` model. `microsoft/Lens-Base` (50-step
/// supervised) is the fallback base if RL-tuning hurts LoRA stability; override
/// via `advanced.baseModelRepo` (sc-1584).
///
/// Unlike Z-Image's separate `to_q/to_k/to_v`, Lens uses *fused* QKV attention
/// (`img_qkv`/`txt_qkv`) plus joint-attention output projections. The image output
/// `to_out` is an `nn.ModuleList([Linear, Identity])`, so the trainable Linear is
/// `to_out.0` (PEFT errors if pointed at the ModuleList — sc-2218); `to_add_out` is
/// a plain Linear. The Z-Image defaults would match nothing and inject no adapter.
fn lens_turbo_lora_target() -> TrainingTarget {
    TrainingTarget {
        id: "lens_turbo_lora".to_owned(),
        name: "Lens LoRA".to_owned(),
        modality: TrainingModality::Image,
        output_kind: TrainingOutputKind::Lora,
        family: "lens".to_owned(),
        base_model: "lens".to_owned(),
        base_model_repo: Some("microsoft/Lens".to_owned()),
        kernel: "lens_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 16,
            alpha: 16,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 3000,
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
                "cacheTextEmbeddings": true,
                "gradientCheckpointing": true,
                "networkType": "lora",
                "timestepType": "sigmoid",
                "timestepBias": "high_noise",
                "lossType": "mse",
                "weightDecay": 0.0001,
                // Learning-rate scheduler (see the Z-Image target); the worker
                // honors `constant`/`linear`/`cosine` with an optional warmup.
                "lrScheduler": "constant",
                // Lens uses fused QKV attention plus joint-attention output
                // projections, NOT Z-Image's separate to_q/to_k/to_v. ``to_out`` is
                // a ModuleList, so its Linear is ``to_out.0`` — PEFT (LoRA + LoKr)
                // errors on the ModuleList itself (sc-2218). Suffixes must match the
                // LensTransformer2DModel module names or PEFT injects nothing.
                "loraTargetModules": ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"],
                // Non-distilled training base; override with "microsoft/Lens-Base"
                // to train on the 50-step supervised checkpoint instead (sc-1584).
                "baseModelRepo": "microsoft/Lens",
                // In-training previews render on the loaded base model (not the
                // distilled Turbo), so they use the base's 20-step / CFG 5.0
                // settings — 4-step / CFG 1.0 on the non-distilled base is garbage.
                // Heavier than Turbo previews, so a wider cadence by default.
                "sampleEvery": 500,
                "sampleSteps": 20,
                "sampleGuidanceScale": 5.0,
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
            // Lens snaps onto 1024/1440 base-resolution buckets; 768 is allowed
            // for lower-VRAM runs.
            "resolutions": [768, 1024, 1440],
            "batchSize": [1, 4],
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
            // Lens trains + infers in a separate sidecar venv; both the trainer
            // (lens_train_runner) and inference (lens_runner) gained PEFT LoKr
            // build/save + injection (epic 2193, sc-2218), keyed off the fused-QKV
            // target modules above.
            "networkTypes": ["lora", "lokr"],
            "lrSchedulers": ["constant", "linear", "cosine"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Lens LoRA",
            "description": "Train an image LoRA for Microsoft Lens (gpt-oss text encoder + FLUX.2 VAE). Trains on the non-distilled base and applies to Lens-Turbo.",
            "recommendedFor": ["character", "style"]
        })),
        extra: ExtraFields::new(),
    }
}

/// Native MLX LoRA training for LTX-2.3 video, trained from a still-image dataset.
///
/// Apple-Silicon only: the `ltx_mlx_lora` kernel runs a native MLX (mlx.core /
/// mlx.optimizers) QLoRA loop against the quantized LTX transformer, so the
/// worker only advertises/accepts it on macOS with MLX (gated in story 1538).
/// `modality` is the *output* modality (a video LoRA); the consumed dataset is
/// images (the only dataset modality the store supports), surfaced via
/// `ui.datasetModality`. The output registers as an `ltx-video` family LoRA the
/// MLX LTX video adapter loads at inference.
fn ltx_video_lora_target() -> TrainingTarget {
    TrainingTarget {
        id: "ltx_video_lora".to_owned(),
        name: "LTX-2.3 Video LoRA".to_owned(),
        modality: TrainingModality::Video,
        output_kind: TrainingOutputKind::Lora,
        family: "ltx-video".to_owned(),
        base_model: "ltx_2_3".to_owned(),
        // Mirrors the generation load path (sc-5608): the turnkey SceneWorks LTX-2.3 MLX bundle,
        // replacing the third-party notapalindrome mirror. Informational here — the MLX kernel loads
        // the base from `base_model_path` (the app-managed dir), not this repo.
        base_model_repo: Some("SceneWorks/ltx-2.3-mlx".to_owned()),
        kernel: "ltx_mlx_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 32,
            alpha: 32,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 1500,
            batch_size: 1,
            gradient_accumulation: 1,
            resolution: 768,
            save_every: 250,
            seed: 42,
            // MLX trains with mlx.optimizers.AdamW; bitsandbytes 8-bit is CUDA-only
            // and never applies here.
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                "cacheLatents": true,
                "networkType": "lora",
                // Learning-rate scheduler (see the Z-Image target). MLX honors the
                // same `constant`/`linear`/`cosine` set via mlx.optimizers schedules.
                "lrScheduler": "constant",
                "backend": "mlx",
                // Still-image training: each item encodes to a single latent frame.
                "numFrames": 1,
                "loraTargetModules": ["to_q", "to_k", "to_v", "to_out"],
                "sampleEvery": 250,
                "qualityPreset": "balanced",
                "outputScope": "project",
                "requestedGpu": "auto"
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [4, 128],
            "alpha": [1, 128],
            "steps": [200, 4000],
            "resolutions": [512, 768, 1024],
            "batchSize": [1, 2],
            // MLX backend: the LoKr inference path (Kronecker merge) is out of
            // scope for epic 2193 v1, so this target stays `lora`-only.
            "networkTypes": ["lora"],
            "lrSchedulers": ["constant", "linear", "cosine"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"],
            "requiresBackend": "mlx",
            "appleSiliconOnly": true
        })),
        ui: object(json!({
            "label": "LTX-2.3 Video LoRA",
            "description": "Train an LTX-2.3 video LoRA from still images. Apple Silicon only (native MLX).",
            "recommendedFor": ["character", "style"],
            "appleSiliconOnly": true,
            "backend": "mlx",
            "datasetModality": "image"
        })),
        extra: ExtraFields::new(),
    }
}

/// Wan2.2-TI2V-5B video LoRA training, applied at inference to the Wan video
/// models.
///
/// Like the LTX video LoRA it trains from a still-image dataset (each item
/// encodes to a single Wan-VAE latent frame, `numFrames: 1`), but the `wan_lora`
/// kernel is torch/diffusers — CUDA *and* Apple-Silicon MPS — not MLX: a
/// flow-matching velocity loop on the `WanTransformer3DModel` attention
/// projections (`to_q`/`to_k`/`to_v`/`to_out.0`). On MPS it runs in fp32 (Wan's
/// Conv3d patch embedding has no bf16 Metal kernel; the kernel forces it). The
/// output registers as a `wan-video` family LoRA the Wan video adapter loads at
/// generation. This dense 5B target is the base the 14B (A14B MoE) trainer
/// extends for the two-expert (high/low-noise) case.
fn wan_lora_target() -> TrainingTarget {
    TrainingTarget {
        id: "wan_lora".to_owned(),
        name: "Wan2.2 Video LoRA".to_owned(),
        modality: TrainingModality::Video,
        output_kind: TrainingOutputKind::Lora,
        family: "wan-video".to_owned(),
        base_model: "wan_2_2".to_owned(),
        base_model_repo: Some("Wan-AI/Wan2.2-TI2V-5B-Diffusers".to_owned()),
        kernel: "wan_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 32,
            alpha: 32,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 1500,
            batch_size: 1,
            gradient_accumulation: 1,
            resolution: 512,
            save_every: 250,
            seed: 42,
            // torch AdamW: cross-platform (CUDA + MPS). adamw8bit is CUDA-only
            // (bitsandbytes) and falls back to AdamW elsewhere, so the default
            // stays plain adamw like the LTX video target.
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                "cacheLatents": true,
                "cacheTextEmbeddings": true,
                "gradientCheckpointing": true,
                "networkType": "lora",
                // Flow-matching noise sampling (see the Z-Image target). Video uses
                // a neutral balanced bias rather than the high-noise bias the
                // distilled image models prefer.
                "timestepType": "sigmoid",
                "timestepBias": "balanced",
                "lossType": "mse",
                "weightDecay": 0.0001,
                "lrScheduler": "constant",
                // Still-image training: each item encodes to a single latent frame
                // (mirrors the LTX video LoRA). The kernel keeps the Wan-VAE 5D
                // latent shape so a future clip dataset can pass numFrames > 1.
                "numFrames": 1,
                // Wan transformer attention projections (attn1/attn2 to_q/k/v + the
                // attention output projection), matching the diffusers Wan LoRA.
                "loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"],
                // In-training video sampling is not implemented for the first cut
                // (per-step Wan video gen is expensive), so previews stay off.
                "sampleEvery": 0,
                "qualityPreset": "balanced",
                "outputScope": "project",
                "requestedGpu": "auto"
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [4, 128],
            "alpha": [1, 128],
            "steps": [200, 4000],
            "resolutions": [512, 768],
            "batchSize": [1, 2],
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
            // Wan2.2 TI2V-5B is a torch/diffusers backend: the PEFT LoKr trainer
            // (sc-2196) applies and the diffusers video inference loads it via PEFT
            // injection (sc-2197/2211). MLX video has no Kronecker merge yet, so a
            // LoKr Wan job falls back to the torch path (sc-2211; native MLX is sc-2213).
            "networkTypes": ["lora", "lokr"],
            "lrSchedulers": ["constant", "linear", "cosine"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Wan2.2 Video LoRA",
            "description": "Train a Wan2.2 video LoRA from still images (Wan2.2-TI2V-5B). Runs on CUDA and Apple Silicon (MPS).",
            "recommendedFor": ["character", "style"],
            "datasetModality": "image"
        })),
        extra: ExtraFields::new(),
    }
}

/// Shared builder for the Wan2.2 A14B (MoE) video LoRA targets (T2V + I2V).
///
/// A14B is a two-expert mixture: a high-noise expert (`transformer`) and a
/// low-noise expert (`transformer_2`), so the `wan_moe_lora` kernel trains a
/// separate LoRA on each, split at the pipeline `boundary_ratio` (0.875), and
/// saves two `wan-video` family files. Same diffusers flow-matching recipe as the
/// dense 5B `wan_lora` target (still-image dataset, fp32 on MPS), but the A14B
/// bf16 base is GPU-only (~56GB of transformers); the Q8_0 GGUF base path fits
/// memory-bound hosts. `base_model` records the specific A14B variant so the
/// inference loader gates the LoRA to the matching model (not the 5B).
fn wan_moe_lora_target(
    id: &str,
    name: &str,
    base_model: &str,
    base_model_repo: &str,
    description: &str,
) -> TrainingTarget {
    TrainingTarget {
        id: id.to_owned(),
        name: name.to_owned(),
        modality: TrainingModality::Video,
        output_kind: TrainingOutputKind::Lora,
        family: "wan-video".to_owned(),
        base_model: base_model.to_owned(),
        base_model_repo: Some(base_model_repo.to_owned()),
        kernel: "wan_moe_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 32,
            alpha: 32,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 1500,
            batch_size: 1,
            gradient_accumulation: 1,
            resolution: 512,
            save_every: 250,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                "cacheLatents": true,
                "cacheTextEmbeddings": true,
                "gradientCheckpointing": true,
                "networkType": "lora",
                "timestepType": "sigmoid",
                "timestepBias": "balanced",
                "lossType": "mse",
                "weightDecay": 0.0001,
                "lrScheduler": "constant",
                "numFrames": 1,
                "loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"],
                "sampleEvery": 0,
                "qualityPreset": "balanced",
                "outputScope": "project",
                "requestedGpu": "auto"
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [4, 128],
            "alpha": [1, 128],
            "steps": [200, 4000],
            "resolutions": [512, 768],
            "batchSize": [1, 2],
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
            // Wan is a torch backend, but epic 2193 v1 validates LoKr on the
            // image backends (Z-Image/SDXL) first; video stays `lora`-only.
            "networkTypes": ["lora"],
            "lrSchedulers": ["constant", "linear", "cosine"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": name,
            "description": description,
            "recommendedFor": ["character", "style"],
            "datasetModality": "image"
        })),
        extra: ExtraFields::new(),
    }
}

fn wan_t2v_14b_lora_target() -> TrainingTarget {
    wan_moe_lora_target(
        "wan_t2v_14b_lora",
        "Wan2.2 14B T2V Video LoRA",
        "wan_2_2_t2v_14b",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Train a Wan2.2 14B (A14B MoE, text-to-video) LoRA from still images. Trains both noise experts; GPU-only at bf16 (Q8_0 GGUF base fits smaller hosts).",
    )
}

fn wan_i2v_14b_lora_target() -> TrainingTarget {
    wan_moe_lora_target(
        "wan_i2v_14b_lora",
        "Wan2.2 14B I2V Video LoRA",
        "wan_2_2_i2v_14b",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Train a Wan2.2 14B (A14B MoE, image-to-video) LoRA from still images. Trains both noise experts; GPU-only at bf16 (Q8_0 GGUF base fits smaller hosts).",
    )
}

/// Generic SDXL-UNet image LoRA training, applied at inference to the SDXL base
/// model (and the shared foundation epic 1929 extends for Kolors).
///
/// SDXL is the first U-Net (non-DiT) training target: the `sdxl_lora` kernel
/// runs an epsilon/v-prediction loop on a DDPM noise schedule with the SDXL
/// `added_cond_kwargs` (pooled text embeds + add_time_ids), unlike the
/// flow-matching transformer kernels. The LoRA injects into the SDXL UNet's
/// separate `to_q`/`to_k`/`to_v` attention projections plus the `to_out.0`
/// output projection; the output registers as an `sdxl` family LoRA the SDXL
/// adapter loads at generation time.
fn sdxl_lora_target() -> TrainingTarget {
    TrainingTarget {
        id: "sdxl_lora".to_owned(),
        name: "Stable Diffusion XL LoRA".to_owned(),
        modality: TrainingModality::Image,
        output_kind: TrainingOutputKind::Lora,
        family: "sdxl".to_owned(),
        base_model: "sdxl".to_owned(),
        base_model_repo: Some("stabilityai/stable-diffusion-xl-base-1.0".to_owned()),
        kernel: "sdxl_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 16,
            alpha: 16,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 1500,
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
                "cacheTextEmbeddings": true,
                "gradientCheckpointing": true,
                "networkType": "lora",
                "lossType": "mse",
                "weightDecay": 0.0001,
                // Learning-rate scheduler (see the Z-Image target); the worker
                // honors `constant`/`linear`/`cosine` with an optional warmup.
                "lrScheduler": "constant",
                // SDXL UNet attention modules: separate q/k/v projections plus the
                // attention output projection. These match the PEFT target names
                // the diffusers SDXL DreamBooth-LoRA reference uses.
                "loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"],
                // SDXL uses real classifier-free guidance, so in-training previews
                // render at a positive guidance (unlike distilled Z-Image at 0.0).
                "sampleEvery": 250,
                "sampleSteps": 30,
                "sampleGuidanceScale": 7.0,
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
            "resolutions": [768, 1024],
            "batchSize": [1, 4],
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
            // See the Z-Image target: `lokr` (LyCORIS Kronecker) is advertised on
            // the validated torch/PEFT image backends (epic 2193).
            "networkTypes": ["lora", "lokr"],
            "lrSchedulers": ["constant", "linear", "cosine"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Stable Diffusion XL LoRA",
            "description": "Train an image LoRA for Stable Diffusion XL. The generic SDXL-UNet trainer and the shared foundation for SDXL-family models.",
            "recommendedFor": ["character", "style"]
        })),
        extra: ExtraFields::new(),
    }
}

/// Kolors LoRA training target (epic 1929).
///
/// Kolors (Kwai-Kolors) is an SDXL-architecture U-Net with a ChatGLM3-6B text
/// encoder + SDXL VAE and the same epsilon/v-prediction objective, so it reuses
/// the generic SDXL-UNet trainer wholesale (same DDPM loop, same `added_cond_kwargs`,
/// same `to_q`/`to_k`/`to_v`/`to_out.0` attention target modules) via a thin
/// `kolors_lora` kernel that only swaps the pipeline class + the ChatGLM3 prompt
/// encoder. The output registers as a `kolors` family LoRA the Kolors image
/// adapter loads at generation time.
fn kolors_lora_target() -> TrainingTarget {
    TrainingTarget {
        id: "kolors_lora".to_owned(),
        name: "Kolors LoRA".to_owned(),
        modality: TrainingModality::Image,
        output_kind: TrainingOutputKind::Lora,
        family: "kolors".to_owned(),
        base_model: "kolors".to_owned(),
        base_model_repo: Some("Kwai-Kolors/Kolors-diffusers".to_owned()),
        kernel: "kolors_lora".to_owned(),
        defaults: TrainingConfig {
            rank: 16,
            alpha: 16,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            steps: 1500,
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
                "cacheTextEmbeddings": true,
                "gradientCheckpointing": true,
                "networkType": "lora",
                "lossType": "mse",
                "weightDecay": 0.0001,
                "lrScheduler": "constant",
                // Kolors shares the SDXL UNet attention module names.
                "loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"],
                "sampleEvery": 250,
                "sampleSteps": 30,
                "sampleGuidanceScale": 7.0,
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
            "resolutions": [768, 1024],
            "batchSize": [1, 4],
            "optimizers": ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
            // Kolors is an SDXL-architecture torch/PEFT backend, so it advertises
            // `lokr` like the SDXL target — inherited from the shared SDXL backend's
            // LoKr save + PEFT-injection inference path (epic 2193, sc-2217).
            "networkTypes": ["lora", "lokr"],
            "lrSchedulers": ["constant", "linear", "cosine"],
            "qualityPresets": ["speed", "balanced", "quality"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Kolors LoRA",
            "description": "Train an image LoRA for Kolors (Kwai-Kolors). Runs the SDXL-UNet trainer with the Kolors pipeline + ChatGLM3 text encoder, on CUDA and Apple Silicon (MPS).",
            "recommendedFor": ["character", "style"]
        })),
        extra: ExtraFields::new(),
    }
}

fn number(value: f64) -> ContractNumber {
    ContractNumber::from_f64(value).expect("builtin preset number is finite")
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
    /// Selected preset metadata, when the user started from a named preset.
    pub preset: Option<TrainingPresetProvenance>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingPresetProvenance {
    pub preset_id: String,
    pub preset_version: u32,
    pub preset_name: String,
    pub preset_config_snapshot: JsonObject,
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
        .map(|item| {
            Ok(TrainingPlanItem {
                image_path: resolve_item_path(input.dataset_root, &item.path)?,
                caption: caption_with_trigger_words(
                    &item.caption.text,
                    &combined_trigger_words(&item.caption.trigger_words, &trigger_words),
                ),
                width: item.width,
                height: item.height,
                extra: ExtraFields::new(),
            })
        })
        .collect::<Result<Vec<_>, TrainingPlanError>>()?;

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
            preset_id: input.preset.as_ref().map(|preset| preset.preset_id.clone()),
            preset_version: input.preset.as_ref().map(|preset| preset.preset_version),
            preset_name: input
                .preset
                .as_ref()
                .map(|preset| preset.preset_name.clone()),
            preset_config_snapshot: input
                .preset
                .as_ref()
                .map(|preset| preset.preset_config_snapshot.clone()),
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
fn resolve_item_path(
    dataset_root: &Path,
    relative_path: &str,
) -> Result<String, TrainingPlanError> {
    if !is_safe_relative_path(relative_path) {
        return Err(TrainingPlanError::InvalidConfig(
            "Dataset item path must be relative to the dataset root.".to_owned(),
        ));
    }
    let mut path = dataset_root.to_path_buf();
    for component in Path::new(relative_path).components() {
        path.push(component);
    }
    Ok(path.display().to_string())
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
    validate_lr_scheduler(config)?;
    Ok(())
}

/// Validates the learning-rate scheduler knobs in `advanced`. The scheduler name
/// must be one of [`SUPPORTED_LR_SCHEDULERS`] (so a non-constant choice is never
/// silently ignored), and an optional `lrWarmupSteps` ramp must be a non-negative
/// integer shorter than the run. Absent keys keep the constant-LR default.
fn validate_lr_scheduler(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    if let Some(value) = config.advanced.get("lrScheduler") {
        let name = value.as_str().ok_or_else(|| {
            TrainingPlanError::InvalidConfig("lrScheduler must be a string.".to_owned())
        })?;
        let normalized = name.trim().to_lowercase();
        if !SUPPORTED_LR_SCHEDULERS.contains(&normalized.as_str()) {
            return Err(TrainingPlanError::InvalidConfig(format!(
                "Unsupported lrScheduler '{name}'. Supported schedulers: {}.",
                SUPPORTED_LR_SCHEDULERS.join(", ")
            )));
        }
    }
    if let Some(value) = config.advanced.get("lrWarmupSteps") {
        let warmup = value
            .as_u64()
            .or_else(|| match value.as_f64() {
                Some(number) if number.is_finite() && number >= 0.0 && number.fract() == 0.0 => {
                    Some(number as u64)
                }
                _ => None,
            })
            .ok_or_else(|| {
                TrainingPlanError::InvalidConfig(
                    "lrWarmupSteps must be a non-negative integer.".to_owned(),
                )
            })?;
        if warmup >= u64::from(config.steps) {
            return Err(TrainingPlanError::InvalidConfig(format!(
                "lrWarmupSteps ({warmup}) must be less than steps ({}).",
                config.steps
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_item_path_rejects_paths_outside_dataset_root() {
        let root = Path::new("/dataset");

        assert!(matches!(
            resolve_item_path(root, "../outside.png"),
            Err(TrainingPlanError::InvalidConfig(_))
        ));
        assert!(matches!(
            resolve_item_path(root, "/absolute.png"),
            Err(TrainingPlanError::InvalidConfig(_))
        ));
        // A safe relative path resolves under the dataset root using the host separator (the plan is
        // built + consumed on the same machine). Build the expectation the same way so the assertion
        // holds on Windows too — a hardcoded `/dataset/images/photo.png` only matches on `/`-sep hosts.
        let mut expected = root.to_path_buf();
        for component in Path::new("images/photo.png").components() {
            expected.push(component);
        }
        assert_eq!(
            resolve_item_path(root, "images/photo.png").expect("safe path"),
            expected.display().to_string()
        );
    }
}
