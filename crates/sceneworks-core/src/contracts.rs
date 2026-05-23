use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// Unknown JSON object keys captured during deserialization.
///
/// These fields are reserved for forward-compatible contract data. Do not add
/// keys here that duplicate declared struct fields; write the typed field
/// instead so serialized JSON stays unambiguous.
pub type ExtraFields = BTreeMap<String, Value>;
pub type JsonObject = serde_json::Map<String, Value>;
/// JSON number token preserved exactly for fixture parity.
///
/// Using `serde_json::Number` keeps integer JSON values as integers on
/// round-trip, but it means owning structs derive `PartialEq` rather than `Eq`.
pub type ContractNumber = serde_json::Number;

#[derive(Debug, Clone, Default, PartialEq)]
pub enum MaybePresent<T> {
    #[default]
    Missing,
    Present(Option<T>),
}

impl<T> MaybePresent<T> {
    pub const fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

impl<T> Serialize for MaybePresent<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Missing => serializer.serialize_none(),
            Self::Present(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for MaybePresent<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self::Present)
    }
}

macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $($variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[non_exhaustive]
        #[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub enum $name {
            $($variant,)+
            Unknown(String),
        }

        impl $name {
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $value,)+
                    Self::Unknown(value) => value.as_str(),
                }
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Ok(match value.as_str() {
                    $($value => Self::$variant,)+
                    _ => Self::Unknown(value),
                })
            }
        }
    };
}

// Re-exported within the crate so sibling contract modules (for example
// `training`) can declare their own forward-compatible string enums with the
// same `Unknown(String)` fallback semantics.
pub(crate) use string_enum;

string_enum! {
    pub enum ContractMode {
        TextToImage => "text_to_image",
        EditImage => "edit_image",
        CharacterImage => "character_image",
        StyleVariations => "style_variations",
        ImageToVideo => "image_to_video",
        TextToVideo => "text_to_video",
        FirstLastFrame => "first_last_frame",
        ExtendClip => "extend_clip",
        VideoBridge => "video_bridge",
        ReplacePerson => "replace_person",
        PersonDetect => "person_detect",
        PersonTrack => "person_track",
        FrameExtract => "frame_extract",
        TimelineExport => "timeline_export",
        ModelDownload => "model_download",
        ModelImport => "model_import",
        LoraImport => "lora_import",
    }
}

string_enum! {
    pub enum RecipeAdapter {
        ProceduralPreview => "procedural_preview",
        ZImageDiffusers => "z_image_diffusers",
        QwenImage => "qwen_image",
        LensTurbo => "lens_turbo",
        SenseNovaU1 => "sensenova_u1",
        ProceduralVideo => "procedural_video",
        ProceduralPersonTracking => "procedural_person_tracking",
        FfmpegFrameExtract => "ffmpeg-frame-extract",
        TimelineExporter => "timeline-exporter",
    }
}

string_enum! {
    pub enum PersonTrackMaskState {
        // "deferred" is the legacy procedural-preview state. Real tracks report
        // segmentation mask states: active (masks for all detected frames),
        // generated (partial), degraded (box fallback / no segmenter), or missing.
        Deferred => "deferred",
        Active => "active",
        Generated => "generated",
        Degraded => "degraded",
        Missing => "missing",
    }
}

string_enum! {
    pub enum PersonTrackCorrectionState {
        // A freshly tracked sidecar advertises that it accepts box corrections.
        ReadyForBoxCorrections => "ready_for_box_corrections",
        // Set once the correction UI has persisted at least one box adjustment or
        // frame rejection into the sidecar's `corrections` array (sc-1485).
        BoxCorrectionsApplied => "box_corrections_applied",
    }
}

string_enum! {
    pub enum CharacterReferenceRole {
        Reference => "reference",
        Hero => "hero",
        TestOutput => "test-output",
    }
}

string_enum! {
    pub enum CharacterLoraCategory {
        Character => "character",
    }
}

string_enum! {
    pub enum JobType {
        Placeholder => "placeholder",
        ImageGenerate => "image_generate",
        ImageEdit => "image_edit",
        ImageVqa => "image_vqa",
        VideoGenerate => "video_generate",
        VideoExtend => "video_extend",
        VideoBridge => "video_bridge",
        PersonDetect => "person_detect",
        PersonTrack => "person_track",
        PersonReplace => "person_replace",
        FrameExtract => "frame_extract",
        TimelineExport => "timeline_export",
        ModelDownload => "model_download",
        ModelImport => "model_import",
        ModelConvert => "model_convert",
        LoraImport => "lora_import",
        LoraTrain => "lora_train",
        TrainingCaption => "training_caption",
    }
}

string_enum! {
    pub enum JobStatus {
        Queued => "queued",
        Preparing => "preparing",
        Downloading => "downloading",
        LoadingModel => "loading_model",
        Running => "running",
        Saving => "saving",
        Completed => "completed",
        Failed => "failed",
        Canceled => "canceled",
        Interrupted => "interrupted",
    }
}

string_enum! {
    pub enum ProgressStage {
        Queued => "queued",
        Preparing => "preparing",
        Downloading => "downloading",
        Importing => "importing",
        LoadingModel => "loading_model",
        Estimating => "estimating",
        Generating => "generating",
        Running => "running",
        // LoRA training stages (status stays `running`); see
        // apps/worker/scene_worker/training_adapters.py.
        CachingLatents => "caching_latents",
        Training => "training",
        Checkpointing => "checkpointing",
        Rendering => "rendering",
        Extracting => "extracting",
        Tracking => "tracking",
        Muxing => "muxing",
        Saving => "saving",
        Completed => "completed",
        Failed => "failed",
        Canceled => "canceled",
        Interrupted => "interrupted",
    }
}

string_enum! {
    pub enum WorkerStatus {
        Idle => "idle",
        Busy => "busy",
        Offline => "offline",
    }
}

string_enum! {
    pub enum WorkerCapability {
        Cpu => "cpu",
        Gpu => "gpu",
        Placeholder => "placeholder",
        ImageGenerate => "image_generate",
        ImageEdit => "image_edit",
        ImageVqa => "image_vqa",
        VideoGenerate => "video_generate",
        VideoExtend => "video_extend",
        VideoBridge => "video_bridge",
        PersonDetect => "person_detect",
        PersonTrack => "person_track",
        PersonReplace => "person_replace",
        // Procedural detection/tracking previews served by the Rust utility worker.
        // Real, model-backed PersonDetect/PersonTrack jobs are served by the Python
        // GPU worker (YOLO/ByteTrack/SAM2); these preview capabilities keep the CPU
        // procedural path claimable only for explicit `preview: true` jobs, so a
        // real job never routes to the placeholder. Segment availability is its own
        // capability for replacement readiness. See jobs_store::worker_supports_job
        // and apps/worker/scene_worker/runtime.py.
        PersonDetectPreview => "person_detect_preview",
        PersonTrackPreview => "person_track_preview",
        PersonSegment => "person_segment",
        FrameExtract => "frame_extract",
        TimelineExport => "timeline_export",
        ModelDownload => "model_download",
        ModelImport => "model_import",
        ModelConvert => "model_convert",
        LoraImport => "lora_import",
        LoraTrain => "lora_train",
        TrainingCaption => "training_caption",
        // Real (non-dry-run) LoRA training execution. Advertised separately from
        // `LoraTrain` (dry-run plan validation, which needs no inference backend)
        // so a real run only routes to a worker that can actually train. See
        // jobs_store::worker_supports_job and apps/worker/scene_worker/runtime.py.
        LoraTrainExecute => "lora_train_execute",
    }
}

string_enum! {
    pub enum SseEvent {
        Ready => "ready",
        Heartbeat => "heartbeat",
        JobUpdated => "job.updated",
        QueueUpdated => "queue.updated",
        WorkerUpdated => "worker.updated",
    }
}

string_enum! {
    /// Asset resource types allowed by the JSON schema.
    ///
    /// Current writers emit `image`, `video`, and frame extraction sidecars;
    /// `upload` and `render` are reserved by the schema and project folders.
    pub enum AssetType {
        Image => "image",
        Video => "video",
        Upload => "upload",
        Frame => "frame",
        Render => "render",
    }
}

string_enum! {
    pub enum CharacterType {
        Person => "person",
        Creature => "creature",
        Object => "object",
    }
}

string_enum! {
    pub enum TimelineAspectRatio {
        Landscape => "16:9",
        Portrait => "9:16",
        Square => "1:1",
    }
}

string_enum! {
    pub enum TimelineTrackKind {
        Video => "video",
        Overlay => "overlay",
        Audio => "audio",
    }
}

string_enum! {
    pub enum TimelineItemType {
        Video => "video",
        Image => "image",
        Audio => "audio",
    }
}

string_enum! {
    pub enum TimelineFitMode {
        Fit => "fit",
        Fill => "fill",
        Stretch => "stretch",
    }
}

string_enum! {
    pub enum TimelineVersionSource {
        Original => "original",
        Replacement => "replacement",
        Extension => "extension",
        Bridge => "bridge",
        Restore => "restore",
        Manual => "manual",
    }
}

string_enum! {
    pub enum TimelineTransitionType {
        Cut => "cut",
        Crossfade => "crossfade",
        FadeFromBlack => "fade_from_black",
        FadeToBlack => "fade_to_black",
    }
}

string_enum! {
    pub enum CharacterLoraScope {
        Project => "project",
        Global => "global",
        External => "external",
    }
}

string_enum! {
    pub enum RecipePresetScope {
        Builtin => "builtin",
        Global => "global",
        Project => "project",
    }
}

string_enum! {
    pub enum RecipePresetWorkflow {
        TextToImage => "text_to_image",
        ImageEdit => "edit_image",
        ImageToVideo => "image_to_video",
        TextToVideo => "text_to_video",
        FirstLastFrame => "first_last_frame",
    }
}

string_enum! {
    pub enum ModelKind {
        Image => "image",
        Video => "video",
        Utility => "utility",
    }
}

string_enum! {
    pub enum ModelCapability {
        TextToImage => "text_to_image",
        ImageEdit => "image_edit",
        CharacterImage => "character_image",
        StyleVariations => "style_variations",
        TextToVideo => "text_to_video",
        ImageToVideo => "image_to_video",
        VideoExtend => "video_extend",
        VideoBridge => "video_bridge",
        PersonReplace => "person_replace",
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProtocolFixture {
    pub schema_version: u32,
    pub job_types: Vec<JobType>,
    pub statuses: Vec<JobStatus>,
    pub active_statuses: Vec<JobStatus>,
    pub terminal_statuses: Vec<JobStatus>,
    pub non_gpu_job_types: Vec<JobType>,
    pub worker_statuses: Vec<WorkerStatus>,
    pub progress_stages: Vec<ProgressStage>,
    pub sse_events: Vec<SseEvent>,
    pub worker_capability_profiles: BTreeMap<String, Vec<WorkerCapability>>,
    pub requests: JobProtocolRequests,
    pub job_snapshot: JobSnapshot,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProtocolRequests {
    pub create_job: JobCreateRequest,
    pub register_worker: WorkerRegisterRequest,
    pub heartbeat: WorkerHeartbeatRequest,
    pub progress: ProgressRequest,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobCreateRequest {
    #[serde(rename = "type")]
    pub job_type: JobType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,
    pub payload: JsonObject,
    pub requested_gpu: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerRegisterRequest {
    pub worker_id: String,
    pub gpu_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_name: Option<String>,
    pub capabilities: Vec<WorkerCapability>,
    pub loaded_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utilization: Option<WorkerUtilizationSnapshot>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerHeartbeatRequest {
    pub status: WorkerStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_job_id: Option<String>,
    pub loaded_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utilization: Option<WorkerUtilizationSnapshot>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressRequest {
    pub status: JobStatus,
    pub stage: ProgressStage,
    pub progress: ContractNumber,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<ContractNumber>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSnapshot {
    pub id: String,
    #[serde(rename = "type")]
    pub job_type: JobType,
    pub status: JobStatus,
    pub project_id: Option<String>,
    pub project_name: Option<String>,
    pub payload: JsonObject,
    pub result: JsonObject,
    pub requested_gpu: String,
    pub assigned_gpu: Option<String>,
    pub worker_id: Option<String>,
    pub progress: ContractNumber,
    pub stage: ProgressStage,
    pub message: String,
    pub error: Option<String>,
    pub eta_seconds: Option<ContractNumber>,
    pub elapsed_seconds: Option<ContractNumber>,
    pub attempts: u32,
    pub source_job_id: Option<String>,
    pub duplicate_of_job_id: Option<String>,
    pub cancel_requested: bool,
    pub created_at: String,
    pub updated_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub canceled_at: Option<String>,
    pub last_heartbeat_at: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceSidecarsFixture {
    pub schema_version: u32,
    pub project_folders: Vec<String>,
    pub sidecar_patterns: SidecarPatterns,
    pub fixtures: Vec<SidecarFixtureDescriptor>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueSummary {
    pub counts: BTreeMap<JobStatus, u32>,
    pub active_jobs: Vec<JobSnapshot>,
    pub workers: Vec<WorkerSnapshot>,
    pub max_job_attempts: u32,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerSnapshot {
    pub id: String,
    pub gpu_id: String,
    pub gpu_name: Option<String>,
    pub status: WorkerStatus,
    pub current_job_id: Option<String>,
    pub capabilities: Vec<WorkerCapability>,
    pub loaded_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utilization: Option<WorkerUtilizationSnapshot>,
    pub registered_at: String,
    pub last_seen_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerUtilizationSnapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_total_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_used_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_free_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_load_percent: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaimRequest {
    pub worker_id: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaimResponse {
    pub job: Option<JobSnapshot>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DuplicateJobRequest {
    pub payload_changes: JsonObject,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_gpu: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarPatterns {
    pub project: String,
    pub asset: String,
    pub character: String,
    pub timeline: String,
    pub person_track: String,
    pub generation_set: String,
    pub recipe: String,
    pub model_install_marker: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarFixtureDescriptor {
    pub name: String,
    pub path: String,
    pub required_top_level_keys: Vec<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub schema_version: u32,
    pub app_version: String,
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub folders: BTreeMap<String, String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub generation_set_id: Option<String>,
    #[serde(rename = "type")]
    pub asset_type: AssetType,
    pub display_name: String,
    pub created_at: String,
    pub file: AssetFile,
    pub status: AssetStatus,
    pub recipe: Recipe,
    pub lineage: AssetLineage,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetFile {
    pub path: String,
    pub mime_type: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration: Option<ContractNumber>,
    pub fps: Option<ContractNumber>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetStatus {
    pub favorite: bool,
    pub rating: u8,
    pub rejected: bool,
    pub trashed: bool,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetLineage {
    pub parents: Vec<String>,
    pub source_asset_id: Option<String>,
    pub source_timestamp: Option<ContractNumber>,
    pub job_id: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSet {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub job_id: Option<String>,
    pub mode: ContractMode,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub count: u32,
    pub created_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Recipe {
    pub mode: ContractMode,
    pub model: String,
    pub adapter: RecipeAdapter,
    pub prompt: String,
    pub negative_prompt: String,
    pub seed: i64,
    pub loras: Vec<JsonObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style_preset: Option<String>,
    pub normalized_settings: JsonObject,
    pub raw_adapter_settings: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Character {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub character_type: CharacterType,
    pub description: String,
    pub created_at: String,
    pub updated_at: String,
    pub status: CharacterStatus,
    pub references: Vec<CharacterReference>,
    pub looks: Vec<CharacterLook>,
    pub loras: Vec<CharacterLora>,
    pub trained_loras: Vec<JsonObject>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterStatus {
    pub archived: bool,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterReference {
    pub asset_id: String,
    pub approved: bool,
    pub role: CharacterReferenceRole,
    pub notes: String,
    pub added_at: String,
    pub approved_at: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterLook {
    pub id: String,
    pub name: String,
    pub description: String,
    pub approved_reference_ids: Vec<String>,
    pub recipe_settings: JsonObject,
    pub created_at: String,
    pub updated_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterLora {
    pub id: String,
    pub lora_id: Option<String>,
    pub name: String,
    pub source_path: Option<String>,
    pub project_path: Option<String>,
    pub copied_into_project: bool,
    pub category: CharacterLoraCategory,
    pub scope: CharacterLoraScope,
    pub trigger_words: Vec<String>,
    pub default_weight: ContractNumber,
    pub compatibility: JsonObject,
    pub created_at: String,
    pub updated_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Timeline {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub aspect_ratio: TimelineAspectRatio,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration: ContractNumber,
    pub tracks: Vec<TimelineTrack>,
    pub transitions: Vec<TimelineTransition>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineTrack {
    pub id: String,
    pub name: String,
    pub kind: TimelineTrackKind,
    pub locked: bool,
    pub muted: bool,
    pub items: Vec<TimelineItem>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineItem {
    pub id: String,
    pub track_id: String,
    pub asset_id: String,
    #[serde(rename = "type")]
    pub item_type: TimelineItemType,
    pub display_name: String,
    pub source_in: ContractNumber,
    pub source_out: ContractNumber,
    pub timeline_start: ContractNumber,
    pub timeline_end: ContractNumber,
    pub speed: ContractNumber,
    pub fit: TimelineFitMode,
    pub volume: ContractNumber,
    pub version_asset_ids: Vec<String>,
    pub current_version_asset_id: Option<String>,
    pub version_history: Vec<TimelineVersion>,
    pub transition_in: Option<TimelineTransition>,
    pub transition_out: Option<TimelineTransition>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineVersion {
    pub asset_id: String,
    pub created_at: Option<String>,
    pub source: TimelineVersionSource,
    pub job_id: Option<String>,
    pub note: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineTransition {
    pub id: String,
    #[serde(rename = "type")]
    pub transition_type: TimelineTransitionType,
    pub from_item_id: Option<String>,
    pub to_item_id: Option<String>,
    pub duration: ContractNumber,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonTrack {
    pub schema_version: u32,
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub created_at: String,
    pub source_asset_id: String,
    pub source_display_name: Option<String>,
    pub representative_frame_asset_id: String,
    pub selected_detection: PersonDetection,
    pub frames: Vec<PersonTrackFrame>,
    pub corrections: Vec<Value>,
    pub status: PersonTrackStatus,
    pub recipe: Recipe,
    pub lineage: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonDetection {
    pub id: String,
    #[serde(rename = "box")]
    pub box_: NormalizedBox,
    pub confidence: ContractNumber,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonTrackFrame {
    pub timestamp: ContractNumber,
    #[serde(rename = "box")]
    pub box_: NormalizedBox,
    pub confidence: ContractNumber,
    #[serde(default, skip_serializing_if = "MaybePresent::is_missing")]
    pub mask: MaybePresent<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedBox {
    pub x: ContractNumber,
    pub y: ContractNumber,
    pub width: ContractNumber,
    pub height: ContractNumber,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonTrackStatus {
    pub sample_rate_fps: ContractNumber,
    pub mask_state: PersonTrackMaskState,
    pub average_confidence: ContractNumber,
    pub correction_state: PersonTrackCorrectionState,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    pub schema_version: u32,
    pub models: Vec<ModelManifestEntry>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifestEntry {
    pub id: String,
    pub name: String,
    pub family: String,
    #[serde(rename = "type")]
    pub model_type: ModelKind,
    pub adapter: String,
    pub capabilities: Vec<ModelCapability>,
    pub downloads: Vec<ModelDownload>,
    pub paths: BTreeMap<String, String>,
    pub defaults: JsonObject,
    pub limits: JsonObject,
    pub lora_compatibility: JsonObject,
    pub ui: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownload {
    pub provider: String,
    pub repo: String,
    pub files: Vec<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoraManifest {
    pub schema_version: u32,
    pub loras: Vec<LoraManifestEntry>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoraManifestEntry {
    pub id: String,
    pub name: String,
    pub family: String,
    pub trigger_words: Vec<String>,
    pub compatibility: JsonObject,
    pub source: JsonObject,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipePresetManifest {
    pub schema_version: u32,
    pub presets: Vec<RecipePresetManifestEntry>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipePresetManifestEntry {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<RecipePresetScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<i64>,
    pub workflow: RecipePresetWorkflow,
    pub model: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modes: Vec<ContractMode>,
    #[serde(default, skip_serializing_if = "RecipePresetDefaults::is_empty")]
    pub defaults: RecipePresetDefaults,
    #[serde(default, skip_serializing_if = "RecipePresetPrompt::is_empty")]
    pub prompt: RecipePresetPrompt,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<RecipePresetLora>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipePresetDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

impl RecipePresetDefaults {
    pub fn is_empty(&self) -> bool {
        self.count.is_none()
            && self.resolution.is_none()
            && self.negative_prompt.is_none()
            && self.extra.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipePresetPrompt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

impl RecipePresetPrompt {
    pub fn is_empty(&self) -> bool {
        self.prefix.is_none() && self.suffix.is_none() && self.extra.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecipePresetLora {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<ContractNumber>,
    #[serde(flatten)]
    pub extra: ExtraFields,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInstallMarker {
    pub repo: String,
    pub model_id: String,
    pub model_name: String,
    pub job_id: String,
    pub completed_at: String,
    #[serde(flatten)]
    pub extra: ExtraFields,
}
