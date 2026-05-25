use super::*;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JobsQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AssetsQuery {
    pub(crate) include_rejected: Option<bool>,
    pub(crate) include_trashed: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharactersQuery {
    pub(crate) include_archived: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LorasQuery {
    pub(crate) model_family: Option<String>,
    pub(crate) project_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CatalogDeleteQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RecipePresetsQuery {
    pub(crate) project_id: Option<String>,
    pub(crate) include_archived: Option<bool>,
    pub(crate) model: Option<String>,
    pub(crate) workflow: Option<String>,
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EventsQuery {
    pub(crate) ticket: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HealthResponse {
    pub(crate) status: &'static str,
    pub(crate) service: &'static str,
    pub(crate) runtime: String,
    pub(crate) version: String,
    pub(crate) auth_required: bool,
    // Absolute host paths are withheld from the public health endpoint when a token is
    // configured, so a LAN client can't map the host filesystem despite auth being on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) directories: Option<DirectoriesResponse>,
    pub(crate) interrupted_jobs_on_startup: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DirectoriesResponse {
    pub(crate) data: String,
    pub(crate) config: String,
    pub(crate) projects: String,
    pub(crate) jobs_db: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AccessResponse {
    pub(crate) auth_required: bool,
    pub(crate) token_header: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct VerifyResponse {
    pub(crate) ok: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ProjectCreateRequest {
    pub(crate) name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterCreateRequest {
    pub(crate) name: String,
    #[serde(default = "default_character_type", rename = "type")]
    pub(crate) character_type: String,
    #[serde(default)]
    pub(crate) description: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterUpdateRequest {
    pub(crate) name: Option<String>,
    #[serde(default, rename = "type")]
    pub(crate) character_type: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) archived: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterReferenceRequest {
    pub(crate) asset_id: String,
    #[serde(default)]
    pub(crate) approved: bool,
    #[serde(default = "default_reference_role")]
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) notes: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterReferenceUpdateRequest {
    pub(crate) approved: Option<bool>,
    pub(crate) role: Option<String>,
    pub(crate) notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLookRequest {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) description: String,
    #[serde(default)]
    pub(crate) approved_reference_ids: Vec<String>,
    #[serde(default)]
    pub(crate) recipe_settings: JsonObject,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLookUpdateRequest {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) approved_reference_ids: Option<Vec<String>>,
    pub(crate) recipe_settings: Option<JsonObject>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLoraRequest {
    #[serde(default)]
    pub(crate) lora_id: Option<String>,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) source_path: Option<String>,
    #[serde(default)]
    pub(crate) trigger_words: Vec<String>,
    #[serde(default = "default_character_lora_weight")]
    pub(crate) default_weight: f64,
    #[serde(default)]
    pub(crate) compatibility: JsonObject,
    #[serde(default = "default_project_lora_scope")]
    pub(crate) scope: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterLoraUpdateRequest {
    pub(crate) name: Option<String>,
    pub(crate) trigger_words: Option<Vec<String>>,
    pub(crate) default_weight: Option<f64>,
    pub(crate) compatibility: Option<JsonObject>,
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharacterTestRequest {
    pub(crate) prompt: String,
    #[serde(default = "default_image_model")]
    pub(crate) model: String,
    #[serde(default = "default_image_count")]
    pub(crate) count: u32,
    #[serde(default = "default_image_size")]
    pub(crate) width: u32,
    #[serde(default = "default_image_size")]
    pub(crate) height: u32,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) look_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimelineCreateRequest {
    #[serde(default = "default_timeline_name")]
    pub(crate) name: String,
    #[serde(default = "default_aspect_ratio")]
    pub(crate) aspect_ratio: String,
    #[serde(default = "default_timeline_fps")]
    pub(crate) fps: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimelineSaveRequest {
    pub(crate) timeline: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimelineExportRequest {
    #[serde(default = "default_export_resolution")]
    pub(crate) resolution: u32,
    #[serde(default = "default_timeline_fps")]
    pub(crate) fps: u32,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FrameExtractRequest {
    pub(crate) playhead_seconds: f64,
    #[serde(default = "default_frame_intended_use")]
    pub(crate) intended_use: String,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersonDetectionJobRequest {
    pub(crate) source_asset_id: String,
    #[serde(default)]
    pub(crate) source_timestamp: Option<f64>,
    /// Opt into the Rust utility worker's procedural preview instead of real,
    /// model-backed detection on the Python GPU worker. Defaults to real.
    #[serde(default)]
    pub(crate) preview: bool,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersonTrackJobRequest {
    pub(crate) source_asset_id: String,
    pub(crate) representative_frame_asset_id: String,
    pub(crate) detection: JsonObject,
    #[serde(default = "default_track_name")]
    pub(crate) track_name: String,
    /// Opt into the Rust utility worker's procedural preview instead of real,
    /// model-backed tracking on the Python GPU worker. Defaults to real.
    #[serde(default)]
    pub(crate) preview: bool,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersonTrackCorrectionsRequest {
    /// The UI's full correction set for the track. Each entry targets a frame by
    /// index and adjusts its box and/or rejects the frame; the store validates
    /// ranges and stamps author/createdAt/source. Kept as raw values so the
    /// schema-flexible `corrections` array can evolve without an API change.
    #[serde(default)]
    pub(crate) corrections: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrainingCaptionJobRequest {
    #[serde(default = "default_training_captioner")]
    pub(crate) captioner: String,
    #[serde(default = "default_training_caption_model")]
    pub(crate) model_name_or_path: String,
    #[serde(default)]
    pub(crate) recaption: bool,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) options: TrainingCaptionOptions,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrainingCaptionOptions {
    #[serde(default = "default_training_caption_type")]
    pub(crate) caption_type: String,
    #[serde(default = "default_training_caption_length")]
    pub(crate) caption_length: String,
    #[serde(default)]
    pub(crate) extra_options: Vec<String>,
    #[serde(default)]
    pub(crate) name_input: String,
    #[serde(default = "default_training_caption_temperature")]
    pub(crate) temperature: f64,
    #[serde(default = "default_training_caption_top_p")]
    pub(crate) top_p: f64,
    #[serde(default = "default_training_caption_max_new_tokens")]
    pub(crate) max_new_tokens: u32,
    #[serde(default)]
    pub(crate) caption_prompt: String,
    #[serde(default)]
    pub(crate) low_vram: bool,
}

impl Default for TrainingCaptionOptions {
    fn default() -> Self {
        Self {
            caption_type: default_training_caption_type(),
            caption_length: default_training_caption_length(),
            extra_options: Vec::new(),
            name_input: String::new(),
            temperature: default_training_caption_temperature(),
            top_p: default_training_caption_top_p(),
            max_new_tokens: default_training_caption_max_new_tokens(),
            caption_prompt: String::new(),
            low_vram: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ImageJobRequest {
    pub(crate) project_id: String,
    #[serde(default)]
    pub(crate) project_name: Option<String>,
    #[serde(default = "default_image_mode")]
    pub(crate) mode: String,
    pub(crate) prompt: String,
    #[serde(default)]
    pub(crate) negative_prompt: String,
    #[serde(default = "default_image_model")]
    pub(crate) model: String,
    #[serde(default = "default_image_count")]
    pub(crate) count: u32,
    #[serde(default)]
    pub(crate) seed: Option<i64>,
    #[serde(default = "default_image_size")]
    pub(crate) width: u32,
    #[serde(default = "default_image_size")]
    pub(crate) height: u32,
    #[serde(default = "default_style_preset")]
    pub(crate) style_preset: String,
    #[serde(default)]
    pub(crate) recipe_preset_id: Option<String>,
    #[serde(default)]
    pub(crate) loras: Vec<Value>,
    #[serde(default)]
    pub(crate) character_id: Option<String>,
    #[serde(default)]
    pub(crate) character_look_id: Option<String>,
    #[serde(default)]
    pub(crate) source_asset_id: Option<String>,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) advanced: JsonObject,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VqaJobRequest {
    pub(crate) project_id: String,
    #[serde(default)]
    pub(crate) project_name: Option<String>,
    pub(crate) source_asset_id: String,
    pub(crate) question: String,
    #[serde(default = "default_vqa_model")]
    pub(crate) model: String,
    #[serde(default = "default_vqa_max_new_tokens")]
    pub(crate) max_new_tokens: u32,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) advanced: JsonObject,
}

pub(crate) fn default_vqa_model() -> String {
    "sensenova_u1_8b".to_owned()
}

pub(crate) fn default_vqa_max_new_tokens() -> u32 {
    256
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InterleaveJobRequest {
    pub(crate) project_id: String,
    #[serde(default)]
    pub(crate) project_name: Option<String>,
    pub(crate) prompt: String,
    // Optional input images for grounded (it2i) interleaved generation.
    #[serde(default)]
    pub(crate) source_asset_ids: Vec<String>,
    #[serde(default = "default_interleave_model")]
    pub(crate) model: String,
    #[serde(default = "default_interleave_max_images")]
    pub(crate) max_images: u32,
    #[serde(default = "default_image_size")]
    pub(crate) width: u32,
    #[serde(default = "default_image_size")]
    pub(crate) height: u32,
    #[serde(default)]
    pub(crate) seed: Option<i64>,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) advanced: JsonObject,
}

pub(crate) fn default_interleave_model() -> String {
    "sensenova_u1_8b".to_owned()
}

pub(crate) fn default_interleave_max_images() -> u32 {
    6
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoJobRequest {
    pub(crate) project_id: String,
    #[serde(default)]
    pub(crate) project_name: Option<String>,
    #[serde(default = "default_video_mode")]
    pub(crate) mode: String,
    pub(crate) prompt: String,
    #[serde(default)]
    pub(crate) negative_prompt: String,
    #[serde(default = "default_video_model")]
    pub(crate) model: String,
    #[serde(default = "default_video_duration")]
    pub(crate) duration: ContractNumber,
    #[serde(default = "default_video_fps")]
    pub(crate) fps: u32,
    #[serde(default = "default_video_width")]
    pub(crate) width: u32,
    #[serde(default = "default_video_height")]
    pub(crate) height: u32,
    #[serde(default = "default_video_quality")]
    pub(crate) quality: String,
    #[serde(default)]
    pub(crate) seed: Option<i64>,
    #[serde(default)]
    pub(crate) recipe_preset_id: Option<String>,
    #[serde(default)]
    pub(crate) loras: Vec<Value>,
    #[serde(default)]
    pub(crate) character_id: Option<String>,
    #[serde(default)]
    pub(crate) character_look_id: Option<String>,
    #[serde(default)]
    pub(crate) person_track_id: Option<String>,
    #[serde(default = "default_replacement_mode")]
    pub(crate) replacement_mode: String,
    #[serde(default)]
    pub(crate) source_asset_id: Option<String>,
    #[serde(default)]
    pub(crate) last_frame_asset_id: Option<String>,
    #[serde(default)]
    pub(crate) source_clip_asset_id: Option<String>,
    #[serde(default)]
    pub(crate) bridge_right_clip_asset_id: Option<String>,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default)]
    pub(crate) advanced: JsonObject,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelDownloadRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelConvertRequest {
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelImportRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, alias = "type", skip_serializing_if = "Option::is_none")]
    pub(crate) model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_path: Option<String>,
    #[serde(default)]
    pub(crate) files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) family: Option<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    pub(crate) uploaded_source_path: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LoraImportRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) lora_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_path: Option<String>,
    #[serde(default)]
    pub(crate) files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) family: Option<String>,
    #[serde(default = "default_lora_scope")]
    pub(crate) scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) project_id: Option<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    pub(crate) uploaded_source_path: bool,
}
