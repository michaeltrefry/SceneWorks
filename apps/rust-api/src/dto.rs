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
pub(crate) struct PromptRefineRequest {
    pub(crate) prompt: String,
    pub(crate) model_id: Option<String>,
    pub(crate) workflow: Option<String>,
    /// The selected model's Markdown prompt guide, sent by the client so the
    /// worker can build a guide-aware system prompt without filesystem access to
    /// the web assets.
    pub(crate) guide: Option<String>,
    /// `"refine"` (default) or `"magic_prompt"` — the latter expands a plain idea
    /// into a structured Ideogram 4 JSON caption (epic 4725, sc-5997).
    pub(crate) task: Option<String>,
    /// Target image aspect ratio as `"W:H"` (magic-prompt only); drives bbox/layout
    /// decisions in the caption. Defaults to `"1:1"` worker-side when absent.
    pub(crate) aspect_ratio: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AssetsQuery {
    pub(crate) include_rejected: Option<bool>,
    pub(crate) include_trashed: Option<bool>,
    /// Scope to one character: assets generated in association with it
    /// (recipe.normalizedSettings.characterId) or referencing it
    /// (metadata.characterReferences[].characterId).
    pub(crate) character_id: Option<String>,
    /// View scope (sc-2024). `library` excludes Character Studio test outputs;
    /// anything else (the default) returns all assets.
    pub(crate) scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CharactersQuery {
    pub(crate) include_archived: Option<bool>,
}

/// Query parameters for the dataset readiness report (sc-6533). The training UI passes what it has
/// selected; all are optional and fall back to per-kind defaults in `readiness_context`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReadinessQuery {
    /// The chosen training target's resolution; drives the bucket the scalars are measured at.
    pub(crate) target_resolution: Option<u32>,
    /// Comma-separated preset/target `recommendedFor` tags (e.g. `"character,style"`).
    pub(crate) recommended_for: Option<String>,
    /// The dataset's character kind (e.g. `"person"`).
    pub(crate) character_type: Option<String>,
    /// Override for the preset's minimum item count.
    pub(crate) min_items: Option<u32>,
}

/// Body for the per-image quality override (sc-6534): the checks the user dismissed for one image.
/// The store strips the non-acknowledgeable ones (`decode`, `count`); an empty list clears the ack.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct QualityAckBody {
    #[serde(default)]
    pub(crate) checks: Vec<sceneworks_core::dataset_quality::QualityCheck>,
    /// Client-side freshness guard. The ack applies only to the exact image bytes the user reviewed.
    #[serde(default)]
    pub(crate) expected_content_hash: Option<String>,
    /// Freshness guard for caption-dependent checks such as `caption_alignment`.
    #[serde(default)]
    pub(crate) expected_caption_hash: Option<String>,
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

/// Host memory for remote-browser model gating (epic 4484 story 9). A remote browser
/// can't call the desktop-only Tauri `get_gpu_info`, so it reads the host's memory
/// from here — derived from the registered GPU worker's reported utilization. Carries
/// only aggregate memory totals + the platform string (no paths/secrets), and is
/// auth-protected (not a public path).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HostCapabilitiesResponse {
    /// API host OS (`macos`/`windows`/`linux`) so the client gates with the right
    /// memory concept (unified vs VRAM).
    pub(crate) platform: &'static str,
    /// Unified system memory (GB) — the macOS MLX worker reports `sysctl hw.memsize`.
    /// `None` when no MLX worker is registered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) unified_memory_gb: Option<f64>,
    /// Largest discrete GPU VRAM (GB) across registered GPU workers (Windows candle /
    /// CUDA). `None` when no such worker is registered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) gpu_memory_gb: Option<f64>,
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
    /// Restrict the job to these dataset item ids (sc-2025 per-image Re-Caption).
    /// When present, only those items are captioned and they are always recaptioned;
    /// when absent, the dataset-wide `recaption`/missing-caption rule applies.
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// Request to run the Dataset Doctor CLIP-embedding analysis over a training dataset (sc-6535).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetAnalysisJobRequest {
    /// The image-embedder provider id to run. Only `clip_vit_l14` is supported today (validated
    /// server-side); the field exists so a future EVA-CLIP/SigLIP swap is a payload change.
    #[serde(default = "default_dataset_analysis_embedder")]
    pub(crate) embedder: String,
    /// Optional weights snapshot path/id for the embedder; resolved worker-side when absent.
    #[serde(default)]
    pub(crate) model_name_or_path: Option<String>,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// Restrict the analysis to these dataset item ids; when absent, every item is embedded.
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// The analysis worker POSTs its computed CLIP embeddings here to persist the sidecar (sc-6535) —
/// the embedding-side analog of the caption job's `/caption-sidecars` write.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetEmbeddingsBody {
    /// The embedding space (e.g. `clip-vit-l14`).
    pub(crate) space: String,
    pub(crate) items: Vec<DatasetEmbeddingRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetEmbeddingRecord {
    /// The item's content hash (the sidecar key — survives dataset edits).
    pub(crate) content_hash: String,
    /// The raw (un-normalized) embedding vector.
    pub(crate) embedding: Vec<f32>,
    /// SHA-256 of the caption text embedded by the worker. Required when `textEmbedding` is present.
    #[serde(default)]
    pub(crate) caption_hash: Option<String>,
    /// The raw (un-normalized) CLIP text embedding for the caption.
    #[serde(default)]
    pub(crate) text_embedding: Option<Vec<f32>>,
}

/// Body for the synchronous one-tap image fixes (sc-6539 smart-crop / EXIF-strip). `itemIds` scopes
/// the fix; when absent it applies to every item in the dataset (EXIF-strip-all).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetImageFixBody {
    #[serde(default)]
    pub(crate) item_ids: Option<Vec<String>>,
}

/// Request to upscale flagged low-resolution items in a training dataset (sc-6539 one-tap fix).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetUpscaleJobRequest {
    /// Integer upscale factor (Real-ESRGAN x2 / x4). Defaults to x2.
    #[serde(default = "default_dataset_upscale_factor")]
    pub(crate) factor: u8,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    /// The dataset item ids to upscale — the readiness report's low-resolution-flagged items.
    pub(crate) item_ids: Vec<String>,
}

pub(crate) fn default_dataset_upscale_factor() -> u8 {
    2
}

/// The upscale worker POSTs its results here to re-point each item at the upscaled child asset
/// (sc-6539) — the dataset-mutation analog of the caption job's `/caption-sidecars` write.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetRepointBody {
    pub(crate) items: Vec<DatasetRepointRecord>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DatasetRepointRecord {
    pub(crate) item_id: String,
    /// The upscaled child asset (lineage back to the original).
    #[serde(default)]
    pub(crate) asset_id: Option<String>,
    /// Project-relative path of the upscaled bytes the worker wrote.
    pub(crate) source_path: String,
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
    // Reference image for IP-Adapter (style/identity conditioning) — distinct from
    // source_asset_id (the img2img edit base). Drives the Character Studio
    // "many images from one reference" flow. ip_adapter_scale rides in `advanced`.
    #[serde(default)]
    pub(crate) reference_asset_id: Option<String>,
    // Multi-image reference set for a FLUX.2 multi-reference edit (sc-6211 / sc-6107): every id is a
    // reference image that jointly conditions the edit (DiT token concat → Conditioning::MultiReference),
    // distinct from the single `reference_asset_id`. Threaded to the worker payload as-is; the worker's
    // `flux2_edit_reference_ids` consumes a non-empty list (capped at MAX_EDIT_REFERENCES) and prefers it
    // over `source_asset_id`. Must be a typed field: without it serde drops the unknown top-level key on
    // deserialize and `to_json_object` never forwards it, so the references never reach the worker
    // (sc-6358). Mirrors VideoJobRequest.reference_asset_ids.
    #[serde(default)]
    pub(crate) reference_asset_ids: Vec<String>,
    // Optional inpaint mask asset (sc-2476): white = edit region. Honored only by
    // inpaint-capable models (the `image_inpaint` capability, SDXL family); others
    // ignore it and run the whole-image edit. Threaded to the worker payload as-is.
    #[serde(default)]
    pub(crate) mask_asset_id: Option<String>,
    // How the source is fitted to the output W×H on an edit (epic 2551): "crop"
    // (cover, default), "pad" (letterbox), "outpaint" (pad + generate the border —
    // inpaint-capable models only), "stretch" (legacy). Threaded to the worker as-is;
    // the worker normalizes unknown values back to crop.
    #[serde(default = "default_fit_mode")]
    pub(crate) fit_mode: String,
    #[serde(default = "default_requested_gpu")]
    pub(crate) requested_gpu: String,
    #[serde(default, skip_serializing_if = "ImageUpscaleRequest::is_disabled")]
    pub(crate) upscale: ImageUpscaleRequest,
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
    // How the starting image is fitted to the output W×H for the image-conditioned
    // modes (image_to_video / first_last_frame), mirroring Image Studio Edit (sc-6139):
    // "crop" (cover, default) or "pad" (letterbox) — never distort. Threaded to the
    // worker as-is; the worker normalizes unknown values back to crop. Outpaint is
    // inpaint-only and not offered for video.
    #[serde(default = "default_fit_mode")]
    pub(crate) fit_mode: String,
    #[serde(default)]
    pub(crate) last_frame_asset_id: Option<String>,
    #[serde(default)]
    pub(crate) source_clip_asset_id: Option<String>,
    /// Multiple source clips for Bernini's multi-source-video edit mode
    /// (`multi_video_to_video` / mv2v, sc-5425). The worker pushes one
    /// `Conditioning::VideoClip` per clip; v2v/rv2v/ads2v use the single
    /// `source_clip_asset_id` instead.
    #[serde(default)]
    pub(crate) source_clip_asset_ids: Vec<String>,
    #[serde(default)]
    pub(crate) bridge_right_clip_asset_id: Option<String>,
    /// Subject reference images for Bernini's reference-driven video modes
    /// (`reference_to_video` / `reference_video_to_video` / `ads2v`, sc-4703 /
    /// sc-5425). Carried as a list so the mode can supply multiple references; the
    /// worker VAE/ViT-encodes each into the planner's `MultiReference` conditioning.
    #[serde(default)]
    pub(crate) reference_asset_ids: Vec<String>,
    /// The reference *video* slot for Bernini's `ads2v` mode (sc-5425): a second
    /// source video distinct from `source_clip_asset_id` (the clip being edited).
    /// The worker pushes it as a second `Conditioning::VideoClip`.
    #[serde(default)]
    pub(crate) reference_clip_asset_id: Option<String>,
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
    /// MLX quantization bit width (4 or 8). When set, the convert job quantizes the
    /// MLX weights; absent leaves them at the requested float dtype.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) quantize_bits: Option<u32>,
    /// MLX quantization group size (defaults to 64 in the convert tool when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) quantize_group_size: Option<u32>,
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
    /// Optional caller-supplied SHA-256 of the imported file. When present, the worker
    /// verifies the downloaded file against it and fails on a mismatch (sc-6137). HF
    /// repo imports are verified automatically from HF's own per-file digests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expected_sha256: Option<String>,
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
    /// Specific base model the LoRA targets (e.g. `wan_2_2_t2v_14b`). Recorded on
    /// the manifest entry so the loader's base-model gating can match it (sc-1955).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) base_model: Option<String>,
    /// Optional caller-supplied SHA-256 of the imported file. When present, the worker
    /// verifies the downloaded file against it and fails on a mismatch (sc-6137). HF
    /// repo imports are verified automatically from HF's own per-file digests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expected_sha256: Option<String>,
    #[serde(default = "default_lora_scope")]
    pub(crate) scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) project_id: Option<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    pub(crate) uploaded_source_path: bool,
    /// Staged path of the low-noise expert half for a Wan A14B MoE upload (sc-1991).
    /// Set server-side from the `secondaryFile` multipart part only; never from
    /// client JSON. Its presence flags the import as a paired MoE write.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub(crate) secondary_source_path: Option<String>,
}
