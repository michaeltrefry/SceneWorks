use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{
    DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request as AxumRequest, State,
};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use futures_util::future::join_all;
use parking_lot::Mutex;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, DuplicateJobRequest, JobCreateRequest,
    JobSnapshot, JobType, JsonObject, ProgressRequest, QueueSummary, WorkerHeartbeatRequest,
    WorkerRegisterRequest, WorkerSnapshot,
};
use sceneworks_core::jobs_store::{
    CreateJob, DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker,
    WorkerHeartbeat, JOB_STATUSES,
};
use sceneworks_core::project_store::{
    AssetStatusPatch, CharacterCreateInput, CharacterLookInput, CharacterLookUpdateInput,
    CharacterLoraInput, CharacterLoraUpdateInput, CharacterReferenceInput,
    CharacterReferenceUpdateInput, CharacterUpdateInput, ProjectStore, ProjectStoreError,
    UploadAsset,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::time::{Instant as TokioInstant, MissedTickBehavior};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;

const PUBLIC_PATHS: &[&str] = &[
    "/api/v1/health",
    "/api/v1/access",
    "/api/v1/auth/verify",
    "/api/v1/jobs/events",
];
const DEFAULT_CORS_ORIGINS: &str = concat!(
    "http://localhost:5173,http://127.0.0.1:5173,",
    "http://localhost:5174,http://127.0.0.1:5174,",
    "http://localhost:5175,http://127.0.0.1:5175,",
    "http://localhost:5176,http://127.0.0.1:5176"
);
const EVENT_BUFFER_SIZE: usize = 100;
const HEARTBEAT_SSE_DATA: &str = "{}";
#[cfg(test)]
const HEARTBEAT_SSE_WIRE: &str = "event: heartbeat\ndata: {}\n\n";
const MAX_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MANIFEST_CACHE_LIMIT: usize = 16;
const MODEL_SIZE_CACHE_LIMIT: usize = 64;

#[derive(Debug, Clone)]
pub struct Settings {
    pub api_runtime: String,
    pub app_version: String,
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub access_token: String,
    pub cors_origins: Vec<String>,
    pub worker_timeout_seconds: u64,
    pub jobs_db_path: PathBuf,
}

impl Settings {
    pub fn from_env() -> Self {
        let data_dir = env_path("SCENEWORKS_DATA_DIR", "data");
        let jobs_db_path = std::env::var("SCENEWORKS_JOBS_DB_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("cache").join("jobs.db"));
        Self {
            api_runtime: env_string("SCENEWORKS_API_RUNTIME", "rust"),
            app_version: env_string("SCENEWORKS_APP_VERSION", "0.2.0"),
            host: env_string("SCENEWORKS_API_HOST", "0.0.0.0"),
            port: std::env::var("SCENEWORKS_API_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8000),
            data_dir,
            config_dir: env_path("SCENEWORKS_CONFIG_DIR", "config"),
            access_token: std::env::var("SCENEWORKS_ACCESS_TOKEN")
                .unwrap_or_default()
                .trim()
                .to_owned(),
            cors_origins: env_string("SCENEWORKS_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect(),
            worker_timeout_seconds: std::env::var("SCENEWORKS_WORKER_TIMEOUT_SECONDS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(90),
            jobs_db_path,
        }
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.data_dir.join("projects")
    }
}

#[derive(Clone)]
pub struct AppState {
    settings: Settings,
    jobs_store: Arc<JobsStore>,
    project_store: Arc<ProjectStore>,
    events: Arc<EventHub>,
    event_tickets: Arc<EventTicketStore>,
    manifest_cache: Arc<Mutex<ManifestCache>>,
    model_size_cache: Arc<Mutex<ModelSizeCache>>,
    http_client: reqwest::Client,
    interrupted_jobs_on_startup: usize,
}

struct ApiJson<T>(T);

#[axum::async_trait]
impl<S, T> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(request: AxumRequest, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(json_rejection_response(rejection)),
        }
    }
}

fn json_rejection_response(rejection: JsonRejection) -> Response {
    let detail = match rejection {
        JsonRejection::JsonDataError(error) => error.body_text(),
        JsonRejection::JsonSyntaxError(error) => error.body_text(),
        other => other.body_text(),
    };
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({
            "detail": [{
                "type": "json_invalid",
                "loc": ["body", 0],
                "msg": "JSON decode error",
                "input": {},
                "ctx": { "error": detail }
            }]
        })),
    )
        .into_response()
}

#[derive(Debug, Clone)]
struct EventMessage {
    event: String,
    data: String,
}

#[derive(Debug, Default)]
struct EventHub {
    state: Mutex<EventHubState>,
}

#[derive(Debug, Default)]
struct EventHubState {
    next_subscriber_id: u64,
    subscribers: HashMap<u64, mpsc::Sender<EventMessage>>,
}

impl EventHub {
    fn subscribe(&self) -> ReceiverStream<EventMessage> {
        let (sender, receiver) = mpsc::channel(EVENT_BUFFER_SIZE);
        let mut state = self.state.lock();
        let subscriber_id = state.next_subscriber_id;
        state.next_subscriber_id = state.next_subscriber_id.wrapping_add(1);
        state.subscribers.insert(subscriber_id, sender);
        ReceiverStream::new(receiver)
    }

    fn publish(&self, message: EventMessage) {
        let mut state = self.state.lock();
        state.subscribers.retain(|_, sender| {
            sender
                .try_send(message.clone())
                .map(|_| true)
                .unwrap_or(false)
        });
    }
}

#[derive(Debug)]
struct EventTicketStore {
    ttl: Duration,
    tickets: Mutex<HashMap<String, Instant>>,
}

impl EventTicketStore {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            ttl: Duration::from_secs(ttl_seconds),
            tickets: Mutex::new(HashMap::new()),
        }
    }

    fn issue(&self) -> Result<EventTicket, ApiError> {
        let now = Instant::now();
        let mut tickets = self.tickets.lock();
        prune_tickets(&mut tickets, now);
        let ticket = Uuid::new_v4().simple().to_string();
        tickets.insert(ticket.clone(), now + self.ttl);
        Ok(EventTicket {
            ticket,
            expires_in_seconds: self.ttl.as_secs(),
        })
    }

    fn consume(&self, ticket: &str) -> Result<(), ApiError> {
        let now = Instant::now();
        let mut tickets = self.tickets.lock();
        prune_tickets(&mut tickets, now);
        match tickets.remove(ticket) {
            Some(expires_at) if expires_at >= now => Ok(()),
            _ => Err(ApiError::unauthorized(
                "Invalid or expired event stream ticket",
            )),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EventTicket {
    ticket: String,
    expires_in_seconds: u64,
}

#[derive(Debug, Default)]
struct ModelSizeCache {
    entries: HashMap<ModelSizeCacheKey, u64>,
    order: VecDeque<ModelSizeCacheKey>,
}

type ModelSizeCacheKey = (String, Vec<String>);

impl ModelSizeCache {
    fn get(&mut self, key: &ModelSizeCacheKey) -> Option<u64> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get(key).copied()
    }

    fn insert(&mut self, key: ModelSizeCacheKey, value: u64) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
        while self.order.len() > MODEL_SIZE_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &ModelSizeCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct ManifestCacheKey {
    path: PathBuf,
    field: String,
    modified_ns: u128,
    size: u64,
}

#[derive(Debug, Default)]
struct ManifestCache {
    entries: HashMap<ManifestCacheKey, Vec<Value>>,
    order: VecDeque<ManifestCacheKey>,
}

impl ManifestCache {
    fn get(&mut self, key: &ManifestCacheKey) -> Option<Vec<Value>> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: ManifestCacheKey, value: Vec<Value>) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
        while self.order.len() > MANIFEST_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &ManifestCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::from_env();
    let address: SocketAddr = format!("{}:{}", settings.host, settings.port).parse()?;
    let app = create_app(settings)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("SceneWorks Rust API listening on http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn create_app(settings: Settings) -> Result<Router, JobsStoreError> {
    let jobs_store = Arc::new(JobsStore::new(&settings.jobs_db_path));
    jobs_store.initialize()?;
    let interrupted_jobs_on_startup = jobs_store.mark_interrupted_on_startup()?.len();
    let project_store = Arc::new(ProjectStore::new(
        settings.data_dir.clone(),
        settings.app_version.clone(),
    ));
    let state = AppState {
        settings,
        jobs_store,
        project_store,
        events: Arc::new(EventHub::default()),
        event_tickets: Arc::new(EventTicketStore::new(30)),
        manifest_cache: Arc::new(Mutex::new(ManifestCache::default())),
        model_size_cache: Arc::new(Mutex::new(ModelSizeCache::default())),
        http_client: reqwest::Client::new(),
        interrupted_jobs_on_startup,
    };
    let cors = cors_layer(&state.settings);

    Ok(Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/access", get(access))
        .route("/api/v1/auth/verify", post(verify_access))
        .route("/api/v1/projects", get(list_projects).post(create_project))
        .route("/api/v1/projects/:project_id", get(get_project))
        .route(
            "/api/v1/projects/:project_id/reindex",
            post(reindex_project_endpoint),
        )
        .route(
            "/api/v1/projects/:project_id/assets",
            get(list_assets).post(import_asset),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id",
            get(get_asset).delete(delete_asset),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/purge",
            delete(purge_asset),
        )
        .route(
            "/api/v1/projects/:project_id/assets/:asset_id/status",
            patch(update_asset_status),
        )
        .route(
            "/api/v1/projects/:project_id/files/*relative_path",
            get(get_project_file),
        )
        .route(
            "/api/v1/projects/:project_id/characters",
            get(list_characters).post(create_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id",
            get(get_character)
                .patch(update_character)
                .delete(archive_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/archive",
            post(archive_character_explicit),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/purge",
            delete(purge_character),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/references",
            post(add_character_reference),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/references/:asset_id",
            patch(update_character_reference).delete(remove_character_reference),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/looks",
            post(create_character_look),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/looks/:look_id",
            patch(update_character_look).delete(delete_character_look),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/loras",
            post(attach_character_lora),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/loras/:link_id",
            patch(update_character_lora).delete(detach_character_lora),
        )
        .route(
            "/api/v1/projects/:project_id/characters/:character_id/test-jobs",
            post(create_character_test_job),
        )
        .route(
            "/api/v1/projects/:project_id/timelines",
            get(list_timelines).post(create_timeline),
        )
        .route(
            "/api/v1/projects/:project_id/timelines/:timeline_id",
            get(get_timeline).put(update_timeline),
        )
        .route(
            "/api/v1/projects/:project_id/timelines/:timeline_id/exports",
            post(create_timeline_export),
        )
        .route(
            "/api/v1/projects/:project_id/timelines/:timeline_id/items/:item_id/frames",
            post(extract_timeline_frame),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks",
            get(list_person_tracks),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/detections",
            post(create_person_detection_job),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/jobs",
            post(create_person_track_job),
        )
        .route(
            "/api/v1/projects/:project_id/person-tracks/:track_id",
            get(get_person_track),
        )
        .route("/api/v1/image/jobs", post(create_image_job))
        .route("/api/v1/video/jobs", post(create_video_job))
        .route("/api/v1/models", get(list_models))
        .route(
            "/api/v1/models/:model_id/download",
            post(create_model_download_job),
        )
        .route("/api/v1/loras", get(list_loras))
        .route("/api/v1/loras/import", post(create_lora_import_job))
        .route("/api/v1/recipe-presets", get(list_recipe_presets))
        .route("/api/v1/jobs", get(list_jobs).post(create_job))
        .route("/api/v1/jobs/claim", post(claim_job))
        .route("/api/v1/jobs/events", get(job_events))
        .route("/api/v1/jobs/events/ticket", post(create_event_ticket))
        .route("/api/v1/jobs/:job_id", get(get_job))
        .route("/api/v1/jobs/:job_id/cancel", post(cancel_job))
        .route("/api/v1/jobs/:job_id/retry", post(retry_job))
        .route("/api/v1/jobs/:job_id/duplicate", post(duplicate_job))
        .route("/api/v1/jobs/:job_id/progress", post(update_job_progress))
        .route("/api/v1/queue", get(queue_summary))
        .route("/api/v1/workers", get(list_workers))
        .route("/api/v1/workers/register", post(register_worker))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_worker),
        )
        .fallback(route_not_found)
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(middleware::from_fn_with_state(state, access_control))
        .layer(cors))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobsQuery {
    project_id: Option<String>,
    status: Option<String>,
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssetsQuery {
    include_rejected: Option<bool>,
    include_trashed: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharactersQuery {
    include_archived: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LorasQuery {
    model_family: Option<String>,
    project_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecipePresetsQuery {
    project_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EventsQuery {
    ticket: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    runtime: String,
    version: String,
    auth_required: bool,
    directories: DirectoriesResponse,
    interrupted_jobs_on_startup: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DirectoriesResponse {
    data: String,
    config: String,
    projects: String,
    jobs_db: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AccessResponse {
    auth_required: bool,
    token_header: &'static str,
}

#[derive(Debug, Serialize)]
struct VerifyResponse {
    ok: bool,
}

#[derive(Debug, Deserialize)]
struct ProjectCreateRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterCreateRequest {
    name: String,
    #[serde(default = "default_character_type", rename = "type")]
    character_type: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterUpdateRequest {
    name: Option<String>,
    #[serde(default, rename = "type")]
    character_type: Option<String>,
    description: Option<String>,
    archived: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterReferenceRequest {
    asset_id: String,
    #[serde(default)]
    approved: bool,
    #[serde(default = "default_reference_role")]
    role: String,
    #[serde(default)]
    notes: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterReferenceUpdateRequest {
    approved: Option<bool>,
    role: Option<String>,
    notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterLookRequest {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    approved_reference_ids: Vec<String>,
    #[serde(default)]
    recipe_settings: JsonObject,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterLookUpdateRequest {
    name: Option<String>,
    description: Option<String>,
    approved_reference_ids: Option<Vec<String>>,
    recipe_settings: Option<JsonObject>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterLoraRequest {
    #[serde(default)]
    lora_id: Option<String>,
    name: String,
    #[serde(default)]
    source_path: Option<String>,
    #[serde(default)]
    trigger_words: Vec<String>,
    #[serde(default = "default_character_lora_weight")]
    default_weight: f64,
    #[serde(default)]
    compatibility: JsonObject,
    #[serde(default = "default_project_lora_scope")]
    scope: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterLoraUpdateRequest {
    name: Option<String>,
    trigger_words: Option<Vec<String>>,
    default_weight: Option<f64>,
    compatibility: Option<JsonObject>,
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CharacterTestRequest {
    prompt: String,
    #[serde(default = "default_image_model")]
    model: String,
    #[serde(default = "default_image_count")]
    count: u32,
    #[serde(default = "default_image_size")]
    width: u32,
    #[serde(default = "default_image_size")]
    height: u32,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
    #[serde(default)]
    look_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimelineCreateRequest {
    #[serde(default = "default_timeline_name")]
    name: String,
    #[serde(default = "default_aspect_ratio")]
    aspect_ratio: String,
    #[serde(default = "default_timeline_fps")]
    fps: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimelineSaveRequest {
    timeline: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimelineExportRequest {
    #[serde(default = "default_export_resolution")]
    resolution: u32,
    #[serde(default = "default_timeline_fps")]
    fps: u32,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameExtractRequest {
    playhead_seconds: f64,
    #[serde(default = "default_frame_intended_use")]
    intended_use: String,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersonDetectionJobRequest {
    source_asset_id: String,
    #[serde(default)]
    source_timestamp: Option<f64>,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersonTrackJobRequest {
    source_asset_id: String,
    representative_frame_asset_id: String,
    detection: JsonObject,
    #[serde(default = "default_track_name")]
    track_name: String,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageJobRequest {
    project_id: String,
    #[serde(default)]
    project_name: Option<String>,
    #[serde(default = "default_image_mode")]
    mode: String,
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_image_model")]
    model: String,
    #[serde(default = "default_image_count")]
    count: u32,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default = "default_image_size")]
    width: u32,
    #[serde(default = "default_image_size")]
    height: u32,
    #[serde(default = "default_style_preset")]
    style_preset: String,
    #[serde(default)]
    recipe_preset_id: Option<String>,
    #[serde(default)]
    loras: Vec<Value>,
    #[serde(default)]
    character_id: Option<String>,
    #[serde(default)]
    character_look_id: Option<String>,
    #[serde(default)]
    source_asset_id: Option<String>,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
    #[serde(default)]
    advanced: JsonObject,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoJobRequest {
    project_id: String,
    #[serde(default)]
    project_name: Option<String>,
    #[serde(default = "default_video_mode")]
    mode: String,
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_video_model")]
    model: String,
    #[serde(default = "default_video_duration")]
    duration: ContractNumber,
    #[serde(default = "default_video_fps")]
    fps: u32,
    #[serde(default = "default_video_width")]
    width: u32,
    #[serde(default = "default_video_height")]
    height: u32,
    #[serde(default = "default_video_quality")]
    quality: String,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    loras: Vec<Value>,
    #[serde(default)]
    character_id: Option<String>,
    #[serde(default)]
    character_look_id: Option<String>,
    #[serde(default)]
    person_track_id: Option<String>,
    #[serde(default = "default_replacement_mode")]
    replacement_mode: String,
    #[serde(default)]
    source_asset_id: Option<String>,
    #[serde(default)]
    last_frame_asset_id: Option<String>,
    #[serde(default)]
    source_clip_asset_id: Option<String>,
    #[serde(default)]
    bridge_right_clip_asset_id: Option<String>,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
    #[serde(default)]
    advanced: JsonObject,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelDownloadRequest {
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoraImportRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lora_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    family: Option<String>,
    #[serde(default = "default_lora_scope")]
    scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    project_id: Option<String>,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "sceneworks-api",
        runtime: state.settings.api_runtime.clone(),
        version: state.settings.app_version.clone(),
        auth_required: !state.settings.access_token.is_empty(),
        directories: DirectoriesResponse {
            data: state.settings.data_dir.display().to_string(),
            config: state.settings.config_dir.display().to_string(),
            projects: state.settings.projects_dir().display().to_string(),
            jobs_db: state.settings.jobs_db_path.display().to_string(),
        },
        interrupted_jobs_on_startup: state.interrupted_jobs_on_startup,
    })
}

async fn access(State(state): State<AppState>) -> Json<AccessResponse> {
    Json(AccessResponse {
        auth_required: !state.settings.access_token.is_empty(),
        token_header: "X-SceneWorks-Token",
    })
}

async fn verify_access(State(state): State<AppState>, headers: HeaderMap) -> Json<VerifyResponse> {
    Json(VerifyResponse {
        ok: is_authorized(&headers, &state.settings),
    })
}

async fn list_projects(
    State(state): State<AppState>,
) -> Result<Json<Vec<sceneworks_core::project_store::ProjectSummary>>, ApiError> {
    Ok(Json(
        project_call(state, |store| store.list_projects()).await?,
    ))
}

async fn create_project(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ProjectCreateRequest>,
) -> Result<
    (
        StatusCode,
        Json<sceneworks_core::project_store::ProjectSummary>,
    ),
    ApiError,
> {
    let project = project_call(state, move |store| store.create_project(&payload.name)).await?;
    Ok((StatusCode::CREATED, Json(project)))
}

async fn get_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<sceneworks_core::project_store::ProjectSummary>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.get_project(&project_id)).await?,
    ))
}

async fn reindex_project_endpoint(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<sceneworks_core::project_store::ReindexResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.reindex_project(&project_id)).await?,
    ))
}

async fn list_assets(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<AssetsQuery>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.list_assets(
                &project_id,
                query.include_rejected.unwrap_or(false),
                query.include_trashed.unwrap_or(false),
            )
        })
        .await?,
    ))
}

async fn get_asset(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.get_asset(&project_id, &asset_id)).await?,
    ))
}

async fn import_asset(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        if field.name() != Some("file") {
            continue;
        }
        let filename = field.file_name().unwrap_or("upload").to_owned();
        let content_type = field.content_type().map(str::to_owned);
        let temp_path = write_upload_field_to_temp_file(&state, field).await?;
        let source_path = temp_path.clone();
        let asset = project_call(state, move |store| {
            store.import_asset(
                &project_id,
                UploadAsset {
                    filename,
                    content_type,
                    source_path,
                },
            )
        })
        .await
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&temp_path);
        })?;
        return Ok((StatusCode::CREATED, Json(asset)));
    }
    Err(ApiError::bad_request("Upload file field is required"))
}

async fn write_upload_field_to_temp_file(
    state: &AppState,
    mut field: axum::extract::multipart::Field<'_>,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state.settings.data_dir.join("cache").join("uploads");
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(format!("upload-{}.tmp", Uuid::new_v4().simple()));
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut uploaded_bytes = 0usize;
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        uploaded_bytes = uploaded_bytes.saturating_add(chunk.len());
        if uploaded_bytes > MAX_UPLOAD_BYTES {
            let _ = tokio::fs::remove_file(&temp_path).await;
            return Err(ApiError::payload_too_large("Uploaded file is too large"));
        }
        file.write_all(&chunk)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
    }
    file.flush()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(temp_path)
}

async fn update_asset_status(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<AssetStatusPatch>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_asset_status(&project_id, &asset_id, payload)
        })
        .await?,
    ))
}

async fn delete_asset(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::AssetMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_asset(&project_id, &asset_id)
        })
        .await?,
    ))
}

async fn purge_asset(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::AssetMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.purge_asset(&project_id, &asset_id)
        })
        .await?,
    ))
}

async fn get_project_file(
    State(state): State<AppState>,
    Path((project_id, relative_path)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let project_file = project_call(state, move |store| {
        store.project_file(&project_id, &relative_path)
    })
    .await?;
    let file = tokio::fs::File::open(&project_file.path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let stream = ReaderStream::new(file);
    Ok((
        [(header::CONTENT_TYPE, project_file.content_type)],
        Body::from_stream(stream),
    )
        .into_response())
}

async fn list_characters(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<CharactersQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.list_characters(&project_id, query.include_archived.unwrap_or(false))
        })
        .await?,
    ))
}

async fn create_character(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<CharacterCreateRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.create_character(
            &project_id,
            CharacterCreateInput {
                name: payload.name,
                character_type: payload.character_type,
                description: payload.description,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

async fn get_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_character(&project_id, &character_id)
        })
        .await?,
    ))
}

async fn update_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character(
                &project_id,
                &character_id,
                CharacterUpdateInput {
                    name: payload.name,
                    character_type: payload.character_type,
                    description: payload.description,
                    archived: payload.archived,
                },
            )
        })
        .await?,
    ))
}

async fn archive_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::CharacterMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.archive_character(&project_id, &character_id)
        })
        .await?,
    ))
}

async fn archive_character_explicit(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::CharacterMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.archive_character(&project_id, &character_id)
        })
        .await?,
    ))
}

async fn purge_character(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::project_store::CharacterMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.purge_character(&project_id, &character_id)
        })
        .await?,
    ))
}

async fn add_character_reference(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterReferenceRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.add_character_reference(
            &project_id,
            &character_id,
            CharacterReferenceInput {
                asset_id: payload.asset_id,
                approved: payload.approved,
                role: payload.role,
                notes: payload.notes,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

async fn update_character_reference(
    State(state): State<AppState>,
    Path((project_id, character_id, asset_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<CharacterReferenceUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character_reference(
                &project_id,
                &character_id,
                &asset_id,
                CharacterReferenceUpdateInput {
                    approved: payload.approved,
                    role: payload.role,
                    notes: payload.notes,
                },
            )
        })
        .await?,
    ))
}

async fn remove_character_reference(
    State(state): State<AppState>,
    Path((project_id, character_id, asset_id)): Path<(String, String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.remove_character_reference(&project_id, &character_id, &asset_id)
        })
        .await?,
    ))
}

async fn create_character_look(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterLookRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.create_character_look(
            &project_id,
            &character_id,
            CharacterLookInput {
                name: payload.name,
                description: payload.description,
                approved_reference_ids: payload.approved_reference_ids,
                recipe_settings: payload.recipe_settings,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

async fn update_character_look(
    State(state): State<AppState>,
    Path((project_id, character_id, look_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<CharacterLookUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character_look(
                &project_id,
                &character_id,
                &look_id,
                CharacterLookUpdateInput {
                    name: payload.name,
                    description: payload.description,
                    approved_reference_ids: payload.approved_reference_ids,
                    recipe_settings: payload.recipe_settings,
                },
            )
        })
        .await?,
    ))
}

async fn delete_character_look(
    State(state): State<AppState>,
    Path((project_id, character_id, look_id)): Path<(String, String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_character_look(&project_id, &character_id, &look_id)
        })
        .await?,
    ))
}

async fn attach_character_lora(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterLoraRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let character = project_call(state, move |store| {
        store.attach_character_lora(
            &project_id,
            &character_id,
            CharacterLoraInput {
                lora_id: payload.lora_id,
                name: payload.name,
                source_path: payload.source_path,
                trigger_words: payload.trigger_words,
                default_weight: payload.default_weight,
                compatibility: payload.compatibility,
                scope: payload.scope,
            },
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(character)))
}

async fn update_character_lora(
    State(state): State<AppState>,
    Path((project_id, character_id, link_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<CharacterLoraUpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_character_lora(
                &project_id,
                &character_id,
                &link_id,
                CharacterLoraUpdateInput {
                    name: payload.name,
                    trigger_words: payload.trigger_words,
                    default_weight: payload.default_weight,
                    compatibility: payload.compatibility,
                    scope: payload.scope,
                },
            )
        })
        .await?,
    ))
}

async fn detach_character_lora(
    State(state): State<AppState>,
    Path((project_id, character_id, link_id)): Path<(String, String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.detach_character_lora(&project_id, &character_id, &link_id)
        })
        .await?,
    ))
}

async fn create_character_test_job(
    State(state): State<AppState>,
    Path((project_id, character_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<CharacterTestRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_character_test_job(&payload)?;
    let character = project_call(state.clone(), {
        let project_id = project_id.clone();
        let character_id = character_id.clone();
        move |store| store.get_character(&project_id, &character_id)
    })
    .await?;
    let look = payload.look_id.as_deref().and_then(|look_id| {
        character
            .get("looks")
            .and_then(Value::as_array)
            .and_then(|looks| {
                looks
                    .iter()
                    .find(|look| look.get("id").and_then(Value::as_str) == Some(look_id))
                    .cloned()
            })
    });
    let approved_reference_ids = character
        .get("references")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|reference| {
            reference
                .get("approved")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|reference| reference.get("assetId").and_then(Value::as_str))
        .map(|asset_id| Value::String(asset_id.to_owned()))
        .collect::<Vec<_>>();
    let mut advanced = JsonObject::new();
    advanced.insert(
        "characterName".to_owned(),
        character.get("name").cloned().unwrap_or(Value::Null),
    );
    advanced.insert(
        "characterType".to_owned(),
        character.get("type").cloned().unwrap_or(Value::Null),
    );
    advanced.insert(
        "approvedReferenceIds".to_owned(),
        Value::Array(approved_reference_ids),
    );
    advanced.insert("look".to_owned(), look.unwrap_or(Value::Null));

    let mut job_payload = JsonObject::new();
    job_payload.insert(
        "mode".to_owned(),
        Value::String("character_image".to_owned()),
    );
    job_payload.insert("prompt".to_owned(), Value::String(payload.prompt));
    job_payload.insert("negativePrompt".to_owned(), Value::String(String::new()));
    job_payload.insert("model".to_owned(), Value::String(payload.model));
    job_payload.insert("count".to_owned(), json!(payload.count));
    job_payload.insert("seed".to_owned(), Value::Null);
    job_payload.insert("width".to_owned(), json!(payload.width));
    job_payload.insert("height".to_owned(), json!(payload.height));
    job_payload.insert(
        "stylePreset".to_owned(),
        Value::String("character-test".to_owned()),
    );
    job_payload.insert("sourceAssetId".to_owned(), Value::Null);
    job_payload.insert(
        "loras".to_owned(),
        character.get("loras").cloned().unwrap_or_else(|| json!([])),
    );
    job_payload.insert("characterId".to_owned(), Value::String(character_id));
    job_payload.insert(
        "characterLookId".to_owned(),
        payload.look_id.map(Value::String).unwrap_or(Value::Null),
    );
    job_payload.insert("advanced".to_owned(), Value::Object(advanced));
    let job = create_generation_job(
        state,
        JobType::ImageGenerate,
        Some(project_id),
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn list_timelines(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<sceneworks_core::project_store::TimelineSummary>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.list_timelines(&project_id)).await?,
    ))
}

async fn create_timeline(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<TimelineCreateRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let timeline = project_call(state, move |store| {
        store.create_timeline(
            &project_id,
            &payload.name,
            &payload.aspect_ratio,
            payload.fps,
        )
    })
    .await?;
    Ok((StatusCode::CREATED, Json(timeline)))
}

async fn get_timeline(
    State(state): State<AppState>,
    Path((project_id, timeline_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_timeline(&project_id, &timeline_id)
        })
        .await?,
    ))
}

async fn update_timeline(
    State(state): State<AppState>,
    Path((project_id, timeline_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TimelineSaveRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.save_existing_timeline(&project_id, &timeline_id, payload.timeline)
        })
        .await?,
    ))
}

async fn create_timeline_export(
    State(state): State<AppState>,
    Path((project_id, timeline_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TimelineExportRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_timeline_export(&payload)?;
    let timeline_result = project_call(state.clone(), {
        let project_id = project_id.clone();
        let timeline_id = timeline_id.clone();
        move |store| store.timeline_file_and_document(&project_id, &timeline_id)
    })
    .await?;
    let timeline_name = timeline_result
        .document
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("Timeline")
        .to_owned();
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert("timelineId".to_owned(), Value::String(timeline_id));
    job_payload.insert("timelineName".to_owned(), Value::String(timeline_name));
    job_payload.insert(
        "timelinePath".to_owned(),
        Value::String(timeline_result.file.relative_path),
    );
    job_payload.insert("resolution".to_owned(), json!(payload.resolution));
    job_payload.insert("fps".to_owned(), json!(payload.fps));
    let job = create_generation_job(
        state,
        JobType::TimelineExport,
        Some(project_id),
        None,
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn extract_timeline_frame(
    State(state): State<AppState>,
    Path((project_id, timeline_id, item_id)): Path<(String, String, String)>,
    ApiJson(payload): ApiJson<FrameExtractRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_frame_extract(&payload)?;
    let timeline_result = project_call(state.clone(), {
        let project_id = project_id.clone();
        let timeline_id = timeline_id.clone();
        move |store| store.timeline_file_and_document(&project_id, &timeline_id)
    })
    .await?;
    let item = find_timeline_item(&timeline_result.document, &item_id)?;
    let source_asset_id = required_string_field(item, "assetId")?.to_owned();
    let timestamp = source_timestamp_for_item(item, payload.playhead_seconds)?;
    let timeline_name = timeline_result
        .document
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("Timeline")
        .to_owned();
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert("timelineId".to_owned(), Value::String(timeline_id));
    job_payload.insert("timelineName".to_owned(), Value::String(timeline_name));
    job_payload.insert(
        "timelinePath".to_owned(),
        Value::String(timeline_result.file.relative_path),
    );
    job_payload.insert("timelineItemId".to_owned(), Value::String(item_id));
    job_payload.insert("sourceAssetId".to_owned(), Value::String(source_asset_id));
    job_payload.insert("sourceTimestamp".to_owned(), json!(timestamp));
    job_payload.insert(
        "playheadSeconds".to_owned(),
        json!(payload.playhead_seconds),
    );
    job_payload.insert(
        "intendedUse".to_owned(),
        Value::String(payload.intended_use),
    );
    let job = create_generation_job(
        state,
        JobType::FrameExtract,
        Some(project_id),
        None,
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn list_person_tracks(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.list_person_tracks(&project_id)).await?,
    ))
}

async fn get_person_track(
    State(state): State<AppState>,
    Path((project_id, track_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_person_track(&project_id, &track_id)
        })
        .await?,
    ))
}

async fn create_person_detection_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<PersonDetectionJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_person_detection_job(&payload)?;
    let project_name = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.project_stem(&project_id)
    })
    .await?;
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert(
        "sourceAssetId".to_owned(),
        Value::String(payload.source_asset_id),
    );
    job_payload.insert(
        "sourceTimestamp".to_owned(),
        payload.source_timestamp.map_or(Value::Null, Value::from),
    );
    let job = create_generation_job(
        state,
        JobType::PersonDetect,
        Some(project_id),
        Some(project_name),
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn create_person_track_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<PersonTrackJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_person_track_job(&payload)?;
    let project_name = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.project_stem(&project_id)
    })
    .await?;
    let mut job_payload = JsonObject::new();
    job_payload.insert("projectId".to_owned(), Value::String(project_id.clone()));
    job_payload.insert(
        "sourceAssetId".to_owned(),
        Value::String(payload.source_asset_id),
    );
    job_payload.insert(
        "representativeFrameAssetId".to_owned(),
        Value::String(payload.representative_frame_asset_id),
    );
    job_payload.insert("detection".to_owned(), Value::Object(payload.detection));
    job_payload.insert("trackName".to_owned(), Value::String(payload.track_name));
    let job = create_generation_job(
        state,
        JobType::PersonTrack,
        Some(project_id),
        Some(project_name),
        job_payload,
        payload.requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn create_image_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ImageJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_image_job(&payload)?;
    let job_type = if payload.mode == "edit_image" {
        JobType::ImageEdit
    } else {
        JobType::ImageGenerate
    };
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    if payload.recipe_preset_id.is_none() {
        job_payload.remove("recipePresetId");
    }
    apply_recipe_preset_to_image_payload(&state, &payload, &mut job_payload).await?;
    if payload.seed.is_none() {
        let count = job_payload
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(payload.count);
        job_payload.insert("seeds".to_owned(), random_image_seeds(count));
    }
    let job = create_generation_job(
        state,
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn apply_recipe_preset_to_image_payload(
    state: &AppState,
    payload: &ImageJobRequest,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(preset_id) = payload.recipe_preset_id.as_deref() else {
        return Ok(());
    };
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    let presets = recipe_preset_catalog(state, Some(&payload.project_id)).await?;
    let preset = presets
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(preset_id))
        .ok_or_else(|| ApiError::bad_request("Recipe preset not found"))?;

    let expanded_prompt = preset_prompt(&payload.prompt, preset);
    job_payload.insert("prompt".to_owned(), Value::String(expanded_prompt));
    if let Some(model) = preset
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
    }
    apply_recipe_preset_defaults(preset, job_payload)?;
    job_payload.insert(
        "stylePreset".to_owned(),
        Value::String(preset_id.to_owned()),
    );
    let loras = lora_catalog(state, Some(&payload.project_id)).await?;
    let existing_lora_ids = job_payload
        .get("loras")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut seen_lora_ids = existing_lora_ids;
    let mut preset_loras = Vec::new();
    let mut missing_lora_ids = Vec::new();
    for preset_lora in preset
        .get("builtInLoras")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let Some(lora_id) = preset_lora_id(&preset_lora) else {
            continue;
        };
        let Some(lora) = loras
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(lora_id))
        else {
            missing_lora_ids.push(Value::String(lora_id.to_owned()));
            continue;
        };
        if seen_lora_ids.iter().any(|seen_id| seen_id == lora_id) {
            continue;
        }
        preset_loras.push(serialize_preset_lora(lora, &preset_lora, lora_id));
        seen_lora_ids.push(lora_id.to_owned());
    }
    let advanced = job_payload
        .entry("advanced".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if !advanced.is_object() {
        *advanced = Value::Object(JsonObject::new());
    }
    let advanced = advanced
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("advanced payload must be an object"))?;
    advanced.insert(
        "recipePresetId".to_owned(),
        Value::String(preset_id.to_owned()),
    );
    advanced.remove("recipePresetName");
    if missing_lora_ids.is_empty() {
        advanced.remove("presetMissingLoras");
    } else {
        advanced.insert(
            "presetMissingLoras".to_owned(),
            Value::Array(missing_lora_ids),
        );
    }

    let user_loras = job_payload
        .remove("loras")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    preset_loras.extend(user_loras);
    job_payload.insert("loras".to_owned(), Value::Array(preset_loras));
    Ok(())
}

fn apply_recipe_preset_defaults(
    preset: &Value,
    job_payload: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(defaults) = preset.get("defaults").and_then(Value::as_object) else {
        return Ok(());
    };
    if let Some(count) = defaults.get("count").and_then(Value::as_u64) {
        let count = u32::try_from(count)
            .map_err(|_| ApiError::bad_request("Recipe preset count is out of range"))?;
        if !(1..=8).contains(&count) {
            return Err(ApiError::bad_request(
                "Recipe preset count must be between 1 and 8",
            ));
        }
        job_payload.insert("count".to_owned(), json!(count));
    }
    if let Some(resolution) = defaults.get("resolution").and_then(Value::as_str) {
        let (width, height) = parse_recipe_preset_resolution(resolution)?;
        validate_dimension(width, "width", 2048)?;
        validate_dimension(height, "height", 2048)?;
        job_payload.insert("width".to_owned(), json!(width));
        job_payload.insert("height".to_owned(), json!(height));
        let advanced = job_payload
            .entry("advanced".to_owned())
            .or_insert_with(|| Value::Object(JsonObject::new()));
        if !advanced.is_object() {
            *advanced = Value::Object(JsonObject::new());
        }
        advanced
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("advanced payload must be an object"))?
            .insert(
                "resolution".to_owned(),
                Value::String(resolution.to_owned()),
            );
    }
    if let Some(negative_prompt) = defaults.get("negativePrompt").and_then(Value::as_str) {
        job_payload.insert(
            "negativePrompt".to_owned(),
            Value::String(negative_prompt.to_owned()),
        );
    }
    Ok(())
}

fn parse_recipe_preset_resolution(value: &str) -> Result<(u32, u32), ApiError> {
    let Some((width, height)) = value.split_once('x') else {
        return Err(ApiError::bad_request(
            "Recipe preset resolution must use WIDTHxHEIGHT",
        ));
    };
    let width = width
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("Recipe preset width must be a number"))?;
    let height = height
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("Recipe preset height must be a number"))?;
    Ok((width, height))
}

async fn create_video_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VideoJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_video_job(&payload)?;
    let job_type = match payload.mode.as_str() {
        "extend_clip" => JobType::VideoExtend,
        "video_bridge" => JobType::VideoBridge,
        "replace_person" => JobType::PersonReplace,
        _ => JobType::VideoGenerate,
    };
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        job_type,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn route_not_found(request: Request<axum::body::Body>) -> Response {
    let path = request.uri().path();
    let lower_path = path.to_ascii_lowercase();
    if path.contains("/files/")
        && (path.contains("..")
            || lower_path.contains("%2e")
            || lower_path.contains("%2f")
            || lower_path.contains("%5c"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid project file path" })),
        )
            .into_response();
    }
    if path.contains("/person-tracks/")
        && (path.contains("..")
            || lower_path.contains("%2e")
            || lower_path.contains("%2f")
            || lower_path.contains("%5c"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid person track ID" })),
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": "Not Found" })),
    )
        .into_response()
}

async fn list_models(State(state): State<AppState>) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(model_catalog(&state).await?))
}

async fn create_model_download_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let download = model_download(&model)
        .ok_or_else(|| ApiError::bad_request("Model does not define a Hugging Face download"))?;
    let repo = required_string_field(&download, "repo")?.to_owned();
    let files = download
        .get("files")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.clone()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&model_id)
                .to_owned(),
        ),
    );
    job_payload.insert(
        "provider".to_owned(),
        Value::String(required_string_field(&download, "provider")?.to_owned()),
    );
    job_payload.insert("repo".to_owned(), Value::String(repo.clone()));
    job_payload.insert("files".to_owned(), json!(files));
    job_payload.insert(
        "targetDir".to_owned(),
        Value::String(
            state
                .settings
                .data_dir
                .join("models")
                .join(safe_download_dir(&repo))
                .display()
                .to_string(),
        ),
    );

    let job = create_generation_job(
        state,
        JobType::ModelDownload,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn list_loras(
    State(state): State<AppState>,
    Query(query): Query<LorasQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let mut items = lora_catalog(&state, query.project_id.as_deref()).await?;
    if let Some(model_family) = query.model_family {
        items.retain(|item| {
            lora_families(item)
                .iter()
                .any(|family| family == &model_family)
        });
    }
    Ok(Json(items))
}

async fn list_recipe_presets(
    State(state): State<AppState>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        recipe_preset_catalog(&state, query.project_id.as_deref()).await?,
    ))
}

async fn create_lora_import_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<LoraImportRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo or source path",
        ));
    }
    if !matches!(payload.scope.as_str(), "global" | "project") {
        return Err(ApiError::bad_request(
            "LoRA scope must be global or project",
        ));
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            payload.source_path.as_deref().and_then(|path| {
                FsPath::new(path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "Imported LoRA".to_owned());
    let lora_id = payload
        .lora_id
        .clone()
        .unwrap_or_else(|| slugify_lora_id(&name));
    let target_name = safe_download_dir(&lora_id);
    let (target_dir, manifest_path, source_path, project_id, project_name) =
        if payload.scope == "project" {
            let Some(project_id) = payload.project_id.clone() else {
                return Err(ApiError::bad_request(
                    "Project LoRA imports require projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), &project_id).await?;
            (
                project_path
                    .join("loras")
                    .join("imports")
                    .join(&target_name),
                project_path.join("loras").join("manifest.jsonc"),
                format!("loras/imports/{target_name}"),
                Some(project_id),
                None,
            )
        } else {
            (
                state.settings.data_dir.join("loras").join(&target_name),
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
                format!("loras/{target_name}"),
                None,
                None,
            )
        };
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": lora_id,
        "name": name,
        "scope": payload.scope.clone(),
        "source": {
            "provider": if payload.repo.is_some() { "huggingface" } else { "local" },
            "repo": payload.repo.clone(),
            "path": source_path,
        },
        "files": payload.files.clone(),
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
    if let Some(family) = payload.family.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("family".to_owned(), Value::String(family));
        }
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("loraId".to_owned(), manifest_entry["id"].clone());
    payload.insert("name".to_owned(), manifest_entry["name"].clone());
    payload.insert(
        "targetDir".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    let job = create_generation_job(
        state,
        JobType::LoraImport,
        project_id,
        project_name,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn list_jobs(
    State(state): State<AppState>,
    Query(query): Query<JobsQuery>,
) -> Result<Json<Vec<JobSnapshot>>, ApiError> {
    if let Some(status) = &query.status {
        if !JOB_STATUSES.contains(&status.as_str()) {
            return Err(ApiError::bad_request("Unsupported job status"));
        }
    }
    Ok(Json(
        store_call(state, move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            store.list_jobs(
                query.project_id.as_deref(),
                query.status.as_deref(),
                query.limit.unwrap_or(100),
            )
        })
        .await?,
    ))
}

async fn create_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<JobCreateRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type: payload.job_type,
            project_id: payload.project_id,
            project_name: payload.project_name,
            payload: payload.payload,
            requested_gpu: payload.requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn claim_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<ClaimRequest>,
) -> Result<Json<ClaimResponse>, ApiError> {
    let response = store_call(state.clone(), move |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.claim_next_job(&payload.worker_id)
    })
    .await?;
    if let Some(job) = &response {
        publish(&state, "job.updated", job);
        publish_queue(&state).await?;
    }
    Ok(Json(ClaimResponse {
        job: response,
        extra: Default::default(),
    }))
}

async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    Ok(Json(
        store_call(state, move |store, _timeout| store.get_job(&job_id)).await?,
    ))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.cancel_job(&job_id)
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

async fn retry_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.retry_job(&job_id)
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn duplicate_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(payload): ApiJson<DuplicateJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.duplicate_job(
            &job_id,
            DuplicateJob {
                payload_changes: payload.payload_changes,
                requested_gpu: payload.requested_gpu,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn update_job_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    ApiJson(payload): ApiJson<ProgressRequest>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let progress = number_to_f64(&payload.progress, "progress")?;
    let eta_seconds = optional_number_to_f64(payload.eta_seconds.as_ref(), "etaSeconds")?;
    let job = store_call(state.clone(), move |store, _timeout| {
        store.update_job_progress(
            &job_id,
            ProgressUpdate {
                status: payload.status,
                stage: payload.stage,
                progress,
                message: payload.message,
                error: payload.error,
                result: payload.result,
                eta_seconds,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

async fn queue_summary(State(state): State<AppState>) -> Result<Json<QueueSummary>, ApiError> {
    Ok(Json(queue_summary_snapshot(state).await?))
}

async fn list_workers(
    State(state): State<AppState>,
) -> Result<Json<Vec<WorkerSnapshot>>, ApiError> {
    Ok(Json(
        store_call(state, move |store, timeout| {
            store.mark_stale_workers_interrupted(timeout)?;
            store.list_workers()
        })
        .await?,
    ))
}

async fn register_worker(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<WorkerRegisterRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    let worker = store_call(state.clone(), move |store, _timeout| {
        store.register_worker(RegisterWorker {
            worker_id: payload.worker_id,
            gpu_id: payload.gpu_id,
            gpu_name: payload.gpu_name,
            capabilities: payload.capabilities,
            loaded_models: payload.loaded_models,
        })
    })
    .await?;
    publish(&state, "worker.updated", &worker);
    publish_queue(&state).await?;
    Ok(Json(worker))
}

async fn heartbeat_worker(
    State(state): State<AppState>,
    Path(worker_id): Path<String>,
    ApiJson(payload): ApiJson<WorkerHeartbeatRequest>,
) -> Result<Json<WorkerSnapshot>, ApiError> {
    let worker = store_call(state.clone(), move |store, _timeout| {
        store.heartbeat_worker(WorkerHeartbeat {
            worker_id,
            status: payload.status,
            current_job_id: payload.current_job_id,
            loaded_models: payload.loaded_models,
        })
    })
    .await?;
    publish(&state, "worker.updated", &worker);
    Ok(Json(worker))
}

async fn create_event_ticket(State(state): State<AppState>) -> Result<Json<EventTicket>, ApiError> {
    Ok(Json(state.event_tickets.issue()?))
}

async fn job_events(
    State(state): State<AppState>,
    Query(query): Query<EventsQuery>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    if !state.settings.access_token.is_empty() {
        state
            .event_tickets
            .consume(query.ticket.as_deref().unwrap_or_default())?;
    }
    Ok(Sse::new(sse_event_stream(state.events.subscribe())))
}

fn sse_event_stream(
    messages: ReceiverStream<EventMessage>,
) -> impl futures_util::Stream<Item = Result<Event, Infallible>> {
    let mut heartbeat = tokio::time::interval_at(
        TokioInstant::now() + Duration::from_secs(15),
        Duration::from_secs(15),
    );
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
    futures_util::stream::unfold(
        (messages, heartbeat, true),
        |(mut messages, mut heartbeat, send_ready)| async move {
            if send_ready {
                return Some((Ok(ready_event()), (messages, heartbeat, false)));
            }
            tokio::select! {
                message = messages.next() => {
                    message.map(|message| (Ok(sse_message_event(message)), (messages, heartbeat, false)))
                }
                _ = heartbeat.tick() => {
                    Some((Ok(heartbeat_event()), (messages, heartbeat, false)))
                }
            }
        },
    )
}

fn ready_event() -> Event {
    Event::default()
        .event("ready")
        .data(json!({ "status": "connected" }).to_string())
}

fn sse_message_event(message: EventMessage) -> Event {
    Event::default().event(message.event).data(message.data)
}

fn heartbeat_event() -> Event {
    Event::default().event("heartbeat").data(HEARTBEAT_SSE_DATA)
}

async fn access_control(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if request.method() == Method::OPTIONS
        || PUBLIC_PATHS.contains(&request.uri().path())
        || is_authorized(request.headers(), &state.settings)
    {
        return next.run(request).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "detail": "SceneWorks access token required",
            "authRequired": true
        })),
    )
        .into_response()
}

fn cors_layer(settings: &Settings) -> CorsLayer {
    let origins = settings
        .cors_origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect::<Vec<_>>();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            HeaderName::from_static("x-sceneworks-token"),
        ])
}

fn is_authorized(headers: &HeaderMap, settings: &Settings) -> bool {
    if settings.access_token.is_empty() {
        return true;
    }
    constant_time_eq(
        token_from_headers(headers).as_bytes(),
        settings.access_token.as_bytes(),
    )
}

fn token_from_headers(headers: &HeaderMap) -> String {
    if let Some(token) = headers
        .get("x-sceneworks-token")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return token.to_owned();
    }
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .fold(0, |difference, (left, right)| difference | (left ^ right))
        == 0
}

async fn store_call<T, F>(state: AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<JobsStore>, u64) -> Result<T, JobsStoreError> + Send + 'static,
{
    let timeout = state.settings.worker_timeout_seconds;
    let store = state.jobs_store.clone();
    tokio::task::spawn_blocking(move || operation(store, timeout))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(Into::into)
}

async fn project_call<T, F>(state: AppState, operation: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(Arc<ProjectStore>) -> Result<T, ProjectStoreError> + Send + 'static,
{
    let store = state.project_store.clone();
    tokio::task::spawn_blocking(move || operation(store))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(Into::into)
}

async fn queue_summary_snapshot(state: AppState) -> Result<QueueSummary, ApiError> {
    store_call(state, |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.queue_summary()
    })
    .await
}

async fn create_generation_job(
    state: AppState,
    job_type: JobType,
    project_id: Option<String>,
    project_name: Option<String>,
    payload: JsonObject,
    requested_gpu: String,
) -> Result<JobSnapshot, ApiError> {
    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job(CreateJob {
            job_type,
            project_id,
            project_name,
            payload,
            requested_gpu,
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
        })
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(job)
}

async fn publish_queue(state: &AppState) -> Result<(), ApiError> {
    let queue = queue_summary_snapshot(state.clone()).await?;
    publish(state, "queue.updated", &queue);
    Ok(())
}

fn publish<T: Serialize>(state: &AppState, event: &str, data: &T) {
    if let Ok(data) = serde_json::to_string(data) {
        // Publishing with no subscribers is expected; slow subscribers are dropped so they reconnect.
        state.events.publish(EventMessage {
            event: event.to_owned(),
            data,
        });
    }
}

#[derive(Debug, Clone)]
struct DownloadContext {
    repo: String,
    files: Vec<String>,
}

async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let mut models = merge_entries_by_id(builtin, user);
    let download_contexts = models
        .iter()
        .map(model_download_context)
        .collect::<Result<Vec<_>, _>>()?;
    let download_size_bytes = join_all(download_contexts.iter().map(|context| async move {
        match context {
            Some(context) => {
                estimate_huggingface_download_size(state, &context.repo, &context.files).await
            }
            None => None,
        }
    }))
    .await;

    for (model, (download_context, download_size_bytes)) in models
        .iter_mut()
        .zip(download_contexts.into_iter().zip(download_size_bytes))
    {
        let (downloadable, installed_path, installed) =
            if let Some(download_context) = download_context {
                let installed_path = state
                    .settings
                    .data_dir
                    .join("models")
                    .join(safe_download_dir(&download_context.repo));
                let installed = model_is_installed(&installed_path);
                (true, Some(installed_path.display().to_string()), installed)
            } else {
                (false, None, false)
            };
        let object = model
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
        object.insert("downloadable".to_owned(), Value::Bool(downloadable));
        object.insert(
            "downloadSizeBytes".to_owned(),
            download_size_bytes
                .map(|value| json!(value))
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeLabel".to_owned(),
            download_size_bytes
                .map(format_bytes)
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        object.insert(
            "installState".to_owned(),
            Value::String(if installed { "installed" } else { "missing" }.to_owned()),
        );
        object.insert(
            "installedPath".to_owned(),
            installed_path.map(Value::String).unwrap_or(Value::Null),
        );
    }
    models.sort_by(|left, right| {
        let left_key = (
            left.get("type").and_then(Value::as_str).unwrap_or_default(),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(models)
}

async fn lora_catalog(state: &AppState, project_id: Option<&str>) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.loras.jsonc"), "loras").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.loras.jsonc"), "loras").await?;
    let mut loras = Vec::new();
    for lora in builtin {
        loras.push(normalize_lora_entry(
            lora,
            "builtin",
            &manifest_dir.join("builtin.loras.jsonc"),
            &state.settings.data_dir,
        )?);
    }
    let user = user
        .into_iter()
        .map(|lora| {
            normalize_lora_entry(
                lora,
                "global",
                &manifest_dir.join("user.loras.jsonc"),
                &state.settings.data_dir,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    loras = merge_entries_by_id(loras, user);
    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("loras").join("manifest.jsonc");
        let project_loras = load_manifest_entries(state, &project_manifest, "loras")
            .await?
            .into_iter()
            .map(|lora| normalize_lora_entry(lora, "project", &project_manifest, &project_path))
            .collect::<Result<Vec<_>, _>>()?;
        loras = merge_entries_by_id(loras, project_loras);
    }
    loras.sort_by(|left, right| {
        let left_key = (
            left.get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("family")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("family")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(loras)
}

async fn recipe_preset_catalog(
    state: &AppState,
    project_id: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin_manifest = manifest_dir.join("builtin.recipe-presets.jsonc");
    let user_manifest = manifest_dir.join("user.recipe-presets.jsonc");
    let builtin = load_manifest_entries(state, &builtin_manifest, "presets")
        .await?
        .into_iter()
        .map(|preset| normalize_recipe_preset_entry(preset, "builtin", &builtin_manifest))
        .collect::<Result<Vec<_>, _>>()?;
    let user = load_manifest_entries(state, &user_manifest, "presets")
        .await?
        .into_iter()
        .map(|preset| normalize_recipe_preset_entry(preset, "global", &user_manifest))
        .collect::<Result<Vec<_>, _>>()?;
    let mut presets = merge_entries_by_id(builtin, user);
    if let Some(project_id) = project_id {
        let project_path = project_path_for_id(state.clone(), project_id).await?;
        let project_manifest = project_path.join("recipes").join("presets.jsonc");
        let project_presets = load_manifest_entries(state, &project_manifest, "presets")
            .await?
            .into_iter()
            .map(|preset| normalize_recipe_preset_entry(preset, "project", &project_manifest))
            .collect::<Result<Vec<_>, _>>()?;
        presets = merge_entries_by_id(presets, project_presets);
    }
    for preset in presets.iter_mut() {
        finalize_recipe_preset_entry(preset)?;
    }
    presets.sort_by(|left, right| {
        let left_key = (
            left.get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            left.get("order").and_then(Value::as_i64).unwrap_or(10_000),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("scope")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right.get("order").and_then(Value::as_i64).unwrap_or(10_000),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });
    Ok(presets)
}

async fn project_path_for_id(state: AppState, project_id: &str) -> Result<PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let project = project_call(state, move |store| store.get_project(&project_id)).await?;
    Ok(PathBuf::from(project.path))
}

fn normalize_lora_entry(
    mut lora: Value,
    scope: &str,
    manifest_path: &FsPath,
    default_root: &FsPath,
) -> Result<Value, ApiError> {
    let object = lora
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("LoRA manifest entry must be an object"))?;
    object
        .entry("scope".to_owned())
        .or_insert_with(|| Value::String(scope.to_owned()));
    let source_path = object
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .or_else(|| object.get("path").and_then(Value::as_str));
    let installed_path = source_path.map(|source_path| {
        let path = PathBuf::from(source_path);
        if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        }
    });
    let install_state = if installed_path.as_ref().is_some_and(|path| !path.exists()) {
        "missing"
    } else {
        "installed"
    };
    object.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    object.insert(
        "installedPath".to_owned(),
        installed_path
            .map(|path| Value::String(path.display().to_string()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "installState".to_owned(),
        Value::String(install_state.to_owned()),
    );
    Ok(lora)
}

fn normalize_recipe_preset_entry(
    mut preset: Value,
    scope: &str,
    manifest_path: &FsPath,
) -> Result<Value, ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Recipe preset manifest entry must be an object"))?;
    object
        .entry("scope".to_owned())
        .or_insert_with(|| Value::String(scope.to_owned()));
    object.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    Ok(preset)
}

fn finalize_recipe_preset_entry(preset: &mut Value) -> Result<(), ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Recipe preset manifest entry must be an object"))?;
    object
        .entry("builtInLoras".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    object
        .entry("defaults".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    object
        .entry("prompt".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    Ok(())
}

fn preset_prompt(prompt: &str, preset: &Value) -> String {
    let fragments = preset.get("prompt").and_then(Value::as_object);
    [
        fragments
            .and_then(|value| value.get("prefix"))
            .and_then(Value::as_str),
        Some(prompt),
        fragments
            .and_then(|value| value.get("suffix"))
            .and_then(Value::as_str),
    ]
    .into_iter()
    .flatten()
    .map(str::trim)
    .filter(|value| !value.is_empty())
    .collect::<Vec<_>>()
    .join(", ")
}

fn preset_lora_id(preset_lora: &Value) -> Option<&str> {
    preset_lora
        .as_str()
        .or_else(|| preset_lora.get("id").and_then(Value::as_str))
}

fn preset_lora_weight(lora: &Value, preset_lora: &Value) -> f64 {
    preset_lora
        .get("weight")
        .and_then(Value::as_f64)
        .or_else(|| lora.get("defaultWeight").and_then(Value::as_f64))
        .or_else(|| lora.get("weight").and_then(Value::as_f64))
        .unwrap_or(0.8)
}

fn serialize_preset_lora(lora: &Value, preset_lora: &Value, lora_id: &str) -> Value {
    json!({
        "id": lora_id,
        "name": lora.get("name").and_then(Value::as_str).unwrap_or(lora_id),
        "scope": lora.get("scope").and_then(Value::as_str).unwrap_or("builtin"),
        "weight": preset_lora_weight(lora, preset_lora),
        "triggerWords": lora.get("triggerWords").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        "compatibility": lora.get("compatibility").cloned().unwrap_or_else(|| Value::Object(JsonObject::new())),
        "presetManaged": true
    })
}

async fn load_manifest_entries(
    state: &AppState,
    path: &FsPath,
    field: &str,
) -> Result<Vec<Value>, ApiError> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(_) => return Err(ApiError::internal("Failed to load manifest")),
    };
    let cache_key = ManifestCacheKey {
        path: path.to_path_buf(),
        field: field.to_owned(),
        modified_ns: metadata_modified_ns(&metadata),
        size: metadata.len(),
    };
    if let Some(entries) = state.manifest_cache.lock().get(&cache_key) {
        return Ok(entries);
    }

    let payload = tokio::fs::read_to_string(path)
        .await
        .map_err(|_| ApiError::internal("Failed to load manifest"))?;
    let manifest: Value = serde_json::from_str(&strip_jsonc_comments(&payload))
        .map_err(|_| ApiError::internal("Failed to load manifest"))?;
    let entries = manifest
        .get(field)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    state
        .manifest_cache
        .lock()
        .insert(cache_key, entries.clone());
    Ok(entries)
}

fn metadata_modified_ns(metadata: &std::fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn merge_entries_by_id(builtin: Vec<Value>, user: Vec<Value>) -> Vec<Value> {
    let mut entries = Vec::<Value>::new();
    for entry in builtin {
        if entry.get("id").and_then(Value::as_str).is_some() {
            entries.push(entry);
        }
    }
    for entry in user {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        if let Some(existing) = entries
            .iter_mut()
            .find(|existing| existing.get("id").and_then(Value::as_str) == Some(id))
        {
            merge_object(existing, entry);
        } else {
            entries.push(entry);
        }
    }
    entries
}

fn merge_object(base: &mut Value, override_value: Value) {
    if let (Some(base_object), Some(override_object)) =
        (base.as_object_mut(), override_value.as_object())
    {
        for (key, value) in override_object {
            base_object.insert(key.clone(), value.clone());
        }
    } else {
        *base = override_value;
    }
}

fn strip_jsonc_comments(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(character) = chars.next() {
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }

        if character == '"' {
            in_string = true;
            output.push(character);
            continue;
        }
        if character == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\r' || next == '\n' {
                    output.push(next);
                    break;
                }
            }
            continue;
        }
        if character == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut previous = '\0';
            for next in chars.by_ref() {
                if previous == '*' && next == '/' {
                    break;
                }
                previous = next;
            }
            continue;
        }
        output.push(character);
    }
    output
}

fn model_download(model: &Value) -> Option<Value> {
    model
        .get("downloads")?
        .as_array()?
        .iter()
        .find(|download| {
            download.get("provider").and_then(Value::as_str) == Some("huggingface")
                && download
                    .get("repo")
                    .and_then(Value::as_str)
                    .is_some_and(|repo| !repo.is_empty())
        })
        .cloned()
}

fn model_download_context(model: &Value) -> Result<Option<DownloadContext>, ApiError> {
    let Some(download) = model_download(model) else {
        return Ok(None);
    };
    Ok(Some(DownloadContext {
        repo: required_string_field(&download, "repo")?.to_owned(),
        files: string_array_field(&download, "files"),
    }))
}

async fn estimate_huggingface_download_size(
    state: &AppState,
    repo: &str,
    files: &[String],
) -> Option<u64> {
    if matches!(
        std::env::var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    ) {
        return None;
    }
    let cache_key = (repo.to_owned(), files.to_vec());
    if let Some(cached) = state.model_size_cache.lock().get(&cache_key) {
        return Some(cached);
    }
    let url = format!(
        "https://huggingface.co/api/models/{}?blobs=true",
        quote_huggingface_repo(repo)
    );
    let estimate =
        estimate_huggingface_download_size_uncached(&state.http_client, &url, files).await;
    if let Some(estimate) = estimate {
        state.model_size_cache.lock().insert(cache_key, estimate);
    }
    estimate
}

async fn estimate_huggingface_download_size_uncached(
    client: &reqwest::Client,
    url: &str,
    files: &[String],
) -> Option<u64> {
    let payload = tokio::time::timeout(Duration::from_secs(8), async {
        client
            .get(url.to_owned())
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json::<Value>()
            .await
            .ok()
    })
    .await
    .ok()??;
    let siblings = payload.get("siblings")?.as_array()?;
    download_size_from_siblings(siblings, files)
}

fn download_size_from_siblings(siblings: &[Value], files: &[String]) -> Option<u64> {
    let mut total = 0_u64;
    let mut found_size = false;
    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !allow_pattern_matches(filename, files) {
            continue;
        }
        let Some(size) = sibling.get("size").and_then(json_size_to_u64) else {
            continue;
        };
        found_size = true;
        total = total.saturating_add(size);
    }
    found_size.then_some(total)
}

fn json_size_to_u64(value: &Value) -> Option<u64> {
    if let Some(value) = value.as_u64() {
        return Some(value);
    }
    value.as_str().and_then(|value| value.parse::<u64>().ok())
}

fn allow_pattern_matches(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    patterns
        .iter()
        .any(|pattern| pattern_matches(pattern, path))
}

fn pattern_matches(pattern: &str, value: &str) -> bool {
    let (pattern, value) = if cfg!(windows) {
        (pattern.to_ascii_lowercase(), value.to_ascii_lowercase())
    } else {
        (pattern.to_owned(), value.to_owned())
    };
    glob::Pattern::new(&pattern).is_ok_and(|pattern| pattern.matches(&value))
}

fn quote_huggingface_repo(repo: &str) -> String {
    let mut output = String::new();
    for byte in repo.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn format_bytes(value: u64) -> String {
    let mut size = value as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if size < 1024.0 || unit == "TB" {
            if unit == "B" {
                return format!("{} {unit}", size as u64);
            }
            return format!("{size:.1} {unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

fn string_array_field(payload: &Value, field: &str) -> Vec<String> {
    payload
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn safe_download_dir(repo: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in repo.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            in_replacement = false;
        } else if !in_replacement {
            output.push_str("__");
            in_replacement = true;
        }
    }
    let output = output.trim_matches('_').to_owned();
    if output.is_empty() {
        "download".to_owned()
    } else {
        output
    }
}

fn slugify_lora_id(value: &str) -> String {
    let mut output = String::new();
    let mut previous_separator = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
            previous_separator = false;
        } else if !previous_separator && !output.is_empty() {
            output.push('_');
            previous_separator = true;
        }
    }
    while output.ends_with('_') {
        output.pop();
    }
    if output.is_empty() {
        "lora".to_owned()
    } else {
        output
    }
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .expect("setting nanoseconds to zero must be valid")
        .format(&Rfc3339)
        .expect("formatting a UTC timestamp as RFC3339 must succeed")
}

fn model_is_installed(path: &FsPath) -> bool {
    path.is_dir() && path.join(".sceneworks-download-complete.json").is_file()
}

fn lora_families(lora: &Value) -> Vec<String> {
    let compatibility = lora.get("compatibility").unwrap_or(&Value::Null);
    let values = lora
        .get("families")
        .or_else(|| lora.get("compatibleFamilies"))
        .or_else(|| lora.get("modelFamilies"))
        .or_else(|| compatibility.get("families"))
        .or_else(|| lora.get("family"));
    match values {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        Some(Value::String(value)) => vec![value.clone()],
        _ => Vec::new(),
    }
}

fn requested_gpu_or_auto(value: String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "auto".to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn option_str_is_empty(value: Option<&str>) -> bool {
    value.map(str::trim).unwrap_or_default().is_empty()
}

fn number_to_f64(number: &serde_json::Number, field: &'static str) -> Result<f64, ApiError> {
    number
        .as_f64()
        .ok_or_else(|| ApiError::bad_request(format!("Invalid numeric value for {field}")))
}

fn optional_number_to_f64(
    number: Option<&serde_json::Number>,
    field: &'static str,
) -> Result<Option<f64>, ApiError> {
    number.map(|value| number_to_f64(value, field)).transpose()
}

fn validate_timeline_export(payload: &TimelineExportRequest) -> Result<(), ApiError> {
    if ![640, 720, 1024, 1280].contains(&payload.resolution) {
        return Err(ApiError::bad_request(
            "Resolution must be one of 640, 720, 1024, or 1280.",
        ));
    }
    if !(1..=60).contains(&payload.fps) {
        return Err(ApiError::bad_request("FPS must be between 1 and 60"));
    }
    Ok(())
}

fn validate_frame_extract(payload: &FrameExtractRequest) -> Result<(), ApiError> {
    if !payload.playhead_seconds.is_finite() || payload.playhead_seconds < 0.0 {
        return Err(ApiError::bad_request(
            "playheadSeconds must be greater than or equal to 0",
        ));
    }
    if ![
        "reuse",
        "first_frame",
        "last_frame",
        "video_studio",
        "image_studio",
        "bridge",
        "extension",
    ]
    .contains(&payload.intended_use.as_str())
    {
        return Err(ApiError::bad_request("Unsupported intendedUse"));
    }
    Ok(())
}

fn validate_person_detection_job(payload: &PersonDetectionJobRequest) -> Result<(), ApiError> {
    if payload.source_asset_id.is_empty() {
        return Err(ApiError::bad_request("Source clip is required"));
    }
    if payload
        .source_timestamp
        .is_some_and(|timestamp| !timestamp.is_finite() || timestamp < 0.0)
    {
        return Err(ApiError::bad_request(
            "sourceTimestamp must be greater than or equal to 0",
        ));
    }
    Ok(())
}

fn validate_person_track_job(payload: &PersonTrackJobRequest) -> Result<(), ApiError> {
    if payload.source_asset_id.is_empty() {
        return Err(ApiError::bad_request("Source clip is required"));
    }
    if payload.representative_frame_asset_id.is_empty() {
        return Err(ApiError::bad_request(
            "Representative frame asset is required",
        ));
    }
    if payload.track_name.is_empty() || payload.track_name.chars().count() > 120 {
        return Err(ApiError::bad_request(
            "trackName must be between 1 and 120 characters",
        ));
    }
    if !payload.detection.contains_key("id") {
        return Err(ApiError::bad_request(
            "Selected detection metadata is required",
        ));
    }
    Ok(())
}

fn validate_image_job(payload: &ImageJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    if ![
        "text_to_image",
        "edit_image",
        "character_image",
        "style_variations",
    ]
    .contains(&payload.mode.as_str())
    {
        return Err(ApiError::bad_request("Unsupported image mode"));
    }
    if !(1..=8).contains(&payload.count) {
        return Err(ApiError::bad_request("count must be between 1 and 8"));
    }
    validate_dimension(payload.width, "width", 2048)?;
    validate_dimension(payload.height, "height", 2048)?;
    Ok(())
}

fn validate_character_test_job(payload: &CharacterTestRequest) -> Result<(), ApiError> {
    if payload.prompt.is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    if !(1..=8).contains(&payload.count) {
        return Err(ApiError::bad_request("count must be between 1 and 8"));
    }
    validate_dimension(payload.width, "width", 2048)?;
    validate_dimension(payload.height, "height", 2048)?;
    Ok(())
}

fn validate_video_job(payload: &VideoJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    if ![
        "image_to_video",
        "text_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "replace_person",
    ]
    .contains(&payload.mode.as_str())
    {
        return Err(ApiError::bad_request("Unsupported video mode"));
    }
    let duration = payload
        .duration
        .as_f64()
        .ok_or_else(|| ApiError::bad_request("duration must be a number between 1 and 30"))?;
    if !duration.is_finite() || !(1.0..=30.0).contains(&duration) {
        return Err(ApiError::bad_request("duration must be between 1 and 30"));
    }
    if !(1..=60).contains(&payload.fps) {
        return Err(ApiError::bad_request("fps must be between 1 and 60"));
    }
    validate_dimension(payload.width, "width", 1920)?;
    validate_dimension(payload.height, "height", 1920)?;
    match payload.mode.as_str() {
        "image_to_video" if payload.source_asset_id.is_none() => Err(ApiError::bad_request(
            "Image to Video requires a source image.",
        )),
        "first_last_frame"
            if payload.source_asset_id.is_none() || payload.last_frame_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "First/Last Frame requires first and last image assets.",
            ))
        }
        "extend_clip" if payload.source_clip_asset_id.is_none() => {
            Err(ApiError::bad_request("Extend Clip requires a source clip."))
        }
        "video_bridge"
            if payload.source_clip_asset_id.is_none()
                || payload.bridge_right_clip_asset_id.is_none() =>
        {
            Err(ApiError::bad_request(
                "Bridge generation requires left and right source clips.",
            ))
        }
        "replace_person" if payload.source_clip_asset_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a source clip.",
        )),
        "replace_person" if payload.person_track_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a selected person track.",
        )),
        "replace_person" if payload.character_id.is_none() => Err(ApiError::bad_request(
            "Replace Person requires a Character.",
        )),
        _ => Ok(()),
    }
}

fn validate_dimension(value: u32, field: &'static str, max: u32) -> Result<(), ApiError> {
    if !(256..=max).contains(&value) {
        return Err(ApiError::bad_request(format!(
            "{field} must be between 256 and {max}"
        )));
    }
    Ok(())
}

fn to_json_object<T: Serialize>(payload: &T) -> Result<JsonObject, ApiError> {
    serde_json::to_value(payload)
        .map_err(|error| ApiError::internal(error.to_string()))?
        .as_object()
        .cloned()
        .ok_or_else(|| ApiError::internal("Serialized payload was not an object"))
}

fn random_image_seeds(count: u32) -> Value {
    Value::Array(
        (0..count)
            .map(|_| {
                let bytes = *Uuid::new_v4().as_bytes();
                Value::Number(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).into())
            })
            .collect(),
    )
}

fn find_timeline_item<'a>(timeline: &'a Value, item_id: &str) -> Result<&'a Value, ApiError> {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(item_id))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Timeline item not found".to_owned(),
        })
}

fn source_timestamp_for_item(item: &Value, playhead_seconds: f64) -> Result<f64, ApiError> {
    let timeline_start = required_finite_f64_field(item, "timelineStart")?;
    let timeline_end = required_finite_f64_field(item, "timelineEnd")?;
    let source_in = required_finite_f64_field(item, "sourceIn")?;
    let speed = required_finite_f64_field(item, "speed")?;
    if timeline_end <= timeline_start {
        return Err(ApiError::bad_request(
            "timelineEnd must be greater than timelineStart.",
        ));
    }
    let clamped = playhead_seconds.clamp(timeline_start, timeline_end);
    Ok(source_in + ((clamped - timeline_start) * speed))
}

fn required_string_field<'a>(payload: &'a Value, field: &str) -> Result<&'a str, ApiError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request(format!("Missing required field: {field}")))
}

fn optional_f64_field(payload: &Value, field: &str) -> Option<f64> {
    payload.get(field).and_then(Value::as_f64)
}

fn required_finite_f64_field(payload: &Value, field: &str) -> Result<f64, ApiError> {
    let value = optional_f64_field(payload, field)
        .ok_or_else(|| ApiError::bad_request(format!("Missing required field: {field}")))?;
    if !value.is_finite() {
        return Err(ApiError::bad_request(format!(
            "Invalid numeric value for {field}"
        )));
    }
    Ok(value)
}

fn default_timeline_name() -> String {
    "Main timeline".to_owned()
}

fn default_aspect_ratio() -> String {
    "16:9".to_owned()
}

fn default_timeline_fps() -> u32 {
    30
}

fn default_export_resolution() -> u32 {
    720
}

fn default_frame_intended_use() -> String {
    "reuse".to_owned()
}

fn default_requested_gpu() -> String {
    "auto".to_owned()
}

fn default_lora_scope() -> String {
    "global".to_owned()
}

fn default_project_lora_scope() -> String {
    "project".to_owned()
}

fn default_character_type() -> String {
    "person".to_owned()
}

fn default_reference_role() -> String {
    "reference".to_owned()
}

fn default_character_lora_weight() -> f64 {
    0.8
}

fn default_track_name() -> String {
    "Selected person".to_owned()
}

fn default_image_mode() -> String {
    "text_to_image".to_owned()
}

fn default_image_model() -> String {
    "z_image_turbo".to_owned()
}

fn default_image_count() -> u32 {
    4
}

fn default_image_size() -> u32 {
    1024
}

fn default_style_preset() -> String {
    "cinematic".to_owned()
}

fn default_video_mode() -> String {
    "image_to_video".to_owned()
}

fn default_video_model() -> String {
    "ltx_2_3".to_owned()
}

fn default_video_duration() -> ContractNumber {
    ContractNumber::from(6)
}

fn default_video_fps() -> u32 {
    25
}

fn default_video_width() -> u32 {
    768
}

fn default_video_height() -> u32 {
    512
}

fn default_video_quality() -> String {
    "balanced".to_owned()
}

fn default_replacement_mode() -> String {
    "face_only".to_owned()
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn env_path(name: &str, default: &str) -> PathBuf {
    PathBuf::from(env_string(name, default))
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    detail: String,
}

impl ApiError {
    fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
        }
    }

    fn unauthorized(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            detail: detail.into(),
        }
    }

    fn payload_too_large(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            detail: detail.into(),
        }
    }

    fn internal(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            detail: detail.into(),
        }
    }
}

impl From<JobsStoreError> for ApiError {
    fn from(error: JobsStoreError) -> Self {
        match error {
            JobsStoreError::NotFound(_) => Self {
                status: StatusCode::NOT_FOUND,
                detail: "Record not found".to_owned(),
            },
            JobsStoreError::InvalidStatus(status) => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Unsupported job status: {status}"),
            },
            JobsStoreError::InvalidNumber(field) => {
                Self::bad_request(format!("Invalid numeric value for {field}"))
            }
            JobsStoreError::RetryLimit { max_attempts } => Self {
                status: StatusCode::BAD_REQUEST,
                detail: format!("Job retry limit reached after {max_attempts} attempts."),
            },
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<ProjectStoreError> for ApiError {
    fn from(error: ProjectStoreError) -> Self {
        match error {
            ProjectStoreError::BadRequest(detail) => Self::bad_request(detail),
            ProjectStoreError::NotFound(detail) => Self {
                status: StatusCode::NOT_FOUND,
                detail,
            },
            other => Self::internal(other.to_string()),
        }
    }
}

fn prune_tickets(tickets: &mut HashMap<String, Instant>, now: Instant) {
    tickets.retain(|_, expires_at| *expires_at >= now);
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        create_app, EventHub, EventMessage, Settings, EVENT_BUFFER_SIZE, HEARTBEAT_SSE_DATA,
        HEARTBEAT_SSE_WIRE,
    };
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tokio_stream::StreamExt;
    use tower::ServiceExt;

    fn test_settings(temp_dir: &tempfile::TempDir) -> Settings {
        Settings {
            api_runtime: "rust".to_owned(),
            app_version: "test".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 0,
            data_dir: temp_dir.path().join("data"),
            config_dir: temp_dir.path().join("config"),
            access_token: String::new(),
            cors_origins: vec![
                "http://localhost:5173".to_owned(),
                "http://127.0.0.1:5173".to_owned(),
            ],
            worker_timeout_seconds: 90,
            jobs_db_path: temp_dir.path().join("jobs.db"),
        }
    }

    async fn request(
        app: axum::Router,
        method: &str,
        uri: &str,
        body: Value,
    ) -> (StatusCode, Value) {
        request_with_headers(app, method, uri, body, &[]).await
    }

    async fn request_with_headers(
        app: axum::Router,
        method: &str,
        uri: &str,
        body: Value,
        headers: &[(&str, &str)],
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let request = builder
            .body(Body::from(body.to_string()))
            .expect("request builds");
        let response = app.oneshot(request).await.expect("response returns");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body buffers");
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).expect("json body parses")
        };
        (status, value)
    }

    async fn request_raw(
        app: axum::Router,
        method: &str,
        uri: &str,
        body: impl Into<Body>,
        headers: &[(&str, &str)],
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let mut builder = Request::builder().method(method).uri(uri);
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let response = app
            .oneshot(builder.body(body.into()).expect("request builds"))
            .await
            .expect("response returns");
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body buffers")
            .to_vec();
        (status, headers, bytes)
    }

    async fn request_multipart_upload(
        app: axum::Router,
        uri: &str,
        filename: &str,
        content_type: &str,
        bytes: &[u8],
    ) -> (StatusCode, Value) {
        let boundary = "SCENEWORKS_BOUNDARY";
        let mut body = Vec::new();
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let (status, _, bytes) = request_raw(
            app,
            "POST",
            uri,
            body,
            &[(
                "content-type",
                &format!("multipart/form-data; boundary={boundary}"),
            )],
        )
        .await;
        let value = serde_json::from_slice(&bytes).expect("json body parses");
        (status, value)
    }

    #[tokio::test]
    async fn worker_can_register_claim_and_complete_job_through_http() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/workers/register",
            json!({
                "workerId": "worker-1",
                "gpuId": "gpu-0",
                "gpuName": "GPU 0",
                "capabilities": ["image_generate"],
                "loadedModels": []
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (status, created) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "image_generate",
                "projectId": "project-1",
                "projectName": "Project 1",
                "payload": { "prompt": "mist over hills" },
                "requestedGpu": "auto"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let (status, claimed) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs/claim",
            json!({ "workerId": "worker-1" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(claimed["job"]["id"], created["id"]);
        assert_eq!(claimed["job"]["status"], "preparing");

        let job_id = created["id"].as_str().expect("job id is string");
        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Done",
                "result": { "assetIds": ["asset-1"] }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(completed["status"], "completed");
        assert_eq!(completed["result"], json!({ "assetIds": ["asset-1"] }));

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["completed"], 1);
        assert_eq!(queue["workers"][0]["status"], "idle");
    }

    #[tokio::test]
    async fn project_and_asset_routes_persist_python_compatible_state() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, created) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "My Project" }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert!(created["id"]
            .as_str()
            .is_some_and(|value| value.starts_with("project_")));
        assert!(created["path"]
            .as_str()
            .unwrap()
            .ends_with("my-project.sceneworks"));

        let project_id = created["id"].as_str().expect("project id").to_owned();
        let (status, projects) = request(app.clone(), "GET", "/api/v1/projects", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(projects[0]["id"], project_id);

        let (status, uploaded) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Hero Image.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(uploaded["projectId"], project_id);
        assert_eq!(uploaded["type"], "image");
        assert_eq!(uploaded["status"]["trashed"], false);
        assert!(uploaded["url"]
            .as_str()
            .unwrap()
            .contains("/files/assets/uploads/"));

        let (status, heic_upload) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Plate.HEIC",
            "application/octet-stream",
            b"heic-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(heic_upload["type"], "image");
        assert_eq!(heic_upload["file"]["mimeType"], "image/heic");

        let asset_id = uploaded["id"].as_str().expect("asset id").to_owned();
        let (status, assets) = request(
            app.clone(),
            "GET",
            &format!(
                "/api/v1/projects/{project_id}/assets?includeRejected=true&includeTrashed=true"
            ),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(assets.as_array().unwrap().len(), 2);

        let (status, detail) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(detail["id"], asset_id);

        let (status, updated) = request(
            app.clone(),
            "PATCH",
            &format!("/api/v1/projects/{project_id}/assets/{asset_id}/status"),
            json!({ "favorite": true, "rating": 4, "rejected": true }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["status"]["favorite"], true);
        assert_eq!(updated["status"]["rating"], 4);
        assert_eq!(updated["status"]["rejected"], true);

        let (status, deleted) = request(
            app.clone(),
            "DELETE",
            &format!("/api/v1/projects/{project_id}/assets/{asset_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted, json!({ "id": asset_id, "status": "trashed" }));

        let (status, reindex) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/reindex"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reindex["assets"], 2);

        let (status, purged) = request(
            app,
            "DELETE",
            &format!("/api/v1/projects/{project_id}/assets/{asset_id}/purge"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(purged, json!({ "id": asset_id, "status": "purged" }));
    }

    #[tokio::test]
    async fn timeline_routes_persist_and_create_worker_jobs() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, created_project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Timeline Project" }),
        )
        .await;
        let project_id = created_project["id"]
            .as_str()
            .expect("project id")
            .to_owned();

        let (status, mut timeline) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines"),
            json!({ "name": "Main timeline", "aspectRatio": "16:9", "fps": 30 }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(timeline["projectId"], project_id);
        assert_eq!(timeline["tracks"].as_array().unwrap().len(), 3);

        let timeline_id = timeline["id"].as_str().expect("timeline id").to_owned();
        timeline["tracks"][0]["items"] = json!([
            {
                "id": "item-1",
                "trackId": "track_main",
                "assetId": "asset-1",
                "type": "video",
                "displayName": "Clip",
                "sourceIn": 2,
                "sourceOut": 6,
                "timelineStart": 10,
                "timelineEnd": 14,
                "speed": 1,
                "fit": "fit",
                "volume": 1
            }
        ]);
        let (status, saved) = request(
            app.clone(),
            "PUT",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            json!({ "timeline": timeline }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(saved["duration"].as_f64(), Some(14.0));
        assert_eq!(
            saved["tracks"][0]["items"][0]["currentVersionAssetId"],
            "asset-1"
        );
        assert_eq!(
            saved["tracks"][0]["items"][0]["versionHistory"][0]["source"],
            "original"
        );

        let (status, timelines) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/timelines"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(timelines[0]["id"], timeline_id);
        assert_eq!(
            timelines[0]["filePath"],
            format!(
                "timelines/main-timeline-{}.sceneworks.timeline.json",
                &timeline_id[timeline_id.len() - 8..]
            )
        );

        let (status, export_job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/exports"),
            json!({ "resolution": 720, "fps": 30, "requestedGpu": "auto" }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(export_job["type"], "timeline_export");
        assert_eq!(export_job["payload"]["timelineId"], timeline_id);

        let (status, frame_job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}/items/item-1/frames"),
            json!({ "playheadSeconds": 12.5, "intendedUse": "first_frame" }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(frame_job["type"], "frame_extract");
        assert_eq!(frame_job["payload"]["sourceAssetId"], "asset-1");
        assert_eq!(frame_job["payload"]["sourceTimestamp"], 4.5);

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["queued"], 2);
    }

    #[tokio::test]
    async fn image_and_video_job_routes_normalize_payloads() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, image_job) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": "project-1",
                "projectName": "Project 1",
                "mode": "text_to_image",
                "prompt": "mist over hills",
                "count": 2
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(image_job["type"], "image_generate");
        assert_eq!(image_job["projectId"], "project-1");
        assert!(image_job["payload"].get("requestedGpu").is_none());
        assert_eq!(image_job["payload"]["seed"], Value::Null);
        assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 2);

        let (status, edit_job) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": "project-1",
                "mode": "edit_image",
                "prompt": "make it dusk",
                "sourceAssetId": "asset-1",
                "seed": 42
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(edit_job["type"], "image_edit");
        assert!(edit_job["payload"].get("seeds").is_none());

        let (status, wide_seed_job) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": " ",
                "mode": "text_to_image",
                "prompt": "space project id stays Python-compatible",
                "seed": -42
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(wide_seed_job["payload"]["projectId"], " ");
        assert_eq!(wide_seed_job["payload"]["seed"], -42);

        let (status, video_job) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "mode": "replace_person",
                "prompt": "hero walks through rain",
                "sourceClipAssetId": "asset-video",
                "personTrackId": "track-1",
                "characterId": "character-1"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(video_job["type"], "person_replace");
        assert!(video_job["payload"].get("requestedGpu").is_none());

        let (status, integer_duration_job) = request(
            app.clone(),
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "integer duration stays an integer",
                "duration": 6
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(integer_duration_job["payload"]["duration"], 6);

        let (status, queue) = request(app, "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["queued"], 5);
    }

    #[tokio::test]
    async fn person_tracking_routes_match_python_contracts() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Tracking Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();
        let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
        std::fs::write(
            project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
            serde_json::to_string_pretty(&json!({
                "schemaVersion": 1,
                "id": "track_1",
                "projectId": project_id,
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

        let (status, tracks) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/person-tracks"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(tracks[0]["id"], "track_1");
        assert_eq!(
            tracks[0]["path"],
            "person-tracks/track_1.sceneworks.person-track.json"
        );

        let (status, track) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/person-tracks/track_1"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(track["name"], "Hero");

        let (status, detection_job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/person-tracks/detections"),
            json!({ "sourceAssetId": "asset-video", "sourceTimestamp": 1.25 }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(detection_job["type"], "person_detect");
        assert_eq!(detection_job["payload"]["sourceTimestamp"], 1.25);
        assert!(detection_job["projectName"]
            .as_str()
            .is_some_and(|value| value.starts_with("tracking")));

        let detection = json!({
            "id": "person_1",
            "box": { "x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6 }
        });
        let (status, track_job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/person-tracks/jobs"),
            json!({
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "detection": detection,
                "trackName": "Hero"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(track_job["type"], "person_track");
        assert_eq!(track_job["payload"]["trackName"], "Hero");

        for invalid_path in [
            format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E"),
            format!("/api/v1/projects/{project_id}/person-tracks/%2E%2E%2Fescape"),
            format!("/api/v1/projects/{project_id}/person-tracks/track~bad"),
        ] {
            let (status, error) = request(app.clone(), "GET", &invalid_path, Value::Null).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(error["detail"], "Invalid person track ID");
        }

        let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["queued"], 2);
    }

    #[tokio::test]
    async fn model_and_lora_routes_match_python_manifest_behavior() {
        std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "base-model",
                  "name": "Base Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [{ "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"] }],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": { "label": "Base" }
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                { "id": "base-model", "name": "User Model", "ui": { "label": "User" } }
              ]
            }
            "#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style-lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": ["style"],
                  "compatibility": { "families": ["z-image", "wan-video"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                }
              ]
            }
            "#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "cinematic",
                  "name": "Cinematic",
                  "model": "preset-model",
                  "defaults": { "count": 4, "resolution": "1280x720", "negativePrompt": "flat lighting" },
                  "prompt": { "suffix": "cinematic lighting" },
                  "builtInLoras": [{ "id": "style-lora", "weight": 0.5 }]
                }
              ]
            }
            "#,
        )
        .expect("builtin recipe presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                { "id": "cinematic", "name": "My Cinematic", "defaults": { "count": 2, "resolution": "1280x720", "negativePrompt": "flat lighting" } }
              ]
            }
            "#,
        )
        .expect("user recipe presets writes");
        let marker_dir = temp_dir.path().join("data/models/owner__model");
        std::fs::create_dir_all(&marker_dir).expect("model dir creates");
        std::fs::write(marker_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("marker writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(models[0]["name"], "User Model");
        assert_eq!(models[0]["adapter"], "z_image_diffusers");
        assert_eq!(models[0]["downloadable"], true);
        assert_eq!(models[0]["installState"], "installed");
        assert!(models[0]["installedPath"]
            .as_str()
            .is_some_and(|value| value.ends_with("owner__model")));

        let (status, loras) = request(
            app.clone(),
            "GET",
            "/api/v1/loras?modelFamily=wan-video",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(loras.as_array().unwrap().len(), 1);
        assert_eq!(loras[0]["id"], "style-lora");

        let (status, presets) =
            request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(presets.as_array().unwrap().len(), 1);
        assert_eq!(presets[0]["id"], "cinematic");
        assert_eq!(presets[0]["name"], "My Cinematic");
        assert_eq!(presets[0]["scope"], "global");
        assert_eq!(presets[0]["defaults"]["count"], 2);
        assert_eq!(presets[0]["builtInLoras"][0]["id"], "style-lora");

        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Preset Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id");
        let (status, image_job) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "prompt": "city at night",
                "model": "client-model",
                "count": 1,
                "width": 512,
                "height": 512,
                "negativePrompt": "client negative prompt",
                "recipePresetId": "cinematic"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(
            image_job["payload"]["prompt"],
            "city at night, cinematic lighting"
        );
        assert_eq!(image_job["payload"]["loras"][0]["id"], "style-lora");
        assert_eq!(image_job["payload"]["model"], "preset-model");
        assert_eq!(image_job["payload"]["count"], 2);
        assert_eq!(image_job["payload"]["seeds"].as_array().unwrap().len(), 2);
        assert_eq!(image_job["payload"]["width"], 1280);
        assert_eq!(image_job["payload"]["height"], 720);
        assert_eq!(image_job["payload"]["negativePrompt"], "flat lighting");
        assert_eq!(image_job["payload"]["advanced"]["resolution"], "1280x720");
        assert_eq!(
            image_job["payload"]["advanced"]["recipePresetId"],
            "cinematic"
        );

        let (status, job) = request(
            app.clone(),
            "POST",
            "/api/v1/models/base-model/download",
            json!({ "requestedGpu": "" }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(job["type"], "model_download");
        assert_eq!(job["requestedGpu"], "auto");
        assert_eq!(job["payload"]["modelName"], "User Model");
        assert_eq!(job["payload"]["targetDir"], models[0]["installedPath"]);

        let (status, job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({ "repo": "owner/lora", "name": "Imported LoRA", "files": ["adapter.safetensors"] }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(job["type"], "lora_import");
        assert_eq!(job["payload"]["repo"], "owner/lora");
        assert_eq!(job["payload"]["loraId"], "imported_lora");
        assert_eq!(job["payload"]["scope"], "global");
        assert!(job["payload"]["targetDir"]
            .as_str()
            .is_some_and(|value| value.ends_with("data/loras/imported_lora")
                || value.ends_with("data\\loras\\imported_lora")));
        assert_eq!(job["payload"]["manifestEntry"]["scope"], "global");
        assert!(job["payload"].get("sourcePath").is_none());
    }

    #[test]
    fn model_download_size_helpers_match_python_shapes() {
        let siblings = json!([
            { "rfilename": "model-00001.safetensors", "size": 100 },
            { "rfilename": "model-00002.safetensors", "size": "200" },
            { "rfilename": "README.md", "size": 50 },
            { "rfilename": "unknown.bin" }
        ]);
        let siblings = siblings.as_array().expect("siblings array");
        assert_eq!(
            super::download_size_from_siblings(siblings, &["*.safetensors".to_owned()]),
            Some(300)
        );
        assert_eq!(
            super::download_size_from_siblings(siblings, &["*.ckpt".to_owned()]),
            None
        );
        assert_eq!(super::json_size_to_u64(&json!("200.5")), None);
        assert_eq!(super::format_bytes(0), "0 B");
        assert_eq!(super::format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(
            super::quote_huggingface_repo("owner/model name"),
            "owner/model%20name"
        );
        assert!(super::model_download(&json!({
            "downloads": [{ "repo": "owner/model" }]
        }))
        .is_none());
        let mut cache = super::ModelSizeCache::default();
        let key = ("owner/model".to_owned(), vec!["*.safetensors".to_owned()]);
        cache.insert(key.clone(), 300);
        assert_eq!(cache.get(&key), Some(300));
        assert!(super::allow_pattern_matches(
            "model-7.safetensors",
            &["model-[0-9].safetensors".to_owned()]
        ));
        if cfg!(windows) {
            assert!(super::allow_pattern_matches(
                "Model.SAFETENSORS",
                &["*.safetensors".to_owned()]
            ));
        }
    }

    #[test]
    fn lora_family_filter_shapes_match_python_fallbacks() {
        let shapes = [
            json!({ "families": ["z-image"] }),
            json!({ "compatibleFamilies": ["z-image"] }),
            json!({ "modelFamilies": ["z-image"] }),
            json!({ "compatibility": { "families": ["z-image"] } }),
            json!({ "family": "z-image" }),
        ];
        for lora in shapes {
            assert_eq!(super::lora_families(&lora), vec!["z-image".to_owned()]);
        }
    }

    #[tokio::test]
    async fn malformed_manifest_returns_stable_server_error() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"{ "models": [ /*"#,
        )
        .expect("manifest writes");
        std::fs::write(config_dir.join("user.models.jsonc"), r#"{ "models": [] }"#)
            .expect("manifest writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, error) = request(app, "GET", "/api/v1/models", Value::Null).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error["detail"], "Failed to load manifest");
    }

    #[tokio::test]
    async fn generation_routes_reject_invalid_payloads() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({ "projectId": "project-1", "prompt": "x".repeat(4001) }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = request(
            app,
            "POST",
            "/api/v1/video/jobs",
            json!({
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "missing source image"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn timeline_routes_reject_invalid_payloads() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, created_project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Invalid Timeline Project" }),
        )
        .await;
        let project_id = created_project["id"]
            .as_str()
            .expect("project id")
            .to_owned();

        let (status, _) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines"),
            json!({ "name": "Main timeline", "aspectRatio": "4:3", "fps": 30 }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (_, mut timeline) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/timelines"),
            json!({ "name": "Main timeline" }),
        )
        .await;
        let timeline_id = timeline["id"].as_str().expect("timeline id").to_owned();
        timeline["tracks"][0]["items"] = json!([
            {
                "id": "item-1",
                "trackId": "track_main",
                "assetId": "asset-1",
                "type": "video",
                "displayName": "Clip",
                "sourceIn": 4,
                "sourceOut": 2,
                "timelineStart": 0,
                "timelineEnd": 4
            }
        ]);
        let (status, _) = request(
            app.clone(),
            "PUT",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            json!({ "timeline": timeline.clone() }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        timeline["tracks"][0]["items"][0]["sourceOut"] = json!(6);
        timeline["tracks"][0]["kind"] = json!("audio_v2");
        let (status, _) = request(
            app,
            "PUT",
            &format!("/api/v1/projects/{project_id}/timelines/{timeline_id}"),
            json!({ "timeline": timeline }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn frame_extract_rejects_non_finite_playhead() {
        let result = super::validate_frame_extract(&super::FrameExtractRequest {
            playhead_seconds: f64::NAN,
            intended_use: "reuse".to_owned(),
            requested_gpu: "auto".to_owned(),
        });

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn project_file_route_serves_files_and_rejects_traversal() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, created) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Files" }),
        )
        .await;
        let project_id = created["id"].as_str().expect("project id").to_owned();
        let project_path = std::path::PathBuf::from(created["path"].as_str().unwrap());
        let media_path = project_path.join("assets/images/image.png");
        std::fs::write(&media_path, b"image-bytes").expect("media writes");
        let outside_path = temp_dir.path().join("data").join("outside.txt");
        std::fs::write(outside_path, b"nope").expect("outside writes");

        let (status, headers, bytes) = request_raw(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/files/assets/images/image.png"),
            Body::empty(),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bytes, b"image-bytes");
        assert_eq!(
            headers
                .get("content-type")
                .and_then(|value| value.to_str().ok()),
            Some("image/png")
        );

        let (status, _, bytes) = request_raw(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/files/%2E%2E%2F%2E%2E%2Foutside.txt"),
            Body::empty(),
            &[],
        )
        .await;
        let error: Value = serde_json::from_slice(&bytes).expect("json error parses");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Invalid project file path");

        let (status, _, bytes) = request_raw(
            app,
            "GET",
            &format!("/api/v1/projects/{project_id}/files/%2E%2E%5C%2E%2E%5Coutside.txt"),
            Body::empty(),
            &[],
        )
        .await;
        let error: Value = serde_json::from_slice(&bytes).expect("json error parses");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Invalid project file path");
    }

    #[tokio::test]
    async fn character_studio_routes_manage_references_loras_and_test_jobs() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let data_dir = settings.data_dir.clone();
        let app = create_app(settings).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Characters" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();
        let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

        let (status, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "reference.png",
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        let (status, character) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters"),
            json!({ "name": "Mira", "type": "person" }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(character["name"], "Mira");
        let character_id = character["id"].as_str().expect("character id").to_owned();

        let (status, with_reference) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/references"),
            json!({ "assetId": asset_id, "approved": false }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(
            with_reference["references"][0]["asset"]["displayName"],
            "reference.png"
        );

        let (status, updated) = request(
            app.clone(),
            "PATCH",
            &format!(
                "/api/v1/projects/{project_id}/characters/{character_id}/references/{asset_id}"
            ),
            json!({ "approved": true, "role": "hero" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["approvedReferences"][0]["assetId"], asset_id);

        let sidecar_path = project_path.join(
            asset["sidecarPath"]
                .as_str()
                .expect("asset sidecar path")
                .replace('/', std::path::MAIN_SEPARATOR_STR),
        );
        let asset_sidecar: Value = serde_json::from_str(
            &std::fs::read_to_string(sidecar_path).expect("asset sidecar reads"),
        )
        .expect("asset sidecar parses");
        assert_eq!(
            asset_sidecar["metadata"]["characterReferences"][0]["characterId"],
            character_id
        );
        assert_eq!(
            asset_sidecar["metadata"]["characterReferences"][0]["approved"],
            true
        );

        let (status, with_look) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/looks"),
            json!({ "name": "Rain coat", "approvedReferenceIds": [asset_id], "recipeSettings": { "style": "noir" } }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(with_look["looks"][0]["recipeSettings"]["style"], "noir");
        let look_id = with_look["looks"][0]["id"]
            .as_str()
            .expect("look id")
            .to_owned();

        let lora_dir = data_dir.join("loras");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        let lora_source = lora_dir.join("mira.safetensors");
        std::fs::write(&lora_source, b"lora").expect("lora writes");
        let (status, with_lora) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
            json!({
                "name": "Mira LoRA",
                "sourcePath": lora_source.display().to_string(),
                "compatibility": { "families": ["sdxl"] },
                "triggerWords": ["mira"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(with_lora["loras"][0]["copiedIntoProject"], true);
        let project_lora_path = project_path.join(
            with_lora["loras"][0]["projectPath"]
                .as_str()
                .expect("project lora path")
                .replace('/', std::path::MAIN_SEPARATOR_STR),
        );
        assert_eq!(
            std::fs::read(project_lora_path).expect("lora copied"),
            b"lora"
        );

        let (status, test_job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
            json!({ "prompt": "portrait", "lookId": look_id, "count": 2 }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(test_job["type"], "image_generate");
        assert_eq!(test_job["payload"]["mode"], "character_image");
        assert_eq!(test_job["payload"]["characterId"], character_id);
        assert_eq!(
            test_job["payload"]["advanced"]["approvedReferenceIds"][0],
            asset_id
        );

        let (status, _) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/archive"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, visible) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/characters"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(visible.as_array().unwrap().len(), 0);
        let (status, archived) = request(
            app,
            "GET",
            &format!("/api/v1/projects/{project_id}/characters?includeArchived=true"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(archived.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn worker_heartbeat_interrupts_previous_active_job_through_http() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        request(
            app.clone(),
            "POST",
            "/api/v1/workers/register",
            json!({
                "workerId": "worker-1",
                "gpuId": "gpu-0",
                "gpuName": null,
                "capabilities": ["image_generate"],
                "loadedModels": []
            }),
        )
        .await;
        let (_, created) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({ "type": "image_generate", "payload": {}, "requestedGpu": "auto" }),
        )
        .await;
        request(
            app.clone(),
            "POST",
            "/api/v1/jobs/claim",
            json!({ "workerId": "worker-1" }),
        )
        .await;

        let (status, worker) = request(
            app.clone(),
            "POST",
            "/api/v1/workers/worker-1/heartbeat",
            json!({ "status": "idle", "currentJobId": null, "loadedModels": [] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(worker["currentJobId"], Value::Null);

        let job_id = created["id"].as_str().expect("job id is string");
        let (status, job) =
            request(app, "GET", &format!("/api/v1/jobs/{job_id}"), Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(job["status"], "interrupted");
        assert_eq!(job["workerId"], Value::Null);
    }

    #[tokio::test]
    async fn access_token_is_enforced_on_protected_routes() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        let (status, access) = request(app.clone(), "GET", "/api/v1/access", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(access["authRequired"], true);

        let (status, error) = request(app.clone(), "GET", "/api/v1/jobs", Value::Null).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(error["detail"], "SceneWorks access token required");

        let (status, jobs) = request_with_headers(
            app,
            "GET",
            "/api/v1/jobs",
            Value::Null,
            &[("x-sceneworks-token", "secret-token")],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(jobs, json!([]));
    }

    #[tokio::test]
    async fn bearer_token_is_accepted_for_access_verification() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        let (status, verified) = request_with_headers(
            app,
            "POST",
            "/api/v1/auth/verify",
            Value::Null,
            &[("authorization", "Bearer secret-token")],
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(verified["ok"], true);
    }

    #[tokio::test]
    async fn event_tickets_are_protected_and_match_python_shape() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs/events/ticket",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(error["detail"], "SceneWorks access token required");

        let (status, ticket) = request_with_headers(
            app.clone(),
            "POST",
            "/api/v1/jobs/events/ticket",
            Value::Null,
            &[("x-sceneworks-token", "secret-token")],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            ticket["ticket"].as_str().is_some_and(
                |value| value.len() == 32 && value.chars().all(|c| c.is_ascii_hexdigit())
            )
        );
        assert_eq!(ticket["expiresInSeconds"], 30);

        let (status, error) = request(
            app,
            "GET",
            "/api/v1/jobs/events?ticket=missing",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(error["detail"], "Invalid or expired event stream ticket");
    }

    #[tokio::test]
    async fn lagged_event_subscribers_are_disconnected() {
        let hub = EventHub::default();
        let mut stream = hub.subscribe();

        for index in 0..EVENT_BUFFER_SIZE {
            hub.publish(EventMessage {
                event: "job.updated".to_owned(),
                data: json!({ "index": index }).to_string(),
            });
        }
        hub.publish(EventMessage {
            event: "job.updated".to_owned(),
            data: json!({ "index": EVENT_BUFFER_SIZE }).to_string(),
        });

        for _ in 0..EVENT_BUFFER_SIZE {
            assert!(stream.next().await.is_some());
        }
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn heartbeat_event_matches_python_wire_shape() {
        assert_eq!(HEARTBEAT_SSE_DATA, "{}");
        assert_eq!(HEARTBEAT_SSE_WIRE, "event: heartbeat\ndata: {}\n\n");
    }

    #[tokio::test]
    async fn cors_preflight_allows_frontend_origin_and_token_header() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let request = Request::builder()
            .method("OPTIONS")
            .uri("/api/v1/jobs")
            .header("origin", "http://localhost:5173")
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "X-SceneWorks-Token")
            .body(Body::empty())
            .expect("request builds");

        let response = app.oneshot(request).await.expect("response returns");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("http://localhost:5173")
        );
        assert!(response
            .headers()
            .get("access-control-allow-headers")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.to_ascii_lowercase().contains("x-sceneworks-token")));
    }
}
