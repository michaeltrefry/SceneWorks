use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    JobSnapshot, JobStatus, JobType, JsonObject, ProgressRequest, QueueSummary, WorkerCapability,
    WorkerHeartbeatRequest, WorkerRegisterRequest, WorkerSnapshot, WorkerStatus,
};
use sceneworks_core::jobs_store::{
    CreateJob, DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker,
    WorkerHeartbeat, JOB_STATUSES,
};
use sceneworks_core::lora_family::{
    apply_model_manifest_defaults, detect_lora_family, detect_model_family, first_safetensors_path,
    read_safetensors_header, reconcile_detected_family, SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{lora_source_url_file_stem, parse_lora_source_url, LoraUrlError};
use sceneworks_core::project_store::{
    AssetStatusPatch, CharacterCreateInput, CharacterLookInput, CharacterLookUpdateInput,
    CharacterLoraInput, CharacterLoraUpdateInput, CharacterReferenceInput,
    CharacterReferenceUpdateInput, CharacterUpdateInput, ProjectStore, ProjectStoreError,
    UploadAsset,
};
use sceneworks_core::training::{
    build_training_plan, builtin_training_targets, BuildTrainingPlan, LoraTrainingRequest,
    TrainingDataset, TrainingTarget, TrainingTargetRegistry,
};
use sceneworks_core::training_store::{
    TrainingCaptionSidecarsResult, TrainingDatasetBatchRenameInput,
    TrainingDatasetCaptionSidecarsInput, TrainingDatasetCreateInput, TrainingDatasetMutationResult,
    TrainingDatasetSummary, TrainingDatasetUpdateInput,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, Mutex as AsyncMutex};
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
const MAX_MODEL_UPLOAD_BYTES: usize = 256 * 1024 * 1024 * 1024;
const MAX_LORA_MULTIPART_BODY_BYTES: usize = MAX_UPLOAD_BYTES + 16 * 1024 * 1024;
const MAX_MODEL_MULTIPART_BODY_BYTES: usize = MAX_MODEL_UPLOAD_BYTES + 16 * 1024 * 1024;
const STALE_LORA_UPLOAD_SECONDS: u64 = 24 * 60 * 60;
const MANIFEST_CACHE_LIMIT: usize = 16;
const MODEL_SIZE_CACHE_LIMIT: usize = 64;
const API_MANAGED_MANIFEST_HEADER: &str = "// This file is rewritten by the SceneWorks API. Inline JSONC comments are not preserved across writes.";
#[cfg(test)]
static TEST_MAX_LORA_UPLOAD_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_MAX_MODEL_UPLOAD_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Settings {
    pub app_version: String,
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub access_token: String,
    pub cors_origins: Vec<String>,
    pub worker_timeout_seconds: u64,
    pub jobs_db_path: PathBuf,
    pub run_utility_inprocess: bool,
}

impl Settings {
    pub fn from_env() -> Self {
        let defaults = sceneworks_core::app_paths::AppPaths::platform_default();
        let data_dir = env_path_or("SCENEWORKS_DATA_DIR", &defaults.data_dir);
        let jobs_db_path = std::env::var("SCENEWORKS_JOBS_DB_PATH")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("cache").join("jobs.db"));
        Self {
            app_version: env_string("SCENEWORKS_APP_VERSION", "0.2.0"),
            host: env_string("SCENEWORKS_API_HOST", "0.0.0.0"),
            port: std::env::var("SCENEWORKS_API_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8000),
            data_dir,
            config_dir: env_path_or("SCENEWORKS_CONFIG_DIR", &defaults.config_dir),
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
            run_utility_inprocess: std::env::var("SCENEWORKS_RUN_UTILITY_INPROCESS")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
                .unwrap_or(false),
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
    manifest_write_locks: Arc<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>>,
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
    let run_utility_inprocess = settings.run_utility_inprocess;
    let app = create_app(settings)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    // Use the actual bound address so port 0 (OS-assigned) is reported and the
    // in-process worker connects to the real port.
    let bound = listener.local_addr()?;
    let port = bound.port();
    println!("SceneWorks Rust API listening on http://{bound}");

    let utility_worker = run_utility_inprocess.then(|| spawn_inprocess_utility_worker(port));

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    if let Some(worker) = utility_worker {
        worker.shutdown().await;
    }
    Ok(())
}

/// Spawns the utility worker loop ([`sceneworks_worker::run_worker_loop`]) as a
/// tokio task in this process, pointed at the local API over loopback. The loop
/// observes the same Ctrl+C/SIGTERM as the HTTP server (via the worker's own
/// shutdown handling), so `shutdown()` only bounds the wait by the worker's
/// configured grace period.
fn spawn_inprocess_utility_worker(port: u16) -> InProcessUtilityWorker {
    let mut worker_settings = sceneworks_worker::Settings::from_env();
    worker_settings.api_url = format!("http://127.0.0.1:{port}");
    worker_settings.gpu_id =
        inprocess_worker_gpu_id(std::env::var("SCENEWORKS_RUST_WORKER_GPU_ID").ok());
    let grace = Duration::from_secs(worker_settings.shutdown_timeout_seconds.max(1));
    println!(
        "SceneWorks utility worker running in-process (loopback {})",
        worker_settings.api_url
    );
    let handle =
        tokio::spawn(async move { sceneworks_worker::run_worker_loop(worker_settings).await });
    InProcessUtilityWorker { handle, grace }
}

/// GPU id for the in-process utility worker. Defaults to `cpu` so the embedded
/// worker advertises CPU utility capabilities (downloads, imports, ffmpeg,
/// person detect/track) regardless of the ambient `SCENEWORKS_GPU_ID` — which on
/// a GPU host would otherwise make it register as a GPU worker that never claims
/// utility jobs. `SCENEWORKS_RUST_WORKER_GPU_ID` overrides for the rare case of
/// wanting the embedded worker on a specific GPU.
fn inprocess_worker_gpu_id(override_var: Option<String>) -> String {
    override_var
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "cpu".to_owned())
}

struct InProcessUtilityWorker {
    handle: tokio::task::JoinHandle<sceneworks_worker::WorkerResult<()>>,
    grace: Duration,
}

impl InProcessUtilityWorker {
    async fn shutdown(self) {
        match tokio::time::timeout(self.grace, self.handle).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(error))) => eprintln!("in-process utility worker exited with error: {error}"),
            Ok(Err(join_error)) => eprintln!("in-process utility worker task failed: {join_error}"),
            Err(_) => eprintln!(
                "in-process utility worker did not stop within {}s grace period",
                self.grace.as_secs()
            ),
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

pub fn create_app(settings: Settings) -> Result<Router, JobsStoreError> {
    let _ = std::fs::create_dir_all(&settings.data_dir);
    let _ = std::fs::create_dir_all(&settings.config_dir);
    if let Some(jobs_db_parent) = settings.jobs_db_path.parent() {
        let _ = std::fs::create_dir_all(jobs_db_parent);
    }
    let _ = sweep_stale_lora_uploads(&settings.data_dir);
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
        manifest_write_locks: Arc::new(Mutex::new(HashMap::new())),
        model_size_cache: Arc::new(Mutex::new(ModelSizeCache::default())),
        http_client: reqwest::Client::new(),
        interrupted_jobs_on_startup,
    };
    let cors = cors_layer(&state.settings);

    Ok(Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/access", get(access))
        .route("/api/v1/auth/verify", post(verify_access))
        .route("/api/v1/training/targets", get(list_training_targets))
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
            "/api/v1/projects/:project_id/training/datasets",
            get(list_training_datasets).post(create_training_dataset),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id",
            get(get_training_dataset)
                .patch(update_training_dataset)
                .delete(delete_training_dataset),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/batch-rename",
            post(batch_rename_training_dataset_items),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/caption-sidecars",
            post(write_training_dataset_caption_sidecars),
        )
        .route(
            "/api/v1/projects/:project_id/training/jobs",
            post(create_training_job),
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
        .route(
            "/api/v1/projects/:project_id/person-tracks/:track_id/corrections",
            post(save_person_track_corrections),
        )
        .route("/api/v1/image/jobs", post(create_image_job))
        .route("/api/v1/video/jobs", post(create_video_job))
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models/:model_id", delete(delete_model))
        .route(
            "/api/v1/models/:model_id/download",
            post(create_model_download_job),
        )
        .route(
            "/api/v1/models/:model_id/convert",
            post(create_model_convert_job),
        )
        .route(
            "/api/v1/models/import",
            post(create_model_import_job)
                .layer(DefaultBodyLimit::max(MAX_MODEL_MULTIPART_BODY_BYTES)),
        )
        .route("/api/v1/loras", get(list_loras))
        .route("/api/v1/loras/:lora_id", delete(delete_lora))
        .route(
            "/api/v1/loras/import",
            post(create_lora_import_job)
                .layer(DefaultBodyLimit::max(MAX_LORA_MULTIPART_BODY_BYTES)),
        )
        .route(
            "/api/v1/recipe-presets",
            get(list_recipe_presets).post(create_recipe_preset),
        )
        .route(
            "/api/v1/recipe-presets/:preset_id",
            get(get_recipe_preset)
                .patch(update_recipe_preset)
                .delete(delete_recipe_preset),
        )
        .route(
            "/api/v1/recipe-presets/:preset_id/duplicate",
            post(duplicate_recipe_preset),
        )
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
        .route(
            "/api/v1/capabilities/person",
            get(person_capability_readiness),
        )
        .route("/api/v1/workers/register", post(register_worker))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_worker),
        )
        .fallback(app_fallback)
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
struct CatalogDeleteQuery {
    project_id: Option<String>,
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecipePresetsQuery {
    project_id: Option<String>,
    include_archived: Option<bool>,
    model: Option<String>,
    workflow: Option<String>,
    scope: Option<String>,
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
    /// Opt into the Rust utility worker's procedural preview instead of real,
    /// model-backed detection on the Python GPU worker. Defaults to real.
    #[serde(default)]
    preview: bool,
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
    /// Opt into the Rust utility worker's procedural preview instead of real,
    /// model-backed tracking on the Python GPU worker. Defaults to real.
    #[serde(default)]
    preview: bool,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersonTrackCorrectionsRequest {
    /// The UI's full correction set for the track. Each entry targets a frame by
    /// index and adjusts its box and/or rejects the frame; the store validates
    /// ranges and stamps author/createdAt/source. Kept as raw values so the
    /// schema-flexible `corrections` array can evolve without an API change.
    #[serde(default)]
    corrections: Vec<Value>,
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
    recipe_preset_id: Option<String>,
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelConvertRequest {
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelImportRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, alias = "type", skip_serializing_if = "Option::is_none")]
    model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    family: Option<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    uploaded_source_path: bool,
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
    source_url: Option<String>,
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
    #[serde(default, skip_deserializing, skip_serializing_if = "bool_is_false")]
    uploaded_source_path: bool,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "sceneworks-api",
        runtime: "rust".to_owned(),
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

async fn list_training_targets() -> Json<TrainingTargetRegistry> {
    Json(builtin_training_targets())
}

async fn list_training_datasets(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<TrainingDatasetSummary>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.list_training_datasets(&project_id)
        })
        .await?,
    ))
}

async fn create_training_dataset(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<TrainingDatasetCreateInput>,
) -> Result<(StatusCode, Json<TrainingDataset>), ApiError> {
    let dataset = project_call(state, move |store| {
        store.create_training_dataset(&project_id, payload)
    })
    .await?;
    Ok((StatusCode::CREATED, Json(dataset)))
}

async fn get_training_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
) -> Result<Json<TrainingDataset>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.get_training_dataset(&project_id, &dataset_id)
        })
        .await?,
    ))
}

async fn update_training_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingDatasetUpdateInput>,
) -> Result<Json<TrainingDataset>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.update_training_dataset(&project_id, &dataset_id, payload)
        })
        .await?,
    ))
}

async fn batch_rename_training_dataset_items(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingDatasetBatchRenameInput>,
) -> Result<Json<TrainingDataset>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.batch_rename_training_dataset_items(&project_id, &dataset_id, payload)
        })
        .await?,
    ))
}

async fn write_training_dataset_caption_sidecars(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<TrainingDatasetCaptionSidecarsInput>,
) -> Result<Json<TrainingCaptionSidecarsResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.write_training_dataset_caption_sidecars(&project_id, &dataset_id, payload)
        })
        .await?,
    ))
}

async fn delete_training_dataset(
    State(state): State<AppState>,
    Path((project_id, dataset_id)): Path<(String, String)>,
) -> Result<Json<TrainingDatasetMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_training_dataset(&project_id, &dataset_id)
        })
        .await?,
    ))
}

async fn create_training_job(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<LoraTrainingRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    // Both dry-run plan validation and real execution exist (story 1417). A
    // dry-run resolves and validates the plan without producing weights; a real
    // run hands the same plan to the worker's Z-Image LoRA kernel.
    //
    // Targets come from the Rust-owned registry; the request only names one.
    let registry = builtin_training_targets();
    let target = registry
        .targets
        .iter()
        .find(|target| target.id == payload.target_id)
        .ok_or_else(|| {
            ApiError::bad_request(format!("Unknown training target: {}", payload.target_id))
        })?;

    let output_name = payload.output_name.trim().to_owned();
    if output_name.is_empty() {
        return Err(ApiError::bad_request("Training output name is required."));
    }

    // Load the dataset, its absolute root, and the project name for the queue.
    let dataset_id = payload.dataset_id.clone();
    let (dataset, dataset_root, project_name) = project_call(state.clone(), {
        let project_id = project_id.clone();
        move |store| store.training_dataset_for_plan(&project_id, &dataset_id)
    })
    .await?;

    // We persist only the dataset's current version, so an older pin is unrunnable.
    if let Some(requested_version) = payload.dataset_version {
        if requested_version != dataset.version {
            return Err(ApiError::bad_request(format!(
                "Dataset {} is at version {}, but the request pinned version {requested_version}.",
                dataset.id, dataset.version
            )));
        }
    }

    // Resolve absolute on-host paths and ids the kernel will consume. The job id
    // is pre-allocated so the plan can embed its own `jobId`/`sourceJobId`.
    let data_dir = state.settings.data_dir.clone();
    let base_model_path = resolve_base_model_path(target, &data_dir);
    let lora_id = format!("lora_{}", Uuid::new_v4().simple());
    let file_name = format!("{}.safetensors", slugify_lora_id(&output_name));
    let job_id = format!("job_{}", Uuid::new_v4().simple());
    let requested_gpu = training_requested_gpu(&payload.config.advanced);

    // Where the produced adapter is written. The target's default `outputScope`
    // (project) lives in the config's `advanced` bag: project outputs land in the
    // project's LoRA store, global outputs in the shared data dir. The manifest
    // `source.path` stays relative to the scope's root so `normalize_lora_entry`
    // resolves the installed path on either side. The matching manifest path is
    // recomputed from the same trusted inputs at registration time
    // (`register_trained_lora`), never read back from the job payload.
    let output_scope = training_output_scope(&payload.config.advanced)?;
    let (output_dir, _manifest_path) =
        resolve_training_output_location(&state, &output_scope, Some(&project_id), &lora_id)
            .await?;
    let source_relpath = format!("loras/{lora_id}");

    // Operational guardrails (story 1419): fail fast with actionable errors for
    // the common setup problems, before a job is queued.
    //
    // The produced LoRA's family must be one an installed model accepts, or the
    // output would never be selectable in Image Studio. When no model manifests
    // are present the set is empty and this is a no-op. Families come straight
    // from the manifests (not `model_catalog`) so this guardrail — which runs on
    // every submit, including the offline dry-run path — makes no network calls.
    let normalized_family = normalize_lora_family(&target.family);
    let known_families = known_lora_families_from_manifests(&state).await?;
    if !known_families.is_empty()
        && !known_families
            .iter()
            .any(|family| family == &normalized_family)
    {
        return Err(ApiError::bad_request(format!(
            "Training target '{}' produces LoRA family '{}', which no installed model accepts ({}). Install a compatible base model first.",
            target.id,
            target.family,
            known_families.join(", ")
        )));
    }

    // A real run loads the base model and writes weights, so it must have the
    // model installed and room on disk. A dry run only resolves the plan, so it
    // is exempt — that is how you preview a plan before installing the model.
    if !payload.dry_run {
        if !training_base_model_installed(&data_dir, target) {
            return Err(ApiError::bad_request(format!(
                "Base model '{}' is not installed. Install it from the model catalog before starting a real training run (dry runs work without it).",
                target.base_model
            )));
        }
        if let Some(message) = training_disk_space_error(&output_dir) {
            return Err(ApiError::bad_request(message));
        }
    }

    // Rust resolves and validates the normalized plan before any job is queued.
    let plan = build_training_plan(BuildTrainingPlan {
        job_id: &job_id,
        target,
        dataset: &dataset,
        config: payload.config,
        lora_id: &lora_id,
        base_model_path,
        dataset_root: &dataset_root,
        output_dir: &output_dir,
        file_name,
        created_at: now_rfc3339(),
    })
    .map_err(|error| ApiError::bad_request(error.to_string()))?;

    // Pre-build the LoRA registry entry the completed job will register, mirroring
    // the `lora_import` pattern: Rust owns LoRA registration. The descriptive
    // metadata is captured here; the manifest path and the security-sensitive
    // fields (id, scope, source path) are recomputed from trusted inputs when the
    // entry is upserted on completion (story 1418), so this stays purely
    // informational. Dry runs never register.
    let timestamp = now_rfc3339();
    let manifest_entry = json!({
        "id": lora_id.clone(),
        "name": output_name.clone(),
        "scope": output_scope,
        "family": target.family.clone(),
        "baseModel": target.base_model.clone(),
        "triggerWords": plan.output.trigger_words.clone(),
        "source": {
            "provider": "training",
            "path": source_relpath,
        },
        "files": [plan.output.file_name.clone()],
        "provenance": {
            "kind": "training",
            "trainingJobId": job_id.clone(),
            "targetId": plan.provenance.target_id.clone(),
            "datasetId": plan.provenance.dataset_id.clone(),
            "datasetVersion": plan.provenance.dataset_version,
            "baseModel": plan.provenance.base_model.clone(),
            "configSnapshot": plan.provenance.config_snapshot.clone(),
            "createdAt": timestamp.clone(),
        },
        "createdAt": timestamp.clone(),
        "updatedAt": timestamp,
    });

    let plan_value =
        serde_json::to_value(&plan).map_err(|error| ApiError::internal(error.to_string()))?;
    let mut job_payload = JsonObject::new();
    job_payload.insert("dryRun".to_owned(), Value::Bool(payload.dry_run));
    job_payload.insert("outputName".to_owned(), Value::String(output_name));
    job_payload.insert("plan".to_owned(), plan_value);
    job_payload.insert("manifestEntry".to_owned(), manifest_entry);

    let job = store_call(state.clone(), move |store, _timeout| {
        store.create_job_with_id(
            job_id,
            CreateJob {
                job_type: JobType::LoraTrain,
                project_id: Some(project_id),
                project_name: Some(project_name),
                payload: job_payload,
                requested_gpu,
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Absolute path to the target's base model weights on the worker host. Prefers
/// the Hugging Face hub cache for the target's repo, falling back to the local
/// models directory. The path need not exist yet — model installation is a
/// separate job; the dry-run plan only records where the kernel will read from.
fn resolve_base_model_path(target: &TrainingTarget, data_dir: &FsPath) -> String {
    if let Some(repo) = target
        .base_model_repo
        .as_deref()
        .map(str::trim)
        .filter(|repo| !repo.is_empty())
    {
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, repo) {
            return cache_path.display().to_string();
        }
    }
    data_dir
        .join("models")
        .join(safe_download_dir(&target.base_model))
        .display()
        .to_string()
}

/// GPU selection for a training job, read from the config's advanced bag (the
/// request has no top-level field). Defaults to `auto`; `lora_train` is
/// GPU-required, so a `cpu` value is rejected downstream when the job is created.
fn training_requested_gpu(advanced: &JsonObject) -> String {
    let raw = advanced
        .get("requestedGpu")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    requested_gpu_or_auto(raw)
}

/// Resolves the output scope for a training run from the config's `advanced` bag.
/// Defaults to `project` (the target default) and rejects anything other than the
/// two scopes the LoRA registry understands.
fn training_output_scope(advanced: &JsonObject) -> Result<String, ApiError> {
    let scope = advanced
        .get("outputScope")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("project");
    match scope {
        "project" | "global" => Ok(scope.to_owned()),
        other => Err(ApiError::bad_request(format!(
            "Unsupported training outputScope: {other}. Use project or global."
        ))),
    }
}

/// Single source of truth for where a training run's adapter is written and its
/// LoRA registered, derived only from trusted inputs: the scope, the owning
/// project, and the pre-allocated LoRA id. `create_training_job` uses it to place
/// the plan's output dir at submit time, and `register_trained_lora` recomputes
/// it at completion so a (mutable) job payload can never redirect a manifest
/// write outside the two canonical LoRA manifests. Returns
/// `(output_dir, manifest_path)`.
async fn resolve_training_output_location(
    state: &AppState,
    scope: &str,
    project_id: Option<&str>,
    lora_id: &str,
) -> Result<(PathBuf, PathBuf), ApiError> {
    match scope {
        "project" => {
            let project_id = project_id.ok_or_else(|| {
                ApiError::bad_request("Project-scoped training requires a project id")
            })?;
            let loras_dir = project_path_for_id(state.clone(), project_id)
                .await?
                .join("loras");
            Ok((loras_dir.join(lora_id), loras_dir.join("manifest.jsonc")))
        }
        "global" => {
            let loras_dir = state.settings.data_dir.join("loras");
            Ok((
                loras_dir.join(lora_id),
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
            ))
        }
        other => Err(ApiError::bad_request(format!(
            "Unsupported training outputScope: {other}. Use project or global."
        ))),
    }
}

/// Rejects a LoRA id that is not a single safe path component, so a crafted job
/// payload cannot escape the LoRA output/manifest tree via `..` or path
/// separators. Server-generated ids (`lora_<hex>`) always pass.
fn validate_lora_id_component(lora_id: &str) -> Result<(), ApiError> {
    let invalid = lora_id.is_empty()
        || lora_id == "."
        || lora_id == ".."
        || lora_id.contains("..")
        || lora_id.chars().any(|character| {
            !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.'))
        });
    if invalid {
        return Err(ApiError::bad_request(format!(
            "Invalid LoRA id for training output: {lora_id}"
        )));
    }
    Ok(())
}

/// Whether the training target's base model weights are present on disk, using
/// the same resolution `model_catalog` reports: the shared Hugging Face hub
/// cache for the target's repo, or a SceneWorks-managed `data/models/<id>`
/// install marker. A real run requires this; a dry run does not.
fn training_base_model_installed(data_dir: &FsPath, target: &TrainingTarget) -> bool {
    if let Some(repo) = target
        .base_model_repo
        .as_deref()
        .map(str::trim)
        .filter(|repo| !repo.is_empty())
    {
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, repo) {
            if huggingface_repo_cache_exists(&cache_path) {
                return true;
            }
        }
    }
    let managed = data_dir
        .join("models")
        .join(safe_download_dir(&target.base_model));
    model_is_installed(&managed)
}

/// Minimum free space we require at the output location before queuing a real
/// training run: enough headroom for periodic checkpoints plus the final
/// adapter. Conservative so it only trips when a disk is genuinely low.
const MIN_FREE_TRAINING_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Returns an actionable error message when the output volume is too low on
/// space for a real run, or `None` when there is room (or free space cannot be
/// determined — we do not block on an unknowable answer).
fn training_disk_space_error(output_dir: &FsPath) -> Option<String> {
    // `output_dir` itself rarely exists yet; probe the nearest existing parent.
    let probe = nearest_existing_ancestor(output_dir)?;
    let available = fs2::available_space(&probe).ok()?;
    insufficient_disk_space(available, MIN_FREE_TRAINING_BYTES).then(|| {
        format!(
            "Not enough free disk space to train: {} available on the volume holding {}, but at least {} is recommended. Free up space and try again.",
            human_gib(available),
            probe.display(),
            human_gib(MIN_FREE_TRAINING_BYTES)
        )
    })
}

/// Pure decision split out so the threshold logic is unit-testable without
/// touching a real filesystem.
fn insufficient_disk_space(available: u64, required: u64) -> bool {
    available < required
}

/// Nearest ancestor of `path` (including itself) that exists on disk.
fn nearest_existing_ancestor(path: &FsPath) -> Option<PathBuf> {
    let mut current = Some(path);
    while let Some(candidate) = current {
        if candidate.exists() {
            return Some(candidate.to_path_buf());
        }
        current = candidate.parent();
    }
    None
}

fn human_gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// The trusted `files` list for a trained LoRA: the adapter file names the plan
/// declared (staged into `manifestEntry.files` at submit), each validated as a
/// plain in-tree file and confirmed to exist under the recomputed output dir.
/// Returns `None` when none qualify.
///
/// Trusting the declared name rather than the first `.safetensors` on disk
/// matters: the trainer leaves step checkpoints (`<stem>-stepNNN.safetensors`)
/// in the same directory as the final `<stem>.safetensors`, and an arbitrary
/// pick could register an under-trained checkpoint. Requiring plain components
/// also keeps a crafted `..`-traversing `files` value from pointing generation
/// at a safetensors outside `installedPath`.
fn trusted_adapter_files(declared: Option<&Value>, output_dir: &FsPath) -> Option<Vec<String>> {
    let declared = declared?.as_array()?;
    let files = declared
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| is_plain_relative_file(name) && output_dir.join(name).is_file())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

/// Whether `name` is a single relative file path made only of normal components
/// (no `..`, root, drive prefix, or `.`), so joining it to an output dir cannot
/// escape that dir.
fn is_plain_relative_file(name: &str) -> bool {
    let path = FsPath::new(name);
    !name.is_empty()
        && path.file_name().is_some()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
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
    validate_job_lora_compatibility(&state, Some(&project_id), &mut job_payload, true).await?;
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

async fn save_person_track_corrections(
    State(state): State<AppState>,
    Path((project_id, track_id)): Path<(String, String)>,
    ApiJson(payload): ApiJson<PersonTrackCorrectionsRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.set_person_track_corrections(&project_id, &track_id, payload.corrections)
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
    if payload.preview {
        job_payload.insert("preview".to_owned(), Value::Bool(true));
    }
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
    if payload.preview {
        job_payload.insert("preview".to_owned(), Value::Bool(true));
    }
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
    validate_job_lora_compatibility(&state, Some(&payload.project_id), &mut job_payload, false)
        .await?;
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
    if payload.model == default_image_model() {
        if let Some(model) = preset
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            job_payload.insert("model".to_owned(), Value::String(model.to_owned()));
        }
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
    for preset_lora in recipe_preset_loras(preset) {
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
    if payload.recipe_preset_id.is_none() {
        job_payload.remove("recipePresetId");
    }
    validate_job_lora_compatibility(&state, Some(&payload.project_id), &mut job_payload, false)
        .await?;
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

/// Embedded production web bundle (apps/web/dist), compiled in only under the
/// `embed-web` feature so default/server/test builds need no web build.
#[cfg(feature = "embed-web")]
mod web_assets {
    use axum::http::{header, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use rust_embed::RustEmbed;

    #[derive(RustEmbed)]
    #[folder = "../web/dist"]
    struct WebAssets;

    pub(super) async fn serve(uri: Uri) -> Response {
        let requested = uri.path().trim_start_matches('/');
        let requested = if requested.is_empty() {
            "index.html"
        } else {
            requested
        };
        if let Some(file) = WebAssets::get(requested) {
            let mime = mime_guess::from_path(requested).first_or_octet_stream();
            return (
                [(header::CONTENT_TYPE, mime.as_ref())],
                file.data.into_owned(),
            )
                .into_response();
        }
        // Single-page app: unknown non-API paths resolve to index.html so
        // client-side deep links (e.g. project routes) load correctly.
        match WebAssets::get("index.html") {
            Some(index) => (
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                index.data.into_owned(),
            )
                .into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }
}

/// Router fallback. With `embed-web`, non-API paths are served from the embedded
/// web bundle (SPA fallback); API paths and all default-feature builds keep the
/// existing JSON not-found behavior.
async fn app_fallback(request: Request<axum::body::Body>) -> Response {
    #[cfg(feature = "embed-web")]
    {
        if !request.uri().path().starts_with("/api/") {
            return web_assets::serve(request.uri().clone()).await;
        }
    }
    route_not_found(request).await
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

/// Convert a model's native checkpoint into the local MLX format (macOS/Apple
/// Silicon). Only valid for models whose manifest declares `mlx.requiresConversion`
/// (Wan TI2V-5B, Wan I2V-A14B); turnkey MLX models need no conversion. The native
/// source checkpoint must already be downloaded; the Rust utility worker shells out
/// to the Python/MLX `mlx_video.convert_wan` tool.
async fn create_model_convert_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelConvertRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let mlx = model
        .get("mlx")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("Model has no MLX variant to convert"))?;
    if !mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(ApiError::bad_request(
            "Model does not require MLX conversion",
        ));
    }
    let source_repo = mlx
        .get("convertSourceRepo")
        .and_then(Value::as_str)
        .filter(|repo| !repo.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("MLX conversion source repo is not configured"))?
        .to_owned();
    let output_dir = state
        .settings
        .data_dir
        .join("models")
        .join("mlx")
        .join(&model_id);
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
    job_payload.insert("sourceRepo".to_owned(), Value::String(source_repo));
    job_payload.insert(
        "outputDir".to_owned(),
        Value::String(output_dir.display().to_string()),
    );
    job_payload.insert("dtype".to_owned(), Value::String("bfloat16".to_owned()));

    let job = create_generation_job(
        state,
        JobType::ModelConvert,
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

async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let catalog = model_catalog(&state).await?;
    let model = catalog
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    let removed_entry =
        remove_catalog_manifest_entry(&state, &manifest_path, "models", &model_id).await?;
    let cleanup_source = removed_entry.as_ref().unwrap_or(&model);
    let mut removed_paths = Vec::new();
    let mut retained_paths = Vec::new();
    let allowed_roots = vec![
        state.settings.data_dir.join("models"),
        huggingface_hub_cache_dir(&state.settings.data_dir),
    ];
    for path in model_artifact_paths(cleanup_source, &state.settings.data_dir) {
        remove_owned_artifact_path(
            path,
            &allowed_roots,
            &mut removed_paths,
            &mut retained_paths,
        )
        .await?;
    }
    if removed_entry.is_none() && removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in model catalog entries are read-only unless local files are installed",
        ));
    }
    let warnings = catalog_delete_warnings(&state, "model", &model_id, None).await?;
    let policy = if removed_entry.is_some() {
        "Removed the model registry entry and SceneWorks-owned local model files."
    } else {
        "Built-in model catalog entries are retained; SceneWorks-owned local model files were removed."
    };
    Ok(Json(json!({
        "id": model_id,
        "kind": "model",
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removed_paths.is_empty(),
        "removedPaths": removed_paths,
        "retainedPaths": retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

async fn delete_lora(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
    Query(query): Query<CatalogDeleteQuery>,
) -> Result<Json<Value>, ApiError> {
    let catalog = lora_catalog(&state, query.project_id.as_deref()).await?;
    let lora = catalog
        .into_iter()
        .find(|item| {
            item.get("id").and_then(Value::as_str) == Some(lora_id.as_str())
                && query.scope.as_deref().map_or(true, |scope| {
                    item.get("scope").and_then(Value::as_str) == Some(scope)
                })
        })
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "LoRA not found".to_owned(),
        })?;
    let scope = query
        .scope
        .as_deref()
        .or_else(|| lora.get("scope").and_then(Value::as_str))
        .unwrap_or("global");
    let (manifest_path, allowed_roots, default_root) = match scope {
        "global" => (
            Some(
                state
                    .settings
                    .config_dir
                    .join("manifests")
                    .join("user.loras.jsonc"),
            ),
            vec![state.settings.data_dir.join("loras")],
            state.settings.data_dir.clone(),
        ),
        "project" => {
            let Some(project_id) = query.project_id.as_deref() else {
                return Err(ApiError::bad_request(
                    "Project LoRA deletion requires projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), project_id).await?;
            (
                Some(project_path.join("loras").join("manifest.jsonc")),
                vec![
                    state.settings.data_dir.join("loras"),
                    project_path.join("loras"),
                ],
                project_path,
            )
        }
        "builtin" => (
            None,
            vec![state.settings.data_dir.join("loras")],
            state.settings.data_dir.clone(),
        ),
        _ => return Err(ApiError::bad_request("Unsupported LoRA scope")),
    };
    let removed_entry = if let Some(manifest_path) = manifest_path.as_deref() {
        remove_catalog_manifest_entry(&state, manifest_path, "loras", &lora_id).await?
    } else {
        None
    };
    let cleanup_source = removed_entry.as_ref().unwrap_or(&lora);
    let mut removed_paths = Vec::new();
    let mut retained_paths = Vec::new();
    for path in lora_artifact_paths(cleanup_source, &default_root) {
        remove_owned_artifact_path(
            path,
            &allowed_roots,
            &mut removed_paths,
            &mut retained_paths,
        )
        .await?;
    }
    if removed_entry.is_none() && removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in LoRA catalog entries are read-only unless local files are installed",
        ));
    }
    let warnings =
        catalog_delete_warnings(&state, "lora", &lora_id, query.project_id.as_deref()).await?;
    let policy = if removed_entry.is_some() {
        "Removed the LoRA registry entry and SceneWorks-owned local LoRA files."
    } else {
        "Built-in LoRA catalog entries are retained; SceneWorks-owned local LoRA files were removed."
    };
    Ok(Json(json!({
        "id": lora_id,
        "kind": "lora",
        "scope": scope,
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removed_paths.is_empty(),
        "removedPaths": removed_paths,
        "retainedPaths": retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

async fn list_recipe_presets(
    State(state): State<AppState>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let mut presets = recipe_preset_catalog(&state, query.project_id.as_deref()).await?;
    if !query.include_archived.unwrap_or(false) {
        presets.retain(|preset| !recipe_preset_archived(preset));
    }
    if let Some(model) = query.model.as_deref() {
        presets.retain(|preset| preset.get("model").and_then(Value::as_str) == Some(model));
    }
    if let Some(workflow) = query.workflow.as_deref() {
        presets.retain(|preset| preset.get("workflow").and_then(Value::as_str) == Some(workflow));
    }
    if let Some(scope) = query.scope.as_deref() {
        presets.retain(|preset| preset.get("scope").and_then(Value::as_str) == Some(scope));
    }
    Ok(Json(presets))
}

async fn get_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let preset = recipe_preset_catalog(&state, query.project_id.as_deref())
        .await?
        .into_iter()
        .find(|preset| preset.get("id").and_then(Value::as_str) == Some(preset_id.as_str()))
        .filter(|preset| {
            query.scope.as_deref().map_or(true, |scope| {
                preset.get("scope").and_then(Value::as_str) == Some(scope)
            })
        })
        .filter(|preset| query.include_archived.unwrap_or(false) || !recipe_preset_archived(preset))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Recipe preset not found".to_owned(),
        })?;
    Ok(Json(preset))
}

async fn create_recipe_preset(
    State(state): State<AppState>,
    Query(query): Query<RecipePresetsQuery>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    validate_recipe_preset_query(&query)?;
    let mut preset = recipe_preset_from_payload(payload)?;
    let scope = recipe_preset_write_scope(query.scope.as_deref(), recipe_preset_scope(&preset))?;
    let project_id = recipe_preset_context_project_id(&query, &mut preset);
    let manifest_path =
        recipe_preset_write_manifest_path(&state, &scope, project_id.as_deref()).await?;
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Recipe preset must be an object"))?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            object
                .get("name")
                .and_then(Value::as_str)
                .map(slugify_preset_id)
        })
        .ok_or_else(|| ApiError::bad_request("Recipe preset name is required"))?;
    object.insert("id".to_owned(), Value::String(id.clone()));
    let timestamp = now_rfc3339();
    object
        .entry("createdAt".to_owned())
        .or_insert_with(|| Value::String(timestamp.clone()));
    object.insert("updatedAt".to_owned(), Value::String(timestamp));
    let models = model_catalog(&state).await?;
    let loras = lora_catalog(&state, project_id.as_deref()).await?;
    let preset = mutate_manifest_entries(&state, &manifest_path, "presets", |mut entries| {
        let preset = normalize_recipe_preset_for_write(preset, &scope, true)?;
        validate_recipe_preset_model_workflow(&models, &preset)?;
        validate_recipe_preset_lora_compatibility(&models, &loras, &preset)?;
        if entries
            .iter()
            .any(|entry| entry.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            return Err(ApiError::bad_request("Recipe preset already exists"));
        }
        entries.push(preset.clone());
        Ok((entries, preset))
    })
    .await?;
    Ok((StatusCode::CREATED, Json(finalized_recipe_preset(preset)?)))
}

async fn update_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
    ApiJson(payload): ApiJson<Value>,
) -> Result<Json<Value>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let mut patch = recipe_preset_from_payload(payload)?;
    let project_id = recipe_preset_context_project_id(&query, &mut patch);
    strip_recipe_preset_write_context(&mut patch);
    let location = find_recipe_preset_write_location(
        &state,
        &preset_id,
        project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let models = model_catalog(&state).await?;
    let loras = lora_catalog(&state, project_id.as_deref()).await?;
    let preset =
        mutate_manifest_entries(&state, &location.manifest_path, "presets", |mut entries| {
            let Some(index) = entries.iter().position(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(preset_id.as_str())
            }) else {
                return Err(recipe_preset_not_found());
            };
            let mut preset = entries[index].clone();
            merge_object(&mut preset, patch);
            if let Some(object) = preset.as_object_mut() {
                object.insert("id".to_owned(), Value::String(preset_id.clone()));
                object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
            }
            let preset = normalize_recipe_preset_for_write(preset, &location.scope, false)?;
            validate_recipe_preset_model_workflow(&models, &preset)?;
            validate_recipe_preset_lora_compatibility(&models, &loras, &preset)?;
            entries[index] = preset.clone();
            Ok((entries, preset))
        })
        .await?;
    Ok(Json(finalized_recipe_preset(preset)?))
}

async fn delete_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_recipe_preset_query(&query)?;
    let location = find_recipe_preset_write_location(
        &state,
        &preset_id,
        query.project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let preset =
        mutate_manifest_entries(&state, &location.manifest_path, "presets", |mut entries| {
            let Some(index) = entries.iter().position(|entry| {
                entry.get("id").and_then(Value::as_str) == Some(preset_id.as_str())
            }) else {
                return Err(recipe_preset_not_found());
            };
            let mut preset = entries[index].clone();
            if let Some(object) = preset.as_object_mut() {
                object.insert("archived".to_owned(), Value::Bool(true));
                object.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));
            }
            let preset = normalize_recipe_preset_for_write(preset, &location.scope, false)?;
            entries[index] = preset.clone();
            Ok((entries, preset))
        })
        .await?;
    Ok(Json(finalized_recipe_preset(preset)?))
}

async fn duplicate_recipe_preset(
    State(state): State<AppState>,
    Path(preset_id): Path<String>,
    Query(query): Query<RecipePresetsQuery>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    validate_recipe_preset_query(&query)?;
    let location = find_recipe_preset_write_location(
        &state,
        &preset_id,
        query.project_id.as_deref(),
        query.scope.as_deref(),
    )
    .await?;
    let models = model_catalog(&state).await?;
    let loras = lora_catalog(&state, query.project_id.as_deref()).await?;
    let preset =
        mutate_manifest_entries(&state, &location.manifest_path, "presets", |mut entries| {
            let Some(source) = entries
                .iter()
                .find(|entry| entry.get("id").and_then(Value::as_str) == Some(preset_id.as_str()))
                .cloned()
            else {
                return Err(recipe_preset_not_found());
            };
            let mut duplicate = source;
            strip_recipe_preset_runtime_fields(&mut duplicate);
            let base_id = duplicate
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(preset_id.as_str());
            let duplicate_id = next_duplicate_preset_id(&entries, base_id);
            let duplicate_name = next_duplicate_preset_name(
                &entries,
                duplicate
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(base_id),
            );
            let timestamp = now_rfc3339();
            if let Some(object) = duplicate.as_object_mut() {
                object.insert("id".to_owned(), Value::String(duplicate_id));
                object.insert("name".to_owned(), Value::String(duplicate_name));
                object.insert("scope".to_owned(), Value::String(location.scope.clone()));
                object.insert("archived".to_owned(), Value::Bool(false));
                object.insert("createdAt".to_owned(), Value::String(timestamp.clone()));
                object.insert("updatedAt".to_owned(), Value::String(timestamp));
            }
            let duplicate = normalize_recipe_preset_for_write(duplicate, &location.scope, true)?;
            validate_recipe_preset_model_workflow(&models, &duplicate)?;
            validate_recipe_preset_lora_compatibility(&models, &loras, &duplicate)?;
            entries.push(duplicate.clone());
            Ok((entries, duplicate))
        })
        .await?;
    Ok((StatusCode::CREATED, Json(finalized_recipe_preset(preset)?)))
}

async fn create_lora_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if is_multipart {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()).into_response())?;
        let (payload, staged_path) = lora_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_lora_import_job(state, payload).await;
        if result.is_err() {
            cleanup_staged_lora_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<LoraImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_lora_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

async fn queue_lora_import_job(
    state: AppState,
    mut payload: LoraImportRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, or source path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    if !matches!(payload.scope.as_str(), "global" | "project") {
        return Err(ApiError::bad_request(
            "LoRA scope must be global or project",
        ));
    }
    if let Some(family) = payload.family.take() {
        let models = model_catalog(&state).await?;
        payload.family = Some(validate_lora_family(&models, &family)?);
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            payload
                .source_url
                .as_deref()
                .and_then(|value| lora_source_url_file_stem(value).ok())
        })
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
    let (target_dir, manifest_path, source_path, project_id, project_name, allowed_source_roots) =
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
                vec![
                    state.settings.data_dir.join("loras"),
                    project_path.join("loras"),
                ],
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
                vec![state.settings.data_dir.join("loras")],
            )
        };
    if let Some(source_path) = payload.source_path.as_deref() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("lora-uploads")]
        } else {
            allowed_source_roots
        };
        validate_lora_import_source_path(source_path, &allowed_source_roots)?;
        let detected = detect_family_from_local_path(source_path)?;
        payload.family = reconcile_lora_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": lora_id,
        "name": name,
        "scope": payload.scope.clone(),
        "source": {
            "provider": lora_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path,
        },
        "files": payload.files.clone(),
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
    if let Some(source_url) = payload.source_url.clone() {
        if let Some(source) = manifest_entry
            .get_mut("source")
            .and_then(Value::as_object_mut)
        {
            source.insert("url".to_owned(), Value::String(source_url));
        }
    }
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

async fn lora_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(LoraImportRequest, PathBuf), ApiError> {
    let mut payload = LoraImportRequest {
        lora_id: None,
        name: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        family: None,
        scope: default_lora_scope(),
        project_id: None,
        uploaded_source_path: false,
    };
    let mut staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one LoRA file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("lora.safetensors"));
                let path =
                    write_lora_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }

            let value = field
                .text()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match field_name.as_str() {
                "loraId" => payload.lora_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "scope" => payload.scope = value.to_owned(),
                "projectId" => payload.project_id = Some(value.to_owned()),
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = parse_result {
        if let Some(path) = staged_path.as_deref() {
            cleanup_staged_lora_upload(path).await;
        }
        return Err(error);
    }

    let Some(staged_path) = staged_path else {
        return Err(ApiError::bad_request("Upload file field is required"));
    };
    Ok((payload, staged_path))
}

async fn write_lora_upload_field_to_staged_file(
    state: &AppState,
    mut field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("lora-uploads")
        .join(format!("upload-{}", Uuid::new_v4().simple()));
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(filename);
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut uploaded_bytes = 0usize;
    let write_result = async {
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            uploaded_bytes = uploaded_bytes.saturating_add(chunk.len());
            if uploaded_bytes > max_lora_upload_bytes() {
                return Err(ApiError::payload_too_large(
                    "Uploaded LoRA file exceeds the 2GB limit",
                ));
            }
            file.write_all(&chunk)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
        }
        file.flush()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))
    }
    .await;
    if let Err(error) = write_result {
        cleanup_staged_lora_upload(&temp_path).await;
        return Err(error);
    }
    Ok(temp_path)
}

async fn cleanup_staged_lora_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

const ALLOWED_MODEL_TYPES: &[&str] = &["image", "video", "utility"];

async fn create_model_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if is_multipart {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()).into_response())?;
        let (payload, staged_path) = model_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_model_import_job(state, payload).await;
        if result.is_err() {
            cleanup_staged_model_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<ModelImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_model_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

async fn queue_model_import_job(
    state: AppState,
    mut payload: ModelImportRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, or source path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    let model_type = match payload.model_type.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => {
            let normalized = value.to_ascii_lowercase();
            if !ALLOWED_MODEL_TYPES.contains(&normalized.as_str()) {
                return Err(ApiError::bad_request(format!(
                    "Model type must be one of {}",
                    ALLOWED_MODEL_TYPES.join(", ")
                )));
            }
            normalized
        }
        _ => "image".to_owned(),
    };
    payload.model_type = Some(model_type.clone());
    if let Some(family) = payload.family.take() {
        let models = model_catalog(&state).await?;
        payload.family = Some(validate_lora_family(&models, &family)?);
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            payload
                .source_url
                .as_deref()
                .and_then(|value| lora_source_url_file_stem(value).ok())
        })
        .or_else(|| {
            payload.source_path.as_deref().and_then(|path| {
                FsPath::new(path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "Imported Model".to_owned());
    let model_id = payload
        .model_id
        .clone()
        .unwrap_or_else(|| slugify_lora_id(&name));
    let existing_ids = model_catalog(&state)
        .await?
        .into_iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    if existing_ids.contains(&model_id) {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' already exists. Pick a different id or delete the existing model first."
        )));
    }
    let target_name = safe_download_dir(&model_id);
    let target_dir = state
        .settings
        .data_dir
        .join("models")
        .join("imports")
        .join(&target_name);
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    let source_path_rel = format!("models/imports/{target_name}");
    let allowed_source_roots = vec![state.settings.data_dir.join("models")];
    if let Some(source_path) = payload.source_path.as_deref() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("model-uploads")]
        } else {
            allowed_source_roots
        };
        validate_lora_import_source_path(source_path, &allowed_source_roots)?;
        let detected =
            detect_model_family(FsPath::new(source_path)).map_err(model_family_inspection_error)?;
        payload.family = reconcile_model_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": model_id,
        "name": name,
        "type": model_type,
        "source": {
            "provider": model_import_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path_rel,
        },
        "files": payload.files.clone(),
        "paths": {
            "model": target_dir.display().to_string(),
        },
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
    if let Some(source_url) = payload.source_url.clone() {
        if let Some(source) = manifest_entry
            .get_mut("source")
            .and_then(Value::as_object_mut)
        {
            source.insert("url".to_owned(), Value::String(source_url));
        }
    }
    if let Some(family) = payload.family.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("family".to_owned(), Value::String(family));
        }
    }
    if let Some(object) = manifest_entry.as_object_mut() {
        apply_model_manifest_defaults(object, &model_type, payload.family.as_deref());
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("modelId".to_owned(), manifest_entry["id"].clone());
    payload.insert("modelName".to_owned(), manifest_entry["name"].clone());
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
        JobType::ModelImport,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn model_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(ModelImportRequest, PathBuf), ApiError> {
    let mut payload = ModelImportRequest {
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        family: None,
        uploaded_source_path: false,
    };
    let mut staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one model file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("model.safetensors"));
                let path =
                    write_model_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }

            let value = field
                .text()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match field_name.as_str() {
                "modelId" => payload.model_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "type" => payload.model_type = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "repo" => payload.repo = Some(value.to_owned()),
                "sourceUrl" => payload.source_url = Some(value.to_owned()),
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = parse_result {
        if let Some(path) = staged_path.as_deref() {
            cleanup_staged_model_upload(path).await;
        }
        return Err(error);
    }

    let Some(staged_path) = staged_path else {
        return Err(ApiError::bad_request("Upload file field is required"));
    };
    Ok((payload, staged_path))
}

async fn write_model_upload_field_to_staged_file(
    state: &AppState,
    mut field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("model-uploads")
        .join(format!("upload-{}", Uuid::new_v4().simple()));
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(filename);
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut uploaded_bytes = 0usize;
    let write_result = async {
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            uploaded_bytes = uploaded_bytes.saturating_add(chunk.len());
            if uploaded_bytes > max_model_upload_bytes() {
                return Err(ApiError::payload_too_large(format!(
                    "Uploaded model file exceeds the {} limit",
                    format_bytes(max_model_upload_bytes() as u64)
                )));
            }
            file.write_all(&chunk)
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?;
        }
        file.flush()
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        Ok(())
    }
    .await;
    if let Err(error) = write_result {
        drop(file);
        cleanup_staged_model_upload(&temp_path).await;
        return Err(error);
    }
    Ok(temp_path)
}

async fn cleanup_staged_model_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

fn model_import_source_provider(payload: &ModelImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

fn model_family_inspection_error(error: SafetensorsHeaderError) -> ApiError {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect model file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("Model file has an invalid safetensors header".to_owned())
        }
    }
}

/// Applies the import-time policy for base models: confident detection rejects
/// a mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection accepts the supplied family
/// unchanged (and leaves things unset if none was supplied).
fn reconcile_model_family(
    supplied: Option<String>,
    detected: Option<String>,
    _context: &str,
) -> Result<Option<String>, ApiError> {
    reconcile_detected_family(supplied, detected).map_err(|mismatch| {
        ApiError::bad_request(format!(
            "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
            mismatch.detected, mismatch.supplied, mismatch.detected
        ))
    })
}

fn max_lora_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_LORA_UPLOAD_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        if limit > 0 {
            return limit;
        }
    }
    MAX_UPLOAD_BYTES
}

fn max_model_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_MODEL_UPLOAD_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        if limit > 0 {
            return limit;
        }
    }
    MAX_MODEL_UPLOAD_BYTES
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
    let mut result = payload.result;
    // On a completing real training run, register the produced adapter as a
    // SceneWorks LoRA *before* recording completion, and fold the outcome into
    // the job result (story 1418). Doing it here keeps the result write atomic
    // and makes a registration failure visible in the job record rather than
    // silently dropping the trained output.
    if matches!(payload.status, JobStatus::Completed) {
        if let Some(status) = register_completed_training_lora(&state, &job_id).await {
            result.get_or_insert_with(JsonObject::new).extend(status);
        }
    }
    let job = store_call(state.clone(), move |store, _timeout| {
        store.update_job_progress(
            &job_id,
            ProgressUpdate {
                status: payload.status,
                stage: payload.stage,
                progress,
                message: payload.message,
                error: payload.error,
                result,
                eta_seconds,
            },
        )
    })
    .await?;
    publish(&state, "job.updated", &job);
    publish_queue(&state).await?;
    Ok(Json(job))
}

/// Attempts LoRA registration for a job reporting completion, returning result
/// fields that describe the outcome — or `None` when the job is not a real
/// training run with a staged output. Never errors the progress update: a
/// registration failure is logged and surfaced via `loraRegistered: false` +
/// `loraRegistrationError` so the trained output is not silently lost.
async fn register_completed_training_lora(state: &AppState, job_id: &str) -> Option<JsonObject> {
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await
    .ok()?;
    if !matches!(job.job_type, JobType::LoraTrain) {
        return None;
    }
    match register_trained_lora(state, &job).await {
        Ok(None) => None,
        Ok(Some((lora_id, manifest_path))) => {
            let mut status = JsonObject::new();
            status.insert("loraRegistered".to_owned(), Value::Bool(true));
            status.insert("loraId".to_owned(), Value::String(lora_id));
            status.insert(
                "loraManifestPath".to_owned(),
                Value::String(manifest_path.display().to_string()),
            );
            Some(status)
        }
        Err(error) => {
            eprintln!(
                "Failed to register trained LoRA for job {}: {}",
                job.id, error.detail
            );
            let mut status = JsonObject::new();
            status.insert("loraRegistered".to_owned(), Value::Bool(false));
            status.insert(
                "loraRegistrationError".to_owned(),
                Value::String(error.detail),
            );
            Some(status)
        }
    }
}

/// Registers a completed real training run's output as a normal SceneWorks LoRA,
/// returning the registered `(lora_id, manifest_path)` or `None` when there is
/// nothing to register (a dry run, or a job without a staged entry).
///
/// Security: the manifest path and output directory are recomputed from the
/// run's scope, owning project, and a validated LoRA id — never from the
/// (mutable) job payload — so a crafted or duplicated `lora_train` job cannot
/// redirect the manifest write outside the two canonical LoRA manifests
/// (`config_dir/manifests/user.loras.jsonc` or `<project>/loras/manifest.jsonc`).
/// A run whose adapter is missing under the recomputed output dir registers
/// nothing, so a failed/canceled/unwritten job never leaves a broken entry. The
/// entry shows up in `/api/v1/loras` and is selectable in Image Studio.
async fn register_trained_lora(
    state: &AppState,
    job: &JobSnapshot,
) -> Result<Option<(String, PathBuf)>, ApiError> {
    if job
        .payload
        .get("dryRun")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        return Ok(None);
    }
    let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    else {
        return Ok(None);
    };
    // Derive the security-sensitive fields from the entry but trust nothing: the
    // scope is validated by `resolve_training_output_location`, and the id must be
    // a safe single path component before it can name an output dir / manifest.
    let scope = manifest_entry
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("project")
        .to_owned();
    let lora_id = manifest_entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("Training manifest entry requires an id"))?
        .to_owned();
    validate_lora_id_component(&lora_id)?;

    // Recompute the output dir and manifest path from trusted inputs; the job
    // payload's own manifest/output paths are deliberately ignored.
    let (output_dir, manifest_path) =
        resolve_training_output_location(state, &scope, job.project_id.as_deref(), &lora_id)
            .await?;
    // Register the adapter file(s) the plan declared, validated as plain in-tree
    // files that exist under the recomputed output dir. Using the declared final
    // name (not the first `.safetensors` on disk) means a step checkpoint sharing
    // the directory is never registered in place of the final adapter, while the
    // validation still rejects any `..`-traversing name a crafted payload injects.
    let Some(files) = trusted_adapter_files(manifest_entry.get("files"), &output_dir) else {
        return Err(ApiError::internal(format!(
            "No declared trained adapter found under {}; skipping LoRA registration",
            output_dir.display()
        )));
    };

    // Overwrite the security-sensitive fields with the trusted values, keeping
    // the descriptive metadata (name, family, triggerWords, baseModel,
    // provenance) the submit step captured. `source.path` stays relative so
    // `normalize_lora_entry` resolves it under the scope root.
    let mut entry = manifest_entry;
    entry.insert("id".to_owned(), Value::String(lora_id.clone()));
    entry.insert("scope".to_owned(), Value::String(scope));
    entry.insert(
        "source".to_owned(),
        json!({ "provider": "training", "path": format!("loras/{lora_id}") }),
    );
    entry.insert(
        "files".to_owned(),
        Value::Array(files.into_iter().map(Value::String).collect()),
    );
    entry.insert("updatedAt".to_owned(), Value::String(now_rfc3339()));

    let upsert_id = lora_id.clone();
    mutate_manifest_entries(state, &manifest_path, "loras", move |entries| {
        // Replace any prior entry with this id (re-run) so provenance refreshes
        // without duplicating, preserving the original createdAt.
        let created_at = entries
            .iter()
            .find(|item| item.get("id").and_then(Value::as_str) == Some(upsert_id.as_str()))
            .and_then(|item| item.get("createdAt").cloned());
        let mut entries = entries
            .into_iter()
            .filter(|item| item.get("id").and_then(Value::as_str) != Some(upsert_id.as_str()))
            .collect::<Vec<_>>();
        let mut entry = entry;
        if let Some(created_at) = created_at {
            entry.insert("createdAt".to_owned(), created_at);
        }
        entries.push(Value::Object(entry));
        Ok((entries, ()))
    })
    .await?;
    Ok(Some((lora_id, manifest_path)))
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

/// Person-workflow readiness derived from the live (non-offline) workers: a
/// capability is ready when some live worker advertises it. Surfaces, per
/// dependency, whether real detection/tracking/segmentation/replacement (and the
/// procedural previews) can actually run, so the UI can gate Replace Person and
/// explain why an action is unavailable (sc-1484).
fn person_readiness_from_workers(workers: &[WorkerSnapshot]) -> Value {
    let live: Vec<&WorkerSnapshot> = workers
        .iter()
        .filter(|worker| worker.status != WorkerStatus::Offline)
        .collect();
    let entry = |capability: WorkerCapability| {
        let cap = capability.as_str();
        let ready = live.iter().any(|worker| {
            worker
                .capabilities
                .iter()
                .any(|owned| owned.as_str() == cap)
        });
        json!({ "capability": cap, "ready": ready })
    };
    json!({
        "detect": entry(WorkerCapability::PersonDetect),
        "track": entry(WorkerCapability::PersonTrack),
        "segment": entry(WorkerCapability::PersonSegment),
        "replace": entry(WorkerCapability::PersonReplace),
        "detectPreview": entry(WorkerCapability::PersonDetectPreview),
        "trackPreview": entry(WorkerCapability::PersonTrackPreview),
    })
}

async fn person_capability_readiness(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let workers = store_call(state, move |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.list_workers()
    })
    .await?;
    Ok(Json(
        json!({ "person": person_readiness_from_workers(&workers) }),
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
            utilization: payload.utilization,
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
            utilization: payload.utilization,
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
        || !requires_token(request.uri().path())
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

/// Whether a path is gated by the access token. Only `/api/*` routes are
/// protected (minus the explicitly public ones); everything else is the
/// embedded web bundle / SPA fallback, which a browser must be able to load
/// before it can attach the token header.
fn requires_token(path: &str) -> bool {
    path.starts_with("/api/") && !PUBLIC_PATHS.contains(&path)
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
    fallback_size_bytes: Option<u64>,
}

async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
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
        let fallback_size_bytes = download_context
            .as_ref()
            .and_then(|context| context.fallback_size_bytes);
        let effective_download_size_bytes = download_size_bytes.or(fallback_size_bytes);
        let download_size_estimated =
            download_size_bytes.is_none() && fallback_size_bytes.is_some();
        let (downloadable, installed_path, installed) =
            if let Some(download_context) = download_context {
                let managed_path = state
                    .settings
                    .data_dir
                    .join("models")
                    .join(safe_download_dir(&download_context.repo));
                let cache_path =
                    huggingface_repo_cache_path(&state.settings.data_dir, &download_context.repo);
                let cache_installed = cache_path
                    .as_ref()
                    .is_some_and(|path| huggingface_repo_cache_exists(path));
                let managed_installed = model_is_installed(&managed_path);
                let installed_path = if cache_installed {
                    cache_path
                } else {
                    Some(managed_path)
                };
                (
                    true,
                    installed_path.map(|path| path.display().to_string()),
                    managed_installed || cache_installed,
                )
            } else if let Some(installed_path) =
                model_manifest_installed_path(model, &state.settings.data_dir)
            {
                let installed = model_is_installed(&installed_path);
                (false, Some(installed_path.display().to_string()), installed)
            } else {
                (false, None, false)
            };
        let object = model
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
        let model_id = object.get("id").and_then(Value::as_str).unwrap_or_default();
        let user_managed = user_model_ids.contains(model_id);
        object.insert(
            "catalogScope".to_owned(),
            Value::String(if user_managed { "user" } else { "builtin" }.to_owned()),
        );
        object.insert("downloadable".to_owned(), Value::Bool(downloadable));
        object.insert(
            "downloadSizeBytes".to_owned(),
            effective_download_size_bytes
                .map(|value| json!(value))
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeLabel".to_owned(),
            effective_download_size_bytes
                .map(format_bytes)
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeEstimated".to_owned(),
            Value::Bool(download_size_estimated),
        );
        object.insert(
            "installState".to_owned(),
            Value::String(if installed { "installed" } else { "missing" }.to_owned()),
        );
        object.insert(
            "installedPath".to_owned(),
            installed_path.map(Value::String).unwrap_or(Value::Null),
        );
        object.insert(
            "removable".to_owned(),
            Value::Bool(user_managed || installed),
        );
        // macOS Model Manager: MLX availability + conversion status for models that
        // declare an `mlx` variant. Additive fields the web/Docker build ignores; the
        // probes are cheap and portable, so a const `cfg!` check gates them rather
        // than per-OS compilation. minMemoryGb passes through from the raw manifest.
        let mlx_status = if cfg!(target_os = "macos") {
            mlx_catalog_status(object, &state.settings.data_dir)
        } else {
            None
        };
        if let Some(status) = mlx_status {
            object.insert(
                "mlxInstallState".to_owned(),
                Value::String(status.install_state.to_owned()),
            );
            object.insert(
                "mlxConversionState".to_owned(),
                Value::String(status.conversion_state.to_owned()),
            );
            object.insert(
                "mlxConvertedPath".to_owned(),
                status
                    .converted_path
                    .map(|path| Value::String(path.display().to_string()))
                    .unwrap_or(Value::Null),
            );
        }
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
            .map(|lora| {
                normalize_lora_entry(
                    lora,
                    "project",
                    &project_manifest,
                    &project_path,
                    &state.settings.data_dir,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        loras = merge_entries_by_id(loras, project_loras);
    }
    for lora in &mut loras {
        let object = lora
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("LoRA manifest entry must be an object"))?;
        let scope = object
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("builtin");
        let installed = object
            .get("installState")
            .and_then(Value::as_str)
            .is_some_and(|state| state == "installed");
        object.insert(
            "removable".to_owned(),
            Value::Bool(scope != "builtin" || installed),
        );
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
    let models = model_catalog(state).await?;
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
        finalize_recipe_preset_entry(preset, &models)?;
    }
    presets.sort_by(|left, right| {
        let left_key = (
            recipe_preset_scope_order(left.get("scope").and_then(Value::as_str)),
            left.get("order").and_then(Value::as_i64).unwrap_or(10_000),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            recipe_preset_scope_order(right.get("scope").and_then(Value::as_str)),
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

fn recipe_preset_scope_order(scope: Option<&str>) -> u8 {
    match scope {
        Some("builtin") => 0,
        Some("global") => 1,
        Some("project") => 2,
        _ => 3,
    }
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
    data_dir: &FsPath,
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
    let local_path = source_path.map(|source_path| {
        let path = PathBuf::from(source_path);
        if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        }
    });
    let lora_snapshot = Value::Object(object.clone());
    let installed_path = match local_path.as_ref() {
        Some(path) if lora_is_installed(path) => Some(path.clone()),
        _ => match lora_huggingface_cached_file(&lora_snapshot, data_dir) {
            Some(path) if lora_is_installed(&path) => Some(path),
            _ => local_path.clone(),
        },
    };
    let install_state = match installed_path.as_ref() {
        Some(path) if lora_is_installed(path) => "installed",
        _ => "missing",
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

fn finalize_recipe_preset_entry(preset: &mut Value, models: &[Value]) -> Result<(), ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Recipe preset manifest entry must be an object"))?;
    let mut migration_notes = Vec::new();
    if !object.contains_key("workflow") {
        if let Some(workflow) = inferred_recipe_preset_workflow(object) {
            object.insert("workflow".to_owned(), Value::String(workflow.to_owned()));
            migration_notes.push(Value::String(format!(
                "workflow inferred from legacy modes as {workflow}"
            )));
        }
    }
    if !object.contains_key("model") {
        if let Some(model) = object
            .get("workflow")
            .and_then(Value::as_str)
            .and_then(|workflow| default_recipe_preset_model_for_workflow(models, workflow))
        {
            object.insert("model".to_owned(), Value::String(model.clone()));
            migration_notes.push(Value::String(format!(
                "model defaulted to {model} for legacy preset"
            )));
        }
    }
    if !object.contains_key("modes") {
        if let Some(workflow) = object.get("workflow").and_then(Value::as_str) {
            object.insert(
                "modes".to_owned(),
                Value::Array(
                    default_recipe_preset_modes_for_workflow(workflow)
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
    }
    if !object.contains_key("loras") {
        if let Some(loras) = object.get("builtInLoras").cloned() {
            let migrated_count = loras.as_array().map(Vec::len).unwrap_or_default();
            object.insert("loras".to_owned(), loras);
            if migrated_count > 0 {
                migration_notes.push(Value::String("builtInLoras migrated to loras".to_owned()));
            }
        }
    }
    let loras = object
        .get("loras")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    object.entry("builtInLoras".to_owned()).or_insert(loras);
    object
        .entry("defaults".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    object
        .entry("prompt".to_owned())
        .or_insert_with(|| Value::Object(JsonObject::new()));
    if !migration_notes.is_empty() {
        object.insert(
            "appliedDefaults".to_owned(),
            json!({
                "notes": migration_notes
            }),
        );
    }
    Ok(())
}

fn default_recipe_preset_model_for_workflow(models: &[Value], workflow: &str) -> Option<String> {
    models
        .iter()
        .find(|model| {
            model_supports_recipe_workflow(model, workflow)
                && model.get("installState").and_then(Value::as_str) == Some("installed")
        })
        .and_then(|model| model.get("id").and_then(Value::as_str))
        .map(str::to_owned)
}

fn model_supports_recipe_workflow(model: &Value, workflow: &str) -> bool {
    model
        .get("capabilities")
        .and_then(Value::as_array)
        .is_some_and(|capabilities| {
            capabilities
                .iter()
                .filter_map(Value::as_str)
                .any(|capability| capability == workflow)
        })
}

fn default_recipe_preset_modes_for_workflow(workflow: &str) -> Vec<String> {
    match workflow {
        "text_to_image" => vec!["text_to_image", "character_image", "style_variations"],
        "edit_image" => vec!["edit_image"],
        "image_to_video" => vec!["image_to_video"],
        "text_to_video" => vec!["text_to_video"],
        "first_last_frame" => vec!["first_last_frame"],
        _ => vec![workflow],
    }
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn inferred_recipe_preset_workflow(object: &JsonObject) -> Option<&'static str> {
    object
        .get("modes")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_str)
        .find_map(|mode| match mode {
            "text_to_image" => Some("text_to_image"),
            "edit_image" => Some("edit_image"),
            "image_to_video" => Some("image_to_video"),
            "text_to_video" => Some("text_to_video"),
            "first_last_frame" => Some("first_last_frame"),
            _ => None,
        })
}

fn recipe_preset_loras(preset: &Value) -> Vec<Value> {
    preset
        .get("loras")
        .or_else(|| preset.get("builtInLoras"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn recipe_preset_archived(preset: &Value) -> bool {
    preset
        .get("archived")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn finalized_recipe_preset(mut preset: Value) -> Result<Value, ApiError> {
    // Write paths require an explicit model before this point, so single-preset
    // response finalization does not need the read-side model catalog fallback.
    finalize_recipe_preset_entry(&mut preset, &[])?;
    Ok(preset)
}

fn recipe_preset_from_payload(payload: Value) -> Result<Value, ApiError> {
    match payload {
        Value::Null => Ok(Value::Object(JsonObject::new())),
        Value::Object(_) => Ok(payload),
        _ => Err(ApiError::bad_request(
            "Recipe preset payload must be an object",
        )),
    }
}

fn take_string_field(payload: &mut Value, field: &str) -> Option<String> {
    payload
        .as_object_mut()
        .and_then(|object| object.remove(field))
        .and_then(|value| value.as_str().map(str::to_owned))
}

fn recipe_preset_scope(preset: &Value) -> Option<&str> {
    preset.get("scope").and_then(Value::as_str)
}

fn recipe_preset_context_project_id(
    query: &RecipePresetsQuery,
    payload: &mut Value,
) -> Option<String> {
    query
        .project_id
        .clone()
        .or_else(|| take_string_field(payload, "projectId"))
}

fn strip_recipe_preset_write_context(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("projectId");
        object.remove("scope");
        object.remove("manifestPath");
        object.remove("builtInLoras");
        object.remove("appliedDefaults");
    }
}

fn strip_recipe_preset_runtime_fields(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("manifestPath");
        object.remove("builtInLoras");
        object.remove("appliedDefaults");
    }
}

fn recipe_preset_write_scope(
    query_scope: Option<&str>,
    payload_scope: Option<&str>,
) -> Result<String, ApiError> {
    let scope = query_scope.or(payload_scope).unwrap_or("global").trim();
    match scope {
        "global" | "project" => Ok(scope.to_owned()),
        "builtin" => Err(ApiError::bad_request(
            "Built-in recipe presets are read-only",
        )),
        _ => Err(ApiError::bad_request(
            "Recipe preset scope must be global or project",
        )),
    }
}

fn validate_recipe_preset_query(query: &RecipePresetsQuery) -> Result<(), ApiError> {
    if let Some(workflow) = query.workflow.as_deref() {
        validate_recipe_preset_workflow(Some(workflow), false)?;
    }
    if let Some(scope) = query.scope.as_deref() {
        match scope {
            "builtin" | "global" | "project" => {}
            _ => return Err(ApiError::bad_request("Unsupported recipe preset scope")),
        }
    }
    Ok(())
}

async fn recipe_preset_write_manifest_path(
    state: &AppState,
    scope: &str,
    project_id: Option<&str>,
) -> Result<PathBuf, ApiError> {
    match scope {
        "global" => Ok(state
            .settings
            .config_dir
            .join("manifests")
            .join("user.recipe-presets.jsonc")),
        "project" => {
            let Some(project_id) = project_id else {
                return Err(ApiError::bad_request(
                    "Project recipe presets require projectId",
                ));
            };
            let project_path = project_path_for_id(state.clone(), project_id).await?;
            Ok(project_path.join("recipes").join("presets.jsonc"))
        }
        _ => Err(ApiError::bad_request(
            "Recipe preset scope must be global or project",
        )),
    }
}

#[derive(Debug, Clone)]
struct RecipePresetWriteLocation {
    scope: String,
    manifest_path: PathBuf,
}

fn recipe_preset_not_found() -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        detail: "Recipe preset not found".to_owned(),
    }
}

async fn find_recipe_preset_write_location(
    state: &AppState,
    preset_id: &str,
    project_id: Option<&str>,
    scope: Option<&str>,
) -> Result<RecipePresetWriteLocation, ApiError> {
    match scope {
        Some("builtin") => {
            return recipe_preset_readonly_or_not_found(state, preset_id, project_id).await
        }
        Some("global") => {
            return recipe_preset_location_if_present(state, preset_id, "global", project_id).await;
        }
        Some("project") => {
            return recipe_preset_location_if_present(state, preset_id, "project", project_id)
                .await;
        }
        Some(_) => return Err(ApiError::bad_request("Unsupported recipe preset scope")),
        None => {}
    }

    if project_id.is_some() {
        match recipe_preset_location_if_present(state, preset_id, "project", project_id).await {
            Ok(location) => return Ok(location),
            Err(error) if error.status == StatusCode::NOT_FOUND => {}
            Err(error) => return Err(error),
        }
    }
    match recipe_preset_location_if_present(state, preset_id, "global", project_id).await {
        Ok(location) => Ok(location),
        Err(error) if error.status == StatusCode::NOT_FOUND => {
            recipe_preset_readonly_or_not_found(state, preset_id, project_id).await
        }
        Err(error) => Err(error),
    }
}

async fn recipe_preset_location_if_present(
    state: &AppState,
    preset_id: &str,
    scope: &str,
    project_id: Option<&str>,
) -> Result<RecipePresetWriteLocation, ApiError> {
    let manifest_path = recipe_preset_write_manifest_path(state, scope, project_id).await?;
    let entries = load_manifest_entries(state, &manifest_path, "presets").await?;
    if entries
        .iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(preset_id))
    {
        Ok(RecipePresetWriteLocation {
            scope: scope.to_owned(),
            manifest_path,
        })
    } else {
        Err(recipe_preset_not_found())
    }
}

async fn recipe_preset_readonly_or_not_found(
    state: &AppState,
    preset_id: &str,
    project_id: Option<&str>,
) -> Result<RecipePresetWriteLocation, ApiError> {
    let catalog = recipe_preset_catalog(state, project_id).await?;
    if catalog.iter().any(|preset| {
        preset.get("id").and_then(Value::as_str) == Some(preset_id)
            && preset.get("scope").and_then(Value::as_str) == Some("builtin")
    }) {
        Err(ApiError::bad_request(
            "Built-in recipe presets are read-only",
        ))
    } else {
        Err(recipe_preset_not_found())
    }
}

async fn mutate_manifest_entries<F, R>(
    state: &AppState,
    path: &FsPath,
    field: &str,
    operation: F,
) -> Result<R, ApiError>
where
    F: FnOnce(Vec<Value>) -> Result<(Vec<Value>, R), ApiError>,
{
    let lock = manifest_write_lock(state, path);
    let _guard = lock.lock().await;
    let entries = load_manifest_entries(state, path, field).await?;
    let (entries, result) = operation(entries)?;
    save_manifest_entries(path, field, entries).await?;
    Ok(result)
}

async fn remove_catalog_manifest_entry(
    state: &AppState,
    path: &FsPath,
    field: &str,
    id: &str,
) -> Result<Option<Value>, ApiError> {
    mutate_manifest_entries(state, path, field, |entries| {
        let mut removed = None;
        let entries = entries
            .into_iter()
            .filter(|entry| {
                if entry.get("id").and_then(Value::as_str) == Some(id) {
                    removed = Some(entry.clone());
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        Ok((entries, removed))
    })
    .await
}

fn manifest_write_lock(state: &AppState, path: &FsPath) -> Arc<AsyncMutex<()>> {
    let mut locks = state.manifest_write_locks.lock();
    locks
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

async fn save_manifest_entries(
    path: &FsPath,
    field: &str,
    entries: Vec<Value>,
) -> Result<(), ApiError> {
    let Some(parent) = path.parent() else {
        return Err(ApiError::internal("Manifest path has no parent directory"));
    };
    tokio::fs::create_dir_all(parent).await.map_err(|error| {
        ApiError::internal(format!(
            "Failed to create manifest directory {}: {error}",
            parent.display()
        ))
    })?;
    let mut manifest = load_manifest_root(path).await?;
    manifest.entry("$schema".to_owned()).or_insert_with(|| {
        Value::String("https://sceneworks.local/schemas/recipe-preset.schema.json".to_owned())
    });
    manifest
        .entry("schemaVersion".to_owned())
        .or_insert_with(|| json!(1));
    manifest.insert(field.to_owned(), Value::Array(entries));
    let payload = serde_json::to_string_pretty(&Value::Object(manifest))
        .map_err(|error| ApiError::internal(format!("Failed to encode manifest: {error}")))?;
    write_manifest_atomic(path, &format!("{API_MANAGED_MANIFEST_HEADER}\n{payload}\n")).await
}

async fn load_manifest_root(path: &FsPath) -> Result<JsonObject, ApiError> {
    let payload = match tokio::fs::read_to_string(path).await {
        Ok(payload) => payload,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(JsonObject::new()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to load manifest {}: {error}",
                path.display()
            )))
        }
    };
    serde_json::from_str::<Value>(&strip_jsonc_comments(&payload))
        .map_err(|error| {
            ApiError::internal(format!(
                "Failed to parse manifest {}: {error}",
                path.display()
            ))
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            ApiError::internal(format!("Manifest {} must be a JSON object", path.display()))
        })
}

async fn write_manifest_atomic(path: &FsPath, payload: &str) -> Result<(), ApiError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("jsonc");
    let tmp_path = path.with_extension(format!("{extension}.{}.tmp", Uuid::new_v4().simple()));
    tokio::fs::write(&tmp_path, payload)
        .await
        .map_err(|error| {
            ApiError::internal(format!(
                "Failed to write manifest temp file {}: {error}",
                tmp_path.display()
            ))
        })?;
    tokio::fs::rename(&tmp_path, path).await.map_err(|error| {
        let _ = std::fs::remove_file(&tmp_path);
        ApiError::internal(format!(
            "Failed to replace manifest {}: {error}",
            path.display()
        ))
    })
}

fn normalize_recipe_preset_for_write(
    mut preset: Value,
    scope: &str,
    require_all: bool,
) -> Result<Value, ApiError> {
    let object = preset
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("Recipe preset must be an object"))?;
    object.insert("scope".to_owned(), Value::String(scope.to_owned()));
    validate_recipe_preset_id(object.get("id").and_then(Value::as_str))?;
    validate_required_string_field(
        object,
        "name",
        require_all,
        "Recipe preset name is required",
    )?;
    validate_required_string_field(
        object,
        "model",
        require_all,
        "Recipe preset model is required",
    )?;
    validate_recipe_preset_workflow(object.get("workflow").and_then(Value::as_str), require_all)?;
    validate_recipe_preset_order(object.get("order"))?;
    validate_recipe_preset_defaults(object.get("defaults"))?;
    validate_recipe_preset_prompt(object.get("prompt"))?;
    normalize_recipe_preset_loras(object)?;
    Ok(preset)
}

fn validate_recipe_preset_id(value: Option<&str>) -> Result<(), ApiError> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Err(ApiError::bad_request("Recipe preset id is required"));
    };
    let valid = value.chars().enumerate().all(|(index, character)| {
        character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || (index > 0 && matches!(character, '_' | '-'))
    });
    if !valid {
        return Err(ApiError::bad_request(
            "Recipe preset id must use lowercase letters, numbers, dashes, or underscores",
        ));
    }
    Ok(())
}

fn validate_required_string_field(
    object: &JsonObject,
    field: &str,
    require: bool,
    message: &'static str,
) -> Result<(), ApiError> {
    match object.get(field).and_then(Value::as_str).map(str::trim) {
        Some(value) if !value.is_empty() => Ok(()),
        _ if require => Err(ApiError::bad_request(message)),
        _ => Ok(()),
    }
}

fn validate_recipe_preset_workflow(value: Option<&str>, require: bool) -> Result<(), ApiError> {
    match value {
        Some(
            "text_to_image" | "edit_image" | "image_to_video" | "text_to_video"
            | "first_last_frame",
        ) => Ok(()),
        Some(_) => Err(ApiError::bad_request("Unsupported recipe preset workflow")),
        None if require => Err(ApiError::bad_request("Recipe preset workflow is required")),
        None => Ok(()),
    }
}

fn validate_recipe_preset_order(value: Option<&Value>) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_i64()) {
        return Err(ApiError::bad_request(
            "Recipe preset order must be an integer",
        ));
    }
    Ok(())
}

fn validate_recipe_preset_defaults(value: Option<&Value>) -> Result<(), ApiError> {
    let Some(defaults) = value else {
        return Ok(());
    };
    let object = defaults
        .as_object()
        .ok_or_else(|| ApiError::bad_request("Recipe preset defaults must be an object"))?;
    if let Some(resolution) = object.get("resolution").and_then(Value::as_str) {
        let (width, height) = parse_recipe_preset_resolution(resolution)?;
        validate_dimension(width, "width", 2048)?;
        validate_dimension(height, "height", 2048)?;
    }
    if let Some(count) = object.get("count").and_then(Value::as_u64) {
        if !(1..=8).contains(&count) {
            return Err(ApiError::bad_request(
                "Recipe preset count must be between 1 and 8",
            ));
        }
    }
    Ok(())
}

fn validate_recipe_preset_model_workflow(models: &[Value], preset: &Value) -> Result<(), ApiError> {
    let Some(model_id) = preset.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(workflow) = preset.get("workflow").and_then(Value::as_str) else {
        return Ok(());
    };
    let model = models
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .ok_or_else(|| {
            ApiError::bad_request(format!("Recipe preset model not found: {model_id}"))
        })?;
    let capabilities = model
        .get("capabilities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if capabilities
        .iter()
        .filter_map(Value::as_str)
        .any(|capability| capability == workflow)
    {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "Model {model_id} does not support workflow {workflow}"
        )))
    }
}

fn validate_recipe_preset_lora_compatibility(
    models: &[Value],
    loras: &[Value],
    preset: &Value,
) -> Result<(), ApiError> {
    let Some(model_id) = preset.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    validate_lora_specs_for_model(
        models,
        loras,
        model_id,
        &recipe_preset_loras(preset),
        false,
        "Recipe preset LoRA",
    )?;
    Ok(())
}

async fn validate_job_lora_compatibility(
    state: &AppState,
    project_id: Option<&str>,
    job_payload: &mut JsonObject,
    allow_inline_loras: bool,
) -> Result<(), ApiError> {
    let Some(loras) = job_payload
        .get("loras")
        .and_then(Value::as_array)
        .filter(|loras| !loras.is_empty())
        .cloned()
    else {
        return Ok(());
    };
    let model_id = job_payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("Model is required for LoRA compatibility"))?;
    let models = model_catalog(state).await?;
    let catalog_loras = lora_catalog(state, project_id).await?;
    let normalized = validate_lora_specs_for_model(
        &models,
        &catalog_loras,
        model_id,
        &loras,
        allow_inline_loras,
        "LoRA",
    )?;
    job_payload.insert("loras".to_owned(), Value::Array(normalized));
    Ok(())
}

fn validate_lora_specs_for_model(
    models: &[Value],
    catalog_loras: &[Value],
    model_id: &str,
    attached_loras: &[Value],
    allow_inline_loras: bool,
    lora_label: &str,
) -> Result<Vec<Value>, ApiError> {
    if attached_loras.is_empty() {
        return Ok(Vec::new());
    }
    let Some(model) = models
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
    else {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} not found; cannot verify LoRA compatibility"
        )));
    };
    let model_families = model_lora_families(model);
    if model_families.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Model {model_id} has no declared LoRA families"
        )));
    }
    let mut normalized_loras = Vec::new();
    for attached_lora in attached_loras {
        let Some((lora_id, lora, normalized_lora, catalog_backed)) =
            hydrate_lora_spec(catalog_loras, attached_lora, allow_inline_loras, lora_label)?
        else {
            continue;
        };
        let install_state = lora.get("installState").and_then(Value::as_str);
        if install_state.is_some_and(|state| state != "installed")
            || (catalog_backed && install_state.is_none())
        {
            return Err(ApiError::bad_request(format!(
                "{lora_label} is not installed: {lora_id}"
            )));
        }
        let header = validate_lora_safetensors_header(lora_id, lora)?;
        if let Some(detected_family) = header.as_ref().and_then(detect_lora_family) {
            if !model_families
                .iter()
                .any(|model_family| model_family == &detected_family)
            {
                let model_family_list = model_families.join(", ");
                return Err(ApiError::bad_request(format!(
                    "LoRA {lora_id} appears to be a {detected_family} LoRA, which is not compatible with model {model_id} ({model_family_list})"
                )));
            }
        }
        let families = lora_families(lora);
        if families.is_empty() {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} has no declared family; cannot verify compatibility with model {model_id}"
            )));
        }
        if !families.iter().any(|family| {
            model_families
                .iter()
                .any(|model_family| model_family == family)
        }) {
            return Err(ApiError::bad_request(format!(
                "LoRA {lora_id} is not compatible with model {model_id}"
            )));
        }
        normalized_loras.push(normalized_lora);
    }
    Ok(normalized_loras)
}

fn hydrate_lora_spec<'a>(
    catalog_loras: &'a [Value],
    attached_lora: &'a Value,
    allow_inline_loras: bool,
    lora_label: &str,
) -> Result<Option<(&'a str, &'a Value, Value, bool)>, ApiError> {
    let Some(lora_id) = job_lora_id(attached_lora) else {
        return Ok(None);
    };
    let catalog_lora = if allow_inline_loras {
        None
    } else {
        catalog_loras
            .iter()
            .find(|lora| lora.get("id").and_then(Value::as_str) == Some(lora_id))
    };
    if catalog_lora.is_none() && !allow_inline_loras {
        return Err(ApiError::bad_request(format!(
            "{lora_label} not found: {lora_id}"
        )));
    }
    let source_lora = catalog_lora.unwrap_or(attached_lora);
    let normalized_lora = match catalog_lora {
        Some(catalog_lora) => serialize_job_lora(catalog_lora, attached_lora, lora_id),
        None => normalize_inline_job_lora(attached_lora, lora_id),
    };
    Ok(Some((
        lora_id,
        source_lora,
        normalized_lora,
        catalog_lora.is_some(),
    )))
}

/// Returns the parsed safetensors header for `lora` when one is available
/// on disk. Returns `Ok(None)` when the manifest entry has no installed
/// path or no `.safetensors` file is present under it (the same "skip"
/// semantics this helper has always had). Returns an error if the file
/// exists but the header is malformed.
fn validate_lora_safetensors_header(
    lora_id: &str,
    lora: &Value,
) -> Result<Option<Value>, ApiError> {
    let Some(path) = lora
        .get("installedPath")
        .or_else(|| lora.get("sourcePath"))
        .or_else(|| lora.get("path"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let Some(safetensors_path) = first_safetensors_path(&path) else {
        return Ok(None);
    };
    read_safetensors_header_for_api(lora_id, &safetensors_path).map(Some)
}

fn read_safetensors_header_for_api(lora_id: &str, path: &FsPath) -> Result<Value, ApiError> {
    read_safetensors_header(path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect LoRA {lora_id}: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request(format!("LoRA {lora_id} has an invalid safetensors header"))
        }
    })
}

fn model_lora_families(model: &Value) -> Vec<String> {
    families_from_value_chain(
        model,
        &["families", "compatibleFamilies", "modelFamilies"],
        Some("loraCompatibility"),
    )
}

fn families_from_value_chain(
    value: &Value,
    direct_fields: &[&str],
    compatibility_field: Option<&str>,
) -> Vec<String> {
    let compatibility = compatibility_field
        .and_then(|field| value.get(field))
        .unwrap_or(&Value::Null);
    let values = direct_fields
        .iter()
        .find_map(|field| value.get(*field).filter(|value| !value.is_null()))
        .or_else(|| {
            compatibility
                .get("families")
                .filter(|value| !value.is_null())
        })
        .or_else(|| value.get("family").filter(|value| !value.is_null()));
    let mut families = match values {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(normalize_lora_family)
            .collect(),
        Some(Value::String(value)) => vec![normalize_lora_family(value)],
        _ => Vec::new(),
    };
    families.sort();
    families.dedup();
    families
}

fn validate_recipe_preset_prompt(value: Option<&Value>) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_object()) {
        return Err(ApiError::bad_request(
            "Recipe preset prompt must be an object",
        ));
    }
    Ok(())
}

fn normalize_recipe_preset_loras(object: &mut JsonObject) -> Result<(), ApiError> {
    if !object.contains_key("loras") {
        if let Some(loras) = object.remove("builtInLoras") {
            object.insert("loras".to_owned(), loras);
        }
    } else {
        object.remove("builtInLoras");
    }
    let Some(loras) = object.get_mut("loras") else {
        return Ok(());
    };
    let items = loras
        .as_array_mut()
        .ok_or_else(|| ApiError::bad_request("Recipe preset loras must be an array"))?;
    if items.len() > 3 {
        return Err(ApiError::bad_request(
            "Recipe presets can include at most 3 LoRAs",
        ));
    }
    for item in items {
        if let Some(id) = item.as_str().map(str::to_owned) {
            *item = json!({ "id": id });
        }
        let object = item
            .as_object()
            .ok_or_else(|| ApiError::bad_request("Recipe preset LoRA must be an object"))?;
        validate_recipe_preset_id(object.get("id").and_then(Value::as_str))?;
        if let Some(lora_id) = object.get("loraId").and_then(Value::as_str) {
            validate_recipe_preset_id(Some(lora_id))?;
        }
        if object
            .get("compatibility")
            .is_some_and(|value| !value.is_object())
        {
            return Err(ApiError::bad_request(
                "Recipe preset LoRA compatibility must be an object",
            ));
        }
        if let Some(weight) = object.get("weight").and_then(Value::as_f64) {
            if !(-2.0..=2.0).contains(&weight) {
                return Err(ApiError::bad_request(
                    "Recipe preset LoRA weight must be between -2 and 2",
                ));
            }
        }
    }
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

fn job_lora_id(lora: &Value) -> Option<&str> {
    lora.as_str()
        .or_else(|| lora.get("id").and_then(Value::as_str))
        .or_else(|| lora.get("loraId").and_then(Value::as_str))
}

async fn catalog_delete_warnings(
    state: &AppState,
    kind: &str,
    id: &str,
    project_id: Option<&str>,
) -> Result<Vec<String>, ApiError> {
    let mut warnings = Vec::new();
    let presets = recipe_preset_catalog(state, project_id).await?;
    let preset_names = presets
        .iter()
        .filter(|preset| match kind {
            "model" => preset.get("model").and_then(Value::as_str) == Some(id),
            "lora" => recipe_preset_loras(preset)
                .iter()
                .any(|lora| job_lora_id(lora) == Some(id) || preset_lora_id(lora) == Some(id)),
            _ => false,
        })
        .filter_map(|preset| {
            preset
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| preset.get("id").and_then(Value::as_str))
        })
        .take(5)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !preset_names.is_empty() {
        warnings.push(format!(
            "Recipe presets reference this {kind}: {}",
            preset_names.join(", ")
        ));
    }

    let item_id = id.to_owned();
    let jobs = store_call(state.clone(), move |store, timeout| {
        store.mark_stale_workers_interrupted(timeout)?;
        store.list_jobs(None, None, 100)
    })
    .await?;
    let job_ids = jobs
        .iter()
        .filter(|job| job_references_catalog_item(job, kind, &item_id))
        .filter_map(|job| {
            if job.id.is_empty() {
                None
            } else {
                Some(job.id.clone())
            }
        })
        .take(5)
        .collect::<Vec<_>>();
    if !job_ids.is_empty() {
        warnings.push(format!(
            "Recent or queued jobs reference this {kind}: {}",
            job_ids.join(", ")
        ));
    }
    Ok(warnings)
}

fn job_references_catalog_item(job: &JobSnapshot, kind: &str, id: &str) -> bool {
    match kind {
        "model" => {
            job.payload.get("model").and_then(Value::as_str) == Some(id)
                || job.payload.get("modelId").and_then(Value::as_str) == Some(id)
        }
        "lora" => {
            job.payload.get("loraId").and_then(Value::as_str) == Some(id)
                || job
                    .payload
                    .get("loras")
                    .and_then(Value::as_array)
                    .is_some_and(|loras| loras.iter().any(|lora| job_lora_id(lora) == Some(id)))
        }
        _ => false,
    }
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
        "family": lora.get("family").cloned().unwrap_or(Value::Null),
        "families": lora.get("families").cloned().unwrap_or(Value::Null),
        "compatibleFamilies": lora.get("compatibleFamilies").cloned().unwrap_or(Value::Null),
        "modelFamilies": lora.get("modelFamilies").cloned().unwrap_or(Value::Null),
        "triggerWords": lora.get("triggerWords").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        "compatibility": lora.get("compatibility").cloned().unwrap_or_else(|| Value::Object(JsonObject::new())),
        "icLora": lora.get("icLora").cloned().unwrap_or(Value::Bool(false)),
        "conditioningRole": lora.get("conditioningRole").cloned().unwrap_or(Value::Null),
        "installedPath": lora.get("installedPath").cloned().unwrap_or(Value::Null),
        "source": lora.get("source").cloned().unwrap_or(Value::Null),
        "presetManaged": true
    })
}

fn serialize_job_lora(lora: &Value, selected_lora: &Value, lora_id: &str) -> Value {
    json!({
        "id": lora_id,
        "name": preferred_lora_str(selected_lora, lora, "name", lora_id),
        "scope": preferred_lora_str(selected_lora, lora, "scope", "global"),
        "weight": preset_lora_weight(lora, selected_lora),
        "family": preferred_lora_value(selected_lora, lora, "family"),
        "families": preferred_lora_value(selected_lora, lora, "families"),
        "compatibleFamilies": preferred_lora_value(selected_lora, lora, "compatibleFamilies"),
        "modelFamilies": preferred_lora_value(selected_lora, lora, "modelFamilies"),
        "triggerWords": preferred_lora_array(selected_lora, lora, "triggerWords"),
        "compatibility": preferred_lora_object(selected_lora, lora, "compatibility"),
        "icLora": preferred_lora_value(selected_lora, lora, "icLora"),
        "conditioningRole": preferred_lora_value(selected_lora, lora, "conditioningRole"),
        "installedPath": preferred_lora_value(selected_lora, lora, "installedPath"),
        "sourcePath": preferred_lora_value(selected_lora, lora, "sourcePath"),
        "source": preferred_lora_value(selected_lora, lora, "source"),
        "presetManaged": selected_lora.get("presetManaged").and_then(Value::as_bool).unwrap_or(false)
    })
}

fn preferred_lora_str<'a>(
    selected_lora: &'a Value,
    catalog_lora: &'a Value,
    field: &str,
    fallback: &'a str,
) -> &'a str {
    selected_lora
        .get(field)
        .and_then(Value::as_str)
        .or_else(|| catalog_lora.get(field).and_then(Value::as_str))
        .unwrap_or(fallback)
}

fn preferred_lora_value(selected_lora: &Value, catalog_lora: &Value, field: &str) -> Value {
    selected_lora
        .get(field)
        .filter(|value| !value.is_null())
        .or_else(|| catalog_lora.get(field))
        .cloned()
        .unwrap_or(Value::Null)
}

fn preferred_lora_array(selected_lora: &Value, catalog_lora: &Value, field: &str) -> Value {
    selected_lora
        .get(field)
        .filter(|value| value.is_array())
        .or_else(|| catalog_lora.get(field).filter(|value| value.is_array()))
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

fn preferred_lora_object(selected_lora: &Value, catalog_lora: &Value, field: &str) -> Value {
    selected_lora
        .get(field)
        .filter(|value| value.is_object())
        .or_else(|| catalog_lora.get(field).filter(|value| value.is_object()))
        .cloned()
        .unwrap_or_else(|| Value::Object(JsonObject::new()))
}

fn normalize_inline_job_lora(lora: &Value, lora_id: &str) -> Value {
    match lora {
        Value::Object(object) => {
            let mut object = object.clone();
            object.insert("id".to_owned(), Value::String(lora_id.to_owned()));
            Value::Object(object)
        }
        _ => json!({ "id": lora_id }),
    }
}

async fn load_manifest_entries(
    state: &AppState,
    path: &FsPath,
    field: &str,
) -> Result<Vec<Value>, ApiError> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to stat manifest {}: {error}",
                path.display()
            )))
        }
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

    let payload = tokio::fs::read_to_string(path).await.map_err(|error| {
        ApiError::internal(format!(
            "Failed to load manifest {}: {error}",
            path.display()
        ))
    })?;
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&payload)).map_err(|error| {
            ApiError::internal(format!(
                "Failed to parse manifest {}: {error}",
                path.display()
            ))
        })?;
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
    let downloads = model.get("downloads")?.as_array()?;
    let mut fallback = None;
    for download in downloads {
        if !is_supported_model_download(download) {
            continue;
        }
        fallback.get_or_insert(download);
        if download.get("default").and_then(Value::as_bool) == Some(true) {
            return Some(download.clone());
        }
    }
    fallback.cloned()
}

fn is_supported_model_download(download: &Value) -> bool {
    download.get("provider").and_then(Value::as_str) == Some("huggingface")
        && download
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| !repo.is_empty())
}

fn model_download_context(model: &Value) -> Result<Option<DownloadContext>, ApiError> {
    let Some(download) = model_download(model) else {
        return Ok(None);
    };
    Ok(Some(DownloadContext {
        repo: required_string_field(&download, "repo")?.to_owned(),
        files: string_array_field(&download, "files"),
        fallback_size_bytes: manifest_download_size_bytes(model, &download),
    }))
}

fn manifest_download_size_bytes(model: &Value, download: &Value) -> Option<u64> {
    // Prefer the selected download entry, then fall back to legacy model-level metadata.
    ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
        .iter()
        .find_map(|field| download.get(*field).and_then(json_size_to_u64))
        .or_else(|| {
            ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
                .iter()
                .find_map(|field| model.get(*field).and_then(json_size_to_u64))
        })
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

fn sanitized_upload_filename(filename: &str) -> String {
    let filename = filename
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(filename)
        .trim();
    let sanitized = safe_download_dir(filename);
    if sanitized.is_empty() || sanitized == "download" {
        "lora.safetensors".to_owned()
    } else {
        sanitized
    }
}

fn validate_lora_import_source_path(
    source_path: &str,
    allowed_roots: &[PathBuf],
) -> Result<(), ApiError> {
    let source = FsPath::new(source_path);
    if !source.is_absolute() {
        return Err(ApiError::bad_request("LoRA sourcePath must be absolute"));
    }
    let source = std::fs::canonicalize(source)
        .map_err(|_| ApiError::bad_request(format!("LoRA sourcePath not found: {source_path}")))?;
    let metadata = std::fs::metadata(&source)
        .map_err(|error| ApiError::bad_request(format!("Invalid LoRA sourcePath: {error}")))?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(ApiError::bad_request(
            "LoRA sourcePath must point to a file or directory",
        ));
    }
    for root in allowed_roots {
        if let Ok(root) = std::fs::canonicalize(root) {
            if source.starts_with(root) {
                return Ok(());
            }
        }
    }
    Err(ApiError::bad_request(
        "LoRA sourcePath must be inside app-managed data/loras, project/loras, or staged upload folders",
    ))
}

fn sweep_stale_lora_uploads(data_dir: &FsPath) -> std::io::Result<usize> {
    sweep_stale_lora_uploads_before(
        data_dir,
        SystemTime::now() - Duration::from_secs(STALE_LORA_UPLOAD_SECONDS),
    )
}

fn sweep_stale_lora_uploads_before(
    data_dir: &FsPath,
    cutoff: SystemTime,
) -> std::io::Result<usize> {
    let upload_root = data_dir.join("cache").join("lora-uploads");
    let entries = match std::fs::read_dir(upload_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut removed = 0usize;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() {
            continue;
        }
        let filename = entry.file_name();
        let filename = filename.to_string_lossy();
        if !filename.starts_with("upload-") {
            continue;
        }
        let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
        if modified <= cutoff {
            std::fs::remove_dir_all(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn validate_source_url(source_url: &str) -> Result<(), ApiError> {
    parse_lora_source_url(source_url)
        .map(|_| ())
        .map_err(|error| ApiError::bad_request(lora_url_error_message(error)))
}

fn lora_source_provider(payload: &LoraImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

fn lora_url_error_message(error: LoraUrlError) -> &'static str {
    error.message()
}

/// Parses the safetensors header at `source_path` (or the first
/// `.safetensors` file under it) and runs the architecture detector.
/// Returns `Ok(None)` when no header is available or the signature is
/// inconclusive. Returns `Err` only when the file exists but its header
/// is malformed — that mirrors the pre-existing validation behaviour and
/// gives the user a clear "the file is broken" message instead of a
/// silent acceptance.
fn detect_family_from_local_path(source_path: &str) -> Result<Option<String>, ApiError> {
    let path = FsPath::new(source_path);
    let Some(safetensors_path) = first_safetensors_path(path) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect LoRA file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("LoRA file has an invalid safetensors header".to_owned())
        }
    })?;
    Ok(detect_lora_family(&header))
}

/// Applies the import-time family policy: confident detection rejects a
/// mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection logs a warning and accepts the
/// supplied family unchanged.
fn reconcile_lora_family(
    supplied: Option<String>,
    detected: Option<String>,
    context: &str,
) -> Result<Option<String>, ApiError> {
    match (supplied, detected) {
        (Some(supplied), Some(detected)) => {
            if supplied == detected {
                Ok(Some(supplied))
            } else {
                Err(ApiError::bad_request(format!(
                    "LoRA file appears to be a {detected} model, but family was declared as {supplied}. Re-import with family {detected} or pick a different file."
                )))
            }
        }
        (None, Some(detected)) => Ok(Some(detected)),
        (Some(supplied), None) => {
            println!(
                "LoRA import {context}: architecture detection inconclusive; accepting supplied family {supplied}"
            );
            Ok(Some(supplied))
        }
        (None, None) => Ok(None),
    }
}

fn validate_lora_family(models: &[Value], family: &str) -> Result<String, ApiError> {
    let normalized = normalize_lora_family(family);
    if normalized.is_empty() {
        return Err(ApiError::bad_request(
            "LoRA family is required when provided",
        ));
    }
    let known = known_lora_families(models);
    if !known.is_empty() && !known.iter().any(|known_family| known_family == &normalized) {
        return Err(ApiError::bad_request(format!(
            "Unsupported LoRA family: {family}"
        )));
    }
    Ok(normalized)
}

fn normalize_lora_family(family: &str) -> String {
    family.trim().to_ascii_lowercase().replace('_', "-")
}

fn known_lora_families(models: &[Value]) -> Vec<String> {
    let mut families = Vec::new();
    for model in models {
        families.extend(model_lora_families(model));
    }
    families.sort();
    families.dedup();
    families
}

/// LoRA families accepted by installed models, read directly from the model
/// manifests. Unlike `known_lora_families(&model_catalog(..))`, this does no
/// Hugging Face size-estimation, so callers on hot/offline paths (the training
/// submit guardrail) stay local.
async fn known_lora_families_from_manifests(state: &AppState) -> Result<Vec<String>, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let mut models =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    models.extend(
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?,
    );
    Ok(known_lora_families(&models))
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

fn slugify_preset_id(value: &str) -> String {
    let id = slugify_lora_id(value);
    if id == "lora" {
        "preset".to_owned()
    } else {
        id
    }
}

fn next_duplicate_preset_id(entries: &[Value], base_id: &str) -> String {
    let base_id = base_id.trim().trim_end_matches("_copy");
    let first = format!("{base_id}_copy");
    if !preset_id_exists(entries, &first) {
        return first;
    }
    for index in 2.. {
        let candidate = format!("{base_id}_copy_{index}");
        if !preset_id_exists(entries, &candidate) {
            return candidate;
        }
    }
    unreachable!("infinite iterator should return a duplicate preset id")
}

fn preset_id_exists(entries: &[Value], id: &str) -> bool {
    entries
        .iter()
        .any(|entry| entry.get("id").and_then(Value::as_str) == Some(id))
}

fn next_duplicate_preset_name(entries: &[Value], base_name: &str) -> String {
    let first = format!("{base_name} Copy");
    if !preset_name_exists(entries, &first) {
        return first;
    }
    for index in 2.. {
        let candidate = format!("{base_name} Copy {index}");
        if !preset_name_exists(entries, &candidate) {
            return candidate;
        }
    }
    unreachable!("infinite iterator should return a duplicate preset name")
}

fn preset_name_exists(entries: &[Value], name: &str) -> bool {
    entries
        .iter()
        .any(|entry| entry.get("name").and_then(Value::as_str) == Some(name))
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

fn huggingface_hub_cache_dir(data_dir: &FsPath) -> PathBuf {
    if let Some(path) = std::env::var("HF_HUB_CACHE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var("HUGGINGFACE_HUB_CACHE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var("HF_HOME")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(path).join("hub");
    }
    data_dir.join("cache").join("huggingface").join("hub")
}

fn huggingface_repo_cache_path(data_dir: &FsPath, repo: &str) -> Option<PathBuf> {
    let safe_repo = repo
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character.to_string()
            } else {
                "--".to_owned()
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    if safe_repo.is_empty() {
        return None;
    }
    Some(huggingface_hub_cache_dir(data_dir).join(format!("models--{safe_repo}")))
}

fn huggingface_repo_cache_exists(path: &FsPath) -> bool {
    path.join("snapshots").is_dir() || path.join("blobs").is_dir()
}

struct MlxCatalogStatus {
    install_state: &'static str,
    conversion_state: &'static str,
    converted_path: Option<PathBuf>,
}

/// macOS Model Manager status for a model's `mlx` variant. Returns `None` when the
/// model declares no `mlx` block.
///
/// `conversion_state`:
/// - `ready`            turnkey MLX repo (no conversion needed)
/// - `converted`        requiresConversion and the local MLX dir exists
/// - `needs_conversion` source checkpoint present, MLX dir absent
/// - `needs_source`     source checkpoint not downloaded yet
///
/// `install_state` is `installed` when the usable MLX artifact exists.
fn mlx_catalog_status(
    model: &serde_json::Map<String, Value>,
    data_dir: &FsPath,
) -> Option<MlxCatalogStatus> {
    let mlx = model.get("mlx").and_then(Value::as_object)?;
    let repo_cached = |repo: &str| {
        huggingface_repo_cache_path(data_dir, repo)
            .as_deref()
            .is_some_and(huggingface_repo_cache_exists)
    };
    if mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let converted_dir = data_dir.join("models").join("mlx").join(model_id);
        if converted_dir.join("config.json").is_file() {
            return Some(MlxCatalogStatus {
                install_state: "installed",
                conversion_state: "converted",
                converted_path: Some(converted_dir),
            });
        }
        let source_present = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: "missing",
            conversion_state: if source_present {
                "needs_conversion"
            } else {
                "needs_source"
            },
            converted_path: None,
        })
    } else {
        let installed = mlx
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: if installed { "installed" } else { "missing" },
            conversion_state: "ready",
            converted_path: None,
        })
    }
}

fn lora_is_installed(path: &FsPath) -> bool {
    first_safetensors_path(path).is_some()
}

fn model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = model_manifest_installed_path(model, data_dir) {
        paths.push(path);
    }
    if let Some(repo) = model_download(model).and_then(|download| {
        download
            .get("repo")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }) {
        paths.push(data_dir.join("models").join(safe_download_dir(&repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, &repo) {
            paths.push(cache_path);
        }
    }
    if let Some(source_path) = model
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        paths.push(if path.is_absolute() {
            path
        } else {
            data_dir.join(path)
        });
    }
    unique_paths(paths)
}

fn lora_artifact_paths(lora: &Value, default_root: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let is_huggingface_source = lora
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)
        == Some("huggingface");
    if !is_huggingface_source {
        if let Some(installed_path) = lora
            .get("installedPath")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && !value.contains("${"))
        {
            paths.push(PathBuf::from(installed_path));
        }
    }
    if let Some(source_path) = lora
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .or_else(|| lora.get("path").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        paths.push(if path.is_absolute() {
            path
        } else {
            default_root.join(path)
        });
    }
    unique_paths(paths)
}

fn lora_huggingface_cached_file(lora: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let source = lora.get("source").and_then(Value::as_object);
    let provider = source
        .and_then(|source| source.get("provider"))
        .or_else(|| lora.get("provider"))
        .and_then(Value::as_str)?;
    if provider != "huggingface" {
        return None;
    }
    let repo = source
        .and_then(|source| source.get("repo"))
        .or_else(|| lora.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    if !repo_root.exists() {
        return None;
    }
    let file_name = source
        .and_then(|source| source.get("file"))
        .or_else(|| lora.get("file"))
        .and_then(Value::as_str)
        .or_else(|| {
            source
                .and_then(|source| source.get("files"))
                .or_else(|| lora.get("files"))
                .and_then(Value::as_array)
                .and_then(|files| files.first())
                .and_then(Value::as_str)
        });
    if let Some(file_name) = file_name {
        for snapshot in huggingface_snapshot_dirs(&repo_root) {
            let candidate = snapshot.join(file_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    huggingface_main_snapshot_dir(&repo_root)
        .and_then(|snapshot| first_safetensors_path(&snapshot))
        .or_else(|| first_safetensors_path(&repo_root))
}

fn huggingface_snapshot_dirs(repo_root: &FsPath) -> Vec<PathBuf> {
    let snapshots = repo_root.join("snapshots");
    let mut snapshot_dirs = std::fs::read_dir(&snapshots)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    snapshot_dirs.sort();
    if let Some(main_snapshot) = huggingface_main_snapshot_dir(repo_root) {
        let mut ordered = vec![main_snapshot.clone()];
        ordered.extend(
            snapshot_dirs
                .into_iter()
                .filter(|path| path != &main_snapshot),
        );
        return ordered;
    }
    snapshot_dirs
}

fn huggingface_main_snapshot_dir(repo_root: &FsPath) -> Option<PathBuf> {
    let revision = std::fs::read_to_string(repo_root.join("refs").join("main")).ok()?;
    let revision = revision.trim();
    if revision.is_empty() {
        return None;
    }
    let snapshot = repo_root.join("snapshots").join(revision);
    snapshot.is_dir().then_some(snapshot)
}

fn unique_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::new();
    for path in paths {
        if !unique.iter().any(|item| item == &path) {
            unique.push(path);
        }
    }
    unique
}

async fn remove_owned_artifact_path(
    path: PathBuf,
    allowed_roots: &[PathBuf],
    removed_paths: &mut Vec<String>,
    retained_paths: &mut Vec<String>,
) -> Result<(), ApiError> {
    let metadata = match tokio::fs::symlink_metadata(&path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to inspect artifact path {}: {error}",
                path.display()
            )))
        }
    };
    let canonical_path = tokio::fs::canonicalize(&path).await.map_err(|error| {
        ApiError::internal(format!(
            "Failed to resolve artifact path {}: {error}",
            path.display()
        ))
    })?;
    let mut owned = false;
    for root in allowed_roots {
        if let Ok(canonical_root) = tokio::fs::canonicalize(root).await {
            if canonical_path.starts_with(&canonical_root) && canonical_path != canonical_root {
                owned = true;
                break;
            }
        }
    }
    if !owned {
        retained_paths.push(path.display().to_string());
        return Ok(());
    }
    if metadata.is_dir() {
        tokio::fs::remove_dir_all(&path).await.map_err(|error| {
            ApiError::internal(format!(
                "Failed to remove artifact directory {}: {error}",
                path.display()
            ))
        })?;
    } else {
        tokio::fs::remove_file(&path).await.map_err(|error| {
            ApiError::internal(format!(
                "Failed to remove artifact file {}: {error}",
                path.display()
            ))
        })?;
    }
    removed_paths.push(path.display().to_string());
    Ok(())
}

fn model_manifest_installed_path(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let raw_path = model
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if raw_path.contains("${") {
        return None;
    }
    let path = PathBuf::from(raw_path);
    Some(if path.is_absolute() {
        path
    } else {
        data_dir.join(path)
    })
}

fn lora_families(lora: &Value) -> Vec<String> {
    families_from_value_chain(
        lora,
        &["families", "compatibleFamilies", "modelFamilies"],
        Some("compatibility"),
    )
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

fn bool_is_false(value: &bool) -> bool {
    !*value
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

fn env_path_or(name: &str, default: &FsPath) -> PathBuf {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default.to_path_buf())
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
            JobsStoreError::InvalidRequestedGpu(detail) => Self::bad_request(detail),
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
        create_app, huggingface_repo_cache_path, inprocess_worker_gpu_id, insufficient_disk_space,
        lora_artifact_paths, mlx_catalog_status, person_readiness_from_workers, requires_token,
        strip_jsonc_comments, sweep_stale_lora_uploads_before, EventHub, EventMessage, Settings,
        WorkerCapability, WorkerSnapshot, WorkerStatus, API_MANAGED_MANIFEST_HEADER,
        EVENT_BUFFER_SIZE, HEARTBEAT_SSE_DATA, HEARTBEAT_SSE_WIRE, TEST_MAX_LORA_UPLOAD_BYTES,
    };
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, SystemTime};
    use tokio_stream::StreamExt;
    use tower::ServiceExt;

    fn readiness_worker(
        id: &str,
        status: WorkerStatus,
        capabilities: Vec<WorkerCapability>,
    ) -> WorkerSnapshot {
        WorkerSnapshot {
            id: id.to_owned(),
            gpu_id: "0".to_owned(),
            gpu_name: None,
            status,
            current_job_id: None,
            capabilities,
            loaded_models: Vec::new(),
            utilization: None,
            registered_at: "2026-05-21T00:00:00Z".to_owned(),
            last_seen_at: "2026-05-21T00:00:00Z".to_owned(),
            extra: Default::default(),
        }
    }

    #[test]
    fn person_readiness_reflects_live_worker_capabilities() {
        let workers = vec![
            readiness_worker(
                "gpu",
                WorkerStatus::Idle,
                vec![
                    WorkerCapability::PersonDetect,
                    WorkerCapability::PersonTrack,
                    WorkerCapability::PersonReplace,
                ],
            ),
            readiness_worker(
                "cpu",
                WorkerStatus::Idle,
                vec![
                    WorkerCapability::PersonDetectPreview,
                    WorkerCapability::PersonTrackPreview,
                ],
            ),
            // Segment capability exists only on an offline worker -> not ready.
            readiness_worker(
                "dead",
                WorkerStatus::Offline,
                vec![WorkerCapability::PersonSegment],
            ),
        ];
        let readiness = person_readiness_from_workers(&workers);
        assert_eq!(readiness["detect"]["ready"], json!(true));
        assert_eq!(readiness["detect"]["capability"], json!("person_detect"));
        assert_eq!(readiness["track"]["ready"], json!(true));
        assert_eq!(readiness["replace"]["ready"], json!(true));
        assert_eq!(readiness["detectPreview"]["ready"], json!(true));
        assert_eq!(readiness["segment"]["ready"], json!(false));
    }

    fn test_settings(temp_dir: &tempfile::TempDir) -> Settings {
        Settings {
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
            run_utility_inprocess: false,
        }
    }

    #[test]
    fn mlx_catalog_status_reports_turnkey_and_conversion_states() {
        let temp = tempfile::tempdir().expect("tempdir");
        let data_dir = temp.path().join("data");

        // No `mlx` block -> None.
        let plain = json!({ "id": "z_image_turbo" });
        assert!(mlx_catalog_status(plain.as_object().unwrap(), &data_dir).is_none());

        // Turnkey model, repo not cached -> missing / ready.
        let ltx = json!({
            "id": "ltx_2_3",
            "mlx": { "minMemoryGb": 31, "repo": "notapalindrome/ltx23-mlx-av-q4" }
        });
        let status = mlx_catalog_status(ltx.as_object().unwrap(), &data_dir).expect("ltx status");
        assert_eq!(status.install_state, "missing");
        assert_eq!(status.conversion_state, "ready");

        // Turnkey model with the repo cached -> installed / ready.
        let repo_dir = huggingface_repo_cache_path(&data_dir, "notapalindrome/ltx23-mlx-av-q4")
            .expect("repo cache path");
        std::fs::create_dir_all(repo_dir.join("snapshots")).expect("create snapshots");
        let status = mlx_catalog_status(ltx.as_object().unwrap(), &data_dir).expect("ltx status");
        assert_eq!(status.install_state, "installed");
        assert_eq!(status.conversion_state, "ready");

        // Conversion model, native source missing -> missing / needs_source.
        let wan5b = json!({
            "id": "wan_2_2",
            "mlx": {
                "minMemoryGb": 45,
                "requiresConversion": true,
                "convertSourceRepo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            }
        });
        let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
        assert_eq!(status.install_state, "missing");
        assert_eq!(status.conversion_state, "needs_source");

        // Native source cached -> missing / needs_conversion.
        let source_dir = huggingface_repo_cache_path(&data_dir, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            .expect("source cache path");
        std::fs::create_dir_all(source_dir.join("snapshots")).expect("create source snapshots");
        let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
        assert_eq!(status.conversion_state, "needs_conversion");

        // Converted MLX dir present -> installed / converted.
        let converted = data_dir.join("models").join("mlx").join("wan_2_2");
        std::fs::create_dir_all(&converted).expect("create converted dir");
        std::fs::write(converted.join("config.json"), "{}").expect("write config");
        let status = mlx_catalog_status(wan5b.as_object().unwrap(), &data_dir).expect("wan status");
        assert_eq!(status.install_state, "installed");
        assert_eq!(status.conversion_state, "converted");
        assert_eq!(status.converted_path.unwrap(), converted);
    }

    fn write_test_safetensors(path: &std::path::Path) {
        std::fs::write(path, test_safetensors_bytes()).expect("test safetensors writes");
    }

    fn write_test_safetensors_with_keys(path: &std::path::Path, tensor_keys: &[String]) {
        std::fs::write(path, test_safetensors_bytes_with_keys(tensor_keys))
            .expect("test safetensors writes");
    }

    fn test_safetensors_bytes() -> Vec<u8> {
        test_safetensors_bytes_with_keys(&[])
    }

    fn test_safetensors_bytes_with_keys(tensor_keys: &[String]) -> Vec<u8> {
        let mut object = serde_json::Map::new();
        object.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for key in tensor_keys {
            object.insert(
                key.clone(),
                json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, 32768]}),
            );
        }
        let header = serde_json::to_vec(&Value::Object(object)).expect("header serializes");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(b"tensor-bytes");
        bytes
    }

    fn z_image_tensor_keys() -> Vec<String> {
        mm_dit_tensor_keys(24)
    }

    fn qwen_image_tensor_keys() -> Vec<String> {
        mm_dit_tensor_keys(60)
    }

    fn mm_dit_tensor_keys(block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "img_mlp.net.0.proj",
                "txt_mlp.net.0.proj",
            ] {
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        keys
    }

    fn wan_video_tensor_keys() -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
                keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
                keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
            }
        }
        keys
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

    async fn request_multipart_lora_upload(
        app: axum::Router,
        fields: &[(&str, &str)],
        filename: &str,
        bytes: &[u8],
    ) -> (StatusCode, Value) {
        let boundary = "SCENEWORKS_LORA_BOUNDARY";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let (status, _, bytes) = request_raw(
            app,
            "POST",
            "/api/v1/loras/import",
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

    #[test]
    fn stale_lora_upload_sweep_removes_only_upload_dirs_before_cutoff() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let upload_root = temp_dir.path().join("data/cache/lora-uploads");
        let expired = upload_root.join("upload-expired");
        let fresh = upload_root.join("upload-fresh");
        let unrelated = upload_root.join("keep-me");
        std::fs::create_dir_all(&expired).expect("expired dir creates");
        std::fs::create_dir_all(&fresh).expect("fresh dir creates");
        std::fs::create_dir_all(&unrelated).expect("unrelated dir creates");

        let removed = sweep_stale_lora_uploads_before(
            &temp_dir.path().join("data"),
            SystemTime::now() + Duration::from_secs(1),
        )
        .expect("stale uploads sweep");

        assert_eq!(removed, 2);
        assert!(!expired.exists());
        assert!(!fresh.exists());
        assert!(unrelated.exists());
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
    async fn canceling_queued_job_finishes_without_worker_acknowledgement() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");

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

        let job_id = created["id"].as_str().expect("job id is string");
        let (status, canceled) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/cancel"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(canceled["status"], "canceled");
        assert_eq!(canceled["stage"], "canceled");
        assert_eq!(canceled["progress"], 1.0);
        assert_eq!(canceled["cancelRequested"], true);
        assert_eq!(canceled["message"], "Canceled before a worker started.");
        assert!(canceled["canceledAt"].is_string());
        assert!(canceled["completedAt"].is_string());
        assert_eq!(canceled["workerId"], Value::Null);

        let (status, queue) = request(app.clone(), "GET", "/api/v1/queue", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queue["counts"]["canceled"], 1);
        assert_eq!(queue["counts"]["queued"], 0);

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

        let (status, claimed) = request(
            app,
            "POST",
            "/api/v1/jobs/claim",
            json!({ "workerId": "worker-1" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(claimed["job"], Value::Null);
    }

    #[tokio::test]
    async fn project_and_asset_routes_persist_contract_state() {
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
    async fn training_targets_route_returns_builtin_registry() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let app = create_app(settings).expect("app creates");

        let (status, registry) = request(app, "GET", "/api/v1/training/targets", Value::Null).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(registry["schemaVersion"], 1);
        assert_eq!(registry["targets"][0]["id"], "z_image_turbo_lora");
        assert_eq!(registry["targets"][0]["defaults"]["rank"], 16);
        assert_eq!(
            registry["targets"][0]["defaults"]["advanced"]["qualityPreset"],
            "balanced"
        );
    }

    #[tokio::test]
    async fn training_dataset_routes_persist_and_validate_project_assets() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let app = create_app(settings.clone()).expect("app creates");

        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();
        let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

        let (status, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Portrait.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        let (status, dataset) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Mira LoRA set",
                "items": [{
                    "assetId": asset_id,
                    "caption": {
                        "text": "miraStyle portrait",
                        "triggerWords": ["miraStyle"]
                    }
                }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(dataset["projectId"], project_id);
        assert_eq!(dataset["version"], 1);
        assert_eq!(dataset["items"][0]["assetId"], asset_id);
        assert_eq!(dataset["items"][0]["path"], "images/item_0001.png");
        assert_eq!(dataset["items"][0]["caption"]["source"], "manual");
        let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
        assert!(project_path
            .join("training")
            .join("datasets")
            .join(&dataset_id)
            .join("images")
            .join("item_0001.png")
            .exists());

        let (status, listed) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(listed[0]["id"], dataset_id);
        assert_eq!(listed[0]["itemCount"], 1);

        let reloaded_app = create_app(settings).expect("app reloads");
        let (status, detail) = request(
            reloaded_app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(detail["items"][0]["caption"]["text"], "miraStyle portrait");

        let (status, updated) = request(
            reloaded_app.clone(),
            "PATCH",
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
            json!({
                "items": [{
                    "assetId": asset_id,
                    "caption": { "text": "miraStyle close portrait" }
                }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["version"], 2);
        assert_eq!(
            updated["items"][0]["caption"]["text"],
            "miraStyle close portrait"
        );
        let dataset_image_path = project_path
            .join("training")
            .join("datasets")
            .join(&dataset_id)
            .join("images")
            .join("item_0001.png");
        assert_eq!(
            std::fs::read(&dataset_image_path).expect("dataset image remains"),
            b"png-bytes"
        );

        let (status, error) = request(
            reloaded_app.clone(),
            "PATCH",
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
            json!({
                "items": [
                    { "assetId": asset_id },
                    { "assetId": "asset_missing" }
                ]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error["detail"], "Asset not found");
        assert_eq!(
            std::fs::read(&dataset_image_path).expect("old dataset image survives failed update"),
            b"png-bytes"
        );
        let (status, detail_after_failed_update) = request(
            reloaded_app.clone(),
            "GET",
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(detail_after_failed_update["version"], 2);

        let (status, sidecars) = request(
            reloaded_app.clone(),
            "POST",
            &format!(
                "/api/v1/projects/{project_id}/training/datasets/{dataset_id}/caption-sidecars"
            ),
            json!({
                "items": [{
                    "itemId": "item_0001",
                    "caption": {
                        "text": "miraStyle studio portrait",
                        "triggerWords": ["miraStyle"]
                    }
                }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(sidecars["dataset"]["version"], 3);
        assert_eq!(
            sidecars["sidecars"][0]["captionPath"],
            format!("training/datasets/{dataset_id}/images/item_0001.txt")
        );
        assert_eq!(
            std::fs::read_to_string(
                project_path
                    .join("training")
                    .join("datasets")
                    .join(&dataset_id)
                    .join("images")
                    .join("item_0001.txt")
            )
            .expect("caption sidecar writes"),
            "miraStyle studio portrait\n"
        );

        let (status, renamed) = request(
            reloaded_app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/batch-rename"),
            json!({
                "items": [{
                    "itemId": "item_0001",
                    "newItemId": "item_0007",
                    "fileStem": "mira_0007",
                    "displayName": "mira_0007.png"
                }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(renamed["version"], 4);
        assert_eq!(renamed["items"][0]["id"], "item_0007");
        assert_eq!(renamed["items"][0]["path"], "images/mira_0007.png");
        assert_eq!(renamed["items"][0]["displayName"], "mira_0007.png");
        let renamed_image_path = project_path
            .join("training")
            .join("datasets")
            .join(&dataset_id)
            .join("images")
            .join("mira_0007.png");
        assert_eq!(
            std::fs::read(&renamed_image_path).expect("renamed dataset image remains"),
            b"png-bytes"
        );
        assert!(!dataset_image_path.exists());
        assert_eq!(
            std::fs::read_to_string(renamed_image_path.with_extension("txt"))
                .expect("caption sidecar follows rename"),
            "miraStyle studio portrait\n"
        );

        let (status, error) = request(
            reloaded_app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Bad Path",
                "items": [{ "path": "../outside.png" }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Invalid dataset item path");

        let (status, error) = request(
            reloaded_app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Duplicate Items",
                "items": [
                    { "id": "same_item", "assetId": asset_id },
                    { "id": "same_item", "assetId": asset_id }
                ]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["detail"], "Training dataset item IDs must be unique");

        let (_, other_project) = request(
            reloaded_app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Other Training Project" }),
        )
        .await;
        let other_project_id = other_project["id"].as_str().expect("project id").to_owned();
        let (status, other_asset) = request_multipart_upload(
            reloaded_app.clone(),
            &format!("/api/v1/projects/{other_project_id}/assets"),
            "Other.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let other_asset_id = other_asset["id"].as_str().expect("asset id").to_owned();
        let (status, error) = request(
            reloaded_app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Cross Project",
                "items": [{ "assetId": other_asset_id }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error["detail"], "Asset not found");

        let (status, deleted) = request(
            reloaded_app.clone(),
            "DELETE",
            &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted, json!({ "id": dataset_id, "status": "deleted" }));
        let (status, listed_after_delete) = request(
            reloaded_app,
            "GET",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(listed_after_delete, json!([]));
    }

    #[tokio::test]
    async fn create_training_job_resolves_plan_and_queues_lora_train() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let app = create_app(settings).expect("app creates");

        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();
        let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Portrait.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        let (_, dataset) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Aurora set",
                "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
            }),
        )
        .await;
        let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

        let (_, registry) =
            request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
        let target = registry["targets"][0].clone();
        let target_id = target["id"].as_str().expect("target id").to_owned();
        let config = target["defaults"].clone();

        let (status, job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": dataset_id,
                "config": config,
                "outputName": "Aurora Style",
                "dryRun": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(job["type"], "lora_train");
        assert_eq!(job["status"], "queued");
        assert_eq!(job["requestedGpu"], "auto");
        assert_eq!(job["projectId"], project_id);
        assert_eq!(job["payload"]["dryRun"], true);

        let job_id = job["id"].as_str().expect("job id").to_owned();
        let plan = &job["payload"]["plan"];
        // The resolved plan is self-referential and fully normalized in Rust.
        assert_eq!(plan["jobId"], job_id);
        assert_eq!(plan["provenance"]["sourceJobId"], job_id);
        assert_eq!(plan["target"]["targetId"], target_id);
        assert_eq!(plan["dataset"]["datasetId"], dataset_id);
        assert_eq!(plan["dataset"]["datasetVersion"], 1);
        assert_eq!(plan["dataset"]["items"].as_array().unwrap().len(), 1);
        let lora_id = plan["output"]["loraId"].as_str().expect("lora id");
        assert!(lora_id.starts_with("lora_"));
        assert_eq!(plan["output"]["fileName"], "aurora_style.safetensors");

        // Item image paths resolve under the dataset root on disk.
        let expected_image = project_path
            .join("training")
            .join("datasets")
            .join(&dataset_id)
            .join("images")
            .join("item_0001.png");
        assert_eq!(
            plan["dataset"]["items"][0]["imagePath"]
                .as_str()
                .expect("image path"),
            expected_image.display().to_string()
        );
        // The default target scope is `project`, so the adapter is written into
        // the project's LoRA store (not the shared data dir).
        assert_eq!(
            plan["output"]["outputDir"].as_str().expect("output dir"),
            project_path
                .join("loras")
                .join(lora_id)
                .display()
                .to_string()
        );
        // The submit-time manifest entry carries provenance for the LoRA that
        // registration will recompute and upsert on completion. The manifest path
        // itself is intentionally NOT persisted in the payload — it is recomputed
        // from trusted inputs at completion so a tampered payload cannot redirect
        // the write.
        assert_eq!(job["payload"]["manifestEntry"]["scope"], "project");
        assert_eq!(job["payload"]["manifestEntry"]["family"], "z-image");
        assert_eq!(
            job["payload"]["manifestEntry"]["source"]["path"],
            format!("loras/{lora_id}")
        );
        assert_eq!(
            job["payload"]["manifestEntry"]["provenance"]["datasetId"],
            dataset_id
        );
        assert_eq!(
            job["payload"]["manifestEntry"]["provenance"]["trainingJobId"],
            job_id
        );
        assert!(job["payload"]["manifestPath"].is_null());

        // The job is queued and visible to the queue/worker surface.
        let (status, queued) = request(
            app.clone(),
            "GET",
            "/api/v1/jobs?status=queued",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queued[0]["id"], job_id);
        assert_eq!(queued[0]["type"], "lora_train");
    }

    #[tokio::test]
    async fn create_training_job_rejects_unknown_target_and_missing_dataset() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();

        let (_, registry) =
            request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
        let config = registry["targets"][0]["defaults"].clone();

        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": "not_a_target",
                "datasetId": "ds_missing",
                "config": config,
                "outputName": "Aurora"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error["detail"]
            .as_str()
            .unwrap()
            .contains("Unknown training target"));

        let target_id = registry["targets"][0]["id"].as_str().unwrap().to_owned();
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": "ds_missing",
                "config": registry["targets"][0]["defaults"].clone(),
                "outputName": "Aurora"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error["detail"], "Training dataset not found");
    }

    #[tokio::test]
    async fn create_training_job_queues_real_run_when_not_dry_run() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let app = create_app(settings.clone()).expect("app creates");
        // A real run requires the base model installed (story 1419 guardrail).
        seed_installed_base_model(&settings.data_dir);
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();

        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Portrait.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        let (_, dataset) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Aurora set",
                "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
            }),
        )
        .await;
        let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

        let (_, registry) =
            request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
        let target = registry["targets"][0].clone();
        let target_id = target["id"].as_str().expect("target id").to_owned();
        let config = target["defaults"].clone();

        // Real execution exists (story 1417): a non-dry-run job resolves the same
        // plan and queues for the worker's Z-Image LoRA kernel.
        let (status, job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": dataset_id,
                "config": config,
                "outputName": "Aurora Style",
                "dryRun": false
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(job["type"], "lora_train");
        assert_eq!(job["status"], "queued");
        assert_eq!(job["payload"]["dryRun"], false);
        // The plan is resolved and embedded just like the dry-run path.
        assert_eq!(job["payload"]["plan"]["planVersion"], 1);
        assert_eq!(job["payload"]["plan"]["target"]["kernel"], "z_image_lora");

        let job_id = job["id"].as_str().expect("job id").to_owned();
        let (status, queued) = request(
            app.clone(),
            "GET",
            "/api/v1/jobs?status=queued",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(queued[0]["id"], job_id);
        assert_eq!(queued[0]["type"], "lora_train");
    }

    /// Seeds the Z-Image-Turbo base model as installed (a managed-download
    /// marker) so a real training run clears the missing-model guardrail.
    fn seed_installed_base_model(data_dir: &std::path::Path) {
        let model_dir = data_dir.join("models").join("z_image_turbo");
        std::fs::create_dir_all(&model_dir).expect("model dir creates");
        std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("model marker writes");
    }

    /// Drives a project-scoped training job from submission to a completed result
    /// and asserts the produced adapter is registered as a normal SceneWorks LoRA.
    /// Seeds the base model so the real-run guardrails pass.
    async fn submit_real_training_job(
        app: axum::Router,
        project_id: &str,
        data_dir: &std::path::Path,
    ) -> (String, std::path::PathBuf, std::path::PathBuf) {
        seed_installed_base_model(data_dir);
        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Portrait.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();

        let (_, dataset) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Aurora set",
                "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
            }),
        )
        .await;
        let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();

        let (_, registry) =
            request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
        let target_id = registry["targets"][0]["id"]
            .as_str()
            .expect("target id")
            .to_owned();
        let mut config = registry["targets"][0]["defaults"].clone();
        // A trigger word flows from the config into the plan and the LoRA entry.
        config["triggerWord"] = json!("auroraStyle");

        let (status, job) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": dataset_id,
                "config": config,
                "outputName": "Aurora Style",
                "dryRun": false
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let job_id = job["id"].as_str().expect("job id").to_owned();
        let output_dir = std::path::PathBuf::from(
            job["payload"]["plan"]["output"]["outputDir"]
                .as_str()
                .unwrap(),
        );
        let file_name = job["payload"]["plan"]["output"]["fileName"]
            .as_str()
            .unwrap()
            .to_owned();
        let adapter_path = output_dir.join(file_name);
        (job_id, output_dir, adapter_path)
    }

    #[tokio::test]
    async fn completed_training_job_registers_lora_with_provenance() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let app = create_app(settings.clone()).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();

        let (job_id, output_dir, adapter_path) =
            submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;

        // The worker writes the final adapter into the resolved output dir before
        // it reports completion, alongside step checkpoints it does not clean up.
        // Registration must pick the declared final adapter, not a checkpoint.
        std::fs::create_dir_all(&output_dir).expect("output dir creates");
        write_test_safetensors(&adapter_path);
        let final_name = adapter_path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("final adapter name")
            .to_owned();
        let stem = adapter_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("adapter stem");
        write_test_safetensors(&output_dir.join(format!("{stem}-step000250.safetensors")));

        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Trained LoRA saved.",
                "result": { "outputPath": adapter_path.display().to_string() }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(completed["status"], "completed");
        // The registration outcome is folded into the job result so it is
        // observable (rather than silently dropped on failure).
        assert_eq!(completed["result"]["loraRegistered"], true);
        assert!(completed["result"]["loraId"]
            .as_str()
            .is_some_and(|id| id.starts_with("lora_")));
        assert!(completed["result"]["loraManifestPath"]
            .as_str()
            .is_some_and(|path| path.ends_with("manifest.jsonc")));

        // The trained adapter is now a normal, installed, project-scoped LoRA.
        let (status, loras) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/loras?projectId={project_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let entry = loras
            .as_array()
            .expect("loras array")
            .iter()
            .find(|item| item["name"] == json!("Aurora Style"))
            .expect("trained LoRA appears in catalog")
            .clone();
        assert_eq!(entry["scope"], "project");
        assert_eq!(entry["family"], "z-image");
        assert_eq!(entry["baseModel"], "z_image_turbo");
        assert_eq!(entry["triggerWords"], json!(["auroraStyle"]));
        assert_eq!(entry["installState"], "installed");
        // The final adapter is registered, not the step checkpoint that shares
        // the output directory.
        assert_eq!(entry["files"], json!([final_name]));
        // installedPath resolves to the trained adapter's directory (the same
        // convention as imported LoRAs), and the adapter file lives under it.
        let lora_id = entry["id"].as_str().expect("lora id");
        let installed_path = entry["installedPath"].as_str().expect("installed path");
        assert!(
            installed_path.contains(lora_id),
            "installed path {installed_path} should point at the trained adapter dir"
        );
        assert!(adapter_path.exists());
        assert_eq!(entry["source"]["provider"], "training");
        assert_eq!(entry["provenance"]["trainingJobId"], job_id);
        assert!(entry["provenance"]["configSnapshot"].is_object());

        // Provenance survives an app restart (manifest is on disk).
        let reloaded = create_app(settings).expect("app reloads");
        let (status, loras) = request(
            reloaded,
            "GET",
            &format!("/api/v1/loras?projectId={project_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(loras
            .as_array()
            .expect("loras array")
            .iter()
            .any(|item| item["name"] == json!("Aurora Style")
                && item["provenance"]["trainingJobId"] == json!(job_id)));
    }

    #[tokio::test]
    async fn failed_or_unwritten_training_job_registers_no_lora() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let settings = test_settings(&temp_dir);
        let app = create_app(settings.clone()).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();

        // A failed job never registers, even though a manifest entry was staged.
        let (failed_job_id, _output_dir, _adapter_path) =
            submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
        let (status, _) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{failed_job_id}/progress"),
            json!({
                "status": "failed",
                "stage": "failed",
                "progress": 1,
                "message": "Training failed.",
                "error": "CUDA out of memory"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // A job that reports completed but produced no weights must not leave a
        // broken registry entry either, and the failure is surfaced in the result.
        let (completed_no_weights_id, _, _) =
            submit_real_training_job(app.clone(), &project_id, &settings.data_dir).await;
        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{completed_no_weights_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Reported complete without weights."
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(completed["result"]["loraRegistered"], false);
        assert!(completed["result"]["loraRegistrationError"].is_string());

        let (status, loras) = request(
            app,
            "GET",
            &format!("/api/v1/loras?projectId={project_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(loras
            .as_array()
            .expect("loras array")
            .iter()
            .all(|item| item["name"] != json!("Aurora Style")));
    }

    #[tokio::test]
    async fn crafted_training_job_cannot_register_outside_canonical_manifest() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();
        let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());

        // A `lora_train` job can be crafted directly via the generic job endpoint
        // with an attacker-chosen payload. Stage a real adapter under the canonical
        // project output dir so registration would succeed if (and only if) it uses
        // the recomputed path.
        let crafted_lora_id = "lora_crafted01";
        let adapter_dir = project_path.join("loras").join(crafted_lora_id);
        std::fs::create_dir_all(&adapter_dir).expect("adapter dir creates");
        write_test_safetensors(&adapter_dir.join("crafted.safetensors"));

        // The payload points the manifest write and the source path at locations
        // outside the canonical project manifest. Both must be ignored.
        let evil_manifest = temp_dir.path().join("evil-manifest.jsonc");
        let (status, job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "lora_train",
                "projectId": project_id,
                "projectName": "Training Project",
                "requestedGpu": "auto",
                "payload": {
                    "dryRun": false,
                    "manifestPath": evil_manifest.display().to_string(),
                    "manifestEntry": {
                        "id": crafted_lora_id,
                        "name": "Crafted LoRA",
                        "scope": "project",
                        "family": "z-image",
                        "source": { "provider": "evil", "path": "../../../../escape/loras" },
                        "files": ["crafted.safetensors"]
                    }
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let job_id = job["id"].as_str().expect("job id").to_owned();

        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Crafted completion."
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // The write went to the canonical project manifest, not the payload path.
        assert!(
            !evil_manifest.exists(),
            "payload manifestPath must be ignored"
        );
        assert_eq!(completed["result"]["loraRegistered"], true);
        assert_eq!(
            completed["result"]["loraManifestPath"]
                .as_str()
                .expect("manifest path"),
            project_path
                .join("loras")
                .join("manifest.jsonc")
                .display()
                .to_string()
        );

        // The registered entry's source path was recomputed, not taken from the
        // attacker payload.
        let (status, loras) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/loras?projectId={project_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let entry = loras
            .as_array()
            .expect("loras array")
            .iter()
            .find(|item| item["id"] == json!(crafted_lora_id))
            .expect("crafted LoRA registered under canonical manifest")
            .clone();
        assert_eq!(entry["scope"], "project");
        assert_eq!(entry["source"]["provider"], "training");
        assert_eq!(entry["source"]["path"], format!("loras/{crafted_lora_id}"));
        // `files` was validated against the recomputed output dir (the declared
        // name exists there), so the registered entry points only inside the
        // canonical LoRA directory.
        assert_eq!(entry["files"], json!(["crafted.safetensors"]));

        // A traversal id is rejected outright: nothing registers and the failure
        // is visible.
        let (_, evil_job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "lora_train",
                "projectId": project_id,
                "projectName": "Training Project",
                "requestedGpu": "auto",
                "payload": {
                    "dryRun": false,
                    "manifestEntry": {
                        "id": "../../pwned",
                        "name": "Traversal",
                        "scope": "project",
                        "source": { "provider": "evil", "path": "loras/x" }
                    }
                }
            }),
        )
        .await;
        let evil_job_id = evil_job["id"].as_str().expect("job id").to_owned();
        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{evil_job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Traversal completion."
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(completed["result"]["loraRegistered"], false);
        assert!(completed["result"]["loraRegistrationError"].is_string());

        // A `..`-traversing files entry is rejected even when a valid adapter
        // exists under the canonical output dir: registration only accepts plain
        // in-tree file names, so generation can never be pointed outside it.
        let traversal_lora_id = "lora_filestrav01";
        let traversal_dir = project_path.join("loras").join(traversal_lora_id);
        std::fs::create_dir_all(&traversal_dir).expect("adapter dir creates");
        write_test_safetensors(&traversal_dir.join("real.safetensors"));
        let (_, files_job) = request(
            app.clone(),
            "POST",
            "/api/v1/jobs",
            json!({
                "type": "lora_train",
                "projectId": project_id,
                "projectName": "Training Project",
                "requestedGpu": "auto",
                "payload": {
                    "dryRun": false,
                    "manifestEntry": {
                        "id": traversal_lora_id,
                        "name": "Files Traversal",
                        "scope": "project",
                        "source": { "provider": "evil", "path": "loras/x" },
                        "files": ["../../../../escape/evil.safetensors"]
                    }
                }
            }),
        )
        .await;
        let files_job_id = files_job["id"].as_str().expect("job id").to_owned();
        let (status, completed) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{files_job_id}/progress"),
            json!({
                "status": "completed",
                "stage": "completed",
                "progress": 1,
                "message": "Files traversal completion."
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(completed["result"]["loraRegistered"], false);
        assert!(completed["result"]["loraRegistrationError"].is_string());
    }

    #[tokio::test]
    async fn real_training_job_rejected_when_base_model_missing() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();

        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Portrait.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();
        let (_, dataset) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Aurora set",
                "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
            }),
        )
        .await;
        let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
        let (_, registry) =
            request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
        let target_id = registry["targets"][0]["id"]
            .as_str()
            .expect("target id")
            .to_owned();
        let config = registry["targets"][0]["defaults"].clone();

        // No base model is installed: a real run is rejected with an actionable
        // message, but a dry run (plan preview) still succeeds.
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": dataset_id,
                "config": config,
                "outputName": "Aurora Style",
                "dryRun": false
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error["detail"]
            .as_str()
            .unwrap()
            .contains("is not installed"));

        let (status, _) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": dataset_id,
                "config": config,
                "outputName": "Aurora Style",
                "dryRun": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
    }

    #[tokio::test]
    async fn training_job_rejects_cpu_target() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Training Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id").to_owned();

        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            "Portrait.PNG",
            "image/png",
            b"png-bytes",
        )
        .await;
        let asset_id = asset["id"].as_str().expect("asset id").to_owned();
        let (_, dataset) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/datasets"),
            json!({
                "name": "Aurora set",
                "items": [{ "assetId": asset_id, "caption": { "text": "auroraStyle portrait" } }]
            }),
        )
        .await;
        let dataset_id = dataset["id"].as_str().expect("dataset id").to_owned();
        let (_, registry) =
            request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
        let target_id = registry["targets"][0]["id"]
            .as_str()
            .expect("target id")
            .to_owned();
        let mut config = registry["targets"][0]["defaults"].clone();
        // Targeting a CPU worker for a GPU-only job is rejected with an
        // actionable message (a dry run is GPU-routed too, so this also holds).
        config["advanced"]["requestedGpu"] = json!("cpu");

        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/training/jobs"),
            json!({
                "targetId": target_id,
                "datasetId": dataset_id,
                "config": config,
                "outputName": "Aurora Style",
                "dryRun": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error["detail"]
            .as_str()
            .unwrap()
            .contains("cannot target CPU workers"));
    }

    #[test]
    fn insufficient_disk_space_threshold_is_strict_less_than() {
        assert!(insufficient_disk_space(100, 200));
        assert!(!insufficient_disk_space(200, 200));
        assert!(!insufficient_disk_space(300, 200));
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
    async fn person_tracking_routes_match_contracts() {
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
    async fn generation_job_routes_reject_incompatible_loras() {
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
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "edit_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                },
                {
                  "id": "ltx_2_3",
                  "name": "LTX",
                  "family": "ltx-video",
                  "type": "video",
                  "adapter": "ltx_video",
                  "capabilities": ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["ltx-video"] },
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                }
              ]
            }
            "#,
        )
        .expect("user loras writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "bad_qwen",
                  "name": "Bad Qwen",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo",
                  "loras": [{ "id": "qwen_style" }]
                }
              ]
            }
            "#,
        )
        .expect("builtin recipe presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("user recipe presets writes");
        let lora_dir = temp_dir.path().join("data/loras");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        write_test_safetensors(&lora_dir.join("qwen.safetensors"));

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Compatibility" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id");
        let (status, image_error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "prompt": "mist",
                "model": "z_image_turbo",
                "loras": [{ "id": "qwen_style" }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            image_error["detail"],
            "LoRA qwen_style is not compatible with model z_image_turbo"
        );

        let (status, unknown_model_error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "prompt": "mist",
                "model": "missing_model",
                "loras": [{ "id": "qwen_style" }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            unknown_model_error["detail"],
            "Model missing_model not found; cannot verify LoRA compatibility"
        );

        let (status, preset_error) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "prompt": "mist",
                "model": "z_image_turbo",
                "recipePresetId": "bad_qwen"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            preset_error["detail"],
            "LoRA qwen_style is not compatible with model z_image_turbo"
        );

        for (mode, extra) in [
            ("image_to_video", json!({ "sourceAssetId": "asset-image" })),
            ("text_to_video", json!({})),
            (
                "first_last_frame",
                json!({ "sourceAssetId": "asset-first", "lastFrameAssetId": "asset-last" }),
            ),
            ("extend_clip", json!({ "sourceClipAssetId": "asset-video" })),
            (
                "video_bridge",
                json!({ "sourceClipAssetId": "asset-left", "bridgeRightClipAssetId": "asset-right" }),
            ),
            (
                "replace_person",
                json!({ "sourceClipAssetId": "asset-video", "personTrackId": "track-1", "characterId": "character-1" }),
            ),
        ] {
            let mut payload = json!({
                "projectId": project_id,
                "mode": mode,
                "prompt": "motion",
                "model": "ltx_2_3",
                "loras": [{ "id": "qwen_style" }]
            });
            payload
                .as_object_mut()
                .expect("video payload object")
                .extend(extra.as_object().expect("extra payload object").clone());
            let (status, video_error) =
                request(app.clone(), "POST", "/api/v1/video/jobs", payload).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{mode}");
            assert_eq!(
                video_error["detail"],
                "LoRA qwen_style is not compatible with model ltx_2_3"
            );
        }

        let (_, character) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters"),
            json!({ "name": "Mira", "type": "person" }),
        )
        .await;
        let character_id = character["id"].as_str().expect("character id");
        let character_lora = temp_dir
            .path()
            .join("data/loras/character-qwen.safetensors");
        write_test_safetensors(&character_lora);
        let (status, _) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
            json!({
                "name": "Character Qwen",
                "sourcePath": character_lora.display().to_string(),
                "compatibility": { "families": ["qwen-image"] }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        let (status, character_error) = request(
            app,
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/test-jobs"),
            json!({ "prompt": "portrait", "model": "z_image_turbo" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(character_error["detail"]
            .as_str()
            .unwrap()
            .contains("is not compatible with model z_image_turbo"));
    }

    #[tokio::test]
    async fn model_and_lora_routes_match_manifest_behavior() {
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
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [
                    { "provider": "huggingface", "repo": "owner/alternate-model", "files": ["*.bin"], "estimatedSizeBytes": 536870912 },
                    { "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"], "default": true, "estimatedSizeBytes": 12884901888 }
                  ],
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
                  "workflow": "text_to_image",
                  "model": "base-model",
                  "defaults": { "count": 4, "resolution": "1280x720", "negativePrompt": "flat lighting" },
                  "prompt": { "suffix": "cinematic lighting" },
                  "loras": [{ "id": "style-lora", "weight": 0.5 }]
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
                { "id": "cinematic", "name": "My Cinematic", "defaults": { "count": 2, "resolution": "1280x720", "negativePrompt": "flat lighting" } },
                { "id": "legacy_edit", "name": "Legacy Edit", "modes": ["edit_image"], "builtInLoras": [{ "id": "style-lora", "weight": 0.25 }] }
              ]
            }
            "#,
        )
        .expect("user recipe presets writes");
        let marker_dir = temp_dir.path().join("data/models/owner__model");
        std::fs::create_dir_all(&marker_dir).expect("model dir creates");
        std::fs::write(marker_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("marker writes");
        let lora_dir = temp_dir.path().join("data/loras");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        write_test_safetensors(&lora_dir.join("style.safetensors"));

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(models[0]["name"], "User Model");
        assert_eq!(models[0]["adapter"], "z_image_diffusers");
        assert_eq!(models[0]["downloadable"], true);
        assert_eq!(models[0]["downloadSizeBytes"], 12884901888_u64);
        assert_eq!(models[0]["downloadSizeLabel"], "12.0 GB");
        assert_eq!(models[0]["downloadSizeEstimated"], true);
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
        let presets = presets.as_array().unwrap();
        assert_eq!(presets.len(), 2);
        let cinematic = presets
            .iter()
            .find(|preset| preset["id"] == "cinematic")
            .expect("cinematic preset");
        assert_eq!(cinematic["name"], "My Cinematic");
        assert_eq!(cinematic["scope"], "global");
        assert_eq!(cinematic["workflow"], "text_to_image");
        assert_eq!(cinematic["defaults"]["count"], 2);
        assert_eq!(cinematic["loras"][0]["id"], "style-lora");
        assert_eq!(cinematic["builtInLoras"][0]["id"], "style-lora");
        let legacy_edit = presets
            .iter()
            .find(|preset| preset["id"] == "legacy_edit")
            .expect("legacy edit preset");
        assert_eq!(legacy_edit["workflow"], "edit_image");
        assert_eq!(legacy_edit["model"], "base-model");
        assert_eq!(legacy_edit["loras"][0]["id"], "style-lora");
        assert_eq!(
            legacy_edit["appliedDefaults"]["notes"][0],
            "workflow inferred from legacy modes as edit_image"
        );
        assert_eq!(
            legacy_edit["appliedDefaults"]["notes"][1],
            "model defaulted to base-model for legacy preset"
        );
        assert_eq!(
            legacy_edit["appliedDefaults"]["notes"][2],
            "builtInLoras migrated to loras"
        );

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
                "model": "base-model",
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
        assert!(image_job["payload"]["loras"][0]["installedPath"]
            .as_str()
            .is_some_and(|value| value.ends_with("data/loras/style.safetensors")
                || value.ends_with("data\\loras\\style.safetensors")
                || value.ends_with("loras/style.safetensors")
                || value.ends_with("loras\\style.safetensors")));
        assert_eq!(
            image_job["payload"]["loras"][0]["source"]["path"],
            "loras/style.safetensors"
        );
        assert_eq!(image_job["payload"]["model"], "base-model");
        assert_eq!(image_job["payload"]["loras"][0]["family"], "z-image");
        assert_eq!(
            image_job["payload"]["loras"][0]["compatibility"]["families"][0],
            "z-image"
        );
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

        let (status, null_path_job) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "prompt": "city with selected lora",
                "model": "base-model",
                "count": 1,
                "width": 512,
                "height": 512,
                "loras": [{
                    "id": "style-lora",
                    "name": null,
                    "triggerWords": null,
                    "compatibility": null,
                    "installedPath": null,
                    "sourcePath": null
                }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(null_path_job["payload"]["loras"][0]["id"], "style-lora");
        assert_eq!(null_path_job["payload"]["loras"][0]["name"], "Style LoRA");
        assert_eq!(
            null_path_job["payload"]["loras"][0]["triggerWords"][0],
            "style"
        );
        assert_eq!(
            null_path_job["payload"]["loras"][0]["compatibility"]["families"][0],
            "z-image"
        );
        assert!(null_path_job["payload"]["loras"][0]["installedPath"]
            .as_str()
            .is_some_and(|value| value.ends_with("data/loras/style.safetensors")
                || value.ends_with("data\\loras\\style.safetensors")
                || value.ends_with("loras/style.safetensors")
                || value.ends_with("loras\\style.safetensors")));

        let (status, preset_model_job) = request(
            app.clone(),
            "POST",
            "/api/v1/image/jobs",
            json!({
                "projectId": project_id,
                "prompt": "city at dawn",
                "count": 1,
                "width": 512,
                "height": 512,
                "recipePresetId": "cinematic"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(preset_model_job["payload"]["model"], "base-model");

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
        assert_eq!(job["payload"]["repo"], "owner/model");
        assert_eq!(job["payload"]["files"][0], "*.safetensors");
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

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, url_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourceUrl": "https://example.com/loras/detail.safetensors",
                "name": "Detail LoRA",
                "family": "z-image"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(url_job["type"], "lora_import");
        assert_eq!(
            url_job["payload"]["sourceUrl"],
            "https://example.com/loras/detail.safetensors"
        );
        assert_eq!(url_job["payload"]["loraId"], "detail_lora");
        assert_eq!(
            url_job["payload"]["manifestEntry"]["source"]["provider"],
            "url"
        );
        assert_eq!(
            url_job["payload"]["manifestEntry"]["source"]["url"],
            "https://example.com/loras/detail.safetensors"
        );
        assert_eq!(url_job["payload"]["manifestEntry"]["family"], "z-image");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let upload_bytes = test_safetensors_bytes();
        let (status, upload_job) = request_multipart_lora_upload(
            app,
            &[
                ("name", "Uploaded Detail"),
                ("scope", "global"),
                ("family", "z-image"),
            ],
            "detail.safetensors",
            &upload_bytes,
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(upload_job["type"], "lora_import");
        assert_eq!(upload_job["payload"]["loraId"], "uploaded_detail");
        assert_eq!(upload_job["payload"]["uploadedSourcePath"], true);
        assert_eq!(
            upload_job["payload"]["manifestEntry"]["source"]["provider"],
            "local"
        );
        assert_eq!(
            upload_job["payload"]["manifestEntry"]["files"][0],
            "detail.safetensors"
        );
        let source_path = std::path::PathBuf::from(
            upload_job["payload"]["sourcePath"]
                .as_str()
                .expect("source path"),
        );
        assert_eq!(
            std::fs::read(&source_path).expect("staged upload reads"),
            upload_bytes
        );
        assert_eq!(
            source_path.file_name().and_then(|value| value.to_str()),
            Some("detail.safetensors")
        );

        TEST_MAX_LORA_UPLOAD_BYTES.store(4, Ordering::SeqCst);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (bad_status, bad_error) = request_multipart_lora_upload(
            app,
            &[("name", "Too Large"), ("scope", "global")],
            "too-large.safetensors",
            b"12345",
        )
        .await;
        TEST_MAX_LORA_UPLOAD_BYTES.store(0, Ordering::SeqCst);
        assert_eq!(bad_status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            bad_error["detail"],
            "Uploaded LoRA file exceeds the 2GB limit"
        );

        let lora_source_dir = temp_dir.path().join("data").join("loras");
        std::fs::create_dir_all(&lora_source_dir).expect("lora source dir creates");
        let lora_source = lora_source_dir.join("safe-local.safetensors");
        write_test_safetensors(&lora_source);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, source_path_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": lora_source.display().to_string(),
                "name": "Safe Local Source"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(
            source_path_job["payload"]["manifestEntry"]["source"]["provider"],
            "local"
        );

        let outside_source = temp_dir.path().join("outside.safetensors");
        write_test_safetensors(&outside_source);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (bad_status, bad_error) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": outside_source.display().to_string(),
                "name": "Unsafe Local Source"
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "LoRA sourcePath must be inside app-managed data/loras, project/loras, or staged upload folders"
        );

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/loras/import",
            json!({ "sourceUrl": "file:///tmp/detail.safetensors" }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(bad_error["detail"], "LoRA sourceUrl must use http or https");

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/loras/import",
            json!({ "sourceUrl": "https://example.com/loras/detail.safetensors", "family": "unknown-family" }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Unsupported LoRA family: unknown-family"
        );

        let (status, normalized_family) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourceUrl": "https://example.com/loras/z-detail.safetensors",
                "family": "Z_Image"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(
            normalized_family["payload"]["manifestEntry"]["family"],
            "z-image"
        );

        // sc-1378: architecture detection at import time. The detection
        // policy lets the user supply any family the catalog declares, so
        // expand the catalog now to include the families we exercise below.
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
                  "capabilities": ["text_to_image", "edit_image"],
                  "downloads": [
                    { "provider": "huggingface", "repo": "owner/alternate-model", "files": ["*.bin"], "estimatedSizeBytes": 536870912 },
                    { "provider": "huggingface", "repo": "owner/model", "files": ["*.safetensors"], "default": true, "estimatedSizeBytes": 12884901888 }
                  ],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": { "label": "Base" }
                },
                {
                  "id": "qwen-image-base",
                  "name": "Qwen Image",
                  "family": "qwen-image",
                  "type": "image",
                  "adapter": "qwen_image",
                  "capabilities": ["text_to_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                },
                {
                  "id": "wan-video-base",
                  "name": "Wan Video",
                  "family": "wan-video",
                  "type": "video",
                  "adapter": "wan_video",
                  "capabilities": ["text_to_video"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models rewrites for detection tests");

        let detect_dir = temp_dir.path().join("data").join("loras");
        std::fs::create_dir_all(&detect_dir).expect("detect dir creates");

        // Qwen-Image-shaped file with a mismatched user-supplied family is
        // rejected with both values surfaced in the error message.
        let mismatch_path = detect_dir.join("qwen-as-wan.safetensors");
        write_test_safetensors_with_keys(&mismatch_path, &qwen_image_tensor_keys());
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, mismatch_error) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": mismatch_path.display().to_string(),
                "family": "wan-video",
                "name": "Mismatched"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let detail = mismatch_error["detail"].as_str().expect("detail string");
        assert!(
            detail.contains("qwen-image") && detail.contains("wan-video"),
            "mismatch error must surface both detected and supplied families, got: {detail}"
        );

        // Low-block MMDiT tensors are inconclusive rather than treated as
        // Z-Image; sparse Qwen LoRAs can target only early blocks.
        let auto_path = detect_dir.join("low-mmdit-no-autofill.safetensors");
        write_test_safetensors_with_keys(&auto_path, &z_image_tensor_keys());
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, auto_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": auto_path.display().to_string(),
                "name": "Auto Family"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert!(auto_job["payload"]["manifestEntry"].get("family").is_none());

        // Supplied family + inconclusive MMDiT detection succeeds, and the
        // user-supplied family is kept.
        let match_path = detect_dir.join("z-match.safetensors");
        write_test_safetensors_with_keys(&match_path, &z_image_tensor_keys());
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, match_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": match_path.display().to_string(),
                "family": "z-image",
                "name": "Matched"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(match_job["payload"]["manifestEntry"]["family"], "z-image");

        // Wan-shaped tensors are detected and accepted when the user agrees.
        let wan_match_path = detect_dir.join("wan-match.safetensors");
        write_test_safetensors_with_keys(&wan_match_path, &wan_video_tensor_keys());
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, wan_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": wan_match_path.display().to_string(),
                "family": "wan-video",
                "name": "Wan Matched"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(wan_job["payload"]["manifestEntry"]["family"], "wan-video");

        // Inconclusive header (only `__metadata__`) + supplied family is
        // accepted unchanged — the user-supplied label survives.
        let inconclusive_path = detect_dir.join("inconclusive.safetensors");
        write_test_safetensors(&inconclusive_path);
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, inconclusive_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": inconclusive_path.display().to_string(),
                "family": "z-image",
                "name": "Inconclusive"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(
            inconclusive_job["payload"]["manifestEntry"]["family"],
            "z-image"
        );

        // Confident Qwen-Image detection (block count > 40) auto-fills.
        let qwen_path = detect_dir.join("qwen-autofill.safetensors");
        write_test_safetensors_with_keys(&qwen_path, &qwen_image_tensor_keys());
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, qwen_job) = request(
            app,
            "POST",
            "/api/v1/loras/import",
            json!({
                "sourcePath": qwen_path.display().to_string(),
                "name": "Qwen Auto"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(qwen_job["payload"]["manifestEntry"]["family"], "qwen-image");
    }

    async fn request_multipart_model_upload(
        app: axum::Router,
        fields: &[(&str, &str)],
        filename: &str,
        bytes: &[u8],
    ) -> (StatusCode, Value) {
        let boundary = "SCENEWORKS_MODEL_BOUNDARY";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let (status, _, bytes) = request_raw(
            app,
            "POST",
            "/api/v1/models/import",
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
    async fn model_import_routes_handle_url_upload_and_family_detection() {
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
                { "id": "z_image_turbo", "name": "Z-Image Turbo", "family": "z-image", "type": "image", "loraCompatibility": { "families": ["z-image"] } }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("user presets writes");

        // URL import without supplied family is accepted, payload echoes
        // the request fields, and target dir lives under data/models/imports.
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, url_job) = request(
            app,
            "POST",
            "/api/v1/models/import",
            json!({
                "sourceUrl": "https://example.com/models/custom.safetensors",
                "name": "Custom Model",
                "type": "image"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(url_job["type"], "model_import");
        assert_eq!(url_job["requestedGpu"], "auto");
        assert_eq!(url_job["payload"]["modelId"], "custom_model");
        assert_eq!(url_job["payload"]["modelName"], "Custom Model");
        assert_eq!(
            url_job["payload"]["manifestEntry"]["source"]["provider"],
            "url"
        );
        assert_eq!(url_job["payload"]["manifestEntry"]["type"], "image");
        assert!(url_job["payload"]["targetDir"]
            .as_str()
            .is_some_and(|value| value.contains("models/imports/custom_model")
                || value.contains("models\\imports\\custom_model")));
        assert!(url_job["payload"]["manifestEntry"].get("family").is_none());

        // Duplicate id is rejected with an actionable error.
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (dup_status, dup_error) = request(
            app,
            "POST",
            "/api/v1/models/import",
            json!({
                "modelId": "z_image_turbo",
                "sourceUrl": "https://example.com/models/clone.safetensors",
                "name": "Clone"
            }),
        )
        .await;
        assert_eq!(dup_status, StatusCode::BAD_REQUEST);
        assert!(dup_error["detail"]
            .as_str()
            .unwrap_or("")
            .contains("already exists"));

        // Local upload stages bytes, queues an import job, and family
        // detection from a diffusers-style header auto-fills the manifest.
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let upload_bytes = test_safetensors_bytes_with_keys(&qwen_image_tensor_keys());
        let (upload_status, upload_job) = request_multipart_model_upload(
            app,
            &[("name", "Auto Family"), ("type", "image")],
            "auto-family.safetensors",
            &upload_bytes,
        )
        .await;
        assert_eq!(upload_status, StatusCode::CREATED);
        assert_eq!(upload_job["type"], "model_import");
        assert_eq!(upload_job["payload"]["modelId"], "auto_family");
        assert_eq!(upload_job["payload"]["uploadedSourcePath"], true);
        assert_eq!(
            upload_job["payload"]["manifestEntry"]["family"],
            "qwen-image"
        );
        assert_eq!(
            upload_job["payload"]["manifestEntry"]["adapter"],
            "qwen_image"
        );
        assert_eq!(
            upload_job["payload"]["manifestEntry"]["capabilities"],
            json!(["text_to_image", "style_variations"])
        );
        assert_eq!(
            upload_job["payload"]["manifestEntry"]["loraCompatibility"]["families"],
            json!(["qwen-image"])
        );
        let source_path = std::path::PathBuf::from(
            upload_job["payload"]["sourcePath"]
                .as_str()
                .expect("source path"),
        );
        assert_eq!(
            std::fs::read(&source_path).expect("staged upload reads"),
            upload_bytes
        );

        // Declared family must match detected family when detection is
        // confident — mismatch is rejected at the API boundary.
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (mismatch_status, mismatch_error) = request_multipart_model_upload(
            app,
            &[
                ("name", "Mismatch"),
                ("type", "image"),
                ("family", "z-image"),
            ],
            "mismatch.safetensors",
            &test_safetensors_bytes_with_keys(&qwen_image_tensor_keys()),
        )
        .await;
        assert_eq!(mismatch_status, StatusCode::BAD_REQUEST);
        assert!(mismatch_error["detail"]
            .as_str()
            .unwrap_or("")
            .contains("qwen-image"));

        // Missing source produces an actionable error.
        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (empty_status, empty_error) = request(
            app,
            "POST",
            "/api/v1/models/import",
            json!({ "name": "Nothing" }),
        )
        .await;
        assert_eq!(empty_status, StatusCode::BAD_REQUEST);
        assert!(empty_error["detail"]
            .as_str()
            .unwrap_or("")
            .contains("Hugging Face repo"));
    }

    #[tokio::test]
    async fn imported_model_catalog_uses_paths_model_install_marker() {
        std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        let model_dir = temp_dir.path().join("data/models/imports/custom_model");
        std::fs::create_dir_all(&model_dir).expect("model dir creates");
        std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("marker writes");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            format!(
                r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "custom_model",
                    "name": "Custom Model",
                    "type": "image",
                    "family": "z-image",
                    "paths": {{ "model": "{}" }}
                  }}]
                }}"#,
                model_dir.display().to_string().replace('\\', "\\\\")
            ),
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("user presets writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(models[0]["id"], "custom_model");
        assert_eq!(models[0]["downloadable"], false);
        assert_eq!(models[0]["installState"], "installed");
        assert_eq!(models[0]["installedPath"], model_dir.display().to_string());
    }

    #[tokio::test]
    async fn downloadable_model_catalog_uses_huggingface_cache_install_state() {
        std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [{
                "id": "base_model",
                "name": "Base Model",
                "type": "image",
                "family": "z-image",
                "downloads": [{ "provider": "huggingface", "repo": "owner/model" }]
              }]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("user presets writes");
        let cache_dir = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--owner--model/snapshots/abc123");
        std::fs::create_dir_all(&cache_dir).expect("hf cache creates");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, models) = request(app, "GET", "/api/v1/models", Value::Null).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(models[0]["id"], "base_model");
        assert_eq!(models[0]["downloadable"], true);
        assert_eq!(models[0]["installState"], "installed");
        assert!(models[0]["installedPath"]
            .as_str()
            .is_some_and(|value| value.contains("models--owner--model")));
    }

    #[tokio::test]
    async fn lora_catalog_uses_huggingface_cache_install_state() {
        std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "loras": [{
                "id": "ltx_ic_union",
                "name": "LTX IC Union",
                "family": "ltx-video",
                "icLora": true,
                "conditioningRole": "ic_lora",
                "compatibility": { "families": ["ltx-video"] },
                "source": {
                  "provider": "huggingface",
                  "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                  "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
                }
              }]
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
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("user presets writes");
        let stale_cache_file = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/snapshots/aaa111/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors");
        std::fs::create_dir_all(
            stale_cache_file
                .parent()
                .expect("stale cache file has parent"),
        )
        .expect("stale hf cache creates");
        std::fs::write(&stale_cache_file, b"stale-lora").expect("stale lora cache writes");
        let cache_file = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/snapshots/zzz999/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors");
        std::fs::create_dir_all(cache_file.parent().expect("cache file has parent"))
            .expect("hf cache creates");
        std::fs::write(&cache_file, b"lora").expect("lora cache writes");
        let refs_main = temp_dir
            .path()
            .join("data/cache/huggingface/hub/models--Lightricks--LTX-2.3-22b-IC-LoRA-Union-Control/refs/main");
        std::fs::create_dir_all(refs_main.parent().expect("refs main has parent"))
            .expect("refs dir creates");
        std::fs::write(&refs_main, b"zzz999").expect("refs main writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(loras[0]["id"], "ltx_ic_union");
        assert_eq!(loras[0]["icLora"], true);
        assert_eq!(loras[0]["conditioningRole"], "ic_lora");
        assert_eq!(loras[0]["installState"], "installed");
        assert_eq!(
            std::path::PathBuf::from(loras[0]["installedPath"].as_str().expect("installed path")),
            cache_file
        );
    }

    #[test]
    fn lora_artifact_paths_exclude_shared_huggingface_cache_files() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let cache_file = temp_dir.path().join(
            "data/cache/huggingface/hub/models--owner--repo/snapshots/abc123/lora.safetensors",
        );
        let lora = json!({
            "id": "hf_lora",
            "installedPath": cache_file.display().to_string(),
            "source": {
                "provider": "huggingface",
                "repo": "owner/repo",
                "file": "lora.safetensors"
            }
        });

        assert!(lora_artifact_paths(&lora, temp_dir.path()).is_empty());
    }

    #[tokio::test]
    async fn catalog_delete_routes_remove_manifest_entries_and_owned_artifacts() {
        std::env::set_var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", "1");
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        let model_dir = temp_dir.path().join("data/models/imports/delete_me");
        let lora_dir = temp_dir.path().join("data/loras/delete_style");
        std::fs::create_dir_all(&model_dir).expect("model dir creates");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        std::fs::write(model_dir.join(".sceneworks-download-complete.json"), "{}")
            .expect("marker writes");
        std::fs::write(lora_dir.join("adapter.safetensors"), b"lora").expect("lora writes");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            format!(
                r#"{{
                  "schemaVersion": 1,
                  "models": [{{
                    "id": "delete_me",
                    "name": "Delete Me",
                    "type": "image",
                    "family": "z-image",
                    "paths": {{ "model": "{}" }}
                  }}]
                }}"#,
                model_dir.display().to_string().replace('\\', "\\\\")
            ),
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            format!(
                r#"{{
                  "schemaVersion": 1,
                  "loras": [{{
                    "id": "delete_style",
                    "name": "Delete Style",
                    "family": "z-image",
                    "source": {{ "provider": "local", "path": "{}" }}
                  }}]
                }}"#,
                lora_dir.display().to_string().replace('\\', "\\\\")
            ),
        )
        .expect("user loras writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "moody",
                  "name": "Moody",
                  "workflow": "text_to_image",
                  "model": "delete_me",
                  "loras": [{ "id": "delete_style" }]
                }
              ]
            }
            "#,
        )
        .expect("user presets writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (model_status, model_delete) = request(
            app.clone(),
            "DELETE",
            "/api/v1/models/delete_me",
            Value::Null,
        )
        .await;
        assert_eq!(model_status, StatusCode::OK);
        assert_eq!(model_delete["removedManifestEntry"], true);
        assert_eq!(model_delete["removedLocalArtifacts"], true);
        assert!(model_delete["warnings"][0]
            .as_str()
            .is_some_and(|warning| warning.contains("Moody")));
        assert!(!model_dir.exists());
        let models_manifest =
            std::fs::read_to_string(config_dir.join("user.models.jsonc")).expect("models reads");
        assert!(!models_manifest.contains("delete_me"));

        let (lora_status, lora_delete) = request(
            app.clone(),
            "DELETE",
            "/api/v1/loras/delete_style?scope=global",
            Value::Null,
        )
        .await;
        assert_eq!(lora_status, StatusCode::OK);
        assert_eq!(lora_delete["removedManifestEntry"], true);
        assert_eq!(lora_delete["removedLocalArtifacts"], true);
        assert!(lora_delete["warnings"][0]
            .as_str()
            .is_some_and(|warning| warning.contains("Moody")));
        assert!(!lora_dir.exists());
        let loras_manifest =
            std::fs::read_to_string(config_dir.join("user.loras.jsonc")).expect("loras reads");
        assert!(!loras_manifest.contains("delete_style"));

        let (models_status, models) =
            request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
        assert_eq!(models_status, StatusCode::OK);
        assert_eq!(models.as_array().expect("models array").len(), 0);
        let (loras_status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
        assert_eq!(loras_status, StatusCode::OK);
        assert_eq!(loras.as_array().expect("loras array").len(), 0);
    }

    #[tokio::test]
    async fn recipe_preset_crud_routes_persist_global_and_project_presets() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                {
                  "id": "builtin_readonly",
                  "name": "Built-in Readonly",
                  "scope": "builtin",
                  "workflow": "text_to_image",
                  "model": "z_image_turbo"
                }
              ]
            }
            "#,
        )
        .expect("builtin recipe presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            // user preset notes survive API writes
            {
              "schemaVersion": 1,
              /* preserve unknown root fields too */
              "futureRoot": true,
              "presets": []
            }
            "#,
        )
        .expect("user recipe presets writes");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z Image Turbo",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "loras": [
                {
                  "id": "style_lora",
                  "name": "Style LoRA",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/style.safetensors" }
                },
                {
                  "id": "qwen_style",
                  "name": "Qwen Style",
                  "family": "qwen-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["qwen-image"] },
                  "source": { "provider": "local", "path": "loras/qwen.safetensors" }
                },
                {
                  "id": "deleted_style",
                  "name": "Deleted Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/deleted.safetensors" }
                },
                {
                  "id": "empty_dir_style",
                  "name": "Empty Dir Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local", "path": "loras/empty-dir" }
                },
                {
                  "id": "unknown_family",
                  "name": "Unknown Family",
                  "triggerWords": [],
                  "compatibility": {},
                  "source": { "provider": "local", "path": "loras/unknown.safetensors" }
                },
                {
                  "id": "no_path_style",
                  "name": "No Path Style",
                  "family": "z-image",
                  "triggerWords": [],
                  "compatibility": { "families": ["z-image"] },
                  "source": { "provider": "local" }
                }
              ]
            }
            "#,
        )
        .expect("user loras writes");
        let lora_dir = temp_dir.path().join("data/loras");
        std::fs::create_dir_all(&lora_dir).expect("lora dir creates");
        std::fs::create_dir_all(lora_dir.join("empty-dir")).expect("empty lora dir creates");
        write_test_safetensors(&lora_dir.join("style.safetensors"));
        write_test_safetensors(&lora_dir.join("qwen.safetensors"));
        write_test_safetensors(&lora_dir.join("unknown.safetensors"));

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, loras) = request(app.clone(), "GET", "/api/v1/loras", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        let empty_dir_style = loras
            .as_array()
            .expect("loras array")
            .iter()
            .find(|lora| lora["id"] == "empty_dir_style")
            .expect("empty dir lora listed");
        assert_eq!(empty_dir_style["installState"], "missing");

        // This also pins the positive compatibility path: style_lora is installed and compatible with z_image_turbo.
        let (status, created) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Soft Glow",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "order": 30,
                "defaults": { "resolution": "1024x1024" },
                "prompt": { "suffix": "soft glow" },
                "loras": [{ "id": "style_lora", "weight": 0.5 }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(created["id"], "soft_glow");
        assert_eq!(created["scope"], "global");
        assert_eq!(created["builtInLoras"][0]["id"], "style_lora");

        let (status, updated) = request(
            app.clone(),
            "PATCH",
            "/api/v1/recipe-presets/soft_glow",
            json!({
                "defaults": { "negativePrompt": "noise" },
                "loras": [{ "id": "style_lora", "weight": 0.75 }]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["defaults"]["negativePrompt"], "noise");
        assert_eq!(updated["loras"][0]["weight"], 0.75);

        let (status, duplicate) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets/soft_glow/duplicate",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(duplicate["id"], "soft_glow_copy");
        assert_eq!(duplicate["name"], "Soft Glow Copy");
        assert_eq!(duplicate["loras"][0]["id"], "style_lora");

        let (_, project) = request(
            app.clone(),
            "POST",
            "/api/v1/projects",
            json!({ "name": "Preset Project" }),
        )
        .await;
        let project_id = project["id"].as_str().expect("project id");
        let project_path = std::path::PathBuf::from(project["path"].as_str().unwrap());
        let (status, project_preset) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "id": "project_soft_glow",
                "name": "Project Soft Glow",
                "scope": "project",
                "projectId": project_id,
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "order": 1
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(project_preset["scope"], "project");
        assert!(project_path.join("recipes/presets.jsonc").is_file());

        for (id, name) in [("beta_order", "Beta Order"), ("alpha_order", "Alpha Order")] {
            let (status, _) = request(
                app.clone(),
                "POST",
                "/api/v1/recipe-presets",
                json!({
                    "id": id,
                    "name": name,
                    "model": "z_image_turbo",
                    "workflow": "text_to_image",
                    "order": 10
                }),
            )
            .await;
            assert_eq!(status, StatusCode::CREATED);
        }

        let (status, ordered) = request(
            app.clone(),
            "GET",
            &format!(
                "/api/v1/recipe-presets?projectId={project_id}&workflow=text_to_image&model=z_image_turbo"
            ),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let ordered_ids = ordered
            .as_array()
            .unwrap()
            .iter()
            .map(|preset| preset["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_ids,
            vec![
                "builtin_readonly",
                "alpha_order",
                "beta_order",
                "soft_glow",
                "soft_glow_copy",
                "project_soft_glow"
            ]
        );

        let (status, scoped) = request(
            app.clone(),
            "GET",
            "/api/v1/recipe-presets?scope=global&workflow=text_to_image&model=z_image_turbo",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(scoped
            .as_array()
            .unwrap()
            .iter()
            .all(|preset| preset["scope"] == "global"));

        let (status, readonly_error) = request(
            app.clone(),
            "PATCH",
            "/api/v1/recipe-presets/builtin_readonly",
            json!({ "name": "Nope" }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            readonly_error["detail"],
            "Built-in recipe presets are read-only"
        );

        let (status, project_updated) = request(
            app.clone(),
            "PATCH",
            &format!("/api/v1/recipe-presets/project_soft_glow?projectId={project_id}"),
            json!({ "prompt": { "suffix": "project update" } }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(project_updated["prompt"]["suffix"], "project update");

        let (status, _, bytes) = request_raw(
            app.clone(),
            "DELETE",
            "/api/v1/recipe-presets/soft_glow",
            Body::empty(),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let archived: Value = serde_json::from_slice(&bytes).expect("archive response parses");
        assert_eq!(archived["archived"], true);

        let (status, _, bytes) = request_raw(
            app.clone(),
            "DELETE",
            &format!("/api/v1/recipe-presets/project_soft_glow?projectId={project_id}"),
            Body::empty(),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let archived: Value =
            serde_json::from_slice(&bytes).expect("project archive response parses");
        assert_eq!(archived["archived"], true);

        let (status, visible) =
            request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!visible
            .as_array()
            .unwrap()
            .iter()
            .any(|preset| preset["id"] == "soft_glow"));

        let (status, archived_visible) = request(
            app.clone(),
            "GET",
            "/api/v1/recipe-presets?includeArchived=true",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(archived_visible
            .as_array()
            .unwrap()
            .iter()
            .any(|preset| preset["id"] == "soft_glow" && preset["archived"] == true));

        let (status, unarchived) = request(
            app.clone(),
            "PATCH",
            "/api/v1/recipe-presets/soft_glow",
            json!({ "archived": false }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(unarchived["archived"], false);

        let saved_manifest_text =
            std::fs::read_to_string(config_dir.join("user.recipe-presets.jsonc"))
                .expect("user recipe preset manifest reads");
        assert!(saved_manifest_text.starts_with(API_MANAGED_MANIFEST_HEADER));
        assert!(!saved_manifest_text.contains("// user preset notes survive API writes"));
        assert!(!saved_manifest_text.contains("/* preserve unknown root fields too */"));
        let saved_manifest: Value =
            serde_json::from_str(&strip_jsonc_comments(&saved_manifest_text))
                .expect("saved manifest parses");
        assert_eq!(saved_manifest["futureRoot"], true);

        let (status, second_update) = request(
            app.clone(),
            "PATCH",
            "/api/v1/recipe-presets/soft_glow",
            json!({ "order": 31 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(second_update["order"], 31);
        let second_manifest_text =
            std::fs::read_to_string(config_dir.join("user.recipe-presets.jsonc"))
                .expect("user recipe preset manifest reads after second write");
        assert!(second_manifest_text.starts_with(API_MANAGED_MANIFEST_HEADER));
        assert_eq!(
            second_manifest_text
                .matches(API_MANAGED_MANIFEST_HEADER)
                .count(),
            1
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "id": "Bad Id",
                "name": "Bad Id",
                "model": "z_image_turbo",
                "workflow": "text_to_image"
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe preset id must use lowercase letters, numbers, dashes, or underscores"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Bad Order",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "order": "high"
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe preset order must be an integer"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Bad Workflow",
                "model": "z_image_turbo",
                "workflow": "text_to_video"
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Model z_image_turbo does not support workflow text_to_video"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Too Many LoRAs",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [
                    { "id": "style_one" },
                    { "id": "style_two" },
                    { "id": "style_three" },
                    { "id": "style_four" }
                ]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe presets can include at most 3 LoRAs"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Overweighted LoRA",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [{ "id": "style_one", "weight": 2.5 }]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe preset LoRA weight must be between -2 and 2"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Missing LoRA",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [{ "id": "missing_lora" }]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe preset LoRA not found: missing_lora"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Deleted LoRA",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [{ "id": "deleted_style" }]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe preset LoRA is not installed: deleted_style"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "No Path LoRA",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [{ "id": "no_path_style" }]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "Recipe preset LoRA is not installed: no_path_style"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Unknown Family LoRA",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [{ "id": "unknown_family" }]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "LoRA unknown_family has no declared family; cannot verify compatibility with model z_image_turbo"
        );

        let (bad_status, bad_error) = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "name": "Wrong Family LoRA",
                "model": "z_image_turbo",
                "workflow": "text_to_image",
                "loras": [{ "id": "qwen_style" }]
            }),
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            bad_error["detail"],
            "LoRA qwen_style is not compatible with model z_image_turbo"
        );

        let create_one = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "id": "concurrent_one",
                "name": "Concurrent One",
                "model": "z_image_turbo",
                "workflow": "text_to_image"
            }),
        );
        let create_two = request(
            app.clone(),
            "POST",
            "/api/v1/recipe-presets",
            json!({
                "id": "concurrent_two",
                "name": "Concurrent Two",
                "model": "z_image_turbo",
                "workflow": "text_to_image"
            }),
        );
        let ((status_one, _), (status_two, _)) = tokio::join!(create_one, create_two);
        assert_eq!(status_one, StatusCode::CREATED);
        assert_eq!(status_two, StatusCode::CREATED);
        let (status, concurrent_presets) = request(
            app.clone(),
            "GET",
            "/api/v1/recipe-presets?scope=global&includeArchived=true",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(concurrent_presets
            .as_array()
            .unwrap()
            .iter()
            .any(|preset| preset["id"] == "concurrent_one"));
        assert!(concurrent_presets
            .as_array()
            .unwrap()
            .iter()
            .any(|preset| preset["id"] == "concurrent_two"));

        let (bad_status, bad_error) = request(
            app,
            "GET",
            "/api/v1/recipe-presets?workflow=bogus",
            Value::Null,
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(bad_error["detail"], "Unsupported recipe preset workflow");
    }

    #[tokio::test]
    async fn empty_builtin_preset_and_lora_manifests_ship_empty_catalogs() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let config_dir = temp_dir.path().join("config/manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin recipe presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("user recipe presets writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, presets) =
            request(app.clone(), "GET", "/api/v1/recipe-presets", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(presets.as_array().expect("presets array").len(), 0);

        let (status, loras) = request(app, "GET", "/api/v1/loras", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(loras.as_array().expect("loras array").len(), 0);
    }

    #[tokio::test]
    async fn legacy_preset_read_defaults_do_not_select_uninstalled_models() {
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
                  "id": "missing_image_model",
                  "name": "Missing Image Model",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image"],
                  "downloads": [{ "provider": "huggingface", "repo": "owner/missing-model", "files": ["*.safetensors"] }],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": {},
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
        std::fs::write(
            config_dir.join("builtin.recipe-presets.jsonc"),
            r#"{ "schemaVersion": 1, "presets": [] }"#,
        )
        .expect("builtin recipe presets writes");
        std::fs::write(
            config_dir.join("user.recipe-presets.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "presets": [
                { "id": "legacy_text", "name": "Legacy Text", "modes": ["text_to_image"] }
              ]
            }
            "#,
        )
        .expect("user recipe presets writes");
        std::fs::write(
            config_dir.join("builtin.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("builtin loras writes");
        std::fs::write(
            config_dir.join("user.loras.jsonc"),
            r#"{ "schemaVersion": 1, "loras": [] }"#,
        )
        .expect("user loras writes");

        let app = create_app(test_settings(&temp_dir)).expect("app creates");
        let (status, presets) = request(app, "GET", "/api/v1/recipe-presets", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        let preset = &presets.as_array().expect("presets array")[0];
        assert_eq!(preset["workflow"], "text_to_image");
        assert!(preset.get("model").is_none());
        assert_eq!(
            preset["appliedDefaults"]["notes"][0],
            "workflow inferred from legacy modes as text_to_image"
        );
        assert!(preset["appliedDefaults"]["notes"]
            .as_array()
            .expect("notes array")
            .iter()
            .all(|note| !note
                .as_str()
                .unwrap_or_default()
                .contains("model defaulted")));
    }

    #[test]
    fn model_download_size_helpers_match_contract_shapes() {
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
            super::manifest_download_size_bytes(
                &json!({ "downloads": [] }),
                &json!({ "estimatedSizeBytes": "4096" })
            ),
            Some(4096)
        );
        assert_eq!(
            super::manifest_download_size_bytes(&json!({ "sizeBytes": 2048 }), &json!({})),
            Some(2048)
        );
        assert_eq!(
            super::quote_huggingface_repo("owner/model name"),
            "owner/model%20name"
        );
        assert!(super::model_download(&json!({
            "downloads": [{ "repo": "owner/model" }]
        }))
        .is_none());
        assert_eq!(
            super::model_download(&json!({
                "downloads": [
                    { "provider": "huggingface", "repo": "owner/fallback", "estimatedSizeBytes": 1024 },
                    { "provider": "huggingface", "repo": "owner/default", "default": true, "estimatedSizeBytes": 4096 }
                ]
            }))
            .and_then(|download| download.get("repo").and_then(Value::as_str).map(str::to_owned)),
            Some("owner/default".to_owned())
        );
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
    fn lora_family_filter_shapes_match_contract_fallbacks() {
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
        assert!(error["detail"]
            .as_str()
            .is_some_and(|detail| detail.starts_with("Failed to parse manifest")));
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
        let config_dir = settings.config_dir.join("manifests");
        std::fs::create_dir_all(&config_dir).expect("manifest dir creates");
        std::fs::write(
            config_dir.join("builtin.models.jsonc"),
            r#"
            {
              "schemaVersion": 1,
              "models": [
                {
                  "id": "z_image_turbo",
                  "name": "Z-Image",
                  "family": "z-image",
                  "type": "image",
                  "adapter": "z_image_diffusers",
                  "capabilities": ["text_to_image", "character_image"],
                  "downloads": [],
                  "paths": {},
                  "defaults": {},
                  "limits": {},
                  "loraCompatibility": { "families": ["z-image"] },
                  "ui": {}
                }
              ]
            }
            "#,
        )
        .expect("builtin models writes");
        std::fs::write(
            config_dir.join("user.models.jsonc"),
            r#"{ "schemaVersion": 1, "models": [] }"#,
        )
        .expect("user models writes");
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
        write_test_safetensors(&lora_source);
        let (status, with_lora) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/projects/{project_id}/characters/{character_id}/loras"),
            json!({
                "name": "Mira LoRA",
                "sourcePath": lora_source.display().to_string(),
                "compatibility": { "families": ["z-image"] },
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
            test_safetensors_bytes()
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

    #[test]
    fn requires_token_only_gates_api_paths() {
        // Non-API paths (embedded UI / SPA fallback) must never require a token,
        // or the browser cannot load the bundle to prompt for one.
        assert!(!requires_token("/"));
        assert!(!requires_token("/assets/index-abc.js"));
        assert!(!requires_token("/projects/some-id"));
        // Public API paths stay open; other API paths stay gated.
        assert!(!requires_token("/api/v1/health"));
        assert!(requires_token("/api/v1/jobs"));
        assert!(requires_token("/api/v1/projects"));
    }

    #[tokio::test]
    async fn embedded_ui_root_is_reachable_with_access_token_set() {
        let temp_dir = tempfile::tempdir().expect("temp dir creates");
        let mut settings = test_settings(&temp_dir);
        settings.access_token = "secret-token".to_owned();
        let app = create_app(settings).expect("app creates");

        // With a token configured and no header, the embedded UI root and assets
        // must not be blocked by auth (404 here under default features since the
        // bundle isn't embedded; the point is it is NOT 401).
        let (status, _) = request(app.clone(), "GET", "/", Value::Null).await;
        assert_ne!(status, StatusCode::UNAUTHORIZED);
        let (status, _) = request(app.clone(), "GET", "/assets/app.js", Value::Null).await;
        assert_ne!(status, StatusCode::UNAUTHORIZED);
        // API routes stay protected.
        let (status, _) = request(app, "GET", "/api/v1/jobs", Value::Null).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn inprocess_worker_defaults_to_cpu_utility() {
        // Default (and blank override) → cpu so utility capabilities are
        // advertised regardless of the ambient SCENEWORKS_GPU_ID.
        assert_eq!(inprocess_worker_gpu_id(None), "cpu");
        assert_eq!(inprocess_worker_gpu_id(Some("   ".to_owned())), "cpu");
        // Explicit override is honored.
        assert_eq!(inprocess_worker_gpu_id(Some("auto".to_owned())), "auto");
        assert_eq!(inprocess_worker_gpu_id(Some("0".to_owned())), "0");
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
    async fn event_tickets_are_protected_and_match_contract_shape() {
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
    fn heartbeat_event_matches_contract_wire_shape() {
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
