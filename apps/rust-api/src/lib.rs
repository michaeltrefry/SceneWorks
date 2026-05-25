use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::io::SeekFrom;
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
use sceneworks_core::time::{format_unix_seconds, now_unix_seconds};
use sceneworks_core::training::{
    build_training_plan, builtin_training_presets, builtin_training_targets, BuildTrainingPlan,
    LoraTrainingRequest, TrainingDataset, TrainingPresetProvenance, TrainingTarget,
    TrainingTargetRegistry,
};
use sceneworks_core::training_store::{
    TrainingCaptionSidecarsResult, TrainingDatasetBatchRenameInput,
    TrainingDatasetCaptionSidecarsInput, TrainingDatasetCreateInput, TrainingDatasetMutationResult,
    TrainingDatasetSummary, TrainingDatasetUpdateInput,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::time::{Instant as TokioInstant, MissedTickBehavior};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;

mod auth;
use auth::{access_control, cors_layer, is_authorized};
mod characters;
use characters::*;
mod timelines;
use timelines::*;
mod person;
use person::*;
mod projects;
use projects::*;
mod assets;
use assets::*;
mod training;
use training::*;

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

/// Poll cadence for the parent-death watchdog (see [`shutdown_signal`]).
#[cfg(unix)]
const PARENT_POLL_INTERVAL: Duration = Duration::from_secs(3);

/// The parent PID this process should watch, parsed from `SCENEWORKS_PARENT_PID`.
/// `None` when the var is unset/blank/unparseable or `<= 1`: a value of 0 or 1
/// (init/launchd) means "already reparented or no real parent", so the watchdog
/// must not fire. Server/Docker deployments leave the var unset.
#[cfg(unix)]
fn parent_pid_to_watch() -> Option<i32> {
    let pid: i64 = std::env::var("SCENEWORKS_PARENT_PID")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (pid > 1 && pid <= i64::from(i32::MAX)).then_some(pid as i32)
}

/// True while `pid` names a live process. `kill(pid, None)` checks for the
/// process without delivering a signal: `Ok` means it's alive; `EPERM` means it
/// exists but we may not signal it (still alive); `ESRCH` is the only "gone"
/// case and yields false.
#[cfg(unix)]
fn pid_alive(pid: i32) -> bool {
    match nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None) {
        Ok(()) => true,
        Err(errno) => errno == nix::errno::Errno::EPERM,
    }
}

/// Resolves once the watched parent process disappears, polling every
/// [`PARENT_POLL_INTERVAL`]. With no parent to watch (`None`) it stays pending
/// forever, so the `select!` branch in [`shutdown_signal`] never fires.
#[cfg(unix)]
async fn parent_death(parent_pid: Option<i32>) {
    let Some(parent_pid) = parent_pid else {
        std::future::pending::<()>().await;
        return;
    };
    while pid_alive(parent_pid) {
        tokio::time::sleep(PARENT_POLL_INTERVAL).await;
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

    // Parent-death watchdog: when launched as a desktop sidecar the Tauri shell
    // sets SCENEWORKS_PARENT_PID to its own PID. A force-quit/crash skips the
    // shell's graceful teardown (`begin_shutdown`), so without this the API
    // orphans to launchd (PPID=1) — holding its OS-assigned port and a jobs.db
    // handle until the next launch reaps it. Unset (server/Docker) -> the future
    // stays pending and this branch never fires.
    #[cfg(unix)]
    let parent_gone = parent_death(parent_pid_to_watch());

    #[cfg(not(unix))]
    let parent_gone = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
        _ = parent_gone => {
            eprintln!("SceneWorks API: parent process exited; shutting down");
        }
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
        .route("/api/v1/training/presets", get(list_training_presets))
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
            "/api/v1/projects/:project_id/training/datasets/:dataset_id/caption-jobs",
            post(create_training_dataset_caption_job),
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
        .route("/api/v1/image/vqa/jobs", post(create_vqa_job))
        .route("/api/v1/image/interleave/jobs", post(create_interleave_job))
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
    // Absolute host paths are withheld from the public health endpoint when a token is
    // configured, so a LAN client can't map the host filesystem despite auth being on.
    #[serde(skip_serializing_if = "Option::is_none")]
    directories: Option<DirectoriesResponse>,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainingCaptionJobRequest {
    #[serde(default = "default_training_captioner")]
    captioner: String,
    #[serde(default = "default_training_caption_model")]
    model_name_or_path: String,
    #[serde(default)]
    recaption: bool,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
    #[serde(default)]
    options: TrainingCaptionOptions,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct TrainingCaptionOptions {
    #[serde(default = "default_training_caption_type")]
    caption_type: String,
    #[serde(default = "default_training_caption_length")]
    caption_length: String,
    #[serde(default)]
    extra_options: Vec<String>,
    #[serde(default)]
    name_input: String,
    #[serde(default = "default_training_caption_temperature")]
    temperature: f64,
    #[serde(default = "default_training_caption_top_p")]
    top_p: f64,
    #[serde(default = "default_training_caption_max_new_tokens")]
    max_new_tokens: u32,
    #[serde(default)]
    caption_prompt: String,
    #[serde(default)]
    low_vram: bool,
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
struct VqaJobRequest {
    project_id: String,
    #[serde(default)]
    project_name: Option<String>,
    source_asset_id: String,
    question: String,
    #[serde(default = "default_vqa_model")]
    model: String,
    #[serde(default = "default_vqa_max_new_tokens")]
    max_new_tokens: u32,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
    #[serde(default)]
    advanced: JsonObject,
}

fn default_vqa_model() -> String {
    "sensenova_u1_8b".to_owned()
}

fn default_vqa_max_new_tokens() -> u32 {
    256
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct InterleaveJobRequest {
    project_id: String,
    #[serde(default)]
    project_name: Option<String>,
    prompt: String,
    // Optional input images for grounded (it2i) interleaved generation.
    #[serde(default)]
    source_asset_ids: Vec<String>,
    #[serde(default = "default_interleave_model")]
    model: String,
    #[serde(default = "default_interleave_max_images")]
    max_images: u32,
    #[serde(default = "default_image_size")]
    width: u32,
    #[serde(default = "default_image_size")]
    height: u32,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default = "default_requested_gpu")]
    requested_gpu: String,
    #[serde(default)]
    advanced: JsonObject,
}

fn default_interleave_model() -> String {
    "sensenova_u1_8b".to_owned()
}

fn default_interleave_max_images() -> u32 {
    6
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
    let token_configured = !state.settings.access_token.is_empty();
    Json(HealthResponse {
        status: "ok",
        service: "sceneworks-api",
        runtime: "rust".to_owned(),
        version: state.settings.app_version.clone(),
        auth_required: token_configured,
        // When a token is configured the endpoint is public but the deployment expects
        // auth, so don't leak absolute host paths to unauthenticated LAN callers.
        directories: if token_configured {
            None
        } else {
            Some(DirectoriesResponse {
                data: state.settings.data_dir.display().to_string(),
                config: state.settings.config_dir.display().to_string(),
                projects: state.settings.projects_dir().display().to_string(),
                jobs_db: state.settings.jobs_db_path.display().to_string(),
            })
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

async fn get_project_file(
    State(state): State<AppState>,
    Path((project_id, relative_path)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let project_file = project_call(state, move |store| {
        store.project_file(&project_id, &relative_path)
    })
    .await?;
    let mut file = tokio::fs::File::open(&project_file.path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let total = file
        .metadata()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .len();
    let content_type = project_file.content_type;

    // WebKit/WKWebView (the macOS desktop webview) requires HTTP byte-range
    // responses to play <video>: it probes with `Range: bytes=0-1` and treats
    // any 200 reply as a non-seekable source it won't play. Honor a single
    // range with 206 Partial Content; advertise Accept-Ranges otherwise.
    if let Some(range_header) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        match parse_single_byte_range(range_header, total) {
            Some((start, end)) => {
                let len = end - start + 1;
                file.seek(SeekFrom::Start(start))
                    .await
                    .map_err(|error| ApiError::internal(error.to_string()))?;
                let stream = ReaderStream::new(file.take(len));
                return Ok((
                    StatusCode::PARTIAL_CONTENT,
                    [
                        (header::CONTENT_TYPE, content_type),
                        (header::ACCEPT_RANGES, "bytes".to_string()),
                        (
                            header::CONTENT_RANGE,
                            format!("bytes {start}-{end}/{total}"),
                        ),
                        (header::CONTENT_LENGTH, len.to_string()),
                    ],
                    Body::from_stream(stream),
                )
                    .into_response());
            }
            None => {
                return Ok((
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [(header::CONTENT_RANGE, format!("bytes */{total}"))],
                )
                    .into_response());
            }
        }
    }

    let stream = ReaderStream::new(file);
    Ok((
        [
            (header::CONTENT_TYPE, content_type),
            (header::ACCEPT_RANGES, "bytes".to_string()),
            (header::CONTENT_LENGTH, total.to_string()),
        ],
        Body::from_stream(stream),
    )
        .into_response())
}

/// Parse a single HTTP byte range (`bytes=start-end`, `bytes=start-`, or
/// `bytes=-suffix`) against a known total size, returning an inclusive
/// `(start, end)` clamped to the file. Returns `None` for unsatisfiable or
/// multi-range requests (callers answer 416).
fn parse_single_byte_range(value: &str, total: u64) -> Option<(u64, u64)> {
    let spec = value.strip_prefix("bytes=")?.trim();
    if spec.is_empty() || spec.contains(',') || total == 0 {
        return None;
    }
    let (start_str, end_str) = spec.split_once('-')?;
    let (start, end) = if start_str.is_empty() {
        // Suffix range: last `suffix` bytes.
        let suffix: u64 = end_str.parse().ok()?;
        if suffix == 0 {
            return None;
        }
        let start = total.saturating_sub(suffix);
        (start, total - 1)
    } else {
        let start: u64 = start_str.parse().ok()?;
        let end = if end_str.is_empty() {
            total - 1
        } else {
            end_str.parse::<u64>().ok()?.min(total - 1)
        };
        (start, end)
    };
    if start > end || start >= total {
        return None;
    }
    Some((start, end))
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

async fn create_vqa_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<VqaJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_vqa_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        JobType::ImageVqa,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

fn validate_vqa_job(payload: &VqaJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.source_asset_id.trim().is_empty() {
        return Err(ApiError::bad_request("sourceAssetId is required"));
    }
    let question = payload.question.trim();
    if question.is_empty() || question.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "question must be between 1 and 4000 characters",
        ));
    }
    if !(16..=2048).contains(&payload.max_new_tokens) {
        return Err(ApiError::bad_request(
            "maxNewTokens must be between 16 and 2048",
        ));
    }
    Ok(())
}

async fn create_interleave_job(
    State(state): State<AppState>,
    ApiJson(payload): ApiJson<InterleaveJobRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    validate_interleave_job(&payload)?;
    let requested_gpu = payload.requested_gpu.clone();
    let project_id = Some(payload.project_id.clone());
    let project_name = payload.project_name.clone();
    let mut job_payload = to_json_object(&payload)?;
    job_payload.remove("requestedGpu");
    let job = create_generation_job(
        state,
        JobType::ImageInterleave,
        project_id,
        project_name,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

fn validate_interleave_job(payload: &InterleaveJobRequest) -> Result<(), ApiError> {
    if payload.project_id.is_empty() {
        return Err(ApiError::bad_request("projectId is required"));
    }
    if payload.prompt.trim().is_empty() || payload.prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request(
            "prompt must be between 1 and 4000 characters",
        ));
    }
    // Upstream interleave_gen caps the run at 10 generated images.
    if !(1..=10).contains(&payload.max_images) {
        return Err(ApiError::bad_request("maxImages must be between 1 and 10"));
    }
    if payload
        .source_asset_ids
        .iter()
        .any(|id| id.trim().is_empty())
    {
        return Err(ApiError::bad_request(
            "sourceAssetIds must not contain blank ids",
        ));
    }
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
    Ok(())
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
        validate_dimension(width, "width", MAX_IMAGE_DIMENSION)?;
        validate_dimension(height, "height", MAX_IMAGE_DIMENSION)?;
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
    // Resolve the model manifest entry here so the GPU worker never re-parses
    // builtin/user.models.jsonc itself — Rust owns manifest parsing/merging
    // (story 1653). An unknown model resolves to {}, matching the worker's
    // existing fallback to the model's default repo.
    let model_manifest_entry = resolve_model_manifest_entry(&state, &payload.model).await?;
    job_payload.insert("modelManifestEntry".to_owned(), model_manifest_entry);
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

    // The desktop shell navigates its privileged webview to this server, so the embedded
    // UI runs from this origin and its CSP must come from here (tauri.conf.json only
    // governs the bundled setup screen). Kept narrow: scripts only from this origin (the
    // theme bootstrap was moved to /theme-init.js so no inline script is needed), Google
    // Fonts allowed, images/media as self/data/blob, IPC for the Tauri webview. Same-origin
    // API + SSE are covered by connect-src 'self'.
    pub(super) const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; \
script-src 'self'; \
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; \
font-src 'self' https://fonts.gstatic.com data:; \
img-src 'self' data: blob:; \
media-src 'self' data: blob:; \
connect-src 'self' ipc: http://ipc.localhost; \
object-src 'none'; \
base-uri 'self'; \
frame-ancestors 'none'; \
form-action 'self'";

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
                [
                    (header::CONTENT_TYPE, mime.as_ref()),
                    (header::CONTENT_SECURITY_POLICY, CONTENT_SECURITY_POLICY),
                ],
                file.data.into_owned(),
            )
                .into_response();
        }
        // Single-page app: unknown non-API paths resolve to index.html so
        // client-side deep links (e.g. project routes) load correctly.
        match WebAssets::get("index.html") {
            Some(index) => (
                [
                    (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                    (header::CONTENT_SECURITY_POLICY, CONTENT_SECURITY_POLICY),
                ],
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
    // Forward the catalog-declared family so the worker can re-verify the downloaded
    // weights match it (parity with model import). The catalog is project-curated, but
    // a mis-declared family would otherwise silently mismatch downstream adapter
    // selection; the worker reconciles and fails on a confident conflict (sc-1663).
    if let Some(family) = model.get("family").and_then(Value::as_str) {
        if !family.trim().is_empty() {
            job_payload.insert("family".to_owned(), Value::String(family.to_owned()));
        }
    }
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
    // Persist any generated assets the worker reported as `assetWrites` facts and
    // re-inject the built sidecars into the result so the UI keeps streaming them
    // (story 1656 — Rust is the single project-store writer).
    if let Some(result_obj) = result.as_mut() {
        persist_reported_assets(&state, &job_id, result_obj).await?;
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

/// Persist the generated assets a worker reports as `assetWrites` facts in its
/// progress result, then re-inject the built sidecars into `result.assets` /
/// `result.assetIds` so ImageStudio's live preview and the library refresh keep
/// streaming (story 1656). Idempotent: re-applied progress updates upsert the
/// same rows/files. No-op when there are no `assetWrites` (status-only updates,
/// or job types that still write their own assets).
async fn persist_reported_assets(
    state: &AppState,
    job_id: &str,
    result: &mut JsonObject,
) -> Result<(), ApiError> {
    let Some(asset_writes) = result.get("assetWrites").and_then(Value::as_array) else {
        return Ok(());
    };
    if asset_writes.is_empty() {
        return Ok(());
    }
    let asset_writes = asset_writes.clone();
    let generation_set_id = result
        .get("generationSetId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let generation_set = result.get("generationSet").cloned();
    // The job row is authoritative for the project id (never the worker payload).
    let job = store_call(state.clone(), {
        let job_id = job_id.to_owned();
        move |store, _timeout| store.get_job(&job_id)
    })
    .await?;
    let Some(project_id) = job.project_id.clone() else {
        return Ok(());
    };
    let job_id_owned = job_id.to_owned();
    let built = project_call(state.clone(), move |store| {
        if let Some(generation_set) = generation_set.as_ref() {
            store.write_generation_set(&project_id, &job_id_owned, generation_set)?;
        }
        let mut built = Vec::with_capacity(asset_writes.len());
        for fact in &asset_writes {
            built.push(store.persist_generated_asset(
                &project_id,
                &job_id_owned,
                &generation_set_id,
                fact,
            )?);
        }
        Ok(built)
    })
    .await?;
    let asset_ids: Vec<Value> = built
        .iter()
        .filter_map(|asset| asset.get("id").cloned())
        .collect();
    result.insert("assets".to_owned(), Value::Array(built));
    result.insert("assetIds".to_owned(), Value::Array(asset_ids));
    result.remove("assetWrites");
    result.remove("generationSet");
    Ok(())
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
/// entry shows up in `/api/v1/loras` and is selectable in the Studio (Image or
/// Video Studio, by LoRA family).
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
        validate_dimension(width, "width", MAX_IMAGE_DIMENSION)?;
        validate_dimension(height, "height", MAX_IMAGE_DIMENSION)?;
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

/// Resolve the merged model manifest entry for `model_id` so the GPU worker no
/// longer re-parses `builtin.models.jsonc`/`user.models.jsonc` itself — Rust is
/// the single owner of manifest parsing/merging (story 1653). The merged entry
/// is injected into video job payloads as `modelManifestEntry`. Returns `{}`
/// when the model is absent from both manifests, which the worker treats the
/// same as before (fall back to the model's default repo).
async fn resolve_model_manifest_entry(state: &AppState, model_id: &str) -> Result<Value, ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let find = |entries: &[Value]| -> Option<Value> {
        entries
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
    };
    Ok(merge_model_manifest_entry(find(&builtin), find(&user)))
}

/// One-level-deep merge of the builtin and user manifest entries for a single
/// model id. Mirrors the worker's former `ltx_model_manifest_entry` exactly so
/// this migration is behavior-preserving: user top-level keys override builtin
/// (shallow), and the nested config blocks the adapters read are merged
/// key-by-key rather than replaced wholesale. (This is intentionally deeper than
/// `merge_entries_by_id`, which the model catalog uses for display.)
fn merge_model_manifest_entry(builtin: Option<Value>, user: Option<Value>) -> Value {
    const NESTED_KEYS: [&str; 6] = [
        "paths",
        "resources",
        "defaults",
        "limits",
        "loraCompatibility",
        "ui",
    ];
    match (builtin, user) {
        (builtin, None) => builtin.unwrap_or_else(|| Value::Object(JsonObject::new())),
        (None, Some(user)) => user,
        (Some(builtin), Some(user)) => {
            let mut merged = builtin.clone();
            merge_object(&mut merged, user.clone());
            for key in NESTED_KEYS {
                let builtin_nested = builtin.get(key).and_then(Value::as_object);
                let user_nested = user.get(key).and_then(Value::as_object);
                if builtin_nested.is_none() && user_nested.is_none() {
                    continue;
                }
                let mut nested = builtin_nested.cloned().unwrap_or_default();
                if let Some(user_nested) = user_nested {
                    for (nested_key, value) in user_nested {
                        nested.insert(nested_key.clone(), value.clone());
                    }
                }
                if let Some(object) = merged.as_object_mut() {
                    object.insert(key.to_owned(), Value::Object(nested));
                }
            }
            merged
        }
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
    format_unix_seconds(now_unix_seconds())
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

/// The `<X>` in Hugging Face hub's `models--<X>` cache directory name: every
/// character outside `[A-Za-z0-9._-]` becomes `--`, then surrounding `-` are
/// trimmed. `None` when nothing survives. Kept byte-identical to the Python
/// worker (`hf_cache.safe_repo_dir_name`) and the Rust CPU worker — pinned by
/// the `repo_slugs.json` cross-language contract (story 1667).
fn safe_repo_dir_name(repo: &str) -> Option<String> {
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
        None
    } else {
        Some(safe_repo)
    }
}

fn huggingface_repo_cache_path(data_dir: &FsPath, repo: &str) -> Option<PathBuf> {
    let safe_repo = safe_repo_dir_name(repo)?;
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
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
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
    validate_dimension(payload.width, "width", MAX_IMAGE_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_IMAGE_DIMENSION)?;
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

/// Upper bound for image width/height. A backstop only — per-model resolution is
/// governed by manifest `limits.resolutions` + the UI. Covers SenseNova-U1's
/// largest trained bucket (3456) with headroom; video uses its own lower cap.
const MAX_IMAGE_DIMENSION: u32 = 4096;

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

fn default_training_captioner() -> String {
    "joy_caption".to_owned()
}

fn default_training_caption_model() -> String {
    "fancyfeast/llama-joycaption-beta-one-hf-llava".to_owned()
}

fn default_training_caption_type() -> String {
    "Descriptive".to_owned()
}

fn default_training_caption_length() -> String {
    "long".to_owned()
}

fn default_training_caption_temperature() -> f64 {
    0.6
}

fn default_training_caption_top_p() -> f64 {
    0.9
}

fn default_training_caption_max_new_tokens() -> u32 {
    256
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
mod tests;
