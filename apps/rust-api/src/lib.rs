use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::io::SeekFrom;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::rejection::JsonRejection;
use axum::extract::{
    DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request as AxumRequest, State,
};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post, put};
use axum::{Json, Router};
use futures_util::future::join_all;
use parking_lot::Mutex;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, DuplicateJobRequest, ImageUpscaleRequest,
    JobCreateRequest, JobSnapshot, JobStatus, JobType, JsonObject, ProgressRequest, QueueSummary,
    RetryJobRequest, WorkerCapability, WorkerHeartbeatRequest, WorkerRegisterRequest,
    WorkerSnapshot, WorkerStatus,
};
use sceneworks_core::hf_home::{huggingface_hub_cache_dir, huggingface_repo_cache_path};
use sceneworks_core::jobs_store::{
    mac_capabilities, mac_rust_supported, model_mac_support, CreateJob, DuplicateJob, JobsStore,
    JobsStoreError, MacCapabilities, ProgressUpdate, RegisterWorker, RetryJob, RouteDecision,
    UnsupportedReason, WorkerHeartbeat, JOB_STATUSES,
};
use sceneworks_core::lora_family::{
    apply_model_manifest_defaults, detect_lora_family, detect_model_family, first_safetensors_path,
    read_safetensors_header, reconcile_detected_family, SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{lora_source_url_file_stem, parse_lora_source_url, LoraUrlError};
use sceneworks_core::project_store::{
    AssetStatusPatch, AssetTagsPatch, CharacterCreateInput, CharacterLookInput,
    CharacterLookUpdateInput, CharacterLoraInput, CharacterLoraUpdateInput,
    CharacterReferenceInput, CharacterReferenceUpdateInput, CharacterUpdateInput, ProjectStore,
    ProjectStoreError, UploadAsset,
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
mod generation;
use generation::*;
mod jobs;
use jobs::*;
mod workers;
use workers::*;
mod events;
use events::*;
mod dto;
use dto::*;
mod manifest;
use manifest::*;
mod models;
use models::*;
mod loras;
use loras::*;
mod recipe_presets;
use recipe_presets::*;
mod credentials;
use credentials::*;
mod preferences;
use preferences::*;
mod prompts;
use prompts::*;
mod poses;
use poses::*;
mod keypoints;
use keypoints::*;
mod logs;
use logs::*;

const PUBLIC_PATHS: &[&str] = &[
    "/api/v1/health",
    "/api/v1/access",
    "/api/v1/auth/verify",
    "/api/v1/jobs/events",
    // Non-sensitive UI state (theme); loaded before auth to avoid a flash.
    "/api/v1/ui-preferences",
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
// sc-4201 (F-API-1): default to loopback so a bare/server run that doesn't set
// SCENEWORKS_API_HOST isn't exposed to the whole LAN with auth off. Docker and the
// desktop wrapper set the host explicitly (0.0.0.0 / 127.0.0.1 respectively), so this
// only changes the out-of-the-box default for a direct binary run.
const DEFAULT_API_HOST: &str = "127.0.0.1";
const MAX_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_MODEL_UPLOAD_BYTES: usize = 256 * 1024 * 1024 * 1024;
const MAX_LORA_MULTIPART_BODY_BYTES: usize = MAX_UPLOAD_BYTES + 16 * 1024 * 1024;
const MAX_MODEL_MULTIPART_BODY_BYTES: usize = MAX_MODEL_UPLOAD_BYTES + 16 * 1024 * 1024;
const STALE_LORA_UPLOAD_SECONDS: u64 = 24 * 60 * 60;
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
    /// Epic 3482 — macOS "MLX-required" mode. When set (the desktop sets it on macOS,
    /// where it spawns the in-process `mlx` worker), the MPS torch worker never claims an
    /// MLX-eligible job: it defers unconditionally to the `mlx` worker, and a job no live
    /// `mlx` worker takes within the grace window fails terminal with `mlx_unavailable`
    /// instead of silently falling back to MPS (sc-3483). Absent on Windows/Linux/Docker
    /// (no `mlx` worker) → today's behaviour unchanged. Ships default OFF (observe); the
    /// final cutover (sc-3492) flips it on for the packaged Mac build.
    pub mlx_required: bool,
    /// Epic 3482 / sc-3484 — when MLX-required, what to do with a job the Rust/MLX flow can't
    /// run (`mac_rust_supported` returns `Err`). **false = warn-only** (default): log a
    /// structured `mlx_unsupported` gap event at claim time but still run the job on the
    /// existing torch path, so flipping `mlx_required` on for observation materializes the gap
    /// list without breaking anything. **true = enforce**: fail the job terminal with
    /// `mlx_unsupported`. Read from `SCENEWORKS_MLX_UNSUPPORTED_MODE` (`enforce` vs anything
    /// else). Irrelevant unless `mlx_required`.
    pub mlx_enforce_unsupported: bool,
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
            host: env_string("SCENEWORKS_API_HOST", DEFAULT_API_HOST),
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
            mlx_required: std::env::var("SCENEWORKS_MLX_REQUIRED")
                .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
                .unwrap_or(false),
            mlx_enforce_unsupported: std::env::var("SCENEWORKS_MLX_UNSUPPORTED_MODE")
                .map(|value| value.trim().eq_ignore_ascii_case("enforce"))
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

// sc-4201 (F-API-1): true when the API would serve every endpoint without auth to
// the network — no access token AND a non-loopback bind address. Pure so the security
// decision is unit-tested without spinning up a listener.
fn should_warn_open_bind(access_token: &str, ip: std::net::IpAddr) -> bool {
    access_token.trim().is_empty() && !ip.is_loopback()
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

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Host mode (no HF cache env set): default HF_HOME to the shared ~/.cache/
    // huggingface so the catalog and downloads agree on the OS cache rather than
    // the private data dir (sc-1904 follow-up). Desktop/Compose already inject it.
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        println!(
            "SceneWorks Rust API defaulting HF_HOME to {}",
            home.display()
        );
    }
    let settings = Settings::from_env();
    let address: SocketAddr = format!("{}:{}", settings.host, settings.port).parse()?;
    // sc-4201 (F-API-1): a non-loopback bind with no access token serves every
    // endpoint — file reads, credential writes, job creation, large uploads — to the
    // whole network without authentication. The default is now loopback; warn loudly
    // when an operator opts into a wider bind (e.g. Docker's 0.0.0.0) without a token.
    if should_warn_open_bind(&settings.access_token, address.ip()) {
        eprintln!(
            "WARNING: SceneWorks API is binding to {address} with no SCENEWORKS_ACCESS_TOKEN set — \
             every endpoint is reachable without authentication from the network. Set \
             SCENEWORKS_ACCESS_TOKEN, or bind to 127.0.0.1, before exposing this beyond a trusted host."
        );
    }
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

/// Run this binary as a standalone worker process instead of the HTTP API.
/// Dispatched from `main` when `SCENEWORKS_WORKER_ONLY=1`; the desktop app uses
/// it to launch the Apple-Silicon MLX GPU worker (`SCENEWORKS_GPU_ID=mlx`,
/// sc-3289) as a crash-isolated sibling of the API process — reusing this binary
/// because it already links the mlx-gen engine.
///
/// Delegates to [`sceneworks_worker::run`] (which reads `SCENEWORKS_GPU_ID` +
/// `SCENEWORKS_API_URL` and, for a non-`auto`/non-`cpu` id, runs a single worker
/// loop), raced against the same parent-death watchdog the API uses: the desktop
/// sets `SCENEWORKS_PARENT_PID` to its own PID, and a force-quit/crash skips the
/// shell's graceful teardown — so without this a worker would orphan to launchd
/// with its multi-GB MLX model resident.
pub async fn run_worker() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        println!(
            "SceneWorks Rust worker defaulting HF_HOME to {}",
            home.display()
        );
    }
    #[cfg(unix)]
    {
        tokio::select! {
            result = sceneworks_worker::run() => result?,
            _ = parent_death(parent_pid_to_watch()) => {
                eprintln!("SceneWorks Rust worker: watched parent process gone, exiting");
            }
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        sceneworks_worker::run().await?;
        Ok(())
    }
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
    Ok(create_app_with_state(settings)?.0)
}

// Like create_app but also returns a clone of the AppState (the same Arc-shared
// stores + event hub the router uses), so tests can subscribe to the event hub and
// assert on what the handlers publish (sc-4203).
pub(crate) fn create_app_with_state(
    settings: Settings,
) -> Result<(Router, AppState), JobsStoreError> {
    let _ = std::fs::create_dir_all(&settings.data_dir);
    let _ = std::fs::create_dir_all(&settings.config_dir);
    if let Some(jobs_db_parent) = settings.jobs_db_path.parent() {
        let _ = std::fs::create_dir_all(jobs_db_parent);
    }
    let _ = sweep_stale_lora_uploads(&settings.data_dir);
    let _ = sweep_stale_pose_uploads(&settings.data_dir);
    let _ = sweep_stale_keypoint_uploads(&settings.data_dir);
    // sc-4204 (F-API-6): asset-import temp files (cache/uploads) had no startup sweep.
    let _ = sweep_stale_asset_uploads(&settings.data_dir);
    let jobs_store = Arc::new(JobsStore::new(&settings.jobs_db_path));
    jobs_store.initialize()?;
    let interrupted_jobs_on_startup = jobs_store.mark_interrupted_on_startup()?.len();
    let project_store = Arc::new(ProjectStore::new(
        settings.data_dir.clone(),
        settings.app_version.clone(),
    ));
    // Reserved global pose library (epic 2282): created up front so its assets
    // endpoint returns [] (not 404) before any pose is saved. Best-effort.
    if let Err(error) = project_store.ensure_global_poses_project() {
        eprintln!("SceneWorks API: could not ensure global pose library project: {error}");
    }
    // Reserved global Key Point Library (epic 4422): created up front so its assets +
    // collections endpoints return seeded data before any preset is saved. Best-effort.
    if let Err(error) = project_store.ensure_global_keypoints_project() {
        eprintln!("SceneWorks API: could not ensure global keypoint library project: {error}");
    }
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
    let returned_state = state.clone();

    let router = Router::new()
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
            "/api/v1/projects/:project_id/assets/:asset_id/tags",
            patch(update_asset_tags),
        )
        .route(
            "/api/v1/projects/:project_id/training/datasets",
            get(list_training_datasets).post(create_training_dataset),
        )
        .route(
            "/api/v1/projects/:project_id/training/uploads",
            post(upload_training_dataset_item),
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
        .route("/api/v1/prompts/refine", post(create_prompt_refine_job))
        .route("/api/v1/poses", post(create_poses))
        .route("/api/v1/poses/sources", post(create_pose_sources))
        .route(
            "/api/v1/poses/preview/:job_id/:file_name",
            get(get_pose_preview),
        )
        .route("/api/v1/keypoints", post(create_keypoint))
        .route("/api/v1/keypoints/sources", post(create_keypoint_sources))
        .route("/api/v1/keypoints/presets", get(list_keypoint_presets))
        .route(
            "/api/v1/keypoints/collections",
            get(list_keypoint_collections).post(upsert_keypoint_collection),
        )
        .route(
            "/api/v1/keypoints/collections/:collection_id",
            delete(delete_keypoint_collection),
        )
        .route(
            "/api/v1/keypoints/collections/:collection_id/default",
            put(set_default_keypoint_collection),
        )
        .route(
            "/api/v1/credentials",
            get(list_credentials).put(set_credential),
        )
        .route("/api/v1/credentials/:host", delete(delete_credential))
        .route(
            "/api/v1/ui-preferences",
            get(get_ui_preferences).put(set_ui_preferences),
        )
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
        .route("/api/v1/logs", get(list_logs))
        .route("/api/v1/workers", get(list_workers))
        .route(
            "/api/v1/capabilities/person",
            get(person_capability_readiness),
        )
        .route("/api/v1/capabilities/mac", get(mac_capability_support))
        .route("/api/v1/workers/register", post(register_worker))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_worker),
        )
        .fallback(app_fallback)
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(middleware::from_fn_with_state(state, access_control))
        .layer(cors);
    Ok((router, returned_state))
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

async fn project_path_for_id(state: AppState, project_id: &str) -> Result<PathBuf, ApiError> {
    let project_id = project_id.to_owned();
    let project = project_call(state, move |store| store.get_project(&project_id)).await?;
    Ok(PathBuf::from(project.path))
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
        // The specific base model the LoRA was trained for (e.g. wan_2_2 vs
        // wan_2_2_t2v_14b). The worker gates Wan 5B-vs-14B on this since both share
        // family `wan-video`. Absent for LoRAs that don't record one.
        "baseModel": preferred_lora_value(selected_lora, lora, "baseModel"),
        // Adapter network type (epic 2193). Carried into the generation payload so
        // the worker can route LoKr off the MLX backend without opening the file.
        "networkType": preferred_lora_value(selected_lora, lora, "networkType"),
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

fn validate_source_url(source_url: &str) -> Result<(), ApiError> {
    parse_lora_source_url(source_url)
        .map(|_| ())
        .map_err(|error| ApiError::bad_request(lora_url_error_message(error)))
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

fn now_rfc3339() -> String {
    format_unix_seconds(now_unix_seconds())
}

fn huggingface_repo_cache_exists(path: &FsPath) -> bool {
    path.join("snapshots").is_dir() || path.join("blobs").is_dir()
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
    if payload.upscale.enabled {
        if ![2, 4].contains(&payload.upscale.factor) {
            return Err(ApiError::bad_request("upscale.factor must be 2 or 4"));
        }
        if payload.upscale.engine.trim().is_empty() {
            return Err(ApiError::bad_request("upscale.engine is required"));
        }
    }
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
    validate_dimension(payload.width, "width", MAX_VIDEO_DIMENSION)?;
    validate_dimension(payload.height, "height", MAX_VIDEO_DIMENSION)?;
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

/// Upper bound for video width/height — a lower backstop than images, matching
/// the cap enforced when validating a video job request.
const MAX_VIDEO_DIMENSION: u32 = 1920;

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

fn default_fit_mode() -> String {
    // epic 2551: never stretch by default. "crop" covers the frame undistorted; the
    // worker normalizes unknown values back to crop, so this is just the wire default.
    "crop".to_owned()
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
            // 409 tells the worker its report lost a race with cancel/sweep/
            // reclaim: abandon the job instead of retrying (sc-4172).
            JobsStoreError::TerminalJobImmutable { job_id, status } => Self {
                status: StatusCode::CONFLICT,
                detail: format!(
                    "Job {job_id} is already {status}; terminal jobs cannot be updated."
                ),
            },
            JobsStoreError::NotJobOwner { job_id } => Self {
                status: StatusCode::CONFLICT,
                detail: format!(
                    "Progress rejected: the reporting worker no longer owns job {job_id}."
                ),
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

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}

#[cfg(test)]
mod tests;
