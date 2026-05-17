use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, DuplicateJobRequest, JobCreateRequest, JobSnapshot,
    ProgressRequest, QueueSummary, WorkerHeartbeatRequest, WorkerRegisterRequest, WorkerSnapshot,
};
use sceneworks_core::jobs_store::{
    CreateJob, DuplicateJob, JobsStore, JobsStoreError, ProgressUpdate, RegisterWorker,
    WorkerHeartbeat, JOB_STATUSES,
};
use sceneworks_core::project_store::{
    AssetStatusPatch, ProjectStore, ProjectStoreError, UploadAsset,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
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
const HEARTBEAT_SSE_TEXT: &str = "event: heartbeat\ndata: {}\n\n";

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
            app_version: env_string("SCENEWORKS_APP_VERSION", "0.1.0"),
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
    interrupted_jobs_on_startup: usize,
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
struct EventsQuery {
    ticket: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
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

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "sceneworks-api",
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
    Json(payload): Json<ProjectCreateRequest>,
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
        let bytes = field
            .bytes()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
            .to_vec();
        let asset = project_call(state, move |store| {
            store.import_asset(
                &project_id,
                UploadAsset {
                    filename,
                    content_type,
                    bytes,
                },
            )
        })
        .await?;
        return Ok((StatusCode::CREATED, Json(asset)));
    }
    Err(ApiError::bad_request("Upload file field is required"))
}

async fn update_asset_status(
    State(state): State<AppState>,
    Path((project_id, asset_id)): Path<(String, String)>,
    Json(payload): Json<AssetStatusPatch>,
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
    let bytes = tokio::fs::read(&project_file.path)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(([(header::CONTENT_TYPE, project_file.content_type)], bytes).into_response())
}

async fn route_not_found(request: Request<axum::body::Body>) -> Response {
    let path = request.uri().path();
    let lower_path = path.to_ascii_lowercase();
    if path.contains("/files/")
        && (path.contains("..") || lower_path.contains("%2e") || lower_path.contains("%2f"))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid project file path" })),
        )
            .into_response();
    }
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": "Not Found" })),
    )
        .into_response()
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
    Json(payload): Json<JobCreateRequest>,
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
    Json(payload): Json<ClaimRequest>,
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
    Json(payload): Json<DuplicateJobRequest>,
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
    Json(payload): Json<ProgressRequest>,
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
    Json(payload): Json<WorkerRegisterRequest>,
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
    Json(payload): Json<WorkerHeartbeatRequest>,
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
    let ready = tokio_stream::iter([Ok(Event::default()
        .event("ready")
        .data(json!({ "status": "connected" }).to_string()))]);
    let messages = state
        .events
        .subscribe()
        .map(|message| Ok(Event::default().event(message.event).data(message.data)));
    Ok(Sse::new(ready.chain(messages)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text(HEARTBEAT_SSE_TEXT),
    ))
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
        create_app, EventHub, EventMessage, Settings, EVENT_BUFFER_SIZE, HEARTBEAT_SSE_TEXT,
    };
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tokio_stream::StreamExt;
    use tower::ServiceExt;

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
        assert_eq!(assets.as_array().unwrap().len(), 1);

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
        assert_eq!(reindex["assets"], 1);

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
        assert_eq!(HEARTBEAT_SSE_TEXT, "event: heartbeat\ndata: {}\n\n");
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
