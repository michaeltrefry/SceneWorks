use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use reqwest::header;
use reqwest::StatusCode;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, JobSnapshot, JobStatus, JobType, JsonObject,
    ProgressRequest, ProgressStage, WorkerCapability, WorkerHeartbeatRequest,
    WorkerRegisterRequest, WorkerSnapshot, WorkerStatus, WorkerUtilizationSnapshot,
};
use sceneworks_core::lora_family::{
    apply_model_manifest_defaults, detect_lora_family, detect_model_family, first_safetensors_path,
    read_safetensors_header, reconcile_detected_family, FamilyMismatch, SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{
    lora_source_url_file_name, lora_source_url_file_stem, parse_lora_source_url_with_private,
    validate_public_ip,
};
use sceneworks_core::project_store::{ProjectStore, ProjectStoreError};
use sceneworks_core::slug::slugify;
use sceneworks_core::time::{format_unix_seconds, now_unix_seconds};
use serde::Deserialize;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time::MissedTickBehavior;
use uuid::Uuid;

mod api_client;
use api_client::*;
mod gpu;
use gpu::*;
mod supervisor;
use supervisor::*;
mod model_jobs;
use model_jobs::*;
mod media_jobs;
use media_jobs::*;
mod downloads;
use downloads::*;

const INSTALL_MARKER: &str = ".sceneworks-download-complete.json";
const DEFAULT_API_URL: &str = "http://localhost:8000";
const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const DEFAULT_MAX_LORA_URL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_MODEL_URL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const DEFAULT_TRANSITION_DURATION_SECONDS: f64 = 0.5;
const PERSON_TRACK_SAMPLE_RATE_FPS: f64 = 2.0;
const PERSON_TRACK_MAX_SAMPLES: usize = 24;
const PERSON_TRACK_X_DRIFT: f64 = 0.018;

/// How a stored download credential authenticates to its host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialScheme {
    /// `Authorization: Bearer <token>`.
    Bearer,
    /// `?token=<token>` query parameter.
    Query,
}

/// A per-host download credential injected via `SCENEWORKS_CREDENTIALS`, matched
/// against LoRA/model `sourceUrl` hosts.
#[derive(Debug, Clone)]
pub struct WorkerCredential {
    pub host: String,
    pub token: String,
    pub scheme: CredentialScheme,
}

/// Parse the `SCENEWORKS_CREDENTIALS` env value: a JSON object mapping host to
/// `{ "token": "...", "scheme": "bearer" | "query" }`. Empty entries are skipped,
/// an unrecognized/absent scheme defaults to bearer, and invalid JSON yields none.
pub(crate) fn parse_credentials_env(raw: &str) -> Vec<WorkerCredential> {
    #[derive(serde::Deserialize)]
    struct RawCredential {
        token: String,
        #[serde(default)]
        scheme: Option<String>,
    }
    let parsed: std::collections::HashMap<String, RawCredential> =
        serde_json::from_str(raw).unwrap_or_default();
    parsed
        .into_iter()
        .filter_map(|(host, credential)| {
            let host = host.trim().to_ascii_lowercase();
            let token = credential.token.trim().to_owned();
            if host.is_empty() || token.is_empty() {
                return None;
            }
            let scheme = match credential.scheme.as_deref() {
                Some("query") => CredentialScheme::Query,
                _ => CredentialScheme::Bearer,
            };
            Some(WorkerCredential {
                host,
                token,
                scheme,
            })
        })
        .collect()
}

/// Merge two credential sets keyed by host, with `env` overriding `file` per host.
/// Desktop injects credentials via the env (from the keychain); server/Docker reads
/// the config-dir file store; an operator env override wins over the file.
fn merge_credentials(
    file_credentials: Vec<WorkerCredential>,
    env_credentials: Vec<WorkerCredential>,
) -> Vec<WorkerCredential> {
    let mut by_host: std::collections::HashMap<String, WorkerCredential> =
        std::collections::HashMap::new();
    for credential in file_credentials {
        by_host.insert(credential.host.clone(), credential);
    }
    for credential in env_credentials {
        by_host.insert(credential.host.clone(), credential);
    }
    by_host.into_values().collect()
}

/// Worker credentials from the server/Docker file store (`<config>/credentials.json`)
/// overlaid with the `SCENEWORKS_CREDENTIALS` env (desktop injection / operator
/// override). Same parser for both (the file carries an extra `label` the worker
/// ignores). Picked up at startup, so changing credentials needs a worker restart —
/// consistent with the desktop, which already re-injects on restart.
fn load_worker_credentials(config_dir: &Path) -> Vec<WorkerCredential> {
    let file = config_dir.join(sceneworks_core::credentials::CREDENTIALS_FILENAME);
    let file_credentials = std::fs::read_to_string(&file)
        .ok()
        .map(|body| parse_credentials_env(&body))
        .unwrap_or_default();
    let env_credentials = std::env::var("SCENEWORKS_CREDENTIALS")
        .ok()
        .map(|raw| parse_credentials_env(&raw))
        .unwrap_or_default();
    merge_credentials(file_credentials, env_credentials)
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub api_url: String,
    pub access_token: Option<String>,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub worker_id: String,
    pub gpu_id: String,
    pub is_child_worker: bool,
    pub poll_seconds: u64,
    pub heartbeat_seconds: u64,
    pub shutdown_timeout_seconds: u64,
    pub huggingface_base_url: String,
    pub huggingface_token: Option<String>,
    /// Per-host download credentials from `SCENEWORKS_CREDENTIALS`, matched against
    /// LoRA/model `sourceUrl` hosts. HF auth still flows through `huggingface_token`.
    pub credentials: Vec<WorkerCredential>,
    pub max_lora_url_bytes: u64,
    pub max_model_url_bytes: u64,
    pub allow_private_lora_urls: bool,
    /// Number of CPU/utility worker processes to run when this worker is in
    /// dedicated `cpu` mode. Utility jobs (downloads, imports, frame extraction,
    /// timeline export, person detect/track) are I/O-bound and serialize per
    /// worker, so a small pool lets e.g. a quick upload run alongside a long
    /// download instead of queueing behind it.
    pub utility_workers: usize,
}

impl Settings {
    pub fn from_env() -> Self {
        let defaults = sceneworks_core::app_paths::AppPaths::platform_default();
        let config_dir = env_path_or("SCENEWORKS_CONFIG_DIR", &defaults.config_dir);
        Self {
            api_url: env_string("SCENEWORKS_API_URL", DEFAULT_API_URL),
            access_token: std::env::var("SCENEWORKS_ACCESS_TOKEN")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            data_dir: env_path_or("SCENEWORKS_DATA_DIR", &defaults.data_dir),
            config_dir: config_dir.clone(),
            worker_id: env_string("SCENEWORKS_WORKER_ID", "rust-utility-worker"),
            gpu_id: env_string("SCENEWORKS_GPU_ID", "cpu"),
            is_child_worker: std::env::var("SCENEWORKS_WORKER_CHILD")
                .is_ok_and(|value| value.trim() == "1"),
            poll_seconds: env_u64_any(
                &["SCENEWORKS_POLL_SECONDS", "SCENEWORKS_WORKER_POLL_SECONDS"],
                2,
            ),
            heartbeat_seconds: env_u64_any(
                &[
                    "SCENEWORKS_HEARTBEAT_SECONDS",
                    "SCENEWORKS_WORKER_HEARTBEAT_SECONDS",
                ],
                10,
            ),
            shutdown_timeout_seconds: env_u64_any(
                &["SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS"],
                10,
            ),
            huggingface_base_url: env_string(
                "SCENEWORKS_HUGGINGFACE_BASE_URL",
                DEFAULT_HUGGINGFACE_BASE_URL,
            ),
            huggingface_token: std::env::var("HF_TOKEN")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            credentials: load_worker_credentials(&config_dir),
            max_lora_url_bytes: env_u64_any(
                &["SCENEWORKS_MAX_LORA_URL_BYTES"],
                DEFAULT_MAX_LORA_URL_BYTES,
            ),
            max_model_url_bytes: env_u64_any(
                &["SCENEWORKS_MAX_MODEL_URL_BYTES"],
                DEFAULT_MAX_MODEL_URL_BYTES,
            ),
            allow_private_lora_urls: std::env::var("SCENEWORKS_ALLOW_PRIVATE_LORA_URLS")
                .is_ok_and(|value| value.trim() == "1"),
            utility_workers: env_u64_any(&["SCENEWORKS_UTILITY_WORKERS"], 4).max(1) as usize,
        }
    }
}

#[derive(Debug)]
pub enum WorkerError {
    Http(reqwest::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    ProjectStore(ProjectStoreError),
    Api { status: StatusCode, detail: String },
    InvalidPayload(String),
    Canceled(String),
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(error) => write!(formatter, "{error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::ProjectStore(error) => write!(formatter, "{error}"),
            Self::Api { status, detail } => write!(formatter, "API {status}: {detail}"),
            Self::InvalidPayload(detail) => formatter.write_str(detail),
            Self::Canceled(detail) => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<reqwest::Error> for WorkerError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<std::io::Error> for WorkerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for WorkerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<ProjectStoreError> for WorkerError {
    fn from(value: ProjectStoreError) -> Self {
        Self::ProjectStore(value)
    }
}

pub type WorkerResult<T> = Result<T, WorkerError>;

#[derive(Debug, Clone, PartialEq)]
struct DiscoveredGpu {
    id: String,
    name: String,
    capabilities: Vec<WorkerCapability>,
    utilization: Option<WorkerUtilizationSnapshot>,
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn emit_json(payload: Value) {
    println!("{payload}");
}

pub async fn run() -> WorkerResult<()> {
    let settings = Settings::from_env();
    if !settings.is_child_worker {
        if settings.gpu_id == "auto" {
            return supervise_auto_workers(settings).await;
        }
        if settings.gpu_id == "cpu" && settings.utility_workers > 1 {
            let specs = utility_worker_specs(&settings.worker_id, settings.utility_workers);
            return supervise_children(settings, specs).await;
        }
    }
    run_worker_loop(settings).await
}

pub async fn run_worker_loop(settings: Settings) -> WorkerResult<()> {
    let gpu = discover_gpu(&settings.gpu_id).await;
    let api = ApiClient::new(&settings);
    let http_client = reqwest::Client::new();
    register_worker_with_retry(&api, &settings, &gpu).await?;
    loop {
        tokio::select! {
            result = poll_once(&api, &settings, &http_client) => {
                if let Err(error) = result {
                    eprintln!("rust_worker_poll_failed: {error}");
                    tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))).await;
                }
            }
            _ = shutdown_signal() => {
                let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                return Ok(());
            }
        }
    }
}

async fn register_worker_with_retry(
    api: &ApiClient,
    settings: &Settings,
    gpu: &DiscoveredGpu,
) -> WorkerResult<()> {
    let mut attempt = 0_u32;
    loop {
        match register_worker(api, settings, gpu).await {
            Ok(_) => return Ok(()),
            Err(error) => {
                attempt = attempt.saturating_add(1);
                let delay = retry_delay(settings.poll_seconds, attempt);
                eprintln!(
                    "rust_worker_register_failed: attempt={attempt} retryInSeconds={delay} error={error}"
                );
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                    _ = shutdown_signal() => return Err(WorkerError::Canceled(
                        "Worker shutdown requested before registration completed.".to_owned(),
                    )),
                }
            }
        }
    }
}

async fn poll_once(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Idle, None).await?;
    let claim: ClaimResponse = api
        .post_json(
            "/api/v1/jobs/claim",
            &ClaimRequest {
                worker_id: settings.worker_id.clone(),
                extra: BTreeMap::new(),
            },
        )
        .await?;
    let Some(job) = claim.job else {
        tokio::time::sleep(Duration::from_secs(settings.poll_seconds)).await;
        return Ok(());
    };
    run_utility_job(api, settings, http_client, job).await;
    Ok(())
}

async fn register_worker(
    api: &ApiClient,
    settings: &Settings,
    gpu: &DiscoveredGpu,
) -> WorkerResult<WorkerSnapshot> {
    api.post_json(
        "/api/v1/workers/register",
        &WorkerRegisterRequest {
            worker_id: settings.worker_id.clone(),
            gpu_id: gpu.id.clone(),
            gpu_name: Some(gpu.name.clone()),
            capabilities: worker_capabilities(gpu),
            loaded_models: Vec::new(),
            utilization: gpu.utilization.clone(),
            extra: BTreeMap::new(),
        },
    )
    .await
}

async fn heartbeat(
    api: &ApiClient,
    settings: &Settings,
    status: WorkerStatus,
    current_job_id: Option<&str>,
) -> WorkerResult<WorkerSnapshot> {
    api.post_json(
        &format!("/api/v1/workers/{}/heartbeat", settings.worker_id),
        &WorkerHeartbeatRequest {
            status,
            current_job_id: current_job_id.map(str::to_owned),
            loaded_models: Vec::new(),
            utilization: gpu_utilization(&settings.gpu_id).await,
            extra: BTreeMap::new(),
        },
    )
    .await
}

async fn run_utility_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: JobSnapshot,
) {
    let result = match job.job_type {
        JobType::Placeholder => run_placeholder_job(api, settings, &job)
            .await
            .map_err(|error| ("Placeholder job failed.", error)),
        JobType::ModelDownload => run_model_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model download failed.", error)),
        JobType::LoraImport => run_lora_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA import failed.", error)),
        JobType::ModelImport => run_model_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model import failed.", error)),
        JobType::ModelConvert => run_model_convert_job(api, settings, &job)
            .await
            .map_err(|error| ("Model conversion failed.", error)),
        JobType::FrameExtract => run_frame_extract_job(api, settings, &job)
            .await
            .map_err(|error| ("Frame extraction failed.", error)),
        JobType::TimelineExport => run_timeline_export_job(api, settings, &job)
            .await
            .map_err(|error| ("Timeline export failed.", error)),
        JobType::PersonDetect => run_person_detect_job(api, settings, &job)
            .await
            .map_err(|error| ("Person detection failed.", error)),
        JobType::PersonTrack => run_person_track_job(api, settings, &job)
            .await
            .map_err(|error| ("Person tracking failed.", error)),
        _ => {
            let result = fail_job(
                api,
                &job.id,
                "No Rust utility exists for this job type.",
                Some(format!(
                    "Unsupported utility job type: {}",
                    job.job_type.as_str()
                )),
            )
            .await;
            result.map_err(|error| ("Utility job failed.", error))
        }
    };
    if matches!(job.job_type, JobType::LoraImport | JobType::ModelImport) {
        let _ = cleanup_uploaded_import_source(&job.payload).await;
    }
    if let Err((message, error)) = result {
        match error {
            WorkerError::Canceled(_) => {}
            error => {
                let _ = fail_job(api, &job.id, message, Some(error.to_string())).await;
                eprintln!("{error}");
            }
        }
    }
    let _ = heartbeat(api, settings, WorkerStatus::Idle, None).await;
}

async fn run_placeholder_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let stages = [
        (
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Preparing placeholder job.",
        ),
        (
            JobStatus::Running,
            ProgressStage::Running,
            0.35,
            "Running placeholder step 1.",
        ),
        (
            JobStatus::Running,
            ProgressStage::Running,
            0.65,
            "Running placeholder step 2.",
        ),
        (
            JobStatus::Saving,
            ProgressStage::Saving,
            0.9,
            "Saving placeholder result.",
        ),
    ];

    for (status, stage, progress, message) in stages {
        let snapshot: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{}", job.id)).await?;
        if snapshot.cancel_requested {
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Canceled,
                    ProgressStage::Canceled,
                    progress,
                    "Worker canceled the job before completion.",
                    None,
                    None,
                    None,
                ),
            )
            .await?;
            return Err(WorkerError::Canceled(
                "Worker canceled the job before completion.".to_owned(),
            ));
        }

        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
        update_job(
            api,
            &job.id,
            progress_payload(status, stage, progress, message, None, None, None),
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }

    let mut result = JsonObject::new();
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    result.insert("output".to_owned(), Value::String("placeholder".to_owned()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Placeholder job completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn fail_job(
    api: &ApiClient,
    job_id: &str,
    message: &str,
    error: Option<String>,
) -> WorkerResult<()> {
    update_job(
        api,
        job_id,
        progress_payload(
            JobStatus::Failed,
            ProgressStage::Failed,
            1.0,
            message,
            error,
            None,
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn check_cancel(api: &ApiClient, job_id: &str, message: &str) -> WorkerResult<()> {
    let job: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{job_id}")).await?;
    if job.cancel_requested {
        update_job(
            api,
            job_id,
            progress_payload(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                None,
                None,
                None,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    Ok(())
}

async fn update_job(
    api: &ApiClient,
    job_id: &str,
    payload: ProgressRequest,
) -> WorkerResult<JobSnapshot> {
    api.post_json(&format!("/api/v1/jobs/{job_id}/progress"), &payload)
        .await
}

pub async fn copy_lora_source(source: &Path, target_dir: &Path) -> WorkerResult<()> {
    import_lora_source_path(source, target_dir, false).await
}

async fn import_lora_source_path(
    source: &Path,
    target_dir: &Path,
    prefer_move: bool,
) -> WorkerResult<()> {
    let source = source.canonicalize()?;
    if !source.exists() {
        return Err(WorkerError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("LoRA source not found: {}", source.display()),
        )));
    }
    tokio::fs::create_dir_all(target_dir).await?;
    if source.is_dir() {
        copy_dir_recursive(&source, target_dir).await?;
    } else {
        let target = target_dir.join(source.file_name().ok_or_else(|| {
            WorkerError::InvalidPayload("LoRA source has no filename".to_owned())
        })?);
        if prefer_move {
            match tokio::fs::rename(&source, &target).await {
                Ok(()) => return Ok(()),
                Err(error) if is_cross_device_rename_error(&error) => {}
                Err(error) => return Err(error.into()),
            }
        }
        tokio::fs::copy(source, target).await?;
    }
    Ok(())
}

fn is_cross_device_rename_error(error: &std::io::Error) -> bool {
    matches!(error.raw_os_error(), Some(17 | 18))
}

async fn copy_dir_recursive(source: &Path, target: &Path) -> WorkerResult<()> {
    let mut stack = vec![(source.to_path_buf(), target.to_path_buf())];
    while let Some((source_dir, target_dir)) = stack.pop() {
        tokio::fs::create_dir_all(&target_dir).await?;
        let mut entries = tokio::fs::read_dir(&source_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let file_type = entry.file_type().await?;
            let destination = target_dir.join(entry.file_name());
            if file_type.is_dir() {
                stack.push((entry.path(), destination));
            } else if file_type.is_file() {
                tokio::fs::copy(entry.path(), destination).await?;
            }
        }
    }
    Ok(())
}

async fn write_model_install_marker(
    target_dir: &Path,
    payload: &JsonObject,
    repo: &str,
    job_id: &str,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    let marker = json!({
        "repo": repo,
        "modelId": payload.get("modelId").cloned().unwrap_or(Value::Null),
        "modelName": payload.get("modelName").cloned().unwrap_or(Value::Null),
        "jobId": job_id,
        "completedAt": now_rfc3339(),
    });
    let bytes = serde_json::to_vec_pretty(&marker)?;
    tokio::fs::write(target_dir.join(INSTALL_MARKER), bytes).await?;
    Ok(())
}

async fn write_lora_install_marker(
    target_dir: &Path,
    payload: &JsonObject,
    job_id: &str,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    let marker = json!({
        "loraId": payload.get("loraId").cloned().unwrap_or(Value::Null),
        "loraName": payload.get("name").cloned().unwrap_or(Value::Null),
        "repo": payload.get("repo").cloned().unwrap_or(Value::Null),
        "sourceUrl": payload.get("sourceUrl").cloned().unwrap_or(Value::Null),
        "sourcePath": payload.get("sourcePath").cloned().unwrap_or(Value::Null),
        "jobId": job_id,
        "completedAt": now_rfc3339(),
    });
    let bytes = serde_json::to_vec_pretty(&marker)?;
    tokio::fs::write(target_dir.join(INSTALL_MARKER), bytes).await?;
    Ok(())
}

pub fn allow_pattern_matches(path: &str, patterns: &[String]) -> bool {
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

pub fn safe_download_dir(value: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in value.chars() {
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

fn huggingface_hub_cache_dir(data_dir: &Path) -> PathBuf {
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
/// worker (`hf_cache.safe_repo_dir_name`) and the Rust API — pinned by the
/// `repo_slugs.json` cross-language contract (story 1667).
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

fn huggingface_repo_cache_path(data_dir: &Path, repo: &str) -> Option<PathBuf> {
    let safe_repo = safe_repo_dir_name(repo)?;
    Some(huggingface_hub_cache_dir(data_dir).join(format!("models--{safe_repo}")))
}

async fn directory_size(path: &Path) -> u64 {
    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(path) = stack.pop() {
        let mut entries = match tokio::fs::read_dir(&path).await {
            Ok(entries) => entries,
            Err(error) => {
                eprintln!(
                    "rust_worker_directory_size_failed: path={} error={error}",
                    path.display()
                );
                continue;
            }
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let Ok(file_type) = entry.file_type().await else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() && entry.file_name() != INSTALL_MARKER {
                if let Ok(metadata) = entry.metadata().await {
                    total = total.saturating_add(metadata.len());
                }
            }
        }
    }
    total
}

fn safe_join(base: &Path, relative: &str) -> WorkerResult<PathBuf> {
    let mut target = base.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            std::path::Component::Normal(value) => target.push(value),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Unsafe snapshot path: {relative}"
                )))
            }
        }
    }
    Ok(target)
}

fn progress_payload(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    error: Option<String>,
    result: Option<JsonObject>,
    eta_seconds: Option<ContractNumber>,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error,
        result,
        eta_seconds,
        extra: BTreeMap::new(),
    }
}

fn number_from_f64(value: f64) -> ContractNumber {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}

fn json_size_to_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn required_payload_string<'a>(payload: &'a JsonObject, field: &str) -> WorkerResult<&'a str> {
    optional_payload_string(payload, field)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload(format!("Missing payload.{field}")))
}

fn optional_payload_string<'a>(payload: &'a JsonObject, field: &str) -> Option<&'a str> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

fn payload_bool(payload: &JsonObject, field: &str) -> bool {
    payload.get(field).and_then(Value::as_bool).unwrap_or(false)
}

async fn cleanup_uploaded_import_source(payload: &JsonObject) -> WorkerResult<()> {
    if !payload_bool(payload, "uploadedSourcePath") {
        return Ok(());
    }
    let Some(source_path) = optional_payload_string(payload, "sourcePath") else {
        return Ok(());
    };
    let source_path = PathBuf::from(source_path);
    let _ = tokio::fs::remove_file(&source_path).await;
    if let Some(parent) = source_path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
    Ok(())
}

fn normalize_absolute_path(path: &Path) -> WorkerResult<PathBuf> {
    let mut output = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => output.push(prefix.as_os_str()),
            std::path::Component::RootDir => output.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !output.pop() {
                    return Err(WorkerError::InvalidPayload(format!(
                        "Unsafe absolute path: {}",
                        path.display()
                    )));
                }
            }
            std::path::Component::Normal(value) => output.push(value),
        }
    }
    Ok(output)
}

fn project_path_for_payload(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Option<PathBuf>> {
    let Some(project_id) = optional_payload_string(payload, "projectId") else {
        return Ok(None);
    };
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    Ok(Some(PathBuf::from(project.path)))
}

fn resolve_lora_import_target(
    settings: &Settings,
    payload: &JsonObject,
    fallback_target: PathBuf,
) -> WorkerResult<PathBuf> {
    let target = normalize_absolute_path(
        &optional_payload_string(payload, "targetDir")
            .map(PathBuf::from)
            .unwrap_or(fallback_target),
    )?;
    let mut allowed_roots = vec![normalize_absolute_path(&settings.data_dir.join("loras"))?];
    if let Some(project_path) = project_path_for_payload(settings, payload)? {
        allowed_roots.push(normalize_absolute_path(
            &project_path.join("loras").join("imports"),
        )?);
    }
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "LoRA import targetDir must be inside app-managed data/loras or project/loras/imports"
            .to_owned(),
    ))
}

fn resolve_model_import_target(
    settings: &Settings,
    payload: &JsonObject,
    fallback_target: PathBuf,
) -> WorkerResult<PathBuf> {
    let target = normalize_absolute_path(
        &optional_payload_string(payload, "targetDir")
            .map(PathBuf::from)
            .unwrap_or(fallback_target),
    )?;
    let allowed_roots = [normalize_absolute_path(&settings.data_dir.join("models"))?];
    if allowed_roots.iter().any(|root| target.starts_with(root)) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "Model import targetDir must be inside app-managed data/models".to_owned(),
    ))
}

fn resolve_model_convert_output(settings: &Settings, output_dir: &str) -> WorkerResult<PathBuf> {
    let target = normalize_absolute_path(&PathBuf::from(output_dir))?;
    let allowed_root = normalize_absolute_path(&settings.data_dir.join("models"))?;
    if target.starts_with(&allowed_root) {
        return Ok(target);
    }
    Err(WorkerError::InvalidPayload(
        "Model convert outputDir must be inside app-managed data/models".to_owned(),
    ))
}

fn model_manifest_target(settings: &Settings, payload: &JsonObject) -> WorkerResult<PathBuf> {
    let manifest_path = normalize_absolute_path(&PathBuf::from(required_payload_string(
        payload,
        "manifestPath",
    )?))?;
    let allowed = [normalize_absolute_path(
        &settings
            .config_dir
            .join("manifests")
            .join("user.models.jsonc"),
    )?];
    if allowed.iter().any(|path| path == &manifest_path) {
        return Ok(manifest_path);
    }
    Err(WorkerError::InvalidPayload(
        "Model manifestPath must target the global user model manifest".to_owned(),
    ))
}

fn lora_manifest_target(settings: &Settings, payload: &JsonObject) -> WorkerResult<PathBuf> {
    let manifest_path = normalize_absolute_path(&PathBuf::from(required_payload_string(
        payload,
        "manifestPath",
    )?))?;
    let mut allowed = vec![normalize_absolute_path(
        &settings
            .config_dir
            .join("manifests")
            .join("user.loras.jsonc"),
    )?];
    if let Some(project_path) = project_path_for_payload(settings, payload)? {
        allowed.push(normalize_absolute_path(
            &project_path.join("loras").join("manifest.jsonc"),
        )?);
    }
    if allowed.iter().any(|path| path == &manifest_path) {
        return Ok(manifest_path);
    }
    Err(WorkerError::InvalidPayload(
        "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest"
            .to_owned(),
    ))
}

fn payload_string_array(payload: &JsonObject, field: &str) -> Vec<String> {
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

fn progress_report_interval(settings: &Settings) -> Duration {
    Duration::from_secs(settings.heartbeat_seconds.clamp(5, 15))
}

pub fn format_bytes(value: u64) -> String {
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

fn quote_path(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn now_rfc3339() -> String {
    format_unix_seconds(now_unix_seconds())
}

fn bounded_tail(value: &str, max_lines: usize, max_chars: usize) -> String {
    let mut lines = value.lines().rev().take(max_lines).collect::<Vec<_>>();
    lines.reverse();
    let mut output = lines.join("\n");
    if output.len() > max_chars {
        let start = output
            .char_indices()
            .rev()
            .nth(max_chars)
            .map_or(0, |(index, _)| index);
        output = output[start..].to_owned();
    }
    output
}

async fn read_json_value(path: &Path) -> WorkerResult<Value> {
    Ok(serde_json::from_slice(&tokio::fs::read(path).await?)?)
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

async fn upsert_lora_manifest_entry(
    path: &Path,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    let mut manifest = match tokio::fs::read_to_string(path).await {
        Ok(payload) => serde_json::from_str(&strip_jsonc_comments(&payload))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            json!({ "schemaVersion": 1, "loras": [] })
        }
        Err(error) => return Err(error.into()),
    };
    let lora_id = entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| WorkerError::InvalidPayload("LoRA manifest entry requires id".to_owned()))?;
    let loras = manifest
        .as_object_mut()
        .ok_or_else(|| WorkerError::InvalidPayload("LoRA manifest must be an object".to_owned()))?
        .entry("loras")
        .or_insert_with(|| Value::Array(Vec::new()));
    let loras = loras.as_array_mut().ok_or_else(|| {
        WorkerError::InvalidPayload("LoRA manifest loras must be an array".to_owned())
    })?;
    let mut found = false;
    for item in loras.iter_mut() {
        if item.get("id").and_then(Value::as_str) != Some(lora_id) {
            continue;
        }
        found = true;
        let created_at = item.get("createdAt").cloned();
        let Some(object) = item.as_object_mut() else {
            return Err(WorkerError::InvalidPayload(
                "LoRA manifest entry must be an object".to_owned(),
            ));
        };
        for (key, value) in entry.clone() {
            object.insert(key, value);
        }
        if let Some(created_at) = created_at {
            object.insert("createdAt".to_owned(), created_at);
        }
    }
    if !found {
        loras.push(Value::Object(entry));
    }
    write_json_value(path, &manifest).await
}

async fn upsert_model_manifest_entry(
    path: &Path,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    let mut manifest = match tokio::fs::read_to_string(path).await {
        Ok(payload) => serde_json::from_str(&strip_jsonc_comments(&payload))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            json!({ "schemaVersion": 1, "models": [] })
        }
        Err(error) => return Err(error.into()),
    };
    let model_id = entry.get("id").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload("Model manifest entry requires id".to_owned())
    })?;
    let models = manifest
        .as_object_mut()
        .ok_or_else(|| WorkerError::InvalidPayload("Model manifest must be an object".to_owned()))?
        .entry("models")
        .or_insert_with(|| Value::Array(Vec::new()));
    let models = models.as_array_mut().ok_or_else(|| {
        WorkerError::InvalidPayload("Model manifest models must be an array".to_owned())
    })?;
    let mut found = false;
    for item in models.iter_mut() {
        if item.get("id").and_then(Value::as_str) != Some(model_id) {
            continue;
        }
        found = true;
        let created_at = item.get("createdAt").cloned();
        let Some(object) = item.as_object_mut() else {
            return Err(WorkerError::InvalidPayload(
                "Model manifest entry must be an object".to_owned(),
            ));
        };
        for (key, value) in entry.clone() {
            object.insert(key, value);
        }
        if let Some(created_at) = created_at {
            object.insert("createdAt".to_owned(), created_at);
        }
    }
    if !found {
        models.push(Value::Object(entry));
    }
    write_json_value(path, &manifest).await
}

async fn write_json_value(path: &Path, value: &Value) -> WorkerResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut output = serde_json::to_vec_pretty(value)?;
    output.push(b'\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    tokio::fs::write(&tmp_path, output).await?;
    tokio::fs::rename(tmp_path, path).await?;
    Ok(())
}

fn safe_project_path(project_path: &Path, relative: &str) -> WorkerResult<PathBuf> {
    if relative.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Project-relative path is required.".to_owned(),
        ));
    }
    let mut path = project_path.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            std::path::Component::Normal(value) => path.push(value),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Unsafe project-relative path: {relative}"
                )))
            }
        }
    }
    Ok(path)
}

fn relative_path(root: &Path, path: &Path) -> WorkerResult<String> {
    // Project media paths are app-created filenames; keep recipe metadata best-effort
    // if a host path contains non-UTF-8 bytes.
    Ok(path
        .strip_prefix(root)
        .map_err(|_| WorkerError::InvalidPayload("Path is outside project.".to_owned()))?
        .to_string_lossy()
        .replace('\\', "/"))
}

fn required_value_str<'a>(value: &'a Value, key: &str) -> WorkerResult<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| WorkerError::InvalidPayload(format!("Missing {key}")))
}

fn payload_u32(payload: &JsonObject, field: &str, default: u32) -> u32 {
    payload
        .get(field)
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(default)
}

fn payload_f64(payload: &JsonObject, field: &str, default: f64) -> f64 {
    payload
        .get(field)
        .map_or(default, |value| value_f64(value, default))
}

fn item_f64(item: &Value, field: &str, default: f64) -> f64 {
    item.get(field)
        .map_or(default, |value| value_f64(value, default))
}

fn value_f64(value: &Value, default: f64) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
        .filter(|value: &f64| value.is_finite())
        .unwrap_or(default)
}

fn fresh_asset_id() -> String {
    format!("asset_{}", Uuid::new_v4().simple())
}

fn asset_suffix(value: &str) -> String {
    let safe = safe_download_dir(value);
    let chars = safe.chars().rev().take(8).collect::<Vec<_>>();
    chars.into_iter().rev().collect::<String>()
}

async fn existing_download_bytes(path: &Path, expected_size: Option<u64>) -> WorkerResult<u64> {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return Ok(0);
    };
    let existing = metadata.len();
    if expected_size.is_some_and(|expected_size| existing > expected_size) {
        tokio::fs::remove_file(path).await?;
        return Ok(0);
    }
    Ok(existing)
}

fn with_hf_auth(settings: &Settings, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match &settings.huggingface_token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

fn retry_delay(poll_seconds: u64, attempt: u32) -> u64 {
    let multiplier = 2_u64.saturating_pow(attempt.saturating_sub(1).min(4));
    poll_seconds.max(1).saturating_mul(multiplier).clamp(1, 30)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn env_path_or(key: &str, default: &std::path::Path) -> PathBuf {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default.to_path_buf())
}

fn env_u64_any(keys: &[&str], default: u64) -> u64 {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse().ok()))
        .unwrap_or(default)
}

#[cfg(test)]
mod tests;
