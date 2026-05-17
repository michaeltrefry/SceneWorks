use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use reqwest::header;
use reqwest::StatusCode;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, JobSnapshot, JobStatus, JobType, JsonObject,
    ProgressRequest, ProgressStage, WorkerCapability, WorkerHeartbeatRequest,
    WorkerRegisterRequest, WorkerSnapshot, WorkerStatus,
};
use serde::Deserialize;
use serde_json::{json, Number, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tokio::io::AsyncWriteExt;
use tokio::time::MissedTickBehavior;

const INSTALL_MARKER: &str = ".sceneworks-download-complete.json";
const DEFAULT_API_URL: &str = "http://localhost:8000";
const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";

#[derive(Debug, Clone)]
pub struct Settings {
    pub api_url: String,
    pub access_token: Option<String>,
    pub data_dir: PathBuf,
    pub worker_id: String,
    pub poll_seconds: u64,
    pub heartbeat_seconds: u64,
    pub huggingface_base_url: String,
    pub huggingface_token: Option<String>,
}

impl Settings {
    pub fn from_env() -> Self {
        Self {
            api_url: env_string("SCENEWORKS_API_URL", DEFAULT_API_URL),
            access_token: std::env::var("SCENEWORKS_ACCESS_TOKEN")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            data_dir: env_path("SCENEWORKS_DATA_DIR", "data"),
            worker_id: env_string("SCENEWORKS_WORKER_ID", "rust-utility-worker"),
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
            huggingface_base_url: env_string(
                "SCENEWORKS_HUGGINGFACE_BASE_URL",
                DEFAULT_HUGGINGFACE_BASE_URL,
            ),
            huggingface_token: std::env::var("HF_TOKEN")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
        }
    }
}

#[derive(Debug)]
pub enum WorkerError {
    Http(reqwest::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
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

type WorkerResult<T> = Result<T, WorkerError>;

#[derive(Clone)]
struct ApiClient {
    client: reqwest::Client,
    api_url: String,
    access_token: Option<String>,
}

impl ApiClient {
    fn new(settings: &Settings) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_url: settings.api_url.trim_end_matches('/').to_owned(),
            access_token: settings.access_token.clone(),
        }
    }

    async fn get_json<T>(&self, path: &str) -> WorkerResult<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let response = self
            .with_auth(self.client.get(self.url(path)))
            .send()
            .await?;
        decode_api_response(response).await
    }

    async fn post_json<T, U>(&self, path: &str, payload: &T) -> WorkerResult<U>
    where
        T: serde::Serialize + ?Sized,
        U: for<'de> Deserialize<'de>,
    {
        let response = self
            .with_auth(self.client.post(self.url(path)).json(payload))
            .send()
            .await?;
        decode_api_response(response).await
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_url, path)
    }

    fn with_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.access_token {
            Some(token) => request.header("X-SceneWorks-Token", token),
            None => request,
        }
    }
}

async fn decode_api_response<T>(response: reqwest::Response) -> WorkerResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    let status = response.status();
    if !status.is_success() {
        let detail = response
            .text()
            .await
            .unwrap_or_else(|_| "request failed".to_owned());
        return Err(WorkerError::Api { status, detail });
    }
    Ok(response.json::<T>().await?)
}

pub async fn run() -> WorkerResult<()> {
    let settings = Settings::from_env();
    let api = ApiClient::new(&settings);
    let http_client = reqwest::Client::new();
    register_worker_with_retry(&api, &settings).await;
    loop {
        if let Err(error) = poll_once(&api, &settings, &http_client).await {
            eprintln!("rust_worker_poll_failed: {error}");
            tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))).await;
        }
    }
}

async fn register_worker_with_retry(api: &ApiClient, settings: &Settings) {
    let mut attempt = 0_u32;
    loop {
        match register_worker(api, settings).await {
            Ok(_) => return,
            Err(error) => {
                attempt = attempt.saturating_add(1);
                let delay = retry_delay(settings.poll_seconds, attempt);
                eprintln!(
                    "rust_worker_register_failed: attempt={attempt} retryInSeconds={delay} error={error}"
                );
                tokio::time::sleep(Duration::from_secs(delay)).await;
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

async fn register_worker(api: &ApiClient, settings: &Settings) -> WorkerResult<WorkerSnapshot> {
    api.post_json(
        "/api/v1/workers/register",
        &WorkerRegisterRequest {
            worker_id: settings.worker_id.clone(),
            gpu_id: "cpu".to_owned(),
            gpu_name: Some("Rust CPU utility worker".to_owned()),
            capabilities: vec![
                WorkerCapability::Cpu,
                WorkerCapability::ModelDownload,
                WorkerCapability::LoraImport,
            ],
            loaded_models: Vec::new(),
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
        JobType::ModelDownload => run_model_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model download failed.", error)),
        JobType::LoraImport => run_lora_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA import failed.", error)),
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

async fn run_model_download_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = match required_payload_string(&job.payload, "repo") {
        Ok(repo) => repo,
        Err(error) => {
            fail_job(
                api,
                &job.id,
                "Model download is missing a repository.",
                Some(error.to_string()),
            )
            .await?;
            return Ok(());
        }
    };
    let files = payload_string_array(&job.payload, "files");
    let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
    let target_dir = optional_payload_string(&job.payload, "targetDir")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            settings
                .data_dir
                .join("models")
                .join(safe_download_dir(repo))
        });

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.1,
            &format!("Downloading {repo}: estimating size."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Model download canceled before transfer started.",
    )
    .await?;

    let snapshot =
        HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
    if let Some(total_bytes) = snapshot.total_bytes() {
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Downloading,
                ProgressStage::Downloading,
                0.1,
                &format!("Downloading {repo}: 0 B of {}.", format_bytes(total_bytes)),
                None,
                None,
                None,
            ),
        )
        .await?;
    }

    let mut progress = DownloadProgress::new(
        repo,
        directory_size(&target_dir).await,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    download_snapshot(
        &DownloadContext {
            api,
            client: http_client,
            settings,
            job_id: &job.id,
            cancel_message: "Model download canceled by user.",
        },
        &target_dir,
        &snapshot,
        &mut progress,
    )
    .await?;
    write_model_install_marker(&target_dir, &job.payload, repo, &job.id).await?;

    let mut result = JsonObject::new();
    result.insert(
        "modelId".to_owned(),
        job.payload.get("modelId").cloned().unwrap_or(Value::Null),
    );
    result.insert("repo".to_owned(), Value::String(repo.to_owned()));
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Model download completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn run_lora_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = optional_payload_string(&job.payload, "repo");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "loraId")
        .or_else(|| optional_payload_string(&job.payload, "name"))
        .or(repo)
        .map(safe_download_dir)
        .unwrap_or_else(|| {
            source_path
                .and_then(|path| {
                    Path::new(path)
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .map(safe_download_dir)
                })
                .unwrap_or_else(|| "lora".to_owned())
        });
    let target_dir = settings.data_dir.join("loras").join(target_name);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Importing,
            0.1,
            "Importing LoRA.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "LoRA import canceled before transfer started.",
    )
    .await?;

    if let Some(repo) = repo {
        let files = payload_string_array(&job.payload, "files");
        let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
        let snapshot =
            HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
        let mut progress = DownloadProgress::new(
            repo,
            directory_size(&target_dir).await,
            snapshot.total_bytes(),
            progress_report_interval(settings),
        );
        // LoRA HF imports intentionally skip the model install marker for parity with the Python worker.
        download_snapshot(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "LoRA import canceled by user.",
            },
            &target_dir,
            &snapshot,
            &mut progress,
        )
        .await?;
    } else if let Some(source_path) = source_path {
        copy_lora_source(Path::new(source_path), &target_dir).await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "LoRA import failed.",
            Some("Provide repo or sourcePath for LoRA import".to_owned()),
        )
        .await;
    }

    let mut result = JsonObject::new();
    result.insert(
        "repo".to_owned(),
        repo.map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "LoRA import completed.",
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

#[derive(Debug, Clone)]
struct SnapshotFile {
    path: String,
    size: Option<u64>,
    download_url: String,
}

#[derive(Debug, Clone)]
struct HuggingFaceSnapshot {
    files: Vec<SnapshotFile>,
}

impl HuggingFaceSnapshot {
    async fn resolve(
        client: &reqwest::Client,
        settings: &Settings,
        repo: &str,
        revision: &str,
        files: &[String],
    ) -> WorkerResult<Self> {
        let base_url = settings.huggingface_base_url.trim_end_matches('/');
        let tree_url = format!(
            "{base_url}/api/models/{}/tree/{}?recursive=1&expand=1",
            quote_path(repo),
            quote_path(revision)
        );
        let payload = with_hf_auth(settings, client.get(tree_url))
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        let entries = if let Some(entries) = payload.as_array() {
            entries.clone()
        } else {
            payload
                .get("siblings")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let snapshot_files = entries
            .iter()
            .filter_map(|entry| snapshot_file_from_entry(base_url, repo, revision, entry))
            .filter(|file| allow_pattern_matches(&file.path, files))
            .collect();
        Ok(Self {
            files: snapshot_files,
        })
    }

    fn total_bytes(&self) -> Option<u64> {
        self.files
            .iter()
            .try_fold(0_u64, |total, file| Some(total.saturating_add(file.size?)))
    }
}

fn snapshot_file_from_entry(
    base_url: &str,
    repo: &str,
    revision: &str,
    entry: &Value,
) -> Option<SnapshotFile> {
    let kind = entry.get("type").and_then(Value::as_str);
    if kind.is_some_and(|kind| kind != "file") {
        return None;
    }
    let path = entry
        .get("path")
        .or_else(|| entry.get("rfilename"))
        .and_then(Value::as_str)?;
    Some(SnapshotFile {
        path: path.to_owned(),
        size: entry.get("size").and_then(json_size_to_u64),
        download_url: format!(
            "{base_url}/{}/resolve/{}/{}",
            quote_path(repo),
            quote_path(revision),
            quote_path(path)
        ),
    })
}

struct DownloadContext<'a> {
    api: &'a ApiClient,
    client: &'a reqwest::Client,
    settings: &'a Settings,
    job_id: &'a str,
    cancel_message: &'a str,
}

async fn download_snapshot(
    context: &DownloadContext<'_>,
    target_dir: &Path,
    snapshot: &HuggingFaceSnapshot,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    for file in &snapshot.files {
        check_cancel(context.api, context.job_id, context.cancel_message).await?;
        let target_path = safe_join(target_dir, &file.path)?;
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let existing_bytes = existing_download_bytes(&target_path, file.size).await?;
        if file.size.is_some_and(|size| existing_bytes == size) {
            continue;
        }
        let mut request = context.client.get(&file.download_url);
        if existing_bytes > 0 {
            request = request.header(header::RANGE, format!("bytes={existing_bytes}-"));
        }
        let response = with_hf_auth(context.settings, request).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(WorkerError::Http(response.error_for_status().unwrap_err()));
        }
        let appending = existing_bytes > 0 && status == StatusCode::PARTIAL_CONTENT;
        if existing_bytes > 0 && !appending {
            progress.discard_started_bytes(existing_bytes);
        }
        let mut response = response;
        let mut output = if appending {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&target_path)
                .await?
        } else {
            tokio::fs::File::create(&target_path).await?
        };
        let mut interval = tokio::time::interval(progress.report_interval());
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                chunk = response.chunk() => {
                    let Some(chunk) = chunk? else {
                        break;
                    };
                    output.write_all(&chunk).await?;
                    progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                }
                _ = interval.tick() => {
                    report_download_progress(context, progress).await?;
                }
            }
        }
    }
    Ok(())
}

async fn report_download_progress(
    context: &DownloadContext<'_>,
    progress: &DownloadProgress<'_>,
) -> WorkerResult<()> {
    heartbeat(
        context.api,
        context.settings,
        WorkerStatus::Busy,
        Some(context.job_id),
    )
    .await?;
    update_job(context.api, context.job_id, progress.payload()).await?;
    check_cancel(context.api, context.job_id, context.cancel_message).await
}

struct DownloadProgress<'a> {
    repo: &'a str,
    started_bytes: u64,
    transferred_bytes: u64,
    total_bytes: Option<u64>,
    started_at: Instant,
    report_interval: Duration,
}

impl<'a> DownloadProgress<'a> {
    fn new(
        repo: &'a str,
        started_bytes: u64,
        total_bytes: Option<u64>,
        report_interval: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            repo,
            started_bytes,
            transferred_bytes: 0,
            total_bytes,
            started_at: now,
            report_interval,
        }
    }

    fn downloaded_bytes(&self) -> u64 {
        self.started_bytes.saturating_add(self.transferred_bytes)
    }

    fn record_transferred(&mut self, bytes: u64) {
        self.transferred_bytes = self.transferred_bytes.saturating_add(bytes);
    }

    fn discard_started_bytes(&mut self, bytes: u64) {
        self.started_bytes = self.started_bytes.saturating_sub(bytes);
    }

    fn report_interval(&self) -> Duration {
        self.report_interval
    }

    fn payload(&self) -> ProgressRequest {
        download_progress_payload(
            self.repo,
            self.downloaded_bytes(),
            self.total_bytes,
            self.started_bytes,
            self.started_at.elapsed(),
        )
    }
}

pub fn download_progress_payload(
    repo: &str,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    started_bytes: u64,
    elapsed: Duration,
) -> ProgressRequest {
    let transferred_bytes = downloaded_bytes.saturating_sub(started_bytes);
    let elapsed_seconds = elapsed.as_secs_f64().max(0.001);
    let rate = transferred_bytes as f64 / elapsed_seconds;
    let eta_seconds = total_bytes.and_then(|total| {
        if rate > 0.0 {
            let remaining = total.saturating_sub(downloaded_bytes) as f64;
            Some(number_from_f64((remaining / rate).max(0.0)))
        } else {
            None
        }
    });

    let (progress, message) = if let Some(total) = total_bytes {
        let ratio = if total == 0 {
            1.0
        } else {
            (downloaded_bytes as f64 / total as f64).clamp(0.0, 1.0)
        };
        let remaining = total.saturating_sub(downloaded_bytes);
        (
            0.1 + ratio * 0.85,
            format!(
                "Downloading {repo}: {} of {} ({} left).",
                format_bytes(downloaded_bytes),
                format_bytes(total),
                format_bytes(remaining)
            ),
        )
    } else {
        (
            0.1,
            format!(
                "Downloading {repo}: {} written.",
                format_bytes(downloaded_bytes)
            ),
        )
    };

    progress_payload(
        JobStatus::Downloading,
        ProgressStage::Downloading,
        progress,
        &message,
        None,
        None,
        eta_seconds,
    )
}

pub async fn copy_lora_source(source: &Path, target_dir: &Path) -> WorkerResult<()> {
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
        tokio::fs::copy(source, target).await?;
    }
    Ok(())
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
    OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .unwrap_or(OffsetDateTime::UNIX_EPOCH)
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_owned())
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

fn env_path(key: &str, default: &str) -> PathBuf {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn env_u64_any(keys: &[&str], default: u64) -> u64 {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse().ok()))
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode as AxumStatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        allow_pattern_matches, copy_lora_source, download_progress_payload, now_rfc3339,
        safe_download_dir, write_model_install_marker, HuggingFaceSnapshot, Settings,
        INSTALL_MARKER,
    };

    #[test]
    fn download_progress_payload_matches_python_shape() {
        let payload = download_progress_payload(
            "owner/model",
            512 * 1024 * 1024,
            Some(1024 * 1024 * 1024),
            0,
            Duration::from_secs(2),
        );

        assert_eq!(payload.status.as_str(), "downloading");
        assert_eq!(payload.stage.as_str(), "downloading");
        assert_eq!(payload.progress.as_f64(), Some(0.525));
        assert!(payload.message.contains("512.0 MB of 1.0 GB"));
        assert!(payload.eta_seconds.is_some());
    }

    #[test]
    fn pattern_filtering_and_download_dir_match_python_behavior() {
        assert!(allow_pattern_matches(
            "nested/model.safetensors",
            &["*.safetensors".to_owned()]
        ));
        assert!(!allow_pattern_matches(
            "nested/model.ckpt",
            &["*.safetensors".to_owned()]
        ));
        assert_eq!(safe_download_dir("owner/model name"), "owner__model__name");
        assert_eq!(safe_download_dir("///"), "download");
    }

    #[tokio::test]
    async fn writes_model_install_marker_with_expected_keys() {
        let temp = tempdir().expect("tempdir creates");
        let mut payload = serde_json::Map::new();
        payload.insert("modelId".to_owned(), json!("base-model"));
        payload.insert("modelName".to_owned(), json!("Base Model"));

        write_model_install_marker(temp.path(), &payload, "owner/model", "job-1")
            .await
            .expect("marker writes");

        let marker_path = temp.path().join(INSTALL_MARKER);
        let marker: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(marker_path).await.unwrap()).unwrap();
        assert_eq!(marker["repo"], "owner/model");
        assert_eq!(marker["modelId"], "base-model");
        assert_eq!(marker["modelName"], "Base Model");
        assert_eq!(marker["jobId"], "job-1");
        assert!(marker["completedAt"].as_str().is_some());
    }

    #[tokio::test]
    async fn lora_file_and_directory_import_preserve_copy_semantics() {
        let temp = tempdir().expect("tempdir creates");
        let source_file = temp.path().join("mira.safetensors");
        tokio::fs::write(&source_file, b"lora").await.unwrap();
        let file_target = temp.path().join("file-target");

        copy_lora_source(&source_file, &file_target).await.unwrap();

        assert_eq!(
            tokio::fs::read(file_target.join("mira.safetensors"))
                .await
                .unwrap(),
            b"lora"
        );

        let source_dir = temp.path().join("source-dir");
        tokio::fs::create_dir_all(source_dir.join("nested"))
            .await
            .unwrap();
        tokio::fs::write(source_dir.join("nested/adapter.safetensors"), b"adapter")
            .await
            .unwrap();
        let dir_target = temp.path().join("dir-target");

        copy_lora_source(&source_dir, &dir_target).await.unwrap();

        assert_eq!(
            tokio::fs::read(dir_target.join("nested/adapter.safetensors"))
                .await
                .unwrap(),
            b"adapter"
        );
    }

    #[test]
    fn now_matches_python_second_precision() {
        let value = now_rfc3339();

        assert!(value.ends_with('Z'));
        assert!(!value.trim_end_matches('Z').contains('.'));
    }

    #[tokio::test]
    async fn huggingface_snapshot_resolve_accepts_tree_and_sibling_shapes_with_auth() {
        let array_url = spawn_hf_stub(
            json!([
                { "type": "file", "path": "nested/model.safetensors", "size": 7 },
                { "type": "file", "path": "nested/model.ckpt", "size": 9 },
                { "type": "directory", "path": "nested" }
            ]),
            Some("hf_test"),
        )
        .await;
        let client = reqwest::Client::new();
        let array_settings = test_settings(array_url, Some("hf_test"));

        let snapshot = HuggingFaceSnapshot::resolve(
            &client,
            &array_settings,
            "owner/model",
            "main",
            &["*.safetensors".to_owned()],
        )
        .await
        .expect("tree snapshot resolves");

        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.files[0].path, "nested/model.safetensors");
        assert_eq!(snapshot.total_bytes(), Some(7));

        let siblings_url = spawn_hf_stub(
            json!({
                "siblings": [
                    { "rfilename": "adapter.safetensors", "size": "5" }
                ]
            }),
            None,
        )
        .await;
        let siblings_settings = test_settings(siblings_url, None);

        let snapshot = HuggingFaceSnapshot::resolve(
            &client,
            &siblings_settings,
            "owner/lora",
            "main",
            &["*.safetensors".to_owned()],
        )
        .await
        .expect("siblings snapshot resolves");

        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.files[0].path, "adapter.safetensors");
        assert_eq!(snapshot.total_bytes(), Some(5));
    }

    #[derive(Clone)]
    struct HfStubState {
        payload: serde_json::Value,
        token: Option<String>,
    }

    async fn spawn_hf_stub(payload: serde_json::Value, token: Option<&str>) -> String {
        let state = HfStubState {
            payload,
            token: token.map(str::to_owned),
        };
        let app = Router::new()
            .route("/api/models/:owner/:repo/tree/:revision", get(hf_stub))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener binds");
        let address = listener.local_addr().expect("listener has address");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub serves");
        });
        format!("http://{address}")
    }

    async fn hf_stub(State(state): State<HfStubState>, headers: HeaderMap) -> Response {
        if let Some(token) = &state.token {
            let expected = format!("Bearer {token}");
            let authorized = headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                == Some(expected.as_str());
            if !authorized {
                return (
                    AxumStatusCode::UNAUTHORIZED,
                    Json(json!({ "error": "missing token" })),
                )
                    .into_response();
            }
        }
        Json(state.payload).into_response()
    }

    fn test_settings(huggingface_base_url: String, huggingface_token: Option<&str>) -> Settings {
        Settings {
            api_url: "http://127.0.0.1:8000".to_owned(),
            access_token: None,
            data_dir: PathBuf::from("data"),
            worker_id: "test-worker".to_owned(),
            poll_seconds: 1,
            heartbeat_seconds: 5,
            huggingface_base_url,
            huggingface_token: huggingface_token.map(str::to_owned),
        }
    }
}
