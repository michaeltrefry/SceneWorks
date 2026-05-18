use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use reqwest::header;
use reqwest::StatusCode;
use sceneworks_core::contracts::{
    ClaimRequest, ClaimResponse, ContractNumber, JobSnapshot, JobStatus, JobType, JsonObject,
    ProgressRequest, ProgressStage, WorkerCapability, WorkerHeartbeatRequest,
    WorkerRegisterRequest, WorkerSnapshot, WorkerStatus,
};
use sceneworks_core::project_store::{ProjectStore, ProjectStoreError};
use serde::Deserialize;
use serde_json::{json, Number, Value};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time::MissedTickBehavior;
use uuid::Uuid;

const INSTALL_MARKER: &str = ".sceneworks-download-complete.json";
const DEFAULT_API_URL: &str = "http://localhost:8000";
const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const DEFAULT_TRANSITION_DURATION_SECONDS: f64 = 0.5;

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
            config_dir: env_path("SCENEWORKS_CONFIG_DIR", "config"),
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
        }
    }

    fn for_worker(&self, worker_id: String, gpu_id: String) -> Self {
        let mut settings = self.clone();
        settings.worker_id = worker_id;
        settings.gpu_id = gpu_id;
        settings.is_child_worker = true;
        settings
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

#[derive(Debug, Clone, Eq, PartialEq)]
struct DiscoveredGpu {
    id: String,
    name: String,
    capabilities: Vec<WorkerCapability>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct WorkerSpec {
    worker_id: String,
    gpu_id: String,
}

struct SupervisedChild {
    spec: WorkerSpec,
    process: Child,
    restart_attempt: u32,
}

async fn supervise_auto_workers(settings: Settings) -> WorkerResult<()> {
    let gpus = discover_gpus().await;
    if gpus.is_empty() {
        let cpu_settings =
            settings.for_worker(cpu_worker_id(&settings.worker_id), "cpu".to_owned());
        return run_worker_loop(cpu_settings).await;
    }

    let mut children = HashMap::new();
    for spec in auto_worker_specs(&settings.worker_id, &gpus) {
        let process = start_child_worker(&settings, &spec)?;
        children.insert(
            spec.worker_id.clone(),
            SupervisedChild {
                spec,
                process,
                restart_attempt: 0,
            },
        );
    }

    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = shutdown_signal() => {
                stop_children(&settings, &mut children).await;
                return Ok(());
            }
            _ = interval.tick() => {
                restart_exited_children(&settings, &mut children).await?;
            }
        }
    }
}

async fn restart_exited_children(
    settings: &Settings,
    children: &mut HashMap<String, SupervisedChild>,
) -> WorkerResult<()> {
    restart_exited_children_with_spawner(settings, children, start_child_worker).await
}

async fn restart_exited_children_with_spawner<F>(
    settings: &Settings,
    children: &mut HashMap<String, SupervisedChild>,
    mut spawner: F,
) -> WorkerResult<()>
where
    F: FnMut(&Settings, &WorkerSpec) -> WorkerResult<Child>,
{
    let mut exited = Vec::new();
    for (worker_id, child) in children.iter_mut() {
        if let Some(status) = child.process.try_wait()? {
            let restart_attempt = child.restart_attempt.saturating_add(1);
            let delay = retry_delay(settings.poll_seconds, restart_attempt);
            emit_json(json!({
                "event": "worker_exited",
                "workerId": worker_id,
                "gpuId": child.spec.gpu_id,
                "exitCode": status.code(),
                "restartInSeconds": delay,
                "reportedAt": now_rfc3339(),
            }));
            exited.push(worker_id.clone());
        }
    }
    for worker_id in exited {
        let Some(mut child) = children.remove(&worker_id) else {
            continue;
        };
        child.restart_attempt = child.restart_attempt.saturating_add(1);
        let delay = retry_delay(settings.poll_seconds, child.restart_attempt);
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
            _ = shutdown_signal() => {
                children.insert(worker_id, child);
                stop_children(settings, children).await;
                return Ok(());
            }
        }
        let process = spawner(settings, &child.spec)?;
        child.process = process;
        children.insert(child.spec.worker_id.clone(), child);
    }
    Ok(())
}

async fn stop_children(settings: &Settings, children: &mut HashMap<String, SupervisedChild>) {
    for child in children.values_mut() {
        terminate_child(&mut child.process).await;
    }
    let deadline = tokio::time::sleep(Duration::from_secs(
        settings.shutdown_timeout_seconds.max(1),
    ));
    tokio::pin!(deadline);
    loop {
        let mut remaining = 0_usize;
        for child in children.values_mut() {
            match child.process.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => remaining += 1,
                Err(_) => {}
            }
        }
        if remaining == 0 {
            children.clear();
            return;
        }
        tokio::select! {
            _ = &mut deadline => break,
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
    for child in children.values_mut() {
        let _ = child.process.start_kill();
        let _ = child.process.wait().await;
    }
    children.clear();
}

async fn terminate_child(child: &mut Child) {
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGTERM,
            );
            return;
        }
    }
    let _ = child.start_kill();
}

fn start_child_worker(_settings: &Settings, spec: &WorkerSpec) -> WorkerResult<Child> {
    let executable = std::env::current_exe()?;
    emit_json(json!({
        "event": "starting_worker",
        "workerId": spec.worker_id,
        "gpuId": spec.gpu_id,
        "reportedAt": now_rfc3339(),
    }));
    let mut command = Command::new(executable);
    command.envs(child_environment(spec));
    command.spawn().map_err(Into::into)
}

fn child_environment(spec: &WorkerSpec) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    env.insert("SCENEWORKS_WORKER_CHILD".to_owned(), "1".to_owned());
    env.insert("SCENEWORKS_WORKER_ID".to_owned(), spec.worker_id.clone());
    env.insert("SCENEWORKS_GPU_ID".to_owned(), spec.gpu_id.clone());
    if spec.gpu_id == "cpu" {
        env.insert("CUDA_VISIBLE_DEVICES".to_owned(), String::new());
        env.insert("SCENEWORKS_UTILITY_JOBS".to_owned(), "1".to_owned());
    } else {
        env.insert("CUDA_VISIBLE_DEVICES".to_owned(), spec.gpu_id.clone());
        env.insert("SCENEWORKS_UTILITY_JOBS".to_owned(), "0".to_owned());
    }
    env
}

fn auto_worker_specs(base_worker_id: &str, gpus: &[DiscoveredGpu]) -> Vec<WorkerSpec> {
    let mut specs = gpus
        .iter()
        .map(|gpu| WorkerSpec {
            worker_id: gpu_worker_id(base_worker_id, &gpu.id),
            gpu_id: gpu.id.clone(),
        })
        .collect::<Vec<_>>();
    specs.push(WorkerSpec {
        worker_id: cpu_worker_id(base_worker_id),
        gpu_id: "cpu".to_owned(),
    });
    specs
}

async fn discover_gpu(requested_gpu_id: &str) -> DiscoveredGpu {
    if requested_gpu_id == "cpu" {
        return cpu_gpu();
    }
    let gpus = discover_gpus().await;
    if requested_gpu_id.is_empty() || requested_gpu_id == "auto" {
        return gpus.into_iter().next().unwrap_or_else(cpu_gpu);
    }
    gpus.into_iter()
        .find(|gpu| gpu.id == requested_gpu_id)
        .unwrap_or_else(|| fallback_gpu(requested_gpu_id))
}

async fn discover_gpus() -> Vec<DiscoveredGpu> {
    let visible_ids = visible_gpu_ids_from_env();
    if visible_ids.as_ref().is_some_and(Vec::is_empty) {
        return Vec::new();
    }
    let gpus = query_nvidia_gpus().await;
    if let Some(ids) = visible_ids {
        let by_id = gpus
            .into_iter()
            .map(|gpu| (gpu.id.clone(), gpu))
            .collect::<BTreeMap<_, _>>();
        return ids
            .into_iter()
            .map(|gpu_id| {
                by_id
                    .get(&gpu_id)
                    .cloned()
                    .unwrap_or_else(|| fallback_gpu(&gpu_id))
            })
            .collect();
    }
    gpus
}

async fn query_nvidia_gpus() -> Vec<DiscoveredGpu> {
    let output = tokio::time::timeout(
        Duration::from_secs(3),
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output(),
    )
    .await;
    match output {
        Ok(Ok(output)) if output.status.success() => {
            parse_nvidia_smi_gpus(&String::from_utf8_lossy(&output.stdout))
        }
        _ => Vec::new(),
    }
}

fn parse_nvidia_smi_gpus(output: &str) -> Vec<DiscoveredGpu> {
    output
        .trim()
        .lines()
        .filter_map(|line| {
            let parts = line.splitn(3, ',').map(str::trim).collect::<Vec<_>>();
            let [index, name, memory_mb] = parts.as_slice() else {
                return None;
            };
            Some(DiscoveredGpu {
                id: (*index).to_owned(),
                name: format!("{name} ({memory_mb} MB)"),
                capabilities: vec![
                    WorkerCapability::Placeholder,
                    WorkerCapability::Gpu,
                    WorkerCapability::Unknown("nvidia".to_owned()),
                ],
            })
        })
        .collect()
}

fn visible_gpu_ids_from_env() -> Option<Vec<String>> {
    visible_gpu_ids(std::env::var("NVIDIA_VISIBLE_DEVICES").ok().as_deref())
}

fn visible_gpu_ids(value: Option<&str>) -> Option<Vec<String>> {
    let value = value.map(str::trim).filter(|value| !value.is_empty())?;
    match value {
        "all" => None,
        "void" | "none" => Some(Vec::new()),
        _ => Some(
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect(),
        ),
    }
}

fn cpu_gpu() -> DiscoveredGpu {
    DiscoveredGpu {
        id: "cpu".to_owned(),
        name: "Rust CPU utility worker".to_owned(),
        capabilities: vec![WorkerCapability::Placeholder, WorkerCapability::Cpu],
    }
}

fn fallback_gpu(gpu_id: &str) -> DiscoveredGpu {
    DiscoveredGpu {
        id: gpu_id.to_owned(),
        name: format!("GPU {gpu_id}"),
        capabilities: vec![WorkerCapability::Placeholder, WorkerCapability::Gpu],
    }
}

fn worker_capabilities(gpu: &DiscoveredGpu) -> Vec<WorkerCapability> {
    let utility_jobs_enabled =
        std::env::var("SCENEWORKS_UTILITY_JOBS").map_or(true, |value| value.trim() != "0");
    worker_capabilities_with_utility(gpu, utility_jobs_enabled)
}

fn worker_capabilities_with_utility(
    gpu: &DiscoveredGpu,
    utility_jobs_enabled: bool,
) -> Vec<WorkerCapability> {
    let mut capabilities = gpu.capabilities.clone();
    let is_cpu = capabilities.contains(&WorkerCapability::Cpu);
    if is_cpu && utility_jobs_enabled {
        capabilities.extend([
            WorkerCapability::FrameExtract,
            WorkerCapability::TimelineExport,
            WorkerCapability::ModelDownload,
            WorkerCapability::LoraImport,
        ]);
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

fn gpu_worker_id(base_worker_id: &str, gpu_id: &str) -> String {
    let safe_gpu_id = slugify_worker_id_part(gpu_id, "gpu");
    if safe_gpu_id == "0" && base_worker_id.ends_with("-0") {
        return base_worker_id.to_owned();
    }
    if base_worker_id.ends_with("-0") && safe_gpu_id.chars().all(|value| value.is_ascii_digit()) {
        return format!(
            "{}{}",
            &base_worker_id[..base_worker_id.len() - 1],
            safe_gpu_id
        );
    }
    format!("{base_worker_id}-gpu-{safe_gpu_id}")
}

fn cpu_worker_id(base_worker_id: &str) -> String {
    let base = base_worker_id.strip_suffix("-0").unwrap_or(base_worker_id);
    format!("{base}-cpu")
}

fn slugify_worker_id_part(value: &str, fallback: &str) -> String {
    let mut output = String::new();
    let mut previous_dash = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            previous_dash = false;
        } else if !previous_dash && !output.is_empty() {
            output.push('-');
            previous_dash = true;
        }
    }
    while output.ends_with('-') {
        output.pop();
    }
    if output.is_empty() {
        fallback.to_owned()
    } else {
        output
    }
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
    if settings.gpu_id == "auto" && !settings.is_child_worker {
        return supervise_auto_workers(settings).await;
    }
    run_worker_loop(settings).await
}

async fn run_worker_loop(settings: Settings) -> WorkerResult<()> {
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
        JobType::Placeholder => run_placeholder_job(api, settings, &job)
            .await
            .map_err(|error| ("Placeholder job failed.", error)),
        JobType::ModelDownload => run_model_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model download failed.", error)),
        JobType::LoraImport => run_lora_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA import failed.", error)),
        JobType::FrameExtract => run_frame_extract_job(api, settings, &job)
            .await
            .map_err(|error| ("Frame extraction failed.", error)),
        JobType::TimelineExport => run_timeline_export_job(api, settings, &job)
            .await
            .map_err(|error| ("Timeline export failed.", error)),
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
    let target_dir = resolve_lora_import_target(
        settings,
        &job.payload,
        settings.data_dir.join("loras").join(target_name),
    )?;

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

    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let manifest_path = lora_manifest_target(settings, &job.payload)?;
        upsert_lora_manifest_entry(&manifest_path, manifest_entry).await?;
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

#[derive(Debug, Clone)]
struct TimelineExportRequest {
    project_id: String,
    timeline_id: String,
    timeline_name: String,
    timeline_path: String,
    resolution: u32,
    fps: u32,
}

#[derive(Clone, Copy)]
struct FfmpegContext<'a> {
    api: &'a ApiClient,
    settings: &'a Settings,
    job_id: &'a str,
    cancel_message: &'a str,
}

async fn run_frame_extract_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.08,
            "Preparing frame extraction.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Frame extraction canceled before reading media.",
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Extracting,
            0.25,
            "Extracting timeline frame.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_frame_extract(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Timeline frame saved as an asset.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn run_frame_extract(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let timestamp = payload_f64(&job.payload, "sourceTimestamp", 0.0).clamp(0.0, 3600.0);
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_media_rel = required_value_str(
        source_asset.get("file").ok_or_else(|| {
            WorkerError::InvalidPayload("Source asset file is missing.".to_owned())
        })?,
        "path",
    )?;
    let source_media_path = safe_project_path(&project_path, source_media_rel)?;
    if !source_media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Source media not found: {}",
            source_media_path.display()
        )));
    }

    let frames_dir = project_path.join("assets").join("frames");
    tokio::fs::create_dir_all(&frames_dir).await?;
    tokio::fs::create_dir_all(project_path.join("recipes")).await?;
    let asset_id = fresh_asset_id(&job.id);
    let created_at = now_rfc3339();
    let filename = format!(
        "{}_frame_{}.png",
        &created_at[..10],
        asset_suffix(&asset_id)
    );
    let media_rel = format!("assets/frames/{filename}");
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");

    let ffmpeg_context = FfmpegContext {
        api,
        settings,
        job_id: &job.id,
        cancel_message: "Frame extraction canceled by user.",
    };
    render_frame_png(
        "ffmpeg",
        &source_media_path,
        &temp_path,
        timestamp,
        1920,
        1080,
        Some(ffmpeg_context),
    )
    .await?;
    tokio::fs::rename(&temp_path, &media_path).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.85,
            "Saving extracted frame asset.",
            None,
            None,
            None,
        ),
    )
    .await?;
    if let Err(error) = check_cancel(
        api,
        &job.id,
        "Frame extraction canceled before asset promotion.",
    )
    .await
    {
        let _ = tokio::fs::remove_file(&media_path).await;
        return Err(error);
    }

    let timeline_id = job
        .payload
        .get("timelineId")
        .cloned()
        .unwrap_or(Value::Null);
    let timeline_item_id = job
        .payload
        .get("timelineItemId")
        .cloned()
        .unwrap_or(Value::Null);
    let playhead_seconds = job
        .payload
        .get("playheadSeconds")
        .cloned()
        .unwrap_or(Value::Null);
    let intended_use = optional_payload_string(&job.payload, "intendedUse").unwrap_or("reuse");
    let source_display_name = source_asset
        .get("displayName")
        .and_then(Value::as_str)
        .unwrap_or("clip");
    let source_rel = relative_path(&project_path, &source_media_path)?;
    let asset = json!({
        "schemaVersion": 1,
        "id": asset_id.clone(),
        "projectId": project_id,
        "generationSetId": Value::Null,
        "type": "frame",
        "displayName": format!("Frame {timestamp:.2}s from {source_display_name}"),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": 1920,
            "height": 1080,
            "duration": Value::Null,
            "fps": Value::Null
        },
        "status": {
            "favorite": false,
            "rating": 0,
            "rejected": false,
            "trashed": false
        },
        "recipe": {
            "mode": "frame_extract",
            "model": "timeline-frame-extract",
            "adapter": "ffmpeg-frame-extract",
            "prompt": format!("Extract frame at {timestamp:.2}s"),
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "timelineId": timeline_id,
                "timelineItemId": timeline_item_id,
                "playheadSeconds": playhead_seconds,
                "sourceTimestamp": timestamp,
                "intendedUse": intended_use
            },
            "rawAdapterSettings": { "sourcePath": source_rel }
        },
        "lineage": {
            "parents": [source_asset_id],
            "sourceAssetId": source_asset_id,
            "sourceTimestamp": timestamp,
            "timelineId": job.payload.get("timelineId").cloned().unwrap_or(Value::Null),
            "timelineItemId": job.payload.get("timelineItemId").cloned().unwrap_or(Value::Null),
            "intendedUse": intended_use,
            "jobId": job.id
        }
    });
    let sidecar_path = media_path.with_extension("sceneworks.json");
    write_json_value(&sidecar_path, &asset).await?;
    write_json_value(
        &project_path
            .join("recipes")
            .join(format!("{asset_id}.recipe.json")),
        &asset["recipe"],
    )
    .await?;
    store.index_asset_sidecar(project_id, &sidecar_path)?;

    let mut result = JsonObject::new();
    result.insert("assetIds".to_owned(), json!([asset_id]));
    result.insert("assets".to_owned(), json!([asset]));
    result.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    result.insert("sourceTimestamp".to_owned(), json!(timestamp));
    result.insert(
        "timelineId".to_owned(),
        job.payload
            .get("timelineId")
            .cloned()
            .unwrap_or(Value::Null),
    );
    result.insert(
        "timelineItemId".to_owned(),
        job.payload
            .get("timelineItemId")
            .cloned()
            .unwrap_or(Value::Null),
    );
    Ok(result)
}

async fn run_timeline_export_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.06,
            "Preparing timeline export.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Timeline export canceled before rendering.").await?;
    let request = export_request_from_job(job)?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let timeline_path = safe_project_path(&project_path, &request.timeline_path)?;
    let timeline = read_json_value(&timeline_path).await?;
    let (width, height) = output_dimensions(
        timeline
            .get("aspectRatio")
            .and_then(Value::as_str)
            .unwrap_or("16:9"),
        request.resolution,
    );
    let mut items = main_track_items(&timeline);
    items.sort_by(|left, right| {
        item_f64(left, "timelineStart", 0.0).total_cmp(&item_f64(right, "timelineStart", 0.0))
    });
    if items.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no main video items to export.".to_owned(),
        ));
    }

    let temp_dir = tempfile::Builder::new()
        .prefix(&format!(
            "sceneworks_export_{}_",
            safe_download_dir(&job.id)
        ))
        .tempdir()?;
    let tmp_path = temp_dir.path().to_path_buf();

    let mut segments = Vec::new();
    let mut cursor = 0.0_f64;
    let total = items.len().max(1);
    let render_spec = RenderSpec {
        width,
        height,
        fps: request.fps,
    };
    let ffmpeg_context = FfmpegContext {
        api,
        settings,
        job_id: &job.id,
        cancel_message: "Timeline export canceled by user.",
    };
    let render_result = async {
        for (index, item) in items.iter().enumerate() {
            check_cancel(api, &job.id, "Timeline export canceled by user.").await?;
            let start = item_f64(item, "timelineStart", 0.0);
            let item_end = item_f64(item, "timelineEnd", start);
            if item_end <= start {
                return Err(WorkerError::InvalidPayload(
                    "timelineEnd must be greater than timelineStart.".to_owned(),
                ));
            }
            if start > cursor {
                let gap_duration = start - cursor;
                let gap_path = tmp_path.join(format!("segment_{:04}_gap.mp4", segments.len()));
                render_black_segment(
                    "ffmpeg",
                    &gap_path,
                    gap_duration,
                    render_spec,
                    Some(ffmpeg_context),
                )
                .await?;
                segments.push(TimelineSegment {
                    path: gap_path,
                    duration: gap_duration,
                    transition: None,
                    transition_duration: 0.0,
                });
                cursor = start;
            }

            let asset_id = required_value_str(item, "assetId")?;
            let asset = store.get_asset(&request.project_id, asset_id)?;
            let display_name = item
                .get("displayName")
                .and_then(Value::as_str)
                .unwrap_or("item");
            let segment_path = tmp_path.join(format!(
                "segment_{:04}_{}.mp4",
                segments.len(),
                slugify(display_name, "timeline-export", Some(48))
            ));
            let duration = render_item_segment(
                "ffmpeg",
                &project_path,
                item,
                &asset,
                &segment_path,
                render_spec,
                Some(ffmpeg_context),
            )
            .await?;
            let transition_in = item.get("transitionIn").unwrap_or(&Value::Null);
            segments.push(TimelineSegment {
                path: segment_path,
                duration,
                transition: transition_in
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                transition_duration: value_f64(
                    transition_in.get("duration").unwrap_or(&Value::Null),
                    DEFAULT_TRANSITION_DURATION_SECONDS,
                ),
            });
            cursor = cursor.max(item_end);
            update_job(
                api,
                &job.id,
                progress_payload(
                    JobStatus::Running,
                    ProgressStage::Rendering,
                    0.12 + (((index + 1) as f64 / total as f64) * 0.58),
                    "Rendering timeline segments.",
                    None,
                    None,
                    None,
                ),
            )
            .await?;
        }
        WorkerResult::Ok(())
    }
    .await;

    render_result?;

    let output_rel = format!(
        "assets/renders/{}_{}_{}.mp4",
        &now_rfc3339()[..10],
        slugify(&request.timeline_name, "timeline-export", Some(48)),
        asset_suffix(&job.id)
    );
    let output_path = project_path.join(&output_rel);
    tokio::fs::create_dir_all(output_path.parent().ok_or_else(|| {
        WorkerError::InvalidPayload("Render output has no parent directory.".to_owned())
    })?)
    .await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Muxing,
            0.78,
            "Muxing MP4 export.",
            None,
            None,
            None,
        ),
    )
    .await?;
    mux_segments(
        "ffmpeg",
        &segments,
        &tmp_path,
        &output_path,
        Some(ffmpeg_context),
    )
    .await?;

    let asset = build_render_asset(
        &request,
        &timeline,
        &job.id,
        &output_rel,
        width,
        height,
        cursor,
    );
    let sidecar_path = output_path.with_extension("sceneworks.json");
    write_json_value(&sidecar_path, &asset).await?;
    tokio::fs::create_dir_all(project_path.join("recipes")).await?;
    let asset_id = required_value_str(&asset, "id")?.to_owned();
    write_json_value(
        &project_path
            .join("recipes")
            .join(format!("{asset_id}.recipe.json")),
        &asset["recipe"],
    )
    .await?;
    store.index_asset_sidecar(&request.project_id, &sidecar_path)?;

    let mut result = JsonObject::new();
    result.insert("assetIds".to_owned(), json!([asset_id]));
    result.insert("assets".to_owned(), json!([asset]));
    result.insert(
        "timelineId".to_owned(),
        Value::String(request.timeline_id.clone()),
    );
    result.insert("renderPath".to_owned(), Value::String(output_rel));
    result.insert(
        "adapter".to_owned(),
        Value::String("ffmpeg_timeline".to_owned()),
    );
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Timeline MP4 export saved.",
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
    OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .expect("setting nanoseconds to zero must be valid")
        .format(&Rfc3339)
        .expect("formatting a UTC timestamp as RFC3339 must succeed")
}

fn export_request_from_job(job: &JobSnapshot) -> WorkerResult<TimelineExportRequest> {
    Ok(TimelineExportRequest {
        project_id: required_payload_string(&job.payload, "projectId")?.to_owned(),
        timeline_id: required_payload_string(&job.payload, "timelineId")?.to_owned(),
        timeline_name: optional_payload_string(&job.payload, "timelineName")
            .unwrap_or("Timeline")
            .to_owned(),
        timeline_path: required_payload_string(&job.payload, "timelinePath")?.to_owned(),
        resolution: payload_u32(&job.payload, "resolution", 720).clamp(240, 2160),
        fps: payload_u32(&job.payload, "fps", 30).clamp(1, 60),
    })
}

async fn render_frame_png(
    ffmpeg: &str,
    source_path: &Path,
    output_path: &Path,
    timestamp: f64,
    width: u32,
    height: u32,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x12110f,format=rgb24"
    );
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-ss".to_owned(),
            format!("{:.3}", timestamp.max(0.0)),
            "-i".to_owned(),
            source_path.display().to_string(),
            "-frames:v".to_owned(),
            "1".to_owned(),
            "-vf".to_owned(),
            filters,
            "-f".to_owned(),
            "image2".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await?;
    if !tokio::fs::try_exists(output_path).await? {
        return Err(WorkerError::InvalidPayload(format!(
            "FFmpeg did not produce frame output: {}",
            output_path.display()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct TimelineSegment {
    path: PathBuf,
    duration: f64,
    transition: Option<String>,
    transition_duration: f64,
}

fn main_track_items(timeline: &Value) -> Vec<Value> {
    timeline
        .get("tracks")
        .and_then(Value::as_array)
        .and_then(|tracks| {
            tracks
                .iter()
                .find(|track| {
                    track.get("id").and_then(Value::as_str) == Some("track_main")
                        || track.get("kind").and_then(Value::as_str) == Some("video")
                })
                .and_then(|track| track.get("items").and_then(Value::as_array))
        })
        .cloned()
        .unwrap_or_default()
}

fn output_dimensions(aspect_ratio: &str, resolution: u32) -> (u32, u32) {
    let resolution = resolution.max(2);
    let (width, height) = match aspect_ratio {
        "9:16" => (resolution, ((resolution as f64) * 16.0 / 9.0).ceil() as u32),
        "1:1" => (resolution, resolution),
        _ => (((resolution as f64) * 16.0 / 9.0).ceil() as u32, resolution),
    };
    (even(width), even(height))
}

fn even(value: u32) -> u32 {
    if value % 2 == 0 {
        value
    } else {
        value + 1
    }
}

#[derive(Debug, Clone, Copy)]
struct RenderSpec {
    width: u32,
    height: u32,
    fps: u32,
}

async fn render_black_segment(
    ffmpeg: &str,
    output_path: &Path,
    duration: f64,
    spec: RenderSpec,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "lavfi".to_owned(),
            "-i".to_owned(),
            format!(
                "color=c=black:s={}x{}:r={}",
                spec.width, spec.height, spec.fps
            ),
            "-t".to_owned(),
            format!("{duration:.3}"),
            "-pix_fmt".to_owned(),
            "yuv420p".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await
}

async fn render_item_segment(
    ffmpeg: &str,
    project_path: &Path,
    item: &Value,
    asset: &Value,
    output_path: &Path,
    spec: RenderSpec,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<f64> {
    let file = asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Timeline asset file is missing.".to_owned()))?;
    let media_rel = required_value_str(file, "path")?;
    let media_path = safe_project_path(project_path, media_rel)?;
    if !media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Timeline source file is missing: {}",
            media_path.display()
        )));
    }

    let source_in = item_f64(item, "sourceIn", 0.0);
    let source_out = item_f64(item, "sourceOut", item_f64(item, "timelineEnd", 4.0));
    let timeline_duration =
        item_f64(item, "timelineEnd", 4.0) - item_f64(item, "timelineStart", 0.0);
    let source_duration = (source_out - source_in).max(0.1);
    let speed = item_f64(item, "speed", 1.0).max(0.1);
    let duration = if timeline_duration > 0.0 {
        timeline_duration.max(0.1)
    } else {
        (source_duration / speed).max(0.1)
    };
    let mut vf = vec![
        format!(
            "scale={}:{}:force_original_aspect_ratio=decrease",
            spec.width, spec.height
        ),
        format!(
            "pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=black",
            spec.width, spec.height
        ),
        format!("fps={}", spec.fps),
        "format=yuv420p".to_owned(),
    ];
    let transition_in = item.get("transitionIn").unwrap_or(&Value::Null);
    let transition_out = item.get("transitionOut").unwrap_or(&Value::Null);
    if transition_in.get("type").and_then(Value::as_str) == Some("fade_from_black") {
        let fade_duration = duration.min(value_f64(
            transition_in.get("duration").unwrap_or(&Value::Null),
            0.5,
        ));
        vf.push(format!("fade=t=in:st=0:d={fade_duration:.3}"));
    }
    if transition_out.get("type").and_then(Value::as_str) == Some("fade_to_black") {
        let fade_duration = duration.min(value_f64(
            transition_out.get("duration").unwrap_or(&Value::Null),
            0.5,
        ));
        vf.push(format!(
            "fade=t=out:st={:.3}:d={fade_duration:.3}",
            (duration - fade_duration).max(0.0)
        ));
    }

    let media_type = asset.get("type").and_then(Value::as_str);
    let mime_type = file
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let is_image_source = media_type != Some("video")
        && (media_type == Some("image") || mime_type.starts_with("image/"));
    if is_image_source {
        run_ffmpeg(
            vec![
                ffmpeg.to_owned(),
                "-y".to_owned(),
                "-loop".to_owned(),
                "1".to_owned(),
                "-framerate".to_owned(),
                spec.fps.to_string(),
                "-i".to_owned(),
                media_path.display().to_string(),
                "-t".to_owned(),
                format!("{duration:.3}"),
                "-vf".to_owned(),
                vf.join(","),
                "-an".to_owned(),
                output_path.display().to_string(),
            ],
            context,
        )
        .await?;
        return Ok(duration);
    }

    let setpts = format!("setpts={:.6}*PTS", 1.0 / speed);
    let filters = std::iter::once(setpts)
        .chain(vf)
        .collect::<Vec<_>>()
        .join(",");
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-ss".to_owned(),
            format!("{source_in:.3}"),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-t".to_owned(),
            format!("{source_duration:.3}"),
            "-vf".to_owned(),
            filters,
            "-an".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await?;
    Ok(duration)
}

async fn mux_segments(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    tmp_path: &Path,
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    if segments
        .iter()
        .skip(1)
        .any(|segment| segment.transition.as_deref() == Some("crossfade"))
    {
        return mux_with_crossfades(ffmpeg, segments, tmp_path, output_path, context).await;
    }
    let list_path = tmp_path.join("concat.txt");
    tokio::fs::write(
        &list_path,
        concat_file_contents(segments.iter().map(|segment| &segment.path)),
    )
    .await?;
    run_ffmpeg(
        vec![
            ffmpeg.to_owned(),
            "-y".to_owned(),
            "-f".to_owned(),
            "concat".to_owned(),
            "-safe".to_owned(),
            "0".to_owned(),
            "-i".to_owned(),
            list_path.display().to_string(),
            "-c".to_owned(),
            "copy".to_owned(),
            output_path.display().to_string(),
        ],
        context,
    )
    .await
}

async fn mux_with_crossfades(
    ffmpeg: &str,
    segments: &[TimelineSegment],
    tmp_path: &Path,
    output_path: &Path,
    context: Option<FfmpegContext<'_>>,
) -> WorkerResult<()> {
    let Some(first) = segments.first() else {
        return Err(WorkerError::InvalidPayload(
            "Timeline has no rendered segments to mux.".to_owned(),
        ));
    };
    let mut current = first.path.clone();
    let mut current_duration = first.duration;
    for (index, segment) in segments.iter().enumerate().skip(1) {
        let merged = tmp_path.join(format!("xfade_{index:04}.mp4"));
        if segment.transition.as_deref() == Some("crossfade") {
            let duration = crossfade_duration(segment.transition_duration);
            let offset = (current_duration - duration).max(0.0);
            run_ffmpeg(
                vec![
                    ffmpeg.to_owned(),
                    "-y".to_owned(),
                    "-i".to_owned(),
                    current.display().to_string(),
                    "-i".to_owned(),
                    segment.path.display().to_string(),
                    "-filter_complex".to_owned(),
                    format!(
                    "[0:v][1:v]xfade=transition=fade:duration={duration:.3}:offset={offset:.3},format=yuv420p[v]"
                ),
                    "-map".to_owned(),
                    "[v]".to_owned(),
                    merged.display().to_string(),
                ],
                context,
            )
            .await?;
            current_duration += segment.duration - duration;
        } else {
            let list_path = tmp_path.join(format!("concat_{index:04}.txt"));
            tokio::fs::write(
                &list_path,
                concat_file_contents([&current, &segment.path].into_iter()),
            )
            .await?;
            run_ffmpeg(
                vec![
                    ffmpeg.to_owned(),
                    "-y".to_owned(),
                    "-f".to_owned(),
                    "concat".to_owned(),
                    "-safe".to_owned(),
                    "0".to_owned(),
                    "-i".to_owned(),
                    list_path.display().to_string(),
                    "-c".to_owned(),
                    "copy".to_owned(),
                    merged.display().to_string(),
                ],
                context,
            )
            .await?;
            current_duration += segment.duration;
        }
        current = merged;
    }
    tokio::fs::rename(current, output_path).await?;
    Ok(())
}

fn crossfade_duration(duration: f64) -> f64 {
    duration.clamp(0.1, 1.5)
}

fn concat_file_contents<'a>(paths: impl Iterator<Item = &'a PathBuf>) -> String {
    paths
        .map(|path| {
            let path = path
                .display()
                .to_string()
                .replace('\\', "/")
                .replace('\'', "'\\''");
            format!("file '{path}'\n")
        })
        .collect()
}

fn build_render_asset(
    request: &TimelineExportRequest,
    timeline: &Value,
    job_id: &str,
    media_rel: &str,
    width: u32,
    height: u32,
    duration: f64,
) -> Value {
    let asset_id = fresh_asset_id(job_id);
    let created_at = now_rfc3339();
    let source_asset_ids = timeline
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|track| track.get("items").and_then(Value::as_array))
        .flatten()
        .filter_map(|item| item.get("assetId").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let aspect_ratio = timeline
        .get("aspectRatio")
        .and_then(Value::as_str)
        .unwrap_or("16:9");
    json!({
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": request.project_id,
        "generationSetId": Value::Null,
        "type": "render",
        "displayName": format!("{} export", request.timeline_name),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "video/mp4",
            "width": width,
            "height": height,
            "duration": (duration * 1000.0).round() / 1000.0,
            "fps": request.fps
        },
        "status": {
            "favorite": false,
            "rating": 0,
            "rejected": false,
            "trashed": false
        },
        "recipe": {
            "mode": "timeline_export",
            "model": "ffmpeg",
            "adapter": "ffmpeg_timeline",
            "prompt": request.timeline_name,
            "negativePrompt": "",
            "seed": Value::Null,
            "loras": [],
            "normalizedSettings": {
                "timelineId": request.timeline_id,
                "resolution": request.resolution,
                "width": width,
                "height": height,
                "fps": request.fps,
                "aspectRatio": aspect_ratio
            },
            "rawAdapterSettings": {
                "timelinePath": request.timeline_path,
                "renderer": "ffmpeg segment concat"
            }
        },
        "lineage": {
            "parents": source_asset_ids,
            "sourceAssetId": request.timeline_id,
            "sourceTimestamp": Value::Null,
            "jobId": job_id
        }
    })
}

async fn run_ffmpeg(args: Vec<String>, context: Option<FfmpegContext<'_>>) -> WorkerResult<()> {
    let Some((program, arguments)) = args.split_first() else {
        return Err(WorkerError::InvalidPayload(
            "FFmpeg command is empty.".to_owned(),
        ));
    };
    let mut child = Command::new(program)
        .args(arguments)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Failed to start FFmpeg. Ensure ffmpeg is installed and on PATH: {error}"
            ))
        })?;

    let mut stderr = child.stderr.take();
    let stderr_task = tokio::spawn(async move {
        let mut bytes = Vec::new();
        if let Some(stderr) = stderr.as_mut() {
            let _ = stderr.read_to_end(&mut bytes).await;
        }
        bytes
    });

    let status = if let Some(context) = context {
        let mut interval = tokio::time::interval(progress_report_interval(context.settings));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                status = child.wait() => break status?,
                _ = interval.tick() => {
                    heartbeat(context.api, context.settings, WorkerStatus::Busy, Some(context.job_id)).await?;
                    if let Err(error) = check_cancel(context.api, context.job_id, context.cancel_message).await {
                        let _ = child.kill().await;
                        let _ = child.wait().await;
                        return Err(error);
                    }
                }
            }
        }
    } else {
        child.wait().await?
    };

    let stderr = stderr_task.await.unwrap_or_default();
    if status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&stderr);
    let bounded = bounded_tail(&stderr, 10, 2000);
    if bounded.trim().is_empty() {
        Err(WorkerError::InvalidPayload(
            "FFmpeg command failed without stderr output.".to_owned(),
        ))
    } else {
        Err(WorkerError::InvalidPayload(bounded))
    }
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

fn fresh_asset_id(job_id: &str) -> String {
    let _ = job_id;
    format!("asset_{}", Uuid::new_v4().simple())
}

fn asset_suffix(value: &str) -> String {
    let safe = safe_download_dir(value);
    let chars = safe.chars().rev().take(8).collect::<Vec<_>>();
    chars.into_iter().rev().collect::<String>()
}

fn slugify(value: &str, fallback: &str, max_length: Option<usize>) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash && !slug.is_empty() {
            slug.push('-');
            previous_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        slug = fallback.to_owned();
    }
    if let Some(max_length) = max_length {
        slug.truncate(max_length);
    }
    slug
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
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::process::Stdio as StdStdio;
    use std::time::Duration;

    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode as AxumStatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        allow_pattern_matches, auto_worker_specs, bounded_tail, child_environment,
        concat_file_contents, copy_lora_source, cpu_gpu, cpu_worker_id, crossfade_duration,
        download_progress_payload, fallback_gpu, fresh_asset_id, gpu_worker_id, now_rfc3339,
        output_dimensions, parse_nvidia_smi_gpus, restart_exited_children_with_spawner, run_ffmpeg,
        safe_download_dir, safe_project_path, value_f64, visible_gpu_ids,
        worker_capabilities_with_utility, write_model_install_marker, HuggingFaceSnapshot,
        Settings, SupervisedChild, WorkerError, WorkerSpec, DEFAULT_TRANSITION_DURATION_SECONDS,
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

    #[test]
    fn nvidia_smi_parsing_and_visible_device_filtering_match_python_worker() {
        let gpus = parse_nvidia_smi_gpus(
            "0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887\n\
             1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887\n",
        );

        assert_eq!(
            gpus.iter().map(|gpu| gpu.id.as_str()).collect::<Vec<_>>(),
            ["0", "1"]
        );
        assert_eq!(
            gpus[0].name,
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (97887 MB)"
        );
        assert!(gpus[1]
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == "nvidia"));
        assert!(gpus[1]
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == "placeholder"));

        assert_eq!(visible_gpu_ids(None), None);
        assert_eq!(visible_gpu_ids(Some("all")), None);
        assert_eq!(visible_gpu_ids(Some("none")), Some(Vec::new()));
        assert_eq!(
            visible_gpu_ids(Some("0, GPU-abcd")),
            Some(vec!["0".to_owned(), "GPU-abcd".to_owned()])
        );
    }

    #[test]
    fn auto_worker_ids_and_child_environment_match_python_supervisor() {
        assert_eq!(gpu_worker_id("worker-gpu-auto-0", "0"), "worker-gpu-auto-0");
        assert_eq!(gpu_worker_id("worker-gpu-auto-0", "1"), "worker-gpu-auto-1");
        assert_eq!(cpu_worker_id("worker-gpu-auto-0"), "worker-gpu-auto-cpu");

        let gpus = vec![fallback_gpu("0"), fallback_gpu("1")];
        let specs = auto_worker_specs("worker-gpu-auto-0", &gpus);
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.worker_id.as_str())
                .collect::<Vec<_>>(),
            [
                "worker-gpu-auto-0",
                "worker-gpu-auto-1",
                "worker-gpu-auto-cpu"
            ]
        );
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.gpu_id.as_str())
                .collect::<Vec<_>>(),
            ["0", "1", "cpu"]
        );

        let gpu_env = child_environment(&WorkerSpec {
            worker_id: "worker-gpu-auto-1".to_owned(),
            gpu_id: "1".to_owned(),
        });
        assert_eq!(gpu_env["SCENEWORKS_UTILITY_JOBS"], "0");
        assert_eq!(gpu_env["CUDA_VISIBLE_DEVICES"], "1");

        let cpu_env = child_environment(&WorkerSpec {
            worker_id: "worker-gpu-auto-cpu".to_owned(),
            gpu_id: "cpu".to_owned(),
        });
        assert_eq!(cpu_env["SCENEWORKS_UTILITY_JOBS"], "1");
        assert_eq!(cpu_env["CUDA_VISIBLE_DEVICES"], "");
    }

    #[test]
    fn rust_cpu_capabilities_do_not_claim_gpu_generation_jobs() {
        let cpu_capabilities = worker_capabilities_with_utility(&cpu_gpu(), true);

        assert!(cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "model_download"));
        assert!(cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "placeholder"));
        assert!(cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "timeline_export"));
        assert!(!cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "image_generate"));
        assert!(!cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "video_generate"));

        let gpu_capabilities = worker_capabilities_with_utility(&fallback_gpu("0"), false);
        assert!(gpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "gpu"));
        assert!(gpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "placeholder"));
        assert!(!gpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "model_download"));
        assert!(!gpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "image_generate"));
    }

    #[tokio::test]
    async fn supervisor_restarts_exited_children_with_backoff_state() {
        let settings = test_settings("http://127.0.0.1".to_owned(), None);
        let spec = WorkerSpec {
            worker_id: "worker-gpu-auto-0".to_owned(),
            gpu_id: "0".to_owned(),
        };
        let mut exited = spawn_exit_child();
        for _ in 0..20 {
            if exited.try_wait().expect("child status checks").is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let mut children = HashMap::from([(
            spec.worker_id.clone(),
            SupervisedChild {
                spec,
                process: exited,
                restart_attempt: 0,
            },
        )]);
        let mut spawns = 0_u32;

        restart_exited_children_with_spawner(&settings, &mut children, |_settings, _spec| {
            spawns += 1;
            Ok(spawn_sleep_child())
        })
        .await
        .expect("child restarts");

        assert_eq!(spawns, 1);
        let child = children
            .get_mut("worker-gpu-auto-0")
            .expect("restarted child is tracked");
        assert_eq!(child.restart_attempt, 1);
        assert!(child
            .process
            .try_wait()
            .expect("child status checks")
            .is_none());
        let _ = child.process.start_kill();
        let _ = child.process.wait().await;
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

    #[test]
    fn ffmpeg_helper_shapes_match_python_timeline_exporter() {
        assert_eq!(output_dimensions("16:9", 720), (1280, 720));
        assert_eq!(output_dimensions("9:16", 720), (720, 1280));
        assert_eq!(output_dimensions("1:1", 721), (722, 722));

        let concat = concat_file_contents(
            [
                PathBuf::from(r"C:\renders\clip one's.mp4"),
                PathBuf::from("nested/two.mp4"),
            ]
            .iter(),
        );
        assert!(concat.contains("C:/renders/clip one'\\''s.mp4"));
        assert!(concat.contains("file 'nested/two.mp4'"));

        let asset_id = fresh_asset_id("job-ignored");
        assert!(asset_id.starts_with("asset_"));
        assert_eq!(asset_id.len(), "asset_".len() + 32);
        assert!(asset_id["asset_".len()..]
            .chars()
            .all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn missing_crossfade_duration_defaults_to_python_mux_duration() {
        let missing = json!(null);
        assert_eq!(
            value_f64(&missing, DEFAULT_TRANSITION_DURATION_SECONDS),
            0.5
        );
        assert_eq!(crossfade_duration(0.5), 0.5);
        assert_eq!(crossfade_duration(0.0), 0.1);
        assert_eq!(crossfade_duration(2.0), 1.5);
    }

    #[test]
    fn path_and_error_helpers_are_bounded_and_defensive() {
        let temp = tempdir().expect("tempdir creates");
        let error = safe_project_path(temp.path(), "").expect_err("empty relative path rejects");
        assert!(error
            .to_string()
            .contains("Project-relative path is required"));

        let noisy = (0..100)
            .map(|index| format!("line {index} caf\u{e9}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tail = bounded_tail(&noisy, 10, 37);

        assert!(tail.contains("caf\u{e9}"));
        assert!(!tail.contains("line 1 "));
    }

    #[tokio::test]
    async fn ffmpeg_runner_surfaces_bounded_stderr_from_failing_process() {
        let args = if cfg!(windows) {
            let command = (1..=30)
                .map(|index| format!("echo ffmpeg-line-{index} 1>&2"))
                .collect::<Vec<_>>()
                .join(" & ");
            vec![
                "cmd".to_owned(),
                "/C".to_owned(),
                format!("{command} & exit /B 7"),
            ]
        } else {
            vec![
                "sh".to_owned(),
                "-c".to_owned(),
                "for i in $(seq 1 30); do echo ffmpeg-line-$i >&2; done; exit 7".to_owned(),
            ]
        };

        let error = run_ffmpeg(args, None)
            .await
            .expect_err("non-zero process returns an error");

        match error {
            WorkerError::InvalidPayload(message) => {
                assert!(message.contains("ffmpeg-line-30"));
                assert!(!message.contains("ffmpeg-line-1"));
                assert!(message.len() <= 2000);
            }
            other => panic!("expected InvalidPayload, got {other:?}"),
        }
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
            config_dir: PathBuf::from("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: true,
            poll_seconds: 1,
            heartbeat_seconds: 5,
            shutdown_timeout_seconds: 1,
            huggingface_base_url,
            huggingface_token: huggingface_token.map(str::to_owned),
        }
    }

    fn spawn_exit_child() -> tokio::process::Child {
        let mut command = if cfg!(windows) {
            let mut command = tokio::process::Command::new("cmd");
            command.args(["/C", "exit /B 0"]);
            command
        } else {
            let mut command = tokio::process::Command::new("sh");
            command.args(["-c", "exit 0"]);
            command
        };
        command
            .stdout(StdStdio::null())
            .stderr(StdStdio::null())
            .spawn()
            .expect("test child starts")
    }

    fn spawn_sleep_child() -> tokio::process::Child {
        let mut command = if cfg!(windows) {
            let mut command = tokio::process::Command::new("cmd");
            command.args(["/C", "ping -n 30 127.0.0.1 >NUL"]);
            command
        } else {
            let mut command = tokio::process::Command::new("sh");
            command.args(["-c", "sleep 30"]);
            command
        };
        command
            .stdout(StdStdio::null())
            .stderr(StdStdio::null())
            .spawn()
            .expect("test child starts")
    }
}
