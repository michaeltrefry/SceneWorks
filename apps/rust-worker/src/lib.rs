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
    read_safetensors_header, reconcile_detected_family, SafetensorsHeaderError,
};
use sceneworks_core::lora_url::{
    lora_source_url_file_name, lora_source_url_file_stem, parse_lora_source_url_with_private,
    validate_public_ip,
};
use sceneworks_core::project_store::{ProjectStore, ProjectStoreError};
use serde::Deserialize;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::time::MissedTickBehavior;
use uuid::Uuid;

const INSTALL_MARKER: &str = ".sceneworks-download-complete.json";
const DEFAULT_API_URL: &str = "http://localhost:8000";
const DEFAULT_HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";
const DEFAULT_MAX_LORA_URL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_MODEL_URL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const DEFAULT_TRANSITION_DURATION_SECONDS: f64 = 0.5;
const PERSON_TRACK_SAMPLE_RATE_FPS: f64 = 2.0;
const PERSON_TRACK_MAX_SAMPLES: usize = 24;
const PERSON_TRACK_X_DRIFT: f64 = 0.018;

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
    pub max_lora_url_bytes: u64,
    pub max_model_url_bytes: u64,
    pub allow_private_lora_urls: bool,
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

#[derive(Debug, Clone, PartialEq)]
struct DiscoveredGpu {
    id: String,
    name: String,
    capabilities: Vec<WorkerCapability>,
    utilization: Option<WorkerUtilizationSnapshot>,
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
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
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
            let parts = line.split(',').map(str::trim).collect::<Vec<_>>();
            if parts.len() < 3 {
                return None;
            }
            let index = parts[0];
            let name = parts[1];
            let memory_mb = parts[2];
            Some(DiscoveredGpu {
                id: index.to_owned(),
                name: format!("{name} ({memory_mb} MB)"),
                capabilities: vec![
                    WorkerCapability::Placeholder,
                    WorkerCapability::Gpu,
                    WorkerCapability::Unknown("nvidia".to_owned()),
                ],
                utilization: utilization_from_parts(&parts),
            })
        })
        .collect()
}

fn utilization_from_parts(parts: &[&str]) -> Option<WorkerUtilizationSnapshot> {
    if parts.len() < 6 {
        return None;
    }
    Some(WorkerUtilizationSnapshot {
        memory_total_mb: parse_u64(parts[2]),
        memory_used_mb: parse_u64(parts[3]),
        memory_free_mb: parse_u64(parts[4]),
        gpu_load_percent: parse_f64(parts[5]),
    })
}

fn parse_u64(value: &str) -> Option<u64> {
    value.parse().ok()
}

fn parse_f64(value: &str) -> Option<f64> {
    value.parse().ok()
}

async fn gpu_utilization(gpu_id: &str) -> Option<WorkerUtilizationSnapshot> {
    if gpu_id == "cpu" {
        return None;
    }
    query_nvidia_gpus()
        .await
        .into_iter()
        .find(|gpu| gpu.id == gpu_id)
        .and_then(|gpu| gpu.utilization)
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
        utilization: None,
    }
}

fn fallback_gpu(gpu_id: &str) -> DiscoveredGpu {
    DiscoveredGpu {
        id: gpu_id.to_owned(),
        name: format!("GPU {gpu_id}"),
        capabilities: vec![WorkerCapability::Placeholder, WorkerCapability::Gpu],
        utilization: None,
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
            WorkerCapability::ModelImport,
            WorkerCapability::LoraImport,
            WorkerCapability::PersonDetect,
            WorkerCapability::PersonTrack,
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

/// Locates the first `.safetensors` under `dir`, reads its header, and
/// runs the architecture detector. Returns `Ok(None)` when no header is
/// available or the signature is inconclusive. Returns `Err(message)`
/// only when a file was found but its header is unreadable or malformed —
/// the caller surfaces that message via `fail_job`.
fn detect_family_in_target_dir(dir: &Path) -> Result<Option<String>, String> {
    let Some(safetensors_path) = first_safetensors_path(dir) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path).map_err(|error| match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect downloaded LoRA file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Downloaded LoRA file has an invalid safetensors header.".to_owned()
        }
    })?;
    Ok(detect_lora_family(&header))
}

async fn run_lora_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = optional_payload_string(&job.payload, "repo");
    let source_url = optional_payload_string(&job.payload, "sourceUrl");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "loraId")
        .or_else(|| optional_payload_string(&job.payload, "name"))
        .map(str::to_owned)
        .or_else(|| repo.map(str::to_owned))
        .or_else(|| source_url.and_then(|value| lora_source_url_file_stem(value).ok()))
        .map(|value| safe_download_dir(&value))
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
        import_lora_source_path(
            Path::new(source_path),
            &target_dir,
            payload_bool(&job.payload, "uploadedSourcePath"),
        )
        .await?;
    } else if let Some(source_url) = source_url {
        download_lora_source_url(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "LoRA import canceled by user.",
            },
            source_url,
            &target_dir,
        )
        .await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "LoRA import failed.",
            Some("Provide repo, sourceUrl, or sourcePath for LoRA import".to_owned()),
        )
        .await;
    }

    let detected_family = match detect_family_in_target_dir(&target_dir) {
        Ok(detected) => detected,
        Err(detail) => {
            return fail_job(api, &job.id, "LoRA import failed.", Some(detail)).await;
        }
    };
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    let resolved_family = match (supplied_family, detected_family) {
        (Some(supplied), Some(detected)) => {
            if supplied != detected {
                return fail_job(
                    api,
                    &job.id,
                    "LoRA import failed.",
                    Some(format!(
                        "LoRA file appears to be a {detected} model, but family was declared as {supplied}. Re-import with family {detected} or pick a different file."
                    )),
                )
                .await;
            }
            Some(supplied)
        }
        (None, Some(detected)) => Some(detected),
        (Some(supplied), None) => {
            eprintln!(
                "LoRA import job {}: architecture detection inconclusive; accepting supplied family {supplied}",
                job.id
            );
            Some(supplied)
        }
        (None, None) => None,
    };

    write_lora_install_marker(&target_dir, &job.payload, &job.id).await?;
    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let mut manifest_entry = manifest_entry;
        if let Some(family) = resolved_family {
            manifest_entry
                .entry("family")
                .or_insert(Value::String(family));
        }
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
        "sourceUrl".to_owned(),
        source_url
            .map(|value| Value::String(value.to_owned()))
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

fn model_family_detection_error(error: SafetensorsHeaderError) -> String {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect imported model file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Imported model file has an invalid safetensors header.".to_owned()
        }
    }
}

async fn run_model_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = optional_payload_string(&job.payload, "repo");
    let source_url = optional_payload_string(&job.payload, "sourceUrl");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "modelId")
        .map(safe_download_dir)
        .unwrap_or_else(|| "model".to_owned());
    let target_dir = resolve_model_import_target(
        settings,
        &job.payload,
        settings
            .data_dir
            .join("models")
            .join("imports")
            .join(target_name),
    )?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Importing,
            0.1,
            "Importing model.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Model import canceled before transfer started.",
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
        download_snapshot(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "Model import canceled by user.",
            },
            &target_dir,
            &snapshot,
            &mut progress,
        )
        .await?;
    } else if let Some(source_path) = source_path {
        import_lora_source_path(
            Path::new(source_path),
            &target_dir,
            payload_bool(&job.payload, "uploadedSourcePath"),
        )
        .await?;
    } else if let Some(source_url) = source_url {
        download_model_source_url(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "Model import canceled by user.",
            },
            source_url,
            &target_dir,
        )
        .await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "Model import failed.",
            Some("Provide repo, sourceUrl, or sourcePath for model import".to_owned()),
        )
        .await;
    }

    let detected_family = match detect_model_family(&target_dir) {
        Ok(detected) => detected,
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(model_family_detection_error(error)),
            )
            .await;
        }
    };
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    let resolved_family = match reconcile_detected_family(supplied_family, detected_family) {
        Ok(family) => family,
        Err(mismatch) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(format!(
                    "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await;
        }
    };

    write_model_install_marker(&target_dir, &job.payload, repo.unwrap_or(""), &job.id).await?;
    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let mut manifest_entry = manifest_entry;
        if let Some(family) = resolved_family.clone() {
            manifest_entry
                .entry("family")
                .or_insert(Value::String(family));
        }
        let model_type = manifest_entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("image")
            .to_owned();
        let family = manifest_entry
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_owned);
        apply_model_manifest_defaults(&mut manifest_entry, &model_type, family.as_deref());
        if let Some(paths) = manifest_entry
            .entry("paths")
            .or_insert_with(|| json!({}))
            .as_object_mut()
        {
            paths.insert(
                "model".to_owned(),
                Value::String(target_dir.display().to_string()),
            );
        }
        let manifest_path = model_manifest_target(settings, &job.payload)?;
        upsert_model_manifest_entry(&manifest_path, manifest_entry).await?;
    }

    let mut result = JsonObject::new();
    result.insert(
        "modelId".to_owned(),
        job.payload.get("modelId").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "repo".to_owned(),
        repo.map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "sourceUrl".to_owned(),
        source_url
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert(
        "family".to_owned(),
        resolved_family.map(Value::String).unwrap_or(Value::Null),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Model import completed.",
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
    let asset_id = fresh_asset_id();
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

async fn run_person_detect_job(
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
            "Preparing representative frame analysis.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Person detection canceled before frame extraction.",
    )
    .await?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Extracting,
            0.25,
            "Extracting representative frame.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_person_detect(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Person candidates detected.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn run_person_detect(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_file = source_asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset file is missing.".to_owned()))?;
    let source_media_rel = required_value_str(source_file, "path")?;
    let source_media_path = safe_project_path(&project_path, source_media_rel)?;
    if !source_media_path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "Source media not found: {}",
            source_media_path.display()
        )));
    }

    let duration = source_file
        .get("duration")
        .map_or(6.0, |value| value_f64(value, 6.0))
        .clamp(0.0, 3600.0);
    let timestamp = payload_f64(
        &job.payload,
        "sourceTimestamp",
        if duration > 0.0 { duration * 0.25 } else { 0.0 },
    )
    .clamp(0.0, duration.max(3600.0));

    let frames_dir = project_path.join("assets").join("frames");
    tokio::fs::create_dir_all(&frames_dir).await?;
    tokio::fs::create_dir_all(project_path.join("recipes")).await?;
    let asset_id = fresh_asset_id();
    let created_at = now_rfc3339();
    let filename = format!(
        "{}_person-frame_{}.png",
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
        cancel_message: "Person detection canceled by user.",
    };
    render_frame_png(
        "ffmpeg",
        &source_media_path,
        &temp_path,
        timestamp,
        1280,
        720,
        Some(ffmpeg_context),
    )
    .await?;
    tokio::fs::rename(&temp_path, &media_path).await?;

    let detections = candidate_people(1280, 720, source_asset_id, timestamp);
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
        "displayName": format!("Person selection frame from {source_display_name}"),
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": 1280,
            "height": 720,
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
            "mode": "person_detect",
            "model": "procedural-person-detector",
            "adapter": "procedural_person_tracking",
            "prompt": "Detect selectable people in representative frame",
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sourceTimestamp": timestamp,
                "detectionCount": detections.len(),
                "personDetectionActive": false
            },
            "rawAdapterSettings": { "sourcePath": source_rel }
        },
        "lineage": {
            "parents": [source_asset_id],
            "sourceAssetId": source_asset_id,
            "sourceTimestamp": timestamp,
            "jobId": job.id
        }
    });

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.78,
            "Saving representative frame and candidate boxes.",
            None,
            None,
            None,
        ),
    )
    .await?;
    if let Err(error) = check_cancel(
        api,
        &job.id,
        "Person detection canceled before asset promotion.",
    )
    .await
    {
        let _ = tokio::fs::remove_file(&media_path).await;
        return Err(error);
    }
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
    result.insert("frameAssetId".to_owned(), Value::String(asset_id));
    result.insert("frameAsset".to_owned(), asset);
    result.insert(
        "sourceAssetId".to_owned(),
        Value::String(source_asset_id.to_owned()),
    );
    result.insert("sourceTimestamp".to_owned(), json!(timestamp));
    result.insert("detections".to_owned(), Value::Array(detections));
    result.insert(
        "limits".to_owned(),
        json!({
            "maskStorage": "deferred",
            "correction": "single selected box corrections can be added to the track sidecar later"
        }),
    );
    Ok(result)
}

async fn run_person_track_job(
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
            "Preparing selected-person tracking.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(api, &job.id, "Person tracking canceled before saving.").await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Tracking,
            0.35,
            "Tracking selected person through sampled frames.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let result = run_person_track(api, settings, job).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Reusable person track saved.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

async fn run_person_track(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<JsonObject> {
    let project_id = required_payload_string(&job.payload, "projectId")?;
    let source_asset_id = required_payload_string(&job.payload, "sourceAssetId")?;
    let detection = job
        .payload
        .get("detection")
        .cloned()
        .filter(Value::is_object)
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Selected detection metadata is required".to_owned())
        })?;
    if detection
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .is_none()
    {
        return Err(WorkerError::InvalidPayload(
            "Selected detection metadata is required".to_owned(),
        ));
    }
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store.get_project(project_id)?;
    let project_path = PathBuf::from(project.path);
    let source_asset = store.get_asset(project_id, source_asset_id)?;
    let source_file = source_asset
        .get("file")
        .ok_or_else(|| WorkerError::InvalidPayload("Source asset file is missing.".to_owned()))?;
    let duration = source_file
        .get("duration")
        .map_or(6.0, |value| value_f64(value, 6.0))
        .clamp(1.0, 3600.0);
    let frames = track_frames_from_detection(&detection, duration);
    let average_confidence = frames
        .iter()
        .map(|frame| {
            frame
                .get("confidence")
                .map_or(0.0, |value| value_f64(value, 0.0))
        })
        .sum::<f64>()
        / (frames.len().max(1) as f64);
    let track_id = format!("track_{}", Uuid::new_v4().simple());
    let track_name =
        optional_payload_string(&job.payload, "trackName").unwrap_or("Selected person");
    let representative_frame_asset_id = job
        .payload
        .get("representativeFrameAssetId")
        .cloned()
        .unwrap_or(Value::Null);
    let raw_selected_detection = detection.clone();
    let created_at = now_rfc3339();
    let source_display_name = source_asset
        .get("displayName")
        .cloned()
        .unwrap_or(Value::Null);
    let track = json!({
        "schemaVersion": 1,
        "id": track_id.clone(),
        "projectId": project_id,
        "name": track_name,
        "createdAt": created_at,
        "sourceAssetId": source_asset_id,
        "sourceDisplayName": source_display_name,
        "representativeFrameAssetId": representative_frame_asset_id,
        "selectedDetection": detection,
        "frames": frames,
        "corrections": [],
        "status": {
            "sampleRateFps": PERSON_TRACK_SAMPLE_RATE_FPS,
            "maskState": "deferred",
            "averageConfidence": round_to(average_confidence, 3),
            "correctionState": "ready_for_box_corrections",
            "personTrackingActive": false
        },
        "recipe": {
            "mode": "person_track",
            "model": "procedural-person-tracker",
            "adapter": "procedural_person_tracking",
            "prompt": format!("Track {track_name}"),
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sampleRateFps": PERSON_TRACK_SAMPLE_RATE_FPS,
                "personDetectionActive": false,
                "personTrackingActive": false
            },
            "rawAdapterSettings": { "selectedDetection": raw_selected_detection }
        },
        "lineage": {
            "jobId": job.id,
            "parents": [source_asset_id, job.payload.get("representativeFrameAssetId").cloned().unwrap_or(Value::Null)]
        }
    });

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Saving,
            ProgressStage::Saving,
            0.82,
            "Saving reusable person track metadata.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Person tracking canceled before sidecar write.",
    )
    .await?;
    let track_path = project_path
        .join("person-tracks")
        .join(format!("{track_id}.sceneworks.person-track.json"));
    write_json_value(&track_path, &track).await?;
    let relative = relative_path(&project_path, &track_path)?;
    let mut result = JsonObject::new();
    result.insert("trackId".to_owned(), Value::String(track_id));
    result.insert("track".to_owned(), track);
    result.insert("path".to_owned(), Value::String(relative));
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

async fn download_lora_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "LoRA",
        context.settings.max_lora_url_bytes,
    )
    .await
}

async fn download_model_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "Model",
        context.settings.max_model_url_bytes,
    )
    .await
}

async fn download_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
    source_label: &str,
    max_bytes: u64,
) -> WorkerResult<()> {
    let url =
        parse_lora_source_url_with_private(source_url, context.settings.allow_private_lora_urls)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    validate_lora_url_dns(context.settings, &url).await?;
    let file_name = lora_source_url_file_name(source_url)
        .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    tokio::fs::create_dir_all(target_dir).await?;
    let target_path = target_dir.join(file_name);
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let total_bytes = lora_source_content_length(&client, source_url).await?;
    if total_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let existing_bytes = existing_download_bytes(&target_path, total_bytes).await?;
    if total_bytes.is_some_and(|total| total > 0 && existing_bytes == total) {
        return Ok(());
    }
    let mut request = client.get(source_url);
    if existing_bytes > 0 {
        request = request.header(header::RANGE, format!("bytes={existing_bytes}-"));
    }
    let mut response = request.send().await?;
    if response.status().is_redirection() {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl redirects are not allowed".to_owned(),
        ));
    }
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        let range_total = response
            .headers()
            .get(header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(content_range_total);
        if total_bytes
            .or(range_total)
            .is_some_and(|total| total > 0 && existing_bytes == total)
        {
            return Ok(());
        }
    }
    response = response.error_for_status()?;
    let appending = existing_bytes > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    let expected_bytes = total_bytes.or_else(|| {
        response.content_length().map(|remaining| {
            if appending {
                existing_bytes + remaining
            } else {
                remaining
            }
        })
    });
    if expected_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let mut progress = DownloadProgress::new(
        source_url,
        if appending { existing_bytes } else { 0 },
        expected_bytes,
        progress_report_interval(context.settings),
    );
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
    interval.tick().await;
    loop {
        tokio::select! {
            chunk = response.chunk() => {
                let Some(chunk) = chunk? else {
                    break;
                };
                check_cancel(context.api, context.job_id, context.cancel_message).await?;
                output.write_all(&chunk).await?;
                progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                if progress.downloaded_bytes() > max_bytes {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{source_label} sourceUrl exceeds the {} limit",
                        format_bytes(max_bytes)
                    )));
                }
            }
            _ = interval.tick() => {
                report_download_progress(context, &progress).await?;
            }
        }
    }
    if expected_bytes.is_some_and(|expected| progress.downloaded_bytes() != expected) {
        return Err(WorkerError::InvalidPayload(format!(
            "LoRA sourceUrl download ended at {} but expected {}",
            format_bytes(progress.downloaded_bytes()),
            format_bytes(expected_bytes.unwrap_or_default())
        )));
    }
    Ok(())
}

async fn lora_source_content_length(
    client: &reqwest::Client,
    source_url: &str,
) -> WorkerResult<Option<u64>> {
    let response = client.head(source_url).send().await?;
    if response.status().is_success() {
        return Ok(response.content_length().filter(|value| *value > 0));
    }
    if response.status().is_redirection() {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl redirects are not allowed".to_owned(),
        ));
    }
    if matches!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED | StatusCode::FORBIDDEN
    ) {
        return Ok(None);
    }
    response.error_for_status()?;
    Ok(None)
}

fn content_range_total(value: &str) -> Option<u64> {
    value
        .rsplit_once('/')
        .and_then(|(_, total)| total.trim().parse::<u64>().ok())
}

async fn validate_lora_url_dns(settings: &Settings, url: &reqwest::Url) -> WorkerResult<()> {
    if settings.allow_private_lora_urls {
        return Ok(());
    }
    let Some(host) = url.host_str() else {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host is not allowed".to_owned(),
        ));
    };
    if let Ok(address) = host.parse::<IpAddr>() {
        validate_public_ip(address)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
        return Ok(());
    }
    let port = url.port_or_known_default().unwrap_or(443);
    let mut resolved_any = false;
    for address in tokio::net::lookup_host((host, port)).await? {
        resolved_any = true;
        validate_public_ip(address.ip())
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    }
    if resolved_any {
        Ok(())
    } else {
        Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host did not resolve".to_owned(),
        ))
    }
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
    OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .expect("setting nanoseconds to zero must be valid")
        .format(&Rfc3339)
        .expect("formatting a UTC timestamp as RFC3339 must succeed")
}

fn candidate_people(width: u32, height: u32, source_asset_id: &str, timestamp: f64) -> Vec<Value> {
    let seed = format!("{source_asset_id}:{timestamp:.3}:{width}x{height}");
    let digest = Sha256::digest(seed.as_bytes());
    let templates = [
        (0.34, 0.16, 0.24, 0.68, 0.91),
        (0.58, 0.20, 0.20, 0.58, 0.78),
        (0.14, 0.26, 0.17, 0.50, 0.66),
    ];
    templates
        .iter()
        .enumerate()
        .map(|(index, (x, y, box_width, box_height, confidence))| {
            let jitter = ((digest[index] % 13) as f64 - 6.0) / 1000.0;
            json!({
                "id": format!("person_{}", index + 1),
                "label": format!("Person {}", index + 1),
                "confidence": round_to(*confidence - index as f64 * 0.04, 2),
                "box": {
                    "x": (*x + jitter).clamp(0.02, 0.92),
                    "y": *y,
                    "width": *box_width,
                    "height": *box_height
                },
                "maskState": "deferred",
                "frameWidth": width,
                "frameHeight": height
            })
        })
        .collect()
}

fn track_frames_from_detection(detection: &Value, duration: f64) -> Vec<Value> {
    let sample_count = ((duration.max(1.0) * PERSON_TRACK_SAMPLE_RATE_FPS).round() as usize)
        .clamp(3, PERSON_TRACK_MAX_SAMPLES);
    let base_confidence =
        value_f64(detection.get("confidence").unwrap_or(&Value::Null), 0.82).clamp(0.0, 1.0);
    (0..sample_count)
        .map(|index| {
            let t = index as f64 / (sample_count.saturating_sub(1).max(1) as f64);
            json!({
                "timestamp": round_to(t * duration.max(0.0), 3),
                "box": {
                    "x": round_to(detection_box_f64(detection, "x", 0.35, 0.0, 1.0) + (t - 0.5) * PERSON_TRACK_X_DRIFT, 4),
                    "y": round_to(detection_box_f64(detection, "y", 0.16, 0.0, 1.0), 4),
                    "width": round_to(detection_box_f64(detection, "width", 0.24, 0.01, 1.0), 4),
                    "height": round_to(detection_box_f64(detection, "height", 0.68, 0.01, 1.0), 4)
                },
                "confidence": 0.5_f64.max(round_to(base_confidence - index as f64 * 0.006, 3)),
                "mask": Value::Null
            })
        })
        .collect()
}

fn detection_box_f64(
    detection: &Value,
    field: &str,
    default: f64,
    min_value: f64,
    max_value: f64,
) -> f64 {
    detection
        .get("box")
        .and_then(|value| value.get(field))
        .map_or(default, |value| value_f64(value, default))
        .clamp(min_value, max_value)
}

fn round_to(value: f64, places: u32) -> f64 {
    let factor = 10_f64.powi(i32::try_from(places).unwrap_or(0));
    (value * factor).round() / factor
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
    let asset_id = fresh_asset_id();
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

    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode as AxumStatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use sceneworks_core::contracts::WorkerUtilizationSnapshot;
    use serde_json::{json, Value};
    use tempfile::tempdir;

    use super::{
        allow_pattern_matches, auto_worker_specs, bounded_tail, candidate_people,
        child_environment, cleanup_uploaded_import_source, concat_file_contents, copy_lora_source,
        cpu_gpu, cpu_worker_id, crossfade_duration, download_lora_source_url,
        download_progress_payload, fallback_gpu, fresh_asset_id, gpu_worker_id,
        import_lora_source_path, now_rfc3339, output_dimensions, parse_nvidia_smi_gpus,
        restart_exited_children_with_spawner, run_ffmpeg, safe_download_dir, safe_project_path,
        value_f64, visible_gpu_ids, worker_capabilities_with_utility, write_model_install_marker,
        ApiClient, DownloadContext, HuggingFaceSnapshot, Settings, SupervisedChild, WorkerError,
        WorkerSpec, DEFAULT_MAX_LORA_URL_BYTES, DEFAULT_MAX_MODEL_URL_BYTES,
        DEFAULT_TRANSITION_DURATION_SECONDS, INSTALL_MARKER,
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
            "0, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 4096, 93791, 12\n\
             1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887, 8192, 89695, 25\n",
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
        assert_eq!(
            gpus[0].utilization,
            Some(WorkerUtilizationSnapshot {
                memory_total_mb: Some(97887),
                memory_used_mb: Some(4096),
                memory_free_mb: Some(93791),
                gpu_load_percent: Some(12.0),
            })
        );

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
        assert!(cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "person_detect"));
        assert!(cpu_capabilities
            .iter()
            .any(|capability| capability.as_str() == "person_track"));
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

    #[tokio::test]
    async fn uploaded_lora_source_cleanup_removes_staged_file_and_parent() {
        let temp = tempdir().expect("tempdir creates");
        let upload_dir = temp.path().join("upload-1");
        tokio::fs::create_dir_all(&upload_dir).await.unwrap();
        let source_file = upload_dir.join("detail.safetensors");
        tokio::fs::write(&source_file, b"lora").await.unwrap();
        let mut payload = serde_json::Map::new();
        payload.insert(
            "sourcePath".to_owned(),
            json!(source_file.display().to_string()),
        );
        payload.insert("uploadedSourcePath".to_owned(), json!(true));

        cleanup_uploaded_import_source(&payload).await.unwrap();

        assert!(!source_file.exists());
        assert!(!upload_dir.exists());
    }

    #[tokio::test]
    async fn uploaded_lora_file_import_prefers_move_over_copy() {
        let temp = tempdir().expect("tempdir creates");
        let source_file = temp.path().join("uploaded.safetensors");
        tokio::fs::write(&source_file, b"lora").await.unwrap();
        let target_dir = temp.path().join("target");

        import_lora_source_path(&source_file, &target_dir, true)
            .await
            .unwrap();

        assert!(!source_file.exists());
        assert_eq!(
            tokio::fs::read(target_dir.join("uploaded.safetensors"))
                .await
                .unwrap(),
            b"lora"
        );
    }

    #[tokio::test]
    async fn lora_url_import_downloads_to_named_file() {
        let temp = tempdir().expect("tempdir creates");
        let source_url = spawn_binary_stub(b"url-lora".to_vec()).await;
        let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
        settings.api_url = source_url.clone();
        let api = ApiClient::new(&settings);
        let client = reqwest::Client::new();
        let target_dir = temp.path().join("url-target");

        download_lora_source_url(
            &DownloadContext {
                api: &api,
                client: &client,
                settings: &settings,
                job_id: "job-1",
                cancel_message: "canceled",
            },
            &format!("{source_url}/loras/style.safetensors"),
            &target_dir,
        )
        .await
        .expect("url LoRA downloads");

        assert_eq!(
            tokio::fs::read(target_dir.join("style.safetensors"))
                .await
                .unwrap(),
            b"url-lora"
        );
    }

    #[tokio::test]
    async fn lora_url_import_skips_existing_matching_file() {
        let temp = tempdir().expect("tempdir creates");
        let source_url = spawn_binary_stub(b"new-lora".to_vec()).await;
        let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
        settings.api_url = source_url.clone();
        let api = ApiClient::new(&settings);
        let client = reqwest::Client::new();
        let target_dir = temp.path().join("url-target");
        tokio::fs::create_dir_all(&target_dir).await.unwrap();
        tokio::fs::write(target_dir.join("style.safetensors"), b"old-lora")
            .await
            .unwrap();

        download_lora_source_url(
            &DownloadContext {
                api: &api,
                client: &client,
                settings: &settings,
                job_id: "job-1",
                cancel_message: "canceled",
            },
            &format!("{source_url}/loras/style.safetensors"),
            &target_dir,
        )
        .await
        .expect("existing LoRA is accepted");

        assert_eq!(
            tokio::fs::read(target_dir.join("style.safetensors"))
                .await
                .unwrap(),
            b"old-lora"
        );
    }

    #[tokio::test]
    async fn lora_url_import_rejects_failed_and_oversized_downloads() {
        let temp = tempdir().expect("tempdir creates");
        let missing_url =
            spawn_binary_stub_with_options(b"missing".to_vec(), AxumStatusCode::NOT_FOUND, false)
                .await;
        let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
        settings.api_url = missing_url.clone();
        let api = ApiClient::new(&settings);
        let client = reqwest::Client::new();

        let error = download_lora_source_url(
            &DownloadContext {
                api: &api,
                client: &client,
                settings: &settings,
                job_id: "job-1",
                cancel_message: "canceled",
            },
            &format!("{missing_url}/loras/missing.safetensors"),
            &temp.path().join("missing-target"),
        )
        .await
        .expect_err("failed URL returns an error");
        assert!(error.to_string().contains("404"));

        let large_url = spawn_binary_stub(b"too-large".to_vec()).await;
        settings.api_url = large_url.clone();
        settings.max_lora_url_bytes = 4;
        let api = ApiClient::new(&settings);
        let error = download_lora_source_url(
            &DownloadContext {
                api: &api,
                client: &client,
                settings: &settings,
                job_id: "job-1",
                cancel_message: "canceled",
            },
            &format!("{large_url}/loras/large.safetensors"),
            &temp.path().join("large-target"),
        )
        .await
        .expect_err("oversized URL returns an error");
        assert!(error.to_string().contains("exceeds"));
    }

    #[tokio::test]
    async fn lora_url_import_honors_midstream_cancel() {
        let temp = tempdir().expect("tempdir creates");
        let source_url =
            spawn_binary_stub_with_options(b"url-lora".to_vec(), AxumStatusCode::OK, true).await;
        let mut settings = test_settings("http://127.0.0.1".to_owned(), None);
        settings.api_url = source_url.clone();
        let api = ApiClient::new(&settings);
        let client = reqwest::Client::new();

        let error = download_lora_source_url(
            &DownloadContext {
                api: &api,
                client: &client,
                settings: &settings,
                job_id: "job-1",
                cancel_message: "LoRA import canceled by user.",
            },
            &format!("{source_url}/loras/style.safetensors"),
            &temp.path().join("cancel-target"),
        )
        .await
        .expect_err("cancel request interrupts the URL import");

        assert!(matches!(error, WorkerError::Canceled(_)));
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

        let asset_id = fresh_asset_id();
        assert!(asset_id.starts_with("asset_"));
        assert_eq!(asset_id.len(), "asset_".len() + 32);
        assert!(asset_id["asset_".len()..]
            .chars()
            .all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn person_detection_jitter_uses_python_sha256_bytes() {
        let detections = candidate_people(1280, 720, "asset_source_clip", 1.25);

        assert_eq!(detections[0]["box"]["x"].as_f64(), Some(0.338));
        assert_eq!(detections[1]["box"]["x"].as_f64(), Some(0.579));
        assert_eq!(detections[2]["box"]["x"].as_f64(), Some(0.134));
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

    #[derive(Clone)]
    struct BinaryStubState {
        bytes: Vec<u8>,
        status: AxumStatusCode,
        cancel_requested: bool,
    }

    async fn spawn_binary_stub(bytes: Vec<u8>) -> String {
        spawn_binary_stub_with_options(bytes, AxumStatusCode::OK, false).await
    }

    async fn spawn_binary_stub_with_options(
        bytes: Vec<u8>,
        status: AxumStatusCode,
        cancel_requested: bool,
    ) -> String {
        let state = BinaryStubState {
            bytes,
            status,
            cancel_requested,
        };
        let app = Router::new()
            .route("/api/v1/jobs/:job_id", get(job_stub))
            .route("/api/v1/jobs/:job_id/progress", post(progress_stub))
            .route("/*path", get(binary_stub).head(binary_head_stub))
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

    async fn binary_stub(State(state): State<BinaryStubState>, headers: HeaderMap) -> Response {
        let length = state.bytes.len();
        if headers
            .get(axum::http::header::RANGE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value == format!("bytes={length}-"))
        {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = AxumStatusCode::RANGE_NOT_SATISFIABLE;
            response.headers_mut().insert(
                axum::http::header::CONTENT_RANGE,
                axum::http::HeaderValue::from_str(&format!("bytes */{length}"))
                    .expect("content range header"),
            );
            return response;
        }
        let mut response = state.bytes.into_response();
        *response.status_mut() = state.status;
        response.headers_mut().insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&length.to_string()).expect("content length header"),
        );
        response
    }

    async fn binary_head_stub(State(state): State<BinaryStubState>) -> Response {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = state.status;
        response.headers_mut().insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&state.bytes.len().to_string())
                .expect("content length header"),
        );
        response
    }

    async fn job_stub(
        State(state): State<BinaryStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
    }

    async fn progress_stub(
        State(state): State<BinaryStubState>,
        axum::extract::Path(job_id): axum::extract::Path<String>,
    ) -> Response {
        Json(job_snapshot_json(&job_id, state.cancel_requested)).into_response()
    }

    fn job_snapshot_json(job_id: &str, cancel_requested: bool) -> Value {
        json!({
            "id": job_id,
            "type": "lora_import",
            "status": "running",
            "projectId": null,
            "projectName": null,
            "payload": {},
            "result": {},
            "requestedGpu": "auto",
            "assignedGpu": null,
            "workerId": "test-worker",
            "progress": 0.1,
            "stage": "importing",
            "message": "running",
            "error": null,
            "etaSeconds": null,
            "elapsedSeconds": null,
            "attempts": 1,
            "sourceJobId": null,
            "duplicateOfJobId": null,
            "cancelRequested": cancel_requested,
            "createdAt": "2026-05-18T00:00:00Z",
            "updatedAt": "2026-05-18T00:00:00Z",
            "startedAt": null,
            "completedAt": null,
            "canceledAt": null,
            "lastHeartbeatAt": null
        })
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
            max_lora_url_bytes: DEFAULT_MAX_LORA_URL_BYTES,
            max_model_url_bytes: DEFAULT_MAX_MODEL_URL_BYTES,
            allow_private_lora_urls: true,
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
