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
use sceneworks_core::hf_home::{huggingface_hub_cache_dir, huggingface_repo_cache_path};
use sceneworks_core::jsonc::strip_jsonc_comments;
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
use tracing::Level;
use uuid::Uuid;

// Shared `advanced` knob accessors (sc-4281). The MLX image/video job paths are macOS-gated; the
// candle InstantID lane (sc-5491) is the first off-Mac caller, so the module also compiles on the
// Windows candle build. The candle lane calls only a subset (`flag`/`str`/`f32_clamped`), so allow
// dead_code there (the rest are MLX-only) — same pattern as `openpose_skeleton`. On a non-candle
// Windows/Linux build it stays excluded, so its accessors are never uncalled-dead there.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod advanced;
mod api_client;
// Lazy, on-demand download-credential pull from the macOS desktop credential socket
// (sc-5891). Compiles on all targets; the socket I/O is `cfg(unix)` and inert unless
// the desktop injects `SCENEWORKS_CRED_IPC_*`, so server/Docker/Windows are unaffected.
mod credentials_ipc;
// Backend-neutral generator load/run cache (epic 3720, sc-3724). Typed entirely against
// `gen_core::*` (no tensor types leak), so it links on ALL targets — the production load seam
// (`with_cached_generator`) is reached only from the macOS image/video paths, but the all-targets
// stub test exercises the load→progress→cancel→output contract with no backend linked. Off macOS
// the production caller is cfg'd out, so allow dead_code there (the engines.rs precedent).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod generator_cache;
use api_client::*;
// Backend-neutral engine dispatch table + registry-derived capability advertisement
// (sc-3723). All-targets: the table is pure data and the derivation runs off-macOS off an
// (empty) registry, so a future candle backend lights up with zero worker changes. Off
// macOS the only consumers are the (all-targets) registry-derivation tests — the production
// caller (`mlx_gpu`) is macOS-gated — so allow dead_code on the non-macOS lib build (the
// person_replace pattern); the stub test still exercises it on every target.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod engines;
mod gpu;
use gpu::*;
mod supervisor;
use supervisor::*;
mod model_jobs;
use model_jobs::*;
mod media_jobs;
use media_jobs::*;
// Image-decode backstop (sc-6143): transcodes a valid-but-unsupported image (AVIF/HEIC/HEIF/TIFF/
// BMP/GIF) to PNG at decode time. Compiles on all targets; the transcoder is the shared
// `sceneworks_core::media_convert` routine (sips on macOS, ffmpeg elsewhere).
mod image_decode;
mod image_jobs;
use image_jobs::*;
// SenseNova-U1 understanding + interleave jobs (epic 3180, sc-3905 — Path B). VQA + Document
// Studio (interleave) consume the concrete `T2iModel` directly (the `Generator` contract emits
// Images/Video only). The handlers are compiled cross-platform (with non-macOS error stubs); the
// real in-process MLX work is macOS-gated inside the module.
mod sensenova_jobs;
use sensenova_jobs::*;
mod video_jobs;
use video_jobs::*;
// Replace-person mask pipeline (epic 3040, sc-3521): cross-platform mask rasterization /
// resample / stored-seg-mask load, so the mask-port-vs-Python parity test runs on the
// Linux CI lane. Its masks are consumed only by the macOS Wan-VACE path in `video_jobs`,
// so off macOS the items are otherwise unused (the parity tests still build + run).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod person_replace;
mod training_jobs;
use training_jobs::*;
mod caption_jobs;
use caption_jobs::*;
mod prompt_refine_jobs;
use prompt_refine_jobs::*;
mod downloads;
// The DWPose skeleton rasterizer is consumed only by the macOS Z-Image strict-pose
// control path; on Mac AND the off-Mac candle DWPose lane (sc-5496) it backs the
// `pose_jobs` skeleton render; on a candle-disabled box off Mac it still builds +
// unit-tests (cross-platform raster) but its items are otherwise unused — so allow
// dead_code only there.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
mod openpose_skeleton;
// DWPose pose detection via onnxruntime (epic 3482, sc-3487). On Mac the CoreML EP +
// on the off-Mac candle GPU-worker lane the CUDA EP (sc-5496, epic 5482) run the same
// RTMW detector in-process; on a candle-disabled box the Python rtmlib path stays the
// Windows/Linux backend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod pose_jobs;
// CUDA execution-provider dependency preloading for the off-Mac candle `ort` paths
// (sc-6209, epic 5482): `ort::ep::cuda::preload_dylibs` dlopens the CUDA-12 runtime +
// cuDNN-9 DLLs the onnxruntime CUDA EP needs, so it engages the GPU regardless of PATH
// (the Mac CoreML path needs no equivalent). Shared by pose_jobs (DWPose, sc-5496) +
// person_jobs (YOLO, sc-5498), and Real-ESRGAN (sc-5499) next — gated to the candle GPU
// lane only.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod ort_cuda;
// SCRFD 5-point face-landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac, plus the
// candle SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482) — the same
// InstantID face-stack detector reused in-process for the Key Point Library "extract kps from this
// image" capability. So the module compiles on Mac AND the candle lane; on a candle-disabled box the
// Python InsightFace path stays the Windows/Linux backend.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod kps_jobs;
// Image upscaling: Real-ESRGAN (epic 3482, sc-3489) RRDBNet x2/x4 via `ort`/CoreML on Mac, plus the
// SeedVR2 one-step diffusion upscaler — native MLX on Mac (sc-4815) and the candle CUDA backend on
// Windows (sc-5928). So the module compiles on Mac AND the Windows/CUDA candle lane; the ort/CoreML
// Real-ESRGAN path inside stays Mac-gated (the Python torch Real-ESRGAN / AuraSR path is the
// Windows/Linux backend), while the SeedVR2 path is backend-neutral (`gen_core::load("seedvr2")`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod upscale_jobs;
// YOLO11 person detection + selected-person ByteTrack tracking (epic 3482, sc-3488/sc-3633;
// off-Mac candle lane sc-5498, epic 5482). Native-MLX YOLO11m on Mac, `ort`/CUDA on the off-Mac
// candle GPU-worker lane (the pure-Rust ByteTrack in `person_track` is backend-neutral). So both
// modules compile on Mac AND the candle lane; on a candle-disabled box the Python Ultralytics
// path stays the Windows/Linux backend. Person *segmentation* (SAM masks) stays Mac-only
// (`person_segment*` below) — off-Mac tracks are box-only; a candle SAM backport is epic 3792.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_jobs;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod person_track;
// Native-MLX SAM2 person segmentation (epic 3704, sc-3709): the `mlx-gen-sam2`
// box-prompt segmenter generates per-frame masks in `run_person_track`. macOS-only
// like person_jobs (mlx-gen builds MLX from source); the Python SAM2 path stays the
// Windows/Linux backend.
#[cfg(target_os = "macos")]
mod person_segment;
// SAM3 text-concept (PCS) person segmenter — the box-prompt-free upgrade of `person_segment`
// (epic 4910, sc-4926). macOS-only (native MLX `mlx-gen-sam3`); the off-Mac Windows/CUDA candle
// sibling is `person_segment_sam3_candle` below.
#[cfg(target_os = "macos")]
mod person_segment_sam3;
// Smart-select image segmentation (epic 6087, sc-6105): the `image_segment` job runs SAM3
// box-prompt segmentation in-process to produce an inpaint mask asset for the Image Editor.
// macOS-only like its `person_segment_sam3` (SAM3) dependency; no torch/candle image-segment path.
#[cfg(target_os = "macos")]
mod segment_jobs;
// Off-Mac candle SAM3 text-concept person segmenter (sc-6247, epic 5482 under sc-5062) — the
// Windows/CUDA sibling of `person_segment_sam3`, driving `candle-gen-sam3`'s `Sam3VideoModel` to
// replace the SAM2 box-prompt STUB in the off-Mac person-track (`media_jobs` `maskState = "missing"`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
mod person_segment_sam3_candle;
// SCAIL-2 color-coded segmentation-mask painting (epic 5439, sc-5448): turns native SAM3
// per-person masks into the palette-painted RGB masks the SCAIL-2 engine consumes. macOS-only
// like its SAM3 dependency.
#[cfg(target_os = "macos")]
mod scail2_masks;
use downloads::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use kps_jobs::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use pose_jobs::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use upscale_jobs::*;

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
    /// Whether the MLX (Apple Silicon) tensor backend is enabled when deriving the worker's
    /// advertised capabilities from the linked engine registry (sc-3723). Default `true`.
    pub backend_mlx_enabled: bool,
    /// Whether the candle (Windows/CUDA) tensor backend is enabled for capability derivation
    /// (sc-3723). Default `false` — no candle provider crate ships yet; flipping this on once
    /// one is linked lights up its descriptors with no further worker change.
    pub backend_candle_enabled: bool,
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
            backend_mlx_enabled: env_bool("SCENEWORKS_BACKEND_MLX_ENABLED", true),
            backend_candle_enabled: env_bool("SCENEWORKS_BACKEND_CANDLE_ENABLED", false),
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
    Engine(String),
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
            Self::Engine(detail) => formatter.write_str(detail),
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

fn task_join_error(label: &str, error: tokio::task::JoinError) -> WorkerError {
    WorkerError::Io(std::io::Error::other(format!("{label}: {error}")))
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

/// Emit a pre-built structured-event object (already carrying its `event` key) at a
/// **declared** level through the `tracing` backbone. The format-adaptive subscriber
/// renders the `{ event, level, reportedAt, ... }` line on stdout (captured into the
/// per-process log file + the in-app Logs buffer); `reportedAt` is stamped at render
/// time. Replaces the old `println!` of the same JSON so the level is now authoritative
/// rather than inferred from the line text downstream.
fn emit_event_value(level: Level, payload: Value) {
    sceneworks_core::observability::emit_event(level, payload);
}

/// Emit a structured worker event at **info** level (the per-generation lifecycle
/// events — pipeline load / inference start+complete — that the Rust MLX path mirrors
/// from the torch worker, sc-3450). `event` is injected into `payload`.
// Only the macOS image-generation path emits these today; on other targets the
// generation code is cfg'd out, so the helper would be dead code.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn emit_event(event: &str, payload: Value) {
    let mut value = payload;
    if let Some(object) = value.as_object_mut() {
        object.insert("event".to_owned(), Value::String(event.to_owned()));
    }
    emit_event_value(Level::INFO, value);
}

pub async fn run() -> WorkerResult<()> {
    // Install the tracing backbone before anything emits (covers both the
    // standalone `sceneworks-rust-worker` binary and the API's GPU-worker path,
    // which both funnel here). Idempotent — a second call is a no-op.
    sceneworks_core::observability::init_logging();
    // Host mode (no HF cache env set): default HF_HOME to the shared ~/.cache/
    // huggingface so downloads land in the OS cache rather than the private data
    // dir (sc-1904 follow-up). Set before spawning child workers so they inherit
    // it; desktop/Compose already inject HF_HOME, making this a no-op there.
    if let Some(home) = sceneworks_core::hf_home::ensure_default_huggingface_home() {
        tracing::info!(
            event = "hf_home_defaulted",
            home = %home.display(),
            "rust_worker defaulting HF_HOME"
        );
    }
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
    // sc-4482 (epic 3720): log the resolved backend-neutral gen-core contract version at startup
    // so a pin skew that slips past the CI guard (`scripts/check-gen-core-skew.sh`) is
    // diagnosable from one log line. One shared contract version backs every linked backend.
    tracing::info!(
        event = "gen_core_contract_version",
        version = %gen_core::VERSION,
        gpuId = %settings.gpu_id,
        "rust_worker gen-core contract version"
    );
    let gpu = discover_gpu(&settings).await;
    let api = ApiClient::new(&settings);
    let http_client = reqwest::Client::new();
    register_worker_with_retry(&api, &settings, &gpu).await?;
    let mut lock_failures = 0_u32;
    let mut idle_heartbeat = IdleHeartbeat::new(progress_report_interval(&settings));
    loop {
        tokio::select! {
            result = poll_once(&api, &settings, &http_client, &mut idle_heartbeat) => {
                match result {
                    Ok(()) => lock_failures = 0,
                    Err(error) if is_database_locked(&error) => {
                        // SQLite claim contention. With busy_timeout + BEGIN IMMEDIATE in the
                        // store this should be rare, but back off (instead of hammering at the
                        // flat poll interval) and make it visible so an MLX-eligible job lost to
                        // lock contention is explained rather than silently retried into torch.
                        lock_failures = lock_failures.saturating_add(1);
                        let delay = retry_delay(settings.poll_seconds, lock_failures);
                        emit_event_value(
                            Level::WARN,
                            json!({
                                "event": "claim_lock_contention",
                                "workerId": settings.worker_id,
                                "gpuId": settings.gpu_id,
                                "consecutiveFailures": lock_failures,
                                "retryInSeconds": delay,
                                "error": error.to_string(),
                            }),
                        );
                        tokio::time::sleep(Duration::from_secs(delay)).await;
                    }
                    Err(error) => {
                        lock_failures = 0;
                        tracing::error!(
                            event = "rust_worker_poll_failed",
                            error = %error,
                            "worker claim poll failed"
                        );
                        tokio::time::sleep(Duration::from_secs(settings.poll_seconds.max(1))).await;
                    }
                }
            }
            _ = shutdown_signal() => {
                let _ = heartbeat(&api, &settings, WorkerStatus::Offline, None).await;
                return Ok(());
            }
        }
    }
}

/// True when an error ultimately stems from SQLite reporting the jobs database as locked.
/// The claim travels worker→API→store, so a lock surfaces as an `Api { detail }` whose
/// message embeds the SQLite text; match on the rendered string rather than a typed variant.
fn is_database_locked(error: &WorkerError) -> bool {
    error
        .to_string()
        .to_ascii_lowercase()
        .contains("database is locked")
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
                tracing::warn!(
                    event = "rust_worker_register_failed",
                    attempt,
                    retryInSeconds = delay,
                    error = %error,
                    "worker registration failed; will retry"
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
    idle_heartbeat: &mut IdleHeartbeat,
) -> WorkerResult<()> {
    if idle_heartbeat.should_send() {
        heartbeat(api, settings, WorkerStatus::Idle, None).await?;
        idle_heartbeat.mark_sent();
    }
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
    idle_heartbeat.mark_due();
    Ok(())
}

struct IdleHeartbeat {
    interval: Duration,
    next_due: Instant,
}

impl IdleHeartbeat {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            next_due: Instant::now(),
        }
    }

    fn should_send(&self) -> bool {
        Instant::now() >= self.next_due
    }

    fn mark_sent(&mut self) {
        self.next_due = Instant::now() + self.interval;
    }

    fn mark_due(&mut self) {
        self.next_due = Instant::now();
    }
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

/// Post a worker heartbeat. A transport-level failure (`WorkerError::Http`: the API
/// is briefly unreachable — a restart, a transient network blip) is logged and
/// swallowed rather than propagated: a running job must not be torn down for
/// telemetry we can simply resend. The next heartbeat (≤15s) refreshes the worker's
/// `last_seen` well inside the API's stale-sweep window (default 90s), so a brief
/// outage no longer false-positives a live job to `interrupted`; a sustained outage
/// (> the timeout) still lets the sweep fire — the API stays the authority on
/// declaring a worker gone. A non-transport error (the API answered and rejected
/// the heartbeat, e.g. the worker is no longer registered) is a real signal and is
/// still propagated. (sc-6320)
async fn heartbeat(
    api: &ApiClient,
    settings: &Settings,
    status: WorkerStatus,
    current_job_id: Option<&str>,
) -> WorkerResult<()> {
    // Capture the label before `status` is moved into the request, for the log line.
    let status_label = status.as_str().to_owned();
    let outcome: WorkerResult<WorkerSnapshot> = api
        .post_json(
            &format!("/api/v1/workers/{}/heartbeat", settings.worker_id),
            &WorkerHeartbeatRequest {
                status,
                current_job_id: current_job_id.map(str::to_owned),
                loaded_models: Vec::new(),
                utilization: gpu_utilization(&settings.gpu_id).await,
                extra: BTreeMap::new(),
            },
        )
        .await;
    match outcome {
        Ok(_) => Ok(()),
        Err(WorkerError::Http(error)) => {
            emit_event_value(
                Level::ERROR,
                json!({
                    "event": "worker_heartbeat_transport_failed",
                    "workerId": settings.worker_id,
                    "jobId": current_job_id,
                    "status": status_label,
                    "error": error.to_string(),
                }),
            );
            Ok(())
        }
        Err(other) => Err(other),
    }
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
        // Native MLX image generation, served in-process by the linked mlx-gen
        // engine on the macOS Apple-Silicon GPU worker (epic 3018). Off macOS the
        // capability is never advertised, so this arm is unreachable there.
        JobType::ImageGenerate => run_image_generate_job(api, settings, &job)
            .await
            .map_err(|error| ("Image generation failed.", error)),
        // Plain Image Edit (sc-3513): the distinct `image_edit` job type (`mode=edit_image`
        // + `sourceAssetId`, epic 2427) shares the generate handler — it dispatches on
        // payload model+mode (qwen/flux2/sdxl edit streams), not job type. The API only
        // routes MLX-eligible edit models here (jobs_store::image_job_is_mlx_eligible); off
        // macOS the `image_edit` capability is never advertised, so this arm is unreachable.
        JobType::ImageEdit => run_image_generate_job(api, settings, &job)
            .await
            .map_err(|error| ("Image edit failed.", error)),
        // Native MLX tile-ControlNet detail refine (epic 3041, sc-3060), served in-process
        // by the engine on the macOS Apple-Silicon GPU worker. Off macOS the capability is
        // never advertised, so this arm is unreachable there (image_detail runs on torch).
        JobType::ImageDetail => run_image_detail_job(api, settings, &job)
            .await
            .map_err(|error| ("Image detail enhancement failed.", error)),
        // SenseNova-U1 visual question answering + Document Studio interleave (epic 3180,
        // sc-3905). These bypass the `Generator` registry and call the concrete `T2iModel`
        // directly (text / text+images output the `GenerationOutput` contract can't express).
        // The API routes them here only on Mac (`understanding_job_is_mlx_eligible`); off macOS
        // the `image_vqa`/`image_interleave` capabilities are never advertised, so these arms
        // are unreachable there (the Python torch worker serves them on Windows/Linux).
        JobType::ImageVqa => run_vqa_job(api, settings, &job)
            .await
            .map_err(|error| ("Visual question answering failed.", error)),
        JobType::ImageInterleave => run_interleave_job(api, settings, &job)
            .await
            .map_err(|error| ("Interleaved generation failed.", error)),
        // Native MLX video generation, served in-process by the linked mlx-gen engine
        // on the macOS Apple-Silicon GPU worker (epic 3018). sc-3033 ships the runtime
        // + procedural stub; the real Wan (sc-3034) / LTX+audio (sc-3035) models link
        // their provider crates. Off macOS the capability is never advertised, so this
        // arm is unreachable there.
        // The clip-conditioning advanced video modes (epic 3040, sc-3522) share the video
        // generation handler — `run_video_generate_job` dispatches `extend_clip` /
        // `video_bridge` by the request `mode` into the LTX IC-LoRA `VideoClip` path. The API
        // only routes the LTX-eligible jobs here (`video_job_is_mlx_eligible`); off macOS the
        // VideoExtend/VideoBridge capabilities are never advertised, so these arms are
        // unreachable there (the procedural stub would otherwise ignore the conditioning).
        JobType::VideoGenerate | JobType::VideoExtend | JobType::VideoBridge => {
            run_video_generate_job(api, settings, &job)
                .await
                .map_err(|error| ("Video generation failed.", error))
        }
        // replace_person → native Wan-VACE (epic 3040, sc-3521): the `PersonReplace` job
        // type (and `video_generate` mode=`replace_person`) shares the video handler, which
        // dispatches on `mode == "replace_person"` to the engine `wan_vace` provider — the
        // native equivalent of the torch `WanVACEPipeline` path. The API routes only
        // MLX-eligible replace_person jobs here (`jobs_store::video_job_is_mlx_eligible`);
        // off macOS the `person_replace` capability is never advertised, so this arm only
        // produces a real video on the macOS MLX worker (and the Python torch path serves
        // Windows/Linux + non-VACE replacement).
        JobType::PersonReplace => run_video_generate_job(api, settings, &job)
            .await
            .map_err(|error| ("Person replacement failed.", error)),
        // Native MLX LoRA/LoKr training (epic 3039, sc-3043/3049), served in-process
        // by the linked mlx-gen engine on the macOS Apple-Silicon GPU worker. The API
        // routes only MLX-native families here (jobs_store::training_job_is_mlx_eligible);
        // kolors/lens + LoKr-on-Wan stay on the Python torch worker, which is also the
        // Windows/Linux path. Off macOS the execute capability is never advertised.
        JobType::LoraTrain => run_lora_train_job(api, settings, &job)
            .await
            .map_err(|error| ("LoRA training failed.", error)),
        // Native MLX JoyCaption dataset captioning (epic 3550, sc-3556). The API
        // routes only `captioner=joy_caption` jobs here; Windows/Linux and
        // explicit non-MLX GPU choices keep the Python torch captioner fallback.
        JobType::TrainingCaption => run_training_caption_job(api, settings, &job)
            .await
            .map_err(|error| ("Training captioning failed.", error)),
        // Native candle prompt refinement (epic 5095, sc-5525): routes `prompt_refine` to the candle
        // `TextLlm` provider (Llama-3.2-3B) via `gen_core::load_textllm`. The candle worker advertises
        // `prompt_refine` only when `backend_candle_enabled` (engines::registry_capabilities from the
        // registered TextLlm); off the Windows candle build the capability is never advertised, so this
        // arm is unreachable there and the Python torch refiner serves the job (sc-5525 keeps it as the
        // Mac + default-installer fallback).
        JobType::PromptRefine => run_prompt_refine_job(api, settings, &job)
            .await
            .map_err(|error| ("Prompt refinement failed.", error)),
        JobType::ModelDownload => run_model_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Model download failed.", error)),
        JobType::LoraImport => run_lora_import_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA import failed.", error)),
        JobType::LoraDownload => run_lora_download_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("LoRA download failed.", error)),
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
        JobType::PersonDetect => run_person_detect_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Person detection failed.", error)),
        // DWPose whole-body pose detection (epic 3482, sc-3487 Mac / sc-5496 off-Mac):
        // RTMW via onnxruntime, replacing the Python rtmlib path — CoreML EP on the
        // macOS MLX worker, CUDA EP on the off-Mac candle GPU worker. Available on Mac
        // AND the candle lane; on a candle-disabled box `PoseDetect` is never advertised
        // by the Rust worker (the Python worker handles it), so this falls to the `_`
        // arm there.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::PoseDetect => run_pose_detect_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Pose detection failed.", error)),
        // SCRFD 5-point landmark extraction (epic 4422, sc-4433): native-MLX SCRFD on Mac + the candle
        // SCRFD/ArcFace stack on the Windows/Linux candle lane (sc-5497, epic 5482), served in-process
        // for the Key Point Library. Available on Mac AND the candle lane; on a candle-disabled box
        // `KpsExtract` is never advertised by the Rust worker (the Python InsightFace path handles it),
        // so this falls to the `_` arm there.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::KpsExtract => run_kps_extract_job(api, settings, &job)
            .await
            .map_err(|error| ("Keypoint extraction failed.", error)),
        // Image upscaling, served in-process by `upscale_jobs::run_image_upscale_job`: Real-ESRGAN
        // RRDBNet x2/x4 via onnxruntime/CoreML (epic 3482, sc-3489, Mac) + SeedVR2 one-step diffusion
        // (native MLX on Mac sc-4815 / candle CUDA on Windows sc-5928). Available on Mac AND the
        // Windows/CUDA candle lane; on a plain Windows/Linux box `ImageUpscale` is never advertised by
        // the Rust worker, so it falls to the `_` arm (Python Real-ESRGAN/AuraSR). The routing oracle
        // refuses `engine=seedvr2` on torch and `engine=real-esrgan`/`aura-sr` on the candle worker.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::ImageUpscale => run_image_upscale_job(api, settings, http_client, &job)
            .await
            .map_err(|error| ("Image upscale failed.", error)),
        // Smart-select segmentation (epic 6087, sc-6105): native-MLX SAM3 box-prompt segmentation,
        // served in-process by `segment_jobs::run_image_segment_job` — a box prompt → a binary
        // inpaint mask asset for the Image Editor. macOS-only (the capability is advertised only by
        // `mlx_gpu`), so off-Mac this arm is absent and a segment job is never claimed there.
        #[cfg(target_os = "macos")]
        JobType::ImageSegment => {
            segment_jobs::run_image_segment_job(api, settings, http_client, &job)
                .await
                .map_err(|error| ("Smart-select segmentation failed.", error))
        }
        // SeedVR2 video upscaling (epic 4811): one-step super-resolution — native MLX on Mac (sc-4816)
        // / candle CUDA on Windows (sc-5928). SceneWorks' first video upscaler: decodes the source
        // clip, runs the temporal-chunked 5D upscale, re-encodes, and passes the source audio through.
        // Available on Mac + the Windows/CUDA candle lane; elsewhere `VideoUpscale` is never advertised
        // (no torch path), so it falls to the `_` arm and the routing oracle reports it unsupported.
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        JobType::VideoUpscale => run_video_upscale_job(api, settings, &job)
            .await
            .map_err(|error| ("Video upscale failed.", error)),
        JobType::PersonTrack => run_person_track_job(api, settings, http_client, &job)
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
        let _ = cleanup_uploaded_import_source(settings, &job.payload).await;
    }
    if let Err((message, error)) = result {
        match error {
            WorkerError::Canceled(_) => {}
            error => {
                let _ = fail_job(api, &job.id, message, Some(error.to_string())).await;
                tracing::error!(
                    event = "utility_job_failed",
                    jobId = %job.id,
                    error = %error,
                    "{message}"
                );
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
        mark_job_canceled(api, job_id, message).await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    Ok(())
}

async fn mark_job_canceled(api: &ApiClient, job_id: &str, message: &str) -> WorkerResult<()> {
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
    Ok(())
}

/// Check-only cancel poll (sc-5515): returns `true` when the user requested
/// cancellation, WITHOUT posting any status. Unlike [`check_cancel`] this never
/// writes the terminal `Canceled`. In-loop generation/training pollers that sit in
/// front of a long, un-interruptible compute use this so the job stays non-terminal
/// ("Cancelling…") until the in-flight work actually stops; they post the terminal
/// `Canceled` themselves only once it does (sc-5515 image, sc-5516 video/training/detail).
/// Posting terminal at acknowledgement time frees the worker row
/// (`jobs_store::update_job_progress`) while the worker process is still busy, so
/// the next queued job is told a worker is free that isn't — deferring the
/// terminal write to actual-stop keeps the two in sync. Transient GET failures are
/// tolerated (read as "not canceled", retried on the next poll) so an API hiccup
/// never aborts a multi-minute run by being misread as a user cancel (sc-4174).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
async fn cancel_requested_peek(api: &ApiClient, job_id: &str) -> bool {
    let outcome: WorkerResult<JobSnapshot> = api.get_json(&format!("/api/v1/jobs/{job_id}")).await;
    match outcome {
        Ok(job) => job.cancel_requested,
        Err(error) => {
            tracing::warn!(
                event = "cancel_poll_failed",
                jobId = %job_id,
                error = %error,
                "cancel poll failed; retrying on the next poll"
            );
            false
        }
    }
}

async fn update_job(
    api: &ApiClient,
    job_id: &str,
    mut payload: ProgressRequest,
) -> WorkerResult<JobSnapshot> {
    // Stamp the reporting worker so the server can reject the write if this
    // worker no longer owns the job (swept stale / canceled / reclaimed). The
    // resulting 409 propagates as WorkerError::Api and aborts the local job
    // handling — i.e. the worker abandons the job (sc-4172).
    payload.worker_id = Some(api.worker_id.clone());
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

/// Write one staged LoRA upload into `target_dir` under an explicit
/// `target_filename`, renaming it from its uploaded name. Used for paired Wan A14B
/// MoE imports (sc-1991): the two staged uploads must land as
/// `<stem>.high_noise.safetensors` / `<stem>.low_noise.safetensors` so the Python
/// worker's filename-convention split detects them as one two-expert pair.
async fn import_lora_source_file_as(
    source: &Path,
    target_dir: &Path,
    target_filename: &str,
    prefer_move: bool,
) -> WorkerResult<()> {
    let source = source.canonicalize()?;
    tokio::fs::create_dir_all(target_dir).await?;
    let target = target_dir.join(target_filename);
    if prefer_move {
        match tokio::fs::rename(&source, &target).await {
            Ok(()) => return Ok(()),
            Err(error) if is_cross_device_rename_error(&error) => {}
            Err(error) => return Err(error.into()),
        }
    }
    tokio::fs::copy(source, target).await?;
    Ok(())
}

/// The `<stem>.high_noise.safetensors` / `<stem>.low_noise.safetensors` filenames
/// for a Wan A14B MoE LoRA pair stored under one record. The high-noise file sorts
/// first alphabetically, so it resolves as the primary (transformer) and the
/// low-noise file as the `transformer_2` sibling.
pub(crate) fn wan_moe_pair_filenames(stem: &str) -> (String, String) {
    (
        format!("{stem}.high_noise.safetensors"),
        format!("{stem}.low_noise.safetensors"),
    )
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
                // A missing directory is the normal start-of-a-fresh-download state (the HF
                // `blobs/` dir does not exist until the first file lands), so it means "0 bytes
                // so far", not a failure — don't log it at error level. Only surface genuine I/O
                // problems (permissions, etc.).
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        event = "rust_worker_directory_size_failed",
                        path = %path.display(),
                        error = %error,
                        "failed to read a directory while sizing a download"
                    );
                }
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
        // The Rust utility worker doesn't run GPU work, so it never reports
        // per-job peak GPU stats. The Python GPU worker (scene_worker) sets
        // these (sc-2086). Same for `backend` — utility jobs run on the CPU
        // worker which never advertises a GPU runtime.
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some("cpu".to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
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

async fn cleanup_uploaded_import_source(
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<()> {
    if !payload_bool(payload, "uploadedSourcePath") {
        return Ok(());
    }
    let Some(source_path) = optional_payload_string(payload, "sourcePath") else {
        return Ok(());
    };
    let source_path = normalize_absolute_path(Path::new(source_path))?;
    let allowed_roots = [
        normalize_absolute_path(&settings.data_dir.join("cache").join("lora-uploads"))?,
        normalize_absolute_path(&settings.data_dir.join("cache").join("model-uploads"))?,
    ];
    let source_path = ensure_path_under(source_path, &allowed_roots, "Uploaded sourcePath")?;
    let _ = tokio::fs::remove_file(&source_path).await;
    if let Some(parent) = source_path.parent() {
        if allowed_roots
            .iter()
            .any(|root| parent.starts_with(root) && parent != root)
        {
            let _ = tokio::fs::remove_dir(parent).await;
        }
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

fn normalized_data_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    normalize_absolute_path(&settings.data_dir)
}

fn ensure_path_under(path: PathBuf, roots: &[PathBuf], label: &str) -> WorkerResult<PathBuf> {
    if roots.iter().any(|root| path.starts_with(root)) {
        return Ok(path);
    }
    let allowed = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkerError::InvalidPayload(format!(
        "{label} must be inside an app-managed directory ({allowed})."
    )))
}

fn normalize_app_managed_path(
    settings: &Settings,
    raw_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let data_dir = normalized_data_dir(settings)?;
    let path = normalize_absolute_path(Path::new(raw_path))?;
    ensure_path_under(path, &[data_dir], label)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn normalize_app_managed_cache_path(
    settings: &Settings,
    raw_path: &str,
    cache_dir: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let root = settings.data_dir.join("cache").join(cache_dir);
    let normalized_root = normalize_absolute_path(&root)?;
    let canonical_root = normalize_existing_or_absolute(&root)?;
    let normalized = normalize_absolute_path(Path::new(raw_path))?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(resolved, &[normalized_root, canonical_root], label)
}

/// A model's weights are a read-only source the rust-api resolves (e.g.
/// `resolve_base_model_path`) from either the app data dir *or* the shared
/// Hugging Face hub cache — the default `HF_HOME` the desktop injects points the
/// cache at `~/.cache/huggingface`, outside `data_dir`. Unlike output dirs and
/// dataset roots (write targets, confined to `data_dir`), model weights may
/// legitimately live in that cache, so they are allowed under either root. Used
/// for the training base model and every other read-only model dir (captioner,
/// image/InstantID). Without this, an HF-cache-resident model (e.g. z_image_turbo)
/// fails the data-dir-only check even though the install/resolve gates accepted it.
fn normalize_app_managed_model_path(
    settings: &Settings,
    raw_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let raw_path = raw_path.trim();
    if raw_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let data_dir = normalized_data_dir(settings)?;
    let hf_cache = normalize_absolute_path(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let path = normalize_absolute_path(Path::new(raw_path))?;
    ensure_path_under(path, &[data_dir, hf_cache], label)
}

/// Confine a LoRA adapter path taken from a job payload to an app-managed root
/// (sc-5723 / WKA-002). The path arrives untrusted (`installedPath`/`sourcePath`/
/// `path`/`source.path` on a LoRA spec) and is loaded as adapter weights, so —
/// like every other on-disk model input — it must resolve under the app data dir
/// or the shared Hugging Face hub cache (installed LoRAs live in `<data>/loras` or
/// a project tree under `<data>`; HF-cached adapters live in the hub cache).
/// Without this a crafted payload could point a LoRA at any `.safetensors` on the
/// host, giving the worker an arbitrary-file read primitive across the API boundary.
/// Mirrors `normalize_app_managed_model_path` (model weights share the same roots).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn normalize_app_managed_lora_path(
    settings: &Settings,
    path: &Path,
) -> WorkerResult<PathBuf> {
    let data_dir = normalized_data_dir(settings)?;
    let canonical_data_dir = normalize_existing_or_absolute(&settings.data_dir)?;
    let hf_cache = normalize_absolute_path(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let canonical_hf_cache =
        normalize_existing_or_absolute(&huggingface_hub_cache_dir(&settings.data_dir))?;
    let normalized = normalize_absolute_path(path)?;
    let resolved = normalize_existing_or_absolute(&normalized)?;
    ensure_path_under(
        resolved,
        &[data_dir, canonical_data_dir, hf_cache, canonical_hf_cache],
        "LoRA path",
    )
}

fn normalize_existing_or_absolute(path: &Path) -> WorkerResult<PathBuf> {
    match std::fs::canonicalize(path) {
        Ok(canonical) => normalize_absolute_path(&canonical),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => normalize_absolute_path(path),
        Err(error) => Err(error.into()),
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn looks_like_huggingface_repo(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() || value.contains('\\') || Path::new(value).is_absolute() {
        return false;
    }
    let mut parts = value.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(repo) = parts.next() else {
        return false;
    };
    !owner.is_empty()
        && !repo.is_empty()
        && parts.next().is_none()
        && ![owner, repo]
            .iter()
            .any(|part| *part == "." || *part == "..")
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn resolve_app_managed_model_dir(
    settings: &Settings,
    model_name_or_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let model_name_or_path = model_name_or_path.trim();
    if model_name_or_path.is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, model_name_or_path) {
        return Ok(snapshot);
    }
    if looks_like_huggingface_repo(model_name_or_path) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} snapshot is not cached for {model_name_or_path}."
        )));
    }
    let path = normalize_app_managed_model_path(settings, model_name_or_path, label)?;
    if path.is_dir() {
        return Ok(path);
    }
    if path.exists() {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} must be a snapshot directory, not a file: {}",
            path.display()
        )));
    }
    Err(WorkerError::InvalidPayload(format!(
        "{label} is not installed at {}.",
        path.display()
    )))
}

fn resolve_training_output_dir(
    settings: &Settings,
    output_dir: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let path = normalize_app_managed_path(settings, output_dir, label)?;
    let data_dir = normalized_data_dir(settings)?;
    // Global-scope outputs land in `<data>/loras` (or `<data>/models` for full
    // fine-tunes); project-scope outputs — the default — land in the owning
    // project's tree, `<data>/projects/<slug>.sceneworks/loras/<lora_id>`, which
    // `resolve_training_output_location` computes API-side from trusted inputs.
    // All three stay inside the app data dir, so allow the projects tree too
    // rather than rejecting every project-scoped run.
    let allowed_roots = [
        data_dir.join("loras"),
        data_dir.join("models"),
        data_dir.join("projects"),
    ];
    ensure_path_under(path, &allowed_roots, label)
}

fn resolve_dataset_item_path(
    settings: &Settings,
    dataset_root: &str,
    image_path: &str,
    label: &str,
) -> WorkerResult<PathBuf> {
    let root = normalize_app_managed_path(settings, dataset_root, "Dataset root")?;
    let raw_image = Path::new(image_path.trim());
    if image_path.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(format!("{label} is required.")));
    }
    let path = if raw_image.is_absolute() {
        normalize_absolute_path(raw_image)?
    } else {
        normalize_absolute_path(&root.join(raw_image))?
    };
    ensure_path_under(path, &[root], label)
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

/// Upsert `entry` (keyed by its `id`) into the `collection_key` array of a JSONC
/// manifest at `path`, creating the manifest when absent. An existing entry with
/// the same id is merged (incoming fields win) but keeps its original `createdAt`.
/// Shared by the LoRA (`"loras"`) and model (`"models"`) manifests, which differed
/// only by this array key (sc-4279 / F-MLXW-15).
async fn upsert_manifest_entry(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    let mut manifest = match tokio::fs::read_to_string(path).await {
        Ok(payload) => serde_json::from_str(&strip_jsonc_comments(&payload))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut object = serde_json::Map::new();
            object.insert("schemaVersion".to_owned(), json!(1));
            object.insert(collection_key.to_owned(), Value::Array(Vec::new()));
            Value::Object(object)
        }
        Err(error) => return Err(error.into()),
    };
    let entry_id = entry.get("id").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{collection_key} manifest entry requires id"))
    })?;
    let collection = manifest
        .as_object_mut()
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("{collection_key} manifest must be an object"))
        })?
        .entry(collection_key.to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let collection = collection.as_array_mut().ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{collection_key} manifest array must be an array"))
    })?;
    let mut found = false;
    for item in collection.iter_mut() {
        if item.get("id").and_then(Value::as_str) != Some(entry_id) {
            continue;
        }
        found = true;
        let created_at = item.get("createdAt").cloned();
        let Some(object) = item.as_object_mut() else {
            return Err(WorkerError::InvalidPayload(format!(
                "{collection_key} manifest entry must be an object"
            )));
        };
        for (key, value) in entry.clone() {
            object.insert(key, value);
        }
        if let Some(created_at) = created_at {
            object.insert("createdAt".to_owned(), created_at);
        }
    }
    if !found {
        collection.push(Value::Object(entry));
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

async fn with_hf_auth(
    settings: &Settings,
    request: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    // Resolves the HF token lazily: the env `HF_TOKEN` (server/Docker/Windows) or,
    // on the macOS desktop, a one-time pull of the recorded `huggingface.co`
    // credential from the desktop socket (sc-5891). `None` ⇒ unauthenticated.
    match credentials_ipc::resolve_hf_token(settings).await {
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

/// Parse a boolean env toggle: `1`/`true`/`yes`/`on` → true, `0`/`false`/`no`/`off` → false,
/// empty or unrecognized → `default` (and an unset var → `default`). Used by the per-backend
/// capability toggles (sc-3723).
fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            "" => default,
            _ => default,
        },
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests;
