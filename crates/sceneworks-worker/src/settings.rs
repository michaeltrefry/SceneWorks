//! Worker [`Settings`] and the environment-variable accessors that populate it.
use super::*;

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
    /// Whether the candle (Windows/CUDA + Linux/NVIDIA) tensor backend is enabled for capability
    /// derivation (sc-3723). **Defaults to on whenever the worker is built `--features
    /// backend-candle`** (sc-5502, epic 5483 Phase-7 cutover): the candle providers are at parity
    /// off-Mac, so a candle build advertises them by default — no `SCENEWORKS_BACKEND_CANDLE_ENABLED`
    /// needed. A non-candle build (Mac mlx, the desktop installer with no CUDA, any CPU/Linux build
    /// without the feature) defaults `false` and links no candle crate, so this is inert there. The
    /// env var still overrides either way (set `0` to force a candle build back onto the Python
    /// torch fallback during a staged rollout).
    pub backend_candle_enabled: bool,
    /// Soft ceiling, in bytes, on the shared/unified GPU memory the MLX runtime may use, from
    /// `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES` (epic 7819, sc-7820). `0` (the default / unset) leaves
    /// MLX at its own budget — byte-identical to prior behavior. When non-zero it is applied
    /// **process-globally** at worker startup via `generator_cache::apply_gpu_memory_limit`, so a
    /// single value covers generations, upscales, AND LoRA training in this process. macOS/MLX only;
    /// inert on candle/CPU builds (the cross-platform path is tracked separately as sc-7826).
    pub gpu_memory_limit_bytes: u64,
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
            // Phase-7 cutover (sc-5502): a candle build defaults the backend ON (the providers are at
            // off-Mac parity); a non-candle build defaults it off. The env var overrides either way.
            backend_candle_enabled: env_bool(
                "SCENEWORKS_BACKEND_CANDLE_ENABLED",
                cfg!(feature = "backend-candle"),
            ),
            gpu_memory_limit_bytes: env_u64_any(&["SCENEWORKS_GPU_MEMORY_LIMIT_BYTES"], 0),
        }
    }
}

pub(crate) fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

pub(crate) fn env_path_or(key: &str, default: &std::path::Path) -> PathBuf {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| default.to_path_buf())
}

pub(crate) fn env_u64_any(keys: &[&str], default: u64) -> u64 {
    keys.iter()
        .find_map(|key| std::env::var(key).ok().and_then(|value| value.parse().ok()))
        .unwrap_or(default)
}

/// Parse a boolean env toggle: `1`/`true`/`yes`/`on` → true, `0`/`false`/`no`/`off` → false,
/// empty or unrecognized → `default` (and an unset var → `default`). Used by the per-backend
/// capability toggles (sc-3723).
pub(crate) fn env_bool(key: &str, default: bool) -> bool {
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
