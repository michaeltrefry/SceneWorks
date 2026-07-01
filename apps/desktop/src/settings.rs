//! Desktop settings surface (sc-1350): data directory, Hugging Face token (OS
//! keychain), detected GPU info, and a worker restart. Commands are invoked from
//! the React settings screen when running inside the Tauri shell.

use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use tauri::AppHandle;
use tauri_plugin_dialog::DialogExt;

use crate::setup::{app_support_dir, default_data_dir, shared_huggingface_home};

const KEYRING_SERVICE: &str = "SceneWorks";
/// Pre-migration account that held the single Hugging Face token. Retained only so
/// `delete_credential` can also clear it when the user removes Hugging Face — it is
/// never *read* on a startup/spawn path, because probing it unconditionally is an
/// unguarded keychain touch (and macOS prompt) on installs that have no token
/// recorded (sc-5891).
const HF_TOKEN_ACCOUNT: &str = "huggingface_token";
/// Host of the migrated Hugging Face credential.
const HF_HOST: &str = "huggingface.co";
/// Keychain account for the LAN remote-access password (epic 4484, story 1). The
/// password the user sets to gate LAN access is stored as a secret here — never in
/// `settings.json`, which holds only the `remote_password_set` boolean. Resolves to
/// macOS Keychain / Windows Credential Manager via the same `keyring::Entry` path as
/// the download credentials above. Read only when the user opts into LAN mode, so a
/// default (LAN-off) install never touches the keychain for it.
const REMOTE_PASSWORD_ACCOUNT: &str = "remote-access-password";
/// Suggested default LAN port the Settings UI pre-fills the first time remote access
/// is enabled (story 4). Kept here so the launcher and UI agree on the suggestion.
pub const DEFAULT_REMOTE_PORT: u16 = 8787;

/// How a stored credential is attached to a download request for its host.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CredentialScheme {
    /// `Authorization: Bearer <token>` — Hugging Face and most token APIs.
    #[default]
    Bearer,
    /// `?token=<token>` query parameter — e.g. Civit.ai download URLs.
    Query,
}

/// Non-secret metadata for a stored credential, persisted in `settings.json`. The
/// secret token itself lives in the OS keychain under account `cred:<host>`; this
/// list lets the UI enumerate hosts/labels without ever reading secrets back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialMeta {
    /// Host the credential authenticates, lower-cased (e.g. `huggingface.co`).
    pub host: String,
    /// Human label shown in the UI (e.g. "Hugging Face").
    pub label: String,
    #[serde(default)]
    pub scheme: CredentialScheme,
}

/// What `list_credentials` returns: metadata plus whether the secret is actually
/// present in the keychain. Never includes the token itself.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialStatus {
    host: String,
    label: String,
    scheme: CredentialScheme,
    present: bool,
}

fn cred_account(host: &str) -> String {
    format!("cred:{host}")
}

/// Normalize a user-entered host or URL to a bare lower-cased host (strip scheme
/// and any path) so `https://Civitai.com/foo` and `civitai.com` collapse to one
/// keychain account.
fn normalize_host(input: &str) -> String {
    input
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// The token stored for `host`, from the OS keychain. `None` when unset/unreadable.
fn read_credential_secret(host: &str) -> Option<String> {
    keyring::Entry::new(KEYRING_SERVICE, &cred_account(host))
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

/// Insert a credential's metadata, or update label/scheme in place if its host is
/// already recorded.
fn upsert_credential_meta(settings: &mut AppSettings, meta: CredentialMeta) {
    if let Some(existing) = settings
        .credentials
        .iter_mut()
        .find(|entry| entry.host == meta.host)
    {
        existing.label = meta.label;
        existing.scheme = meta.scheme;
    } else {
        settings.credentials.push(meta);
    }
}

/// Drop a host's metadata. Returns whether anything was removed.
fn remove_credential_meta(settings: &mut AppSettings, host: &str) -> bool {
    let before = settings.credentials.len();
    settings.credentials.retain(|entry| entry.host != host);
    settings.credentials.len() != before
}

// Gates the eager-push HF read below; on macOS that path is replaced by the lazy
// credential socket (sc-5891), so this is only reached off macOS (and in tests).
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn hf_credential_recorded(settings: &AppSettings) -> bool {
    settings
        .credentials
        .iter()
        .any(|entry| entry.host == HF_HOST)
}

// NOTE (sc-5891): there used to be a startup `migrate_legacy_hf_token` here that
// moved the pre-host-keyed single token (`huggingface_token` account) into the
// `cred:huggingface.co` store. It had to probe the legacy keychain entry to do so,
// and the only signal that such a token might exist *is* the keychain itself — the
// pre-migration `settings.json` recorded nothing about it. That unconditional probe
// is exactly the unguarded keychain touch (and macOS password prompt) this story
// removes, so the startup auto-migration is gone: a fresh/no-token install must
// never touch the keychain. Any install that launched a build after the host-keyed
// store landed already auto-migrated (the migration was idempotent and ran at
// startup); the rare install that skipped every such build can re-add its Hugging
// Face token once in Settings.

#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    /// Override for the workspace data directory (projects, generated assets,
    /// imported/non-HF models, jobs.db); `None` uses the platform default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
    /// Hugging Face cache home (`HF_HOME`) for HF-downloaded model weights;
    /// `None` uses the shared per-user cache (`~/.cache/huggingface`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_home: Option<String>,
    /// Set once the first-run splash storage step (sc-1473 Step 1) has run, so
    /// later launches skip straight to provisioning instead of re-prompting.
    #[serde(default)]
    pub storage_configured: bool,
    /// Set once the in-app setup wizard (sc-1473 Steps 2-3) has completed, so the
    /// studio shows directly. Cleared by `reset_setup` to re-run the wizard.
    #[serde(default)]
    pub setup_completed: bool,
    /// Non-secret metadata for stored service credentials (HF, Civit.ai, …). The
    /// secret token for each host lives in the OS keychain; this only records
    /// host/label/scheme so the UI can enumerate them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<CredentialMeta>,
    /// LAN remote-access master switch (epic 4484). OFF by default: the API sidecar
    /// binds loopback-only on a dynamic port exactly as before. When ON *and* a
    /// password is set, the launcher binds `0.0.0.0:<remote_port>` with the password
    /// as the API access token so other devices on the LAN can reach the host.
    #[serde(default)]
    pub remote_access_enabled: bool,
    /// Fixed port for LAN remote access. `None` until the user picks one; the UI
    /// pre-fills [`DEFAULT_REMOTE_PORT`] (8787) the first time remote access is
    /// enabled. Only consulted when `remote_access_enabled`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_port: Option<u16>,
    /// Whether a LAN password secret is recorded in the OS keychain. Non-secret
    /// metadata (the password itself never lands in `settings.json`); gates the
    /// keychain read so a no-password install never touches it (mirrors the HF
    /// `credentials` gate, sc-5891). LAN cannot be enabled while this is false.
    #[serde(default)]
    pub remote_password_set: bool,
    /// User-set GPU memory cap as a fraction (0.1–1.0) of total unified memory (epic 7819).
    /// `None` = no limit (MLX keeps its own default budget). Stored as a fraction so it's portable
    /// across machines; the MLX worker spawn derives `fraction × hw.memsize` bytes and applies it to
    /// the MLX runtime process-globally, covering generations, upscales, AND LoRA training. macOS/MLX
    /// only today — the candle/Windows path is tracked separately (sc-7826).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_memory_limit_fraction: Option<f32>,
}

fn settings_path() -> PathBuf {
    app_support_dir().join("settings.json")
}

pub fn load_settings() -> AppSettings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
        .unwrap_or_default()
}

fn save_settings(settings: &AppSettings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let body = serde_json::to_string_pretty(settings).map_err(|error| error.to_string())?;
    std::fs::write(&path, body).map_err(|error| error.to_string())
}

/// Hugging Face token for injecting `HF_TOKEN` into the worker, from the host-keyed
/// `huggingface.co` credential. Gated on the non-secret `settings.json` metadata
/// first: if no `huggingface.co` credential is recorded, returns `None` *without
/// constructing a `keyring::Entry` or calling `get_password()`*, so a no-token
/// install never touches the OS keychain (and never triggers a macOS password
/// prompt). The pre-migration `huggingface_token` fallback was removed for the same
/// reason — it probed the keychain unconditionally (sc-5891).
///
/// macOS no longer uses this eager push at all — the MLX worker pulls credentials
/// lazily from the desktop credential socket (`cred_ipc`) — so it's compiled but
/// uncalled there (the Python/candle spawn sites on other platforms still use it).
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn read_hf_token() -> Option<String> {
    if !hf_credential_recorded(&load_settings()) {
        return None;
    }
    read_credential_secret(HF_HOST)
}

/// The non-secret hosts that have a credential recorded, handed to the MLX worker so
/// it knows which hosts it may request from the credential socket (sc-5891). Reads
/// `settings.json` metadata only — no keychain access — so an empty list (nothing
/// recorded) keeps the worker from ever asking, hence no keychain touch.
#[cfg(target_os = "macos")]
pub fn recorded_credential_hosts() -> Vec<String> {
    load_settings()
        .credentials
        .into_iter()
        .map(|meta| meta.host)
        .collect()
}

/// The secret token + scheme for a single recorded host, for the on-demand
/// credential socket to serve when a worker download actually needs it (sc-5891).
/// Gated on the non-secret `settings.json` metadata: if the host isn't recorded this
/// returns `None` without reading the keychain. This is the single *lazy* keychain
/// read that replaces the eager spawn-time reads on macOS.
#[cfg(target_os = "macos")]
pub fn resolve_credential_secret(host: &str) -> Option<(String, CredentialScheme)> {
    let host = host.trim().to_ascii_lowercase();
    let settings = load_settings();
    let meta = settings
        .credentials
        .iter()
        .find(|entry| entry.host == host)?;
    let token = read_credential_secret(&host)?;
    Some((token, meta.scheme))
}

/// All stored credentials serialized as the worker's `SCENEWORKS_CREDENTIALS` JSON
/// map (`{ host: { token, scheme } }`), reading each secret from the keychain.
/// `None` when no credentials are stored. Injected into the worker at spawn.
///
/// macOS uses the lazy credential socket (`cred_ipc`) instead, so this is compiled
/// but uncalled there (the Python/candle spawn sites on other platforms use it).
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn credentials_env_json() -> Option<String> {
    let settings = load_settings();
    let mut map = serde_json::Map::new();
    for meta in &settings.credentials {
        if let Some(token) = read_credential_secret(&meta.host) {
            let scheme = match meta.scheme {
                CredentialScheme::Bearer => "bearer",
                CredentialScheme::Query => "query",
            };
            map.insert(
                meta.host.clone(),
                serde_json::json!({ "token": token, "scheme": scheme }),
            );
        }
    }
    if map.is_empty() {
        None
    } else {
        serde_json::to_string(&serde_json::Value::Object(map)).ok()
    }
}

// ---------------------------------------------------------------------------
// LAN remote-access (epic 4484, story 1)
// ---------------------------------------------------------------------------

/// Whether a LAN password is recorded, from the non-secret `settings.json`
/// metadata. The keychain gate for [`read_remote_password`]: a no-password install
/// must short-circuit here so it never constructs a `keyring::Entry` / prompts —
/// the same pattern as `hf_credential_recorded` (sc-5891).
fn remote_password_recorded(settings: &AppSettings) -> bool {
    settings.remote_password_set
}

/// Apply a password set/clear to the non-secret metadata. Pure (no keychain/disk)
/// so the metadata transitions — including the fail-closed disable on clear — are
/// unit-tested; the keychain write/delete is done by the set/clear wrappers around
/// this. Clearing the password also disables LAN remote access: with no password the
/// launcher must never bind non-loopback (epic 4484 security constraint, story 3).
fn mark_remote_password(settings: &mut AppSettings, present: bool) {
    settings.remote_password_set = present;
    if !present {
        settings.remote_access_enabled = false;
    }
}

/// The LAN remote-access password from the OS keychain, used as the API access token
/// when the launcher binds non-loopback (story 2). Gated on the non-secret
/// `settings.json` metadata first: if no password is recorded, returns `None`
/// *without constructing a `keyring::Entry` or calling `get_password()`*, so a
/// LAN-off install never touches the OS keychain at launch (mirrors `read_hf_token`,
/// sc-5891). Empty/whitespace secrets are treated as absent.
pub fn read_remote_password() -> Option<String> {
    if !remote_password_recorded(&load_settings()) {
        return None;
    }
    keyring::Entry::new(KEYRING_SERVICE, REMOTE_PASSWORD_ACCOUNT)
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .map(|secret| secret.trim().to_owned())
        .filter(|secret| !secret.is_empty())
}

/// Store (or overwrite) the LAN password secret in the OS keychain and record the
/// `remote_password_set` metadata. The password is write-only — it is never read back
/// to the UI (only used as the sidecar access token). Rejects an empty password so a
/// LAN bind can never end up with an empty token (story 3 backstop).
pub fn set_remote_password(password: &str) -> Result<(), String> {
    let password = password.trim();
    if password.is_empty() {
        return Err("A password is required.".to_owned());
    }
    keyring::Entry::new(KEYRING_SERVICE, REMOTE_PASSWORD_ACCOUNT)
        .map_err(|error| error.to_string())?
        .set_password(password)
        .map_err(|error| error.to_string())?;
    let mut settings = load_settings();
    mark_remote_password(&mut settings, true);
    save_settings(&settings)
}

/// Remove the LAN password from the keychain and clear its metadata, which also
/// disables remote access (fail-closed: no password ⇒ no non-loopback bind). Safe to
/// call when no password is stored.
pub fn clear_remote_password() -> Result<(), String> {
    if let Ok(entry) = keyring::Entry::new(KEYRING_SERVICE, REMOTE_PASSWORD_ACCOUNT) {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    let mut settings = load_settings();
    mark_remote_password(&mut settings, false);
    save_settings(&settings)
}

/// Host OS the desktop shell is running on, so the Settings UI can show the
/// platform-correct firewall guidance (story 11): macOS firewall prompt vs Windows
/// Defender alert.
fn host_platform() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "macos"
    }
    #[cfg(target_os = "windows")]
    {
        "windows"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "linux"
    }
}

/// Snapshot of the LAN remote-access state for the Settings UI (story 4). Carries no
/// secret — only whether a password is set, plus the computed URL/candidates.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteAccessStatus {
    /// Whether LAN remote access is enabled (takes effect on next launch).
    enabled: bool,
    /// Configured port, or the [`DEFAULT_REMOTE_PORT`] suggestion when unset, so the
    /// UI can pre-fill the input.
    port: u16,
    /// Whether a password secret is recorded (gates enabling; never the value).
    password_set: bool,
    /// Best-guess LAN IPv4 for the URL, or `None` if none could be determined.
    lan_address: Option<String>,
    /// All private LAN candidates (so the UI could offer a picker if the guess is off).
    lan_candidates: Vec<String>,
    /// `http://<lan_address>:<port>` when an address is known; `None` otherwise.
    url: Option<String>,
    /// The suggested default port, for the UI's initial value / reset.
    default_port: u16,
    /// Host OS (`macos`/`windows`/`linux`) for platform-conditional firewall copy.
    platform: &'static str,
}

/// Build the current remote-access snapshot from persisted settings + a fresh LAN
/// address probe. Never reads the password secret (only its `*_set` metadata).
fn remote_access_status() -> RemoteAccessStatus {
    let settings = load_settings();
    let port = settings.remote_port.unwrap_or(DEFAULT_REMOTE_PORT);
    let lan = crate::net::lan_addresses();
    let url = lan.primary.as_ref().map(|ip| format!("http://{ip}:{port}"));
    RemoteAccessStatus {
        enabled: settings.remote_access_enabled,
        port,
        password_set: settings.remote_password_set,
        lan_address: lan.primary,
        lan_candidates: lan.candidates,
        url,
        default_port: DEFAULT_REMOTE_PORT,
        platform: host_platform(),
    }
}

/// Current LAN remote-access status for the Settings screen (story 4).
#[tauri::command]
pub fn get_remote_access() -> RemoteAccessStatus {
    remote_access_status()
}

/// Enable/disable LAN remote access and set the port (story 4). Fail-closed: enabling
/// is rejected unless a password is already set (mirrors the launcher guard, story 3),
/// and a privileged/low port the app can't bind is rejected up front rather than
/// bricking startup. The change takes effect on the next launch (sidecar relaunch).
#[tauri::command]
pub fn set_remote_access(enabled: bool, port: u16) -> Result<RemoteAccessStatus, String> {
    let mut settings = load_settings();
    if enabled && !settings.remote_password_set {
        return Err("Set a password before enabling remote access.".to_owned());
    }
    if enabled && port < 1024 {
        return Err(
            "Choose a port between 1024 and 65535 — lower ports need administrator privileges."
                .to_owned(),
        );
    }
    settings.remote_access_enabled = enabled;
    // Persist a valid port choice even while disabled, so it's remembered for next time.
    if port >= 1024 {
        settings.remote_port = Some(port);
    }
    save_settings(&settings)?;
    Ok(remote_access_status())
}

/// Set/change the LAN password (story 4). Stored in the OS keychain; never echoed back.
#[tauri::command]
pub fn set_remote_access_password(password: String) -> Result<RemoteAccessStatus, String> {
    set_remote_password(&password)?;
    Ok(remote_access_status())
}

/// Clear the LAN password, which also disables remote access (fail-closed, story 3).
#[tauri::command]
pub fn clear_remote_access_password() -> Result<RemoteAccessStatus, String> {
    clear_remote_password()?;
    Ok(remote_access_status())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    platform: String,
    devices: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unified_memory_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wired_limit_mb: Option<u64>,
}

fn run_capture(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[tauri::command]
pub fn get_app_settings() -> AppSettings {
    load_settings()
}

/// Floor for the user's GPU memory fraction — below this even small models can't load and the
/// worker would just OOM on every job, so the slider clamps here (epic 7819).
const MIN_GPU_MEMORY_FRACTION: f32 = 0.1;

/// Persist the user's GPU memory cap as a fraction of total unified memory, or clear it. `None`
/// (and a non-finite or ≥ 1.0 value — 100% is "use everything" = no constraint) clears the cap;
/// otherwise the value is clamped to `[MIN_GPU_MEMORY_FRACTION, 0.99]`. Persists the fraction AND
/// (macOS, sc-7824) writes the derived byte ceiling to the live-handoff file the running MLX worker
/// re-reads between jobs, so the change applies within a couple of seconds with no worker restart.
/// Returns the updated settings.
#[tauri::command]
pub fn set_gpu_memory_limit(fraction: Option<f32>) -> Result<AppSettings, String> {
    let mut settings = load_settings();
    settings.gpu_memory_limit_fraction = match fraction {
        Some(value) if value.is_finite() && value < 1.0 => {
            Some(value.clamp(MIN_GPU_MEMORY_FRACTION, 0.99))
        }
        _ => None,
    };
    save_settings(&settings)?;
    #[cfg(target_os = "macos")]
    write_gpu_memory_limit_file();
    Ok(settings)
}

/// Write the resolved byte ceiling (or `0` for "no limit") to the live-handoff file the running MLX
/// worker re-reads between jobs (epic 7819, sc-7824), so a slider change applies without a worker
/// restart. Best-effort: a write failure just means the change waits for the next worker restart
/// (which re-reads `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES` from the persisted fraction anyway).
#[cfg(target_os = "macos")]
fn write_gpu_memory_limit_file() {
    let bytes = gpu_memory_limit_bytes().unwrap_or(0);
    let path = sceneworks_core::app_paths::gpu_memory_limit_file(&crate::setup::config_dir());
    let _ = std::fs::write(&path, bytes.to_string());
}

/// Read the MLX worker's latest GPU memory telemetry for the Settings readout (epic 7819, sc-7825).
/// `None` when the worker hasn't published any yet, or on platforms/workers without MLX telemetry
/// (candle/CPU never write the file). Best-effort: a missing or unparseable file yields `None`.
#[tauri::command]
pub fn get_gpu_telemetry() -> Option<sceneworks_core::app_paths::GpuMemoryTelemetry> {
    let path = sceneworks_core::app_paths::gpu_telemetry_file(&crate::setup::config_dir());
    let raw = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// The configured GPU memory ceiling in BYTES for the MLX worker, or `None` for no limit:
/// `fraction × hw.memsize` (total unified memory). Read at worker spawn (`supervise_mlx_worker`)
/// and passed as `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES`, which `run_worker_loop` applies process-
/// globally. macOS only — the fraction is meaningless without a unified-memory total; the
/// candle/Windows path is tracked separately (sc-7826).
#[cfg(target_os = "macos")]
pub fn gpu_memory_limit_bytes() -> Option<u64> {
    let fraction = load_settings().gpu_memory_limit_fraction?;
    if !(fraction.is_finite() && (MIN_GPU_MEMORY_FRACTION..1.0).contains(&fraction)) {
        return None;
    }
    let total =
        run_capture("sysctl", &["-n", "hw.memsize"]).and_then(|value| value.parse::<u64>().ok())?;
    Some((total as f64 * fraction as f64).round() as u64)
}

/// First-run storage state for the splash Step 1 + the in-app wizard gate. The
/// `*Default` fields let the splash pre-fill the pickers with the locations the
/// app would use today so a new user can just continue.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageSetup {
    data_dir: Option<String>,
    data_dir_default: String,
    hf_home: Option<String>,
    hf_home_default: String,
    storage_configured: bool,
    setup_completed: bool,
}

#[tauri::command]
pub fn get_storage_setup() -> StorageSetup {
    let settings = load_settings();
    StorageSetup {
        data_dir: settings.data_dir,
        data_dir_default: default_data_dir().to_string_lossy().into_owned(),
        hf_home: settings.hf_home,
        hf_home_default: shared_huggingface_home().to_string_lossy().into_owned(),
        storage_configured: settings.storage_configured,
        setup_completed: settings.setup_completed,
    }
}

/// Persist the splash Step 1 storage choice and mark storage configured. Empty
/// strings clear the override (fall back to the platform default). This runs
/// before the API/worker are spawned, so the chosen paths take effect with no
/// restart.
#[tauri::command]
pub fn save_storage_setup(data_dir: String, hf_home: String) -> Result<AppSettings, String> {
    let mut settings = load_settings();
    let data_trimmed = data_dir.trim();
    settings.data_dir = if data_trimmed.is_empty() {
        None
    } else {
        Some(data_trimmed.to_owned())
    };
    let hf_trimmed = hf_home.trim();
    settings.hf_home = if hf_trimmed.is_empty() {
        None
    } else {
        Some(hf_trimmed.to_owned())
    };
    settings.storage_configured = true;
    save_settings(&settings)?;
    Ok(settings)
}

/// Mark the in-app setup wizard (Steps 2-3) complete so the studio shows on
/// subsequent loads.
#[tauri::command]
pub fn complete_setup() -> Result<(), String> {
    let mut settings = load_settings();
    settings.setup_completed = true;
    save_settings(&settings)
}

/// Clear the wizard-completed marker so the wizard re-runs (Settings → Re-run
/// setup wizard). Storage configuration is left in place — relocating the data
/// dir is a separate, restart-bound action handled by the data-directory control.
#[tauri::command]
pub fn reset_setup() -> Result<(), String> {
    let mut settings = load_settings();
    settings.setup_completed = false;
    save_settings(&settings)
}

/// Generic folder picker for the splash storage step (workspace + HF cache
/// pickers). Returns the chosen absolute path, or `None` if the dialog was
/// dismissed.
#[tauri::command]
pub async fn choose_folder(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|file| file.into_path().ok())
        .map(|path| path.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn set_data_dir(path: String) -> Result<AppSettings, String> {
    let mut settings = load_settings();
    let trimmed = path.trim();
    settings.data_dir = if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    };
    save_settings(&settings)?;
    Ok(settings)
}

#[tauri::command]
pub async fn choose_data_dir(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|file| file.into_path().ok())
        .map(|path| path.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn reveal_in_os(path: String) -> Result<(), String> {
    let target = validate_reveal_target(&path)?;
    let result = if cfg!(target_os = "macos") {
        Command::new("open").arg("-R").arg(&target).status()
    } else if cfg!(target_os = "windows") {
        Command::new("explorer")
            .arg(format!("/select,{}", target.display()))
            .status()
    } else {
        let dir = target.parent().unwrap_or(&target);
        Command::new("xdg-open").arg(dir).status()
    };
    result.map(|_| ()).map_err(|error| error.to_string())
}

/// The app-managed roots a revealed/saved file must live inside: the SceneWorks data
/// directory (projects, generated assets) and the Hugging Face cache. Reads the user's
/// persisted overrides, falling back to the platform defaults. Canonicalized so the
/// `starts_with` containment check in [`path_within_roots`] compares real paths (and so
/// a symlinked root is resolved once here rather than per-check).
fn app_managed_roots() -> Vec<PathBuf> {
    let settings = load_settings();
    [
        settings
            .data_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(default_data_dir),
        settings
            .hf_home
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(shared_huggingface_home),
    ]
    .iter()
    .filter_map(|root| std::fs::canonicalize(root).ok())
    .collect()
}

/// Pure containment check: is `target` inside any of `roots`? Extracted so the guard's
/// core logic is unit-testable without touching real settings/disk (the canonicalized
/// inputs are supplied by the caller). Both sides are expected to be already
/// canonicalized so `starts_with` compares real, prefix-aligned components.
fn path_within_roots(target: &std::path::Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| target.starts_with(root))
}

/// Canonicalize `path` and confirm it lives inside an app-managed root (data dir or HF
/// cache). Shared by [`reveal_in_os`] and [`save_asset_as`] so both reject paths outside
/// the SceneWorks-managed trees identically — a save/reveal must never touch an arbitrary
/// file the frontend hands us. Returns the canonicalized path on success.
fn validate_reveal_target(path: &str) -> Result<PathBuf, String> {
    let target = path.trim();
    if target.is_empty() {
        return Err("A path is required.".to_owned());
    }
    let target = std::fs::canonicalize(target).map_err(|error| error.to_string())?;
    if path_within_roots(&target, &app_managed_roots()) {
        return Ok(target);
    }
    Err(
        "Can only reveal files inside the SceneWorks data directory or Hugging Face cache."
            .to_owned(),
    )
}

/// The resolved workspace data directory: the user's persisted override, or the platform
/// default. This is the root under which projects (and thus assets) live on disk; the
/// project registry (`<data_dir>/recent-projects.json`) maps a `projectId` to its
/// `.sceneworks` project directory, onto which a project-relative `file.path` joins.
fn resolved_data_dir() -> PathBuf {
    load_settings()
        .data_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir)
}

/// Turn an asset's project-relative `file.path` (as carried by the frontend alongside its
/// `projectId`) into its absolute on-disk path (sc-8726). Assets are addressed in the API
/// as `/api/v1/projects/<id>/files/<relativePath>`; that route resolves through the shared
/// [`ProjectStore`](sceneworks_core::project_store::ProjectStore), which looks the project
/// up in the data-dir registry and joins the relative path onto its `.sceneworks`
/// directory — with the same traversal guard and canonicalization the API uses. We reuse
/// that exact resolver here (rather than re-deriving the layout) so reveal/save operate on
/// the real file the API would serve. Exposed as a command so the frontend (sc-8727) can
/// turn an asset into an absolute path to feed [`reveal_in_os`] / [`save_asset_as`].
#[tauri::command]
pub fn resolve_asset_path(project_id: String, relative_path: String) -> Result<String, String> {
    let store = sceneworks_core::project_store::ProjectStore::new(
        resolved_data_dir(),
        env!("CARGO_PKG_VERSION"),
    );
    let project_file = store
        .project_file(&project_id, &relative_path)
        .map_err(|error| error.to_string())?;
    Ok(project_file.path.to_string_lossy().into_owned())
}

/// Decide whether a caller-supplied `start_dir` should seed the native save dialog's
/// initial directory (sc-8737). Returns `Some(path)` only when the value is non-empty
/// (after trimming) AND names an existing directory on disk; otherwise `None`, so an
/// empty/missing/non-directory hint is skipped gracefully rather than erroring the save.
/// Extracted as a pure helper so the accept/reject decision is unit-testable without the
/// GUI dialog.
fn sanitized_start_dir(start_dir: Option<&str>) -> Option<PathBuf> {
    let trimmed = start_dir?.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = PathBuf::from(trimmed);
    if path.is_dir() {
        Some(path)
    } else {
        None
    }
}

/// Save an asset file to a user-chosen destination (sc-8726). Opens the native "save as"
/// dialog pre-filled with `suggested_filename`, then copies the bytes from `source_path`
/// to the chosen destination. `source_path` must be an already-resolved absolute path
/// (see [`resolve_asset_path`]); it is validated to live inside an app-managed root by the
/// same guard as [`reveal_in_os`] so the frontend can't ask us to copy an arbitrary file
/// off disk. When `start_dir` names an existing directory the dialog opens there so the
/// user returns to their last-used save location (sc-8737); an empty/missing/non-existent
/// hint is ignored. Returns `Ok(None)` when the user cancels the dialog and `Ok(Some(dest))`
/// with the absolute destination path on success.
#[tauri::command]
pub async fn save_asset_as(
    app: AppHandle,
    source_path: String,
    suggested_filename: String,
    start_dir: Option<String>,
) -> Result<Option<String>, String> {
    // Guard the SOURCE before opening any dialog: only files inside the SceneWorks data
    // dir / HF cache may be copied out.
    let source = validate_reveal_target(&source_path)?;

    let suggested = suggested_filename.trim();
    let mut dialog = app.dialog().file();
    if !suggested.is_empty() {
        dialog = dialog.set_file_name(suggested);
    }
    if let Some(dir) = sanitized_start_dir(start_dir.as_deref()) {
        dialog = dialog.set_directory(dir);
    }
    let Some(destination) = dialog
        .blocking_save_file()
        .and_then(|file| file.into_path().ok())
    else {
        // User dismissed the save dialog.
        return Ok(None);
    };

    std::fs::copy(&source, &destination).map_err(|error| error.to_string())?;
    Ok(Some(destination.to_string_lossy().into_owned()))
}

/// Enumerate stored credentials for the Settings screen: host, label, scheme, and
/// whether the secret is present in the keychain. Never returns the token itself.
#[tauri::command]
pub fn list_credentials() -> Vec<CredentialStatus> {
    load_settings()
        .credentials
        .into_iter()
        .map(|meta| {
            let present = read_credential_secret(&meta.host).is_some();
            CredentialStatus {
                host: meta.host,
                label: meta.label,
                scheme: meta.scheme,
                present,
            }
        })
        .collect()
}

/// Save (or overwrite) the token for a host in the OS keychain and record its
/// non-secret metadata. The token is write-only — it is never read back to the UI.
#[tauri::command]
pub fn set_credential(
    app: AppHandle,
    host: String,
    label: String,
    scheme: CredentialScheme,
    token: String,
) -> Result<Vec<CredentialStatus>, String> {
    let host = normalize_host(&host);
    if host.is_empty() {
        return Err("A host is required (e.g. huggingface.co).".to_owned());
    }
    let token = token.trim();
    if token.is_empty() {
        return Err("A token is required; use Remove to clear a credential.".to_owned());
    }
    keyring::Entry::new(KEYRING_SERVICE, &cred_account(&host))
        .map_err(|error| error.to_string())?
        .set_password(token)
        .map_err(|error| error.to_string())?;
    let label = {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            host.clone()
        } else {
            trimmed.to_owned()
        }
    };
    let mut settings = load_settings();
    upsert_credential_meta(
        &mut settings,
        CredentialMeta {
            host: host.clone(),
            label,
            scheme,
        },
    );
    save_settings(&settings)?;
    // Drop any cached secret for this host so the worker pulls the new token on its
    // next download without an app restart (sc-5891).
    crate::setup::invalidate_credential_cache(&app, &host);
    Ok(list_credentials())
}

/// Remove a host's credential from the keychain and drop its metadata.
#[tauri::command]
pub fn delete_credential(app: AppHandle, host: String) -> Result<Vec<CredentialStatus>, String> {
    let host = normalize_host(&host);
    if let Ok(entry) = keyring::Entry::new(KEYRING_SERVICE, &cred_account(&host)) {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    // Also clear the pre-migration HF account so removing HF can't be undone by a
    // later migration re-importing the legacy token.
    if host == HF_HOST {
        if let Ok(legacy) = keyring::Entry::new(KEYRING_SERVICE, HF_TOKEN_ACCOUNT) {
            let _ = legacy.delete_credential();
        }
    }
    let mut settings = load_settings();
    remove_credential_meta(&mut settings, &host);
    save_settings(&settings)?;
    // Drop the cached secret so a revoked credential stops being served by the
    // credential socket without an app restart (sc-5891).
    crate::setup::invalidate_credential_cache(&app, &host);
    Ok(list_credentials())
}

#[tauri::command]
pub fn restart_worker(app: AppHandle) {
    // Kill the current GPU worker child; its supervisor respawns it. macOS runs the MLX
    // worker; Windows runs the candle worker (the Python worker was retired off-Mac,
    // sc-5563). Shared with the remote REST restart (epic 4484 story 12).
    crate::setup::restart_gpu_worker(&app);
}

#[tauri::command]
pub fn get_gpu_info() -> GpuInfo {
    #[cfg(target_os = "macos")]
    {
        let mut devices = Vec::new();
        if let Some(profile) = run_capture("system_profiler", &["SPDisplaysDataType"]) {
            for line in profile.lines() {
                if let Some((_, model)) = line.trim().split_once("Chipset Model:") {
                    devices.push(model.trim().to_owned());
                }
            }
        }
        let unified_memory_mb = run_capture("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok())
            .map(|bytes| bytes / (1024 * 1024));
        let wired_limit_mb = run_capture("sysctl", &["-n", "iogpu.wired_limit_mb"])
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0);
        GpuInfo {
            platform: "macos".to_owned(),
            devices,
            unified_memory_mb,
            wired_limit_mb,
        }
    }
    #[cfg(target_os = "windows")]
    {
        let mut devices = Vec::new();
        if let Some(output) = run_capture(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
        ) {
            for line in output.lines() {
                let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                match parts.as_slice() {
                    [name, memory, ..] => devices.push(format!("{name} ({memory} MB)")),
                    [name] => devices.push((*name).to_owned()),
                    _ => {}
                }
            }
        }
        GpuInfo {
            platform: "windows".to_owned(),
            devices,
            unified_memory_mb: None,
            wired_limit_mb: None,
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let devices = run_capture("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"])
            .map(|output| output.lines().map(str::to_owned).collect())
            .unwrap_or_default();
        GpuInfo {
            platform: "linux".to_owned(),
            devices,
            unified_memory_mb: None,
            wired_limit_mb: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique scratch directory under the system temp dir, so the path-resolution
    /// tests below don't need a `tempfile` dependency. Cleaned up by the caller.
    fn scratch_dir(tag: &str) -> PathBuf {
        let unique = format!(
            "sceneworks-desktop-test-{tag}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// sc-8726: the source-path guard's containment core accepts a file inside an
    /// app-managed root and rejects one outside it. This is the pure half of
    /// `validate_reveal_target` (the canonicalization + settings read is the shell
    /// around it), so a save/reveal can never be pointed at an arbitrary file.
    #[test]
    fn path_within_roots_accepts_inside_and_rejects_outside() {
        let root = scratch_dir("guard");
        let data_root = root.join("data");
        std::fs::create_dir_all(data_root.join("projects")).expect("data root");
        // Canonicalize both sides exactly as the guard does.
        let canonical_root = std::fs::canonicalize(&data_root).expect("canonical root");
        let inside = data_root.join("projects").join("asset.png");
        std::fs::write(&inside, b"x").expect("inside file");
        let inside = std::fs::canonicalize(&inside).expect("canonical inside");

        let outside = root.join("outside.png");
        std::fs::write(&outside, b"x").expect("outside file");
        let outside = std::fs::canonicalize(&outside).expect("canonical outside");

        let roots = vec![canonical_root];
        assert!(path_within_roots(&inside, &roots));
        assert!(!path_within_roots(&outside, &roots));
        // A root the target is not under is not sufficient.
        assert!(!path_within_roots(&inside, &[]));

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-8726: `resolve_asset_path` turns a (projectId, project-relative file.path)
    /// into the correct absolute on-disk path by delegating to the same `ProjectStore`
    /// resolver the API's `/projects/<id>/files/<path>` route uses. Drives that resolver
    /// against a real temp data dir + created project to prove the layout is right (the
    /// command only adds the `resolved_data_dir()` wiring around this call).
    #[test]
    fn project_store_resolves_relative_asset_to_absolute_disk_path() {
        use sceneworks_core::project_store::ProjectStore;

        let root = scratch_dir("resolve");
        let data_dir = root.join("data");
        let store = ProjectStore::new(&data_dir, "test-version");
        let project = store.create_project("Resolver").expect("project creates");

        // Materialize an asset inside the project's `.sceneworks` dir at a known
        // relative path, mirroring how the worker writes generated assets.
        let project_path = store
            .list_projects()
            .expect("list projects")
            .into_iter()
            .find(|p| p.id == project.id)
            .map(|p| PathBuf::from(p.path))
            .expect("project path");
        let relative = "assets/images/shot.png";
        let asset_path = project_path.join(relative);
        std::fs::create_dir_all(asset_path.parent().unwrap()).expect("asset dir");
        std::fs::write(&asset_path, b"png-bytes").expect("write asset");

        let resolved = store
            .project_file(&project.id, relative)
            .expect("resolve asset");
        let expected = std::fs::canonicalize(&asset_path).expect("canonical asset");
        assert_eq!(resolved.path, expected);

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-8753: the asset Save As / Reveal commands are invoked from the API-served UI
    /// at the remote (`http://127.0.0.1:*`) origin, so each must be granted an
    /// `allow-<kebab-command>` permission in `capabilities/default.json` or the Tauri
    /// ACL silently rejects the invoke in the real app (the unit tests mock `invoke`, so
    /// they can't catch a missing grant). Guards against future command↔ACL drift for
    /// the asset commands specifically: `resolve_asset_path` (Reveal + Save As both call
    /// it first) and `save_asset_as`.
    #[test]
    fn asset_commands_are_granted_in_the_remote_capability() {
        let capability = include_str!("../capabilities/default.json");
        let manifest: serde_json::Value =
            serde_json::from_str(capability).expect("capabilities/default.json parses");
        let permissions = manifest["permissions"]
            .as_array()
            .expect("permissions is an array")
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();
        for permission in ["allow-resolve-asset-path", "allow-save-asset-as"] {
            assert!(
                permissions.contains(&permission),
                "capabilities/default.json is missing `{permission}` — the command would \
                 be rejected by the ACL at the remote UI origin"
            );
        }
    }

    /// sc-8737: the save dialog's initial-directory hint is applied only when it names
    /// an existing directory; empty/whitespace, missing, and non-directory paths are all
    /// skipped (returning `None`) so a stale/bad hint never errors the save — it just
    /// falls back to the OS default location.
    #[test]
    fn start_dir_applied_only_for_existing_directory() {
        // None / empty / whitespace → skipped.
        assert_eq!(sanitized_start_dir(None), None);
        assert_eq!(sanitized_start_dir(Some("")), None);
        assert_eq!(sanitized_start_dir(Some("   ")), None);

        let root = scratch_dir("startdir");
        // A path that doesn't exist → skipped.
        let missing = root.join("does-not-exist");
        assert_eq!(sanitized_start_dir(Some(&missing.to_string_lossy())), None);

        // A file (not a directory) → skipped.
        let file = root.join("a-file.png");
        std::fs::write(&file, b"x").expect("write file");
        assert_eq!(sanitized_start_dir(Some(&file.to_string_lossy())), None);

        // An existing directory → applied (trimmed).
        let dir = root.join("saved-here");
        std::fs::create_dir_all(&dir).expect("create dir");
        let padded = format!("  {}  ", dir.to_string_lossy());
        assert_eq!(sanitized_start_dir(Some(&padded)), Some(dir.clone()));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn normalizes_hosts_and_urls() {
        assert_eq!(normalize_host("  HuggingFace.co "), "huggingface.co");
        assert_eq!(
            normalize_host("https://Civitai.com/api/models"),
            "civitai.com"
        );
        assert_eq!(normalize_host("http://example.com"), "example.com");
        assert_eq!(normalize_host(""), "");
    }

    #[test]
    fn upsert_inserts_then_updates_in_place() {
        let mut settings = AppSettings::default();
        upsert_credential_meta(
            &mut settings,
            CredentialMeta {
                host: "civitai.com".to_owned(),
                label: "Civit".to_owned(),
                scheme: CredentialScheme::Query,
            },
        );
        assert_eq!(settings.credentials.len(), 1);
        // The same host updates label/scheme in place rather than appending.
        upsert_credential_meta(
            &mut settings,
            CredentialMeta {
                host: "civitai.com".to_owned(),
                label: "Civit.ai".to_owned(),
                scheme: CredentialScheme::Bearer,
            },
        );
        assert_eq!(settings.credentials.len(), 1);
        assert_eq!(settings.credentials[0].label, "Civit.ai");
        assert_eq!(settings.credentials[0].scheme, CredentialScheme::Bearer);
    }

    #[test]
    fn remove_reports_whether_anything_changed() {
        let mut settings = AppSettings::default();
        upsert_credential_meta(
            &mut settings,
            CredentialMeta {
                host: HF_HOST.to_owned(),
                label: "Hugging Face".to_owned(),
                scheme: CredentialScheme::Bearer,
            },
        );
        assert!(hf_credential_recorded(&settings));
        assert!(remove_credential_meta(&mut settings, HF_HOST));
        assert!(!hf_credential_recorded(&settings));
        assert!(settings.credentials.is_empty());
        // Removing a host that isn't recorded is a no-op.
        assert!(!remove_credential_meta(&mut settings, HF_HOST));
    }

    /// sc-5891: the spawn-time credential gate must short-circuit on the non-secret
    /// metadata before any `keyring::Entry`/`get_password()` is constructed, so a
    /// no-credential install never touches the OS keychain. `hf_credential_recorded`
    /// is that gate for `read_hf_token`; an empty `credentials` list is likewise the
    /// gate for `credentials_env_json` (its loop body — the only `read_credential_secret`
    /// site — never runs). This headless test asserts the gate predicate without
    /// touching the keychain (which `read_hf_token` itself would, once past the gate).
    #[test]
    fn no_recorded_credential_means_no_keychain_gate() {
        let settings = AppSettings::default();
        assert!(!hf_credential_recorded(&settings));
        assert!(settings.credentials.is_empty());
    }

    /// epic 4484 story 1: LAN remote access is OFF by default with no port and no
    /// recorded password, and a default `settings.json` carries none of the new keys
    /// it doesn't need (so an untouched install reads exactly as before).
    #[test]
    fn remote_access_defaults_off() {
        let settings = AppSettings::default();
        assert!(!settings.remote_access_enabled);
        assert_eq!(settings.remote_port, None);
        assert!(!settings.remote_password_set);
        // The keychain gate short-circuits with no password recorded.
        assert!(!remote_password_recorded(&settings));
        // remote_port is Option → skipped when None; the bools default to false so
        // they round-trip but add no behavior.
        let json = serde_json::to_string(&settings).unwrap();
        assert!(!json.contains("remotePort"));
    }

    /// The new fields round-trip through serde with the expected camelCase keys.
    #[test]
    fn remote_access_settings_round_trip() {
        let settings = AppSettings {
            remote_access_enabled: true,
            remote_port: Some(DEFAULT_REMOTE_PORT),
            remote_password_set: true,
            ..AppSettings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("remoteAccessEnabled"));
        assert!(json.contains("remotePort"));
        assert!(json.contains("remotePasswordSet"));
        let back: AppSettings = serde_json::from_str(&json).unwrap();
        assert!(back.remote_access_enabled);
        assert_eq!(back.remote_port, Some(DEFAULT_REMOTE_PORT));
        assert!(back.remote_password_set);
    }

    /// Setting a password records the metadata; clearing it both drops the metadata
    /// AND disables remote access (fail-closed: no password ⇒ no LAN bind). This is
    /// the pure metadata half of `set_remote_password`/`clear_remote_password` — the
    /// keychain write/delete is exercised on real hardware (story 13).
    #[test]
    fn mark_remote_password_sets_then_clears_and_disables() {
        let mut settings = AppSettings {
            remote_access_enabled: true,
            remote_port: Some(DEFAULT_REMOTE_PORT),
            ..AppSettings::default()
        };
        mark_remote_password(&mut settings, true);
        assert!(settings.remote_password_set);
        assert!(remote_password_recorded(&settings));
        // Still enabled — setting a password doesn't toggle the switch.
        assert!(settings.remote_access_enabled);
        // Clearing the password forces remote access off.
        mark_remote_password(&mut settings, false);
        assert!(!settings.remote_password_set);
        assert!(!remote_password_recorded(&settings));
        assert!(!settings.remote_access_enabled);
        // Port choice is preserved across a clear (the user's preference persists).
        assert_eq!(settings.remote_port, Some(DEFAULT_REMOTE_PORT));
    }
}
