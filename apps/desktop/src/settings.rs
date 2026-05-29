//! Desktop settings surface (sc-1350): data directory, Hugging Face token (OS
//! keychain), detected GPU info, and a worker restart. Commands are invoked from
//! the React settings screen when running inside the Tauri shell.

use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

use crate::setup::{app_support_dir, default_data_dir, shared_huggingface_home, Managed};

const KEYRING_SERVICE: &str = "SceneWorks";
/// Pre-migration account that held the single Hugging Face token. Kept only for
/// one-time migration into the host-keyed store (and as a read fallback).
const HF_TOKEN_ACCOUNT: &str = "huggingface_token";
/// Host of the migrated Hugging Face credential.
const HF_HOST: &str = "huggingface.co";

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

fn hf_credential_recorded(settings: &AppSettings) -> bool {
    settings
        .credentials
        .iter()
        .any(|entry| entry.host == HF_HOST)
}

/// One-time move of the legacy single HF token into the host-keyed store as a
/// `huggingface.co` credential. Idempotent: a no-op once a `huggingface.co`
/// credential is recorded. Returns true when it migrated so the caller persists.
fn migrate_legacy_hf_token(settings: &mut AppSettings) -> bool {
    if hf_credential_recorded(settings) {
        return false;
    }
    let Some(token) = keyring::Entry::new(KEYRING_SERVICE, HF_TOKEN_ACCOUNT)
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
    else {
        return false;
    };
    match keyring::Entry::new(KEYRING_SERVICE, &cred_account(HF_HOST)) {
        Ok(entry) if entry.set_password(&token).is_ok() => {}
        _ => return false,
    }
    // Drop the legacy entry so the token isn't stored twice; best-effort.
    if let Ok(legacy) = keyring::Entry::new(KEYRING_SERVICE, HF_TOKEN_ACCOUNT) {
        let _ = legacy.delete_credential();
    }
    settings.credentials.push(CredentialMeta {
        host: HF_HOST.to_owned(),
        label: "Hugging Face".to_owned(),
        scheme: CredentialScheme::Bearer,
    });
    true
}

/// Load settings, running the one-time HF-token migration and persisting it if it
/// fired.
fn load_settings_migrated() -> AppSettings {
    let mut settings = load_settings();
    if migrate_legacy_hf_token(&mut settings) {
        let _ = save_settings(&settings);
    }
    settings
}

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
    /// Last-used UI theme (`"light"` or `"dark"`). Persisted here because the
    /// desktop webview's `localStorage` is keyed to the API's per-launch random
    /// port and so can't be relied on across runs (same reason as the wizard
    /// state). `None` until the user first toggles the theme.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
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

/// Hugging Face token for injecting `HF_TOKEN` into the worker. Prefers the
/// host-keyed `huggingface.co` credential, falling back to the pre-migration
/// `huggingface_token` account so a not-yet-migrated install still authenticates.
pub fn read_hf_token() -> Option<String> {
    read_credential_secret(HF_HOST).or_else(|| {
        keyring::Entry::new(KEYRING_SERVICE, HF_TOKEN_ACCOUNT)
            .ok()
            .and_then(|entry| entry.get_password().ok())
            .map(|token| token.trim().to_owned())
            .filter(|token| !token.is_empty())
    })
}

/// All stored credentials serialized as the worker's `SCENEWORKS_CREDENTIALS` JSON
/// map (`{ host: { token, scheme } }`), reading each secret from the keychain.
/// `None` when no credentials are stored. Injected into the worker at spawn.
pub fn credentials_env_json() -> Option<String> {
    let settings = load_settings_migrated();
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

/// The valid stored theme for `input`, or `None` if it isn't a theme we
/// recognize (so a stray value can't wedge the persisted setting).
fn normalize_theme(input: &str) -> Option<String> {
    match input.trim() {
        "light" => Some("light".to_owned()),
        "dark" => Some("dark".to_owned()),
        _ => None,
    }
}

/// Persist the last-used UI theme so the desktop shell restores it on the next
/// launch (the webview's `localStorage` is unreliable across runs — see the
/// wizard-state commands). Unrecognized values are ignored.
#[tauri::command]
pub fn save_app_theme(theme: String) -> Result<(), String> {
    let Some(theme) = normalize_theme(&theme) else {
        return Ok(());
    };
    let mut settings = load_settings();
    settings.theme = Some(theme);
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
    let target = PathBuf::from(&path);
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

/// Enumerate stored credentials for the Settings screen: host, label, scheme, and
/// whether the secret is present in the keychain. Never returns the token itself.
#[tauri::command]
pub fn list_credentials() -> Vec<CredentialStatus> {
    load_settings_migrated()
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
    host: String,
    label: String,
    scheme: CredentialScheme,
    token: String,
) -> Result<(), String> {
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
            host,
            label,
            scheme,
        },
    );
    save_settings(&settings)
}

/// Remove a host's credential from the keychain and drop its metadata.
#[tauri::command]
pub fn delete_credential(host: String) -> Result<(), String> {
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
    save_settings(&settings)
}

#[tauri::command]
pub fn restart_worker(app: AppHandle) {
    // Kill the current worker child; the supervisor restarts it.
    if let Some(child) = app
        .state::<Managed>()
        .worker
        .lock()
        .expect("worker lock")
        .take()
    {
        let _ = child.kill();
    }
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

    #[test]
    fn normalize_theme_accepts_only_known_themes() {
        assert_eq!(normalize_theme(" light "), Some("light".to_owned()));
        assert_eq!(normalize_theme("dark"), Some("dark".to_owned()));
        assert_eq!(normalize_theme("blue"), None);
        assert_eq!(normalize_theme(""), None);
    }

    #[test]
    fn migration_is_skipped_once_hf_is_recorded() {
        let mut settings = AppSettings::default();
        upsert_credential_meta(
            &mut settings,
            CredentialMeta {
                host: HF_HOST.to_owned(),
                label: "Hugging Face".to_owned(),
                scheme: CredentialScheme::Bearer,
            },
        );
        // With HF already recorded, migration short-circuits before any keychain
        // access, so this is safe to run in a headless test.
        assert!(!migrate_legacy_hf_token(&mut settings));
    }
}
