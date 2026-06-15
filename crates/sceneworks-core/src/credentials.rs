//! Host-keyed download credential store backed by a `0600` JSON file in the config
//! dir (sc-1893). Used by the server/Docker deployment, where there is no per-user
//! OS keychain: the rust-api manages the file via `/api/v1/credentials` and the
//! worker reads it. The on-disk shape mirrors the worker's `SCENEWORKS_CREDENTIALS`
//! env map (`{ host: { token, scheme } }`) with an extra `label`, so the worker can
//! parse either source the same way.
//!
//! There is no application-level encryption: a headless container has no per-user
//! vault and any key would have to live beside the data, so the realistic
//! protection is the restricted file mode plus orchestrator (Docker/K8s) secrets.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::store_util::random_hex;

/// Filename of the credential store within the config dir. Shared so the rust-api
/// (writer) and the worker (reader) never drift.
pub const CREDENTIALS_FILENAME: &str = "credentials.json";

/// A stored credential's persisted fields. The token is the only secret.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialEntry {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub scheme: String,
    pub token: String,
}

/// Redacted view returned to clients: never includes the token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialStatus {
    pub host: String,
    pub label: String,
    pub scheme: String,
    pub present: bool,
}

/// File-backed credential store. Hosts are the map keys; lookups and writes
/// normalize the host (trim + lower-case).
pub struct CredentialFileStore {
    path: PathBuf,
}

impl CredentialFileStore {
    pub fn new(config_dir: &Path) -> Self {
        Self {
            path: config_dir.join(CREDENTIALS_FILENAME),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load the stored map, or an empty map when the file is absent or unreadable.
    pub fn load(&self) -> BTreeMap<String, CredentialEntry> {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|body| serde_json::from_str(&body).ok())
            .unwrap_or_default()
    }

    /// Redacted listing (host/label/scheme/present), sorted by host. No tokens.
    pub fn list(&self) -> Vec<CredentialStatus> {
        self.load()
            .into_iter()
            .map(|(host, entry)| CredentialStatus {
                present: !entry.token.trim().is_empty(),
                scheme: normalize_scheme(&entry.scheme),
                label: entry.label.trim().to_owned(),
                host,
            })
            .collect()
    }

    /// Upsert a host's credential. Returns the normalized host key.
    pub fn set(&self, host: &str, label: &str, scheme: &str, token: &str) -> io::Result<String> {
        let host = normalize_host(host);
        let mut map = self.load();
        map.insert(
            host.clone(),
            CredentialEntry {
                label: label.trim().to_owned(),
                scheme: normalize_scheme(scheme),
                token: token.trim().to_owned(),
            },
        );
        self.save(&map)?;
        Ok(host)
    }

    /// Remove a host's credential. Returns whether anything was removed.
    pub fn delete(&self, host: &str) -> io::Result<bool> {
        let host = normalize_host(host);
        let mut map = self.load();
        let removed = map.remove(&host).is_some();
        if removed {
            self.save(&map)?;
        }
        Ok(removed)
    }

    fn save(&self, map: &BTreeMap<String, CredentialEntry>) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = serde_json::to_string_pretty(map)?;
        // Stage into a sibling temp file then atomically rename into place, so a
        // crash never leaves a half-written or default-mode secrets file. On Unix
        // the temp is created 0600 up front, closing the window where the token
        // was briefly world/group-readable between a plain write and the chmod
        // (sc-4268 / F-CORE-8).
        let extension = self
            .path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json");
        let token = random_hex(8).map_err(io::Error::other)?;
        let tmp_path = self.path.with_extension(format!("{extension}.{token}.tmp"));
        write_secret_file(&tmp_path, body.as_bytes())?;
        std::fs::rename(&tmp_path, &self.path)?;
        // Backstop: re-assert 0600 on the final path in case it already existed
        // with looser permissions on a platform without atomic-mode creation.
        restrict_permissions(&self.path)?;
        Ok(())
    }
}

/// Write `contents` to `path`, creating the file with `0600` from the start on
/// Unix so a secret is never momentarily readable by other local users. On other
/// platforms this is a plain write (the desktop build keeps secrets in the OS
/// keychain, not this file).
#[cfg(unix)]
fn write_secret_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(contents)
}

#[cfg(not(unix))]
fn write_secret_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    std::fs::write(path, contents)
}

/// Normalize a user-entered host or URL to a bare lower-cased host (strip scheme
/// and any path).
pub fn normalize_host(input: &str) -> String {
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

/// Collapse an arbitrary scheme string to the two we support, defaulting to bearer.
pub fn normalize_scheme(scheme: &str) -> String {
    match scheme.trim().to_ascii_lowercase().as_str() {
        "query" => "query".to_owned(),
        _ => "bearer".to_owned(),
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> io::Result<()> {
    // Windows uses ACLs rather than POSIX mode bits, and the desktop build keeps
    // secrets in the OS keychain rather than this file, so there's nothing to do.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_list_delete_round_trip_redacts_token() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        assert!(store.list().is_empty());

        let host = store
            .set("https://Civitai.com", "Civit.ai", "query", " key ")
            .expect("set");
        assert_eq!(host, "civitai.com");

        let list = store.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].host, "civitai.com");
        assert_eq!(list[0].label, "Civit.ai");
        assert_eq!(list[0].scheme, "query");
        assert!(list[0].present);
        // The redacted listing must never carry the token.
        let serialized = serde_json::to_string(&list).expect("serialize");
        assert!(!serialized.contains("key"), "listing leaked the token");
        // ...but it is persisted (trimmed) for the worker to read.
        assert_eq!(store.load()["civitai.com"].token, "key");

        assert!(store.delete("civitai.com").expect("delete"));
        assert!(!store.delete("civitai.com").expect("delete again"));
        assert!(store.list().is_empty());
    }

    #[test]
    fn unknown_scheme_defaults_to_bearer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        store.set("huggingface.co", "", "weird", "hf").expect("set");
        assert_eq!(store.list()[0].scheme, "bearer");
    }

    /// sc-4268 / F-CORE-8: the secrets file is written via a 0600 staging temp +
    /// atomic rename. Verify the final file is 0600 even when it pre-existed with
    /// loose perms (the backstop chmod), and that no staging temp is left behind.
    #[cfg(unix)]
    #[test]
    fn save_is_atomic_and_0600_over_loose_existing_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        // Pre-create the credentials file with loose 0644 perms.
        std::fs::write(store.path(), "{}").expect("seed");
        std::fs::set_permissions(store.path(), std::fs::Permissions::from_mode(0o644))
            .expect("loosen");

        store.set("example.com", "", "bearer", "tok").expect("set");

        let mode = std::fs::metadata(store.path())
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "final file must be 0600");
        let tmp = store.path().with_extension("json.tmp");
        assert!(
            !tmp.exists(),
            "staging temp must be renamed away, not left behind"
        );
        // Content round-trips through the atomic write.
        assert_eq!(store.load()["example.com"].token, "tok");
    }

    #[cfg(unix)]
    #[test]
    fn stored_file_is_mode_600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        store.set("example.com", "", "bearer", "tok").expect("set");
        let mode = std::fs::metadata(store.path())
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
