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

use crate::store_util::{lock_store_path, random_hex};

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

    /// Load the stored map.
    ///
    /// An absent file is the legitimate first-run case and yields an empty map.
    /// A file that is present but unreadable or unparseable is *not* treated as
    /// empty: it propagates an error (and logs a warning), so that a subsequent
    /// destructive `save` in `set`/`delete` refuses to overwrite corrupt-but-real
    /// data with an empty map (sc-8814 / F-012). One malformed byte (partial
    /// disk write, a botched manual edit) must never silently wipe every stored
    /// credential on the next write.
    pub fn load(&self) -> io::Result<BTreeMap<String, CredentialEntry>> {
        let body = match std::fs::read_to_string(&self.path) {
            Ok(body) => body,
            // Absent file: nothing stored yet — the legitimate empty-map case.
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            // Present but unreadable (permissions, I/O error, etc.): surface it
            // rather than pretend the store is empty.
            Err(error) => {
                tracing::warn!(
                    path = %self.path.display(),
                    error = %error,
                    "failed to read credential store; refusing to treat it as empty",
                );
                return Err(error);
            }
        };
        serde_json::from_str(&body).map_err(|error| {
            tracing::warn!(
                path = %self.path.display(),
                error = %error,
                "credential store is present but unparseable; refusing to treat it as empty \
                 (fix or remove the file to recover)",
            );
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "credential store {} is corrupt: {error}",
                    self.path.display()
                ),
            )
        })
    }

    /// Redacted listing (host/label/scheme/present), sorted by host. No tokens.
    ///
    /// Propagates a load error (corrupt/unreadable file) rather than reporting an
    /// empty list that would hide the corruption from the operator (sc-8814).
    pub fn list(&self) -> io::Result<Vec<CredentialStatus>> {
        Ok(self
            .load()?
            .into_iter()
            .map(|(host, entry)| CredentialStatus {
                present: !entry.token.trim().is_empty(),
                scheme: normalize_scheme(&entry.scheme),
                label: entry.label.trim().to_owned(),
                host,
            })
            .collect())
    }

    /// Upsert a host's credential. Returns the normalized host key.
    ///
    /// The whole read-modify-write is serialized under a process-wide lock keyed
    /// on the file path, so concurrent `set`/`delete` calls (rust-api builds a
    /// fresh store per request) can't clobber each other's entries (sc-8813).
    /// In-process only — sufficient because the rust-api is the file's sole
    /// writer; the worker only reads it, which the atomic rename in `save`
    /// already keeps consistent.
    ///
    /// If the existing store is present but corrupt, `load` errors and the write
    /// is refused rather than overwriting real credentials with an empty map
    /// (sc-8814 / F-012). The load happens under the same guard as the save, so
    /// the read-modify-write stays atomic (sc-8813).
    pub fn set(&self, host: &str, label: &str, scheme: &str, token: &str) -> io::Result<String> {
        let host = normalize_host(host);
        let _guard = lock_store_path(&self.path);
        let mut map = self.load()?;
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
    /// Serialized against concurrent `set`/`delete` like [`Self::set`] (sc-8813).
    /// A corrupt existing store errors instead of being silently replaced by an
    /// empty map (sc-8814 / F-012).
    pub fn delete(&self, host: &str) -> io::Result<bool> {
        let host = normalize_host(host);
        let _guard = lock_store_path(&self.path);
        let mut map = self.load()?;
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

/// True when a Unix file mode grants any access to the group or "other" classes,
/// i.e. it is looser than the `0600` we write. Pure so the security decision is
/// unit-tested without touching the filesystem. The permission bits live in the
/// low 9 bits; `0o077` is the group + other read/write/execute mask.
pub fn mode_is_group_or_world_accessible(mode: u32) -> bool {
    mode & 0o077 != 0
}

/// If the credentials file at `path` exists and its mode is looser than `0600`
/// (group- or world-accessible), return the offending permission bits so the
/// caller can warn the operator to `chmod 600` it. Returns `None` when the file
/// is absent, unreadable, or already restricted. Unix-only: other platforms keep
/// secrets in the OS keychain rather than this file (see module docs).
#[cfg(unix)]
pub fn loose_credentials_mode(path: &Path) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path).ok()?.permissions().mode() & 0o777;
    mode_is_group_or_world_accessible(mode).then_some(mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_list_delete_round_trip_redacts_token() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        assert!(store.list().expect("list").is_empty());

        let host = store
            .set("https://Civitai.com", "Civit.ai", "query", " key ")
            .expect("set");
        assert_eq!(host, "civitai.com");

        let list = store.list().expect("list");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].host, "civitai.com");
        assert_eq!(list[0].label, "Civit.ai");
        assert_eq!(list[0].scheme, "query");
        assert!(list[0].present);
        // The redacted listing must never carry the token.
        let serialized = serde_json::to_string(&list).expect("serialize");
        assert!(!serialized.contains("key"), "listing leaked the token");
        // ...but it is persisted (trimmed) for the worker to read.
        assert_eq!(store.load().expect("load")["civitai.com"].token, "key");

        assert!(store.delete("civitai.com").expect("delete"));
        assert!(!store.delete("civitai.com").expect("delete again"));
        assert!(store.list().expect("list").is_empty());
    }

    #[test]
    fn unknown_scheme_defaults_to_bearer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        store.set("huggingface.co", "", "weird", "hf").expect("set");
        assert_eq!(store.list().expect("list")[0].scheme, "bearer");
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
        assert_eq!(store.load().expect("load")["example.com"].token, "tok");
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

    /// sc-8813 / F-011: `set`/`delete` are read-modify-write over the whole file,
    /// so unsynchronized concurrent calls silently drop each other's entries.
    /// Hammer the store from many threads (fresh store instances, like the
    /// per-request stores in rust-api) and assert no update is lost.
    #[test]
    fn concurrent_sets_do_not_lose_updates() {
        use std::sync::Barrier;

        const THREADS: usize = 8;
        const HOSTS_PER_THREAD: usize = 8;

        let dir = tempfile::tempdir().expect("tempdir");
        let config_dir = dir.path().to_path_buf();
        let barrier = std::sync::Arc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|thread| {
                let config_dir = config_dir.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    // Fresh store per thread, mirroring rust-api's per-request stores.
                    let store = CredentialFileStore::new(&config_dir);
                    barrier.wait();
                    for index in 0..HOSTS_PER_THREAD {
                        store
                            .set(
                                &format!("host-{thread}-{index}.example.com"),
                                "",
                                "bearer",
                                &format!("tok-{thread}-{index}"),
                            )
                            .expect("set");
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread");
        }

        let map = CredentialFileStore::new(&config_dir).load().expect("load");
        assert_eq!(
            map.len(),
            THREADS * HOSTS_PER_THREAD,
            "a concurrent set lost another thread's credential"
        );
    }

    /// sc-8813 / F-011: interleave `set` (new hosts) with `delete` (seeded hosts)
    /// and assert the final file is exactly the surviving set — no deleted host
    /// resurrected, no newly-set host lost.
    #[test]
    fn concurrent_set_and_delete_do_not_lose_updates() {
        use std::sync::Barrier;

        const PAIRS: usize = 8;

        let dir = tempfile::tempdir().expect("tempdir");
        let config_dir = dir.path().to_path_buf();
        let seed_store = CredentialFileStore::new(&config_dir);
        for index in 0..PAIRS {
            seed_store
                .set(&format!("old-{index}.example.com"), "", "bearer", "tok")
                .expect("seed");
        }

        let barrier = std::sync::Arc::new(Barrier::new(PAIRS * 2));
        let mut handles = Vec::new();
        for index in 0..PAIRS {
            let config_dir_set = config_dir.clone();
            let barrier_set = barrier.clone();
            handles.push(std::thread::spawn(move || {
                let store = CredentialFileStore::new(&config_dir_set);
                barrier_set.wait();
                store
                    .set(&format!("new-{index}.example.com"), "", "bearer", "tok")
                    .expect("set");
            }));
            let config_dir_delete = config_dir.clone();
            let barrier_delete = barrier.clone();
            handles.push(std::thread::spawn(move || {
                let store = CredentialFileStore::new(&config_dir_delete);
                barrier_delete.wait();
                assert!(
                    store
                        .delete(&format!("old-{index}.example.com"))
                        .expect("delete"),
                    "seeded host was already gone — a concurrent write clobbered it"
                );
            }));
        }
        for handle in handles {
            handle.join().expect("thread");
        }

        let map = CredentialFileStore::new(&config_dir).load().expect("load");
        let expected: Vec<String> = (0..PAIRS)
            .map(|index| format!("new-{index}.example.com"))
            .collect();
        let actual: Vec<String> = map.keys().cloned().collect();
        assert_eq!(
            actual, expected,
            "set/delete interleaving lost or resurrected an entry"
        );
    }

    #[test]
    fn mode_predicate_flags_only_group_or_world_access() {
        // The 0600 we write is fine; so is the even tighter 0400/0000.
        assert!(!mode_is_group_or_world_accessible(0o600));
        assert!(!mode_is_group_or_world_accessible(0o400));
        assert!(!mode_is_group_or_world_accessible(0o000));
        // Any group or other bit (read, write, or execute) is too loose.
        assert!(mode_is_group_or_world_accessible(0o640)); // group read
        assert!(mode_is_group_or_world_accessible(0o604)); // other read
        assert!(mode_is_group_or_world_accessible(0o660)); // group write
        assert!(mode_is_group_or_world_accessible(0o644)); // typical umask leak
        assert!(mode_is_group_or_world_accessible(0o777));
    }

    /// The startup-warning probe must fire only for a present, too-loose file and
    /// stay silent when the file is absent or already 0600.
    #[cfg(unix)]
    #[test]
    fn loose_mode_reports_only_present_loose_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());

        // Absent file → nothing to warn about.
        assert_eq!(loose_credentials_mode(store.path()), None);

        // Present and properly restricted (0600) → no warning.
        store.set("example.com", "", "bearer", "tok").expect("set");
        assert_eq!(loose_credentials_mode(store.path()), None);

        // Loosened to 0644 → report the offending mode for the operator message.
        std::fs::set_permissions(store.path(), std::fs::Permissions::from_mode(0o644))
            .expect("loosen");
        assert_eq!(loose_credentials_mode(store.path()), Some(0o644));
    }

    /// sc-8814 / F-012: an absent file is the legitimate first-run case — `load`
    /// yields an empty map and a `set` creates the store from scratch.
    #[test]
    fn absent_file_loads_empty_and_first_set_creates_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        assert!(!store.path().exists(), "no file should exist yet");
        assert!(
            store.load().expect("absent file loads as empty").is_empty(),
            "an absent file must load as an empty map, not error",
        );

        store
            .set("huggingface.co", "HF", "bearer", "hf-token")
            .expect("first set must create the store");
        assert!(store.path().exists(), "set must have created the file");
        assert_eq!(
            store.load().expect("load")["huggingface.co"].token,
            "hf-token",
        );
    }

    /// sc-8814 / F-012: a present-but-corrupt file must NOT be treated as empty.
    /// `load`/`list` error, and a subsequent `set`/`delete` refuses to persist —
    /// the corrupt bytes are left on disk rather than clobbered with an empty map.
    #[test]
    fn corrupt_file_errors_and_write_refuses_to_wipe() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());

        // A single malformed edit: valid JSON prefix, then garbage.
        let garbage = "{ \"huggingface.co\": { \"token\": \"real-secret\" ";
        std::fs::write(store.path(), garbage).expect("seed corrupt file");

        // load / list surface the corruption instead of pretending it is empty.
        let load_err = store.load().expect_err("corrupt file must error");
        assert_eq!(load_err.kind(), io::ErrorKind::InvalidData);
        assert!(store.list().is_err(), "list must propagate the corruption");

        // A write must NOT overwrite the corrupt-but-real data with an empty map.
        assert!(
            store.set("civitai.com", "", "bearer", "new").is_err(),
            "set over a corrupt store must refuse rather than wipe",
        );
        assert!(
            store.delete("huggingface.co").is_err(),
            "delete over a corrupt store must refuse rather than wipe",
        );

        // The original bytes are still on disk — nothing was destroyed.
        assert_eq!(
            std::fs::read_to_string(store.path()).expect("read back"),
            garbage,
            "the corrupt file must be preserved verbatim, not overwritten",
        );
    }

    /// sc-8814 / F-012: an *empty* file (0 bytes) is not valid JSON, so it is
    /// treated as corrupt rather than silently empty — a truncated write must not
    /// look like "no credentials".
    #[test]
    fn empty_file_is_treated_as_corrupt_not_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = CredentialFileStore::new(dir.path());
        std::fs::write(store.path(), "").expect("seed empty file");
        assert!(
            store.load().is_err(),
            "a zero-byte file is a truncated write, not an empty store",
        );
    }
}
