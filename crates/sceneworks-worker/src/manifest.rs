//! JSONC manifest read/upsert/write helpers shared by the LoRA and model manifests.
use super::*;

use std::io::Write as _;

use fs2::FileExt as _;

/// Max time to block waiting for the cross-process manifest lock before giving up
/// with a clear error. Manifest RMW is a few KB of JSON, so a real hold is sub-ms;
/// a stall this long means a stuck/crashed peer, and failing loudly beats silently
/// skipping the write and losing the freshly-installed entry (sc-8843).
const MANIFEST_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Poll cadence while spin-waiting on `try_lock_exclusive` (fs2 has no timed
/// blocking-lock API, so we retry rather than block indefinitely).
const MANIFEST_LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(25);

pub(crate) async fn read_json_value(path: &Path) -> WorkerResult<Value> {
    Ok(serde_json::from_slice(&tokio::fs::read(path).await?)?)
}

/// Upsert `entry` (keyed by its `id`) into the `collection_key` array of a JSONC
/// manifest at `path`, creating the manifest when absent. An existing entry with
/// the same id is merged (incoming fields win) but keeps its original `createdAt`.
/// Shared by the LoRA (`"loras"`) and model (`"models"`) manifests, which differed
/// only by this array key (sc-4279 / F-MLXW-15).
///
/// The default utility pool is 4 SEPARATE PROCESSES, so an in-process mutex cannot
/// serialize concurrent installs — two parallel upserts would each read the old
/// file, merge their own entry, and the second rename would clobber the first,
/// losing an entry (F-041 / sc-8843). The entire read→merge→write(rename) therefore
/// runs under a cross-process advisory *exclusive* file lock on a `<manifest>.lock`
/// sibling. The API writer (`apps/rust-api/src/manifest.rs`) takes the SAME sibling
/// lock so worker↔API writes to these shared files serialize too.
pub(crate) async fn upsert_manifest_entry(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    // `id` is validated up front (cheap, no I/O) so a malformed payload fails before
    // we take the lock.
    if entry.get("id").and_then(Value::as_str).is_none() {
        return Err(WorkerError::InvalidPayload(format!(
            "{collection_key} manifest entry requires id"
        )));
    }
    let path = path.to_path_buf();
    let collection_key = collection_key.to_owned();
    // The critical section is blocking (fs2 advisory lock + sync read/write) so it
    // runs on a blocking thread rather than stalling the async runtime.
    tokio::task::spawn_blocking(move || upsert_manifest_entry_locked(&path, &collection_key, entry))
        .await
        .map_err(|error| task_join_error("manifest upsert", error))?
}

/// Blocking read→merge→write of `path`, run under a cross-process exclusive lock on
/// the `<manifest>.lock` sibling. Callers must invoke this off the async runtime.
fn upsert_manifest_entry_locked(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _guard = ManifestLock::acquire(path)?;

    let mut manifest = match std::fs::read_to_string(path) {
        Ok(payload) => serde_json::from_str(&strip_jsonc_comments(&payload))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut object = serde_json::Map::new();
            object.insert("schemaVersion".to_owned(), json!(1));
            object.insert(collection_key.to_owned(), Value::Array(Vec::new()));
            Value::Object(object)
        }
        Err(error) => return Err(error.into()),
    };
    // Re-fetch under the lock; validated above, so this is infallible in practice.
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
    write_json_value_blocking(path, &manifest)
    // `_guard` (and the advisory lock) drops here, after the atomic rename lands.
}

/// RAII holder for a cross-process advisory exclusive lock on a `<manifest>.lock`
/// sibling file. The lock is released when the underlying file handle drops.
struct ManifestLock {
    _file: std::fs::File,
}

impl ManifestLock {
    fn acquire(manifest_path: &Path) -> WorkerResult<Self> {
        let lock_path = manifest_lock_path(manifest_path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;
        let deadline = std::time::Instant::now() + MANIFEST_LOCK_TIMEOUT;
        // fs2 signals lock contention with a platform-specific error: `EWOULDBLOCK`
        // (`ErrorKind::WouldBlock`) on Unix, but `ERROR_LOCK_VIOLATION` (raw OS 33,
        // `ErrorKind::Uncategorized`) on Windows. Matching on `ErrorKind` misses the
        // Windows case and would mis-propagate a real, retryable contention as a hard
        // error (sc-8843). Compare by RAW OS CODE against fs2's own contention error,
        // which is correct on every platform.
        let contended = fs2::lock_contended_error().raw_os_error();
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => return Ok(Self { _file: file }),
                Err(error) if error.raw_os_error() == contended => {
                    if std::time::Instant::now() >= deadline {
                        return Err(WorkerError::Io(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!(
                                "timed out after {:?} waiting for manifest lock {}",
                                MANIFEST_LOCK_TIMEOUT,
                                lock_path.display()
                            ),
                        )));
                    }
                    std::thread::sleep(MANIFEST_LOCK_POLL);
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

/// The `.lock` sibling path for a manifest. Kept alongside the manifest so the lock
/// scope is the exact file being mutated (per-file, not global).
fn manifest_lock_path(manifest_path: &Path) -> PathBuf {
    let mut name = manifest_path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    name.push(".lock");
    manifest_path.with_file_name(name)
}

/// Blocking sibling of [`write_json_value`]: atomic temp-write + rename. Used inside
/// the manifest lock's critical section (which is already off the async runtime).
fn write_json_value_blocking(path: &Path, value: &Value) -> WorkerResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = serde_json::to_vec_pretty(value)?;
    output.push(b'\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    let mut file = std::fs::File::create(&tmp_path)?;
    file.write_all(&output)?;
    file.sync_all()?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub(crate) async fn write_json_value(path: &Path, value: &Value) -> WorkerResult<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str) -> serde_json::Map<String, Value> {
        let mut map = serde_json::Map::new();
        map.insert("id".to_owned(), json!(id));
        map.insert("name".to_owned(), json!(format!("name-{id}")));
        map
    }

    async fn read_ids(path: &Path, collection_key: &str) -> Vec<String> {
        let payload = tokio::fs::read_to_string(path)
            .await
            .expect("read manifest");
        let manifest: Value =
            serde_json::from_str(&strip_jsonc_comments(&payload)).expect("parse manifest");
        manifest
            .get(collection_key)
            .and_then(Value::as_array)
            .expect("collection array")
            .iter()
            .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect()
    }

    /// Two concurrent upserts to the same manifest must BOTH survive. Without the
    /// cross-process lock, the read→merge→rename interleaves and one entry is lost
    /// (F-041 / sc-8843). The flock guard serializes the RMW even across threads
    /// (and, in the real deployment, across the separate utility-worker processes).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_upserts_all_survive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("user.loras.jsonc");

        // Many concurrent writers, each a distinct id, all racing the same file. If
        // the lock is missing this loses entries essentially every run.
        const WRITERS: usize = 24;
        let mut handles = Vec::with_capacity(WRITERS);
        for index in 0..WRITERS {
            let path = path.clone();
            handles.push(tokio::spawn(async move {
                upsert_manifest_entry(&path, "loras", entry(&format!("lora-{index}"))).await
            }));
        }
        for handle in handles {
            handle.await.expect("join").expect("upsert");
        }

        let mut ids = read_ids(&path, "loras").await;
        ids.sort();
        let mut expected: Vec<String> = (0..WRITERS).map(|i| format!("lora-{i}")).collect();
        expected.sort();
        assert_eq!(ids, expected, "every concurrent upsert must persist");
    }

    /// Upserting an existing id merges fields and preserves the original createdAt,
    /// and the lock path is unaffected by repeated calls.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn upsert_merges_and_preserves_created_at() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("user.models.jsonc");

        let mut first = entry("m1");
        first.insert("createdAt".to_owned(), json!("2020-01-01"));
        first.insert("keep".to_owned(), json!("original"));
        upsert_manifest_entry(&path, "models", first)
            .await
            .expect("first upsert");

        let mut second = entry("m1");
        second.insert("createdAt".to_owned(), json!("2099-12-31"));
        second.insert("name".to_owned(), json!("renamed"));
        upsert_manifest_entry(&path, "models", second)
            .await
            .expect("second upsert");

        let payload = tokio::fs::read_to_string(&path).await.expect("read");
        let manifest: Value = serde_json::from_str(&strip_jsonc_comments(&payload)).expect("parse");
        let models = manifest["models"].as_array().expect("array");
        assert_eq!(models.len(), 1, "same id must not duplicate");
        let m1 = &models[0];
        assert_eq!(m1["createdAt"], json!("2020-01-01"), "createdAt preserved");
        assert_eq!(m1["name"], json!("renamed"), "incoming fields win");
        assert_eq!(m1["keep"], json!("original"), "prior fields retained");
    }

    /// A lock timeout surfaces a clear error rather than silently skipping the write.
    #[tokio::test]
    async fn missing_id_errors_before_lock() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("user.loras.jsonc");
        let mut bad = serde_json::Map::new();
        bad.insert("name".to_owned(), json!("no-id"));
        let err = upsert_manifest_entry(&path, "loras", bad)
            .await
            .expect_err("missing id must error");
        assert!(matches!(err, WorkerError::InvalidPayload(_)));
        // Nothing should have been written.
        assert!(!path.exists(), "no manifest written for invalid payload");
    }
}
