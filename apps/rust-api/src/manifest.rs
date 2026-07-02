use super::*;

use fs2::FileExt as _;

const MANIFEST_CACHE_LIMIT: usize = 16;
/// Max time to block on the cross-process manifest lock before erroring. Mirrors the
/// worker's `MANIFEST_LOCK_TIMEOUT` (sc-8843): the API and the utility-worker pool
/// (separate processes) write the SAME `user.*.jsonc` files, so both take the same
/// `<manifest>.lock` sibling to serialize cross-writer read→merge→rename.
const MANIFEST_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const MANIFEST_LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// RAII holder for the cross-process advisory exclusive lock on a `<manifest>.lock`
/// sibling. Released when the file handle drops. Acquired on a blocking thread since
/// flock is synchronous (see [`acquire_manifest_file_lock`]).
struct ManifestFileLock {
    _file: std::fs::File,
}

fn manifest_lock_path(manifest_path: &FsPath) -> PathBuf {
    let mut name = manifest_path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    name.push(".lock");
    manifest_path.with_file_name(name)
}

/// Acquire the cross-process manifest lock, off the async runtime (flock blocks).
async fn acquire_manifest_file_lock(path: &FsPath) -> Result<ManifestFileLock, ApiError> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            ApiError::internal(format!(
                "Failed to create manifest directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let lock_path = manifest_lock_path(path);
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| {
                ApiError::internal(format!(
                    "Failed to open manifest lock {}: {error}",
                    lock_path.display()
                ))
            })?;
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
                Ok(()) => return Ok(ManifestFileLock { _file: file }),
                Err(error) if error.raw_os_error() == contended => {
                    if std::time::Instant::now() >= deadline {
                        return Err(ApiError::internal(format!(
                            "Timed out after {MANIFEST_LOCK_TIMEOUT:?} waiting for manifest lock {}",
                            lock_path.display()
                        )));
                    }
                    std::thread::sleep(MANIFEST_LOCK_POLL);
                }
                Err(error) => {
                    return Err(ApiError::internal(format!(
                        "Failed to acquire manifest lock {}: {error}",
                        lock_path.display()
                    )))
                }
            }
        }
    })
    .await
    .map_err(|error| ApiError::internal(format!("Manifest lock task failed: {error}")))?
}
pub(crate) const API_MANAGED_MANIFEST_HEADER: &str = "// This file is rewritten by the SceneWorks API. Inline JSONC comments are not preserved across writes.";

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub(crate) struct ManifestCacheKey {
    path: PathBuf,
    field: String,
    modified_ns: u128,
    size: u64,
}

#[derive(Debug, Default)]
pub(crate) struct ManifestCache {
    entries: HashMap<ManifestCacheKey, Vec<Value>>,
    order: VecDeque<ManifestCacheKey>,
}

impl ManifestCache {
    fn get(&mut self, key: &ManifestCacheKey) -> Option<Vec<Value>> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: ManifestCacheKey, value: Vec<Value>) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
        while self.order.len() > MANIFEST_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &ManifestCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

pub(crate) async fn mutate_manifest_entries<F, R>(
    state: &AppState,
    path: &FsPath,
    field: &str,
    operation: F,
) -> Result<R, ApiError>
where
    F: FnOnce(Vec<Value>) -> Result<(Vec<Value>, R), ApiError>,
{
    let lock = manifest_write_lock(state, path);
    let _guard = lock.lock().await;
    // Cross-process lock (sc-8843): the utility-worker pool writes the same
    // `user.*.jsonc` files from separate processes, so the in-process mutex above is
    // not enough — hold the shared `<manifest>.lock` across the whole read→merge→save.
    let _file_lock = acquire_manifest_file_lock(path).await?;
    let entries = load_manifest_entries(state, path, field).await?;
    let (entries, result) = operation(entries)?;
    save_manifest_entries(path, field, entries).await?;
    Ok(result)
}

pub(crate) async fn remove_catalog_manifest_entry(
    state: &AppState,
    path: &FsPath,
    field: &str,
    id: &str,
) -> Result<Option<Value>, ApiError> {
    mutate_manifest_entries(state, path, field, |entries| {
        let mut removed = None;
        let entries = entries
            .into_iter()
            .filter(|entry| {
                if entry.get("id").and_then(Value::as_str) == Some(id) {
                    removed = Some(entry.clone());
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        Ok((entries, removed))
    })
    .await
}

pub(crate) fn manifest_write_lock(state: &AppState, path: &FsPath) -> Arc<AsyncMutex<()>> {
    let mut locks = state.manifest_write_locks.lock();
    locks
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

pub(crate) async fn save_manifest_entries(
    path: &FsPath,
    field: &str,
    entries: Vec<Value>,
) -> Result<(), ApiError> {
    let Some(parent) = path.parent() else {
        return Err(ApiError::internal("Manifest path has no parent directory"));
    };
    tokio::fs::create_dir_all(parent).await.map_err(|error| {
        ApiError::internal(format!(
            "Failed to create manifest directory {}: {error}",
            parent.display()
        ))
    })?;
    let mut manifest = load_manifest_root(path).await?;
    manifest.entry("$schema".to_owned()).or_insert_with(|| {
        Value::String("https://sceneworks.local/schemas/recipe-preset.schema.json".to_owned())
    });
    manifest
        .entry("schemaVersion".to_owned())
        .or_insert_with(|| json!(1));
    manifest.insert(field.to_owned(), Value::Array(entries));
    let payload = serde_json::to_string_pretty(&Value::Object(manifest))
        .map_err(|error| ApiError::internal(format!("Failed to encode manifest: {error}")))?;
    write_manifest_atomic(path, &format!("{API_MANAGED_MANIFEST_HEADER}\n{payload}\n")).await
}

pub(crate) async fn load_manifest_root(path: &FsPath) -> Result<JsonObject, ApiError> {
    let payload = match tokio::fs::read_to_string(path).await {
        Ok(payload) => payload,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(JsonObject::new()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to load manifest {}: {error}",
                path.display()
            )))
        }
    };
    serde_json::from_str::<Value>(&strip_jsonc_comments(&payload))
        .map_err(|error| {
            ApiError::internal(format!(
                "Failed to parse manifest {}: {error}",
                path.display()
            ))
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            ApiError::internal(format!("Manifest {} must be a JSON object", path.display()))
        })
}

pub(crate) async fn write_manifest_atomic(path: &FsPath, payload: &str) -> Result<(), ApiError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("jsonc");
    let tmp_path = path.with_extension(format!("{extension}.{}.tmp", Uuid::new_v4().simple()));
    tokio::fs::write(&tmp_path, payload)
        .await
        .map_err(|error| {
            ApiError::internal(format!(
                "Failed to write manifest temp file {}: {error}",
                tmp_path.display()
            ))
        })?;
    tokio::fs::rename(&tmp_path, path).await.map_err(|error| {
        let _ = std::fs::remove_file(&tmp_path);
        ApiError::internal(format!(
            "Failed to replace manifest {}: {error}",
            path.display()
        ))
    })
}

pub(crate) async fn load_manifest_entries(
    state: &AppState,
    path: &FsPath,
    field: &str,
) -> Result<Vec<Value>, ApiError> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(ApiError::internal(format!(
                "Failed to stat manifest {}: {error}",
                path.display()
            )))
        }
    };
    let cache_key = ManifestCacheKey {
        path: path.to_path_buf(),
        field: field.to_owned(),
        modified_ns: metadata_modified_ns(&metadata),
        size: metadata.len(),
    };
    if let Some(entries) = state.manifest_cache.lock().get(&cache_key) {
        return Ok(entries);
    }

    let payload = tokio::fs::read_to_string(path).await.map_err(|error| {
        ApiError::internal(format!(
            "Failed to load manifest {}: {error}",
            path.display()
        ))
    })?;
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(&payload)).map_err(|error| {
            ApiError::internal(format!(
                "Failed to parse manifest {}: {error}",
                path.display()
            ))
        })?;
    let entries = manifest
        .get(field)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    state
        .manifest_cache
        .lock()
        .insert(cache_key, entries.clone());
    Ok(entries)
}

pub(crate) fn metadata_modified_ns(metadata: &std::fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

pub(crate) fn merge_entries_by_id(builtin: Vec<Value>, user: Vec<Value>) -> Vec<Value> {
    let mut entries = Vec::<Value>::new();
    for entry in builtin {
        if entry.get("id").and_then(Value::as_str).is_some() {
            entries.push(entry);
        }
    }
    for entry in user {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        if let Some(existing) = entries
            .iter_mut()
            .find(|existing| existing.get("id").and_then(Value::as_str) == Some(id))
        {
            merge_object(existing, entry);
        } else {
            entries.push(entry);
        }
    }
    entries
}

pub(crate) fn merge_object(base: &mut Value, override_value: Value) {
    if let (Some(base_object), Some(override_object)) =
        (base.as_object_mut(), override_value.as_object())
    {
        for (key, value) in override_object {
            base_object.insert(key.clone(), value.clone());
        }
    } else {
        *base = override_value;
    }
}

// Shared JSONC comment stripper, moved to sceneworks-core (sc-4279 / F-MLXW-15).
// Re-exported `pub(crate)` so the crate-root `use manifest::*` keeps it available
// to the rest of the api (and tests) under the same path as before.
pub(crate) use sceneworks_core::jsonc::strip_jsonc_comments;
