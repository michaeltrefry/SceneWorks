use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::OnceLock;

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use rusqlite::Connection;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::project_store::{ProjectStoreError, ProjectStoreResult};

pub(crate) fn read_json(path: &Path) -> ProjectStoreResult<Value> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

/// Atomically writes `contents` to `path`: stage into a uniquely-named temp file
/// in the same directory, then rename it into place.
///
/// The temp suffix carries random bytes so two threads writing the *same* target
/// never collide on the temp path. Previously the temp name was deterministic
/// (`foo.json` -> `foo.json.tmp`), so overlapping writers interleaved their bytes
/// into one temp file and the second `rename` failed once the first had already
/// renamed the temp away (sc-1633). Pair with [`lock_project_files`] when the
/// caller does a read-modify-write so concurrent updates don't clobber each other.
pub(crate) fn atomic_write(path: &Path, contents: &[u8]) -> ProjectStoreResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("json");
    let token = random_hex(8)?;
    let tmp_path = path.with_extension(format!("{extension}.{token}.tmp"));
    fs::write(&tmp_path, contents)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

pub(crate) fn write_json<T: Serialize>(path: &Path, payload: &T) -> ProjectStoreResult<()> {
    let mut output = serde_json::to_string_pretty(payload)?;
    output.push('\n');
    atomic_write(path, output.as_bytes())
}

const PROJECT_LOCK_STRIPES: usize = 256;

/// Process-wide striped locks that serialize file read-modify-write within a
/// single project. A store resolves the project path, then holds this guard
/// across its read -> mutate -> write so concurrent API/worker mutations to the
/// same project can't lose each other's sidecar/JSON updates (sc-1633).
///
/// Striping (rather than a per-path map) keeps this allocation-free with no
/// unbounded growth; distinct projects only contend on the rare hash collision.
/// The lock is *reentrant* so a mutating method that calls another mutating
/// method on the same project (e.g. `create_timeline` -> `save_timeline`) does
/// not self-deadlock on the single blocking thread that runs the call chain.
///
/// Scope: this guards mutations that flow through `ProjectStore` within one
/// process. It does not coordinate separate OS processes writing the same
/// project directory; that is out of scope for the single-API desktop model.
fn project_locks() -> &'static [ReentrantMutex<()>] {
    static LOCKS: OnceLock<Vec<ReentrantMutex<()>>> = OnceLock::new();
    LOCKS.get_or_init(|| {
        (0..PROJECT_LOCK_STRIPES)
            .map(|_| ReentrantMutex::new(()))
            .collect()
    })
}

pub(crate) fn lock_project_files(project_path: &Path) -> ReentrantMutexGuard<'static, ()> {
    let mut hasher = DefaultHasher::new();
    project_path.hash(&mut hasher);
    let index = (hasher.finish() as usize) % PROJECT_LOCK_STRIPES;
    project_locks()[index].lock()
}

pub(crate) fn relative_string(root: &Path, path: &Path) -> ProjectStoreResult<String> {
    Ok(path
        .strip_prefix(root)
        .map_err(|_| ProjectStoreError::BadRequest("Path is outside project".to_owned()))?
        .to_string_lossy()
        .replace('\\', "/"))
}

pub(crate) fn is_safe_relative_path(relative_path: &str) -> bool {
    !relative_path.trim().is_empty()
        && !relative_path.contains('\\')
        && Path::new(relative_path)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

pub(crate) fn is_safe_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

pub(crate) fn optional_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

pub(crate) fn optional_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

pub(crate) fn optional_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

pub(crate) fn optional_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

pub(crate) fn random_hex(bytes: usize) -> ProjectStoreResult<String> {
    let connection = Connection::open_in_memory()?;
    Ok(connection.query_row(
        &format!("select lower(hex(randomblob({bytes})))"),
        [],
        |row| row.get(0),
    )?)
}

pub(crate) fn parse_string_enum<T>(value: &str) -> T
where
    T: DeserializeOwned,
{
    serde_json::from_value(Value::String(value.to_owned()))
        .expect("string enum deserialization is infallible")
}

#[cfg(test)]
mod tests {
    use super::{atomic_write, lock_project_files};

    #[test]
    fn atomic_write_tolerates_concurrent_writers_to_same_path() {
        // sc-1633: the temp file is uniquely named, so overlapping writers to the
        // same target never collide. With the old deterministic `*.tmp` name they
        // interleaved bytes and the second `rename` failed (the temp was already
        // renamed away) — which would surface here as an Err from `atomic_write`.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("doc.json");
        std::thread::scope(|scope| {
            for writer in 0..8 {
                let path = path.clone();
                scope.spawn(move || {
                    for _ in 0..50 {
                        atomic_write(&path, format!("{{\"writer\":{writer}}}").as_bytes())
                            .expect("atomic_write never collides on its temp file");
                    }
                });
            }
        });
        // The final file is exactly one writer's complete value, never torn.
        let contents = std::fs::read_to_string(&path).expect("file reads");
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).expect("final contents are valid json");
        assert!(parsed
            .get("writer")
            .and_then(|value| value.as_i64())
            .is_some());
    }

    #[test]
    fn lock_project_files_serializes_read_modify_write() {
        // sc-1633: holding the per-project lock across read -> mutate -> write makes
        // overlapping increments serialize. Without it the read/write window races
        // and the total comes up short of the expected count.
        let dir = tempfile::tempdir().expect("temp dir");
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).expect("project dir");
        let counter = project.join("counter.json");
        std::fs::write(&counter, "0").expect("counter seed");

        let threads = 8;
        let per_thread = 50;
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let project = project.clone();
                let counter = counter.clone();
                scope.spawn(move || {
                    for _ in 0..per_thread {
                        let _guard = lock_project_files(&project);
                        let current: i64 = std::fs::read_to_string(&counter)
                            .expect("counter reads")
                            .trim()
                            .parse()
                            .expect("counter parses");
                        atomic_write(&counter, (current + 1).to_string().as_bytes())
                            .expect("counter writes");
                    }
                });
            }
        });

        let total: i64 = std::fs::read_to_string(&counter)
            .expect("counter reads")
            .trim()
            .parse()
            .expect("counter parses");
        assert_eq!(total, (threads * per_thread) as i64);
    }
}
