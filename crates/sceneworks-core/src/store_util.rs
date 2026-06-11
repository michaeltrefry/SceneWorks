use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::OnceLock;

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use serde::Serialize;
use serde_json::Value;

use crate::contracts::StringEnum;

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

/// Generate a random lowercase-hex id of `bytes` random bytes (so `2 * bytes`
/// hex chars). Previously this ran `hex(randomblob(n))` through a fresh
/// in-memory SQLite connection per call, so saving a timeline with N new items
/// opened N connections and turned SQLite-init failures into id-generation
/// failures (sc-4209 / F-CORE-5). Now it pulls from the OS CSPRNG via
/// `getrandom` and hex-encodes in Rust — no connection, no SQLite failure
/// surface.
pub(crate) fn random_hex(bytes: usize) -> ProjectStoreResult<String> {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut buffer = vec![0u8; bytes];
    getrandom::fill(&mut buffer)
        .map_err(|error| ProjectStoreError::Io(std::io::Error::other(error.to_string())))?;
    let mut hex = String::with_capacity(bytes * 2);
    for byte in buffer {
        hex.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        hex.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    Ok(hex)
}

/// Deserialize a single string into one of the crate's [`StringEnum`] types.
///
/// The [`StringEnum`] bound (not a bare `DeserializeOwned`) is what makes the
/// `expect` sound: every `string_enum!`-generated type maps an unrecognized
/// string to its `Unknown(String)` variant, so deserializing a `String` value
/// can't fail. The bound stops this helper from being reused with a stricter
/// enum whose `Deserialize` could error and panic the whole store call on one
/// bad DB row (sc-4269 / F-CORE-9).
pub(crate) fn parse_string_enum<T>(value: &str) -> T
where
    T: StringEnum,
{
    serde_json::from_value(Value::String(value.to_owned()))
        .expect("string enum deserialization is infallible")
}

#[cfg(test)]
mod tests {
    use super::{atomic_write, lock_project_files, parse_string_enum, random_hex};
    use crate::contracts::JobStatus;
    use std::collections::HashSet;

    /// sc-4269 / F-CORE-9: the `StringEnum` bound makes `parse_string_enum`
    /// infallible — a value never written by the app (or a hand-edited DB row)
    /// maps to the enum's `Unknown` variant instead of panicking, and known
    /// values still map to their variant.
    #[test]
    fn parse_string_enum_maps_unknown_without_panicking() {
        assert_eq!(
            parse_string_enum::<JobStatus>("totally-bogus-status"),
            JobStatus::Unknown("totally-bogus-status".to_owned())
        );
        assert_eq!(
            parse_string_enum::<JobStatus>("running"),
            JobStatus::Running
        );
    }

    /// sc-4209 / F-CORE-5: `random_hex(n)` returns `2n` lowercase-hex chars,
    /// generated without a SQLite connection. Covers the length/charset contract
    /// and that successive ids differ (i.e. it is actually random, not constant).
    #[test]
    fn random_hex_produces_lowercase_hex_of_expected_length() {
        for bytes in [1usize, 4, 16] {
            let id = random_hex(bytes).expect("id generates");
            assert_eq!(
                id.len(),
                bytes * 2,
                "{bytes} bytes -> {} hex chars",
                bytes * 2
            );
            assert!(
                id.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "id {id:?} is lowercase hex"
            );
        }

        // Distinct across many calls — a constant/low-entropy generator would collide.
        let ids = (0..1000)
            .map(|_| random_hex(16).expect("id generates"))
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), 1000, "16-byte ids are unique across 1000 draws");
    }

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
