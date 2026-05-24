use std::fs;
use std::path::Path;

use rusqlite::Connection;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::project_store::{ProjectStoreError, ProjectStoreResult};

pub(crate) fn read_json(path: &Path) -> ProjectStoreResult<Value> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

pub(crate) fn write_json<T: Serialize>(path: &Path, payload: &T) -> ProjectStoreResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = serde_json::to_string_pretty(payload)?;
    output.push('\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    fs::write(&tmp_path, output)?;
    fs::rename(tmp_path, path)?;
    Ok(())
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
