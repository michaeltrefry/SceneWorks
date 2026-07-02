//! Shared payload-parsing helpers for the image- and video-generation job requests
//! (sc-8817). Both [`crate::image_request`] and [`crate::video_request`] parse a job
//! `payload` (a [`JsonObject`]) into a typed request, and both need the same small set
//! of "read this key, coerce it, fall back to a default" primitives.
//!
//! Historically these helpers were copy-pasted into each module and had already drifted:
//! `image_request::string_or` returned the raw string value, while
//! `video_request::string_or` trimmed and empty-filtered it (the semantics
//! `image_request` spelled `nonempty_string_or`). Collapsing them into one function would
//! silently change one lane's behavior, so this module deliberately exposes **two**
//! clearly-named string variants:
//!
//! * [`string_or`] — the RAW value (no trimming, a present-but-empty string stays empty).
//! * [`nonempty_string_or`] — trimmed, with a present-but-empty/whitespace value falling
//!   back to the default.
//!
//! Each existing call site is wired to the variant that preserves its prior behavior.

use serde_json::Value;

use crate::contracts::JsonObject;

/// Read a string key, returning the RAW value (no trimming); a present-but-empty string
/// is preserved. Falls back to `default` only when the key is absent or non-string.
pub(crate) fn string_or(payload: &JsonObject, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

/// Read a string key, trimmed, where a present-but-empty (or whitespace-only) value also
/// falls back to `default` (matches the Python `.get(key, default)` where the UI never
/// sends an empty model/mode/etc.).
pub(crate) fn nonempty_string_or(payload: &JsonObject, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

/// Read an optional string id: trimmed, `None` when absent, non-string, or blank.
pub(crate) fn optional_id(payload: &JsonObject, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Read an optional int that may arrive as a JSON number or a numeric string. `None`
/// when absent or unparseable.
pub(crate) fn optional_i64(payload: &JsonObject, key: &str) -> Option<i64> {
    payload.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })
}

/// Parse an int (JSON number or numeric string), clamp to `[min, max]`, default when
/// absent/unparseable — the `safe_int` contract. Takes the raw `Option<&Value>` so both
/// the `payload.get(key)` (image) and pre-fetched `payload.get("width")` (video) call
/// sites share one primitive.
pub(crate) fn clamped_u32(value: Option<&Value>, default: u32, min: u32, max: u32) -> u32 {
    value
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(default)
        .clamp(min, max)
}

/// Collect a JSON int array (numbers or numeric strings), dropping non-numeric entries.
/// Absent / non-array → empty.
pub(crate) fn int_array(payload: &JsonObject, key: &str) -> Vec<i64> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| {
                    value
                        .as_i64()
                        .or_else(|| value.as_str()?.trim().parse().ok())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Collect a JSON string array into trimmed, non-empty owned strings (blanks and
/// non-strings are dropped). Absent / non-array / all-blank → empty.
pub(crate) fn string_list(payload: &JsonObject, key: &str) -> Vec<String> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Clone a JSON array key verbatim; absent / non-array → empty.
pub(crate) fn array_or_empty(payload: &JsonObject, key: &str) -> Vec<Value> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Clone a JSON object key verbatim; absent / non-object → empty.
pub(crate) fn object_or_empty(payload: &JsonObject, key: &str) -> JsonObject {
    payload
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: Value) -> JsonObject {
        value.as_object().cloned().unwrap()
    }

    // --- Characterization tests capturing the CURRENT per-lane behavior (sc-8817). ---
    // The image lane historically used `string_or` = RAW and `nonempty_string_or` =
    // trim+empty-filter; the video lane's `string_or` was actually the trim+filter variant.
    // These lock the raw-vs-trim distinction so the dedup is provably behavior-preserving.

    #[test]
    fn string_or_returns_raw_value_including_present_but_empty() {
        // Present-but-empty stays empty (does NOT fall back to the default).
        let p = payload(json!({ "k": "" }));
        assert_eq!(string_or(&p, "k", "def"), "");
        // Leading/trailing whitespace is preserved verbatim.
        let p = payload(json!({ "k": "  spaced  " }));
        assert_eq!(string_or(&p, "k", "def"), "  spaced  ");
        // Absent / non-string → default.
        let p = payload(json!({ "n": 5 }));
        assert_eq!(string_or(&p, "k", "def"), "def");
        assert_eq!(string_or(&p, "n", "def"), "def");
    }

    #[test]
    fn nonempty_string_or_trims_and_empty_filters() {
        // Present-but-empty / whitespace-only → default.
        let p = payload(json!({ "k": "" }));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "def");
        let p = payload(json!({ "k": "   " }));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "def");
        // Non-blank values are trimmed.
        let p = payload(json!({ "k": "  spaced  " }));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "spaced");
        // Absent → default.
        let p = payload(json!({}));
        assert_eq!(nonempty_string_or(&p, "k", "def"), "def");
    }

    #[test]
    fn optional_id_trims_and_drops_blanks() {
        let p = payload(json!({ "id": "  x  ", "blank": "  ", "n": 3 }));
        assert_eq!(optional_id(&p, "id").as_deref(), Some("x"));
        assert!(optional_id(&p, "blank").is_none());
        assert!(optional_id(&p, "n").is_none());
        assert!(optional_id(&p, "missing").is_none());
    }

    #[test]
    fn optional_i64_reads_numbers_and_numeric_strings() {
        let p = payload(json!({ "a": 5, "b": "42", "c": "nope", "d": null }));
        assert_eq!(optional_i64(&p, "a"), Some(5));
        assert_eq!(optional_i64(&p, "b"), Some(42));
        assert_eq!(optional_i64(&p, "c"), None);
        assert_eq!(optional_i64(&p, "d"), None);
        assert_eq!(optional_i64(&p, "missing"), None);
    }

    #[test]
    fn clamped_u32_parses_and_clamps() {
        let p = payload(json!({ "a": 99, "b": "3", "c": "bad", "d": -1 }));
        assert_eq!(clamped_u32(p.get("a"), 4, 1, 8), 8);
        assert_eq!(clamped_u32(p.get("b"), 4, 1, 8), 3);
        // Unparseable / absent → default (then clamped).
        assert_eq!(clamped_u32(p.get("c"), 4, 1, 8), 4);
        assert_eq!(clamped_u32(None, 4, 1, 8), 4);
        // Negative can't become u32 → default.
        assert_eq!(clamped_u32(p.get("d"), 4, 1, 8), 4);
    }

    #[test]
    fn int_array_reads_numbers_and_numeric_strings_dropping_others() {
        let p = payload(json!({ "seeds": [1, "2", null, "bad", 3] }));
        assert_eq!(int_array(&p, "seeds"), vec![1, 2, 3]);
        let p = payload(json!({ "seeds": "notarray" }));
        assert!(int_array(&p, "seeds").is_empty());
        assert!(int_array(&payload(json!({})), "seeds").is_empty());
    }

    #[test]
    fn string_list_trims_and_drops_blanks_and_nonstrings() {
        let p = payload(json!({ "ids": ["a", "  ", "  b  ", 42, null] }));
        assert_eq!(string_list(&p, "ids"), vec!["a", "b"]);
        assert!(string_list(&payload(json!({ "ids": "x" })), "ids").is_empty());
        assert!(string_list(&payload(json!({})), "ids").is_empty());
    }

    #[test]
    fn array_and_object_or_empty_clone_or_default() {
        let p = payload(json!({ "arr": [1, 2], "obj": { "k": 1 }, "wrong": 5 }));
        assert_eq!(array_or_empty(&p, "arr"), vec![json!(1), json!(2)]);
        assert!(array_or_empty(&p, "wrong").is_empty());
        assert!(array_or_empty(&p, "missing").is_empty());
        assert_eq!(object_or_empty(&p, "obj").get("k"), Some(&json!(1)));
        assert!(object_or_empty(&p, "wrong").is_empty());
        assert!(object_or_empty(&p, "missing").is_empty());
    }
}
