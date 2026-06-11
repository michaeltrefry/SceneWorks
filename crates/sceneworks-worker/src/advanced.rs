//! Shared accessors for a job's `advanced` knob map (the per-request `advanced`
//! JSON object).
//!
//! These were previously re-implemented per job module, and had drifted: image
//! and video both had a private `advanced_f32`, but only the image one clamped —
//! so a reader moving between the files would wrongly assume identical behavior
//! and could add an unclamped user knob believing it was range-protected
//! (sc-4281 / F-MLXW-18). Centralizing them here makes clamping **explicit at the
//! call site**: `f32`/`u32` read raw, `f32_clamped`/`u32_clamped` clamp to a
//! `RangeInclusive`. All accept the parsed `advanced` object directly so they are
//! reusable regardless of the request type that owns it.

use std::ops::RangeInclusive;

use sceneworks_core::contracts::JsonObject;
use serde_json::Value;

/// Permissive truthiness: a JSON `true`, a non-zero number, a non-empty string,
/// or a non-empty array all read as `true`; everything else (incl. absent) is
/// `false`. (Was image's `advanced_flag`.)
pub(crate) fn flag(advanced: &JsonObject, key: &str) -> bool {
    match advanced.get(key) {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(number)) => number.as_f64().map(|value| value != 0.0).unwrap_or(false),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        _ => false,
    }
}

/// Strict boolean: only a JSON `true`/`false`, default `false`. (Was video's
/// `advanced_bool`.) Use [`flag`] for the permissive truthiness reading.
pub(crate) fn bool(advanced: &JsonObject, key: &str) -> bool {
    advanced.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Float (JSON number or numeric string), default `default`, **no clamp**.
pub(crate) fn f32(advanced: &JsonObject, key: &str, default: f32) -> f32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(default)
}

/// Float, parsed like [`f32`] then clamped to `range`.
pub(crate) fn f32_clamped(
    advanced: &JsonObject,
    key: &str,
    default: f32,
    range: RangeInclusive<f32>,
) -> f32 {
    f32(advanced, key, default).clamp(*range.start(), *range.end())
}

/// Signed 32-bit int (JSON int or numeric string), default `default`, **no clamp**.
pub(crate) fn i32(advanced: &JsonObject, key: &str, default: i32) -> i32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as i32)
        .unwrap_or(default)
}

/// Unsigned 32-bit int (JSON uint or numeric string), default `default`, then
/// clamped to `range`.
pub(crate) fn u32_clamped(
    advanced: &JsonObject,
    key: &str,
    default: u32,
    range: RangeInclusive<u32>,
) -> u32 {
    advanced
        .get(key)
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as u32)
        .unwrap_or(default)
        .clamp(*range.start(), *range.end())
}

/// Trimmed non-empty string, else `default`.
pub(crate) fn str(advanced: &JsonObject, key: &str, default: &str) -> String {
    advanced
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(value: serde_json::Value) -> JsonObject {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn f32_clamped_vs_raw_make_the_clamp_explicit() {
        let advanced = obj(json!({ "x": 5.0, "s": "2.5" }));
        // Raw f32 does not clamp; the *_clamped variant does — the call site now
        // says which it wants (sc-4281).
        assert_eq!(f32(&advanced, "x", 0.0), 5.0);
        assert_eq!(f32_clamped(&advanced, "x", 0.0, 0.0..=1.0), 1.0);
        // Numeric strings parse for both.
        assert_eq!(f32(&advanced, "s", 0.0), 2.5);
        // Absent key falls back to default.
        assert_eq!(f32_clamped(&advanced, "missing", 0.3, 0.0..=1.0), 0.3);
    }

    #[test]
    fn flag_is_permissive_bool_is_strict() {
        let advanced = obj(json!({ "n": 2, "s": "hi", "arr": [1], "b": false }));
        assert!(flag(&advanced, "n"));
        assert!(flag(&advanced, "s"));
        assert!(flag(&advanced, "arr"));
        assert!(!flag(&advanced, "missing"));
        // Strict bool only honors a real JSON boolean.
        assert!(!bool(&advanced, "n"));
        assert!(!bool(&advanced, "b"));
        assert!(bool(&obj(json!({ "b": true })), "b"));
    }

    #[test]
    fn int_and_string_accessors_parse_and_default() {
        let advanced = obj(json!({ "i": -4, "u": "7", "name": "  z " }));
        assert_eq!(i32(&advanced, "i", 0), -4);
        assert_eq!(u32_clamped(&advanced, "u", 0, 0..=5), 5);
        assert_eq!(str(&advanced, "name", "d"), "z");
        assert_eq!(str(&advanced, "missing", "d"), "d");
    }
}
