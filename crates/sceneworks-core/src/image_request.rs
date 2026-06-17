//! The image-generation job request (epic 3018, sc-3020).
//!
//! Parses a job `payload` into a typed request, mirroring the Python worker's
//! `image_request_from_job` (apps/worker/scene_worker/image_adapters.py) so the
//! native MLX Rust worker reads the same payload the UI already sends. The `advanced`
//! and `model_manifest_entry` maps pass through verbatim (they carry per-family knobs
//! like steps/guidanceScale/mlxQuantize/poses/angleSet/controlScale/referenceStrength
//! and the resolved model manifest entry), so adding a family needs no DTO change.

use serde_json::Value;

use crate::contracts::JsonObject;

/// Default model when the payload omits one (matches the Python worker).
const DEFAULT_MODEL: &str = "z_image_turbo";
const DEFAULT_MODE: &str = "text_to_image";
const DEFAULT_STYLE_PRESET: &str = "cinematic";
/// Default fit mode (epic 2551): never distort, cover the frame. Shared with the
/// video request (sc-6139) so image- and video-conditioned sources normalize identically.
pub(crate) const DEFAULT_FIT_MODE: &str = "crop";
pub(crate) const FIT_MODES: [&str; 4] = ["crop", "pad", "outpaint", "stretch"];

/// A typed image-generation request, parsed from a job payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRequest {
    pub project_id: String,
    pub mode: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub model: String,
    /// Number of images, clamped to 1..=8.
    pub count: u32,
    /// Base seed (per-image seeds in `seeds` take precedence).
    pub seed: Option<i64>,
    /// Explicit per-image seeds.
    pub seeds: Vec<i64>,
    /// Output dimensions, clamped to 256..=4096.
    pub width: u32,
    pub height: u32,
    pub style_preset: String,
    /// LoRA specs, passed through verbatim (shape resolved per family).
    pub loras: Vec<Value>,
    pub character_id: Option<String>,
    pub character_look_id: Option<String>,
    pub source_asset_id: Option<String>,
    pub reference_asset_id: Option<String>,
    /// Multiple reference images for a multi-reference edit (sc-6211). Plural companion to the
    /// singular `reference_asset_id`; the FLUX.2-dev multi-image Image-Studio picker sends these.
    /// Empty for every single-reference / character flow (which keeps using `reference_asset_id`).
    pub reference_asset_ids: Vec<String>,
    pub mask_asset_id: Option<String>,
    /// One of crop/pad/outpaint/stretch (default crop).
    pub fit_mode: String,
    /// The resolved model manifest entry (repo/quant/limits/…), passed through.
    pub model_manifest_entry: JsonObject,
    /// Per-family advanced knobs (steps, guidanceScale, mlxQuantize, …), passed through.
    pub advanced: JsonObject,
}

impl ImageRequest {
    /// Parse a job payload (the `payload` object of a `JobSnapshot`). Infallible:
    /// missing fields fall back to the Python defaults and `project_id` may be empty
    /// — the caller validates it is present (the worker rejects an empty project id).
    pub fn from_payload(payload: &JsonObject) -> Self {
        Self {
            project_id: string_or(payload, "projectId", ""),
            mode: nonempty_string_or(payload, "mode", DEFAULT_MODE),
            prompt: string_or(payload, "prompt", ""),
            negative_prompt: string_or(payload, "negativePrompt", ""),
            model: nonempty_string_or(payload, "model", DEFAULT_MODEL),
            count: clamped_u32(payload, "count", 4, 1, 8),
            seed: optional_i64(payload, "seed"),
            seeds: int_array(payload, "seeds"),
            width: clamped_u32(payload, "width", 1024, 256, 4096),
            height: clamped_u32(payload, "height", 1024, 256, 4096),
            style_preset: nonempty_string_or(payload, "stylePreset", DEFAULT_STYLE_PRESET),
            loras: array_or_empty(payload, "loras"),
            character_id: optional_id(payload, "characterId"),
            character_look_id: optional_id(payload, "characterLookId"),
            source_asset_id: optional_id(payload, "sourceAssetId"),
            reference_asset_id: optional_id(payload, "referenceAssetId"),
            reference_asset_ids: string_list(payload, "referenceAssetIds"),
            mask_asset_id: optional_id(payload, "maskAssetId"),
            fit_mode: normalize_fit_mode(payload.get("fitMode").and_then(Value::as_str)),
            model_manifest_entry: object_or_empty(payload, "modelManifestEntry"),
            advanced: object_or_empty(payload, "advanced"),
        }
    }

    /// The resolved seed for image `index`: an explicit per-image seed wins, else the
    /// base seed offset by the index (so a multi-image batch from one seed differs),
    /// else `None` (the generator picks a random seed and records it).
    pub fn seed_for(&self, index: usize) -> Option<i64> {
        if let Some(seed) = self.seeds.get(index) {
            return Some(*seed);
        }
        self.seed.map(|base| base.wrapping_add(index as i64))
    }
}

fn string_or(payload: &JsonObject, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

/// Like `string_or` but a present-but-empty value also falls back to the default
/// (matches the Python `.get(key, default)` where the UI never sends an empty model).
fn nonempty_string_or(payload: &JsonObject, key: &str, default: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_owned()
}

fn optional_id(payload: &JsonObject, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Read an int that may arrive as a JSON number or a numeric string, clamp to
/// `[min, max]`, default when absent/unparseable.
fn clamped_u32(payload: &JsonObject, key: &str, default: u32, min: u32, max: u32) -> u32 {
    payload
        .get(key)
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn optional_i64(payload: &JsonObject, key: &str) -> Option<i64> {
    payload.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })
}

fn int_array(payload: &JsonObject, key: &str) -> Vec<i64> {
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

/// Collect a JSON string array into trimmed, non-empty owned strings (sc-6211, mirrors the video
/// request's `referenceAssetIds` parsing). Absent / non-array / all-blank → empty.
fn string_list(payload: &JsonObject, key: &str) -> Vec<String> {
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

fn array_or_empty(payload: &JsonObject, key: &str) -> Vec<Value> {
    payload
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn object_or_empty(payload: &JsonObject, key: &str) -> JsonObject {
    payload
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn normalize_fit_mode(value: Option<&str>) -> String {
    let normalized = value
        .unwrap_or(DEFAULT_FIT_MODE)
        .trim()
        .to_ascii_lowercase();
    if FIT_MODES.contains(&normalized.as_str()) {
        normalized
    } else {
        DEFAULT_FIT_MODE.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: Value) -> JsonObject {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn defaults_when_payload_is_minimal() {
        let request = ImageRequest::from_payload(&payload(json!({ "projectId": "proj_1" })));
        assert_eq!(request.project_id, "proj_1");
        assert_eq!(request.mode, "text_to_image");
        assert_eq!(request.model, "z_image_turbo");
        assert_eq!(request.count, 4);
        assert_eq!(request.width, 1024);
        assert_eq!(request.height, 1024);
        assert_eq!(request.style_preset, "cinematic");
        assert_eq!(request.fit_mode, "crop");
        assert_eq!(request.prompt, "");
        assert!(request.seed.is_none());
        assert!(request.seeds.is_empty());
        assert!(request.loras.is_empty());
        assert!(request.advanced.is_empty());
    }

    #[test]
    fn clamps_count_and_dimensions() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "count": 99, "width": 10, "height": 99999
        })));
        assert_eq!(request.count, 8);
        assert_eq!(request.width, 256);
        assert_eq!(request.height, 4096);

        let low = ImageRequest::from_payload(&payload(json!({ "projectId": "p", "count": 0 })));
        assert_eq!(low.count, 1);
    }

    #[test]
    fn reads_numeric_strings_and_seeds() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "count": "3", "seed": "42", "seeds": [1, "2", null, 3]
        })));
        assert_eq!(request.count, 3);
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.seeds, vec![1, 2, 3]);
    }

    #[test]
    fn seed_for_prefers_explicit_then_offsets_base() {
        let explicit = ImageRequest::from_payload(&payload(json!({
            "projectId": "p", "seed": 100, "seeds": [7, 8]
        })));
        assert_eq!(explicit.seed_for(0), Some(7));
        assert_eq!(explicit.seed_for(1), Some(8));
        // No explicit seed at index 2 -> base + index.
        assert_eq!(explicit.seed_for(2), Some(102));

        let none = ImageRequest::from_payload(&payload(json!({ "projectId": "p" })));
        assert_eq!(none.seed_for(0), None);
    }

    #[test]
    fn normalizes_fit_mode_and_passes_through_maps() {
        let request = ImageRequest::from_payload(&payload(json!({
            "projectId": "p",
            "fitMode": "OUTPAINT",
            "advanced": { "steps": 8, "mlxQuantize": 8 },
            "modelManifestEntry": { "family": "z-image", "repo": "x" },
            "loras": [{ "path": "a.safetensors", "weight": 0.8 }]
        })));
        assert_eq!(request.fit_mode, "outpaint");
        assert_eq!(request.advanced.get("steps"), Some(&json!(8)));
        assert_eq!(
            request.model_manifest_entry.get("family"),
            Some(&json!("z-image"))
        );
        assert_eq!(request.loras.len(), 1);

        let bogus =
            ImageRequest::from_payload(&payload(json!({ "projectId": "p", "fitMode": "weird" })));
        assert_eq!(bogus.fit_mode, "crop");
    }
}
