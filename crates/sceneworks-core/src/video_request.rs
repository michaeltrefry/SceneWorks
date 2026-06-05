//! The video-generation job request (epic 3018, sc-3033).
//!
//! Parses a job `payload` into a typed request, mirroring the Python worker's
//! `video_request_from_job` (apps/worker/scene_worker/video_adapters.py) so the
//! native MLX Rust worker reads the same payload the UI already sends. The
//! `advanced` and `model_manifest_entry` maps pass through verbatim (they carry
//! per-model knobs like steps/guidanceScale/imageConditioningStrength and the
//! resolved manifest entry), so adding a model needs no DTO change.
//!
//! The advanced-mode asset ids (`last_frame` / `source_clip` / `bridge_right_clip`
//! / `person_track`) are parsed and carried so the asset fact + routing layer see
//! them, but the MLX video path (Wan / LTX text-or-image→video) does not consume
//! them — those modes (first_last_frame / extend_clip / video_bridge /
//! replace_person) stay on the Python torch worker (epic 3018 scope boundary;
//! the MLX-vs-Python routing decision is sc-3036).

use serde_json::Value;

use crate::contracts::JsonObject;

/// Defaults matching the Python `video_request_from_job` (`.get(key, default)`).
const DEFAULT_MODE: &str = "image_to_video";
const DEFAULT_MODEL: &str = "ltx_2_3";
const DEFAULT_QUALITY: &str = "balanced";
const DEFAULT_REPLACEMENT_MODE: &str = "face_only";

/// A typed video-generation request, parsed from a job payload. One job produces a
/// single video asset (unlike images, which batch `count`).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoRequest {
    pub project_id: String,
    pub mode: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub model: String,
    /// Clip length in seconds, clamped to 1.0..=30.0 (Python `safe_float`).
    pub duration: f32,
    /// Frames per second, clamped to 1..=60 (Python `safe_int`).
    pub fps: u32,
    /// Output dimensions: clamped to 256..=1920, then floored to a multiple of 32
    /// (Python `normalized_dimensions`). Defaults 768x512.
    pub width: u32,
    pub height: u32,
    pub quality: String,
    /// Base seed; `None` means derive deterministically from the prompt at run time.
    pub seed: Option<i64>,
    /// LoRA specs, passed through verbatim (shape resolved per family).
    pub loras: Vec<Value>,
    pub character_id: Option<String>,
    pub character_look_id: Option<String>,
    /// Image→video conditioning source (consumed by the MLX I2V path).
    pub source_asset_id: Option<String>,
    // --- Advanced-mode fields: parsed + carried, but Python-routed (sc-3036). ---
    pub person_track_id: Option<String>,
    pub replacement_mode: String,
    pub last_frame_asset_id: Option<String>,
    pub source_clip_asset_id: Option<String>,
    pub bridge_right_clip_asset_id: Option<String>,
    /// Per-model advanced knobs (steps, guidanceScale, imageConditioningStrength,
    /// timelineContext, …), passed through.
    pub advanced: JsonObject,
    /// The resolved builtin+user model manifest entry (repo/quant/limits/…).
    pub model_manifest_entry: JsonObject,
}

impl VideoRequest {
    /// Parse a job payload (the `payload` object of a `JobSnapshot`). Infallible:
    /// missing fields fall back to the Python defaults and `project_id` may be empty
    /// — the caller validates it is present (the worker rejects an empty project id).
    pub fn from_payload(payload: &JsonObject) -> Self {
        let (width, height) = normalized_dimensions(payload.get("width"), payload.get("height"));
        Self {
            project_id: string_or(payload, "projectId", ""),
            mode: string_or(payload, "mode", DEFAULT_MODE),
            prompt: string_or(payload, "prompt", ""),
            negative_prompt: string_or(payload, "negativePrompt", ""),
            model: string_or(payload, "model", DEFAULT_MODEL),
            duration: safe_float(payload.get("duration"), 6.0, 1.0, 30.0),
            fps: safe_int(payload.get("fps"), 25, 1, 60),
            width,
            height,
            quality: string_or(payload, "quality", DEFAULT_QUALITY),
            seed: optional_i64(payload, "seed"),
            loras: array_or_empty(payload, "loras"),
            character_id: optional_id(payload, "characterId"),
            character_look_id: optional_id(payload, "characterLookId"),
            source_asset_id: optional_id(payload, "sourceAssetId"),
            person_track_id: optional_id(payload, "personTrackId"),
            replacement_mode: string_or(payload, "replacementMode", DEFAULT_REPLACEMENT_MODE),
            last_frame_asset_id: optional_id(payload, "lastFrameAssetId"),
            source_clip_asset_id: optional_id(payload, "sourceClipAssetId"),
            bridge_right_clip_asset_id: optional_id(payload, "bridgeRightClipAssetId"),
            advanced: object_or_empty(payload, "advanced"),
            model_manifest_entry: object_or_empty(payload, "modelManifestEntry"),
        }
    }

    /// Raw frame count from `duration * fps` (Python `round(duration * fps)`), at
    /// least 1. The per-model snap ([`frame_count`](Self::frame_count)) rounds this
    /// to the model's temporal stride.
    pub fn raw_frame_count(&self) -> u32 {
        ((self.duration * self.fps as f32).round() as i64).max(1) as u32
    }

    /// The frame count the model will actually produce: the raw `duration * fps`
    /// snapped to the model's temporal constraint (LTX `8k + 1`, Wan `4n + 1`).
    /// Unknown / stub models keep the raw count. The engine's `validate()` is
    /// authoritative for the real models (sc-3034 / sc-3035); this mirrors the
    /// Python worker's pre-snap so the stub clip length and the UI estimate agree.
    pub fn frame_count(&self) -> u32 {
        let raw = self.raw_frame_count();
        if is_ltx_model(&self.model) {
            ltx_frame_count(raw)
        } else if is_wan_model(&self.model) {
            wan_frame_count(raw)
        } else {
            raw
        }
    }
}

/// LTX-2.3 temporal stride: frames snap to the nearest `8k + 1`, minimum 9. Direct
/// port of the Python `ltx_frame_count`.
pub fn ltx_frame_count(raw_frames: u32) -> u32 {
    let frame_count = raw_frames.max(9);
    let lower = frame_count - ((frame_count - 1) % 8);
    let upper = lower + 8;
    if lower < 9 {
        return upper;
    }
    let lower_delta = frame_count - lower;
    let upper_delta = upper - frame_count;
    if lower_delta <= upper_delta {
        lower
    } else {
        upper
    }
}

/// Wan2.2 temporal stride: the VAE math `t_lat = (frames − 1) / 4 + 1` requires
/// `frames ≡ 1 (mod 4)`. Frames floor to the largest `4k + 1` at or below `raw`,
/// with a 5-frame minimum — the Python worker's `_run_wan_mlx`
/// `max(5, raw - ((raw - 1) % 4))`.
pub fn wan_frame_count(raw_frames: u32) -> u32 {
    let raw = raw_frames.max(1);
    (raw - ((raw - 1) % 4)).max(5)
}

/// Whether `model` is an LTX-2.3 family id (`ltx_2_3`, `ltx_2_3_eros`, …).
pub fn is_ltx_model(model: &str) -> bool {
    model.starts_with("ltx")
}

/// Whether `model` is a Wan2.2 family id (`wan_2_2`, `wan_2_2_t2v_14b`, …).
pub fn is_wan_model(model: &str) -> bool {
    model.starts_with("wan")
}

/// Clamp + floor dimensions exactly like the Python `normalized_dimensions`: parse
/// (default 768x512), clamp to 256..=1920, floor to a multiple of 32, floor of 256.
fn normalized_dimensions(width: Option<&Value>, height: Option<&Value>) -> (u32, u32) {
    let w = floor_to_32(safe_int(width, 768, 256, 1920));
    let h = floor_to_32(safe_int(height, 512, 256, 1920));
    (w, h)
}

fn floor_to_32(value: u32) -> u32 {
    (value - (value % 32)).max(256)
}

fn string_or(payload: &JsonObject, key: &str, default: &str) -> String {
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

fn optional_i64(payload: &JsonObject, key: &str) -> Option<i64> {
    payload.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    })
}

/// Parse an int (JSON number or numeric string), clamp to `[min, max]`, default
/// when absent/unparseable — the `safe_int` contract.
fn safe_int(value: Option<&Value>, default: u32, min: u32, max: u32) -> u32 {
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

/// Parse a float (JSON number or numeric string), clamp to `[min, max]`, default
/// when absent/unparseable — the `safe_float` contract.
fn safe_float(value: Option<&Value>, default: f32, min: f32, max: f32) -> f32 {
    value
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
        .unwrap_or(default)
        .clamp(min, max)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(value: Value) -> JsonObject {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn defaults_when_payload_is_minimal() {
        let request = VideoRequest::from_payload(&payload(json!({ "projectId": "proj_1" })));
        assert_eq!(request.project_id, "proj_1");
        assert_eq!(request.mode, "image_to_video");
        assert_eq!(request.model, "ltx_2_3");
        assert_eq!(request.duration, 6.0);
        assert_eq!(request.fps, 25);
        assert_eq!(request.width, 768);
        assert_eq!(request.height, 512);
        assert_eq!(request.quality, "balanced");
        assert_eq!(request.replacement_mode, "face_only");
        assert!(request.seed.is_none());
        assert!(request.loras.is_empty());
        assert!(request.advanced.is_empty());
        assert!(request.source_asset_id.is_none());
    }

    #[test]
    fn clamps_duration_fps_and_dimensions() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "duration": 99, "fps": 999, "width": 99999, "height": 100
        })));
        assert_eq!(request.duration, 30.0);
        assert_eq!(request.fps, 60);
        // 1920 is already a multiple of 32; 100 clamps to 256.
        assert_eq!(request.width, 1920);
        assert_eq!(request.height, 256);

        let low = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "duration": 0, "fps": 0
        })));
        assert_eq!(low.duration, 1.0);
        assert_eq!(low.fps, 1);
    }

    #[test]
    fn floors_dimensions_to_multiple_of_32() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "width": 1000, "height": 543
        })));
        // 1000 -> 992 (31*32), 543 -> 512 (16*32).
        assert_eq!(request.width, 992);
        assert_eq!(request.height, 512);
    }

    #[test]
    fn reads_numeric_strings_and_carries_advanced_fields() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p",
            "duration": "4.5",
            "fps": "30",
            "seed": "42",
            "sourceAssetId": "asset_src",
            "personTrackId": "track_1",
            "lastFrameAssetId": "asset_last",
            "advanced": { "guidanceScale": 4.0, "steps": 20 },
            "modelManifestEntry": { "family": "ltx-video", "repo": "x" },
            "loras": [{ "path": "a.safetensors", "weight": 0.8 }]
        })));
        assert_eq!(request.duration, 4.5);
        assert_eq!(request.fps, 30);
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.source_asset_id.as_deref(), Some("asset_src"));
        assert_eq!(request.person_track_id.as_deref(), Some("track_1"));
        assert_eq!(request.last_frame_asset_id.as_deref(), Some("asset_last"));
        assert_eq!(request.advanced.get("steps"), Some(&json!(20)));
        assert_eq!(
            request.model_manifest_entry.get("family"),
            Some(&json!("ltx-video"))
        );
        assert_eq!(request.loras.len(), 1);
    }

    #[test]
    fn raw_frame_count_rounds_duration_times_fps() {
        let request = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "stub", "duration": 2.0, "fps": 24
        })));
        assert_eq!(request.raw_frame_count(), 48);
        // Unknown model keeps the raw count.
        assert_eq!(request.frame_count(), 48);
    }

    #[test]
    fn ltx_frame_count_snaps_to_8k_plus_1() {
        // Exact 8k+1 values are unchanged.
        assert_eq!(ltx_frame_count(9), 9);
        assert_eq!(ltx_frame_count(17), 17);
        assert_eq!(ltx_frame_count(121), 121);
        // Below the floor snaps up to 9.
        assert_eq!(ltx_frame_count(1), 9);
        assert_eq!(ltx_frame_count(8), 9);
        // Nearest wins; ties go to the lower 8k+1.
        assert_eq!(ltx_frame_count(13), 9); // |13-9|=4 == |17-13|=4 -> lower
        assert_eq!(ltx_frame_count(14), 17); // closer to 17
                                             // A real default: 6s * 25fps = 150 -> nearest of 145 / 153 -> 153.
        assert_eq!(ltx_frame_count(150), 153);
    }

    #[test]
    fn wan_frame_count_floors_to_4n_plus_1_min_5() {
        // Exact 1+4k values >= 5 are unchanged.
        assert_eq!(wan_frame_count(5), 5);
        assert_eq!(wan_frame_count(81), 81);
        // Floors to the 1+4k at or below raw (Python `raw - ((raw-1)%4)`).
        assert_eq!(wan_frame_count(48), 45); // 48-(47%4=3)
        assert_eq!(wan_frame_count(83), 81); // 83-(82%4=2)
        assert_eq!(wan_frame_count(8), 5); // 8-(7%4=3)
                                           // 5-frame floor for tiny counts.
        assert_eq!(wan_frame_count(1), 5);
        assert_eq!(wan_frame_count(4), 5);
    }

    #[test]
    fn frame_count_dispatches_by_model_family() {
        let ltx = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "ltx_2_3", "duration": 6.0, "fps": 25
        })));
        assert_eq!(ltx.frame_count(), ltx_frame_count(150));

        let wan = VideoRequest::from_payload(&payload(json!({
            "projectId": "p", "model": "wan_2_2_t2v_14b", "duration": 3.0, "fps": 16
        })));
        assert_eq!(wan.frame_count(), wan_frame_count(48));

        assert!(is_ltx_model("ltx_2_3_eros"));
        assert!(is_wan_model("wan_2_2"));
        assert!(!is_ltx_model("wan_2_2"));
    }
}
