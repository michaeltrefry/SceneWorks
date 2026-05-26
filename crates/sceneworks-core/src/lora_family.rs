//! Detects the architecture family of a LoRA file from its safetensors header.
//!
//! The detector inspects the tensor key names parsed from a safetensors
//! header and matches them against a table of architecture signatures.
//! It is deliberately conservative: it only returns a family when the
//! evidence is strong and unambiguous. Callers should treat `None` as
//! "we cannot prove the user is wrong" and accept a user-supplied family,
//! while a `Some(family)` that disagrees with a user-supplied family is
//! grounds to reject the import.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

/// Maximum allowed safetensors header size, in bytes. Matches the
/// pre-existing 16 MiB cap enforced by the rust-api.
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// Errors returned when reading a safetensors header from disk.
#[derive(Debug)]
pub enum SafetensorsHeaderError {
    /// The header bytes could not be read from disk.
    Io(std::io::Error),
    /// The file did not contain a valid safetensors header (too short,
    /// implausible length, or non-JSON contents).
    InvalidHeader,
}

impl std::fmt::Display for SafetensorsHeaderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidHeader => formatter.write_str("invalid safetensors header"),
        }
    }
}

impl std::error::Error for SafetensorsHeaderError {}

/// Reads and JSON-decodes the safetensors header from `path`. The file
/// layout is: 8-byte little-endian header length, then the JSON header,
/// then tensor data — only the header is read.
pub fn read_safetensors_header(path: &Path) -> Result<Value, SafetensorsHeaderError> {
    let metadata = fs::metadata(path).map_err(SafetensorsHeaderError::Io)?;
    if metadata.len() < 8 {
        return Err(SafetensorsHeaderError::InvalidHeader);
    }
    let mut file = fs::File::open(path).map_err(SafetensorsHeaderError::Io)?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len + 8 > metadata.len() {
        return Err(SafetensorsHeaderError::InvalidHeader);
    }
    let header_len_usize =
        usize::try_from(header_len).map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    let mut header = vec![0_u8; header_len_usize];
    file.read_exact(&mut header)
        .map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    serde_json::from_slice::<Value>(&header).map_err(|_| SafetensorsHeaderError::InvalidHeader)
}

/// Returns the first `.safetensors` file at or below `path`. When `path`
/// itself is a `.safetensors` file it is returned directly. Returns `None`
/// when no file is found or `path` is neither a file nor a directory.
pub fn first_safetensors_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() && has_safetensors_extension(path) {
        return Some(path.to_path_buf());
    }
    if !path.is_dir() {
        return None;
    }
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = fs::read_dir(current).ok()?;
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                stack.push(entry_path);
            } else if has_safetensors_extension(&entry_path) {
                return Some(entry_path);
            }
        }
    }
    None
}

fn has_safetensors_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
}

/// Returns the detected family for a base model directory or file.
///
/// Detection strategy, in priority order:
/// 1. Diffusers `model_index.json` `_class_name` — canonical for diffusers
///    snapshots and the most reliable signal.
/// 2. Safetensors header architecture detection via [`detect_lora_family`] —
///    the architecture-prefix substrings (e.g. `transformer.transformer_blocks.`)
///    appear in base models as well as LoRAs, so the same detector usefully
///    classifies single-file diffusers checkpoints.
///
/// Returns `Ok(None)` when no confident signal is available — callers should
/// treat that as "unassociated" rather than as an error.
pub fn detect_model_family(path: &Path) -> Result<Option<String>, SafetensorsHeaderError> {
    if path.is_dir() {
        if let Some(family) = read_diffusers_model_index_family(path) {
            return Ok(Some(family));
        }
    } else if path.is_file()
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("model_index.json"))
    {
        if let Some(parent) = path.parent() {
            if let Some(family) = read_diffusers_model_index_family(parent) {
                return Ok(Some(family));
            }
        }
    }
    let Some(safetensors_path) = first_safetensors_path(path) else {
        return Ok(None);
    };
    let header = read_safetensors_header(&safetensors_path)?;
    Ok(detect_lora_family(&header))
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FamilyMismatch {
    pub supplied: String,
    pub detected: String,
}

/// Applies the shared import policy for detected architecture families.
///
/// A confident detector result rejects a conflicting user-supplied family; a
/// missing detector result keeps the supplied family, if any.
pub fn reconcile_detected_family(
    supplied: Option<String>,
    detected: Option<String>,
) -> Result<Option<String>, FamilyMismatch> {
    match (supplied, detected) {
        (Some(supplied), Some(detected)) if supplied != detected => {
            Err(FamilyMismatch { supplied, detected })
        }
        (Some(supplied), Some(_)) => Ok(Some(supplied)),
        (None, Some(detected)) => Ok(Some(detected)),
        (Some(supplied), None) => Ok(Some(supplied)),
        (None, None) => Ok(None),
    }
}

/// Adds model-manifest defaults derived from the imported model type/family.
/// Existing author-supplied fields are preserved.
pub fn apply_model_manifest_defaults(
    entry: &mut Map<String, Value>,
    model_type: &str,
    family: Option<&str>,
) {
    entry
        .entry("downloads".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    entry
        .entry("defaults".to_owned())
        .or_insert_with(|| json!({}));
    entry
        .entry("limits".to_owned())
        .or_insert_with(|| json!({}));
    entry.entry("ui".to_owned()).or_insert_with(|| json!({}));

    let Some(family) = family
        .map(normalize_model_family)
        .filter(|value| !value.is_empty())
    else {
        entry
            .entry("loraCompatibility".to_owned())
            .or_insert_with(|| json!({}));
        return;
    };

    if let Some(adapter) = model_adapter_for_family(&family) {
        entry
            .entry("adapter".to_owned())
            .or_insert_with(|| Value::String(adapter.to_owned()));
    }

    let capabilities = model_capabilities_for_type_and_family(model_type, &family);
    if !capabilities.is_empty() {
        entry.entry("capabilities".to_owned()).or_insert_with(|| {
            Value::Array(
                capabilities
                    .into_iter()
                    .map(|capability| Value::String(capability.to_owned()))
                    .collect(),
            )
        });
    }

    let compatibility = entry
        .entry("loraCompatibility".to_owned())
        .or_insert_with(|| json!({}));
    if let Some(object) = compatibility.as_object_mut() {
        object
            .entry("families".to_owned())
            .or_insert_with(|| json!([family]));
    }
}

pub fn model_adapter_for_family(family: &str) -> Option<&'static str> {
    match normalize_model_family(family).as_str() {
        "z-image" => Some("z_image_diffusers"),
        "qwen-image" => Some("qwen_image"),
        "lens" => Some("lens_turbo"),
        "sensenova-u1" => Some("sensenova_u1"),
        "flux" => Some("flux_diffusers"),
        "ltx-video" => Some("ltx_video"),
        "wan-video" => Some("wan_video"),
        _ => None,
    }
}

pub fn model_capabilities_for_type_and_family(model_type: &str, family: &str) -> Vec<&'static str> {
    match (
        model_type.trim().to_ascii_lowercase().as_str(),
        normalize_model_family(family).as_str(),
    ) {
        ("image", "z-image") => vec!["text_to_image", "character_image", "style_variations"],
        ("image", "qwen-image") => vec!["text_to_image", "style_variations"],
        ("image", "lens") => vec!["text_to_image", "style_variations"],
        ("image", "sensenova-u1") => vec!["text_to_image", "edit_image", "vqa", "interleave"],
        ("image", "flux") => vec!["text_to_image", "style_variations"],
        ("video", "ltx-video") => vec![
            "image_to_video",
            "text_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
        ],
        ("video", "wan-video") => vec![
            "image_to_video",
            "text_to_video",
            "first_last_frame",
            "extend_clip",
            "video_bridge",
            "replace_person",
        ],
        _ => Vec::new(),
    }
}

fn normalize_model_family(family: &str) -> String {
    family.trim().to_ascii_lowercase().replace('_', "-")
}

fn read_diffusers_model_index_family(dir: &Path) -> Option<String> {
    let index_path = dir.join("model_index.json");
    let bytes = fs::read(&index_path).ok()?;
    let index: Value = serde_json::from_slice(&bytes).ok()?;
    let class_name = index.get("_class_name").and_then(Value::as_str)?;
    diffusers_class_name_to_family(class_name)
}

/// Maps a diffusers pipeline `_class_name` to a SceneWorks family. The map is
/// intentionally conservative — only return a family when the mapping is
/// unambiguous, and otherwise let the caller treat the model as unassociated.
pub fn diffusers_class_name_to_family(class_name: &str) -> Option<String> {
    let normalized = class_name.trim();
    let lower = normalized.to_ascii_lowercase();
    match lower.as_str() {
        "zimagepipeline"
        | "zimageimg2imgpipeline"
        | "zimageturbopipeline"
        | "zimageturboimg2imgpipeline" => Some("z-image".to_owned()),
        "qwenimagepipeline" | "qwenimageimg2imgpipeline" | "qwenimageeditpipeline" => {
            Some("qwen-image".to_owned())
        }
        "lenspipeline" => Some("lens".to_owned()),
        "fluxpipeline" | "fluximg2imgpipeline" | "fluxinpaintpipeline" => Some("flux".to_owned()),
        "wanpipeline" | "wani2vpipeline" | "wantext2videopipeline" => Some("wan-video".to_owned()),
        "ltxpipeline" | "ltxvideopipeline" | "ltximagetovideopipeline" => {
            Some("ltx-video".to_owned())
        }
        "stablediffusionpipeline" | "stablediffusionimg2imgpipeline" => Some("sd1.5".to_owned()),
        "stablediffusionxlpipeline" | "stablediffusionxlimg2imgpipeline" => Some("sdxl".to_owned()),
        _ => None,
    }
}

/// Returns the detected LoRA architecture family or `None` if the header
/// is ambiguous, empty, or matches no known signature with confidence.
pub fn detect_lora_family(header: &Value) -> Option<String> {
    if let Some(family) = detect_metadata_family(header) {
        return Some(family);
    }
    let keys = collect_tensor_keys(header);
    if keys.is_empty() {
        return None;
    }
    let bucket = detect_bucket(&keys)?;
    match bucket {
        Bucket::WanVideo => Some("wan-video".to_owned()),
        Bucket::Flux => Some("flux".to_owned()),
        Bucket::LtxVideo => Some("ltx-video".to_owned()),
        Bucket::Sdxl => Some("sdxl".to_owned()),
        Bucket::Sd15 => Some("sd1.5".to_owned()),
        Bucket::MmDit => disambiguate_mm_dit(&keys),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    WanVideo,
    Flux,
    LtxVideo,
    MmDit,
    Sdxl,
    Sd15,
}

struct BucketSignature {
    bucket: Bucket,
    /// Every group in this list must be satisfied: at least one tensor key
    /// must contain at least one substring from the inner slice. A signature
    /// with any unmet group does not apply (score = 0). Use this to encode
    /// AND-of-OR conjunctions like "must have both `lora_te1_` and
    /// `lora_te2_` keys" for SDXL.
    require_all_of: &'static [&'static [&'static str]],
    /// If any substring here is present in any tensor key, the signature is
    /// disqualified regardless of marker score.
    disqualifiers: &'static [&'static str],
    /// Substrings that count toward the score when present in a tensor key.
    markers: &'static [&'static str],
}

const SIGNATURES: &[BucketSignature] = &[
    BucketSignature {
        bucket: Bucket::Flux,
        // Flux is the only architecture that ships both double-stream and
        // single-stream transformer blocks together. The single-block prefix
        // alone is enough to identify it. Kohya-style Flux LoRAs flatten the
        // same names into `lora_unet_single_blocks_...` tensor keys.
        require_all_of: &[&["single_transformer_blocks.", "single_blocks_"]],
        disqualifiers: &[],
        markers: &[
            "single_transformer_blocks.",
            "single_blocks_",
            "double_blocks.",
            "double_blocks_",
            "transformer_blocks.",
        ],
    },
    BucketSignature {
        bucket: Bucket::Flux,
        // XLabs / x-flux Flux LoRAs adapt only the double-stream blocks through an
        // attention-processor layout (`double_blocks.<n>.processor.{qkv,proj}_lora{1,2}`)
        // and ship no single-stream keys, so the primary Flux signature (which
        // requires a single-block marker) misses them. `double_blocks.` is unique to
        // Flux among the architectures we detect; pairing it with the x-flux
        // processor/lora naming keeps this tight. Disqualified whenever single-block
        // keys are present so it never co-scores with the primary Flux signature
        // (two same-bucket scores could otherwise trip the runner-up margin and
        // return ambiguous).
        require_all_of: &[
            &["double_blocks."],
            &["qkv_lora", "proj_lora", "processor."],
        ],
        disqualifiers: &["single_transformer_blocks.", "single_blocks_"],
        markers: &["double_blocks.", "processor.", "qkv_lora", "proj_lora"],
    },
    BucketSignature {
        bucket: Bucket::WanVideo,
        // Wan transformers expose their blocks under `transformer.blocks.<n>.`
        // (not `transformer.transformer_blocks.`) and use `self_attn`/`cross_attn`/`ffn`
        // module names. Discriminating against the MMDiT-style key prefix
        // keeps Wan separate from Qwen/Z-Image.
        require_all_of: &[&["transformer.blocks."]],
        disqualifiers: &[
            "transformer.transformer_blocks.",
            "single_transformer_blocks.",
        ],
        markers: &[
            "transformer.blocks.",
            ".self_attn.",
            ".cross_attn.",
            ".ffn.",
        ],
    },
    BucketSignature {
        bucket: Bucket::LtxVideo,
        // LTX-Video uses MMDiT-style `transformer.transformer_blocks.` keys
        // but its attention submodules are named `attn1` and `attn2` (the
        // latter is the cross-attention path). It does not use the
        // dual-stream `img_mlp` / `txt_mlp` naming that Qwen-Image / Z-Image
        // expose.
        require_all_of: &[
            &["transformer.transformer_blocks."],
            &[".attn1."],
            &[".attn2."],
        ],
        disqualifiers: &[
            "single_transformer_blocks.",
            ".img_mlp.",
            ".txt_mlp.",
            "add_q_proj",
            "add_k_proj",
        ],
        markers: &["transformer.transformer_blocks.", ".attn1.", ".attn2."],
    },
    BucketSignature {
        bucket: Bucket::MmDit,
        // Dual-stream MMDiT covers Qwen-Image and Z-Image. They share a key
        // layout in current Diffusers releases; per-family disambiguation
        // happens after this bucket is selected.
        require_all_of: &[
            &["transformer.transformer_blocks."],
            &[
                ".img_mlp.",
                ".txt_mlp.",
                "add_q_proj",
                "add_k_proj",
                ".to_added_q.",
                ".to_added_k.",
            ],
        ],
        disqualifiers: &[
            "single_transformer_blocks.",
            "transformer.blocks.",
            ".attn2.",
        ],
        markers: &[
            "transformer.transformer_blocks.",
            ".img_mlp.",
            ".txt_mlp.",
            "add_q_proj",
            "add_k_proj",
            ".to_added_q.",
            ".to_added_k.",
        ],
    },
    BucketSignature {
        bucket: Bucket::Sdxl,
        // SDXL ships two text encoders, so kohya-style LoRAs always carry
        // both `lora_te1_` and `lora_te2_` prefixes alongside `lora_unet_`.
        require_all_of: &[&["lora_unet_"], &["lora_te1_"], &["lora_te2_"]],
        disqualifiers: &["transformer.transformer_blocks.", "transformer.blocks."],
        markers: &["lora_unet_", "lora_te1_", "lora_te2_"],
    },
    BucketSignature {
        bucket: Bucket::Sd15,
        // SD1.5 only ships a single text encoder, so kohya-style LoRAs
        // never carry the SDXL `lora_te1_` / `lora_te2_` split.
        require_all_of: &[&["lora_unet_"], &["lora_te_"]],
        disqualifiers: &[
            "lora_te1_",
            "lora_te2_",
            "transformer.transformer_blocks.",
            "transformer.blocks.",
        ],
        markers: &["lora_unet_", "lora_te_"],
    },
];

/// Minimum marker hits required for any bucket to win. Below this the file
/// is treated as ambiguous.
const MIN_KEY_MATCHES: usize = 4;

/// Best score must beat the runner-up by at least 1.5×. Encoded as a
/// rational comparison so we never need floats: best * DEN >= second * NUM.
const MARGIN_NUM: usize = 3;
const MARGIN_DEN: usize = 2;

/// Block-index threshold for Qwen-Image (larger, ~60 blocks). Values are
/// zero-indexed block numbers. Low block indices are intentionally not
/// enough to identify Z-Image: a sparse Qwen LoRA may only train early
/// blocks, and false hard rejections are worse than an inconclusive result.
const QWEN_MIN_BLOCK_INDEX: usize = 39;

fn detect_bucket(keys: &[String]) -> Option<Bucket> {
    let mut scored: Vec<(Bucket, usize)> = SIGNATURES
        .iter()
        .map(|sig| (sig.bucket, score_signature(sig, keys)))
        .collect();
    scored.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    let (best_bucket, best_score) = scored[0];
    if best_score < MIN_KEY_MATCHES {
        return None;
    }
    let second_score = scored.get(1).map(|entry| entry.1).unwrap_or(0);
    if second_score > 0 && best_score * MARGIN_DEN < second_score * MARGIN_NUM {
        // best/second < 1.5 → too close to call.
        return None;
    }
    Some(best_bucket)
}

fn score_signature(sig: &BucketSignature, keys: &[String]) -> usize {
    if keys.iter().any(|key| {
        sig.disqualifiers
            .iter()
            .any(|disqualifier| key.contains(disqualifier))
    }) {
        return 0;
    }
    let all_required_groups_satisfied = sig.require_all_of.iter().all(|group| {
        keys.iter()
            .any(|key| group.iter().any(|needle| key.contains(needle)))
    });
    if !all_required_groups_satisfied {
        return 0;
    }
    let mut score = 0_usize;
    for key in keys {
        if sig.markers.iter().any(|marker| key.contains(marker)) {
            score += 1;
        }
    }
    score
}

fn disambiguate_mm_dit(keys: &[String]) -> Option<String> {
    let max_block = max_transformer_block_index(keys)?;
    if max_block >= QWEN_MIN_BLOCK_INDEX {
        Some("qwen-image".to_owned())
    } else {
        None
    }
}

/// Returns the highest `N` seen in keys matching
/// `transformer.transformer_blocks.<N>.` (or just `transformer_blocks.<N>.`),
/// or `None` if no such key exists.
fn max_transformer_block_index(keys: &[String]) -> Option<usize> {
    let mut max_index: Option<usize> = None;
    for key in keys {
        if let Some(index) = parse_block_index(key) {
            max_index = Some(max_index.map_or(index, |current| current.max(index)));
        }
    }
    max_index
}

fn parse_block_index(key: &str) -> Option<usize> {
    let needle = "transformer_blocks.";
    let mut rest = key;
    while let Some(position) = rest.find(needle) {
        let candidate = &rest[position + needle.len()..];
        let digits: String = candidate.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            if let Ok(index) = digits.parse::<usize>() {
                return Some(index);
            }
        }
        rest = &candidate[digits.len()..];
    }
    None
}

fn detect_metadata_family(header: &Value) -> Option<String> {
    let metadata = header.get("__metadata__")?.as_object()?;
    for key in [
        "ss_base_model_version",
        "modelspec.architecture",
        "modelspec.implementation",
    ] {
        let Some(value) = metadata.get(key).and_then(Value::as_str) else {
            continue;
        };
        if let Some(family) = metadata_value_to_family(value) {
            return Some(family);
        }
    }
    None
}

fn metadata_value_to_family(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized.contains("flux") {
        return Some("flux".to_owned());
    }
    if normalized.contains("zimage") || normalized.contains("z-image") {
        return Some("z-image".to_owned());
    }
    if normalized.contains("qwen") && normalized.contains("image") {
        return Some("qwen-image".to_owned());
    }
    if normalized.contains("ltx") {
        return Some("ltx-video".to_owned());
    }
    if normalized.contains("wan") {
        return Some("wan-video".to_owned());
    }
    if normalized.contains("sdxl") {
        return Some("sdxl".to_owned());
    }
    if normalized == "sd1" || normalized == "sd1.5" || normalized.contains("stable-diffusion-v1") {
        return Some("sd1.5".to_owned());
    }
    None
}

/// The safetensors header is a JSON object whose top-level keys are tensor
/// names plus a special `__metadata__` entry. Returns the tensor names only.
fn collect_tensor_keys(header: &Value) -> Vec<String> {
    let Some(object) = header.as_object() else {
        return Vec::new();
    };
    object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header_from_keys(keys: &[&str]) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for key in keys {
            object.insert(
                (*key).to_owned(),
                json!({"dtype": "F16", "shape": [16, 1024], "data_offsets": [0, 32768]}),
            );
        }
        Value::Object(object)
    }

    fn write_safetensors(path: &Path, keys: &[String]) {
        // Emit a minimal valid safetensors layout: 8-byte little-endian header
        // length, then a JSON header whose entries each point at empty tensor
        // slices in the (empty) tensor section. The detector only reads the
        // header, so empty offsets are fine.
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for key in keys {
            header.insert(
                key.clone(),
                json!({"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        std::fs::write(path, buffer).expect("write safetensors");
    }

    fn diffusers_double_stream_keys(prefix: &str, block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
            for module in ["img_mlp.net.0.proj", "txt_mlp.net.0.proj"] {
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "{prefix}.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
            keys.push(format!(
                "{prefix}.transformer_blocks.{block}.attn.add_q_proj.lora_A.weight"
            ));
            keys.push(format!(
                "{prefix}.transformer_blocks.{block}.attn.add_q_proj.lora_B.weight"
            ));
        }
        keys
    }

    #[test]
    fn detects_wan_video() {
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in ["self_attn.q", "self_attn.k", "cross_attn.q", "ffn.0"] {
                keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
                keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_flux() {
        let mut keys = Vec::new();
        for block in 0..19 {
            keys.push(format!(
                "transformer.transformer_blocks.{block}.attn.to_q.lora_A.weight"
            ));
            keys.push(format!(
                "transformer.transformer_blocks.{block}.attn.to_q.lora_B.weight"
            ));
        }
        for block in 0..38 {
            keys.push(format!(
                "transformer.single_transformer_blocks.{block}.proj_mlp.lora_A.weight"
            ));
            keys.push(format!(
                "transformer.single_transformer_blocks.{block}.proj_mlp.lora_B.weight"
            ));
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn detects_kohya_flux() {
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["img_mlp_0", "txt_mlp_0", "img_attn_qkv", "txt_attn_qkv"] {
                keys.push(format!(
                    "lora_unet_double_blocks_{block}_{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "lora_unet_double_blocks_{block}_{module}.lora_up.weight"
                ));
            }
        }
        for block in 0..38 {
            keys.push(format!(
                "lora_unet_single_blocks_{block}_linear1.lora_down.weight"
            ));
            keys.push(format!(
                "lora_unet_single_blocks_{block}_linear1.lora_up.weight"
            ));
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn detects_metadata_family_before_keys() {
        let mut header = header_from_keys(&[
            "lora_unet_double_blocks_0_img_mlp_0.lora_down.weight",
            "lora_unet_double_blocks_0_img_mlp_0.lora_up.weight",
        ]);
        header["__metadata__"] = json!({
            "ss_base_model_version": "flux1",
            "modelspec.architecture": "flux-1-dev/lora"
        });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn detects_xflux_double_blocks_only() {
        // XLabs / x-flux realism-style LoRAs adapt only the double-stream blocks
        // (no single blocks, no metadata) via the attention-processor layout.
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["qkv_lora1", "qkv_lora2", "proj_lora1", "proj_lora2"] {
                keys.push(format!("double_blocks.{block}.processor.{module}.down.weight"));
                keys.push(format!("double_blocks.{block}.processor.{module}.up.weight"));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    #[test]
    fn flux_family_maps_to_flux_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("flux"), Some("flux_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "flux"),
            vec!["text_to_image", "style_variations"],
        );
    }

    #[test]
    fn detects_ltx_video() {
        let mut keys = Vec::new();
        for block in 0..28 {
            for module in ["attn1.to_q", "attn1.to_k", "attn2.to_q", "ff.net.0.proj"] {
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.transformer_blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("ltx-video"));
    }

    #[test]
    fn detects_qwen_image_by_block_count() {
        let keys = diffusers_double_stream_keys("transformer", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn low_mm_dit_block_count_is_inconclusive() {
        let keys = diffusers_double_stream_keys("transformer", 24);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn ambiguous_mm_dit_block_count_returns_none() {
        let keys = diffusers_double_stream_keys("transformer", 32);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn detects_sdxl() {
        let mut keys = Vec::new();
        for block in 0..10 {
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_down.weight"
            ));
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_up.weight"
            ));
        }
        keys.push(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned(),
        );
        keys.push(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned(),
        );
        keys.push(
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned(),
        );
        keys.push(
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned(),
        );
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sdxl"));
    }

    #[test]
    fn detects_sd15() {
        let mut keys = Vec::new();
        for block in 0..10 {
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_down.weight"
            ));
            keys.push(format!(
                "lora_unet_down_blocks_{block}_attentions_0_proj_in.lora_up.weight"
            ));
        }
        keys.push(
            "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned(),
        );
        keys.push("lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned());
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("sd1.5"));
    }

    #[test]
    fn empty_header_returns_none() {
        let header = json!({"__metadata__": {"format": "pt"}});
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn unknown_keys_return_none() {
        let header =
            header_from_keys(&["weird.custom.module.weight", "another.random.tensor.bias"]);
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn non_object_header_returns_none() {
        let header = json!(["not", "an", "object"]);
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn diffusers_class_names_map_to_known_families() {
        assert_eq!(
            diffusers_class_name_to_family("ZImagePipeline").as_deref(),
            Some("z-image")
        );
        assert_eq!(
            diffusers_class_name_to_family("QwenImagePipeline").as_deref(),
            Some("qwen-image")
        );
        assert_eq!(
            diffusers_class_name_to_family("FluxPipeline").as_deref(),
            Some("flux")
        );
        assert_eq!(
            diffusers_class_name_to_family("StableDiffusionXLPipeline").as_deref(),
            Some("sdxl")
        );
        assert!(diffusers_class_name_to_family("UnknownCustomPipeline").is_none());
    }

    #[test]
    fn reconcile_detected_family_rejects_mismatches_only() {
        assert_eq!(
            reconcile_detected_family(Some("z-image".to_owned()), Some("z-image".to_owned()))
                .unwrap()
                .as_deref(),
            Some("z-image")
        );
        assert_eq!(
            reconcile_detected_family(None, Some("qwen-image".to_owned()))
                .unwrap()
                .as_deref(),
            Some("qwen-image")
        );
        assert_eq!(
            reconcile_detected_family(Some("wan-video".to_owned()), None)
                .unwrap()
                .as_deref(),
            Some("wan-video")
        );
        assert_eq!(
            reconcile_detected_family(Some("z-image".to_owned()), Some("qwen-image".to_owned()))
                .unwrap_err(),
            FamilyMismatch {
                supplied: "z-image".to_owned(),
                detected: "qwen-image".to_owned(),
            }
        );
    }

    #[test]
    fn model_manifest_defaults_follow_supported_families() {
        let mut entry = serde_json::Map::new();
        apply_model_manifest_defaults(&mut entry, "video", Some("wan_video"));

        assert_eq!(entry["adapter"], "wan_video");
        assert_eq!(
            entry["capabilities"],
            json!([
                "image_to_video",
                "text_to_video",
                "first_last_frame",
                "extend_clip",
                "video_bridge",
                "replace_person"
            ])
        );
        assert_eq!(entry["loraCompatibility"]["families"], json!(["wan-video"]));
        assert_eq!(entry["downloads"], json!([]));
    }

    #[test]
    fn detect_model_family_reads_diffusers_index() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("model_index.json"),
            br#"{"_class_name": "ZImagePipeline", "_diffusers_version": "0.27.0"}"#,
        )
        .expect("write index");
        let family = detect_model_family(temp.path()).expect("detect");
        assert_eq!(family.as_deref(), Some("z-image"));
    }

    #[test]
    fn detect_model_family_falls_back_to_header() {
        let temp = tempfile::tempdir().expect("tempdir");
        let keys = diffusers_double_stream_keys("transformer", 40);
        write_safetensors(&temp.path().join("checkpoint.safetensors"), &keys);
        let family = detect_model_family(temp.path()).expect("detect");
        assert_eq!(family.as_deref(), Some("qwen-image"));
    }

    #[test]
    fn detect_model_family_returns_none_for_empty_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let family = detect_model_family(temp.path()).expect("detect");
        assert!(family.is_none());
    }

    #[test]
    fn detect_model_family_returns_none_for_unmapped_class_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("model_index.json"),
            br#"{"_class_name": "ExperimentalPipeline"}"#,
        )
        .expect("write index");
        let family = detect_model_family(temp.path()).expect("detect");
        assert!(family.is_none());
    }

    #[test]
    fn sensenova_u1_image_family_supports_text_to_image_and_edit() {
        let caps = model_capabilities_for_type_and_family("image", "sensenova-u1");
        assert!(caps.contains(&"text_to_image"));
        assert!(caps.contains(&"edit_image"));
    }
}
