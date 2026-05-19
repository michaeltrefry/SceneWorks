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

use serde_json::Value;

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

/// Returns the detected LoRA architecture family or `None` if the header
/// is ambiguous, empty, or matches no known signature with confidence.
pub fn detect_lora_family(header: &Value) -> Option<String> {
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
        // alone is enough to identify it.
        require_all_of: &[&["single_transformer_blocks.", "double_blocks."]],
        disqualifiers: &[],
        markers: &[
            "single_transformer_blocks.",
            "double_blocks.",
            "transformer_blocks.",
        ],
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

/// Block-index thresholds separating Z-Image (smaller, ~24 blocks) from
/// Qwen-Image (larger, ~60 blocks). The gap between the two architectures
/// is wide enough that picking middle thresholds and returning `None`
/// inside the no-man's-land keeps the detector honest. Values are
/// zero-indexed block numbers, so 27 means "highest block at index 27"
/// (i.e. up to 28 blocks total).
const ZIMAGE_MAX_BLOCK_INDEX: usize = 27;
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
    } else if max_block <= ZIMAGE_MAX_BLOCK_INDEX {
        Some("z-image".to_owned())
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
                keys.push(format!(
                    "transformer.blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "transformer.blocks.{block}.{module}.lora_B.weight"
                ));
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
    fn detects_z_image_by_block_count() {
        let keys = diffusers_double_stream_keys("transformer", 24);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("z-image"));
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
        keys.push("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned());
        keys.push("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned());
        keys.push("lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned());
        keys.push("lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight".to_owned());
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
        keys.push("lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight".to_owned());
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
        let header = header_from_keys(&[
            "weird.custom.module.weight",
            "another.random.tensor.bias",
        ]);
        assert!(detect_lora_family(&header).is_none());
    }

    #[test]
    fn non_object_header_returns_none() {
        let header = json!(["not", "an", "object"]);
        assert!(detect_lora_family(&header).is_none());
    }
}
