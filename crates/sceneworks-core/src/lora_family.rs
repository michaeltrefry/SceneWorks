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
    /// The header parsed cleanly but the file is too small to hold the tensor
    /// data the header declares — the file is truncated/incomplete (e.g. an
    /// interrupted download). `declared` is the minimum size the header implies
    /// (`8 + header_len + max(tensor data_offsets end)`), `actual` is the size
    /// on disk.
    IncompleteData { declared: u64, actual: u64 },
}

impl std::fmt::Display for SafetensorsHeaderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidHeader => formatter.write_str("invalid safetensors header"),
            Self::IncompleteData { declared, actual } => write!(
                formatter,
                "incomplete or truncated safetensors: file is {actual} bytes but the header \
                 declares tensor data requiring at least {declared} bytes"
            ),
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
    let header = serde_json::from_slice::<Value>(&header)
        .map_err(|_| SafetensorsHeaderError::InvalidHeader)?;
    // A valid header can still front a truncated/incomplete file (an interrupted
    // download): the data section must be large enough to hold every tensor the
    // header declares. The tensor `data_offsets` are relative to the byte buffer
    // that begins right after the 8-byte length and the header JSON, so the file
    // must be at least `8 + header_len + max(data_offsets end)` bytes. Without this
    // the bad file is accepted at import and only fails cryptically at load time
    // ("invalid data offsets exceeding the size of the file"). See sc-6072.
    let declared = 8_u64
        .saturating_add(header_len)
        .saturating_add(max_tensor_data_end(&header));
    if metadata.len() < declared {
        return Err(SafetensorsHeaderError::IncompleteData {
            declared,
            actual: metadata.len(),
        });
    }
    Ok(header)
}

/// The largest `data_offsets` end across all tensor entries in a parsed
/// safetensors header — i.e. the length of the tensor data section the header
/// declares. The `__metadata__` key and any entry without a well-formed
/// two-element `data_offsets` array contribute nothing (they carry no tensor
/// bytes). Returns 0 for a header with no tensors.
fn max_tensor_data_end(header: &Value) -> u64 {
    let Some(entries) = header.as_object() else {
        return 0;
    };
    entries
        .iter()
        .filter(|(key, _)| key.as_str() != "__metadata__")
        .filter_map(|(_, tensor)| {
            tensor
                .get("data_offsets")
                .and_then(Value::as_array)
                .and_then(|offsets| offsets.get(1))
                .and_then(Value::as_u64)
        })
        .max()
        .unwrap_or(0)
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
        "chroma" => Some("chroma_diffusers"),
        "kolors" => Some("kolors_diffusers"),
        "sdxl" => Some("sdxl_diffusers"),
        "ltx-video" => Some("ltx_video"),
        "wan-video" => Some("wan_video"),
        "svd" => Some("svd_video"),
        // Bernini is macOS-only native MLX (epic 4699): there is no Torch/diffusers
        // adapter. This label is recorded in recipe/lineage only; on Mac the job is
        // MLX-routed by engine id, never instantiated through a Torch adapter.
        "bernini" => Some("bernini"),
        // SCAIL-2 (epic 5439) is likewise macOS-only native MLX (engine id
        // "scail2_14b"); no Torch/diffusers adapter. Lineage label only.
        "scail2" => Some("scail2"),
        _ => None,
    }
}

pub fn model_capabilities_for_type_and_family(model_type: &str, family: &str) -> Vec<&'static str> {
    match (
        model_type.trim().to_ascii_lowercase().as_str(),
        normalize_model_family(family).as_str(),
    ) {
        // No `character_image`: the worker's ZImageDiffusersAdapter has no IP-Adapter
        // / reference-conditioning code, and no Z-Image IP-Adapter exists upstream
        // (sc-2005). Custom z-image models that override capabilities can still
        // re-declare it, but the family default shouldn't claim what it can't do.
        ("image", "z-image") => vec!["text_to_image", "style_variations"],
        ("image", "qwen-image") => vec!["text_to_image", "style_variations"],
        ("image", "lens") => vec!["text_to_image", "style_variations"],
        ("image", "sensenova-u1") => vec!["text_to_image", "edit_image", "vqa", "interleave"],
        ("image", "flux") => vec!["text_to_image", "style_variations"],
        ("image", "chroma") => vec!["text_to_image", "style_variations"],
        ("image", "kolors") => vec!["text_to_image", "character_image", "style_variations"],
        ("image", "sdxl") => vec!["text_to_image", "edit_image", "style_variations"],
        // Bernini still-image companion (epic 4699 / sc-5424): the same `Modality::Both`
        // engine the video `bernini` family uses, but the image-typed catalog id
        // (`bernini_image`) exposes only the still tasks — t2i (text→image) and i2i
        // (`edit_image`, the source-image edit via `Conditioning::Reference`). No
        // `character_image`/`style_variations` (no IP-Adapter/style surface) and no LoRA
        // (the descriptor reports `supports_lora: false`).
        ("image", "bernini") => vec!["text_to_image", "edit_image"],
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
        // Stable Video Diffusion is image-conditioned only (no text prompt) and
        // does not support the timeline/replacement modes.
        ("video", "svd") => vec!["image_to_video"],
        // Bernini (epic 4699) is a Wan2.2-T2V-A14B renderer + Qwen2.5-VL semantic
        // planner whose engine descriptor is `Modality::Both` with conditioning
        // `[Reference, MultiReference, VideoClip]`. The video task surface maps onto
        // SceneWorks modes (sc-4703 / sc-5425): `text_to_video` (t2v), `video_to_video`
        // (v2v — a source clip edit), `reference_to_video` (r2v — subject reference
        // images → video), `reference_video_to_video` (rv2v — source clip + reference
        // images), `multi_video_to_video` (mv2v — multiple source clips), and `ads2v`
        // (source video + reference video + reference images). Bernini has no classic
        // still-image-to-video (its renderer is T2V, not I2V). The t2i/i2i image
        // companion is a separate image-typed catalog id (tracked under epic 4699), not
        // declared here.
        ("video", "bernini") => vec![
            "text_to_video",
            "video_to_video",
            "reference_to_video",
            "reference_video_to_video",
            "multi_video_to_video",
            "ads2v",
        ],
        // SCAIL-2 (epic 5439) is a Wan2.1-14B I2V end-to-end character-animation
        // engine: a reference character image + a driving video → an animated clip.
        // Its engine descriptor is `Modality::Video` with conditioning
        // `[Reference, Mask, MultiReference, ControlClip]`. It serves the standalone
        // `animate_character` mode (sc-5448, the worker paints the color-coded masks
        // from native SAM3) and cross-identity `replace_person` (sc-5452, the same
        // engine with replace_flag=true, as the higher-quality backend behind the
        // existing person-track replacement pipeline). LoRA is sc-5451 — not declared
        // until wired. Multi-character (paired ref+mask) awaits the engine
        // request-contract extension (sc-5583).
        ("video", "scail2") => vec!["animate_character", "replace_person"],
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
        "chromapipeline" | "chromaimg2imgpipeline" => Some("chroma".to_owned()),
        "kolorspipeline" | "kolorsimg2imgpipeline" => Some("kolors".to_owned()),
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
        Bucket::Flux2 => Some("flux2".to_owned()),
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
    Flux2,
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
        bucket: Bucket::Flux2,
        // FLUX.2 [klein] native/ComfyUI LoRAs share the FLUX.1 `diffusion_model.`
        // prefix and the same double_blocks/single_blocks split, but FLUX.2 shares
        // modulation ACROSS all blocks via top-level `double_stream_modulation_img`
        // / `double_stream_modulation_txt` / `single_stream_modulation` tensors,
        // whereas FLUX.1 keeps per-block `img_mod`/`txt_mod`/`modulation` inside
        // each block index. Those shared-modulation keys are unique to FLUX.2 and
        // never appear in a FLUX.1 LoRA, so requiring one cleanly separates the two
        // (the primary Flux signature can't fire here anyway — FLUX.2 uses
        // `single_blocks.` with a dot, not the `single_blocks_` underscore form).
        require_all_of: &[&["double_stream_modulation_", "single_stream_modulation."]],
        disqualifiers: &["single_transformer_blocks."],
        markers: &[
            "double_stream_modulation_",
            "single_stream_modulation.",
            "double_blocks.",
            "single_blocks.",
            ".img_attn.",
            ".txt_attn.",
        ],
    },
    BucketSignature {
        bucket: Bucket::WanVideo,
        // Diffusers-format Wan LoRAs expose their blocks under
        // `transformer.blocks.<n>.` (not `transformer.transformer_blocks.`) and
        // use either the native `self_attn`/`cross_attn`/`ffn` module names or
        // the diffusers `attn1`/`attn2` names. The `transformer.blocks.` prefix
        // marker alone scores every key; discriminating against the MMDiT-style
        // key prefix keeps Wan separate from Qwen/Z-Image.
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
        bucket: Bucket::WanVideo,
        // ComfyUI / native Wan checkpoints and LoRAs prefix every block key with
        // `diffusion_model.blocks.<n>.` and keep the native `self_attn`/
        // `cross_attn`/`ffn` module names. Requiring both the prefix and a Wan
        // module marker keeps this from colliding with ComfyUI Flux
        // (`diffusion_model.double_blocks.` / `diffusion_model.single_blocks.`)
        // or LTX (`diffusion_model.transformer_blocks.`), none of which contain
        // the bare `.blocks.` segment.
        require_all_of: &[
            &["diffusion_model.blocks."],
            &[".self_attn.", ".cross_attn.", ".ffn."],
        ],
        disqualifiers: &[
            "transformer.transformer_blocks.",
            "single_transformer_blocks.",
            "double_blocks.",
        ],
        markers: &[
            "diffusion_model.blocks.",
            ".self_attn.",
            ".cross_attn.",
            ".ffn.",
        ],
    },
    BucketSignature {
        bucket: Bucket::WanVideo,
        // Kohya / musubi-tuner Wan LoRAs flatten the native module path into
        // underscore-delimited keys: `lora_unet_blocks_<n>_self_attn_q...`.
        // `lora_unet_blocks_` is Wan-specific — SD/SDXL UNet keys are
        // `lora_unet_down_blocks_` / `lora_unet_up_blocks_` / `lora_unet_mid_block_`,
        // never the bare `lora_unet_blocks_`. Disqualifying the SD/SDXL
        // text-encoder prefixes (which Wan lacks) prevents any collision with the
        // Sd15/Sdxl signatures, whose text-encoder keys also contain `_self_attn_`.
        require_all_of: &[
            &["lora_unet_blocks_"],
            &["_self_attn_", "_cross_attn_", "_ffn_"],
        ],
        disqualifiers: &["lora_te_", "lora_te1_", "lora_te2_"],
        markers: &["lora_unet_blocks_", "_self_attn_", "_cross_attn_", "_ffn_"],
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
        bucket: Bucket::MmDit,
        // kohya / musubi-tuner / LyCORIS (lycoris-lora) Qwen-Image & Z-Image LoRAs
        // flatten the dual-stream MMDiT module path into underscore-delimited keys
        // behind a `lora_unet_` / `lora_transformer_` / `lycoris_` prefix, e.g.
        // `lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1` or
        // `lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lora_down.weight`. The
        // dotted MMDiT signature above can't see these (no `transformer.
        // transformer_blocks.` segment). `_transformer_blocks_` (underscores on both
        // sides) is the discriminator: Wan kohya uses `_blocks_`, SD/SDXL kohya use
        // `_down_blocks_`/`_up_blocks_`/`_mid_block_`, and SDXL's nested
        // `_transformer_blocks_` never carries the joint-attention `add_{q,k}_proj`
        // or dual-stream `_img_mlp_`/`_txt_mlp_` that group two requires. Flux's
        // single-stream keys are disqualified so a (double+single) Flux LoRA never
        // lands here despite sharing `transformer_blocks` + `add_q_proj`.
        require_all_of: &[
            &["_transformer_blocks_"],
            &[
                "_img_mlp_",
                "_txt_mlp_",
                "add_q_proj",
                "add_k_proj",
                "to_added_q",
                "to_added_k",
            ],
        ],
        disqualifiers: &[
            "single_transformer_blocks",
            "single_blocks_",
            ".attn1.",
            ".attn2.",
            "_attn1_",
            "_attn2_",
        ],
        markers: &[
            "_transformer_blocks_",
            "_img_mlp_",
            "_txt_mlp_",
            "add_q_proj",
            "add_k_proj",
            "to_added_q",
            "to_added_k",
            "_attn_to_q",
            "_attn_to_k",
            "_attn_to_v",
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
    // Diffusers separates with dots (`transformer_blocks.<N>.`); kohya / LyCORIS
    // flatten to underscores (`transformer_blocks_<N>_`). Accept either separator.
    let needle = "transformer_blocks";
    let mut rest = key;
    while let Some(position) = rest.find(needle) {
        let after = &rest[position + needle.len()..];
        let candidate = match after.as_bytes().first() {
            Some(b'.' | b'_') => &after[1..],
            _ => after,
        };
        let digits: String = candidate.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() {
            if let Ok(index) = digits.parse::<usize>() {
                return Some(index);
            }
        }
        rest = after;
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
    // Check chroma before flux: Chroma is FLUX.1-schnell-derived, so a Chroma
    // LoRA's metadata may name both. Only metadata can distinguish the two — by
    // tensor keys a Chroma LoRA is identical to a Flux LoRA (same double/single
    // transformer blocks), so the key-based detector classifies it as `flux`.
    if normalized.contains("chroma") {
        return Some("chroma".to_owned());
    }
    // Check flux2 before flux: FLUX.2 architecture strings ("flux2", "flux-2",
    // "flux.2", "flux_2") all contain the "flux" substring, so the generic flux
    // match below would otherwise swallow them. FLUX.2 is a distinct family with
    // its own MLX adapter — a FLUX.1 ("flux") LoRA is not interchangeable with it.
    if normalized.contains("flux2")
        || normalized.contains("flux-2")
        || normalized.contains("flux.2")
        || normalized.contains("flux_2")
    {
        return Some("flux2".to_owned());
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

// ---------------------------------------------------------------------------
// LoRA family-compatibility validation (epic 3018, sc-3027).
//
// Ported from the Python worker's `lora_adapters.py`
// (validate_lora_compatibility / accepted_lora_families / lora_families /
// lora_base_model) so the Rust GPU worker rejects an incompatible LoRA *before*
// a job runs, with the same message, instead of failing deep in the engine's
// strict adapter loader. Pure (no I/O): it reads the LoRA spec's *declared*
// families, exactly like the Python pre-flight.
// ---------------------------------------------------------------------------

/// Maximum LoRAs per job (matches the worker's `MAX_JOB_LORAS` / Python
/// `normalize_lora_specs`).
pub const MAX_JOB_LORAS: usize = 3;

/// Architecture families a model can load LoRAs from *in addition to* its own
/// (Python `EXTRA_COMPATIBLE_LORA_FAMILIES`). Chroma is FLUX.1-derived and shares
/// Flux's block layout, so Flux LoRAs load on Chroma (one-directional). FLUX.2
/// [klein]'s model family is `flux2-klein` but klein LoRAs are detected/declared
/// as `flux2`, so a klein model must accept `flux2` LoRAs. FLUX.2-dev's family is
/// `flux2-dev` (a separate model — Mistral3 TE + 48/48 DiT) but it shares the FLUX.2
/// transformer layout, so dev LoRAs are likewise detected/declared as `flux2` (epic 5914;
/// dev LoRA application is validated in sc-5920).
fn extra_compatible_lora_families(normalized_family: &str) -> &'static [&'static str] {
    match normalized_family {
        "chroma" => &["flux"],
        "flux2-klein" | "flux2-dev" => &["flux2"],
        _ => &[],
    }
}

/// The set of LoRA families a model of `model_family` can load (normalized; the
/// model's own family plus its extra-compatible families). Empty when the family
/// is unknown — callers treat that as "skip validation".
pub fn accepted_lora_families(model_family: &str) -> Vec<String> {
    let normalized = normalize_model_family(model_family);
    if normalized.is_empty() {
        return Vec::new();
    }
    let mut families = vec![normalized.clone()];
    families.extend(
        extra_compatible_lora_families(&normalized)
            .iter()
            .map(|family| (*family).to_owned()),
    );
    families
}

/// The LoRA's declared compatible families (normalized, de-duplicated, sorted),
/// from the first present of `families` / `compatibleFamilies` / `modelFamilies`
/// / `compatibility.families` / `[family]`. Empty when the spec declares none
/// (an unstamped LoRA the user vouches for — never rejected on family grounds).
pub fn lora_declared_families(lora: &Value) -> Vec<String> {
    let compatibility = lora.get("compatibility").and_then(Value::as_object);
    let raw = ["families", "compatibleFamilies", "modelFamilies"]
        .into_iter()
        .find_map(|key| lora.get(key).and_then(Value::as_array).cloned())
        .or_else(|| {
            compatibility
                .and_then(|compat| compat.get("families").and_then(Value::as_array).cloned())
        })
        .or_else(|| {
            lora.get("family")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(|value| vec![Value::String(value.to_owned())])
        })
        .unwrap_or_default();
    let mut families: Vec<String> = raw
        .iter()
        .filter_map(Value::as_str)
        .map(normalize_model_family)
        .filter(|family| !family.is_empty())
        .collect();
    families.sort();
    families.dedup();
    families
}

/// The specific base model a LoRA records (e.g. `wan_2_2`, `wan_2_2_t2v_14b`), or
/// `None`. Used by the base-model gate for families where a matching family alone
/// is not enough (Python `lora_base_model`).
pub fn lora_base_model(lora: &Value) -> Option<String> {
    lora.get("baseModel")
        .or_else(|| lora.get("base_model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Families that share an architecture family but NOT a LoRA-compatible
/// architecture, so the trained base model must also match (Python
/// `_BASE_MODEL_GATED_FAMILIES`). Wan: `wan_2_2` (5B) and `wan_2_2_*_14b` (A14B)
/// are both `wan-video` but cross-applying a LoRA garbles output.
fn is_base_model_gated_family(family: &str) -> bool {
    family == "wan-video"
}

/// A LoRA id for error messages: `id` / `loraId` / `lora_<n>`.
fn lora_display_id(lora: &Value, index: usize) -> String {
    lora.get("id")
        .or_else(|| lora.get("loraId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("lora_{}", index + 1))
}

/// Validate every LoRA in `loras` against `model_family` before a job runs
/// (Python `validate_lora_compatibility`). Errors on a declared family the model
/// cannot load, or — for a base-model-gated family (Wan) — a recorded base model
/// that differs from `model_id`. A LoRA that declares no family is skipped (the
/// user vouches for it). Returns the user-facing message as `Err`.
pub fn validate_lora_compatibility(
    loras: &[Value],
    model_family: Option<&str>,
    adapter_id: &str,
    model_id: Option<&str>,
) -> Result<(), String> {
    let normalized_model_family = model_family.map(normalize_model_family).unwrap_or_default();
    let accepted = model_family.map(accepted_lora_families).unwrap_or_default();
    if loras.is_empty() || accepted.is_empty() {
        return Ok(());
    }
    for (index, lora) in loras.iter().enumerate() {
        let families = lora_declared_families(lora);
        if families.is_empty() {
            continue;
        }
        let lora_id = lora_display_id(lora, index);
        // Accept when any declared family is one the model can load.
        if !families.iter().any(|family| accepted.contains(family)) {
            return Err(format!(
                "LoRA {lora_id} is not compatible with model family {normalized_model_family} for {adapter_id}."
            ));
        }
        // Base-model gating (Wan 5B vs 14B): a LoRA that records its trained base
        // model only applies to that exact model; one without falls back to family.
        if let Some(model_id) = model_id {
            if families
                .iter()
                .any(|family| is_base_model_gated_family(family))
            {
                if let Some(base) = lora_base_model(lora) {
                    if base != model_id {
                        return Err(format!(
                            "LoRA {lora_id} was trained for base model {base}, not {model_id}; \
                             Wan 5B and 14B LoRAs are not interchangeable."
                        ));
                    }
                }
            }
        }
    }
    Ok(())
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
    fn detects_diffusers_wan_video() {
        // The `wan_lora` trainer (epic 1949 sc-1952) saves diffusers-format keys
        // via WanPipeline.save_lora_weights: `transformer.blocks.<n>.attn1|attn2.to_*`
        // (not the native `self_attn`/`cross_attn`/`ffn` names). These must still
        // detect as wan-video so the inference loader gates them correctly.
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in [
                "attn1.to_q",
                "attn1.to_k",
                "attn1.to_v",
                "attn1.to_out.0",
                "attn2.to_q",
                "attn2.to_k",
                "attn2.to_v",
                "attn2.to_out.0",
            ] {
                keys.push(format!("transformer.blocks.{block}.{module}.lora_A.weight"));
                keys.push(format!("transformer.blocks.{block}.{module}.lora_B.weight"));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_comfyui_native_wan_video() {
        // ComfyUI / native Wan LoRAs prefix every block with
        // `diffusion_model.blocks.<n>.` and keep the native self_attn/cross_attn/ffn
        // module names. These contain `.blocks.` but not `transformer.blocks.`, so
        // the diffusers Wan signature misses them — the ComfyUI sibling must catch them.
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in [
                "self_attn.q",
                "self_attn.k",
                "self_attn.v",
                "self_attn.o",
                "cross_attn.q",
                "cross_attn.k",
                "ffn.0",
                "ffn.2",
            ] {
                keys.push(format!(
                    "diffusion_model.blocks.{block}.{module}.lora_A.weight"
                ));
                keys.push(format!(
                    "diffusion_model.blocks.{block}.{module}.lora_B.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("wan-video"));
    }

    #[test]
    fn detects_kohya_wan_video() {
        // Kohya / musubi-tuner Wan LoRAs flatten the path into underscore-delimited
        // keys with a `lora_unet_blocks_<n>_` prefix and no text-encoder keys.
        let mut keys = Vec::new();
        for block in 0..30 {
            for module in [
                "self_attn_q",
                "self_attn_k",
                "self_attn_v",
                "self_attn_o",
                "cross_attn_q",
                "cross_attn_k",
                "ffn_0",
                "ffn_2",
            ] {
                keys.push(format!(
                    "lora_unet_blocks_{block}_{module}.lora_down.weight"
                ));
                keys.push(format!("lora_unet_blocks_{block}_{module}.lora_up.weight"));
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
    fn chroma_metadata_distinguishes_chroma_from_flux_keys() {
        // Chroma is FLUX.1-schnell-derived: its LoRA tensor keys are identical to
        // Flux (single/double transformer blocks), so only metadata can mark a
        // LoRA as chroma. Metadata is checked before keys and chroma before flux.
        let mut header = header_from_keys(&[
            "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        ]);
        header["__metadata__"] = json!({
            "modelspec.architecture": "chroma/lora"
        });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("chroma"));
    }

    #[test]
    fn detects_xflux_double_blocks_only() {
        // XLabs / x-flux realism-style LoRAs adapt only the double-stream blocks
        // (no single blocks, no metadata) via the attention-processor layout.
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["qkv_lora1", "qkv_lora2", "proj_lora1", "proj_lora2"] {
                keys.push(format!(
                    "double_blocks.{block}.processor.{module}.down.weight"
                ));
                keys.push(format!(
                    "double_blocks.{block}.processor.{module}.up.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux"));
    }

    fn flux2_klein_native_keys() -> Vec<String> {
        // Mirrors the real klein_9B_Turbo_r128.safetensors layout: native/ComfyUI
        // `diffusion_model.` prefix, 8 double blocks + 24 single blocks, and FLUX.2's
        // shared (per-stream, not per-block) modulation tensors.
        let mut keys = Vec::new();
        for block in 0..8 {
            for module in [
                "img_attn.proj",
                "img_attn.qkv",
                "txt_attn.proj",
                "txt_attn.qkv",
            ] {
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_up.weight"
                ));
            }
            for module in ["img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"] {
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        for block in 0..24 {
            for module in ["linear1", "linear2"] {
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        for module in [
            "double_stream_modulation_img.lin",
            "double_stream_modulation_txt.lin",
            "single_stream_modulation.lin",
            "img_in",
            "txt_in",
        ] {
            keys.push(format!("diffusion_model.{module}.lora_down.weight"));
            keys.push(format!("diffusion_model.{module}.lora_up.weight"));
        }
        keys
    }

    #[test]
    fn detects_flux2_klein_native() {
        let keys = flux2_klein_native_keys();
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux2"));
    }

    #[test]
    fn flux1_native_keys_do_not_detect_as_flux2() {
        // A FLUX.1 native LoRA shares the double_blocks/single_blocks split but keeps
        // PER-BLOCK modulation (`img_mod`/`txt_mod`/`modulation`) and has none of the
        // shared `*_stream_modulation` tensors, so it must not be misread as FLUX.2.
        let mut keys = Vec::new();
        for block in 0..19 {
            for module in ["img_attn.qkv", "txt_attn.qkv", "img_mod.lin", "txt_mod.lin"] {
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.double_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        for block in 0..38 {
            for module in ["linear1", "modulation.lin"] {
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_down.weight"
                ));
                keys.push(format!(
                    "diffusion_model.single_blocks.{block}.{module}.lora_up.weight"
                ));
            }
        }
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_ne!(detect_lora_family(&header).as_deref(), Some("flux2"));
    }

    #[test]
    fn flux2_metadata_wins_over_generic_flux_substring() {
        // FLUX.2 architecture strings contain "flux"; the flux2 metadata branch must
        // claim them before the generic flux match (mirrors chroma-before-flux).
        for arch in ["flux2", "flux-2", "flux.2-klein", "FLUX_2/lora"] {
            let mut header = header_from_keys(&[
                "lora_unet_double_blocks_0_img_mlp_0.lora_down.weight",
                "lora_unet_double_blocks_0_img_mlp_0.lora_up.weight",
            ]);
            header["__metadata__"] = json!({ "modelspec.architecture": arch });
            assert_eq!(
                detect_lora_family(&header).as_deref(),
                Some("flux2"),
                "architecture {arch} should map to flux2"
            );
        }
    }

    #[test]
    fn ai_toolkit_flux2_klein_base_model_version_detects_flux2() {
        // Real ai-toolkit klein LoRAs (e.g. V3_flux_klein.safetensors) train only
        // attn/mlp/linear — no `*_stream_modulation` tensors — so the key signature
        // can't fire, but they carry `ss_base_model_version: "flux2_klein_9b"`. That
        // value contains "flux", so the generic flux branch used to swallow it; the
        // flux2 branch must claim it first.
        let mut header = header_from_keys(&[
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "diffusion_model.single_blocks.0.linear1.lora_A.weight",
        ]);
        header["__metadata__"] = json!({ "ss_base_model_version": "flux2_klein_9b" });

        assert_eq!(detect_lora_family(&header).as_deref(), Some("flux2"));
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
    fn chroma_family_maps_to_chroma_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("chroma"), Some("chroma_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "chroma"),
            vec!["text_to_image", "style_variations"],
        );
    }

    #[test]
    fn kolors_family_maps_to_kolors_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("kolors"), Some("kolors_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "kolors"),
            vec!["text_to_image", "character_image", "style_variations"],
        );
    }

    #[test]
    fn sdxl_family_maps_to_sdxl_diffusers_adapter_and_image_capabilities() {
        assert_eq!(model_adapter_for_family("sdxl"), Some("sdxl_diffusers"));
        assert_eq!(
            model_capabilities_for_type_and_family("image", "sdxl"),
            vec!["text_to_image", "edit_image", "style_variations"],
        );
    }

    #[test]
    fn svd_family_maps_to_svd_video_adapter_and_image_to_video_only() {
        assert_eq!(model_adapter_for_family("svd"), Some("svd_video"));
        // SVD is image-conditioned only — no text-to-video or timeline modes.
        assert_eq!(
            model_capabilities_for_type_and_family("video", "svd"),
            vec!["image_to_video"],
        );
    }

    #[test]
    fn bernini_family_maps_to_bernini_adapter_and_full_video_task_surface() {
        assert_eq!(model_adapter_for_family("bernini"), Some("bernini"));
        // sc-4703 / sc-5425: the full Bernini video task surface — t2v + the editing
        // (`video_to_video`), reference-driven (`reference_to_video` /
        // `reference_video_to_video`), and multi-source (`multi_video_to_video` /
        // `ads2v`) modes. No still-image-to-video (renderer is T2V).
        assert_eq!(
            model_capabilities_for_type_and_family("video", "bernini"),
            vec![
                "text_to_video",
                "video_to_video",
                "reference_to_video",
                "reference_video_to_video",
                "multi_video_to_video",
                "ads2v",
            ],
        );
        // sc-5424: the image-typed companion (`bernini_image`) shares the `bernini`
        // family/adapter but exposes only the still tasks — t2i + i2i (`edit_image`).
        assert_eq!(
            model_capabilities_for_type_and_family("image", "bernini"),
            vec!["text_to_image", "edit_image"],
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

    /// kohya / musubi-tuner / LyCORIS export of a dual-stream MMDiT (Qwen-Image /
    /// Z-Image) adapter: module paths flattened with underscores behind `prefix`,
    /// carrying LoKr (`lokr_w1`/`lokr_w2`/`alpha`) tensors. Mirrors the real
    /// lycoris-lora file shape from sc-2626.
    fn lycoris_underscore_mmdit_keys(prefix: &str, block_count: usize) -> Vec<String> {
        let mut keys = Vec::new();
        for block in 0..block_count {
            for module in [
                "attn_to_q",
                "attn_to_k",
                "attn_to_v",
                "attn_to_out_0",
                "attn_add_q_proj",
                "attn_add_k_proj",
                "attn_add_v_proj",
                "attn_to_add_out",
                "img_mlp_net_0_proj",
                "img_mlp_net_2",
                "txt_mlp_net_0_proj",
                "txt_mlp_net_2",
            ] {
                let base = format!("{prefix}_transformer_blocks_{block}_{module}");
                keys.push(format!("{base}.lokr_w1"));
                keys.push(format!("{base}.lokr_w2"));
                keys.push(format!("{base}.alpha"));
            }
        }
        keys
    }

    #[test]
    fn detects_lycoris_underscore_qwen_image_lokr() {
        // The real failing upload (sc-2626): a lycoris-lora-exported Qwen-Image LoKr
        // whose keys carry the library's default `lycoris` prefix and underscore-
        // flattened module paths — invisible to the dotted MMDiT signature.
        let keys = lycoris_underscore_mmdit_keys("lycoris", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn detects_kohya_underscore_qwen_image() {
        // kohya / musubi-tuner flatten with a `lora_unet` prefix instead.
        let keys = lycoris_underscore_mmdit_keys("lora_unet", 60);
        let header = header_from_keys(&keys.iter().map(String::as_str).collect::<Vec<_>>());

        assert_eq!(detect_lora_family(&header).as_deref(), Some("qwen-image"));
    }

    #[test]
    fn low_underscore_mm_dit_block_count_is_inconclusive() {
        // Same conservative block-count gate as the dotted path: too few blocks to
        // tell Qwen from Z-Image → inconclusive rather than a wrong guess.
        let keys = lycoris_underscore_mmdit_keys("lycoris", 24);
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
            diffusers_class_name_to_family("ChromaPipeline").as_deref(),
            Some("chroma")
        );
        assert_eq!(
            diffusers_class_name_to_family("KolorsPipeline").as_deref(),
            Some("kolors")
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

    /// Write a safetensors file whose header declares a single tensor spanning
    /// `[0, declared_data_len)` but whose data section on disk is only
    /// `actual_data_len` bytes — so `declared_data_len > actual_data_len`
    /// reproduces a truncated/interrupted download.
    fn write_safetensors_with_data(path: &Path, declared_data_len: u64, actual_data_len: u64) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        header.insert(
            "lora.weight".to_owned(),
            json!({"dtype": "F16", "shape": [1], "data_offsets": [0, declared_data_len]}),
        );
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("serialize header");
        let mut buffer = (header_bytes.len() as u64).to_le_bytes().to_vec();
        buffer.extend_from_slice(&header_bytes);
        buffer.resize(buffer.len() + actual_data_len as usize, 0_u8);
        std::fs::write(path, buffer).expect("write safetensors");
    }

    #[test]
    fn read_safetensors_header_accepts_complete_data_section() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("complete.safetensors");
        write_safetensors_with_data(&path, 32, 32);
        read_safetensors_header(&path).expect("complete file is accepted");
    }

    #[test]
    fn read_safetensors_header_accepts_trailing_padding() {
        // A file larger than the declared data section (trailing padding) is not
        // "incomplete"; only truncation is rejected.
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("padded.safetensors");
        write_safetensors_with_data(&path, 32, 64);
        read_safetensors_header(&path).expect("over-long file is accepted");
    }

    #[test]
    fn read_safetensors_header_rejects_truncated_data_section() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("truncated.safetensors");
        // Header declares 1024 bytes of tensor data, but only 16 are present.
        write_safetensors_with_data(&path, 1024, 16);
        match read_safetensors_header(&path) {
            Err(SafetensorsHeaderError::IncompleteData { declared, actual }) => {
                assert!(
                    actual < declared,
                    "actual {actual} should be below declared minimum {declared}"
                );
            }
            other => panic!("expected IncompleteData, got {other:?}"),
        }
    }

    #[test]
    fn read_safetensors_header_accepts_empty_tensors() {
        // The `write_safetensors` helper emits empty tensors (`data_offsets [0, 0]`);
        // a header-only file with no tensor bytes is complete.
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("empty.safetensors");
        write_safetensors(&path, &["lora.weight".to_owned()]);
        read_safetensors_header(&path).expect("empty-tensor file is accepted");
    }

    #[test]
    fn max_tensor_data_end_skips_metadata_and_takes_max() {
        let header = json!({
            "__metadata__": {"format": "pt"},
            "a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 100]},
            "b": {"dtype": "F16", "shape": [1], "data_offsets": [100, 420]},
        });
        assert_eq!(max_tensor_data_end(&header), 420);
        assert_eq!(max_tensor_data_end(&json!({"__metadata__": {"x": "y"}})), 0);
    }

    #[test]
    fn sensenova_u1_image_family_supports_text_to_image_and_edit() {
        let caps = model_capabilities_for_type_and_family("image", "sensenova-u1");
        assert!(caps.contains(&"text_to_image"));
        assert!(caps.contains(&"edit_image"));
    }

    // --- LoRA family-compat validation (sc-3027) ---

    #[test]
    fn accepted_lora_families_includes_extra_compatible() {
        assert_eq!(accepted_lora_families("flux"), vec!["flux".to_owned()]);
        // chroma additionally accepts flux; flux2-klein accepts flux2.
        assert_eq!(
            accepted_lora_families("chroma"),
            vec!["chroma".to_owned(), "flux".to_owned()]
        );
        assert_eq!(
            accepted_lora_families("flux2_klein"),
            vec!["flux2-klein".to_owned(), "flux2".to_owned()]
        );
        assert!(accepted_lora_families("").is_empty());
    }

    #[test]
    fn lora_declared_families_reads_first_present_source() {
        assert_eq!(
            lora_declared_families(&json!({ "family": "FLUX" })),
            vec!["flux".to_owned()]
        );
        assert_eq!(
            lora_declared_families(&json!({ "compatibility": { "families": ["sdxl"] } })),
            vec!["sdxl".to_owned()]
        );
        // `families` wins over `family`; normalized + de-duplicated + sorted.
        assert_eq!(
            lora_declared_families(&json!({ "families": ["flux2", "Flux2"], "family": "sdxl" })),
            vec!["flux2".to_owned()]
        );
        assert!(lora_declared_families(&json!({ "id": "x" })).is_empty());
    }

    #[test]
    fn validate_lora_compatibility_accepts_matching_and_extra_compatible() {
        // exact family
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "flux" })],
            Some("flux"),
            "mlx_flux",
            Some("flux_dev"),
        )
        .is_ok());
        // flux2-klein model accepts a flux2 LoRA
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "flux2" })],
            Some("flux2-klein"),
            "mlx_flux2",
            Some("flux2_klein_9b"),
        )
        .is_ok());
        // unstamped LoRA (no declared family) is skipped, not rejected
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a" })],
            Some("sdxl"),
            "mlx_sdxl",
            Some("sdxl"),
        )
        .is_ok());
        // unknown model family → skip (no accepted set)
        assert!(validate_lora_compatibility(
            &[json!({ "id": "a", "family": "flux" })],
            None,
            "mlx_x",
            None,
        )
        .is_ok());
    }

    #[test]
    fn validate_lora_compatibility_rejects_incompatible_family() {
        let err = validate_lora_compatibility(
            &[json!({ "id": "fluxlora", "family": "flux" })],
            Some("sdxl"),
            "mlx_sdxl",
            Some("sdxl"),
        )
        .unwrap_err();
        assert!(err.contains("fluxlora"), "got: {err}");
        assert!(err.contains("sdxl"), "got: {err}");
    }

    #[test]
    fn validate_lora_compatibility_gates_wan_base_model() {
        // Same family (wan-video) but a LoRA trained for a different base model is rejected.
        let err = validate_lora_compatibility(
            &[json!({ "id": "w", "family": "wan-video", "baseModel": "wan_2_2_t2v_14b" })],
            Some("wan-video"),
            "mlx_wan",
            Some("wan_2_2"),
        )
        .unwrap_err();
        assert!(err.contains("not interchangeable"), "got: {err}");
        // A matching base model passes; a LoRA without a base model falls back to family.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "w", "family": "wan-video", "baseModel": "wan_2_2" })],
            Some("wan-video"),
            "mlx_wan",
            Some("wan_2_2"),
        )
        .is_ok());
        assert!(validate_lora_compatibility(
            &[json!({ "id": "w", "family": "wan-video" })],
            Some("wan-video"),
            "mlx_wan",
            Some("wan_2_2"),
        )
        .is_ok());
    }

    #[test]
    fn validate_lora_compatibility_accepts_krea_raw_lora_on_turbo() {
        // epic 7565 P3 (sc-7578): a Krea LoRA trained on Krea 2 Raw records
        // `family: krea_2` / `baseModel: krea_2_raw` and applies at Krea 2 Turbo inference
        // by family match. `krea_2` is NOT base-model-gated (only wan-video is), so the Raw
        // base model differing from the served Turbo model id does NOT reject — the Lens /
        // Z-Image train-on-base → infer-on-Turbo precedent.
        assert!(validate_lora_compatibility(
            &[json!({ "id": "k", "family": "krea_2", "baseModel": "krea_2_raw" })],
            Some("krea_2"),
            "mlx_krea",
            Some("krea_2_turbo"),
        )
        .is_ok());
        // A foreign-family LoRA is still rejected on the Krea Turbo model.
        let err = validate_lora_compatibility(
            &[json!({ "id": "sdxllora", "family": "sdxl" })],
            Some("krea_2"),
            "mlx_krea",
            Some("krea_2_turbo"),
        )
        .unwrap_err();
        assert!(err.contains("sdxllora"), "got: {err}");
    }
}
