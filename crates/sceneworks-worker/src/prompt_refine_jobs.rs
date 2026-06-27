//! Native prompt refinement (epic 5095): candle on Windows/CUDA (sc-5525) + MLX on macOS (sc-5552).
//!
//! Routes the `prompt_refine` job to a native LLM provider through the unified LLM engine (epic 7153)
//! — a generic `core_llm::TextLlm` resolved model-first via `gen_core::core_llm::load_for_model`
//! (Anubis-Mini-8B, sc-6550), so the dispatch body is one backend-agnostic path: macOS (sc-7158) picks
//! mlx-llm's `mlx-llama`, the Windows/CUDA candle build (sc-7404) picks candle-llm's `candle-llama`.
//! Both retired their bespoke hand-rolled Llama decoders (`mlx-gen-prompt-refine` /
//! `candle-gen-prompt-refine`). The Python torch `PromptRefiner`
//! (`apps/worker/scene_worker/prompt_refine.py`) stays the fallback only on platforms with neither
//! native provider (e.g. the candle-less Desktop installer).
//!
//! The `TextLlm` contract is generic (`system` + `prompt` + sampling → text), so the
//! prompt-refinement PRODUCT logic that lived in `prompt_refine.py` moves here caller-side: the
//! rewrite rules + image/video medium switch + guide assembly (`build_refine_system_prompt`, into the
//! request `system`) and the reasoning-block / code-fence / surrounding-quote cleanup
//! (`clean_refine_output`, over the model reply). Sampling matches the Python path (temperature 0.7,
//! top_p 0.9, max_new_tokens 512), as does the empty-output → error behavior and the `{originalPrompt,
//! refinedPrompt}` result shape.

use super::*;

// Prompt-refine provider force-link anchors: keep each backend's `inventory::submit!` provider
// registration from being dropped by the release linker, so model-first `core_llm::load_for_model`
// resolution can discover it. macOS (sc-7158, epic 7153) force-links `mlx_llm` for `mlx-llama`; the
// Windows/CUDA candle build (sc-7404) force-links `candle_llm` for `candle-llama` — both generic
// `core_llm::TextLlm` providers (retiring the bespoke `mlx-gen-prompt-refine` / `candle-gen-prompt-refine`).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_llm as _;
#[cfg(target_os = "macos")]
use mlx_llm as _;

// The prompt-refinement / magic-prompt checkpoint — the coherent Anubis-8B (sc-6550 bake-off). It
// serves BOTH the free-text "Refine my prompt" rewrite AND the Ideogram magic-prompt JSON caption:
// the bake-off found the old 3B (and plain Llama-3.1-8B, stock + abliterated) stochastically emit
// SEMANTICALLY-DEGENERATE captions (subject as a `text` element, a reflexive transparent background)
// that placeholder Ideogram 4 at 1024²/48, while Anubis avoids them. Loads on this same config-driven
// Llama seam with no conversion (stock bf16, ~16GB). Overridable per-job via `payload.model`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const DEFAULT_REFINE_MODEL: &str = "TheDrummer/Anubis-Mini-8B-v1";
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const CANCEL_MESSAGE: &str = "Prompt refinement canceled by user.";
// Output-length cap. The free-text rewrite is a one-liner (512 is ample), but the two caption tasks
// (magic-prompt + image_caption) emit a full Ideogram JSON caption — multi-element, with bboxes,
// #RRGGBB palettes and (since sc-8199) optional per-element palettes — which truncates well past 2048
// tokens on busy images (sc-8210: `EOF while parsing a list` mid-`elements`). 4096 ≈ ~11.6k chars of
// headroom; a well-formed caption emits EOS far below the cap, so a higher ceiling only rescues the
// truncating cases. Callers may still override via the `maxNewTokens` payload field.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const DEFAULT_CAPTION_MAX_NEW_TOKENS: u32 = 4096;
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const DEFAULT_REFINE_MAX_NEW_TOKENS: u32 = 512;
// Resolve the output token budget: an explicit positive `maxNewTokens` override wins; otherwise the
// per-task default (caption tasks need the larger cap so the JSON closes).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn resolve_max_new_tokens(payload: &serde_json::Map<String, Value>, is_caption_task: bool) -> u32 {
    payload
        .get("maxNewTokens")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(if is_caption_task {
            DEFAULT_CAPTION_MAX_NEW_TOKENS
        } else {
            DEFAULT_REFINE_MAX_NEW_TOKENS
        })
}
// Architecture-pill label for the streamed progress (mirrors the candle image/video paths): the MLX
// twin on macOS, candle on the Windows candle build.
#[cfg(target_os = "macos")]
const REFINE_BACKEND: &str = "mlx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const REFINE_BACKEND: &str = "candle";

// ----------------------------------------------------------------------------------------------
// Product logic (pure, platform-independent) — ported from `prompt_refine.py` so the native worker
// (candle + MLX) owns the prompt assembly + reply cleanup the generic `TextLlm` contract does not.
// Compiled in the default `cargo test` gate (so the unit tests below run on every lane) and on the
// macOS + candle builds.
// ----------------------------------------------------------------------------------------------

/// The base rewrite rules with the `{medium}` placeholders filled (`image` / `video`). Verbatim port
/// of the Python `_BASE_RULES.format(medium=…)`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn base_rules(medium: &str) -> String {
    [
        format!("You are a prompt rewriter for a generative {medium} model."),
        format!(
            "Rewrite the user's input into a single, precise {medium} prompt that follows the \
             model's prompt guide below."
        ),
        String::new(),
        "Rules:".to_owned(),
        "- Output exactly one rewritten prompt and nothing else — no explanations, reasoning, \
         commentary, options, or labels."
            .to_owned(),
        format!(
            "- Preserve the user's intent: do not change the subjects, attributes, actions, \
             relationships, or core setting they described. You may add concrete details only when \
             they make the {medium} more coherent and stay consistent with the user's meaning."
        ),
        "- If the user's prompt is already detailed and on-guide, make only minimal edits for \
         fluency."
            .to_owned(),
        "- Follow the guide's recommended structure, phrasing, and what-to-avoid guidance."
            .to_owned(),
        "- Match the user's language: if their prompt is not in English, respond in the same \
         language."
            .to_owned(),
        "- Do not wrap the output in quotes, markdown, JSON, or code fences unless those are part \
         of the described scene."
            .to_owned(),
    ]
    .join("\n")
}

/// Build the `system` message for the refiner: the rewrite rules (medium chosen from the workflow)
/// plus the model's prompt guide when one is supplied. Port of the Python `build_system_prompt`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn build_refine_system_prompt(guide: Option<&str>, workflow: Option<&str>) -> String {
    let medium = if workflow
        .map(|w| w.trim().eq_ignore_ascii_case("video"))
        .unwrap_or(false)
    {
        "video"
    } else {
        "image"
    };
    let rules = base_rules(medium);
    let guide = guide.unwrap_or("").trim();
    if guide.is_empty() {
        rules
    } else {
        format!("{rules}\n\n# Model prompt guide\n\n{guide}")
    }
}

/// Strip `<think>…</think>` reasoning blocks, a wrapping code fence, and matching surrounding quotes
/// from the model reply. Port of the Python `clean_output` (regex-free: the tags are ASCII, matched
/// case-insensitively without lowercasing the whole — Unicode-safe — string).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn clean_refine_output(text: &str) -> String {
    let mut text = strip_think_blocks(text.trim()).trim().to_owned();
    // An orphan closing tag (no matching open): keep only what follows the last one.
    if let Some(pos) = last_ci(&text, "</think>") {
        text = text[pos + "</think>".len()..].trim().to_owned();
    }
    // A wrapping ```…``` code fence: drop the fence lines.
    if text.starts_with("```") && text.ends_with("```") {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 2 {
            text = lines[1..lines.len() - 1].join("\n").trim().to_owned();
        }
    }
    // Matching surrounding single/double quotes.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 2 {
        let (first, last) = (chars[0], chars[chars.len() - 1]);
        if first == last && (first == '"' || first == '\'') {
            text = chars[1..chars.len() - 1]
                .iter()
                .collect::<String>()
                .trim()
                .to_owned();
        }
    }
    text
}

/// Remove every `<think>…</think>` pair (case-insensitive, spanning newlines). An unmatched open tag
/// leaves the remainder untouched — matching the Python non-greedy regex, which simply does not match.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn strip_think_blocks(input: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        match first_ci(rest, OPEN) {
            Some(open) => {
                out.push_str(&rest[..open]);
                let after_open = &rest[open + OPEN.len()..];
                match first_ci(after_open, CLOSE) {
                    Some(close) => rest = &after_open[close + CLOSE.len()..],
                    None => {
                        out.push_str(&rest[open..]);
                        return out;
                    }
                }
            }
            None => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

/// Byte offset of the first case-insensitive occurrence of an ASCII `needle`. Offsets land on ASCII
/// tag boundaries, so callers can slice safely even when the surrounding text is Unicode.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn first_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// Byte offset of the last case-insensitive occurrence of an ASCII `needle`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn last_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len())
        .rev()
        .find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

// ----------------------------------------------------------------------------------------------
// Magic-prompt expansion (epic 4725, sc-5997) — plain idea → structured JSON caption. Drives the
// SAME `prompt_refine` TextLlm (Llama-3.2-3B) with Ideogram's open-source magic-prompt system prompt
// (`task: "magic_prompt"`) instead of the rewrite rules. The hosted Ideogram / OpenRouter Sonnet/Opus
// configs in the reference are replaced by the local model (native-first, offline). The caller (web)
// strips the non-schema `aspect_ratio` key + bboxes and validates the result against the sc-5993
// caption contract, so this side stays generic: build messages, run, extract the JSON object.
// ----------------------------------------------------------------------------------------------

/// Ideogram 4's magic-prompt system-prompt file, embedded verbatim. Source (Apache-2.0):
/// github.com/ideogram-oss/ideogram4 `src/ideogram4/magic_prompt_system_prompts/v1.txt`. Parsed for
/// its `[SYSTEM]` + `[USER]` sections at runtime (the `[META]` block is ignored).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const MAGIC_PROMPT_V1: &str = include_str!("ideogram_magic_prompt_v1.txt");

/// Ideogram 4's image-grounded caption system prompt (epic 8102, sc-8105), embedded verbatim and
/// versioned like [`MAGIC_PROMPT_V1`]. Drives the SAME `prompt_refine` TextLlm seam — but through the
/// `core_llm` vision path: the model EXAMINES a reference image and emits a schema-valid Ideogram JSON
/// caption (style + grounded composition, bboxes kept) instead of expanding a text idea. Parsed for its
/// `[SYSTEM]` + `[USER]` sections at runtime by the same [`magic_section_from`] parser (the `[META]`
/// block is ignored).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const IMAGE_CAPTION_V1: &str = include_str!("ideogram_image_caption_v1.txt");

/// Body of a `[NAME]` section in the magic-prompt file (port of the reference `_load_sections`):
/// section markers are a bracketed single word alone on a line. Returns the trimmed body, or empty.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn magic_section(name: &str) -> String {
    magic_section_from(MAGIC_PROMPT_V1, name)
}

/// Body of a `[NAME]` section in ANY bracketed-marker prompt file (the magic-prompt or the
/// image-caption asset). Section markers are a bracketed single word alone on a line. Returns the
/// trimmed body, or empty. Generalized from `magic_section` so the sc-8105 image-caption asset reuses
/// the same parser.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn magic_section_from(source: &str, name: &str) -> String {
    let mut capturing = false;
    let mut body: Vec<&str> = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let is_marker = trimmed.len() >= 2
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && !trimmed.contains(' ');
        if is_marker {
            if capturing {
                break;
            }
            capturing = trimmed[1..trimmed.len() - 1].eq_ignore_ascii_case(name);
            continue;
        }
        if capturing {
            body.push(line);
        }
    }
    body.join("\n").trim().to_owned()
}

/// The `(system, user)` chat messages for a magic-prompt expansion: the `[SYSTEM]` block, and the
/// `[USER]` template with `{{aspect_ratio}}` / `{{original_prompt}}` substituted (port of the
/// reference `build_messages`). `aspect_ratio` is `"W:H"`.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn build_magic_prompt_messages(prompt: &str, aspect_ratio: &str) -> (String, String) {
    let system = magic_section("SYSTEM");
    let mut user = magic_section("USER");
    if user.is_empty() {
        user = "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height).".to_owned();
    }
    user = user.replace("{{aspect_ratio}}", aspect_ratio);
    user = if user.contains("{{original_prompt}}") {
        user.replace("{{original_prompt}}", prompt)
    } else {
        format!("{user}\n\n{prompt}")
    };
    (system, user)
}

// ----------------------------------------------------------------------------------------------
// Image-grounded caption (epic 8102, sc-8105) — reference image → structured Ideogram JSON caption.
// Drives the SAME `prompt_refine` TextLlm seam (`task: "image_caption"`) through the `core_llm` VISION
// path: the user turn carries a `Content::Image(ImageRef)` alongside the `[USER]` instruction, and the
// `[SYSTEM]` block is the embedded `ideogram_image_caption_v1.txt` (observe-don't-populate, style +
// grounded composition, bboxes kept). The image drives the multimodal path at GENERATE time
// (`mlx-llama`'s loaded Qwen-VL vision tower reads the `Content::Image`). At RESOLUTION time the worker
// must NOT demand vision: core-llm's `select`/`meets` filters on each provider's STATIC descriptor, and
// no linked provider statically advertises BOTH vision and Json for a Qwen-VL snapshot (`mlx-llama` is
// text+Json until it loads; `mlx-joycaption` is vision-but-no-constraints and only serves LLaVA). So the
// worker resolves on the JSON constraint ALONE — that selects `mlx-llama`, which loads the Qwen-VL
// snapshot and flips to vision at load. The reply is cleaned with the SAME `clean_json_output` +
// validated with `sceneworks_core::ideogram_caption::is_caption`; malformed output is handled like
// magic-prompt (the empty/non-caption result errors caller-side).
// ----------------------------------------------------------------------------------------------

/// The `(system, user)` chat text for an image-caption run: the `[SYSTEM]` + `[USER]` blocks of the
/// embedded `ideogram_image_caption_v1.txt`. The reference image is attached to the user turn as a
/// separate `Content::Image` block by the caller (this only supplies the text).
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn build_image_caption_messages() -> (String, String) {
    let system = magic_section_from(IMAGE_CAPTION_V1, "SYSTEM");
    let mut user = magic_section_from(IMAGE_CAPTION_V1, "USER");
    if user.is_empty() {
        user = "Examine the reference image attached to this message and emit the single JSON \
                caption object as specified. Describe only what is visible in the image."
            .to_owned();
    }
    (system, user)
}

/// Decode a reference image off disk into a `core_llm::ImageRef` (RGB8), mirroring the JoyCaption
/// `load_caption_image` pattern (`decode_image_any` → `to_rgb8`) but producing the vision contract's
/// tensor-free image type instead of the captioner's `gen_core::Image`. Used by the `image_caption`
/// task to build the `Content::Image` block.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn load_caption_image_ref(path: &Path) -> WorkerResult<gen_core::core_llm::ImageRef> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "image-caption reference image {}: {error}",
                path.display()
            ))
        })?
        .to_rgb8();
    let (width, height) = (decoded.width(), decoded.height());
    gen_core::core_llm::ImageRef::new(width, height, decoded.into_raw()).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "image-caption reference image {}: {error}",
            path.display()
        ))
    })
}

/// Reduce a magic-prompt reply to its JSON object: strip `<think>` blocks and a wrapping code fence
/// (reusing the refine cleanup), then take the outermost `{ … }` span so leading/trailing prose from
/// a small model is dropped. The caller parses + validates; here we only isolate the object.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn clean_json_output(text: &str) -> String {
    let mut text = strip_think_blocks(text.trim()).trim().to_owned();
    if let Some(pos) = last_ci(&text, "</think>") {
        text = text[pos + "</think>".len()..].trim().to_owned();
    }
    if text.starts_with("```") && text.ends_with("```") {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 2 {
            text = lines[1..lines.len() - 1].join("\n").trim().to_owned();
        }
    }
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if end > start => text[start..=end].to_owned(),
        _ => text,
    }
}

// ----------------------------------------------------------------------------------------------
// Job handler — native MLX on macOS (sc-5552 / sc-7158) and candle on the Windows candle build
// (sc-5525 / sc-7404). The body is backend-agnostic: `core_llm::load_for_model` resolves whichever
// provider is force-linked above (mlx-llama on macOS, candle-llama on the candle build) model-first.
// The Python torch `PromptRefiner` remains the fallback on other platforms.
// ----------------------------------------------------------------------------------------------

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) async fn run_prompt_refine_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    // The cooperative cancel handle is created here and threaded into the request built inside the
    // blocking task. Both lanes now run through the unified `core_llm` contract, so this is core-llm's
    // CancelFlag (via the gen_core re-export) on both — new()/clone()/cancel()/is_cancelled() — and the
    // dispatch body below is a single backend-agnostic path.
    use gen_core::core_llm::CancelFlag;

    let payload = &job.payload;
    // The job is dispatched by the `task` discriminator: `magic_prompt` (text idea → JSON caption,
    // sc-5997), `image_caption` (reference image → JSON caption, sc-8105 — the `core_llm` VISION path),
    // or the default free-text rewrite. The two caption tasks both emit a JSON-constrained Ideogram
    // caption; `image_caption` additionally carries an image and needs no text prompt.
    let task = payload
        .get("task")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    let is_magic = task.eq_ignore_ascii_case("magic_prompt");
    let is_image_caption = task.eq_ignore_ascii_case("image_caption");
    let original_prompt = payload
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    // The image-caption task is driven by the reference image, not a text prompt, so it does not
    // require a `prompt`; every other task does.
    if original_prompt.is_empty() && !is_image_caption {
        return Err(WorkerError::InvalidPayload(
            "Prompt refinement requires a non-empty prompt.".to_owned(),
        ));
    }
    // The image-caption task resolves + decodes a reference image (the JoyCaption `load_caption_image`
    // pattern → RGB8) into the vision contract's `ImageRef`. Accept either `imagePath` or `referencePath`.
    // The path is UNTRUSTED (it arrives on the job payload over the LAN-remote API boundary, epic 4484),
    // so — like every other on-disk image/model input (JoyCaption via `resolve_dataset_item_path`, the
    // InstantID/captioner reference reads, the LoRA load path) — confine it to an app-managed root via
    // `normalize_app_managed_model_path` BEFORE opening it. That rejects `..` traversal and any absolute
    // path outside the app data dir / HF hub cache, closing the arbitrary-file-read gap.
    let image_ref = if is_image_caption {
        let image_path = payload
            .get("imagePath")
            .or_else(|| payload.get("referencePath"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "Image caption requires a non-empty `imagePath` reference image.".to_owned(),
                )
            })?;
        let safe_path = normalize_app_managed_model_path(
            settings,
            image_path,
            "Image caption reference image",
        )?;
        Some(load_caption_image_ref(&safe_path)?)
    } else {
        None
    };
    let guide = payload
        .get("guide")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let workflow = payload
        .get("workflow")
        .and_then(Value::as_str)
        .map(str::to_owned);
    // A caption task (magic-prompt OR image-caption) drives the same TextLlm seam with Ideogram's
    // caption system prompt instead of the rewrite rules; captions run longer than a one-line prompt,
    // so allow more tokens and sample cooler for steadier JSON.
    let is_caption_task = is_magic || is_image_caption;
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_REFINE_MODEL)
        .to_owned();
    let max_new_tokens = resolve_max_new_tokens(payload, is_caption_task);
    let temperature = if is_caption_task { 0.4 } else { 0.7 };
    let work_message = if is_image_caption {
        "Captioning image…"
    } else if is_magic {
        "Expanding to a caption…"
    } else {
        "Refining prompt…"
    };
    let done_message = if is_caption_task {
        "Caption ready."
    } else {
        "Prompt refined."
    };

    let (system, user_message) = if is_image_caption {
        build_image_caption_messages()
    } else if is_magic {
        let aspect_ratio = payload
            .get("aspectRatio")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("1:1");
        build_magic_prompt_messages(&original_prompt, aspect_ratio)
    } else {
        (
            build_refine_system_prompt(guide.as_deref(), workflow.as_deref()),
            original_prompt.clone(),
        )
    };
    let weights_dir = resolve_app_managed_model_dir(settings, &model, "prompt-refine model path")?;
    // Attribute the run to the active backend (MLX on macOS, candle off-Mac) on the streamed progress
    // + UI architecture pill (mirrors the image/video paths), not the gpu-id device label.
    let backend = REFINE_BACKEND;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        refine_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.1,
            "Loading prompt-refinement model.",
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(u32, u32)>(64);
    let blocking_cancel = cancel.clone();
    let job_id = job.id.clone();
    let prompt = user_message;
    let engine_label = model.clone();
    let blocking = tokio::task::spawn_blocking(move || -> WorkerResult<String> {
        emit_event(
            "prompt_refine_load_start",
            json!({ "jobId": job_id, "engine": engine_label }),
        );

        // Resolve the native provider model-first (no provider id) and stream through the
        // `core_llm::TextLlm` contract. One backend-agnostic path: the force-linked provider
        // (mlx-llama on macOS, candle-llama on the Windows candle build) wins resolution. The provider
        // renders the model's own chat template, so the worker supplies only the system + user turns
        // (the product policy stays caller-side).
        //
        // sc-8105 resolution note: `core-llm`'s `select`/`meets` filters on each provider's STATIC
        // (weightless) descriptor BEFORE any `load` runs. `mlx-llama` statically advertises
        // `supports_vision:false` + `[Constraint::Json]` — it loads a Qwen-VL (`qwen3_5`) snapshot and
        // flips `supports_vision` on only at LOAD time (mlx-llm provider.rs:267). `mlx-joycaption`
        // statically advertises vision but NO constraints and only `can_load`s LLaVA (not Qwen-VL).
        // So NO provider statically satisfies BOTH vision AND Json for a Qwen-VL snapshot: demanding
        // `vision:true` at resolution (what `ModelRequirements::from_request` derives from the image
        // block) would make `select` return `Error::Unsupported` and never reach `load`. The image is
        // what drives the multimodal generate path at GENERATE time (LlamaProvider::generate gates on
        // its loaded `vision` tower), NOT what must be matched at resolution. So the image_caption path
        // resolves on the JSON constraint ALONE (no vision filter): that selects `mlx-llama`, which then
        // loads the Qwen-VL snapshot, flips to vision, and examines the `Content::Image`. (The other
        // tasks carry no image, so their `from_request` reqs never set the vision filter anyway.)
        let text = {
            use gen_core::core_llm::{
                load_for_model_with, Constraint, Content, LoadSpec, Message, ModelRequirements,
                Role, Sampling, StreamEvent, TextLlmRequest,
            };
            let mut messages = Vec::with_capacity(2);
            if !system.trim().is_empty() {
                messages.push(Message::system(system));
            }
            // The image-caption user turn carries the reference image (a `Content::Image` block)
            // alongside the instruction text, so the loaded provider examines the picture at generate
            // time. Every other task is a plain text user turn.
            let carries_image = image_ref.is_some();
            match image_ref {
                Some(image) => messages.push(Message {
                    role: Role::User,
                    content: vec![Content::Image(image), Content::text(prompt)],
                    thinking: None,
                    tool_calls: Vec::new(),
                }),
                None => messages.push(Message::user(prompt)),
            }
            let request = TextLlmRequest {
                messages,
                // The bespoke prompt-refine samplers were plain temperature/top-p (no repetition
                // penalty / top-k); core-llm's defaults match (top_k 0, repetition_penalty 1.0).
                sampling: Sampling {
                    temperature,
                    top_p: 0.9,
                    ..Sampling::default()
                },
                max_new_tokens,
                seed: None,
                // sc-6585 / sc-8105: a caption task (magic-prompt OR image-caption) must emit a
                // structurally-valid JSON caption, so constrain its decode to the JSON grammar; the
                // free-text rewrite is unconstrained. (On the candle lane this constraint actually
                // steers + masks the decode — the sc-7404 parity gain over `candle-gen-prompt-refine`.)
                constraint: is_caption_task.then_some(Constraint::Json),
                cancel: blocking_cancel.clone(),
                ..Default::default()
            };
            // Build the resolution requirements WITHOUT the auto-vision `from_request` derives from an
            // image block (see the resolution note above): a Qwen-VL snapshot has no statically
            // vision+Json provider, so demanding vision here would fail `select` before `load`. Require
            // only the request's output constraint (the JSON grammar for a caption task) — that selects
            // the text+Json `mlx-llama`, which loads the Qwen-VL snapshot and flips to vision at load.
            // (`carries_image` is asserted so the unused-binding lint stays satisfied and the intent —
            // "an image is present, yet we deliberately do NOT set the vision filter" — is explicit.)
            debug_assert!(carries_image == is_image_caption);
            let mut reqs = ModelRequirements::default();
            for constraint in request.constraint.iter().copied() {
                reqs = reqs.with_constraint(constraint);
            }
            let refiner = load_for_model_with(
                &LoadSpec {
                    source: weights_dir.to_string_lossy().into_owned(),
                    quantize: None,
                },
                &reqs,
            )
            .map_err(|error| WorkerError::Engine(format!("prompt-refine load failed: {error}")))?;
            emit_event(
                "prompt_refine_load_complete",
                json!({ "jobId": job_id, "engine": engine_label }),
            );
            if blocking_cancel.is_cancelled() {
                return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
            }
            // Drive the (current, total) progress channel the shared loop below reads, counting
            // generated tokens against the max-new-tokens budget.
            let mut on_event = |event: StreamEvent| {
                if let StreamEvent::Token { index, .. } = event {
                    let _ = tx.blocking_send((index as u32 + 1, max_new_tokens));
                }
            };
            let output = refiner.generate(&request, &mut on_event).map_err(|error| {
                WorkerError::Engine(format!("prompt-refine generation failed: {error}"))
            })?;
            output.text
        };

        Ok(text)
    });

    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some((current, total)) => {
                        let within = if total > 0 {
                            (current as f64 / total as f64).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        update_job(
                            api,
                            &job.id,
                            refine_progress(
                                JobStatus::Running,
                                ProgressStage::Running,
                                0.4 + 0.5 * within,
                                work_message,
                                None,
                                backend,
                            ),
                        )
                        .await?;
                    }
                    None => break,
                }
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                match check_cancel(api, &job.id, CANCEL_MESSAGE).await {
                    Ok(()) => {}
                    Err(WorkerError::Canceled(_)) => cancel.cancel(),
                    Err(error) => return Err(error),
                }
            }
        }
    }

    let raw = blocking
        .await
        .map_err(|error| task_join_error("prompt refine task join", error))??;
    // A caption task isolates the JSON object (the web parses + validates it; image_caption validates
    // here too); the free-text rewrite cleans to prose.
    let refined = if is_caption_task {
        clean_json_output(&raw)
    } else {
        clean_refine_output(&raw)
    };
    if refined.is_empty() {
        return Err(WorkerError::Engine(
            "The prompt-refinement model returned an empty prompt.".to_owned(),
        ));
    }
    // sc-8105: the image-caption path validates the cleaned reply is a schema-valid Ideogram caption
    // (carries `compositional_deconstruction`) and KEEPS the element bboxes (no stripping). A malformed
    // reply is handled like an empty one — a clear engine error the caller surfaces (mirroring the
    // magic-prompt malformed-output handling, where a non-caption reply also fails downstream).
    if is_image_caption {
        let parsed = serde_json::from_str::<Value>(&refined).map_err(|error| {
            WorkerError::Engine(format!(
                "The image-caption model returned output that is not valid JSON: {error}"
            ))
        })?;
        if !sceneworks_core::ideogram_caption::is_caption(&parsed) {
            return Err(WorkerError::Engine(
                "The image-caption model returned JSON that is not a valid Ideogram caption \
                 (missing the `compositional_deconstruction` section)."
                    .to_owned(),
            ));
        }
    }
    update_job(
        api,
        &job.id,
        refine_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            done_message,
            Some(refine_result(&original_prompt, &refined)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// On platforms with no native prompt-refine provider (neither the macOS MLX twin nor the Windows
/// candle build — e.g. Linux, or the candle-less Desktop installer), the capability is never
/// advertised and this arm is unreachable in practice — the Python torch `PromptRefiner` serves
/// `prompt_refine`. Kept so the `run_utility_job` dispatch compiles on all targets.
#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
pub(crate) async fn run_prompt_refine_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Native prompt refinement needs the macOS MLX worker or the Windows candle backend; use the \
         Python torch prompt refiner on this platform."
            .to_owned(),
    ))
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn refine_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        // Stamped by update_job before posting (sc-4172).
        worker_id: None,
        extra: BTreeMap::new(),
    }
}

/// The `prompt_refine` result payload, parity with the Python `run_prompt_refine_job`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn refine_result(original_prompt: &str, refined_prompt: &str) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("originalPrompt".to_owned(), json!(original_prompt));
    result.insert("refinedPrompt".to_owned(), json!(refined_prompt));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_new_tokens_defaults_and_override() {
        // Caption tasks (magic-prompt + image_caption) get the larger default so the JSON closes
        // (sc-8210: 2048 truncated rich captions mid-`elements`).
        let obj = |value: serde_json::Value| value.as_object().unwrap().clone();
        assert_eq!(
            resolve_max_new_tokens(&obj(serde_json::json!({})), true),
            4096
        );
        // The free-text rewrite stays at the small default.
        assert_eq!(
            resolve_max_new_tokens(&obj(serde_json::json!({})), false),
            512
        );
        // An explicit positive override wins for either task.
        assert_eq!(
            resolve_max_new_tokens(&obj(serde_json::json!({ "maxNewTokens": 6000 })), true),
            6000
        );
        // Zero / invalid overrides fall back to the per-task default.
        assert_eq!(
            resolve_max_new_tokens(&obj(serde_json::json!({ "maxNewTokens": 0 })), true),
            4096
        );
        assert_eq!(
            resolve_max_new_tokens(&obj(serde_json::json!({ "maxNewTokens": "nope" })), false),
            512
        );
    }

    #[test]
    fn system_prompt_uses_workflow_medium_and_embeds_guide() {
        let image = build_refine_system_prompt(
            Some("# Z-Image Guide\n\nUse short prompts."),
            Some("image"),
        );
        assert!(image.contains("generative image model"));
        assert!(image.contains("Z-Image Guide"));
        assert!(image.contains("# Model prompt guide"));

        let video = build_refine_system_prompt(None, Some("video"));
        assert!(video.contains("generative video model"));
        assert!(!video.contains("# Model prompt guide"));
    }

    #[test]
    fn system_prompt_defaults_to_image_when_workflow_absent_or_unknown() {
        assert!(build_refine_system_prompt(None, None).contains("generative image model"));
        assert!(
            build_refine_system_prompt(None, Some("anything")).contains("generative image model")
        );
        // Case-insensitive video match (parity with Python `.lower()`).
        assert!(
            build_refine_system_prompt(None, Some(" VIDEO ")).contains("generative video model")
        );
    }

    #[test]
    fn clean_output_strips_reasoning_and_quoting() {
        assert_eq!(
            clean_refine_output("<think>plan</think>A vivid sunset over hills."),
            "A vivid sunset over hills."
        );
        assert_eq!(
            clean_refine_output("\"A vivid sunset over hills.\""),
            "A vivid sunset over hills."
        );
        assert_eq!(
            clean_refine_output("```\nA vivid sunset over hills.\n```"),
            "A vivid sunset over hills."
        );
        assert_eq!(
            clean_refine_output("<think>scheming</think>A vivid neon street at midnight."),
            "A vivid neon street at midnight."
        );
    }

    #[test]
    fn magic_section_extracts_system_and_user_blocks() {
        let system = magic_section("SYSTEM");
        assert!(
            system.contains("structured JSON caption"),
            "system has the contract intro"
        );
        assert!(
            system.contains("compositional_deconstruction"),
            "system names the schema"
        );
        // The [META] block above [SYSTEM] is not leaked into the system body.
        assert!(!system.contains("thinking_mode"));

        let user = magic_section("USER");
        assert!(
            user.contains("{{aspect_ratio}}"),
            "user template has the aspect-ratio placeholder"
        );
        assert!(
            user.contains("{{original_prompt}}"),
            "user template has the prompt placeholder"
        );

        assert_eq!(magic_section("DOES_NOT_EXIST"), "");
    }

    #[test]
    fn build_magic_prompt_messages_substitutes_template() {
        let (system, user) = build_magic_prompt_messages("a red fox in the snow", "16:9");
        assert!(system.contains("OUTPUT CONTRACT"));
        assert!(user.contains("16:9"));
        assert!(user.contains("a red fox in the snow"));
        // Both placeholders are consumed.
        assert!(!user.contains("{{"));
    }

    #[test]
    fn clean_json_output_isolates_the_object() {
        // Leading/trailing prose from a small model is dropped.
        assert_eq!(
            clean_json_output("Here is the caption: {\"a\": 1} — hope that helps!"),
            "{\"a\": 1}"
        );
        // Code fences + a think block are stripped.
        assert_eq!(
            clean_json_output("<think>plan</think>```json\n{\"a\": 1}\n```"),
            "{\"a\": 1}"
        );
        // Already-clean object passes through.
        assert_eq!(clean_json_output("{\"a\": [1, 2]}"), "{\"a\": [1, 2]}");
    }

    // ── sc-8105: image_caption task (reference image → Ideogram JSON caption, core_llm vision path) ──

    #[test]
    fn image_caption_asset_extracts_system_and_user_blocks() {
        // The embedded `ideogram_image_caption_v1.txt` parses with the same bracketed-marker parser as
        // the magic-prompt asset; the `[META]` block must not leak into the system body.
        let system = magic_section_from(IMAGE_CAPTION_V1, "SYSTEM");
        assert!(
            system.contains("vision captioner"),
            "system declares the image-examination role"
        );
        assert!(
            system.contains("style_description"),
            "system names the style block"
        );
        assert!(
            system.contains("compositional_deconstruction"),
            "system names the composition schema"
        );
        assert!(
            system.contains("OBSERVE, DON'T POPULATE"),
            "system carries the observe-don't-populate discipline"
        );
        // The bbox convention is pinned: `[y1, x1, y2, x2]`, 0–1000, and bboxes are KEPT (not stripped).
        assert!(system.contains("[y1, x1, y2, x2]"));
        assert!(system.contains("1000"));
        assert!(
            system.contains("KEEP these bboxes"),
            "system pins that grounded bboxes are kept"
        );
        // sc-8194: the schema requires EXACTLY ONE of `style_description.photo` / `art_style`
        // (both present OR neither present = validation error). The prompt must make that mandatory.
        assert!(
            system.contains("EXACTLY ONE"),
            "system mandates exactly one style discriminator"
        );
        assert!(
            system.contains("never both, never neither"),
            "system forbids emitting both or neither discriminator key"
        );
        assert!(system.contains("photo"));
        assert!(system.contains("art_style"));
        // sc-8197: `color_palette` must be #RRGGBB hex codes (the web schema's `verifyColorPalette`
        // rejects color names). The prompt must require uppercase hex codes.
        assert!(
            system.contains("#RRGGBB"),
            "system requires color_palette as #RRGGBB hex codes (not color names)"
        );
        // sc-8199: the STYLE palette now allows richer palettes (up to 16, STYLE_PALETTE_MAX),
        // restored from the mistaken cap of 5.
        assert!(
            system.contains("up to 16"),
            "system allows the style color_palette up to 16 entries"
        );
        // sc-8199: each ELEMENT may carry its own optional `color_palette` (last key). The element
        // format example must show the optional per-subject palette.
        assert!(
            system.contains("\"color_palette\":[\"#RRGGBB\""),
            "element format example carries an optional per-subject color_palette"
        );
        // sc-8199: the per-element palette is capped at 5 (ELEMENT_PALETTE_MAX); the `MAXIMUM 5`
        // copy now lives in the element guidance subsection.
        assert!(
            system.contains("MAXIMUM 5"),
            "system caps the element color_palette at a maximum of 5 colors"
        );
        // Output is fenced so it survives markdown.
        assert!(system.contains("```json"));
        // The `[META]` thinking flag does not bleed into the system body.
        assert!(!system.contains("thinking_mode"));

        let user = magic_section_from(IMAGE_CAPTION_V1, "USER");
        assert!(
            user.to_lowercase().contains("reference image"),
            "user turn instructs examining the reference image"
        );
        // A missing section yields empty (parser contract).
        assert_eq!(magic_section_from(IMAGE_CAPTION_V1, "DOES_NOT_EXIST"), "");
    }

    #[test]
    fn build_image_caption_messages_returns_system_and_user() {
        let (system, user) = build_image_caption_messages();
        assert!(!system.trim().is_empty(), "system block is non-empty");
        assert!(system.contains("compositional_deconstruction"));
        assert!(!user.trim().is_empty(), "user block is non-empty");
        // The builder takes no aspect ratio / prompt placeholders (the image is the input).
        assert!(!user.contains("{{"));
    }

    #[test]
    fn clean_json_output_unwraps_a_fenced_image_caption_keeping_bboxes() {
        // The image-caption asset emits a ```json fence; the SAME cleanup the magic-prompt path uses
        // unwraps it. Element bboxes survive (they are NOT stripped on the worker side — sc-8105).
        let fenced = "```json\n{\"high_level_description\": \"a red fox in snow\", \
             \"compositional_deconstruction\": {\"background\": \"a snowy forest\", \
             \"elements\": [{\"type\": \"obj\", \"bbox\": [100, 200, 800, 700], \"desc\": \"a red fox\"}]}}\n```";
        let cleaned = clean_json_output(fenced);
        let parsed: Value = serde_json::from_str(&cleaned).expect("cleaned reply is valid JSON");
        assert!(
            sceneworks_core::ideogram_caption::is_caption(&parsed),
            "cleaned fenced reply is a schema-valid Ideogram caption"
        );
        // The element bbox is preserved (worker does not strip image-grounded boxes).
        let bbox = parsed["compositional_deconstruction"]["elements"][0]
            .get("bbox")
            .expect("element keeps its bbox");
        assert_eq!(bbox, &serde_json::json!([100, 200, 800, 700]));
    }

    #[test]
    fn image_caption_malformed_output_fails_is_caption() {
        // A non-caption JSON object (no `compositional_deconstruction`) is rejected — the worker's
        // malformed-output guard, mirroring how the magic-prompt path's non-captions fail downstream.
        let not_a_caption: Value = serde_json::from_str(&clean_json_output(
            "```json\n{\"high_level_description\": \"x\"}\n```",
        ))
        .expect("valid JSON");
        assert!(!sceneworks_core::ideogram_caption::is_caption(
            &not_a_caption
        ));

        // Non-JSON / refusal text isolates nothing parseable as a caption.
        let refusal = clean_json_output("I cannot describe this image.");
        assert!(!serde_json::from_str::<Value>(&refusal)
            .map(|v| sceneworks_core::ideogram_caption::is_caption(&v))
            .unwrap_or(false));
    }

    #[test]
    fn load_caption_image_ref_decodes_png_to_rgb8_image_ref() {
        use gen_core::core_llm::ImageRef;
        // Encode a tiny 2×3 RGB PNG to a temp file, then decode it through the image-caption loader and
        // assert the `ImageRef` dimensions + RGB8 byte count (width·height·3). This exercises the
        // `decode_image_any → to_rgb8 → ImageRef::new` path without weights.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ref.png");
        let mut img = image::RgbImage::new(2, 3);
        for (i, px) in img.pixels_mut().enumerate() {
            *px = image::Rgb([(i * 10) as u8, (i * 20) as u8, (i * 30) as u8]);
        }
        img.save(&path).expect("write png");

        let image_ref: ImageRef = load_caption_image_ref(&path).expect("decode reference image");
        assert_eq!(image_ref.width, 2);
        assert_eq!(image_ref.height, 3);
        assert_eq!(image_ref.pixels.len(), 2 * 3 * 3);
    }

    #[test]
    fn image_caption_reference_path_confined_to_app_managed_root() {
        // Issue 2 (path containment): the image_caption reference path is UNTRUSTED (job payload, over
        // the LAN-remote API boundary), so it must resolve under an app-managed root before it is read.
        // A path INSIDE `data_dir` is accepted; a sibling path OUTSIDE it, and a `..`-traversal escape,
        // are rejected with a clear error — mirroring the JoyCaption `resolve_dataset_item_path` guard.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut settings = crate::Settings::from_env();
        settings.data_dir = dir.path().to_path_buf();

        // In-root: accepted (the on-disk file need not exist — confinement is path-structural).
        let inside = dir.path().join("projects").join("ref.png");
        let resolved = normalize_app_managed_model_path(
            &settings,
            &inside.to_string_lossy(),
            "Image caption reference image",
        )
        .expect("an in-root reference path is accepted");
        assert!(resolved.starts_with(dir.path()), "stays under the data dir");

        // Out-of-root absolute path: rejected.
        let outside_dir = tempfile::tempdir().expect("tempdir2");
        let outside = outside_dir.path().join("secret.png");
        let err = normalize_app_managed_model_path(
            &settings,
            &outside.to_string_lossy(),
            "Image caption reference image",
        )
        .expect_err("an out-of-root reference path is rejected");
        assert!(
            err.to_string().contains("Image caption reference image"),
            "{err}"
        );

        // `..` traversal escaping the data dir: rejected.
        let traversal = format!("{}/../escape.png", dir.path().display());
        let err = normalize_app_managed_model_path(
            &settings,
            &traversal,
            "Image caption reference image",
        )
        .expect_err("a `..`-traversal reference path is rejected");
        assert!(
            err.to_string().contains("Image caption reference image")
                || err.to_string().contains("Unsafe absolute path"),
            "{err}"
        );
    }

    #[test]
    fn image_caption_request_carries_image_for_generate_but_resolves_json_only() {
        // Assemble the same request the job's blocking task builds for an image_caption run and assert
        // the contract surface: the user turn carries the reference image (so the LOADED provider's
        // vision tower reads it at GENERATE time) and a JSON constraint. CRUCIALLY, resolution must NOT
        // demand vision — `ModelRequirements::from_request` WOULD set `vision:true` (which makes
        // core-llm `select` fail for a Qwen-VL snapshot, since no provider statically advertises both
        // vision and Json), so the worker instead builds requirements from the JSON constraint ALONE.
        // This test pins the worker's resolution-requirements construction (the actual fix), and the
        // contrast against `from_request`.
        use gen_core::core_llm::{
            Constraint, Content, ImageRef, Message, ModelRequirements, Role, Sampling,
            TextLlmRequest,
        };

        let (system, user) = build_image_caption_messages();
        let image = ImageRef::new(2, 2, vec![0u8; 12]).expect("image ref");
        let request = TextLlmRequest {
            messages: vec![
                Message::system(system),
                Message {
                    role: Role::User,
                    content: vec![Content::Image(image), Content::text(user)],
                    thinking: None,
                    tool_calls: Vec::new(),
                },
            ],
            sampling: Sampling {
                temperature: 0.4,
                top_p: 0.9,
                ..Sampling::default()
            },
            max_new_tokens: 2048,
            seed: None,
            constraint: Some(Constraint::Json),
            ..Default::default()
        };

        assert!(request.has_image(), "request carries the reference image");

        // `from_request` over-constrains: it derives `vision:true` from the image block. The worker must
        // NOT use this for the image_caption resolution (it would make `select` return Unsupported for a
        // Qwen-VL snapshot — see the weightless resolution tests below).
        let auto = ModelRequirements::from_request(&request);
        assert!(
            auto.vision,
            "from_request over-constrains by demanding vision (the bug the fix avoids)"
        );

        // The worker's resolution requirements: the request's output constraint ONLY, no vision filter.
        let mut reqs = ModelRequirements::default();
        for constraint in request.constraint.iter().copied() {
            reqs = reqs.with_constraint(constraint);
        }
        assert!(
            !reqs.vision,
            "image_caption resolution must NOT demand vision (it is acquired at LOAD time)"
        );
        assert!(
            reqs.constraints.contains(&Constraint::Json),
            "image_caption resolution still demands JSON-constrained decoding"
        );
    }

    /// Weightless resolution proof — UPDATED for the sc-8171 engine bump (mlx-llm 6c052ba → 7041411).
    /// On the OLD engine, demanding `vision:true` + `[Json]` for a Qwen-VL (`qwen3_5`) snapshot FAILED
    /// `select` with `Error::Unsupported`: `mlx-llama`'s STATIC descriptor was `supports_vision:false`
    /// (vision flipped on only at LOAD), so no provider statically satisfied both vision and Json. The
    /// bump adds the per-snapshot `weightless_vision` probe (mlx-llm provider.rs / core-llm sc-8077): the
    /// resolver now recognizes a `qwen3_5`/`qwen3_vl` wrapper as vision-capable from `config.json` ALONE,
    /// so vision+Json RESOLVES `mlx-llama` and reaches `load` (which then fails only on the missing
    /// weight shards, NOT an `Unsupported` capability gap). This test pins that lifted limitation.
    ///
    /// NB: the PRODUCTION image_caption path is unaffected — it deliberately resolves on the JSON
    /// constraint ALONE and acquires vision at load (see the resolution note on the blocking task), so it
    /// worked before AND after the bump. This test just proves the resolver gained resolution-time vision
    /// recognition. macOS-only (the `mlx-llm` providers link there); no weights — reads only `config.json`.
    #[cfg(target_os = "macos")]
    #[test]
    fn vision_plus_json_resolution_now_resolves_for_qwen_vl_snapshot() {
        use gen_core::core_llm::{load_for_model_with, Constraint, LoadSpec, ModelRequirements};
        use mlx_llm as _; // force-link the providers into core-llm's inventory

        let dir = tempfile::tempdir().expect("tempdir");
        write_qwen_vl_config(dir.path());

        // Demand vision + Json (the `from_request` shape for an image+Json request).
        let reqs = ModelRequirements::default()
            .with_vision()
            .with_constraint(Constraint::Json);
        let err = load_for_model_with(
            &LoadSpec {
                source: dir.path().to_string_lossy().into_owned(),
                quantize: None,
            },
            &reqs,
        )
        .err()
        .expect("no weight shards on disk, so the LOAD fails after the provider is selected");
        let msg = err.to_string();
        // Post-bump: a provider WAS selected (the weightless_vision probe satisfied the vision gate), so
        // the error is a weight-load failure — NOT the pre-bump `Unsupported` capability-resolution gap.
        assert!(
            !msg.contains("no linked provider meets")
                && !msg.contains("no registered provider can serve")
                && !msg.contains("unsupported"),
            "vision+Json must now resolve a provider for a Qwen-VL snapshot (then fail on missing \
             weights) thanks to the weightless_vision probe; got: {msg}"
        );
    }

    /// Weightless resolution proof (the Issue-1 FIX). The SAME Qwen-VL snapshot, resolved with the
    /// worker's actual image_caption requirements — the JSON constraint ALONE, no vision filter —
    /// passes core-llm's `select` (it picks the text+Json `mlx-llama`, which `can_load`s a Qwen-VL
    /// wrapper). Resolution therefore reaches `load`; with no weight shards on disk the load fails, but
    /// with a `Load`/backend error (missing tensors), NOT the `Unsupported` resolution error above —
    /// proving the provider WAS selected. That is exactly what the fixed worker path does.
    #[cfg(target_os = "macos")]
    #[test]
    fn json_only_resolution_selects_a_provider_for_qwen_vl_snapshot() {
        use gen_core::core_llm::{
            load_for_model_with, textllms, Constraint, LoadSpec, ModelRequirements,
        };
        use mlx_llm as _; // force-link the providers into core-llm's inventory

        let dir = tempfile::tempdir().expect("tempdir");
        write_qwen_vl_config(dir.path());

        // The worker's image_caption resolution requirements: JSON constraint only, no vision filter.
        let reqs = ModelRequirements::default().with_constraint(Constraint::Json);
        let spec = LoadSpec {
            source: dir.path().to_string_lossy().into_owned(),
            quantize: None,
        };
        let err = load_for_model_with(&spec, &reqs)
            .err()
            .expect("no weights on disk, so the LOAD fails after the provider is selected");
        let msg = err.to_string();
        // The provider WAS selected (resolution passed): the error is a weight-load failure, not the
        // `Unsupported` capability-resolution error. So `select`/`meets` admitted a provider for the
        // Qwen-VL snapshot under the JSON-only requirements — the fix routes here.
        assert!(
            !msg.contains("no linked provider meets")
                && !msg.contains("no registered provider can serve"),
            "JSON-only requirements must resolve a provider (then fail on missing weights), got: {msg}"
        );

        // Strengthen the proof: the SPECIFIC provider admitted under JSON-only requirements for this
        // Qwen-VL snapshot must be `mlx-llama` — the one whose `generate` reads a `Content::Image` once
        // its Qwen-VL tower is loaded. `load_for_model_with` only returns the loaded provider (impossible
        // here without weights), so replicate core-llm's public selection predicate over the live
        // registry: architecture `can_load` AND the capability filter (vision/constraints) the resolver
        // applies. A future regression that resolved the snapshot to a DIFFERENT Json provider would
        // surface here, not slip past the generic "some provider resolved" check above.
        let viable: Vec<String> = textllms()
            .filter(|r| (r.can_load)(&spec))
            .filter(|r| {
                let caps = (r.descriptor)().capabilities;
                (!reqs.vision || caps.supports_vision)
                    && reqs
                        .constraints
                        .iter()
                        .all(|c| caps.supports_constraint(*c))
            })
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(
            viable.iter().any(|id| id == "mlx-llama"),
            "JSON-only resolution must admit `mlx-llama` (the image-capable Json provider) for a Qwen-VL \
             snapshot; viable providers were: {viable:?}"
        );
    }

    /// A minimal Qwen3.6 (`qwen3_5`) VLM-wrapper `config.json` — enough for the weightless `can_load`
    /// probes (`Architecture::from_config` dispatches on `model_type`/`architectures`; the
    /// `vision_config` marks it a VLM wrapper). No weight shards / tokenizer: resolution reads only this
    /// file, and a subsequent `load` is expected to fail on the missing tensors.
    #[cfg(target_os = "macos")]
    fn write_qwen_vl_config(dir: &Path) {
        std::fs::write(
            dir.join("config.json"),
            br#"{"architectures":["Qwen3_5ForConditionalGeneration"],
                "model_type":"qwen3_5",
                "text_config":{"model_type":"qwen3_5_text"},
                "vision_config":{"model_type":"qwen3_5","depth":27}}"#,
        )
        .expect("write qwen-vl config.json");
    }

    /// A minimal **`qwen3_vl`** VLM-wrapper `config.json` — the architecture of the chosen default
    /// image_caption model (`huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated`). `model_type`
    /// `qwen3_vl` (+ `text_config.model_type` `qwen3_vl_text`) is the family the sc-8171 engine bump
    /// (mlx-llm 6c052ba → 7041411) teaches the inventory to recognize: before the bump `qwen3_vl`
    /// silently fell through to a text-only `Architecture::Qwen3` (no vision tower); after it,
    /// `Architecture::Qwen3Vl` + the per-snapshot `weightless_vision` probe make a `qwen3_vl` snapshot
    /// resolve vision-capable. No weight shards / tokenizer: resolution reads only this file.
    #[cfg(target_os = "macos")]
    fn write_qwen3_vl_config(dir: &Path) {
        std::fs::write(
            dir.join("config.json"),
            br#"{"architectures":["Qwen3VLForConditionalGeneration"],
                "model_type":"qwen3_vl",
                "text_config":{"model_type":"qwen3_vl_text"},
                "vision_config":{"model_type":"qwen3_vl","depth":27}}"#,
        )
        .expect("write qwen3_vl config.json");
    }

    /// Weightless resolution proof for the **`qwen3_vl`** default model (sc-8171, the whole point of
    /// the engine bump). The SAME shape as `json_only_resolution_selects_a_provider_for_qwen_vl_snapshot`
    /// but on a `qwen3_vl` snapshot: under the worker's image_caption requirements (JSON constraint
    /// ALONE), core-llm's `select` admits `mlx-llama` for the `qwen3_vl` wrapper. CRUCIALLY this test
    /// ALSO proves the bump's purpose — that the SAME `qwen3_vl` snapshot resolves to a *vision-capable*
    /// provider — by replicating core-llm's per-snapshot vision predicate (`weightless_vision` probe →
    /// `serves_vision`) over the live registry. On the OLD engine (mlx-llm 6c052ba) a `qwen3_vl`
    /// `config.json` parsed to a text-only `Architecture::Qwen3` with NO `weightless_vision` probe, so
    /// this vision assertion would fail; on the bumped engine (7041411) it passes. macOS-only (the
    /// `mlx-llm` providers link there); no weights — resolution reads only `config.json`.
    #[cfg(target_os = "macos")]
    #[test]
    fn json_only_resolution_selects_vision_capable_provider_for_qwen3_vl_snapshot() {
        use gen_core::core_llm::{
            load_for_model_with, textllms, Constraint, LoadSpec, ModelRequirements,
        };
        use mlx_llm as _; // force-link the providers into core-llm's inventory

        let dir = tempfile::tempdir().expect("tempdir");
        write_qwen3_vl_config(dir.path());
        let spec = LoadSpec {
            source: dir.path().to_string_lossy().into_owned(),
            quantize: None,
        };

        // (1) Selection: the worker's image_caption resolution requirements (JSON constraint only)
        // resolve a provider for the qwen3_vl snapshot — the LOAD then fails on missing weights, NOT
        // the `Unsupported` capability-resolution error (proving a provider WAS selected).
        let reqs = ModelRequirements::default().with_constraint(Constraint::Json);
        let err = load_for_model_with(&spec, &reqs)
            .err()
            .expect("no weights on disk, so the LOAD fails after the provider is selected");
        let msg = err.to_string();
        assert!(
            !msg.contains("no linked provider meets")
                && !msg.contains("no registered provider can serve"),
            "JSON-only requirements must resolve a provider for a qwen3_vl snapshot (then fail on \
             missing weights), got: {msg}"
        );

        // (2) The selected Json provider for this qwen3_vl snapshot is `mlx-llama` — the one whose
        // loaded Qwen3-VL tower reads the `Content::Image` at generate time.
        let json_viable: Vec<String> = textllms()
            .filter(|r| (r.can_load)(&spec))
            .filter(|r| {
                let caps = (r.descriptor)().capabilities;
                reqs.constraints
                    .iter()
                    .all(|c| caps.supports_constraint(*c))
            })
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(
            json_viable.iter().any(|id| id == "mlx-llama"),
            "JSON-only resolution must admit `mlx-llama` for a qwen3_vl snapshot; viable: {json_viable:?}"
        );

        // (3) THE BUMP'S PURPOSE: the same qwen3_vl snapshot is recognized as vision-capable. Replicate
        // core-llm's per-snapshot vision predicate (the `weightless_vision` probe added in mlx-llm
        // 7041411 / sc-8077, OR a statically-vision descriptor) over the live registry. At least one
        // provider that `can_load` this snapshot must serve it WITH vision — impossible on the old
        // engine, where `qwen3_vl` was a text-only `Qwen3` with no probe.
        let vision_capable: Vec<String> = textllms()
            .filter(|r| (r.can_load)(&spec))
            .filter(|r| {
                r.weightless_vision.map(|p| p(&spec)).unwrap_or(false)
                    || (r.descriptor)().capabilities.supports_vision
            })
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(
            vision_capable.iter().any(|id| id == "mlx-llama"),
            "the sc-8171 bump must make a qwen3_vl snapshot resolve to a vision-capable `mlx-llama` \
             provider (weightless_vision probe); vision-capable providers were: {vision_capable:?}"
        );

        // (4) End-to-end resolver proof of the bump: demanding vision + Json at RESOLUTION now RESOLVES
        // a provider for the qwen3_vl snapshot (the `weightless_vision` probe satisfies the vision gate
        // from `config.json` alone), reaching `load` — which fails only on the missing weight shards, NOT
        // the pre-bump `Unsupported` capability gap. (Production still resolves on Json alone and acquires
        // vision at load; this assertion proves the resolver itself gained resolution-time vision.)
        let vision_json = ModelRequirements::default()
            .with_vision()
            .with_constraint(Constraint::Json);
        let vj_msg = load_for_model_with(&spec, &vision_json)
            .err()
            .expect("no weight shards on disk, so the LOAD fails after the provider is selected")
            .to_string();
        assert!(
            !vj_msg.contains("no linked provider meets")
                && !vj_msg.contains("no registered provider can serve")
                && !vj_msg.contains("unsupported"),
            "vision+Json must now resolve a provider for a qwen3_vl snapshot (then fail on missing \
             weights) thanks to the weightless_vision probe; got: {vj_msg}"
        );
    }

    /// Real-weight image-caption smoke (sc-8105 → validated under sc-8113): examines a reference image
    /// and emits an Ideogram JSON caption through the unified mlx-llm VISION engine —
    /// `gen_core::core_llm::load_for_model_with` resolves `mlx-llama` on a Qwen-VL snapshot. The pick is
    /// steered by the JSON constraint ALONE (the request DOES carry the image, but resolution must NOT
    /// demand vision: no provider statically advertises both vision and Json for a Qwen-VL snapshot, so
    /// `mlx-llama` is selected text+Json and flips to vision only at LOAD time — see the resolution note
    /// on the blocking task). The reqs are therefore built the SAME way production builds them — the
    /// request's output constraint only, NOT `ModelRequirements::from_request` (which would over-derive
    /// `vision:true` from the image block and fail `select` with `Error::Unsupported` before any load,
    /// per `vision_plus_json_resolution_fails_for_qwen_vl_snapshot`). The cleaned reply must pass
    /// `is_caption` with element bboxes kept. `#[ignore]` — the weights live outside CI; run on a Mac
    /// with a vision model staged and pointed at by VISION_CAPTION_SNAPSHOT and a reference image at
    /// IMAGE_CAPTION_REF:
    ///   VISION_CAPTION_SNAPSHOT=<snapshot dir> IMAGE_CAPTION_REF=<image path> \
    ///   cargo test -p sceneworks-worker --lib -- --ignored image_caption_examines_reference --nocapture
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight (sc-8113): needs a vision model snapshot + reference image; set VISION_CAPTION_SNAPSHOT + IMAGE_CAPTION_REF"]
    fn image_caption_examines_reference_image() {
        use gen_core::core_llm::{
            load_for_model_with, Constraint, Content, ImageRef, LoadSpec, Message,
            ModelRequirements, Role, Sampling, StreamEvent, TextLlmRequest,
        };
        use mlx_llm as _;

        let snapshot = std::env::var("VISION_CAPTION_SNAPSHOT")
            .expect("set VISION_CAPTION_SNAPSHOT to a vision model snapshot dir");
        let ref_path = std::env::var("IMAGE_CAPTION_REF")
            .expect("set IMAGE_CAPTION_REF to a reference image path");

        let image =
            load_caption_image_ref(std::path::Path::new(&ref_path)).expect("decode ref image");
        let (system, user) = build_image_caption_messages();
        let request = TextLlmRequest {
            messages: vec![
                Message::system(system),
                Message {
                    role: Role::User,
                    content: vec![Content::Image(image), Content::text(user)],
                    thinking: None,
                    tool_calls: Vec::new(),
                },
            ],
            sampling: Sampling {
                temperature: 0.4,
                top_p: 0.9,
                ..Sampling::default()
            },
            max_new_tokens: 2048,
            seed: None,
            constraint: Some(Constraint::Json),
            ..Default::default()
        };
        let _ = ImageRef::new(1, 1, vec![0u8; 3]); // touch the type so the import is load-bearing in all cfgs
                                                   // Build the resolution requirements the SAME way the production image_caption path does — the
                                                   // request's output constraint ALONE, no vision filter (NOT `ModelRequirements::from_request`,
                                                   // which derives `vision:true` from the image block and would fail `select` on a Qwen-VL snapshot
                                                   // before any load). This exercises the real production resolution path the sc-8113 validator runs.
        let mut reqs = ModelRequirements::default();
        for constraint in request.constraint.iter().copied() {
            reqs = reqs.with_constraint(constraint);
        }
        let captioner = load_for_model_with(
            &LoadSpec {
                source: snapshot,
                quantize: None,
            },
            &reqs,
        )
        .expect("load a vision provider via core-llm model-first JSON-only resolution");

        let mut sink = |_event: StreamEvent| {};
        let output = captioner.generate(&request, &mut sink).expect("generate");
        let json = clean_json_output(&output.text);
        eprintln!("image-caption JSON:\n{json}");

        let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
        assert!(
            sceneworks_core::ideogram_caption::is_caption(&parsed),
            "reply is a schema-valid Ideogram caption"
        );
    }

    /// Real-weight magic-prompt smoke (sc-7158): expands a plain idea into a JSON caption through the
    /// unified mlx-llm engine — `gen_core::core_llm::load_for_model` resolves `mlx-llama` on the Anubis
    /// snapshot (the JSON constraint steers the model-first pick) — and asserts the cleaned reply parses
    /// with the caption's required section. `#[ignore]` — the weights live outside CI; run on a Mac with
    /// the model staged in the HF cache:
    ///   cargo test -p sceneworks-worker --lib -- --ignored magic_prompt_expands
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight: needs the Anubis-Mini-8B prompt-refine model in the HF cache"]
    fn magic_prompt_expands_plain_text_to_caption() {
        use gen_core::core_llm::{
            load_for_model_with, Constraint, LoadSpec, Message, ModelRequirements, Sampling,
            StreamEvent, TextLlmRequest,
        };

        let home = std::env::var("HOME").expect("HOME");
        let snapshots = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots");
        let weights_dir = std::fs::read_dir(&snapshots)
            .expect("prompt-refine model staged in the HF cache")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .expect("a snapshot dir");

        let (system, user) = build_magic_prompt_messages(
            "a red fox sitting in a snowy forest at golden hour",
            "1:1",
        );
        let request = TextLlmRequest {
            messages: vec![Message::system(system), Message::user(user)],
            sampling: Sampling {
                temperature: 0.4,
                top_p: 0.9,
                ..Sampling::default()
            },
            max_new_tokens: 2048,
            seed: None,
            constraint: Some(Constraint::Json), // sc-6585: exercise constrained decoding
            ..Default::default()
        };
        let refiner = load_for_model_with(
            &LoadSpec {
                source: weights_dir.to_string_lossy().into_owned(),
                quantize: None,
            },
            &ModelRequirements::from_request(&request),
        )
        .expect("load prompt_refine via core-llm model-first resolution");

        let mut sink = |_event: StreamEvent| {};
        let output = refiner.generate(&request, &mut sink).expect("generate");
        let json = clean_json_output(&output.text);
        eprintln!("magic-prompt JSON:\n{json}");

        let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
        let cd = parsed
            .get("compositional_deconstruction")
            .expect("has compositional_deconstruction");
        assert!(cd.get("background").is_some(), "has a background");
        assert!(
            cd.get("elements").map(Value::is_array).unwrap_or(false),
            "elements is an array"
        );
    }

    /// Real-weight magic-prompt smoke (sc-7404) — the Windows/CUDA twin of the macOS test above. Expands
    /// a plain idea into a JSON caption through the unified candle engine: `gen_core::core_llm::load_for_model`
    /// resolves candle-llm's `candle-llama` on the Anubis snapshot (the JSON constraint steers the
    /// model-first pick + masks the decode), and the cleaned reply must parse with the caption's required
    /// section. This is the candle parity gate for retiring `candle-gen-prompt-refine`. `#[ignore]` — the
    /// weights live outside CI; run on the CUDA box with the model staged (point `PROMPT_REFINE_SNAPSHOT`
    /// at the snapshot dir, else it resolves the standard HF cache under `%USERPROFILE%`):
    ///   cargo test -p sceneworks-worker --lib --features backend-candle --release -- --ignored magic_prompt_expands_plain_text_to_caption_candle --nocapture
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    #[ignore = "real-weight: needs the Anubis-Mini-8B prompt-refine model staged + a CUDA GPU"]
    fn magic_prompt_expands_plain_text_to_caption_candle() {
        use gen_core::core_llm::{
            load_for_model_with, Constraint, LoadSpec, Message, ModelRequirements, Sampling,
            StreamEvent, TextLlmRequest,
        };

        // Force-link `candle-llama` so model-first resolution can discover it in this test binary
        // (the worker's force-link anchor is cfg'd to the job path, not the test module).
        use candle_llm as _;

        // An explicit snapshot override wins; otherwise resolve the HF cache under the Windows home.
        let weights_dir = match std::env::var("PROMPT_REFINE_SNAPSHOT") {
            Ok(dir) => std::path::PathBuf::from(dir),
            Err(_) => {
                let home = std::env::var("USERPROFILE")
                    .or_else(|_| std::env::var("HOME"))
                    .expect("USERPROFILE/HOME");
                let snapshots = std::path::Path::new(&home)
                    .join(".cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots");
                std::fs::read_dir(&snapshots)
                    .expect("prompt-refine model staged in the HF cache (or set PROMPT_REFINE_SNAPSHOT)")
                    .flatten()
                    .map(|entry| entry.path())
                    .find(|path| path.is_dir())
                    .expect("a snapshot dir")
            }
        };

        let (system, user) = build_magic_prompt_messages(
            "a red fox sitting in a snowy forest at golden hour",
            "1:1",
        );
        let request = TextLlmRequest {
            messages: vec![Message::system(system), Message::user(user)],
            sampling: Sampling {
                temperature: 0.4,
                top_p: 0.9,
                ..Sampling::default()
            },
            max_new_tokens: 2048,
            seed: None,
            constraint: Some(Constraint::Json), // sc-6585: exercise constrained decoding on candle
            ..Default::default()
        };
        let refiner = load_for_model_with(
            &LoadSpec {
                source: weights_dir.to_string_lossy().into_owned(),
                quantize: None,
            },
            &ModelRequirements::from_request(&request),
        )
        .expect("load prompt_refine via core-llm model-first resolution (candle-llama)");

        let mut sink = |_event: StreamEvent| {};
        let output = refiner.generate(&request, &mut sink).expect("generate");
        let json = clean_json_output(&output.text);
        eprintln!("magic-prompt JSON (candle):\n{json}");

        let parsed: Value = serde_json::from_str(&json).expect("a valid JSON object");
        let cd = parsed
            .get("compositional_deconstruction")
            .expect("has compositional_deconstruction");
        assert!(cd.get("background").is_some(), "has a background");
        assert!(
            cd.get("elements").map(Value::is_array).unwrap_or(false),
            "elements is an array"
        );
    }

    #[test]
    fn clean_output_handles_orphan_close_case_insensitive_and_whitespace() {
        // Orphan closing tag with no open (case-insensitive): keep only the tail.
        assert_eq!(
            clean_refine_output("reasoning</THINK> Final prompt."),
            "Final prompt."
        );
        // Multiple think blocks all stripped.
        assert_eq!(
            clean_refine_output("<think>a</think>X<think>b</think>Y"),
            "XY"
        );
        // Plain whitespace trim, no decoration.
        assert_eq!(clean_refine_output("  spaced out  "), "spaced out");
        // An unmatched OPEN tag is left untouched (Python non-greedy regex would not match).
        assert_eq!(
            clean_refine_output("<think>no close here"),
            "<think>no close here"
        );
    }

    // ------------------------------------------------------------------------------------------
    // Magic-prompt model bake-off (sc-6550) — a parameterized harness + degeneracy rubric for
    // choosing the magic-prompt model that fixes the sc-6546 caption-quality ceiling. The 3B
    // (`huihui-ai/Llama-3.2-3B-Instruct-abliterated`) stochastically emits SEMANTICALLY-DEGENERATE
    // captions (the subject placed as a `type:"text"` element, or `background` set to a transparent
    // cutout for a prompt that never asked for one) which placeholder at 1024²/48. This harness runs
    // Ideogram's magic-prompt v1 system prompt (the SHIPPING `build_magic_prompt_messages`) over a
    // fixed prompt set through the REAL `prompt_refine` provider and scores the degeneracy from the
    // caption JSON alone — the root-cause signal, no Ideogram render needed (metric (c), the
    // placeholder-escape render confirmation, is the gated downstream step).
    // ------------------------------------------------------------------------------------------

    /// One bake-off prompt + what a CORRECT caption looks like for it, so the analyzer can tell a
    /// degeneracy (subject-as-text / unrequested transparent cutout) apart from a legitimate use of
    /// those features.
    // `text` / `edgy` are read only by the macOS-gated real-weight bakeoff test below; on the Linux
    // clippy lane that test is cfg'd out, so suppress the cross-platform dead-code lint.
    #[allow(dead_code)]
    struct BakeoffPrompt {
        text: &'static str,
        /// The prompt legitimately calls for `type:"text"` (typography / rendered words), so a text
        /// element is NOT degenerate here.
        wants_text: bool,
        /// The prompt legitimately calls for a transparent background (product cutout / sticker), so
        /// `background:"…transparent…"` is NOT degenerate here.
        wants_transparent: bool,
        /// An edgy prompt — present to measure refusals (an uncensored model must still emit a
        /// caption). A refusal shows up as `!caption_valid`.
        edgy: bool,
    }

    /// The fixed bake-off prompt set: the sc-6546 failing prompt + diverse plain subjects (the
    /// degeneracy measurement), two CONTROLS that legitimately want a text element / transparent
    /// background, and two edgy prompts (refusal test for the uncensored requirement).
    const BAKEOFF_PROMPTS: &[BakeoffPrompt] = &[
        // The sc-6546 failing prompt.
        BakeoffPrompt { text: "a red fox sitting in a snowy forest at golden hour", wants_text: false, wants_transparent: false, edgy: false },
        // Diverse plain subjects — none ask for text or transparency, so ANY text element or
        // transparent background is the sc-6546 degeneracy.
        BakeoffPrompt { text: "a steaming bowl of ramen on a rustic wooden table", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a vintage red bicycle leaning against a weathered brick wall", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a snow leopard walking across a rocky mountain ridge", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "an astronaut floating above the earth in low orbit", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a cozy log cabin in a pine forest during a snowstorm", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a golden retriever puppy playing in a pile of autumn leaves", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a futuristic city skyline at night lit by neon signs", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a single sunflower in a glass vase on a sunlit windowsill", wants_text: false, wants_transparent: false, edgy: false },
        BakeoffPrompt { text: "a lone samurai standing in a misty bamboo forest at dawn", wants_text: false, wants_transparent: false, edgy: false },
        // CONTROL: legitimately wants text — the model SHOULD emit a `type:"text"` element here.
        BakeoffPrompt { text: "a bold motivational poster with the headline NEVER GIVE UP in large letters", wants_text: true, wants_transparent: false, edgy: false },
        // CONTROL: legitimately wants a transparent background — a product cutout.
        BakeoffPrompt { text: "a product cutout of a red running sneaker on a transparent background", wants_text: false, wants_transparent: true, edgy: false },
        // Edgy (refusal test for the uncensored requirement).
        BakeoffPrompt { text: "a gritty film-noir crime scene, a detective examining a body under a harsh streetlight", wants_text: false, wants_transparent: false, edgy: true },
        BakeoffPrompt { text: "a fierce barbarian warrior, bloodied and scarred, standing over a fallen foe after battle", wants_text: false, wants_transparent: false, edgy: true },
    ];

    /// Degeneracy metrics for a single caption reply, scored against what the prompt legitimately
    /// expected. This is the sc-6546 rubric in code — pure, so it is unit-tested without weights.
    #[derive(Clone, Copy, Debug, Default)]
    // `parse_ok` is read only by the macOS-gated bakeoff test; suppress the Linux-lane dead-code lint.
    #[allow(dead_code)]
    struct CaptionMetrics {
        /// The cleaned reply parses as JSON.
        parse_ok: bool,
        /// A structured caption — a JSON object with a `compositional_deconstruction` object
        /// (mirrors `sceneworks_core::ideogram_caption::is_caption`).
        caption_valid: bool,
        n_obj: usize,
        n_text: usize,
        has_transparent_bg: bool,
        /// A text element appeared where the prompt did not ask for one (subject-as-`text`
        /// degeneracy).
        text_when_unwanted: bool,
        /// A transparent background appeared where the prompt did not ask for one.
        transparent_when_unwanted: bool,
    }

    impl CaptionMetrics {
        /// The headline placeholder-risk proxy: a malformed/non-caption reply, an unrequested
        /// transparent cutout, an unrequested text element, or a caption with NO object subject.
        fn degenerate(&self) -> bool {
            !self.caption_valid
                || self.transparent_when_unwanted
                || self.text_when_unwanted
                || self.n_obj == 0
        }
    }

    /// Score a raw model reply for `prompt` (applies the shipping `clean_json_output` first).
    fn analyze_caption(prompt: &BakeoffPrompt, raw: &str) -> CaptionMetrics {
        let cleaned = clean_json_output(raw);
        let Ok(value) = serde_json::from_str::<Value>(&cleaned) else {
            return CaptionMetrics::default();
        };
        let mut m = CaptionMetrics {
            parse_ok: true,
            ..Default::default()
        };
        let Some(cd) = value
            .get("compositional_deconstruction")
            .and_then(Value::as_object)
        else {
            return m; // parsed, but not a caption (caption_valid stays false)
        };
        m.caption_valid = true;
        if let Some(bg) = cd.get("background").and_then(Value::as_str) {
            m.has_transparent_bg = bg.to_ascii_lowercase().contains("transparent");
        }
        if let Some(elements) = cd.get("elements").and_then(Value::as_array) {
            for el in elements {
                match el.get("type").and_then(Value::as_str) {
                    Some("text") => m.n_text += 1,
                    _ => m.n_obj += 1, // obj (or an untyped element defaults to obj per the serializer)
                }
            }
        }
        m.text_when_unwanted = m.n_text > 0 && !prompt.wants_text;
        m.transparent_when_unwanted = m.has_transparent_bg && !prompt.wants_transparent;
        m
    }

    #[test]
    fn analyze_caption_flags_the_sc6546_degeneracies() {
        // Healthy: one obj subject, opaque scene background, no text → not degenerate.
        let plain = &BAKEOFF_PROMPTS[0];
        let healthy = analyze_caption(
            plain,
            r#"{"compositional_deconstruction": {"background": "a snowy forest at golden hour", "elements": [{"type": "obj", "desc": "a red fox"}]}}"#,
        );
        assert!(healthy.caption_valid && healthy.n_obj == 1 && !healthy.degenerate());

        // Subject-as-text degeneracy: the fox emitted as a `type:"text"` element.
        let subj_text = analyze_caption(
            plain,
            r#"{"compositional_deconstruction": {"background": "a snowy forest", "elements": [{"type": "text", "text": "a red fox", "desc": "a red fox"}]}}"#,
        );
        assert!(subj_text.text_when_unwanted && subj_text.n_obj == 0 && subj_text.degenerate());

        // Transparent-background degeneracy for a scene prompt that never asked for one.
        let transp = analyze_caption(
            plain,
            r#"{"compositional_deconstruction": {"background": "on a transparent background", "elements": [{"type": "obj", "desc": "a red fox"}]}}"#,
        );
        assert!(transp.transparent_when_unwanted && transp.degenerate());

        // A refusal / non-JSON reply is degenerate (parse fails).
        assert!(analyze_caption(plain, "I cannot help with that request.").degenerate());

        // CONTROL: the transparent-cutout prompt legitimately uses a transparent background.
        let cutout = &BAKEOFF_PROMPTS[11];
        assert!(cutout.wants_transparent);
        let ok_transp = analyze_caption(
            cutout,
            r#"{"compositional_deconstruction": {"background": "on a transparent background", "elements": [{"type": "obj", "desc": "a red sneaker"}]}}"#,
        );
        assert!(!ok_transp.transparent_when_unwanted && !ok_transp.degenerate());

        // CONTROL: the poster prompt legitimately uses a text element.
        let poster = &BAKEOFF_PROMPTS[10];
        assert!(poster.wants_text);
        let ok_text = analyze_caption(
            poster,
            r#"{"compositional_deconstruction": {"background": "a plain studio backdrop", "elements": [{"type": "obj", "desc": "a poster"}, {"type": "text", "text": "NEVER GIVE UP", "desc": "the headline"}]}}"#,
        );
        assert!(!ok_text.text_when_unwanted && !ok_text.degenerate());
    }

    /// Real-weight magic-prompt bake-off (sc-6550, ported to the unified engine in sc-7158). Runs the
    /// shipping magic-prompt system prompt over `BAKEOFF_PROMPTS` × N seeds through mlx-llm's `mlx-llama`
    /// (resolved model-first by `gen_core::core_llm::load_for_model`) and prints per-run + aggregate
    /// degeneracy metrics. `#[ignore]` — the weights live outside CI. Point it at any Llama-3.x-Instruct
    /// snapshot dir (the 3B baseline, a Llama-3.1-8B, an Anubis-8B …). Measure footprint by wrapping the
    /// TEST BINARY with `/usr/bin/time -l` (NOT `cargo test`, which reports the cargo parent's peak):
    ///   BAKEOFF_MODEL_DIR=~/.cache/huggingface/hub/models--TheDrummer--Anubis-Mini-8B-v1/snapshots/<rev> \
    ///   BAKEOFF_SEEDS=3 cargo test -p sceneworks-worker --lib -- --ignored --nocapture magic_prompt_bakeoff
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight: set BAKEOFF_MODEL_DIR to a Llama-3.x-Instruct snapshot dir"]
    fn magic_prompt_bakeoff() {
        use gen_core::core_llm::{
            load_for_model_with, Constraint, LoadSpec, Message, ModelRequirements, Sampling,
            StreamEvent, TextLlmRequest,
        };
        use std::time::Instant;

        let weights_dir = std::path::PathBuf::from(
            std::env::var("BAKEOFF_MODEL_DIR").expect("set BAKEOFF_MODEL_DIR to a snapshot dir"),
        );
        let seeds: u64 = std::env::var("BAKEOFF_SEEDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        // sc-6585: set BAKEOFF_CONSTRAIN=1 to measure caption_valid under grammar-constrained decoding
        // (expect ~100%) vs the unconstrained baseline. Steers the model-first pick to a JSON provider.
        let constrain = std::env::var("BAKEOFF_CONSTRAIN").is_ok();
        eprintln!("BAKEOFF model_dir={}", weights_dir.display());

        let mut reqs = ModelRequirements::default();
        if constrain {
            reqs = reqs.with_constraint(Constraint::Json);
        }
        let refiner = load_for_model_with(
            &LoadSpec {
                source: weights_dir.to_string_lossy().into_owned(),
                quantize: None,
            },
            &reqs,
        )
        .expect("load prompt_refine via core-llm model-first resolution");

        let (mut runs, mut parse_ok, mut caption_valid, mut degenerate) = (0u32, 0u32, 0u32, 0u32);
        let (mut text_bad, mut transp_bad, mut no_obj) = (0u32, 0u32, 0u32);
        // Refusal tracking on the edgy prompts only.
        let (mut edgy_runs, mut edgy_valid) = (0u32, 0u32);
        let mut total_ms = 0u128;

        for prompt in BAKEOFF_PROMPTS {
            let (system, user) = build_magic_prompt_messages(prompt.text, "1:1");
            for seed in 0..seeds {
                let request = TextLlmRequest {
                    messages: vec![Message::system(system.clone()), Message::user(user.clone())],
                    sampling: Sampling {
                        temperature: 0.4, // matches the worker's magic-prompt sampling
                        top_p: 0.9,
                        ..Sampling::default()
                    },
                    max_new_tokens: 2048,
                    seed: Some(seed),
                    constraint: constrain.then_some(Constraint::Json),
                    ..Default::default()
                };
                let mut sink = |_event: StreamEvent| {};
                let start = Instant::now();
                let output = refiner.generate(&request, &mut sink).expect("generate");
                let ms = start.elapsed().as_millis();
                total_ms += ms;

                let m = analyze_caption(prompt, &output.text);
                runs += 1;
                parse_ok += m.parse_ok as u32;
                caption_valid += m.caption_valid as u32;
                degenerate += m.degenerate() as u32;
                text_bad += m.text_when_unwanted as u32;
                transp_bad += m.transparent_when_unwanted as u32;
                no_obj += (m.caption_valid && m.n_obj == 0) as u32;
                if prompt.edgy {
                    edgy_runs += 1;
                    edgy_valid += m.caption_valid as u32;
                }
                eprintln!(
                    "BAKEOFF run seed={seed} {}ms valid={} obj={} text={} transp_bg={} degen={} :: {}",
                    ms, m.caption_valid, m.n_obj, m.n_text, m.has_transparent_bg, m.degenerate(),
                    prompt.text
                );
                // Surface the cleaned JSON so degeneracies are eyeball-auditable.
                eprintln!("BAKEOFF json :: {}", clean_json_output(&output.text));
            }
        }

        let pct = |n: u32| 100.0 * n as f64 / runs.max(1) as f64;
        eprintln!(
            "\nBAKEOFF SUMMARY ({runs} runs, {seeds} seeds × {} prompts)",
            BAKEOFF_PROMPTS.len()
        );
        eprintln!("  parse_ok           {:.1}%", pct(parse_ok));
        eprintln!("  caption_valid      {:.1}%", pct(caption_valid));
        eprintln!(
            "  DEGENERATE         {:.1}%  <- headline placeholder-risk proxy",
            pct(degenerate)
        );
        eprintln!("    text_when_unwanted   {:.1}%", pct(text_bad));
        eprintln!("    transp_when_unwanted {:.1}%", pct(transp_bad));
        eprintln!("    valid_but_no_obj     {:.1}%", pct(no_obj));
        eprintln!(
            "  edgy_caption_valid {:.1}% ({edgy_valid}/{edgy_runs})  (refusal proxy)",
            100.0 * edgy_valid as f64 / edgy_runs.max(1) as f64
        );
        eprintln!("  avg_latency        {}ms", total_ms / runs.max(1) as u128);
    }
}
