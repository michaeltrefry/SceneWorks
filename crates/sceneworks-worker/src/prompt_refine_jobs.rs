//! Native prompt refinement (epic 5095): candle on Windows/CUDA (sc-5525) + MLX on macOS (sc-5552).
//!
//! Routes the `prompt_refine` job to a native `TextLlm` provider (Llama-3.2-3B-Instruct) through the
//! backend-neutral `gen_core::load_textllm` seam (the sc-5500 contract): the candle provider
//! (`backend="candle"`, `candle-gen-prompt-refine`) on the Windows candle build, and the MLX twin
//! (`backend="mlx"`, `mlx-gen-prompt-refine`) on macOS. The Python torch `PromptRefiner`
//! (`apps/worker/scene_worker/prompt_refine.py`) stays the fallback only on platforms with neither
//! native provider (e.g. the candle-less Desktop installer); its physical deletion waits on the candle
//! provider being the default everywhere off-Mac (see sc-5525).
//!
//! The `TextLlm` contract is generic (`system` + `prompt` + sampling → text), so the
//! prompt-refinement PRODUCT logic that lived in `prompt_refine.py` moves here caller-side: the
//! rewrite rules + image/video medium switch + guide assembly (`build_refine_system_prompt`, into the
//! request `system`) and the reasoning-block / code-fence / surrounding-quote cleanup
//! (`clean_refine_output`, over the model reply). Sampling matches the Python path (temperature 0.7,
//! top_p 0.9, max_new_tokens 512), as does the empty-output → error behavior and the `{originalPrompt,
//! refinedPrompt}` result shape.

use super::*;

// Prompt-refine provider force-link anchors: keep each backend's `inventory::submit!` `TextLlm`
// registration (id `prompt_refine`) from being dropped by the release linker. sc-5552 adds the native
// MLX twin (`mlx_gen_prompt_refine`, backend `mlx`) alongside sc-5525's candle anchor; mirrors the
// dual JoyCaption anchors in caption_jobs.rs.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use candle_gen_prompt_refine as _;
#[cfg(target_os = "macos")]
use mlx_gen_prompt_refine as _;

// The registry id both providers register under (`prompt::PROMPT_REFINE_ID`); kept as a local literal
// so the shared dispatch names no backend-specific symbol. `gen_core::load_textllm` resolves the MLX
// twin on macOS and the candle provider on the Windows candle build.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
const PROMPT_REFINE_ENGINE_ID: &str = "prompt_refine";
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

/// Body of a `[NAME]` section in the magic-prompt file (port of the reference `_load_sections`):
/// section markers are a bracketed single word alone on a line. Returns the trimmed body, or empty.
#[cfg(any(
    test,
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn magic_section(name: &str) -> String {
    let mut capturing = false;
    let mut body: Vec<&str> = Vec::new();
    for line in MAGIC_PROMPT_V1.lines() {
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
// Job handler — native MLX on macOS (sc-5552) and candle on the Windows candle build (sc-5525). The
// body is backend-agnostic: `gen_core::load_textllm("prompt_refine", …)` resolves whichever provider
// is force-linked above. The Python torch `PromptRefiner` remains the fallback on other platforms.
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
    use gen_core::{
        CancelFlag, LoadSpec, Progress, TextLlmConstraint, TextLlmRequest, TextLlmSampling,
        WeightsSource,
    };

    let payload = &job.payload;
    let original_prompt = payload
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned();
    if original_prompt.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Prompt refinement requires a non-empty prompt.".to_owned(),
        ));
    }
    let guide = payload
        .get("guide")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let workflow = payload
        .get("workflow")
        .and_then(Value::as_str)
        .map(str::to_owned);
    // Magic-prompt expansion (sc-5997) drives the same TextLlm seam with Ideogram's caption system
    // prompt instead of the rewrite rules; captions run longer than a one-line prompt, so allow more
    // tokens and sample cooler for steadier JSON.
    let is_magic = payload
        .get("task")
        .and_then(Value::as_str)
        .map(|task| task.trim().eq_ignore_ascii_case("magic_prompt"))
        .unwrap_or(false);
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_REFINE_MODEL)
        .to_owned();
    let max_new_tokens = payload
        .get("maxNewTokens")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(if is_magic { 2048 } else { 512 });
    let temperature = if is_magic { 0.4 } else { 0.7 };
    let work_message = if is_magic {
        "Expanding to a caption…"
    } else {
        "Refining prompt…"
    };
    let done_message = if is_magic {
        "Caption ready."
    } else {
        "Prompt refined."
    };

    let (system, user_message) = if is_magic {
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
        let refiner = gen_core::load_textllm(
            PROMPT_REFINE_ENGINE_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .map_err(|error| WorkerError::Engine(format!("prompt-refine load failed: {error}")))?;
        emit_event(
            "prompt_refine_load_complete",
            json!({ "jobId": job_id, "engine": engine_label }),
        );
        if blocking_cancel.is_cancelled() {
            return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
        }
        let request = TextLlmRequest {
            system,
            prompt,
            sampling: TextLlmSampling {
                temperature,
                top_p: 0.9,
                max_new_tokens,
                seed: None,
            },
            // sc-6585: magic-prompt expansion must emit a structurally-valid JSON caption, so
            // constrain its decode to the JSON grammar; the free-text "Refine my prompt" rewrite is
            // unconstrained prose.
            constraint: is_magic.then_some(TextLlmConstraint::Json),
            cancel: blocking_cancel.clone(),
        };
        let mut on_progress = |progress: Progress| {
            if let Progress::Step { current, total } = progress {
                let _ = tx.blocking_send((current, total));
            }
        };
        let output = refiner
            .generate(&request, &mut on_progress)
            .map_err(|error| {
                WorkerError::Engine(format!("prompt-refine generation failed: {error}"))
            })?;
        Ok(output.text)
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
    // Magic-prompt isolates the JSON object (the web parses + validates it); refine cleans to prose.
    let refined = if is_magic {
        clean_json_output(&raw)
    } else {
        clean_refine_output(&raw)
    };
    if refined.is_empty() {
        return Err(WorkerError::Engine(
            "The prompt-refinement model returned an empty prompt.".to_owned(),
        ));
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

    /// Real-weight magic-prompt smoke (sc-5997): expands a plain idea into a JSON caption through the
    /// actual `prompt_refine` Llama-3.2-3B and asserts the cleaned reply parses with the caption's
    /// required section. `#[ignore]` — the weights live outside CI; run on a Mac with the model
    /// staged in the HF cache:
    ///   cargo test -p sceneworks-worker --lib -- --ignored magic_prompt_expands
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight: needs the Llama-3.2-3B prompt-refine model in the HF cache"]
    fn magic_prompt_expands_plain_text_to_caption() {
        use gen_core::{
            CancelFlag, LoadSpec, Progress, TextLlmConstraint, TextLlmRequest, TextLlmSampling,
            WeightsSource,
        };

        let home = std::env::var("HOME").expect("HOME");
        let snapshots = std::path::Path::new(&home).join(
            ".cache/huggingface/hub/models--huihui-ai--Llama-3.2-3B-Instruct-abliterated/snapshots",
        );
        let weights_dir = std::fs::read_dir(&snapshots)
            .expect("prompt-refine model staged in the HF cache")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| path.is_dir())
            .expect("a snapshot dir");

        let refiner = gen_core::load_textllm(
            PROMPT_REFINE_ENGINE_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .expect("load prompt_refine");

        let (system, user) = build_magic_prompt_messages(
            "a red fox sitting in a snowy forest at golden hour",
            "1:1",
        );
        let request = TextLlmRequest {
            system,
            prompt: user,
            sampling: TextLlmSampling {
                temperature: 0.4,
                top_p: 0.9,
                max_new_tokens: 2048,
                seed: None,
            },
            constraint: Some(TextLlmConstraint::Json), // sc-6585: exercise constrained decoding
            cancel: CancelFlag::new(),
        };
        let mut noop = |_progress: Progress| {};
        let output = refiner.generate(&request, &mut noop).expect("generate");
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

    /// Real-weight magic-prompt bake-off (sc-6550). Runs the shipping magic-prompt system prompt over
    /// `BAKEOFF_PROMPTS` × N seeds through the REAL `prompt_refine` provider and prints per-run +
    /// aggregate degeneracy metrics. `#[ignore]` — the weights live outside CI. Point it at any
    /// snapshot dir (the 3B baseline, a Llama-3.1-8B, an Anubis-8B …); the provider is config-driven
    /// Llama, so any Llama-3.x-Instruct shape loads as-is. Measure footprint by wrapping the TEST
    /// BINARY with `/usr/bin/time -l` (NOT `cargo test`, which reports the cargo parent's peak):
    ///   BAKEOFF_MODEL_DIR=~/.cache/huggingface/hub/models--huihui-ai--Meta-Llama-3.1-8B-Instruct-abliterated/snapshots/<rev> \
    ///   BAKEOFF_SEEDS=3 cargo test -p sceneworks-worker --lib -- --ignored --nocapture magic_prompt_bakeoff
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "real-weight: set BAKEOFF_MODEL_DIR to a Llama-3.x-Instruct snapshot dir"]
    fn magic_prompt_bakeoff() {
        use gen_core::{
            CancelFlag, LoadSpec, Progress, TextLlmConstraint, TextLlmRequest, TextLlmSampling,
            WeightsSource,
        };
        use std::time::Instant;

        let weights_dir = std::path::PathBuf::from(
            std::env::var("BAKEOFF_MODEL_DIR").expect("set BAKEOFF_MODEL_DIR to a snapshot dir"),
        );
        let seeds: u64 = std::env::var("BAKEOFF_SEEDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        eprintln!("BAKEOFF model_dir={}", weights_dir.display());

        let refiner = gen_core::load_textllm(
            PROMPT_REFINE_ENGINE_ID,
            &LoadSpec::new(WeightsSource::Dir(weights_dir)),
        )
        .expect("load prompt_refine");

        let (mut runs, mut parse_ok, mut caption_valid, mut degenerate) = (0u32, 0u32, 0u32, 0u32);
        let (mut text_bad, mut transp_bad, mut no_obj) = (0u32, 0u32, 0u32);
        // Refusal tracking on the edgy prompts only.
        let (mut edgy_runs, mut edgy_valid) = (0u32, 0u32);
        let mut total_ms = 0u128;

        for prompt in BAKEOFF_PROMPTS {
            let (system, user) = build_magic_prompt_messages(prompt.text, "1:1");
            for seed in 0..seeds {
                let request = TextLlmRequest {
                    system: system.clone(),
                    prompt: user.clone(),
                    sampling: TextLlmSampling {
                        temperature: 0.4, // matches the worker's magic-prompt sampling
                        top_p: 0.9,
                        max_new_tokens: 2048,
                        seed: Some(seed),
                    },
                    // sc-6585: set BAKEOFF_CONSTRAIN=1 to measure caption_valid under grammar-
                    // constrained decoding (expect ~100%) vs the unconstrained baseline.
                    constraint: std::env::var("BAKEOFF_CONSTRAIN")
                        .is_ok()
                        .then_some(TextLlmConstraint::Json),
                    cancel: CancelFlag::new(),
                };
                let mut noop = |_p: Progress| {};
                let start = Instant::now();
                let output = refiner.generate(&request, &mut noop).expect("generate");
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
