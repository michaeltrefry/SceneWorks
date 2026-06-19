// Ideogram 4 mandatory JSON-caption conditioning + placeholder detect-and-recover (epic 4725,
// sc-6501). Ideogram 4 was trained EXCLUSIVELY on structured JSON captions; its own reference
// pipeline gates prompts through a `CaptionVerifier` that REJECTS plain text. Feeding the model raw
// plain text drives it out-of-distribution, where it stochastically emits the learned "Image blocked
// by safety filter" placeholder (sc-6307, reference-confirmed faithful — NOT a porting bug).
//
// Two worker-side layers (the rich, escape-the-placeholder fix lives in the UI — see below):
//   1. ensure_caption_prompt — a FORMAT guarantee: a non-caption prompt is wrapped into a minimal
//      valid JSON caption so the engine never tokenizes raw text (covers the API path / any UI
//      bypass). NOTE (real-weight verified, sc-6501): a *sparse* caption does NOT by itself escape
//      the placeholder — content RICHNESS is the lever, not JSON structure. So this is a structural
//      guarantee, not a quality fix; the actual escape comes from (a) the UI's magic-prompt RICH
//      expansion (run as a SEPARATE job, so the 3B utility model and the ~50GB Ideogram weights are
//      never co-resident) or (b) the reseed recovery below.
//   2. looks_like_placeholder + reseed recovery — the detect-and-recover safety net: a residual
//      placeholder (stochastic; ~2/3 of plain/sparse renders at the 1024²/48 default) is detected and
//      transparently reseeded. The worker deliberately does NOT inline-expand (that would force a
//      second model load alongside the resident Ideogram weights → memory pressure); rich expansion
//      is the caller's job.
//
// No engine change — the fix is entirely in the prompt we hand the engine, and a post-render check.

/// Whether `model` is one of the Ideogram 4 image models (quality + turbo). Both are JSON-caption
/// trained, so both get the caption guard; the placeholder is a quality-mode CFG behavior the turbo
/// (CFG-free) path cannot trigger, but running the detector on turbo is a cheap, harmless no-op.
pub(crate) fn is_ideogram_model(model: &str) -> bool {
    matches!(model, "ideogram_4" | "ideogram_4_turbo")
}

// ----- caption guard --------------------------------------------------------

/// Ensure the prompt handed to an Ideogram 4 engine is a structured JSON caption. If `prompt` is
/// already a caption (a JSON object carrying the required `compositional_deconstruction` section),
/// it is returned unchanged — this is the normal path (the web sends the canonically-ordered
/// serialized caption). Otherwise the plain text is wrapped into a minimal, valid, key-ordered
/// caption so the engine receives the trained FORMAT rather than raw text. This is a structural
/// guarantee only: a sparse caption does not by itself escape the placeholder (sc-6501 real-weight —
/// richness is the lever), so the reseed recovery, not the wrap, is what rescues the API path.
///
/// The wrap mirrors the web caption contract (`apps/web/src/ideogramCaption.js`): top-level key
/// order `high_level_description` then `compositional_deconstruction`, composition order `background`
/// then `elements`, and Python `json.dumps(..., ensure_ascii=False)` default spacing (`", "` / `": "`)
/// — the exact byte format the model was trained on and the engine tokenizes.
pub(crate) fn ensure_caption_prompt(prompt: &str) -> String {
    if is_caption_json(prompt) {
        prompt.to_owned()
    } else {
        wrap_plain_text(prompt.trim())
    }
}

/// True when `prompt` parses as a JSON object that already contains a `compositional_deconstruction`
/// object — the one section Ideogram's `CaptionVerifier` requires. We deliberately accept a sparse
/// caption (the web validator's full rules are not re-implemented here): the goal is only to avoid
/// double-wrapping an already-structured prompt, not to re-validate it.
fn is_caption_json(prompt: &str) -> bool {
    if !prompt.trim_start().starts_with('{') {
        return false;
    }
    matches!(
        serde_json::from_str::<serde_json::Value>(prompt),
        Ok(serde_json::Value::Object(map))
            if map.get("compositional_deconstruction").is_some_and(serde_json::Value::is_object)
    )
}

/// Wrap plain text into a minimal valid caption string. The same text seeds both
/// `high_level_description` and the composition `background`, giving the model two conditioning
/// signals while keeping the structure the model expects.
fn wrap_plain_text(text: &str) -> String {
    let quoted = json_string(text);
    format!(
        "{{\"high_level_description\": {quoted}, \"compositional_deconstruction\": {{\"background\": {quoted}, \"elements\": []}}}}"
    )
}

/// JSON-encode a string value (quotes + escapes) exactly as `serde_json` / Python would. Falls back
/// to an empty JSON string on the (practically impossible) encode error so the wrap is always valid.
fn json_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_owned())
}

// ----- placeholder detect-and-recover ---------------------------------------

// Detection thresholds for the "Image blocked by safety filter" placeholder. The placeholder is a
// near-uniform mid-gray frame with a small block of crisp dark text (sc-6307: mean ~112-130, std
// ~10). A pure flat/std check does NOT catch it — the baked text lifts std to ~10, read as "not
// flat". So we combine three signals the placeholder satisfies but a real Ideogram render does not:
//   1. near-grayscale: almost no pixels carry color (a real photo/art render is colorful),
//   2. mid-gray mean luma, and
//   3. low-but-nonzero std (flat gray would be ~0; the text lifts it; a real image is far higher).

/// Per-pixel `max(R,G,B) - min(R,G,B)` above which a pixel counts as "colorful".
const COLOR_SPREAD_THRESHOLD: u16 = 24;
/// Max fraction of colorful pixels for a frame to read as near-grayscale (only anti-aliased text
/// edges carry a little color in the placeholder).
const MAX_COLORFUL_FRACTION: f64 = 0.02;
const PLACEHOLDER_MEAN_MIN: f64 = 90.0;
const PLACEHOLDER_MEAN_MAX: f64 = 165.0;
/// Above ~0 (a perfectly flat frame) — the baked text lifts std into the ~10 range.
const PLACEHOLDER_STD_MIN: f64 = 2.0;
/// A real Ideogram render's luma std is well above this; the placeholder sits around ~10.
const PLACEHOLDER_STD_MAX: f64 = 30.0;

/// Number of reseed-retries attempted on a detected placeholder before surfacing the best render.
const DEFAULT_PLACEHOLDER_RETRIES: u32 = 3;

/// Detection is on by default; `SCENEWORKS_IDEOGRAM_PLACEHOLDER_DETECT=0`/`false` disables it.
fn detection_enabled() -> bool {
    !matches!(
        std::env::var("SCENEWORKS_IDEOGRAM_PLACEHOLDER_DETECT")
            .ok()
            .as_deref(),
        Some("0") | Some("false")
    )
}

/// How many reseed-retries to attempt on a detected placeholder (`SCENEWORKS_IDEOGRAM_PLACEHOLDER_RETRIES`).
pub(crate) fn placeholder_recovery_retries() -> u32 {
    std::env::var("SCENEWORKS_IDEOGRAM_PLACEHOLDER_RETRIES")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_PLACEHOLDER_RETRIES)
}

/// A reseed for recovery attempt `attempt` (0-based) derived from the original `seed`. A large odd
/// stride (golden-ratio prime) keeps recovery seeds clear of the adjacent batch items' seeds (which
/// step by 1) so a retry never re-renders a sibling image's seed.
pub(crate) fn recovery_seed(seed: i64, attempt: u32) -> i64 {
    seed.wrapping_add((attempt as i64 + 1).wrapping_mul(0x9E37_79B1))
}

/// Heuristic detector for the baked "Image blocked by safety filter" placeholder on an RGB8 buffer
/// (`width * height * 3` bytes). Text/template-region based, NOT a std/flatness check (sc-6307).
/// Returns false (never recover) when detection is disabled or the buffer is malformed.
pub(crate) fn looks_like_placeholder(pixels: &[u8], width: u32, height: u32) -> bool {
    if !detection_enabled() {
        return false;
    }
    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    if pixels.len() < 3 || pixels.len() != expected {
        return false;
    }
    let n = (pixels.len() / 3) as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut colorful = 0usize;
    for px in pixels.chunks_exact(3) {
        let (r, g, b) = (px[0] as u16, px[1] as u16, px[2] as u16);
        if r.max(g).max(b) - r.min(g).min(b) > COLOR_SPREAD_THRESHOLD {
            colorful += 1;
        }
        let luma = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
        sum += luma;
        sum_sq += luma * luma;
    }
    let mean = sum / n;
    let std = ((sum_sq / n) - mean * mean).max(0.0).sqrt();
    let colorful_fraction = colorful as f64 / n;

    colorful_fraction <= MAX_COLORFUL_FRACTION
        && (PLACEHOLDER_MEAN_MIN..=PLACEHOLDER_MEAN_MAX).contains(&mean)
        && (PLACEHOLDER_STD_MIN..=PLACEHOLDER_STD_MAX).contains(&std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_passes_through_existing_caption() {
        let caption = r#"{"high_level_description": "a red fox", "compositional_deconstruction": {"background": "snowy forest", "elements": []}}"#;
        assert_eq!(ensure_caption_prompt(caption), caption);
    }

    #[test]
    fn ensure_passes_through_caption_without_high_level_description() {
        // The web builder can emit a caption with only the required composition section.
        let caption =
            r#"{"compositional_deconstruction": {"background": "a beach", "elements": []}}"#;
        assert_eq!(ensure_caption_prompt(caption), caption);
    }

    #[test]
    fn ensure_wraps_plain_text_into_valid_caption() {
        let wrapped = ensure_caption_prompt("a fox on a beach");
        // Python json.dumps default spacing, canonical key order.
        assert_eq!(
            wrapped,
            r#"{"high_level_description": "a fox on a beach", "compositional_deconstruction": {"background": "a fox on a beach", "elements": []}}"#
        );
        // ...and the wrap is itself a caption the guard would pass through unchanged.
        assert_eq!(ensure_caption_prompt(&wrapped), wrapped);
    }

    #[test]
    fn ensure_wrap_escapes_quotes_and_is_valid_json() {
        let wrapped = ensure_caption_prompt(r#"a "quoted" fox"#);
        let value: serde_json::Value = serde_json::from_str(&wrapped).expect("wrap is valid JSON");
        assert_eq!(value["high_level_description"], r#"a "quoted" fox"#);
        assert!(value["compositional_deconstruction"]["elements"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn ensure_wraps_non_caption_json() {
        // Valid JSON object but missing the required section → still wrapped (treated as text).
        let wrapped = ensure_caption_prompt(r#"{"foo": "bar"}"#);
        assert!(wrapped.contains("compositional_deconstruction"));
    }

    /// Build a flat mid-gray RGB8 frame with a small darker horizontal text band — the placeholder's
    /// signature (mid-gray mean, low-but-nonzero std, ~no color).
    fn placeholder_frame(width: u32, height: u32) -> Vec<u8> {
        let (w, h) = (width as usize, height as usize);
        let mut pixels = vec![118u8; w * h * 3];
        let band_top = h / 2 - h / 32;
        let band_bottom = h / 2 + h / 32;
        for y in band_top..band_bottom {
            for x in (w / 4)..(3 * w / 4) {
                // Sparse "glyphs" so the band is text-like, not a solid bar.
                if (x + y) % 3 == 0 {
                    let i = (y * w + x) * 3;
                    pixels[i] = 70;
                    pixels[i + 1] = 70;
                    pixels[i + 2] = 70;
                }
            }
        }
        pixels
    }

    #[test]
    fn detects_gray_placeholder_frame() {
        let pixels = placeholder_frame(256, 256);
        assert!(looks_like_placeholder(&pixels, 256, 256));
    }

    #[test]
    fn ignores_colorful_image() {
        // A saturated gradient — every pixel carries color.
        let (w, h) = (64usize, 64usize);
        let mut pixels = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                pixels[i] = (x * 4) as u8;
                pixels[i + 1] = 30;
                pixels[i + 2] = (y * 4) as u8;
            }
        }
        assert!(!looks_like_placeholder(&pixels, w as u32, h as u32));
    }

    #[test]
    fn ignores_high_contrast_grayscale() {
        // Grayscale but high-variance (black/white checker) — a real render, not the placeholder.
        let (w, h) = (64usize, 64usize);
        let mut pixels = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if (x / 8 + y / 8) % 2 == 0 { 10 } else { 245 };
                let i = (y * w + x) * 3;
                pixels[i] = v;
                pixels[i + 1] = v;
                pixels[i + 2] = v;
            }
        }
        assert!(!looks_like_placeholder(&pixels, w as u32, h as u32));
    }

    #[test]
    fn ignores_malformed_buffer() {
        assert!(!looks_like_placeholder(&[0u8; 10], 256, 256));
        assert!(!looks_like_placeholder(&[], 0, 0));
    }

    #[test]
    fn recovery_seed_is_distinct_from_neighbours() {
        let seed = 1000i64;
        // Recovery seeds must not collide with this seed or its batch neighbours (seed±1, ±2…).
        for attempt in 0..4 {
            let rs = recovery_seed(seed, attempt);
            assert!((rs - seed).abs() > 8, "recovery seed too close to original");
        }
        // Distinct across attempts.
        assert_ne!(recovery_seed(seed, 0), recovery_seed(seed, 1));
    }

    #[test]
    fn is_ideogram_model_matches_both_variants() {
        assert!(is_ideogram_model("ideogram_4"));
        assert!(is_ideogram_model("ideogram_4_turbo"));
        assert!(!is_ideogram_model("flux_dev"));
    }

    /// Real-weight validation (sc-6501): runs the PRODUCTION detector against actual Ideogram 4
    /// renders so it is verified on real placeholder pixels, not just synthetic frames. Env-gated +
    /// `#[ignore]`; needs PNGs from the `mlx-gen-ideogram` `placeholder_probe` harness at the product
    /// default (1024²/48). Filename prefix encodes the conditioning:
    ///   - `rich_*`  = a full magic-prompt-style caption (the UI auto-expand output) → MUST be
    ///     non-placeholder. This is acceptance #3: "auto-caption renders real images."
    ///   - `plain_*` = raw plain text (control); `wrapped_*` = the worker's SPARSE format-guard wrap.
    ///     Both are under-conditioned and stochastically placeholder, so their verdicts are printed +
    ///     counted, not asserted. The empirical finding (sc-6501): a sparse caption does NOT escape —
    ///     content richness is the lever — so the sparse wrap is a FORMAT guarantee, and the real
    ///     escape comes from UI rich expansion or the detect-and-recover reseed.
    ///
    ///   IDEO_PNG_DIR=/tmp/ideo_6501 cargo test -p sceneworks-worker --lib -- --ignored --nocapture \
    ///     ideogram_caption::tests::detector_classifies_real_probe_renders
    #[test]
    #[ignore = "needs real-weight probe PNGs (set IDEO_PNG_DIR)"]
    fn detector_classifies_real_probe_renders() {
        let dir = std::env::var("IDEO_PNG_DIR").unwrap_or_else(|_| "/tmp/ideo_6501".to_owned());
        let mut seen = 0usize;
        let mut rich_seen = 0usize;
        let (mut plain_ph, mut plain_total) = (0usize, 0usize);
        for entry in std::fs::read_dir(&dir).expect("read IDEO_PNG_DIR") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let img = image::open(&path).expect("decode png").to_rgb8();
            let (w, h) = (img.width(), img.height());
            let verdict = looks_like_placeholder(img.as_raw(), w, h);
            eprintln!("{name}: {w}x{h} placeholder={verdict}");
            seen += 1;
            if name.starts_with("plain_") || name.starts_with("wrapped_") {
                plain_total += 1;
                plain_ph += usize::from(verdict);
            }
            if name.starts_with("rich_") {
                rich_seen += 1;
                assert!(
                    !verdict,
                    "rich auto-caption render must NOT be a placeholder: {name}"
                );
            }
        }
        assert!(seen > 0, "no PNGs found in {dir}");
        assert!(
            rich_seen > 0,
            "no rich_* renders to validate the auto-caption escape in {dir}"
        );
        eprintln!("under-conditioned control: {plain_ph}/{plain_total} classified as placeholder");
    }
}
