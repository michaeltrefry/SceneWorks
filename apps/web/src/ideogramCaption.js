// Ideogram 4 structured JSON-caption contract (epic 4725, sc-5993).
//
// Ideogram 4 was trained EXCLUSIVELY on structured JSON captions; plain text is
// out-of-distribution and produces prompt-agnostic images. This module is the
// SceneWorks-side implementation of that caption format (the realization of
// spike sc-4727): a typed representation, a serializer that emits keys in the
// EXACT order the model was trained on, a validator that mirrors Ideogram's
// `CaptionVerifier` (src/ideogram4/caption_verifier.py), and the recipe shape
// the studio stores so a generation can be replayed.
//
// The serializer reproduces `json.dumps(caption, ensure_ascii=False)` with
// Python's default separators (", " and ": ") byte-for-byte — that is the
// string the native MLX engine tokenizes, and the format the validated
// reference render was produced with. JS `JSON.stringify(obj)` drops those
// separator spaces, so we emit by hand.
//
// The builder UI (sc-5994), bbox canvas (sc-5995), palette picker (sc-5996) and
// magic-prompt expander (sc-5997) all sit on top of this contract.

// ----- schema constants (mirror CaptionVerifier) ---------------------------

export const TOP_LEVEL_KEYS = Object.freeze([
  "high_level_description",
  "style_description",
  "compositional_deconstruction",
]);

// Note the discriminator's position differs: `photo` sits at index 2, but
// `art_style` sits at index 3 (after `medium`). They are NOT interchangeable in
// place — this is the verifier's actual rule.
export const STYLE_KEY_ORDER_PHOTO = Object.freeze([
  "aesthetics",
  "lighting",
  "photo",
  "medium",
  "color_palette",
]);
export const STYLE_KEY_ORDER_NON_PHOTO = Object.freeze([
  "aesthetics",
  "lighting",
  "medium",
  "art_style",
  "color_palette",
]);
export const STYLE_KNOWN_KEYS = Object.freeze([
  "aesthetics",
  "lighting",
  "photo",
  "art_style",
  "medium",
  "color_palette",
]);

export const COMPOSITION_KEY_ORDER = Object.freeze(["background", "elements"]);

export const ELEMENT_KEY_ORDER_OBJ = Object.freeze(["type", "bbox", "desc", "color_palette"]);
export const ELEMENT_KEY_ORDER_TEXT = Object.freeze([
  "type",
  "bbox",
  "text",
  "desc",
  "color_palette",
]);
export const ELEMENT_KNOWN_KEYS = Object.freeze(["type", "bbox", "text", "desc", "color_palette"]);
export const ELEMENT_TYPES = Object.freeze(["obj", "text"]);

export const BBOX_MIN = 0;
export const BBOX_MAX = 1000;
export const STYLE_PALETTE_MAX = 16;
export const ELEMENT_PALETTE_MAX = 5;

// ----- typed-model factories -----------------------------------------------

// A structurally-valid empty caption: `compositional_deconstruction` is the
// only required section, with a (string) background and an elements list.
// high_level_description / style_description are optional and omitted until the
// user fills them, so an untouched skeleton never serializes empty keys.
export function emptyCaption() {
  return {
    compositional_deconstruction: { background: "", elements: [] },
  };
}

export function makeObjElement({ bbox = null, desc = "", color_palette = null } = {}) {
  const elem = { type: "obj" };
  if (bbox != null) elem.bbox = bbox;
  elem.desc = desc;
  if (color_palette != null) elem.color_palette = color_palette;
  return elem;
}

export function makeTextElement({ bbox = null, text = "", desc = "", color_palette = null } = {}) {
  const elem = { type: "text" };
  if (bbox != null) elem.bbox = bbox;
  elem.text = text;
  elem.desc = desc;
  if (color_palette != null) elem.color_palette = color_palette;
  return elem;
}

// ----- bbox + palette helpers ----------------------------------------------

export function isValidBbox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return false;
  if (!bbox.every((v) => Number.isInteger(v))) return false;
  if (!bbox.every((v) => v >= BBOX_MIN && v <= BBOX_MAX)) return false;
  const [ymin, xmin, ymax, xmax] = bbox;
  return ymin <= ymax && xmin <= xmax;
}

// Coerce a single bbox coordinate into an in-range integer.
export function clampBboxValue(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return BBOX_MIN;
  return Math.min(BBOX_MAX, Math.max(BBOX_MIN, n));
}

const HEX_COLOR_RE = /^#[0-9A-F]{6}$/;

export function isValidHexColor(color) {
  return typeof color === "string" && HEX_COLOR_RE.test(color);
}

// Uppercase + validate a hex color, accepting lowercase input. Returns null when
// it is not a #RRGGBB string (the model only accepts uppercase 6-digit hex).
export function normalizeHexColor(color) {
  if (typeof color !== "string") return null;
  const upper = color.trim().toUpperCase();
  return HEX_COLOR_RE.test(upper) ? upper : null;
}

// ----- canonical serialization ---------------------------------------------

// Python-`json.dumps(value, ensure_ascii=False)` default formatting: compact,
// ", " between items, ": " after keys, literal (non-escaped) UTF-8. JSON
// `JSON.stringify` already escapes strings the ensure_ascii=False way (it keeps
// literal non-ASCII and escapes control chars / `"` / `\`); we only need to
// supply the separator spaces and preserve object key insertion order.
function emit(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "[" + value.map(emit).join(", ") + "]";
  if (typeof value === "object") {
    const parts = Object.keys(value).map((k) => JSON.stringify(k) + ": " + emit(value[k]));
    return "{" + parts.join(", ") + "}";
  }
  return JSON.stringify(value);
}

function styleOrderFor(style) {
  const hasPhoto = "photo" in style;
  const hasArt = "art_style" in style;
  // When the discriminator is ambiguous (both/neither) serialization still has
  // to produce something; validation reports the real problem separately.
  if (hasPhoto && !hasArt) return STYLE_KEY_ORDER_PHOTO;
  if (hasArt && !hasPhoto) return STYLE_KEY_ORDER_NON_PHOTO;
  if (hasPhoto) return STYLE_KEY_ORDER_PHOTO;
  return STYLE_KEY_ORDER_NON_PHOTO;
}

function elementOrderFor(element) {
  return element?.type === "text" ? ELEMENT_KEY_ORDER_TEXT : ELEMENT_KEY_ORDER_OBJ;
}

// Build a deep copy with keys in the model's canonical order, dropping any keys
// not in the schema so the engine payload is always order-clean. This is what
// guarantees we never emit schema-valid-but-order-drifted JSON.
export function orderCaption(caption) {
  const out = {};
  if (!caption || typeof caption !== "object") return out;
  for (const key of TOP_LEVEL_KEYS) {
    if (!(key in caption)) continue;
    if (key === "style_description") {
      out.style_description = orderObjectKeys(caption.style_description, styleOrderFor(caption.style_description));
    } else if (key === "compositional_deconstruction") {
      out.compositional_deconstruction = orderComposition(caption.compositional_deconstruction);
    } else {
      out[key] = caption[key];
    }
  }
  return out;
}

function orderComposition(cd) {
  if (!cd || typeof cd !== "object") return cd;
  const out = {};
  if ("background" in cd) out.background = cd.background;
  if ("elements" in cd) {
    out.elements = Array.isArray(cd.elements)
      ? cd.elements.map((el) => orderObjectKeys(el, elementOrderFor(el)))
      : cd.elements;
  }
  return out;
}

function orderObjectKeys(obj, order) {
  if (!obj || typeof obj !== "object") return obj;
  const out = {};
  for (const key of order) {
    if (key in obj) out[key] = obj[key];
  }
  return out;
}

// The exact string the engine tokenizes. Always canonical order, ensure_ascii=False.
export function serializeCaption(caption) {
  return emit(orderCaption(caption));
}

export function parseCaption(text) {
  try {
    return { caption: JSON.parse(text), error: null };
  } catch (e) {
    return { caption: null, error: e instanceof Error ? e.message : String(e) };
  }
}

// Turn a magic-prompt model reply (sc-5997) into a schema-clean caption: parse it,
// drop the non-schema top-level `aspect_ratio` key the prompt emits, and (by default,
// matching the reference) strip per-element bboxes — the model's box guesses are
// unreliable and the user places boxes themselves. The result is validated by the
// caller; serializeCaption drops any remaining unknown keys.
export function parseMagicPromptCaption(rawText, { stripBboxes = true } = {}) {
  const { caption, error } = parseCaption(rawText);
  if (error) return { caption: null, error };
  if (!caption || typeof caption !== "object" || Array.isArray(caption)) {
    return { caption: null, error: "Magic-prompt did not return a JSON object." };
  }
  const out = { ...caption };
  delete out.aspect_ratio;
  const cd = out.compositional_deconstruction;
  if (stripBboxes && cd && typeof cd === "object" && Array.isArray(cd.elements)) {
    out.compositional_deconstruction = {
      ...cd,
      elements: cd.elements.map((el) => {
        if (el && typeof el === "object" && "bbox" in el) {
          const { bbox: _bbox, ...rest } = el;
          return rest;
        }
        return el;
      }),
    };
  }
  return { caption: out, error: null };
}

// ----- verifier (faithful mirror of CaptionVerifier) -----------------------
//
// Returns structured issues `{ code, path, severity, message }`. `severity` is
// the model-side default used by `validateCaption`; "error" blocks generation,
// "warning" is advisory. Mirrors which checks fire and where, not the exact
// Python message strings.

function pushUnknownKeys(obj, known, path, issues) {
  const unknown = Object.keys(obj).filter((k) => !known.includes(k));
  if (unknown.length) {
    issues.push({
      code: "unknown_keys",
      path,
      severity: "error",
      message: `${path}: unknown keys [${unknown.join(", ")}] (not in schema)`,
    });
  }
}

function pushKeyOrder(obj, expectedOrder, path, issues) {
  const present = Object.keys(obj).filter((k) => expectedOrder.includes(k));
  const sameOrder =
    present.length === expectedOrder.length && present.every((k, i) => k === expectedOrder[i]);
  if (!sameOrder) {
    issues.push({
      code: "key_order",
      path,
      severity: "warning",
      message: `${path}: key order is [${present.join(", ")}], expected [${expectedOrder.join(", ")}]`,
    });
  }
  const extra = Object.keys(obj).filter((k) => !expectedOrder.includes(k));
  if (extra.length) {
    issues.push({
      code: "key_context",
      path,
      severity: "error",
      message: `${path}: keys [${extra.join(", ")}] are not allowed in this context`,
    });
  }
}

function typeName(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function verifyColorPalette(palette, path, max, issues) {
  if (!Array.isArray(palette)) {
    issues.push({ code: "bad_palette", path, severity: "error", message: `${path}: expected a list` });
    return;
  }
  if (palette.length > max) {
    issues.push({
      code: "bad_palette",
      path,
      severity: "error",
      message: `${path}: too many colors (${palette.length}), expected at most ${max}`,
    });
    return;
  }
  palette.forEach((color, i) => {
    if (!isValidHexColor(color)) {
      issues.push({
        code: "bad_palette",
        path: `${path}[${i}]`,
        severity: "error",
        message: `${path}[${i}]: '${color}' is not a valid #RRGGBB hex color`,
      });
    }
  });
}

function verifyBbox(bbox, path, issues) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    issues.push({ code: "bad_bbox", path, severity: "error", message: `${path}: expected [ymin, xmin, ymax, xmax]` });
    return;
  }
  if (!bbox.every((v) => Number.isInteger(v))) {
    issues.push({ code: "bad_bbox", path, severity: "error", message: `${path}: all values must be integers` });
    return;
  }
  const [ymin, xmin, ymax, xmax] = bbox;
  if (!bbox.every((v) => v >= BBOX_MIN && v <= BBOX_MAX)) {
    issues.push({
      code: "bad_bbox",
      path,
      severity: "error",
      message: `${path}: values must be in [${BBOX_MIN}, ${BBOX_MAX}], got [${bbox.join(", ")}]`,
    });
  }
  if (ymin > ymax) {
    issues.push({ code: "bad_bbox", path, severity: "error", message: `${path}: ymin (${ymin}) > ymax (${ymax})` });
  }
  if (xmin > xmax) {
    issues.push({ code: "bad_bbox", path, severity: "error", message: `${path}: xmin (${xmin}) > xmax (${xmax})` });
  }
}

function styleOrderPresentOnly(style) {
  // The verifier's expected order includes color_palette only when present.
  const base = styleOrderFor(style);
  return base.filter((k) => k !== "color_palette" || "color_palette" in style);
}

function elementOrderPresentOnly(element) {
  // bbox + color_palette appear in the expected order only when present.
  return elementOrderFor(element).filter((k) => {
    if (k === "bbox") return "bbox" in element;
    if (k === "color_palette") return "color_palette" in element;
    return true;
  });
}

function verifyStyle(style, issues) {
  if (style === null || typeof style !== "object" || Array.isArray(style)) {
    issues.push({ code: "bad_type", path: "style_description", severity: "error", message: "style_description: expected an object" });
    return;
  }
  pushUnknownKeys(style, STYLE_KNOWN_KEYS, "style_description", issues);
  const hasPhoto = "photo" in style;
  const hasArt = "art_style" in style;
  if (hasPhoto && hasArt) {
    issues.push({
      code: "style_discriminator",
      path: "style_description",
      severity: "error",
      message: "style_description: contains both 'photo' and 'art_style'; expected exactly one",
    });
    return;
  }
  if (!hasPhoto && !hasArt) {
    issues.push({
      code: "style_discriminator",
      path: "style_description",
      severity: "error",
      message: "style_description: expected one of 'photo' (photo captions) or 'art_style' (non-photo captions)",
    });
    return;
  }
  pushKeyOrder(style, styleOrderPresentOnly(style), "style_description", issues);
  if ("color_palette" in style) {
    verifyColorPalette(style.color_palette, "style_description.color_palette", STYLE_PALETTE_MAX, issues);
  }
}

function verifyElement(element, i, issues) {
  const path = `elements[${i}]`;
  if (element === null || typeof element !== "object" || Array.isArray(element)) {
    issues.push({ code: "bad_type", path, severity: "error", message: `${path}: expected an object` });
    return;
  }
  pushUnknownKeys(element, ELEMENT_KNOWN_KEYS, path, issues);
  if (!("type" in element)) {
    issues.push({ code: "missing_section", path, severity: "error", message: `${path}: 'type' must exist` });
    return;
  }
  if (!ELEMENT_TYPES.includes(element.type)) {
    issues.push({
      code: "bad_type",
      path,
      severity: "error",
      message: `${path}: 'type' must be one of [${ELEMENT_TYPES.join(", ")}]`,
    });
    return;
  }
  pushKeyOrder(element, elementOrderPresentOnly(element), path, issues);
  if ("bbox" in element) verifyBbox(element.bbox, `${path}.bbox`, issues);
  if ("color_palette" in element) {
    verifyColorPalette(element.color_palette, `${path}.color_palette`, ELEMENT_PALETTE_MAX, issues);
  }
}

function verifyComposition(cd, issues) {
  if (cd === null || typeof cd !== "object" || Array.isArray(cd)) {
    issues.push({ code: "bad_type", path: "compositional_deconstruction", severity: "error", message: "compositional_deconstruction: expected an object" });
    return;
  }
  if (!("background" in cd)) {
    issues.push({ code: "missing_section", path: "compositional_deconstruction.background", severity: "error", message: "compositional_deconstruction: 'background' must exist" });
    return;
  }
  if (typeof cd.background !== "string") {
    issues.push({
      code: "bad_type",
      path: "compositional_deconstruction.background",
      severity: "error",
      message: `compositional_deconstruction.background: expected a string, got ${typeName(cd.background)}`,
    });
    return;
  }
  if (!("elements" in cd)) {
    issues.push({ code: "missing_section", path: "compositional_deconstruction.elements", severity: "error", message: "compositional_deconstruction: 'elements' must exist" });
    return;
  }
  pushKeyOrder(cd, COMPOSITION_KEY_ORDER, "compositional_deconstruction", issues);
  if (!Array.isArray(cd.elements)) {
    issues.push({ code: "bad_type", path: "compositional_deconstruction.elements", severity: "error", message: "compositional_deconstruction.elements: expected a list" });
    return;
  }
  cd.elements.forEach((el, i) => verifyElement(el, i, issues));
}

// Faithful CaptionVerifier.verify(caption) — structured issues.
export function verifyCaption(caption) {
  const issues = [];
  if (caption === null || typeof caption !== "object" || Array.isArray(caption)) {
    issues.push({ code: "bad_type", path: "root", severity: "error", message: `root: expected a JSON object, got ${typeName(caption)}` });
    return issues;
  }
  pushUnknownKeys(caption, TOP_LEVEL_KEYS, "root", issues);
  if ("high_level_description" in caption && typeof caption.high_level_description !== "string") {
    issues.push({
      code: "bad_type",
      path: "high_level_description",
      severity: "error",
      message: `high_level_description: expected a string, got ${typeName(caption.high_level_description)}`,
    });
  }
  if ("style_description" in caption) verifyStyle(caption.style_description, issues);
  if ("compositional_deconstruction" in caption) {
    verifyComposition(caption.compositional_deconstruction, issues);
  } else {
    issues.push({ code: "missing_section", path: "compositional_deconstruction", severity: "error", message: "root: 'compositional_deconstruction' must exist" });
  }
  return issues;
}

// Detect `\uXXXX` escapes for non-ASCII with no literal non-ASCII — usually a
// sign the JSON was written with ensure_ascii=True (advisory, mirrors verifier).
const NON_ASCII_ESCAPE_RE = /\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})/g;

export function verifyRaw(rawText) {
  const issues = [];
  const escapes = rawText.match(NON_ASCII_ESCAPE_RE);
  const hasLiteralNonAscii = [...rawText].some((c) => c.charCodeAt(0) > 0x7f);
  if (escapes && !hasLiteralNonAscii) {
    issues.push({
      code: "ensure_ascii",
      path: "raw",
      severity: "warning",
      message:
        "raw text: found non-ASCII unicode escapes and no literal non-ASCII characters; " +
        "usually means the JSON was saved with ensure_ascii=True. Prefer literal UTF-8.",
    });
  }
  const { caption, error } = parseCaption(rawText);
  if (error) {
    issues.push({ code: "invalid_json", path: "root", severity: "error", message: `invalid JSON: ${error}` });
    return issues;
  }
  return issues.concat(verifyCaption(caption));
}

// ----- SceneWorks validation policy ----------------------------------------
//
// Wraps the faithful verifier with SceneWorks UX policy and the extra,
// non-verifier checks the story calls for (recommended sections, empty
// elements, conflicting plain-text + JSON). Accepts either a parsed caption
// object or a raw JSON string.
//
// Key-order issues are surfaced as warnings, not errors: `serializeCaption`
// canonicalizes order before the string ever reaches the engine, so drift is
// auto-corrected — we report it so the user knows we reordered, but it does not
// block. Everything else from the verifier blocks.
export function validateCaption(input, { plainText = "" } = {}) {
  const errors = [];
  const warnings = [];

  let caption = input;
  let issues;
  if (typeof input === "string") {
    issues = verifyRaw(input);
    const parsed = parseCaption(input);
    caption = parsed.caption;
  } else {
    issues = verifyCaption(input);
  }

  for (const issue of issues) {
    (issue.severity === "warning" ? warnings : errors).push(issue);
  }

  // SceneWorks advisories (non-blocking) — only meaningful once the caption parsed.
  if (caption && typeof caption === "object" && !Array.isArray(caption)) {
    if (!("high_level_description" in caption) || !caption.high_level_description) {
      warnings.push({
        code: "missing_recommended",
        path: "high_level_description",
        severity: "warning",
        message: "Recommended: add a one-sentence high_level_description for better adherence.",
      });
    }
    if (!("style_description" in caption)) {
      warnings.push({
        code: "missing_recommended",
        path: "style_description",
        severity: "warning",
        message: "Recommended: add a style_description to control the look.",
      });
    }
    const elements = caption?.compositional_deconstruction?.elements;
    if (Array.isArray(elements)) {
      if (elements.length === 0) {
        warnings.push({
          code: "empty_elements",
          path: "compositional_deconstruction.elements",
          severity: "warning",
          message: "No elements placed — the layout is unguided. Add object/text elements with bounding boxes.",
        });
      }
      elements.forEach((el, i) => {
        if (el && typeof el === "object") {
          const hasContent = (el.desc && String(el.desc).trim()) || (el.type === "text" && el.text && String(el.text).trim());
          if (!hasContent) {
            warnings.push({
              code: "empty_element",
              path: `elements[${i}]`,
              severity: "warning",
              message: `elements[${i}]: has no description${el.type === "text" ? "/text" : ""}.`,
            });
          }
        }
      });
    }
  }

  // Conflicting plain-text + JSON: the JSON caption is authoritative; a leftover
  // plain-text prompt that is not just the high_level_description would be
  // silently ignored, so surface it.
  if (caption && typeof plainText === "string" && plainText.trim()) {
    const hld = caption.high_level_description ? String(caption.high_level_description).trim() : "";
    if (plainText.trim() !== hld) {
      warnings.push({
        code: "conflicting_inputs",
        path: "prompt",
        severity: "warning",
        message: "A plain-text prompt is present alongside the JSON caption. The JSON caption is used for generation; the plain text is kept only as the original intent.",
      });
    }
  }

  const ok = errors.length === 0;
  return {
    ok,
    errors,
    warnings,
    issues: errors.concat(warnings),
    caption: caption ?? null,
    serialized: ok && caption ? serializeCaption(caption) : null,
  };
}

// ----- recipe / raw-settings storage ---------------------------------------
//
// Stored under the image recipe so a structured-prompt generation can be
// replayed and re-edited. Captures: the original plain-text intent (the
// magic-prompt seed), the edited JSON caption (which carries bboxes + palette
// choices inside its elements/style), the magic-prompt backend that drafted it,
// whether the user hand-edited it, and the exact runtime payload string sent to
// the engine.

export const STRUCTURED_PROMPT_RECIPE_VERSION = 1;

export function buildStructuredPromptRecipe({
  intent = "",
  caption,
  magicPromptBackend = null,
  edited = false,
} = {}) {
  return {
    version: STRUCTURED_PROMPT_RECIPE_VERSION,
    intent: String(intent ?? ""),
    caption: caption ?? null,
    magicPromptBackend: magicPromptBackend ?? null,
    edited: Boolean(edited),
    runtimePrompt: caption ? serializeCaption(caption) : "",
  };
}

// The exact `prompt` string to send in the job payload. Prefers the stored
// runtime payload, falling back to re-serializing the caption.
export function runtimePromptFromRecipe(recipe) {
  if (!recipe) return "";
  if (typeof recipe.runtimePrompt === "string" && recipe.runtimePrompt) return recipe.runtimePrompt;
  return recipe.caption ? serializeCaption(recipe.caption) : "";
}
