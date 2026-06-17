import { describe, expect, it } from "vitest";
import {
  BBOX_MAX,
  buildStructuredPromptRecipe,
  clampBboxValue,
  emptyCaption,
  isValidBbox,
  isValidHexColor,
  makeObjElement,
  makeTextElement,
  normalizeHexColor,
  orderCaption,
  parseCaption,
  parseMagicPromptCaption,
  runtimePromptFromRecipe,
  serializeCaption,
  STYLE_KEY_ORDER_NON_PHOTO,
  validateCaption,
  verifyCaption,
  verifyRaw,
} from "./ideogramCaption.js";

// The validated reference render's caption, byte-identical to the engine's
// `mlx-gen-ideogram/tests/common/mod.rs` CAPTION_JSON — which is itself
// `json.dumps(CAPTION, ensure_ascii=False)`. Our serializer must reproduce this
// exactly, since this is the string the model tokenizes.
const FOX_JSON =
  '{"high_level_description": "A photograph of a red fox sitting in a snowy forest at golden hour.", "style_description": {"aesthetics": "serene, warm, naturalistic", "lighting": "golden hour, soft warm backlight, long shadows", "photo": "telephoto, shallow depth of field, sharp focus, eye-level", "medium": "photograph"}, "compositional_deconstruction": {"background": "A snowy forest of tall pine trees, soft golden sunlight filtering through the branches, snow on the ground.", "elements": [{"type": "obj", "bbox": [250, 320, 950, 760], "desc": "A red fox with vivid orange fur, white chest and a thick bushy tail, sitting upright in the snow and facing the camera."}]}}';

const FOX = {
  high_level_description: "A photograph of a red fox sitting in a snowy forest at golden hour.",
  style_description: {
    aesthetics: "serene, warm, naturalistic",
    lighting: "golden hour, soft warm backlight, long shadows",
    photo: "telephoto, shallow depth of field, sharp focus, eye-level",
    medium: "photograph",
  },
  compositional_deconstruction: {
    background:
      "A snowy forest of tall pine trees, soft golden sunlight filtering through the branches, snow on the ground.",
    elements: [
      {
        type: "obj",
        bbox: [250, 320, 950, 760],
        desc: "A red fox with vivid orange fur, white chest and a thick bushy tail, sitting upright in the snow and facing the camera.",
      },
    ],
  },
};

describe("serializeCaption — training format", () => {
  it("reproduces the engine CAPTION_JSON byte-for-byte", () => {
    expect(serializeCaption(FOX)).toBe(FOX_JSON);
  });

  it("round-trips: serialize -> parse -> serialize is stable", () => {
    const once = serializeCaption(FOX);
    const { caption } = parseCaption(once);
    expect(serializeCaption(caption)).toBe(once);
  });

  it("uses Python default separators (', ' and ': '), not JSON.stringify defaults", () => {
    const s = serializeCaption({ compositional_deconstruction: { background: "x", elements: [] } });
    expect(s).toBe('{"compositional_deconstruction": {"background": "x", "elements": []}}');
  });

  it("keeps non-ASCII literal (ensure_ascii=False)", () => {
    const s = serializeCaption({ compositional_deconstruction: { background: "café au lait", elements: [] } });
    expect(s).toContain("café");
    expect(s).not.toContain("\\u");
  });
});

describe("orderCaption — key-order preservation", () => {
  it("canonicalizes scrambled top-level + nested keys to training order", () => {
    const scrambled = {
      compositional_deconstruction: {
        elements: [{ desc: "d", bbox: [0, 0, 10, 10], type: "obj" }],
        background: "bg",
      },
      style_description: {
        medium: "photograph",
        photo: "eye-level",
        lighting: "soft",
        aesthetics: "warm",
      },
      high_level_description: "hi",
    };
    expect(serializeCaption(scrambled)).toBe(
      '{"high_level_description": "hi", "style_description": {"aesthetics": "warm", "lighting": "soft", "photo": "eye-level", "medium": "photograph"}, "compositional_deconstruction": {"background": "bg", "elements": [{"type": "obj", "bbox": [0, 0, 10, 10], "desc": "d"}]}}',
    );
  });

  it("orders non-photo style as aesthetics, lighting, medium, art_style, color_palette", () => {
    const ordered = orderCaption({
      style_description: {
        art_style: "watercolor",
        color_palette: ["#FFFFFF"],
        medium: "painting",
        lighting: "soft",
        aesthetics: "dreamy",
      },
      compositional_deconstruction: { background: "b", elements: [] },
    });
    expect(Object.keys(ordered.style_description)).toEqual([...STYLE_KEY_ORDER_NON_PHOTO]);
  });

  it("orders text-element keys as type, bbox, text, desc, color_palette", () => {
    const ordered = orderCaption({
      compositional_deconstruction: {
        background: "b",
        elements: [{ color_palette: ["#000000"], desc: "d", text: "HELLO", bbox: [1, 2, 3, 4], type: "text" }],
      },
    });
    expect(Object.keys(ordered.compositional_deconstruction.elements[0])).toEqual([
      "type",
      "bbox",
      "text",
      "desc",
      "color_palette",
    ]);
  });

  it("drops unknown keys from the serialized payload", () => {
    const s = serializeCaption({
      bogus: "nope",
      compositional_deconstruction: { background: "b", elements: [], extra: 1 },
    });
    expect(s).toBe('{"compositional_deconstruction": {"background": "b", "elements": []}}');
  });
});

describe("bbox + palette helpers", () => {
  it("validates bbox shape, integer-ness, range and ordering", () => {
    expect(isValidBbox([0, 0, 1000, 1000])).toBe(true);
    expect(isValidBbox([250, 320, 950, 760])).toBe(true);
    expect(isValidBbox([0, 0, 10])).toBe(false); // wrong length
    expect(isValidBbox([0, 0, 10.5, 10])).toBe(false); // non-integer
    expect(isValidBbox([0, 0, 1001, 10])).toBe(false); // out of range
    expect(isValidBbox([900, 0, 100, 10])).toBe(false); // ymin > ymax
  });

  it("clamps bbox coordinates into range", () => {
    expect(clampBboxValue(-50)).toBe(0);
    expect(clampBboxValue(5000)).toBe(BBOX_MAX);
    expect(clampBboxValue(250.7)).toBe(251);
  });

  it("validates and normalizes hex colors (uppercase only)", () => {
    expect(isValidHexColor("#FF00AA")).toBe(true);
    expect(isValidHexColor("#ff00aa")).toBe(false); // lowercase not accepted by the model
    expect(isValidHexColor("FF00AA")).toBe(false);
    expect(isValidHexColor("#FFF")).toBe(false);
    expect(normalizeHexColor("#ff00aa")).toBe("#FF00AA");
    expect(normalizeHexColor(" #abcdef ")).toBe("#ABCDEF");
    expect(normalizeHexColor("teal")).toBe(null);
  });
});

describe("validateCaption — failure modes", () => {
  function codes(result) {
    return result.errors.map((e) => e.code);
  }

  it("accepts the canonical fox caption", () => {
    const result = validateCaption(FOX);
    expect(result.ok).toBe(true);
    expect(result.errors).toEqual([]);
    expect(result.serialized).toBe(FOX_JSON);
  });

  it("fails on malformed JSON", () => {
    const result = validateCaption("{not valid json");
    expect(result.ok).toBe(false);
    expect(codes(result)).toContain("invalid_json");
  });

  it("fails when compositional_deconstruction is missing", () => {
    const result = validateCaption({ high_level_description: "hi" });
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.code === "missing_section")).toBe(true);
  });

  it("fails on missing background", () => {
    const result = validateCaption({ compositional_deconstruction: { elements: [] } });
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.path === "compositional_deconstruction.background")).toBe(true);
  });

  it("fails on out-of-range bbox", () => {
    const result = validateCaption({
      compositional_deconstruction: {
        background: "b",
        elements: [{ type: "obj", bbox: [0, 0, 2000, 10], desc: "d" }],
      },
    });
    expect(result.ok).toBe(false);
    expect(codes(result)).toContain("bad_bbox");
  });

  it("fails on inverted bbox (ymin > ymax)", () => {
    const result = validateCaption({
      compositional_deconstruction: {
        background: "b",
        elements: [{ type: "obj", bbox: [900, 0, 100, 10], desc: "d" }],
      },
    });
    expect(result.ok).toBe(false);
    expect(codes(result)).toContain("bad_bbox");
  });

  it("fails on lowercase / over-limit palettes", () => {
    const lower = validateCaption({
      compositional_deconstruction: {
        background: "b",
        elements: [{ type: "obj", bbox: [0, 0, 10, 10], desc: "d", color_palette: ["#ff0000"] }],
      },
    });
    expect(lower.ok).toBe(false);
    expect(codes(lower)).toContain("bad_palette");

    const tooMany = validateCaption({
      compositional_deconstruction: {
        background: "b",
        elements: [
          { type: "obj", bbox: [0, 0, 10, 10], desc: "d", color_palette: ["#000000", "#111111", "#222222", "#333333", "#444444", "#555555"] },
        ],
      },
    });
    expect(tooMany.ok).toBe(false);
    expect(codes(tooMany)).toContain("bad_palette");
  });

  it("fails on invalid element type and unknown keys", () => {
    const badType = validateCaption({
      compositional_deconstruction: { background: "b", elements: [{ type: "widget", desc: "d" }] },
    });
    expect(badType.ok).toBe(false);
    expect(codes(badType)).toContain("bad_type");

    const unknown = validateCaption({
      surprise: 1,
      compositional_deconstruction: { background: "b", elements: [] },
    });
    expect(unknown.ok).toBe(false);
    expect(codes(unknown)).toContain("unknown_keys");
  });

  it("fails when style has both or neither discriminator", () => {
    const both = validateCaption({
      style_description: { aesthetics: "a", lighting: "l", photo: "p", art_style: "x", medium: "m" },
      compositional_deconstruction: { background: "b", elements: [] },
    });
    expect(both.ok).toBe(false);
    expect(codes(both)).toContain("style_discriminator");

    const neither = validateCaption({
      style_description: { aesthetics: "a", lighting: "l", medium: "m" },
      compositional_deconstruction: { background: "b", elements: [] },
    });
    expect(neither.ok).toBe(false);
    expect(codes(neither)).toContain("style_discriminator");
  });

  it("treats key-order drift in pasted JSON as a non-blocking warning (auto-corrected on serialize)", () => {
    // bbox after desc — drifted from canonical type,bbox,desc.
    const drifted =
      '{"compositional_deconstruction": {"elements": [{"type": "obj", "desc": "d", "bbox": [0, 0, 10, 10]}], "background": "b"}}';
    const result = validateCaption(drifted);
    expect(result.ok).toBe(true);
    expect(result.warnings.some((w) => w.code === "key_order")).toBe(true);
    // The emitted engine payload is canonical regardless of the input order.
    expect(result.serialized).toBe(
      '{"compositional_deconstruction": {"background": "b", "elements": [{"type": "obj", "bbox": [0, 0, 10, 10], "desc": "d"}]}}',
    );
  });

  it("warns on conflicting plain-text + JSON inputs without blocking", () => {
    const result = validateCaption(FOX, { plainText: "a totally different idea" });
    expect(result.ok).toBe(true);
    expect(result.warnings.some((w) => w.code === "conflicting_inputs")).toBe(true);
  });

  it("does not warn when plain text equals the high_level_description", () => {
    const result = validateCaption(FOX, { plainText: FOX.high_level_description });
    expect(result.warnings.some((w) => w.code === "conflicting_inputs")).toBe(false);
  });

  it("warns on empty elements and missing recommended sections", () => {
    const result = validateCaption({ compositional_deconstruction: { background: "b", elements: [] } });
    expect(result.ok).toBe(true);
    expect(result.warnings.some((w) => w.code === "empty_elements")).toBe(true);
    expect(result.warnings.some((w) => w.code === "missing_recommended")).toBe(true);
  });
});

describe("parseMagicPromptCaption", () => {
  // What the v1 magic-prompt emits: a top-level aspect_ratio + bboxes on elements.
  const MAGIC_OUTPUT =
    '{"aspect_ratio": "16:9", "high_level_description": "A red fox in the snow", "compositional_deconstruction": {"background": "a snowy forest", "elements": [{"type": "obj", "bbox": [250, 320, 950, 760], "desc": "a red fox"}]}}';

  it("strips the non-schema aspect_ratio key and validates clean", () => {
    const { caption, error } = parseMagicPromptCaption(MAGIC_OUTPUT);
    expect(error).toBe(null);
    expect("aspect_ratio" in caption).toBe(false);
    expect(validateCaption(caption).ok).toBe(true);
  });

  it("strips element bboxes by default but keeps them when asked", () => {
    const stripped = parseMagicPromptCaption(MAGIC_OUTPUT).caption;
    expect("bbox" in stripped.compositional_deconstruction.elements[0]).toBe(false);

    const kept = parseMagicPromptCaption(MAGIC_OUTPUT, { stripBboxes: false }).caption;
    expect(kept.compositional_deconstruction.elements[0].bbox).toEqual([250, 320, 950, 760]);
  });

  it("errors on non-JSON and on non-object JSON", () => {
    expect(parseMagicPromptCaption("not json").error).toBeTruthy();
    expect(parseMagicPromptCaption("[1, 2]").error).toBeTruthy();
  });
});

describe("verifier mirror", () => {
  it("verifyCaption returns no issues for the canonical fox", () => {
    expect(verifyCaption(FOX)).toEqual([]);
  });

  it("verifyRaw flags ensure_ascii escapes as a warning", () => {
    const raw = '{"compositional_deconstruction": {"background": "caf\\u00e9", "elements": []}}';
    const issues = verifyRaw(raw);
    expect(issues.some((i) => i.code === "ensure_ascii" && i.severity === "warning")).toBe(true);
  });

  it("verifyRaw flags both an obj with wrong-context key and missing-type", () => {
    const objWithText = verifyCaption({
      compositional_deconstruction: { background: "b", elements: [{ type: "obj", bbox: [0, 0, 1, 1], text: "X", desc: "d" }] },
    });
    expect(objWithText.some((i) => i.code === "key_context")).toBe(true);
  });
});

describe("factories + recipe storage", () => {
  it("emptyCaption is a valid skeleton", () => {
    const result = validateCaption(emptyCaption());
    expect(result.ok).toBe(true);
  });

  it("element factories order optional keys correctly", () => {
    const obj = makeObjElement({ bbox: [0, 0, 10, 10], desc: "d" });
    expect(Object.keys(obj)).toEqual(["type", "bbox", "desc"]);
    const text = makeTextElement({ text: "HI", desc: "d" });
    expect(Object.keys(text)).toEqual(["type", "text", "desc"]);
  });

  it("builds a structured-prompt recipe carrying intent, caption and runtime payload", () => {
    const recipe = buildStructuredPromptRecipe({
      intent: "a red fox in the snow",
      caption: FOX,
      magicPromptBackend: "prompt_refine",
      edited: true,
    });
    expect(recipe.version).toBe(1);
    expect(recipe.intent).toBe("a red fox in the snow");
    expect(recipe.magicPromptBackend).toBe("prompt_refine");
    expect(recipe.edited).toBe(true);
    expect(recipe.runtimePrompt).toBe(FOX_JSON);
    expect(runtimePromptFromRecipe(recipe)).toBe(FOX_JSON);
  });

  it("runtimePromptFromRecipe falls back to re-serializing the caption", () => {
    expect(runtimePromptFromRecipe({ caption: FOX })).toBe(FOX_JSON);
    expect(runtimePromptFromRecipe(null)).toBe("");
  });

  it("restore round-trip: a stored caption re-serializes byte-identically (sc-6147)", () => {
    // What ImageStudio persists in advanced.structuredPrompt, then restores from
    // rawAdapterSettings.structuredPrompt on "Use this recipe".
    const recipe = buildStructuredPromptRecipe({ intent: "a red fox in the snow", caption: FOX });
    // The restore path runs the stored caption back through orderCaption (key order
    // in the stored object may be insertion-order, not canonical) before re-serializing.
    const scrambled = {
      compositional_deconstruction: recipe.caption.compositional_deconstruction,
      style_description: recipe.caption.style_description,
      high_level_description: recipe.caption.high_level_description,
    };
    expect(validateCaption(scrambled).ok).toBe(true);
    expect(serializeCaption(orderCaption(scrambled))).toBe(FOX_JSON);
    expect(serializeCaption(orderCaption(scrambled))).toBe(recipe.runtimePrompt);
  });
});
