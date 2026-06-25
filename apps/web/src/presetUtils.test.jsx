import { describe, expect, it } from "vitest";
import {
  applyPresetDefault,
  buildStudioPresetPayload,
  cleanPresetDefaults,
  clearPresetDefault,
  editModelForAsset,
  finiteNumberOrUndefined,
  loraWeight,
  presetMatchesModel,
  presetNameTaken,
  slugifyPresetId,
  workflowForMode,
} from "./presetUtils.js";

const ltx = { id: "ltx_2_3", family: "ltx-video" };
const ltxEros = { id: "ltx_2_3_eros", family: "ltx-video" };
const sdxl = { id: "sdxl", family: "sdxl" };
const catalog = [ltx, ltxEros, sdxl];

describe("presetMatchesModel", () => {
  it("matches when the preset pins no model", () => {
    expect(presetMatchesModel({ id: "p" }, ltxEros, catalog)).toBe(true);
  });

  it("matches when the selected model has no id", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, {}, catalog)).toBe(true);
  });

  it("matches on exact model id", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltx, catalog)).toBe(true);
  });

  it("matches a sibling model in the same family (ltx_2_3 preset under ltx_2_3_eros)", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltxEros, catalog)).toBe(true);
  });

  it("does not match across families", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, sdxl, catalog)).toBe(false);
  });

  it("stays strict (no family fallback) when the catalog is unavailable", () => {
    expect(presetMatchesModel({ model: "ltx_2_3" }, ltxEros)).toBe(false);
  });
});

describe("workflowForMode", () => {
  it("folds the character mode into text_to_image", () => {
    expect(workflowForMode("text_to_image")).toBe("text_to_image");
    expect(workflowForMode("character_image")).toBe("text_to_image");
  });

  it("maps the single-mode workflows to themselves", () => {
    expect(workflowForMode("edit_image")).toBe("edit_image");
    expect(workflowForMode("image_to_video")).toBe("image_to_video");
    expect(workflowForMode("text_to_video")).toBe("text_to_video");
    expect(workflowForMode("first_last_frame")).toBe("first_last_frame");
  });

  it("returns an unknown mode unchanged", () => {
    expect(workflowForMode("something_else")).toBe("something_else");
  });
});

describe("slugifyPresetId", () => {
  it("lowercases and replaces runs of invalid characters with a single underscore", () => {
    expect(slugifyPresetId("Atrium Portraits!")).toBe("atrium_portraits");
  });

  it("trims leading and trailing separators so the id starts alphanumeric", () => {
    expect(slugifyPresetId("  --Neon Noir--  ")).toBe("neon_noir");
    expect(slugifyPresetId("neon-noir")).toBe("neon-noir");
  });

  it("returns empty string for non-sluggable input", () => {
    expect(slugifyPresetId("日本語")).toBe("");
    expect(slugifyPresetId("")).toBe("");
  });
});

describe("presetNameTaken", () => {
  const presets = [
    { id: "atrium_portraits", name: "Atrium Portraits" },
    { id: "neon-noir", name: "Neon Noir" },
  ];

  it("matches an existing name case-insensitively", () => {
    expect(presetNameTaken("atrium portraits", presets)).toBe(true);
  });

  it("matches when a different name slugs to an existing id", () => {
    expect(presetNameTaken("Atrium  Portraits!", presets)).toBe(true);
  });

  it("is false for a fresh name and for blank input", () => {
    expect(presetNameTaken("Sunset Pier", presets)).toBe(false);
    expect(presetNameTaken("   ", presets)).toBe(false);
  });
});

describe("finiteNumberOrUndefined", () => {
  it("coerces numeric strings and numbers", () => {
    expect(finiteNumberOrUndefined("30")).toBe(30);
    expect(finiteNumberOrUndefined("4.5")).toBe(4.5);
    expect(finiteNumberOrUndefined(0)).toBe(0);
  });

  it("returns undefined for blank or non-numeric input", () => {
    expect(finiteNumberOrUndefined("")).toBeUndefined();
    expect(finiteNumberOrUndefined(null)).toBeUndefined();
    expect(finiteNumberOrUndefined("abc")).toBeUndefined();
    expect(finiteNumberOrUndefined(undefined)).toBeUndefined();
  });
});

describe("loraWeight", () => {
  it("defaults a generic LoRA to 0.8", () => {
    expect(loraWeight({ id: "sdxl_style", family: "sdxl" })).toBe(0.8);
    expect(loraWeight(null)).toBe(0.8);
  });

  it("defaults a krea-2-family LoRA higher (1.5) for the distilled-Turbo attenuation (sc-7932)", () => {
    // The family token is normalized (krea_2 -> krea-2), and the bump applies via any of the
    // family-bearing shapes extractFamilies() reads.
    expect(loraWeight({ id: "k", family: "krea_2" })).toBe(1.5);
    expect(loraWeight({ id: "k", compatibility: { families: ["krea_2"] } })).toBe(1.5);
  });

  it("lets an explicit weight win over the krea-2 family default", () => {
    expect(loraWeight({ id: "k", family: "krea_2", defaultWeight: 1.0 })).toBe(1.0);
    expect(loraWeight({ id: "k", family: "krea_2", weight: 0.7 })).toBe(0.7);
    expect(loraWeight({ id: "k", family: "krea_2" }, { weight: 2.0 })).toBe(2.0);
  });

  it("falls back to the family default when an explicit value is non-finite", () => {
    expect(loraWeight({ id: "k", family: "krea_2", defaultWeight: "nope" })).toBe(1.5);
    expect(loraWeight({ id: "g", family: "sdxl", defaultWeight: "nope" })).toBe(0.8);
  });
});

describe("cleanPresetDefaults", () => {
  it("drops null/undefined/empty-string but keeps 0 and false", () => {
    expect(
      cleanPresetDefaults({ steps: 0, guidanceScale: "", sampler: null, upscaleEnabled: false, resolution: "1024x1024", motion: undefined }),
    ).toEqual({ steps: 0, upscaleEnabled: false, resolution: "1024x1024" });
  });
});

describe("buildStudioPresetPayload", () => {
  it("snapshots a character-mode image config without the seed", () => {
    const payload = buildStudioPresetPayload({
      name: "Atrium Portraits!",
      scope: "project",
      mode: "character_image",
      model: "instantid_sdxl",
      loras: [{ id: "kelsie", weight: 0.75 }],
      defaults: {
        prompt: "a portrait in the atrium",
        negativePrompt: "",
        resolution: "1024x1024",
        count: 4,
        guidanceScale: 5,
        steps: "",
        sampler: "default",
      },
    });
    expect(payload).toMatchObject({
      id: "atrium_portraits",
      name: "Atrium Portraits!",
      scope: "project",
      workflow: "text_to_image",
      model: "instantid_sdxl",
      loras: [{ id: "kelsie", weight: 0.75 }],
    });
    // modes carry every entry point the picker should surface the preset under.
    expect(payload.modes).toContain("character_image");
    // empty-string knobs are omitted; the literal prompt is preserved.
    expect(payload.defaults).toEqual({
      prompt: "a portrait in the atrium",
      resolution: "1024x1024",
      count: 4,
      guidanceScale: 5,
      sampler: "default",
    });
    expect(payload.defaults).not.toHaveProperty("seed");
  });

  it("coerces a non-finite lora weight to the lora's fallback", () => {
    const payload = buildStudioPresetPayload({
      name: "x",
      mode: "text_to_video",
      model: "ltx_2_3",
      loras: [{ id: "wobble", weight: "not-a-number", defaultWeight: 0.6 }],
      defaults: {},
    });
    expect(payload.workflow).toBe("text_to_video");
    expect(payload.loras).toEqual([{ id: "wobble", weight: 0.6 }]);
  });
});

describe("applyPresetDefault + clearPresetDefault round-trip", () => {
  // Mirrors how the studios drive a state setter: the setter receives either a
  // value or an updater, and the snapshots ref is what the studio keeps in useRef.
  function makeSetter(initial) {
    let value = initial;
    return {
      setter: (updater) => {
        value = typeof updater === "function" ? updater(value) : updater;
      },
      get: () => value,
    };
  }

  it("applies a preset value then restores the user's prior value on clear", () => {
    const snapshots = { current: {} };
    const box = makeSetter("user prompt");
    applyPresetDefault(snapshots, "prompt", box.setter, "preset prompt");
    expect(box.get()).toBe("preset prompt");
    clearPresetDefault(box.setter, snapshots, "prompt");
    expect(box.get()).toBe("user prompt");
  });

  it("leaves a user override in place when they changed the value after applying", () => {
    const snapshots = { current: {} };
    const box = makeSetter(4);
    applyPresetDefault(snapshots, "count", box.setter, 8);
    box.setter(2); // user manually edits after the preset applied
    clearPresetDefault(box.setter, snapshots, "count");
    expect(box.get()).toBe(2);
  });
});

// sc-4162 regression: editModelForAsset previously lived in App.jsx and called
// modelLoraFamilies without importing it — a ReferenceError on the family-sibling
// path (any asset whose generating model can't edit).
describe("editModelForAsset", () => {
  const t2iOnly = { id: "z_image_turbo", family: "z-image", capabilities: ["text_to_image"] };
  const editSibling = { id: "z_image_edit", family: "z-image", capabilities: ["edit_image"] };
  const editSelf = { id: "qwen_image_edit", family: "qwen-image", capabilities: ["image_edit"] };
  const models = [t2iOnly, editSibling, editSelf];

  it("prefers the generating model when it is edit-capable", () => {
    expect(editModelForAsset({ recipe: { model: "qwen_image_edit" } }, models)).toBe("qwen_image_edit");
  });

  it("falls back to a same-family edit-capable sibling when the source model cannot edit", () => {
    expect(editModelForAsset({ recipe: { model: "z_image_turbo" } }, models)).toBe("z_image_edit");
  });

  it("matches a family sibling when the generating model is not in the catalog", () => {
    expect(editModelForAsset({ recipe: { model: "z-image" } }, models)).toBe("z_image_edit");
  });

  it("returns null when no family-matched edit model exists", () => {
    expect(editModelForAsset({ recipe: { model: "z_image_turbo" } }, [t2iOnly, editSelf])).toBe(null);
  });

  it("returns null for assets without a recipe model", () => {
    expect(editModelForAsset({ recipe: {} }, models)).toBe(null);
    expect(editModelForAsset(null, models)).toBe(null);
  });
});
