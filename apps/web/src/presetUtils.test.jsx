import { describe, expect, it } from "vitest";
import {
  applyPresetDefault,
  buildStudioPresetPayload,
  cleanPresetDefaults,
  clearPresetDefault,
  finiteNumberOrUndefined,
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
  it("folds character and variation modes into text_to_image", () => {
    expect(workflowForMode("text_to_image")).toBe("text_to_image");
    expect(workflowForMode("character_image")).toBe("text_to_image");
    expect(workflowForMode("style_variations")).toBe("text_to_image");
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
