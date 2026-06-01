import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// konva's node build pulls in the native `canvas` package (not installed, and not
// usable in jsdom). The empty-state paths under test never mount the <Stage>, so
// stub react-konva to keep konva out of the import graph — mirroring how App.jsx
// lazy-loads the editor to keep konva off the test/initial path.
vi.mock("react-konva", async () => {
  const React = await import("react");
  const passthrough = (name) => ({ children }) => React.createElement("div", { "data-konva": name }, children);
  return { Stage: passthrough("stage"), Layer: passthrough("layer"), Image: () => null, Rect: () => null };
});

import { AppContext } from "../context/AppContext.js";
import {
  ImageEditor,
  cropRatioForKey,
  centeredCropRect,
  upscaleFactorsForEngine,
  buildUpscaleJobBody,
  editedFilename,
  buildSaveProvenance,
  gradePixel,
  applyColorAdjustments,
  isIdentityAdjust,
  IDENTITY_COLOR_ADJUST,
  editCapableModels,
  buildEditJobBody,
  modelIsInpaintCapable,
  maskHasContent,
  detailCapableModels,
  buildDetailJobBody,
} from "./ImageEditor.jsx";

// These tests cover the non-canvas surface of the editor (empty state, the inert
// tool scaffold, and the load affordances). The Konva <Stage> only mounts once a
// working image is present, which needs a real canvas — out of reach for jsdom —
// so canvas behaviour (zoom/pan/fit) is verified in the browser, not here. Simply
// mounting also asserts that importing react-konva/konva doesn't break jsdom.
function baseContext(overrides = {}) {
  return {
    activeProject: null,
    assets: [],
    characters: [],
    setPreviewAsset: vi.fn(),
    token: "",
    requestedGpu: "auto",
    jobs: [],
    importAsset: vi.fn(),
    purgeAsset: vi.fn(),
    registerLeaveGuard: vi.fn(),
    imageModels: [],
    ...overrides,
  };
}

const toolButtons = (container) => [...container.querySelectorAll(".image-editor-tool")];
const barButtons = (container) => [...container.querySelectorAll(".image-editor-bar-actions button")];
const barButton = (container, label) => barButtons(container).find((b) => b.textContent.trim() === label);

describe("ImageEditor scaffold", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageEditor />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("renders the empty state with the tool palette hidden until an image loads", async () => {
    await render(baseContext());

    expect(container.textContent).toContain("Open an image to start editing");
    // No working image → no Konva stage, view controls, or floating tool palette.
    expect(container.querySelector(".image-editor-viewbar")).toBeNull();
    expect(toolButtons(container)).toHaveLength(0);
  });

  it("offers a single always-enabled 'Open' action (source is chosen in the dialog)", async () => {
    await render(baseContext());
    expect(barButtons(container).map((b) => b.textContent.trim())).toEqual(["Open"]);
    expect(barButton(container, "Open").disabled).toBe(false);

    // Still a single, always-enabled Open with a project — there are no separate
    // "Open from project" / "Upload" buttons; the dialog picks the source.
    await render(baseContext({ activeProject: { id: "project_1", name: "My Project" } }));
    expect(barButtons(container).map((b) => b.textContent.trim())).toEqual(["Open"]);
    expect(barButton(container, "Open").disabled).toBe(false);
  });
});

describe("crop geometry", () => {
  it("resolves ratio keys, transposing only non-square ratios on rotate", () => {
    expect(cropRatioForKey("free", false)).toBeNull();
    expect(cropRatioForKey("1:1", false)).toBe(1);
    expect(cropRatioForKey("1:1", true)).toBe(1); // square is unaffected by rotate
    expect(cropRatioForKey("16:9", false)).toBeCloseTo(16 / 9);
    expect(cropRatioForKey("16:9", true)).toBeCloseTo(9 / 16);
    expect(cropRatioForKey("3:4", true)).toBeCloseTo(4 / 3);
  });

  it("centers the largest rect of the ratio that fits in the image", () => {
    // Square in a landscape image → limited by height, centered horizontally.
    expect(centeredCropRect(1000, 500, 1)).toEqual({ x: 250, y: 0, width: 500, height: 500 });
    // 16:9 in a 1000×500 image → limited by height (562.5 > 500), centered.
    const wide = centeredCropRect(1000, 500, 16 / 9);
    expect(wide.height).toBe(500);
    expect(wide.width).toBeCloseTo((500 * 16) / 9);
    expect(wide.x).toBeCloseTo((1000 - (500 * 16) / 9) / 2);
    expect(wide.y).toBe(0);
    // Freeform → centered 80% box.
    expect(centeredCropRect(800, 600, null)).toEqual({ x: 80, y: 60, width: 640, height: 480 });
  });
});

describe("upscale job", () => {
  it("constrains factors per engine", () => {
    expect(upscaleFactorsForEngine("real-esrgan")).toEqual([2, 4]);
    expect(upscaleFactorsForEngine("aura-sr")).toEqual([4]);
    expect(upscaleFactorsForEngine("unknown")).toEqual([2, 4]);
  });

  it("builds the image_upscale job body the worker expects (sourceAssetId/factor/engine)", () => {
    const body = buildUpscaleJobBody({
      project: { id: "project_1", name: "My Project" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      factor: 4,
      engine: "real-esrgan",
      displayName: "shot.png",
    });
    expect(body).toEqual({
      type: "image_upscale",
      projectId: "project_1",
      projectName: "My Project",
      requestedGpu: "auto",
      payload: {
        projectId: "project_1",
        sourceAssetId: "asset_scratch",
        factor: 4,
        engine: "real-esrgan",
        displayName: "shot.png",
      },
    });
  });
});

describe("detail job", () => {
  it("filters to image_detail-capable models", () => {
    const models = [
      { id: "realvisxl", capabilities: ["text_to_image", "edit_image", "image_detail"] },
      { id: "flux", capabilities: ["text_to_image"] },
      { id: "sdxl", capabilities: ["image_detail"] },
    ];
    expect(detailCapableModels(models).map((m) => m.id)).toEqual(["realvisxl", "sdxl"]);
    expect(detailCapableModels([])).toEqual([]);
    expect(detailCapableModels(undefined)).toEqual([]);
  });

  it("builds the image_detail job body the worker expects (model + advanced.strength/cnScale)", () => {
    const body = buildDetailJobBody({
      project: { id: "project_1", name: "My Project" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      model: "realvisxl",
      strength: 0.55,
      cnScale: 0.7,
      displayName: "shot.png",
    });
    expect(body).toEqual({
      type: "image_detail",
      projectId: "project_1",
      projectName: "My Project",
      requestedGpu: "auto",
      payload: {
        projectId: "project_1",
        sourceAssetId: "asset_scratch",
        model: "realvisxl",
        displayName: "shot.png",
        advanced: { strength: 0.55, cnScale: 0.7 },
      },
    });
  });
});

describe("save / export", () => {
  it("derives an -edited.png export filename, always PNG", () => {
    expect(editedFilename({ name: "shot.jpg" })).toBe("shot-edited.png");
    expect(editedFilename({ name: "portrait.png" })).toBe("portrait-edited.png");
    expect(editedFilename({ name: "no-extension" })).toBe("no-extension-edited.png");
    // Falls back to a default when the source has no usable name.
    expect(editedFilename(null)).toBe("image-edited.png");
    expect(editedFilename({})).toBe("image-edited.png");
  });

  it("builds provenance that links a saved edit to its asset source + edit chain", () => {
    const provenance = buildSaveProvenance({
      source: { kind: "asset", assetId: "asset_src", name: "shot.png" },
      edits: [{ op: "crop", width: 100, height: 100 }, { op: "upscale", engine: "real-esrgan", factor: 4 }],
      width: 400,
      height: 400,
    });
    expect(provenance).toEqual({
      editor: "image_editor",
      source: { kind: "asset", assetId: "asset_src", name: "shot.png" },
      edits: [{ op: "crop", width: 100, height: 100 }, { op: "upscale", engine: "real-esrgan", factor: 4 }],
      width: 400,
      height: 400,
    });
  });

  it("records uploads as a source kind with no asset id (nothing to link)", () => {
    const provenance = buildSaveProvenance({
      source: { kind: "upload", name: "drag.png" },
      edits: [],
      width: 10,
      height: 20,
    });
    expect(provenance.source).toEqual({ kind: "upload", name: "drag.png" });
    expect(provenance.edits).toEqual([]);
  });
});

describe("color grade", () => {
  it("treats an all-zero adjustment as the identity", () => {
    expect(isIdentityAdjust(IDENTITY_COLOR_ADJUST)).toBe(true);
    expect(isIdentityAdjust(null)).toBe(true);
    expect(isIdentityAdjust({ brightness: 0.1 })).toBe(false);
    expect(isIdentityAdjust({ contrast: 0, saturation: 0, temperature: 0.01 })).toBe(false);
  });

  it("leaves a pixel unchanged at the identity", () => {
    expect(gradePixel([100, 150, 200], IDENTITY_COLOR_ADJUST)).toEqual([100, 150, 200]);
  });

  it("brightness pushes toward white / black and clamps to [0,255]", () => {
    expect(gradePixel([100, 100, 100], { brightness: 1 })).toEqual([255, 255, 255]);
    expect(gradePixel([100, 100, 100], { brightness: -1 })).toEqual([0, 0, 0]);
  });

  it("contrast keeps mid-gray fixed and spreads extremes", () => {
    // 128 is the pivot — unchanged by any contrast.
    expect(gradePixel([128, 128, 128], { contrast: 0.5 })).toEqual([128, 128, 128]);
    // A darker pixel gets darker as contrast increases.
    expect(gradePixel([100, 100, 100], { contrast: 0.5 })[0]).toBeLessThan(100);
  });

  it("saturation of -1 fully desaturates to a single luma value", () => {
    const [r, g, b] = gradePixel([200, 50, 50], { saturation: -1 });
    expect(r).toBe(g);
    expect(g).toBe(b);
  });

  it("temperature warms (R up, B down) and cools (R down, B up)", () => {
    const warm = gradePixel([120, 120, 120], { temperature: 1 });
    expect(warm[0]).toBeGreaterThan(120);
    expect(warm[2]).toBeLessThan(120);
    const cool = gradePixel([120, 120, 120], { temperature: -1 });
    expect(cool[0]).toBeLessThan(120);
    expect(cool[2]).toBeGreaterThan(120);
  });

  it("applyColorAdjustments edits RGB in place, leaves alpha untouched, and no-ops at identity", () => {
    const data = new Uint8ClampedArray([100, 100, 100, 42]);
    applyColorAdjustments(data, { brightness: 1 });
    expect([data[0], data[1], data[2]]).toEqual([255, 255, 255]);
    expect(data[3]).toBe(42); // alpha preserved

    const untouched = new Uint8ClampedArray([10, 20, 30, 40]);
    applyColorAdjustments(untouched, IDENTITY_COLOR_ADJUST);
    expect([...untouched]).toEqual([10, 20, 30, 40]);
  });
});

describe("AI prompt edit", () => {
  it("filters models to those tagged edit_image / image_edit", () => {
    const models = [
      { id: "z_image_turbo", capabilities: ["text_to_image"] },
      { id: "qwen_image_edit_2511", capabilities: ["text_to_image", "edit_image"] },
      { id: "sdxl", capabilities: ["image_edit"] },
      { id: "no_caps" },
    ];
    expect(editCapableModels(models).map((m) => m.id)).toEqual(["qwen_image_edit_2511", "sdxl"]);
    expect(editCapableModels([])).toEqual([]);
    expect(editCapableModels(undefined)).toEqual([]);
  });

  it("builds the edit_image job body the /api/v1/image/jobs endpoint expects", () => {
    const body = buildEditJobBody({
      project: { id: "project_1", name: "My Project" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      model: "qwen_image_edit_2511",
      prompt: "make it night",
      seed: "42",
      width: 768,
      height: 1024,
    });
    expect(body).toEqual({
      projectId: "project_1",
      projectName: "My Project",
      requestedGpu: "auto",
      mode: "edit_image",
      sourceAssetId: "asset_scratch",
      model: "qwen_image_edit_2511",
      prompt: "make it night",
      negativePrompt: "",
      width: 768,
      height: 1024,
      seed: 42,
      count: 1,
      advanced: {},
    });
  });

  it("treats an empty/blank seed as null (random)", () => {
    const base = {
      project: { id: "p", name: "P" },
      requestedGpu: "auto",
      sourceAssetId: "a",
      model: "m",
      prompt: "x",
      width: 10,
      height: 10,
    };
    expect(buildEditJobBody({ ...base, seed: "" }).seed).toBeNull();
    expect(buildEditJobBody({ ...base, seed: null }).seed).toBeNull();
    expect(buildEditJobBody({ ...base, seed: 7 }).seed).toBe(7);
  });

  it("includes maskAssetId only when an inpaint mask is supplied", () => {
    const base = {
      project: { id: "p", name: "P" },
      requestedGpu: "auto",
      sourceAssetId: "a",
      model: "sdxl",
      prompt: "x",
      width: 10,
      height: 10,
    };
    expect("maskAssetId" in buildEditJobBody(base)).toBe(false);
    expect("maskAssetId" in buildEditJobBody({ ...base, maskAssetId: undefined })).toBe(false);
    expect(buildEditJobBody({ ...base, maskAssetId: "asset_mask" }).maskAssetId).toBe("asset_mask");
  });
});

describe("inpaint mask", () => {
  it("flags only models tagged image_inpaint as mask-capable", () => {
    expect(modelIsInpaintCapable({ capabilities: ["edit_image", "image_inpaint"] })).toBe(true);
    expect(modelIsInpaintCapable({ capabilities: ["edit_image"] })).toBe(false);
    expect(modelIsInpaintCapable(null)).toBe(false);
    expect(modelIsInpaintCapable({})).toBe(false);
  });

  it("treats a mask as having content only with a non-erase stroke", () => {
    expect(maskHasContent([])).toBe(false);
    expect(maskHasContent(null)).toBe(false);
    // A single tap paints a dot — that counts as a mask region.
    expect(maskHasContent([{ points: [10, 10], size: 40, erase: false }])).toBe(true);
    // Erase-only strokes don't make a mask.
    expect(maskHasContent([{ points: [0, 0, 5, 5], size: 40, erase: true }])).toBe(false);
    expect(maskHasContent([{ points: [0, 0, 5, 5], size: 40, erase: false }])).toBe(true);
  });
});
