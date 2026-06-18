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
  upscaleEngineHasSoftness,
  buildUpscaleJobBody,
  editedFilename,
  buildSaveProvenance,
  gradePixel,
  applyColorAdjustments,
  isIdentityAdjust,
  IDENTITY_COLOR_ADJUST,
  editCapableModels,
  buildEditJobBody,
  editOutputDims,
  editOutputAspectRatio,
  modelIsInpaintCapable,
  maskHasContent,
  detailCapableModels,
  buildDetailJobBody,
  rectToBbox,
  bboxToRect,
  isValidHexColor,
  boxPaletteIsValid,
  documentPalette,
  documentPaletteIsValid,
  boxIsValid,
  BOX_TYPES,
  MAX_BOX_PALETTE,
  MAX_DOCUMENT_PALETTE,
  BOX_PALETTE,
  MIN_BOX_PX,
  rectFromPoints,
  clampRectToCanvas,
  makeBox,
  boxFillStyle,
  addPaletteColor,
  removePaletteColor,
  boxMetadataGaps,
  blankCanvasDims,
  BLANK_CANVAS_SIZES,
  paintBoxesOnContext,
  colorName,
  composeColorPrompt,
  boxesToIdeogramElements,
  HISTORY_LIMIT,
  emptyHistory,
  historyCheckpoint,
  historyUndo,
  historyRedo,
  canUndo,
  canRedo,
  buildSegmentJobBody,
  rectToSegmentBox,
  tintMaskRgbaInPlace,
  MASK_PREVIEW_RGBA,
} from "./ImageEditor.jsx";
import { verifyCaption, serializeCaption, ELEMENT_KEY_ORDER_OBJ } from "../ideogramCaption.js";

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

  it("offers always-enabled 'Open' + 'New layout' actions before an image loads", async () => {
    await render(baseContext());
    // Open (dialog picks the source) + New layout (blank canvas, sc-6092). No
    // separate "Open from project" / "Upload" buttons.
    expect(barButtons(container).map((b) => b.textContent.trim())).toEqual(["Open", "New layout"]);
    expect(barButton(container, "Open").disabled).toBe(false);
    expect(barButton(container, "New layout").disabled).toBe(false);

    // Same with a project active.
    await render(baseContext({ activeProject: { id: "project_1", name: "My Project" } }));
    expect(barButtons(container).map((b) => b.textContent.trim())).toEqual(["Open", "New layout"]);
    expect(barButton(container, "New layout").disabled).toBe(false);
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

describe("box layout (Workstream A)", () => {
  it("normalizes a rect to a 0–1000 bbox in exact y_min,x_min,y_max,x_max order", () => {
    // Asymmetric rect in a 1000×1000 image makes the component order unambiguous.
    expect(rectToBbox({ x: 100, y: 200, width: 300, height: 400 }, 1000, 1000)).toEqual([200, 100, 600, 400]);
    // Non-square image: x scales by width, y by height, independently.
    expect(rectToBbox({ x: 250, y: 250, width: 250, height: 250 }, 500, 1000)).toEqual([250, 500, 500, 1000]);
  });

  it("clamps out-of-canvas rects and a full-canvas rect to the grid edges", () => {
    expect(rectToBbox({ x: 0, y: 0, width: 800, height: 600 }, 800, 600)).toEqual([0, 0, 1000, 1000]);
    // Overflow past the right/bottom and a negative origin clamp to [0,1000].
    expect(rectToBbox({ x: -50, y: -50, width: 2000, height: 2000 }, 800, 600)).toEqual([0, 0, 1000, 1000]);
  });

  it("rounds sub-pixel coordinates and survives a flipped (negative-size) rect", () => {
    // 1px in a 2000px canvas → 0.5 → rounds to 1 on the 0–1000 grid.
    expect(rectToBbox({ x: 1, y: 1, width: 0, height: 0 }, 2000, 2000)).toEqual([1, 1, 1, 1]);
    // A rect dragged up-left has negative width/height; min/max keep min ≤ max.
    expect(rectToBbox({ x: 400, y: 600, width: -300, height: -400 }, 1000, 1000)).toEqual([200, 100, 600, 400]);
  });

  it("round-trips a bbox back to image-pixel coordinates", () => {
    const rect = bboxToRect([200, 100, 600, 400], 1000, 1000);
    expect(rect).toEqual({ x: 100, y: 200, width: 300, height: 400 });
    // Quantized round-trip: rect → bbox → rect recovers the rect to grid resolution.
    const back = bboxToRect(rectToBbox({ x: 100, y: 200, width: 300, height: 400 }, 1000, 1000), 1000, 1000);
    expect(back).toEqual({ x: 100, y: 200, width: 300, height: 400 });
  });

  it("validates hex colors as strictly uppercase #RRGGBB", () => {
    expect(isValidHexColor("#FF0000")).toBe(true);
    expect(isValidHexColor("#ff0000")).toBe(false); // lowercase rejected (S3 is case-sensitive)
    expect(isValidHexColor("#F00")).toBe(false); // shorthand rejected
    expect(isValidHexColor("FF0000")).toBe(false); // missing '#'
    expect(isValidHexColor(null)).toBe(false);
  });

  it("enforces ≤5 valid colors per element and ≤16 across the document", () => {
    expect(MAX_BOX_PALETTE).toBe(5);
    expect(MAX_DOCUMENT_PALETTE).toBe(16);
    expect(boxPaletteIsValid(undefined)).toBe(true); // optional
    expect(boxPaletteIsValid(["#FF0000", "#00FF00"])).toBe(true);
    expect(boxPaletteIsValid(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"])).toBe(false);
    expect(boxPaletteIsValid(["#ff0000"])).toBe(false); // lowercase entry invalidates the palette

    // Document palette is the order-preserving, de-duplicated union of per-box palettes.
    const boxes = [
      { colorPalette: ["#FF0000", "#00FF00"] },
      { colorPalette: ["#00FF00", "#0000FF"] }, // #00FF00 already seen
    ];
    expect(documentPalette(boxes)).toEqual(["#FF0000", "#00FF00", "#0000FF"]);
    expect(documentPaletteIsValid(boxes)).toBe(true);

    // 17 distinct colors across boxes exceeds the overall cap of 16.
    const many = Array.from({ length: 17 }, (_, i) => ({
      colorPalette: [`#${i.toString(16).toUpperCase().padStart(6, "0")}`],
    }));
    expect(documentPalette(many)).toHaveLength(17);
    expect(documentPaletteIsValid(many)).toBe(false);
  });

  it("validates a box: positive geometry, known type, required desc, text iff type==='text'", () => {
    expect(BOX_TYPES).toEqual(["obj", "text"]);
    const obj = { rect: { x: 0, y: 0, width: 10, height: 10 }, type: "obj", desc: "a cat" };
    expect(boxIsValid(obj)).toBe(true);

    // Zero/negative geometry is invalid.
    expect(boxIsValid({ ...obj, rect: { x: 0, y: 0, width: 0, height: 10 } })).toBe(false);
    // Unknown type is invalid.
    expect(boxIsValid({ ...obj, type: "scribble" })).toBe(false);
    // Missing/blank desc is invalid.
    expect(boxIsValid({ ...obj, desc: "   " })).toBe(false);

    // A text box additionally requires a non-empty literal string.
    const text = { rect: { x: 0, y: 0, width: 10, height: 10 }, type: "text", desc: "a sign" };
    expect(boxIsValid(text)).toBe(false);
    expect(boxIsValid({ ...text, text: "OPEN" })).toBe(true);
  });
});

describe("box drawing tool (Workstream A, sc-6090)", () => {
  it("ships a palette of distinct, valid uppercase #RRGGBB colors", () => {
    expect(BOX_PALETTE.length).toBeGreaterThanOrEqual(6);
    expect(BOX_PALETTE.every((entry) => isValidHexColor(entry.value))).toBe(true);
    expect(BOX_PALETTE.every((entry) => typeof entry.name === "string" && entry.name)).toBe(true);
    const values = BOX_PALETTE.map((entry) => entry.value);
    expect(new Set(values).size).toBe(values.length); // all distinct
  });

  it("normalizes a drag into a positive-size rect regardless of direction", () => {
    expect(rectFromPoints({ x: 10, y: 20 }, { x: 40, y: 60 })).toEqual({ x: 10, y: 20, width: 30, height: 40 });
    // Dragged up-left → same rect (min origin, abs size).
    expect(rectFromPoints({ x: 40, y: 60 }, { x: 10, y: 20 })).toEqual({ x: 10, y: 20, width: 30, height: 40 });
  });

  it("clamps a box rect inside the canvas with a minimum size", () => {
    // Fully inside → unchanged.
    expect(clampRectToCanvas({ x: 10, y: 10, width: 100, height: 50 }, 800, 600)).toEqual({
      x: 10,
      y: 10,
      width: 100,
      height: 50,
    });
    // Overflowing the right/bottom edge → pushed back inside.
    expect(clampRectToCanvas({ x: 700, y: 500, width: 400, height: 400 }, 800, 600)).toEqual({
      x: 400,
      y: 200,
      width: 400,
      height: 400,
    });
    // Negative origin → clamped to 0.
    expect(clampRectToCanvas({ x: -20, y: -10, width: 100, height: 50 }, 800, 600)).toEqual({
      x: 0,
      y: 0,
      width: 100,
      height: 50,
    });
    // Sub-minimum size → grown to MIN_BOX_PX.
    const tiny = clampRectToCanvas({ x: 5, y: 5, width: 2, height: 1 }, 800, 600);
    expect(tiny.width).toBe(MIN_BOX_PX);
    expect(tiny.height).toBe(MIN_BOX_PX);
  });

  it("builds a box record with the sc-6089 shape and safe metadata defaults", () => {
    const box = makeBox("box_1", { x: 1, y: 2, width: 3, height: 4 }, "#FF0000");
    expect(box).toEqual({
      id: "box_1",
      rect: { x: 1, y: 2, width: 3, height: 4 },
      color: "#FF0000",
      type: "obj",
      desc: "",
      text: "",
      colorPalette: [],
    });
  });

  it("derives a translucent rgba fill from a hex color, with a neutral fallback", () => {
    expect(boxFillStyle("#FF0000", 0.18)).toBe("rgba(255,0,0,0.18)");
    expect(boxFillStyle("#2962FF", 0.18)).toBe("rgba(41,98,255,0.18)");
    // Invalid color → neutral grey so the overlay still renders.
    expect(boxFillStyle("nope", 0.18)).toBe("rgba(127,127,127,0.18)");
  });
});

describe("per-box metadata (Workstream A, sc-6091)", () => {
  it("appends palette colors, uppercasing and ignoring dups / invalid / over-cap", () => {
    expect(addPaletteColor([], "#ff0000")).toEqual(["#FF0000"]); // uppercased
    expect(addPaletteColor(["#FF0000"], "#FF0000")).toEqual(["#FF0000"]); // dup → unchanged
    expect(addPaletteColor(undefined, "#00FF00")).toEqual(["#00FF00"]); // absent palette ok
    expect(addPaletteColor(["#FF0000"], "nope")).toEqual(["#FF0000"]); // invalid → unchanged
    // At the ≤5 cap, a 6th is rejected (returns the same array reference).
    const full = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"];
    expect(addPaletteColor(full, "#00FFFF")).toBe(full);
  });

  it("removes a palette color", () => {
    expect(removePaletteColor(["#FF0000", "#00FF00"], "#FF0000")).toEqual(["#00FF00"]);
    expect(removePaletteColor(undefined, "#FF0000")).toEqual([]);
  });

  it("reports the metadata gaps that block a valid Ideogram element", () => {
    // An obj box needs only a description.
    expect(boxMetadataGaps({ type: "obj", desc: "a cat" })).toEqual([]);
    expect(boxMetadataGaps({ type: "obj", desc: "" })).toContain("a description");
    // A text box also needs the literal text.
    expect(boxMetadataGaps({ type: "text", desc: "a sign", text: "" })).toContain("the literal text");
    expect(boxMetadataGaps({ type: "text", desc: "a sign", text: "OPEN" })).toEqual([]);
    // An over-cap / invalid palette is flagged.
    const tooMany = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"];
    expect(boxMetadataGaps({ type: "obj", desc: "x", colorPalette: tooMany })).toContain("a valid color palette (≤5)");
  });
});

describe("blank-canvas new layout (Workstream A, sc-6092)", () => {
  it("derives W×H from an aspect + long-side, as multiples of 16 in [256,2048]", () => {
    expect(blankCanvasDims("1:1", 1024)).toEqual({ width: 1024, height: 1024 });
    // Landscape: long side is width; 16:9 of 1024 = 576.
    expect(blankCanvasDims("16:9", 1024)).toEqual({ width: 1024, height: 576 });
    // Portrait: long side is height.
    expect(blankCanvasDims("9:16", 1024)).toEqual({ width: 576, height: 1024 });
  });

  it("snaps to a multiple of 16 and clamps out-of-range sizes", () => {
    const dims = blankCanvasDims("3:2", 2048); // 2048 / 1.5 = 1365.33 → snaps to 1360
    expect(dims.width).toBe(2048);
    expect(dims.height % 16).toBe(0);
    expect(dims.height).toBe(1360);
    // Below the floor clamps up to 256; above the ceiling clamps to 2048.
    expect(blankCanvasDims("1:1", 100)).toEqual({ width: 256, height: 256 });
    expect(blankCanvasDims("1:1", 4000)).toEqual({ width: 2048, height: 2048 });
  });

  it("every preset size is itself a valid in-range multiple of 16", () => {
    expect(BLANK_CANVAS_SIZES.every((size) => size % 16 === 0 && size >= 256 && size <= 2048)).toBe(true);
  });
});

describe("bake → pass-through edit (Workstream A, sc-6093)", () => {
  it("paints each box as a solid colored rect onto the context, in order", () => {
    // Fake 2D context recording fillStyle at each fillRect call (jsdom has no canvas).
    const calls = [];
    const ctx = {
      _fill: null,
      set fillStyle(v) {
        this._fill = v;
      },
      get fillStyle() {
        return this._fill;
      },
      fillRect(x, y, w, h) {
        calls.push({ color: this._fill, x, y, w, h });
      },
    };
    const boxes = [
      { color: "#FF0000", rect: { x: 10, y: 20, width: 30, height: 40 } },
      { color: "#2962FF", rect: { x: 5, y: 6, width: 7, height: 8 } },
    ];
    paintBoxesOnContext(ctx, boxes);
    expect(calls).toEqual([
      { color: "#FF0000", x: 10, y: 20, w: 30, h: 40 },
      { color: "#2962FF", x: 5, y: 6, w: 7, h: 8 },
    ]);
  });

  it("no-ops on an empty/absent box list", () => {
    const calls = [];
    const ctx = { fillStyle: null, fillRect: () => calls.push(1) };
    paintBoxesOnContext(ctx, []);
    paintBoxesOnContext(ctx, undefined);
    expect(calls).toHaveLength(0);
  });
});

describe("auto color-prompt (Workstream A, sc-6094)", () => {
  it("maps palette colors to friendly names and falls back to the hex", () => {
    expect(colorName("#FF0000")).toBe("red");
    expect(colorName("#2962FF")).toBe("blue");
    expect(colorName("#123456")).toBe("#123456"); // custom → hex fallback
  });

  it("composes an editable color-keyed prompt, one clause per described box", () => {
    const boxes = [
      { color: "#FF0000", type: "obj", desc: "a sports car" },
      { color: "#2962FF", type: "text", text: "OPEN", desc: "neon sign" },
    ];
    expect(composeColorPrompt(boxes)).toBe(
      'Replace the red region with a sports car. Place the text "OPEN" in the blue region (neon sign).',
    );
  });

  it("skips boxes missing the needed text and returns '' when nothing is describable", () => {
    expect(composeColorPrompt([])).toBe("");
    expect(composeColorPrompt([{ color: "#FF0000", type: "obj", desc: "  " }])).toBe("");
    expect(composeColorPrompt([{ color: "#2962FF", type: "text", text: "", desc: "x" }])).toBe("");
    // A described obj among empty ones still composes.
    expect(
      composeColorPrompt([
        { color: "#FF0000", type: "obj", desc: "" },
        { color: "#00C853", type: "obj", desc: "a tree" },
      ]),
    ).toBe("Replace the green region with a tree.");
  });
});

describe("boxes → Ideogram elements[] adapter (Workstream A, sc-6095)", () => {
  const boxes = [
    {
      id: "box_1",
      type: "obj",
      desc: "a sports car",
      rect: { x: 100, y: 200, width: 300, height: 400 },
      colorPalette: ["#FF0000", "#000000"],
    },
    {
      id: "box_2",
      type: "text",
      text: "OPEN",
      desc: "neon sign",
      rect: { x: 0, y: 0, width: 500, height: 250 },
      colorPalette: [],
    },
  ];

  it("emits one element per box with bbox + S3 keys in canonical order", () => {
    const els = boxesToIdeogramElements(boxes, 1000, 1000);
    expect(els).toHaveLength(2);
    // obj: type, bbox, desc, color_palette — exact key order.
    expect(Object.keys(els[0])).toEqual([...ELEMENT_KEY_ORDER_OBJ]);
    expect(els[0]).toMatchObject({ type: "obj", bbox: [200, 100, 600, 400], desc: "a sports car" });
    expect(els[0].color_palette).toEqual(["#FF0000", "#000000"]);
    // text: type, bbox, text, desc, color_palette — and an empty palette is omitted.
    expect(Object.keys(els[1])).toEqual(["type", "bbox", "text", "desc"]);
    expect(els[1]).toMatchObject({ type: "text", bbox: [0, 0, 250, 500], text: "OPEN", desc: "neon sign" });
    expect("color_palette" in els[1]).toBe(false);
  });

  it("produces elements that pass the epic-4725 S3 verifier and serialize in order", () => {
    const caption = {
      compositional_deconstruction: { background: "a city street", elements: boxesToIdeogramElements(boxes, 1000, 1000) },
    };
    // No verifier ERRORS (key order is a warning at most; here it should be clean).
    expect(verifyCaption(caption).filter((i) => i.severity === "error")).toEqual([]);
    // Canonical serialization keeps the obj/text key order.
    const json = serializeCaption(caption);
    expect(json).toContain('{"type": "obj", "bbox": [200, 100, 600, 400], "desc": "a sports car"');
    expect(json).toContain('{"type": "text", "bbox": [0, 0, 250, 500], "text": "OPEN", "desc": "neon sign"}');
  });

  it("normalizes lowercase palette hex to uppercase and drops invalid entries", () => {
    const els = boxesToIdeogramElements(
      [{ type: "obj", desc: "x", rect: { x: 0, y: 0, width: 10, height: 10 }, colorPalette: ["#ff0000", "nope", "#00ff00"] }],
      100,
      100,
    );
    expect(els[0].color_palette).toEqual(["#FF0000", "#00FF00"]);
  });
});

describe("upscale job", () => {
  it("constrains factors per engine", () => {
    expect(upscaleFactorsForEngine("real-esrgan")).toEqual([2, 4]);
    expect(upscaleFactorsForEngine("seedvr2")).toEqual([2, 4]);
    expect(upscaleFactorsForEngine("aura-sr")).toEqual([4]);
    expect(upscaleFactorsForEngine("unknown")).toEqual([2, 4]);
  });

  it("exposes the softness control only for seedvr2 (sc-4815)", () => {
    expect(upscaleEngineHasSoftness("seedvr2")).toBe(true);
    expect(upscaleEngineHasSoftness("real-esrgan")).toBe(false);
    expect(upscaleEngineHasSoftness("aura-sr")).toBe(false);
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

  it("threads softness into a seedvr2 job, and omits it for engines that ignore it (sc-4815)", () => {
    const seed = buildUpscaleJobBody({
      project: { id: "project_1" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      factor: 2,
      engine: "seedvr2",
      displayName: "shot.png",
      softness: 0.5,
    });
    expect(seed.payload.engine).toBe("seedvr2");
    expect(seed.payload.softness).toBe(0.5);

    // Real-ESRGAN ignores softness even if a value is passed.
    const esrgan = buildUpscaleJobBody({
      project: { id: "project_1" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      factor: 2,
      engine: "real-esrgan",
      softness: 0.5,
    });
    expect(esrgan.payload).not.toHaveProperty("softness");

    // seedvr2 without an explicit softness omits the key (worker defaults to 0).
    const seedNoSoftness = buildUpscaleJobBody({
      project: { id: "project_1" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      factor: 2,
      engine: "seedvr2",
    });
    expect(seedNoSoftness.payload).not.toHaveProperty("softness");
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
      fitMode: "crop",
      seed: 42,
      count: 1,
      advanced: {},
    });
  });

  it("threads a non-default fitMode (outpaint canvas-extend) into the body", () => {
    const body = buildEditJobBody({
      project: { id: "p", name: "P" },
      requestedGpu: "auto",
      sourceAssetId: "asset_scratch",
      model: "sdxl",
      prompt: "extend the scene",
      seed: "",
      width: 1820,
      height: 1024,
      fitMode: "outpaint",
    });
    expect(body.fitMode).toBe("outpaint");
    expect(body.width).toBe(1820);
  });

  it("computes output dims for the canvas-extend control (match / extend / crop)", () => {
    // Match canvas → working size unchanged, fit mode irrelevant.
    expect(editOutputDims(1024, 1024, "match", "outpaint")).toEqual({ width: 1024, height: 1024 });
    // 16:9 outpaint on a square → extend width, keep height at native (add side border).
    expect(editOutputDims(1024, 1024, "16:9", "outpaint")).toEqual({ width: 1820, height: 1024 });
    // 16:9 pad behaves the same geometry as outpaint (extend, then bars vs generate).
    expect(editOutputDims(1024, 1024, "16:9", "pad")).toEqual({ width: 1820, height: 1024 });
    // 16:9 crop on a square → shrink to the aspect inside the image (trim height).
    expect(editOutputDims(1024, 1024, "16:9", "crop")).toEqual({ width: 1024, height: 576 });
    // Portrait target on a square extends height.
    expect(editOutputDims(1024, 1024, "9:16", "outpaint")).toEqual({ width: 1024, height: 1820 });
    // Unknown aspect / zero dims fall back to the working size.
    expect(editOutputDims(800, 600, "bogus", "pad")).toEqual({ width: 800, height: 600 });
    expect(editOutputAspectRatio("1:1")).toBe(1);
    expect(editOutputAspectRatio("match")).toBeNull();
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

// Undo/redo history reducer (sc-6106). Snapshots are opaque to the reducer, so the
// tests use plain marker objects in place of real working-image snapshots.
describe("undo/redo history (sc-6106)", () => {
  const A = { id: "A" };
  const B = { id: "B" };
  const C = { id: "C" };

  it("starts empty with nothing to undo or redo", () => {
    const h = emptyHistory();
    expect(h).toEqual({ past: [], future: [] });
    expect(canUndo(h)).toBe(false);
    expect(canRedo(h)).toBe(false);
  });

  it("a checkpoint records the pre-op snapshot and clears any redo branch", () => {
    let h = emptyHistory();
    h = historyCheckpoint(h, A);
    expect(h).toEqual({ past: [A], future: [] });
    expect(canUndo(h)).toBe(true);
    // A stale redo branch is dropped the moment a new op is committed.
    h = { past: [A], future: [C] };
    expect(historyCheckpoint(h, B)).toEqual({ past: [A, B], future: [] });
  });

  it("undo/redo on an exhausted stack is a no-op with a null restore target", () => {
    expect(historyUndo(emptyHistory(), B)).toEqual({ history: { past: [], future: [] }, restore: null });
    expect(historyRedo(emptyHistory(), B)).toEqual({ history: { past: [], future: [] }, restore: null });
  });

  it("walks a two-op session back and forth restoring each step", () => {
    // open (state A) → op1 (A→B) → op2 (B→C): each op checkpoints the pre-op state.
    let h = emptyHistory();
    h = historyCheckpoint(h, A); // committing op1, present is now B
    h = historyCheckpoint(h, B); // committing op2, present is now C
    expect(h).toEqual({ past: [A, B], future: [] });

    // undo from C → restores B, parks C on the redo branch.
    let step = historyUndo(h, C);
    expect(step.restore).toBe(B);
    expect(step.history).toEqual({ past: [A], future: [C] });
    h = step.history;

    // undo from B → restores A.
    step = historyUndo(h, B);
    expect(step.restore).toBe(A);
    expect(step.history).toEqual({ past: [], future: [B, C] });
    h = step.history;
    expect(canUndo(h)).toBe(false);
    expect(canRedo(h)).toBe(true);

    // redo from A → restores B, then C.
    step = historyRedo(h, A);
    expect(step.restore).toBe(B);
    expect(step.history).toEqual({ past: [A], future: [C] });
    step = historyRedo(step.history, B);
    expect(step.restore).toBe(C);
    expect(step.history).toEqual({ past: [A, B], future: [] });
    expect(canRedo(step.history)).toBe(false);
  });

  it("a new op after undo forks history, dropping the redo branch", () => {
    // ...A, B committed; undo back to B; then commit a different op D.
    let h = { past: [A], future: [C] }; // present is B, C is the undone branch
    const D = { id: "D" };
    h = historyCheckpoint(h, B);
    expect(h).toEqual({ past: [A, B], future: [] });
    // The redo target C is gone — redo now has nothing to restore.
    expect(historyRedo(h, D).restore).toBe(null);
  });

  it("bounds the undo depth, evicting the oldest snapshots", () => {
    let h = emptyHistory();
    for (let i = 0; i < HISTORY_LIMIT + 5; i += 1) h = historyCheckpoint(h, { id: i });
    expect(h.past).toHaveLength(HISTORY_LIMIT);
    // The five oldest (0..4) were evicted; the newest survive.
    expect(h.past[0]).toEqual({ id: 5 });
    expect(h.past[h.past.length - 1]).toEqual({ id: HISTORY_LIMIT + 4 });
  });

  it("honors a custom limit on checkpoint and undo", () => {
    let h = emptyHistory();
    h = historyCheckpoint(h, A, 2);
    h = historyCheckpoint(h, B, 2);
    h = historyCheckpoint(h, C, 2);
    expect(h.past).toEqual([B, C]);
  });

  it("does not mutate the input history", () => {
    const h = emptyHistory();
    const after = historyCheckpoint(h, A);
    expect(h).toEqual({ past: [], future: [] });
    expect(after).not.toBe(h);
  });
});

// Smart-select (sc-3751): box-prompt segment job body + the pure rect/mask helpers.
describe("smart-select (sc-3751)", () => {
  it("builds an image_segment generic-jobs body with the box prompt", () => {
    const body = buildSegmentJobBody({
      project: { id: "proj_1", name: "Demo" },
      requestedGpu: "auto",
      sourceAssetId: "asset_src",
      box: [10, 20, 110, 220],
      displayName: "cat.png",
    });
    expect(body.type).toBe("image_segment");
    expect(body.projectId).toBe("proj_1");
    expect(body.projectName).toBe("Demo");
    expect(body.requestedGpu).toBe("auto");
    expect(body.payload).toEqual({
      projectId: "proj_1",
      sourceAssetId: "asset_src",
      box: [10, 20, 110, 220],
      displayName: "cat.png",
    });
  });

  it("converts a rect to an ordered, rounded [x1,y1,x2,y2] box", () => {
    expect(rectToSegmentBox({ x: 10.4, y: 20.6, width: 100.2, height: 199.9 })).toEqual([10, 21, 111, 221]);
    // Negative width/height (dragged up-left) is ordered to a positive box.
    expect(rectToSegmentBox({ x: 110, y: 220, width: -100, height: -200 })).toEqual([10, 20, 110, 220]);
  });

  it("tints a white-on-black mask to translucent-pink-on-transparent in place", () => {
    // 2 px: one white (foreground), one black (background).
    const data = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
    const out = tintMaskRgbaInPlace(data);
    expect(out).toBe(data); // mutated in place
    // foreground → the preview pink
    expect([data[0], data[1], data[2], data[3]]).toEqual(MASK_PREVIEW_RGBA);
    // background → fully transparent (rgb untouched, alpha 0)
    expect(data[7]).toBe(0);
  });
});
