import React, { useCallback, useEffect, useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line, Rect, Transformer } from "react-konva";
import { apiFetch } from "../api.js";
import { terminalStatuses } from "../jobTypes.js";
import { useAppContext } from "../context/AppContext.js";
import { DEFAULT_MAC_CAPABILITIES, macFeatureBlock, macUpscaleEngineBlocked } from "../macGating.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { makeObjElement, makeTextElement, normalizeHexColor } from "../ideogramCaption.js";

const MIN_SCALE = 0.05;
const MAX_SCALE = 16;
const ZOOM_STEP = 1.2;
const MIN_CROP_PX = 8;

// Future-tool scaffold (epic 2427) — rendered as inert buttons so the frame + the
// next slices' insertion points stay in place. All current epic-2427 tools are live
// (Move + Crop + Upscale + Color + AI Edit + Detail), so this is empty for now.
const UPCOMING_TOOLS = [];

// Models that can edit an existing image with a prompt — the manifest tags them
// with an `edit_image`/`image_edit` capability (same filter the Image Studio uses).
export function editCapableModels(imageModels) {
  return (imageModels ?? []).filter((model) => {
    const caps = model.capabilities ?? [];
    return caps.includes("edit_image") || caps.includes("image_edit");
  });
}

// Models that can run the tile-ControlNet detail refine — the manifest tags them
// `image_detail` (sc-2437/sc-2438; SDXL/RealVisXL only). RealVisXL is the recommended
// photoreal backbone per the spike.
export function detailCapableModels(imageModels) {
  return (imageModels ?? []).filter((model) => (model.capabilities ?? []).includes("image_detail"));
}

// The `POST /api/v1/image/jobs` body for an in-editor prompt edit (sc-2435). Reuses
// the existing `mode:"edit_image"` flow: the working bitmap is staged as a scratch
// asset (sc-2432) and referenced by `sourceAssetId`; the result is the new working
// image at the same dimensions. Pure for unit testing.
export function buildEditJobBody({
  project,
  requestedGpu,
  sourceAssetId,
  maskAssetId,
  model,
  prompt,
  seed,
  width,
  height,
  fitMode = "crop",
}) {
  const body = {
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    mode: "edit_image",
    sourceAssetId,
    model,
    prompt,
    negativePrompt: "",
    width,
    height,
    // How the source is fitted to width×height (epic 2551). For canvas-extend outpaint
    // the dims are the larger target aspect and the worker generates the new border.
    fitMode,
    seed: seed == null || seed === "" ? null : Number(seed),
    count: 1,
    advanced: {},
  };
  // Inpaint mask (sc-2436): only sent for inpaint-capable models with a painted
  // region; the worker confines the edit to it. Omitted entirely otherwise.
  if (maskAssetId) body.maskAssetId = maskAssetId;
  return body;
}

// Output aspect presets for the editor's canvas-extend / outpaint control (sc-2556).
// "match" keeps the working size, so the fit mode then has no border to act on.
export const EDIT_OUTPUT_ASPECTS = [
  { key: "match", label: "Match canvas", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
  { key: "9:16", label: "9:16", ratio: 9 / 16 },
  { key: "4:3", label: "4:3", ratio: 4 / 3 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "3:2", label: "3:2", ratio: 3 / 2 },
  { key: "2:3", label: "2:3", ratio: 2 / 3 },
];

export function editOutputAspectRatio(key) {
  return EDIT_OUTPUT_ASPECTS.find((aspect) => aspect.key === key)?.ratio ?? null;
}

// Output W×H for an editor edit given the target aspect + fit mode, keeping the working
// image at native scale (never upscales). "match"/unknown aspect → working size. crop =
// largest target-aspect rect INSIDE the image (trim the overflow); pad/outpaint =
// smallest target-aspect canvas CONTAINING the image (extend → border to fill). Pure.
export function editOutputDims(workingW, workingH, aspectKey, fitMode) {
  const ratio = editOutputAspectRatio(aspectKey);
  if (!ratio || !workingW || !workingH) return { width: workingW, height: workingH };
  const imageRatio = workingW / workingH;
  let width;
  let height;
  if (fitMode === "crop") {
    // Cover: shrink to the target aspect within the image (trim).
    if (ratio >= imageRatio) {
      width = workingW;
      height = Math.round(workingW / ratio);
    } else {
      height = workingH;
      width = Math.round(workingH * ratio);
    }
  } else {
    // Pad / outpaint: extend to the target aspect around the image (add border).
    if (ratio >= imageRatio) {
      height = workingH;
      width = Math.round(workingH * ratio);
    } else {
      width = workingW;
      height = Math.round(workingW / ratio);
    }
  }
  return { width: Math.max(1, width), height: Math.max(1, height) };
}

// Whether a model accepts an inpaint mask — the manifest tags it `image_inpaint`
// (sc-2476). Gates the mask tool in the editor. Pure.
export function modelIsInpaintCapable(model) {
  return (model?.capabilities ?? []).includes("image_inpaint");
}

// Whether the brush strokes form an actual mask region (at least one non-erase
// stroke with a drawn segment). Erase-only strokes don't count. Pure.
export function maskHasContent(lines) {
  return (lines ?? []).some((line) => !line.erase && (line.points?.length ?? 0) >= 2);
}

// Color-grade controls (sc-2439). Each is a normalized −1..1 slider where 0 is the
// identity; `gradePixel` defines the math. Pure data so the panel + reset are trivial.
const COLOR_ADJUSTMENTS = [
  { key: "brightness", label: "Brightness" },
  { key: "contrast", label: "Contrast" },
  { key: "saturation", label: "Saturation" },
  { key: "temperature", label: "Temperature" },
];

export const IDENTITY_COLOR_ADJUST = { brightness: 0, contrast: 0, saturation: 0, temperature: 0 };

const clamp8 = (value) => (value < 0 ? 0 : value > 255 ? 255 : Math.round(value));

// True when no grade is applied (all sliders at 0) — lets the preview/Apply skip work.
export function isIdentityAdjust(adjust) {
  return COLOR_ADJUSTMENTS.every(({ key }) => !(adjust?.[key]));
}

// Grade one RGB pixel by the −1..1 adjustments, in a fixed order: temperature
// (warm raises R / lowers B), brightness (additive), contrast (around mid-gray),
// then saturation (blend toward/away from luma). Pure + clamped for unit testing.
export function gradePixel([r, g, b], adjust) {
  const { brightness = 0, contrast = 0, saturation = 0, temperature = 0 } = adjust ?? {};
  r += temperature * 30;
  b -= temperature * 30;
  const add = brightness * 255;
  r += add;
  g += add;
  b += add;
  const cf = 1 + contrast;
  r = (r - 128) * cf + 128;
  g = (g - 128) * cf + 128;
  b = (b - 128) * cf + 128;
  const luma = 0.299 * r + 0.587 * g + 0.114 * b;
  const sf = 1 + saturation;
  r = luma + sf * (r - luma);
  g = luma + sf * (g - luma);
  b = luma + sf * (b - luma);
  return [clamp8(r), clamp8(g), clamp8(b)];
}

// Apply the grade to a flat RGBA buffer in place (alpha untouched). Shared by the
// Konva live-preview filter and the Apply bake, so preview === baked result.
export function applyColorAdjustments(data, adjust) {
  if (isIdentityAdjust(adjust)) return;
  for (let i = 0; i < data.length; i += 4) {
    const [r, g, b] = gradePixel([data[i], data[i + 1], data[i + 2]], adjust);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
}

// Konva custom filter for the live preview — reads the grade from the node's
// `colorAdjust` attr (set declaratively by react-konva) and runs the shared math.
function konvaColorFilter(imageData) {
  applyColorAdjustments(imageData.data, this.getAttr("colorAdjust"));
}

// Upscale engines + their valid factors (sc-2433). Real-ESRGAN 2x/4x is the cross-platform default
// (`ort`: CoreML on Mac / CUDA off-Mac, sc-3489 / sc-5499); SeedVR2 2x/4x is the one-step diffusion
// upscaler (native MLX on Mac / candle off-Mac, epic 4811 / sc-5928 / sc-5160) and exposes a
// detail/softness control (`softness: true`). `aura-sr` is kept here only so a stale saved selection
// gracefully falls back — it is hidden on every platform (dropped, sc-3668 / sc-5499) via
// `macUpscaleEngineBlocked`.
const UPSCALE_ENGINES = [
  { key: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
  { key: "seedvr2", label: "SeedVR2", factors: [2, 4], softness: true },
  { key: "aura-sr", label: "AuraSR", factors: [4] },
];

export function upscaleFactorsForEngine(engineKey) {
  const found = UPSCALE_ENGINES.find((entry) => entry.key === engineKey);
  return found ? found.factors : [2, 4];
}

export function upscaleEngineHasSoftness(engineKey) {
  return Boolean(UPSCALE_ENGINES.find((entry) => entry.key === engineKey)?.softness);
}

// The `POST /api/v1/jobs` body for a standalone image_upscale job (sc-2431). The
// worker reads sourceAssetId/factor/engine from the payload; displayName names the
// result. `softness` (0..1) is a SeedVR2-only detail knob (sc-4815) — omitted for engines
// that ignore it. Pure for unit testing.
export function buildUpscaleJobBody({ project, requestedGpu, sourceAssetId, factor, engine, displayName, softness }) {
  const payload = { projectId: project.id, sourceAssetId, factor, engine, displayName };
  if (upscaleEngineHasSoftness(engine) && typeof softness === "number") {
    payload.softness = softness;
  }
  return {
    type: "image_upscale",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload,
  };
}

// The `POST /api/v1/jobs` body for a standalone image_detail job (sc-2438). Same
// generic-jobs shape as upscale; the worker reads model + advanced.strength/cnScale
// from the payload (recipe defaults locked by the sc-2437 spike). Pure for testing.
export function buildDetailJobBody({ project, requestedGpu, sourceAssetId, model, strength, cnScale, displayName }) {
  return {
    type: "image_detail",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload: {
      projectId: project.id,
      sourceAssetId,
      model,
      displayName,
      advanced: { strength, cnScale },
    },
  };
}

// Filename for a Save / Download export (sc-2434): the source name with an
// "-edited" suffix before the extension, always .png — the working image is
// rasterized to PNG, so the original extension would be misleading. Pure.
export function editedFilename(source) {
  const base = (source?.name || "image").replace(/\.[^./\\]+$/, "").trim() || "image";
  return `${base}-edited.png`;
}

// Provenance for a saved edit, stored under the new asset's top-level `extra`
// (sc-2434): which source it was derived from + the ordered edit chain
// (crop/upscale/…) applied this session. Pure for unit testing.
export function buildSaveProvenance({ source, edits, width, height }) {
  return {
    editor: "image_editor",
    source: source?.assetId
      ? { kind: "asset", assetId: source.assetId, name: source.name ?? null }
      : { kind: "upload", name: source?.name ?? null },
    edits: edits ?? [],
    width: width ?? null,
    height: height ?? null,
  };
}

// Predefined crop ratios (width / height). Rotate swaps to the transpose; 1:1 and
// Freeform are unaffected.
const CROP_RATIOS = [
  { key: "free", label: "Freeform", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "5:7", label: "5:7", ratio: 5 / 7 },
  { key: "8:10", label: "8:10", ratio: 8 / 10 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// Resolve a ratio key (+ rotate) to a concrete width/height ratio, or null for
// freeform. Rotating transposes non-square ratios (3:4 → 4:3); 1:1 is a no-op.
export function cropRatioForKey(key, rotated) {
  const found = CROP_RATIOS.find((entry) => entry.key === key);
  const base = found ? found.ratio : null;
  if (base == null || base === 1) return base;
  return rotated ? 1 / base : base;
}

// Largest rect of the given ratio that fits in the image, centered. Freeform
// (null ratio) defaults to a centered 80% box. Returns image-pixel coords.
export function centeredCropRect(imgW, imgH, ratio) {
  if (ratio == null) {
    const w = imgW * 0.8;
    const h = imgH * 0.8;
    return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
  }
  let w = imgW;
  let h = w / ratio;
  if (h > imgH) {
    h = imgH;
    w = h * ratio;
  }
  return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
}

// The four dim rectangles that mask everything outside the crop rect (image coords).
function cropOverlayRects(imgW, imgH, rect) {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  return [
    { x: 0, y: 0, width: imgW, height: rect.y },
    { x: 0, y: bottom, width: imgW, height: imgH - bottom },
    { x: 0, y: rect.y, width: rect.x, height: rect.height },
    { x: right, y: rect.y, width: imgW - right, height: rect.height },
  ];
}

// ── Box layout (Workstream A, sc-6089) ───────────────────────────────────────
// The colored-box layout tool lets the user draw labeled rectangles that drive
// generation two ways: a structured `bbox` for Ideogram 4 (epic 4725) and a
// color-keyed region prompt for any edit model. A box is a pure data record in
// image-pixel coords:
//   { id, rect:{x,y,width,height}, color:"#RRGGBB", type:"obj"|"text",
//     desc, text? /* type==="text" */, colorPalette?:["#RRGGBB",…] /* ≤5 */ }
// The conversion/validation below is pure (no React/Konva) so the box tool, the
// Ideogram elements adapter (sc-6095), and the color-keyed path (sc-6093/6094)
// all share one source of truth.
export const BOX_TYPES = ["obj", "text"];

// Ideogram's structured-caption palette limits (epic 4725 S3): ≤5 colors per
// element, ≤16 across the whole document.
export const MAX_BOX_PALETTE = 5;
export const MAX_DOCUMENT_PALETTE = 16;

// Uppercase `#RRGGBB` only — the Ideogram S3 contract is case-sensitive, so a
// lowercase value is invalid (the per-box metadata editor, sc-6091, normalizes
// user input to uppercase before storing). Pure.
const HEX_COLOR_RE = /^#[0-9A-F]{6}$/;
export function isValidHexColor(color) {
  return typeof color === "string" && HEX_COLOR_RE.test(color);
}

// Normalize one pixel coordinate to Ideogram's 0–1000 grid (origin top-left),
// rounded to an integer and clamped to the canvas. Guards a zero/absent dim.
function normBboxCoord(px, dim) {
  if (!dim) return 0;
  return clamp(Math.round((px / dim) * 1000), 0, 1000);
}

// rect {x,y,width,height} (image-pixel coords) → `[y_min, x_min, y_max, x_max]`,
// integers normalized 0–1000, origin top-left, clamped to the canvas. Component
// order matches epic 4725 S3 exactly. Robust to flipped (negative-size) rects.
export function rectToBbox(rect, imgW, imgH) {
  const x0 = normBboxCoord(rect.x, imgW);
  const x1 = normBboxCoord(rect.x + rect.width, imgW);
  const y0 = normBboxCoord(rect.y, imgH);
  const y1 = normBboxCoord(rect.y + rect.height, imgH);
  return [Math.min(y0, y1), Math.min(x0, x1), Math.max(y0, y1), Math.max(x0, x1)];
}

// Inverse of `rectToBbox` for round-tripping a stored bbox back onto a canvas of
// the given size. Returns image-pixel coords (unrounded, like `centeredCropRect`);
// the 0–1000 quantization means the round-trip is exact only to grid resolution.
export function bboxToRect([yMin, xMin, yMax, xMax], imgW, imgH) {
  return {
    x: (xMin / 1000) * imgW,
    y: (yMin / 1000) * imgH,
    width: ((xMax - xMin) / 1000) * imgW,
    height: ((yMax - yMin) / 1000) * imgH,
  };
}

// A per-element palette is valid when it is ≤5 uppercase `#RRGGBB` colors. An
// absent palette is valid (it's optional). Pure.
export function boxPaletteIsValid(palette) {
  if (palette == null) return true;
  if (!Array.isArray(palette)) return false;
  return palette.length <= MAX_BOX_PALETTE && palette.every(isValidHexColor);
}

// The document-level palette: the de-duplicated union of every box's per-element
// `colorPalette`, order-preserving (Ideogram key order is quality-relevant, S3). Pure.
export function documentPalette(boxes) {
  const seen = [];
  for (const box of boxes ?? []) {
    for (const color of box?.colorPalette ?? []) {
      if (!seen.includes(color)) seen.push(color);
    }
  }
  return seen;
}

// The document palette must stay ≤16 colors overall (epic 4725 S3). Pure.
export function documentPaletteIsValid(boxes) {
  return documentPalette(boxes).length <= MAX_DOCUMENT_PALETTE;
}

// A box is valid for serialization when it has positive geometry, a known type,
// a non-empty description, and — for text elements — a non-empty literal string.
// Color/palette validity is checked separately (`isValidHexColor`/`boxPaletteIsValid`)
// since the color-keyed path needs only color + desc, not a full Ideogram element. Pure.
export function boxIsValid(box) {
  if (!box || !box.rect) return false;
  if (!(box.rect.width > 0) || !(box.rect.height > 0)) return false;
  if (!BOX_TYPES.includes(box.type)) return false;
  if (typeof box.desc !== "string" || box.desc.trim() === "") return false;
  if (box.type === "text" && (typeof box.text !== "string" || box.text.trim() === "")) return false;
  return true;
}

// ── Box drawing tool (Workstream A, sc-6090) ─────────────────────────────────
// A small palette of distinct, nameable colors for the box tool, plus a custom
// `#RRGGBB`. All entries are uppercase #RRGGBB (valid per `isValidHexColor`) so a
// drawn box is well-formed for the color-keyed path and the Ideogram adapter.
export const BOX_PALETTE = [
  { name: "Red", value: "#FF0000" },
  { name: "Green", value: "#00C853" },
  { name: "Blue", value: "#2962FF" },
  { name: "Yellow", value: "#FFD600" },
  { name: "Orange", value: "#FF6D00" },
  { name: "Purple", value: "#AA00FF" },
  { name: "Cyan", value: "#00B8D4" },
  { name: "Pink", value: "#FF4081" },
];

// Smallest box (image pixels) a drag must cover to commit — a click or tiny
// smudge is discarded rather than creating a degenerate box.
export const MIN_BOX_PX = 8;

// Axis-aligned rect spanning two points (image-pixel coords). Pure — the drag
// direction (up-left vs down-right) is normalized to a positive-size rect.
export function rectFromPoints(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

// Clamp a rect to the canvas, keeping width/height ≥ minPx and the rect fully
// inside [0,imgW]×[0,imgH]. Mirrors the crop tool's clamp but pure (takes dims).
export function clampRectToCanvas(rect, imgW, imgH, minPx = MIN_BOX_PX) {
  const width = clamp(rect.width, minPx, imgW);
  const height = clamp(rect.height, minPx, imgH);
  return {
    width,
    height,
    x: clamp(rect.x, 0, imgW - width),
    y: clamp(rect.y, 0, imgH - height),
  };
}

// Build a new box record (the sc-6089 model) from a drawn rect + color. Metadata
// (type/desc/text/colorPalette) starts at safe defaults; the per-box metadata
// editor (sc-6091) fills it in. `id` is supplied by the caller (session-unique).
export function makeBox(id, rect, color) {
  return { id, rect, color, type: "obj", desc: "", text: "", colorPalette: [] };
}

// A semi-transparent CSS rgba() fill from a `#RRGGBB` color for the box overlay.
// Pure; falls back to a neutral fill if the color isn't a valid 6-digit hex.
export function boxFillStyle(hex, alpha) {
  if (!isValidHexColor(hex)) return `rgba(127,127,127,${alpha})`;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Per-box metadata (Workstream A, sc-6091) ─────────────────────────────────
// Append a color to a per-element palette (uppercased), ignoring duplicates,
// invalid hex, and anything past the ≤5 cap. Pure; returns the same array
// reference when nothing changes so callers can no-op cheaply.
export function addPaletteColor(palette, color, max = MAX_BOX_PALETTE) {
  const list = palette ?? [];
  const value = typeof color === "string" ? color.toUpperCase() : color;
  if (!isValidHexColor(value) || list.includes(value) || list.length >= max) return list;
  return [...list, value];
}

// Remove a color from a per-element palette. Pure; returns a new array.
export function removePaletteColor(palette, color) {
  return (palette ?? []).filter((entry) => entry !== color);
}

// What a box still needs to serialize as a valid Ideogram element (S3): a
// description, the literal text for a text element, and a valid ≤5 palette.
// Returns a human list of what's missing ("" when ready). The color-keyed edit
// path only needs color + desc, so this does NOT gate that path. Pure.
export function boxMetadataGaps(box) {
  if (!box) return [];
  const gaps = [];
  if (typeof box.desc !== "string" || box.desc.trim() === "") gaps.push("a description");
  if (box.type === "text" && (typeof box.text !== "string" || box.text.trim() === "")) gaps.push("the literal text");
  if (!boxPaletteIsValid(box.colorPalette)) gaps.push("a valid color palette (≤5)");
  return gaps;
}

// ── Blank-canvas "New layout" (Workstream A, sc-6092) ────────────────────────
// A from-scratch substrate for layout-from-nothing (Ideogram text-to-image). The
// dimensions obey Ideogram's constraints: multiples of 16 within [256, 2048].
export const BLANK_CANVAS_MIN = 256;
export const BLANK_CANVAS_MAX = 2048;
export const BLANK_CANVAS_SIZES = [512, 768, 1024, 1536, 2048];

// Snap a pixel dimension to a multiple of 16 within [256, 2048] (Ideogram limits).
function snapCanvasDim(px) {
  return clamp(Math.round(px / 16) * 16, BLANK_CANVAS_MIN, BLANK_CANVAS_MAX);
}

// Target W×H for a blank layout from an aspect preset + a long-side size. Both
// dims are multiples of 16 in [256, 2048]. "match"/unknown aspect → square. Pure.
export function blankCanvasDims(aspectKey, longSide) {
  const ratio = editOutputAspectRatio(aspectKey) ?? 1;
  let width;
  let height;
  if (ratio >= 1) {
    width = longSide;
    height = longSide / ratio;
  } else {
    height = longSide;
    width = longSide * ratio;
  }
  return { width: snapCanvasDim(width), height: snapCanvasDim(height) };
}

// ── Bake → pass-through edit (Workstream A, sc-6093) ─────────────────────────
// Paint each box as a solid colored rectangle onto a 2D context — the color-keyed
// region signal the edit model reads ("replace the {color} region with …"). The
// caller draws the working image first; this overlays the boxes. Pure given the
// context, so the paint order/coords are unit-testable without a real canvas.
export function paintBoxesOnContext(ctx, boxes) {
  for (const box of boxes ?? []) {
    ctx.fillStyle = box.color;
    ctx.fillRect(box.rect.x, box.rect.y, box.rect.width, box.rect.height);
  }
}

// ── Auto color-prompt (Workstream A, sc-6094) ────────────────────────────────
// Friendly color name for a palette/custom hex — palette colors get their name
// lowercased (#FF0000 → "red"); anything else falls back to the hex itself so the
// prompt still references a concrete color. Pure.
export function colorName(hex) {
  const found = BOX_PALETTE.find((entry) => entry.value === hex);
  return found ? found.name.toLowerCase() : hex;
}

// Compose an editable color-keyed edit prompt from the boxes: one clause per
// described box, referencing it by its visible color so the model maps region →
// element. Boxes missing the needed text (obj → desc; text → literal) are skipped.
// Pure; "" when nothing is describable yet. The user can edit the result freely.
export function composeColorPrompt(boxes) {
  const clauses = [];
  for (const box of boxes ?? []) {
    const name = colorName(box.color);
    if (box.type === "text") {
      const text = (box.text ?? "").trim();
      if (!text) continue;
      const desc = (box.desc ?? "").trim();
      clauses.push(`place the text "${text}" in the ${name} region${desc ? ` (${desc})` : ""}`);
    } else {
      const desc = (box.desc ?? "").trim();
      if (!desc) continue;
      clauses.push(`replace the ${name} region with ${desc}`);
    }
  }
  if (!clauses.length) return "";
  return `${clauses.map((clause) => clause.charAt(0).toUpperCase() + clause.slice(1)).join(". ")}.`;
}

// ── Boxes → Ideogram elements[] adapter (Workstream A, sc-6095) ──────────────
// Convert the editor's boxes into Ideogram 4 structured-caption `elements[]`
// (epic 4725 S3 contract), one element per box, via ideogramCaption.js's factories
// so the canonical key order is guaranteed (obj: type,bbox,desc,color_palette;
// text: type,bbox,text,desc,color_palette). bbox is the 0–1000 grid from
// `rectToBbox`; palette entries are normalized to uppercase #RRGGBB and dropped if
// empty/invalid (an empty palette is omitted entirely). Pure — this supplies only
// the spatial elements; the non-spatial caption fields are epic 4725's (S3/S4/S7).
export function boxesToIdeogramElements(boxes, imgW, imgH) {
  return (boxes ?? []).map((box) => {
    const bbox = rectToBbox(box.rect, imgW, imgH);
    const palette = (box.colorPalette ?? []).map(normalizeHexColor).filter(Boolean);
    const color_palette = palette.length ? palette : null;
    if (box.type === "text") {
      return makeTextElement({ bbox, text: box.text ?? "", desc: box.desc ?? "", color_palette });
    }
    return makeObjElement({ bbox, desc: box.desc ?? "", color_palette });
  });
}

// Decode a blob into an HTMLImageElement via a same-origin object: URL. Asset
// files are served cross-origin from the API in local dev, so loading the bytes
// this way (rather than an <img crossOrigin> against the file URL) guarantees the
// Konva canvas is never tainted — later crop/export (sc-2430/sc-2434) need to read
// pixels back. Resolves { image, objectUrl }; caller owns revoking objectUrl.
function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => resolve({ image, objectUrl });
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not decode image"));
    };
    image.src = objectUrl;
  });
}

export function ImageEditor() {
  const {
    activeProject,
    assets,
    characters,
    setPreviewAsset,
    token,
    requestedGpu,
    jobs,
    importAsset,
    purgeAsset,
    registerLeaveGuard,
    imageModels,
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  // Mac UI gating (sc-3486): the upscale tool itself runs in-process on Rust (Real-ESRGAN,
  // sc-3489), so it is available on a gated Mac — this block is a defensive guard that stays
  // null. The second engine (AuraSR) is dropped on Mac (sc-3668) and gated per-engine below.
  const macUpscaleBlock = macFeatureBlock(macCapabilities, "imageUpscale");

  // The working-image session: the single bitmap every tool operates on, plus its
  // provenance. This state is the contract consumed by crop/upscale/save and the
  // later AI tools (epic 2427). `objectUrl` is tracked so we can revoke it.
  const [working, setWorking] = useState(null);
  const [status, setStatus] = useState({ loading: false, error: "" });
  const [pickerOpen, setPickerOpen] = useState(false);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });

  // Crop tool (sc-2430): client-side, rasterized into a new working image on Apply.
  const [tool, setTool] = useState("move");
  const [ratioKey, setRatioKey] = useState("free");
  const [rotated, setRotated] = useState(false);
  const [cropRect, setCropRect] = useState(null); // image-pixel coords, or null

  // Upscale tool (sc-2433): engine + factor for the in-flight request.
  const [upscaleEngine, setUpscaleEngine] = useState("real-esrgan");
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  // SeedVR2 detail/softness knob (0..1, sc-4815) — only meaningful for the seedvr2 engine.
  const [upscaleSoftness, setUpscaleSoftness] = useState(0);
  // Engines offered in the picker; AuraSR is dropped on every platform (sc-3668 / sc-5499).
  const availableUpscaleEngines = UPSCALE_ENGINES.filter(
    (entry) => !macUpscaleEngineBlocked(macCapabilities, entry.key),
  );
  // If the selected engine got gated out (e.g. a stale saved AuraSR selection), fall back to the
  // default real-esrgan engine (the guaranteed-available cross-platform upscaler) so the tool stays usable.
  useEffect(() => {
    if (macUpscaleEngineBlocked(macCapabilities, upscaleEngine)) {
      setUpscaleEngine("real-esrgan");
      if (!upscaleFactorsForEngine("real-esrgan").includes(upscaleFactor)) {
        setUpscaleFactor(upscaleFactorsForEngine("real-esrgan")[0]);
      }
    }
  }, [macCapabilities, upscaleEngine, upscaleFactor]);

  // Color grade (sc-2439): non-destructive −1..1 adjustments previewed live via a
  // Konva filter, baked into the working image on Apply.
  const [colorAdjust, setColorAdjust] = useState(IDENTITY_COLOR_ADJUST);

  // AI prompt edit (sc-2435): an edit-capable model + instruction + optional seed,
  // run against the working image through the existing edit_image flow.
  const editModels = editCapableModels(imageModels);
  const [editModel, setEditModel] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [editSeed, setEditSeed] = useState("");
  // Canvas-extend / outpaint (sc-2556): target output aspect (default "match" = the
  // working size) and how to fill it (crop trims, pad bars, outpaint generates).
  const [editAspect, setEditAspect] = useState("match");
  const [editFitMode, setEditFitMode] = useState("crop");

  // Detail enhance (sc-2438): tile-ControlNet refine over the working image. Backbone
  // (SDXL/RealVisXL) + strength (the "detail amount" — higher invents more texture) +
  // structure-lock (controlnet scale). Defaults are the sc-2437 spike's locked recipe.
  const detailModels = detailCapableModels(imageModels);
  const [detailModel, setDetailModel] = useState("");
  const [detailStrength, setDetailStrength] = useState(0.55);
  const [detailCnScale, setDetailCnScale] = useState(0.7);

  // Inpaint mask (sc-2436): freehand brush strokes in image-pixel coords, rasterized
  // to a mask asset on Run for inpaint-capable models. `maskMode` is the paint sub-mode
  // of the AI Edit tool (Stage panning is suspended while it's on).
  const [maskLines, setMaskLines] = useState([]); // [{ points:[x,y,…], size, erase }]
  const [maskMode, setMaskMode] = useState(false);
  const [maskBrush, setMaskBrush] = useState(64);
  const [maskErase, setMaskErase] = useState(false);
  const maskPaintingRef = useRef(false);

  // Box layout tool (sc-6090): colored rectangles drawn over the working image in
  // image-pixel coords. They drive the color-keyed edit path (sc-6093) and the
  // Ideogram bbox path (sc-6095). Session-only overlay state — boxes are not baked
  // into the working bitmap here, so they don't mark the session dirty.
  const [boxes, setBoxes] = useState([]); // [{ id, rect, color, type, desc, text, colorPalette }]
  const [selectedBoxId, setSelectedBoxId] = useState(null);
  const [boxColor, setBoxColor] = useState(BOX_PALETTE[0].value);
  const [boxDraft, setBoxDraft] = useState(null); // live rect during a drag-draw
  const boxDrawingRef = useRef(false);
  const boxStartRef = useRef(null);
  const boxIdRef = useRef(0);
  const boxNodeRefs = useRef(new Map()); // id → Konva node, for transformer binding
  const boxTransformerRef = useRef(null);

  // Blank-canvas "New layout" (sc-6092): a from-scratch substrate for box layout
  // (Ideogram text-to-image). The modal picks an aspect + long-side size → W×H.
  const [newLayoutOpen, setNewLayoutOpen] = useState(false);
  const [layoutAspect, setLayoutAspect] = useState("1:1");
  const [layoutSize, setLayoutSize] = useState(1024);

  // Default the edit-model selection to the first edit-capable model once the model
  // list loads, and recover if the current pick stops being edit-capable.
  useEffect(() => {
    const caps = editCapableModels(imageModels);
    if (caps.length && !caps.some((model) => model.id === editModel)) setEditModel(caps[0].id);
  }, [imageModels, editModel]);

  // Same default/self-heal for the detail backbone.
  useEffect(() => {
    const caps = detailCapableModels(imageModels);
    if (caps.length && !caps.some((model) => model.id === detailModel)) setDetailModel(caps[0].id);
  }, [imageModels, detailModel]);

  // The chosen edit model + whether it accepts an inpaint mask (gates the mask tool).
  const selectedEditModel = editModels.find((model) => model.id === editModel) ?? null;
  const canMask = modelIsInpaintCapable(selectedEditModel);

  // Leave paint mode (restoring Stage panning) when the edit tool is closed or the
  // model can't inpaint — otherwise the canvas would stay in a paint state with no UI.
  useEffect(() => {
    if (maskMode && (tool !== "edit" || !canMask)) setMaskMode(false);
  }, [tool, canMask, maskMode]);

  // Save / export (sc-2434). `dirty` tracks edits not yet persisted to the Library;
  // `edits` is the ordered provenance chain; `savedAssetId` flags a completed Save
  // for the bar's "Saved" hint. A fresh open clears all three.
  const [dirty, setDirty] = useState(false);
  const [edits, setEdits] = useState([]);
  const [saving, setSaving] = useState(false);
  const [savedAssetId, setSavedAssetId] = useState(null);
  // An in-flight AI op (upscale now; AI-edit / detail later) on the working image.
  // The seam (sc-2432): stage the working bitmap as a scratch asset, run a worker
  // job against it, load the result back, then purge the scratch + result so the
  // session only persists on Save. { jobId, scratch (asset), source, label } | null.
  const [aiOp, setAiOp] = useState(null);

  const containerRef = useRef(null);
  const objectUrlRef = useRef(null);
  const needsFitRef = useRef(false);
  const cropRectRef = useRef(null);
  const transformerRef = useRef(null);
  const imageNodeRef = useRef(null); // Konva image node — cached for color-grade filtering
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  const imageAssets = (assets ?? []).filter(assetCanRenderAsImage);

  // Track the container size so the Konva stage fills the available canvas area.
  // Measure once up front (a ResizeObserver alone can miss the first layout) and
  // then observe for later window / layout changes.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const measure = () => setStageSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Revoke the live object URL when the editor unmounts.
  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  const fitToView = useCallback(() => {
    if (!working || !stageSize.width || !stageSize.height) return;
    const scale = clamp(
      Math.min(stageSize.width / working.width, stageSize.height / working.height) * 0.92,
      MIN_SCALE,
      MAX_SCALE,
    );
    setView({
      scale,
      x: (stageSize.width - working.width * scale) / 2,
      y: (stageSize.height - working.height * scale) / 2,
    });
  }, [working, stageSize.width, stageSize.height]);

  // Fit a freshly loaded image once the stage has been measured (the stage may be
  // 0×0 on the first render before the ResizeObserver fires).
  useEffect(() => {
    if (needsFitRef.current && working && stageSize.width && stageSize.height) {
      needsFitRef.current = false;
      fitToView();
    }
  }, [working, stageSize.width, stageSize.height, fitToView]);

  const installWorkingImage = useCallback((image, objectUrl, source) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = objectUrl;
    needsFitRef.current = true;
    setTool("move");
    setCropRect(null);
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    // A new working bitmap invalidates the mask (dims/content changed).
    setMaskLines([]);
    setMaskMode(false);
    // Boxes are in image-pixel coords → a new bitmap (open/crop/upscale/AI op) invalidates them.
    setBoxes([]);
    setSelectedBoxId(null);
    setBoxDraft(null);
    boxNodeRefs.current.clear();
    boxDrawingRef.current = false;
    setWorking({
      image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      source,
    });
  }, []);

  const openFromBlob = useCallback(
    async (blob, source) => {
      setStatus({ loading: true, error: "" });
      try {
        const { image, objectUrl } = await blobToImage(blob);
        installWorkingImage(image, objectUrl, source);
        // A freshly opened image is a clean session — clear edit/provenance state.
        setEdits([]);
        setDirty(false);
        setSavedAssetId(null);
        setStatus({ loading: false, error: "" });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not open image" });
      }
    },
    [installWorkingImage],
  );

  const openAsset = useCallback(
    async (assetId) => {
      const asset = imageAssets.find((item) => item.id === assetId);
      if (!asset) return;
      const url = assetUrl(asset);
      if (!url) {
        setStatus({ loading: false, error: "Asset has no media file" });
        return;
      }
      setStatus({ loading: true, error: "" });
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to load asset (${res.status})`);
        const blob = await res.blob();
        await openFromBlob(blob, {
          kind: "asset",
          assetId: asset.id,
          name: asset.displayName ?? asset.id,
        });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not load asset" });
      }
    },
    [imageAssets, openFromBlob],
  );

  const openFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) {
        setStatus({ loading: false, error: "Please choose an image file" });
        return;
      }
      openFromBlob(file, { kind: "upload", name: file.name });
    },
    [openFromBlob],
  );

  // Start a working-image session on a fresh blank (white) canvas (sc-6092). It
  // reuses the same session model as Open, then jumps into the box tool — the
  // point of a blank layout is to draw boxes and generate from them.
  const newBlankLayout = useCallback(
    async ({ width, height }) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) {
        setStatus({ loading: false, error: "Could not create the canvas." });
        return;
      }
      await openFromBlob(blob, { kind: "blank", name: "Untitled layout" });
      setTool("boxes");
    },
    [openFromBlob],
  );

  async function createBlankLayout() {
    if (!confirmDiscardEdits()) return;
    setNewLayoutOpen(false);
    await newBlankLayout(blankCanvasDims(layoutAspect, layoutSize));
  }

  function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file && confirmDiscardEdits()) openFile(file);
  }

  function handleWheel(event) {
    event.evt.preventDefault();
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer) return;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * (event.evt.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP), MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (pointer.x - view.x) / oldScale, y: (pointer.y - view.y) / oldScale };
    setView({ scale: newScale, x: pointer.x - mouseTo.x * newScale, y: pointer.y - mouseTo.y * newScale });
  }

  function zoomAtCenter(factor) {
    const cx = stageSize.width / 2;
    const cy = stageSize.height / 2;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * factor, MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (cx - view.x) / oldScale, y: (cy - view.y) / oldScale };
    setView({ scale: newScale, x: cx - mouseTo.x * newScale, y: cy - mouseTo.y * newScale });
  }

  function actualSize() {
    if (!working) return;
    setView({
      scale: 1,
      x: (stageSize.width - working.width) / 2,
      y: (stageSize.height - working.height) / 2,
    });
  }

  // ── Crop ────────────────────────────────────────────────────────────────
  function startCrop() {
    if (!working) return;
    setTool("crop");
    setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, rotated)));
  }

  function cancelCrop() {
    setTool("move");
    setCropRect(null);
    setColorAdjust(IDENTITY_COLOR_ADJUST); // discard any unbaked color preview
  }

  function chooseRatio(key) {
    setRatioKey(key);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(key, rotated)));
  }

  function toggleRotate() {
    const next = !rotated;
    setRotated(next);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, next)));
  }

  function clampCropToImage(rect) {
    const width = clamp(rect.width, MIN_CROP_PX, working.width);
    const height = clamp(rect.height, MIN_CROP_PX, working.height);
    return {
      width,
      height,
      x: clamp(rect.x, 0, working.width - width),
      y: clamp(rect.y, 0, working.height - height),
    };
  }

  function handleCropDragEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({ ...cropRect, x: node.x(), y: node.y() });
    node.position({ x: next.x, y: next.y });
    setCropRect(next);
  }

  function handleCropTransformEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({
      x: node.x(),
      y: node.y(),
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY(),
    });
    node.scaleX(1);
    node.scaleY(1);
    node.setAttrs(next);
    setCropRect(next);
  }

  // Apply: rasterize the selected region into a fresh working image. The source
  // bitmap is blob-backed (never tainted), so reading pixels back is safe. The
  // result keeps the same source provenance so lineage survives to Save (sc-2434).
  const applyCrop = useCallback(async () => {
    if (!working || !cropRect) return;
    const sx = clamp(Math.round(cropRect.x), 0, working.width - 1);
    const sy = clamp(Math.round(cropRect.y), 0, working.height - 1);
    const sw = clamp(Math.round(cropRect.width), 1, working.width - sx);
    const sh = clamp(Math.round(cropRect.height), 1, working.height - sy);
    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    canvas.getContext("2d").drawImage(working.image, sx, sy, sw, sh, 0, 0, sw, sh);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    const { image, objectUrl } = await blobToImage(blob);
    installWorkingImage(image, objectUrl, working.source);
    setEdits((prev) => [...prev, { op: "crop", width: sw, height: sh }]);
    setDirty(true);
  }, [working, cropRect, installWorkingImage]);

  // Bind the transformer to the crop rect whenever crop mode is active.
  useEffect(() => {
    const transformer = transformerRef.current;
    const node = cropRectRef.current;
    if (tool === "crop" && transformer && node) {
      transformer.nodes([node]);
      transformer.getLayer()?.batchDraw();
    }
  }, [tool, cropRect]);

  // ── Color grade (sc-2439) ─────────────────────────────────────────────────
  function startColorGrade() {
    if (!working) return;
    setTool("color");
    setColorAdjust(IDENTITY_COLOR_ADJUST);
  }

  const setAdjustValue = (key, value) => setColorAdjust((prev) => ({ ...prev, [key]: value }));
  const resetAdjust = (key) => setAdjustValue(key, 0);
  const resetAllAdjust = () => setColorAdjust(IDENTITY_COLOR_ADJUST);

  // Live preview: Konva applies filters only on a cached node, and re-running them
  // needs a re-cache. Cache the image node (re-caching when the grade changes) while
  // the color tool is active with a non-identity grade; clear it otherwise so Move/
  // other tools see the untouched bitmap. The filter reads the `colorAdjust` attr.
  useEffect(() => {
    const node = imageNodeRef.current;
    if (!node) return;
    const active = tool === "color" && !isIdentityAdjust(colorAdjust);
    if (active) {
      node.cache();
    } else {
      node.clearCache();
    }
    node.getLayer()?.batchDraw();
  }, [tool, colorAdjust, working]);

  // Apply: bake the grade into a fresh working image using the SAME pixel math as the
  // preview (a 2D-canvas pass, no Konva-cache readback). Keeps the source provenance
  // so lineage survives to Save; records the grade in the edit chain.
  const applyColorGrade = useCallback(async () => {
    if (!working || isIdentityAdjust(colorAdjust)) return;
    const canvas = document.createElement("canvas");
    canvas.width = working.width;
    canvas.height = working.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(working.image, 0, 0);
    const imageData = ctx.getImageData(0, 0, working.width, working.height);
    applyColorAdjustments(imageData.data, colorAdjust);
    ctx.putImageData(imageData, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    const baked = { ...colorAdjust };
    const { image, objectUrl } = await blobToImage(blob);
    installWorkingImage(image, objectUrl, working.source);
    setEdits((prev) => [...prev, { op: "color", ...baked }]);
    setDirty(true);
  }, [working, colorAdjust, installWorkingImage]);

  // ── Box layout tool (sc-6090) ─────────────────────────────────────────────
  function selectBoxTool() {
    if (working) setTool("boxes");
  }

  const nextBoxId = () => `box_${(boxIdRef.current += 1)}`;

  // Konva node registry so the transformer can bind to the selected box; the ref
  // callback removes a node when its box unmounts (tool switch / delete).
  const registerBoxNode = (id, node) => {
    if (node) boxNodeRefs.current.set(id, node);
    else boxNodeRefs.current.delete(id);
  };

  function boxPointerDown(event) {
    if (tool !== "boxes" || !working) return;
    // Only a click on the canvas background starts a new box — clicks on an
    // existing box (select/drag) or a transformer handle (resize) are left alone.
    const stage = event.target.getStage();
    const name = event.target?.name?.() ?? "";
    const onBackground = event.target === stage || name === "editor-image" || name === "editor-bg";
    if (!onBackground) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    boxDrawingRef.current = true;
    boxStartRef.current = pt;
    setSelectedBoxId(null);
    setBoxDraft({ x: pt.x, y: pt.y, width: 0, height: 0 });
  }

  function boxPointerMove(event) {
    if (!boxDrawingRef.current) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    setBoxDraft(rectFromPoints(boxStartRef.current, pt));
  }

  function boxPointerUp() {
    if (!boxDrawingRef.current) return;
    boxDrawingRef.current = false;
    const draft = boxDraft;
    setBoxDraft(null);
    // Discard a click / sub-minimum smudge; otherwise commit a new colored box.
    if (!draft || draft.width < MIN_BOX_PX || draft.height < MIN_BOX_PX) return;
    const rect = clampRectToCanvas(draft, working.width, working.height);
    const id = nextBoxId();
    setBoxes((prev) => [...prev, makeBox(id, rect, boxColor)]);
    setSelectedBoxId(id);
  }

  const updateBoxRect = (id, rect) =>
    setBoxes((prev) => prev.map((box) => (box.id === id ? { ...box, rect } : box)));

  // Patch a box's metadata (sc-6091): type / desc / text / colorPalette.
  const updateBox = (id, patch) =>
    setBoxes((prev) => prev.map((box) => (box.id === id ? { ...box, ...patch } : box)));

  function handleBoxDragEnd(id, event) {
    const node = event.target;
    const rect = clampRectToCanvas(
      { x: node.x(), y: node.y(), width: node.width(), height: node.height() },
      working.width,
      working.height,
    );
    node.setAttrs(rect);
    updateBoxRect(id, rect);
  }

  function handleBoxTransformEnd(id, event) {
    const node = event.target;
    const rect = clampRectToCanvas(
      { x: node.x(), y: node.y(), width: node.width() * node.scaleX(), height: node.height() * node.scaleY() },
      working.width,
      working.height,
    );
    node.scaleX(1);
    node.scaleY(1);
    node.setAttrs(rect);
    updateBoxRect(id, rect);
  }

  // Selecting a palette color sets the color for new boxes and recolors the
  // selected box (the palette acts on the active box). Stored uppercase so the
  // box stays valid per `isValidHexColor` even from a lowercase <input type=color>.
  function chooseBoxColor(color) {
    const value = color.toUpperCase();
    setBoxColor(value);
    if (selectedBoxId) {
      setBoxes((prev) => prev.map((box) => (box.id === selectedBoxId ? { ...box, color: value } : box)));
    }
  }

  function deleteBox(id) {
    if (!id) return;
    setBoxes((prev) => prev.filter((box) => box.id !== id));
    boxNodeRefs.current.delete(id);
    setSelectedBoxId((cur) => (cur === id ? null : cur));
  }

  function clearBoxes() {
    setBoxes([]);
    boxNodeRefs.current.clear();
    setSelectedBoxId(null);
    setBoxDraft(null);
  }

  // Rasterize the working image + the colored boxes into one PNG File (sc-6093).
  // This is an ephemeral pass-through reference — staged as scratch, never saved
  // to the Library — that the edit model reads as color-keyed regions.
  function bakeBoxesToFile() {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement("canvas");
      canvas.width = working.width;
      canvas.height = working.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(working.image, 0, 0);
      paintBoxesOnContext(ctx, boxes);
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Could not bake the boxes."));
          return;
        }
        resolve(new File([blob], "boxed.png", { type: "image/png" }));
      }, "image/png");
    });
  }

  // Bake the boxes and run them through the existing edit_image flow on the chosen
  // edit model (sc-6093). The baked PNG is the pass-through source; runAiOp stages
  // it as scratch and purges it with the result, so it never lands in the Library.
  async function runBoxEdit() {
    if (!boxes.length || !editModel || !working || aiOp) return;
    const prompt = editPrompt.trim();
    let sourceFile;
    try {
      sourceFile = await bakeBoxesToFile();
    } catch (err) {
      setStatus({ loading: false, error: `Could not bake boxes: ${err.message || err}` });
      return;
    }
    runAiOp({
      label: "edit",
      endpoint: "/api/v1/image/jobs",
      edit: { op: "boxLayout", model: editModel, prompt, boxes: boxes.length },
      sourceFile,
      buildBody: (scratch) =>
        buildEditJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          model: editModel,
          prompt,
          seed: editSeed,
          width: working.width,
          height: working.height,
          fitMode: "crop",
        }),
    });
  }

  // The stage's pointer events drive both the mask brush (edit tool) and box
  // drawing (boxes tool); each handler no-ops unless its tool/mode is active.
  function handleStagePointerDown(event) {
    maskPointerDown(event);
    boxPointerDown(event);
  }
  function handleStagePointerMove(event) {
    maskPointerMove(event);
    boxPointerMove(event);
  }
  function handleStagePointerUp(event) {
    maskPointerUp(event);
    boxPointerUp(event);
  }

  // Bind the transformer to the selected box whenever the box tool is active.
  useEffect(() => {
    const transformer = boxTransformerRef.current;
    if (tool !== "boxes" || !transformer) return;
    const node = selectedBoxId ? boxNodeRefs.current.get(selectedBoxId) : null;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
  }, [tool, selectedBoxId, boxes]);

  // ── Inpaint mask brush (sc-2436) ──────────────────────────────────────────
  // Pointer position in image-pixel coords (undo the stage pan/zoom), clamped.
  function stagePointToImage(event) {
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer || !working) return null;
    return {
      x: clamp((pointer.x - view.x) / view.scale, 0, working.width),
      y: clamp((pointer.y - view.y) / view.scale, 0, working.height),
    };
  }

  function maskPointerDown(event) {
    if (!maskMode || !working) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    maskPaintingRef.current = true;
    setMaskLines((prev) => [...prev, { points: [pt.x, pt.y], size: maskBrush, erase: maskErase }]);
  }

  function maskPointerMove(event) {
    if (!maskMode || !maskPaintingRef.current) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    setMaskLines((prev) => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      return [...prev.slice(0, -1), { ...last, points: [...last.points, pt.x, pt.y] }];
    });
  }

  function maskPointerUp() {
    maskPaintingRef.current = false;
  }

  function clearMask() {
    setMaskLines([]);
  }

  // Rasterize the brush strokes to a mask PNG File aligned to the working bitmap:
  // white = edit region on black. Erase strokes punch holes (destination-out on a
  // transparent scratch), then it's flattened onto black so the worker's convert("L")
  // reads white-on-black. Mirrors the same compositing as the on-canvas preview.
  function rasterizeMaskToFile() {
    return new Promise((resolve, reject) => {
      const scratch = document.createElement("canvas");
      scratch.width = working.width;
      scratch.height = working.height;
      const sctx = scratch.getContext("2d");
      sctx.lineCap = "round";
      sctx.lineJoin = "round";
      sctx.strokeStyle = "#ffffff";
      sctx.fillStyle = "#ffffff";
      for (const line of maskLines) {
        sctx.globalCompositeOperation = line.erase ? "destination-out" : "source-over";
        sctx.lineWidth = line.size;
        const p = line.points;
        if (p.length === 2) {
          sctx.beginPath();
          sctx.arc(p[0], p[1], line.size / 2, 0, Math.PI * 2);
          sctx.fill();
          continue;
        }
        sctx.beginPath();
        sctx.moveTo(p[0], p[1]);
        for (let i = 2; i < p.length; i += 2) sctx.lineTo(p[i], p[i + 1]);
        sctx.stroke();
      }
      // Flatten onto black so erased/holes read as black (= keep).
      const out = document.createElement("canvas");
      out.width = working.width;
      out.height = working.height;
      const octx = out.getContext("2d");
      octx.fillStyle = "#000000";
      octx.fillRect(0, 0, out.width, out.height);
      octx.drawImage(scratch, 0, 0);
      out.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Could not encode the mask."));
          return;
        }
        resolve(new File([blob], "mask.png", { type: "image/png" }));
      }, "image/png");
    });
  }

  // ── AI ops on the working image (sc-2432 seam) ────────────────────────────
  // Rasterize the current working image to a PNG File. `filename` overrides the
  // name (Save/Download use the "-edited" name; the AI-op scratch upload doesn't care).
  const workingImageToFile = useCallback(
    (filename) => {
      return new Promise((resolve, reject) => {
        if (!working) {
          reject(new Error("No working image."));
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = working.width;
        canvas.height = working.height;
        canvas.getContext("2d").drawImage(working.image, 0, 0);
        const base = (working.source.name || "image").replace(/\.[^./\\]+$/, "");
        const name = filename || `${base}.png`;
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error("Could not encode the working image."));
            return;
          }
          resolve(new File([blob], name, { type: "image/png" }));
        }, "image/png");
      });
    },
    [working],
  );

  // Stage the working image as a scratch asset, start a worker job against it, and
  // track it. The watcher below loads the result back and purges scratch + result —
  // intermediates never persist; only Save (sc-2434) lands a Library asset.
  const runAiOp = useCallback(
    async ({ buildBody, label, edit, endpoint = "/api/v1/jobs", maskFile = null, sourceFile = null }) => {
      if (!working || aiOp || !activeProject) return;
      setStatus({ loading: false, error: "" });
      // Stage the source (and, for a masked edit, the mask) as scratch assets. The
      // source defaults to the working bitmap, but callers can pass a derived PNG —
      // e.g. the box-baked pass-through (sc-6093) — to edit that instead.
      let scratch;
      let maskScratch = null;
      try {
        scratch = await importAsset(sourceFile ?? (await workingImageToFile()), { throwOnError: true });
        if (maskFile) maskScratch = await importAsset(maskFile, { throwOnError: true });
      } catch (err) {
        if (scratch) purgeAsset(scratch).catch(() => {});
        setStatus({ loading: false, error: `Could not stage image: ${err.message || err}` });
        return;
      }
      try {
        const job = await apiFetch(endpoint, token, {
          method: "POST",
          body: JSON.stringify(buildBody(scratch, maskScratch)),
        });
        if (!job?.id) throw new Error("The job was not created.");
        setAiOp({ jobId: job.id, scratch, maskScratch, source: working.source, label, edit });
        setTool("move");
      } catch (err) {
        purgeAsset(scratch).catch(() => {});
        if (maskScratch) purgeAsset(maskScratch).catch(() => {});
        setStatus({ loading: false, error: `Could not start ${label}: ${err.message || err}` });
      }
    },
    [working, aiOp, activeProject, workingImageToFile, importAsset, token, purgeAsset],
  );

  function runUpscale() {
    const valid = upscaleFactorsForEngine(upscaleEngine);
    const factor = valid.includes(upscaleFactor) ? upscaleFactor : valid[0];
    const softness = upscaleEngineHasSoftness(upscaleEngine) ? upscaleSoftness : undefined;
    runAiOp({
      label: "upscale",
      edit: {
        op: "upscale",
        engine: upscaleEngine,
        factor,
        ...(softness !== undefined ? { softness } : {}),
      },
      buildBody: (scratch) =>
        buildUpscaleJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          factor,
          engine: upscaleEngine,
          displayName: working?.source?.name,
          softness,
        }),
    });
  }

  function runDetail() {
    if (!detailModel) return;
    runAiOp({
      label: "detail",
      edit: { op: "detail", model: detailModel, strength: detailStrength, cnScale: detailCnScale },
      buildBody: (scratch) =>
        buildDetailJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          model: detailModel,
          strength: detailStrength,
          cnScale: detailCnScale,
          displayName: working?.source?.name,
        }),
    });
  }

  async function runEdit() {
    const prompt = editPrompt.trim();
    if (!prompt || !editModel || !working) return;
    // Canvas-extend / outpaint (sc-2556): resolve the output W×H from the chosen aspect
    // and fit mode (outpaint coerced away when the model can't inpaint). "match" keeps
    // the working size, so the existing same-size edit behavior is unchanged.
    const fitMode = effectiveFitMode(editFitMode, canMask);
    const { width: outWidth, height: outHeight } = editOutputDims(working.width, working.height, editAspect, fitMode);
    // A painted mask is sent only for inpaint-capable models; otherwise it's a
    // whole-image edit (the mask stays as a local guide but isn't uploaded).
    const masked = canMask && maskHasContent(maskLines);
    let maskFile = null;
    if (masked) {
      try {
        maskFile = await rasterizeMaskToFile();
      } catch (err) {
        setStatus({ loading: false, error: `Could not prepare the mask: ${err.message || err}` });
        return;
      }
    }
    runAiOp({
      label: "edit",
      endpoint: "/api/v1/image/jobs",
      edit: { op: "edit", model: editModel, prompt, ...(masked ? { masked: true } : {}) },
      maskFile,
      buildBody: (scratch, maskScratch) =>
        buildEditJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          maskAssetId: maskScratch?.id,
          model: editModel,
          prompt,
          seed: editSeed,
          width: outWidth,
          height: outHeight,
          fitMode,
        }),
    });
  }

  // When the in-flight op's job terminates, load the result back into the working
  // image (on success) and purge the ephemeral scratch + result assets.
  useEffect(() => {
    if (!aiOp?.jobId) return;
    const job = jobs?.find((item) => item.id === aiOp.jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    const { scratch, maskScratch, source, edit } = aiOp;
    setAiOp(null); // stop tracking immediately so this can't re-enter on the next jobs tick
    const resultAsset = job.status === "completed" ? job.result?.assets?.[0] ?? null : null;
    (async () => {
      try {
        if (resultAsset) {
          const res = await fetch(assetUrl(resultAsset));
          if (!res.ok) throw new Error(`Failed to load result (${res.status})`);
          const { image, objectUrl } = await blobToImage(await res.blob());
          installWorkingImage(image, objectUrl, source);
          if (edit) setEdits((prev) => [...prev, edit]);
          setDirty(true);
        } else {
          setStatus({ loading: false, error: job.error ?? job.message ?? "The operation failed." });
        }
      } catch (err) {
        setStatus({ loading: false, error: err.message || "The operation failed." });
      } finally {
        if (scratch) purgeAsset(scratch).catch(() => {});
        if (maskScratch) purgeAsset(maskScratch).catch(() => {});
        if (resultAsset) purgeAsset(resultAsset).catch(() => {});
      }
    })();
  }, [aiOp, jobs, installWorkingImage, purgeAsset]);

  // ── Save / export (sc-2434) ───────────────────────────────────────────────
  // Persist the working image as a NEW Library asset, never overwriting the
  // source. Lineage links it back to the asset it was opened from (uploads have
  // no source to link); the edit chain rides along as provenance.
  const runSave = useCallback(async () => {
    if (!working || saving) return;
    setSaving(true);
    setStatus({ loading: false, error: "" });
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const saved = await importAsset(file, {
        throwOnError: true,
        sourceAssetId: working.source.assetId,
        provenance: buildSaveProvenance({
          source: working.source,
          edits,
          width: working.width,
          height: working.height,
        }),
      });
      setSavedAssetId(saved?.id ?? null);
      setDirty(false);
    } catch (err) {
      setStatus({ loading: false, error: `Could not save: ${err.message || err}` });
    } finally {
      setSaving(false);
    }
  }, [working, saving, workingImageToFile, importAsset, edits]);

  // Export the working image straight to disk as a PNG (no project involvement).
  const runDownload = useCallback(async () => {
    if (!working) return;
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const url = URL.createObjectURL(file);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = file.name;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setStatus({ loading: false, error: `Could not export: ${err.message || err}` });
    }
  }, [working, workingImageToFile]);

  // Confirm before an action that would discard unsaved edits (Open / drag-drop a
  // new image while dirty). Returns true when it's safe to proceed.
  function confirmDiscardEdits() {
    if (!dirty) return true;
    return (
      typeof window.confirm !== "function" ||
      window.confirm("You have unsaved edits. Open a new image and discard them?")
    );
  }

  // Warn before leaving with unsaved edits: a browser unload (close/refresh) and an
  // in-app navigation away (the App nav consults this guard, sc-2434).
  useEffect(() => {
    if (!dirty) return undefined;
    const onBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    const unregister = registerLeaveGuard?.(
      () =>
        typeof window.confirm !== "function" ||
        window.confirm("You have unsaved edits in the Image Editor. Leave and discard them?"),
    );
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      if (typeof unregister === "function") unregister();
    };
  }, [dirty, registerLeaveGuard]);

  const activeAiJob = aiOp ? jobs?.find((item) => item.id === aiOp.jobId) : null;

  // The box currently selected for metadata editing (sc-6091), and what it still
  // needs to be a valid Ideogram element (surfaced as a hint, not a hard block).
  const selectedBox = selectedBoxId ? boxes.find((box) => box.id === selectedBoxId) ?? null : null;
  const selectedBoxGaps = boxMetadataGaps(selectedBox);

  // Live W×H preview for the New-layout modal (sc-6092).
  const layoutDims = blankCanvasDims(layoutAspect, layoutSize);

  // The auto-composed color-keyed prompt from the current boxes (sc-6094). Used to
  // pre-fill the prompt field on demand; "" when no box is describable yet.
  const composedPrompt = composeColorPrompt(boxes);

  return (
    <section className="main-surface image-editor-surface">
      <div className="image-editor-bar">
        <span className="image-editor-title" title={working ? working.source.name : undefined}>
          {working ? working.source.name : "No image open"}
        </span>
        <div className="image-editor-bar-actions">
          <button className={working ? "" : "primary"} onClick={() => setPickerOpen(true)} type="button">
            Open
          </button>
          <button onClick={() => setNewLayoutOpen(true)} title="Start a blank canvas for box layout" type="button">
            New layout
          </button>
          {working && working.source.assetId ? (
            <button
              onClick={() => setPreviewAsset?.(imageAssets.find((item) => item.id === working.source.assetId))}
              title="Preview the source asset"
              type="button"
            >
              Source
            </button>
          ) : null}
          {working ? (
            <>
              <button onClick={runDownload} title="Download a PNG to your computer" type="button">
                Download
              </button>
              {savedAssetId && !dirty ? <span className="image-editor-saved">Saved ✓</span> : null}
              <button
                className="primary"
                disabled={!dirty || saving}
                onClick={runSave}
                title="Save a new image to the project Library"
                type="button"
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {status.error ? <div className="notice notice-error image-editor-notice">{status.error}</div> : null}

      <div
        className="image-editor-canvas-wrap"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        ref={containerRef}
      >
        {working && stageSize.width > 0 && stageSize.height > 0 ? (
          <Stage
            draggable={tool !== "crop" && tool !== "boxes" && !maskMode}
            height={stageSize.height}
            onDragEnd={(event) => {
              if (event.target !== event.target.getStage()) return;
              const stage = event.target.getStage();
              setView((prev) => ({ ...prev, x: stage.x(), y: stage.y() }));
            }}
            onMouseDown={handleStagePointerDown}
            onMouseMove={handleStagePointerMove}
            onMouseUp={handleStagePointerUp}
            onTouchStart={handleStagePointerDown}
            onTouchMove={handleStagePointerMove}
            onTouchEnd={handleStagePointerUp}
            onWheel={handleWheel}
            scaleX={view.scale}
            scaleY={view.scale}
            width={stageSize.width}
            x={view.x}
            y={view.y}
          >
            <Layer>
              <Rect
                fill="#ffffff"
                height={working.height}
                name="editor-bg"
                shadowBlur={12}
                shadowColor="rgba(0,0,0,0.35)"
                width={working.width}
                x={0}
                y={0}
              />
              <KonvaImage
                colorAdjust={colorAdjust}
                filters={[konvaColorFilter]}
                height={working.height}
                image={working.image}
                name="editor-image"
                ref={imageNodeRef}
                width={working.width}
                x={0}
                y={0}
              />
              {tool === "crop" && cropRect ? (
                <>
                  {cropOverlayRects(working.width, working.height, cropRect).map((rect, index) => (
                    <Rect
                      key={index}
                      fill="rgba(0,0,0,0.55)"
                      height={rect.height}
                      listening={false}
                      width={rect.width}
                      x={rect.x}
                      y={rect.y}
                    />
                  ))}
                  <Rect
                    draggable
                    fill="rgba(255,255,255,0.01)"
                    height={cropRect.height}
                    onDragEnd={handleCropDragEnd}
                    onTransformEnd={handleCropTransformEnd}
                    ref={cropRectRef}
                    stroke="#ffffff"
                    strokeScaleEnabled={false}
                    strokeWidth={2}
                    width={cropRect.width}
                    x={cropRect.x}
                    y={cropRect.y}
                  />
                  <Transformer
                    anchorSize={8}
                    borderStroke="#ffffff"
                    boundBoxFunc={(oldBox, newBox) =>
                      newBox.width < MIN_CROP_PX || newBox.height < MIN_CROP_PX ? oldBox : newBox
                    }
                    enabledAnchors={
                      ratioKey === "free"
                        ? ["top-left", "top-center", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-center", "bottom-right"]
                        : ["top-left", "top-right", "bottom-left", "bottom-right"]
                    }
                    keepRatio={ratioKey !== "free"}
                    ref={transformerRef}
                    rotateEnabled={false}
                  />
                </>
              ) : null}
            </Layer>
            {maskLines.length && canMask ? (
              // Isolated layer so the eraser's destination-out clears only the mask
              // overlay, never the image beneath it.
              <Layer listening={false}>
                {maskLines.map((line, index) => (
                  <Line
                    globalCompositeOperation={line.erase ? "destination-out" : "source-over"}
                    key={index}
                    lineCap="round"
                    lineJoin="round"
                    points={line.points}
                    stroke="rgba(255,40,120,0.5)"
                    strokeWidth={line.size}
                  />
                ))}
              </Layer>
            ) : null}
            {tool === "boxes" ? (
              // Box layout overlay (sc-6090): colored rects + a transformer on the
              // selected box + the dashed live-draw preview. Image-pixel coords, so
              // it pans/zooms with the canvas like the crop rect and mask.
              <Layer>
                {boxes.map((box) => (
                  <Rect
                    draggable
                    fill={boxFillStyle(box.color, 0.18)}
                    height={box.rect.height}
                    key={box.id}
                    name="layout-box"
                    onClick={() => setSelectedBoxId(box.id)}
                    onDragEnd={(event) => handleBoxDragEnd(box.id, event)}
                    onMouseDown={() => setSelectedBoxId(box.id)}
                    onTap={() => setSelectedBoxId(box.id)}
                    onTransformEnd={(event) => handleBoxTransformEnd(box.id, event)}
                    ref={(node) => registerBoxNode(box.id, node)}
                    stroke={box.color}
                    strokeScaleEnabled={false}
                    strokeWidth={selectedBoxId === box.id ? 3 : 2}
                    width={box.rect.width}
                    x={box.rect.x}
                    y={box.rect.y}
                  />
                ))}
                {boxDraft ? (
                  <Rect
                    dash={[6, 4]}
                    fill={boxFillStyle(boxColor, 0.18)}
                    height={boxDraft.height}
                    listening={false}
                    stroke={boxColor}
                    strokeScaleEnabled={false}
                    strokeWidth={2}
                    width={boxDraft.width}
                    x={boxDraft.x}
                    y={boxDraft.y}
                  />
                ) : null}
                <Transformer
                  anchorSize={8}
                  borderStroke="#ffffff"
                  boundBoxFunc={(oldBox, newBox) =>
                    newBox.width < MIN_BOX_PX || newBox.height < MIN_BOX_PX ? oldBox : newBox
                  }
                  ref={boxTransformerRef}
                  rotateEnabled={false}
                />
              </Layer>
            ) : null}
          </Stage>
        ) : (
          <div className="image-editor-empty">
            {status.loading ? (
              <p>Loading image…</p>
            ) : (
              <>
                <p className="image-editor-empty-title">Open an image to start editing</p>
                <p className="image-editor-empty-hint">Drag &amp; drop an image here, or click Open.</p>
                <p className="image-editor-empty-hint">
                  Or{" "}
                  <button className="image-editor-linkbtn" onClick={() => setNewLayoutOpen(true)} type="button">
                    start a blank layout
                  </button>{" "}
                  to compose with boxes.
                </p>
              </>
            )}
          </div>
        )}

        {working ? (
          <aside className="image-editor-toolbar" aria-label="Editor tools">
            <button
              className={tool === "move" ? "image-editor-tool active" : "image-editor-tool"}
              onClick={cancelCrop}
              title="Move / pan"
              type="button"
            >
              Move
            </button>
            <button
              className={tool === "crop" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={startCrop}
              title="Crop"
              type="button"
            >
              Crop
            </button>
            <button
              className={tool === "upscale" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp || Boolean(macUpscaleBlock)}
              onClick={() => setTool("upscale")}
              title={macUpscaleBlock ? macUpscaleBlock.text : "Upscale"}
              type="button"
            >
              Upscale
            </button>
            <button
              className={tool === "detail" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp || detailModels.length === 0}
              onClick={() => setTool("detail")}
              title="Detail enhance (tile-ControlNet refine)"
              type="button"
            >
              Detail
            </button>
            <button
              className={tool === "color" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={startColorGrade}
              title="Color grade"
              type="button"
            >
              Color
            </button>
            <button
              className={tool === "edit" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={() => setTool("edit")}
              title="AI prompt edit"
              type="button"
            >
              AI Edit
            </button>
            <button
              className={tool === "boxes" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={selectBoxTool}
              title="Box layout — draw colored regions (color-keyed edit / Ideogram bbox)"
              type="button"
            >
              Boxes
            </button>
            {UPCOMING_TOOLS.map((upcoming) => (
              <button
                className="image-editor-tool"
                disabled
                key={upcoming.id}
                title={`${upcoming.label} — coming soon (${upcoming.story})`}
                type="button"
              >
                {upcoming.label}
              </button>
            ))}
          </aside>
        ) : null}

        {tool === "crop" && cropRect ? (
          <div className="image-editor-cropbar">
            <div className="image-editor-ratios" role="group" aria-label="Crop ratio">
              {CROP_RATIOS.map((entry) => (
                <button
                  className={ratioKey === entry.key ? "active" : ""}
                  key={entry.key}
                  onClick={() => chooseRatio(entry.key)}
                  type="button"
                >
                  {entry.label}
                </button>
              ))}
            </div>
            <button
              className={rotated ? "active" : ""}
              disabled={ratioKey === "free" || ratioKey === "1:1"}
              onClick={toggleRotate}
              title="Rotate ratio (swap orientation)"
              type="button"
            >
              ⟲ Rotate
            </button>
            <span className="image-editor-cropdims">
              {Math.round(cropRect.width)} × {Math.round(cropRect.height)}
            </span>
            <button className="primary" onClick={applyCrop} type="button">
              Apply
            </button>
            <button onClick={cancelCrop} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "upscale" && working ? (
          <div className="image-editor-cropbar">
            <div className="image-editor-ratios" role="group" aria-label="Upscale engine">
              {availableUpscaleEngines.map((entry) => (
                <button
                  className={upscaleEngine === entry.key ? "active" : ""}
                  key={entry.key}
                  onClick={() => {
                    setUpscaleEngine(entry.key);
                    if (!entry.factors.includes(upscaleFactor)) setUpscaleFactor(entry.factors[0]);
                  }}
                  type="button"
                >
                  {entry.label}
                </button>
              ))}
            </div>
            <div className="image-editor-ratios" role="group" aria-label="Upscale factor">
              {upscaleFactorsForEngine(upscaleEngine).map((value) => (
                <button
                  className={upscaleFactor === value ? "active" : ""}
                  key={value}
                  onClick={() => setUpscaleFactor(value)}
                  type="button"
                >
                  {value}×
                </button>
              ))}
            </div>
            {upscaleEngineHasSoftness(upscaleEngine) ? (
              <label className="image-editor-upscale-softness" title="Higher restores more detail from a degraded source; 0 keeps it faithful.">
                Detail
                <input
                  aria-label="SeedVR2 detail (softness)"
                  max="1"
                  min="0"
                  onChange={(event) => setUpscaleSoftness(Number(event.target.value))}
                  step="0.05"
                  type="range"
                  value={upscaleSoftness}
                />
                <span>{upscaleSoftness.toFixed(2)}</span>
              </label>
            ) : null}
            <span className="image-editor-cropdims">
              {working.width * upscaleFactor} × {working.height * upscaleFactor}
            </span>
            <button className="primary" disabled={!!aiOp} onClick={runUpscale} type="button">
              Upscale
            </button>
            <button onClick={() => setTool("move")} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "detail" && working ? (
          <div className="image-editor-cropbar image-editor-detailbar">
            {detailModels.length === 0 ? (
              <span className="image-editor-cropdims">No detail-capable models installed</span>
            ) : (
              <>
                <select
                  aria-label="Detail backbone"
                  className="image-editor-editmodel"
                  onChange={(event) => setDetailModel(event.target.value)}
                  value={detailModel}
                >
                  {detailModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label ?? model.id}
                    </option>
                  ))}
                </select>
                <label className="image-editor-slider" title="Detail amount — higher invents more texture">
                  <span className="image-editor-slider-label">Detail</span>
                  <input
                    aria-label="Detail strength"
                    max={0.8}
                    min={0.3}
                    onChange={(event) => setDetailStrength(Number(event.target.value))}
                    step={0.05}
                    type="range"
                    value={detailStrength}
                  />
                  <span className="image-editor-slider-value">{Math.round(detailStrength * 100)}</span>
                </label>
                <label className="image-editor-slider" title="Structure lock — higher keeps the result closer to the source">
                  <span className="image-editor-slider-label">Structure</span>
                  <input
                    aria-label="Structure lock"
                    max={1}
                    min={0.4}
                    onChange={(event) => setDetailCnScale(Number(event.target.value))}
                    step={0.05}
                    type="range"
                    value={detailCnScale}
                  />
                  <span className="image-editor-slider-value">{Math.round(detailCnScale * 100)}</span>
                </label>
                <button className="primary" disabled={!!aiOp || !detailModel} onClick={runDetail} type="button">
                  Enhance
                </button>
              </>
            )}
            <button onClick={() => setTool("move")} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "color" && working ? (
          <div className="image-editor-cropbar image-editor-colorbar">
            {COLOR_ADJUSTMENTS.map(({ key, label }) => (
              <label className="image-editor-slider" key={key} title="Double-click the slider to reset">
                <span className="image-editor-slider-label">{label}</span>
                <input
                  aria-label={label}
                  max={1}
                  min={-1}
                  onChange={(event) => setAdjustValue(key, Number(event.target.value))}
                  onDoubleClick={() => resetAdjust(key)}
                  step={0.01}
                  type="range"
                  value={colorAdjust[key]}
                />
                <span className="image-editor-slider-value">{Math.round(colorAdjust[key] * 100)}</span>
              </label>
            ))}
            <button disabled={isIdentityAdjust(colorAdjust)} onClick={resetAllAdjust} type="button">
              Reset
            </button>
            <button className="primary" disabled={isIdentityAdjust(colorAdjust)} onClick={applyColorGrade} type="button">
              Apply
            </button>
            <button onClick={cancelCrop} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "edit" && working ? (
          <div className="image-editor-cropbar image-editor-editbar">
            {editModels.length === 0 ? (
              <span className="image-editor-cropdims">No edit-capable models installed</span>
            ) : (
              <>
                <select
                  aria-label="Edit model"
                  className="image-editor-editmodel"
                  onChange={(event) => setEditModel(event.target.value)}
                  value={editModel}
                >
                  {editModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label ?? model.id}
                    </option>
                  ))}
                </select>
                <input
                  aria-label="Edit prompt"
                  className="image-editor-editprompt"
                  onChange={(event) => setEditPrompt(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && editPrompt.trim() && !aiOp) runEdit();
                  }}
                  placeholder="Describe the edit…"
                  type="text"
                  value={editPrompt}
                />
                <input
                  aria-label="Seed (optional)"
                  className="image-editor-editseed"
                  min={0}
                  onChange={(event) => setEditSeed(event.target.value)}
                  placeholder="Seed"
                  type="number"
                  value={editSeed}
                />
                <select
                  aria-label="Output aspect"
                  className="image-editor-editmodel"
                  onChange={(event) => setEditAspect(event.target.value)}
                  title="Output aspect — extend the canvas and fill the new area"
                  value={editAspect}
                >
                  {EDIT_OUTPUT_ASPECTS.map((aspect) => (
                    <option key={aspect.key} value={aspect.key}>
                      {aspect.label}
                    </option>
                  ))}
                </select>
                {editAspect !== "match" ? (
                  <FitModeControl
                    value={effectiveFitMode(editFitMode, canMask)}
                    onChange={setEditFitMode}
                    inpaintCapable={canMask}
                    label="Fill"
                  />
                ) : null}
                {canMask ? (
                  <>
                    <button
                      className={maskMode ? "active" : ""}
                      onClick={() => setMaskMode((on) => !on)}
                      title="Paint a mask to confine the edit to a region (inpaint)"
                      type="button"
                    >
                      {maskHasContent(maskLines) ? "Mask ✓" : "Mask"}
                    </button>
                    {maskMode ? (
                      <>
                        <label className="image-editor-slider" title="Brush size">
                          <span className="image-editor-slider-label">Brush</span>
                          <input
                            aria-label="Brush size"
                            max={300}
                            min={5}
                            onChange={(event) => setMaskBrush(Number(event.target.value))}
                            step={1}
                            type="range"
                            value={maskBrush}
                          />
                        </label>
                        <button
                          className={maskErase ? "active" : ""}
                          onClick={() => setMaskErase((on) => !on)}
                          title="Eraser"
                          type="button"
                        >
                          Eraser
                        </button>
                        <button disabled={!maskLines.length} onClick={clearMask} type="button">
                          Clear
                        </button>
                      </>
                    ) : null}
                  </>
                ) : null}
                <button className="primary" disabled={!editPrompt.trim() || !!aiOp} onClick={runEdit} type="button">
                  {canMask && maskHasContent(maskLines) ? "Inpaint" : "Edit"}
                </button>
              </>
            )}
            <button
              onClick={() => {
                setTool("move");
                setMaskMode(false);
              }}
              type="button"
            >
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "boxes" && working ? (
          <div className="image-editor-cropbar image-editor-boxbar">
            <div className="image-editor-box-palette" role="group" aria-label="Box color">
              {BOX_PALETTE.map((entry) => (
                <button
                  aria-label={entry.name}
                  aria-pressed={boxColor === entry.value}
                  className={boxColor === entry.value ? "image-editor-swatch active" : "image-editor-swatch"}
                  key={entry.value}
                  onClick={() => chooseBoxColor(entry.value)}
                  style={{ background: entry.value }}
                  title={entry.name}
                  type="button"
                />
              ))}
              <label className="image-editor-swatch image-editor-swatch-custom" title="Custom color">
                <input
                  aria-label="Custom box color"
                  onChange={(event) => chooseBoxColor(event.target.value)}
                  type="color"
                  value={boxColor.toLowerCase()}
                />
              </label>
            </div>
            {boxes.length ? (
              <div className="image-editor-box-list" role="group" aria-label="Boxes">
                {boxes.map((box, index) => {
                  const incomplete = boxMetadataGaps(box).length > 0;
                  return (
                    <button
                      className={`image-editor-box-chip${selectedBoxId === box.id ? " active" : ""}${incomplete ? " incomplete" : ""}`}
                      key={box.id}
                      onClick={() => setSelectedBoxId(box.id)}
                      title={box.desc ? `${index + 1}: ${box.desc}` : `Box ${index + 1} — needs a description`}
                      type="button"
                    >
                      <span className="image-editor-box-dot" style={{ background: box.color }} />
                      {index + 1}
                      {incomplete ? <span className="image-editor-box-chip-flag" aria-hidden="true">!</span> : null}
                    </button>
                  );
                })}
              </div>
            ) : (
              <span className="image-editor-cropdims">Drag on the image to draw a box</span>
            )}
            {boxes.length ? (
              editModels.length ? (
                <>
                  <select
                    aria-label="Box edit model"
                    className="image-editor-editmodel"
                    onChange={(event) => setEditModel(event.target.value)}
                    value={editModel}
                  >
                    {editModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label ?? model.id}
                      </option>
                    ))}
                  </select>
                  <button
                    disabled={!composedPrompt}
                    onClick={() => setEditPrompt(composedPrompt)}
                    title="Compose a prompt from the boxes' colors + descriptions (editable)"
                    type="button"
                  >
                    Auto prompt
                  </button>
                  <input
                    aria-label="Box edit prompt"
                    className="image-editor-editprompt"
                    onChange={(event) => setEditPrompt(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !aiOp && editModel) runBoxEdit();
                    }}
                    placeholder="Describe the edit (e.g. replace the red region with…)"
                    type="text"
                    value={editPrompt}
                  />
                  <button className="primary" disabled={!!aiOp || !editModel} onClick={runBoxEdit} type="button">
                    Generate
                  </button>
                </>
              ) : (
                <span className="image-editor-cropdims">No edit-capable models installed</span>
              )
            ) : null}
            <button disabled={!selectedBoxId} onClick={() => deleteBox(selectedBoxId)} type="button">
              Delete
            </button>
            <button disabled={!boxes.length} onClick={clearBoxes} type="button">
              Clear
            </button>
            <button onClick={() => setTool("move")} type="button">
              Cancel
            </button>
          </div>
        ) : null}

        {tool === "boxes" && selectedBox ? (
          <div className="image-editor-boxmeta" aria-label="Box details">
            <div className="image-editor-boxmeta-title">
              <span className="image-editor-box-dot" style={{ background: selectedBox.color }} />
              Box {boxes.indexOf(selectedBox) + 1}
            </div>
            <div className="image-editor-boxmeta-types" role="group" aria-label="Element type">
              <button
                className={selectedBox.type === "obj" ? "active" : ""}
                onClick={() => updateBox(selectedBox.id, { type: "obj" })}
                type="button"
              >
                Object
              </button>
              <button
                className={selectedBox.type === "text" ? "active" : ""}
                onClick={() => updateBox(selectedBox.id, { type: "text" })}
                type="button"
              >
                Text
              </button>
            </div>
            <label className="image-editor-boxmeta-field">
              <span>Description</span>
              <input
                aria-label="Box description"
                onChange={(event) => updateBox(selectedBox.id, { desc: event.target.value })}
                placeholder="What is in this region?"
                type="text"
                value={selectedBox.desc ?? ""}
              />
            </label>
            {selectedBox.type === "text" ? (
              <label className="image-editor-boxmeta-field">
                <span>Text</span>
                <input
                  aria-label="Literal text"
                  onChange={(event) => updateBox(selectedBox.id, { text: event.target.value })}
                  placeholder="Literal text to render"
                  type="text"
                  value={selectedBox.text ?? ""}
                />
              </label>
            ) : null}
            <div className="image-editor-boxmeta-field">
              <span>
                Element colors ({(selectedBox.colorPalette ?? []).length}/{MAX_BOX_PALETTE})
              </span>
              <div className="image-editor-box-palette">
                {(selectedBox.colorPalette ?? []).map((color) => (
                  <button
                    aria-label={`Remove ${color}`}
                    className="image-editor-swatch"
                    key={color}
                    onClick={() =>
                      updateBox(selectedBox.id, { colorPalette: removePaletteColor(selectedBox.colorPalette, color) })
                    }
                    style={{ background: color }}
                    title={`Remove ${color}`}
                    type="button"
                  />
                ))}
                {(selectedBox.colorPalette ?? []).length < MAX_BOX_PALETTE ? (
                  <label className="image-editor-swatch image-editor-swatch-custom" title="Add color">
                    <input
                      aria-label="Add element color"
                      onChange={(event) =>
                        updateBox(selectedBox.id, {
                          colorPalette: addPaletteColor(selectedBox.colorPalette, event.target.value),
                        })
                      }
                      type="color"
                    />
                  </label>
                ) : null}
              </div>
            </div>
            {selectedBoxGaps.length ? (
              <p className="image-editor-boxmeta-hint">
                For Ideogram layout this box still needs {selectedBoxGaps.join(", ")}. The color-keyed edit path only
                needs a color + description.
              </p>
            ) : (
              <p className="image-editor-boxmeta-ready">Ready for Ideogram layout ✓</p>
            )}
          </div>
        ) : null}

        {aiOp ? (
          <div className="image-editor-busy">
            <div className="image-editor-busy-card">
              <p className="image-editor-busy-title">
                {aiOp.label === "upscale"
                  ? "Upscaling…"
                  : aiOp.label === "edit"
                    ? "Editing…"
                    : aiOp.label === "detail"
                      ? "Enhancing detail…"
                      : "Working…"}
              </p>
              <p className="image-editor-busy-msg">
                {activeAiJob?.message ||
                  (activeAiJob?.status === "queued" ? "Queued — waiting for a worker." : "Processing…")}
              </p>
              {typeof activeAiJob?.progress === "number" ? (
                <div className="image-editor-busy-bar">
                  <span style={{ width: `${Math.round(activeAiJob.progress * 100)}%` }} />
                </div>
              ) : null}
            </div>
          </div>
        ) : null}

        {working ? (
          <div className="image-editor-viewbar">
            <button onClick={() => zoomAtCenter(1 / ZOOM_STEP)} title="Zoom out" type="button">
              −
            </button>
            <span className="image-editor-zoom">{Math.round(view.scale * 100)}%</span>
            <button onClick={() => zoomAtCenter(ZOOM_STEP)} title="Zoom in" type="button">
              +
            </button>
            <button onClick={fitToView} type="button">
              Fit
            </button>
            <button onClick={actualSize} type="button">
              100%
            </button>
            <span className="image-editor-dims">
              {working.width} × {working.height}
            </span>
          </div>
        ) : null}
      </div>

      {pickerOpen ? (
        <DatasetAddDialog
          assets={assets ?? []}
          characters={characters ?? []}
          confirmLabel="Open"
          eyebrow="Open"
          fileAccept="image/*"
          fileHint="Drag an image here, or"
          multiple={false}
          onAdd={(ids) => {
            setPickerOpen(false);
            if (ids[0] && confirmDiscardEdits()) openAsset(ids[0]);
          }}
          onClose={() => setPickerOpen(false)}
          onImport={(files) => {
            const file = files?.[0];
            setPickerOpen(false);
            if (file && confirmDiscardEdits()) openFile(file);
          }}
          title="Open image"
        />
      ) : null}

      {newLayoutOpen ? (
        <div
          className="image-editor-modal-backdrop"
          onClick={() => setNewLayoutOpen(false)}
          role="presentation"
        >
          <div
            aria-label="New blank layout"
            className="image-editor-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <h3 className="image-editor-modal-title">New blank layout</h3>
            <div className="image-editor-modal-field">
              <span>Aspect</span>
              <div className="image-editor-ratios" role="group" aria-label="Layout aspect">
                {EDIT_OUTPUT_ASPECTS.filter((aspect) => aspect.key !== "match").map((aspect) => (
                  <button
                    className={layoutAspect === aspect.key ? "active" : ""}
                    key={aspect.key}
                    onClick={() => setLayoutAspect(aspect.key)}
                    type="button"
                  >
                    {aspect.label}
                  </button>
                ))}
              </div>
            </div>
            <label className="image-editor-modal-field">
              <span>Size (long side)</span>
              <select onChange={(event) => setLayoutSize(Number(event.target.value))} value={layoutSize}>
                {BLANK_CANVAS_SIZES.map((size) => (
                  <option key={size} value={size}>
                    {size}px
                  </option>
                ))}
              </select>
            </label>
            <p className="image-editor-modal-dims">
              {layoutDims.width} × {layoutDims.height}px
            </p>
            <div className="image-editor-modal-actions">
              <button onClick={() => setNewLayoutOpen(false)} type="button">
                Cancel
              </button>
              <button className="primary" onClick={createBlankLayout} type="button">
                Create
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
