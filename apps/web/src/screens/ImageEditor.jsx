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
import {
  activeLayerOf,
  addLayer,
  compositeLayersToCanvas,
  createLayer,
  duplicateLayer,
  identityTransform,
  layerById,
  moveLayer,
  removeLayer,
  replaceLayerBitmap,
  sameLayerStack,
  setActiveLayer,
  setLayerProps,
  singleLayerWorking,
  snapshotLayers,
} from "../imageLayers.js";
import { LayersPanel } from "../components/LayersPanel.jsx";
import { CurveEditor } from "../components/CurveEditor.jsx";
import {
  IDENTITY_LEVELS,
  IDENTITY_CURVES,
  isIdentityLevels,
  isIdentityCurves,
  applyLevels,
  applyCurves,
  computeHistogram,
} from "../colorGrade.js";
import {
  maskAlphaFromRgba,
  writeMaskAlphaToRgba,
  invertAlpha,
  dilateAlpha,
  erodeAlpha,
  blurAlpha,
} from "../maskRefine.js";
// Pure job builders + model/engine helpers (sc-6112) — extracted to a konva-free module
// so the Library batch flow can reuse them; imported for internal use and re-exported
// below to keep this module's public surface (and its tests) unchanged.
import {
  buildDetailJobBody,
  buildEditJobBody,
  buildUpscaleJobBody,
  detailCapableModels,
  editCapableModels,
  UPSCALE_ENGINES,
  upscaleEngineHasSoftness,
  upscaleFactorsForEngine,
} from "../imageJobs.js";

export {
  buildDetailJobBody,
  buildEditJobBody,
  buildUpscaleJobBody,
  detailCapableModels,
  editCapableModels,
  upscaleEngineHasSoftness,
  upscaleFactorsForEngine,
};

const MIN_SCALE = 0.05;
const MAX_SCALE = 16;
const ZOOM_STEP = 1.2;
const MIN_CROP_PX = 8;

// Modifier glyph for the shortcut reference (the handler accepts both ⌘ and Ctrl).
const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iP(hone|ad|od)/.test(navigator.platform || navigator.userAgent || "");
const MOD_KEY = IS_MAC ? "⌘" : "Ctrl";

// Keyboard shortcut reference (sc-6111). Single source of truth for the in-editor
// quick reference; the keydown handler implements these exact bindings.
const EDITOR_SHORTCUTS = [
  {
    group: "Tools",
    items: [
      { keys: ["M"], label: "Move / pan" },
      { keys: ["T"], label: "Transform layer" },
      { keys: ["C"], label: "Crop" },
      { keys: ["U"], label: "Upscale" },
      { keys: ["D"], label: "Detail enhance" },
      { keys: ["G"], label: "Color grade" },
      { keys: ["E"], label: "AI edit" },
      { keys: ["B"], label: "Boxes" },
    ],
  },
  {
    group: "View",
    items: [
      { keys: ["+"], label: "Zoom in" },
      { keys: ["−"], label: "Zoom out" },
      { keys: ["0"], label: "Fit to view" },
      { keys: ["1"], label: "Actual size (100%)" },
    ],
  },
  {
    group: "Edit",
    items: [
      { keys: [MOD_KEY, "Z"], label: "Undo" },
      { keys: ["⇧", MOD_KEY, "Z"], label: "Redo" },
      { keys: ["Delete"], label: "Delete selected box" },
      { keys: ["Esc"], label: "Cancel / deselect" },
      { keys: ["?"], label: "Toggle this help" },
    ],
  },
];

// Future-tool scaffold (epic 2427) — rendered as inert buttons so the frame + the
// next slices' insertion points stay in place. All current epic-2427 tools are live
// (Move + Crop + Upscale + Color + AI Edit + Detail), so this is empty for now.
const UPCOMING_TOOLS = [];

// Upper bound on images in a FLUX.2 multi-reference edit, matching the worker's MAX_EDIT_REFERENCES
// (image_jobs/flux2.rs). The working image takes one slot, so the editor allows up to 3 user refs.
export const MAX_EDIT_REFERENCES = 4;

// Compose the multi-reference edit's `referenceAssetIds`: the working image (staged as the scratch
// source) FIRST so it anchors the edit, then the user's reference images — trimmed, de-duped, and
// capped at `max` total. The worker prefers a non-empty `referenceAssetIds` list over `sourceAssetId`
// (image_jobs/flux2.rs `flux2_edit_reference_ids`), so the working scratch id must lead the list or
// the working image is dropped from the joint conditioning. Pure. Empty source → empty list.
export function editReferenceIds(sourceAssetId, refIds, max = MAX_EDIT_REFERENCES) {
  const ids = [sourceAssetId, ...(refIds ?? [])]
    .map((id) => (typeof id === "string" ? id.trim() : ""))
    .filter(Boolean);
  return Array.from(new Set(ids)).slice(0, max);
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

// Konva custom filter for the live preview — reads the grade from the node's attrs
// (set declaratively by react-konva) and runs the shared math, so preview === bake.
// `gradeMode` selects which grade is previewed (sc-6109): the brightness/contrast
// "adjust", levels, or curves.
function konvaColorFilter(imageData) {
  const mode = this.getAttr("gradeMode");
  if (mode === "levels") applyLevels(imageData.data, this.getAttr("gradeLevels"));
  else if (mode === "curves") applyCurves(imageData.data, this.getAttr("gradeCurves"));
  else applyColorAdjustments(imageData.data, this.getAttr("colorAdjust"));
}

// The `POST /api/v1/jobs` body for a smart-select image_segment job (sc-3751 / backend
// sc-6105). Same generic-jobs shape as upscale/detail; the worker (native-MLX SAM3) reads
// `sourceAssetId` + a `box` prompt `[x1,y1,x2,y2]` in source-image pixel coords and returns a
// binary mask asset. Pure for testing.
export function buildSegmentJobBody({ project, requestedGpu, sourceAssetId, box, displayName }) {
  return {
    type: "image_segment",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload: {
      projectId: project.id,
      sourceAssetId,
      box,
      displayName,
    },
  };
}

// Convert an image-pixel rect `{x,y,width,height}` to a SAM3 box prompt `[x1,y1,x2,y2]`, ordered
// (positive width/height) and rounded to whole pixels. Pure for testing.
export function rectToSegmentBox(rect) {
  const x1 = Math.round(Math.min(rect.x, rect.x + rect.width));
  const y1 = Math.round(Math.min(rect.y, rect.y + rect.height));
  const x2 = Math.round(Math.max(rect.x, rect.x + rect.width));
  const y2 = Math.round(Math.max(rect.y, rect.y + rect.height));
  return [x1, y1, x2, y2];
}

// The smart-select mask preview tint: translucent pink, matching the brush-stroke color so the
// auto-mask and any brush refinements read as one selection.
export const MASK_PREVIEW_RGBA = [255, 40, 120, 128];

// Recolor a decoded white-on-black mask's RGBA buffer in place to pink-on-transparent for the
// on-canvas preview: foreground (luminance > 127) → translucent pink, background → transparent.
// Pure (operates on the pixel buffer) so it's unit-testable without a real canvas.
export function tintMaskRgbaInPlace(data) {
  const [r, g, b, a] = MASK_PREVIEW_RGBA;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 127) {
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = a;
    } else {
      data[i + 3] = 0;
    }
  }
  return data;
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
export function buildSaveProvenance({ source, edits, width, height, layers }) {
  const provenance = {
    editor: "image_editor",
    source: source?.assetId
      ? { kind: "asset", assetId: source.assetId, name: source.name ?? null }
      : { kind: "upload", name: source?.name ?? null },
    edits: edits ?? [],
    width: width ?? null,
    height: height ?? null,
  };
  // Layer summary (sc-6121): record what the flattened asset was composited from —
  // bottom→top name / opacity / blend / visibility. Omitted for the degenerate
  // single-layer document (it adds nothing over the flat bitmap, and keeps a plain
  // non-layered save's provenance byte-for-byte as it was before layers).
  if (Array.isArray(layers) && layers.length > 1) {
    provenance.layers = layers.map((layer) => ({
      name: layer.name,
      opacity: layer.opacity,
      blendMode: layer.blendMode,
      visible: layer.visible,
    }));
  }
  return provenance;
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

// ── Undo/redo history (sc-6106) ────────────────────────────────────────────
// A bounded, backend-free history over opaque working-image snapshots. The
// reducer is pure — it only shuffles snapshots between the past (undo) and future
// (redo) stacks; the caller owns capturing a snapshot (the working bitmap blob +
// box/provenance overlay state) and restoring one (decode + install). Snapshots
// hold a Blob, never a live object URL, so an evicted snapshot is plain garbage —
// there is nothing to revoke, which keeps the "no leak of evicted snapshots"
// guarantee trivial. The stack depth is bounded so retained bitmaps stay capped.
export const HISTORY_LIMIT = 30;

export function emptyHistory() {
  return { past: [], future: [] };
}

// Push the current snapshot onto the undo stack and drop the redo stack. Call at
// the START of an operation, before the working state mutates, with a snapshot of
// the pre-operation state. Bounded to the `limit` most-recent entries.
export function historyCheckpoint(history, snapshot, limit = HISTORY_LIMIT) {
  return { past: [...history.past, snapshot].slice(-limit), future: [] };
}

// Step back one operation. `present` is the current on-screen snapshot, captured
// fresh by the caller so a later redo restores exactly what is on screen now.
// Returns the next history plus the snapshot to restore (`restore` is null when
// there is nothing to undo, in which case `history` is returned unchanged).
export function historyUndo(history, present, limit = HISTORY_LIMIT) {
  if (!history.past.length) return { history, restore: null };
  const restore = history.past[history.past.length - 1];
  return {
    history: { past: history.past.slice(0, -1), future: [present, ...history.future].slice(0, limit) },
    restore,
  };
}

// Step forward one operation, symmetric to historyUndo.
export function historyRedo(history, present, limit = HISTORY_LIMIT) {
  if (!history.future.length) return { history, restore: null };
  const [restore, ...rest] = history.future;
  return {
    history: { past: [...history.past, present].slice(-limit), future: rest },
    restore,
  };
}

export const canUndo = (history) => history.past.length > 0;
export const canRedo = (history) => history.future.length > 0;

// Revoke the object URLs of a set of live layers (sc-6117). Undo snapshots hold
// only blobs (no URLs), so the only URLs that ever need revoking are the live
// ones, when their layer is evicted — on delete, on a session replace, and on
// unmount. Tolerant of null/missing URLs so callers don't have to guard.
function revokeLayerUrls(layers) {
  for (const layer of layers ?? []) {
    if (layer?.objectUrl) URL.revokeObjectURL(layer.objectUrl);
  }
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
  // Smart-select (sc-3751) runs native-MLX SAM3 — Mac-only, no torch/candle path. Gate it on the
  // platform-intrinsic `imageSegment` capability (true only on a Mac backend, false off-Mac and
  // pre-load), like the seedvr2 engine — independent of the Mac gating-rollout switch. When false,
  // the mask tool shows only the hand brush (graceful degradation).
  const smartSelectSupported = macCapabilities?.features?.imageSegment?.supported === true;

  // The working document (sc-6117): an ordered raster layer stack composited
  // bottom→top — `{ width, height, source, layers:[Layer], activeLayerId }` (see
  // ../imageLayers.js). A single-layer stack is the degenerate case that behaves
  // exactly like the pre-layers single bitmap, so the existing tools keep operating
  // on the active layer; the per-layer tool matrix + the panel land in sc-6118/6119.
  // Each live layer owns its decoded `image` + `objectUrl` (revoked on eviction).
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
  // Curves + levels (sc-6109): the Color tool has three modes — the brightness/
  // contrast "adjust" (above), per-channel levels, and an editable tone curve. Each
  // previews via the same Konva filter and bakes via the same Canvas-2D pass. The
  // active channel ("master" | "r" | "g" | "b") is shared by the levels + curves UI.
  const [colorMode, setColorMode] = useState("adjust"); // "adjust" | "levels" | "curves"
  const [levels, setLevels] = useState(IDENTITY_LEVELS);
  const [curves, setCurves] = useState(IDENTITY_CURVES);
  const [colorChannel, setColorChannel] = useState("master");
  const histogramRef = useRef(null);

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

  // Keyboard-shortcut quick reference panel (sc-6111).
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  // Inpaint mask (sc-2436): freehand brush strokes in image-pixel coords, rasterized
  // to a mask asset on Run for inpaint-capable models. `maskMode` is the paint sub-mode
  // of the AI Edit tool (Stage panning is suspended while it's on).
  const [maskLines, setMaskLines] = useState([]); // [{ points:[x,y,…], size, erase }]
  const [maskMode, setMaskMode] = useState(false);
  const [maskBrush, setMaskBrush] = useState(64);
  const [maskErase, setMaskErase] = useState(false);
  const maskPaintingRef = useRef(false);
  // Mask refinement (sc-6110): feather / grow / shrink radius in px for the post-process ops.
  const [maskRefineRadius, setMaskRefineRadius] = useState(6);

  // Smart-select (sc-3751): a box-drag sub-mode of the mask tool that runs the SAM3 `image_segment`
  // job (sc-6105) and loads the returned binary mask as an editable base UNDER the brush strokes.
  // `maskBaseImage` is the white-on-black mask bitmap (rasterized into the mask PNG alongside the
  // strokes); `maskOverlay` is its pink-on-transparent preview (rendered in the mask layer, so the
  // eraser's destination-out clears it too). `maskSubTool` toggles brush vs. box-select.
  const [maskBaseImage, setMaskBaseImage] = useState(null); // HTMLImageElement | null
  const [maskOverlay, setMaskOverlay] = useState(null); // HTMLCanvasElement | null
  const [maskSubTool, setMaskSubTool] = useState("brush"); // "brush" | "select"
  const [selectDraft, setSelectDraft] = useState(null); // live selection rect during a drag
  const selectDrawingRef = useRef(false);
  const selectStartRef = useRef(null);

  // Reference-image conditioning (sc-6107): user-attached library images that jointly condition the
  // AI Edit alongside the working image, on a FLUX.2 `multiReference` edit model. The working image is
  // added at run time (it's staged as a scratch source), so this holds only the user's picks.
  const [refAssetIds, setRefAssetIds] = useState([]); // string[] of library asset ids
  const [refPickerOpen, setRefPickerOpen] = useState(false);

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
  // Whether the edit model conditions on extra reference images (FLUX.2 multi-reference edit, sc-6107):
  // the manifest tags it `ui.multiReference`. Gates the reference picker; off-models hide it entirely.
  const multiRefCapable = Boolean(selectedEditModel?.ui?.multiReference);
  // Drop any attached references when the model can't use them (switched away from a multiReference
  // model), so a stale selection never rides a job that would ignore it.
  useEffect(() => {
    if (!multiRefCapable && refAssetIds.length) setRefAssetIds([]);
  }, [multiRefCapable, refAssetIds.length]);

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
  const needsFitRef = useRef(false);
  // Monotonic layer-id source: ids survive an undo (the seq is snapshotted, like
  // boxIdSeq) so a layer added after an undo never collides with a recycled id.
  const layerIdRef = useRef(0);
  const cropRectRef = useRef(null);
  const transformerRef = useRef(null);
  const layerTransformerRef = useRef(null); // Konva transformer bound to the active layer (sc-6120)
  const imageNodeRef = useRef(null); // Konva image node — cached for color-grade filtering + transform
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  // Undo/redo (sc-6106): a bounded snapshot history over the working-image session.
  // The stacks live in a ref for synchronous reads inside the commit handlers; the
  // can-undo/redo flags are mirrored into state so the toolbar buttons re-render.
  const historyRef = useRef(emptyHistory());
  const [historyFlags, setHistoryFlags] = useState({ canUndo: false, canRedo: false });
  // Live mirror of the working document for synchronous reads inside the commit
  // handlers (a checkpoint captures the pre-operation stack; restore reuses the
  // live decoded images for unchanged layers and revokes the URLs it drops).
  const workingRef = useRef(null);
  // Live mirrors of the snapshot-relevant state so a synchronous checkpoint can
  // capture the pre-operation state without stale-closure surprises.
  const editsRef = useRef(edits);
  const dirtyRef = useRef(dirty);
  const savedAssetIdRef = useRef(savedAssetId);
  const boxesRef = useRef(boxes);
  const boxColorRef = useRef(boxColor);
  const aiOpRef = useRef(aiOp);
  useEffect(() => { editsRef.current = edits; }, [edits]);
  useEffect(() => { dirtyRef.current = dirty; }, [dirty]);
  useEffect(() => { savedAssetIdRef.current = savedAssetId; }, [savedAssetId]);
  useEffect(() => { boxesRef.current = boxes; }, [boxes]);
  useEffect(() => { boxColorRef.current = boxColor; }, [boxColor]);
  useEffect(() => { aiOpRef.current = aiOp; }, [aiOp]);
  useEffect(() => { workingRef.current = working; }, [working]);

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

  // Revoke every live layer's object URL when the editor unmounts.
  useEffect(() => () => revokeLayerUrls(workingRef.current?.layers), []);

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

  const nextLayerId = () => `layer_${(layerIdRef.current += 1)}`;

  // Reset the per-bitmap editor overlays/tool state that a new working bitmap
  // invalidates (tool, crop, color preview, mask, references, boxes). Shared by
  // installWorkingImage (open/crop/AI result) and a bitmap-changing undo restore.
  const resetEditorOverlays = useCallback(() => {
    setTool("move");
    setCropRect(null);
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setColorMode("adjust");
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
    setColorChannel("master");
    // A new working bitmap invalidates the mask (dims/content changed) — strokes + smart-select base.
    setMaskLines([]);
    setMaskMode(false);
    setMaskBaseImage(null);
    setMaskOverlay(null);
    setMaskSubTool("brush");
    setSelectDraft(null);
    selectDrawingRef.current = false;
    // A new editing session starts with no attached reference images (sc-6107).
    setRefAssetIds([]);
    setRefPickerOpen(false);
    // Boxes are in image-pixel coords → a new bitmap (open/crop/upscale/AI op) invalidates them.
    setBoxes([]);
    setSelectedBoxId(null);
    setBoxDraft(null);
    boxNodeRefs.current.clear();
    boxDrawingRef.current = false;
  }, []);

  // Replace the whole working document with a fresh single-layer stack from one
  // decoded bitmap (open / blank / crop / color / AI result). Revokes the evicted
  // layers' object URLs first. The single-layer stack is the degenerate case that
  // reproduces the pre-layers single-bitmap behavior; multi-layer creation is the
  // layers panel (sc-6118).
  const installWorkingImage = useCallback(
    (image, objectUrl, blob, source) => {
      revokeLayerUrls(workingRef.current?.layers);
      needsFitRef.current = true;
      resetEditorOverlays();
      setWorking(singleLayerWorking({ id: nextLayerId(), image, objectUrl, blob, source }));
    },
    [resetEditorOverlays],
  );

  // ── Undo/redo plumbing (sc-6106, extended to the layer stack sc-6117) ──────
  // A snapshot is the layer stack (each layer as metadata + its shared blob, no
  // live image/URL) plus the overlay/provenance state that a re-install would
  // otherwise reset (boxes, edit chain, dirty flag). Blobs are shared by reference
  // across snapshots, so retained bitmaps stay bounded like the single-bitmap days.
  const captureSnapshot = useCallback(() => {
    const work = workingRef.current;
    return {
      layers: snapshotLayers(work?.layers),
      activeLayerId: work?.activeLayerId ?? null,
      width: work?.width ?? 0,
      height: work?.height ?? 0,
      source: work?.source ?? null,
      layerIdSeq: layerIdRef.current,
      edits: editsRef.current,
      dirty: dirtyRef.current,
      savedAssetId: savedAssetIdRef.current,
      boxes: boxesRef.current,
      boxColor: boxColorRef.current,
      boxIdSeq: boxIdRef.current,
    };
  }, []);

  const syncHistoryFlags = useCallback(() => {
    setHistoryFlags({ canUndo: canUndo(historyRef.current), canRedo: canRedo(historyRef.current) });
  }, []);

  // Record a step: push the pre-operation snapshot onto the undo stack. Call this
  // BEFORE the working state mutates (crop/color/AI result, layer op, or a box change).
  const checkpoint = useCallback(() => {
    if (!workingRef.current) return;
    historyRef.current = historyCheckpoint(historyRef.current, captureSnapshot());
    syncHistoryFlags();
  }, [captureSnapshot, syncHistoryFlags]);

  // Start a fresh history for a newly opened session (clears both stacks).
  const resetHistory = useCallback(() => {
    historyRef.current = emptyHistory();
    syncHistoryFlags();
  }, [syncHistoryFlags]);

  // Re-apply a snapshot's overlay/provenance state, keeping the live mirrors in
  // sync immediately so an undo→undo chain reads the right "present" each step.
  const applyHistoryAux = useCallback((snap) => {
    setEdits(snap.edits);
    editsRef.current = snap.edits;
    setDirty(snap.dirty);
    dirtyRef.current = snap.dirty;
    setSavedAssetId(snap.savedAssetId);
    savedAssetIdRef.current = snap.savedAssetId;
    setBoxes(snap.boxes);
    boxesRef.current = snap.boxes;
    setBoxColor(snap.boxColor);
    boxColorRef.current = snap.boxColor;
    setSelectedBoxId(null);
    boxIdRef.current = snap.boxIdSeq;
    // Restore the layer-id counter so a layer added after this undo can't recycle
    // an id that a redo would bring back (mirrors boxIdSeq).
    if (typeof snap.layerIdSeq === "number") layerIdRef.current = snap.layerIdSeq;
  }, []);

  const restoreSnapshot = useCallback(
    async (snap) => {
      if (!snap) return;
      try {
        const live = workingRef.current;
        // Overlay-only steps (box edits) keep the stack pixel- and metadata-identical
        // → skip the rebuild, the decode, and the view refit; only a bitmap/structure
        // change (crop/color/AI/layer op) re-installs the stack.
        const stackChanged =
          !live ||
          !sameLayerStack(live.layers, snap.layers) ||
          live.activeLayerId !== snap.activeLayerId ||
          live.width !== snap.width ||
          live.height !== snap.height;
        if (stackChanged) {
          // Rebuild the live stack from the snapshot: reuse a live layer's decoded
          // image when its blob is unchanged (decode ONLY changed/new layers), and
          // revoke the object URLs of live layers the restore drops.
          const liveById = new Map((live?.layers ?? []).map((layer) => [layer.id, layer]));
          const reused = new Set();
          const layers = [];
          for (const sl of snap.layers) {
            const prev = liveById.get(sl.id);
            if (prev && prev.blob === sl.blob && prev.image) {
              reused.add(sl.id);
              layers.push({
                ...prev,
                name: sl.name,
                visible: sl.visible,
                opacity: sl.opacity,
                blendMode: sl.blendMode,
                transform: { ...sl.transform },
              });
            } else {
              const { image, objectUrl } = await blobToImage(sl.blob);
              layers.push(createLayer({ ...sl, image, objectUrl }));
            }
          }
          for (const layer of live?.layers ?? []) {
            if (!reused.has(layer.id) && layer.objectUrl) URL.revokeObjectURL(layer.objectUrl);
          }
          // A true PIXEL change = dims changed, a layer added/removed, or a layer's
          // blob differs. Metadata-only undos (opacity / visibility / blend / transform
          // / reorder — same id→blob set, same dims) keep the current tool, mask, boxes
          // and view; only a pixel change resets the per-bitmap overlays + refits.
          const dimsChanged = !live || live.width !== snap.width || live.height !== snap.height;
          const liveBlobs = new Map((live?.layers ?? []).map((layer) => [layer.id, layer.blob]));
          const bitmapChanged =
            dimsChanged ||
            liveBlobs.size !== snap.layers.length ||
            snap.layers.some((sl) => liveBlobs.get(sl.id) !== sl.blob);
          if (bitmapChanged) resetEditorOverlays();
          if (dimsChanged) needsFitRef.current = true;
          const nextWorking = {
            width: snap.width,
            height: snap.height,
            source: snap.source,
            layers,
            activeLayerId: snap.activeLayerId,
          };
          workingRef.current = nextWorking;
          setWorking(nextWorking);
        }
        applyHistoryAux(snap);
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not restore that step." });
      }
    },
    [resetEditorOverlays, applyHistoryAux],
  );

  const undo = useCallback(async () => {
    if (aiOpRef.current || !workingRef.current) return;
    const { history: next, restore } = historyUndo(historyRef.current, captureSnapshot());
    if (!restore) return;
    historyRef.current = next;
    syncHistoryFlags();
    await restoreSnapshot(restore);
  }, [captureSnapshot, restoreSnapshot, syncHistoryFlags]);

  const redo = useCallback(async () => {
    if (aiOpRef.current || !workingRef.current) return;
    const { history: next, restore } = historyRedo(historyRef.current, captureSnapshot());
    if (!restore) return;
    historyRef.current = next;
    syncHistoryFlags();
    await restoreSnapshot(restore);
  }, [captureSnapshot, restoreSnapshot, syncHistoryFlags]);

  // ── Keyboard shortcuts (sc-6111) ───────────────────────────────────────────
  // One editor-scoped window keydown handler. Held behind a ref so the listener is
  // subscribed once (no add/remove churn during the high-frequency re-renders of a
  // crop / box / mask drag) while always seeing the latest tool + selection state.
  // Never fires while a text field is focused, so typing a prompt / box description
  // / renaming a layer is left to the browser. Undo/redo (sc-6106) are the only
  // modified combos we own; the rest are single keys that mirror the toolbar and the
  // zoom bar. `?` toggles the quick reference and works before an image is open.
  const onEditorKeyDownRef = useRef(null);
  onEditorKeyDownRef.current = (event) => {
    const tag = event.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target?.isContentEditable) return;

    if (event.metaKey || event.ctrlKey) {
      const k = event.key?.toLowerCase();
      if (k === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
      } else if (k === "y") {
        event.preventDefault();
        redo();
      }
      return;
    }
    // Single-key shortcuts only — never with Alt (avoids hijacking OS combos).
    if (event.altKey) return;

    if (event.key === "?") {
      event.preventDefault();
      setShortcutsOpen((on) => !on);
      return;
    }
    if (event.key === "Escape") {
      if (shortcutsOpen) setShortcutsOpen(false);
      else escapeGesture();
      return;
    }

    if (!workingRef.current) return;

    // View shortcuts work regardless of the busy/AI state.
    switch (event.key) {
      case "+":
      case "=":
        event.preventDefault();
        zoomAtCenter(ZOOM_STEP);
        return;
      case "-":
      case "_":
        event.preventDefault();
        zoomAtCenter(1 / ZOOM_STEP);
        return;
      case "0":
        event.preventDefault();
        fitToView();
        return;
      case "1":
        event.preventDefault();
        actualSize();
        return;
      case "Delete":
      case "Backspace":
        if (tool === "boxes" && selectedBoxId) {
          event.preventDefault();
          deleteBox(selectedBoxId);
        }
        return;
      default:
        break;
    }

    // Tool switches. Move always works (it cancels/pans); the rest mirror their
    // toolbar buttons' enabled state and are suppressed while an AI op is running.
    const key = event.key.toLowerCase();
    if (key === "m") {
      cancelCrop();
      return;
    }
    if (aiOpRef.current) return;
    if (key === "t") startTransform();
    else if (key === "c") startCrop();
    else if (key === "u") {
      if (!macUpscaleBlock) setTool("upscale");
    } else if (key === "d") {
      if (detailModels.length) setTool("detail");
    } else if (key === "g") startColorGrade();
    else if (key === "e") setTool("edit");
    else if (key === "b") selectBoxTool();
  };

  useEffect(() => {
    const handler = (event) => onEditorKeyDownRef.current?.(event);
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const openFromBlob = useCallback(
    async (blob, source) => {
      setStatus({ loading: true, error: "" });
      try {
        const { image, objectUrl } = await blobToImage(blob);
        installWorkingImage(image, objectUrl, blob, source);
        // A freshly opened image is a clean session — clear edit/provenance state
        // and start a fresh undo/redo history rooted at this bitmap (sc-6106).
        setEdits([]);
        setDirty(false);
        setSavedAssetId(null);
        resetHistory();
        setStatus({ loading: false, error: "" });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not open image" });
      }
    },
    [installWorkingImage, resetHistory],
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
    // Discard any unbaked color preview (adjust / levels / curves).
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
  }

  // Escape (sc-6111): cancel the most specific in-progress gesture, falling back to
  // deselecting / returning to the Move tool. Highest priority first.
  function escapeGesture() {
    if (boxDrawingRef.current) {
      boxDrawingRef.current = false;
      setBoxDraft(null);
      return;
    }
    if (selectDrawingRef.current) {
      selectDrawingRef.current = false;
      setSelectDraft(null);
      return;
    }
    if (tool === "crop") {
      cancelCrop();
      return;
    }
    if (selectedBoxId) {
      setSelectedBoxId(null);
      return;
    }
    // Any other active tool → back to Move (also discards an unbaked color preview).
    if (tool !== "move") cancelCrop();
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

  // ── Active-layer write-back + flatten plumbing (sc-6119) ───────────────────
  // Encode just the ACTIVE layer's bitmap to a PNG File — the source for an
  // active-layer AI op (same-size edit / detail / smart-select) whose result is
  // written back to that layer, the rest of the stack preserved.
  const activeLayerToFile = useCallback(
    (filename) =>
      new Promise((resolve, reject) => {
        const work = workingRef.current;
        const layer = activeLayerOf(work);
        if (!layer) {
          reject(new Error("No active layer."));
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = layer.image.naturalWidth;
        canvas.height = layer.image.naturalHeight;
        canvas.getContext("2d").drawImage(layer.image, 0, 0);
        const base = (work.source.name || "image").replace(/\.[^./\\]+$/, "");
        canvas.toBlob(
          (blob) =>
            blob
              ? resolve(new File([blob], filename || `${base}.png`, { type: "image/png" }))
              : reject(new Error("Could not encode the layer.")),
          "image/png",
        );
      }),
    [],
  );

  // Write a decoded AI/grade result back into a specific layer, revoking that
  // layer's previous object URL. Preserves the doc dims + the rest of the stack.
  const replaceLayerImage = useCallback((id, image, objectUrl, blob) => {
    const prev = layerById(workingRef.current, id);
    if (prev?.objectUrl && prev.objectUrl !== objectUrl) URL.revokeObjectURL(prev.objectUrl);
    setWorking((cur) => replaceLayerBitmap(cur, id, { image, objectUrl, blob }));
  }, []);

  // A document-level AI op (upscale / outpaint / box-keyed edit) flattens the stack
  // into one base layer; warn before discarding a multi-layer stack.
  const confirmFlatten = useCallback(() => {
    const n = workingRef.current?.layers?.length ?? 0;
    if (n <= 1) return true;
    return (
      typeof window.confirm !== "function" ||
      window.confirm(`This will flatten ${n} layers into a single layer. Continue?`)
    );
  }, []);

  // Apply: document-level crop — crop every layer to the rect, set the doc dims,
  // keep the stack. The bitmaps are blob-backed (never tainted), so reading pixels
  // back is safe; provenance is preserved so lineage survives to Save (sc-2434).
  const applyCrop = useCallback(async () => {
    if (!working || !cropRect || !working.layers.length) return;
    const sx = clamp(Math.round(cropRect.x), 0, working.width - 1);
    const sy = clamp(Math.round(cropRect.y), 0, working.height - 1);
    const sw = clamp(Math.round(cropRect.width), 1, working.width - sx);
    const sh = clamp(Math.round(cropRect.height), 1, working.height - sy);
    // Document-level crop (sc-6119): crop EVERY layer's bitmap to the rect and set the
    // doc dims — the stack is preserved. Layers are doc-aligned in this slice; per-layer
    // transforms (and transform-aware crop) arrive with sc-6120.
    let cropped;
    try {
      cropped = await Promise.all(
        working.layers.map(async (layer) => {
          const canvas = document.createElement("canvas");
          canvas.width = sw;
          canvas.height = sh;
          canvas.getContext("2d").drawImage(layer.image, sx, sy, sw, sh, 0, 0, sw, sh);
          const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
          if (!blob) throw new Error("Could not encode the crop.");
          const { image, objectUrl } = await blobToImage(blob);
          return { id: layer.id, image, objectUrl, blob };
        }),
      );
    } catch (err) {
      setStatus({ loading: false, error: err.message || "Could not crop the layers." });
      return;
    }
    checkpoint();
    const oldLayers = workingRef.current.layers;
    needsFitRef.current = true;
    // Crop invalidates the mask + boxes (old-document pixel coords) and returns to Move.
    resetEditorOverlays();
    const byId = new Map(cropped.map((c) => [c.id, c]));
    setWorking((prev) => ({
      ...prev,
      width: sw,
      height: sh,
      layers: prev.layers.map((layer) => {
        const c = byId.get(layer.id);
        return c ? { ...layer, image: c.image, objectUrl: c.objectUrl, blob: c.blob } : layer;
      }),
    }));
    oldLayers.forEach((layer) => layer.objectUrl && URL.revokeObjectURL(layer.objectUrl));
    setEdits((prev) => [...prev, { op: "crop", width: sw, height: sh }]);
    setDirty(true);
  }, [working, cropRect, checkpoint, resetEditorOverlays]);

  // Bind the transformer to the crop rect whenever crop mode is active.
  useEffect(() => {
    const transformer = transformerRef.current;
    const node = cropRectRef.current;
    if (tool === "crop" && transformer && node) {
      transformer.nodes([node]);
      transformer.getLayer()?.batchDraw();
    }
  }, [tool, cropRect]);

  // Bind the layer transformer to the ACTIVE layer's node whenever the Transform
  // tool is active (sc-6120); re-bind when the active layer changes. `working` in
  // the deps covers an active-layer switch (imageNodeRef reattaches to the new node).
  useEffect(() => {
    const transformer = layerTransformerRef.current;
    if (!transformer) return;
    const node = tool === "transform" ? imageNodeRef.current : null;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
  }, [tool, working]);

  // ── Color grade (sc-2439; curves + levels sc-6109) ─────────────────────────
  function startColorGrade() {
    if (!working) return;
    setTool("color");
    setColorMode("adjust");
    setColorChannel("master");
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
  }

  const setAdjustValue = (key, value) => setColorAdjust((prev) => ({ ...prev, [key]: value }));
  const resetAdjust = (key) => setAdjustValue(key, 0);

  // Patch the active channel's levels (sc-6109).
  const setLevelsValue = (key, value) =>
    setLevels((prev) => ({ ...prev, [colorChannel]: { ...prev[colorChannel], [key]: value } }));

  // Reset the currently-selected color mode (all channels) to its identity.
  function resetActiveColorMode() {
    if (colorMode === "levels") setLevels(IDENTITY_LEVELS);
    else if (colorMode === "curves") setCurves(IDENTITY_CURVES);
    else setColorAdjust(IDENTITY_COLOR_ADJUST);
  }

  // Stroke for the curve editor / channel cue.
  const channelStroke = { master: "var(--accent)", r: "#d44", g: "#4a4", b: "#46d" }[colorChannel];

  // Whether the currently-selected color mode is at its identity (gates Apply + the
  // live-preview cache).
  function activeGradeIsIdentity() {
    if (colorMode === "levels") return isIdentityLevels(levels);
    if (colorMode === "curves") return isIdentityCurves(curves);
    return isIdentityAdjust(colorAdjust);
  }

  // Live preview: Konva applies filters only on a cached node, and re-running them
  // needs a re-cache. Cache the active layer's node (re-caching when ANY grade input
  // changes) while the color tool is active with a non-identity grade; clear it
  // otherwise so Move/other tools see the untouched bitmap. The filter reads the
  // gradeMode + colorAdjust/levels/curves attrs.
  useEffect(() => {
    const node = imageNodeRef.current;
    if (!node) return;
    if (tool === "color" && !activeGradeIsIdentity()) {
      node.cache();
    } else {
      node.clearCache();
    }
    node.getLayer()?.batchDraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, colorMode, colorAdjust, levels, curves, working]);

  // Draw the active layer's histogram for the levels mode (sc-6109). Recomputed when
  // the layer or selected channel changes; cheap (one pass over the layer bitmap).
  useEffect(() => {
    const canvas = histogramRef.current;
    if (!canvas || tool !== "color" || colorMode !== "levels") return;
    const layer = activeLayerOf(working);
    if (!layer) return;
    const off = document.createElement("canvas");
    off.width = layer.image.naturalWidth;
    off.height = layer.image.naturalHeight;
    const octx = off.getContext("2d");
    octx.drawImage(layer.image, 0, 0);
    const hist = computeHistogram(octx.getImageData(0, 0, off.width, off.height).data);
    const series = colorChannel === "master" ? hist.luma : hist[colorChannel];
    const peak = Math.max(1, ...series);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = colorChannel === "r" ? "#d44" : colorChannel === "g" ? "#4a4" : colorChannel === "b" ? "#46d" : "#888";
    const bw = canvas.width / 256;
    for (let i = 0; i < 256; i += 1) {
      const h = (series[i] / peak) * canvas.height;
      ctx.fillRect(i * bw, canvas.height - h, Math.max(1, bw), h);
    }
  }, [tool, colorMode, colorChannel, working]);

  // Apply: bake the active mode's grade (adjust / levels / curves) into the ACTIVE
  // layer using the SAME pixel math as the live preview (a 2D-canvas pass), writing
  // it back in place — stack + dims preserved (sc-6119). Records the grade in the
  // edit chain (sc-6109).
  const applyColorGrade = useCallback(async () => {
    const layer = activeLayerOf(working);
    if (!working || !layer) return;
    // Resolve the active mode's transform + provenance entry; bail if it's identity.
    let transform;
    let edit;
    if (colorMode === "levels") {
      if (isIdentityLevels(levels)) return;
      const baked = levels;
      transform = (data) => applyLevels(data, baked);
      edit = { op: "levels", levels: baked };
    } else if (colorMode === "curves") {
      if (isIdentityCurves(curves)) return;
      const baked = curves;
      transform = (data) => applyCurves(data, baked);
      edit = { op: "curves", curves: baked };
    } else {
      if (isIdentityAdjust(colorAdjust)) return;
      const baked = { ...colorAdjust };
      transform = (data) => applyColorAdjustments(data, baked);
      edit = { op: "color", ...baked };
    }
    const w = layer.image.naturalWidth;
    const h = layer.image.naturalHeight;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(layer.image, 0, 0);
    const imageData = ctx.getImageData(0, 0, w, h);
    transform(imageData.data);
    ctx.putImageData(imageData, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    const { image, objectUrl } = await blobToImage(blob);
    checkpoint();
    replaceLayerImage(layer.id, image, objectUrl, blob);
    // The grade is baked; drop every live preview.
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
    setTool("move");
    setEdits((prev) => [...prev, edit]);
    setDirty(true);
  }, [working, colorMode, colorAdjust, levels, curves, replaceLayerImage, checkpoint]);

  // ── Layer stack ops (sc-6118) ─────────────────────────────────────────────
  // Wire the layers panel to the pure layer-stack ops (../imageLayers.js). Each
  // mutating op checkpoints first (sc-6106 → undoable) and marks the session dirty.
  // Structural ops manage object URLs: delete revokes the evicted layer's URL;
  // add/duplicate decode a fresh blob into the new layer's own image + URL.
  function selectLayer(id) {
    setWorking((prev) => (prev ? setActiveLayer(prev, id) : prev));
  }

  function toggleLayerVisible(id) {
    if (!workingRef.current) return;
    checkpoint();
    setWorking((prev) => {
      const layer = layerById(prev, id);
      return layer ? setLayerProps(prev, id, { visible: !layer.visible }) : prev;
    });
    setDirty(true);
  }

  // One undo step per opacity DRAG: the panel flags the first change of a gesture
  // (`isGestureStart`) → checkpoint once, then the rest of the drag just updates.
  function changeLayerOpacity(id, opacity, isGestureStart) {
    if (!workingRef.current) return;
    if (isGestureStart) checkpoint();
    setWorking((prev) => setLayerProps(prev, id, { opacity }));
    setDirty(true);
  }

  function renameLayer(id, name) {
    const work = workingRef.current;
    const layer = work && layerById(work, id);
    if (!layer || layer.name === name) return;
    checkpoint();
    setWorking((prev) => setLayerProps(prev, id, { name }));
    setDirty(true);
  }

  function reorderLayer(id, toIndex) {
    if (!workingRef.current) return;
    checkpoint();
    setWorking((prev) => moveLayer(prev, id, toIndex));
    setDirty(true);
  }

  function deleteLayer(id) {
    const work = workingRef.current;
    if (!work || work.layers.length <= 1) return;
    checkpoint();
    const { working: next, removed } = removeLayer(work, id);
    if (!removed) return;
    setWorking(next);
    if (removed.objectUrl) URL.revokeObjectURL(removed.objectUrl);
    setDirty(true);
  }

  async function duplicateLayerById(id) {
    const work = workingRef.current;
    const src = work && layerById(work, id);
    if (!src) return;
    const { image, objectUrl } = await blobToImage(src.blob);
    checkpoint();
    setWorking((prev) => duplicateLayer(prev, id, { id: nextLayerId(), image, objectUrl }));
    setDirty(true);
  }

  async function addBlankLayer() {
    const work = workingRef.current;
    if (!work) return;
    // A new transparent layer at the document size — a fresh surface above the
    // active layer. The tools begin targeting it with the sc-6119 per-layer matrix.
    const canvas = document.createElement("canvas");
    canvas.width = work.width;
    canvas.height = work.height;
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) {
      setStatus({ loading: false, error: "Could not create the layer." });
      return;
    }
    const { image, objectUrl } = await blobToImage(blob);
    checkpoint();
    setWorking((prev) =>
      addLayer(
        prev,
        createLayer({ id: nextLayerId(), name: `Layer ${prev.layers.length + 1}`, image, objectUrl, blob }),
      ),
    );
    setDirty(true);
  }

  // Per-layer blend mode (sc-6120): metadata only — the layer's <KonvaImage> node +
  // the flatten compositor both apply it via globalCompositeOperation.
  function setLayerBlend(id, blendMode) {
    if (!workingRef.current) return;
    checkpoint();
    setWorking((prev) => setLayerProps(prev, id, { blendMode }));
    setDirty(true);
  }

  // ── Per-layer transform (sc-6120) ─────────────────────────────────────────
  // The Transform tool binds a Konva Transformer to the ACTIVE layer's node. The
  // node renders from `layer.transform` (x/y/scale/rotation); on drag/transform end
  // we read the node back into the layer's transform metadata. The bitmap is never
  // resampled — the transform is baked only at flatten time (compositeLayersToCanvas
  // already honors it, matching the live node 1:1).
  function startTransform() {
    if (working) setTool("transform");
  }

  function commitActiveLayerTransform() {
    const node = imageNodeRef.current;
    const layer = activeLayerOf(workingRef.current);
    if (!node || !layer) return;
    const transform = {
      x: node.x(),
      y: node.y(),
      scaleX: node.scaleX(),
      scaleY: node.scaleY(),
      rotation: node.rotation(),
    };
    checkpoint();
    setWorking((prev) => setLayerProps(prev, layer.id, { transform }));
    setDirty(true);
  }

  function resetActiveLayerTransform() {
    const layer = activeLayerOf(workingRef.current);
    if (!layer) return;
    checkpoint();
    setWorking((prev) => setLayerProps(prev, layer.id, { transform: identityTransform() }));
    setDirty(true);
  }

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
    checkpoint();
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
    checkpoint();
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
    checkpoint();
    updateBoxRect(id, rect);
  }

  // Selecting a palette color sets the color for new boxes and recolors the
  // selected box (the palette acts on the active box). Stored uppercase so the
  // box stays valid per `isValidHexColor` even from a lowercase <input type=color>.
  function chooseBoxColor(color) {
    const value = color.toUpperCase();
    // Recoloring the active box is an undoable step; setting the color for future
    // boxes (no selection) is not — it changes no committed state.
    if (selectedBoxId) checkpoint();
    setBoxColor(value);
    if (selectedBoxId) {
      setBoxes((prev) => prev.map((box) => (box.id === selectedBoxId ? { ...box, color: value } : box)));
    }
  }

  function deleteBox(id) {
    if (!id) return;
    checkpoint();
    setBoxes((prev) => prev.filter((box) => box.id !== id));
    boxNodeRefs.current.delete(id);
    setSelectedBoxId((cur) => (cur === id ? null : cur));
  }

  function clearBoxes() {
    if (boxes.length) checkpoint();
    setBoxes([]);
    boxNodeRefs.current.clear();
    setSelectedBoxId(null);
    setBoxDraft(null);
  }

  // Flatten the visible layer stack onto a fresh canvas at the document size
  // (sc-6117). The layers' images are already decoded, so this is synchronous;
  // callers toBlob it (Save / Download / AI-op source) or paint overlays on top
  // first (the box-keyed edit). The shared composite behind every editor export.
  function compositeToCanvas(work = working) {
    const canvas = document.createElement("canvas");
    canvas.width = work.width;
    canvas.height = work.height;
    compositeLayersToCanvas(canvas.getContext("2d"), work.layers, { visibleOnly: true });
    return canvas;
  }

  // Rasterize the composited document + the colored boxes into one PNG File (sc-6093).
  // This is an ephemeral pass-through reference — staged as scratch, never saved
  // to the Library — that the edit model reads as color-keyed regions.
  function bakeBoxesToFile() {
    return new Promise((resolve, reject) => {
      const canvas = compositeToCanvas();
      paintBoxesOnContext(canvas.getContext("2d"), boxes);
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
      // The boxes overlay the whole document → the baked composite is the source and
      // the re-rendered result flattens the stack to one base layer (sc-6119).
      layerSource: "composite",
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
    selectPointerDown(event);
    boxPointerDown(event);
  }
  function handleStagePointerMove(event) {
    maskPointerMove(event);
    selectPointerMove(event);
    boxPointerMove(event);
  }
  function handleStagePointerUp(event) {
    maskPointerUp(event);
    selectPointerUp(event);
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
    if (!maskMode || maskSubTool !== "brush" || !working) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    maskPaintingRef.current = true;
    setMaskLines((prev) => [...prev, { points: [pt.x, pt.y], size: maskBrush, erase: maskErase }]);
  }

  function maskPointerMove(event) {
    if (!maskMode || maskSubTool !== "brush" || !maskPaintingRef.current) return;
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
    setMaskBaseImage(null);
    setMaskOverlay(null);
  }

  // ── Smart-select box (sc-3751) ────────────────────────────────────────────
  // A box-drag sub-mode of the mask tool: drag a selection rect, then on release run the SAM3
  // `image_segment` job over it. Mirrors the sc-6090 box draw, but transient (one rect → one run).
  function selectPointerDown(event) {
    if (!maskMode || maskSubTool !== "select" || !working) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    selectDrawingRef.current = true;
    selectStartRef.current = pt;
    setSelectDraft({ x: pt.x, y: pt.y, width: 0, height: 0 });
  }

  function selectPointerMove(event) {
    if (!selectDrawingRef.current) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    setSelectDraft(rectFromPoints(selectStartRef.current, pt));
  }

  function selectPointerUp() {
    if (!selectDrawingRef.current) return;
    selectDrawingRef.current = false;
    const draft = selectDraft;
    setSelectDraft(null);
    // Discard a click / sub-minimum smudge; otherwise run segmentation over the box.
    if (!draft || draft.width < MIN_BOX_PX || draft.height < MIN_BOX_PX) return;
    const rect = clampRectToCanvas(draft, working.width, working.height);
    runSmartSelect(rect);
  }

  // Rasterize the brush strokes to a mask PNG File aligned to the working bitmap:
  // white = edit region on black. Erase strokes punch holes (destination-out on a
  // transparent scratch), then it's flattened onto black so the worker's convert("L")
  // reads white-on-black. Mirrors the same compositing as the on-canvas preview.
  // Composite the current mask (smart-select base + brush strokes) onto a fresh
  // white-on-black canvas at working dims. Shared by the inpaint upload + the mask
  // refine ops (sc-6110). White = edit region; erased holes flatten to black (=keep).
  function rasterizeMaskToCanvas() {
    const scratch = document.createElement("canvas");
    scratch.width = working.width;
    scratch.height = working.height;
    const sctx = scratch.getContext("2d");
    // Smart-select base first (sc-3751): the white-on-black SAM3 mask underlays the brush strokes,
    // so paint strokes add to it and erase strokes (destination-out) carve it back. Its opaque
    // black areas are harmless — the final flatten is onto black anyway.
    if (maskBaseImage) sctx.drawImage(maskBaseImage, 0, 0);
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
    return out;
  }

  function rasterizeMaskToFile() {
    return new Promise((resolve, reject) => {
      rasterizeMaskToCanvas().toBlob((blob) => {
        if (!blob) {
          reject(new Error("Could not encode the mask."));
          return;
        }
        resolve(new File([blob], "mask.png", { type: "image/png" }));
      }, "image/png");
    });
  }

  // ── AI ops on the working image (sc-2432 seam) ────────────────────────────
  // Flatten the composited document to a PNG File. `filename` overrides the name
  // (Save/Download use the "-edited" name; the AI-op scratch upload doesn't care).
  const workingImageToFile = useCallback(
    (filename) => {
      return new Promise((resolve, reject) => {
        if (!working) {
          reject(new Error("No working image."));
          return;
        }
        const canvas = compositeToCanvas(working);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [working],
  );

  // Stage the working image as a scratch asset, start a worker job against it, and
  // track it. The watcher below loads the result back and purges scratch + result —
  // intermediates never persist; only Save (sc-2434) lands a Library asset.
  const runAiOp = useCallback(
    async ({
      buildBody,
      label,
      edit,
      endpoint = "/api/v1/jobs",
      maskFile = null,
      sourceFile = null,
      onComplete = null,
      // The tool-interaction matrix (sc-6119): "activeLayer" ops stage the active
      // layer and write the result back to it (dims unchanged); "composite" ops
      // stage the flattened document and the result becomes a single new base layer.
      layerSource = "activeLayer",
    }) => {
      if (!working || aiOp || !activeProject) return;
      // A composite-source op flattens the stack into one base layer — confirm first.
      if (layerSource === "composite" && !confirmFlatten()) return;
      setStatus({ loading: false, error: "" });
      const targetLayerId = activeLayerOf(working)?.id ?? null;
      // Stage the source (and, for a masked edit, the mask) as scratch assets. An
      // explicit sourceFile (e.g. the box-baked pass-through, sc-6093) wins; else
      // stage the composite (flatten ops) or just the active layer (active-layer ops).
      let scratch;
      let maskScratch = null;
      try {
        const staged =
          sourceFile ?? (layerSource === "composite" ? await workingImageToFile() : await activeLayerToFile());
        scratch = await importAsset(staged, { throwOnError: true });
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
        setAiOp({
          jobId: job.id,
          scratch,
          maskScratch,
          source: working.source,
          label,
          edit,
          onComplete,
          // How the watcher writes the result back: active-layer ops replace the
          // target layer's bitmap; composite ops flatten the stack to one layer.
          writeBack: onComplete ? null : layerSource === "composite" ? "document" : "activeLayer",
          targetLayerId,
        });
        setTool("move");
      } catch (err) {
        purgeAsset(scratch).catch(() => {});
        if (maskScratch) purgeAsset(maskScratch).catch(() => {});
        setStatus({ loading: false, error: `Could not start ${label}: ${err.message || err}` });
      }
    },
    [working, aiOp, activeProject, workingImageToFile, activeLayerToFile, confirmFlatten, importAsset, token, purgeAsset],
  );

  function runUpscale() {
    const valid = upscaleFactorsForEngine(upscaleEngine);
    const factor = valid.includes(upscaleFactor) ? upscaleFactor : valid[0];
    const softness = upscaleEngineHasSoftness(upscaleEngine) ? upscaleSoftness : undefined;
    runAiOp({
      label: "upscale",
      // Upscale changes dimensions → document-level: flatten the stack, upscale once.
      layerSource: "composite",
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
    // Same-size edit → active layer; a canvas-extend / outpaint (dims change) →
    // document-level flatten (sc-6119 tool matrix).
    const dimsChange = outWidth !== working.width || outHeight !== working.height;
    // A mask is sent only for inpaint-capable models; otherwise it's a whole-image edit (the mask
    // stays as a local guide but isn't uploaded). The mask is brush strokes (sc-2436) and/or a
    // smart-select base (sc-3751), composited together by rasterizeMaskToFile.
    const masked = canMask && (maskHasContent(maskLines) || Boolean(maskBaseImage));
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
      layerSource: dimsChange ? "composite" : "activeLayer",
      edit: { op: "edit", model: editModel, prompt, ...(masked ? { masked: true } : {}) },
      maskFile,
      buildBody: (scratch, maskScratch) =>
        buildEditJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          maskAssetId: maskScratch?.id,
          // Multi-reference edit (sc-6107): lead with the working scratch image, then the user's
          // references. Only for a multiReference model with at least one attached reference.
          referenceAssetIds:
            multiRefCapable && refAssetIds.length ? editReferenceIds(scratch.id, refAssetIds) : null,
          model: editModel,
          prompt,
          seed: editSeed,
          width: outWidth,
          height: outHeight,
          fitMode,
        }),
    });
  }

  // Decode a worker mask (white-on-black PNG at working dims) into the editable mask base: a
  // white-on-black canvas for rasterizeMaskToFile + a pink-on-transparent overlay for the preview.
  // Drawn scaled to the working dims defensively (the mask is produced at the working size).
  function loadMaskBase(image) {
    const base = document.createElement("canvas");
    base.width = working.width;
    base.height = working.height;
    base.getContext("2d").drawImage(image, 0, 0, working.width, working.height);
    const overlay = document.createElement("canvas");
    overlay.width = working.width;
    overlay.height = working.height;
    const octx = overlay.getContext("2d");
    octx.drawImage(base, 0, 0);
    const data = octx.getImageData(0, 0, overlay.width, overlay.height);
    tintMaskRgbaInPlace(data.data);
    octx.putImageData(data, 0, 0);
    setMaskBaseImage(base);
    setMaskOverlay(overlay);
  }

  // Install a refined white-on-black mask canvas as the new mask base (sc-6110): it
  // becomes the base + a fresh pink overlay, and the brush strokes are cleared (they
  // are now baked into the canvas). Mirrors loadMaskBase but from a canvas.
  function applyRefinedMask(maskCanvas) {
    const overlay = document.createElement("canvas");
    overlay.width = working.width;
    overlay.height = working.height;
    const octx = overlay.getContext("2d");
    octx.drawImage(maskCanvas, 0, 0);
    const data = octx.getImageData(0, 0, overlay.width, overlay.height);
    tintMaskRgbaInPlace(data.data);
    octx.putImageData(data, 0, 0);
    setMaskBaseImage(maskCanvas);
    setMaskOverlay(overlay);
    setMaskLines([]);
  }

  // Mask refinement (sc-6110): flatten the current mask, run a pure pixel op
  // (feather / grow / shrink / invert), and reinstall it as the base. No-op when no
  // mask exists, except invert (empty mask → select-all).
  function refineMask(op) {
    if (!working) return;
    if (op !== "invert" && !maskHasContent(maskLines) && !maskBaseImage) return;
    const canvas = rasterizeMaskToCanvas();
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, w, h);
    const alpha = maskAlphaFromRgba(imageData.data);
    let refined;
    if (op === "invert") refined = invertAlpha(alpha);
    else if (op === "grow") refined = dilateAlpha(alpha, w, h, maskRefineRadius);
    else if (op === "shrink") refined = erodeAlpha(alpha, w, h, maskRefineRadius);
    else refined = blurAlpha(alpha, w, h, maskRefineRadius);
    writeMaskAlphaToRgba(imageData.data, refined);
    ctx.putImageData(imageData, 0, 0);
    applyRefinedMask(canvas);
  }

  // Run the SAM3 image_segment job over the selection box (sc-3751): stage the working image, post
  // the job, and on completion load the returned binary mask as an editable base under the strokes.
  // It does NOT replace the working image (onComplete owns the result), so the session is unchanged
  // except for the mask layer; the brush/eraser then refines it before Inpaint.
  function runSmartSelect(rect) {
    if (!working || aiOp || !canMask) return;
    const box = rectToSegmentBox(rect);
    runAiOp({
      label: "smart select",
      endpoint: "/api/v1/jobs",
      buildBody: (scratch) =>
        buildSegmentJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          box,
          displayName: working?.source?.name,
        }),
      onComplete: async (resultAsset) => {
        const res = await fetch(assetUrl(resultAsset));
        if (!res.ok) throw new Error(`Failed to load mask (${res.status})`);
        const { image, objectUrl } = await blobToImage(await res.blob());
        loadMaskBase(image);
        URL.revokeObjectURL(objectUrl); // the pixels are copied into the base/overlay canvases
        // Return to the mask tool so the auto-mask can be refined with the brush/eraser.
        setTool("edit");
        setMaskMode(true);
        setMaskSubTool("brush");
      },
    });
  }

  // When the in-flight op's job terminates, load the result back into the working
  // image (on success) and purge the ephemeral scratch + result assets.
  useEffect(() => {
    if (!aiOp?.jobId) return;
    const job = jobs?.find((item) => item.id === aiOp.jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    const { scratch, maskScratch, source, edit, onComplete, writeBack, targetLayerId } = aiOp;
    setAiOp(null); // stop tracking immediately so this can't re-enter on the next jobs tick
    const resultAsset = job.status === "completed" ? job.result?.assets?.[0] ?? null : null;
    (async () => {
      try {
        if (!resultAsset) {
          setStatus({ loading: false, error: job.error ?? job.message ?? "The operation failed." });
          return;
        }
        // Smart-select (sc-3751): the caller's `onComplete` consumes the result asset itself (loads
        // the mask into the mask layer) — it does NOT replace the working image, so skip the install
        // / history / dirty path entirely.
        if (onComplete) {
          await onComplete(resultAsset);
          return;
        }
        const res = await fetch(assetUrl(resultAsset));
        if (!res.ok) throw new Error(`Failed to load result (${res.status})`);
        const blob = await res.blob();
        const { image, objectUrl } = await blobToImage(blob);
        checkpoint();
        // Active-layer op (same-size edit / detail) → write the result back into the
        // target layer, preserving the rest of the stack; document op (upscale /
        // outpaint / box edit) → flatten the stack into one new base layer (sc-6119).
        if (writeBack === "activeLayer" && targetLayerId && layerById(workingRef.current, targetLayerId)) {
          replaceLayerImage(targetLayerId, image, objectUrl, blob);
        } else {
          installWorkingImage(image, objectUrl, blob, source);
        }
        if (edit) setEdits((prev) => [...prev, edit]);
        setDirty(true);
      } catch (err) {
        setStatus({ loading: false, error: err.message || "The operation failed." });
      } finally {
        if (scratch) purgeAsset(scratch).catch(() => {});
        if (maskScratch) purgeAsset(maskScratch).catch(() => {});
        if (resultAsset) purgeAsset(resultAsset).catch(() => {});
      }
    })();
  }, [aiOp, jobs, installWorkingImage, replaceLayerImage, purgeAsset, checkpoint]);

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
          layers: working.layers,
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
          <button
            aria-pressed={shortcutsOpen}
            className={shortcutsOpen ? "image-editor-help active" : "image-editor-help"}
            onClick={() => setShortcutsOpen((on) => !on)}
            title="Keyboard shortcuts (?)"
            type="button"
          >
            ⌨
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
              <button
                className="image-editor-undo"
                disabled={!historyFlags.canUndo || Boolean(aiOp)}
                onClick={undo}
                title="Undo (⌘Z)"
                type="button"
              >
                Undo
              </button>
              <button
                className="image-editor-redo"
                disabled={!historyFlags.canRedo || Boolean(aiOp)}
                onClick={redo}
                title="Redo (⇧⌘Z / Ctrl+Y)"
                type="button"
              >
                Redo
              </button>
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
            draggable={tool !== "crop" && tool !== "boxes" && tool !== "transform" && !maskMode}
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
              {/* Editor layers (sc-6117): one <KonvaImage> per raster layer, bottom→top,
                  honoring per-layer visibility / opacity / blend / transform. The ACTIVE
                  layer carries the color-grade filter + the cached node ref (the live
                  preview) — multi-layer creation + selection arrive with sc-6118/6119. */}
              {working.layers.map((layer) => {
                const isActive = layer.id === working.activeLayerId;
                const t = layer.transform;
                return (
                  <KonvaImage
                    key={layer.id}
                    globalCompositeOperation={layer.blendMode}
                    height={layer.image.naturalHeight}
                    image={layer.image}
                    name="editor-image"
                    opacity={layer.opacity}
                    rotation={t.rotation}
                    scaleX={t.scaleX}
                    scaleY={t.scaleY}
                    visible={layer.visible}
                    width={layer.image.naturalWidth}
                    x={t.x}
                    y={t.y}
                    {...(isActive
                      ? {
                          colorAdjust,
                          gradeMode: colorMode,
                          gradeLevels: levels,
                          gradeCurves: curves,
                          filters: [konvaColorFilter],
                          ref: imageNodeRef,
                        }
                      : {})}
                    {...(isActive && tool === "transform"
                      ? { draggable: true, onDragEnd: commitActiveLayerTransform, onTransformEnd: commitActiveLayerTransform }
                      : {})}
                  />
                );
              })}
              {tool === "transform" ? (
                // Per-layer transform (sc-6120): move / scale / rotate the active layer.
                <Transformer anchorSize={8} borderStroke="#ffffff" ref={layerTransformerRef} rotateEnabled />
              ) : null}
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
            {canMask && (maskLines.length || maskOverlay) ? (
              // Isolated layer so the eraser's destination-out clears only the mask
              // overlay, never the image beneath it. The smart-select base (sc-3751)
              // renders first, with the brush strokes (and their erases) composited on top.
              <Layer listening={false}>
                {maskOverlay ? (
                  <KonvaImage height={working.height} image={maskOverlay} width={working.width} x={0} y={0} />
                ) : null}
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
            {tool === "edit" && maskMode && maskSubTool === "select" && selectDraft ? (
              // Live smart-select box preview (sc-3751), image-pixel coords like the crop rect.
              <Layer listening={false}>
                <Rect
                  dash={[8, 6]}
                  fill="rgba(255,40,120,0.12)"
                  height={selectDraft.height}
                  stroke="rgba(255,40,120,0.9)"
                  strokeWidth={2 / view.scale}
                  width={selectDraft.width}
                  x={selectDraft.x}
                  y={selectDraft.y}
                />
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

        {shortcutsOpen ? (
          <div className="image-editor-shortcuts" role="dialog" aria-label="Keyboard shortcuts">
            <div className="image-editor-shortcuts-head">
              <span>Keyboard shortcuts</span>
              <button onClick={() => setShortcutsOpen(false)} title="Close (Esc)" type="button">
                ✕
              </button>
            </div>
            <div className="image-editor-shortcuts-body">
              {EDITOR_SHORTCUTS.map((section) => (
                <div className="image-editor-shortcuts-group" key={section.group}>
                  <h4>{section.group}</h4>
                  {section.items.map((item) => (
                    <div className="image-editor-shortcut-row" key={item.label}>
                      <span className="image-editor-shortcut-keys">
                        {item.keys.map((cap) => (
                          <kbd key={cap}>{cap}</kbd>
                        ))}
                      </span>
                      <span className="image-editor-shortcut-label">{item.label}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {working ? (
          <LayersPanel
            layers={working.layers}
            activeLayerId={working.activeLayerId}
            busy={Boolean(aiOp)}
            onSelect={selectLayer}
            onToggleVisible={toggleLayerVisible}
            onSetOpacity={changeLayerOpacity}
            onSetBlend={setLayerBlend}
            onRename={renameLayer}
            onReorder={reorderLayer}
            onAdd={addBlankLayer}
            onDelete={deleteLayer}
            onDuplicate={duplicateLayerById}
          />
        ) : null}

        {working ? (
          <aside className="image-editor-toolbar" aria-label="Editor tools">
            <button
              className={tool === "move" ? "image-editor-tool active" : "image-editor-tool"}
              onClick={cancelCrop}
              title="Move / pan (M)"
              type="button"
            >
              Move
            </button>
            <button
              className={tool === "transform" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={startTransform}
              title="Transform the active layer — move / scale / rotate (T)"
              type="button"
            >
              Transform
            </button>
            <button
              className={tool === "crop" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={startCrop}
              title="Crop (C)"
              type="button"
            >
              Crop
            </button>
            <button
              className={tool === "upscale" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp || Boolean(macUpscaleBlock)}
              onClick={() => setTool("upscale")}
              title={macUpscaleBlock ? macUpscaleBlock.text : "Upscale (U)"}
              type="button"
            >
              Upscale
            </button>
            <button
              className={tool === "detail" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp || detailModels.length === 0}
              onClick={() => setTool("detail")}
              title="Detail enhance — tile-ControlNet refine (D)"
              type="button"
            >
              Detail
            </button>
            <button
              className={tool === "color" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={startColorGrade}
              title="Color grade (G)"
              type="button"
            >
              Color
            </button>
            <button
              className={tool === "edit" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={() => setTool("edit")}
              title="AI prompt edit (E)"
              type="button"
            >
              AI Edit
            </button>
            <button
              className={tool === "boxes" ? "image-editor-tool active" : "image-editor-tool"}
              disabled={!!aiOp}
              onClick={selectBoxTool}
              title="Box layout — draw colored regions, color-keyed edit / Ideogram bbox (B)"
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

        {tool === "transform" && working ? (
          <div className="image-editor-cropbar">
            <span className="image-editor-cropdims">
              Drag, scale or rotate <strong>{activeLayerOf(working)?.name}</strong> on the canvas
            </span>
            <button onClick={resetActiveLayerTransform} type="button">
              Reset transform
            </button>
            <button onClick={() => setTool("move")} type="button">
              Done
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
            <div className="image-editor-color-modes" role="group" aria-label="Color mode">
              {[
                ["adjust", "Adjust"],
                ["levels", "Levels"],
                ["curves", "Curves"],
              ].map(([mode, label]) => (
                <button
                  key={mode}
                  className={colorMode === mode ? "active" : ""}
                  onClick={() => setColorMode(mode)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>

            {colorMode === "adjust"
              ? COLOR_ADJUSTMENTS.map(({ key, label }) => (
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
                ))
              : null}

            {colorMode === "levels" || colorMode === "curves" ? (
              <select
                aria-label="Channel"
                className="image-editor-editmodel"
                onChange={(event) => setColorChannel(event.target.value)}
                value={colorChannel}
              >
                <option value="master">Master</option>
                <option value="r">Red</option>
                <option value="g">Green</option>
                <option value="b">Blue</option>
              </select>
            ) : null}

            {colorMode === "levels" ? (
              <>
                <canvas className="image-editor-histogram" height={56} ref={histogramRef} width={200} />
                <label className="image-editor-slider" title="Black point">
                  <span className="image-editor-slider-label">Black</span>
                  <input
                    aria-label="Black point"
                    max={254}
                    min={0}
                    onChange={(event) => setLevelsValue("black", Number(event.target.value))}
                    step={1}
                    type="range"
                    value={levels[colorChannel].black}
                  />
                  <span className="image-editor-slider-value">{levels[colorChannel].black}</span>
                </label>
                <label className="image-editor-slider" title="Gamma (midtones)">
                  <span className="image-editor-slider-label">Gamma</span>
                  <input
                    aria-label="Gamma"
                    max={2.5}
                    min={0.1}
                    onChange={(event) => setLevelsValue("gamma", Number(event.target.value))}
                    step={0.01}
                    type="range"
                    value={levels[colorChannel].gamma}
                  />
                  <span className="image-editor-slider-value">{levels[colorChannel].gamma.toFixed(2)}</span>
                </label>
                <label className="image-editor-slider" title="White point">
                  <span className="image-editor-slider-label">White</span>
                  <input
                    aria-label="White point"
                    max={255}
                    min={1}
                    onChange={(event) => setLevelsValue("white", Number(event.target.value))}
                    step={1}
                    type="range"
                    value={levels[colorChannel].white}
                  />
                  <span className="image-editor-slider-value">{levels[colorChannel].white}</span>
                </label>
              </>
            ) : null}

            {colorMode === "curves" ? (
              <CurveEditor
                points={curves[colorChannel]}
                stroke={channelStroke}
                onChange={(points) => setCurves((prev) => ({ ...prev, [colorChannel]: points }))}
              />
            ) : null}

            <button disabled={activeGradeIsIdentity()} onClick={resetActiveColorMode} type="button">
              Reset
            </button>
            <button className="primary" disabled={activeGradeIsIdentity()} onClick={applyColorGrade} type="button">
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
                      title="Mask a region to confine the edit (inpaint): paint it, or smart-select with a box"
                      type="button"
                    >
                      {maskHasContent(maskLines) || maskBaseImage ? "Mask ✓" : "Mask"}
                    </button>
                    {maskMode ? (
                      <>
                        {smartSelectSupported ? (
                          <>
                            <button
                              className={maskSubTool === "brush" ? "active" : ""}
                              onClick={() => setMaskSubTool("brush")}
                              title="Paint the mask by hand"
                              type="button"
                            >
                              Brush
                            </button>
                            <button
                              className={maskSubTool === "select" ? "active" : ""}
                              disabled={aiOp?.label === "smart select"}
                              onClick={() => {
                                setMaskSubTool("select");
                                setMaskErase(false);
                              }}
                              title="Smart-select: drag a box around an object to auto-mask it (SAM3)"
                              type="button"
                            >
                              {aiOp?.label === "smart select" ? "Segmenting…" : "Smart select"}
                            </button>
                          </>
                        ) : null}
                        {!smartSelectSupported || maskSubTool === "brush" ? (
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
                          </>
                        ) : (
                          <span className="image-editor-hint">Drag a box around an object</span>
                        )}
                        <button
                          disabled={!maskLines.length && !maskBaseImage}
                          onClick={clearMask}
                          type="button"
                        >
                          Clear
                        </button>
                        {/* Mask refinement (sc-6110): post-process the current mask. */}
                        <label className="image-editor-slider" title="Feather / grow / shrink radius (px)">
                          <span className="image-editor-slider-label">Refine</span>
                          <input
                            aria-label="Mask refine radius"
                            max={40}
                            min={1}
                            onChange={(event) => setMaskRefineRadius(Number(event.target.value))}
                            step={1}
                            type="range"
                            value={maskRefineRadius}
                          />
                          <span className="image-editor-slider-value">{maskRefineRadius}</span>
                        </label>
                        <button
                          disabled={!maskHasContent(maskLines) && !maskBaseImage}
                          onClick={() => refineMask("feather")}
                          title="Feather (soften) the mask edges"
                          type="button"
                        >
                          Feather
                        </button>
                        <button
                          disabled={!maskHasContent(maskLines) && !maskBaseImage}
                          onClick={() => refineMask("grow")}
                          title="Grow the selection (dilate)"
                          type="button"
                        >
                          Grow
                        </button>
                        <button
                          disabled={!maskHasContent(maskLines) && !maskBaseImage}
                          onClick={() => refineMask("shrink")}
                          title="Shrink the selection (erode)"
                          type="button"
                        >
                          Shrink
                        </button>
                        <button onClick={() => refineMask("invert")} title="Invert the selection" type="button">
                          Invert
                        </button>
                      </>
                    ) : null}
                  </>
                ) : null}
                {multiRefCapable ? (
                  <span className="image-editor-refs" aria-label="Reference images">
                    <span className="image-editor-slider-label">Reference</span>
                    {refAssetIds.map((id) => {
                      const asset = imageAssets.find((item) => item.id === id);
                      return (
                        <span className="image-editor-ref-chip" key={id}>
                          {asset ? <img alt="" src={assetUrl(asset)} /> : <span>?</span>}
                          <button
                            aria-label="Remove reference"
                            onClick={() => setRefAssetIds((prev) => prev.filter((other) => other !== id))}
                            title="Remove reference"
                            type="button"
                          >
                            ×
                          </button>
                        </span>
                      );
                    })}
                    <button
                      disabled={refAssetIds.length >= MAX_EDIT_REFERENCES - 1}
                      onClick={() => setRefPickerOpen(true)}
                      title="Condition the edit on reference image(s) — identity/style alongside the working image"
                      type="button"
                    >
                      + Reference
                    </button>
                  </span>
                ) : null}
                <button className="primary" disabled={!editPrompt.trim() || !!aiOp} onClick={runEdit} type="button">
                  {canMask && (maskHasContent(maskLines) || maskBaseImage) ? "Inpaint" : "Edit"}
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
            <button onClick={() => zoomAtCenter(1 / ZOOM_STEP)} title="Zoom out (−)" type="button">
              −
            </button>
            <span className="image-editor-zoom">{Math.round(view.scale * 100)}%</span>
            <button onClick={() => zoomAtCenter(ZOOM_STEP)} title="Zoom in (+)" type="button">
              +
            </button>
            <button onClick={fitToView} title="Fit to view (0)" type="button">
              Fit
            </button>
            <button onClick={actualSize} title="Actual size (1)" type="button">
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

      {refPickerOpen ? (
        <DatasetAddDialog
          assets={assets ?? []}
          characters={characters ?? []}
          confirmLabel="Add"
          eyebrow="Reference"
          fileAccept="image/*"
          fileHint="Drag a reference image here, or"
          // Hide images already attached as references so the library tab only offers new picks.
          memberIds={refAssetIds}
          onAdd={(ids) => {
            setRefPickerOpen(false);
            setRefAssetIds((prev) =>
              Array.from(new Set([...prev, ...ids])).slice(0, MAX_EDIT_REFERENCES - 1),
            );
          }}
          onClose={() => setRefPickerOpen(false)}
          onImport={async (files) => {
            // Upload dropped images into the project, then attach them as references (sc-6107).
            setRefPickerOpen(false);
            const imported = await Promise.all(
              Array.from(files ?? []).map((file) => importAsset(file).catch(() => null)),
            );
            const ids = imported.filter(Boolean).map((asset) => asset.id);
            if (ids.length) {
              setRefAssetIds((prev) =>
                Array.from(new Set([...prev, ...ids])).slice(0, MAX_EDIT_REFERENCES - 1),
              );
            }
          }}
          title="Add reference image"
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
