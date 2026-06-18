// Layer-stack model for the Image Editor (sc-6117, Workstream E of epic 6087).
//
// The editor's working document is an ordered raster layer stack composited
// bottom→top. A SINGLE-layer stack is the degenerate case that reproduces the
// pre-layers single-bitmap editor exactly (visually + dimensionally + same Save
// asset). This module is pure — no React, no Konva, no DOM beyond the 2D-canvas
// context the compositor draws into — so the state model and the compositor can
// be unit-tested without mounting the editor.
//
// Shapes:
//   working = { width, height, source, layers: [Layer, …], activeLayerId }
//   Layer   = { id, name, image, objectUrl, blob,
//               visible, opacity, blendMode, transform }
//   transform = { x, y, scaleX, scaleY, rotation }  // rotation in degrees
//
// `image`/`objectUrl` are the LIVE decoded representation (a layer's object URL is
// revoked when the layer is evicted — see the editor's revoke discipline). `blob`
// is the immutable source of truth: it is shared by reference into undo snapshots,
// so retained bitmaps stay bounded the same way the single-bitmap history did.
// A snapshot layer carries only metadata + blob (no image/objectUrl) — see
// `snapshotLayer`.

export const DEFAULT_BLEND_MODE = "source-over";

// Identity transform for a layer that sits 1:1 over the document (the only kind
// sc-6117 produces; non-identity transforms arrive with sc-6120, but the model +
// compositor honor the fields now so the stack is forward-complete).
export function identityTransform() {
  return { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 };
}

function clampOpacity(opacity) {
  if (typeof opacity !== "number" || Number.isNaN(opacity)) return 1;
  return Math.max(0, Math.min(1, opacity));
}

// Normalize a layer spec into a full Layer, filling defaults. Callers supply the
// id (the editor owns a monotonic counter so ids survive undo without colliding).
export function createLayer({
  id,
  name = "Layer",
  image = null,
  objectUrl = null,
  blob = null,
  visible = true,
  opacity = 1,
  blendMode = DEFAULT_BLEND_MODE,
  transform = null,
}) {
  return {
    id,
    name,
    image,
    objectUrl,
    blob,
    visible: visible !== false,
    opacity: clampOpacity(opacity),
    blendMode: blendMode || DEFAULT_BLEND_MODE,
    transform: transform ? { ...identityTransform(), ...transform } : identityTransform(),
  };
}

// Build the degenerate single-layer document from one decoded bitmap. The doc
// dimensions are the bitmap's natural size; the lone layer is the active layer.
export function singleLayerWorking({ id, name = "Background", image, objectUrl, blob, source }) {
  const layer = createLayer({ id, name, image, objectUrl, blob });
  return {
    width: image?.naturalWidth ?? 0,
    height: image?.naturalHeight ?? 0,
    source,
    layers: [layer],
    activeLayerId: layer.id,
  };
}

// The active layer (by id), falling back to the bottom layer, or null.
export function activeLayerOf(working) {
  if (!working || !working.layers?.length) return null;
  return working.layers.find((layer) => layer.id === working.activeLayerId) ?? working.layers[0];
}

export function layerById(working, id) {
  return working?.layers?.find((layer) => layer.id === id) ?? null;
}

function withLayers(working, layers, activeLayerId = working.activeLayerId) {
  return { ...working, layers, activeLayerId };
}

// Insert a new layer on TOP of the stack and make it active (the natural "Add
// layer" semantics — new content lands above and becomes the edit target).
export function addLayer(working, layer) {
  return withLayers(working, [...working.layers, layer], layer.id);
}

// Remove a layer. The last layer cannot be removed (a document always has ≥1
// layer). Returns { working, removed } so the caller can revoke the evicted
// layer's object URL; `removed` is null when the delete was a no-op. When the
// active layer is removed, the active pointer moves to the layer that takes its
// slot (or the new top).
export function removeLayer(working, id) {
  if (!working || working.layers.length <= 1) return { working, removed: null };
  const index = working.layers.findIndex((layer) => layer.id === id);
  if (index < 0) return { working, removed: null };
  const removed = working.layers[index];
  const layers = working.layers.filter((layer) => layer.id !== id);
  let activeLayerId = working.activeLayerId;
  if (activeLayerId === id) {
    const neighbor = layers[Math.min(index, layers.length - 1)];
    activeLayerId = neighbor.id;
  }
  return { working: withLayers(working, layers, activeLayerId), removed };
}

// Duplicate a layer, inserting the clone directly ABOVE the source and making it
// active. The clone shares the source's immutable `blob` by reference but carries
// its OWN freshly-decoded image + object URL (passed in by the caller, which owns
// the decode), so the two layers can later diverge and the URL-revoke-on-evict
// discipline stays one-URL-per-live-layer.
export function duplicateLayer(working, id, { id: cloneId, image, objectUrl }) {
  const index = working.layers.findIndex((layer) => layer.id === id);
  if (index < 0) return working;
  const src = working.layers[index];
  const clone = createLayer({
    id: cloneId,
    name: `${src.name} copy`,
    image,
    objectUrl,
    blob: src.blob,
    visible: src.visible,
    opacity: src.opacity,
    blendMode: src.blendMode,
    transform: { ...src.transform },
  });
  const layers = [...working.layers.slice(0, index + 1), clone, ...working.layers.slice(index + 1)];
  return withLayers(working, layers, clone.id);
}

// Move a layer to a new stack index (clamped). Index 0 = bottom.
export function moveLayer(working, id, toIndex) {
  const from = working.layers.findIndex((layer) => layer.id === id);
  if (from < 0) return working;
  const target = Math.max(0, Math.min(working.layers.length - 1, toIndex));
  if (from === target) return working;
  const layers = [...working.layers];
  const [moved] = layers.splice(from, 1);
  layers.splice(target, 0, moved);
  return withLayers(working, layers);
}

// Patch a single layer's metadata (visible / opacity / name / blendMode /
// transform). Pixel data (image/blob) is never patched here.
export function setLayerProps(working, id, patch) {
  const layers = working.layers.map((layer) => {
    if (layer.id !== id) return layer;
    const next = { ...layer, ...patch };
    if (patch.opacity !== undefined) next.opacity = clampOpacity(patch.opacity);
    if (patch.transform) next.transform = { ...layer.transform, ...patch.transform };
    return next;
  });
  return withLayers(working, layers);
}

export function setActiveLayer(working, id) {
  if (!layerById(working, id)) return working;
  return { ...working, activeLayerId: id };
}

// Strip a live layer down to the snapshot representation: metadata + the shared
// blob, no live image/objectUrl. Undo snapshots hold these (blobs shared by
// reference), so an evicted snapshot is plain garbage with nothing to revoke.
export function snapshotLayer(layer) {
  return {
    id: layer.id,
    name: layer.name,
    blob: layer.blob,
    visible: layer.visible,
    opacity: layer.opacity,
    blendMode: layer.blendMode,
    transform: { ...layer.transform },
  };
}

export function snapshotLayers(layers) {
  return (layers ?? []).map(snapshotLayer);
}

function sameTransform(a, b) {
  const x = a ?? identityTransform();
  const y = b ?? identityTransform();
  return x.x === y.x && x.y === y.y && x.scaleX === y.scaleX && x.scaleY === y.scaleY && x.rotation === y.rotation;
}

// Whether two layer lists are pixel- and metadata-identical (same ids/order, same
// blob by reference, same visible/opacity/blend/transform). Used by restore to
// tell an overlay-only step (no decode, no refit) from a bitmap/structure change.
// Ignores live image identity — only the blob (source of truth) matters.
export function sameLayerStack(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i];
    const y = b[i];
    if (
      x.id !== y.id ||
      x.blob !== y.blob ||
      x.visible !== y.visible ||
      x.opacity !== y.opacity ||
      x.blendMode !== y.blendMode ||
      !sameTransform(x.transform, y.transform)
    ) {
      return false;
    }
  }
  return true;
}

// Composite a layer stack into a 2D-canvas context, bottom→top, honoring
// visibility, opacity, blend mode, and per-layer transform. The shared flatten
// behind Save / Download / Download-as-source for AI ops. `visibleOnly` (default)
// skips hidden and fully-transparent layers. The context must already be sized to
// the document; layers' `image` must be decoded (HTMLImageElement / canvas).
export function compositeLayersToCanvas(ctx, layers, { visibleOnly = true } = {}) {
  for (const layer of layers ?? []) {
    if (!layer.image) continue;
    if (visibleOnly && (!layer.visible || layer.opacity <= 0)) continue;
    const t = layer.transform ?? identityTransform();
    ctx.save();
    ctx.globalAlpha = clampOpacity(layer.opacity);
    ctx.globalCompositeOperation = layer.blendMode || DEFAULT_BLEND_MODE;
    ctx.translate(t.x || 0, t.y || 0);
    if (t.rotation) ctx.rotate((t.rotation * Math.PI) / 180);
    ctx.scale(t.scaleX ?? 1, t.scaleY ?? 1);
    ctx.drawImage(layer.image, 0, 0);
    ctx.restore();
  }
}
