import { describe, expect, it } from "vitest";
import {
  DEFAULT_BLEND_MODE,
  identityTransform,
  createLayer,
  singleLayerWorking,
  activeLayerOf,
  layerById,
  addLayer,
  removeLayer,
  duplicateLayer,
  moveLayer,
  setLayerProps,
  setActiveLayer,
  snapshotLayer,
  snapshotLayers,
  sameLayerStack,
  compositeLayersToCanvas,
} from "./imageLayers.js";

// A minimal stand-in for a decoded bitmap — only naturalWidth/Height are read by
// the model, and the compositor passes the object straight to ctx.drawImage.
const fakeImage = (w = 100, h = 80) => ({ naturalWidth: w, naturalHeight: h });

const liveLayer = (id, overrides = {}) =>
  createLayer({ id, image: fakeImage(), objectUrl: `blob:${id}`, blob: { id }, ...overrides });

describe("layer model — creation + degenerate single-layer document (sc-6117)", () => {
  it("createLayer fills defaults and clamps opacity", () => {
    const layer = createLayer({ id: "a", blob: {}, opacity: 5 });
    expect(layer.name).toBe("Layer");
    expect(layer.visible).toBe(true);
    expect(layer.opacity).toBe(1); // clamped from 5
    expect(layer.blendMode).toBe(DEFAULT_BLEND_MODE);
    expect(layer.transform).toEqual(identityTransform());
    expect(createLayer({ id: "b", blob: {}, opacity: -2 }).opacity).toBe(0);
  });

  it("singleLayerWorking mirrors the old single-bitmap document", () => {
    const image = fakeImage(640, 480);
    const work = singleLayerWorking({ id: "bg", image, objectUrl: "blob:x", blob: {}, source: { kind: "upload", name: "p.png" } });
    expect(work.width).toBe(640);
    expect(work.height).toBe(480);
    expect(work.layers).toHaveLength(1);
    expect(work.activeLayerId).toBe("bg");
    expect(activeLayerOf(work).id).toBe("bg");
    expect(work.layers[0].name).toBe("Background");
  });

  it("activeLayerOf falls back to the bottom layer, then null", () => {
    const work = { layers: [liveLayer("a"), liveLayer("b")], activeLayerId: "missing" };
    expect(activeLayerOf(work).id).toBe("a");
    expect(activeLayerOf({ layers: [], activeLayerId: null })).toBeNull();
    expect(activeLayerOf(null)).toBeNull();
  });
});

describe("layer operations — add / delete / duplicate / reorder (sc-6117)", () => {
  const base = () => ({
    width: 100,
    height: 80,
    source: { kind: "blank", name: "x" },
    layers: [liveLayer("a"), liveLayer("b")],
    activeLayerId: "a",
  });

  it("addLayer inserts on top and activates the new layer", () => {
    const next = addLayer(base(), liveLayer("c"));
    expect(next.layers.map((l) => l.id)).toEqual(["a", "b", "c"]);
    expect(next.activeLayerId).toBe("c");
  });

  it("removeLayer drops the layer, returns it for URL revocation, and reseats active", () => {
    const { working, removed } = removeLayer(base(), "a");
    expect(working.layers.map((l) => l.id)).toEqual(["b"]);
    expect(removed.id).toBe("a");
    expect(removed.objectUrl).toBe("blob:a"); // caller revokes this
    expect(working.activeLayerId).toBe("b"); // active followed off the deleted layer
  });

  it("removeLayer refuses to delete the last layer (a document keeps ≥1)", () => {
    const single = { ...base(), layers: [liveLayer("a")], activeLayerId: "a" };
    const { working, removed } = removeLayer(single, "a");
    expect(removed).toBeNull();
    expect(working.layers).toHaveLength(1);
  });

  it("duplicateLayer clones metadata above the source, shares blob, takes a fresh image/url", () => {
    const src = base();
    src.layers[0] = liveLayer("a", { name: "Sky", opacity: 0.5, blendMode: "multiply" });
    const cloneImage = fakeImage();
    const next = duplicateLayer(src, "a", { id: "a2", image: cloneImage, objectUrl: "blob:a2" });
    expect(next.layers.map((l) => l.id)).toEqual(["a", "a2", "b"]);
    const clone = layerById(next, "a2");
    expect(clone.name).toBe("Sky copy");
    expect(clone.opacity).toBe(0.5);
    expect(clone.blendMode).toBe("multiply");
    expect(clone.blob).toBe(src.layers[0].blob); // shared by reference
    expect(clone.image).toBe(cloneImage); // its own decoded bitmap
    expect(clone.objectUrl).toBe("blob:a2");
    expect(next.activeLayerId).toBe("a2");
  });

  it("moveLayer reorders with clamping and is a no-op at the same index", () => {
    expect(moveLayer(base(), "a", 1).layers.map((l) => l.id)).toEqual(["b", "a"]);
    expect(moveLayer(base(), "b", 99).layers.map((l) => l.id)).toEqual(["a", "b"]); // clamped to top
    expect(moveLayer(base(), "a", 0)).toEqual(base()); // unchanged
  });

  it("setLayerProps patches metadata (clamping opacity, merging transform) without touching pixels", () => {
    const next = setLayerProps(base(), "b", { visible: false, opacity: 9, transform: { x: 10 } });
    const b = layerById(next, "b");
    expect(b.visible).toBe(false);
    expect(b.opacity).toBe(1);
    expect(b.transform).toEqual({ ...identityTransform(), x: 10 });
    expect(b.image).toBe(layerById(base(), "b") ? next.layers[1].image : null); // image untouched
  });

  it("setActiveLayer ignores unknown ids", () => {
    expect(setActiveLayer(base(), "b").activeLayerId).toBe("b");
    expect(setActiveLayer(base(), "ghost").activeLayerId).toBe("a");
  });
});

describe("snapshot representation + equality (sc-6117 undo integration)", () => {
  it("snapshotLayer keeps metadata + shared blob, drops live image/url", () => {
    const blob = { tag: 1 };
    const snap = snapshotLayer(liveLayer("a", { blob, opacity: 0.3, name: "n" }));
    expect(snap).toEqual({
      id: "a",
      name: "n",
      blob,
      visible: true,
      opacity: 0.3,
      blendMode: DEFAULT_BLEND_MODE,
      transform: identityTransform(),
    });
    expect(snap.blob).toBe(blob); // shared by reference → bounded memory
    expect("image" in snap).toBe(false);
    expect("objectUrl" in snap).toBe(false);
  });

  it("sameLayerStack tells overlay-only steps from pixel/structure changes", () => {
    const a = liveLayer("a");
    const b = liveLayer("b");
    const live = [a, b];
    // Identical content (snapshot of the same blobs) → equal.
    expect(sameLayerStack(live, snapshotLayers(live))).toBe(true);
    // A different blob (a bitmap edit) → not equal.
    expect(sameLayerStack(live, [{ ...snapshotLayer(a), blob: { other: 1 } }, snapshotLayer(b)])).toBe(false);
    // A metadata change (opacity) → not equal.
    expect(sameLayerStack(live, [{ ...snapshotLayer(a), opacity: 0.5 }, snapshotLayer(b)])).toBe(false);
    // Reorder → not equal.
    expect(sameLayerStack(live, [snapshotLayer(b), snapshotLayer(a)])).toBe(false);
    // Different length → not equal.
    expect(sameLayerStack(live, [snapshotLayer(a)])).toBe(false);
  });
});

describe("compositor — bottom→top, honoring visibility / opacity / blend / order (sc-6117)", () => {
  // A recording 2D-context stand-in: drawImage logs the image drawn and the alpha
  // + blend in force at draw time, so the test sees exactly what was composited.
  const recordingCtx = () => {
    const calls = [];
    let alpha = 1;
    let gco = "source-over";
    return {
      calls,
      get globalAlpha() {
        return alpha;
      },
      set globalAlpha(v) {
        alpha = v;
      },
      get globalCompositeOperation() {
        return gco;
      },
      set globalCompositeOperation(v) {
        gco = v;
      },
      save() {},
      restore() {},
      translate() {},
      rotate() {},
      scale() {},
      drawImage(image) {
        calls.push({ image, alpha, gco });
      },
    };
  };

  it("draws every visible layer bottom→top with its alpha + blend", () => {
    const bottom = liveLayer("a");
    const top = liveLayer("b", { opacity: 0.4, blendMode: "multiply" });
    const ctx = recordingCtx();
    compositeLayersToCanvas(ctx, [bottom, top]);
    expect(ctx.calls).toHaveLength(2);
    expect(ctx.calls[0]).toMatchObject({ image: bottom.image, alpha: 1, gco: "source-over" });
    expect(ctx.calls[1]).toMatchObject({ image: top.image, alpha: 0.4, gco: "multiply" });
  });

  it("skips hidden and fully-transparent layers (visibleOnly), but draws them when asked", () => {
    const visible = liveLayer("a");
    const hidden = liveLayer("b", { visible: false });
    const transparent = liveLayer("c", { opacity: 0 });
    const ctx = recordingCtx();
    compositeLayersToCanvas(ctx, [visible, hidden, transparent]);
    expect(ctx.calls.map((c) => c.image)).toEqual([visible.image]);

    const all = recordingCtx();
    compositeLayersToCanvas(all, [visible, hidden, transparent], { visibleOnly: false });
    expect(all.calls).toHaveLength(3);
  });

  it("skips layers with no decoded image", () => {
    const ctx = recordingCtx();
    compositeLayersToCanvas(ctx, [createLayer({ id: "x", blob: {} })]);
    expect(ctx.calls).toHaveLength(0);
  });
});
