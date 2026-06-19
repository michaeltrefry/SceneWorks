import { describe, expect, it } from "vitest";
import {
  maskAlphaFromRgba,
  writeMaskAlphaToRgba,
  invertAlpha,
  dilateAlpha,
  erodeAlpha,
  blurAlpha,
} from "./maskRefine.js";

// Build a w*h single-channel mask from a 2D array of 0/255 rows.
const mask = (rows) => Uint8ClampedArray.from(rows.flat());

describe("mask refine (sc-6110)", () => {
  it("round-trips a single-channel mask through RGBA", () => {
    const alpha = mask([[0, 255, 128]]);
    const rgba = new Uint8ClampedArray(3 * 4);
    writeMaskAlphaToRgba(rgba, alpha);
    expect([...rgba]).toEqual([0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255]);
    expect([...maskAlphaFromRgba(rgba)]).toEqual([0, 255, 128]);
  });

  it("invert swaps selected and unselected", () => {
    expect([...invertAlpha(mask([[0, 255, 64]]))]).toEqual([255, 0, 191]);
  });

  it("dilate (grow) spreads the white selection by the radius", () => {
    // 5x5, a single white pixel in the center.
    const w = 5;
    const h = 5;
    const a = new Uint8ClampedArray(w * h);
    a[2 * w + 2] = 255;
    const grown = dilateAlpha(a, w, h, 1);
    // A radius-1 dilate fills the 3x3 block around the center (square structuring elt).
    for (let y = 1; y <= 3; y += 1) for (let x = 1; x <= 3; x += 1) expect(grown[y * w + x]).toBe(255);
    expect(grown[0]).toBe(0); // far corner untouched
    // The grown selection has strictly more white than the seed.
    const count = (m) => m.reduce((n, v) => n + (v === 255 ? 1 : 0), 0);
    expect(count(grown)).toBeGreaterThan(count(a));
  });

  it("erode (shrink) eats the selection edges and can clear a thin region", () => {
    const w = 5;
    const h = 5;
    const a = new Uint8ClampedArray(w * h).fill(0);
    // a 3x3 white block centered.
    for (let y = 1; y <= 3; y += 1) for (let x = 1; x <= 3; x += 1) a[y * w + x] = 255;
    const eroded = erodeAlpha(a, w, h, 1);
    // Only the very center survives a radius-1 erode of a 3x3 block.
    expect(eroded[2 * w + 2]).toBe(255);
    expect(eroded[1 * w + 1]).toBe(0);
    const count = (m) => m.reduce((n, v) => n + (v === 255 ? 1 : 0), 0);
    expect(count(eroded)).toBeLessThan(count(a));
  });

  it("dilate then erode (radius 0) are no-ops", () => {
    const a = mask([[0, 255, 0, 255]]);
    expect([...dilateAlpha(a, 4, 1, 0)]).toEqual([...a]);
    expect([...erodeAlpha(a, 4, 1, 0)]).toEqual([...a]);
    expect([...blurAlpha(a, 4, 1, 0)]).toEqual([...a]);
  });

  it("feather (blur) softens a hard edge into a ramp, bounded 0..255", () => {
    const w = 8;
    const h = 1;
    const a = mask([[0, 0, 0, 0, 255, 255, 255, 255]]);
    const soft = blurAlpha(a, w, h, 2, 1);
    // The hard 0→255 step becomes intermediate values straddling the boundary.
    expect(soft[3]).toBeGreaterThan(0);
    expect(soft[3]).toBeLessThan(255);
    expect(soft[4]).toBeGreaterThan(0);
    expect(soft[4]).toBeLessThan(255);
    // Still monotonic non-decreasing left→right and in range.
    for (let i = 1; i < w; i += 1) {
      expect(soft[i]).toBeGreaterThanOrEqual(soft[i - 1]);
      expect(soft[i]).toBeGreaterThanOrEqual(0);
      expect(soft[i]).toBeLessThanOrEqual(255);
    }
  });
});
