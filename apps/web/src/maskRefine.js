// Mask refinement (sc-6110, Workstream F of epic 6087). Pure morphology / blur /
// invert over a single-channel mask — white (255) = edit region, black (0) = keep —
// represented as a flat Uint8 intensity array of length w*h. The editor flattens its
// current mask (smart-select base + brush strokes) to white-on-black, extracts the
// intensity with maskAlphaFromRgba, applies one of these ops, then writes it back as
// the new mask base. React/DOM-free so the pixel math is unit-tested in isolation.

// Extract mask intensity (the R channel — the mask is grayscale) from an RGBA buffer.
export function maskAlphaFromRgba(data) {
  const out = new Uint8ClampedArray(data.length / 4);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) out[j] = data[i];
  return out;
}

// Write a single-channel mask back to RGBA (R=G=B=value, opaque).
export function writeMaskAlphaToRgba(data, alpha) {
  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
    data[i] = alpha[j];
    data[i + 1] = alpha[j];
    data[i + 2] = alpha[j];
    data[i + 3] = 255;
  }
}

// Invert the selection (255 - v): selected ↔ unselected.
export function invertAlpha(alpha) {
  const out = new Uint8ClampedArray(alpha.length);
  for (let i = 0; i < alpha.length; i += 1) out[i] = 255 - alpha[i];
  return out;
}

// Separable grayscale morphology over a (2r+1) window — extreme = Math.max (dilate /
// grow the white selection) or Math.min (erode / shrink). Edges clamp.
function morph(alpha, w, h, radius, extreme) {
  const r = Math.max(0, Math.round(radius));
  if (r === 0) return Uint8ClampedArray.from(alpha);
  const seed = extreme === Math.max ? 0 : 255;
  const tmp = new Uint8ClampedArray(alpha.length);
  for (let y = 0; y < h; y += 1) {
    const row = y * w;
    for (let x = 0; x < w; x += 1) {
      let v = seed;
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      for (let xx = x0; xx <= x1; xx += 1) v = extreme(v, alpha[row + xx]);
      tmp[row + x] = v;
    }
  }
  const out = new Uint8ClampedArray(alpha.length);
  for (let x = 0; x < w; x += 1) {
    for (let y = 0; y < h; y += 1) {
      let v = seed;
      const y0 = Math.max(0, y - r);
      const y1 = Math.min(h - 1, y + r);
      for (let yy = y0; yy <= y1; yy += 1) v = extreme(v, tmp[yy * w + x]);
      out[y * w + x] = v;
    }
  }
  return out;
}

// Grow / shrink the selection by `radius` px.
export function dilateAlpha(alpha, w, h, radius) {
  return morph(alpha, w, h, radius, Math.max);
}
export function erodeAlpha(alpha, w, h, radius) {
  return morph(alpha, w, h, radius, Math.min);
}

// Separable box-blur pass over a (2r+1) window with edge clamping (averaged over the
// in-bounds count so edges don't darken).
function blurPass(src, w, h, r) {
  const tmp = new Float32Array(src.length);
  for (let y = 0; y < h; y += 1) {
    const row = y * w;
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      for (let xx = x0; xx <= x1; xx += 1) sum += src[row + xx];
      tmp[row + x] = sum / (x1 - x0 + 1);
    }
  }
  const out = new Float32Array(src.length);
  for (let x = 0; x < w; x += 1) {
    for (let y = 0; y < h; y += 1) {
      let sum = 0;
      const y0 = Math.max(0, y - r);
      const y1 = Math.min(h - 1, y + r);
      for (let yy = y0; yy <= y1; yy += 1) sum += tmp[yy * w + x];
      out[y * w + x] = sum / (y1 - y0 + 1);
    }
  }
  return out;
}

// Feather (Gaussian-approx edge softening): `passes` box blurs (3 ≈ Gaussian).
export function blurAlpha(alpha, w, h, radius, passes = 3) {
  const r = Math.max(0, Math.round(radius));
  if (r === 0) return Uint8ClampedArray.from(alpha);
  let src = Float32Array.from(alpha);
  for (let p = 0; p < passes; p += 1) src = blurPass(src, w, h, r);
  const out = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 1) out[i] = Math.round(src[i]);
  return out;
}
