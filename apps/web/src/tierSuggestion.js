// RAM-based quant-tier download suggestion (sc-8509, epic 8506). The Models page lets a user pick
// which quant tier(s) of a model to DOWNLOAD, with a suggested default: the highest-fidelity tier
// that should fit the host's memory. This module is the pure, unit-testable core of that logic —
// deliberately separate from the React screen (like sc-8515's quantTier.js) so sc-8516 can later
// refine the thresholds/constants in ONE place once measured footprints land.
//
// SUGGEST, NEVER WITHHOLD (epic 8506 decision 1): every tier stays installable regardless of RAM.
// The suggestion only preselects/highlights the recommended tier; it never removes a tier from the
// installable set. `suggestTier` returning a smaller tier does not make bf16 un-installable — the
// screen keeps every tier's checkbox enabled.
//
// Consumes the sc-8508 per-variant catalog shape: each `model.variants[]` entry carries a `variant`
// key (bf16/q8/q4) and a `footprint` object. `footprint.diskSizeBytes` is always populated;
// `footprint.residentMemoryBytes` / `peakMemoryBytes` are currently nullable (sc-8516 fills them in).

import { tierQuantize } from "./quantTier.js";

// Fidelity order, HIGHEST first. The suggestion walks this list and picks the first tier that both
// exists on the model AND fits memory; bf16 is preferred, then q8, then q4 (the smallest, always-fits
// fallback). Any declared tier not in this list is considered lowest-fidelity and only chosen last.
const TIER_FIDELITY = ["bf16", "q8", "q4"];

// Resident-memory estimate multiplier over on-disk size, used ONLY when a variant lacks a measured
// `footprint.residentMemoryBytes`. The weights are resident at roughly their on-disk size, but a
// generation also needs headroom for the text encoder, activations/latents, the framework's working
// buffers, and the OS. 1.5× is a deliberately conservative middle estimate: enough margin that a
// suggested tier is unlikely to OOM, without being so pessimistic it under-suggests on right-sized
// hardware. sc-8516 replaces this path entirely by populating measured residentMemoryBytes.
export const DISK_TO_RESIDENT_MULTIPLIER = 1.5;

// Fraction of detected unified/GPU memory a tier's resident footprint must fit UNDER to be
// suggested. The remainder is left for the OS, other apps, and margin against our own estimate. So
// on a 32 GB Mac a tier is "fits" only if its footprint is under 32 × 0.8 = 25.6 GB. This is the
// headroom knob sc-8516 can tune alongside the multiplier.
export const MEMORY_HEADROOM_FRACTION = 0.8;

const BYTES_PER_GB = 1024 * 1024 * 1024;

// A variant's estimated RESIDENT memory footprint in bytes, or null when it can't be estimated.
// Basis, in priority order:
//   1. `footprint.residentMemoryBytes` — the measured value, used verbatim when present (sc-8516).
//   2. `footprint.diskSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER — the estimate (the common case today,
//      since measured memory is still null).
//   3. `downloadSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER — a last-resort estimate when the footprint
//      object is absent but the catalog still knows the tier's download size.
// `measured` reports which basis was used, so the UI/tests can distinguish measured vs estimated.
export function variantFootprintBytes(variant) {
  const footprint = variant?.footprint;
  const resident = numberOrNull(footprint?.residentMemoryBytes);
  if (resident !== null && resident > 0) {
    return { bytes: resident, measured: true };
  }
  const disk = numberOrNull(footprint?.diskSizeBytes);
  if (disk !== null && disk > 0) {
    return { bytes: Math.round(disk * DISK_TO_RESIDENT_MULTIPLIER), measured: false };
  }
  const download = numberOrNull(variant?.downloadSizeBytes);
  if (download !== null && download > 0) {
    return { bytes: Math.round(download * DISK_TO_RESIDENT_MULTIPLIER), measured: false };
  }
  return null;
}

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// The model's real quant tiers (bf16/q8/q4), in fidelity order (highest first). Excludes the
// single-variant "default" pseudo-tier and any unknown key. Every declared tier is included whether
// or not it's installed — the suggestion is about what to DOWNLOAD, so uninstalled tiers count.
export function declaredTiers(model) {
  if (!model?.hasVariantMatrix || !Array.isArray(model.variants)) {
    return [];
  }
  const keys = model.variants
    .map((variant) => variant?.variant)
    .filter((key) => tierQuantize(key) !== null);
  const unique = [...new Set(keys)];
  return unique.sort((a, b) => fidelityRank(a) - fidelityRank(b));
}

// Rank a tier by fidelity, HIGHEST first (bf16=0). Unknown tiers sort last.
function fidelityRank(tier) {
  const index = TIER_FIDELITY.indexOf(tier);
  return index === -1 ? TIER_FIDELITY.length : index;
}

// Whether a variant's estimated footprint fits within `unifiedMemoryGb` with headroom. When the
// memory signal is unknown (null) OR the tier has no estimable footprint, we treat it as fitting —
// we never withhold or block a tier on missing data; the worst case is suggesting a heavier tier.
export function tierFits(variant, unifiedMemoryGb) {
  if (unifiedMemoryGb == null || !Number.isFinite(unifiedMemoryGb)) {
    return true;
  }
  const footprint = variantFootprintBytes(variant);
  if (footprint === null) {
    return true;
  }
  const budgetBytes = unifiedMemoryGb * BYTES_PER_GB * MEMORY_HEADROOM_FRACTION;
  return footprint.bytes <= budgetBytes;
}

// The suggested default download tier for `model` given the host's `unifiedMemoryGb` (GPU VRAM off
// Mac). Picks the HIGHEST-FIDELITY tier that fits memory with headroom; if none fits (a tiny host,
// or every tier over budget) it falls back to the SMALLEST declared tier so there's always a
// suggestion. Returns null only when the model has no quant matrix.
//
// This never affects installability — it just picks which tier to pre-select/highlight. A 32 GB host
// lands on q4, a 512 GB Studio on bf16, and either can override.
export function suggestTier(model, unifiedMemoryGb) {
  const tiers = declaredTiers(model);
  if (tiers.length === 0) {
    return null;
  }
  const byKey = new Map(
    (model.variants ?? [])
      .filter((variant) => tierQuantize(variant?.variant) !== null)
      .map((variant) => [variant.variant, variant]),
  );
  // `tiers` is already highest-fidelity first; the first that fits wins.
  for (const tier of tiers) {
    const variant = byKey.get(tier);
    if (variant && tierFits(variant, unifiedMemoryGb)) {
      return tier;
    }
  }
  // Nothing fit (every tier over budget) — suggest the smallest so we still preselect something.
  return tiers[tiers.length - 1];
}
