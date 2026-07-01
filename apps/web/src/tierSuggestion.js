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
// `footprint.residentMemoryBytes` / `peakMemoryBytes` are the MEASURED memory fields sc-8516 populates
// (for the tiers it measured on-device; the rest stay null and fall back to the calibrated estimate).
//
// sc-8516 CALIBRATION (basis for the constants below): on-device measurement of steady-state resident
// + load+gen peak GPU memory (harness crates/sceneworks-worker/src/footprint_measure.rs — ONE tier per
// fresh process, resident sampled post-gen AFTER releasing the transient cache, peak = load+gen
// high-water — using the MLX counters mlx_rs::memory::{get_active_memory, get_peak_memory} the worker
// already publishes). Measured set: sdxl/q8, z_image/q4, z_image_turbo/q4, lens_turbo/q4. Two findings:
//   1. resident ≈ on-disk size (ratio 0.81–1.01, mean 0.94) — the old disk×1.5 resident estimate was
//      too high; packed weights sit resident at roughly their on-disk size.
//   2. peak − resident is a FIXED 14.04 GiB transient (1024² VAE decode + attention working set),
//      measured to within ~4 MiB across a 3.9→16.5 GiB resident spread over 3 different VAEs. It is
//      genuinely resolution-bound, NOT weight-bound (a real physical property of the 1024² decode, not
//      a "resident + constant" measurement artifact) — so peak = resident + a fixed addend, not ×N.
// The suggestion therefore budgets against PEAK (the real ceiling a generation must fit).

import { tierQuantize } from "./quantTier.js";

// Fidelity order, HIGHEST first. The suggestion walks this list and picks the first tier that both
// exists on the model AND fits memory; bf16 is preferred, then q8, then q4 (the smallest, always-fits
// fallback). Any declared tier not in this list is considered lowest-fidelity and only chosen last.
const TIER_FIDELITY = ["bf16", "q8", "q4"];

// Resident-memory estimate multiplier over on-disk size, used ONLY when a variant lacks any measured
// footprint. CALIBRATED by sc-8516 from on-device measurement (harness: crates/sceneworks-worker/src/
// footprint_measure.rs). Across sdxl-q8 / z-image-q4 / z-image-turbo-q4 / lens-turbo-q4 the measured
// steady-state RESIDENT/disk ratio was 0.81–1.01 (mean 0.94): packed weights sit resident at roughly
// their on-disk size, NOT the old 1.5× guess. We estimate resident ≈ disk × 1.0, then add the fixed
// transient below (which is where the real generation headroom lives).
export const DISK_TO_RESIDENT_MULTIPLIER = 1.0;

// Fixed transient working-set (bytes) a single 1024² generation needs ON TOP OF the resident weights —
// VAE decode buffers + attention activations/latents + framework scratch. sc-8516 measured this as the
// PEAK − RESIDENT gap and found it 14.04 GiB to within ~4 MiB across a 3.9→16.5 GiB resident spread
// (sdxl / z-image / lens, 3 different VAEs) — genuinely size-INDEPENDENT and resolution-bound, so it is
// modeled as a fixed addend, not a multiplier. 14 GiB ≈ the measured value, used directly. This is what
// makes the estimated budget (disk×MULT + transient) track the MEASURED peak the RAM suggestion must
// actually fit — the true install-time/run ceiling.
export const TRANSIENT_HEADROOM_BYTES = 14 * 1024 * 1024 * 1024;

// Fraction of detected unified/GPU memory a tier's peak footprint must fit UNDER to be suggested. The
// remainder is left for the OS, other apps, and margin. sc-8516 raised this from 0.8 → 0.9: because the
// budget is now peak-inclusive (weights + the fixed transient) rather than resident-only, the extra
// slack the old 0.8 baked in for un-modeled transient is already accounted for explicitly, and 0.9
// keeps the suggestion from needlessly under-picking on right-sized hardware. So on a 32 GB Mac a tier
// "fits" only if its peak is under 32 × 0.9 = 28.8 GB.
export const MEMORY_HEADROOM_FRACTION = 0.9;

const BYTES_PER_GB = 1024 * 1024 * 1024;

// A variant's PEAK memory requirement in bytes (the ceiling the suggestion must fit), or null when it
// can't be estimated. Basis, in priority order:
//   1. `footprint.peakMemoryBytes` — the MEASURED load+gen high-water mark, used verbatim (sc-8516).
//      This is the true ceiling: a tier whose peak OOMs during generation must not be suggested.
//   2. `footprint.residentMemoryBytes` + TRANSIENT_HEADROOM_BYTES — measured resident weights plus the
//      fixed transient working set, when resident was measured but peak was not.
//   3. `footprint.diskSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER + TRANSIENT_HEADROOM_BYTES — the
//      estimate (the common case for un-measured tiers): weights ≈ on-disk size, plus the transient.
//   4. `downloadSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER + TRANSIENT_HEADROOM_BYTES — last-resort when
//      the footprint object is absent but the catalog still knows the tier's download size.
// `measured` reports whether the value came from a measured field (peak or resident) vs the estimate.
export function variantFootprintBytes(variant) {
  const footprint = variant?.footprint;
  const peak = numberOrNull(footprint?.peakMemoryBytes);
  if (peak !== null && peak > 0) {
    return { bytes: peak, measured: true };
  }
  const resident = numberOrNull(footprint?.residentMemoryBytes);
  if (resident !== null && resident > 0) {
    return { bytes: resident + TRANSIENT_HEADROOM_BYTES, measured: true };
  }
  const disk = numberOrNull(footprint?.diskSizeBytes);
  if (disk !== null && disk > 0) {
    return {
      bytes: Math.round(disk * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
      measured: false,
    };
  }
  const download = numberOrNull(variant?.downloadSizeBytes);
  if (download !== null && download > 0) {
    return {
      bytes: Math.round(download * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
      measured: false,
    };
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

// Whether a variant's PEAK footprint fits within `unifiedMemoryGb` with headroom. When the memory
// signal is unknown (null) OR the tier has no estimable footprint, we treat it as fitting — we never
// withhold or block a tier on missing data; the worst case is suggesting a heavier tier.
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
