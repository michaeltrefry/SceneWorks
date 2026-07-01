import { describe, expect, it } from "vitest";
import {
  DISK_TO_RESIDENT_MULTIPLIER,
  MEMORY_HEADROOM_FRACTION,
  TRANSIENT_HEADROOM_BYTES,
  declaredTiers,
  suggestTier,
  tierFits,
  variantFootprintBytes,
} from "./tierSuggestion.js";

const GB = 1024 * 1024 * 1024;

// Build a /models-shaped quant-matrix model. Each entry in `tiers` may be a bare tier key (which
// gets a disk-only footprint sized from `diskGb`) or an object { variant, diskGb, residentGb } to
// exercise the measured-vs-estimate paths. Defaults roughly mirror a real z-image-class model:
// q4 ~4 GB on disk, q8 ~8 GB, bf16 ~16 GB.
function matrixModel(tiers = defaultTiers()) {
  return {
    id: "z_image_turbo",
    hasVariantMatrix: true,
    variants: tiers.map((tier) => {
      const spec = typeof tier === "string" ? { variant: tier } : tier;
      const footprint = {};
      if (spec.diskGb != null) {
        footprint.diskSizeBytes = spec.diskGb * GB;
      }
      if (spec.residentGb != null) {
        footprint.residentMemoryBytes = spec.residentGb * GB;
      }
      if (spec.peakGb != null) {
        footprint.peakMemoryBytes = spec.peakGb * GB;
      }
      return {
        variant: spec.variant,
        installState: spec.installed ? "installed" : "missing",
        downloadSizeBytes: spec.downloadGb != null ? spec.downloadGb * GB : null,
        footprint: Object.keys(footprint).length ? footprint : null,
      };
    }),
  };
}

function defaultTiers() {
  return [
    { variant: "q4", diskGb: 4 },
    { variant: "q8", diskGb: 8 },
    { variant: "bf16", diskGb: 16 },
  ];
}

describe("variantFootprintBytes", () => {
  it("uses measured peakMemoryBytes verbatim when present (the true ceiling)", () => {
    // sc-8516: peak is the memory a generation must fit, so it wins over resident when measured.
    const result = variantFootprintBytes({
      variant: "bf16",
      footprint: {
        diskSizeBytes: 16 * GB,
        residentMemoryBytes: 20 * GB,
        peakMemoryBytes: 34 * GB,
      },
    });
    expect(result).toEqual({ bytes: 34 * GB, measured: true });
  });

  it("uses measured residentMemoryBytes + fixed transient when peak is absent", () => {
    const result = variantFootprintBytes({
      variant: "bf16",
      footprint: { diskSizeBytes: 16 * GB, residentMemoryBytes: 20 * GB },
    });
    expect(result).toEqual({ bytes: 20 * GB + TRANSIENT_HEADROOM_BYTES, measured: true });
  });

  it("estimates diskSizeBytes × multiplier + transient when memory is not measured", () => {
    const result = variantFootprintBytes({ variant: "q4", footprint: { diskSizeBytes: 4 * GB } });
    expect(result.measured).toBe(false);
    expect(result.bytes).toBe(
      Math.round(4 * GB * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
    );
  });

  it("falls back to downloadSizeBytes × multiplier + transient when no footprint object", () => {
    const result = variantFootprintBytes({ variant: "q4", downloadSizeBytes: 4 * GB, footprint: null });
    expect(result.measured).toBe(false);
    expect(result.bytes).toBe(
      Math.round(4 * GB * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
    );
  });

  it("returns null when nothing is estimable", () => {
    expect(variantFootprintBytes({ variant: "q4", footprint: null })).toBe(null);
    expect(variantFootprintBytes({ variant: "q4" })).toBe(null);
    expect(variantFootprintBytes(undefined)).toBe(null);
  });
});

describe("declaredTiers", () => {
  it("returns real quant tiers highest-fidelity first, regardless of install state", () => {
    expect(declaredTiers(matrixModel())).toEqual(["bf16", "q8", "q4"]);
  });

  it("excludes the single-variant default pseudo-tier and unknown keys", () => {
    const model = {
      hasVariantMatrix: true,
      variants: [{ variant: "default" }, { variant: "q4" }, { variant: "mystery" }],
    };
    expect(declaredTiers(model)).toEqual(["q4"]);
  });

  it("returns [] for a non-matrix model", () => {
    expect(declaredTiers({ hasVariantMatrix: false, variants: [] })).toEqual([]);
    expect(declaredTiers(undefined)).toEqual([]);
  });
});

describe("tierFits", () => {
  it("treats an unknown memory signal as fitting (never withhold)", () => {
    const bf16 = { variant: "bf16", footprint: { diskSizeBytes: 16 * GB } };
    expect(tierFits(bf16, null)).toBe(true);
    expect(tierFits(bf16, undefined)).toBe(true);
  });

  it("treats an unestimable footprint as fitting", () => {
    expect(tierFits({ variant: "q4", footprint: null }, 8)).toBe(true);
  });

  it("fits when the estimated peak is under the headroom budget", () => {
    // q4: 4 GB disk × 1.0 + 14 GB transient = 18 GB peak. 32 GB × 0.9 = 28.8 GB budget → fits.
    const q4 = { variant: "q4", footprint: { diskSizeBytes: 4 * GB } };
    expect(tierFits(q4, 32)).toBe(true);
  });

  it("does not fit when the estimated peak exceeds the headroom budget", () => {
    // bf16: 16 GB disk × 1.0 + 14 GB transient = 30 GB peak. On a 24 GB host budget is 24 × 0.9 =
    // 21.6 GB → no.
    const bf16 = { variant: "bf16", footprint: { diskSizeBytes: 16 * GB } };
    expect(tierFits(bf16, 24)).toBe(false);
  });

  it("respects the exact headroom boundary", () => {
    // A peak exactly at budget fits; a byte over does not. Solve for the disk size whose estimated
    // peak (disk × MULT + transient) lands exactly on the budget.
    const budgetGb = 30;
    const budgetBytes = budgetGb * GB * MEMORY_HEADROOM_FRACTION; // 27 GB peak we target
    const diskBytes = (budgetBytes - TRANSIENT_HEADROOM_BYTES) / DISK_TO_RESIDENT_MULTIPLIER;
    const atBudget = { variant: "q8", footprint: { diskSizeBytes: diskBytes } };
    const overBudget = { variant: "q8", footprint: { diskSizeBytes: diskBytes + GB } };
    expect(tierFits(atBudget, budgetGb)).toBe(true);
    expect(tierFits(overBudget, budgetGb)).toBe(false);
  });
});

describe("suggestTier", () => {
  it("suggests q4 on a 32 GB host when the larger tiers overflow the budget (acceptance)", () => {
    // Peak-based budget on 32 GB = 32 × 0.9 = 28.8 GB. Estimated peak = diskGb × 1.0 + 14 GB transient.
    // Size q8/bf16 to exceed the budget so only q4 fits (a 32 GB user sees q4 pre-selected).
    const model = matrixModel([
      { variant: "q4", diskGb: 4 }, // 4 + 14 = 18 GB peak < 28.8 → fits
      { variant: "q8", diskGb: 18 }, // 18 + 14 = 32 GB peak > 28.8 → no
      { variant: "bf16", diskGb: 24 }, // 24 + 14 = 38 GB peak > 28.8 → no
    ]);
    expect(suggestTier(model, 32)).toBe("q4");
  });

  it("suggests q8 when bf16 is too big but q8 fits on a 32 GB host", () => {
    // Budget 28.8 GB (32 × 0.9); estimated peak = diskGb + 14 GB transient.
    const model = matrixModel([
      { variant: "q4", diskGb: 4 }, // 18 GB peak
      { variant: "q8", diskGb: 10 }, // 10 + 14 = 24 GB peak < 28.8 → fits
      { variant: "bf16", diskGb: 24 }, // 38 GB peak → no
    ]);
    // q8 is higher fidelity than q4 and fits → preferred over q4.
    expect(suggestTier(model, 32)).toBe("q8");
  });

  it("suggests bf16 on a 512 GB Studio", () => {
    expect(suggestTier(matrixModel(), 512)).toBe("bf16");
  });

  it("prefers a measured peak footprint over the disk estimate", () => {
    // bf16 disk is small (would estimate as fitting) but MEASURED peak is huge → excluded on 32 GB
    // (budget 28.8 GB). Exercises the measured-footprint path the way real manifest data flows.
    const model = matrixModel([
      { variant: "q4", diskGb: 4 },
      { variant: "bf16", diskGb: 8, peakGb: 40 }, // measured peak 40 GB > 28.8 → no
    ]);
    expect(suggestTier(model, 32)).toBe("q4");
  });

  it("suggests a MEASURED tier whose peak fits on a right-sized host (sc-8516 calibration)", () => {
    // Real measured lens_turbo-class numbers: q4 peak ≈ 30.5 GB. On a 48 GB Mac (budget 43.2 GB) it
    // fits; on a 32 GB Mac (budget 28.8 GB) it does not — the exact threshold behavior the harness
    // measured. Only q4 is declared here (the only lens tier sc-8516 measured).
    const lensQ4 = matrixModel([{ variant: "q4", diskGb: 20, peakGb: 30.5 }]);
    expect(suggestTier(lensQ4, 48)).toBe("q4");
    expect(tierFits(lensQ4.variants[0], 32)).toBe(false);
    expect(tierFits(lensQ4.variants[0], 48)).toBe(true);
  });

  it("falls back to the smallest tier when nothing fits", () => {
    const model = matrixModel([
      { variant: "q4", diskGb: 40 }, // 40 + 14 = 54 GB peak est
      { variant: "bf16", diskGb: 80 },
    ]);
    // Tiny 8 GB host, every tier over budget → smallest declared tier (q4).
    expect(suggestTier(model, 8)).toBe("q4");
  });

  it("suggests the highest-fidelity tier when memory is unknown", () => {
    expect(suggestTier(matrixModel(), null)).toBe("bf16");
  });

  it("returns null for a non-matrix model", () => {
    expect(suggestTier({ hasVariantMatrix: false, variants: [] }, 32)).toBe(null);
  });
});
