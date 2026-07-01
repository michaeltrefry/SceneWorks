import { describe, expect, it } from "vitest";
import {
  DISK_TO_RESIDENT_MULTIPLIER,
  MEMORY_HEADROOM_FRACTION,
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
  it("uses measured residentMemoryBytes verbatim when present", () => {
    const result = variantFootprintBytes({
      variant: "bf16",
      footprint: { diskSizeBytes: 16 * GB, residentMemoryBytes: 20 * GB },
    });
    expect(result).toEqual({ bytes: 20 * GB, measured: true });
  });

  it("estimates from diskSizeBytes × multiplier when memory is not measured", () => {
    const result = variantFootprintBytes({ variant: "q4", footprint: { diskSizeBytes: 4 * GB } });
    expect(result.measured).toBe(false);
    expect(result.bytes).toBe(Math.round(4 * GB * DISK_TO_RESIDENT_MULTIPLIER));
  });

  it("falls back to downloadSizeBytes × multiplier when no footprint object", () => {
    const result = variantFootprintBytes({ variant: "q4", downloadSizeBytes: 4 * GB, footprint: null });
    expect(result.measured).toBe(false);
    expect(result.bytes).toBe(Math.round(4 * GB * DISK_TO_RESIDENT_MULTIPLIER));
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

  it("fits when the estimated footprint is under the headroom budget", () => {
    // q4: 4 GB disk × 1.5 = 6 GB resident. 32 GB × 0.8 = 25.6 GB budget → fits.
    const q4 = { variant: "q4", footprint: { diskSizeBytes: 4 * GB } };
    expect(tierFits(q4, 32)).toBe(true);
  });

  it("does not fit when the estimated footprint exceeds the headroom budget", () => {
    // bf16: 16 GB disk × 1.5 = 24 GB resident. On a 16 GB host budget is 16 × 0.8 = 12.8 GB → no.
    const bf16 = { variant: "bf16", footprint: { diskSizeBytes: 16 * GB } };
    expect(tierFits(bf16, 16)).toBe(false);
  });

  it("respects the exact headroom boundary", () => {
    // footprint exactly at budget fits; a byte over does not.
    const budgetGb = 10;
    const footBytes = budgetGb * GB * MEMORY_HEADROOM_FRACTION; // resident bytes we want
    const diskBytes = footBytes / DISK_TO_RESIDENT_MULTIPLIER;
    const atBudget = { variant: "q8", footprint: { diskSizeBytes: diskBytes } };
    const overBudget = { variant: "q8", footprint: { diskSizeBytes: diskBytes + GB } };
    expect(tierFits(atBudget, budgetGb)).toBe(true);
    expect(tierFits(overBudget, budgetGb)).toBe(false);
  });
});

describe("suggestTier", () => {
  it("suggests q4 on a 32 GB host when the larger tiers overflow the budget (acceptance)", () => {
    // Budget on 32 GB = 32 × 0.8 = 25.6 GB. Size q8/bf16 to exceed it so only q4 fits, matching the
    // acceptance criterion (a 32 GB user sees q4 pre-selected).
    const model = matrixModel([
      { variant: "q4", diskGb: 4 }, // 6 GB est → fits
      { variant: "q8", diskGb: 18 }, // 27 GB est > 25.6 → no
      { variant: "bf16", diskGb: 24 }, // 36 GB est > 25.6 → no
    ]);
    expect(suggestTier(model, 32)).toBe("q4");
  });

  it("suggests q8 when bf16 is too big but q8 fits on a 32 GB host", () => {
    const model = matrixModel([
      { variant: "q4", diskGb: 4 }, // 6 GB
      { variant: "q8", diskGb: 10 }, // 15 GB < 25.6 → fits
      { variant: "bf16", diskGb: 24 }, // 36 GB → no
    ]);
    // q8 is higher fidelity than q4 and fits → preferred over q4.
    expect(suggestTier(model, 32)).toBe("q8");
  });

  it("suggests bf16 on a 512 GB Studio", () => {
    expect(suggestTier(matrixModel(), 512)).toBe("bf16");
  });

  it("prefers measured resident memory over the disk estimate", () => {
    // bf16 disk is small (would estimate as fitting) but MEASURED resident is huge → excluded.
    const model = matrixModel([
      { variant: "q4", diskGb: 4 },
      { variant: "bf16", diskGb: 8, residentGb: 40 }, // measured 40 GB > 25.6 → no
    ]);
    expect(suggestTier(model, 32)).toBe("q4");
  });

  it("falls back to the smallest tier when nothing fits", () => {
    const model = matrixModel([
      { variant: "q4", diskGb: 40 }, // 60 GB est
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
