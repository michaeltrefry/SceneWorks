import { describe, expect, it } from "vitest";
import {
  defaultTierSelection,
  installedTiers,
  shouldShowTierPicker,
  tierLabel,
  tierQuantize,
} from "./quantTier.js";

// Build a /models-shaped model with a variant matrix. `installed` is the set of tier keys whose
// files are present (installState "installed"); every other declared tier reports "missing".
function matrixModel({ tiers = ["q4", "q8", "bf16"], installed = [], defaultTier = "q4" } = {}) {
  return {
    id: "z_image_turbo",
    hasVariantMatrix: true,
    variants: tiers.map((tier) => ({
      variant: tier,
      default: tier === defaultTier,
      installState: installed.includes(tier) ? "installed" : "missing",
    })),
  };
}

describe("quantTier mapping", () => {
  it("maps known tiers to mlxQuantize values (bf16→0, q8→8, q4→4)", () => {
    expect(tierQuantize("bf16")).toBe(0);
    expect(tierQuantize("q8")).toBe(8);
    expect(tierQuantize("q4")).toBe(4);
  });

  it("returns null for the 'default' pseudo-variant and unknown keys", () => {
    expect(tierQuantize("default")).toBe(null);
    expect(tierQuantize("q2")).toBe(null);
    expect(tierQuantize(undefined)).toBe(null);
  });

  it("labels known tiers and falls back to the raw key", () => {
    expect(tierLabel("bf16")).toBe("Full precision (bf16)");
    expect(tierLabel("q8")).toBe("Q8 (balanced)");
    expect(tierLabel("q4")).toBe("Q4 (smallest)");
    expect(tierLabel("mystery")).toBe("mystery");
  });
});

describe("installedTiers", () => {
  it("returns only installed quant tiers, in smallest→largest order", () => {
    const model = matrixModel({ installed: ["bf16", "q4"] });
    expect(installedTiers(model)).toEqual(["q4", "bf16"]);
  });

  it("returns [] for a model with no variant matrix", () => {
    expect(installedTiers({ id: "boogu", hasVariantMatrix: false, variants: [] })).toEqual([]);
    expect(installedTiers({ id: "x" })).toEqual([]);
    expect(installedTiers(undefined)).toEqual([]);
  });

  it("excludes the single-variant 'default' pseudo-tier", () => {
    const single = {
      id: "single",
      hasVariantMatrix: false,
      variants: [{ variant: "default", installState: "installed", default: true }],
    };
    expect(installedTiers(single)).toEqual([]);
  });

  it("excludes tiers that are declared but not installed", () => {
    const model = matrixModel({ installed: ["q4"] });
    expect(installedTiers(model)).toEqual(["q4"]);
  });
});

describe("shouldShowTierPicker", () => {
  it("shows the picker only when more than one tier is installed", () => {
    expect(shouldShowTierPicker(matrixModel({ installed: ["q4", "bf16"] }))).toBe(true);
    expect(shouldShowTierPicker(matrixModel({ installed: ["q4"] }))).toBe(false);
    expect(shouldShowTierPicker(matrixModel({ installed: [] }))).toBe(false);
    expect(shouldShowTierPicker({ id: "x", hasVariantMatrix: false })).toBe(false);
  });
});

describe("defaultTierSelection", () => {
  it("prefers the last-used tier when it is still installed", () => {
    const model = matrixModel({ installed: ["q4", "q8", "bf16"] });
    expect(defaultTierSelection(model, "q8")).toBe("q8");
    expect(defaultTierSelection(model, "bf16")).toBe("bf16");
  });

  it("ignores a last-used tier that is no longer installed", () => {
    const model = matrixModel({ installed: ["q4", "bf16"] });
    // q8 was last used but is now uninstalled → fall through to the declared default (q4).
    expect(defaultTierSelection(model, "q8")).toBe("q4");
  });

  it("falls back to the declared default tier when installed", () => {
    const model = matrixModel({ installed: ["q8", "bf16"], defaultTier: "q8" });
    expect(defaultTierSelection(model, null)).toBe("q8");
  });

  it("falls back to q4 when installed and no default/last-used applies", () => {
    const model = matrixModel({ installed: ["q4", "bf16"], defaultTier: "q4" });
    // Declared default q4 is installed → picked.
    expect(defaultTierSelection(model, undefined)).toBe("q4");
  });

  it("falls back to the first installed tier when neither default nor q4 is present", () => {
    const model = matrixModel({ tiers: ["q8", "bf16"], installed: ["q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("q8");
  });

  it("returns null when nothing is installed", () => {
    expect(defaultTierSelection(matrixModel({ installed: [] }), null)).toBe(null);
    expect(defaultTierSelection({ id: "x", hasVariantMatrix: false }, null)).toBe(null);
  });
});
