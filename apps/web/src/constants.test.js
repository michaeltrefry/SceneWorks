import { describe, expect, it } from "vitest";
import { fallbackModels } from "./constants.js";

// SD3.5 Image Studio surfacing (epic 7841 / sc-7873). The fallbackModels list seeds the picker
// before the live catalog loads; the three SD3.5 variants must be present as text-to-image image
// models with the SD3.5 prompt guide and per-variant defaults (Turbo fast, Large high-fidelity,
// Medium smaller-RAM). The real macOnly/gated/minMemoryGb gating comes from the manifest/macSupport
// at runtime — these fallback entries only carry the picker shape + per-variant defaults.
describe("fallbackModels SD3.5 surfacing (sc-7873)", () => {
  const byId = (id) => fallbackModels.find((model) => model.id === id);

  it("includes all three SD3.5 variants as text-to-image image models", () => {
    for (const id of ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]) {
      const model = byId(id);
      expect(model, `${id} must be present in fallbackModels`).toBeTruthy();
      expect(model.type).toBe("image");
      expect(model.capabilities).toContain("text_to_image");
      expect(model.ui?.promptGuide?.path).toBe("/prompt-guides/sd3-5.md");
    }
  });

  it("Turbo is the fast default: few-step, CFG-free", () => {
    const turbo = byId("sd3_5_large_turbo");
    expect(turbo.defaults.steps).toBe(4);
    expect(turbo.defaults.guidanceScale).toBe(1.0);
  });

  it("Large is high-fidelity: more steps + true CFG", () => {
    const large = byId("sd3_5_large");
    expect(large.defaults.steps).toBe(28);
    expect(large.defaults.guidanceScale).toBe(3.5);
  });

  it("Medium is the smaller-RAM mid-tier: its own recipe", () => {
    const medium = byId("sd3_5_medium");
    expect(medium.defaults.steps).toBe(40);
    expect(medium.defaults.guidanceScale).toBe(4.5);
  });
});
