import { describe, expect, it } from "vitest";
import {
  GUIDANCE_METHOD_LABELS,
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  guidanceMethodDefaultFromModel,
  guidanceMethodOptionsFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  schedulerShiftDefaultFromModel,
  stepsDefaultFromModel,
} from "./samplerOptions.js";

describe("samplerOptions", () => {
  it("falls back to default-only when limits are missing", () => {
    expect(samplerOptionsFromModel(undefined)).toEqual(["default"]);
    expect(samplerOptionsFromModel({})).toEqual(["default"]);
    expect(schedulerOptionsFromModel({})).toEqual(["default"]);
  });

  it("returns options in canonical order regardless of manifest ordering", () => {
    const model = { limits: { samplers: ["uni_pc", "default", "euler"] } };
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "uni_pc"]);
  });

  it("preserves unknown sampler keys (forward-compat)", () => {
    const model = { limits: { samplers: ["default", "euler", "future"] } };
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "future"]);
  });

  it("scheduler options are ordered canonically", () => {
    const model = {
      limits: {
        schedulers: ["beta", "default", "karras", "sgm_uniform"],
      },
    };
    expect(schedulerOptionsFromModel(model)).toEqual([
      "default",
      "karras",
      "sgm_uniform",
      "beta",
    ]);
  });

  it("resolves the per-backend limits override (epic 7114)", () => {
    // SDXL-shaped asymmetry: full curated on MLX (base), narrow on candle.
    const model = {
      limits: { samplers: ["default", "euler", "dpmpp_2m"] },
      candle: { limits: { samplers: ["default", "ddim"] } },
    };
    expect(samplerOptionsFromModel(model, "mlx")).toEqual(["default", "euler", "dpmpp_2m"]);
    expect(samplerOptionsFromModel(model, "candle")).toEqual(["default", "ddim"]);
    // No / unknown backend falls back to the base menu.
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "dpmpp_2m"]);
    // Lens-shaped asymmetry: default-only on MLX (base), curated on candle.
    const lens = {
      limits: { samplers: ["default"] },
      candle: { limits: { samplers: ["default", "euler", "heun"] } },
    };
    expect(samplerOptionsFromModel(lens, "mlx")).toEqual(["default"]);
    expect(samplerOptionsFromModel(lens, "candle")).toEqual(["default", "euler", "heun"]);
  });

  it("default helpers read defaults block with sensible fallbacks", () => {
    const model = {
      defaults: {
        sampler: "dpmpp_2m",
        scheduler: "karras",
        schedulerShift: 4.5,
        steps: 20,
        guidanceScale: 3.5,
      },
    };
    expect(samplerDefaultFromModel(model)).toBe("dpmpp_2m");
    expect(schedulerDefaultFromModel(model)).toBe("karras");
    expect(schedulerShiftDefaultFromModel(model)).toBe(4.5);
    expect(stepsDefaultFromModel(model)).toBe(20);
    expect(guidanceDefaultFromModel(model)).toBe(3.5);
  });

  it("invalid default values fall back gracefully", () => {
    expect(samplerDefaultFromModel({ defaults: { sampler: "" } })).toBe("default");
    expect(schedulerShiftDefaultFromModel({ defaults: { schedulerShift: -1 } })).toBe(3.0);
    expect(stepsDefaultFromModel({ defaults: { steps: 0 } })).toBeNull();
    expect(guidanceDefaultFromModel({ defaults: { guidanceScale: "n/a" } })).toBeNull();
  });

  it("guidance methods fall back to cfg-only and order canonically (epic 7434)", () => {
    // No advertisement → cfg-only, so the studio hides the picker (length 1).
    expect(guidanceMethodOptionsFromModel(undefined)).toEqual(["cfg"]);
    expect(guidanceMethodOptionsFromModel({})).toEqual(["cfg"]);
    // Manifest ordering is normalized to the canonical guidance order.
    const model = { limits: { guidanceMethods: ["cfg_pp", "cfg"] } };
    expect(guidanceMethodOptionsFromModel(model)).toEqual(["cfg", "cfg_pp"]);
  });

  it("resolves guidance methods per-backend — CFG++ is MLX-only (sc-7447)", () => {
    // SDXL-shaped: MLX advertises cfg_pp via the mlx override; candle (base) is cfg-only.
    const sdxl = {
      limits: { guidanceMethods: ["cfg"] },
      mlx: { limits: { guidanceMethods: ["cfg", "cfg_pp"] } },
    };
    expect(guidanceMethodOptionsFromModel(sdxl, "mlx")).toEqual(["cfg", "cfg_pp"]);
    expect(guidanceMethodOptionsFromModel(sdxl, "candle")).toEqual(["cfg"]);
    // No / unknown backend falls back to the base (cfg-only) menu.
    expect(guidanceMethodOptionsFromModel(sdxl)).toEqual(["cfg"]);
  });

  it("guidance default is cfg unless the model pins one", () => {
    expect(guidanceMethodDefaultFromModel(undefined)).toBe("cfg");
    expect(guidanceMethodDefaultFromModel({})).toBe("cfg");
    expect(guidanceMethodDefaultFromModel({ defaults: { guidanceMethod: "" } })).toBe("cfg");
    expect(guidanceMethodDefaultFromModel({ defaults: { guidanceMethod: "cfg_pp" } })).toBe(
      "cfg_pp",
    );
  });

  it("guidance labels cover the full vocabulary (epic 7434)", () => {
    for (const key of ["cfg", "cfg_rescale", "apg", "cfg_pp"]) {
      expect(GUIDANCE_METHOD_LABELS[key]).toBeTruthy();
    }
  });

  it("labels cover the full curated menu (epic 7114)", () => {
    for (const key of [
      "default",
      "euler",
      "euler_ancestral",
      "heun",
      "dpmpp_2m",
      "dpmpp_sde",
      "uni_pc",
      "lcm",
      "ddim",
    ]) {
      expect(SAMPLER_LABELS[key]).toBeTruthy();
    }
    for (const key of [
      "default",
      "normal",
      "simple",
      "karras",
      "exponential",
      "sgm_uniform",
      "beta",
      "ddim_uniform",
    ]) {
      expect(SCHEDULER_LABELS[key]).toBeTruthy();
    }
  });
});
