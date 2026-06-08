import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAC_CAPABILITIES,
  macAvailableModels,
  macBlockedModels,
  macFeatureBlock,
  macModelBlock,
  macModelFeatureBlock,
  macReasonText,
  macTrainingKernelBlocked,
  macVideoModeBlock,
} from "./macGating.js";

const gating = {
  macGatingActive: true,
  platform: "macos",
  notAvailableLabel: "Not available on Mac (Rust/MLX only)",
  features: {
    imageUpscale: {
      supported: false,
      reason: { feature: "image_upscale (Real-ESRGAN)", detail: "torch path.", suggestedEpic: "sc-3489" },
    },
    // LyCORIS is ported to MLX (epic 3641) → no longer a capability feature entry.
  },
  training: { supportedKernels: ["z_image_lora", "sdxl_lora"], lokrOnWanSupported: false },
};

const torchModel = { id: "kolors", macSupport: { supported: false, reason: { detail: "no MLX engine.", suggestedEpic: "epic 3532" } } };
const mlxModel = { id: "z_image_turbo", macSupport: { supported: true, features: { pose: true, reference: false, edit: false, lycoris: true } } };

describe("macGating helpers", () => {
  it("are all inert when gating is not active (Windows/Linux/observe mode)", () => {
    const caps = DEFAULT_MAC_CAPABILITIES;
    expect(macModelBlock(torchModel, caps)).toBeNull();
    expect(macModelFeatureBlock(mlxModel, caps, "reference")).toBeNull();
    expect(macFeatureBlock(caps, "imageUpscale")).toBeNull();
    expect(macTrainingKernelBlocked(caps, "kolors_lora")).toBe(false);
    expect(macAvailableModels([torchModel, mlxModel], caps)).toHaveLength(2);
    expect(macBlockedModels([torchModel, mlxModel], caps)).toHaveLength(0);
  });

  it("blocks a torch-only model and names its port epic when gating is active", () => {
    const block = macModelBlock(torchModel, gating);
    expect(block?.blocked).toBe(true);
    expect(block?.text).toContain("Not available on Mac (Rust/MLX only)");
    expect(block?.text).toContain("epic 3532");
    expect(macModelBlock(mlxModel, gating)).toBeNull();
  });

  it("partitions a model list into available/blocked", () => {
    expect(macAvailableModels([torchModel, mlxModel], gating).map((m) => m.id)).toEqual(["z_image_turbo"]);
    expect(macBlockedModels([torchModel, mlxModel], gating).map((m) => m.id)).toEqual(["kolors"]);
  });

  it("blocks a per-model feature only when its flag is false", () => {
    expect(macModelFeatureBlock(mlxModel, gating, "pose")).toBeNull();
    const refBlock = macModelFeatureBlock(mlxModel, gating, "reference");
    expect(refBlock?.blocked).toBe(true);
    expect(refBlock?.text).toContain("torch path");
  });

  it("blocks a global feature and surfaces its reason text", () => {
    const block = macFeatureBlock(gating, "imageUpscale");
    expect(block?.blocked).toBe(true);
    expect(block?.text).toContain("sc-3489");
    // LyCORIS is ported (epic 3641) → not a capability feature, so never blocked.
    expect(macFeatureBlock(gating, "lycoris")).toBeNull();
  });

  it("blocks a training kernel without a native Rust trainer", () => {
    expect(macTrainingKernelBlocked(gating, "kolors_lora")).toBe(true);
    expect(macTrainingKernelBlocked(gating, "z_image_lora")).toBe(false);
  });

  it("blocks a torch-only video mode by per-model eligibility", () => {
    const video = { id: "wan_2_2", macSupport: { features: { videoModes: { image_to_video: true, replace_person: false } } } };
    expect(macVideoModeBlock(video, gating, "image_to_video")).toBeNull();
    expect(macVideoModeBlock(video, gating, "replace_person")?.blocked).toBe(true);
  });

  it("falls back to the bare label when a reason is missing", () => {
    expect(macReasonText(gating, null)).toBe("Not available on Mac (Rust/MLX only)");
  });
});
