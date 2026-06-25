import { describe, expect, it } from "vitest";
import {
  pidCheckpointId,
  pidDecoderAvailable,
  pidToggleEligible,
  pidToggleVisible,
} from "./pidEligibility.js";

const qwen = { id: "qwen_image", ui: { pid: { checkpointId: "pid_qwenimage" } } };
const krea = { id: "krea_2_turbo", ui: { pid: { checkpointId: "pid_qwenimage" } } };
const sensenova = { id: "sensenova_u1_8b", ui: {} };
const installedCkpt = { id: "pid_qwenimage", installState: "installed" };
const missingCkpt = { id: "pid_qwenimage", installState: "missing" };

describe("pidEligibility", () => {
  it("pidCheckpointId reads ui.pid.checkpointId, null when absent", () => {
    expect(pidCheckpointId(qwen)).toBe("pid_qwenimage");
    expect(pidCheckpointId(sensenova)).toBe(null);
    expect(pidCheckpointId({ ui: { pid: { checkpointId: "" } } })).toBe(null);
    expect(pidCheckpointId(undefined)).toBe(null);
  });

  it("pidToggleEligible is true only for models with a PiD backbone", () => {
    expect(pidToggleEligible(qwen)).toBe(true);
    expect(pidToggleEligible(krea)).toBe(true);
    expect(pidToggleEligible(sensenova)).toBe(false);
  });

  it("pidDecoderAvailable requires the checkpoint catalog entry to be installed", () => {
    expect(pidDecoderAvailable(qwen, [qwen, installedCkpt])).toBe(true);
    // Eligible but checkpoint not installed → not available (fail-closed).
    expect(pidDecoderAvailable(qwen, [qwen, missingCkpt])).toBe(false);
    // Eligible but checkpoint entry absent from the catalog (pre-sc-7852) → not available.
    expect(pidDecoderAvailable(qwen, [qwen])).toBe(false);
    // Non-eligible model is never available even if some pid checkpoint is installed.
    expect(pidDecoderAvailable(sensenova, [sensenova, installedCkpt])).toBe(false);
  });

  it("pidToggleVisible requires both eligibility and an installed checkpoint", () => {
    expect(pidToggleVisible(qwen, [qwen, installedCkpt])).toBe(true);
    expect(pidToggleVisible(qwen, [qwen, missingCkpt])).toBe(false);
    expect(pidToggleVisible(qwen, [qwen])).toBe(false);
    expect(pidToggleVisible(sensenova, [sensenova, installedCkpt])).toBe(false);
  });
});
