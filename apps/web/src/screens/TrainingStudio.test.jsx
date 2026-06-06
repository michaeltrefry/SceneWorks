import { describe, expect, it } from "vitest";

import { configDraftFromTarget, trainingConfigSnapshot, trainingSampleGroups } from "./TrainingStudio.jsx";

// Minimal target mirroring a LoKr-capable builtin (sc-2195 gating): networkType
// lives in defaults.advanced, the choice is gated by limits.networkTypes.
const lokrTarget = {
  id: "sdxl_lora",
  defaults: {
    rank: 8,
    alpha: 8,
    learningRate: 0.0001,
    steps: 1000,
    batchSize: 1,
    gradientAccumulation: 1,
    resolution: 1024,
    saveEvery: 0,
    seed: 42,
    optimizer: "adamw",
    advanced: { networkType: "lora" },
  },
  limits: { networkTypes: ["lora", "lokr"] },
};

const dataset = { id: "ds-1", version: 1, name: "Kelsie" };

function snapshot(configDraft) {
  return trainingConfigSnapshot({
    activeDataset: dataset,
    configDraft: { ...configDraft, outputName: "Kelsie LoRA" },
    selectedTarget: lokrTarget,
  });
}

describe("TrainingStudio network type", () => {
  it("draft defaults to lora with an auto LoKr factor", () => {
    const draft = configDraftFromTarget(lokrTarget, dataset, []);
    expect(draft.networkType).toBe("lora");
    expect(draft.decomposeFactor).toBe("-1");
  });

  it("draft reflects a lokr default from the target's advanced bag", () => {
    const lokrDefault = {
      ...lokrTarget,
      defaults: { ...lokrTarget.defaults, advanced: { networkType: "lokr", decomposeFactor: 16 } },
    };
    const draft = configDraftFromTarget(lokrDefault, dataset, []);
    expect(draft.networkType).toBe("lokr");
    expect(draft.decomposeFactor).toBe("16");
  });

  it("serializes lora without a LoKr factor", () => {
    const draft = configDraftFromTarget(lokrTarget, dataset, []);
    const snap = snapshot(draft);
    expect(snap.config.advanced.networkType).toBe("lora");
    expect(snap.config.advanced).not.toHaveProperty("decomposeFactor");
  });

  it("serializes lokr with the chosen factor", () => {
    const draft = configDraftFromTarget(lokrTarget, dataset, []);
    const snap = snapshot({ ...draft, networkType: "lokr", decomposeFactor: "16" });
    expect(snap.config.advanced.networkType).toBe("lokr");
    expect(snap.config.advanced.decomposeFactor).toBe(16);
  });

  it("omits a blank LoKr factor so the worker applies its -1 default", () => {
    const draft = configDraftFromTarget(lokrTarget, dataset, []);
    const snap = snapshot({ ...draft, networkType: "lokr", decomposeFactor: "" });
    expect(snap.config.advanced.networkType).toBe("lokr");
    expect(snap.config.advanced).not.toHaveProperty("decomposeFactor");
  });
});

describe("TrainingStudio training samples", () => {
  it("groups every sampling cycle newest first", () => {
    const groups = trainingSampleGroups(
      {
        id: "job-1",
        payload: { plan: { config: { triggerWord: "Mira" } } },
        result: {
          samplePrompts: ["front", "side"],
          trainingSamples: [
            { step: 250, prompt: "front", relativePath: "training/job-1/samples/step-000250/front.png" },
            { step: 250, prompt: "side", relativePath: "training/job-1/samples/step-000250/side.png" },
            { step: 500, prompt: "front", relativePath: "training/job-1/samples/step-000500/front.png" },
            { step: 500, prompt: "side", relativePath: "training/job-1/samples/step-000500/side.png" },
          ],
        },
      },
      "project-1",
    );

    expect(groups).toHaveLength(2);
    expect(groups[0].label).toBe("Sample #2 - Step 500");
    expect(groups[0].assets).toHaveLength(2);
    expect(groups[0].assets[0].file.path).toBe("training/job-1/samples/step-000500/front.png");
    expect(groups[1].label).toBe("Sample #1 - Step 250");
  });

  it("deduplicates latest samples already present in the cumulative list", () => {
    const groups = trainingSampleGroups(
      {
        id: "job-1",
        payload: {},
        result: {
          trainingSamples: [{ step: 250, prompt: "front", relativePath: "training/job-1/samples/step-000250/front.png" }],
          latestTrainingSamples: [{ step: 250, prompt: "front", relativePath: "training/job-1/samples/step-000250/front.png" }],
        },
      },
      "project-1",
    );

    expect(groups).toHaveLength(1);
    expect(groups[0].assets).toHaveLength(1);
  });
});
