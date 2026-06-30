import { describe, expect, it } from "vitest";

import { configDraftFromTarget, trainingConfigSnapshot } from "./trainingConfig.js";

// sc-4199: configDraftFromTarget (target/preset → form draft) and
// trainingConfigSnapshot (form draft → worker payload) were pure builders buried
// in the 2.1k-line TrainingStudio screen. Extracted to ../training/trainingConfig.js,
// they are directly testable.

const target = {
  id: "sdxl_lora",
  outputKind: "lora",
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

const dataset = { id: "ds-1", version: 3, name: "Kelsie" };

describe("configDraftFromTarget", () => {
  it("seeds the output name from the dataset name + output kind label", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.outputName).toBe("Kelsie LoRA");
  });

  it("stringifies numeric defaults into form drafts", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.rank).toBe("8");
    expect(draft.learningRate).toBe("0.0001");
    expect(draft.steps).toBe("1000");
    expect(draft.seed).toBe("42");
  });

  it("falls back to the first GPU when the advanced requestedGpu is not offered", () => {
    const gpuTarget = { ...target, defaults: { ...target.defaults, advanced: { ...target.defaults.advanced, requestedGpu: "7" } } };
    expect(configDraftFromTarget(gpuTarget, dataset, ["auto", "0"]).requestedGpu).toBe("auto");
    expect(configDraftFromTarget(gpuTarget, dataset, ["auto", "7"]).requestedGpu).toBe("7");
  });

  it("prefers the explicit trigger phrase over the target default", () => {
    const triggerTarget = { ...target, defaults: { ...target.defaults, triggerWord: "fallback" } };
    expect(configDraftFromTarget(triggerTarget, dataset, ["auto"], "ohwx woman").triggerWord).toBe("ohwx woman");
    expect(configDraftFromTarget(triggerTarget, dataset, ["auto"], "").triggerWord).toBe("fallback");
  });

  it("carries a preset's outputName through previousDraft instead of reseeding", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "", null, { outputName: "Custom name" });
    expect(draft.outputName).toBe("Custom name");
  });

  it("defaults the LoKr factor to an auto -1 string and normalizes the adapter version", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(draft.decomposeFactor).toBe("-1");
    const versioned = {
      ...target,
      defaults: { ...target.defaults, advanced: { ...target.defaults.advanced, trainingAdapterVersion: "v2-default" } },
    };
    expect(configDraftFromTarget(versioned, dataset, ["auto"]).trainingAdapterVersion).toBe("v2");
  });
});

describe("trainingConfigSnapshot", () => {
  function snapshot(configDraft, extra = {}) {
    return trainingConfigSnapshot({
      activeDataset: dataset,
      configDraft: { ...configDraft, outputName: "Kelsie LoRA" },
      selectedTarget: target,
      ...extra,
    });
  }

  it("coerces form drafts back into numbers for the worker config", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    const snap = snapshot(draft);
    expect(snap.config.rank).toBe(8);
    expect(snap.config.learningRate).toBe(0.0001);
    expect(snap.config.steps).toBe(1000);
    expect(snap.config.optimizer).toBe("adamw");
  });

  it("threads dataset + output identity and defaults to a dry run", () => {
    const snap = snapshot(configDraftFromTarget(target, dataset, ["auto"]));
    expect(snap.targetId).toBe("sdxl_lora");
    expect(snap.datasetId).toBe("ds-1");
    expect(snap.datasetVersion).toBe(3);
    expect(snap.outputName).toBe("Kelsie LoRA");
    expect(snap.dryRun).toBe(true);
  });

  it("honors an explicit dryRun=false for a real run", () => {
    const snap = snapshot(configDraftFromTarget(target, dataset, ["auto"]), { dryRun: false });
    expect(snap.dryRun).toBe(false);
  });

  it("omits the LoKr factor for a lora network but keeps it for lokr", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    expect(snapshot(draft).config.advanced).not.toHaveProperty("decomposeFactor");
    const lokr = snapshot({ ...draft, networkType: "lokr", decomposeFactor: "16" });
    expect(lokr.config.advanced.networkType).toBe("lokr");
    expect(lokr.config.advanced.decomposeFactor).toBe(16);
  });

  it("drops a blank LoKr factor so the worker applies its own -1 default", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"]);
    const snap = snapshot({ ...draft, networkType: "lokr", decomposeFactor: "" });
    expect(snap.config.advanced).not.toHaveProperty("decomposeFactor");
  });

  it("derives sample prompts from the trigger word", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot(draft);
    expect(snap.config.advanced.samplePrompts[0]).toContain("ohwx woman");
    expect(snap.config.advanced.samplePrompts).toHaveLength(4);
  });

  it("prefills the sample-prompts draft and defaults the sample count to 4 (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    expect(draft.sampleCount).toBe("4");
    expect(draft.samplePrompts.split("\n")).toHaveLength(4);
    expect(draft.samplePrompts).toContain("ohwx woman");
  });

  it("sends the user's edited prompt pool verbatim and a custom count (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot({ ...draft, samplePrompts: "a cat\n  a dog  \n\na bird", sampleCount: "6" });
    expect(snap.config.advanced.sampleCount).toBe(6);
    // Blank lines dropped, surviving lines trimmed; backends cycle this pool to the count.
    expect(snap.config.advanced.samplePrompts).toEqual(["a cat", "a dog", "a bird"]);
  });

  it("falls back to trigger-derived prompts when the pool is cleared (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot({ ...draft, samplePrompts: "   \n  " });
    expect(snap.config.advanced.samplePrompts).toHaveLength(4);
    expect(snap.config.advanced.samplePrompts[0]).toContain("ohwx woman");
  });

  it("drops a blank sample count so the worker applies its own default (sc-8671)", () => {
    const draft = configDraftFromTarget(target, dataset, ["auto"], "ohwx woman");
    const snap = snapshot({ ...draft, sampleCount: "" });
    expect(snap.config.advanced).not.toHaveProperty("sampleCount");
  });

  it("carries the preset id/version when a preset is selected", () => {
    const snap = snapshot(configDraftFromTarget(target, dataset, ["auto"]), {
      selectedPreset: { id: "preset-1", version: 5 },
    });
    expect(snap.presetId).toBe("preset-1");
    expect(snap.presetVersion).toBe(5);
  });
});
