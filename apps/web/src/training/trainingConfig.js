// Training target/preset config helpers + label maps (sc-4199). Extracted
// verbatim from TrainingStudio.jsx: the option/label lookup tables, the
// preset selection helpers, and the two pure config builders the screen used to
// bury — configDraftFromTarget (target/preset → form draft) and
// trainingConfigSnapshot (form draft → worker payload). No React, no app state.

import {
  asText,
  compactObject,
  normalizeTrainingAdapterVersion,
  numberFromDraft,
  numericDraft,
} from "./drafts.js";

export const defaultGpuOptions = ["auto"];
export const defaultOptimizerOptions = ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"];
export const timestepTypeOptions = ["sigmoid", "linear", "weighted"];
export const timestepBiasOptions = ["balanced", "high_noise", "low_noise"];
export const lossTypeOptions = ["mse", "mae"];
// Learning-rate schedulers the worker actually honors (constant holds the LR
// fixed; linear/cosine decay it over the run). Distinct from the timestep/noise
// scheduler above. The target's `limits.lrSchedulers` overrides this fallback.
export const lrSchedulerOptions = ["constant", "linear", "cosine"];
export const optimizerLabels = {
  adam: "Adam",
  adamw: "AdamW",
  adamw8bit: "AdamW 8-bit",
  prodigy: "Prodigy",
  prodigyopt: "Prodigy",
  rose: "Rose",
};
// Adapter network parameterization. `lora` is the universal default; `lokr`
// (LyCORIS Kronecker) is offered only on targets whose `limits.networkTypes`
// advertise it (epic 2193).
export const networkTypeLabels = {
  lora: "LoRA",
  lokr: "LoKr (LyCORIS Kronecker)",
};
// Versions of the ostris de-distill training adapter (Z-Image-Turbo only). The
// worker maps these to the matching repo file; legacy "v2-default" normalizes to v2.
export const trainingAdapterVersionOptions = ["v1", "v2"];
export const trainingAdapterVersionLabels = {
  v1: "v1 — stable (smaller)",
  v2: "v2 — experimental (heavier de-distill)",
};
export const configFieldLabels = {
  outputName: "LoRA name",
  triggerWord: "Trigger phrase",
  outputScope: "Output scope",
  qualityPreset: "Quality preset",
  requestedGpu: "Requested GPU",
  rank: "Rank",
  alpha: "Alpha",
  networkType: "Network type",
  decomposeFactor: "LoKr factor",
  optimizer: "Optimizer",
  learningRate: "Learning rate",
  weightDecay: "Weight decay",
  lrScheduler: "LR scheduler",
  lrWarmupSteps: "LR warmup steps",
  steps: "Steps",
  timestepType: "Timestep type",
  timestepBias: "Timestep bias",
  lossType: "Loss type",
  trainingAdapterVersion: "De-distill adapter",
  gradientCheckpointing: "Gradient checkpointing",
  resolution: "Resolution",
  precision: "Precision",
  saveEvery: "Checkpoint cadence",
  sampleEvery: "Sample cadence",
  sampleSteps: "Sample steps",
  sampleGuidanceScale: "Guidance scale",
  sampleCount: "Sample count",
  samplePrompts: "Sample prompts",
  batchSize: "Batch size",
  gradientAccumulation: "Gradient accumulation",
  seed: "Seed",
};

export function rangeOptions(limits, key) {
  return Array.isArray(limits?.[key]) ? limits[key] : [];
}

export function optimizerLabel(value) {
  return optimizerLabels[value] ?? value;
}

export function networkTypeLabel(value) {
  return networkTypeLabels[value] ?? value;
}

export function optionLabel(value) {
  return String(value ?? "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function presetSortValue(preset) {
  const order = Number(preset?.ui?.order);
  return Number.isFinite(order) ? order : 999;
}

export function presetsForTarget(presets, targetId) {
  return (presets ?? [])
    .filter((preset) => preset.targetId === targetId)
    .slice()
    .sort((left, right) => presetSortValue(left) - presetSortValue(right) || left.name.localeCompare(right.name));
}

export function defaultPresetForTarget(presets, targetId) {
  const targetPresets = presetsForTarget(presets, targetId);
  return targetPresets.find((preset) => preset.ui?.default) ?? targetPresets[0] ?? null;
}

export function outputKindLabel(target) {
  const kind = String(target?.outputKind ?? "output").toLowerCase();
  if (kind === "lora") {
    return "LoRA";
  }
  return kind.replaceAll("_", " ");
}

export function configDraftFromTarget(target, dataset, gpuOptions, triggerPhrase = "", preset = null, previousDraft = {}) {
  const defaults = preset?.config ?? target?.defaults ?? {};
  const advanced = defaults.advanced ?? {};
  const firstGpu = gpuOptions[0] ?? "";
  const requestedGpu = asText(advanced.requestedGpu || firstGpu);
  const outputLabel = outputKindLabel(target);
  return {
    outputName: previousDraft.outputName ?? (dataset?.name ? `${dataset.name} ${outputLabel}` : ""),
    triggerWord: triggerPhrase || asText(defaults.triggerWord),
    outputScope: asText(advanced.outputScope),
    qualityPreset: asText(advanced.qualityPreset),
    requestedGpu: gpuOptions.includes(requestedGpu) ? requestedGpu : firstGpu,
    rank: numericDraft(defaults.rank),
    alpha: numericDraft(defaults.alpha),
    networkType: asText(advanced.networkType || "lora"),
    // LoKr block-decomposition factor; -1 = auto. Only consumed when networkType
    // is lokr (the worker ignores it otherwise).
    decomposeFactor: numericDraft(advanced.decomposeFactor ?? -1),
    optimizer: asText(defaults.optimizer),
    learningRate: numericDraft(defaults.learningRate),
    weightDecay: numericDraft(advanced.weightDecay),
    lrScheduler: asText(advanced.lrScheduler || "constant"),
    lrWarmupSteps: numericDraft(advanced.lrWarmupSteps),
    steps: numericDraft(defaults.steps),
    timestepType: asText(advanced.timestepType || "sigmoid"),
    timestepBias: asText(advanced.timestepBias || "balanced"),
    lossType: asText(advanced.lossType || "mse"),
    trainingAdapterRepo: asText(advanced.trainingAdapterRepo),
    trainingAdapterVersion: normalizeTrainingAdapterVersion(advanced.trainingAdapterVersion),
    gradientCheckpointing: advanced.gradientCheckpointing !== false,
    resolution: numericDraft(defaults.resolution),
    precision: asText(advanced.mixedPrecision),
    saveEvery: numericDraft(defaults.saveEvery),
    sampleEvery: numericDraft(advanced.sampleEvery),
    sampleSteps: numericDraft(advanced.sampleSteps),
    sampleGuidanceScale: numericDraft(advanced.sampleGuidanceScale),
    sampleCount: numericDraft(advanced.sampleCount ?? defaultSampleCount),
    // Prefilled with the preset's prompts when it carries them, otherwise the
    // trigger-derived defaults. The screen keeps this in sync with the trigger
    // phrase until the user edits it (configPromptsFollowTrigger).
    samplePrompts: promptListToLines(
      Array.isArray(advanced.samplePrompts) && advanced.samplePrompts.length
        ? advanced.samplePrompts
        : samplePromptsFromTrigger(triggerPhrase || asText(defaults.triggerWord)),
    ),
    batchSize: numericDraft(defaults.batchSize),
    gradientAccumulation: numericDraft(defaults.gradientAccumulation),
    seed: numericDraft(defaults.seed),
  };
}

export function configValidation({ activeDataset, configDraft, selectedTarget }) {
  const warnings = [];
  if (!selectedTarget) {
    warnings.push("Select a training target");
  }
  if (!activeDataset?.id) {
    warnings.push("Select a saved dataset");
  }
  if (!configDraft.outputName?.trim()) {
    warnings.push(`Name the ${outputKindLabel(selectedTarget)} output`);
  }
  if (!configDraft.triggerWord?.trim()) {
    warnings.push("Add a trigger phrase");
  }
  for (const [field, label] of [
    ["rank", "Rank"],
    ["alpha", "Alpha"],
    ["learningRate", "Learning rate"],
    ["steps", "Steps"],
    ["resolution", "Resolution"],
    ["batchSize", "Batch size"],
    ["gradientAccumulation", "Gradient accumulation"],
    ["saveEvery", "Checkpoint cadence"],
  ]) {
    const value = numberFromDraft(configDraft[field]);
    if (!value || value <= 0) {
      warnings.push(`${label} must be greater than zero`);
    }
  }
  return warnings;
}

export function samplePromptsFromTrigger(triggerWord) {
  const trigger = String(triggerWord ?? "").trim() || "the trained subject";
  return [
    `${trigger}, studio portrait, soft key light, detailed face`,
    `${trigger}, full body fashion editorial photo, natural pose`,
    `${trigger}, cinematic outdoor portrait, golden hour`,
    `${trigger}, close-up character portrait, dramatic rim light`,
  ];
}

// Default number of preview images rendered per sample step (sc-8671). Matches
// the four trigger-derived default prompts, so the out-of-the-box behavior is
// unchanged when neither knob is touched. The backends clamp/cycle the prompt
// pool to this count, so it can differ from the number of prompts supplied.
export const defaultSampleCount = 4;

// The sample-prompts textarea holds one prompt per line; the worker payload wants
// a string array. These two convert between the draft string and the array.
export function promptLinesToList(text) {
  return String(text ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function promptListToLines(list) {
  return (Array.isArray(list) ? list : []).join("\n");
}

export function trainingConfigSnapshot({ activeDataset, configDraft, selectedPreset, selectedTarget, dryRun = true }) {
  const defaults = selectedTarget?.defaults ?? {};
  const networkType = asText(configDraft.networkType).trim() || "lora";
  // The user-edited prompt pool, one per line. Empty falls back to the trigger-derived
  // defaults so previews still render (and {trigger} substitution is preserved). The
  // backends cycle/truncate this pool to sampleCount, so its length need not equal the count.
  const editedPrompts = promptLinesToList(configDraft.samplePrompts);
  const samplePrompts = editedPrompts.length ? editedPrompts : samplePromptsFromTrigger(configDraft.triggerWord);
  const advanced = compactObject({
    ...(defaults.advanced ?? {}),
    networkType,
    // LoKr factor only matters for lokr; omit it otherwise so lora jobs stay clean.
    decomposeFactor: networkType === "lokr" ? numberFromDraft(configDraft.decomposeFactor) : undefined,
    weightDecay: numberFromDraft(configDraft.weightDecay),
    lrScheduler: asText(configDraft.lrScheduler).trim() || "constant",
    lrWarmupSteps: numberFromDraft(configDraft.lrWarmupSteps),
    timestepType: asText(configDraft.timestepType).trim(),
    timestepBias: asText(configDraft.timestepBias).trim(),
    lossType: asText(configDraft.lossType).trim(),
    // Preset-only advanced keys (the submit spreads target defaults, not the
    // preset), so carry the de-distill adapter through explicitly — the worker
    // only fuses it when config.advanced.trainingAdapterRepo is present.
    trainingAdapterRepo: asText(configDraft.trainingAdapterRepo).trim(),
    trainingAdapterVersion: asText(configDraft.trainingAdapterVersion).trim(),
    gradientCheckpointing: Boolean(configDraft.gradientCheckpointing),
    mixedPrecision: asText(configDraft.precision).trim(),
    sampleEvery: numberFromDraft(configDraft.sampleEvery),
    sampleSteps: numberFromDraft(configDraft.sampleSteps),
    sampleGuidanceScale: numberFromDraft(configDraft.sampleGuidanceScale),
    sampleCount: numberFromDraft(configDraft.sampleCount),
    samplePrompts,
    qualityPreset: configDraft.qualityPreset,
    outputScope: configDraft.outputScope,
    requestedGpu: configDraft.requestedGpu,
  });
  return {
    targetId: selectedTarget.id,
    datasetId: activeDataset.id,
    datasetVersion: activeDataset.version,
    outputName: configDraft.outputName.trim(),
    dryRun,
    outputScope: configDraft.outputScope,
    qualityPreset: configDraft.qualityPreset,
    requestedGpu: configDraft.requestedGpu,
    presetId: selectedPreset?.id,
    presetVersion: selectedPreset?.version,
    config: {
      rank: numberFromDraft(configDraft.rank),
      alpha: numberFromDraft(configDraft.alpha),
      learningRate: numberFromDraft(configDraft.learningRate),
      steps: numberFromDraft(configDraft.steps),
      batchSize: numberFromDraft(configDraft.batchSize),
      gradientAccumulation: numberFromDraft(configDraft.gradientAccumulation),
      resolution: numberFromDraft(configDraft.resolution),
      saveEvery: numberFromDraft(configDraft.saveEvery),
      seed: numberFromDraft(configDraft.seed),
      optimizer: asText(configDraft.optimizer).trim(),
      triggerWord: asText(configDraft.triggerWord).trim(),
      advanced,
    },
  };
}
