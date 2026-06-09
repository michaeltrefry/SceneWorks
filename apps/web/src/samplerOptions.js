// Shared labels + helpers for the configurable sampler/scheduler controls
// (epic 1753). The worker's flow-compatible registry lives in
// apps/worker/scene_worker/sampler_registry.py; the studios source their
// dropdown contents from each model's manifest `limits.samplers` /
// `limits.schedulers` arrays so invalid combos are never selectable.

export const SAMPLER_LABELS = {
  default: "Model default",
  euler: "Euler",
  euler_a: "Euler ancestral",
  heun: "Heun (2nd-order)",
  dpmpp: "DPM++ (2M)",
  dpmpp_sde: "DPM++ SDE",
  unipc: "UniPC",
};

export const SCHEDULER_LABELS = {
  default: "Model default",
  simple: "Simple (uniform)",
  shift: "Shift (timestep)",
  karras: "Karras",
  exponential: "Exponential",
  beta: "Beta",
};

const SAMPLER_ORDER = ["default", "euler", "euler_a", "heun", "dpmpp", "dpmpp_sde", "unipc"];
const SCHEDULER_ORDER = ["default", "simple", "shift", "karras", "exponential", "beta"];

function uniqueOrdered(values, order) {
  const seen = new Set();
  const result = [];
  for (const key of order) {
    if (values.includes(key) && !seen.has(key)) {
      seen.add(key);
      result.push(key);
    }
  }
  // Append any keys not in our canonical ordering (forward-compat).
  for (const value of values) {
    if (typeof value === "string" && !seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  }
  return result;
}

// Pull the menu out of a model manifest entry, falling back to default-only.
// When the menu has fewer than 2 entries, the studio hides the dropdown — the
// caller can use `samplerMenu.length > 1` to gate rendering.
export function samplerOptionsFromModel(model) {
  const values = Array.isArray(model?.limits?.samplers) ? model.limits.samplers : ["default"];
  return uniqueOrdered(values, SAMPLER_ORDER);
}

export function schedulerOptionsFromModel(model) {
  const values = Array.isArray(model?.limits?.schedulers) ? model.limits.schedulers : ["default"];
  return uniqueOrdered(values, SCHEDULER_ORDER);
}

// Per-model defaults (UI initial values). The worker never reads these — they
// only set the studio form's initial sampler/scheduler choice. Falls back to
// "default" when the manifest doesn't pin one.
export function samplerDefaultFromModel(model) {
  const value = model?.defaults?.sampler;
  return typeof value === "string" && value.length ? value : "default";
}

export function schedulerDefaultFromModel(model) {
  const value = model?.defaults?.scheduler;
  return typeof value === "string" && value.length ? value : "default";
}

export function schedulerShiftDefaultFromModel(model) {
  const value = Number(model?.defaults?.schedulerShift);
  return Number.isFinite(value) && value > 0 ? value : 3.0;
}

export function stepsDefaultFromModel(model) {
  const value = Number(model?.defaults?.steps);
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function guidanceDefaultFromModel(model) {
  const value = Number(model?.defaults?.guidanceScale);
  return Number.isFinite(value) ? value : null;
}
