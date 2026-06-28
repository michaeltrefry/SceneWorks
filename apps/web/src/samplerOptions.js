// Shared labels + helpers for the configurable sampler/scheduler controls. The
// curated vocabulary (epic 7114) matches the native mlx-gen / candle-gen gen-core
// Solver / Scheduler registries exactly — the names sent in the job's `advanced`
// block ARE the engine's sampler/scheduler ids. The studios source their dropdown
// contents from each model's per-backend manifest `limits.samplers` /
// `limits.schedulers` arrays (base `limits` overridden by `mlx.limits` /
// `candle.limits` for the active backend) so invalid combos are never selectable.

export const SAMPLER_LABELS = {
  default: "Model default",
  euler: "Euler",
  euler_ancestral: "Euler ancestral",
  heun: "Heun (2nd-order)",
  dpmpp_2m: "DPM++ (2M)",
  dpmpp_sde: "DPM++ SDE",
  uni_pc: "UniPC",
  lcm: "LCM",
  ddim: "DDIM",
};

export const SCHEDULER_LABELS = {
  default: "Model default",
  normal: "Normal",
  simple: "Simple (uniform)",
  karras: "Karras",
  exponential: "Exponential",
  sgm_uniform: "SGM uniform",
  beta: "Beta",
  ddim_uniform: "DDIM uniform",
};

// Classifier-free-guidance policy (the orthogonal guidance axis, epic 7434). Names
// match gen-core's GuidanceMethod registry exactly. "cfg" is the engine's standard
// guidance (the N1 no-op the studio defaults to); the rest are advertised per-model-
// per-backend only where the linked engine honors them — CFG++ (cfg_pp) is MLX-only
// today (sc-8256), so the picker only appears for the SDXL family on the MLX backend.
export const GUIDANCE_METHOD_LABELS = {
  cfg: "CFG (standard)",
  cfg_rescale: "CFG rescale",
  apg: "APG (adaptive)",
  cfg_pp: "CFG++",
};

const SAMPLER_ORDER = [
  "default",
  "euler",
  "euler_ancestral",
  "heun",
  "dpmpp_2m",
  "dpmpp_sde",
  "uni_pc",
  "lcm",
  "ddim",
];
const SCHEDULER_ORDER = [
  "default",
  "normal",
  "simple",
  "karras",
  "exponential",
  "sgm_uniform",
  "beta",
  "ddim_uniform",
];
const GUIDANCE_METHOD_ORDER = ["cfg", "cfg_rescale", "apg", "cfg_pp"];

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

// The effective `limits` for the active backend: the per-backend `mlx.limits` /
// `candle.limits` override when present, else the base `limits` (epic 7114 P5
// per-model-per-backend gating). `backend` is the active worker's backend
// ("mlx" | "candle"); a null/unknown backend falls back to the base menu.
function effectiveLimits(model, backend) {
  const override = backend ? model?.[backend]?.limits : null;
  return override ?? model?.limits;
}

// Pull the menu out of a model manifest entry, falling back to default-only.
// When the menu has fewer than 2 entries, the studio hides the dropdown — the
// caller can use `samplerMenu.length > 1` to gate rendering.
export function samplerOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.samplers) ? limits.samplers : ["default"];
  return uniqueOrdered(values, SAMPLER_ORDER);
}

export function schedulerOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.schedulers) ? limits.schedulers : ["default"];
  return uniqueOrdered(values, SCHEDULER_ORDER);
}

// The guidance-method menu the active backend honors for this model (epic 7434).
// Falls back to the engine's standard ["cfg"] when the model advertises none — the
// studio hides the picker when fewer than 2 methods are offered (length > 1 gate),
// so only models that actually expose an alternative (CFG++ on SDXL/MLX) show it.
export function guidanceMethodOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.guidanceMethods) ? limits.guidanceMethods : ["cfg"];
  return uniqueOrdered(values, GUIDANCE_METHOD_ORDER);
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

// The guidance method the studio selects initially. Defaults to "cfg" (the engine's
// standard guidance, the N1 no-op) so existing recipes are unchanged; a model may
// pin a different default via `defaults.guidanceMethod`.
export function guidanceMethodDefaultFromModel(model) {
  const value = model?.defaults?.guidanceMethod;
  return typeof value === "string" && value.length ? value : "cfg";
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
