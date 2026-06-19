// Mac UI gating (sc-3486, epic 3482). The API's GET /api/v1/capabilities/mac surfaces the
// `mac_rust_supported` oracle to the client: a master switch (`macGatingActive`, the
// SCENEWORKS_MLX_REQUIRED rollout flag) plus per-feature support for the non-model Python
// surfaces, and each model on GET /api/v1/models carries a `macSupport` block. When gating is
// active (a Mac in MLX-required mode) the studios hide/disable models and feature controls that
// can only run on the Python torch path, so a Mac user never reaches a `mlx_unsupported` error
// after submit. On Windows/Linux (and a Mac still in observe mode) `macGatingActive` is false and
// every helper here is a no-op, so non-Mac pickers are untouched.

export const MAC_NOT_AVAILABLE_LABEL = "Not available on Mac (Rust/MLX only)";

// The inert default used until the capabilities endpoint responds (and the permanent value on
// non-Mac / observe-mode deployments). Every helper below short-circuits to "not blocked".
export const DEFAULT_MAC_CAPABILITIES = {
  macGatingActive: false,
  platform: "",
  notAvailableLabel: MAC_NOT_AVAILABLE_LABEL,
  features: {},
  training: { supportedKernels: [], lokrOnWanSupported: false },
};

function label(caps) {
  return caps?.notAvailableLabel || MAC_NOT_AVAILABLE_LABEL;
}

// Compose the user-facing affordance text from a backend UnsupportedReason
// ({ feature, detail, suggestedEpic }).
export function macReasonText(caps, reason) {
  const base = label(caps);
  if (!reason) return base;
  const detail = reason.detail || reason.feature || "";
  const epic = reason.suggestedEpic ? ` (${reason.suggestedEpic})` : "";
  return detail ? `${base} — ${detail}${epic}` : base;
}

function featureLabel(feature) {
  switch (feature) {
    case "pose":
      return "pose conditioning";
    case "reference":
      return "reference / identity conditioning";
    case "edit":
      return "image editing";
    case "lycoris":
      return "third-party LyCORIS LoRA";
    default:
      return feature;
  }
}

// Whether gating is engaged at all (a Mac in MLX-required mode).
export function macGatingActive(caps) {
  return Boolean(caps?.macGatingActive);
}

// A whole model that can't run in the Rust/MLX flow on Mac (torch-only) → hidden/disabled in
// pickers. Returns `{ blocked, reason, text }` or `null`.
export function macModelBlock(model, caps) {
  if (!macGatingActive(caps)) return null;
  if (model?.macSupport?.supported === false) {
    const reason = model.macSupport.reason ?? null;
    return { blocked: true, reason, text: macReasonText(caps, reason) };
  }
  return null;
}

// A per-model feature control (pose / reference / edit / lycoris) the model can't run on MLX.
// Returns `{ blocked, text }` or `null`.
export function macModelFeatureBlock(model, caps, feature) {
  if (!macGatingActive(caps)) return null;
  const features = model?.macSupport?.features;
  if (features && features[feature] === false) {
    return {
      blocked: true,
      text: `${label(caps)} — this model's ${featureLabel(feature)} is not available in the native flow on Mac.`,
    };
  }
  return null;
}

// Whether a video_generate mode routes to MLX for the given (supported) video model.
export function macVideoModeBlock(model, caps, mode) {
  if (!macGatingActive(caps)) return null;
  const modes = model?.macSupport?.features?.videoModes;
  if (modes && modes[mode] === false) {
    return {
      blocked: true,
      // On Mac the runtime is MLX-only (epic 3482) — there is no torch fallback, so a
      // blocked mode simply isn't served for this model here (some modes have no MLX path
      // on this engine; others, like Bernini's editing/reference modes, are MLX-only and
      // exclusive to another model). Don't call it "torch-only".
      text: `${label(caps)} — the selected model doesn't support this mode on macOS.`,
    };
  }
  return null;
}

// A non-model Python surface (imageUpscale, poseFromPhoto, personDetect, datasetCaptioning).
// Returns `{ blocked, reason, text }` or `null`. Video modes are gated per-model via
// macVideoModeBlock, not here (sc-3773).
export function macFeatureBlock(caps, key) {
  if (!macGatingActive(caps)) return null;
  const feature = caps?.features?.[key];
  if (feature && feature.supported === false) {
    const reason = feature.reason ?? null;
    return { blocked: true, reason, text: macReasonText(caps, reason) };
  }
  return null;
}

// Whether a specific image_upscale engine should be hidden from the picker on this platform.
// Two engines are platform-restricted (ids match the worker):
//   * `aura-sr` — a torch-only GigaGAN DROPPED on Mac (sc-3668) and now dropped as an offered engine
//     on EVERY platform (sc-5499): it has no native (MLX/candle) path and the Python torch backend
//     that served it off-Mac is retired in Phase 7 (epic 5483), so the picker hides it everywhere
//     (users don't build on a path about to disappear). Gated off the platform-intrinsic
//     `imageUpscaleAuraSr` capability (`supported: false` on every platform), so — like SeedVR2 — the
//     check is gating-independent (hidden even pre-load, when caps haven't arrived). Real-ESRGAN is
//     the cross-platform default upscaler.
//   * `seedvr2` — the one-step diffusion upscaler (epic 4811 / sc-4815), the INVERSE of AuraSR:
//     backed by native MLX on Mac and the Candle CUDA/NVIDIA port on Windows (sc-5928) + Linux
//     (sc-5160 — candle is CPU+CUDA cross-platform, so Linux rides the Windows port). Gated off the
//     platform-intrinsic `imageUpscaleSeedvr2` capability (true on macOS + Windows + Linux, in any
//     mode), so it is hidden only pre-load (no caps yet) and shown on every GPU platform —
//     independent of the gating switch.
export function macUpscaleEngineBlocked(caps, engine) {
  if (engine === "seedvr2") {
    return caps?.features?.imageUpscaleSeedvr2?.supported !== true;
  }
  if (engine === "aura-sr") {
    return caps?.features?.imageUpscaleAuraSr?.supported !== true;
  }
  return false;
}

// Whether a training kernel has no native Rust trainer (so its base model is gated in Training).
export function macTrainingKernelBlocked(caps, kernel) {
  if (!macGatingActive(caps) || !kernel) return false;
  const supported = caps?.training?.supportedKernels ?? [];
  return !supported.includes(kernel);
}

// Partition a model list into the Mac-available ones (for the picker) — a no-op when gating is off.
export function macAvailableModels(models, caps) {
  if (!macGatingActive(caps)) return models ?? [];
  return (models ?? []).filter((model) => model?.macSupport?.supported !== false);
}

// The models hidden from a picker on Mac, so a screen can show a "N unavailable" affordance.
export function macBlockedModels(models, caps) {
  if (!macGatingActive(caps)) return [];
  return (models ?? []).filter((model) => model?.macSupport?.supported === false);
}
