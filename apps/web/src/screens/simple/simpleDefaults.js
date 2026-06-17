// Smart-default data for Simple mode's "Make a picture" surface. Keeping the
// data here (not in the screen) makes the proxy → engine mappings auditable and
// reusable by the upcoming "Make a video" screen.
//
// LOOKS: each look is, eventually, a curated recipe preset (model + params +
// prompt fragments) surfaced via `recipePresetId`. Those built-in presets aren't
// seeded yet (Phase 7), so for now a look applies a `promptSuffix` — functional
// against the real engine today, and `presetId` is the forward hook.
export const LOOKS = [
  { id: "photo", label: "Photo", tone: "look-photo", presetId: null, promptSuffix: "professional photograph, realistic, natural lighting, fine detail" },
  { id: "cinematic", label: "Cinematic", tone: "look-cine", presetId: null, promptSuffix: "cinematic film still, dramatic lighting, shallow depth of field, color graded" },
  { id: "illustration", label: "Illustration", tone: "look-illus", presetId: null, promptSuffix: "illustration, clean linework, flat bold colors" },
  { id: "anime", label: "Anime", tone: "look-anime", presetId: null, promptSuffix: "anime style, cel shaded, vibrant colors" },
  { id: "render3d", label: "3D", tone: "look-3d", presetId: null, promptSuffix: "3D render, soft global illumination, subsurface scattering" },
  { id: "watercolor", label: "Watercolor", tone: "look-water", presetId: null, promptSuffix: "watercolor painting, soft washes, textured paper" },
];

// Friendly shapes → a target aspect we snap to the model's allowed resolutions.
export const SHAPES = [
  { id: "square", label: "Square", width: 1024, height: 1024 },
  { id: "portrait", label: "Portrait", width: 768, height: 1024 },
  { id: "landscape", label: "Landscape", width: 1024, height: 768 },
  { id: "wide", label: "Wide", width: 1280, height: 720 },
];

export const COUNT_OPTIONS = [1, 2, 4];

// "Make it sharper" → the createImageJob `upscale` payload (off = omit entirely).
export const UPSCALE_OPTIONS = [
  { id: "off", label: "Off", factor: null },
  { id: "x2", label: "2× larger", factor: 2 },
  { id: "x4", label: "4× larger", factor: 4 },
];

// Fallback resolution menu when a model doesn't advertise `limits.resolutions`.
export const FALLBACK_RESOLUTIONS = ["1024x1024", "768x1024", "1024x768", "1280x720", "720x1280"];

// Friendly "Movement" chips for Make a video → the worker's `advanced.motion`
// vocabulary (see MOTIONS in VideoStudio). Kept to a calm subset.
export const VIDEO_MOTIONS = [
  { id: "static", label: "Hold steady", motion: "static" },
  { id: "push", label: "Push in", motion: "slow push-in" },
  { id: "pull", label: "Pull out", motion: "pull out" },
  { id: "pan", label: "Pan across", motion: "pan right" },
  { id: "handheld", label: "Handheld", motion: "handheld" },
];

// Quality tiers → the worker's `quality` value (jobTypes qualityChoices).
export const QUALITY_CHOICES = [
  { id: "fast", label: "Draft (fast)" },
  { id: "balanced", label: "Balanced" },
  { id: "best", label: "Final (slow)" },
];

export const FALLBACK_VIDEO_RESOLUTIONS = ["1280x720", "768x1280", "768x768"];
export const FALLBACK_DURATIONS = [4, 6, 8];

// Compose the prompt the engine receives: the user's words plus the look's
// descriptive suffix (deduped if the user already typed it).
export function composePrompt(prompt, look) {
  const base = String(prompt ?? "").trim();
  if (!look?.promptSuffix) return base;
  if (base.toLowerCase().includes(look.promptSuffix.toLowerCase())) return base;
  return base ? `${base}, ${look.promptSuffix}` : look.promptSuffix;
}
