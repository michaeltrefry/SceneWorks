// Pure image-job builders + model/engine helpers (extracted from ImageEditor.jsx,
// sc-6112). React/Konva/DOM-free so they can be shared by the editor AND the Library
// batch flow WITHOUT pulling react-konva into the eagerly-loaded Library bundle (the
// editor is deliberately lazy-loaded to keep konva off the main/test path). ImageEditor.jsx
// re-exports these so its public surface (and its tests) are unchanged.

// Models that can edit an existing image with a prompt — the manifest tags them
// with an `edit_image`/`image_edit` capability (same filter the Image Studio uses).
export function editCapableModels(imageModels) {
  return (imageModels ?? []).filter((model) => {
    const caps = model.capabilities ?? [];
    return caps.includes("edit_image") || caps.includes("image_edit");
  });
}

// Models that can run the tile-ControlNet detail refine — the manifest tags them
// `image_detail` (sc-2437/sc-2438; SDXL/RealVisXL only). RealVisXL is the recommended
// photoreal backbone per the spike.
export function detailCapableModels(imageModels) {
  return (imageModels ?? []).filter((model) => (model.capabilities ?? []).includes("image_detail"));
}

// The `POST /api/v1/image/jobs` body for a prompt edit (sc-2435). Reuses the existing
// `mode:"edit_image"` flow: a `sourceAssetId` image is edited to the new `width`×`height`
// (the worker fits the source per `fitMode`). Pure for unit testing.
export function buildEditJobBody({
  project,
  requestedGpu,
  sourceAssetId,
  maskAssetId,
  referenceAssetIds = null,
  model,
  prompt,
  seed,
  width,
  height,
  fitMode = "crop",
}) {
  const body = {
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    mode: "edit_image",
    sourceAssetId,
    model,
    prompt,
    negativePrompt: "",
    width,
    height,
    // How the source is fitted to width×height (epic 2551). For canvas-extend outpaint
    // the dims are the larger target aspect and the worker generates the new border.
    fitMode,
    seed: seed == null || seed === "" ? null : Number(seed),
    count: 1,
    advanced: {},
  };
  // Inpaint mask (sc-2436): only sent for inpaint-capable models with a painted
  // region; the worker confines the edit to it. Omitted entirely otherwise.
  if (maskAssetId) body.maskAssetId = maskAssetId;
  // Multi-reference edit (sc-6107): a FLUX.2 `multiReference` model conditions the edit jointly on
  // the working image + the user's reference image(s). The list already leads with the working
  // scratch id (see `editReferenceIds`); the worker prefers a non-empty `referenceAssetIds` over
  // `sourceAssetId`, so the working image must ride in the list or it's dropped from the edit.
  if (referenceAssetIds && referenceAssetIds.length) body.referenceAssetIds = referenceAssetIds;
  return body;
}

// Upscale engines + their valid factors (sc-2433). Real-ESRGAN 2x/4x is the cross-platform default
// (`ort`: CoreML on Mac / CUDA off-Mac, sc-3489 / sc-5499); SeedVR2 2x/4x is the one-step diffusion
// upscaler (native MLX on Mac / candle off-Mac, epic 4811 / sc-5928 / sc-5160) and exposes a
// detail/softness control (`softness: true`). `aura-sr` is kept here only so a stale saved selection
// gracefully falls back — it is hidden on every platform (dropped, sc-3668 / sc-5499) via
// `macUpscaleEngineBlocked`.
export const UPSCALE_ENGINES = [
  { key: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
  { key: "seedvr2", label: "SeedVR2", factors: [2, 4], softness: true },
  { key: "aura-sr", label: "AuraSR", factors: [4] },
];

export function upscaleFactorsForEngine(engineKey) {
  const found = UPSCALE_ENGINES.find((entry) => entry.key === engineKey);
  return found ? found.factors : [2, 4];
}

export function upscaleEngineHasSoftness(engineKey) {
  return Boolean(UPSCALE_ENGINES.find((entry) => entry.key === engineKey)?.softness);
}

// The `POST /api/v1/jobs` body for a standalone image_upscale job (sc-2431). The
// worker reads sourceAssetId/factor/engine from the payload; displayName names the
// result. `softness` (0..1) is a SeedVR2-only detail knob (sc-4815) — omitted for engines
// that ignore it. Pure for unit testing.
export function buildUpscaleJobBody({ project, requestedGpu, sourceAssetId, factor, engine, displayName, softness }) {
  const payload = { projectId: project.id, sourceAssetId, factor, engine, displayName };
  if (upscaleEngineHasSoftness(engine) && typeof softness === "number") {
    payload.softness = softness;
  }
  return {
    type: "image_upscale",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload,
  };
}

// The `POST /api/v1/jobs` body for a standalone image_detail job (sc-2438). Same
// generic-jobs shape as upscale; the worker reads model + advanced.strength/cnScale
// from the payload (recipe defaults locked by the sc-2437 spike). Pure for testing.
export function buildDetailJobBody({ project, requestedGpu, sourceAssetId, model, strength, cnScale, displayName }) {
  return {
    type: "image_detail",
    projectId: project.id,
    projectName: project.name ?? null,
    requestedGpu,
    payload: {
      projectId: project.id,
      sourceAssetId,
      model,
      displayName,
      advanced: { strength, cnScale },
    },
  };
}
