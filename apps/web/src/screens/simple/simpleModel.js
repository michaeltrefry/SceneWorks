import { useCallback, useMemo, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";

// Which model the Make-a-picture / Make-a-video surfaces use. Simple mode
// previously hard-wired imageModels[0]/videoModels[0] — whatever the manifest
// listed first — with no way to change it. These hooks expose a friendly picker
// and a sensible default, persisted per browser.
export const SIMPLE_IMAGE_MODEL_KEY = "sceneworks-simple-image-model";
export const SIMPLE_VIDEO_MODEL_KEY = "sceneworks-simple-video-model";

// Default preference for each surface: most versatile / recommended first; the
// first installed one wins.
const IMAGE_MODEL_PREFERENCE = ["sdxl", "realvisxl", "sensenova_u1_8b", "sensenova_u1_8b_fast", "z_image_turbo"];
const VIDEO_MODEL_PREFERENCE = ["ltx_2_3"];

// Only genuine text-to-image models belong in the picture picker — not the
// edit-only or identity/reference models, which can't run a plain prompt.
export function textToImageModels(imageModels = []) {
  return imageModels.filter((model) => (model.capabilities ?? []).includes("text_to_image"));
}

// First model from the preference order that's installed, else the first one.
export function defaultModelId(models = [], preference = []) {
  for (const id of preference) {
    if (models.some((model) => model.id === id)) return id;
  }
  return models[0]?.id ?? null;
}

// Back-compat: the look-exemplar previews resolve the picture default this way.
export function defaultImageModelId(models = []) {
  return defaultModelId(models, IMAGE_MODEL_PREFERENCE);
}

export function modelLabel(model) {
  return model?.ui?.label ?? model?.name ?? model?.id ?? "Model";
}

function readKey(key) {
  if (typeof localStorage === "undefined") return null;
  try {
    return localStorage.getItem(key) || null;
  } catch {
    return null;
  }
}

function writeKey(key, value) {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, value);
  } catch {
    // Private mode — the picker still works for the session, just doesn't persist.
  }
}

// Shared picker state: the saved default if it's still installed, else the
// preference default. Selection is session-only; makeDefault persists it.
function useModelChoice(models, storageKey, preference) {
  const [chosenId, setChosenId] = useState(() => readKey(storageKey));
  const [savedDefault, setSavedDefault] = useState(() => readKey(storageKey));

  const modelId = useMemo(() => {
    if (chosenId && models.some((model) => model.id === chosenId)) return chosenId;
    return defaultModelId(models, preference);
  }, [chosenId, models, preference]);

  const model = useMemo(() => models.find((entry) => entry.id === modelId) ?? null, [models, modelId]);
  const select = useCallback((id) => setChosenId(id), []);
  const makeDefault = useCallback((id = modelId) => {
    const nextId = typeof id === "string" ? id : modelId;
    if (!nextId) return;
    writeKey(storageKey, nextId);
    setSavedDefault(nextId);
    setChosenId(nextId);
  }, [modelId, storageKey]);
  const isDefaultId = useCallback((id = modelId) => Boolean(id) && id === savedDefault, [modelId, savedDefault]);
  const isDefault = isDefaultId(modelId);

  return { models, model, modelId, select, makeDefault, isDefault, isDefaultId };
}

export function useSimpleImageModel() {
  const { imageModels = [] } = useAppContext();
  const models = useMemo(() => textToImageModels(imageModels), [imageModels]);
  return useModelChoice(models, SIMPLE_IMAGE_MODEL_KEY, IMAGE_MODEL_PREFERENCE);
}

export function useSimpleVideoModel() {
  const { videoModels = [] } = useAppContext();
  return useModelChoice(videoModels, SIMPLE_VIDEO_MODEL_KEY, VIDEO_MODEL_PREFERENCE);
}
