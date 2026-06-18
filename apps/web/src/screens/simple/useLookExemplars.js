import { useCallback, useEffect, useMemo, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { LOOKS, composePrompt } from "./simpleDefaults.js";
import { defaultImageModelId, textToImageModels } from "./simpleModel.js";
import { readLookExemplars, writeLookExemplar } from "./lookExemplars.js";
import { errorStatuses, terminalStatuses } from "../../jobTypes.js";

// One canonical subject rendered in each look, so the tiles read as a true
// before/after of the style — and it echoes the cabin-at-dusk placeholder it
// replaces, for visual continuity.
const SUBJECT = "a small cabin by a mountain lake at dusk";
// Square and small: the tiles are tiny, so there's no reason to spend a full
// 1024² render on a preview.
const PREVIEW_DIMS = { width: 768, height: 768, resolution: "768x768" };

function randomSeed() {
  return Math.floor(Math.random() * 2_000_000_000);
}

// Renders + caches one image per look on demand. Shared by Make a picture and
// Make a video so both grids show the same engine-rendered exemplars. Rendering
// is event-driven: we submit createImageJob and resolve the resulting asset by
// watching the shared jobs/asset lists (the same SSE feed the rest of the app
// uses) rather than polling.
export function useLookExemplars(preferredModelId = null) {
  const {
    activeProject,
    imageModels = [],
    createImageJob,
    jobs = [],
    recentImageAssets = [],
    mediaAssets = [],
  } = useAppContext();

  const projectId = activeProject?.id ?? null;
  // Preview with the model the user picked for Make a picture, so the look tiles
  // match what Create renders; fall back to the same default when none is passed.
  const model = useMemo(() => {
    const choices = textToImageModels(imageModels);
    if (preferredModelId) {
      const match = choices.find((entry) => entry.id === preferredModelId);
      if (match) return match;
    }
    const id = defaultImageModelId(choices);
    return choices.find((entry) => entry.id === id) ?? null;
  }, [imageModels, preferredModelId]);
  const modelKey = model?.id ?? null;
  const [exemplars, setExemplars] = useState(() => readLookExemplars(projectId, modelKey));
  const [pending, setPending] = useState({}); // lookId → jobId in flight

  // Re-seed from storage when the workspace/model changes; drop any in-flight markers.
  useEffect(() => {
    setExemplars(readLookExemplars(projectId, modelKey));
    setPending({});
  }, [projectId, modelKey]);

  // Resolve completed render jobs to their asset and cache them.
  useEffect(() => {
    if (!projectId || Object.keys(pending).length === 0) return;
    const pool = [...recentImageAssets, ...mediaAssets];
    let changed = false;
    const next = { ...pending };
    for (const [lookId, jobId] of Object.entries(pending)) {
      const job = jobs.find((entry) => entry.id === jobId);
      if (!job || !terminalStatuses.has(job.status)) continue;
      if (!errorStatuses.has(job.status)) {
        const assetId = job.result?.assetIds?.[0] ?? pool.find((asset) => asset.lineage?.jobId === jobId)?.id ?? null;
        const asset = pool.find((entry) => entry.id === assetId) ?? job.result?.assets?.[0] ?? null;
        if (assetId) {
          const entry = { assetId, url: asset?.url ?? null, seed: job.result?.seed ?? null };
          writeLookExemplar(projectId, modelKey, lookId, entry);
          setExemplars((current) => ({ ...current, [lookId]: entry }));
        }
      }
      delete next[lookId];
      changed = true;
    }
    if (changed) setPending(next);
  }, [jobs, pending, projectId, modelKey, recentImageAssets, mediaAssets]);

  const canRender = Boolean(model && projectId && typeof createImageJob === "function");

  const refresh = useCallback(
    async (lookIds) => {
      if (!canRender) return;
      const ids = Array.isArray(lookIds) && lookIds.length ? lookIds : LOOKS.map((look) => look.id);
      for (const lookId of ids) {
        const look = LOOKS.find((entry) => entry.id === lookId);
        if (!look) continue;
        const job = await createImageJob({
          mode: "text_to_image",
          prompt: composePrompt(SUBJECT, look),
          negativePrompt: "",
          model: model.id,
          count: 1,
          seed: randomSeed(),
          width: PREVIEW_DIMS.width,
          height: PREVIEW_DIMS.height,
          recipePresetId: look.presetId ?? null,
          loras: [],
          advanced: { resolution: PREVIEW_DIMS.resolution },
        });
        if (job?.id) {
          setPending((current) => ({ ...current, [lookId]: job.id }));
        }
      }
    },
    [canRender, createImageJob, model],
  );

  // A renderable asset for a look: prefer the live sidecar (best for the shared
  // thumbnail component), fall back to a minimal { url } from the cache.
  const assetForLook = useCallback(
    (lookId) => {
      const entry = exemplars[lookId];
      if (!entry) return null;
      const pool = [...recentImageAssets, ...mediaAssets];
      const live = pool.find((asset) => asset.id === entry.assetId);
      if (live) return live;
      return entry.url ? { id: entry.assetId, type: "image", url: entry.url } : null;
    },
    [exemplars, recentImageAssets, mediaAssets],
  );

  const refreshing = useMemo(() => Object.keys(pending).length > 0, [pending]);
  const hasAny = useMemo(() => Object.keys(exemplars).length > 0, [exemplars]);

  return { assetForLook, pending, refresh, refreshing, canRender, hasAny };
}
