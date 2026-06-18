// Per-project cache of engine-rendered "look" exemplars (look id → the image we
// rendered to preview that style). Mirrors the tiny localStorage modules used
// for ui-mode/theme/accent: keyed by project so a deleted project can't leave
// dangling asset ids behind, and every access is guarded for private-mode.
export const LOOK_EXEMPLARS_STORAGE_KEY = "sceneworks-look-exemplars";

function readAll() {
  if (typeof localStorage === "undefined") return {};
  try {
    const parsed = JSON.parse(localStorage.getItem(LOOK_EXEMPLARS_STORAGE_KEY) || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function readLookExemplars(projectId, modelId) {
  if (!projectId || !modelId) return {};
  const all = readAll();
  const project = all[projectId];
  const byModel = project && typeof project === "object" ? project[modelId] : null;
  return byModel && typeof byModel === "object" ? byModel : {};
}

// Persist one look's exemplar. entry = { assetId, url, seed }.
export function writeLookExemplar(projectId, modelId, lookId, entry) {
  if (!projectId || !modelId || !lookId || typeof localStorage === "undefined") return;
  try {
    const all = readAll();
    const project = all[projectId] && typeof all[projectId] === "object" ? all[projectId] : {};
    all[projectId] = {
      ...project,
      [modelId]: { ...(project[modelId] || {}), [lookId]: entry },
    };
    localStorage.setItem(LOOK_EXEMPLARS_STORAGE_KEY, JSON.stringify(all));
  } catch {
    // Private mode or quota — exemplars are a cache, so a failed write is fine.
  }
}
