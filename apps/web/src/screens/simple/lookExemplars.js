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

export function readLookExemplars(projectId) {
  if (!projectId) return {};
  const all = readAll();
  return all[projectId] && typeof all[projectId] === "object" ? all[projectId] : {};
}

// Persist one look's exemplar. entry = { assetId, url, seed }.
export function writeLookExemplar(projectId, lookId, entry) {
  if (!projectId || !lookId || typeof localStorage === "undefined") return;
  try {
    const all = readAll();
    all[projectId] = { ...(all[projectId] || {}), [lookId]: entry };
    localStorage.setItem(LOOK_EXEMPLARS_STORAGE_KEY, JSON.stringify(all));
  } catch {
    // Private mode or quota — exemplars are a cache, so a failed write is fine.
  }
}
