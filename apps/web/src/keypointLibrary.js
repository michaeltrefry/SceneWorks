import { useCallback, useEffect, useState } from "react";
import { API_BASE_URL, apiFetch } from "./api.js";
import { useAppContext } from "./context/AppContext.js";

// The reserved global project that holds user-created keypoint (face-angle) preset assets
// (epic 4422, sc-4434). Mirrors crates/sceneworks-core::GLOBAL_KEYPOINTS_PROJECT_ID; hidden
// from the project switcher and addressed directly by the Key Point Library screen. The
// face-angle sibling of GLOBAL_POSES_PROJECT_ID.
export const GLOBAL_KEYPOINTS_PROJECT_ID = "project_global_keypoints";

// The id of the seeded virtual default collection (the built-in 11). Mirrors
// sceneworks_core::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID.
export const BUILTIN_DEFAULT_COLLECTION_ID = "builtin_default";

// The Key Point Library is the union of two preset sources, both served by
// GET /api/v1/keypoints/presets:
//  - BUILT-IN: the 11 validated angle presets (front, three-quarters, profiles, up/down,
//    diagonals). `builtin: true`, `sourceImageRef: null` (their framing was tuned, not
//    captured from a committed photo — render the kps overlay on a neutral canvas).
//  - USER: type:"keypoint" assets captured from an uploaded photo via SCRFD extraction
//    (sc-4433). `builtin: false`, `sourceImageRef` points at the retained source image.
// A preset record: { id, name, angle?, kps:[[x,y]×5], builtin, sourceImageRef, sourceAssetId? }.
// A collection record: { id, name, orderedPresetIds[], isDefault, builtin? }.

// Build a displayable URL for a user preset's retained source image. `sourceImageRef` is a
// path relative to the reserved keypoints project (e.g. "assets/keypoints/asset_x.png");
// mirrors assetMedia.assetUrl's projectId+path fallback. Null/built-in refs → "".
export function keypointSourceImageUrl(sourceImageRef) {
  if (!sourceImageRef) {
    return "";
  }
  const normalized = String(sourceImageRef).replaceAll("\\", "/");
  return `${API_BASE_URL}/api/v1/projects/${GLOBAL_KEYPOINTS_PROJECT_ID}/files/${normalized}`;
}

export async function loadKeypointPresets(token, options = {}) {
  const presets = await apiFetch("/api/v1/keypoints/presets", token, options);
  return Array.isArray(presets) ? presets : [];
}

export async function loadKeypointCollections(token, options = {}) {
  const collections = await apiFetch("/api/v1/keypoints/collections", token, options);
  return Array.isArray(collections) ? collections : [];
}

// Stage an uploaded photo to the TRANSIENT keypoint-source area (NOT a workspace asset). The
// worker reads it by path for kps_extract; saving a preset copies it into the library, and a
// startup sweep reclaims paths that are never saved. Returns { path, displayName }.
export async function stageKeypointSource(token, file) {
  const body = new FormData();
  body.append("file", file);
  const result = await apiFetch("/api/v1/keypoints/sources", token, { method: "POST", body });
  const staged = Array.isArray(result?.sources) ? result.sources : [];
  return staged[0] ?? null;
}

// Persist a captured preset from an extracted kps + a staged source image.
export async function saveKeypointPreset(token, spec) {
  return apiFetch("/api/v1/keypoints", token, { method: "POST", body: JSON.stringify(spec) });
}

export async function deleteKeypointPreset(token, presetId) {
  return apiFetch(`/api/v1/projects/${GLOBAL_KEYPOINTS_PROJECT_ID}/assets/${presetId}`, token, {
    method: "DELETE",
  });
}

// Create or update a user angle-set collection: { id?, name, orderedPresetIds[], isDefault? }.
export async function upsertKeypointCollection(token, spec) {
  return apiFetch("/api/v1/keypoints/collections", token, {
    method: "POST",
    body: JSON.stringify(spec),
  });
}

export async function setDefaultKeypointCollection(token, collectionId) {
  return apiFetch(`/api/v1/keypoints/collections/${encodeURIComponent(collectionId)}/default`, token, {
    method: "PUT",
  });
}

export async function deleteKeypointCollection(token, collectionId) {
  return apiFetch(`/api/v1/keypoints/collections/${encodeURIComponent(collectionId)}`, token, {
    method: "DELETE",
  });
}

// Load the preset list (built-in 11 + user) once per token. Best-effort: a fetch failure
// surfaces via `error`; built-ins always come back from the API so the list is rarely empty.
export function useKeypointPresets() {
  const { token } = useAppContext();
  const [state, setState] = useState({ presets: [], loading: true, error: "" });
  const reload = useCallback(
    async (signal) => {
      try {
        setState((prev) => ({ ...prev, loading: true }));
        const presets = await loadKeypointPresets(token, signal ? { signal } : {});
        setState({ presets, loading: false, error: "" });
      } catch (error) {
        if (signal?.aborted) return;
        setState({ presets: [], loading: false, error: String(error?.message ?? error) });
      }
    },
    [token],
  );
  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal);
    return () => controller.abort();
  }, [reload]);
  return { ...state, reload };
}

// Load the collection list (built-in default + user collections). Used by the Collections tab
// and by the per-generation override picker in Character Studio's angle set.
export function useKeypointCollections() {
  const { token } = useAppContext();
  const [state, setState] = useState({ collections: [], loading: true, error: "" });
  const reload = useCallback(
    async (signal) => {
      try {
        setState((prev) => ({ ...prev, loading: true }));
        const collections = await loadKeypointCollections(token, signal ? { signal } : {});
        setState({ collections, loading: false, error: "" });
      } catch (error) {
        if (signal?.aborted) return;
        setState({ collections: [], loading: false, error: String(error?.message ?? error) });
      }
    },
    [token],
  );
  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal);
    return () => controller.abort();
  }, [reload]);
  return { ...state, reload };
}
