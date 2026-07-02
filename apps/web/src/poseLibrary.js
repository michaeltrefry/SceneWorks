import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api.js";
import { assetUrl } from "./components/assetMedia.jsx";
import { useAppContext } from "./context/AppContext.js";

// The reserved global project that holds user-created pose assets (epic 2282). Mirrors
// crates/sceneworks-core::GLOBAL_POSES_PROJECT_ID. Hidden from the project switcher;
// addressed directly by the Pose Library screen + the user-pose picker fetcher.
export const GLOBAL_POSES_PROJECT_ID = "project_global_poses";

// The pose library is the union of two sources:
//  - BUILT-IN: the bundled OpenPose library (apps/web/public/poses/index.json) —
//    normalized COCO-18 skeletons + preview thumbnails, shipped read-only, cached.
//  - USER: type:"pose" assets in the reserved global poses project (epic 2282),
//    fetched fresh via an injected `loadUserPoses` fetcher (they change as users
//    create/trash poses). Optional + best-effort: any failure degrades to built-ins.
// Selected poses' keypoints (+ optional hands/face) ride advanced.poses on a job.
let builtinCache = null;

export function loadBuiltinPoses() {
  if (!builtinCache) {
    // Promise.resolve().then(...) so a missing/throwing fetch becomes a rejection the
    // caller's .catch handles (rather than a synchronous throw at the call site).
    builtinCache = Promise.resolve()
      .then(() => {
        if (typeof fetch !== "function") {
          throw new Error("fetch unavailable");
        }
        return fetch("/poses/index.json");
      })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`pose library unavailable (${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        const poses = Array.isArray(data?.poses) ? data.poses : [];
        return poses.map((pose) => ({
          ...pose,
          source: "builtin",
          previewUrl: `/${pose.preview}`,
        }));
      })
      .catch((error) => {
        builtinCache = null; // allow a retry on next mount
        throw error;
      });
  }
  return builtinCache;
}

// Map a reserved-project type:"pose" asset into a pose record the picker understands.
// The asset's `pose` field carries keypoints/hands/face/category; the rendered skeleton
// preview is resolved through the shared `assetUrl` helper. Built by the DWPose detector
// + Create tab (sc-2285/sc-2287).
export function poseAssetToRecord(asset) {
  const pose = asset?.pose ?? {};
  // Route the preview through the shared asset-URL helper so it gets the API_BASE_URL
  // prefix (split-origin / Vite dev), the correct /api/v1/projects/:id/files/ route,
  // and the short-lived media ticket in remote-auth mode (sc-8810/sc-8859). The raw
  // asset already carries `url` + `projectId` + `file.path` (asset_index injects `url`),
  // which is exactly the shape assetUrl consumes; `""` when unresolvable.
  const previewUrl = assetUrl(asset) || undefined;
  return {
    id: asset.id,
    label: asset.displayName || asset.id,
    category: pose.category || "my poses",
    keypoints: pose.keypoints ?? [],
    hands: pose.hands,
    face: pose.face,
    tags: asset.tags ?? [],
    source: "user",
    assetId: asset.id,
    previewUrl,
  };
}

export async function loadPoseLibrary({ loadUserPoses } = {}) {
  const builtin = await loadBuiltinPoses();
  let user = [];
  if (typeof loadUserPoses === "function") {
    try {
      user = (await loadUserPoses()) || [];
    } catch {
      user = []; // best-effort: never let user-pose fetch failures hide the built-ins
    }
  }
  const poses = [...builtin, ...user];
  const categories = [...new Set(poses.map((pose) => pose.category).filter(Boolean))];
  const byId = Object.fromEntries(poses.map((pose) => [pose.id, pose]));
  return { poses, categories, byId };
}

// A memoized fetcher for the user's saved poses (the reserved global project),
// mapped into picker records. Pass its result to BOTH `usePoseLibrary` (so a
// selected user pose resolves to keypoints when building the job) and
// `PoseLibraryPicker` (so it appears in the grid). Best-effort: any fetch failure
// is swallowed by `loadPoseLibrary`, degrading to the built-in library (sc-2287).
export function useUserPoseLoader() {
  const { token } = useAppContext();
  return useCallback(async () => {
    const items = await apiFetch(`/api/v1/projects/${GLOBAL_POSES_PROJECT_ID}/assets`, token);
    return (Array.isArray(items) ? items : [])
      .filter((asset) => asset?.type === "pose")
      .map(poseAssetToRecord);
  }, [token]);
}

// `loadUserPoses` should be a memoized (useCallback) async fetcher or undefined.
export function usePoseLibrary({ loadUserPoses } = {}) {
  const [state, setState] = useState({ poses: [], categories: [], byId: {}, loading: true, error: "" });
  useEffect(() => {
    let active = true;
    loadPoseLibrary({ loadUserPoses })
      .then((library) => active && setState({ ...library, loading: false, error: "" }))
      .catch((error) => active && setState({ poses: [], categories: [], byId: {}, loading: false, error: String(error.message ?? error) }));
    return () => {
      active = false;
    };
  }, [loadUserPoses]);
  return state;
}
