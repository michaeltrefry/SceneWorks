import { afterEach, describe, expect, it, vi } from "vitest";

import { API_BASE_URL, setMediaTicket } from "./api.js";
import { loadBuiltinPoses, loadPoseLibrary, poseAssetToRecord } from "./poseLibrary.js";

const BUILTIN = {
  version: 1,
  categories: ["standing"],
  poses: [{ id: "standing_01", category: "standing", label: "Standing 01", preview: "poses/standing_01.png", keypoints: [[0.5, 0.1]] }],
};

function mockBuiltinFetch() {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => BUILTIN })));
}

afterEach(() => {
  vi.unstubAllGlobals();
  setMediaTicket("");
});

describe("poseLibrary", () => {
  it("tags built-in poses with source + a usable previewUrl", async () => {
    mockBuiltinFetch();
    const builtin = await loadBuiltinPoses();
    expect(builtin[0].source).toBe("builtin");
    expect(builtin[0].previewUrl).toBe("/poses/standing_01.png");
  });

  it("maps a reserved-project pose asset into a pose record", () => {
    const record = poseAssetToRecord({
      id: "asset_pose_1",
      displayName: "Arm Raised",
      projectId: "project_global_poses",
      url: "/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png",
      file: { path: "assets/poses/asset_pose_1.png", mimeType: "image/png" },
      tags: ["dynamic"],
      pose: { category: "dance", keypoints: [[0.4, 0.2]], hands: [[], []], face: [] },
    });
    expect(record).toMatchObject({
      id: "asset_pose_1",
      label: "Arm Raised",
      category: "dance",
      source: "user",
      assetId: "asset_pose_1",
    });
    expect(record.keypoints).toEqual([[0.4, 0.2]]);
    expect(record.previewUrl).toContain("/files/assets/poses/");
  });

  // sc-8859 (F-057): the preview must be built through the shared assetUrl helper so
  // it carries the API_BASE_URL prefix — bare `asset.url` 404s under Vite dev / any
  // split-origin (VITE_API_BASE_URL) deployment, and the old raw-path fallback omitted
  // the /api/v1/projects/:id/files/ route entirely.
  it("builds previewUrl through assetUrl (API_BASE_URL-prefixed, not a bare relative path)", () => {
    const record = poseAssetToRecord({
      id: "asset_pose_1",
      displayName: "Arm Raised",
      projectId: "project_global_poses",
      url: "/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png",
      file: { path: "assets/poses/asset_pose_1.png", mimeType: "image/png" },
      pose: { category: "dance", keypoints: [[0.4, 0.2]] },
    });
    expect(record.previewUrl).toBe(
      `${API_BASE_URL}/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png`,
    );
    // API_BASE_URL is non-empty under the test/dev config, so the URL is absolute
    // rather than a bare relative path that would resolve against the dev origin.
    expect(API_BASE_URL).not.toBe("");
    expect(record.previewUrl.startsWith(API_BASE_URL)).toBe(true);
  });

  // sc-8859: in remote-auth mode assetUrl (sc-8810) appends a short-lived media
  // ticket so the element-driven <img> preview authenticates. The pose preview must
  // inherit that ticket by riding the shared helper.
  it("carries the media ticket in remote-auth mode (sc-8810)", () => {
    setMediaTicket("t0ken");
    const record = poseAssetToRecord({
      id: "asset_pose_1",
      displayName: "Arm Raised",
      projectId: "project_global_poses",
      url: "/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png",
      file: { path: "assets/poses/asset_pose_1.png", mimeType: "image/png" },
      pose: { category: "dance", keypoints: [[0.4, 0.2]] },
    });
    expect(record.previewUrl).toBe(
      `${API_BASE_URL}/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png?ticket=t0ken`,
    );
  });

  it("merges built-in + injected user poses (categories + byId)", async () => {
    mockBuiltinFetch();
    const userPose = poseAssetToRecord({ id: "asset_u1", displayName: "User Pose", pose: { category: "my poses", keypoints: [[0.1, 0.2]] } });
    const lib = await loadPoseLibrary({ loadUserPoses: async () => [userPose] });
    expect(lib.poses).toHaveLength(2);
    expect(lib.categories).toEqual(expect.arrayContaining(["standing", "my poses"]));
    expect(lib.byId.asset_u1.source).toBe("user");
    expect(lib.byId.standing_01.source).toBe("builtin");
  });

  it("degrades to built-ins when the user-pose fetch fails", async () => {
    mockBuiltinFetch();
    const lib = await loadPoseLibrary({
      loadUserPoses: async () => {
        throw new Error("network");
      },
    });
    expect(lib.poses).toHaveLength(1);
    expect(lib.poses[0].id).toBe("standing_01");
  });
});
