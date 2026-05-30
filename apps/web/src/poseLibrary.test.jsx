import { afterEach, describe, expect, it, vi } from "vitest";

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
      url: "/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png",
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
