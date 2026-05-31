import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Control the mocked API: GET /assets returns `poseAssets`, mutations resolve and are
// recorded in `apiCalls`.
const apiCalls = [];
let poseAssets = [];

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual, // keep API_BASE_URL / eventUrl / isAbortError for assetMedia etc.
    apiFetch: vi.fn(async (path, _token, options = {}) => {
      const method = options.method ?? "GET";
      apiCalls.push({ path, method, body: options.body });
      if (method === "GET" && path.includes("/assets")) {
        return poseAssets;
      }
      if (method === "POST" && path === "/api/v1/jobs") {
        return { id: "job_pose_1", status: "queued" };
      }
      if (method === "POST" && path === "/api/v1/poses/sources") {
        return { sources: [{ path: "/data/cache/pose-uploads/upload-x.png", displayName: "photo.png" }] };
      }
      return {};
    }),
  };
});

import { AppContext } from "../context/AppContext.js";
import { PoseLibraryScreen } from "./PoseLibraryScreen.jsx";

function poseAsset(overrides = {}) {
  return {
    id: "asset_pose_1",
    projectId: "project_global_poses",
    type: "pose",
    displayName: "Arm Raised",
    tags: ["dynamic"],
    file: { path: "assets/poses/asset_pose_1.png", mimeType: "image/png", width: 768, height: 1280 },
    url: "/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png",
    status: { favorite: false, rating: 0, rejected: false, trashed: false },
    pose: { category: "dance", keypoints: [[0.5, 0.1]] },
    recipe: {},
    lineage: {},
    ...overrides,
  };
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

describe("PoseLibraryScreen", () => {
  let container;
  let root;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiCalls.length = 0;
    poseAssets = [];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "test-token" }}>
          <PoseLibraryScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush the refresh() effect
  }

  it("fetches reserved-project poses and groups them by category", async () => {
    poseAssets = [poseAsset()];
    await render();
    expect(apiCalls[0].path).toContain("/api/v1/projects/project_global_poses/assets");
    expect(container.textContent).toContain("Arm Raised");
    expect(container.textContent).toContain("dance");
  });

  it("shows an empty state when there are no saved poses", async () => {
    poseAssets = [];
    await render();
    expect(container.textContent).toContain("No saved poses yet");
  });

  it("discards a selected pose against the reserved project", async () => {
    poseAssets = [poseAsset()];
    await render();
    const tile = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Arm Raised"));
    await click(tile);
    const discard = [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === "Discard");
    expect(discard).toBeTruthy();
    await click(discard);
    expect(
      apiCalls.some((c) => c.method === "DELETE" && c.path === "/api/v1/projects/project_global_poses/assets/asset_pose_1"),
    ).toBe(true);
  });

  it("opens a pose in the shared fullscreen preview on double-click", async () => {
    poseAssets = [poseAsset()];
    await render();
    const tile = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Arm Raised"));
    await act(async () => {
      tile.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
    });
    expect(container.querySelector(".preview-modal")).toBeTruthy();
  });

  it("switches to the Create tab", async () => {
    poseAssets = [poseAsset()];
    await render();
    const createTab = container.querySelector("#pose-library-tab-create");
    await click(createTab);
    expect(container.querySelector("#pose-library-panel-poses").hidden).toBe(true);
    expect(container.querySelector("#pose-library-panel-create").hidden).toBe(false);
  });
});

const libAsset = {
  id: "asset_lib_1",
  type: "image",
  displayName: "Photo",
  origin: "upload",
  status: { trashed: false },
  file: { path: "assets/images/p.png", mimeType: "image/png" },
  url: "/api/v1/projects/project_1/files/assets/images/p.png",
};

const completedJob = {
  id: "job_pose_1",
  type: "pose_detect",
  status: "completed",
  result: {
    sources: [
      {
        sourceAssetId: "asset_lib_1",
        displayName: "Photo",
        sourcePath: "assets/images/p.png",
        sourceWidth: 768,
        sourceHeight: 1280,
        sourceAspect: 0.6,
        poses: [
          {
            personIndex: 0,
            facing: "front",
            bbox: [0.1, 0.1, 0.9, 0.9],
            meanConf: { body: 5.0, hands: 4.0, face: 4.5 },
            keypoints: [[0.5, 0.1, 5.0]],
            hands: [[], []],
            face: [],
            skeletonPreview: "/data/cache/pose_detect/job_pose_1/p_p0_skel.png",
          },
        ],
      },
    ],
  },
};

function makeContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "Proj" },
    assets: [libAsset],
    characters: [],
    importAsset: vi.fn(),
    requestedGpu: "auto",
    jobs: [completedJob],
    ...overrides,
  };
}

describe("PoseLibraryScreen — Create tab", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiCalls.length = 0;
    poseAssets = [];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context = makeContext()) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <PoseLibraryScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush the refresh() effect
  }

  const byText = (text, selector = "button") =>
    [...container.querySelectorAll(selector)].find((el) => el.textContent.includes(text));
  const exactBtn = (text) =>
    [...container.querySelectorAll("button")].find((el) => el.textContent.trim() === text);

  async function setInputValue(input, value) {
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setter.call(input, value);
      input.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  }

  it("runs photo → detect → categorize → save and posts to /api/v1/poses", async () => {
    await render();
    await click(container.querySelector("#pose-library-tab-create"));

    // Pick a Library image via the shared DatasetAddDialog.
    await click(byText("Add images"));
    await click(byText("Asset Library"));
    await click(byText("Photo")); // candidate card
    await click(exactBtn("Add 1")); // commit selection
    await click(exactBtn("Done")); // close dialog

    // A source chip + Generate button appear; fire the detector job.
    expect(container.textContent).toContain("Generate poses");
    await click(byText("Generate poses"));

    // The completed job (from context.jobs) yields one candidate to categorize.
    const categoryInput = container.querySelector('input[list="pose-category-suggestions"]');
    expect(categoryInput).toBeTruthy();
    await setInputValue(categoryInput, "standing");

    await click(byText("Save 1 pose"));

    const post = apiCalls.find((c) => c.method === "POST" && c.path === "/api/v1/poses");
    expect(post).toBeTruthy();
    const body = JSON.parse(post.body);
    expect(body.poses).toHaveLength(1);
    expect(body.poses[0]).toMatchObject({
      jobId: "job_pose_1",
      skeletonFile: "p_p0_skel.png",
      category: "standing",
      width: 768,
      height: 1280,
    });
    expect(body.poses[0].pose.keypoints).toEqual([[0.5, 0.1, 5.0]]);
    // After save, control returns to the Poses tab.
    expect(container.querySelector("#pose-library-panel-poses").hidden).toBe(false);
  });

  it("fires the detector with assetId sources, not browser paths", async () => {
    await render();
    await click(container.querySelector("#pose-library-tab-create"));
    await click(byText("Add images"));
    await click(byText("Asset Library"));
    await click(byText("Photo"));
    await click(exactBtn("Add 1"));
    await click(exactBtn("Done"));
    await click(byText("Generate poses"));

    const job = apiCalls.find((c) => c.method === "POST" && c.path === "/api/v1/jobs");
    expect(job).toBeTruthy();
    const body = JSON.parse(job.body);
    expect(body.type).toBe("pose_detect");
    expect(body.projectId).toBe("project_1");
    expect(body.payload.sources).toEqual([{ assetId: "asset_lib_1", displayName: "Photo" }]);
  });

  it("stages File-Upload sources as temporary uploads, not workspace assets", async () => {
    window.URL.createObjectURL = () => "blob:test";
    window.URL.revokeObjectURL = () => {};
    await render();
    await click(container.querySelector("#pose-library-tab-create"));
    await click(byText("Add images"));
    // File tab is the default; fire a change on its <input type=file> with an image.
    const fileInput = container.querySelector('input[type="file"]');
    const file = new File(["x"], "photo.png", { type: "image/png" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    await act(async () => {
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });

    // Staged via the transient endpoint — NEVER imported as a workspace asset.
    expect(apiCalls.some((c) => c.method === "POST" && c.path === "/api/v1/poses/sources")).toBe(true);
    expect(apiCalls.some((c) => c.method === "POST" && c.path.includes("/assets"))).toBe(false);

    // Generate forwards it as a temp path source (not an assetId).
    await click(byText("Generate poses"));
    const job = apiCalls.find((c) => c.method === "POST" && c.path === "/api/v1/jobs");
    expect(JSON.parse(job.body).payload.sources[0]).toMatchObject({
      path: "/data/cache/pose-uploads/upload-x.png",
      temp: true,
    });
  });

  it("requires a workspace before creating poses", async () => {
    await render(makeContext({ activeProject: null }));
    await click(container.querySelector("#pose-library-tab-create"));
    expect(container.querySelector("#pose-library-panel-create").textContent).toContain("Open a workspace");
  });
});
