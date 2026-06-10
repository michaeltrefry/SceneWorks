import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Control the mocked API: GETs return the fixture lists; mutations resolve and are recorded.
const apiCalls = [];
let presets = [];
let collections = [];

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual, // keep API_BASE_URL etc. for keypointSourceImageUrl
    apiFetch: vi.fn(async (path, _token, options = {}) => {
      const method = options.method ?? "GET";
      apiCalls.push({ path, method, body: options.body });
      if (method === "GET" && path === "/api/v1/keypoints/presets") return presets;
      if (method === "GET" && path === "/api/v1/keypoints/collections") return collections;
      if (method === "POST" && path === "/api/v1/keypoints/sources") {
        return { sources: [{ path: "/data/cache/keypoint-uploads/upload-x.png", displayName: "face.png" }] };
      }
      if (method === "POST" && path === "/api/v1/jobs") return { id: "job_kps_1", status: "queued" };
      return {};
    }),
  };
});

import { AppContext } from "../context/AppContext.js";
import { KeyPointLibraryScreen } from "./KeyPointLibraryScreen.jsx";

const FRONT_KPS = [
  [0.4, 0.34],
  [0.6, 0.34],
  [0.5, 0.43],
  [0.43, 0.53],
  [0.57, 0.53],
];

function builtinPreset(overrides = {}) {
  return { id: "builtin_front", name: "Front", angle: "front", kps: FRONT_KPS, builtin: true, sourceImageRef: null, ...overrides };
}
function customPreset(overrides = {}) {
  return {
    id: "asset_k1",
    name: "My Side",
    kps: FRONT_KPS,
    builtin: false,
    sourceImageRef: "assets/keypoints/asset_k1.png",
    sourceAssetId: null,
    ...overrides,
  };
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}
async function setInputValue(input, value) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(input, value);
    input.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}
const byText = (text, selector = "button") =>
  [...document.querySelectorAll(selector)].find((el) => el.textContent.includes(text));

function makeContext(overrides = {}) {
  return { token: "test-token", requestedGpu: "auto", jobs: [], ...overrides };
}

describe("KeyPointLibraryScreen", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiCalls.length = 0;
    presets = [];
    collections = [];
    window.URL.createObjectURL = () => "blob:test";
    window.URL.revokeObjectURL = () => {};
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
          <KeyPointLibraryScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush the load effects
  }

  it("renders each preset's landmarks as a 5-point overlay, with a source image for captures", async () => {
    presets = [builtinPreset(), customPreset()];
    await render();
    // Each preset card carries an SVG overlay with exactly 5 landmark dots.
    const overlays = container.querySelectorAll(".keypoint-card .kps-overlay");
    expect(overlays.length).toBeGreaterThanOrEqual(2);
    const dots = overlays[0].querySelectorAll(".kps-overlay-dot");
    expect(dots.length).toBe(5);
    // Built-ins render on a neutral canvas (no source image); captures show their photo.
    expect(container.textContent).toContain("Front");
    expect(container.textContent).toContain("My Side");
    const image = [...container.querySelectorAll("image")].find((el) =>
      (el.getAttribute("href") ?? "").includes("assets/keypoints/asset_k1.png"),
    );
    expect(image).toBeTruthy();
  });

  it("protects built-ins (no delete) and deletes a user preset against the reserved project", async () => {
    presets = [builtinPreset(), customPreset()];
    await render();
    const deletes = [...container.querySelectorAll("button")].filter((b) => b.textContent.trim() === "Delete");
    // Only the one custom preset is deletable; the built-in has no delete control.
    expect(deletes).toHaveLength(1);
    await click(deletes[0]);
    expect(
      apiCalls.some(
        (c) => c.method === "DELETE" && c.path === "/api/v1/projects/project_global_keypoints/assets/asset_k1",
      ),
    ).toBe(true);
  });

  it("captures a preset: upload → extract → preview → save", async () => {
    presets = [builtinPreset()];
    const job = {
      id: "job_kps_1",
      type: "kps_extract",
      status: "completed",
      result: { detected: true, kps: FRONT_KPS, lowConfidence: false, sourceWidth: 800, sourceHeight: 1000 },
    };
    await render(makeContext({ jobs: [job] }));
    await click(container.querySelector("#keypoint-tab-capture"));

    const fileInput = container.querySelector('#keypoint-panel-capture input[type="file"]');
    const file = new File(["x"], "face.png", { type: "image/png" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    await act(async () => fileInput.dispatchEvent(new window.Event("change", { bubbles: true })));
    await act(async () => {}); // flush staging + job post + the job-watch effect

    // Staged to the transient endpoint and a kps_extract job fired.
    expect(apiCalls.some((c) => c.method === "POST" && c.path === "/api/v1/keypoints/sources")).toBe(true);
    const jobPost = apiCalls.find((c) => c.method === "POST" && c.path === "/api/v1/jobs");
    expect(JSON.parse(jobPost.body).type).toBe("kps_extract");
    expect(JSON.parse(jobPost.body).payload.sourcePath).toBe("/data/cache/keypoint-uploads/upload-x.png");

    // The completed job yields a preview + a name seeded from the filename.
    const nameInput = container.querySelector('#keypoint-panel-capture input[type="text"], #keypoint-panel-capture input:not([type])');
    expect(nameInput.value).toBe("face");
    await setInputValue(nameInput, "Captured front");
    await click(byText("Save preset"));

    const save = apiCalls.find((c) => c.method === "POST" && c.path === "/api/v1/keypoints");
    expect(save).toBeTruthy();
    const body = JSON.parse(save.body);
    expect(body).toMatchObject({
      name: "Captured front",
      sourceUploadPath: "/data/cache/keypoint-uploads/upload-x.png",
      sourceWidth: 800,
      sourceHeight: 1000,
    });
    expect(body.kps).toEqual(FRONT_KPS);
  });

  it("explains an extraction failure instead of saving silently", async () => {
    presets = [builtinPreset()];
    const job = { id: "job_kps_1", type: "kps_extract", status: "completed", result: { detected: false, reason: "no_face" } };
    await render(makeContext({ jobs: [job] }));
    await click(container.querySelector("#keypoint-tab-capture"));

    const fileInput = container.querySelector('#keypoint-panel-capture input[type="file"]');
    const file = new File(["x"], "face.png", { type: "image/png" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    await act(async () => fileInput.dispatchEvent(new window.Event("change", { bubbles: true })));
    await act(async () => {});

    expect(container.querySelector("#keypoint-panel-capture").textContent).toContain("No usable face");
    // No Save control offered and nothing posted to the preset endpoint.
    expect(byText("Save preset")).toBeFalsy();
    expect(apiCalls.some((c) => c.method === "POST" && c.path === "/api/v1/keypoints")).toBe(false);
  });

  it("builds an ordered collection from selected presets", async () => {
    presets = [builtinPreset(), customPreset()];
    collections = [
      { id: "builtin_default", name: "Default angles", orderedPresetIds: ["builtin_front"], isDefault: true, builtin: true },
    ];
    await render();
    await click(container.querySelector("#keypoint-tab-collections"));

    const nameInput = container.querySelector('#keypoint-panel-collections input');
    await setInputValue(nameInput, "LoRA coverage");
    // Add both presets from the picker grid (in order).
    await click(byText("Front", ".keypoint-pick"));
    await click(byText("My Side", ".keypoint-pick"));
    await click(byText("Create collection"));

    const post = apiCalls.find((c) => c.method === "POST" && c.path === "/api/v1/keypoints/collections");
    expect(post).toBeTruthy();
    expect(JSON.parse(post.body)).toMatchObject({
      name: "LoRA coverage",
      orderedPresetIds: ["builtin_front", "asset_k1"],
    });
  });

  it("sets a non-default collection as the default", async () => {
    presets = [builtinPreset()];
    collections = [
      { id: "builtin_default", name: "Default angles", orderedPresetIds: ["builtin_front"], isDefault: true, builtin: true },
      { id: "col_user", name: "My Set", orderedPresetIds: ["builtin_front"], isDefault: false },
    ];
    await render();
    await click(container.querySelector("#keypoint-tab-collections"));

    await click(byText("Set default"));
    expect(
      apiCalls.some((c) => c.method === "PUT" && c.path === "/api/v1/keypoints/collections/col_user/default"),
    ).toBe(true);
  });
});
