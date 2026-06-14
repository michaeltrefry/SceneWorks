import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import { VideoStudio } from "./VideoStudio.jsx";

const LTX = {
  id: "ltx_2_3",
  name: "LTX 2.3",
  type: "video",
  family: "ltx-video",
  capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
  defaults: { duration: 6, resolution: "768x512", fps: 25 },
  limits: {},
  quantization: {},
  loraCompatibility: {},
  ui: {},
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createPersonDetectionJob: vi.fn(),
    createPersonTrackJob: vi.fn(),
    createVideoJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    latestVideoAssets: [],
    recentVideoAssets: [],
    studioLaunch: null,
    loras: [],
    jobs: [],
    videoLocalJobs: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setPreviewAsset: vi.fn(),
    personTracks: [],
    personReadiness: {},
    presets: [],
    requestedGpu: "",
    saveTrackCorrections: vi.fn(),
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    videoModels: [LTX],
    ...overrides,
  };
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

async function doubleClick(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true }));
  });
}

const buttonWithText = (root, text) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

function setInput(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

const saveButton = (container) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Save as Preset"));
const nameInput = (container) => container.querySelector('input[aria-label="Preset name"]');

describe("VideoStudio Save as Preset", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("snapshots the video config into an image_to_video preset without the seed", async () => {
    const context = baseContext();
    await render(context);

    const input = nameInput(container);
    expect(input).toBeTruthy();
    await act(async () => setInput(input, "Push In"));
    await click(saveButton(container));

    expect(context.createPreset).toHaveBeenCalledTimes(1);
    const payload = context.createPreset.mock.calls[0][0];
    expect(payload).toMatchObject({
      id: "push_in",
      name: "Push In",
      scope: "project",
      workflow: "image_to_video",
      model: "ltx_2_3",
    });
    expect(payload.defaults.prompt).toBe("Camera slowly pushes in while the scene comes alive");
    expect(payload.defaults).not.toHaveProperty("seed");
    expect(container.textContent).toContain('Saved "Push In" to this project.');
  });

  it("blocks a duplicate name client-side before calling the API", async () => {
    const context = baseContext({
      presets: [
        {
          id: "push_in",
          name: "Push In",
          scope: "project",
          workflow: "image_to_video",
          model: "ltx_2_3",
          modes: ["image_to_video"],
        },
      ],
    });
    await render(context);

    await act(async () => setInput(nameInput(container), "Push In"));
    await click(saveButton(container));

    expect(context.createPreset).not.toHaveBeenCalled();
    expect(container.textContent).toContain("already exists");
  });
});

describe("VideoStudio video_bridge", () => {
  let container;
  let root;

  // A bridge-capable video model with a non-LTX id so the IC-LoRA preset gate
  // (requiresLtxIcLora, which keys on ltx_2_3) doesn't block submission here —
  // this test exercises the new input wiring, not the LTX IC-LoRA requirement.
  const BRIDGE_MODEL = {
    id: "bridge_model",
    name: "Bridge Model",
    type: "video",
    family: "ltx-video",
    capabilities: ["image_to_video", "text_to_video", "extend_clip", "video_bridge"],
    defaults: { duration: 6, resolution: "768x512", fps: 25 },
    limits: {},
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  const leftClip = { id: "vid_left", type: "video", projectId: "project_1", displayName: "Left Clip" };
  const rightClip = { id: "vid_right", type: "video", projectId: "project_1", displayName: "Right Clip" };

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("submits both clip ids when bridging two clips", async () => {
    const context = baseContext({
      videoModels: [BRIDGE_MODEL],
      assets: [leftClip, rightClip],
      selectedAsset: leftClip,
    });
    await render(context);

    // Switch to Bridge mode; the left clip is pre-filled from the selected asset.
    const modeControl = container.querySelector(".mode-control");
    await click(buttonWithText(modeControl, "Bridge"));

    // Drive the right-clip picker (the only "Select clip" button; the left
    // picker shows "Change" because it already has a value).
    await click(buttonWithText(container, "Select clip"));
    const modal = document.querySelector(".asset-picker-modal");
    expect(modal).toBeTruthy();
    const rightOption = [...modal.querySelectorAll('[role="option"]')].find((el) =>
      el.textContent.includes("Right Clip"),
    );
    await doubleClick(rightOption);

    // Render.
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "video_bridge",
      sourceClipAssetId: "vid_left",
      bridgeRightClipAssetId: "vid_right",
    });
  });
});

describe("VideoStudio Bernini task modes", () => {
  let container;
  let root;

  // Bernini exposes the full planner video surface (sc-4703). No `macSupport` here so
  // the (gating-off) test env leaves the mode buttons enabled — `capabilities` gates submit.
  const BERNINI = {
    id: "bernini",
    name: "Bernini",
    type: "video",
    family: "bernini",
    capabilities: [
      "text_to_video",
      "video_to_video",
      "reference_to_video",
      "reference_video_to_video",
      "multi_video_to_video",
      "ads2v",
    ],
    defaults: { duration: 5, resolution: "848x480", fps: 16 },
    limits: { durations: [3, 4, 5], fps: [16], resolutions: ["848x480", "480x848"] },
    quantization: {},
    loraCompatibility: {},
    ui: {},
  };

  const clip = { id: "vid_src", type: "video", projectId: "project_1", displayName: "Source Clip" };
  const clip2 = { id: "vid_src_2", type: "video", projectId: "project_1", displayName: "Source Clip Two" };
  const refClip = { id: "vid_ref", type: "video", projectId: "project_1", displayName: "Reference Clip" };
  const refA = { id: "img_ref_a", type: "image", projectId: "project_1", displayName: "Reference A" };
  const refB = { id: "img_ref_b", type: "image", projectId: "project_1", displayName: "Reference B" };

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <VideoStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // Scope to the mode's media-input band so the unrelated upscaler card (which also
  // has a "Source clip" picker in the results rail) doesn't leak into the assertions.
  const pickerLabels = () =>
    [...(container.querySelector(".studio-source-band")?.querySelectorAll(".asset-picker-label") ?? [])].map(
      (el) => el.textContent,
    );
  const modeButton = (label) => buttonWithText(container.querySelector(".mode-control"), label);

  it("shows the right media slots for each mode", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA, refB] });
    await render(context);

    await click(modeButton("Video → Video"));
    expect(pickerLabels()).toContain("Source clip");
    expect(pickerLabels()).not.toContain("Reference images");

    await click(modeButton("Reference → Video"));
    expect(pickerLabels()).toContain("Reference images");
    expect(pickerLabels()).not.toContain("Source clip");

    await click(modeButton("Reference + Video"));
    expect(pickerLabels()).toEqual(expect.arrayContaining(["Source clip", "Reference images"]));

    // mv2v: a single multi-clip picker, no single "Source clip" or reference images.
    await click(modeButton("Multi-Clip → Video"));
    expect(pickerLabels()).toContain("Source clips");
    expect(pickerLabels()).not.toContain("Source clip");
    expect(pickerLabels()).not.toContain("Reference images");

    // ads2v: source clip + reference video + reference images.
    await click(modeButton("Clip + Ref Video"));
    expect(pickerLabels()).toEqual(
      expect.arrayContaining(["Source clip", "Reference video", "Reference images"]),
    );
  });

  it("keeps Render disabled until the required reference image is selected", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA, refB] });
    await render(context);
    await click(modeButton("Reference → Video"));

    // No reference selected yet.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    await click(buttonWithText(container, "Select images"));
    const modal = document.querySelector(".asset-picker-modal");
    const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Reference A"));
    await click(option);
    await click(buttonWithText(modal, "Use Selection"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
  });

  it("submits the source clip for video_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip], selectedAsset: clip });
    await render(context);
    await click(modeButton("Video → Video"));
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({ mode: "video_to_video", sourceClipAssetId: "vid_src" });
    expect(payload.referenceAssetIds).toEqual([]);
  });

  it("submits all chosen reference images for reference_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [refA, refB] });
    await render(context);
    await click(modeButton("Reference → Video"));

    await click(buttonWithText(container, "Select images"));
    const modal = document.querySelector(".asset-picker-modal");
    for (const name of ["Reference A", "Reference B"]) {
      const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes(name));
      await click(option);
    }
    await click(buttonWithText(modal, "Use Selection"));
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("reference_to_video");
    expect(payload.referenceAssetIds).toEqual(["img_ref_a", "img_ref_b"]);
    expect(payload.sourceClipAssetId).toBeNull();
  });

  it("submits both clip and references for reference_video_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, refA], selectedAsset: clip });
    await render(context);
    await click(modeButton("Reference + Video"));

    await click(buttonWithText(container, "Select images"));
    const modal = document.querySelector(".asset-picker-modal");
    const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes("Reference A"));
    await click(option);
    await click(buttonWithText(modal, "Use Selection"));
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "reference_video_to_video",
      sourceClipAssetId: "vid_src",
    });
    expect(payload.referenceAssetIds).toEqual(["img_ref_a"]);
  });

  it("requires at least two clips before submitting multi_video_to_video", async () => {
    const context = baseContext({ videoModels: [BERNINI], assets: [clip, clip2], selectedAsset: clip });
    await render(context);
    await click(modeButton("Multi-Clip → Video"));

    // One clip auto-selected from selectedAsset isn't enough — mv2v needs >=2.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    await click(buttonWithText(container, "Select clips"));
    const modal = document.querySelector(".asset-picker-modal");
    for (const name of ["Source Clip", "Source Clip Two"]) {
      const option = [...modal.querySelectorAll('[role="option"]')].find((el) => el.textContent.includes(name));
      await click(option);
    }
    await click(buttonWithText(modal, "Use Selection"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));

    expect(context.createVideoJob).toHaveBeenCalledTimes(1);
    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload.mode).toBe("multi_video_to_video");
    expect(payload.sourceClipAssetIds).toEqual(["vid_src", "vid_src_2"]);
    expect(payload.sourceClipAssetId).toBeNull();
    expect(payload.referenceAssetIds).toEqual([]);
  });

  it("submits source clip, reference video, and references for ads2v", async () => {
    const context = baseContext({
      videoModels: [BERNINI],
      assets: [clip, refClip, refA],
      selectedAsset: clip,
    });
    await render(context);
    await click(modeButton("Clip + Ref Video"));

    // Source clip auto-selected; reference video + a reference image still required.
    expect(buttonWithText(container, "Render clip").disabled).toBe(true);

    // The source clip picker shows "Change" (auto-selected), so the first "Select clip"
    // button in document order is the empty reference-video picker.
    await click(buttonWithText(container, "Select clip"));
    let modal = document.querySelector(".asset-picker-modal");
    const refClipOption = [...modal.querySelectorAll('[role="option"]')].find((el) =>
      el.textContent.includes("Reference Clip"),
    );
    await click(refClipOption);
    await click(buttonWithText(modal, "Use Selection"));

    // Pick a reference image.
    await click(buttonWithText(container, "Select images"));
    modal = document.querySelector(".asset-picker-modal");
    const refImageOption = [...modal.querySelectorAll('[role="option"]')].find((el) =>
      el.textContent.includes("Reference A"),
    );
    await click(refImageOption);
    await click(buttonWithText(modal, "Use Selection"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(false);
    await click(buttonWithText(container, "Render clip"));

    const payload = context.createVideoJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "ads2v",
      sourceClipAssetId: "vid_src",
      referenceClipAssetId: "vid_ref",
    });
    expect(payload.referenceAssetIds).toEqual(["img_ref_a"]);
    expect(payload.sourceClipAssetIds).toEqual([]);
  });

  it("disables an editing mode on a model that does not support it", async () => {
    const context = baseContext({ videoModels: [LTX], assets: [clip] });
    await render(context);
    await click(modeButton("Reference → Video"));

    expect(buttonWithText(container, "Render clip").disabled).toBe(true);
    expect(container.textContent).toContain("does not support this mode");
  });
});
