import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MyCreations } from "./MyCreations.jsx";
import { AppContext } from "../../context/AppContext.js";

const IMAGE_TO_VIDEO_MODEL = {
  id: "ltx_2_3",
  name: "LTX",
  type: "video",
  capabilities: ["image_to_video", "text_to_video"],
  defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
  limits: { resolutions: ["768x512"] },
};

function imageAsset(overrides = {}) {
  return {
    id: "img_1",
    projectId: "project-a",
    type: "image",
    createdAt: "2026-01-01T00:00:00Z",
    displayName: "Uploaded image",
    file: { path: "assets/img_1.png", mimeType: "image/png", width: 1024, height: 1024 },
    status: {},
    ...overrides,
  };
}

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function baseContext(overrides = {}) {
  return {
    recentImageAssets: [imageAsset()],
    recentVideoAssets: [],
    imageModels: [],
    videoModels: [IMAGE_TO_VIDEO_MODEL],
    createImageJob: vi.fn(),
    createVideoJob: vi.fn(async () => ({ id: "video_job_1" })),
    updateAssetStatus: vi.fn(),
    setPreviewAsset: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    setUiMode: vi.fn(),
    ...overrides,
  };
}

async function render(value) {
  await act(async () => {
    root.render(
      <AppContext.Provider value={value}>
        <MyCreations />
      </AppContext.Provider>,
    );
  });
}

function buttonText(text) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.includes(text));
}

describe("MyCreations", () => {
  it("does not offer image-to-video for an image without a prompt recipe", async () => {
    await render(baseContext());
    expect(buttonText("Make a video from this")).toBeUndefined();
  });

  it("offers image-to-video when the selected image has a prompt recipe", async () => {
    const ctx = baseContext({
      recentImageAssets: [imageAsset({ recipe: { prompt: "a cabin at dusk" } })],
    });
    await render(ctx);

    const action = buttonText("Make a video from this");
    expect(action).toBeTruthy();

    await act(async () => {
      action.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(ctx.createVideoJob).toHaveBeenCalledWith(expect.objectContaining({
      mode: "image_to_video",
      model: "ltx_2_3",
      prompt: "a cabin at dusk",
    }));
  });
});
