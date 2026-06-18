import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MakeVideo } from "./MakeVideo.jsx";
import { AppContext } from "../../context/AppContext.js";

const TEXT_VIDEO_MODEL = {
  id: "wan_t2v",
  name: "Wan T2V",
  type: "video",
  capabilities: ["text_to_video"],
  defaults: { duration: 5, fps: 16, resolution: "1280x720", quality: "balanced" },
  limits: { durations: [3, 4, 5], recommendedMaxDuration: 5, fps: [16], resolutions: ["1280x720"] },
};

const IMAGE_VIDEO_MODEL = {
  id: "svd",
  name: "SVD",
  type: "video",
  capabilities: ["image_to_video"],
  defaults: { duration: 4, fps: 7, resolution: "1024x576", quality: "balanced" },
  limits: { durations: [4], recommendedMaxDuration: 4, fps: [7], resolutions: ["1024x576"] },
};

let container;
let root;

beforeEach(() => {
  localStorage.clear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  localStorage.clear();
});

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-a", name: "Project A" },
    videoModels: [TEXT_VIDEO_MODEL],
    imageModels: [],
    createVideoJob: vi.fn(async () => ({ id: "video_job_1" })),
    createImageJob: vi.fn(),
    refinePrompt: vi.fn(),
    mediaAssets: [],
    recentVideoAssets: [],
    videoLocalJobs: [],
    jobs: [],
    recentImageAssets: [],
    setPreviewAsset: vi.fn(),
    ...overrides,
  };
}

async function render(value) {
  await act(async () => {
    root.render(
      <AppContext.Provider value={value}>
        <MakeVideo />
      </AppContext.Provider>,
    );
  });
}

async function setPrompt(value) {
  const textarea = container.querySelector("textarea");
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  await act(async () => {
    setter.call(textarea, value);
    textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

describe("MakeVideo", () => {
  it("does not submit when no video model is available", async () => {
    const ctx = baseContext({ videoModels: [] });
    await render(ctx);
    await setPrompt("snow drifting past the windows");

    const create = container.querySelector(".sw-cta");
    expect(create.disabled).toBe(true);

    await act(async () => {
      create.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(ctx.createVideoJob).not.toHaveBeenCalled();
  });

  it("does not submit text-to-video with an image-only video model", async () => {
    const ctx = baseContext({ videoModels: [IMAGE_VIDEO_MODEL] });
    await render(ctx);
    await setPrompt("snow drifting past the windows");

    const create = container.querySelector(".sw-cta");
    expect(create.disabled).toBe(true);

    await act(async () => {
      create.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(ctx.createVideoJob).not.toHaveBeenCalled();
  });
});
