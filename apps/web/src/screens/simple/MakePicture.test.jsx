import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MakePicture } from "./MakePicture.jsx";
import { AppContext } from "../../context/AppContext.js";

const IMAGE_MODEL = {
  id: "sdxl",
  name: "SDXL",
  type: "image",
  capabilities: ["text_to_image"],
  defaults: { guidanceScale: 5 },
  limits: { resolutions: ["1024x1024"] },
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
    imageModels: [IMAGE_MODEL],
    createImageJob: vi.fn(async () => ({ id: "image_job_1" })),
    refinePrompt: vi.fn(),
    recentImageAssets: [],
    imageLocalJobs: [],
    jobs: [],
    mediaAssets: [],
    setPreviewAsset: vi.fn(),
    ...overrides,
  };
}

async function render(value) {
  await act(async () => {
    root.render(
      <AppContext.Provider value={value}>
        <MakePicture />
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

describe("MakePicture", () => {
  it("does not submit when no picture model is available", async () => {
    const ctx = baseContext({ imageModels: [] });
    await render(ctx);
    await setPrompt("a cabin at dusk");

    const create = container.querySelector(".sw-cta");
    expect(create.disabled).toBe(true);

    await act(async () => {
      create.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(ctx.createImageJob).not.toHaveBeenCalled();
  });
});
