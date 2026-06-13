import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { VideoUpscalePanel } from "./VideoUpscalePanel.jsx";

const VIDEO = {
  id: "vid_1",
  type: "video",
  displayName: "Clip A",
  file: { width: 512, height: 384 },
};

let container;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  container.remove();
});

async function renderPanel(ui) {
  await act(async () => {
    createRoot(container).render(ui);
  });
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

function findButton(text) {
  return [...container.querySelectorAll("button")].find((button) =>
    button.textContent.includes(text),
  );
}

describe("VideoUpscalePanel", () => {
  it("submits a video_upscale job for the selected clip (SeedVR2, default 2x)", async () => {
    const createVideoUpscaleJob = vi.fn(async () => ({ id: "job_1" }));
    await renderPanel(
      <VideoUpscalePanel
        createVideoUpscaleJob={createVideoUpscaleJob}
        selectedAsset={VIDEO}
        videoAssets={[VIDEO]}
      />,
    );
    await click(findButton("Upscale clip"));
    expect(createVideoUpscaleJob).toHaveBeenCalledTimes(1);
    expect(createVideoUpscaleJob.mock.calls[0][0]).toMatchObject({
      sourceAssetId: "vid_1",
      factor: 2,
      engine: "seedvr2",
      model: "seedvr2_3b",
      softness: 0,
    });
  });

  it("honors a 4x scale selection", async () => {
    const createVideoUpscaleJob = vi.fn(async () => ({ id: "job_2" }));
    await renderPanel(
      <VideoUpscalePanel
        createVideoUpscaleJob={createVideoUpscaleJob}
        selectedAsset={VIDEO}
        videoAssets={[VIDEO]}
      />,
    );
    await click([...container.querySelectorAll("button")].find((b) => b.textContent.trim() === "4×"));
    await click(findButton("Upscale clip"));
    expect(createVideoUpscaleJob.mock.calls[0][0]).toMatchObject({ factor: 4 });
  });

  it("does not submit without a source clip", async () => {
    const createVideoUpscaleJob = vi.fn(async () => ({ id: "job_3" }));
    await renderPanel(
      <VideoUpscalePanel createVideoUpscaleJob={createVideoUpscaleJob} videoAssets={[VIDEO]} />,
    );
    const button = findButton("Upscale clip");
    expect(button.disabled).toBe(true);
    await click(button);
    expect(createVideoUpscaleJob).not.toHaveBeenCalled();
  });
});
