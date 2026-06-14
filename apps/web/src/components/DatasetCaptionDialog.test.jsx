import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetCaptionDialog } from "./DatasetCaptionDialog.jsx";

const baseSettings = {
  captioner: "joy_caption",
  modelNameOrPath: "",
  requestedGpu: "auto",
  captionType: "Descriptive",
  captionLength: "long",
  nameInput: "",
  temperature: 0.6,
  topP: 0.9,
  maxNewTokens: 256,
  lowVram: false,
  recaption: false,
  extraOptions: [],
};

function buttonByText(container, text) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === text);
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("DatasetCaptionDialog", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function render(ui) {
    root = createRoot(container);
    act(() => root.render(ui));
  }

  it("offers a download + blocks Run when the captioning model is missing (sc-5620)", async () => {
    const onDownloadModel = vi.fn(async () => ({ id: "job-1" }));
    render(
      <DatasetCaptionDialog
        settings={baseSettings}
        onChange={vi.fn()}
        onRun={vi.fn()}
        onToggleExtra={vi.fn()}
        onClose={vi.fn()}
        scope={{ type: "all" }}
        modelMissing
        onDownloadModel={onDownloadModel}
        modelSizeLabel="17.0 GB"
        modelName="JoyCaption (beta one)"
      />,
    );

    expect(container.querySelector(".caption-missing-model").textContent).toContain("isn’t installed");
    expect(container.querySelector(".caption-missing-model").textContent).toContain("17.0 GB");
    // Run is blocked while the model is missing.
    expect(buttonByText(container, "Caption missing").disabled).toBe(true);

    const download = buttonByText(container, "Download captioning model");
    expect(download).toBeTruthy();
    await act(async () => {
      download.click();
    });
    await settle();

    expect(onDownloadModel).toHaveBeenCalledTimes(1);
    expect(container.querySelector(".caption-missing-model").textContent).toContain("Downloading");
  });

  it("shows no affordance and enables Run when the captioning model is present", () => {
    render(
      <DatasetCaptionDialog
        settings={baseSettings}
        onChange={vi.fn()}
        onRun={vi.fn()}
        onToggleExtra={vi.fn()}
        onClose={vi.fn()}
        scope={{ type: "all" }}
        modelMissing={false}
      />,
    );

    expect(container.querySelector(".caption-missing-model")).toBeNull();
    expect(buttonByText(container, "Caption missing").disabled).toBe(false);
  });
});
