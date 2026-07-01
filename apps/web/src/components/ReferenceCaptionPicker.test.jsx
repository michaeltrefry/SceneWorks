import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ReferenceCaptionPicker from "./ReferenceCaptionPicker.jsx";

// Shared reference-image → prompt picker (epic 8203, sc-8208). These cover the prose/tags "describe"
// surface; the Ideogram JSON surface is covered through StructuredPromptBuilder's reference tests.

function click(el) {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

describe("ReferenceCaptionPicker", () => {
  let container;
  let root;

  async function clickAndSettle(el) {
    await act(async () => {
      click(el);
      await new Promise((r) => setTimeout(r, 0));
    });
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const buttonByText = (text) =>
    [...document.body.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

  const refAsset = {
    id: "ref-1",
    type: "image",
    projectId: "proj-1",
    file: { path: "uploads/ref.png", mimeType: "image/png" },
  };

  async function selectReference() {
    await clickAndSettle(buttonByText("Select reference image"));
    const card = document.body.querySelector(".asset-picker-card");
    await clickAndSettle(card);
    await clickAndSettle(buttonByText("Use Selection"));
  }

  async function mount(props = {}) {
    await act(async () =>
      root.render(
        <ReferenceCaptionPicker
          onCaption={props.onCaption}
          onApply={props.onApply ?? (() => {})}
          referenceAssets={props.referenceAssets ?? [refAsset]}
          projectId="proj-1"
          buttonLabel="✨ Describe image"
          busyLabel="Describing…"
          {...props}
        />,
      ),
    );
  }

  it("keeps the describe button disabled until a reference is selected", async () => {
    await mount({ onCaption: vi.fn(async () => "a fox") });
    expect(buttonByText("✨ Describe image").disabled).toBe(true);
    await selectReference();
    expect(buttonByText("✨ Describe image").disabled).toBe(false);
  });

  it("runs the caption for the picked asset and applies the result", async () => {
    const onCaption = vi.fn(async () => "a red fox in snow, cinematic photograph");
    const onApply = vi.fn();
    await mount({ onCaption, onApply });
    await selectReference();
    await clickAndSettle(buttonByText("✨ Describe image"));

    expect(onCaption).toHaveBeenCalledWith("ref-1");
    expect(onApply).toHaveBeenCalledWith("a red fox in snow, cinematic photograph");
  });

  it("shows the empty message and does not apply when the result is falsy", async () => {
    const onApply = vi.fn();
    await mount({
      onCaption: vi.fn(async () => ""),
      onApply,
      emptyMessage: "No usable description.",
    });
    await selectReference();
    await clickAndSettle(buttonByText("✨ Describe image"));

    expect(onApply).not.toHaveBeenCalled();
    expect(document.body.querySelector(".structured-error")?.textContent).toContain(
      "No usable description.",
    );
  });

  it("surfaces a thrown error from the caption call", async () => {
    await mount({
      onCaption: vi.fn(async () => {
        throw new Error("describe blew up");
      }),
    });
    await selectReference();
    await clickAndSettle(buttonByText("✨ Describe image"));

    expect(document.body.querySelector(".structured-error")?.textContent).toContain("describe blew up");
  });

  it("gates behind the download offer (no picker/button) when the captioner is missing", async () => {
    const onDownloadModel = vi.fn();
    const offer = { id: "vision_caption_qwen3vl_8b", name: "Vision Captioner", downloadSizeLabel: "18 GB" };
    await mount({
      onCaption: vi.fn(async () => "x"),
      visionCaptionReady: false,
      visionCaptionOffers: [offer],
      onDownloadModel,
    });

    expect(buttonByText("✨ Describe image")).toBeFalsy();
    expect(buttonByText("Select reference image")).toBeFalsy();
    expect(document.body.querySelector(".model-availability-gate")).toBeTruthy();
    const download = buttonByText("Download");
    expect(download).toBeTruthy();
    await clickAndSettle(download);
    expect(onDownloadModel).toHaveBeenCalledWith(offer);
  });
});
