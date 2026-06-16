import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "./ModelAvailabilityGate.jsx";

describe("ModelAvailabilityGate", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  async function renderGate(props) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ workersById: new Map(), visibleWorkers: [] }}>
          <ModelAvailabilityGate {...props} />
        </AppContext.Provider>,
      );
    });
  }

  async function click(el) {
    await act(async () => {
      el.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
  }

  it("renders children when ready", async () => {
    await renderGate({ ready: true, children: <div className="studio-body">Studio</div> });
    expect(container.querySelector(".studio-body")).toBeTruthy();
    expect(container.querySelector(".model-availability-gate")).toBeNull();
  });

  it("offers recommended models for download when gated and queues on click", async () => {
    const onDownload = vi.fn();
    await renderGate({
      ready: false,
      title: "Image Studio needs a model",
      offers: [{ id: "z_image_turbo", name: "Z-Image-Turbo", downloadSizeLabel: "30.6 GB", downloadSizeEstimated: true }],
      downloadJobs: [],
      onDownload,
      onOpenModels: () => {},
      children: <div className="studio-body">Studio</div>,
    });
    expect(container.querySelector(".studio-body")).toBeNull();
    expect(container.textContent).toContain("Image Studio needs a model");
    expect(container.textContent).toContain("~30.6 GB");
    const downloadButton = [...container.querySelectorAll(".model-availability-offer button")].find(
      (button) => button.textContent === "Download",
    );
    expect(downloadButton).toBeTruthy();
    await click(downloadButton);
    expect(onDownload).toHaveBeenCalledWith(expect.objectContaining({ id: "z_image_turbo" }));
  });

  it("shows progress and disables the button while an offer is downloading", async () => {
    await renderGate({
      ready: false,
      title: "Video Studio needs a model",
      offers: [{ id: "ltx_2_3", name: "LTX-2.3", downloadSizeLabel: "146 GB" }],
      downloadJobs: [{ id: "j1", type: "model_download", status: "downloading", progress: 0.3, payload: { modelId: "ltx_2_3" } }],
      onDownload: vi.fn(),
      children: <div className="studio-body">Studio</div>,
    });
    const button = container.querySelector(".model-availability-offer button");
    expect(button.disabled).toBe(true);
    expect(button.textContent).toBe("downloading");
  });
});
