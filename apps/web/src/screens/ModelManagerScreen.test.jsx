import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// ModelManagerScreen reads `window.__TAURI__` at module load (via ../credentials.js)
// to pick the keychain transport, so we set the Tauri bridge and re-import the
// module fresh in each test. Credentials are served by the mocked `list_credentials`
// command. AppContext is imported from the same fresh graph so the provider matches.
async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

const GATED_MODEL = {
  id: "flux_dev",
  name: "FLUX.1 [dev]",
  type: "image",
  family: "flux",
  installState: "missing",
  downloadable: true,
  gated: true,
  credentialHost: "huggingface.co",
  licenseUrl: "https://huggingface.co/black-forest-labs/FLUX.1-dev",
  ui: { description: "Gated FLUX model." },
};

const PLAIN_MODEL = {
  id: "z_image_turbo",
  name: "Z-Image-Turbo",
  type: "image",
  family: "z-image",
  installState: "missing",
  downloadable: true,
  ui: { description: "Open model." },
};

describe("ModelManagerScreen gated-model notice", () => {
  let container;
  let root;
  let invoke;
  let credentials;
  let setActiveView;
  let ModelManagerScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    credentials = [];
    setActiveView = vi.fn();
    invoke = vi.fn(async (command) => {
      switch (command) {
        case "get_gpu_info":
          return { platform: "windows", devices: [] };
        case "list_credentials":
          return credentials;
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    ({ AppContext } = await import("../context/AppContext.js"));
    ({ ModelManagerScreen } = await import("./ModelManagerScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function render(models) {
    const value = {
      activeProject: null,
      jobs: [],
      loras: [],
      models,
      presets: [],
      jobAction: () => {},
      setActiveView,
      deleteLora: () => {},
      deleteModel: () => {},
      createModelDownloadJob: () => {},
      createModelConvertJob: () => {},
      createLoraImportJob: () => {},
      createModelImportJob: () => {},
    };
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    // Flush the gated-credential and GPU-info effect promises.
    await act(async () => {});
  }

  it("shows a gated notice with a Settings link when no credential is saved", async () => {
    await render([GATED_MODEL]);
    expect(invoke).toHaveBeenCalledWith("list_credentials", undefined);
    expect(container.textContent).toContain("Gated download");
    expect(container.textContent).toContain("huggingface.co");

    const settingsButton = [...container.querySelectorAll(".model-gated-actions button")].find(
      (button) => button.textContent === "Add token in Settings",
    );
    expect(settingsButton).toBeTruthy();
    await click(settingsButton);
    expect(setActiveView).toHaveBeenCalledWith("Settings");

    const license = container.querySelector(".model-gated-actions a");
    expect(license?.getAttribute("href")).toBe(
      "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    );
  });

  it("softens the notice once a matching credential is present", async () => {
    credentials = [{ host: "huggingface.co", label: "Hugging Face", scheme: "bearer", present: true }];
    await render([GATED_MODEL]);
    expect(container.textContent).toContain("ready to download");
    expect(container.textContent).not.toContain("Add token in Settings");
  });

  it("renders no gated notice for a non-gated model", async () => {
    await render([PLAIN_MODEL]);
    expect(invoke).not.toHaveBeenCalledWith("list_credentials", undefined);
    expect(container.textContent).not.toContain("Gated download");
    expect(container.querySelector(".model-gated-notice")).toBeNull();
  });
});
