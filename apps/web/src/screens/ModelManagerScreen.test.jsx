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

const WAN_MOE_MODEL = {
  id: "wan_2_2_t2v_14b",
  name: "Wan 2.2 T2V A14B",
  type: "video",
  family: "wan-video",
  installState: "ready",
  ui: { description: "Wan A14B MoE video model." },
};

function setNativeValue(element, value) {
  const proto = element.tagName === "SELECT" ? window.HTMLSelectElement.prototype : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(element, value);
}

describe("ModelManagerScreen Wan A14B MoE LoRA import (sc-1991)", () => {
  let container;
  let root;
  let createLoraImportJob;
  let ModelManagerScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    createLoraImportJob = vi.fn(async () => ({ payload: { loraId: "wan_moe", manifestEntry: { family: "wan-video" } } }));
    window.__TAURI__ = { core: { invoke: vi.fn(async () => null) } };
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

  async function render() {
    const value = {
      activeProject: null,
      jobs: [],
      loras: [],
      models: [WAN_MOE_MODEL],
      presets: [],
      jobAction: () => {},
      setActiveView: () => {},
      deleteLora: () => {},
      deleteModel: () => {},
      createModelDownloadJob: () => {},
      createModelConvertJob: () => {},
      createLoraImportJob,
      createModelImportJob: () => {},
    };
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // Both the model and LoRA import forms share the .lora-import-panel class, so
  // scope queries to the LoRA form by its aria-label.
  function loraForm() {
    return container.querySelector('form[aria-label="Import LoRA"]');
  }

  function labelStartingWith(prefix) {
    return [...loraForm().querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith(prefix),
    );
  }

  async function change(element, val) {
    await act(async () => {
      setNativeValue(element, val);
      // Dispatch a single event so React's value tracker doesn't pre-record the new
      // value and then skip onChange: "change" for <select>, "input" for inputs.
      element.dispatchEvent(
        new window.Event(element.tagName === "SELECT" ? "change" : "input", { bubbles: true }),
      );
    });
  }

  async function selectWanVideoFamily() {
    const familySelect = labelStartingWith("Family").querySelector("select");
    await change(familySelect, "wan-video");
  }

  it("reveals the base-model selector only after the wan-video family is chosen", async () => {
    await render();
    expect(labelStartingWith("Base model")).toBeUndefined();
    await selectWanVideoFamily();
    const baseModelLabel = labelStartingWith("Base model");
    expect(baseModelLabel).toBeTruthy();
    expect(baseModelLabel.textContent).toContain("Wan 2.2 T2V A14B");
  });

  it("surfaces the low-noise slot and a warning for an A14B upload missing its second half", async () => {
    await render();
    await selectWanVideoFamily();
    await change(labelStartingWith("Base model").querySelector("select"), "wan_2_2_t2v_14b");
    // Switch from URL to Upload mode.
    const uploadButton = [...loraForm().querySelectorAll("button")].find(
      (button) => button.textContent === "Upload",
    );
    await act(async () => {
      uploadButton.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(labelStartingWith("Low-noise expert")).toBeTruthy();
    expect(container.textContent).toContain("two-expert model");
  });

  it("threads the chosen base model into the import request", async () => {
    await render();
    await selectWanVideoFamily();
    await change(labelStartingWith("Base model").querySelector("select"), "wan_2_2_t2v_14b");
    await change(labelStartingWith("Source URL").querySelector("input"), "https://example.com/wan-moe.safetensors");
    const form = container.querySelector('form[aria-label="Import LoRA"]');
    await act(async () => {
      form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(createLoraImportJob).toHaveBeenCalledTimes(1);
    const payload = createLoraImportJob.mock.calls[0][0];
    expect(payload.family).toBe("wan-video");
    expect(payload.baseModel).toBe("wan_2_2_t2v_14b");
  });

  // Distilled variants whose model `family` differs from their LoRA-compatibility
  // set (FLUX.2 [klein]: family "flux2-klein" but accepts "flux2" LoRAs) must offer
  // the compatibility family in the dropdown — the import validator + generation
  // matcher both key off loraCompatibility.families, so offering "flux2-klein"
  // would let the user pick a family the backend rejects ("Unsupported LoRA family").
  it("offers the LoRA-compatibility family, not the model identity, for distilled variants", async () => {
    const value = {
      activeProject: null,
      jobs: [],
      loras: [],
      models: [
        {
          id: "flux2_klein_9b",
          name: "FLUX.2 [klein] 9B",
          type: "image",
          family: "flux2-klein",
          loraCompatibility: { families: ["flux2"] },
          installState: "ready",
          ui: { description: "Distilled FLUX.2 variant." },
        },
      ],
      presets: [],
      jobAction: () => {},
      setActiveView: () => {},
      deleteLora: () => {},
      deleteModel: () => {},
      createModelDownloadJob: () => {},
      createModelConvertJob: () => {},
      createLoraImportJob,
      createModelImportJob: () => {},
    };
    await act(async () => {
      root.render(
        <AppContext.Provider value={value}>
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    const familyOptions = [...labelStartingWith("Family").querySelectorAll("option")].map((option) => option.value);
    expect(familyOptions).toContain("flux2");
    expect(familyOptions).not.toContain("flux2-klein");
  });
});

describe("ModelManagerScreen type-grouped layout", () => {
  let container;
  let root;
  let ModelManagerScreen;
  let AppContext;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.__TAURI__ = { core: { invoke: vi.fn(async () => null) } };
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

  async function render({ models = [], loras = [] } = {}) {
    const value = {
      activeProject: null,
      jobs: [],
      loras,
      models,
      presets: [],
      jobAction: () => {},
      setActiveView: () => {},
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
    await act(async () => {});
  }

  const MODELS = [
    { id: "z_image_turbo", name: "Z-Image-Turbo", type: "image", family: "z-image", capabilities: ["text_to_image", "style_variations"], installState: "missing" },
    { id: "wan_t2v", name: "Wan T2V", type: "video", family: "wan-video", capabilities: ["text_to_video"], installState: "missing" },
    { id: "real_esrgan", name: "Real-ESRGAN", type: "utility", family: "real-esrgan", capabilities: [], installState: "missing" },
  ];

  function groupHeadings() {
    return [...container.querySelectorAll(".model-type-group-heading h3")].map((h) => h.textContent);
  }

  it("renders one section per populated model type, in fixed order", async () => {
    await render({ models: MODELS });
    expect(groupHeadings()).toEqual(["Image Models", "Video Models", "Utility Models"]);
    // Each group holds exactly its own type's card.
    const groups = container.querySelectorAll(".model-type-group");
    expect(groups.length).toBe(3);
    expect(groups[0].querySelectorAll(".model-card").length).toBe(1);
    expect(groups[0].textContent).toContain("Z-Image-Turbo");
    expect(groups[1].textContent).toContain("Wan T2V");
    expect(groups[2].textContent).toContain("Real-ESRGAN");
  });

  it("omits a type section when no model has that type", async () => {
    await render({ models: [MODELS[0]] });
    expect(groupHeadings()).toEqual(["Image Models"]);
  });

  it("describes each model's capabilities as chips on the card", async () => {
    await render({ models: [MODELS[0]] });
    const chips = [...container.querySelectorAll(".model-capabilities .chip")].map((c) => c.textContent);
    expect(chips).toEqual(["Text to Image", "Style Variations"]);
  });

  it("groups LoRAs by family with a heading per family", async () => {
    await render({
      models: MODELS,
      loras: [
        { id: "a", name: "Flux A", family: "flux", installState: "installed" },
        { id: "b", name: "Flux B", family: "flux", installState: "installed" },
        { id: "c", name: "Wan C", family: "wan-video", installState: "installed" },
      ],
    });
    const families = [...container.querySelectorAll(".lora-family-group-heading h3")].map((h) => h.textContent);
    expect(families).toEqual(["flux", "wan-video"]);
    const groups = container.querySelectorAll(".lora-family-group");
    expect(groups[0].querySelectorAll(".lora-row").length).toBe(2);
    expect(groups[1].querySelectorAll(".lora-row").length).toBe(1);
  });

  it("buckets family-less LoRAs under a trailing 'Other / compatible' group", async () => {
    await render({
      models: MODELS,
      loras: [
        { id: "a", name: "Flux A", family: "flux", installState: "installed" },
        { id: "x", name: "Loose", installState: "installed" },
      ],
    });
    const families = [...container.querySelectorAll(".lora-family-group-heading h3")].map((h) => h.textContent);
    expect(families).toEqual(["flux", "Other / compatible"]);
  });
});
