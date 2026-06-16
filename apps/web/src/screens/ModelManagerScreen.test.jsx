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

  async function render({
    models = [],
    loras = [],
    jobs = [],
    createModelDownloadJob = () => {},
    createLoraDownloadJob = () => {},
  } = {}) {
    const value = {
      activeProject: null,
      jobs,
      loras,
      models,
      presets: [],
      workersById: new Map(),
      visibleWorkers: [],
      jobAction: () => {},
      setActiveView: () => {},
      deleteLora: () => {},
      deleteModel: () => {},
      createModelDownloadJob,
      createLoraDownloadJob,
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

  it("splits a type into Recommended and a collapsed Additional Supported when both exist", async () => {
    await render({
      models: [
        { id: "z_image_turbo", name: "Z-Image-Turbo", type: "image", family: "z-image", capabilities: ["text_to_image"], installState: "missing", recommended: true },
        { id: "flux_dev", name: "FLUX.1 [dev]", type: "image", family: "flux", capabilities: ["text_to_image"], installState: "missing" },
      ],
    });
    const imageGroup = container.querySelector(".model-type-group");
    const recommendedGrid = imageGroup.querySelector(".model-subgroup .model-grid");
    expect(imageGroup.textContent).toContain("Recommended Models");
    expect(recommendedGrid.querySelectorAll(".model-card").length).toBe(1);
    expect(recommendedGrid.textContent).toContain("Z-Image-Turbo");
    const additional = imageGroup.querySelector(".model-subgroup-additional");
    expect(additional.tagName).toBe("DETAILS");
    expect(additional.open).toBe(false); // collapsed by default to cut clutter
    expect(additional.textContent).toContain("Additional Supported Models");
    expect(additional.querySelectorAll(".model-card").length).toBe(1);
    expect(additional.textContent).toContain("FLUX.1 [dev]");
  });

  it("shows a single grid (no Recommended/Additional split) when a type has no recommended model", async () => {
    await render({ models: [MODELS[0]] }); // fixture has no recommended flag
    const imageGroup = container.querySelector(".model-type-group");
    expect(imageGroup.textContent).not.toContain("Recommended Models");
    expect(imageGroup.querySelector(".model-subgroup-additional")).toBeNull();
    expect(imageGroup.querySelectorAll(".model-card").length).toBe(1);
  });

  it("describes each model's capabilities as chips on the card", async () => {
    await render({ models: [MODELS[0]] });
    const chips = [...container.querySelectorAll(".model-capabilities .chip")].map((c) => c.textContent);
    expect(chips).toEqual(["Text to Image", "Style Variations"]);
  });

  it("offers a Fix action when a cached model is incomplete", async () => {
    const createModelDownloadJob = vi.fn();
    await render({
      createModelDownloadJob,
      models: [
        {
          ...MODELS[0],
          cacheState: "incomplete",
          repairAvailable: true,
          missingRequiredFiles: ["vae/config.json"],
          downloadable: true,
        },
      ],
    });
    expect(container.textContent).toContain("incomplete");
    expect(container.textContent).toContain("vae/config.json");
    const fixButton = [...container.querySelectorAll(".model-card-actions button")].find(
      (button) => button.textContent === "Fix",
    );
    expect(fixButton).toBeTruthy();
    await click(fixButton);
    expect(createModelDownloadJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "z_image_turbo" }),
    );
  });

  it("offers a Fix action when an installed model has an incomplete cache", async () => {
    const createModelDownloadJob = vi.fn();
    await render({
      createModelDownloadJob,
      models: [
        {
          ...MODELS[0],
          installState: "installed",
          cacheState: "incomplete",
          repairAvailable: true,
          missingRequiredFiles: ["model_index.json"],
          downloadable: true,
        },
      ],
    });
    const fixButton = [...container.querySelectorAll(".model-card-actions button")].find(
      (button) => button.textContent === "Fix",
    );
    expect(fixButton).toBeTruthy();
    expect(fixButton.disabled).toBe(false);
    await click(fixButton);
    expect(createModelDownloadJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "z_image_turbo" }),
    );
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
        { id: "a", name: "Flux A", family: "flux", scope: "global", installState: "installed" },
        { id: "x", name: "Loose", scope: "global", installState: "installed" },
      ],
    });
    const families = [...container.querySelectorAll(".lora-family-group-heading h3")].map((h) => h.textContent);
    expect(families).toEqual(["flux", "Other / compatible"]);
  });

  it("separates built-in LoRAs from user LoRAs into their own sections", async () => {
    await render({
      models: MODELS,
      loras: [
        { id: "ltx_ic", name: "LTX IC", family: "ltx-video", scope: "builtin", installState: "missing" },
        { id: "u1", name: "My LoRA", family: "flux", scope: "global", installState: "installed" },
      ],
    });
    const scopeHeadings = [...container.querySelectorAll(".lora-scope-group-heading h3")].map((h) => h.textContent);
    expect(scopeHeadings).toEqual(["Built-In LoRAs", "User LoRAs"]);
    const builtin = [...container.querySelectorAll(".lora-scope-group")].find((group) =>
      group.querySelector("h3")?.textContent === "Built-In LoRAs",
    );
    expect(builtin.querySelectorAll(".lora-row").length).toBe(1);
    expect(builtin.textContent).toContain("LTX IC");
    expect(builtin.textContent).not.toContain("My LoRA");
    // The user LoRA is family-grouped inside the User section only.
    const families = [...container.querySelectorAll(".lora-family-group-heading h3")].map((h) => h.textContent);
    expect(families).toEqual(["flux"]);
  });

  function builtinSection() {
    return [...container.querySelectorAll(".lora-scope-group")].find(
      (group) => group.querySelector("h3")?.textContent === "Built-In LoRAs",
    );
  }

  it("offers a Download button on a built-in HF LoRA and queues a download", async () => {
    const createLoraDownloadJob = vi.fn();
    await render({
      models: MODELS,
      createLoraDownloadJob,
      loras: [
        {
          id: "ltx_ic",
          name: "LTX IC",
          family: "ltx-video",
          scope: "builtin",
          installState: "missing",
          source: { provider: "huggingface", repo: "Lightricks/LTX-2.3-IC", file: "ic.safetensors" },
        },
      ],
    });
    const downloadButton = [...builtinSection().querySelectorAll(".lora-row-actions button")].find(
      (button) => button.textContent === "Download",
    );
    expect(downloadButton).toBeTruthy();
    await click(downloadButton);
    expect(createLoraDownloadJob).toHaveBeenCalledWith(expect.objectContaining({ id: "ltx_ic" }));
  });

  it("shows progress and disables the button while a built-in LoRA download runs", async () => {
    await render({
      models: MODELS,
      loras: [
        {
          id: "ltx_ic",
          name: "LTX IC",
          family: "ltx-video",
          scope: "builtin",
          installState: "missing",
          source: { provider: "huggingface", repo: "r", file: "f" },
        },
      ],
      jobs: [{ id: "j1", type: "lora_download", status: "downloading", progress: 0.4, payload: { loraId: "ltx_ic" } }],
    });
    const actionButton = builtinSection().querySelector(".lora-row-actions button");
    expect(actionButton.disabled).toBe(true);
    expect(actionButton.textContent).toBe("downloading");
    expect(builtinSection().querySelector(".lora-row-progress")).toBeTruthy();
  });

  it("does not offer Download on a built-in LoRA without a Hugging Face source", async () => {
    await render({
      models: MODELS,
      createLoraDownloadJob: vi.fn(),
      loras: [
        {
          id: "local_builtin",
          name: "Local Builtin",
          family: "z-image",
          scope: "builtin",
          installState: "missing",
          source: { provider: "local", path: "loras/x.safetensors" },
        },
      ],
    });
    const downloadButton = [...builtinSection().querySelectorAll(".lora-row-actions button")].find(
      (button) => button.textContent === "Download",
    );
    expect(downloadButton).toBeUndefined();
  });
});
