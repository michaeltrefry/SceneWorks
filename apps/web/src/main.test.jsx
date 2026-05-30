import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, ErrorBoundary, eventUrl } from "./main.jsx";
import { AssetPickerField } from "./components/AssetPicker.jsx";
import { AssetDetail, FullscreenPreview } from "./components/assetPanels.jsx";
import { liveElapsedSeconds } from "./formatting.js";
import { foldUpscaledAssetVariants } from "./assetVariants.js";
import { extractFamilies } from "./presetUtils.js";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { CharacterAssets, CharacterDatasets } from "./screens/characterPanels.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { DocumentStudio } from "./screens/DocumentStudio.jsx";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { SetupWizard } from "./screens/SetupWizard.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { ReplacePersonPanel } from "./screens/ReplacePersonPanel.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { TrainingDataSetsLibrary, TrainingStudio } from "./screens/TrainingStudio.jsx";
import { AppContext } from "./context/AppContext.js";
import { qualityChoices, GPU_REQUIRED_JOB_TYPES, errorStatuses } from "./jobTypes.js";

// sc-1651 Phase B: screens converted to useAppContext() read their data from the
// provider instead of props. Tests wrap the screen in a provider carrying only the
// values that screen reads.
function withAppContext(value, ui) {
  return <AppContext.Provider value={value}>{ui}</AppContext.Provider>;
}

// ModelManagerScreen (sc-1651 Phase B) reads primitives from context and derives
// its own on* callbacks. This adapter lets the existing tests keep their old
// prop-shaped objects (and their assertions on those fns) while feeding the
// screen via the provider.
// ImageStudio (sc-1651 Phase B) — same adapter idea as ModelManager: keep the
// old prop-shaped fixtures (and their assertions) and map onto the provider.
function withImageStudioContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      assets: p.assets,
      characters: p.characters,
      createImageJob: p.createImageJob,
      refinePrompt: p.refinePrompt,
      deleteAsset: p.deleteAsset,
      purgeAsset: p.purgeAsset,
      gpuOptions: p.gpuOptions,
      imageModels: p.imageModels,
      latestImageAssets: p.latestAssets,
      studioLaunch: p.launchRequest,
      imageLocalJobs: p.localJobs,
      loras: p.loras,
      presets: p.presets,
      requestedGpu: p.requestedGpu,
      selectedAsset: p.selectedAsset,
      setRequestedGpu: p.setRequestedGpu,
      updateAssetStatus: p.updateAssetStatus,
      setPreviewAsset: p.onPreview ?? (() => {}),
      jobAction: p.onCancelJob ? (job) => p.onCancelJob(job) : () => {},
      rememberLocalGenerationJob: p.onLocalJobCreated ? (_kind, job) => p.onLocalJobCreated(job) : () => {},
      setActiveView: (view) => {
        if (view === "Presets") p.onOpenPresets?.();
        else if (view === "Queue") p.onOpenQueue?.();
      },
    },
    <ImageStudio />,
  );
}

function withModelManagerContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      jobs: p.jobs,
      loras: p.loras,
      models: p.models,
      presets: p.recipePresets,
      jobAction: p.onCancelJob ? (job) => p.onCancelJob(job) : () => {},
      setActiveView: p.onOpenQueue ? () => p.onOpenQueue() : () => {},
      deleteLora: p.onDeleteLora,
      deleteModel: p.onDeleteModel,
      createModelDownloadJob: p.onDownloadModel,
      createModelConvertJob: p.onConvertModel,
      createLoraImportJob: p.onImportLora,
      createModelImportJob: p.onImportModel,
    },
    <ModelManagerScreen />,
  );
}

// TrainingStudio (sc-1651 Phase B) — maps the old prop-shaped fixtures onto the
// provider. The screen unwraps catalogs ({presets}/{targets}) and project-scopes
// datasets, so feed those raw, and route the derived callbacks back to the
// fixture's on* / wrapped fns.
function withTrainingStudioContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      authenticated: p.authenticated,
      assets: p.assets,
      gpuOptions: p.gpuOptions,
      jobs: p.jobs,
      setPreviewAsset: p.onPreview ?? (() => {}),
      importAsset: p.importAsset,
      trainingDatasets: p.datasets,
      trainingDatasetsProjectId: p.activeProject?.id,
      trainingDatasetsError: p.datasetsError,
      loadingTrainingDatasets: p.loadingDatasets,
      refreshTrainingDatasets: p.onRefreshDatasets ? () => p.onRefreshDatasets() : () => {},
      loadTrainingDataset: p.loadDataset,
      createTrainingDataset: p.createDataset,
      uploadTrainingDatasetItem: p.uploadDatasetItem,
      updateTrainingDataset: p.updateDataset,
      batchRenameTrainingDataset: p.batchRenameDataset,
      writeTrainingDatasetCaptionSidecars: p.writeCaptionSidecars,
      createTrainingDatasetCaptionJob: p.createCaptionJob,
      createTrainingJob: p.createTrainingJob,
      trainingPresets: { presets: p.trainingPresets },
      trainingPresetsError: p.trainingPresetsError,
      trainingTargets: { targets: p.trainingTargets },
      trainingTargetsError: p.trainingTargetsError,
    },
    <TrainingStudio />,
  );
}

function withTrainingDataSetsLibraryContext(p) {
  return withAppContext(
    {
      activeProject: p.activeProject,
      authenticated: p.authenticated,
      assets: p.assets,
      characters: p.characters,
      gpuOptions: p.gpuOptions,
      jobs: p.jobs,
      setPreviewAsset: p.onPreview ?? (() => {}),
      importAsset: p.importAsset,
      trainingDatasets: p.datasets,
      trainingDatasetsProjectId: p.activeProject?.id,
      trainingDatasetsError: p.datasetsError,
      loadingTrainingDatasets: p.loadingDatasets,
      refreshTrainingDatasets: p.onRefreshDatasets ? () => p.onRefreshDatasets() : () => {},
      loadTrainingDataset: p.loadDataset,
      createTrainingDataset: p.createDataset,
      uploadTrainingDatasetItem: p.uploadDatasetItem,
      updateTrainingDataset: p.updateDataset,
      batchRenameTrainingDataset: p.batchRenameDataset,
      writeTrainingDatasetCaptionSidecars: p.writeCaptionSidecars,
      createTrainingDatasetCaptionJob: p.createCaptionJob,
      createTrainingJob: p.createTrainingJob,
      trainingPresets: { presets: p.trainingPresets },
      trainingPresetsError: p.trainingPresetsError,
      trainingTargets: { targets: p.trainingTargets },
      trainingTargetsError: p.trainingTargetsError,
      setActiveView: p.setActiveView,
    },
    <TrainingDataSetsLibrary />,
  );
}

// jsdom 27 omits Blob.text(); all real browsers implement it. Polyfill so the
// dataset import flow (which reads .txt caption sidecars) is exercisable here.
if (typeof Blob !== "undefined" && typeof Blob.prototype.text !== "function") {
  Blob.prototype.text = function text() {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ""));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(this);
    });
  };
}

class FakeEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.listeners = {};
    FakeEventSource.instances.push(this);
  }

  addEventListener(event, handler) {
    this.listeners[event] = handler;
  }

  close() {}
}

function response(payload) {
  return {
    ok: true,
    json: async () => payload,
  };
}

function errorResponse(status, detail) {
  return {
    ok: false,
    status,
    json: async () => ({ detail }),
  };
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 6; index += 1) {
      await Promise.resolve();
    }
  });
}

function field(container, labelText) {
  const label = [...container.querySelectorAll("label")].find((item) => item.childNodes[0]?.textContent.trim() === labelText);
  return label?.querySelector("input, select, textarea");
}

function loraPanel(container) {
  return container.querySelector("form[aria-label='Import LoRA']");
}

function modelImportPanel(container) {
  return container.querySelector("form[aria-label='Import model']");
}

function buttonInside(scope, label) {
  return [...scope.querySelectorAll("button")].find((button) => button.textContent === label);
}

function navLabels(container, sectionLabel) {
  const section = [...container.querySelectorAll(".sidebar-section")].find(
    (item) => item.querySelector(".sidebar-section-title")?.textContent === sectionLabel,
  );
  return [...(section?.querySelectorAll(".nav-label") ?? [])].map((item) => item.textContent);
}

const zImageTrainingTarget = {
  id: "z_image_turbo_lora",
  name: "Z-Image-Turbo LoRA",
  modality: "image",
  outputKind: "lora",
  family: "z-image",
  baseModel: "z_image_turbo",
  kernel: "z_image_lora",
  defaults: {
    rank: 16,
    alpha: 16,
    learningRate: 0.0001,
    steps: 3000,
    batchSize: 1,
    gradientAccumulation: 1,
    resolution: 1024,
    saveEvery: 250,
    seed: 42,
    optimizer: "adamw8bit",
    advanced: {
      mixedPrecision: "bf16",
      cacheTextEmbeddings: true,
      gradientCheckpointing: true,
      timestepType: "sigmoid",
      timestepBias: "high_noise",
      lossType: "mse",
      weightDecay: 0.0001,
      sampleEvery: 250,
      sampleSteps: 8,
      sampleGuidanceScale: 0.0,
      qualityPreset: "balanced",
      outputScope: "project",
      requestedGpu: "auto",
    },
  },
  limits: {
    resolutions: [512, 768, 1024],
    optimizers: ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"],
    qualityPresets: ["speed", "balanced", "quality"],
    outputScopes: ["project", "global"],
  },
  ui: { label: "Z-Image-Turbo LoRA" },
};

const zImageTrainingPresets = [
  {
    id: "z_image_turbo_lora.character.adamw8bit.balanced",
    version: 1,
    targetId: "z_image_turbo_lora",
    name: "Character balanced",
    recommendedFor: ["character"],
    optimizer: "adamw8bit",
    qualityPreset: "balanced",
    config: {
      ...zImageTrainingTarget.defaults,
      advanced: {
        ...zImageTrainingTarget.defaults.advanced,
        trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
        trainingAdapterVersion: "v2-default",
      },
    },
    ui: { default: true, order: 10 },
  },
  {
    id: "z_image_turbo_lora.character.adamw8bit.conservative",
    version: 1,
    targetId: "z_image_turbo_lora",
    name: "Character conservative",
    recommendedFor: ["character"],
    optimizer: "adamw8bit",
    qualityPreset: "conservative",
    config: {
      ...zImageTrainingTarget.defaults,
      rank: 8,
      alpha: 8,
      learningRate: 0.00005,
      advanced: {
        ...zImageTrainingTarget.defaults.advanced,
        qualityPreset: "conservative",
        trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
        trainingAdapterVersion: "v2-default",
      },
    },
    ui: { order: 20 },
  },
  {
    id: "z_image_turbo_lora.character.prodigyopt.balanced",
    version: 1,
    targetId: "z_image_turbo_lora",
    name: "Prodigy character (experimental)",
    recommendedFor: ["character"],
    optimizer: "prodigyopt",
    qualityPreset: "balanced",
    config: {
      ...zImageTrainingTarget.defaults,
      optimizer: "prodigyopt",
      learningRate: 1.0,
      steps: 1600,
      saveEvery: 200,
      advanced: {
        ...zImageTrainingTarget.defaults.advanced,
        sampleEvery: 200,
        trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
        trainingAdapterVersion: "v2-default",
      },
    },
    ui: { experimental: true, order: 40 },
  },
];

async function changeField(input, value) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new window.Event(input.tagName === "SELECT" ? "change" : "input", { bubbles: true }));
  });
}

async function changeFile(input, file) {
  await act(async () => {
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    input.dispatchEvent(new window.Event("change", { bubbles: true }));
  });
}

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it("shows a fallback instead of a blank screen when rendering fails", async () => {
    function BrokenScreen() {
      throw new Error("Render smoke signal");
    }

    const preventExpectedError = (event) => event.preventDefault();
    window.addEventListener("error", preventExpectedError);
    vi.spyOn(console, "error").mockImplementation(() => {});
    root = createRoot(container);
    try {
      await act(async () => {
        root.render(
          <ErrorBoundary>
            <BrokenScreen />
          </ErrorBoundary>,
        );
      });
    } finally {
      window.removeEventListener("error", preventExpectedError);
    }

    expect(container.textContent).toContain("Something went wrong");
    expect(container.textContent).toContain("Render smoke signal");
  });

  const wizardModels = [
    {
      id: "z_image_turbo",
      name: "Z-Image-Turbo",
      type: "image",
      downloadable: true,
      installState: "missing",
      downloadSizeLabel: "30.6 GB",
      downloadSizeBytes: 32899667397,
      downloadSizeEstimated: true,
      downloads: [{ repo: "Tongyi-MAI/Z-Image-Turbo" }],
    },
    {
      id: "qwen_image",
      name: "Qwen Image",
      type: "image",
      downloadable: true,
      installState: "installed",
      downloadSizeLabel: "53.7 GB",
      downloadSizeBytes: 57704594653,
      downloads: [{ repo: "Qwen/Qwen-Image" }],
    },
    {
      id: "wan_2_2",
      name: "Wan2.2",
      type: "video",
      downloadable: true,
      installState: "missing",
      downloadSizeLabel: "31.8 GB",
      downloadSizeBytes: 34203021834,
      downloadSizeEstimated: true,
      downloads: [{ repo: "Wan-AI/Wan2.2-TI2V-5B" }],
    },
    {
      id: "ltx_2_3",
      name: "LTX-2.3",
      type: "video",
      downloadable: true,
      installState: "missing",
      downloadSizeLabel: "146 GB",
      downloadSizeBytes: 157004895813,
      downloads: [{ repo: "Lightricks/LTX-2.3" }],
    },
  ];

  function renderWizard(overrides = {}) {
    const props = {
      models: wizardModels,
      jobs: [],
      onDownloadModel: vi.fn(),
      onCreateProject: vi.fn(async (name) => ({ id: "project-new", name })),
      onComplete: vi.fn(async () => {}),
      onOpenQueue: vi.fn(),
      ...overrides,
    };
    root = createRoot(container);
    return props;
  }

  it("groups downloadable models, pre-checks recommended ones, and flags installed", async () => {
    const props = renderWizard();
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();

    expect(container.textContent).toContain("Image models");
    expect(container.textContent).toContain("Video models");
    expect(container.textContent).toContain("Recommended");
    expect(container.textContent).toContain("Already installed");
    expect(container.textContent).toContain("~30.6 GB");

    const checkboxes = [...container.querySelectorAll("input[type=checkbox]")];
    // DOM order: image group (z_image_turbo, qwen_image[installed]), then video (wan_2_2, ltx_2_3).
    expect(checkboxes[0].checked).toBe(true); // z_image_turbo — recommended, small enough to auto-select
    expect(checkboxes[1].disabled).toBe(true); // qwen_image — installed, not selectable
    expect(checkboxes[2].checked).toBe(false); // wan_2_2 — not recommended
    expect(checkboxes[3].checked).toBe(false); // ltx_2_3 — recommended but too large to auto-select (~146 GB)
    // LTX-2.3 is still surfaced as recommended (badge) and shows its size so the choice is informed.
    expect(container.textContent).toContain("146 GB");
    expect(container.querySelectorAll(".setup-wizard-tag").length).toBe(2); // z_image_turbo + ltx_2_3
  });

  it("auto-selects only the small recommended model, leaving huge ones opt-in", async () => {
    const props = renderWizard();
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();

    const downloadButton = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Download"),
    );
    expect(downloadButton.textContent).toContain("1");
    await act(async () => {
      downloadButton.click();
    });
    await settle();

    // Only the pre-checked image model downloads; LTX-2.3 stays opt-in.
    expect(props.onDownloadModel).toHaveBeenCalledTimes(1);
    expect(props.onDownloadModel.mock.calls[0][0].id).toBe("z_image_turbo");
    // Re-firing is guarded: the button disables once nothing is pending.
    expect(downloadButton.disabled).toBe(true);
  });

  it("advances to the project step and creates a project then marks setup complete", async () => {
    const props = renderWizard();
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Continue").click();
    });
    await settle();
    expect(container.textContent).toContain("Create your first project");
    // Skipping downloads is allowed — Continue advanced without firing any.
    expect(props.onDownloadModel).not.toHaveBeenCalled();

    const input = container.querySelector("input[type=text]") ?? container.querySelector("input");
    await changeField(input, "My First Project");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Finish setup").click();
    });
    await settle();

    expect(props.onCreateProject).toHaveBeenCalledWith("My First Project");
    expect(props.onComplete).toHaveBeenCalledTimes(1);
  });

  it("does not mark setup complete when project creation fails", async () => {
    const props = renderWizard({ onCreateProject: vi.fn(async () => null) });
    await act(async () => {
      root.render(<SetupWizard {...props} />);
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Continue").click();
    });
    await settle();
    const input = container.querySelector("input");
    await changeField(input, "Doomed Project");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Finish setup").click();
    });
    await settle();

    expect(props.onCreateProject).toHaveBeenCalledWith("Doomed Project");
    expect(props.onComplete).not.toHaveBeenCalled();
  });

  it("renders the app navigation against mocked API calls", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(container.textContent).toContain("Assets");
    expect(navLabels(container, "Workspace")).not.toContain("Library");
    expect(navLabels(container, "Library")).toContain("Assets");
    expect(container.textContent).toContain("Train");
    expect(container.textContent).toContain("Queue");
  });

  it("gates the studios behind workspace creation when no projects exist", async () => {
    const requests = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      requests.push({ path, method: options.method ?? "GET" });
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects") && options.method === "POST") {
        return Promise.resolve(response({ id: "project-new", name: "My First Project" }));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    // With zero workspaces the studio area is replaced by the create gate.
    expect(container.textContent).toContain("Create your first workspace");

    await changeField(container.querySelector('[aria-label="Workspace name"]'), "My First Project");
    await act(async () => {
      buttonInside(container, "Create workspace").click();
    });
    await settle();

    // Creating the first workspace clears the gate and lands in a studio.
    expect(requests.some((request) => request.path.endsWith("/projects") && request.method === "POST")).toBe(true);
    expect(container.textContent).not.toContain("Create your first workspace");
  });

  it("opens the Train navigation item without exposing a queue action", async () => {
    global.fetch = vi.fn((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-a", name: "Project A" }]));
      }
      if (path.includes("/training/datasets")) {
        return Promise.resolve(response([{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 2 }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Train").click();
    });
    await settle();

    expect(container.textContent).toContain("Training Studio");
    expect(container.textContent).toContain("Configure Job");
    expect(container.textContent).toContain("Data Sets");
    expect(container.textContent).not.toContain("Rename & Caption");
    expect([...container.querySelectorAll("button")].some((button) => /queue training/i.test(button.textContent))).toBe(false);
  });

  it("keeps Training Studio focused on selecting existing datasets", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 2 }],
          onRefreshDatasets: () => {},
        }),
      );
    });

    expect(container.querySelector("#training-tab-configure").getAttribute("aria-selected")).toBe("true");
    expect(container.textContent).toContain("A dry run validates the Rust-resolved training plan");
    expect(container.querySelector("#training-tab-dataset")).toBeNull();
    expect(container.querySelector("#training-tab-rename-caption")).toBeNull();
    expect(container.textContent).not.toContain("Import images & captions");
  });

  it("creates a training dataset from selected image assets", async () => {
    const createDataset = vi.fn(async (payload) => ({
      id: "dataset-new",
      name: payload.name,
      version: 1,
      items: payload.items.map((item) => ({ ...item, caption: { text: "", triggerWords: [] } })),
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          createDataset,
          datasets: [],
        }),
      );
    });

    await changeField(field(container, "Dataset name"), "Mira Set");
    // Add the image via the Asset Library tab of the add dialog.
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    await act(async () => {
      [...container.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Asset Library").click();
    });
    await act(async () => {
      container.querySelector(".dataset-add-card").click();
    });
    await act(async () => {
      container.querySelector(".dataset-add-footer button.primary-action").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Create dataset").click();
    });

    expect(createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Mira Set",
        modality: "image",
        items: [expect.objectContaining({ assetId: "asset-a", displayName: "Mira.png" })],
      }),
    );
    expect(container.textContent).toContain("Dataset created");
  });

  it("imports caption sidecars alongside images and bakes them into the saved dataset", async () => {
    const createDataset = vi.fn(async (payload) => ({
      id: "dataset-new",
      name: payload.name,
      version: 1,
      items: payload.items,
    }));
    const uploadDatasetItem = vi.fn(async (file) => ({
      id: "dataset-upload-mira",
      datasetOnly: true,
      displayName: file.name,
      file: { path: "training/uploads/mira.png", mimeType: "image/png" },
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-mira", type: "image", displayName: "mira.png", file: { path: "assets/images/mira.png", mimeType: "image/png" } }],
          createDataset,
          datasets: [],
          uploadDatasetItem,
        }),
      );
    });

    const imageFile = new File([new Uint8Array([1, 2, 3])], "mira.png", { type: "image/png" });
    const captionFile = new File(["a portrait of mira"], "mira.txt", { type: "text/plain" });
    // Import via the File tab of the add dialog (default tab).
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    const fileInput = container.querySelector(".dataset-add-dropzone input[type=file]");
    await act(async () => {
      Object.defineProperty(fileInput, "files", { configurable: true, value: [imageFile, captionFile] });
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await settle();

    // Only the image is uploaded to dataset-owned staging; the .txt is parsed locally.
    expect(uploadDatasetItem).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("Imported 1 image with 1 caption");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Close").click();
    });
    await changeField(field(container, "Dataset name"), "Mira Set");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Create dataset").click();
    });

    expect(createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Mira Set",
        items: [
          expect.objectContaining({
            path: "training/uploads/mira.png",
            caption: expect.objectContaining({ text: "a portrait of mira", source: "imported" }),
          }),
        ],
      }),
    );
  });

  it("opens and saves an existing training dataset membership", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [{ assetId: "asset-a", displayName: "Mira.png", caption: { text: "mira portrait", triggerWords: [] } }],
    }));
    const updateDataset = vi.fn(async (datasetId, payload) => ({
      id: datasetId,
      name: payload.name,
      version: 4,
      items: payload.items.map((item) => ({ ...item, caption: item.caption ?? { text: "", triggerWords: [] } })),
    }));
    const assets = [
      { id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } },
      { id: "asset-b", type: "image", displayName: "Mira close.png", file: { path: "assets/images/Mira-close.png", mimeType: "image/png" } },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset,
          updateDataset,
        }),
      );
    });

    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();
    expect(loadDataset).toHaveBeenCalledWith("dataset-a");
    // The editor body shows only the dataset's own member (asset-a), not all assets.
    expect(container.querySelectorAll(".training-caption-card")).toHaveLength(1);
    expect(container.querySelector(".training-caption-grid").textContent).toContain("Mira.png");

    // Add the second asset through the Asset Library tab (current members excluded).
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });
    await act(async () => {
      [...container.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Asset Library").click();
    });
    await act(async () => {
      container.querySelector(".dataset-add-card").click();
    });
    await act(async () => {
      container.querySelector(".dataset-add-footer button.primary-action").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").click();
    });

    expect(updateDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        name: "Portrait Set",
        items: [
          expect.objectContaining({ assetId: "asset-a" }),
          expect.objectContaining({ assetId: "asset-b" }),
        ],
      }),
    );
    expect(container.textContent).toContain("Dataset changes saved");
  });

  it("scopes the add dialog: Library excludes Character Studio outputs, Character tab pulls them in (sc-2026)", async () => {
    const createDataset = vi.fn(async (payload) => ({ id: "dataset-new", name: payload.name, version: 1, items: payload.items }));
    const assets = [
      {
        id: "asset-lib",
        type: "image",
        displayName: "Studio render.png",
        origin: "image_studio",
        file: { path: "assets/images/studio.png", mimeType: "image/png" },
      },
      {
        id: "asset-char",
        type: "image",
        displayName: "Kelsie hero.png",
        origin: "character_studio",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        file: { path: "assets/images/kelsie.png", mimeType: "image/png" },
      },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets,
          characters: [{ id: "char-1", name: "Kelsie" }],
          createDataset,
          datasets: [],
        }),
      );
    });

    await changeField(field(container, "Dataset name"), "Kelsie Set");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add images").click();
    });

    // Asset Library tab is scoped: the Character Studio output is hidden.
    await act(async () => {
      [...container.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Asset Library").click();
    });
    const libraryCards = [...container.querySelectorAll(".dataset-add-card")];
    expect(libraryCards).toHaveLength(1);
    expect(libraryCards[0].textContent).toContain("Studio render.png");

    // Character tab intentionally surfaces the character's image (its character_studio output).
    await act(async () => {
      [...container.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === "Character").click();
    });
    const characterCards = [...container.querySelectorAll(".dataset-add-card")];
    expect(characterCards).toHaveLength(1);
    expect(characterCards[0].textContent).toContain("Kelsie hero.png");
    await act(async () => {
      characterCards[0].click();
    });
    await act(async () => {
      container.querySelector(".dataset-add-footer button.primary-action").click();
    });

    // The editor body is the member grid only — no all-asset picker remains.
    expect(container.querySelector(".training-asset-picker")).toBeNull();
    expect(container.querySelector(".training-caption-grid").textContent).toContain("Kelsie hero.png");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Create dataset").click();
    });
    // Importing from the Character tab associates the dataset with that
    // character (sc-2022).
    expect(createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        characterId: "char-1",
        items: [expect.objectContaining({ assetId: "asset-char" })],
      }),
    );
  });

  it("shows a cover thumbnail per dataset and a New dataset item in the selector (sc-2025)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [],
          datasets: [
            { id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 3, coverPath: "training/datasets/dataset-a/images/item_0001.png" },
          ],
        }),
      );
    });

    // Pill shows the New-dataset draft placeholder before anything is opened.
    expect(container.querySelector(".compact-selector-pill").textContent).toContain("New dataset");

    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });

    // A "New dataset" create item sits at the top of the dropdown.
    expect(container.querySelector(".compact-selector-create").textContent).toContain("New dataset");

    // Every dataset row renders its server-provided cover thumbnail.
    const datasetItem = [...container.querySelectorAll(".compact-selector-item")].find((button) =>
      button.textContent.includes("Portrait Set"),
    );
    const cover = datasetItem.querySelector("img");
    expect(cover).not.toBeNull();
    expect(cover.getAttribute("src")).toContain("training/datasets/dataset-a/images/item_0001.png");
  });

  it("lets users remove unavailable dataset assets before saving", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [
        { assetId: "asset-a", displayName: "Mira.png", caption: { text: "mira portrait", triggerWords: [] } },
        { assetId: "asset-missing", displayName: "Missing.png", caption: { text: "missing portrait", triggerWords: [] } },
      ],
    }));
    const updateDataset = vi.fn(async (datasetId, payload) => ({
      id: datasetId,
      name: payload.name,
      version: 4,
      items: payload.items,
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 2 }],
          loadDataset,
          updateDataset,
        }),
      );
    });

    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();

    expect(container.textContent).toContain("Asset is no longer available");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").disabled).toBe(true);

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Remove").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").click();
    });

    expect(updateDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        items: [expect.objectContaining({ assetId: "asset-a" })],
      }),
    );
  });

  it("does not save unchanged existing datasets", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [{ assetId: "asset-a", displayName: "Mira.png", caption: { text: "mira portrait", triggerWords: [] } }],
    }));
    const updateDataset = vi.fn();

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset,
          updateDataset,
        }),
      );
    });

    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();

    const saveButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save dataset");
    expect(saveButton.disabled).toBe(true);
    expect(updateDataset).not.toHaveBeenCalled();
  });

  // Open the lone "Portrait Set" dataset through the compact selector.
  async function openPortraitSet() {
    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Portrait Set")).click();
    });
    await settle();
  }

  function singleItemDataset() {
    return {
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: [] },
        },
      ],
    };
  }

  it("edits a caption inline and saves it with the dataset (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: payload.items }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    const caption = container.querySelector(".training-caption-card-text");
    expect(caption.value).toBe("mira portrait");
    await changeField(caption, "mira studio portrait");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Save dataset").click();
    });

    expect(updateDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        items: [expect.objectContaining({ assetId: "asset-a", caption: expect.objectContaining({ text: "mira studio portrait", source: "manual" }) })],
      }),
    );
  });

  it("queues a caption job for all images via the caption dialog (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: singleItemDataset().items }));
    const createCaptionJob = vi.fn(async () => ({ id: "job-caption-1", type: "training_caption" }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          createCaptionJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Caption all").click();
    });
    // The dialog prefills the character name from the dataset.
    expect(field(container, "Character name").value).toBe("Portrait Set");
    await act(async () => {
      [...container.querySelectorAll(".dataset-caption-footer button")].find((button) => button.textContent.startsWith("Caption")).click();
    });
    await settle();

    expect(createCaptionJob).toHaveBeenCalledWith("dataset-a", expect.objectContaining({ captioner: "joy_caption" }));
    expect(createCaptionJob.mock.calls[0][1].itemIds).toBeUndefined();
    expect(container.textContent).toContain("Caption job queued (job-caption-1)");
  });

  it("re-captions a single image with the itemIds filter (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: singleItemDataset().items }));
    const createCaptionJob = vi.fn(async () => ({ id: "job-caption-2", type: "training_caption" }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          createCaptionJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Re-Caption").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".dataset-caption-footer button")].find((button) => button.textContent.startsWith("Re-caption")).click();
    });
    await settle();

    expect(createCaptionJob).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({ itemIds: ["item_0001"], recaption: true }),
    );
  });

  it("applies ordered names to the dataset from the toolbar (sc-2025)", async () => {
    const updateDataset = vi.fn(async (datasetId, payload) => ({ id: datasetId, name: payload.name, version: 4, items: singleItemDataset().items }));
    const batchRenameDataset = vi.fn(async (datasetId) => ({ ...singleItemDataset(), id: datasetId, version: 5 }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingDataSetsLibraryContext({
          activeProject: { id: "project-a", name: "Project A" },
          assets: [{ id: "asset-a", type: "image", displayName: "Mira.png", file: { path: "assets/images/Mira.png", mimeType: "image/png" } }],
          batchRenameDataset,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset: vi.fn(async () => singleItemDataset()),
          updateDataset,
        }),
      );
    });
    await openPortraitSet();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Apply ordered names")).click();
    });
    await settle();

    expect(batchRenameDataset).toHaveBeenCalledWith(
      "dataset-a",
      expect.objectContaining({
        items: [expect.objectContaining({ itemId: "item_0001", newItemId: "portrait_set_0001", fileStem: "portrait_set_0001" })],
      }),
    );
  });

  it("defaults the training trigger phrase from the selected dataset name until edited", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 3,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "first portrait", source: "manual", triggerWords: ["oldOne"] },
        },
      ],
    }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });

    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    expect(field(container, "Trigger phrase").value).toBe("Portrait Set");

    await changeField(field(container, "Trigger phrase"), "manualTrigger");
    expect(container.querySelector("#training-tab-rename-caption")).toBeNull();
    expect(field(container, "Trigger phrase").value).toBe("manualTrigger");
  });

  it("shows active training progress with live sample previews", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          jobs: [
            {
              id: "job-train-1",
              type: "lora_train",
              status: "running",
              stage: "rendering",
              progress: 0.42,
              elapsedSeconds: 31,
              projectId: "project-a",
              requestedGpu: "0",
              payload: { outputName: "Portrait Set LoRA" },
              result: {
                latestTrainingSamples: [
                  { step: 250, prompt: "Portrait Set, studio portrait", relativePath: "loras/lora_1/samples/sample-1.png" },
                  { step: 250, prompt: "Portrait Set, full body", relativePath: "loras/lora_1/samples/sample-2.png" },
                  { step: 250, prompt: "Portrait Set, outdoor", relativePath: "loras/lora_1/samples/sample-3.png" },
                  { step: 250, prompt: "Portrait Set, close-up", relativePath: "loras/lora_1/samples/sample-4.png" },
                ],
              },
            },
          ],
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });

    expect(container.textContent).toContain("Training in progress");
    expect(container.textContent).toContain("Portrait Set LoRA");
    // The unified WorkerProgressCard renders the stage with title-case via
    // defaultChipLabel ("rendering" -> "Rendering"); same content, different style.
    expect(container.textContent).toContain("Rendering");
    expect([...container.querySelectorAll(".worker-progress-card__thumb-media")].map((image) => image.src)).toEqual([
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-1.png",
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-2.png",
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-3.png",
      "http://localhost:8000/api/v1/projects/project-a/files/loras/lora_1/samples/sample-4.png",
    ]);
  });

  it("builds a training config snapshot from registry defaults", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          createTrainingJob,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");

    expect(field(container, "Target").value).toBe("z_image_turbo_lora");
    expect(field(container, "Base model").value).toBe("z_image_turbo");
    expect(field(container, "Guidance scale").value).toBe("0");
    expect(field(container, "Rank").value).toBe("16");
    expect(field(container, "Precision").value).toBe("bf16");
    expect([...field(container, "Optimizer").options].map((option) => option.value)).toEqual(["adamw8bit", "adamw", "adam", "prodigyopt", "rose"]);
    await changeField(field(container, "Optimizer"), "prodigyopt");
    await changeField(field(container, "Guidance scale"), "1.2");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        targetId: "z_image_turbo_lora",
        datasetId: "dataset-a",
        datasetVersion: 5,
        outputName: "Portrait Set LoRA",
        dryRun: true,
        config: expect.objectContaining({
          rank: 16,
          alpha: 16,
          learningRate: 0.0001,
          optimizer: "prodigyopt",
          triggerWord: "miraStyle",
          advanced: expect.objectContaining({
            mixedPrecision: "bf16",
            qualityPreset: "balanced",
            outputScope: "project",
            requestedGpu: "auto",
            sampleSteps: 8,
            sampleGuidanceScale: 1.2,
            samplePrompts: expect.arrayContaining([expect.stringContaining("miraStyle")]),
          }),
        }),
      }),
    );
    expect(container.textContent).toContain("Queued dry-run job");
  });

  it("applies training presets and includes selected preset metadata", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingPresets: zImageTrainingPresets,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();

    expect(field(container, "Preset").value).toBe("z_image_turbo_lora.character.adamw8bit.balanced");
    expect(container.textContent).toContain("Character balanced");
    await changeField(field(container, "Optimizer"), "prodigyopt");
    expect(field(container, "Preset").value).toBe("z_image_turbo_lora.character.prodigyopt.balanced");
    expect(field(container, "Learning rate").value).toBe("1");
    expect(field(container, "Sample cadence").value).toBe("200");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        presetId: "z_image_turbo_lora.character.prodigyopt.balanced",
        presetVersion: 1,
        config: expect.objectContaining({
          learningRate: 1,
          optimizer: "prodigyopt",
          steps: 1600,
          advanced: expect.objectContaining({ sampleEvery: 200 }),
        }),
      }),
    );
  });

  it("submits the selected de-distill adapter version for Z-Image-Turbo", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingPresets: zImageTrainingPresets,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();

    // The selector appears for Z-Image-Turbo and normalizes the preset's legacy
    // "v2-default" value to "v2".
    const adapterSelect = field(container, "De-distill adapter");
    expect(adapterSelect).toBeTruthy();
    expect(adapterSelect.value).toBe("v2");

    await changeField(adapterSelect, "v1");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    // The repo + chosen version must reach config.advanced — the worker only fuses
    // the de-distill adapter when trainingAdapterRepo is present.
    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          advanced: expect.objectContaining({
            trainingAdapterRepo: "ostris/zimage_turbo_training_adapter",
            trainingAdapterVersion: "v1",
          }),
        }),
      }),
    );
  });

  it("marks manual training preset edits as customizations", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          trainingPresets: zImageTrainingPresets,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Rank"), "24");

    expect(container.textContent).toContain("Customized: Rank");
  });

  it("queues a dry-run training job from the configure tab", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_dryrun_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        targetId: "z_image_turbo_lora",
        datasetId: "dataset-a",
        datasetVersion: 5,
        outputName: "Portrait Set LoRA",
        dryRun: true,
        config: expect.objectContaining({ rank: 16, triggerWord: "miraStyle" }),
      }),
    );
    expect(container.textContent).toContain("Queued dry-run job job_dryrun_1");
  });

  it("queues a real training job when run mode is set to training", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn(async () => ({ id: "job_train_1", type: "lora_train", status: "queued" }));

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          createTrainingJob,
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions: ["auto", "0"],
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();

    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");
    await changeField(field(container, "Run mode"), "real");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Start training").click();
    });
    await settle();

    expect(createTrainingJob).toHaveBeenCalledWith(
      expect.objectContaining({
        targetId: "z_image_turbo_lora",
        datasetId: "dataset-a",
        outputName: "Portrait Set LoRA",
        dryRun: false,
        config: expect.objectContaining({ rank: 16, triggerWord: "miraStyle" }),
      }),
    );
    expect(container.textContent).toContain("Queued training job job_train_1");
  });

  it("keeps config edits when GPU options are recomputed", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));

    function render(gpuOptions) {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          gpuOptions,
          loadDataset,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    }

    root = createRoot(container);
    await act(async () => {
      render(["auto", "0"]);
    });
    await settle();
    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });
    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");
    await changeField(field(container, "Rank"), "24");
    await changeField(field(container, "Requested GPU"), "0");

    await act(async () => {
      render(["auto", "0"]);
    });
    await settle();

    expect(field(container, "Trigger phrase").value).toBe("miraStyle");
    expect(field(container, "Rank").value).toBe("24");
    expect(field(container, "Requested GPU").value).toBe("0");

    await act(async () => {
      render(["auto"]);
    });
    await settle();

    expect(field(container, "Trigger phrase").value).toBe("miraStyle");
    expect(field(container, "Rank").value).toBe("24");
    expect(field(container, "Requested GPU").value).toBe("auto");
  });

  it("blocks job submission until required fields are valid", async () => {
    const loadDataset = vi.fn(async () => ({
      id: "dataset-a",
      name: "Portrait Set",
      version: 5,
      items: [
        {
          id: "item_0001",
          assetId: "asset-a",
          path: "images/item_0001.png",
          displayName: "item_0001.png",
          caption: { text: "mira portrait", source: "manual", triggerWords: ["mira"] },
        },
      ],
    }));
    const createTrainingJob = vi.fn();

    root = createRoot(container);
    await act(async () => {
      root.render(
        withTrainingStudioContext({
          activeProject: { id: "project-a", name: "Project A" },
          datasets: [{ id: "dataset-a", name: "Portrait Set", modality: "image", itemCount: 1 }],
          loadDataset,
          createTrainingJob,
          trainingTargets: [zImageTrainingTarget],
        }),
      );
    });
    await settle();
    await act(async () => {
      container.querySelector("#training-tab-configure").click();
    });

    expect(container.textContent).toContain("Select a saved dataset");
    expect(container.textContent).toContain("Add a trigger phrase");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job").disabled).toBe(true);

    await changeField(field(container, "Dataset"), "dataset-a");
    await settle();
    await changeField(field(container, "Trigger phrase"), "miraStyle");
    await changeField(field(container, "Checkpoint cadence"), "");

    const submitButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue dry-run job");
    expect(container.textContent).toContain("Checkpoint cadence must be greater than zero");
    expect(submitButton.disabled).toBe(true);

    await act(async () => {
      submitButton.click();
    });
    await settle();

    expect(createTrainingJob).not.toHaveBeenCalled();
  });

  it("selects duplicate-titled assets through the thumbnail asset picker", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Shot", createdAt: "2026-05-19T09:00:00Z", recipe: { mode: "text_to_image" } },
      { id: "image-beta", type: "image", displayName: "Shot", createdAt: "2026-05-19T09:05:00Z", recipe: { mode: "edit_image" } },
      { id: "clip-gamma", type: "video", displayName: "Shot", createdAt: "2026-05-19T09:10:00Z", file: { mimeType: "video/mp4" } },
      { id: "upload-delta", type: "upload", displayName: "Plate", createdAt: "2026-05-19T09:15:00Z" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        <AssetPickerField
          assets={assets}
          buttonLabel="Select image"
          emptyLabel="No source image selected"
          label="Source"
          onChange={onChange}
          value=""
        />,
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Select image").click();
    });

    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.textContent).toContain("Images 2");
    expect(container.textContent).toContain("Video 1");
    expect(container.textContent).toContain("Uploads 1");
    expect(container.textContent).toContain("Renders 2");

    await act(async () => {
      [...container.querySelectorAll(".asset-picker-toolbar button")].find((button) => button.textContent.includes("Video")).click();
    });

    expect(container.querySelectorAll(".asset-picker-card")).toHaveLength(1);
    expect(container.querySelector('[title="clip-gamma"]')).not.toBeNull();

    await act(async () => {
      [...container.querySelectorAll(".asset-picker-toolbar button")].find((button) => button.textContent.includes("All")).click();
    });
    await changeField(container.querySelector('[aria-label="Search assets"]'), "plate");

    expect(container.querySelectorAll(".asset-picker-card")).toHaveLength(1);
    expect(container.textContent).toContain("Plate");

    await act(async () => {
      container.querySelector(".modal-backdrop").dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    });

    expect(container.querySelector('[role="dialog"]')).toBeNull();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Select image").click();
    });

    const cards = [...container.querySelectorAll(".asset-picker-card")];
    await act(async () => {
      cards[1].click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });

    expect(onChange).toHaveBeenCalledWith("image-beta");

    await act(async () => {
      root.render(
        <AssetPickerField
          assets={assets}
          buttonLabel="Select image"
          emptyLabel="No source image selected"
          label="Source"
          onChange={onChange}
          value="image-beta"
        />,
      );
    });

    expect(container.textContent).toContain("image-beta".slice(-6));

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Change").click();
    });
    await act(async () => {
      container.querySelector('[role="dialog"]').dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });

    expect(container.querySelector('[role="dialog"]')).toBeNull();
  });

  it("dismisses FullscreenPreview via Escape and backdrop click", async () => {
    const onClose = vi.fn();
    const noop = () => {};
    const asset = {
      id: "asset-a",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/plate.png" },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={null}
          onClose={onClose}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    expect(container.querySelector('[role="dialog"]')).not.toBeNull();

    await act(async () => {
      container.querySelector('[role="dialog"]').dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);

    await act(async () => {
      container.querySelector(".modal-backdrop").dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("toggles FullscreenPreview between original and upscaled variants", async () => {
    const noop = () => {};
    const original = {
      id: "asset-original",
      projectId: "project-1",
      displayName: "Plate",
      type: "image",
      status: {},
      file: { path: "assets/images/original.png" },
    };
    const upscaled = {
      id: "asset-upscaled",
      projectId: "project-1",
      displayName: "Plate (2x upscaled)",
      type: "image",
      status: {},
      file: { path: "assets/images/upscaled.png" },
      lineage: { sourceAssetId: "asset-original", parents: ["asset-original"] },
      extra: { isUpscaled: true, upscaledFromAssetId: "asset-original", factor: 2, engine: "real-esrgan" },
      variants: { original, upscaled: null },
    };
    upscaled.variants.upscaled = upscaled;

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={upscaled}
          deleteAsset={noop}
          nextAsset={null}
          onClose={noop}
          onPreviewAsset={noop}
          previousAsset={null}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    expect(container.querySelector(".preview-modal img").getAttribute("src")).toContain("upscaled.png");
    expect(container.textContent).toContain("Original");
    expect(container.textContent).toContain("Upscaled");

    await act(async () => {
      [...container.querySelectorAll(".preview-variant-toggle button")].find((button) => button.textContent === "Original").click();
    });

    expect(container.querySelector(".preview-modal img").getAttribute("src")).toContain("original.png");
  });

  it("reports the scroll direction when navigating the FullscreenPreview", async () => {
    const noop = () => {};
    const onPreviewAsset = vi.fn();
    const asset = { id: "asset-b", displayName: "Plate", type: "image", status: {}, file: { path: "b.png" } };
    const previous = { id: "asset-a", displayName: "Prev", type: "image", status: {}, file: { path: "a.png" } };
    const next = { id: "asset-c", displayName: "Next", type: "image", status: {}, file: { path: "c.png" } };

    root = createRoot(container);
    await act(async () => {
      root.render(
        <FullscreenPreview
          asset={asset}
          deleteAsset={noop}
          nextAsset={next}
          onClose={noop}
          onPreviewAsset={onPreviewAsset}
          previousAsset={previous}
          purgeAsset={noop}
          updateAssetStatus={noop}
        />,
      );
    });

    await act(async () => {
      container.querySelector(".preview-nav-button.next").click();
    });
    expect(onPreviewAsset).toHaveBeenLastCalledWith(next, "next");

    await act(async () => {
      container.querySelector(".preview-nav-button.previous").click();
    });
    expect(onPreviewAsset).toHaveBeenLastCalledWith(previous, "previous");
  });

  it("keeps in-progress picker selection across parent rerenders", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha" },
      { id: "image-beta", type: "image", displayName: "Beta" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(<AssetPickerField assets={assets} label="Source" onChange={onChange} value="image-alpha" />);
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Change").click();
    });
    await act(async () => {
      container.querySelectorAll(".asset-picker-card")[1].click();
    });
    await act(async () => {
      root.render(<AssetPickerField assets={[...assets]} label="Source" onChange={onChange} value="image-alpha" />);
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });

    expect(onChange).toHaveBeenCalledWith("image-beta");
  });

  it("toggles and confirms multiple assets through the thumbnail picker", async () => {
    const onChange = vi.fn();
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha" },
      { id: "image-beta", type: "image", displayName: "Beta" },
      { id: "image-gamma", type: "image", displayName: "Gamma" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(<AssetPickerField assets={assets} label="Reference assets" multiple onChange={onChange} values={[]} />);
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Select").click();
    });

    const cards = [...container.querySelectorAll(".asset-picker-card")];
    await act(async () => {
      cards[0].click();
      cards[1].click();
      cards[0].click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });

    expect(onChange).toHaveBeenCalledWith(["image-beta"]);

    await act(async () => {
      root.render(<AssetPickerField assets={assets} label="Reference assets" multiple onChange={onChange} values={["image-beta"]} />);
    });

    expect(container.querySelectorAll(".asset-preview-chip")).toHaveLength(1);
    expect(container.textContent).toContain("Beta");
  });

  it("keeps unsaved character reference selections when a multi-add partially fails", async () => {
    const addCharacterReference = vi.fn(async (_characterId, reference) => {
      if (reference.assetId === "image-beta") {
        throw new Error("network hiccup");
      }
      return {};
    });
    const assets = [
      { id: "image-alpha", type: "image", displayName: "Alpha" },
      { id: "image-beta", type: "image", displayName: "Beta" },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference,
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add image or frame").click();
    });

    const cards = [...container.querySelectorAll(".asset-picker-card")];
    await act(async () => {
      cards[0].click();
      cards[1].click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Use Selection").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add").click();
    });

    expect(addCharacterReference).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("Added 1 reference");
    expect(container.textContent).toContain("network hiccup");
    expect(container.querySelectorAll(".asset-preview-chip")).toHaveLength(1);
    expect(container.textContent).toContain("Beta");
  });

  it("switches the active character via the compact selector (sc-2025)", async () => {
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        { id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
        { id: "char-2", name: "Dax", type: "person", references: [], approvedReferences: [], looks: [], loras: [] },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // The full-height character list is replaced by a compact selector pill.
    expect(container.querySelector(".character-list")).toBeNull();
    const pill = container.querySelector(".compact-selector-pill");
    expect(pill.textContent).toContain("Mira");
    expect(field(container, "Name").value).toBe("Mira");

    // Open the dropdown and switch to the second character.
    await act(async () => {
      pill.click();
    });
    await act(async () => {
      [...container.querySelectorAll(".compact-selector-item")].find((button) => button.textContent.includes("Dax")).click();
    });

    expect(field(container, "Name").value).toBe("Dax");
    expect(container.querySelector(".compact-selector-pill").textContent).toContain("Dax");
  });

  it("creates a character from the selector's New item (sc-2025)", async () => {
    const createCharacter = vi.fn(async (payload) => ({
      id: "char-new",
      ...payload,
      references: [],
      approvedReferences: [],
      looks: [],
      loras: [],
    }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
      createCharacter,
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // No inline header create form anymore — creation lives in the dropdown.
    expect([...container.querySelectorAll("input")].some((input) => input.getAttribute("aria-label") === "Character name")).toBe(false);

    await act(async () => {
      container.querySelector(".compact-selector-pill").click();
    });
    await act(async () => {
      container.querySelector(".compact-selector-create").click();
    });

    expect(createCharacter).toHaveBeenCalledWith(expect.objectContaining({ name: "New character", type: "person" }));
  });

  it("fires a one-click angle-set batch job from the Character Studio panel", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-angle" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [
        {
          id: "instantid_realvisxl",
          name: "InstantID (RealVisXL)",
          type: "image",
          ui: { viewAngles: [{ id: "front", label: "Front" }, { id: "left_profile", label: "Left profile" }, { id: "up", label: "Looking up" }] },
        },
      ],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });

    // The angle-set panel renders with the model's angle count in the button.
    const generateButton = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate angle set"),
    );
    expect(generateButton).toBeTruthy();
    expect(generateButton.textContent).toContain("3 views");

    await act(async () => {
      generateButton.click();
    });

    // One batch job with a valid count (worker expands to all pack angles) +
    // advanced.angleSet; the job is tracked for live in-panel progress.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        model: "instantid_realvisxl",
        characterId: "char-1",
        referenceAssetId: "ref-1",
        count: 1,
        advanced: expect.objectContaining({ angleSet: true, ipAdapterScale: 0.8 }),
      }),
    );
    expect(baseContext.rememberLocalGenerationJob).toHaveBeenCalledWith("image", { id: "job-angle" });
  });

  it("fires a pose-library batch job from the Character Studio pose picker", async () => {
    const poseKeypoints = Array.from({ length: 18 }, (_, i) => [0.5, i / 18]);
    global.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        version: 1,
        categories: ["standing"],
        poses: [
          { id: "standing_01", category: "standing", label: "Standing 01", preview: "poses/standing_01.png", keypoints: poseKeypoints },
        ],
      }),
    }));
    const createImageJob = vi.fn(async () => ({ id: "job-pose" }));
    const baseContext = {
      activeProject: { id: "project-1", name: "Noir" },
      addCharacterReference: () => {},
      archiveCharacter: () => {},
      assets: [],
      attachCharacterLora: () => {},
      characters: [
        {
          id: "char-1",
          name: "Mira",
          type: "person",
          references: [],
          approvedReferences: [{ assetId: "ref-1", role: "hero", asset: { id: "ref-1", type: "image", displayName: "Mira ref" } }],
          looks: [],
          loras: [],
        },
      ],
      createCharacter: () => {},
      createCharacterLook: () => {},
      createCharacterTestJob: () => {},
      createImageJob,
      importAsset: vi.fn(),
      imageLocalJobs: [],
      rememberLocalGenerationJob: vi.fn(),
      deleteAsset: () => {},
      deleteCharacterLook: () => {},
      detachCharacterLora: () => {},
      imageModels: [
        {
          id: "instantid_realvisxl",
          name: "InstantID (RealVisXL)",
          type: "image",
          ui: { poseLibrary: true },
        },
      ],
      latestImageAssets: [],
      loras: [],
      setPreviewAsset: () => {},
      sendCharacterToImage: () => {},
      sendCharacterToVideo: () => {},
      purgeAsset: () => {},
      removeCharacterReference: () => {},
      updateAssetStatus: () => {},
      updateCharacter: () => {},
      updateCharacterLook: () => {},
      updateCharacterLora: () => {},
      updateCharacterReference: () => {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(baseContext, <CharacterStudio />));
    });
    // Let the bundled pose library fetch resolve and the picker render.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    // The pose thumbnail loads from the library; select it.
    const poseButton = [...container.querySelectorAll("button")].find((button) =>
      (button.getAttribute("aria-label") ?? "").includes("pose Standing 01"),
    );
    expect(poseButton).toBeTruthy();
    await act(async () => {
      poseButton.click();
    });

    const generateButton = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.startsWith("Generate") && button.textContent.includes("pose"),
    );
    expect(generateButton).toBeTruthy();
    await act(async () => {
      generateButton.click();
    });

    // One batch job carrying the selected pose's keypoints in advanced.poses; the worker
    // emits one image per pose. count stays within the API's 1-8 guard.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        model: "instantid_realvisxl",
        characterId: "char-1",
        referenceAssetId: "ref-1",
        count: 1,
        advanced: expect.objectContaining({
          ipAdapterScale: 0.8,
          poses: [{ id: "standing_01", keypoints: poseKeypoints }],
          faceRestore: true,
        }),
      }),
    );
    expect(baseContext.rememberLocalGenerationJob).toHaveBeenCalledWith("image", { id: "job-pose" });
  });

  it("collects all character-associated assets in the Character Studio gallery (sc-2076)", async () => {
    const onPreview = vi.fn();
    const selectedCharacter = { id: "char-1", name: "Mira" };
    const assets = [
      { id: "a1", type: "image", displayName: "by recipe", recipe: { normalizedSettings: { characterId: "char-1" } } },
      { id: "a2", type: "image", displayName: "by reference", metadata: { characterReferences: [{ characterId: "char-1" }] } },
      { id: "a3", type: "image", displayName: "other character", recipe: { normalizedSettings: { characterId: "char-2" } } },
      { id: "a4", type: "image", displayName: "unassociated" },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(<CharacterAssets assets={assets} onPreview={onPreview} selectedCharacter={selectedCharacter} />);
    });

    // Counts only assets associated with this character (by recipe characterId or by
    // characterReferences) — not other characters' or unassociated assets.
    expect(container.textContent).toContain("Generated for Mira (2)");
    const previewButtons = [...container.querySelectorAll("button")].filter((button) =>
      (button.getAttribute("aria-label") ?? "").startsWith("Preview "),
    );
    expect(previewButtons).toHaveLength(2);
    await act(async () => {
      previewButtons[0].click();
    });
    expect(onPreview).toHaveBeenCalled();
  });

  it("lists a character's associated datasets and opens one (sc-2022)", async () => {
    const onOpenDataset = vi.fn();
    const onCreateDataset = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <CharacterDatasets
          datasets={[
            { id: "ds-1", name: "Mira identity set", itemCount: 12, status: "ready", characterId: "char-1" },
          ]}
          imageCount={5}
          onCreateDataset={onCreateDataset}
          onOpenDataset={onOpenDataset}
          projectId="project-1"
          selectedCharacter={{ id: "char-1", name: "Mira" }}
        />,
      );
    });

    expect(container.textContent).toContain("For Mira (1)");
    const row = container.querySelector(".character-dataset-row");
    expect(row.textContent).toContain("Mira identity set");
    expect(row.textContent).toContain("12 images · ready");

    await act(async () => {
      [...row.querySelectorAll("button")].find((button) => button.textContent === "Open").click();
    });
    expect(onOpenDataset).toHaveBeenCalledWith("ds-1");

    // The create button reflects how many of the character's images would seed it.
    const createButton = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Create dataset from 5 images"),
    );
    await act(async () => {
      createButton.click();
    });
    expect(onCreateDataset).toHaveBeenCalled();
  });

  it("creates a dataset from a character's images and opens it (sc-2022)", async () => {
    const createTrainingDataset = vi.fn(async () => ({ id: "ds-new" }));
    const openDatasetInLibrary = vi.fn();
    const assets = [
      { id: "img-1", type: "image", displayName: "hero", recipe: { normalizedSettings: { characterId: "char-1" } } },
      { id: "img-2", type: "image", displayName: "ref", metadata: { characterReferences: [{ characterId: "char-1" }] } },
      { id: "img-3", type: "image", displayName: "other", recipe: { normalizedSettings: { characterId: "char-2" } } },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            createTrainingDataset,
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [],
            loras: [],
            openDatasetInLibrary,
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            trainingDatasets: [],
            trainingDatasetsProjectId: "project-1",
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    // Two of three images belong to this character.
    const createButton = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Create dataset from 2 images"),
    );
    expect(createButton).toBeTruthy();
    await act(async () => {
      createButton.click();
    });

    expect(createTrainingDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        characterId: "char-1",
        name: "Mira dataset",
        items: [{ assetId: "img-1" }, { assetId: "img-2" }],
      }),
    );
    expect(openDatasetInLibrary).toHaveBeenCalledWith("ds-new");
  });

  it("hides discarded character images from the grid and surfaces them in the Trashcan", async () => {
    const assets = [
      { id: "img-active", type: "image", displayName: "keep", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "img-trashed",
        type: "image",
        displayName: "discarded",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    // The active toggle counts only non-trashed images for this character.
    const showButton = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Show this character's images (1)"),
    );
    expect(showButton).toBeTruthy();
    await act(async () => {
      showButton.click();
    });

    // Active grid shows the kept image, not the discarded one.
    expect(container.querySelectorAll(".review-grid .review-card").length).toBe(1);
    expect(container.querySelectorAll(".review-grid .review-card.trashed").length).toBe(0);

    // Switch to the Trashcan view and the discarded image becomes reachable.
    // Scope to the Sample-outputs panel, since CharacterAssets also has a toggle.
    const testPanel = container.querySelector(".test-character-panel");
    const trashButton = [...testPanel.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Trashcan (1)"),
    );
    expect(trashButton).toBeTruthy();
    await act(async () => {
      trashButton.click();
    });
    expect(testPanel.querySelectorAll(".review-grid .review-card.trashed").length).toBe(1);
    expect([...testPanel.querySelectorAll(".review-grid button")].some((button) => button.textContent === "Purge")).toBe(true);
  });

  it("hides discarded images from the Character assets grid and exposes Trashcan restore/purge", async () => {
    const purgeAsset = vi.fn();
    const updateAssetStatus = vi.fn();
    const assets = [
      { id: "img-active", type: "image", displayName: "keep", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "img-trashed",
        type: "image",
        displayName: "discarded",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset,
            removeCharacterReference: () => {},
            updateAssetStatus,
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    // The Character assets section renders thumbnails; the discarded one is hidden.
    const findSection = () =>
      [...container.querySelectorAll(".character-section")].find((section) =>
        section.querySelector(".eyebrow")?.textContent === "Character assets",
      );
    const section = findSection();
    expect(section).toBeTruthy();
    expect(section.querySelectorAll(".character-asset-thumb").length).toBe(1);
    // Heading count reflects active images only.
    expect(section.querySelector("h2").textContent).toContain("(1)");

    // Switch to the Trashcan and the discarded image surfaces with restore/purge.
    const trashButton = [...section.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Trashcan (1)"),
    );
    expect(trashButton).toBeTruthy();
    await act(async () => {
      trashButton.click();
    });
    const trashSection = findSection();
    expect(trashSection.querySelectorAll(".character-asset-thumb").length).toBe(1);
    const restore = [...trashSection.querySelectorAll("button")].find((button) => button.textContent === "Restore");
    const purge = [...trashSection.querySelectorAll("button")].find((button) => button.textContent === "Purge");
    expect(restore).toBeTruthy();
    expect(purge).toBeTruthy();
    await act(async () => {
      purge.click();
    });
    expect(purgeAsset).toHaveBeenCalledWith(expect.objectContaining({ id: "img-trashed" }));
  });

  it("Empty Trash purges all discarded images for the character and only in the Trashcan view", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const purgeAsset = vi.fn();
    const assets = [
      { id: "img-active", type: "image", displayName: "keep", recipe: { normalizedSettings: { characterId: "char-1" } } },
      {
        id: "img-trash-1",
        type: "image",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
      {
        id: "img-trash-2",
        type: "image",
        status: { trashed: true },
        recipe: { normalizedSettings: { characterId: "char-1" } },
      },
      // Belongs to another character — must never be purged by this view.
      { id: "img-other", type: "image", status: { trashed: true }, recipe: { normalizedSettings: { characterId: "char-2" } } },
    ];
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets,
            attachCharacterLora: () => {},
            characters: [{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: assets,
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage: () => {},
            sendCharacterToVideo: () => {},
            purgeAsset,
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    const findSection = () =>
      [...container.querySelectorAll(".character-section")].find((section) =>
        section.querySelector(".eyebrow")?.textContent === "Character assets",
      );
    // No Empty Trash in the active Images view.
    expect([...findSection().querySelectorAll("button")].some((button) => button.textContent.startsWith("Empty Trash"))).toBe(false);

    await act(async () => {
      [...findSection().querySelectorAll("button")].find((button) => button.textContent.includes("Trashcan (2)")).click();
    });
    const emptyButton = [...findSection().querySelectorAll("button")].find((button) => button.textContent.startsWith("Empty Trash"));
    expect(emptyButton).toBeTruthy();
    expect(emptyButton.textContent).toContain("(2)");

    await act(async () => {
      emptyButton.click();
    });
    expect(purgeAsset).toHaveBeenCalledTimes(2);
    expect(purgeAsset).toHaveBeenCalledWith(expect.objectContaining({ id: "img-trash-1" }));
    expect(purgeAsset).toHaveBeenCalledWith(expect.objectContaining({ id: "img-trash-2" }));
    expect(purgeAsset).not.toHaveBeenCalledWith(expect.objectContaining({ id: "img-other" }));
    confirm.mockRestore();
  });

  it("launches reference-based generation from an approved character reference", async () => {
    const sendCharacterToImage = vi.fn();
    const reference = { assetId: "ref-1", approved: true, asset: { id: "ref-1", type: "image", displayName: "Mira ref" } };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            addCharacterReference: () => {},
            archiveCharacter: () => {},
            assets: [],
            attachCharacterLora: () => {},
            characters: [
              { id: "char-1", name: "Mira", type: "person", references: [reference], approvedReferences: [reference], looks: [], loras: [] },
            ],
            createCharacter: () => {},
            createCharacterLook: () => {},
            createCharacterTestJob: () => {},
            deleteAsset: () => {},
            deleteCharacterLook: () => {},
            detachCharacterLora: () => {},
            imageModels: [],
            latestImageAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            sendCharacterToImage,
            sendCharacterToVideo: () => {},
            purgeAsset: () => {},
            removeCharacterReference: () => {},
            updateAssetStatus: () => {},
            updateCharacter: () => {},
            updateCharacterLook: () => {},
            updateCharacterLora: () => {},
            updateCharacterReference: () => {},
          },
          <CharacterStudio />,
        ),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate variations").click();
    });

    expect(sendCharacterToImage).toHaveBeenCalledWith(expect.objectContaining({ id: "char-1" }), null, "ref-1");
  });

  it("keeps the shell usable when presets are unavailable", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/recipe-presets")) {
        return Promise.resolve(errorResponse(404, "Not Found"));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(container.textContent).toContain("Library");
    expect(container.textContent).toContain("Assets");
    expect(container.textContent).toContain("Project One");
    expect(container.textContent).not.toContain("Not Found");
  });

  it("does not show a stale timeline lookup error after creating a workspace", async () => {
    const requests = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      requests.push({ method: options.method ?? "GET", path });
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects") && options.method === "POST") {
        return Promise.resolve(response({ id: "project-2", name: "Fresh Workspace" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/projects/project-1/timelines/timeline-1")) {
        return Promise.resolve(
          response({
            id: "timeline-1",
            projectId: "project-1",
            name: "Main timeline",
            aspectRatio: "16:9",
            width: 1280,
            height: 720,
            fps: 30,
            duration: 0,
            tracks: [],
            transitions: [],
          }),
        );
      }
      if (path.endsWith("/projects/project-1/timelines")) {
        return Promise.resolve(
          response([
            {
              id: "timeline-1",
              name: "Main timeline",
              filePath: "timelines/main.sceneworks.timeline.json",
              aspectRatio: "16:9",
              width: 1280,
              height: 720,
              fps: 30,
              duration: 0,
              createdAt: "2026-05-19T12:00:00Z",
              updatedAt: "2026-05-19T12:00:00Z",
            },
          ]),
        );
      }
      if (path.endsWith("/projects/project-2/timelines/timeline-1")) {
        return Promise.resolve(errorResponse(404, "Timeline not found"));
      }
      if (path.endsWith("/projects/project-2/timelines")) {
        return Promise.resolve(response([]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      container.querySelector(".project-pill").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "New workspace").click();
    });
    await changeField(container.querySelector('[aria-label="New workspace name"]'), "Fresh Workspace");
    await act(async () => {
      [...container.querySelectorAll(".project-menu-create button")].find((button) => button.textContent === "Create").click();
    });
    await settle();

    expect(requests.some((request) => request.path.endsWith("/projects/project-2/timelines/timeline-1"))).toBe(false);
    expect(container.textContent).toContain("Fresh Workspace");
    expect(container.textContent).not.toContain("Timeline not found");
  });

  it("shows the real Replace Person panel for a replacement-capable model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    // LTX-2.3 is now the primary replacement-capable model, and the placeholder
    // copy is gone in favor of the real-tracking guidance (sc-1487).
    expect(container.textContent).toContain("Real person tracking");
    expect(container.textContent).not.toContain("V1 placeholder tracking");
  });

  it("updates Replace Person readiness when a capable worker registers over SSE", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    // No live detector/tracker worker yet -> detection is gated with a reason.
    expect(container.textContent).toContain("Detection unavailable");

    // A GPU worker registers with the real person capabilities (sc-1484 finding:
    // readiness must track live worker updates, not just the initial load).
    await act(async () => {
      FakeEventSource.instances[0].listeners["worker.updated"]({
        data: JSON.stringify({
          id: "python-gpu-0",
          gpuId: "0",
          gpuName: "GPU 0",
          status: "idle",
          capabilities: ["gpu", "person_detect", "person_track", "person_segment", "person_replace"],
        }),
      });
    });
    await settle();

    // Readiness is recomputed from the live worker list, so the gate clears.
    expect(container.textContent).not.toContain("Detection unavailable");
    expect(container.textContent).not.toContain("Replacement unavailable");
  });

  it("surfaces replacement-unavailable when no live worker can run person replacement", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    // Detector/tracker are live, but no worker advertises person_replace.
    await act(async () => {
      FakeEventSource.instances[0].listeners["worker.updated"]({
        data: JSON.stringify({
          id: "python-gpu-0",
          gpuId: "0",
          gpuName: "GPU 0",
          status: "idle",
          capabilities: ["gpu", "person_detect", "person_track"],
        }),
      });
    });
    await settle();

    expect(container.textContent).not.toContain("Detection unavailable");
    // The same replace-readiness flag that gates the submit button (sc-1484 finding).
    expect(container.textContent).toContain("Replacement unavailable");
  });

  it("keeps completed Replace Person detections visible in Video Studio", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "wan_replace",
              name: "Wan Replace",
              type: "video",
              capabilities: ["replace_person", "image_to_video", "text_to_video"],
              defaults: { duration: 4, fps: 24, resolution: "1280x720", quality: "balanced" },
              limits: { durations: [4], fps: [24], resolutions: ["1280x720"] },
            },
          ]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(
          response([
            {
              id: "detect-job-1",
              type: "person_detect",
              status: "completed",
              projectId: "project-1",
              payload: { sourceAssetId: "clip-1" },
              result: {
                frameAssetId: "frame-1",
                detections: [{ id: "person-1", label: "person", confidence: 0.82, box: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 } }],
              },
              createdAt: "2026-05-18T22:00:00Z",
            },
          ]),
        );
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(
          response([
            { id: "clip-1", type: "video", displayName: "Source Clip", file: { mimeType: "video/mp4" }, status: {} },
            { id: "frame-1", type: "image", displayName: "Detection Frame", file: { mimeType: "image/png" }, status: {} },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace person").click();
    });
    await settle();

    expect(container.textContent).toContain("1 candidates");
    expect(container.textContent).not.toContain("No analysis yet");
  });

  it("keeps image generation in the studio and shows local progress", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          elapsedSeconds: 4,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      container.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    expect(container.textContent).toContain("Generate Image");
    expect(container.textContent).toContain("Running");
    expect(container.textContent).not.toContain("Jobs and GPUs");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();

    expect(container.textContent).toContain("Generate Image");
    expect(container.textContent).toContain("Running");
  });

  it("shows completed image batch items before the whole job finishes", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          elapsedSeconds: 4,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight", count: 4 },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      container.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    const partialAsset = {
      id: "asset-1",
      projectId: "project-1",
      generationSetId: "genset-1",
      type: "image",
      displayName: "Generated #1",
      file: { path: "assets/images/generated-1.png", mimeType: "image/png" },
      status: { favorite: false, rejected: false, trashed: false },
    };
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          ...createdJobs[0],
          status: "saving",
          stage: "saving",
          progress: 0.82,
          result: {
            generationSetId: "genset-1",
            assetIds: ["asset-1"],
            assets: [partialAsset],
            expectedCount: 4,
          },
        }),
      });
    });
    await settle();

    expect(container.querySelector(".worker-progress-card__thumb-media")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated-1.png",
    );
    // 1 completed asset + 3 pending slots = 4 total cells; skeletons fill the gaps.
    expect(container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBe(3);
  });

  it("reconstructs running image batch slots from partial asset records", async () => {
    const createdJobs = [];
    let currentAssets = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/assets")) {
        return Promise.resolve(response(currentAssets));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.35,
          elapsedSeconds: 4,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight", count: 4 },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      container.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    currentAssets = [
      {
        id: "asset-1",
        projectId: "project-1",
        generationSetId: "genset-1",
        type: "image",
        displayName: "Generated #1",
        file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
        status: { favorite: false, rejected: false, trashed: false },
      },
    ];
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          ...createdJobs[0],
          status: "running",
          stage: "generating",
          progress: 0.48,
          message: "Running Z-Image 2 of 4.",
          result: {
            generationSetId: "genset-1",
            assetIds: ["asset-1"],
            expectedCount: 4,
          },
        }),
      });
    });
    await settle();

    expect(container.textContent).toContain("Running Z-Image 2 of 4.");
    expect(container.querySelector(".worker-progress-card__thumb-media")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated_0001.png",
    );
    // 1 completed thumbnail + 3 pending skeleton slots = 4 total cells.
    expect(container.querySelectorAll(".worker-progress-card__thumb-cell:not(.skeleton)").length).toBe(1);
    expect(container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBe(3);
  });

  it("shows local generation failures without duplicating the global banner", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/image/jobs") && options.method === "POST") {
        const job = {
          id: "image-job-1",
          type: "image_generate",
          status: "running",
          stage: "running",
          progress: 0.25,
          elapsedSeconds: 3,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "A cinematic frame of a neon street at midnight" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();
    await act(async () => {
      container.querySelector(".image-studio form").requestSubmit();
    });
    await settle();

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({ ...createdJobs[0], status: "failed", stage: "failed", progress: 0.25, error: "Adapter crashed" }),
      });
    });
    await settle();

    expect(container.textContent).toContain("Adapter crashed");
    expect(container.textContent).not.toContain("image generate: Adapter crashed");
  });

  it("keeps video generation in the studio and shows local progress", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ]),
        );
      }
      if (path.endsWith("/video/jobs") && options.method === "POST") {
        const job = {
          id: "video-job-1",
          type: "video_generate",
          status: "queued",
          stage: "queued",
          progress: 0,
          elapsedSeconds: 0,
          projectId: "project-1",
          projectName: "Noir",
          requestedGpu: "auto",
          payload: { prompt: "Camera slowly pushes in while the scene comes alive" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Text → Video").click();
    });
    await settle();
    await act(async () => {
      container.querySelector(".video-studio form").requestSubmit();
    });
    await settle();

    expect(container.textContent).toContain("Generate Video");
    expect(container.textContent).toContain("Queued");
    expect(container.textContent).not.toContain("Jobs and GPUs");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Assets").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();

    expect(container.textContent).toContain("Generate Video");
    expect(container.textContent).toContain("Queued");
  });

  it("keeps model downloads on the Models page and shows local progress", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            {
              id: "z_image_turbo",
              name: "Z-Image Turbo",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloadSizeLabel: "12 GB",
              downloads: [{ provider: "huggingface", repo: "Tongyi-MAI/Z-Image-Turbo" }],
            },
          ]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      if (path.endsWith("/models/z_image_turbo/download") && options.method === "POST") {
        const job = {
          id: "download-job-1",
          type: "model_download",
          status: "downloading",
          stage: "downloading",
          progress: 0.5,
          elapsedSeconds: 12,
          requestedGpu: "auto",
          assignedGpu: "cpu",
          payload: { modelId: "z_image_turbo", modelName: "Z-Image Turbo" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Download 12 GB").click();
    });
    await settle();

    expect(container.textContent).toContain("Model Import");
    expect(container.textContent).toContain("Downloading");
    expect(container.textContent).not.toContain("Jobs and GPUs");
  });

  it("keeps LoRA imports on the Models page and shows local progress", async () => {
    const createdJobs = [];
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        const job = {
          id: "lora-import-job-1",
          type: "lora_import",
          status: "running",
          stage: "downloading",
          progress: 0.25,
          payload: { loraId: "detail_lora" },
        };
        createdJobs.unshift(job);
        return Promise.resolve(response(job));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    const panel = loraPanel(container);
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });
    await settle();

    expect(container.textContent).toContain("LoRA imports in progress");
    expect(container.textContent).toContain("Running");
    expect(container.textContent).not.toContain("Jobs and GPUs");
  });

  it("refreshes the project LoRA overlay when a LoRA import completes", async () => {
    global.fetch.mockImplementation((url) => {
      const parsed = new URL(url);
      const path = parsed.pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    global.fetch.mockClear();
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "completed",
          projectId: "project-1",
          payload: { loraId: "detail_lora" },
        }),
      });
    });
    await settle();
    await settle();

    const loraRequests = global.fetch.mock.calls
      .map(([url]) => new URL(url))
      .filter((url) => url.pathname.endsWith("/loras"));
    expect(loraRequests.some((url) => url.search === "")).toBe(true);
    expect(loraRequests.some((url) => url.search === "?projectId=project-1")).toBe(true);
  });

  it("refreshes the project LoRA overlay when LoRA training completes", async () => {
    global.fetch.mockImplementation((url) => {
      const parsed = new URL(url);
      const path = parsed.pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    global.fetch.mockClear();
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "train-job-1",
          type: "lora_train",
          status: "completed",
          projectId: "project-1",
          payload: { dryRun: false, outputName: "Portrait Set LoRA" },
          result: { loraRegistered: true, loraId: "lora_portrait_set" },
        }),
      });
    });
    await settle();
    await settle();

    const loraRequests = global.fetch.mock.calls
      .map(([url]) => new URL(url))
      .filter((url) => url.pathname.endsWith("/loras"));
    expect(loraRequests.some((url) => url.search === "")).toBe(true);
    expect(loraRequests.some((url) => url.search === "?projectId=project-1")).toBe(true);
  });

  it("shows the global banner for failed LoRA imports on the Models page", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(
          response([
            { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await changeField(field(container, "LoRA family"), "z-image");
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "failed",
          error: "Import worker crashed",
          payload: { loraId: "qwen_detail", family: "qwen-image" },
        }),
      });
    });
    await settle();

    expect(container.textContent).toContain("lora import: Import worker crashed");
    expect(container.textContent).toContain("1 LoRA import is hidden by this family filter.");
  });

  it("clears a stale LoRA import banner when a later import completes", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-1",
          type: "lora_import",
          status: "failed",
          createdAt: "2026-05-18T00:00:00Z",
          error: "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest",
          payload: { loraId: "detail_lora", family: "z-image" },
        }),
      });
    });
    await settle();
    expect(container.textContent).toContain("lora import: LoRA manifestPath must target");

    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({
        data: JSON.stringify({
          id: "lora-import-job-2",
          type: "lora_import",
          status: "completed",
          createdAt: "2026-05-18T00:00:01Z",
          payload: { loraId: "detail_lora", family: "z-image" },
        }),
      });
    });
    await settle();

    expect(container.textContent).not.toContain("lora import: LoRA manifestPath must target");
  });

  it("rejects oversized LoRA uploads before posting from the Models page", async () => {
    let importCalls = 0;
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        importCalls += 1;
        return Promise.resolve(response({ id: "should-not-create" }));
      }
      return Promise.resolve(response([]));
    });
    const loraFile = new File(["lora"], "too-large.safetensors", { type: "application/octet-stream" });
    Object.defineProperty(loraFile, "size", { configurable: true, value: 2 * 1024 * 1024 * 1024 + 1 });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Models").click();
    });
    await settle();
    await act(async () => {
      buttonInside(loraPanel(container), "Upload").click();
    });
    const panel = loraPanel(container);
    await changeFile(field(panel, "LoRA File"), loraFile);
    await act(async () => {
      buttonInside(loraPanel(container), "Queue Import").click();
    });

    expect(container.textContent).toContain("Uploaded LoRA file exceeds the 2GB limit");
    expect(importCalls).toBe(0);
  });

  it("keeps Preset Manager LoRA acquisition on the Models page", async () => {
    let importCalls = 0;
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Noir" }]));
      }
      if (path.endsWith("/models")) {
        return Promise.resolve(response([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]));
      }
      if (path.endsWith("/recipe-presets")) {
        return Promise.resolve(
          response([{ id: "moody", name: "Moody", scope: "global", workflow: "text_to_image", model: "z_image_turbo" }]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        importCalls += 1;
        return Promise.resolve(response({ id: "lora-import-job-1", type: "lora_import", status: "running" }));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Presets").click();
    });
    await settle();

    expect(container.textContent).toContain("Preset Manager");
    expect(container.textContent).toContain("No uploaded LoRAs yet. Manage LoRAs on the Models page.");
    expect(container.textContent).not.toContain("Import LoRA");
    expect(container.textContent).not.toContain("Queue Import");
    expect(field(container, "Source URL")).toBeUndefined();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Open Models").click();
    });
    await settle();

    expect(container.textContent).toContain("Models");
    expect(container.textContent).toContain("Import LoRA");
    expect(importCalls).toBe(0);
  });

  it("queues LoRA URL imports from the Models page", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "detail_lora" } }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    const panel = loraPanel(container);
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await changeField(field(panel, "Name"), "Detail LoRA");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceUrl: "https://example.com/loras/detail.safetensors",
        name: "Detail LoRA",
        scope: "global",
      }),
    );
    expect(onImportLora.mock.calls[0][0]).not.toHaveProperty("family");
    expect(container.textContent).toContain("LoRA import queued for detail_lora.");
  });

  it("keeps Models LoRA import family independent from the list filter", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "detail_lora" } }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [
            { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    expect(field(container, "LoRA family").value).toBe("all");
    await changeField(field(container, "LoRA family"), "qwen-image");
    const panel = loraPanel(container);
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportLora.mock.calls[0][0]).not.toHaveProperty("family");

    await changeField(field(panel, "Family"), "z-image");
    await changeField(field(panel, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportLora.mock.calls[1][0]).toEqual(
      expect.objectContaining({
        family: "z-image",
      }),
    );
  });

  it("clears an explicit Models LoRA import family when the model family disappears", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "detail_lora" } }));
    const renderScreen = (models) =>
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models,
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );

    root = createRoot(container);
    await act(async () => {
      renderScreen([
        { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
        { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
      ]);
    });

    const panel = loraPanel(container);
    await changeField(field(panel, "Family"), "qwen-image");
    expect(field(panel, "Family").value).toBe("qwen-image");

    await act(async () => {
      renderScreen([{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]);
    });

    expect(field(loraPanel(container), "Family").value).toBe("");
  });

  it("shows model download size estimates and unavailable states before download", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [
            {
              id: "z_image_turbo",
              name: "Z-Image Turbo",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloadSizeLabel: "30.6 GB",
              downloadSizeEstimated: true,
              downloads: [{ provider: "huggingface", repo: "Tongyi-MAI/Z-Image-Turbo" }],
            },
            {
              id: "local_unknown",
              name: "Unknown Size",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloads: [{ provider: "huggingface", repo: "owner/unknown" }],
            },
            {
              id: "exact_size",
              name: "Exact Size",
              type: "image",
              family: "z-image",
              downloadable: true,
              installState: "missing",
              downloadSizeLabel: "8.0 GB",
              downloadSizeEstimated: false,
              downloads: [{ provider: "huggingface", repo: "owner/exact" }],
            },
          ],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(container.textContent).toContain("Download size");
    expect(container.textContent).toContain("~30.6 GB");
    expect(container.textContent).toContain("8.0 GB");
    expect(container.textContent).toContain("Unavailable");
    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Download ~30.6 GB")).toBe(true);
    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Download 8.0 GB")).toBe(true);
    expect(container.textContent).not.toContain("~8.0 GB");
  });

  it("marks listed LoRAs unavailable when the backend reports missing install state", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [
            { id: "ready_style", name: "Ready Style", family: "z-image", scope: "global", installState: "installed" },
            { id: "broken_style", name: "Broken Style", family: "z-image", scope: "global", installState: "missing" },
          ],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image", installState: "installed" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    const rows = [...container.querySelectorAll(".lora-row")];
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("Ready Style");
    expect(rows[0].textContent).toContain("installed");
    expect(rows[0].classList.contains("warning")).toBe(false);
    expect(rows[1].textContent).toContain("Broken Style");
    expect(rows[1].textContent).toContain("unavailable");
    expect(rows[1].classList.contains("warning")).toBe(true);
    expect(container.textContent).toContain("1 installed · 1 unavailable");
  });

  it("confirms and deletes models and LoRAs from the Models page", async () => {
    const onDeleteModel = vi.fn(async () => ({ removedManifestEntry: true, warnings: ["Recipe presets reference this model: Moody"] }));
    const onDeleteLora = vi.fn(async () => ({ removedManifestEntry: true, warnings: ["Recipe presets reference this lora: Moody"] }));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [{ id: "ready_style", name: "Ready Style", family: "z-image", scope: "global", installState: "installed", removable: true }],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image", installState: "installed", removable: true }],
          onDeleteLora,
          onDeleteModel,
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
          recipePresets: [{ id: "moody", name: "Moody", model: "z_image_turbo", loras: [{ id: "ready_style" }] }],
        }),
      );
    });

    await act(async () => {
      container.querySelector(".model-card .danger-action").click();
    });

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Delete model "Z-Image Turbo"?'));
    expect(confirm.mock.calls[0][0]).toContain("Referenced by presets: Moody.");
    expect(onDeleteModel).toHaveBeenCalledWith(expect.objectContaining({ id: "z_image_turbo" }));
    expect(container.textContent).toContain("Removed the registry entry for Z-Image Turbo.");

    await act(async () => {
      container.querySelector(".lora-row .danger-action").click();
    });

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('Delete lora "Ready Style"?'));
    expect(onDeleteLora).toHaveBeenCalledWith(expect.objectContaining({ id: "ready_style" }));
    expect(container.textContent).toContain("Removed the registry entry for Ready Style.");
  });

  it("advances elapsed seconds for active job snapshots between server updates", () => {
    const job = {
      id: "image-job-1",
      status: "running",
      elapsedSeconds: 57,
      startedAt: "2026-05-18T20:00:00Z",
    };

    expect(liveElapsedSeconds(job, Date.parse("2026-05-18T20:02:05Z"))).toBe(125);
  });

  it("resets the Models LoRA form after queueing and allows another import while one is pending", async () => {
    const onImportLora = vi.fn(async (payload) => ({
      id: `lora-import-job-${onImportLora.mock.calls.length}`,
      type: "lora_import",
      status: "queued",
      progress: 0,
      payload: { ...payload, loraId: `detail_lora_${onImportLora.mock.calls.length}` },
    }));

    function Harness() {
      const [jobs, setJobs] = React.useState([]);
      async function importLora(payload) {
        const job = await onImportLora(payload);
        setJobs((items) => [job, ...items]);
        return job;
      }
      return withModelManagerContext({
        activeProject: { id: "project-1", name: "Noir" },
        jobs,
        loras: [],
        models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
        onDownloadModel: () => {},
        onImportLora: importLora,
        onOpenQueue: () => {},
      });
    }

    root = createRoot(container);
    await act(async () => {
      root.render(<Harness />);
    });

    const panel = () => loraPanel(container);
    await changeField(field(panel(), "Source URL"), "https://example.com/loras/one.safetensors");
    await changeField(field(panel(), "Name"), "First Detail");
    await act(async () => {
      buttonInside(panel(), "Queue Import").click();
    });

    expect(field(panel(), "Source URL").value).toBe("");
    expect(field(panel(), "Name").value).toBe("");
    expect(container.textContent).toContain("LoRA imports");
    expect(container.textContent).toContain("detail_lora_1");
    expect(container.textContent).not.toContain("No LoRAs in this view");

    await changeField(field(panel(), "Source URL"), "https://example.com/loras/two.safetensors");
    await act(async () => {
      buttonInside(panel(), "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("detail_lora_2");
  });

  it("queues LoRA file uploads from the Models page with project scope", async () => {
    const onImportLora = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "uploaded_detail" } }));
    const loraFile = new File(["lora"], "detail.safetensors", { type: "application/octet-stream" });
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "running",
              stage: "downloading",
              progress: 0.3,
              payload: { loraId: "existing_import" },
            },
          ],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    await act(async () => {
      buttonInside(loraPanel(container), "Upload").click();
    });
    const panel = loraPanel(container);
    await changeField(field(panel, "Scope"), "project");
    await changeFile(field(panel, "LoRA File"), loraFile);
    await act(async () => {
      buttonInside(loraPanel(container), "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledWith(
      expect.objectContaining({
        file: loraFile,
        scope: "project",
      }),
    );
    expect(onImportLora.mock.calls[0][0]).not.toHaveProperty("family");
    expect(container.textContent).toContain("LoRA Import");
    expect(container.textContent).toContain("Running");
  });

  it("keeps failed Models LoRA imports visible inline", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "failed",
              stage: "failed",
              progress: 0.4,
              error: "Adapter crashed",
              payload: { loraId: "broken_detail", family: "z-image" },
            },
          ],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(container.textContent).toContain("LoRA imports");
    expect(container.textContent).toContain("broken_detail");
    expect(container.textContent).toContain("Adapter crashed");
    expect(container.textContent).not.toContain("No LoRAs in this view");
  });

  it("hides failed Models LoRA imports superseded by a completed retry", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [
            {
              id: "lora-import-job-2",
              type: "lora_import",
              status: "completed",
              stage: "completed",
              progress: 1,
              createdAt: "2026-05-18T00:00:01Z",
              payload: { loraId: "detail_lora", family: "z-image" },
            },
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "failed",
              stage: "failed",
              progress: 0.4,
              createdAt: "2026-05-18T00:00:00Z",
              error: "LoRA manifestPath must target the global user manifest or the selected project's LoRA manifest",
              payload: { loraId: "detail_lora", family: "z-image" },
            },
          ],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(container.textContent).not.toContain("LoRA manifestPath must target");
    expect(container.textContent).not.toContain("LoRA imports");
    expect(container.textContent).toContain("No LoRAs in this view");
  });

  it("shows Models page LoRA import errors and resets the queueing state", async () => {
    const onImportLora = vi.fn(async () => {
      throw new Error("LoRA sourceUrl must use http or https");
    });
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: { id: "project-1", name: "Noir" },
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora,
          onOpenQueue: () => {},
        }),
      );
    });

    const panel = loraPanel(container);
    await changeField(field(panel, "Source URL"), "file:///tmp/detail.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(container.textContent).toContain("LoRA sourceUrl must use http or https");
    expect(buttonInside(loraPanel(container), "Queue Import").disabled).toBe(false);
  });

  it("queues model URL imports from the Models page", async () => {
    const onImportModel = vi.fn(async (payload) => ({
      payload: { ...payload, modelId: "custom_model", manifestEntry: { family: "z-image" } },
    }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel,
          onOpenQueue: () => {},
        }),
      );
    });

    const panel = modelImportPanel(container);
    await changeField(field(panel, "Source URL"), "https://example.com/models/custom.safetensors");
    await changeField(field(panel, "Name"), "Custom Model");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportModel).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceUrl: "https://example.com/models/custom.safetensors",
        name: "Custom Model",
        modelType: "image",
      }),
    );
    expect(onImportModel.mock.calls[0][0]).not.toHaveProperty("family");
    expect(container.textContent).toContain("Model import queued for custom_model.");
    expect(container.textContent).toContain("Detected family: z-image.");
  });

  it("sends an explicit family override on model imports when chosen", async () => {
    const onImportModel = vi.fn(async (payload) => ({
      payload: { ...payload, modelId: "custom_model", manifestEntry: { family: payload.family } },
    }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [],
          loras: [],
          models: [{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel,
          onOpenQueue: () => {},
        }),
      );
    });

    const panel = modelImportPanel(container);
    await changeField(field(panel, "Family"), "z-image");
    await changeField(field(panel, "Source URL"), "https://example.com/models/custom.safetensors");
    await act(async () => {
      buttonInside(panel, "Queue Import").click();
    });

    expect(onImportModel.mock.calls[0][0]).toEqual(
      expect.objectContaining({ family: "z-image" }),
    );
  });

  it("renders unassociated models with a needs-family badge", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [],
          loras: [],
          models: [{ id: "imported_custom", name: "Imported Custom", type: "image" }],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(container.textContent).toContain("needs family");
    expect(container.textContent).toContain("unassociated");
  });

  it("shows in-progress model imports inline", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withModelManagerContext({
          activeProject: null,
          jobs: [
            {
              id: "model-import-job-1",
              type: "model_import",
              status: "downloading",
              stage: "downloading",
              progress: 0.42,
              payload: { modelId: "custom_model", name: "Custom Model" },
            },
          ],
          loras: [],
          models: [],
          onDownloadModel: () => {},
          onImportLora: () => {},
          onImportModel: () => {},
          onOpenQueue: () => {},
        }),
      );
    });

    expect(container.textContent).toContain("Model imports in progress");
    expect(container.textContent).toContain("Model Import");
    expect(container.textContent).toContain("Downloading");
  });

  it("adds the SSE ticket as a query parameter", () => {
    expect(eventUrl("/api/v1/jobs/events", "stream-ticket")).toContain("ticket=stream-ticket");
  });

  it("uses one worker-canonical quality enum across studios (no draft/final drift)", () => {
    // sc-1657: PresetManager previously used draft/balanced/final while VideoStudio
    // and the worker (video_adapters.py step maps) use fast/balanced/best, so saved
    // presets didn't match the studio control. Pin the shared values.
    expect(qualityChoices.map(([value]) => value)).toEqual(["fast", "balanced", "best"]);
    expect(qualityChoices.map(([, label]) => label)).toEqual(["Draft", "Balanced", "Final"]);
  });

  it("keeps the centralized job-type/status enums consistent", () => {
    // GPU-required job types must stay aligned with the Rust dispatch gate
    // (jobs_store.rs::job_requires_gpu); errorStatuses is terminal minus completed.
    expect(GPU_REQUIRED_JOB_TYPES.has("video_generate")).toBe(true);
    expect(GPU_REQUIRED_JOB_TYPES.has("model_download")).toBe(false);
    expect([...errorStatuses].sort()).toEqual(["canceled", "failed", "interrupted"]);
    expect(errorStatuses.has("completed")).toBe(false);
  });

  it("filters stale and placeholder-only GPU workers from the queue view", async () => {
    global.fetch.mockImplementation((url) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project One" }]));
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(
          response([
            {
              id: "python-gpu-0",
              gpuId: "0",
              gpuName: "Fixture GPU 0",
              status: "idle",
              capabilities: ["placeholder", "gpu", "image_generate"],
              utilization: { memoryTotalMb: 24576, memoryUsedMb: 4096, memoryFreeMb: 20480, gpuLoadPercent: 12 },
            },
            {
              id: "rust-gpu-1",
              gpuId: "1",
              gpuName: "Rust placeholder GPU",
              status: "idle",
              capabilities: ["placeholder", "gpu", "nvidia"],
            },
            {
              id: "stale-gpu-2",
              gpuId: "2",
              gpuName: "Stale GPU",
              status: "offline",
              capabilities: ["placeholder", "gpu", "image_generate"],
            },
            {
              id: "rust-cpu",
              gpuId: "cpu",
              gpuName: "Rust CPU utility worker",
              status: "idle",
              capabilities: ["placeholder", "cpu", "model_download"],
            },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    expect(container.textContent).toContain("Fixture GPU 0");
    expect(container.textContent).toContain("20.0 GB");
    expect(container.textContent).toContain("4.0 GB / 24.0 GB");
    expect(container.textContent).toContain("12%");
    expect(container.textContent).not.toContain("Rust CPU utility worker");
    expect(container.textContent).not.toContain("Rust placeholder GPU");
    expect(container.textContent).not.toContain("Stale GPU");
    expect([...container.querySelector("#queue-gpu").options].map((option) => option.value)).toEqual(["auto", "0"]);
  });

  it("shows queued job cancellation from the action response even when the list refresh is stale", async () => {
    const queuedJob = {
      id: "job-queued",
      type: "image_generate",
      status: "queued",
      stage: "queued",
      progress: 0,
      projectId: "project-1",
      projectName: "Project 1",
      requestedGpu: "auto",
      payload: { prompt: "mist" },
      attempts: 1,
      cancelRequested: false,
      createdAt: "2026-05-19T09:00:00Z",
      updatedAt: "2026-05-19T09:00:00Z",
    };
    const canceledJob = {
      ...queuedJob,
      status: "canceled",
      stage: "canceled",
      progress: 1,
      cancelRequested: true,
      message: "Canceled before a worker started.",
      updatedAt: "2026-05-19T09:01:00Z",
    };
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/job-queued/cancel") && options.method === "POST") {
        return Promise.resolve(response(canceledJob));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project 1" }]));
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response([queuedJob]));
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    expect(container.textContent).toContain("Queued");

    await act(async () => {
      [...container.querySelectorAll(".worker-progress-card__actions button")].find((button) => button.textContent === "Cancel").click();
    });
    await settle();

    expect(container.textContent).toContain("Cancelled");
    expect(container.textContent).toContain("Canceled before a worker started.");
    expect(container.textContent).not.toContain("Queued");
    expect(container.textContent).not.toContain("Waiting for an available GPU worker.");
  });

  it("keeps fresher SSE job state when a post-action refresh returns stale data", async () => {
    const failedJob = {
      id: "job-failed",
      type: "image_generate",
      status: "failed",
      stage: "failed",
      progress: 1,
      projectId: "project-1",
      projectName: "Project 1",
      requestedGpu: "auto",
      payload: { prompt: "mist" },
      attempts: 1,
      cancelRequested: false,
      createdAt: "2026-05-19T09:00:00Z",
      updatedAt: "2026-05-19T09:00:00Z",
    };
    const retryJob = {
      ...failedJob,
      id: "job-retry",
      status: "running",
      stage: "generating",
      progress: 0.1,
      attempts: 2,
      updatedAt: "2026-05-19T09:01:00Z",
    };
    const fresherRetryJob = {
      ...retryJob,
      progress: 0.4,
      message: "Worker advanced during refresh.",
      updatedAt: "2026-05-19T09:02:00Z",
    };
    let jobsRequestCount = 0;
    let resolvePostRetryJobs;
    const postRetryJobs = new Promise((resolve) => {
      resolvePostRetryJobs = resolve;
    });
    global.fetch.mockImplementation((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: false }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: false }));
      }
      if (path.endsWith("/jobs/job-failed/retry") && options.method === "POST") {
        return Promise.resolve(response(retryJob));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-1", name: "Project 1" }]));
      }
      if (path.endsWith("/workers")) {
        return Promise.resolve(response([]));
      }
      if (path.endsWith("/jobs")) {
        jobsRequestCount += 1;
        return jobsRequestCount === 1 ? Promise.resolve(response([failedJob])) : postRetryJobs;
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll(".worker-progress-card__actions button")].find((button) => button.textContent === "Retry").click();
      await Promise.resolve();
    });
    await act(async () => {
      FakeEventSource.instances[0].listeners["job.updated"]({ data: JSON.stringify(fresherRetryJob) });
    });
    await act(async () => {
      resolvePostRetryJobs(response([retryJob]));
    });
    await settle();

    expect(container.querySelector(".progress-track")?.getAttribute("aria-label")).toBe("40% complete");
    expect(container.textContent).toContain("Worker advanced during refresh.");
  });

  it("explains queued GPU jobs that are waiting on capability or busy workers", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Project 1" },
          createPlaceholderJob: (event) => event.preventDefault(),
          filteredJobs: [
            {
              id: "job-waiting",
              type: "image_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "mist", model: "z_image_turbo" },
              attempts: 1,
            },
            {
              id: "job-blocked",
              type: "video_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "clip" },
              attempts: 1,
            },
            {
              id: "job-download",
              type: "model_download",
              status: "downloading",
              stage: "downloading",
              progress: 0.4,
              projectId: null,
              projectName: null,
              requestedGpu: "auto",
              assignedGpu: "cpu",
              payload: {
                modelId: "qwen_image_edit",
                modelName: "Qwen Image Edit",
                repo: "Qwen/Qwen-Image-Edit",
              },
              attempts: 1,
            },
            {
              id: "job-waiting-download",
              type: "image_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "edit", model: "qwen_image_edit" },
              attempts: 1,
            },
            {
              id: "job-dependency",
              type: "image_generate",
              status: "running",
              stage: "generating",
              progress: 0.5,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "0",
              assignedGpu: "0",
              payload: { prompt: "source" },
              attempts: 1,
            },
            {
              id: "job-waiting-dependency",
              type: "image_generate",
              status: "queued",
              stage: "queued",
              progress: 0,
              projectId: "project-1",
              projectName: "Project 1",
              requestedGpu: "auto",
              payload: { prompt: "dependent", dependsOnJobId: "job-dependency" },
              attempts: 1,
            },
          ],
          gpuOptions: ["auto", "0"],
          jobAction: () => {},
          jobPrompt: "Placeholder generation",
          projectFilter: "all",
          projects: [{ id: "project-1", name: "Project 1" }],
          requestedGpu: "auto",
          setJobPrompt: () => {},
          setProjectFilter: () => {},
          setRequestedGpu: () => {},
          visibleWorkers: [
            {
              id: "misregistered-cpu",
              gpuId: "cpu",
              gpuName: "CPU worker",
              status: "idle",
              currentJobId: null,
              capabilities: ["placeholder", "cpu", "video_generate"],
              loadedModels: [],
            },
            {
              id: "python-gpu-0",
              gpuId: "0",
              gpuName: "Fixture GPU 0",
              status: "busy",
              currentJobId: "job-active",
              capabilities: ["placeholder", "gpu", "image_generate"],
              loadedModels: ["z_image_turbo"],
            },
          ],
          },
          <QueueScreen />,
        ),
      );
    });

    expect(container.textContent).toContain("Waiting: an eligible worker is busy.");
    expect(container.textContent).toContain("Blocked: no active worker supports video generate.");
    expect(container.textContent).toContain("Waiting for model download Qwen Image Edit to finish.");
    expect(container.textContent).toContain("Waiting for dependency job-dependency to finish.");
    expect(container.textContent).toContain("Warm: z_image_turbo");
  });

  it("updates Queue GPU utilization when worker props change", async () => {
    const queueProps = {
      activeProject: { id: "project-1", name: "Project 1" },
      createPlaceholderJob: (event) => event.preventDefault(),
      filteredJobs: [],
      gpuOptions: ["auto", "0"],
      jobAction: () => {},
      jobPrompt: "Placeholder generation",
      projectFilter: "all",
      projects: [{ id: "project-1", name: "Project 1" }],
      requestedGpu: "auto",
      setJobPrompt: () => {},
      setProjectFilter: () => {},
      setRequestedGpu: () => {},
    };
    const worker = {
      id: "python-gpu-0",
      gpuId: "0",
      gpuName: "Fixture GPU 0",
      status: "idle",
      capabilities: ["placeholder", "gpu", "image_generate"],
      loadedModels: [],
      utilization: { memoryTotalMb: 24576, memoryUsedMb: 4096, memoryFreeMb: 20480, gpuLoadPercent: 12 },
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext({ ...queueProps, visibleWorkers: [worker] }, <QueueScreen />));
    });

    expect(container.textContent).toContain("20.0 GB");
    expect(container.textContent).toContain("12%");

    await act(async () => {
      root.render(
        withAppContext(
          {
            ...queueProps,
            visibleWorkers: [
              {
                ...worker,
                utilization: { memoryTotalMb: 24576, memoryUsedMb: 12288, memoryFreeMb: 12288, gpuLoadPercent: 67 },
              },
            ],
          },
          <QueueScreen />,
        ),
      );
    });

    expect(container.textContent).toContain("12.0 GB / 24.0 GB");
    expect(container.textContent).toContain("67%");
    expect(container.textContent).not.toContain("20.0 GB");
  });

  it("ignores duplicate image submits while job creation is in flight", async () => {
    let resolveJob;
    const createImageJob = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveJob = resolve;
        }),
    );
    const onLocalJobCreated = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onLocalJobCreated,
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      container.querySelector(".image-studio form").requestSubmit();
    });
    await settle();
    await act(async () => {
      container.querySelector(".image-studio form").requestSubmit();
    });

    expect(createImageJob).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveJob({ id: "image-job-1" });
    });
    await settle();

    expect(onLocalJobCreated).toHaveBeenCalledWith({ id: "image-job-1" });
  });

  it("remembers studio settings per workspace across remounts", async () => {
    const imageProps = {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [],
      characters: [],
      createImageJob: () => {},
      deleteAsset: () => {},
      gpuOptions: ["auto"],
      imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      latestAssets: [],
      localJobs: [],
      loras: [],
      onPreview: () => {},
      purgeAsset: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    };
    const promptField = () => container.querySelector("textarea[aria-label='Prompt']");

    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext(imageProps));
    });
    const defaultPrompt = promptField().value;
    await changeField(promptField(), "a custom remembered prompt");
    await settle();

    // Leaving the studio and returning to the same workspace restores the prompt.
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext(imageProps));
    });
    expect(promptField().value).toBe("a custom remembered prompt");

    // A different workspace starts from its own settings, not workspace-1's.
    await act(async () => root.unmount());
    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext({ ...imageProps, activeProject: { id: "project-2", name: "Other" } }));
    });
    expect(promptField().value).toBe(defaultPrompt);
  });

  it("keeps completed image progress visible until the generated asset renders", async () => {
    const completedJob = {
      id: "image-job-1",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      elapsedSeconds: 8,
      requestedGpu: "auto",
      payload: { prompt: "long alley" },
      result: { generationSetId: "gen-1", assetIds: ["asset-1"] },
    };
    const imageProps = {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [],
      characters: [],
      createImageJob: () => {},
      deleteAsset: () => {},
      gpuOptions: ["auto"],
      imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      latestAssets: [],
      localJobs: [completedJob],
      loras: [],
      onPreview: () => {},
      purgeAsset: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(withImageStudioContext(imageProps));
    });

    // Card stays visible while the completed job waits for its asset to arrive.
    expect(container.querySelector(".worker-progress-card")).not.toBeNull();
    expect(container.textContent).not.toContain("No fresh image batch");

    const generatedAsset = {
      id: "asset-1",
      type: "image",
      displayName: "Generated Image",
      generationSetId: "gen-1",
      status: {},
    };
    await act(async () => {
      root.render(withImageStudioContext({ ...imageProps, assets: [generatedAsset], latestAssets: [generatedAsset] }));
    });

    // Once the asset surfaces in latestAssets the card collapses out of the stack
    // (selectStackedJobs + resultVisible). The asset itself is in Recent Assets.
    expect(container.querySelector(".worker-progress-card")).toBeNull();
    expect(container.textContent).not.toContain("No fresh image batch");
  });

  it("reconstructs running image batch slots from a generation set without asset ids", async () => {
    const localJob = {
      id: "image-job-1",
      type: "image_generate",
      status: "running",
      stage: "generating",
      progress: 0.82,
      elapsedSeconds: 8,
      requestedGpu: "auto",
      payload: { prompt: "long alley", count: 3 },
      result: { generationSetId: "gen-1", expectedCount: 3 },
    };
    const assets = [
      {
        id: "asset-2",
        projectId: "project-1",
        type: "image",
        displayName: "Generated",
        generationSetId: "gen-1",
        file: { path: "runs/run_0007/assets/images/generated_0002.png", mimeType: "image/png" },
        status: {},
      },
      {
        id: "asset-1",
        projectId: "project-1",
        type: "image",
        displayName: "Generated",
        generationSetId: "gen-1",
        file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
        status: {},
      },
    ];

    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets,
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [localJob],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    const images = [...container.querySelectorAll(".worker-progress-card__thumb-media")].map((image) => image.getAttribute("src"));
    expect(images[0]).toContain("/api/v1/projects/project-1/files/assets/images/generated_0001.png");
    expect(images[1]).toContain("/api/v1/projects/project-1/files/runs/run_0007/assets/images/generated_0002.png");
    // 2 completed thumbnails + 1 skeleton slot = 3 total cells.
    expect(container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBe(1);
  });

  it("cancels a running image job from the studio progress card", async () => {
    const runningJob = {
      id: "image-job-cancel",
      type: "image_generate",
      status: "running",
      stage: "generating",
      progress: 0.4,
      requestedGpu: "auto",
      payload: { prompt: "cancel me" },
      result: {},
    };
    const onCancelJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [runningJob],
          loras: [],
          onCancelJob,
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    const cancelButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Cancel");
    expect(cancelButton).not.toBeUndefined();

    await act(async () => {
      cancelButton.click();
    });

    expect(onCancelJob).toHaveBeenCalledWith(expect.objectContaining({ id: "image-job-cancel" }));
  });

  it("hides the cancel control once an image job reaches a terminal state", async () => {
    const completedJob = {
      id: "image-job-done",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      requestedGpu: "auto",
      payload: { prompt: "all done" },
      result: {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [completedJob],
          loras: [],
          onCancelJob: () => {},
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Cancel run")).toBe(false);
  });

  it("hides completed image progress with stale missing result metadata", async () => {
    const staleCompletedJob = {
      id: "image-job-stale",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      elapsedSeconds: 8,
      requestedGpu: "auto",
      updatedAt: "2026-05-18T00:00:00Z",
      payload: { prompt: "missing result metadata" },
      result: {},
    };

    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [staleCompletedJob],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    expect(container.textContent).not.toContain("Finished. Fetching result...");
    expect(container.textContent).toContain("No fresh image batch");
  });

  it("removes a canceled image job's progress card and placeholder thumbnails", async () => {
    const canceledJob = {
      id: "image-job-canceled",
      type: "image_generate",
      status: "canceled",
      stage: "canceled",
      progress: 0.5,
      requestedGpu: "auto",
      payload: { prompt: "abandon ship", count: 4 },
      result: { expectedCount: 4 },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [canceledJob],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.querySelector(".worker-progress-card")).toBeNull();
    expect(container.querySelector(".review-placeholder")).toBeNull();
    expect(container.textContent).not.toContain("Canceled #");
    expect(container.textContent).toContain("No fresh image batch");
  });

  it("stacks multiple image runs, each with its own progress card and slots", async () => {
    const runningJob = {
      id: "image-job-run",
      type: "image_generate",
      status: "running",
      stage: "generating",
      progress: 0.5,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:00:00Z",
      payload: { prompt: "first run", count: 2 },
      result: { generationSetId: "gen-1", expectedCount: 2 },
    };
    const queuedJob = {
      id: "image-job-queue",
      type: "image_generate",
      status: "queued",
      progress: 0,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:01:00Z",
      payload: { prompt: "second run", count: 3 },
      result: { expectedCount: 3 },
    };
    const renderedAsset = {
      id: "asset-1",
      projectId: "project-1",
      type: "image",
      displayName: "Generated",
      generationSetId: "gen-1",
      file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
      status: {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [renderedAsset],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          localJobs: [runningJob, queuedJob],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.querySelectorAll(".worker-progress-card").length).toBe(2);
    // Running run renders its one finished image alongside its remaining slot.
    expect(container.querySelector(".worker-progress-card__thumb-media")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated_0001.png",
    );
    // Queued run shows its own pending skeleton slots while it waits.
    expect(container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBeGreaterThan(0);
    expect(container.textContent).not.toContain("No fresh image batch");
  });

  it("holds a completed run above the queue until the next run starts", async () => {
    const completedJob = {
      id: "image-job-done",
      type: "image_generate",
      status: "completed",
      stage: "completed",
      progress: 1,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:00:00Z",
      completedAt: "2026-05-27T10:00:30Z",
      payload: { prompt: "finished run" },
      result: { generationSetId: "gen-1", assetIds: ["asset-1"] },
    };
    const nextJob = {
      id: "image-job-next",
      type: "image_generate",
      status: "queued",
      progress: 0,
      requestedGpu: "auto",
      createdAt: "2026-05-27T10:01:00Z",
      payload: { prompt: "next run", count: 2 },
      result: { expectedCount: 2 },
    };
    const renderedAsset = {
      id: "asset-1",
      projectId: "project-1",
      type: "image",
      displayName: "Generated",
      generationSetId: "gen-1",
      file: { path: "assets/images/generated_0001.png", mimeType: "image/png" },
      status: {},
    };
    const baseProps = {
      activeProject: { id: "project-1", name: "Noir" },
      assets: [renderedAsset],
      characters: [],
      createImageJob: () => {},
      deleteAsset: () => {},
      gpuOptions: ["auto"],
      imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
      latestAssets: [renderedAsset],
      loras: [],
      onPreview: () => {},
      purgeAsset: () => {},
      requestedGpu: "auto",
      selectedAsset: null,
      setRequestedGpu: () => {},
      updateAssetStatus: () => {},
    };

    root = createRoot(container);
    // Completed run with a run still queued behind it: both stay, completed on top.
    await act(async () => {
      root.render(withImageStudioContext({ ...baseProps, localJobs: [completedJob, nextJob] }));
    });
    await settle();
    expect(container.querySelectorAll(".worker-progress-card").length).toBe(2);
    // The queued next run shows skeleton slots for its expected outputs.
    expect(container.querySelectorAll(".worker-progress-card__thumb-cell.skeleton").length).toBeGreaterThan(0);

    // The next run starts: the completed run slides out and the running run remains.
    await act(async () => {
      root.render(
        withImageStudioContext({
          ...baseProps,
          localJobs: [completedJob, { ...nextJob, status: "running", stage: "generating", progress: 0.3 }],
        }),
      );
    });
    await settle();
    expect(container.querySelectorAll(".worker-progress-card").length).toBe(1);
    expect(container.querySelector(".worker-progress-card.running")).not.toBeNull();
    expect(container.querySelector(".worker-progress-card.completed")).toBeNull();
  });

  it("submits compatible image LoRAs while capping simple user selections at two", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [
            { id: "built_in", name: "Built In", family: "z-image", scope: "builtin", defaultWeight: 0.6 },
            { id: "global_style", name: "Global Style", family: "z-image", scope: "global" },
            { id: "project_mira", name: "Project Mira", family: "z-image", scope: "project", files: ["mira.safetensors"] },
            { id: "third_user", name: "Third User", family: "z-image", scope: "global" },
            { id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "global" },
            { id: "missing_lora", name: "Missing LoRA", family: "z-image", scope: "global", installState: "missing" },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    expect(container.textContent).not.toContain("Built In");
    expect(container.textContent).not.toContain("Qwen Only");
    expect(container.textContent).not.toContain("Missing LoRA");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });

    expect(container.textContent).toContain("Built In");
    const checkboxes = [...container.querySelectorAll('.lora-choice input[type="checkbox"]')];
    await act(async () => {
      checkboxes[0].click();
      checkboxes[1].click();
      checkboxes[2].click();
    });

    expect(checkboxes[3].disabled).toBe(true);

    await act(async () => {
      container.querySelector('.lora-picker .checkline input[type="checkbox"]').click();
    });

    expect(container.textContent).toContain("Qwen Only");
    expect(container.textContent).not.toContain("Missing LoRA");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        loras: [
          expect.objectContaining({ id: "built_in", scope: "builtin", weight: 0.6 }),
          expect.objectContaining({ id: "global_style", scope: "global" }),
          expect.objectContaining({ id: "project_mira", scope: "project", files: ["mira.safetensors"] }),
        ],
      }),
    );
  });

  it("excludes cross-family LoRAs from a Kolors selection (sc-1927)", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          // Mirrors the Kolors manifest entry: family "kolors", LoRA families ["kolors"].
          imageModels: [
            { id: "kolors", name: "Kolors", type: "image", family: "kolors", loraCompatibility: { families: ["kolors"] }, capabilities: ["text_to_image"] },
          ],
          latestAssets: [],
          loras: [
            { id: "z_style", name: "Z Style", family: "z-image", scope: "global" },
            { id: "kolors_style", name: "Kolors Style", family: "kolors", scope: "global" },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });

    const loraNames = [...container.querySelectorAll(".lora-choice strong")].map((node) => node.textContent);
    // A kolors-family model must not offer a z-image LoRA as compatible.
    expect(loraNames).toContain("Kolors Style");
    expect(loraNames).not.toContain("Z Style");
  });

  it("blocks image submit when a visible incompatible LoRA is selected", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [{ id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "builtin" }],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });
    await act(async () => {
      container.querySelector('.lora-picker .checkline input[type="checkbox"]').click();
    });
    await act(async () => {
      container.querySelector('.lora-choice input[type="checkbox"]').click();
    });

    const generate = [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate");
    expect(container.textContent).toContain("Generate is blocked");
    expect(container.textContent).toContain("Qwen Only");
    expect(generate.disabled).toBe(true);

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Hide advanced").click();
    });
    await settle();

    expect([...container.querySelectorAll("button")].some((button) => button.textContent === "Hide advanced")).toBe(true);
    expect(container.textContent).toContain("Qwen Only");

    await act(async () => {
      generate.click();
    });

    expect(createImageJob).not.toHaveBeenCalled();
  });

  it("applies preset defaults and hidden preset LoRAs to image jobs", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [
            {
              id: "cinematic_detail",
              name: "Cinematic Detail",
              family: "z-image",
              scope: "builtin",
              defaultWeight: 0.55,
              presetManaged: true,
            },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              defaults: { count: 2, resolution: "1280x720", negativePrompt: "flat lighting" },
              prompt: { suffix: "cinematic lighting" },
              builtInLoras: [{ id: "cinematic_detail", weight: 0.4 }],
              ui: { description: "Balanced cinematic color, contrast, and detail." },
            },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    expect(container.textContent).toContain("Cinematic");
    expect(container.textContent).toContain("Balanced cinematic color, contrast, and detail.");
    expect(container.textContent).toContain("Adds: cinematic lighting");
    expect(container.textContent).toContain("Preset LoRA applied at generation: Cinematic Detail");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        count: 2,
        width: 1280,
        height: 720,
        negativePrompt: "flat lighting",
        prompt: "A cinematic frame of a neon street at midnight",
        recipePresetId: "cinematic",
        loras: [],
        advanced: { resolution: "1280x720" },
      }),
    );
  });

  it("aspect picker reflects the selected model's trained buckets", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto", "1"],
          imageModels: [
            {
              id: "sensenova_u1_8b",
              name: "SenseNova-U1 8B",
              type: "image",
              family: "sensenova-u1",
              defaults: { resolution: "2048x2048" },
              limits: { resolutions: ["2048x2048", "2720x1536", "1536x2720"] },
            },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    const aspect = field(container, "Aspect");
    const optionValues = [...aspect.querySelectorAll("option")].map((option) => option.value);
    expect(optionValues).toEqual(["2048x2048", "2720x1536", "1536x2720"]);
    // 1024x1024 isn't a SenseNova bucket, so the picker snaps to the model default.
    expect(aspect.value).toBe("2048x2048");
  });

  it("surfaces model and preset first and lets image generation run with no preset", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto", "1"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [{ id: "cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", presetManaged: true }],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              defaults: { count: 2, negativePrompt: "flat lighting" },
              builtInLoras: [{ id: "cinematic_detail" }],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // Primary preset controls are surfaced in the rail (no longer behind Advanced),
    // alongside the hero-mounted prompt + preset chip strip.
    const railLabels = [...container.querySelectorAll(".preset-rail > label, .preset-rail .preset-rail-row label")].map(
      (label) => label.childNodes[0]?.textContent.trim(),
    );
    expect(railLabels).toEqual(expect.arrayContaining(["Model", "Variations", "Aspect"]));
    expect(container.querySelector(".prompt-input")).not.toBeNull();
    expect(container.querySelector(".preset-chips").textContent).toContain("None");
    expect(field(container, "Variations").value).toBe("2");
    expect(field(container, "GPU")).toBeUndefined();
    expect(container.textContent).not.toContain("LoRAs");

    await act(async () => {
      [...container.querySelectorAll(".preset-chip")]
        .find((chip) => chip.textContent.trim() === "None")
        .click();
    });
    await settle();

    expect(container.textContent).toContain("No preset selected");
    expect(field(container, "Variations").value).toBe("4");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });

    expect(field(container, "GPU")).not.toBeUndefined();
    expect(container.textContent).toContain("LoRAs");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        count: 4,
        negativePrompt: "",
        recipePresetId: null,
        loras: [],
      }),
    );
    expect(createImageJob.mock.calls[0][0]).not.toHaveProperty("stylePreset");
  });

  it("threads the Image Studio upscale controls into enabled image jobs", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });

    expect(container.querySelector('.upscale-toggle input[type="checkbox"]').checked).toBe(false);
    expect(field(container, "Scale").disabled).toBe(true);
    expect(field(container, "Engine").disabled).toBe(true);

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledTimes(1);
    expect(createImageJob.mock.calls[0][0]).not.toHaveProperty("upscale");

    await act(async () => {
      container.querySelector('.upscale-toggle input[type="checkbox"]').click();
    });
    await changeField(field(container, "Scale"), "4");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        upscale: {
          enabled: true,
          factor: 4,
          engine: "real-esrgan",
        },
      }),
    );

    await changeField(field(container, "Engine"), "aura-sr");
    expect(field(container, "Scale").value).toBe("4");
    expect([...field(container, "Scale").querySelectorAll("option")].map((option) => option.value)).toEqual(["4"]);

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenLastCalledWith(
      expect.objectContaining({
        upscale: {
          enabled: true,
          factor: 4,
          engine: "aura-sr",
        },
      }),
    );
  });

  it("submits a Kolors character job with the approved reference and IP-Adapter scale", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "kolors",
              name: "Kolors",
              type: "image",
              family: "kolors",
              capabilities: ["text_to_image", "edit_image", "character_image", "style_variations"],
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-1", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // The approved reference renders as a selected identity thumbnail.
    expect(container.textContent).toContain("Reference identity");
    expect(container.querySelector(".reference-thumb.active")).not.toBeNull();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        characterId: "char-1",
        referenceAssetId: "ref-1",
        count: 4,
        advanced: { resolution: "1024x1024", ipAdapterScale: 0.6 },
      }),
    );
  });

  it("exposes the InstantID Identity structure slider and submits its tuned defaults", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "instantid_realvisxl",
              name: "InstantID (RealVisXL)",
              type: "image",
              family: "sdxl",
              capabilities: ["character_image"],
              ui: {
                referenceStrengthDefault: 0.8,
                identityStructure: { label: "Identity structure", default: 0.8, min: 0.3, max: 1.0, step: 0.05 },
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-iid", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // The second (InstantID-only) slider renders; the strength slider is relabeled.
    expect(container.textContent).toContain("Identity structure");
    expect(container.textContent).toContain("Identity strength");
    expect(container.textContent).not.toContain("Reference strength");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // Tuned InstantID defaults flow through advanced: ipAdapterScale 0.8 (raised from
    // the global 0.6) + controlnetConditioningScale 0.8 (the IdentityNet lock).
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        referenceAssetId: "ref-1",
        advanced: { resolution: "1024x1024", ipAdapterScale: 0.8, controlnetConditioningScale: 0.8 },
      }),
    );
  });

  it("offers the InstantID View angle picker and submits the chosen angle", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "instantid_realvisxl",
              name: "InstantID (RealVisXL)",
              type: "image",
              family: "sdxl",
              capabilities: ["character_image"],
              ui: {
                referenceStrengthDefault: 0.8,
                identityStructure: { label: "Identity structure", default: 0.8, min: 0.3, max: 1.0, step: 0.05 },
                viewAngles: [
                  { id: "three_quarter_left", label: "Three-quarter left" },
                  { id: "left_profile", label: "Left profile" },
                ],
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-va", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // The dropdown lists "Match reference" plus the model's declared angles.
    const angleOptions = [...field(container, "View angle").options].map((option) => option.textContent);
    expect(angleOptions).toContain("Match reference");
    expect(angleOptions).toContain("Left profile");

    await changeField(field(container, "View angle"), "left_profile");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // The chosen angle rides advanced.viewAngle for the worker's landmark pack.
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        referenceAssetId: "ref-1",
        advanced: expect.objectContaining({ viewAngle: "left_profile", ipAdapterScale: 0.8 }),
      }),
    );
  });

  it("surfaces the FLUX Variation slider alongside Reference strength and submits both knobs", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "flux_dev",
              name: "FLUX.1 [dev]",
              type: "image",
              family: "flux",
              capabilities: ["character_image"],
              ui: {
                // FLUX exposes BOTH the IP-Adapter reference-strength slider
                // (no override → global 0.6 default; the manifest sets 0.7 in
                // production but this fixture intentionally omits that to verify
                // the picker still renders correctly without a tuned default)
                // AND the Variation slider for true_cfg_scale (sc-2017).
                variationStrength: { label: "Variation", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-flux", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // Both sliders are visible for FLUX: Reference strength (IP-Adapter) AND
    // Variation (the trueCfgScale knob, which is FLUX's real-CFG lever since
    // base FLUX is guidance-distilled).
    expect(container.textContent).toContain("Reference strength");
    expect(container.textContent).toContain("Variation");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // Both knobs ride advanced: ipAdapterScale falls back to the global 0.6
    // default (no per-model override) and trueCfgScale follows the model's
    // declared default (4.0).
    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        referenceAssetId: "ref-1",
        advanced: expect.objectContaining({ ipAdapterScale: 0.6, trueCfgScale: 4.0 }),
      }),
    );
  });

  it("hides the Reference strength slider for Qwen and submits trueCfgScale alone", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            {
              id: "char-1",
              name: "Mira",
              type: "person",
              looks: [],
              approvedReferences: [
                {
                  assetId: "ref-1",
                  approved: true,
                  asset: {
                    id: "ref-1",
                    type: "image",
                    displayName: "Mira ref",
                    projectId: "project-1",
                    file: { path: "assets/images/ref_0001.png", mimeType: "image/png" },
                  },
                },
              ],
            },
          ],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "qwen_image_edit_2511",
              name: "Qwen Image Edit (2511)",
              type: "image",
              family: "qwen-image",
              capabilities: ["character_image"],
              ui: {
                // Qwen-Image-Edit's variation knob is trueCfgScale; the IP-Adapter
                // reference-strength slider would be a no-op here (the worker
                // adapter doesn't read ipAdapterScale). Hide the slider AND drop
                // it from the submit payload (sc-2017).
                hideReferenceStrength: true,
                variationStrength: { label: "Variation", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
              },
            },
          ],
          latestAssets: [],
          launchRequest: { id: "launch-qwen", view: "Image", characterId: "char-1", referenceAssetId: "ref-1", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    // The no-op Reference-strength slider is hidden; only Variation renders.
    expect(container.textContent).not.toContain("Reference strength");
    expect(container.textContent).not.toContain("Identity strength");
    expect(container.textContent).toContain("Variation");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    // advanced carries trueCfgScale but explicitly NOT ipAdapterScale.
    const lastCall = createImageJob.mock.calls.at(-1)[0];
    expect(lastCall.advanced.trueCfgScale).toBe(4.0);
    expect(lastCall.advanced).not.toHaveProperty("ipAdapterScale");
  });

  it("limits the character image model picker to reference-capable models", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [{ id: "char-1", name: "Mira", type: "person", looks: [], approvedReferences: [] }],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] },
            { id: "flux_dev", name: "FLUX", type: "image", capabilities: ["text_to_image"] },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    // Text mode lists every image model.
    let modelOptions = [...field(container, "Model").options].map((option) => option.textContent);
    expect(modelOptions).toContain("Kolors");
    expect(modelOptions).toContain("FLUX");
    const sceneSuggestions = [...container.querySelectorAll(".suggestion")].map((button) => button.textContent).join("|");

    await act(async () => {
      [...container.querySelectorAll(".segmented-control button")].find((button) => button.textContent === "With character").click();
    });
    await settle();

    // Character mode hides models without a reference (IP-Adapter) engine.
    modelOptions = [...field(container, "Model").options].map((option) => option.textContent);
    expect(modelOptions).toContain("Kolors");
    expect(modelOptions).not.toContain("FLUX");

    // Suggestions swap to the variation-oriented set in character mode.
    const characterSuggestions = [...container.querySelectorAll(".suggestion")].map((button) => button.textContent).join("|");
    expect(characterSuggestions).not.toBe(sceneSuggestions);
  });

  it("seeds a character-aware default prompt from the character's notes", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            { id: "char-1", name: "Mira", type: "person", description: "A grizzled detective in a trench coat", looks: [], approvedReferences: [] },
          ],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          launchRequest: { id: "launch-prompt-1", view: "Image", characterId: "char-1", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.querySelector(".prompt-input").value).toBe("A grizzled detective in a trench coat");
  });

  it("falls back to a type-specific default prompt when the character has no notes", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [{ id: "char-2", name: "Echo", type: "creature", looks: [], approvedReferences: [] }],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          launchRequest: { id: "launch-prompt-2", view: "Image", characterId: "char-2", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.querySelector(".prompt-input").value).toBe("The creature in a new setting, varied pose, natural lighting");
  });

  it("keeps an edited prompt when switching into character mode", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [
            { id: "char-1", name: "Mira", type: "person", description: "A grizzled detective", looks: [], approvedReferences: [] },
          ],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await changeField(container.querySelector(".prompt-input"), "my own deliberate scene");
    await act(async () => {
      [...container.querySelectorAll(".segmented-control button")].find((button) => button.textContent === "With character").click();
    });
    await changeField(field(container, "Character"), "char-1");
    await settle();

    // The user's wording survives entering character mode.
    expect(container.querySelector(".prompt-input").value).toBe("my own deliberate scene");
  });

  it("generates without a reference and warns when the character has no approved reference image", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [{ id: "char-2", name: "Echo", type: "creature", looks: [], approvedReferences: [] }],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "kolors", name: "Kolors", type: "image", capabilities: ["text_to_image", "character_image"] }],
          latestAssets: [],
          launchRequest: { id: "launch-2", view: "Image", characterId: "char-2", mode: "character_image" },
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.textContent).toContain("No approved reference");
    expect(container.querySelector(".reference-thumb")).toBeNull();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "character_image",
        characterId: "char-2",
        referenceAssetId: null,
        advanced: { resolution: "1024x1024" },
      }),
    );
  });

  it("blocks image presets whose managed LoRAs do not match the selected model", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ],
          latestAssets: [],
          loras: [
            {
              id: "qwen_detail",
              name: "Qwen Detail",
              family: "qwen-image",
              scope: "builtin",
              presetManaged: true,
            },
          ],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              workflow: "text_to_image",
              builtInLoras: [{ id: "qwen_detail", weight: 0.4 }],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    const generate = [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate");
    expect(container.textContent).toContain("Preset cannot run with Z-Image");
    expect(container.textContent).toContain("qwen_detail");
    expect(generate.disabled).toBe(true);

    await act(async () => {
      generate.click();
    });

    expect(createImageJob).not.toHaveBeenCalled();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });
    await changeField(field(container, "Model"), "qwen_image");
    await settle();

    expect(container.textContent).not.toContain("Preset cannot run with Qwen Image");
    expect(generate.disabled).toBe(false);

    await act(async () => {
      generate.click();
    });

    expect(createImageJob).toHaveBeenCalledWith(expect.objectContaining({ model: "qwen_image", recipePresetId: "cinematic" }));
  });

  it("blocks video presets whose managed LoRAs do not match the selected model", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [{ id: "wan_motion", name: "Wan Motion", family: "wan-video", scope: "builtin", presetManaged: true }],
          setPreviewAsset: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            {
              id: "dream_motion",
              name: "Dream Motion",
              workflow: "image_to_video",
              model: "ltx_2_3",
              builtInLoras: [{ id: "wan_motion" }],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              family: "ltx-video",
              capabilities: ["image_to_video"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              loraCompatibility: { families: ["ltx-video"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });

    const generate = [...container.querySelectorAll("button")].find((button) => button.textContent === "Render clip");
    expect(container.textContent).toContain("Preset cannot run with LTX");
    expect(container.textContent).toContain("wan_motion");
    expect(generate.disabled).toBe(true);

    await act(async () => {
      generate.click();
    });

    expect(createVideoJob).not.toHaveBeenCalled();
  });

  it("offers a Wan A14B quantization selector and threads the choice into the video job (sc-1982)", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob,
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            rememberLocalGenerationJob: () => {},
            requestedGpu: "auto",
            selectedAsset: null,
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "wan_2_2_t2v_14b",
                name: "Wan2.2 14B (T2V)",
                type: "video",
                family: "wan-video",
                capabilities: ["text_to_video"],
                defaults: { duration: 5, fps: 16, resolution: "832x480", quality: "balanced" },
                limits: { durations: [3, 4, 5], fps: [16], resolutions: ["832x480"] },
                loraCompatibility: { families: ["wan-video"] },
                quantization: {
                  defaults: { mps: "gguf-q8_0", cuda: "gguf-q4_k_m" },
                  variants: {
                    "gguf-q8_0": { format: "gguf", label: "GGUF Q8_0 (near-lossless)" },
                    "gguf-q4_k_m": { format: "gguf", label: "GGUF Q4_K_M (smallest)" },
                  },
                },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    await act(async () => {
      [...container.querySelectorAll(".mode-control button")].find((button) => button.textContent === "Text → Video").click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });

    const quantSelect = field(container, "Quantization");
    expect(quantSelect).toBeTruthy();
    const optionLabels = [...quantSelect.querySelectorAll("option")].map((option) => option.textContent);
    expect(optionLabels).toContain("GGUF Q8_0 (near-lossless)");
    expect(optionLabels).toContain("GGUF Q4_K_M (smallest)");
    expect(optionLabels[0]).toContain("Auto");

    await changeField(quantSelect, "gguf-q4_k_m");
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Render clip").click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "wan_2_2_t2v_14b",
        advanced: expect.objectContaining({ quantization: "gguf-q4_k_m" }),
      }),
    );
  });

  it("surfaces compatible LoRAs in the Video Studio picker and sends the selection to the job", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob,
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [
              { id: "ltx_style", name: "LTX Style", family: "ltx-video", scope: "global", installState: "installed" },
              { id: "z_glow", name: "Z Glow", family: "z-image", scope: "global", installState: "installed" },
            ],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            rememberLocalGenerationJob: () => {},
            requestedGpu: "auto",
            selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                family: "ltx-video",
                capabilities: ["image_to_video"],
                defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
                loraCompatibility: { families: ["ltx-video"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });
    await settle();

    // Only the ltx-video LoRA is compatible; the z-image one is filtered out.
    expect(container.textContent).toContain("LTX Style");
    expect(container.textContent).not.toContain("Z Glow");

    const loraCheckbox = container.querySelector(".lora-choice-list input[type=checkbox]");
    await act(async () => {
      loraCheckbox.click();
    });
    await settle();

    // Selecting a LoRA reveals its weight slider, defaulting to the LoRA weight (0.8).
    const weightSlider = container.querySelector(".lora-weight-row input[type=range]");
    expect(weightSlider).toBeTruthy();
    expect(container.querySelector(".lora-weight-value").textContent).toBe("0.80");
    await changeField(weightSlider, "0.5");

    const generate = [...container.querySelectorAll("button")].find((button) => button.textContent === "Render clip");
    expect(generate.disabled).toBe(false);

    await act(async () => {
      generate.click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx_2_3",
        loras: [expect.objectContaining({ id: "ltx_style", weight: 0.5 })],
      }),
    );
  });

  it("always exposes the preset selector in the Video Studio even with no presets", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob: () => {},
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            requestedGpu: "auto",
            selectedAsset: null,
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                family: "ltx-video",
                capabilities: ["image_to_video", "text_to_video"],
                defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
                loraCompatibility: { families: ["ltx-video"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    expect(container.textContent).toContain("Style preset");
    const noneChip = [...container.querySelectorAll(".preset-chip")].find((chip) => chip.textContent === "None");
    expect(noneChip).toBeTruthy();
  });

  it("keeps Qwen selected when applying a Qwen image preset", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            { id: "qwen_detail", name: "Qwen Detail", model: "qwen_image", workflow: "text_to_image", defaults: { count: 1 } },
            { id: "cinematic", name: "Cinematic", model: "z_image_turbo", workflow: "text_to_image", defaults: { count: 4 } },
          ],
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.textContent).toContain("Qwen Detail");
    expect(container.textContent).not.toContain("Cinematic");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate").click();
    });

    expect(createImageJob).toHaveBeenCalledWith(expect.objectContaining({ model: "qwen_image", recipePresetId: "qwen_detail" }));
  });

  it("offers SenseNova-U1 in edit mode via its edit_image capability", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["text_to_image"] },
            {
              id: "sensenova_u1_8b",
              name: "SenseNova-U1 8B",
              type: "image",
              family: "sensenova-u1",
              capabilities: ["text_to_image", "edit_image"],
              limits: { resolutions: ["2048x2048"] },
            },
            {
              id: "sensenova_u1_8b_fast",
              name: "SenseNova-U1 8B Fast",
              type: "image",
              family: "sensenova-u1",
              capabilities: ["text_to_image", "edit_image"],
              limits: { resolutions: ["2048x2048"] },
            },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll(".segmented-control button")].find((button) => button.textContent === "Edit").click();
    });
    await settle();

    const modelValues = [...field(container, "Model").querySelectorAll("option")].map((option) => option.value);
    expect(modelValues).toContain("sensenova_u1_8b");
    // The distilled fast variant also edits, so it appears in the edit picker.
    expect(modelValues).toContain("sensenova_u1_8b_fast");
    // The text-to-image-only model is filtered out of the edit-mode picker.
    expect(modelValues).not.toContain("z_image_turbo");
    // The selected model resets to an edit-capable one, so Generate doesn't submit
    // the (filtered-out) text default and get rejected by the worker.
    expect(field(container, "Model").value).toBe("sensenova_u1_8b");
  });

  it("uses preset modes as the Image Studio picker surface", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["edit_image"] }],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              modes: ["text_to_image", "edit_image", "character_image"],
            },
            {
              id: "portrait_only",
              name: "Portrait Only",
              model: "z_image_turbo",
              workflow: "text_to_image",
              modes: ["character_image"],
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(container.textContent).toContain("Cinematic");
    expect(container.textContent).not.toContain("Portrait Only");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Edit").click();
    });
    await settle();

    expect(container.textContent).toContain("Cinematic");
    expect(container.textContent).not.toContain("Portrait Only");
  });

  it("drops variations to 1 in edit mode and restores 4 for text", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["text_to_image", "edit_image"] },
          ],
          latestAssets: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          presets: [],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });
    await settle();

    expect(field(container, "Variations").value).toBe("4");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Edit").click();
    });
    await settle();

    expect(field(container, "Variations").value).toBe("1");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Text").click();
    });
    await settle();

    expect(field(container, "Variations").value).toBe("4");
  });

  it("applies preset defaults to video jobs", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [{ id: "video_motion", name: "Video Motion" }],
          setPreviewAsset: () => {},
          rememberLocalGenerationJob: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            {
              id: "dream_motion",
              name: "Dream Motion",
              workflow: "image_to_video",
              model: "ltx_2_3",
              defaults: { duration: 8, fps: 30, resolution: "1280x720", quality: "best", negativePrompt: "jitter" },
              prompt: { suffix: "smooth camera motion" },
              builtInLoras: [{ id: "video_motion" }],
              ui: { description: "Soft camera motion." },
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    expect(container.textContent).toContain("Dream Motion");
    expect(container.textContent).toContain("Soft camera motion.");
    expect(container.textContent).toContain("Adds: smooth camera motion");
    expect(container.textContent).toContain("Preset LoRA applied at generation: Video Motion");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Render clip").click();
    });

    expect(createVideoJob).toHaveBeenCalledWith(
      expect.objectContaining({
        duration: 8,
        fps: 30,
        width: 1280,
        height: 720,
        quality: "best",
        negativePrompt: "jitter",
        recipePresetId: "dream_motion",
        // Preset prompt prefix/suffix and preset LoRAs are now folded in
        // server-side from recipePresetId, so the client sends the raw prompt
        // and only its own picker selections (none here).
        prompt: "Camera slowly pushes in while the scene comes alive",
        loras: [],
        advanced: expect.objectContaining({
          resolution: "1280x720",
        }),
      }),
    );
    const submittedAdvanced = createVideoJob.mock.calls[0][0].advanced;
    expect(submittedAdvanced).not.toHaveProperty("recipePresetName");
    expect(submittedAdvanced).not.toHaveProperty("recipePresetPrompt");
  });

  it("lets a promptless video model (SVD) submit without a text prompt", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob,
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            rememberLocalGenerationJob: () => {},
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            requestedGpu: "auto",
            selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "svd",
                name: "Stable Video Diffusion",
                type: "video",
                family: "svd",
                capabilities: ["image_to_video"],
                promptless: true,
                defaults: { duration: 4, fps: 7, resolution: "1024x576", quality: "balanced" },
                limits: { durations: [4], fps: [6, 7, 8], resolutions: ["1024x576", "576x1024"] },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    // The prompt field advertises that no prompt is needed for promptless models.
    const promptField = container.querySelector("textarea[aria-label='Prompt']");
    expect(promptField.placeholder).toContain("No prompt needed");

    // With a source image selected and an empty prompt, Render clip is enabled
    // and submits (a text-prompted model would be blocked here).
    const generate = [...container.querySelectorAll("button")].find((button) => button.textContent === "Render clip");
    expect(generate.disabled).toBe(false);
    await act(async () => {
      generate.click();
    });
    expect(createVideoJob).toHaveBeenCalled();
  });

  it("filters video presets by mode and selected model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [{ id: "image-1", type: "image", displayName: "Frame One" }],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [],
          setPreviewAsset: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            { id: "ltx_motion", name: "LTX Motion", workflow: "image_to_video", model: "ltx_2_3" },
            { id: "ltx_story", name: "LTX Story", workflow: "text_to_video", model: "ltx_2_3" },
            { id: "wan_motion", name: "Wan Motion", workflow: "image_to_video", model: "wan_2_2" },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
            {
              id: "wan_2_2",
              name: "Wan2.2",
              type: "video",
              capabilities: ["image_to_video", "text_to_video"],
              defaults: { duration: 5, fps: 24, resolution: "1280x720", quality: "balanced" },
              limits: { durations: [4, 5], fps: [24], resolutions: ["1280x720"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    expect(container.textContent).toContain("LTX Motion");
    expect(container.textContent).not.toContain("Wan Motion");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Text → Video").click();
    });
    await settle();

    expect(container.textContent).toContain("LTX Story");
    expect(container.textContent).not.toContain("LTX Motion");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });
    await changeField(field(container, "Model"), "wan_2_2");
    await settle();

    expect(container.textContent).toContain("No preset selected");
    expect(container.textContent).not.toContain("LTX Story");
  });

  it("uses preset modes as the Video Studio picker surface", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          assets: [
            { id: "image-1", type: "image", displayName: "Frame One" },
            { id: "image-2", type: "image", displayName: "Frame Two" },
          ],
          characters: [],
          createPersonDetectionJob: () => {},
          createPersonTrackJob: () => {},
          createVideoJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          latestVideoAssets: [],
          loras: [],
          setPreviewAsset: () => {},
          personTracks: [],
          purgeAsset: () => {},
          presets: [
            {
              id: "camera_bridge",
              name: "Camera Bridge",
              workflow: "image_to_video",
              modes: ["image_to_video", "first_last_frame"],
              model: "ltx_2_3",
            },
            {
              id: "start_frame",
              name: "Start Frame",
              workflow: "image_to_video",
              modes: ["image_to_video"],
              model: "ltx_2_3",
            },
          ],
          requestedGpu: "auto",
          selectedAsset: { id: "image-1", type: "image", displayName: "Frame One" },
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
          videoModels: [
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ],
          },
          <VideoStudio />,
        ),
      );
    });
    await settle();

    expect(container.textContent).toContain("Camera Bridge");
    expect(container.textContent).toContain("Start Frame");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "First → Last").click();
    });
    await settle();

    expect(container.textContent).toContain("Camera Bridge");
    expect(container.textContent).not.toContain("Start Frame");
  });

  it("creates, edits, duplicates, and archives presets from the manager", async () => {
    const createPreset = vi.fn(async (payload) => payload);
    const updatePreset = vi.fn(async (id, payload) => ({ ...payload, id }));
    const duplicatePreset = vi.fn(async (id) => ({ id: `${id}_copy` }));
    const deletePreset = vi.fn(async (id) => ({ id, archived: true }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          createPreset,
          deletePreset,
          duplicatePreset,
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }],
          loras: [
            { id: "cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", defaultWeight: 0.55 },
            { id: "global_detail", name: "Global Detail", family: "z-image", scope: "global", defaultWeight: 0.7 },
            { id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "global" },
          ],
          presets: [
            {
              id: "cinematic",
              name: "Cinematic",
              scope: "builtin",
              workflow: "text_to_image",
              model: "z_image_turbo",
              loras: [{ id: "cinematic_detail", weight: 0.5 }],
              ui: { description: "Built in cinematic finish." },
            },
            {
              id: "moody",
              name: "Moody",
              scope: "global",
              workflow: "text_to_image",
              model: "z_image_turbo",
              ui: { description: "Low key color." },
            },
          ],
          updatePreset,
          videoModels: [{ id: "ltx_2_3", name: "LTX", type: "video" }],
          setActiveView: () => {},
          },
          <PresetManagerScreen />,
        ),
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await changeField(field(container, "Name"), "Soft Morning");
    // New flow: open the LoRA picker, then click the compatible LoRA row to add it.
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("Global Detail"))
        .click();
    });
    await changeField(field(container, "Weight"), "0.35");
    expect(field(container, "ID").value).toBe("soft_morning");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Create Preset").click();
    });
    expect(createPreset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "soft_morning",
        name: "Soft Morning",
        scope: "global",
        loras: [{ id: "global_detail", weight: 0.35 }],
        modes: ["text_to_image", "character_image", "style_variations"],
      }),
    );
    expect(container.textContent).toContain("Ready");
    expect(container.textContent).not.toContain("Qwen Only");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await changeField(field(container, "Name"), "Plain Morning");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".lora-pick-row")]
        .find((button) => button.textContent.includes("Global Detail"))
        .click();
    });
    expect(container.querySelector(".lora-choice-list").textContent).toContain("Global Detail");
    await act(async () => {
      [...container.querySelectorAll(".lora-choice button")].find((button) => button.textContent === "Remove").click();
    });
    expect(container.querySelector(".lora-choice-list")).toBeNull();

    await act(async () => {
      [...container.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await changeField(field(container, "Description"), "Richer low key color.");
    expect(container.textContent).not.toContain("Queue Import");
    expect(field(container, "Source URL")).toBeUndefined();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await act(async () => {
      [...container.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await changeField(field(container, "Description"), "Richer low key color.");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Save Preset").click();
    });
    expect(updatePreset).toHaveBeenCalledWith("moody", expect.objectContaining({ ui: { description: "Richer low key color." } }), "global");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Duplicate").click();
    });
    expect(duplicatePreset).toHaveBeenCalledWith("moody", "global");

    await act(async () => {
      [...container.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Archive").click();
    });
    expect(deletePreset).toHaveBeenCalledWith("moody", "global");
  });

  it("explains preset save blockers and selected LoRA warning states", async () => {
    const updatePreset = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
          activeProject: { id: "project-1", name: "Noir" },
          createPreset: () => {},
          deletePreset: () => {},
          duplicatePreset: () => {},
          imageModels: [],
          loras: [{ id: "pending_style", name: "Pending Style", family: "z-image", scope: "global", installState: "missing" }],
          presets: [
            {
              id: "blocked",
              name: "Blocked",
              scope: "global",
              workflow: "text_to_image",
              model: "z_image_turbo",
              loras: [{ id: "pending_style" }],
            },
          ],
          updatePreset,
          videoModels: [],
          setActiveView: () => {},
          },
          <PresetManagerScreen />,
        ),
      );
    });

    expect(container.textContent).toContain("No models");
    expect(container.textContent).not.toContain("No model selected");
    expect(container.textContent).toContain("Pending Style");
    expect(container.textContent).toContain("Missing or still importing");
    expect(container.textContent).toContain("Save blocked: pending_style has not finished importing.");
    expect(field(container, "Weight").disabled).toBe(true);
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Save Preset").disabled).toBe(true);
    expect(updatePreset).not.toHaveBeenCalled();
  });

  function replacePanelProps(overrides = {}) {
    const track = {
      id: "track_1",
      projectId: "project-1",
      name: "Hero",
      sourceAssetId: "clip-1",
      frames: [
        { timestamp: 0, box: { x: 0.1, y: 0.1, width: 0.2, height: 0.5 }, confidence: 0.92, detected: true, mask: "person-tracks/track_1/masks/frame_000001.png", flags: [] },
        { timestamp: 0.5, box: { x: 0.3, y: 0.1, width: 0.2, height: 0.5 }, confidence: 0.3, detected: true, mask: null, flags: ["low_confidence"] },
      ],
      corrections: [],
      status: { maskState: "active", averageConfidence: 0.6, correctionState: "ready_for_box_corrections" },
    };
    const props = {
      createPersonDetectionJob: () => {},
      createPersonTrackJob: () => {},
      detectionResult: null,
      matchingTracks: [track],
      representativeFrame: null,
      selectedDetection: null,
      selectedTrack: track,
      setPersonTrackId: () => {},
      setReplacementMode: () => {},
      setSelectedDetectionId: () => {},
      setSourceClipAssetId: () => {},
      setTrackName: () => {},
      sourceClipAssetId: "clip-1",
      trackName: "Hero",
      personTrackId: "track_1",
      replacementMode: "full_person_keep_outfit",
      videoAssets: [{ id: "clip-1", type: "video", projectId: "project-1", file: { path: "clip.mp4", mimeType: "video/mp4" } }],
      personReadiness: {},
      ...overrides,
    };
    return { track, props };
  }

  it("scrubs tracked frames and persists a corrected box", async () => {
    const saveTrackCorrections = vi.fn(() => Promise.resolve(null));
    const { props } = replacePanelProps({ saveTrackCorrections });
    root = createRoot(container);
    await act(async () => {
      root.render(<ReplacePersonPanel {...props} />);
    });

    expect(container.textContent).toContain("Review & correct track");
    expect(container.textContent).toContain("Frame 1 / 2");

    // Scrub to the second (low-confidence) frame and confirm the quality flag shows.
    const scrubber = container.querySelector('input[type="range"]');
    await changeField(scrubber, "1");
    expect(container.textContent).toContain("Frame 2 / 2");
    expect(container.textContent).toContain("low confidence");

    // Scrub back and nudge the box X, then save the correction set.
    await changeField(scrubber, "0");
    await changeField(container.querySelector('input[aria-label="Box x"]'), "0.5");

    const save = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save corrections");
    expect(save.disabled).toBe(false);
    await act(async () => {
      save.click();
    });

    expect(saveTrackCorrections).toHaveBeenCalledWith("track_1", [
      { frameIndex: 0, rejected: false, author: "ui", source: "manual", box: { x: 0.5, y: 0.1, width: 0.2, height: 0.5 } },
    ]);
  });

  it("rejects a low-quality frame and records it as a correction", async () => {
    const saveTrackCorrections = vi.fn(() => Promise.resolve(null));
    const { props } = replacePanelProps({ saveTrackCorrections });
    root = createRoot(container);
    await act(async () => {
      root.render(<ReplacePersonPanel {...props} />);
    });

    const scrubber = container.querySelector('input[type="range"]');
    await changeField(scrubber, "1");

    const reject = container.querySelector('.person-correction-reject input[type="checkbox"]');
    await act(async () => {
      reject.click();
    });

    // Rejecting a frame disables its box inputs — replacement borrows a neighbor box.
    expect(container.querySelector('input[aria-label="Box x"]').disabled).toBe(true);

    const save = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save corrections");
    await act(async () => {
      save.click();
    });

    expect(saveTrackCorrections).toHaveBeenCalledWith("track_1", [
      { frameIndex: 1, rejected: true, author: "ui", source: "manual" },
    ]);
  });

  it("shows VQA history and asks a question from the asset detail panel", async () => {
    const createVqaJob = vi.fn();
    const asset = { id: "asset-1", type: "image", displayName: "Frame One", recipe: { prompt: "neon street" } };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [asset],
            jobs: [
              {
                id: "job-vqa-1",
                type: "image_vqa",
                status: "completed",
                payload: { sourceAssetId: "asset-1", question: "What time of day is it?" },
                result: { question: "What time of day is it?", answer: "It appears to be nighttime." },
              },
            ],
            imageModels: [{ id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "vqa"] }],
            createVqaJob,
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: asset,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    // The prior answer is surfaced on the asset.
    expect(container.textContent).toContain("It appears to be nighttime.");

    // Asking a new question dispatches a VQA job for this asset.
    const input = container.querySelector('textarea[aria-label="Ask about this image"]');
    expect(input).not.toBeNull();
    await changeField(input, "What is the person wearing?");
    const askButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "Ask");
    await act(async () => {
      askButton.click();
    });
    // Defaults to the short (256-token) response length.
    expect(createVqaJob).toHaveBeenCalledWith(asset, "What is the person wearing?", 256);

    // Choosing a longer response length is passed through to the job.
    await changeField(container.querySelector('select[aria-label="Response length"]'), "512");
    await changeField(input, "Write a detailed critique of this image.");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Ask").click();
    });
    expect(createVqaJob).toHaveBeenLastCalledWith(asset, "Write a detailed critique of this image.", 512);
  });

  it("filters library assets by tag and edits selected asset tags", async () => {
    const updateAssetTags = vi.fn();
    const portrait = {
      id: "asset-portrait",
      projectId: "project-1",
      type: "image",
      displayName: "Portrait One",
      tags: ["portrait"],
      recipe: { prompt: "studio portrait" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const landscape = {
      id: "asset-landscape",
      projectId: "project-1",
      type: "image",
      displayName: "Wide Hill",
      tags: ["landscape"],
      recipe: { prompt: "wide hill" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Tagged" },
            assets: [portrait, landscape],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: portrait,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags,
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    await changeField(container.querySelector('select[aria-label="Asset tag"]'), "landscape");
    const filteredTiles = [...container.querySelectorAll(".asset-tile")];
    expect(filteredTiles).toHaveLength(1);
    expect(filteredTiles[0].textContent).toContain("Wide Hill");

    await changeField(container.querySelector('input[aria-label="Add asset tag"]'), "  Moody  ");
    await act(async () => {
      [...container.querySelectorAll(".asset-tag-form button")].find((button) => button.textContent === "Add").click();
    });
    expect(updateAssetTags).toHaveBeenCalledWith(portrait, ["portrait", "moody"]);

    await act(async () => {
      container.querySelector('button[aria-label="Remove portrait tag"]').click();
    });
    expect(updateAssetTags).toHaveBeenLastCalledWith(portrait, []);
  });

  it("excludes Character Studio outputs from the Asset Library (sc-2024)", async () => {
    const studioImage = {
      id: "asset-studio",
      projectId: "project-1",
      type: "image",
      displayName: "Studio Render",
      origin: "image_studio",
      recipe: { prompt: "studio render" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const characterImage = {
      id: "asset-character",
      projectId: "project-1",
      type: "image",
      displayName: "Character Test",
      origin: "character_studio",
      recipe: { prompt: "character test" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Scoped" },
            assets: [studioImage, characterImage],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: studioImage,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    const tiles = [...container.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(1);
    expect(tiles[0].textContent).toContain("Studio Render");
    expect(container.textContent).not.toContain("Character Test");
  });

  it("folds original and upscaled library variants into one representative tile", async () => {
    const setPreviewAsset = vi.fn();
    const original = {
      id: "asset-original",
      projectId: "project-1",
      type: "image",
      displayName: "Castle original",
      file: { path: "assets/images/castle-original.png", mimeType: "image/png" },
      recipe: { prompt: "castle" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const upscaled = {
      id: "asset-upscaled",
      projectId: "project-1",
      type: "image",
      displayName: "Castle upscaled",
      file: { path: "assets/images/castle-upscaled.png", mimeType: "image/png" },
      lineage: { sourceAssetId: "asset-original", parents: ["asset-original"] },
      extra: { isUpscaled: true, upscaledFromAssetId: "asset-original", factor: 2, engine: "real-esrgan" },
      recipe: { prompt: "castle" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const other = {
      id: "asset-other",
      projectId: "project-1",
      type: "image",
      displayName: "Other frame",
      file: { path: "assets/images/other.png", mimeType: "image/png" },
      recipe: { prompt: "other" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };

    expect(foldUpscaledAssetVariants([original, upscaled, other]).map((asset) => asset.id)).toEqual([
      "asset-upscaled",
      "asset-other",
    ]);

    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Variants" },
            assets: [original, upscaled, other],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset: () => {},
            importAsset: () => {},
            setPreviewAsset,
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: upscaled,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });

    const tiles = [...container.querySelectorAll(".asset-tile")];
    expect(tiles).toHaveLength(2);
    expect(tiles.map((tile) => tile.textContent).join(" ")).toContain("Castle upscaled");
    expect(tiles.map((tile) => tile.textContent).join(" ")).not.toContain("Castle original");
    expect(tiles[0].querySelector("img").getAttribute("src")).toContain("castle-upscaled.png");

    await act(async () => {
      tiles[0].dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    });

    expect(setPreviewAsset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "asset-upscaled",
        variants: {
          original,
          upscaled,
        },
      }),
    );
  });

  it("shows discarded assets in Trashcan and exposes restore and purge actions", async () => {
    const updateAssetStatus = vi.fn();
    const purgeAsset = vi.fn();
    const active = {
      id: "asset-active",
      projectId: "project-1",
      type: "image",
      displayName: "Active Frame",
      recipe: { prompt: "active" },
      status: { favorite: false, rating: 0, rejected: false, trashed: false },
    };
    const trashed = {
      id: "asset-trash",
      projectId: "project-1",
      type: "image",
      displayName: "Discarded Frame",
      recipe: { prompt: "discarded" },
      status: { favorite: false, rating: 0, rejected: false, trashed: true },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Trash" },
            assets: [active, trashed],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset,
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: trashed,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus,
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    expect([...container.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ")).toContain("Active Frame");
    expect([...container.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ")).not.toContain("Discarded Frame");

    await act(async () => {
      [...container.querySelectorAll('button')].find((button) => button.textContent === "Trashcan").click();
    });
    expect([...container.querySelectorAll(".asset-tile")].map((tile) => tile.textContent).join(" ")).toContain("Discarded Frame");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Restore").click();
    });
    expect(updateAssetStatus).toHaveBeenCalledWith(trashed, { trashed: false });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Purge").click();
    });
    expect(purgeAsset).toHaveBeenCalledWith(trashed);
  });

  it("Empty Trash purges every discarded asset in the Library Trashcan view only", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const purgeAsset = vi.fn();
    const active = {
      id: "asset-active",
      projectId: "project-1",
      type: "image",
      displayName: "Active Frame",
      status: { trashed: false },
    };
    const trashedA = {
      id: "trash-a",
      projectId: "project-1",
      type: "image",
      displayName: "Trash A",
      status: { trashed: true },
    };
    const trashedB = {
      id: "trash-b",
      projectId: "project-1",
      type: "image",
      displayName: "Trash B",
      status: { trashed: true },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Trash" },
            assets: [active, trashedA, trashedB],
            jobs: [],
            imageModels: [],
            createVqaJob: vi.fn(),
            deleteAsset: () => {},
            purgeAsset,
            importAsset: () => {},
            setPreviewAsset: () => {},
            sendAssetToImage: () => {},
            sendAssetToVideo: () => {},
            selectedAsset: null,
            setSelectedAssetId: () => {},
            setActiveView: () => {},
            updateAssetStatus: () => {},
            updateAssetTags: () => {},
          },
          <LibraryScreen />,
        ),
      );
    });
    await settle();

    // Empty Trash only appears in the Trashcan view.
    expect([...container.querySelectorAll("button")].some((button) => button.textContent.startsWith("Empty Trash"))).toBe(false);
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Trashcan").click();
    });
    const emptyButton = [...container.querySelectorAll("button")].find((button) => button.textContent.startsWith("Empty Trash"));
    expect(emptyButton).toBeTruthy();
    expect(emptyButton.textContent).toContain("(2)");

    await act(async () => {
      emptyButton.click();
    });
    expect(confirm).toHaveBeenCalled();
    expect(purgeAsset).toHaveBeenCalledTimes(2);
    expect(purgeAsset).toHaveBeenCalledWith(trashedA);
    expect(purgeAsset).toHaveBeenCalledWith(trashedB);
    expect(purgeAsset).not.toHaveBeenCalledWith(active);
    confirm.mockRestore();
  });

  it("DocumentStudio renders an interleaved document and submits a compose job", async () => {
    const createInterleaveJob = vi.fn(() =>
      Promise.resolve({ id: "job-il-new", type: "image_interleave", status: "queued" }),
    );
    const setActiveView = vi.fn();
    const rememberLocalGenerationJob = vi.fn();
    const imageAsset = {
      id: "img-1",
      type: "image",
      projectId: "project-1",
      file: { path: "assets/images/a.png" },
      url: "/api/v1/projects/project-1/files/assets/images/a.png",
    };
    const completedJob = {
      id: "job-il-done",
      type: "image_interleave",
      status: "completed",
      payload: { prompt: "tea guide" },
      result: {
        segments: [
          { type: "text", text: "Boil the water." },
          { type: "image", assetId: "img-1", path: "assets/images/a.png" },
          { type: "text", text: "Steep three minutes." },
        ],
      },
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [imageAsset],
            createInterleaveJob,
            documentLocalJobs: [completedJob],
            gpuOptions: ["auto"],
            imageModels: [
              { id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "interleave"] },
              { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", capabilities: ["text_to_image"] },
            ],
            jobAction: () => {},
            rememberLocalGenerationJob,
            setActiveView,
            requestedGpu: "auto",
            setRequestedGpu: () => {},
          },
          <DocumentStudio />,
        ),
      );
    });
    await settle();

    // The completed document renders text segments in order + the image segment.
    expect(container.textContent).toContain("Boil the water.");
    expect(container.textContent).toContain("Steep three minutes.");
    const image = container.querySelector("img.document-image");
    expect(image).not.toBeNull();
    expect(image.getAttribute("src")).toContain("assets/images/a.png");

    // Only interleave-capable models are offered (Z-Image filtered out).
    const optionValues = [...container.querySelectorAll("select option")].map((option) => option.value);
    expect(optionValues).toContain("sensenova_u1_8b");
    expect(optionValues).not.toContain("z_image_turbo");
    // The size control offers the interleave buckets.
    expect(optionValues).toContain("2048x1152");
    // The system prompt is exposed and prefilled with the default.
    const textareas = [...container.querySelectorAll("textarea")];
    expect(textareas.some((field) => field.value.includes("multimodal assistant capable of reasoning"))).toBe(true);

    // Submitting composes an interleave job with prompt, model, size, and max images.
    await changeField(container.querySelector("textarea"), "An illustrated guide to brewing tea");
    const submit = [...container.querySelectorAll("button")].find((button) =>
      button.textContent.includes("Compose document"),
    );
    await act(async () => {
      submit.click();
    });
    expect(createInterleaveJob).toHaveBeenCalledTimes(1);
    const payload = createInterleaveJob.mock.calls[0][0];
    expect(payload.prompt).toBe("An illustrated guide to brewing tea");
    expect(payload.model).toBe("sensenova_u1_8b");
    expect(payload.maxImages).toBe(6);
    expect(payload.width).toBe(2048);
    expect(payload.height).toBe(1152);
    // Unedited system prompt is not sent (worker uses its own default).
    expect(payload.advanced?.systemMessage).toBeUndefined();

    // Submitting stacks the run in the studio rather than routing to the Queue.
    await settle();
    expect(rememberLocalGenerationJob).toHaveBeenCalledWith(
      "document",
      expect.objectContaining({ id: "job-il-new" }),
    );
    expect(setActiveView).not.toHaveBeenCalledWith("Queue");
  });

  it("DocumentStudio stacks a queued compose run beneath the active document", async () => {
    const completedJob = {
      id: "doc-job-done",
      type: "image_interleave",
      status: "completed",
      createdAt: "2026-05-27T10:00:00Z",
      payload: { prompt: "tea guide" },
      result: {
        segments: [
          { type: "text", text: "Boil the water." },
          { type: "text", text: "Steep three minutes." },
        ],
      },
    };
    const queuedJob = {
      id: "doc-job-queued",
      type: "image_interleave",
      status: "queued",
      createdAt: "2026-05-27T10:01:00Z",
      payload: { prompt: "coffee guide" },
      result: {},
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            createInterleaveJob: () => {},
            documentLocalJobs: [completedJob, queuedJob],
            gpuOptions: ["auto"],
            imageModels: [
              { id: "sensenova_u1_8b", name: "SenseNova-U1 8B", type: "image", capabilities: ["text_to_image", "interleave"] },
            ],
            jobAction: () => {},
            rememberLocalGenerationJob: () => {},
            setActiveView: () => {},
            requestedGpu: "auto",
            setRequestedGpu: () => {},
          },
          <DocumentStudio />,
        ),
      );
    });
    await settle();

    // Both runs stack: the finished document plus the queued run's progress card.
    expect(container.querySelectorAll(".local-job-group").length).toBe(2);
    expect(container.textContent).toContain("Boil the water.");
    expect(container.querySelector(".document-view")).not.toBeNull();
    expect(container.querySelector(".worker-progress-card.queued")).not.toBeNull();
    expect(container.textContent).not.toContain("Your generated document will appear here.");
  });

  it("AssetDetail reopens a saved document from the Library", async () => {
    global.fetch.mockImplementation((url) => {
      if (String(url).includes("assets/documents")) {
        return Promise.resolve(
          response({
            schemaVersion: 1,
            id: "doc_1",
            segments: [
              { type: "text", text: "Boil the water." },
              { type: "image", assetId: "img-1", path: "assets/images/a.png" },
              { type: "text", text: "Steep three minutes." },
            ],
          }),
        );
      }
      return Promise.resolve(response([]));
    });
    const documentAsset = {
      id: "doc_1",
      type: "document",
      projectId: "project-1",
      displayName: "Tea guide",
      file: { path: "assets/documents/doc_1.json", mimeType: "application/json" },
      url: "/api/v1/projects/project-1/files/assets/documents/doc_1.json",
    };
    root = createRoot(container);
    await act(async () => {
      root.render(
        <AssetDetail
          asset={documentAsset}
          deleteAsset={() => {}}
          purgeAsset={() => {}}
          onPreview={() => {}}
          onSendImage={() => {}}
          onSendVideo={() => {}}
          onSendEditor={() => {}}
          updateAssetStatus={() => {}}
        />,
      );
    });
    await settle();

    // The saved document's text + image segments render in order.
    expect(container.textContent).toContain("Boil the water.");
    expect(container.textContent).toContain("Steep three minutes.");
    const image = container.querySelector("img.document-image");
    expect(image).not.toBeNull();
    expect(image.getAttribute("src")).toContain("assets/images/a.png");
    // The document JSON was fetched from its file path (with an abort signal).
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("assets/documents/doc_1.json"),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });
});

describe("extractFamilies", () => {
  it("reads the supported shapes in precedence order", () => {
    expect(extractFamilies({ families: ["a"], compatibleFamilies: ["b"] })).toEqual(["a"]);
    expect(extractFamilies({ compatibleFamilies: ["b"], modelFamilies: ["c"] })).toEqual(["b"]);
    expect(extractFamilies({ modelFamilies: ["c"] })).toEqual(["c"]);
    expect(extractFamilies({ compatibility: { families: ["d"] } })).toEqual(["d"]);
    expect(extractFamilies({ family: "e" })).toEqual(["e"]);
  });

  it("returns an empty array when nothing is set", () => {
    expect(extractFamilies(undefined)).toEqual([]);
    expect(extractFamilies({})).toEqual([]);
  });

  it("returns raw values without normalizing casing or separators", () => {
    expect(extractFamilies({ families: ["Z_Image", "Qwen-Image"] })).toEqual(["Z_Image", "Qwen-Image"]);
  });

  it("ignores manifest metadata unless includeManifest is set", () => {
    const job = { payload: { manifestEntry: { families: ["z-image"] }, family: "qwen-image" } };
    expect(extractFamilies(job)).toEqual([]);
    expect(extractFamilies(job, { includeManifest: true })).toEqual(["z-image"]);
  });

  it("falls back through manifest fields then payload.family", () => {
    expect(extractFamilies({ payload: { family: "qwen-image" } }, { includeManifest: true })).toEqual(["qwen-image"]);
    expect(
      extractFamilies({ payload: { manifestEntry: { compatibility: { families: ["z-image"] } } } }, { includeManifest: true }),
    ).toEqual(["z-image"]);
  });

  it("prefers top-level fields over manifest even with includeManifest", () => {
    const job = { families: ["top"], payload: { manifestEntry: { families: ["manifest"] } } };
    expect(extractFamilies(job, { includeManifest: true })).toEqual(["top"]);
  });
});

describe("prompt guide popup (sc-1817)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    // Prompt guides are static assets fetched by path; echo the path back so
    // each guide's rendered body is distinguishable in assertions.
    global.fetch = vi.fn((url) => Promise.resolve({ ok: true, text: async () => `# Guide\n\nfetched ${url}` }));
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  const guideButton = () =>
    [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === "Prompt guide");

  it("opens the selected image model's guide without submitting the form", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Z Guide", path: "/prompt-guides/z-image-turbo.md" } },
            },
            {
              id: "qwen_image",
              name: "Qwen",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Qwen Guide", path: "/prompt-guides/qwen-image.md" } },
            },
          ],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();

    expect(container.querySelector("[role=dialog]")).not.toBeNull();
    expect(container.querySelector("#prompt-guide-title").textContent).toBe("Z Guide");
    expect(container.textContent).toContain("fetched /prompt-guides/z-image-turbo.md");
    expect(createImageJob).not.toHaveBeenCalled();
  });

  it("renders the new model's guide after switching models", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Z Guide", path: "/prompt-guides/z-image-turbo.md" } },
            },
            {
              id: "qwen_image",
              name: "Qwen",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Qwen Guide", path: "/prompt-guides/qwen-image.md" } },
            },
          ],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();
    expect(container.querySelector("#prompt-guide-title").textContent).toBe("Z Guide");

    await act(async () => {
      container.querySelector(".modal-close").click();
    });
    await changeField(field(container, "Model"), "qwen_image");
    await settle();

    await act(async () => {
      guideButton().click();
    });
    await settle();
    expect(container.querySelector("#prompt-guide-title").textContent).toBe("Qwen Guide");
    expect(container.textContent).toContain("fetched /prompt-guides/qwen-image.md");
  });

  it("falls back to the generic image guide when the model declares none", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [{ id: "z_image_turbo", name: "Z-Image", type: "image", capabilities: ["text_to_image"] }],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();

    expect(global.fetch).toHaveBeenCalledWith("/prompt-guides/generic-image.md");
    expect(container.querySelector("#prompt-guide-title").textContent).toBe("Image Prompt Guide");
  });

  it("falls back to the generic video guide and does not submit the video form", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        withAppContext(
          {
            activeProject: { id: "project-1", name: "Noir" },
            assets: [],
            characters: [],
            createPersonDetectionJob: () => {},
            createPersonTrackJob: () => {},
            createVideoJob,
            deleteAsset: () => {},
            gpuOptions: ["auto"],
            latestVideoAssets: [],
            loras: [],
            setPreviewAsset: () => {},
            personTracks: [],
            purgeAsset: () => {},
            presets: [],
            rememberLocalGenerationJob: () => {},
            requestedGpu: "auto",
            selectedAsset: null,
            setRequestedGpu: () => {},
            updateAssetStatus: () => {},
            videoModels: [
              {
                id: "ltx_2_3",
                name: "LTX",
                type: "video",
                family: "ltx-video",
                capabilities: ["text_to_video"],
                defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              },
            ],
          },
          <VideoStudio />,
        ),
      );
    });

    await act(async () => {
      guideButton().click();
    });
    await settle();

    expect(global.fetch).toHaveBeenCalledWith("/prompt-guides/generic-video.md");
    expect(container.querySelector("#prompt-guide-title").textContent).toBe("Video Prompt Guide");
    expect(createVideoJob).not.toHaveBeenCalled();
  });
});

describe("refine my prompt (sc-2041)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, text: async () => "# Guide\n\nWrite vividly." }));
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("refines the image prompt and applies it to the textarea on Apply", async () => {
    const refinePrompt = vi.fn(async () => "A cinematic neon street at midnight, rain-slick.");
    root = createRoot(container);
    await act(async () => {
      root.render(
        withImageStudioContext({
          activeProject: { id: "project-1", name: "Noir" },
          assets: [],
          characters: [],
          createImageJob: () => {},
          refinePrompt,
          deleteAsset: () => {},
          gpuOptions: ["auto"],
          imageModels: [
            {
              id: "z_image_turbo",
              name: "Z-Image",
              type: "image",
              capabilities: ["text_to_image"],
              ui: { promptGuide: { title: "Z Guide", path: "/prompt-guides/z-image-turbo.md" } },
            },
          ],
          latestAssets: [],
          localJobs: [],
          loras: [],
          onPreview: () => {},
          purgeAsset: () => {},
          requestedGpu: "auto",
          selectedAsset: null,
          setRequestedGpu: () => {},
          updateAssetStatus: () => {},
        }),
      );
    });

    const refine = [...container.querySelectorAll("button")].find((button) => button.textContent.includes("Refine my prompt"));
    await act(async () => {
      refine.click();
    });
    await settle();

    expect(refinePrompt).toHaveBeenCalledWith({
      prompt: "A cinematic frame of a neon street at midnight",
      modelId: "z_image_turbo",
      workflow: "image",
      guide: "# Guide\n\nWrite vividly.",
    });
    expect(container.querySelector(".refine-review-text").textContent).toBe("A cinematic neon street at midnight, rain-slick.");
    // Original prompt unchanged until the user applies.
    expect(container.querySelector(".prompt-input").value).toBe("A cinematic frame of a neon street at midnight");

    const apply = [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === "Apply");
    await act(async () => {
      apply.click();
    });
    await settle();

    expect(container.querySelector(".prompt-input").value).toBe("A cinematic neon street at midnight, rain-slick.");
    expect(container.querySelector(".refine-review")).toBeNull();
  });
});
