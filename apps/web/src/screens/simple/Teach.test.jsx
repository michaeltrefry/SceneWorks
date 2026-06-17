import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Teach } from "./Teach.jsx";
import { AppContext } from "../../context/AppContext.js";

const DEFAULTS = {
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
  triggerWord: "",
  advanced: { mixedPrecision: "bf16", networkType: "lora", qualityPreset: "balanced", outputScope: "project", requestedGpu: "auto" },
};

const TRAINING_TARGETS = {
  schemaVersion: 1,
  targets: [
    {
      id: "z_image_turbo_lora",
      name: "Z-Image-Turbo LoRA",
      modality: "image",
      outputKind: "lora",
      family: "z-image",
      baseModel: "z_image_turbo",
      kernel: "z_image_lora",
      defaults: DEFAULTS,
      limits: { qualityPresets: ["fast", "balanced", "best"], resolutions: [768, 1024], outputScopes: ["project"] },
      ui: {},
    },
  ],
};

const TRAINING_PRESETS = {
  schemaVersion: 1,
  presets: [
    { id: "zi_person", version: 1, targetId: "z_image_turbo_lora", name: "Person", recommendedFor: ["person", "character"], optimizer: "adamw8bit", qualityPreset: "balanced", config: DEFAULTS, ui: { order: 0, default: true } },
  ],
};

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function render(value) {
  return act(() => {
    root.render(<AppContext.Provider value={value}>{<Teach />}</AppContext.Provider>);
  });
}

function setInputValue(input, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(input, value);
  input.dispatchEvent(new window.Event("input", { bubbles: true }));
}

function findButton(text) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent.includes(text));
}

function imageFiles(count) {
  return Array.from({ length: count }, (_, index) => new File([new Uint8Array([index + 1])], `ex_${index}.png`, { type: "image/png" }));
}

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-a", name: "Project A" },
    models: [{ id: "z_image_turbo", installState: "installed" }],
    loras: [],
    jobs: [],
    trainingTargets: TRAINING_TARGETS,
    trainingPresets: TRAINING_PRESETS,
    uploadTrainingDatasetItem: vi.fn(async (file) => ({
      id: `up_${file.name}`,
      datasetOnly: true,
      displayName: file.name,
      file: { path: `training/uploads/${file.name}`, mimeType: "image/png" },
    })),
    createTrainingDataset: vi.fn(async () => ({ id: "ds_1", version: 1 })),
    createTrainingDatasetCaptionJob: vi.fn(async () => ({ id: "caption_1" })),
    createTrainingJob: vi.fn(async () => ({ id: "train_1" })),
    loadTrainingDataset: vi.fn(async () => ({ id: "ds_1", version: 2 })),
    createModelDownloadJob: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

describe("Teach", () => {
  it("renders the guided form when a trainable base model is installed", async () => {
    await render(baseContext());
    expect(container.textContent).toContain("What are you teaching it?");
    expect(container.textContent).toContain("A person");
    expect(container.textContent).toContain("A style");
    expect(container.textContent).toContain("An object");
    expect(findButton("Start teaching")).toBeTruthy();
  });

  it("points the user to Settings when no base model is installed", async () => {
    const ctx = baseContext({ models: [{ id: "z_image_turbo", installState: "missing" }] });
    await render(ctx);
    expect(container.textContent).toContain("add a model to teach from");
    await act(() => findButton("Add a model").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(ctx.setActiveView).toHaveBeenCalledWith("SimpleSettings");
  });

  it("runs upload → dataset → describe → train, and trains once captioning completes", async () => {
    const ctx = baseContext();
    await render(ctx);

    // Add examples via the hidden file input.
    const fileInput = container.querySelector("input[type=file]");
    const files = imageFiles(5);
    await act(() => {
      Object.defineProperty(fileInput, "files", { configurable: true, value: files });
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });

    // Name it — the trigger word auto-derives from the name.
    const nameInput = container.querySelector(".sw-input");
    await act(() => setInputValue(nameInput, "Mara"));

    await act(() => findButton("Start teaching").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    await settle();

    // Each example uploaded; dataset created from path-based (dataset-owned) items.
    expect(ctx.uploadTrainingDatasetItem).toHaveBeenCalledTimes(5);
    expect(ctx.createTrainingDataset).toHaveBeenCalledTimes(1);
    const datasetPayload = ctx.createTrainingDataset.mock.calls[0][0];
    expect(datasetPayload.name).toBe("Mara");
    expect(datasetPayload.items).toHaveLength(5);
    expect(datasetPayload.items[0]).toMatchObject({ path: "training/uploads/ex_0.png" });

    // Captioning is queued; training has NOT started yet.
    expect(ctx.createTrainingDatasetCaptionJob).toHaveBeenCalledTimes(1);
    expect(ctx.createTrainingJob).not.toHaveBeenCalled();

    // Caption job completes → training starts against the reloaded dataset version.
    await render({ ...ctx, jobs: [{ id: "caption_1", type: "training_caption", status: "completed", progress: 1 }] });
    await settle();

    expect(ctx.loadTrainingDataset).toHaveBeenCalledWith("ds_1");
    expect(ctx.createTrainingJob).toHaveBeenCalledTimes(1);
    const request = ctx.createTrainingJob.mock.calls[0][0];
    expect(request).toMatchObject({ targetId: "z_image_turbo_lora", datasetId: "ds_1", datasetVersion: 2, dryRun: false });
    expect(request.config.triggerWord).toBe("mara_person");
    expect(request.config.steps).toBeGreaterThan(0);
  });

  it("skips captioning and trains directly when the describer model is unavailable on a gated Mac", async () => {
    const ctx = baseContext({
      macCapabilities: { macGatingActive: true, training: { supportedKernels: ["z_image_lora"], lokrOnWanSupported: false } },
      models: [
        { id: "z_image_turbo", installState: "installed" },
        { id: "joycaption_beta_one", installState: "missing" },
      ],
    });
    await render(ctx);

    const fileInput = container.querySelector("input[type=file]");
    await act(() => {
      Object.defineProperty(fileInput, "files", { configurable: true, value: imageFiles(5) });
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await act(() => setInputValue(container.querySelector(".sw-input"), "Vale"));
    await act(() => findButton("Start teaching").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    await settle();

    expect(ctx.createTrainingDatasetCaptionJob).not.toHaveBeenCalled();
    expect(ctx.createTrainingJob).toHaveBeenCalledTimes(1);
  });
});
