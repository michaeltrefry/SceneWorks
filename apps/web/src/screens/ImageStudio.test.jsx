import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Pose loaders fetch best-effort on mount; stub the API so render never touches
// the network. The studio's own mutations go through context fns, not apiFetch.
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import {
  buildStructuredPromptRecipe,
  parseMagicPromptCaption,
  serializeCaption,
} from "../ideogramCaption.js";
import { PROMPT_REFINE_MODEL_ID } from "../constants.js";
import { ImageStudio } from "./ImageStudio.jsx";

const Z_IMAGE = {
  id: "z_image_turbo",
  name: "Z Image Turbo",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image"],
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024", "1536x1024"] },
  loraCompatibility: {},
  ui: {},
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [Z_IMAGE],
    importAsset: vi.fn(),
    latestImageAssets: [],
    recentImageAssets: [],
    studioLaunch: null,
    imageLocalJobs: [],
    loras: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setPreviewAsset: vi.fn(),
    presets: [],
    requestedGpu: "",
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    ...overrides,
  };
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

function setInput(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

function setSelect(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("change", { bubbles: true }));
}

function setFileInput(element, files) {
  Object.defineProperty(element, "files", {
    configurable: true,
    value: files,
  });
  element.dispatchEvent(new window.Event("change", { bubbles: true }));
}

const saveButton = (container) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Save as Preset"));
const nameInput = (container) => container.querySelector('input[aria-label="Preset name"]');
const field = (container, labelText) => {
  const label = [...container.querySelectorAll("label")].find((node) =>
    node.textContent.trim().startsWith(labelText),
  );
  return label?.querySelector("input, select");
};

describe("ImageStudio Save as Preset", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush mount effects (pose loaders, etc.)
  }

  it("snapshots the current config into a preset payload without the seed", async () => {
    const context = baseContext();
    await render(context);

    const input = nameInput(container);
    expect(input).toBeTruthy();
    await act(async () => setInput(input, "Atrium Look"));
    await click(saveButton(container));

    expect(context.createPreset).toHaveBeenCalledTimes(1);
    const payload = context.createPreset.mock.calls[0][0];
    expect(payload).toMatchObject({
      id: "atrium_look",
      name: "Atrium Look",
      scope: "project",
      workflow: "text_to_image",
      model: "z_image_turbo",
    });
    // The literal prompt rides in defaults; the seed never does.
    expect(payload.defaults.prompt).toBe("A cinematic frame of a neon street at midnight");
    expect(payload.defaults).not.toHaveProperty("seed");
    expect(container.textContent).toContain('Saved "Atrium Look" to this project.');
  });

  it("blocks a duplicate name client-side before calling the API", async () => {
    const context = baseContext({
      presets: [
        {
          id: "atrium_look",
          name: "Atrium Look",
          scope: "project",
          workflow: "text_to_image",
          model: "z_image_turbo",
          modes: ["text_to_image", "character_image", "style_variations"],
        },
      ],
    });
    await render(context);

    await act(async () => setInput(nameInput(container), "Atrium Look"));
    await click(saveButton(container));

    expect(context.createPreset).not.toHaveBeenCalled();
    expect(container.textContent).toContain("already exists");
  });
});

describe("ImageStudio advanced model defaults", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("resets advanced overrides to the newly selected model defaults", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [
          {
            ...Z_IMAGE,
            defaults: {
              resolution: "1024x1024",
              sampler: "euler",
              scheduler: "shift",
              schedulerShift: 1.5,
              steps: 12,
              guidanceScale: 2.5,
            },
            limits: {
              resolutions: ["1024x1024", "1536x1024"],
              samplers: ["default", "euler", "unipc"],
              schedulers: ["default", "shift", "karras"],
            },
          },
          {
            id: "qwen_image",
            name: "Qwen Image",
            type: "image",
            family: "qwen-image",
            capabilities: ["text_to_image"],
            defaults: {
              resolution: "1536x1024",
              sampler: "unipc",
              scheduler: "shift",
              schedulerShift: 4.2,
              steps: 28,
              guidanceScale: 6.5,
            },
            limits: {
              resolutions: ["1024x1024", "1536x1024"],
              samplers: ["default", "euler", "unipc"],
              schedulers: ["default", "shift", "karras"],
            },
            loraCompatibility: {},
            ui: {},
          },
        ],
      }),
    );

    await click([...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced"));
    await act(async () => setSelect(field(container, "Sampler"), "euler"));
    await act(async () => setSelect(field(container, "Scheduler"), "shift"));
    await act(async () => setInput(field(container, "Schedule shift"), "7.7"));
    await act(async () => setInput(field(container, "Steps"), "44"));
    await act(async () => setInput(field(container, "Guidance"), "11"));

    await act(async () => setSelect(field(container, "Model"), "qwen_image"));
    await act(async () => {});

    expect(field(container, "Sampler").value).toBe("unipc");
    expect(field(container, "Scheduler").value).toBe("shift");
    expect(field(container, "Schedule shift").value).toBe("4.2");
    expect(field(container, "Steps").value).toBe("");
    expect(field(container, "Steps").placeholder).toBe("28");
    expect(field(container, "Guidance").value).toBe("");
    expect(field(container, "Guidance").placeholder).toBe("6.5");

    await click([...container.querySelectorAll("button")].find((button) => button.textContent === "Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.model).toBe("qwen_image");
    expect(payload.advanced).toMatchObject({
      resolution: "1536x1024",
      sampler: "unipc",
      scheduler: "shift",
      schedulerShift: 4.2,
    });
    expect(payload.advanced).not.toHaveProperty("steps");
    expect(payload.advanced).not.toHaveProperty("guidanceScale");
  });
});

describe("ImageStudio edit source picker", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  async function openEditSourcePicker(context) {
    await render(context);
    await click([...container.querySelectorAll(".segmented-control button")].find((button) => button.textContent === "Edit"));
    await click([...container.querySelectorAll(".asset-picker-head button")].find((button) => button.textContent === "Select image"));
    return container.querySelector('[role="dialog"]');
  }

  it("limits Image Edit source selection to active project images and shows the requested source tabs", async () => {
    const active = { id: "asset-active", projectId: "project_1", type: "image", displayName: "Active Plate", status: { trashed: false } };
    const trashed = { id: "asset-trashed", projectId: "project_1", type: "image", displayName: "Discarded Plate", status: { trashed: true } };
    const rejected = { id: "asset-rejected", projectId: "project_1", type: "image", displayName: "Rejected Plate", status: { rejected: true } };
    const otherProject = { id: "asset-other", projectId: "project_2", type: "image", displayName: "Other Project Plate", status: {} };
    const video = { id: "asset-video", projectId: "project_1", type: "video", displayName: "Video Clip", status: {} };

    const dialog = await openEditSourcePicker(
      baseContext({
        assets: [active, trashed, rejected, otherProject, video],
        imageModels: [{ ...Z_IMAGE, capabilities: ["edit_image"] }],
        selectedAsset: null,
      }),
    );

    const sourceTabs = [...dialog.querySelectorAll('[role="tab"]')].map((button) => button.textContent.trim());
    expect(sourceTabs).toEqual(["Assets1", "File Upload", "Character0"]);
    expect(dialog.textContent).toContain("Active Plate");
    expect(dialog.textContent).not.toContain("Discarded Plate");
    expect(dialog.textContent).not.toContain("Rejected Plate");
    expect(dialog.textContent).not.toContain("Other Project Plate");
    expect(dialog.textContent).not.toContain("Video Clip");
    expect(dialog.textContent).not.toContain("Renders");
  });

  it("filters the Character source tab by project, character, and active status", async () => {
    const mira = { id: "char-1", name: "Mira", approvedReferences: [{ assetId: "ref-mira" }] };
    const echo = { id: "char-2", name: "Echo", approvedReferences: [] };
    const assets = [
      { id: "ref-mira", projectId: "project_1", type: "image", displayName: "Mira Reference", status: {} },
      {
        id: "mira-render",
        projectId: "project_1",
        type: "image",
        displayName: "Mira Render",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: {},
      },
      {
        id: "mira-trash",
        projectId: "project_1",
        type: "image",
        displayName: "Mira Discarded",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: true },
      },
      {
        id: "echo-render",
        projectId: "project_1",
        type: "image",
        displayName: "Echo Render",
        recipe: { normalizedSettings: { characterId: "char-2" } },
        status: {},
      },
      {
        id: "mira-other-project",
        projectId: "project_2",
        type: "image",
        displayName: "Mira Elsewhere",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: {},
      },
    ];

    const dialog = await openEditSourcePicker(
      baseContext({
        assets,
        characters: [mira, echo],
        imageModels: [{ ...Z_IMAGE, capabilities: ["edit_image"] }],
        selectedAsset: null,
      }),
    );

    await click([...dialog.querySelectorAll('[role="tab"]')].find((button) => button.textContent.includes("Character")));
    expect(dialog.textContent).toContain("Mira Reference");
    expect(dialog.textContent).toContain("Mira Render");
    expect(dialog.textContent).not.toContain("Mira Discarded");
    expect(dialog.textContent).not.toContain("Echo Render");
    expect(dialog.textContent).not.toContain("Mira Elsewhere");

    await act(async () => {
      dialog.querySelector(".asset-picker-card").click();
    });
    await click([...dialog.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));

    expect(container.textContent).toContain("Mira Reference");
  });

  it("imports a File Upload source and submits it as the edit source image", async () => {
    const imported = {
      id: "uploaded-source",
      projectId: "project_1",
      type: "image",
      displayName: "uploaded.png",
      status: {},
    };
    const importAsset = vi.fn(async () => imported);
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));

    const dialog = await openEditSourcePicker(
      baseContext({
        assets: [],
        createImageJob,
        imageModels: [{ ...Z_IMAGE, capabilities: ["edit_image"] }],
        importAsset,
        selectedAsset: null,
      }),
    );

    await click([...dialog.querySelectorAll('[role="tab"]')].find((button) => button.textContent === "File Upload"));
    const file = new File(["image"], "source.png", { type: "image/png" });
    await act(async () => setFileInput(dialog.querySelector('input[type="file"]'), [file]));
    await act(async () => {});

    expect(importAsset).toHaveBeenCalledWith(file, { throwOnError: true });
    expect(container.querySelector('[role="dialog"]')).toBeNull();

    await click([...container.querySelectorAll("button")].find((button) => button.textContent === "Generate"));
    expect(createImageJob).toHaveBeenCalledWith(expect.objectContaining({ mode: "edit_image", sourceAssetId: "uploaded-source" }));
  });

  it("uses the multi-image reference picker for a multiReference model and submits referenceAssetIds (sc-6211)", async () => {
    const refA = { id: "ref-a", projectId: "project_1", type: "image", displayName: "Ref A", status: {} };
    const refB = { id: "ref-b", projectId: "project_1", type: "image", displayName: "Ref B", status: {} };
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const FLUX2_DEV = {
      ...Z_IMAGE,
      id: "flux2_dev",
      name: "FLUX.2 dev",
      capabilities: ["text_to_image", "edit_image"],
      ui: { multiReference: true },
    };

    await render(
      baseContext({
        assets: [refA, refB],
        createImageJob,
        imageModels: [FLUX2_DEV],
        selectedAsset: null,
      }),
    );
    await click([...container.querySelectorAll(".segmented-control button")].find((button) => button.textContent === "Edit"));

    // The multi-image picker ("Select images") replaces the single source picker ("Select image").
    const headButtons = () => [...container.querySelectorAll(".asset-picker-head button")];
    expect(headButtons().some((button) => button.textContent === "Select images")).toBe(true);
    expect(headButtons().some((button) => button.textContent === "Select image")).toBe(false);

    await click(headButtons().find((button) => button.textContent === "Select images"));
    const dialog = container.querySelector('[role="dialog"]');
    const cards = [...dialog.querySelectorAll(".asset-picker-card")];
    await click(cards[0]);
    await click(cards[1]);
    await click([...dialog.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));

    await click([...container.querySelectorAll("button")].find((button) => button.textContent === "Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.mode).toBe("edit_image");
    expect(payload.referenceAssetIds).toEqual(["ref-a", "ref-b"]);
    expect(payload.sourceAssetId).toBeNull();
  });
});

describe("ImageStudio model picker capability gating", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // One model per capability class so each mode's picker can be checked in isolation.
  const T2I = { ...Z_IMAGE, id: "t2i_only", name: "T2I Only", capabilities: ["text_to_image"] };
  const VARIATIONS = {
    ...Z_IMAGE,
    id: "variations_model",
    name: "Variations Model",
    capabilities: ["text_to_image", "style_variations"],
  };
  const EDIT_ONLY = { ...Z_IMAGE, id: "edit_only", name: "Edit Only", capabilities: ["edit_image", "image_to_image"] };
  const CHARACTER_ONLY = { ...Z_IMAGE, id: "character_only", name: "Character Only", capabilities: ["character_image"] };
  const MAC_CAPS = {
    macGatingActive: true,
    platform: "darwin",
    notAvailableLabel: "Not available on Mac (Rust/MLX only)",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };
  const LENS_TURBO = {
    ...Z_IMAGE,
    id: "lens_turbo",
    name: "Lens-Turbo",
    capabilities: ["text_to_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };
  const QWEN_EDIT = {
    ...Z_IMAGE,
    id: "qwen_image_edit",
    name: "Qwen Image Edit",
    capabilities: ["edit_image"],
    macSupport: { supported: true, features: { edit: true, reference: false } },
  };
  const TORCH_ONLY_EDIT = {
    ...Z_IMAGE,
    id: "torch_only_edit",
    name: "Torch-only Edit",
    capabilities: ["edit_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };

  const modelOptionValues = () => [...field(container, "Model").options].map((option) => option.value);
  const modeButton = (label) =>
    [...container.querySelectorAll(".segmented-control button")].find((button) => button.textContent === label);

  it("Text tab lists only text_to_image models, excluding edit-only and character-only (sc-5549)", async () => {
    await render(baseContext({ imageModels: [EDIT_ONLY, T2I, VARIATIONS, CHARACTER_ONLY] }));

    const options = modelOptionValues();
    expect(options).toContain("t2i_only");
    expect(options).toContain("variations_model"); // declares text_to_image
    expect(options).not.toContain("edit_only");
    expect(options).not.toContain("character_only");
  });

  it("enables the Mac Edit tab when any available model supports edit mode (sc-5589)", async () => {
    await render(
      baseContext({
        imageModels: [LENS_TURBO, TORCH_ONLY_EDIT, QWEN_EDIT],
        macCapabilities: MAC_CAPS,
      }),
    );

    expect(field(container, "Model").value).toBe("lens_turbo");
    expect(modeButton("Edit").disabled).toBe(false);

    await click(modeButton("Edit"));
    await act(async () => {});

    expect(modeButton("Edit").className).toContain("active");
    expect(field(container, "Model").value).toBe("qwen_image_edit");
    expect(modelOptionValues()).toEqual(["qwen_image_edit"]);
  });

  it("disables the Mac Edit tab when no available model supports edit mode", async () => {
    await render(
      baseContext({
        imageModels: [LENS_TURBO, TORCH_ONLY_EDIT],
        macCapabilities: MAC_CAPS,
      }),
    );

    expect(modeButton("Edit").disabled).toBe(true);
    expect(modeButton("Edit").title).toBe("No available Mac model supports this mode.");
  });

  // Boogu-Image-0.1 (epic 6387 / sc-6400) is backend-driven — no dedicated JSX. Base/Turbo are
  // text-to-image, Edit is the instruction-edit checkpoint, and (unlike Ideogram) Boogu is
  // natural-language, so it must render the plain prompt textarea, NOT the structured caption builder.
  const BOOGU_BASE = {
    ...Z_IMAGE,
    id: "boogu_image",
    name: "Boogu Image",
    family: "boogu",
    capabilities: ["text_to_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };
  const BOOGU_TURBO = {
    ...Z_IMAGE,
    id: "boogu_image_turbo",
    name: "Boogu Image Turbo",
    family: "boogu",
    capabilities: ["text_to_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };
  const BOOGU_EDIT = {
    ...Z_IMAGE,
    id: "boogu_image_edit",
    name: "Boogu Image Edit",
    family: "boogu",
    capabilities: ["edit_image"],
    macSupport: { supported: true, features: { edit: true, reference: false } },
  };

  it("surfaces Boogu Base/Turbo in Text (plain prompt, not the structured builder) and Edit in the Edit tab (sc-6400)", async () => {
    await render(
      baseContext({
        imageModels: [BOOGU_BASE, BOOGU_TURBO, BOOGU_EDIT],
        macCapabilities: MAC_CAPS,
      }),
    );

    // Text tab: Base + Turbo (text_to_image); the Edit checkpoint is excluded.
    const textOptions = modelOptionValues();
    expect(textOptions).toContain("boogu_image");
    expect(textOptions).toContain("boogu_image_turbo");
    expect(textOptions).not.toContain("boogu_image_edit");

    // Boogu is natural-language (no `structuredPrompt`) → the plain prompt textarea, NOT the
    // Ideogram structured-caption builder.
    expect(container.querySelector('textarea[aria-label="Prompt"]')).toBeTruthy();

    // Edit tab enabled, and lists only the Edit checkpoint.
    expect(modeButton("Edit").disabled).toBe(false);
    await click(modeButton("Edit"));
    await act(async () => {});
    expect(field(container, "Model").value).toBe("boogu_image_edit");
    expect(modelOptionValues()).toEqual(["boogu_image_edit"]);
  });

  it("offers the Refine-my-prompt control for Boogu — prompt enhancement reuses prompt_refine (sc-6401)", async () => {
    await render(baseContext({ imageModels: [BOOGU_BASE, BOOGU_TURBO], macCapabilities: MAC_CAPS }));

    // Boogu is non-structured, so the plain-prompt path renders RefinePromptControl ("Refine my
    // prompt"). It drives the prompt_refine utility with Boogu's prompt guide as the rewriter context
    // (S4) — the optional, user-editable enhancement step; raw prompt remains the fallback.
    const refineButton = [...container.querySelectorAll("button")].find((b) =>
      b.textContent.includes("Refine my prompt"),
    );
    expect(refineButton).toBeTruthy();
  });

  const precisionLabel = (root) =>
    [...root.querySelectorAll("label")].find((l) =>
      l.textContent.includes("Full precision (bf16)"),
    );
  const openAdvanced = async (root) =>
    click([...root.querySelectorAll("button")].find((b) => b.textContent === "Advanced"));

  it("exposes the Full-precision (bf16) toggle for Boogu in Advanced when ui.precisionToggle is set (sc-6568)", async () => {
    await render(
      baseContext({
        imageModels: [{ ...BOOGU_BASE, ui: { precisionToggle: true } }],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    const toggle = precisionLabel(container);
    expect(toggle).toBeTruthy();
    // Default off → the packed Q8 build (no mlxQuantize emitted).
    expect(toggle.querySelector('input[type="checkbox"]').checked).toBe(false);
  });

  it("hides the precision toggle when the model omits ui.precisionToggle — catalog-gated, not family-hardcoded (sc-6568)", async () => {
    await render(baseContext({ imageModels: [BOOGU_BASE], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    expect(precisionLabel(container)).toBeFalsy();
  });
});

describe("ImageStudio structured-prompt recipe round-trip (sc-6147)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const IDEOGRAM = {
    ...Z_IMAGE,
    id: "ideogram_4",
    name: "Ideogram 4",
    family: "ideogram",
    capabilities: ["text_to_image"],
    structuredPrompt: true,
  };

  const CAPTION = {
    high_level_description: "A red fox in the snow",
    compositional_deconstruction: {
      background: "A snowy pine forest",
      elements: [{ type: "obj", desc: "a red fox sitting upright" }],
    },
  };

  const generateButton = () =>
    [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate");

  it("restores the builder from a recipe, then re-emits the same caption + blob on Generate", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const structuredPrompt = buildStructuredPromptRecipe({
      intent: "a red fox in the snow",
      caption: CAPTION,
      magicPromptBackend: "prompt_refine",
    });
    await render(
      baseContext({
        createImageJob,
        imageModels: [IDEOGRAM],
        studioLaunch: {
          id: "launch-1",
          view: "Image",
          assetId: "asset-1",
          // Mirrors a stored Ideogram asset: prompt = serialized caption, with the
          // full structured-prompt blob under rawAdapterSettings.structuredPrompt.
          recipe: {
            model: "ideogram_4",
            mode: "text_to_image",
            prompt: serializeCaption(CAPTION),
            rawAdapterSettings: { structuredPrompt },
          },
        },
      }),
    );

    // Restore selected the structured model and rehydrated the builder (Generate is
    // enabled, which requires a valid, non-empty caption in the form — not plain text).
    expect(field(container, "Model").value).toBe("ideogram_4");
    expect(generateButton().disabled).toBe(false);

    await click(generateButton());

    const payload = createImageJob.mock.calls[0][0];
    // Top-level prompt is the canonical serialized caption — byte-identical to source.
    expect(payload.prompt).toBe(serializeCaption(CAPTION));
    // The full structured-prompt blob round-trips through advanced (→ rawAdapterSettings).
    expect(payload.advanced.structuredPrompt.caption).toEqual(CAPTION);
    expect(payload.advanced.structuredPrompt.intent).toBe("a red fox in the snow");
    expect(payload.advanced.structuredPrompt.magicPromptBackend).toBe("prompt_refine");
    expect(payload.advanced.structuredPrompt.runtimePrompt).toBe(serializeCaption(CAPTION));
  });

  it("does not attach a structured-prompt blob for non-structured models", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(baseContext({ createImageJob, imageModels: [Z_IMAGE] }));

    await click(generateButton());

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.advanced.structuredPrompt).toBeUndefined();
  });
});

describe("ImageStudio Ideogram 4 auto-expand on plain-text Generate (sc-6501)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const IDEOGRAM = {
    ...Z_IMAGE,
    id: "ideogram_4",
    name: "Ideogram 4",
    family: "ideogram",
    capabilities: ["text_to_image"],
    structuredPrompt: true,
  };

  const REFINE_READY = { id: PROMPT_REFINE_MODEL_ID, name: "Prompt Refiner", installState: "ready" };

  // A raw magic-prompt model reply (JSON string), as the worker would return it. `onMagicExpand`
  // runs it through parseMagicPromptCaption, so the caption the studio sends is EXPANDED.
  const RAW_CAPTION = JSON.stringify({
    aspect_ratio: "1:1",
    high_level_description: "A red fox on a sunny beach",
    compositional_deconstruction: {
      background: "a sunlit sandy beach with gentle waves",
      elements: [{ type: "obj", desc: "a red fox sitting on the sand" }],
    },
  });
  const EXPANDED = parseMagicPromptCaption(RAW_CAPTION).caption;

  const buttonByText = (text) =>
    [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === text);
  const generateButton = () => buttonByText("Generate");

  function setTextArea(element, value) {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    ).set;
    setter.call(element, value);
    element.dispatchEvent(new window.Event("input", { bubbles: true }));
  }

  async function enterPlainText(text) {
    // Switch the builder to its Plain text tab, then type the idea.
    await click(buttonByText("Plain text"));
    await act(async () => {
      setTextArea(container.querySelector('textarea[aria-label="Plain prompt"]'), text);
    });
  }

  it("auto-expands plain text to a JSON caption and never submits raw text", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const magicPrompt = vi.fn(async () => RAW_CAPTION);
    await render(
      baseContext({ createImageJob, magicPrompt, imageModels: [IDEOGRAM], models: [REFINE_READY] }),
    );

    await enterPlainText("a fox on a beach");
    await click(generateButton());
    await act(async () => {});

    expect(magicPrompt).toHaveBeenCalledTimes(1);
    const payload = createImageJob.mock.calls[0][0];
    // The engine receives the serialized JSON caption — NEVER the raw plain text.
    expect(payload.prompt).toBe(serializeCaption(EXPANDED));
    expect(payload.prompt).not.toBe("a fox on a beach");
    // Recipe records the expanded caption, the original idea, and the magic-prompt backend.
    expect(payload.advanced.structuredPrompt.caption).toEqual(EXPANDED);
    expect(payload.advanced.structuredPrompt.intent).toBe("a fox on a beach");
    expect(payload.advanced.structuredPrompt.magicPromptBackend).toBe(PROMPT_REFINE_MODEL_ID);
  });

  it("blocks generation (never raw text) when the prompt-refiner model is missing", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const magicPrompt = vi.fn(async () => RAW_CAPTION);
    await render(
      baseContext({
        createImageJob,
        magicPrompt,
        createModelDownloadJob: vi.fn(),
        imageModels: [IDEOGRAM],
        models: [{ id: PROMPT_REFINE_MODEL_ID, installState: "missing" }],
      }),
    );

    await enterPlainText("a fox on a beach");
    await click(generateButton());
    await act(async () => {});

    expect(magicPrompt).not.toHaveBeenCalled();
    expect(createImageJob).not.toHaveBeenCalled();
    // The block is surfaced (not silently dropped, never sent as raw text).
    const surfaced = [...container.querySelectorAll('[role="alert"]')].some((n) =>
      /download the prompt-refiner model/i.test(n.textContent),
    );
    expect(surfaced).toBe(true);
  });
});

describe("ImageStudio PiD decoder toggle (sc-7851)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // PiD-eligible image model: declares the qwenimage backbone via ui.pid (mirrors the
  // manifest + worker pid_backbone_for). The checkpoint rides the full catalog (`models`)
  // as its own installable entry (sc-7852), distinct from the image-model picker.
  const PID_QWEN = { ...Z_IMAGE, id: "qwen_image", name: "Qwen Image", ui: { pid: { checkpointId: "pid_qwenimage" } } };
  const PID_CKPT = (installState) => ({ id: "pid_qwenimage", type: "utility", installState });

  const openAdvanced = async () =>
    click([...container.querySelectorAll("button")].find((b) => b.textContent === "Advanced"));
  const pidLabel = () =>
    [...container.querySelectorAll("label")].find((l) => l.textContent.includes("PiD decoder"));
  const generateButton = () =>
    [...container.querySelectorAll("button")].find((b) => b.textContent === "Generate");

  it("shows the toggle (default off) when eligible AND the checkpoint is installed", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("installed")] }));
    await openAdvanced();
    await act(async () => {});
    const toggle = pidLabel();
    expect(toggle).toBeTruthy();
    // Non-commercial marker is surfaced on the toggle copy.
    expect(toggle.textContent).toContain("Non-Commercial");
    expect(toggle.querySelector('input[type="checkbox"]').checked).toBe(false);
  });

  it("hides the toggle when the checkpoint is present but not installed (fail-closed)", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("missing")] }));
    await openAdvanced();
    await act(async () => {});
    expect(pidLabel()).toBeFalsy();
  });

  it("hides the toggle when the checkpoint entry is absent from the catalog (today's pre-sc-7852 state)", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN] }));
    await openAdvanced();
    await act(async () => {});
    expect(pidLabel()).toBeFalsy();
  });

  it("hides the toggle for a non-eligible model even when a PiD checkpoint is installed", async () => {
    await render(baseContext({ imageModels: [Z_IMAGE], models: [Z_IMAGE, PID_CKPT("installed")] }));
    await openAdvanced();
    await act(async () => {});
    expect(pidLabel()).toBeFalsy();
  });

  it("emits advanced.usePid:true only when shown AND toggled on", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("installed")] }),
    );
    await openAdvanced();
    await act(async () => {});

    // Default off → no usePid in the payload.
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("usePid");

    // Toggle on → usePid:true rides advanced.
    await act(async () => pidLabel().querySelector('input[type="checkbox"]').click());
    await click(generateButton());
    expect(createImageJob.mock.calls[1][0].advanced.usePid).toBe(true);
  });
});
