import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, eventUrl } from "./main.jsx";
import { AssetPickerField } from "./components/AssetPicker.jsx";
import { liveElapsedSeconds } from "./formatting.js";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";

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

  it("renders the app navigation against mocked API calls", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(container.textContent).toContain("Library");
    expect(container.textContent).toContain("Queue");
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
        <CharacterStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          addCharacterReference={addCharacterReference}
          archiveCharacter={() => {}}
          assets={assets}
          attachCharacterLora={() => {}}
          characters={[{ id: "char-1", name: "Mira", type: "person", references: [], approvedReferences: [], looks: [], loras: [] }]}
          createCharacter={() => {}}
          createCharacterLook={() => {}}
          createCharacterTestJob={() => {}}
          deleteAsset={() => {}}
          deleteCharacterLook={() => {}}
          detachCharacterLora={() => {}}
          imageModels={[]}
          latestAssets={[]}
          loras={[]}
          onPreview={() => {}}
          onSendImage={() => {}}
          onSendVideo={() => {}}
          purgeAsset={() => {}}
          removeCharacterReference={() => {}}
          updateAssetStatus={() => {}}
          updateCharacter={() => {}}
          updateCharacterLook={() => {}}
          updateCharacterLora={() => {}}
          updateCharacterReference={() => {}}
        />,
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

  it("keeps the shell usable when recipe presets are unavailable", async () => {
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
    expect(container.textContent).toContain("Project One");
    expect(container.textContent).not.toContain("Not Found");
  });

  it("switches Replace Person to the replacement-capable video model", async () => {
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

    expect(container.textContent).toContain("Wan2.2");
    expect(container.textContent).toContain("V1 placeholder tracking");
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

    expect(container.textContent).toContain("Image generation");
    expect(container.textContent).toContain("running");
    expect(container.textContent).not.toContain("Jobs and GPUs");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Library").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Image").click();
    });
    await settle();

    expect(container.textContent).toContain("Image generation");
    expect(container.textContent).toContain("running");
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

    expect(container.querySelector(".review-grid img")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated-1.png",
    );
    expect(container.textContent).toContain("Pending #2");
    expect(container.textContent).toContain("Pending #4");
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
    expect(container.querySelector(".review-grid img")?.getAttribute("src")).toContain(
      "/api/v1/projects/project-1/files/assets/images/generated_0001.png",
    );
    expect(container.textContent).not.toContain("Pending #1");
    expect(container.textContent).toContain("Pending #2");
    expect(container.textContent).toContain("Pending #4");
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

    expect(container.textContent).toContain("Video generation");
    expect(container.textContent).toContain("queued");
    expect(container.textContent).not.toContain("Jobs and GPUs");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Library").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();

    expect(container.textContent).toContain("Video generation");
    expect(container.textContent).toContain("queued");
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

    expect(container.textContent).toContain("Model download");
    expect(container.textContent).toContain("downloading");
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
    expect(container.textContent).toContain("running");
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={onImportLora}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[]}
          loras={[]}
          models={[
            { id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ]}
          onDownloadModel={() => {}}
          onImportLora={onImportLora}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[]}
          loras={[]}
          models={models}
          onDownloadModel={() => {}}
          onImportLora={onImportLora}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[]}
          loras={[]}
          models={[
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
          ]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[]}
          loras={[
            { id: "ready_style", name: "Ready Style", family: "z-image", scope: "global", installState: "installed" },
            { id: "broken_style", name: "Broken Style", family: "z-image", scope: "global", installState: "missing" },
          ]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image", installState: "installed" }]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onOpenQueue={() => {}}
        />,
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
      return (
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={jobs}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={importLora}
          onOpenQueue={() => {}}
        />
      );
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "running",
              stage: "downloading",
              progress: 0.3,
              payload: { loraId: "existing_import" },
            },
          ]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={onImportLora}
          onOpenQueue={() => {}}
        />,
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
    expect(container.textContent).toContain("LoRA import");
    expect(container.textContent).toContain("running");
  });

  it("keeps failed Models LoRA imports visible inline", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[
            {
              id: "lora-import-job-1",
              type: "lora_import",
              status: "failed",
              stage: "failed",
              progress: 0.4,
              error: "Adapter crashed",
              payload: { loraId: "broken_detail", family: "z-image" },
            },
          ]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[
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
          ]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          jobs={[]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={onImportLora}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={null}
          jobs={[]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onImportModel={onImportModel}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={null}
          jobs={[]}
          loras={[]}
          models={[{ id: "z_image_turbo", name: "Z-Image Turbo", type: "image", family: "z-image" }]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onImportModel={onImportModel}
          onOpenQueue={() => {}}
        />,
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
        <ModelManagerScreen
          activeProject={null}
          jobs={[]}
          loras={[]}
          models={[{ id: "imported_custom", name: "Imported Custom", type: "image" }]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onImportModel={() => {}}
          onOpenQueue={() => {}}
        />,
      );
    });

    expect(container.textContent).toContain("needs family");
    expect(container.textContent).toContain("unassociated");
  });

  it("shows in-progress model imports inline", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ModelManagerScreen
          activeProject={null}
          jobs={[
            {
              id: "model-import-job-1",
              type: "model_import",
              status: "downloading",
              stage: "downloading",
              progress: 0.42,
              payload: { modelId: "custom_model", name: "Custom Model" },
            },
          ]}
          loras={[]}
          models={[]}
          onDownloadModel={() => {}}
          onImportLora={() => {}}
          onImportModel={() => {}}
          onOpenQueue={() => {}}
        />,
      );
    });

    expect(container.textContent).toContain("Model imports in progress");
    expect(container.textContent).toContain("Model import");
    expect(container.textContent).toContain("downloading");
  });

  it("adds the SSE ticket as a query parameter", () => {
    expect(eventUrl("/api/v1/jobs/events", "stream-ticket")).toContain("ticket=stream-ticket");
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

    expect(container.textContent).toContain("queued");

    await act(async () => {
      [...container.querySelectorAll(".job-actions button")].find((button) => button.textContent === "Cancel").click();
    });
    await settle();

    expect(container.textContent).toContain("canceled");
    expect(container.textContent).toContain("Canceled before a worker started.");
    expect(container.textContent).not.toContain("queued");
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
      [...container.querySelectorAll(".job-actions button")].find((button) => button.textContent === "Retry").click();
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
        <QueueScreen
          activeProject={{ id: "project-1", name: "Project 1" }}
          createJob={(event) => event.preventDefault()}
          filteredJobs={[
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
          ]}
          gpuOptions={["auto", "0"]}
          jobAction={() => {}}
          jobPrompt="Placeholder generation"
          projectFilter="all"
          projects={[{ id: "project-1", name: "Project 1" }]}
          requestedGpu="auto"
          setJobPrompt={() => {}}
          setProjectFilter={() => {}}
          setRequestedGpu={() => {}}
          workers={[
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
          ]}
        />,
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
      createJob: (event) => event.preventDefault(),
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
      root.render(<QueueScreen {...queueProps} workers={[worker]} />);
    });

    expect(container.textContent).toContain("20.0 GB");
    expect(container.textContent).toContain("12%");

    await act(async () => {
      root.render(
        <QueueScreen
          {...queueProps}
          workers={[
            {
              ...worker,
              utilization: { memoryTotalMb: 24576, memoryUsedMb: 12288, memoryFreeMb: 12288, gpuLoadPercent: 67 },
            },
          ]}
        />,
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
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          localJobs={[]}
          loras={[]}
          onLocalJobCreated={onLocalJobCreated}
          onPreview={() => {}}
          purgeAsset={() => {}}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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
      root.render(<ImageStudio {...imageProps} />);
    });

    expect(container.textContent).toContain("Finished. Fetching result...");
    expect(container.textContent).not.toContain("No fresh image batch");

    const generatedAsset = {
      id: "asset-1",
      type: "image",
      displayName: "Generated Image",
      generationSetId: "gen-1",
      status: {},
    };
    await act(async () => {
      root.render(<ImageStudio {...imageProps} assets={[generatedAsset]} latestAssets={[generatedAsset]} />);
    });

    expect(container.textContent).not.toContain("Finished. Fetching result...");
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
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={assets}
          characters={[]}
          createImageJob={() => {}}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          localJobs={[localJob]}
          loras={[]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
      );
    });

    const images = [...container.querySelectorAll(".review-grid img")].map((image) => image.getAttribute("src"));
    expect(images[0]).toContain("/api/v1/projects/project-1/files/assets/images/generated_0001.png");
    expect(images[1]).toContain("/api/v1/projects/project-1/files/runs/run_0007/assets/images/generated_0002.png");
    expect(container.textContent).toContain("Pending #3");
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
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={() => {}}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          localJobs={[staleCompletedJob]}
          loras={[]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
      );
    });

    expect(container.textContent).not.toContain("Finished. Fetching result...");
    expect(container.textContent).toContain("No fresh image batch");
  });

  it("submits compatible image LoRAs while capping simple user selections at two", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          loras={[
            { id: "built_in", name: "Built In", family: "z-image", scope: "builtin", defaultWeight: 0.6 },
            { id: "global_style", name: "Global Style", family: "z-image", scope: "global" },
            { id: "project_mira", name: "Project Mira", family: "z-image", scope: "project" },
            { id: "third_user", name: "Third User", family: "z-image", scope: "global" },
            { id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "global" },
            { id: "missing_lora", name: "Missing LoRA", family: "z-image", scope: "global", installState: "missing" },
          ]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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
          expect.objectContaining({ id: "project_mira", scope: "project" }),
        ],
      }),
    );
  });

  it("blocks image submit when a visible incompatible LoRA is selected", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          loras={[{ id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "builtin" }]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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

  it("applies recipe preset defaults and hidden preset LoRAs to image jobs", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          loras={[
            {
              id: "cinematic_detail",
              name: "Cinematic Detail",
              family: "z-image",
              scope: "builtin",
              defaultWeight: 0.55,
              presetManaged: true,
            },
          ]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          recipePresets={[
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
          ]}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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

  it("surfaces model and preset first and lets image generation run with no preset", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto", "1"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          latestAssets={[]}
          loras={[{ id: "cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", presetManaged: true }]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          recipePresets={[
            {
              id: "cinematic",
              name: "Cinematic",
              model: "z_image_turbo",
              workflow: "text_to_image",
              defaults: { count: 2, negativePrompt: "flat lighting" },
              builtInLoras: [{ id: "cinematic_detail" }],
            },
          ]}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
      );
    });
    await settle();

    // Primary recipe controls are surfaced in the rail (no longer behind Advanced),
    // alongside the hero-mounted prompt + preset chip strip.
    const railLabels = [...container.querySelectorAll(".recipe-rail > label, .recipe-rail .recipe-row label")].map(
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

  it("blocks image presets whose managed LoRAs do not match the selected model", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" },
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
          ]}
          latestAssets={[]}
          loras={[
            {
              id: "qwen_detail",
              name: "Qwen Detail",
              family: "qwen-image",
              scope: "builtin",
              presetManaged: true,
            },
          ]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          recipePresets={[
            {
              id: "cinematic",
              name: "Cinematic",
              workflow: "text_to_image",
              builtInLoras: [{ id: "qwen_detail", weight: 0.4 }],
            },
          ]}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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
        <VideoStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[{ id: "image-1", type: "image", displayName: "Frame One" }]}
          characters={[]}
          createPersonDetectionJob={() => {}}
          createPersonTrackJob={() => {}}
          createVideoJob={createVideoJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          latestAssets={[]}
          loras={[{ id: "wan_motion", name: "Wan Motion", family: "wan-video", scope: "builtin", presetManaged: true }]}
          onPreview={() => {}}
          personTracks={[]}
          purgeAsset={() => {}}
          recipePresets={[
            {
              id: "dream_motion",
              name: "Dream Motion",
              workflow: "image_to_video",
              model: "ltx_2_3",
              builtInLoras: [{ id: "wan_motion" }],
            },
          ]}
          requestedGpu="auto"
          selectedAsset={{ id: "image-1", type: "image", displayName: "Frame One" }}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
          videoModels={[
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              family: "ltx-video",
              capabilities: ["image_to_video"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              loraCompatibility: { families: ["ltx-video"] },
            },
          ]}
        />,
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

  it("keeps Qwen selected when applying a Qwen image preset", async () => {
    const createImageJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[]}
          characters={[]}
          createImageJob={createImageJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[
            { id: "qwen_image", name: "Qwen Image", type: "image", family: "qwen-image" },
            { id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" },
          ]}
          latestAssets={[]}
          loras={[]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          recipePresets={[
            { id: "qwen_detail", name: "Qwen Detail", model: "qwen_image", workflow: "text_to_image", defaults: { count: 1 } },
            { id: "cinematic", name: "Cinematic", model: "z_image_turbo", workflow: "text_to_image", defaults: { count: 4 } },
          ]}
          requestedGpu="auto"
          selectedAsset={null}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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

  it("uses preset modes as the Image Studio picker surface", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        <ImageStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[{ id: "image-1", type: "image", displayName: "Frame One" }]}
          characters={[]}
          createImageJob={() => {}}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image", capabilities: ["edit_image"] }]}
          latestAssets={[]}
          loras={[]}
          onPreview={() => {}}
          purgeAsset={() => {}}
          recipePresets={[
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
          ]}
          requestedGpu="auto"
          selectedAsset={{ id: "image-1", type: "image", displayName: "Frame One" }}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
        />,
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

  it("applies recipe preset defaults to video jobs", async () => {
    const createVideoJob = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <VideoStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[{ id: "image-1", type: "image", displayName: "Frame One" }]}
          characters={[]}
          createPersonDetectionJob={() => {}}
          createPersonTrackJob={() => {}}
          createVideoJob={createVideoJob}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          latestAssets={[]}
          loras={[{ id: "video_motion", name: "Video Motion" }]}
          onPreview={() => {}}
          personTracks={[]}
          purgeAsset={() => {}}
          recipePresets={[
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
          ]}
          requestedGpu="auto"
          selectedAsset={{ id: "image-1", type: "image", displayName: "Frame One" }}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
          videoModels={[
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ]}
        />,
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
        loras: [expect.objectContaining({ id: "video_motion", weight: 0.8, presetManaged: true })],
        advanced: expect.objectContaining({
          recipePresetName: "Dream Motion",
          recipePresetPrompt: { suffix: "smooth camera motion" },
          resolution: "1280x720",
        }),
      }),
    );
  });

  it("filters video presets by mode and selected model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(
        <VideoStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[{ id: "image-1", type: "image", displayName: "Frame One" }]}
          characters={[]}
          createPersonDetectionJob={() => {}}
          createPersonTrackJob={() => {}}
          createVideoJob={() => {}}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          latestAssets={[]}
          loras={[]}
          onPreview={() => {}}
          personTracks={[]}
          purgeAsset={() => {}}
          recipePresets={[
            { id: "ltx_motion", name: "LTX Motion", workflow: "image_to_video", model: "ltx_2_3" },
            { id: "ltx_story", name: "LTX Story", workflow: "text_to_video", model: "ltx_2_3" },
            { id: "wan_motion", name: "Wan Motion", workflow: "image_to_video", model: "wan_2_2" },
          ]}
          requestedGpu="auto"
          selectedAsset={{ id: "image-1", type: "image", displayName: "Frame One" }}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
          videoModels={[
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
          ]}
        />,
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
        <VideoStudio
          activeProject={{ id: "project-1", name: "Noir" }}
          assets={[
            { id: "image-1", type: "image", displayName: "Frame One" },
            { id: "image-2", type: "image", displayName: "Frame Two" },
          ]}
          characters={[]}
          createPersonDetectionJob={() => {}}
          createPersonTrackJob={() => {}}
          createVideoJob={() => {}}
          deleteAsset={() => {}}
          gpuOptions={["auto"]}
          latestAssets={[]}
          loras={[]}
          onPreview={() => {}}
          personTracks={[]}
          purgeAsset={() => {}}
          recipePresets={[
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
          ]}
          requestedGpu="auto"
          selectedAsset={{ id: "image-1", type: "image", displayName: "Frame One" }}
          setRequestedGpu={() => {}}
          updateAssetStatus={() => {}}
          videoModels={[
            {
              id: "ltx_2_3",
              name: "LTX",
              type: "video",
              capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
              defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
              limits: { durations: [4, 6, 8], fps: [24, 25, 30], resolutions: ["768x512", "1280x720"] },
            },
          ]}
        />,
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

  it("creates, edits, duplicates, and archives recipe presets from the manager", async () => {
    const createRecipePreset = vi.fn(async (payload) => payload);
    const updateRecipePreset = vi.fn(async (id, payload) => ({ ...payload, id }));
    const duplicateRecipePreset = vi.fn(async (id) => ({ id: `${id}_copy` }));
    const deleteRecipePreset = vi.fn(async (id) => ({ id, archived: true }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PresetManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          createRecipePreset={createRecipePreset}
          deleteRecipePreset={deleteRecipePreset}
          duplicateRecipePreset={duplicateRecipePreset}
          imageModels={[{ id: "z_image_turbo", name: "Z-Image", type: "image", family: "z-image" }]}
          loras={[
            { id: "cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", defaultWeight: 0.55 },
            { id: "global_detail", name: "Global Detail", family: "z-image", scope: "global", defaultWeight: 0.7 },
            { id: "qwen_only", name: "Qwen Only", family: "qwen-image", scope: "global" },
          ]}
          recipePresets={[
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
          ]}
          updateRecipePreset={updateRecipePreset}
          videoModels={[{ id: "ltx_2_3", name: "LTX", type: "video" }]}
        />,
      );
    });

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "New Preset").click();
    });
    await changeField(field(container, "Name"), "Soft Morning");
    await changeField(field(container, "Add LoRA"), "global_detail");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
    });
    await changeField(field(container, "Weight"), "0.35");
    expect(field(container, "ID").value).toBe("soft_morning");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Create Preset").click();
    });
    expect(createRecipePreset).toHaveBeenCalledWith(
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
    await changeField(field(container, "Add LoRA"), "global_detail");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Add LoRA").click();
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
    expect(updateRecipePreset).toHaveBeenCalledWith("moody", expect.objectContaining({ ui: { description: "Richer low key color." } }), "global");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Duplicate").click();
    });
    expect(duplicateRecipePreset).toHaveBeenCalledWith("moody", "global");

    await act(async () => {
      [...container.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Archive").click();
    });
    expect(deleteRecipePreset).toHaveBeenCalledWith("moody", "global");
  });

  it("explains preset save blockers and selected LoRA warning states", async () => {
    const updateRecipePreset = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PresetManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          createRecipePreset={() => {}}
          deleteRecipePreset={() => {}}
          duplicateRecipePreset={() => {}}
          imageModels={[]}
          loras={[{ id: "pending_style", name: "Pending Style", family: "z-image", scope: "global", installState: "missing" }]}
          recipePresets={[
            {
              id: "blocked",
              name: "Blocked",
              scope: "global",
              workflow: "text_to_image",
              model: "z_image_turbo",
              loras: [{ id: "pending_style" }],
            },
          ]}
          updateRecipePreset={updateRecipePreset}
          videoModels={[]}
        />,
      );
    });

    expect(container.textContent).toContain("No models");
    expect(container.textContent).not.toContain("No model selected");
    expect(container.textContent).toContain("Pending Style");
    expect(container.textContent).toContain("Missing or still importing");
    expect(container.textContent).toContain("Save blocked: pending_style has not finished importing.");
    expect(field(container, "Weight").disabled).toBe(true);
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Save Preset").disabled).toBe(true);
    expect(updateRecipePreset).not.toHaveBeenCalled();
  });
});
