import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, eventUrl } from "./main.jsx";
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
    await Promise.resolve();
    await Promise.resolve();
  });
}

function field(container, labelText) {
  const label = [...container.querySelectorAll("label")].find((item) => item.childNodes[0]?.textContent.trim() === labelText);
  return label?.querySelector("input, select, textarea");
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace Person").click();
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace Person").click();
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Text to Video").click();
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
    await changeField(field(container, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });
    await settle();

    expect(container.textContent).toContain("LoRA imports in progress");
    expect(container.textContent).toContain("running");
    expect(container.textContent).not.toContain("Jobs and GPUs");
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Upload").click();
    });
    await changeFile(field(container, "LoRA File"), loraFile);
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });

    expect(container.textContent).toContain("Uploaded LoRA file exceeds the 2GB limit");
    expect(importCalls).toBe(0);
  });

  it("keeps Preset Manager LoRA imports in context", async () => {
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
      if (path.endsWith("/recipe-presets")) {
        return Promise.resolve(
          response([{ id: "moody", name: "Moody", scope: "global", workflow: "text_to_image", model: "z_image_turbo" }]),
        );
      }
      if (path.endsWith("/jobs")) {
        return Promise.resolve(response(createdJobs));
      }
      if (path.endsWith("/loras/import") && options.method === "POST") {
        const job = {
          id: "lora-import-job-1",
          type: "lora_import",
          status: "running",
          payload: { loraId: "preset_detail" },
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Presets").click();
    });
    await settle();
    await changeField(field(container, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });
    await settle();

    expect(container.textContent).toContain("Preset Manager");
    expect(createdJobs).toHaveLength(1);
    expect(container.textContent).not.toContain("Jobs and GPUs");
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

    await changeField(field(container, "Source URL"), "https://example.com/loras/detail.safetensors");
    await changeField(field(container, "Name"), "Detail LoRA");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceUrl: "https://example.com/loras/detail.safetensors",
        name: "Detail LoRA",
        scope: "global",
        family: "z-image",
      }),
    );
    expect(container.textContent).toContain("LoRA import queued for detail_lora.");
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Upload").click();
    });
    await changeField(field(container, "Scope"), "project");
    await changeFile(field(container, "LoRA File"), loraFile);
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });

    expect(onImportLora).toHaveBeenCalledWith(
      expect.objectContaining({
        file: loraFile,
        scope: "project",
        family: "z-image",
      }),
    );
    expect(container.textContent).toContain("LoRA import");
    expect(container.textContent).toContain("running");
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

    await changeField(field(container, "Source URL"), "file:///tmp/detail.safetensors");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });

    expect(container.textContent).toContain("LoRA sourceUrl must use http or https");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").disabled).toBe(false);
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
    expect(container.textContent).toContain("Rust CPU utility worker");
    expect(container.textContent).not.toContain("Rust placeholder GPU");
    expect(container.textContent).not.toContain("Stale GPU");
    expect([...container.querySelector("#queue-gpu").options].map((option) => option.value)).toEqual(["auto", "0"]);
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

    expect(container.textContent).toContain("Built In");
    expect(container.textContent).not.toContain("Qwen Only");
    expect(container.textContent).not.toContain("Missing LoRA");

    const checkboxes = [...container.querySelectorAll('.lora-choice input[type="checkbox"]')];
    await act(async () => {
      checkboxes[0].click();
      checkboxes[1].click();
      checkboxes[2].click();
    });

    expect(checkboxes[3].disabled).toBe(true);

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Generate Clip").click();
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Text to Video").click();
    });
    await settle();

    expect(container.textContent).toContain("LTX Story");
    expect(container.textContent).not.toContain("LTX Motion");

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Advanced").click();
    });
    await changeField(field(container, "Model"), "wan_2_2");
    await settle();

    expect(container.textContent).toContain("Presets unavailable");
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
      [...container.querySelectorAll("button")].find((button) => button.textContent === "First/Last Frame").click();
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
    const createLoraImportJob = vi.fn(async (payload) => ({ payload: { ...payload, loraId: "imported_detail" } }));
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PresetManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          createLoraImportJob={createLoraImportJob}
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
    await act(async () => {
      [...container.querySelectorAll('.lora-choice input[type="checkbox"]')].find((input) => input.closest(".lora-choice").textContent.includes("Global Detail")).click();
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
      [...container.querySelectorAll(".preset-row")].find((button) => button.textContent.includes("Moody")).click();
    });
    await changeField(field(container, "Description"), "Richer low key color.");
    await changeField(field(container, "Source URL"), "https://example.com/loras/detail.safetensors");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });
    expect(createLoraImportJob).toHaveBeenCalledWith(
      expect.objectContaining({ sourceUrl: "https://example.com/loras/detail.safetensors", scope: "global", family: "z-image" }),
    );
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Local File").click();
    });
    const loraFile = new File(["lora"], "detail.safetensors", { type: "application/octet-stream" });
    const importPanel = container.querySelector(".lora-import-panel");
    await changeFile(field(importPanel, "Local File"), loraFile);
    await changeField(field(importPanel, "Name"), "Uploaded Detail");
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue Import").click();
    });
    expect(createLoraImportJob).toHaveBeenLastCalledWith(
      expect.objectContaining({ file: loraFile, name: "Uploaded Detail", scope: "global", family: "z-image" }),
    );
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

  it("explains preset save blockers and no-model empty states", async () => {
    const updateRecipePreset = vi.fn();
    root = createRoot(container);
    await act(async () => {
      root.render(
        <PresetManagerScreen
          activeProject={{ id: "project-1", name: "Noir" }}
          createLoraImportJob={() => {}}
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
    expect(container.textContent).toContain("No model selected");
    expect(container.textContent).toContain("Save blocked: pending_style has not finished importing.");
    expect([...container.querySelectorAll("button")].find((button) => button.textContent === "Save Preset").disabled).toBe(true);
    expect(updateRecipePreset).not.toHaveBeenCalled();
  });
});
