import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, eventUrl } from "./main.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";

class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.listeners = {};
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

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
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
              id: "builtin_cinematic_detail",
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
              defaults: { count: 2, resolution: "1280x720", negativePrompt: "flat lighting" },
              prompt: { suffix: "cinematic lighting" },
              builtInLoras: [{ id: "builtin_cinematic_detail", weight: 0.4 }],
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
    expect(container.textContent).toContain("Uses LoRA: Cinematic Detail");

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
});
