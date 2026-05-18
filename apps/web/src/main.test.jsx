import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, eventUrl } from "./main.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";

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
              workflow: "text_to_image",
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
    expect(container.textContent).toContain("Uses LoRA: Video Motion");

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
            { id: "builtin_cinematic_detail", name: "Cinematic Detail", family: "z-image", scope: "builtin", defaultWeight: 0.55 },
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
              loras: [{ id: "builtin_cinematic_detail", weight: 0.5 }],
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
    expect(container.textContent).toContain("Valid");
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
});
