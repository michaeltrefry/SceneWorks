import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkerProgressCard, deriveJobTitle, getJobTypeChip } from "./WorkerProgressCard.jsx";
import { AppContext } from "../context/AppContext.js";
import { buildWorkersById } from "../workers.js";

const cudaWorker = {
  id: "worker-cuda-1",
  gpuId: "gpu-0",
  gpuName: "NVIDIA GeForce RTX 4090",
  capabilities: ["gpu", "image_generate"],
  utilization: { memoryUsedMb: 18000, memoryTotalMb: 24000, gpuLoadPercent: 67 },
};

const appleWorker = {
  id: "worker-mps-1",
  gpuId: "mps-0",
  gpuName: "Apple M2 Ultra",
  capabilities: ["gpu", "image_generate", "lora_train"],
  utilization: { memoryUsedMb: 30000, memoryTotalMb: 64000, gpuLoadPercent: 41 },
};

const cpuWorker = {
  id: "worker-cpu-1",
  gpuId: "cpu-0",
  gpuName: null,
  capabilities: ["cpu", "prompt_refine"],
  utilization: null,
};

function makeContext(workers) {
  return {
    visibleWorkers: workers,
    workersById: buildWorkersById(workers),
  };
}

function render(ui, contextValue) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(<AppContext.Provider value={contextValue}>{ui}</AppContext.Provider>);
  });
  return {
    container,
    cleanup() {
      act(() => root.unmount());
      container.remove();
    },
  };
}

describe("deriveJobTitle", () => {
  it("prefers job.title when present", () => {
    expect(deriveJobTitle({ title: "Override", type: "image_generate" })).toBe("Override");
  });

  it("formats lora_train jobs with the LoRA name", () => {
    const job = { type: "lora_train", payload: { loraName: "kelsie-v3" } };
    expect(deriveJobTitle(job)).toBe("Training Run — kelsie-v3");
  });

  it("formats captioning jobs with the dataset name", () => {
    const job = { type: "training_caption", payload: { datasetName: "kelsie-set" } };
    expect(deriveJobTitle(job)).toBe("Dataset Captioning — kelsie-set");
  });

  it("formats image generation jobs and truncates long prompts", () => {
    const longPrompt = "a ".repeat(120);
    const job = { type: "image_generate", payload: { prompt: longPrompt } };
    const title = deriveJobTitle(job);
    expect(title.startsWith("Generate Image — ")).toBe(true);
    expect(title.endsWith("…")).toBe(true);
    expect(title.length).toBeLessThan(120);
  });

  it("formats character turnaround when characterId is set", () => {
    const job = {
      type: "image_generate",
      payload: { prompt: "anything", characterId: "char-1", characterName: "Aria" },
    };
    expect(deriveJobTitle(job)).toBe("Character Turnaround — Aria");
  });

  it("formats lora_import jobs with the filename when name absent", () => {
    const job = { type: "lora_import", payload: { filename: "kelsie.safetensors" } };
    expect(deriveJobTitle(job)).toBe("LoRA Import — kelsie.safetensors");
  });

  it("falls back to (unnamed) when no subject is present", () => {
    const job = { type: "lora_train", payload: {} };
    expect(deriveJobTitle(job)).toBe("Training Run — (unnamed LoRA)");
  });
});

describe("getJobTypeChip", () => {
  it("maps known types to display labels", () => {
    expect(getJobTypeChip("lora_train")).toBe("Training Run");
    expect(getJobTypeChip("training_caption")).toBe("Dataset Captioning");
    expect(getJobTypeChip("video_generate")).toBe("Generate Video");
    expect(getJobTypeChip("model_download")).toBe("Model Import");
  });

  it("falls back to a capitalized type for unknown values", () => {
    expect(getJobTypeChip("custom_job")).toBe("Custom job");
  });
});

describe("WorkerProgressCard layout", () => {
  let api;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-28T12:00:30Z"));
  });

  afterEach(() => {
    api?.cleanup();
    api = null;
    vi.useRealTimers();
  });

  it("renders the type chip, status badge, title, and id", () => {
    const job = {
      id: "job-abcdef123456",
      type: "image_generate",
      status: "running",
      stage: "denoising",
      progress: 0.4,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cudaWorker.id,
      payload: { prompt: "a sunset over the mountains" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([cudaWorker]));
    const card = api.container.querySelector(".worker-progress-card");
    expect(card.classList.contains("running")).toBe(true);
    expect(card.querySelector(".worker-progress-card__type").textContent).toBe("Generate Image");
    expect(card.querySelector(".status-badge").textContent).toBe("Running");
    expect(card.querySelector(".worker-progress-card__title").textContent).toBe(
      "Generate Image — a sunset over the mountains",
    );
    expect(card.querySelector(".worker-progress-card__id").getAttribute("title")).toBe("job-abcdef123456");
    expect(card.querySelector(".worker-progress-card__id").textContent).toBe("job-ab…3456");
  });

  it("renders live GPU meters for a running job assigned to a CUDA worker", () => {
    const job = {
      id: "job-1",
      type: "image_generate",
      status: "running",
      progress: 0.25,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cudaWorker.id,
      payload: {},
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([cudaWorker]));
    const arch = api.container.querySelector(".worker-progress-card__pill.arch");
    expect(arch.textContent).toBe("cuda");
    const meters = api.container.querySelectorAll(".worker-progress-card__meter-value");
    expect(meters).toHaveLength(2);
    expect(meters[0].textContent).toBe("75%"); // 18000/24000
    expect(meters[1].textContent).toBe("67%");
  });

  it("renders MPS arch for Apple Silicon workers", () => {
    const job = {
      id: "j",
      type: "lora_train",
      status: "running",
      progress: 0.1,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: appleWorker.id,
      payload: { loraName: "x" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([appleWorker]));
    expect(api.container.querySelector(".worker-progress-card__pill.arch").textContent).toBe("mps");
  });

  it("uses peak meters and hides cancel for completed jobs", () => {
    const job = {
      id: "j",
      type: "image_generate",
      status: "completed",
      progress: 1,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cudaWorker.id,
      payload: {},
      peakGpuMemoryPct: 88,
      peakGpuLoadPct: 95,
    };
    const onCancel = vi.fn();
    const onRetry = vi.fn();
    const onDuplicate = vi.fn();
    api = render(
      <WorkerProgressCard job={job} onCancel={onCancel} onRetry={onRetry} onDuplicate={onDuplicate} />,
      makeContext([cudaWorker]),
    );
    const metersHost = api.container.querySelector(".worker-progress-card__meters");
    expect(metersHost.getAttribute("data-meter-source")).toBe("peak");
    const values = api.container.querySelectorAll(".worker-progress-card__meter-value");
    expect(values[0].textContent).toBe("88%");
    expect(values[1].textContent).toBe("95%");
    const buttons = api.container.querySelectorAll(".worker-progress-card__actions button");
    const labels = Array.from(buttons).map((b) => b.textContent);
    expect(labels).not.toContain("Cancel");
    expect(labels).toContain("Retry");
    expect(labels).toContain("Duplicate");
  });

  it("shows only Cancel for queued jobs (no Retry/Duplicate)", () => {
    const job = {
      id: "j",
      type: "image_generate",
      status: "queued",
      progress: 0,
      attempts: 1,
      payload: { prompt: "" },
    };
    const handlers = { onCancel: vi.fn(), onRetry: vi.fn(), onDuplicate: vi.fn() };
    api = render(<WorkerProgressCard job={job} {...handlers} />, makeContext([]));
    const buttons = api.container.querySelectorAll(".worker-progress-card__actions button");
    const labels = Array.from(buttons).map((b) => b.textContent);
    expect(labels).toEqual(["Cancel"]);
  });

  it("hides View in Queue when hideOpenQueue is true", () => {
    const job = { id: "j", type: "image_generate", status: "running", progress: 0, attempts: 1, payload: {} };
    const onOpenQueue = vi.fn();
    api = render(
      <WorkerProgressCard job={job} onOpenQueue={onOpenQueue} hideOpenQueue />,
      makeContext([]),
    );
    const labels = Array.from(api.container.querySelectorAll(".worker-progress-card__actions button")).map(
      (b) => b.textContent,
    );
    expect(labels).not.toContain("View in Queue");
  });

  it("shows an indeterminate progress bar when running with no progress value", () => {
    const job = {
      id: "j",
      type: "image_generate",
      status: "running",
      progress: 0,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      payload: {},
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([]));
    expect(api.container.querySelector(".worker-progress-card__progress.indeterminate")).not.toBeNull();
  });

  it("hides GPU pills and meters for CPU-only workers", () => {
    const job = {
      id: "j",
      type: "prompt_refine",
      status: "running",
      progress: 0,
      attempts: 1,
      startedAt: "2026-05-28T12:00:00Z",
      workerId: cpuWorker.id,
      payload: { prompt: "refine this" },
    };
    api = render(<WorkerProgressCard job={job} />, makeContext([cpuWorker]));
    const pills = api.container.querySelectorAll(".worker-progress-card__pill");
    expect(Array.from(pills).map((p) => p.textContent)).toEqual(["CPU"]);
    expect(api.container.querySelector(".worker-progress-card__meters")).toBeNull();
  });

  it("invokes onCancel when the Cancel button is clicked", () => {
    const job = { id: "j", type: "image_generate", status: "running", progress: 0.3, attempts: 1, payload: {} };
    const onCancel = vi.fn();
    api = render(<WorkerProgressCard job={job} onCancel={onCancel} />, makeContext([]));
    const button = api.container.querySelector(".worker-progress-card__actions button");
    act(() => {
      button.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onCancel.mock.calls[0][0]).toBe(job);
  });
});
