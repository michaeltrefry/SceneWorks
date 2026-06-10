import React, { useMemo } from "react";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";
import { GPU_REQUIRED_JOB_TYPES, NON_GPU_JOB_TYPES } from "../jobTypes.js";
import { useAppContext } from "../context/AppContext.js";

function formatJobType(type) {
  return String(type ?? "job").replaceAll("_", " ");
}

function workerSupports(worker, type) {
  return Array.isArray(worker.capabilities) && worker.capabilities.includes(type);
}

function workerCanClaim(job, worker) {
  if (!workerSupports(worker, job.type)) {
    return false;
  }
  if (NON_GPU_JOB_TYPES.has(job.type)) {
    return true;
  }
  if (GPU_REQUIRED_JOB_TYPES.has(job.type) && worker.gpuId === "cpu") {
    return false;
  }
  return job.requestedGpu === "auto" || job.requestedGpu === worker.gpuId;
}

function modelKeys(job) {
  const keys = new Set();
  if (job.payload?.model) {
    keys.add(job.payload.model);
  }
  if (job.payload?.repo) {
    keys.add(job.payload.repo);
  }
  if (job.payload?.advanced?.modelRepo) {
    keys.add(job.payload.advanced.modelRepo);
  }
  if (job.payload?.advanced?.repo) {
    keys.add(job.payload.advanced.repo);
  }
  return keys;
}

function activeModelDownloadFor(job, jobs) {
  const keys = modelKeys(job);
  if (!keys.size) {
    return null;
  }
  return jobs.find(
    (candidate) =>
      candidate.type === "model_download" &&
      !terminalStatuses.has(candidate.status) &&
      (keys.has(candidate.payload?.modelId) || keys.has(candidate.payload?.repo)),
  );
}

function dependencyJobId(job) {
  return job.payload?.dependsOnJobId ?? job.payload?.dependencyJobId ?? job.dependsOnJobId ?? job.sourceJobId ?? null;
}

function activeDependencyFor(job, jobs) {
  const id = dependencyJobId(job);
  if (!id) {
    return null;
  }
  const dependency = jobs.find((candidate) => candidate.id === id);
  return dependency && !terminalStatuses.has(dependency.status) ? dependency : null;
}

function jobWaitingMessage(job, workers, jobs) {
  if (job.status !== "queued") {
    return job.error ?? job.message;
  }
  const dependency = activeDependencyFor(job, jobs);
  if (dependency) {
    return `Waiting for dependency ${dependency.id} to finish.`;
  }
  const download = activeModelDownloadFor(job, jobs);
  if (download) {
    return `Waiting for model download ${download.payload?.modelName ?? download.payload?.modelId ?? download.id} to finish.`;
  }
  const candidates = workers.filter((worker) => workerCanClaim(job, worker));
  if (!candidates.length) {
    if (job.requestedGpu && job.requestedGpu !== "auto") {
      return `Blocked: no active worker can run ${formatJobType(job.type)} on GPU ${job.requestedGpu}.`;
    }
    return `Blocked: no active worker supports ${formatJobType(job.type)}.`;
  }
  if (candidates.every((worker) => worker.status === "busy")) {
    const target = job.requestedGpu && job.requestedGpu !== "auto" ? `GPU ${job.requestedGpu}` : "an eligible worker";
    return `Waiting: ${target} is busy.`;
  }
  if (job.requestedGpu && job.requestedGpu !== "auto") {
    return `Waiting for GPU ${job.requestedGpu} to claim the job.`;
  }
  return NON_GPU_JOB_TYPES.has(job.type) ? "Waiting for a utility worker." : "Waiting for an available GPU worker.";
}

function workerStatusLine(worker) {
  if (worker.status === "busy") {
    return `Busy${worker.currentJobId ? ` with ${worker.currentJobId}` : ""}`;
  }
  return worker.status === "idle" ? "Ready" : worker.status;
}

function isGpuWorker(worker) {
  // Queue resource cards are for live GPU capacity; CPU utility workers stay out of this panel.
  return worker.gpuId && worker.gpuId !== "cpu" && Array.isArray(worker.capabilities) && worker.capabilities.includes("gpu");
}

function formatMemory(mb) {
  if (!Number.isFinite(mb)) {
    return "Unknown";
  }
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${Math.round(mb)} MB`;
}

function boundedPercent(value) {
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.min(100, Math.max(0, value));
}

function memoryUsagePercent(utilization) {
  const total = Number(utilization?.memoryTotalMb);
  const used = Number(utilization?.memoryUsedMb);
  if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(used)) {
    return null;
  }
  return boundedPercent((used / total) * 100);
}

function utilizationLabel(value) {
  return Number.isFinite(value) ? `${Math.round(value)}%` : "Unknown";
}

function WorkerCard({ worker }) {
  const utilization = worker.utilization ?? {};
  const memoryPercent = memoryUsagePercent(utilization);
  const loadPercent = boundedPercent(Number(utilization.gpuLoadPercent));
  const freeMb = Number(utilization.memoryFreeMb);
  const usedMb = Number(utilization.memoryUsedMb);
  const totalMb = Number(utilization.memoryTotalMb);
  return (
    <div className="worker-card">
      <div className="worker-card-header">
        <strong>{worker.gpuName ?? `GPU ${worker.gpuId}`}</strong>
        <span>{workerStatusLine(worker)}</span>
      </div>
      <div className="worker-stat-grid">
        <span>
          <small>Available</small>
          <strong>{formatMemory(freeMb)}</strong>
        </span>
        <span>
          <small>Memory</small>
          <strong>{Number.isFinite(usedMb) && Number.isFinite(totalMb) ? `${formatMemory(usedMb)} / ${formatMemory(totalMb)}` : "Unknown"}</strong>
        </span>
        <span>
          <small>Load</small>
          <strong>{utilizationLabel(loadPercent)}</strong>
        </span>
      </div>
      {memoryPercent === null ? null : (
        <div className="worker-meter" aria-label={`GPU memory usage ${utilizationLabel(memoryPercent)}`}>
          <span style={{ width: `${memoryPercent}%` }} />
        </div>
      )}
      {loadPercent === null ? null : (
        <div className="worker-meter gpu-load" aria-label={`GPU load ${utilizationLabel(loadPercent)}`}>
          <span style={{ width: `${loadPercent}%` }} />
        </div>
      )}
      <small>{worker.loadedModels?.length ? `Warm: ${worker.loadedModels.join(", ")}` : "No warm model"}</small>
    </div>
  );
}

export function QueueScreen() {
  const {
    activeProject,
    assets = [],
    createPlaceholderJob,
    filteredJobs,
    gpuOptions,
    jobAction,
    jobs = filteredJobs,
    jobPrompt,
    projectFilter,
    projects,
    requestedGpu,
    setJobPrompt,
    setProjectFilter,
    setPreviewAsset,
    setRequestedGpu,
    visibleWorkers,
  } = useAppContext();
  const createJob = createPlaceholderJob;
  const workers = visibleWorkers;
  // Prefer the shared index from context (sc-2082); fall back for legacy
  // contexts that may not yet expose it (test harnesses, etc.).
  const gpuWorkers = useMemo(() => workers.filter(isGpuWorker), [workers]);
  return (
    <section className="main-surface queue-surface">
      <div className="queue-header">
        <div className="section-heading">
          <p className="eyebrow">Jobs and GPUs</p>
          <h2>Queue</h2>
        </div>
        <form className="job-composer" onSubmit={createJob}>
          <label htmlFor="queue-job-prompt">Prompt</label>
          <input id="queue-job-prompt" onChange={(event) => setJobPrompt(event.target.value)} value={jobPrompt} />
          <label htmlFor="queue-gpu">GPU</label>
          <select id="queue-gpu" onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
            {gpuOptions.map((gpu) => (
              <option key={gpu} value={gpu}>
                {gpu === "auto" ? "Auto" : gpu}
              </option>
            ))}
          </select>
          <button disabled={!activeProject} type="submit">
            Add job
          </button>
        </form>
      </div>

      <div className="queue-tools">
        <label htmlFor="project-filter">Project</label>
        <select id="project-filter" onChange={(event) => setProjectFilter(event.target.value)} value={projectFilter}>
          <option value="all">All projects</option>
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
      </div>

      <div className="worker-grid">
        {gpuWorkers.length === 0 ? (
          <div className="worker-card">
            <strong>No GPU workers registered</strong>
            <span>Start a GPU worker to claim generation jobs.</span>
          </div>
        ) : (
          gpuWorkers.map((worker) => <WorkerCard key={worker.id} worker={worker} />)
        )}
      </div>

      <div className="job-list">
        {filteredJobs.length === 0 ? (
          <div className="empty-panel">No jobs in this view</div>
        ) : (
          filteredJobs.map((job) => {
            const message = jobWaitingMessage(job, workers, jobs);
            const variant = thumbnailVariantForJob(job);
            const thumbnails = variant === "hidden" ? [] : resolveJobAssets(job, assets);
            // Inject the queue's context-aware waiting/error message via job.message
            // so the shared WorkerProgressCard surface it without per-screen plumbing.
            const enrichedJob = message && message !== job.message ? { ...job, message } : job;
            return (
              <WorkerProgressCard
                key={job.id}
                job={enrichedJob}
                thumbnailsVariant={variant}
                thumbnailAssets={thumbnails}
                onThumbnailClick={setPreviewAsset ? (asset) => setPreviewAsset(asset, thumbnails) : undefined}
                onCancel={(j) => jobAction(j, "cancel")}
                onRetry={(j, payload) => jobAction(j, "retry", { body: payload ?? {} })}
                onFreshRetry={(j, payload) => jobAction(j, "retry", { body: payload ?? {} })}
                onDuplicate={(j) => jobAction(j, "duplicate")}
                hideOpenQueue
              />
            );
          })
        )}
      </div>
    </section>
  );
}

// Variants per job type: asset-producing jobs get the compact small-row of
// thumbnails; caption / import / prompt-refine jobs hide thumbnails per the
// design spec (docs/design/worker-progress-card.md).
function thumbnailVariantForJob(job) {
  switch (job?.type) {
    case "training_caption":
    case "model_download":
    case "model_import":
    case "model_convert":
    case "lora_import":
    case "prompt_refine":
      return "hidden";
    default:
      return "small-row";
  }
}

// Resolve a job's produced asset records against the live catalog. Generic over
// image/video so the queue's small-row works for both. Matches the resolution
// strategy used by ImageStudio.jobResultAssets / VideoStudio.jobVideoResultAssets.
function resolveJobAssets(job, assets) {
  if (!job?.result) return [];
  const catalogById = new Map((assets ?? []).map((asset) => [asset.id, asset]));
  const resultAssets = Array.isArray(job.result.assets) ? job.result.assets : [];
  const resultById = new Map(resultAssets.map((asset) => [asset.id, catalogById.get(asset.id) ?? asset]));
  const assetIds = job.result.assetIds ?? [];
  if (assetIds.length) {
    return assetIds.map((id) => resultById.get(id) ?? catalogById.get(id)).filter(Boolean);
  }
  if (resultAssets.length) {
    return resultAssets.map((asset) => catalogById.get(asset.id) ?? asset);
  }
  if (job.result.generationSetId) {
    return (assets ?? []).filter((asset) => asset.generationSetId === job.result.generationSetId);
  }
  return [];
}
