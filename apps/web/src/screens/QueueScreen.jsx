import React, { useMemo } from "react";
import { actionStatuses, terminalStatuses } from "../constants.js";
import { formatSeconds, percent } from "../formatting.js";

const nonGpuJobTypes = new Set(["model_download", "lora_import"]);
const terminalStatusesForBlocking = new Set(["completed", "failed", "canceled", "interrupted"]);

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
  if (nonGpuJobTypes.has(job.type)) {
    return true;
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
      !terminalStatusesForBlocking.has(candidate.status) &&
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
  return dependency && !terminalStatusesForBlocking.has(dependency.status) ? dependency : null;
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
  return nonGpuJobTypes.has(job.type) ? "Waiting for a utility worker." : "Waiting for an available GPU worker.";
}

function workerStatusLine(worker) {
  if (worker.status === "busy") {
    return `Busy${worker.currentJobId ? ` with ${worker.currentJobId}` : ""}`;
  }
  return worker.status === "idle" ? "Ready" : worker.status;
}

export function QueueScreen({
  activeProject,
  createJob,
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
  setRequestedGpu,
  workers,
}) {
  const workersById = useMemo(() => new Map(workers.map((worker) => [worker.id, worker])), [workers]);
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
        {workers.length === 0 ? (
          <div className="worker-card">
            <strong>No workers registered</strong>
            <span>Start the worker service to claim queued jobs.</span>
          </div>
        ) : (
          workers.map((worker) => (
            <div className="worker-card" key={worker.id}>
              <strong>{worker.gpuName ?? worker.gpuId}</strong>
              <span>{workerStatusLine(worker)}</span>
              <small>{worker.loadedModels?.length ? `Warm: ${worker.loadedModels.join(", ")}` : "No warm model"}</small>
            </div>
          ))
        )}
      </div>

      <div className="job-list">
        {filteredJobs.length === 0 ? (
          <div className="empty-panel">No jobs in this view</div>
        ) : (
          filteredJobs.map((job) => (
            <JobRow
              assignedWorker={workersById.get(job.workerId)}
              job={job}
              jobAction={jobAction}
              key={job.id}
              jobs={jobs}
              workers={workers}
            />
          ))
        )}
      </div>
    </section>
  );
}

function JobRow({ assignedWorker, job, jobAction, jobs, workers }) {
  const canCancel = !terminalStatuses.has(job.status);
  const maxAttempts = 5;
  const attempts = job.attempts ?? 1;
  const canRepeat = actionStatuses.has(job.status) && attempts < maxAttempts;
  const displayMessage = jobWaitingMessage(job, workers, jobs);
  return (
    <article className={`job-row ${job.status}`}>
      <div className="job-main">
        <div>
          <p className="eyebrow">{job.type}</p>
          <h3>{job.payload.prompt ?? job.id}</h3>
        </div>
        <span className="status-badge">{job.status}</span>
      </div>
      <div className="job-meta">
        <span>{job.projectName ?? "Global"}</span>
        <span>Stage {job.stage}</span>
        <span>Elapsed {formatSeconds(job.elapsedSeconds)}</span>
        <span>GPU {job.assignedGpu ?? job.requestedGpu}</span>
        {assignedWorker ? <span>{assignedWorker.gpuName ?? assignedWorker.id}</span> : null}
        <span>Attempt {attempts}/{maxAttempts}</span>
      </div>
      <div className="progress-track" aria-label={`${percent(job.progress)} complete`}>
        <span style={{ width: percent(job.progress) }} />
      </div>
      <p className={job.error ? "job-message error-text" : "job-message"}>{displayMessage}</p>
      <div className="job-actions">
        <button disabled={!canCancel || job.cancelRequested} onClick={() => jobAction(job, "cancel")} type="button">
          Cancel
        </button>
        <button disabled={!canRepeat} onClick={() => jobAction(job, "retry")} type="button">
          Retry
        </button>
        <button disabled={!canRepeat} onClick={() => jobAction(job, "duplicate")} type="button">
          Duplicate
        </button>
      </div>
    </article>
  );
}
