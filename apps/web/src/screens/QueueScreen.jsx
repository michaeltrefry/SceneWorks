import React from "react";
import { actionStatuses, terminalStatuses } from "../constants.js";
import { formatSeconds, percent } from "../formatting.js";

export function QueueScreen({
  activeProject,
  createJob,
  filteredJobs,
  gpuOptions,
  jobAction,
  jobPrompt,
  projectFilter,
  projects,
  requestedGpu,
  setJobPrompt,
  setProjectFilter,
  setRequestedGpu,
  workers,
}) {
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
              <span>{worker.status}</span>
              <small>{worker.currentJobId ?? "Idle"}</small>
            </div>
          ))
        )}
      </div>

      <div className="job-list">
        {filteredJobs.length === 0 ? (
          <div className="empty-panel">No jobs in this view</div>
        ) : (
          filteredJobs.map((job) => <JobRow job={job} jobAction={jobAction} key={job.id} />)
        )}
      </div>
    </section>
  );
}

function JobRow({ job, jobAction }) {
  const canCancel = !terminalStatuses.has(job.status);
  const maxAttempts = 5;
  const attempts = job.attempts ?? 1;
  const canRepeat = actionStatuses.has(job.status) && attempts < maxAttempts;
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
        <span>Attempt {attempts}/{maxAttempts}</span>
      </div>
      <div className="progress-track" aria-label={`${percent(job.progress)} complete`}>
        <span style={{ width: percent(job.progress) }} />
      </div>
      <p className={job.error ? "job-message error-text" : "job-message"}>{job.error ?? job.message}</p>
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
