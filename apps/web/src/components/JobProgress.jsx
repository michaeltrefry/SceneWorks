import React from "react";
import { formatSeconds, percent } from "../formatting.js";

const localErrorStatuses = new Set(["failed", "canceled", "interrupted"]);

function formatJobType(type) {
  return String(type ?? "job").replaceAll("_", " ");
}

function jobTitle(job) {
  return job.payload?.prompt ?? job.payload?.modelName ?? job.payload?.modelId ?? job.id;
}

function jobMessage(job) {
  if (job.error || job.message) {
    return job.error ?? job.message;
  }
  if (job.status === "queued") {
    return "Queued and waiting for an eligible worker.";
  }
  if (job.stage && job.stage !== job.status) {
    return `Stage: ${formatJobType(job.stage)}.`;
  }
  return "";
}

export function JobProgressCard({ job, label, onOpenQueue }) {
  const isError = localErrorStatuses.has(job.status);
  const progressLabel = percent(job.progress);
  const message = jobMessage(job);
  return (
    <article className={`local-job-card ${job.status}`}>
      <div className="local-job-main">
        <div>
          <p className="eyebrow">{label ?? formatJobType(job.type)}</p>
          <h3>{jobTitle(job)}</h3>
        </div>
        <span className="status-badge">{job.status}</span>
      </div>
      <div className="progress-track" aria-label={`${progressLabel} complete`}>
        <span style={{ width: progressLabel }} />
      </div>
      <div className="job-meta">
        <span>{formatJobType(job.stage ?? job.status)}</span>
        <span>{formatSeconds(job.elapsedSeconds)}</span>
        <span>GPU {job.assignedGpu ?? job.requestedGpu ?? "auto"}</span>
      </div>
      {message ? <p className={isError ? "job-message error-text" : "job-message"}>{message}</p> : null}
      {isError && onOpenQueue ? (
        <button className="secondary-action" onClick={onOpenQueue} type="button">
          Open Queue
        </button>
      ) : null}
    </article>
  );
}
