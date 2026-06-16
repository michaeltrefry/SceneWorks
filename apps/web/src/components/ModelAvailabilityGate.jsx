import React from "react";
import { WorkerProgressCard } from "./WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";

// Per-Studio model-availability gate (sc-5947). When a Studio has no installed model that
// supports its functions, it renders this instead of its body: a short explanation plus the
// screen's recommended models with an inline Download. A completed download refreshes the
// catalog (App.jsx SSE handler), so `ready` flips and the Studio renders without a reload.
//
// Props:
//   ready          — when true, render `children` (the Studio body) unchanged.
//   title/description — gate copy.
//   offers         — models to offer for download (downloadOffersFor in modelEligibility.js).
//   downloadJobs   — model_download jobs, to show progress for an in-flight offer.
//   onDownload(model) / onOpenModels() / onOpenQueue() / onCancelJob(job) — wired from context.
function offerSizeText(model) {
  if (!model?.downloadSizeLabel) {
    return "Size unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

export function ModelAvailabilityGate({
  ready,
  title,
  description,
  offers = [],
  downloadJobs = [],
  onDownload,
  onOpenModels,
  onOpenQueue,
  onCancelJob,
  children,
}) {
  if (ready) {
    return children;
  }
  const activeJobFor = (model) =>
    downloadJobs.find((job) => job.payload?.modelId === model.id && !terminalStatuses.has(job.status));
  return (
    <section className="model-availability-gate">
      <div className="model-availability-gate-card">
        <div className="section-heading">
          <p className="eyebrow">No supported model installed</p>
          <h2>{title}</h2>
        </div>
        {description ? <p>{description}</p> : null}
        {offers.length ? (
          <div className="model-availability-offers">
            {offers.map((model) => {
              const job = activeJobFor(model);
              return (
                <article className="model-availability-offer" key={model.id}>
                  <div className="model-availability-offer-head">
                    <span>
                      <strong>{model.name ?? model.id}</strong>
                      <small>{offerSizeText(model)}</small>
                    </span>
                    <button
                      disabled={!onDownload || Boolean(job)}
                      onClick={() => onDownload?.(model)}
                      type="button"
                    >
                      {job ? job.status : "Download"}
                    </button>
                  </div>
                  {job ? (
                    <WorkerProgressCard job={job} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                  ) : null}
                </article>
              );
            })}
          </div>
        ) : (
          <p className="empty-panel compact-panel">No downloadable model in the catalog supports this screen yet.</p>
        )}
        {onOpenModels ? (
          <button className="model-availability-browse" onClick={onOpenModels} type="button">
            Browse all models
          </button>
        ) : null}
      </div>
    </section>
  );
}
