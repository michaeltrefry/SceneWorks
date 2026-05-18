import React, { useEffect, useState } from "react";
import { JobProgressCard } from "../components/JobProgress.jsx";
import { terminalStatuses } from "../constants.js";

export function ModelManagerScreen({ jobs, loras, models, onDownloadModel, onOpenQueue }) {
  const families = Array.from(new Set(models.map((model) => model.family).filter(Boolean))).sort();
  const [familyFilter, setFamilyFilter] = useState(families[0] ?? "all");
  const visibleLoras =
    familyFilter === "all"
      ? loras
      : loras.filter((lora) => {
          const compatibility = lora.compatibility ?? {};
          const values =
            lora.families ??
            lora.compatibleFamilies ??
            lora.modelFamilies ??
            compatibility.families ??
            (lora.family ? [lora.family] : []);
          const families = Array.isArray(values) ? values : [values];
          return families.includes(familyFilter);
        });

  useEffect(() => {
    if (familyFilter !== "all" && !families.includes(familyFilter)) {
      setFamilyFilter(families[0] ?? "all");
    }
  }, [families.join("|"), familyFilter]);

  function activeDownloadFor(model) {
    return jobs.find(
      (job) =>
        job.type === "model_download" &&
        job.payload?.modelId === model.id &&
        !terminalStatuses.has(job.status),
    );
  }

  function localDownloadFor(model) {
    return jobs.find(
      (job) =>
        job.type === "model_download" &&
        job.payload?.modelId === model.id &&
        job.status !== "completed",
    );
  }

  return (
    <section className="main-surface models-surface">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Runtime assets</p>
          <h2>Models</h2>
        </div>
        <label>
          LoRA family
          <select onChange={(event) => setFamilyFilter(event.target.value)} value={familyFilter}>
            <option value="all">All families</option>
            {families.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="model-grid">
        {models.map((model) => {
          const downloadJob = activeDownloadFor(model);
          const installed = model.installState === "installed";
          const localDownloadJob = installed ? null : localDownloadFor(model);
          const failedDownload = localDownloadJob && terminalStatuses.has(localDownloadJob.status);
          return (
            <article className="model-card" key={model.id}>
              <div>
                <p className="eyebrow">{model.type}</p>
                <h3>{model.name}</h3>
              </div>
              <span className={installed ? "status-badge installed" : "status-badge"}>{installed ? "installed" : "missing"}</span>
              <p>{model.ui?.description ?? model.family ?? model.id}</p>
              <dl>
                <div>
                  <dt>Family</dt>
                  <dd>{model.family ?? "unknown"}</dd>
                </div>
                <div>
                  <dt>Repo</dt>
                  <dd>{model.downloads?.[0]?.repo ?? "none"}</dd>
                </div>
                <div>
                  <dt>Download</dt>
                  <dd>{model.downloadSizeLabel ?? "unknown"}</dd>
                </div>
              </dl>
              {localDownloadJob ? (
                <JobProgressCard job={localDownloadJob} label="Model download" onOpenQueue={onOpenQueue} />
              ) : null}
              <button disabled={installed || !model.downloadable || Boolean(downloadJob)} onClick={() => onDownloadModel(model)} type="button">
                {downloadJob
                  ? downloadJob.status
                  : installed
                    ? "Ready"
                    : failedDownload
                      ? "Retry Download"
                      : model.downloadSizeLabel
                        ? `Download ${model.downloadSizeLabel}`
                        : "Download"}
              </button>
            </article>
          );
        })}
      </div>

      <section className="lora-panel">
        <div className="section-heading">
          <p className="eyebrow">LoRAs</p>
          <h2>{familyFilter === "all" ? "All compatible" : familyFilter}</h2>
        </div>
        {visibleLoras.length ? (
          <div className="lora-list">
            {visibleLoras.map((lora) => (
              <article className="lora-row" key={lora.id ?? lora.name}>
                <strong>{lora.name ?? lora.id}</strong>
                <span>{[lora.scope, lora.family ?? "compatible"].filter(Boolean).join(" | ")}</span>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-panel compact-panel">No LoRAs in this view</div>
        )}
      </section>
    </section>
  );
}
