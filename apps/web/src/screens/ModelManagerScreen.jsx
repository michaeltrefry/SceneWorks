import React, { useEffect, useState } from "react";
import { JobProgressCard } from "../components/JobProgress.jsx";
import { terminalStatuses } from "../constants.js";

function loraFamilies(item) {
  // Accept either a LoRA catalog entry or a lora_import job snapshot.
  const compatibility = item.compatibility ?? {};
  const values =
    item.families ??
    item.compatibleFamilies ??
    item.modelFamilies ??
    compatibility.families ??
    item.payload?.manifestEntry?.families ??
    item.payload?.manifestEntry?.compatibleFamilies ??
    item.payload?.manifestEntry?.modelFamilies ??
    item.payload?.manifestEntry?.compatibility?.families ??
    item.payload?.family ??
    item.payload?.manifestEntry?.family ??
    item.family ??
    [];
  return Array.isArray(values) ? values : [values].filter(Boolean);
}

function matchesFamily(item, familyFilter) {
  if (familyFilter === "all") {
    return true;
  }
  const families = loraFamilies(item);
  // Import jobs can briefly lack family metadata; completed catalog entries should not.
  return item.type === "lora_import" && families.length === 0 ? true : families.includes(familyFilter);
}

function loraImportKey(job) {
  return job.payload?.loraId ?? job.payload?.sourceUrl ?? job.payload?.sourcePath ?? job.payload?.name ?? null;
}

function completedLoraImportTimes(jobs) {
  const completed = new Map();
  jobs
    .filter((job) => job.type === "lora_import" && job.status === "completed")
    .forEach((job) => {
      const key = loraImportKey(job);
      if (!key || !job.createdAt) {
        return;
      }
      const previous = completed.get(key);
      if (!previous || job.createdAt.localeCompare(previous) > 0) {
        completed.set(key, job.createdAt);
      }
    });
  return completed;
}

function isSupersededLoraImport(job, completedTimes) {
  const key = loraImportKey(job);
  const completedAt = key ? completedTimes.get(key) : null;
  return Boolean(completedAt) && terminalStatuses.has(job.status) && job.status !== "completed" && completedAt.localeCompare(job.createdAt ?? "") > 0;
}

function downloadSizeText(model) {
  if (!model.downloadSizeLabel) {
    return "Unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

export function ModelManagerScreen({ activeProject, jobs, loras, models, onDownloadModel, onImportLora, onOpenQueue }) {
  const families = Array.from(new Set(models.map((model) => model.family).filter(Boolean))).sort();
  const familiesKey = families.join("|");
  const [familyFilter, setFamilyFilter] = useState(families[0] ?? "all");
  const [importingLora, setImportingLora] = useState(false);
  const [importMessage, setImportMessage] = useState({ tone: "neutral", text: "" });
  const [importForm, setImportForm] = useState({
    mode: "url",
    sourceUrl: "",
    file: null,
    name: "",
    scope: "global",
    family: families[0] ?? "",
  });
  const [fileInputKey, setFileInputKey] = useState(0);
  const visibleLoras = loras.filter((lora) => matchesFamily(lora, familyFilter));

  useEffect(() => {
    if (familyFilter !== "all" && !families.includes(familyFilter)) {
      setFamilyFilter(families[0] ?? "all");
    }
  }, [familiesKey, familyFilter]);

  useEffect(() => {
    setImportForm((current) => {
      if (current.family && families.includes(current.family)) {
        return current;
      }
      const family = familyFilter !== "all" && families.includes(familyFilter) ? familyFilter : families[0] ?? "";
      return { ...current, family };
    });
  }, [familiesKey, familyFilter]);

  function downloadJobsFor(model) {
    return jobs.filter((job) => job.type === "model_download" && job.payload?.modelId === model.id);
  }

  async function importLora(event) {
    event.preventDefault();
    const isFileImport = importForm.mode === "file";
    if ((!isFileImport && !importForm.sourceUrl.trim()) || (isFileImport && !importForm.file) || !onImportLora) {
      return;
    }
    setImportingLora(true);
    setImportMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading LoRA file before queueing import." : "",
    });
    try {
      const job = await onImportLora({
        ...(isFileImport ? { file: importForm.file } : { sourceUrl: importForm.sourceUrl.trim() }),
        name: importForm.name.trim() || undefined,
        scope: importForm.scope,
        family: importForm.family || undefined,
      });
      const loraId = job?.payload?.loraId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detectionNote =
        !importForm.family && resolvedFamily ? ` Detected family: ${resolvedFamily}.` : "";
      setImportForm((current) => ({ ...current, sourceUrl: "", file: null, name: "" }));
      // Force a re-mount so choosing the same file again still emits a change event.
      setFileInputKey((current) => current + 1);
      setImportMessage({
        tone: "success",
        text: `${loraId ? `LoRA import queued for ${loraId}.` : "LoRA import queued."}${detectionNote}`,
      });
    } catch (err) {
      setImportMessage({ tone: "error", text: err.message });
    } finally {
      setImportingLora(false);
    }
  }

  const completedImportTimes = completedLoraImportTimes(jobs);
  const pendingLoraImportJobs = jobs.filter((job) => job.type === "lora_import" && !isSupersededLoraImport(job, completedImportTimes));
  const localLoraImportJobs = pendingLoraImportJobs.filter((job) => job.status !== "completed" && matchesFamily(job, familyFilter));
  const hiddenImportCount =
    familyFilter === "all" ? 0 : pendingLoraImportJobs.filter((job) => job.status !== "completed" && !matchesFamily(job, familyFilter)).length;
  const visibleLoraCount = visibleLoras.length + localLoraImportJobs.length;
  const isFileImport = importForm.mode === "file";
  const importDisabled =
    importingLora ||
    !onImportLora ||
    (importForm.scope === "project" && !activeProject) ||
    (isFileImport ? !importForm.file : !importForm.sourceUrl.trim());

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
          const downloadJobs = downloadJobsFor(model);
          const downloadJob = downloadJobs.find((job) => !terminalStatuses.has(job.status));
          const installed = model.installState === "installed";
          const localDownloadJob = installed ? null : downloadJobs.find((job) => job.status !== "completed");
          const failedDownload = localDownloadJob && terminalStatuses.has(localDownloadJob.status);
          const downloadSize = downloadSizeText(model);
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
                  <dt>Download size</dt>
                  <dd>{downloadSize}</dd>
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
                        ? `Download ${downloadSize}`
                        : "Download"}
              </button>
            </article>
          );
        })}
      </div>

      <section className="lora-panel">
        <div className="lora-panel-header">
          <div className="section-heading">
            <p className="eyebrow">LoRAs</p>
            <h2>{familyFilter === "all" ? "All compatible" : familyFilter}</h2>
          </div>
          <span>{visibleLoraCount} installed or pending</span>
        </div>
        <form className="lora-import-panel models-import-panel" aria-label="Import LoRA" onSubmit={importLora}>
          <div>
            <strong>Import LoRA</strong>
            <span>{importForm.family || "choose a family"}</span>
          </div>
          <div className="segmented-control compact-segment" aria-label="LoRA import source">
            <button
              className={importForm.mode === "url" ? "active" : ""}
              disabled={importingLora}
              onClick={() => setImportForm((current) => ({ ...current, mode: "url" }))}
              type="button"
            >
              URL
            </button>
            <button
              className={importForm.mode === "file" ? "active" : ""}
              disabled={importingLora}
              onClick={() => setImportForm((current) => ({ ...current, mode: "file" }))}
              type="button"
            >
              Upload
            </button>
          </div>
          <div className="models-import-grid">
            <label>
              Scope
              <select
                disabled={importingLora}
                onChange={(event) => setImportForm((current) => ({ ...current, scope: event.target.value }))}
                value={importForm.scope}
              >
                <option value="global">Global</option>
                <option disabled={!activeProject} value="project">
                  Project
                </option>
              </select>
            </label>
            <label>
              Family
              <select
                disabled={importingLora || !families.length}
                onChange={(event) => setImportForm((current) => ({ ...current, family: event.target.value }))}
                value={importForm.family}
              >
                {families.length ? (
                  <>
                    <option value="">Auto-detect from file</option>
                    {families.map((family) => (
                      <option key={family} value={family}>
                        {family}
                      </option>
                    ))}
                  </>
                ) : (
                  <option value="">No model families</option>
                )}
              </select>
            </label>
            {isFileImport ? (
              <label>
                LoRA File
                <span className="file-picker-row">
                  <span className="file-upload-button">
                    Choose
                    <input
                      accept=".safetensors,.ckpt,.pt,.bin"
                      disabled={importingLora}
                      key={fileInputKey}
                      onChange={(event) => setImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                      type="file"
                    />
                  </span>
                  <span className="selected-file-name">{importForm.file?.name ?? "No file selected"}</span>
                </span>
              </label>
            ) : (
              <label>
                Source URL
                <input
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                  placeholder="https://..."
                  value={importForm.sourceUrl}
                />
              </label>
            )}
            <label>
              Name
              <input
                disabled={importingLora}
                onChange={(event) => setImportForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Optional"
                value={importForm.name}
              />
            </label>
            <button disabled={importDisabled} type="submit">
              {importingLora ? (isFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
            </button>
          </div>
          {importForm.scope === "project" && !activeProject ? <p className="helper-copy">Open a project before importing a project LoRA.</p> : null}
          {importMessage.text ? <p className={importMessage.tone === "success" ? "inline-success" : "inline-warning"}>{importMessage.text}</p> : null}
        </form>
        {localLoraImportJobs.length ? (
          <div className="lora-import-progress">
            <strong>LoRA imports in progress</strong>
            <div className="local-job-stack">
              {localLoraImportJobs.map((job) => (
                <JobProgressCard job={job} key={job.id} label="LoRA import" onOpenQueue={onOpenQueue} />
              ))}
            </div>
          </div>
        ) : null}
        {hiddenImportCount ? <p className="helper-copy">{hiddenImportCount} LoRA import{hiddenImportCount === 1 ? " is" : "s are"} hidden by this family filter.</p> : null}
        {visibleLoras.length ? (
          <div className="lora-list">
            {visibleLoras.map((lora) => (
              <article className="lora-row" key={lora.id ?? lora.name}>
                <strong>{lora.name ?? lora.id}</strong>
                <span>{[lora.scope, lora.family ?? "compatible"].filter(Boolean).join(" | ")}</span>
              </article>
            ))}
          </div>
        ) : localLoraImportJobs.length ? null : loras.length && familyFilter !== "all" ? (
          <div className="empty-panel compact-panel">No LoRAs match {familyFilter}</div>
        ) : (
          <div className="empty-panel compact-panel">No LoRAs in this view</div>
        )}
      </section>
    </section>
  );
}
