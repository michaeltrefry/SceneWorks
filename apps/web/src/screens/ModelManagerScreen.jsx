import React, { useEffect, useState } from "react";
import { JobProgressCard } from "../components/JobProgress.jsx";
import { terminalStatuses } from "../constants.js";
import { presetLoraId, presetLoras } from "../presetUtils.js";

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

// MLX status text, keyed off the macOS catalog's mlxConversionState. Turnkey
// ("ready") models fetch their MLX weights automatically on first generation;
// convert-required models need the native checkpoint downloaded, then converted.
function mlxStatusText(model) {
  switch (model.mlxConversionState) {
    case "ready":
      return model.mlxInstallState === "installed"
        ? "MLX weights installed."
        : "MLX weights download automatically on first generation.";
    case "needs_source":
      return "Download the model first, then convert it to MLX.";
    case "needs_conversion":
      return "Native checkpoint downloaded — ready to convert to MLX.";
    case "converted":
      return "Converted to MLX and ready.";
    default:
      return "";
  }
}

const MODEL_TYPE_OPTIONS = [
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
  { value: "utility", label: "Utility" },
];

function referencedPresetNames(recipePresets, kind, id) {
  return recipePresets
    .filter((preset) => {
      if (kind === "model") {
        return preset.model === id;
      }
      return presetLoras(preset).some((lora) => presetLoraId(lora) === id);
    })
    .map((preset) => preset.name ?? preset.id)
    .filter(Boolean);
}

function deleteConfirmation(kind, item, recipePresets) {
  const name = item.name ?? item.id;
  const presetNames = referencedPresetNames(recipePresets, kind, item.id);
  const lines = [
    `Delete ${kind} "${name}"?`,
    "This removes the registry entry and SceneWorks-owned local files when available.",
  ];
  if (presetNames.length) {
    lines.push(`Referenced by presets: ${presetNames.slice(0, 5).join(", ")}.`);
    lines.push("Those presets will keep a broken reference until updated.");
  }
  if (item.scope === "builtin" || item.catalogScope === "builtin") {
    lines.push("Built-in catalog entries stay protected; only local installed files can be removed.");
  }
  return lines.join("\n\n");
}

function deleteResultText(result, name) {
  const removed = result?.removedManifestEntry ? "Removed the registry entry" : "Removed local files";
  const warnings = result?.warnings?.length ? ` ${result.warnings.join(" ")}` : "";
  return `${removed} for ${name}.${warnings}`;
}

export function ModelManagerScreen({
  activeProject,
  jobs,
  loras,
  models,
  onConvertModel,
  onDeleteLora,
  onDeleteModel,
  onDownloadModel,
  onImportLora,
  onImportModel,
  onOpenQueue,
  recipePresets = [],
}) {
  const families = Array.from(new Set(models.map((model) => model.family).filter(Boolean))).sort();
  const familiesKey = families.join("|");
  const [familyFilter, setFamilyFilter] = useState("all");
  const [importingLora, setImportingLora] = useState(false);
  const [importMessage, setImportMessage] = useState({ tone: "neutral", text: "" });
  const [importForm, setImportForm] = useState({
    mode: "url",
    sourceUrl: "",
    file: null,
    name: "",
    scope: "global",
    family: "",
  });
  const [fileInputKey, setFileInputKey] = useState(0);
  const [importingModel, setImportingModel] = useState(false);
  const [modelImportMessage, setModelImportMessage] = useState({ tone: "neutral", text: "" });
  const [modelImportForm, setModelImportForm] = useState({
    mode: "url",
    sourceUrl: "",
    file: null,
    name: "",
    type: "image",
    family: "",
  });
  const [modelFileInputKey, setModelFileInputKey] = useState(0);
  const [deletingItem, setDeletingItem] = useState("");
  const [deleteMessage, setDeleteMessage] = useState({ tone: "neutral", text: "" });
  // Desktop only: read the host's unified memory so MLX models can be gated against
  // their memory tier. Browser/Docker builds have no Tauri bridge and skip this.
  const isDesktop = typeof window !== "undefined" && Boolean(window.__TAURI__);
  const [unifiedMemoryGb, setUnifiedMemoryGb] = useState(null);
  const visibleLoras = loras.filter((lora) => matchesFamily(lora, familyFilter));

  useEffect(() => {
    if (familyFilter !== "all" && !families.includes(familyFilter)) {
      setFamilyFilter("all");
    }
  }, [familiesKey, familyFilter]);

  useEffect(() => {
    setImportForm((current) => (current.family && !families.includes(current.family) ? { ...current, family: "" } : current));
  }, [familiesKey]);

  useEffect(() => {
    if (!isDesktop) {
      return undefined;
    }
    let cancelled = false;
    window.__TAURI__.core
      .invoke("get_gpu_info")
      .then((info) => {
        if (!cancelled && info && typeof info.unifiedMemoryMb === "number") {
          setUnifiedMemoryGb(info.unifiedMemoryMb / 1024);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [isDesktop]);

  function downloadJobsFor(model) {
    return jobs.filter((job) => job.type === "model_download" && job.payload?.modelId === model.id);
  }

  function convertJobsFor(model) {
    return jobs.filter((job) => job.type === "model_convert" && job.payload?.modelId === model.id);
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
      const familyOverride = importForm.family ? { family: importForm.family } : {};
      const job = await onImportLora({
        ...(isFileImport ? { file: importForm.file } : { sourceUrl: importForm.sourceUrl.trim() }),
        name: importForm.name.trim() || undefined,
        scope: importForm.scope,
        ...familyOverride,
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

  async function importModel(event) {
    event.preventDefault();
    const isFileImport = modelImportForm.mode === "file";
    if ((!isFileImport && !modelImportForm.sourceUrl.trim()) || (isFileImport && !modelImportForm.file) || !onImportModel) {
      return;
    }
    setImportingModel(true);
    setModelImportMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading model file before queueing import." : "",
    });
    try {
      const familyOverride = modelImportForm.family ? { family: modelImportForm.family } : {};
      const job = await onImportModel({
        ...(isFileImport ? { file: modelImportForm.file } : { sourceUrl: modelImportForm.sourceUrl.trim() }),
        name: modelImportForm.name.trim() || undefined,
        modelType: modelImportForm.type,
        ...familyOverride,
      });
      const modelId = job?.payload?.modelId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      const detectionNote =
        !modelImportForm.family && resolvedFamily ? ` Detected family: ${resolvedFamily}.` : "";
      setModelImportForm((current) => ({ ...current, sourceUrl: "", file: null, name: "" }));
      setModelFileInputKey((current) => current + 1);
      setModelImportMessage({
        tone: "success",
        text: `${modelId ? `Model import queued for ${modelId}.` : "Model import queued."}${detectionNote}`,
      });
    } catch (err) {
      setModelImportMessage({ tone: "error", text: err.message });
    } finally {
      setImportingModel(false);
    }
  }

  async function deleteModel(model) {
    if (!onDeleteModel || model.removable === false) {
      return;
    }
    if (typeof window.confirm === "function" && !window.confirm(deleteConfirmation("model", model, recipePresets))) {
      return;
    }
    setDeletingItem(`model:${model.id}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await onDeleteModel(model);
      setDeleteMessage({ tone: "success", text: deleteResultText(result, model.name ?? model.id) });
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  async function deleteLora(lora) {
    if (!onDeleteLora || lora.removable === false) {
      return;
    }
    if (typeof window.confirm === "function" && !window.confirm(deleteConfirmation("lora", lora, recipePresets))) {
      return;
    }
    setDeletingItem(`lora:${lora.scope ?? "global"}:${lora.id}`);
    setDeleteMessage({ tone: "neutral", text: "" });
    try {
      const result = await onDeleteLora(lora);
      setDeleteMessage({ tone: "success", text: deleteResultText(result, lora.name ?? lora.id) });
    } catch (err) {
      setDeleteMessage({ tone: "error", text: err.message });
    } finally {
      setDeletingItem("");
    }
  }

  const completedImportTimes = completedLoraImportTimes(jobs);
  const pendingLoraImportJobs = jobs.filter((job) => job.type === "lora_import" && !isSupersededLoraImport(job, completedImportTimes));
  const localLoraImportJobs = pendingLoraImportJobs.filter((job) => job.status !== "completed" && matchesFamily(job, familyFilter));
  const pendingModelImportJobs = jobs.filter((job) => job.type === "model_import" && job.status !== "completed");
  const isModelFileImport = modelImportForm.mode === "file";
  const modelImportDisabled =
    importingModel ||
    !onImportModel ||
    (isModelFileImport ? !modelImportForm.file : !modelImportForm.sourceUrl.trim());
  const hiddenImportCount =
    familyFilter === "all" ? 0 : pendingLoraImportJobs.filter((job) => job.status !== "completed" && !matchesFamily(job, familyFilter)).length;
  const visibleLoraCount = visibleLoras.length + localLoraImportJobs.length;
  const installedLoraCount = visibleLoras.filter((lora) => lora.installState === "installed").length;
  const unavailableLoraCount = visibleLoras.filter((lora) => lora.installState === "missing").length;
  const pendingLoraCount = visibleLoraCount - installedLoraCount - unavailableLoraCount;
  const loraCountText = [
    installedLoraCount ? `${installedLoraCount} installed` : null,
    unavailableLoraCount ? `${unavailableLoraCount} unavailable` : null,
    pendingLoraCount ? `${pendingLoraCount} pending` : null,
  ].filter(Boolean).join(" · ") || "0 visible";
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
          const unassociated = !model.family;
          const deleteKey = `model:${model.id}`;
          const canDelete = Boolean(onDeleteModel) && model.removable !== false;
          // MLX (macOS) variant: only present when the catalog computed mlxConversionState.
          const mlxState = model.mlxConversionState;
          const mlxMinGb = model.mlx?.minMemoryGb ?? null;
          const mlxEnoughMemory =
            unifiedMemoryGb == null || mlxMinGb == null || unifiedMemoryGb >= mlxMinGb;
          const convertJobs = convertJobsFor(model);
          const convertJob = convertJobs.find((job) => !terminalStatuses.has(job.status));
          const failedConvert = convertJobs.find(
            (job) => terminalStatuses.has(job.status) && job.status !== "completed",
          );
          const showConvertButton = mlxState === "needs_conversion" || mlxState === "converted";
          return (
            <article className="model-card" key={model.id}>
              <div>
                <p className="eyebrow">{model.type}</p>
                <h3>{model.name}</h3>
              </div>
              <span className={installed ? "status-badge installed" : "status-badge"}>{installed ? "installed" : "missing"}</span>
              {unassociated ? (
                <span className="status-badge warning" title="Set this model's family in user.models.jsonc before using it for generation.">
                  needs family
                </span>
              ) : null}
              <p>{model.ui?.description ?? model.family ?? model.id}</p>
              <dl>
                <div>
                  <dt>Family</dt>
                  <dd>{model.family ?? "unassociated"}</dd>
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
              {mlxState ? (
                <div className="mlx-status">
                  <div className="mlx-status-badges">
                    <span className="status-badge">MLX</span>
                    {mlxMinGb != null ? (
                      <span className={mlxEnoughMemory ? "status-badge" : "status-badge warning"}>
                        needs ≥ {mlxMinGb} GB
                      </span>
                    ) : null}
                  </div>
                  <p>{mlxStatusText(model)}</p>
                  {!mlxEnoughMemory ? (
                    <p className="inline-warning">
                      Needs ≥ {mlxMinGb} GB unified memory; this Mac has ~{Math.round(unifiedMemoryGb)} GB. It may run out of memory.
                    </p>
                  ) : null}
                  {convertJob ? (
                    <JobProgressCard job={convertJob} label="MLX conversion" onOpenQueue={onOpenQueue} />
                  ) : null}
                  {showConvertButton ? (
                    <button
                      disabled={mlxState === "converted" || Boolean(convertJob) || !mlxEnoughMemory}
                      onClick={() => onConvertModel?.(model)}
                      type="button"
                    >
                      {convertJob
                        ? convertJob.status
                        : mlxState === "converted"
                          ? "MLX ready"
                          : failedConvert
                            ? "Retry MLX Conversion"
                            : "Convert to MLX"}
                    </button>
                  ) : null}
                </div>
              ) : null}
              <div className="model-card-actions">
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
                <button className="danger-action" disabled={!canDelete || deletingItem === deleteKey} onClick={() => deleteModel(model)} type="button">
                  {model.removable === false ? "Protected" : deletingItem === deleteKey ? "Deleting" : "Delete"}
                </button>
              </div>
            </article>
          );
        })}
      </div>
      {deleteMessage.text ? <p className={deleteMessage.tone === "success" ? "inline-success" : "inline-warning"}>{deleteMessage.text}</p> : null}

      <section className="model-import-panel-section">
        <form className="lora-import-panel models-import-panel" aria-label="Import model" onSubmit={importModel}>
          <div>
            <strong>Import model</strong>
            <span>{modelImportForm.family || "auto-detect family"}</span>
          </div>
          <div className="segmented-control compact-segment" aria-label="Model import source">
            <button
              className={modelImportForm.mode === "url" ? "active" : ""}
              disabled={importingModel}
              onClick={() => setModelImportForm((current) => ({ ...current, mode: "url" }))}
              type="button"
            >
              URL
            </button>
            <button
              className={modelImportForm.mode === "file" ? "active" : ""}
              disabled={importingModel}
              onClick={() => setModelImportForm((current) => ({ ...current, mode: "file" }))}
              type="button"
            >
              Upload
            </button>
          </div>
          <div className="models-import-grid">
            <label>
              Type
              <select
                disabled={importingModel}
                onChange={(event) => setModelImportForm((current) => ({ ...current, type: event.target.value }))}
                value={modelImportForm.type}
              >
                {MODEL_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Family
              <select
                disabled={importingModel || !families.length}
                onChange={(event) => setModelImportForm((current) => ({ ...current, family: event.target.value }))}
                value={modelImportForm.family}
              >
                {families.length ? (
                  <>
                    <option value="">Auto-detect</option>
                    {families.map((family) => (
                      <option key={family} value={family}>
                        {family}
                      </option>
                    ))}
                  </>
                ) : (
                  <option value="">No known families</option>
                )}
              </select>
            </label>
            {isModelFileImport ? (
              <label>
                Model File
                <span className="file-picker-row">
                  <span className="file-upload-button">
                    Choose
                    <input
                      accept=".safetensors,.ckpt,.pt,.bin"
                      disabled={importingModel}
                      key={modelFileInputKey}
                      onChange={(event) => setModelImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                      type="file"
                    />
                  </span>
                  <span className="selected-file-name">{modelImportForm.file?.name ?? "No file selected"}</span>
                </span>
              </label>
            ) : (
              <label>
                Source URL
                <input
                  disabled={importingModel}
                  onChange={(event) => setModelImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                  placeholder="https://..."
                  value={modelImportForm.sourceUrl}
                />
              </label>
            )}
            <label>
              Name
              <input
                disabled={importingModel}
                onChange={(event) => setModelImportForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Optional"
                value={modelImportForm.name}
              />
            </label>
            <button disabled={modelImportDisabled} type="submit">
              {importingModel ? (isModelFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
            </button>
          </div>
          {modelImportMessage.text ? <p className={modelImportMessage.tone === "success" ? "inline-success" : "inline-warning"}>{modelImportMessage.text}</p> : null}
        </form>
        {pendingModelImportJobs.length ? (
          <div className="lora-import-progress">
            <strong>Model imports in progress</strong>
            <div className="local-job-stack">
              {pendingModelImportJobs.map((job) => (
                <JobProgressCard job={job} key={job.id} label="Model import" onOpenQueue={onOpenQueue} />
              ))}
            </div>
          </div>
        ) : null}
      </section>

      <section className="lora-panel">
        <div className="lora-panel-header">
          <div className="section-heading">
            <p className="eyebrow">LoRAs</p>
            <h2>{familyFilter === "all" ? "All compatible" : familyFilter}</h2>
          </div>
          <span>{loraCountText}</span>
        </div>
        <form className="lora-import-panel models-import-panel" aria-label="Import LoRA" onSubmit={importLora}>
          <div>
            <strong>Import LoRA</strong>
            <span>{importForm.family || "auto-detect"}</span>
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
                    <option value="">Auto-detect</option>
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
            {visibleLoras.map((lora) => {
              const installed = lora.installState === "installed";
              const missing = lora.installState === "missing";
              const statusText = missing ? "unavailable" : installed ? "installed" : "pending";
              return (
                <article className={missing ? "lora-row warning" : "lora-row"} key={lora.id ?? lora.name}>
                  <span>
                    <strong>{lora.name ?? lora.id}</strong>
                    <small>{[lora.scope, lora.family ?? "compatible"].filter(Boolean).join(" | ")}</small>
                  </span>
                  <span className={installed ? "status-badge installed" : "status-badge"}>{statusText}</span>
                  <button
                    className="danger-action"
                    disabled={!onDeleteLora || lora.removable === false || deletingItem === `lora:${lora.scope ?? "global"}:${lora.id}`}
                    onClick={() => deleteLora(lora)}
                    type="button"
                  >
                    {lora.removable === false ? "Protected" : deletingItem === `lora:${lora.scope ?? "global"}:${lora.id}` ? "Deleting" : "Delete"}
                  </button>
                </article>
              );
            })}
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
