import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";

const tabs = [
  { id: "dataset", label: "Dataset", title: "Dataset intake", status: "Rust dataset store" },
  { id: "rename-caption", label: "Rename & Caption", title: "Rename and caption pass", status: "Needs valid dataset" },
  { id: "configure", label: "Configure Job", title: "Configure training job", status: "Queue dry run" },
];
const defaultGpuOptions = ["auto"];

function formatDatasetModality(dataset) {
  return String(dataset.modality ?? "image").replaceAll("_", " ");
}

function datasetItemCount(dataset) {
  const value = Number(dataset.itemCount ?? dataset.items?.length ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function summarizeDatasets(datasets) {
  return datasets.reduce((summary, dataset) => ({ items: summary.items + datasetItemCount(dataset) }), { items: 0 });
}

function imageAssetName(asset) {
  const path = asset?.file?.path ?? asset?.path ?? asset?.displayName ?? asset?.id ?? "asset";
  return String(path).replaceAll("\\", "/").split("/").pop() || "asset";
}

function captionText(item) {
  return String(item?.caption?.text ?? "").trim();
}

function itemFileStem(item) {
  const name = String(item?.path ?? item?.displayName ?? item?.id ?? "item").replaceAll("\\", "/").split("/").pop() || "item";
  const dotIndex = name.lastIndexOf(".");
  return dotIndex > 0 ? name.slice(0, dotIndex) : name;
}

function triggerWordsText(caption) {
  return (caption?.triggerWords ?? caption?.trigger_words ?? []).join(", ");
}

function parseTriggerWords(value) {
  return String(value ?? "")
    .split(",")
    .map((word) => word.trim())
    .filter(Boolean);
}

function safeSlug(value, fallback = "item") {
  const slug = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return slug || fallback;
}

function orderedName(index, prefix) {
  return `${safeSlug(prefix, "item")}_${String(index + 1).padStart(4, "0")}`;
}

function renameCaptionDrafts(dataset) {
  return (dataset?.items ?? []).map((item, index) => ({
    originalItemId: item.id ?? `item_${String(index + 1).padStart(4, "0")}`,
    itemId: item.id ?? `item_${String(index + 1).padStart(4, "0")}`,
    fileStem: itemFileStem(item),
    displayName: item.displayName ?? imageAssetName(item),
    captionText: item.caption?.text ?? "",
    captionSource: item.caption?.source ?? "manual",
    triggerWords: triggerWordsText(item.caption),
    assetId: item.assetId ?? "",
    path: item.path ?? "",
  }));
}

function renameFieldsDirty(drafts, dataset) {
  const items = dataset?.items ?? [];
  if (drafts.length !== items.length) {
    return true;
  }
  return drafts.some((draft, index) => {
    const item = items[index] ?? {};
    return (
      draft.itemId !== item.id ||
      draft.fileStem !== itemFileStem(item) ||
      draft.displayName !== (item.displayName ?? imageAssetName(item))
    );
  });
}

function normalizeDatasetAssetIds(dataset) {
  return (dataset?.items ?? []).map((item) => item.assetId).filter(Boolean);
}

function datasetHealth({ activeDataset, imageAssets, selectedAssetIds }) {
  const assetsById = new Map(imageAssets.map((asset) => [asset.id, asset]));
  const selectedAssets = selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean);
  const missingAssets = selectedAssetIds.filter((id) => !assetsById.has(id)).length;
  const disabledItems = selectedAssets.filter((asset) => asset.status?.rejected || asset.status?.trashed).length + missingAssets;
  const names = selectedAssets.map((asset) => imageAssetName(asset).toLowerCase());
  const duplicateFilenames = names.filter((name, index) => names.indexOf(name) !== index).length;
  const captionsByAssetId = new Map((activeDataset?.items ?? []).map((item) => [item.assetId, captionText(item)]));
  const missingCaptions = selectedAssetIds.filter((id) => !captionsByAssetId.get(id)).length;
  const valid = selectedAssetIds.length > 0 && disabledItems === 0;

  return {
    disabledItems,
    duplicateFilenames,
    itemCount: selectedAssetIds.length,
    missingCaptions,
    valid,
  };
}

function datasetPayload({ activeDataset, assetsById, name, selectedAssetIds }) {
  const itemsByAssetId = new Map((activeDataset?.items ?? []).map((item) => [item.assetId, item]));
  return {
    name: name.trim(),
    modality: "image",
    items: selectedAssetIds
      .map((assetId) => {
        const asset = assetsById.get(assetId);
        if (!asset) {
          return null;
        }
        const previous = itemsByAssetId.get(assetId);
        return {
          assetId,
          displayName: asset.displayName ?? imageAssetName(asset),
          caption: previous?.caption
            ? {
                text: previous.caption.text ?? "",
                source: previous.caption.source ?? "manual",
                triggerWords: previous.caption.triggerWords ?? [],
              }
            : undefined,
        };
      })
      .filter(Boolean),
  };
}

function asText(value) {
  return value === null || value === undefined ? "" : String(value);
}

function numericDraft(value) {
  return value === null || value === undefined ? "" : String(value);
}

function numberFromDraft(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const number = Number(trimmed);
  return Number.isFinite(number) ? number : null;
}

function compactObject(object) {
  return Object.fromEntries(
    Object.entries(object).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  );
}

function rangeOptions(limits, key) {
  return Array.isArray(limits?.[key]) ? limits[key] : [];
}

function configDraftFromTarget(target, dataset, gpuOptions) {
  const defaults = target?.defaults ?? {};
  const advanced = defaults.advanced ?? {};
  const firstGpu = gpuOptions[0] ?? "";
  const requestedGpu = asText(advanced.requestedGpu || firstGpu);
  const outputLabel = outputKindLabel(target);
  return {
    outputName: dataset?.name ? `${dataset.name} ${outputLabel}` : "",
    triggerWord: asText(defaults.triggerWord),
    outputScope: asText(advanced.outputScope),
    qualityPreset: asText(advanced.qualityPreset),
    requestedGpu: gpuOptions.includes(requestedGpu) ? requestedGpu : firstGpu,
    rank: numericDraft(defaults.rank),
    alpha: numericDraft(defaults.alpha),
    optimizer: asText(defaults.optimizer),
    learningRate: numericDraft(defaults.learningRate),
    scheduler: asText(advanced.scheduler),
    steps: numericDraft(defaults.steps),
    epochs: numericDraft(advanced.epochs),
    repeats: numericDraft(advanced.repeats),
    resolution: numericDraft(defaults.resolution),
    bucketStrategy: asText(advanced.bucketStrategy),
    precision: asText(advanced.mixedPrecision),
    saveEvery: numericDraft(defaults.saveEvery),
    sampleEvery: numericDraft(advanced.sampleEvery),
    batchSize: numericDraft(defaults.batchSize),
    gradientAccumulation: numericDraft(defaults.gradientAccumulation),
    seed: numericDraft(defaults.seed),
  };
}

function configValidation({ activeDataset, configDraft, selectedTarget }) {
  const warnings = [];
  if (!selectedTarget) {
    warnings.push("Select a training target");
  }
  if (!activeDataset?.id) {
    warnings.push("Select a saved dataset");
  }
  if (!configDraft.outputName?.trim()) {
    warnings.push(`Name the ${outputKindLabel(selectedTarget)} output`);
  }
  if (!configDraft.triggerWord?.trim()) {
    warnings.push("Add a trigger phrase");
  }
  for (const [field, label] of [
    ["rank", "Rank"],
    ["alpha", "Alpha"],
    ["learningRate", "Learning rate"],
    ["steps", "Steps"],
    ["resolution", "Resolution"],
    ["batchSize", "Batch size"],
    ["gradientAccumulation", "Gradient accumulation"],
    ["saveEvery", "Checkpoint cadence"],
  ]) {
    const value = numberFromDraft(configDraft[field]);
    if (!value || value <= 0) {
      warnings.push(`${label} must be greater than zero`);
    }
  }
  return warnings;
}

function outputKindLabel(target) {
  const kind = String(target?.outputKind ?? "output").toLowerCase();
  if (kind === "lora") {
    return "LoRA";
  }
  return kind.replaceAll("_", " ");
}

function trainingConfigSnapshot({ activeDataset, configDraft, selectedTarget }) {
  const defaults = selectedTarget?.defaults ?? {};
  const advanced = compactObject({
    ...(defaults.advanced ?? {}),
    scheduler: asText(configDraft.scheduler).trim(),
    epochs: numberFromDraft(configDraft.epochs),
    repeats: numberFromDraft(configDraft.repeats),
    bucketStrategy: asText(configDraft.bucketStrategy).trim(),
    mixedPrecision: asText(configDraft.precision).trim(),
    sampleEvery: numberFromDraft(configDraft.sampleEvery),
    qualityPreset: configDraft.qualityPreset,
    outputScope: configDraft.outputScope,
    requestedGpu: configDraft.requestedGpu,
  });
  return {
    targetId: selectedTarget.id,
    datasetId: activeDataset.id,
    datasetVersion: activeDataset.version,
    outputName: configDraft.outputName.trim(),
    dryRun: true,
    outputScope: configDraft.outputScope,
    qualityPreset: configDraft.qualityPreset,
    requestedGpu: configDraft.requestedGpu,
    config: {
      rank: numberFromDraft(configDraft.rank),
      alpha: numberFromDraft(configDraft.alpha),
      learningRate: numberFromDraft(configDraft.learningRate),
      steps: numberFromDraft(configDraft.steps),
      batchSize: numberFromDraft(configDraft.batchSize),
      gradientAccumulation: numberFromDraft(configDraft.gradientAccumulation),
      resolution: numberFromDraft(configDraft.resolution),
      saveEvery: numberFromDraft(configDraft.saveEvery),
      seed: numberFromDraft(configDraft.seed),
      optimizer: asText(configDraft.optimizer).trim(),
      triggerWord: asText(configDraft.triggerWord).trim(),
      advanced,
    },
  };
}

function DatasetHealth({ health }) {
  return (
    <div className="training-health-grid" aria-label="Dataset health">
      <div>
        <strong>{health.itemCount}</strong>
        <span>Items</span>
      </div>
      <div className={health.missingCaptions ? "needs-attention" : ""}>
        <strong>{health.missingCaptions}</strong>
        <span>Missing captions</span>
      </div>
      <div className={health.duplicateFilenames ? "needs-attention" : ""}>
        <strong>{health.duplicateFilenames}</strong>
        <span>Duplicate filenames</span>
      </div>
      <div className={health.disabledItems ? "needs-attention" : ""}>
        <strong>{health.disabledItems}</strong>
        <span>Disabled items</span>
      </div>
    </div>
  );
}

export function TrainingStudio({
  activeProject,
  authenticated = true,
  assets = [],
  batchRenameDataset = async () => null,
  createDataset = async () => null,
  createTrainingJob = async () => null,
  datasets = [],
  datasetsError = "",
  gpuOptions = defaultGpuOptions,
  importAsset = async () => null,
  loadDataset = async () => null,
  loadingDatasets = false,
  onPreview = () => {},
  onRefreshDatasets = () => {},
  prepareTrainingConfig = async (snapshot) => snapshot,
  trainingTargets = [],
  trainingTargetsError = "",
  updateDataset = async () => null,
  writeCaptionSidecars = async () => null,
}) {
  const [activeTab, setActiveTab] = useState("dataset");
  const [activeDataset, setActiveDataset] = useState(null);
  const [datasetError, setDatasetError] = useState("");
  const [datasetMessage, setDatasetMessage] = useState("");
  const [draftName, setDraftName] = useState("");
  const [busyDatasetId, setBusyDatasetId] = useState("");
  const [importingAssets, setImportingAssets] = useState(false);
  const [savingDataset, setSavingDataset] = useState(false);
  const [selectedAssetIds, setSelectedAssetIds] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [renamePrefix, setRenamePrefix] = useState("");
  const [renameCaptionDraftItems, setRenameCaptionDraftItems] = useState([]);
  const [savingRenameCaption, setSavingRenameCaption] = useState(false);
  const [selectedTargetId, setSelectedTargetId] = useState("");
  const [configDraft, setConfigDraft] = useState({});
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [configSnapshot, setConfigSnapshot] = useState(null);
  const [configMessage, setConfigMessage] = useState("");
  const [configError, setConfigError] = useState("");
  const [preparingConfig, setPreparingConfig] = useState(false);
  const [submittingJob, setSubmittingJob] = useState(false);
  const configBasisRef = useRef("");
  const tabRefs = useRef({});

  const activeIndex = tabs.findIndex((tab) => tab.id === activeTab);
  const active = tabs[activeIndex] ?? tabs[0];
  const datasetSummary = useMemo(() => summarizeDatasets(datasets), [datasets]);
  const imageAssets = useMemo(() => assets.filter((asset) => assetCanRenderAsImage(asset)), [assets]);
  const assetsById = useMemo(() => new Map(imageAssets.map((asset) => [asset.id, asset])), [imageAssets]);
  const unavailableAssetIds = useMemo(
    () => selectedAssetIds.filter((assetId) => !assetsById.has(assetId)),
    [assetsById, selectedAssetIds],
  );
  const health = useMemo(
    () => datasetHealth({ activeDataset, imageAssets, selectedAssetIds }),
    [activeDataset, imageAssets, selectedAssetIds],
  );
  const originalAssetIds = useMemo(() => normalizeDatasetAssetIds(activeDataset), [activeDataset]);
  const dirty =
    Boolean(activeDataset) &&
    (draftName.trim() !== activeDataset.name ||
      selectedAssetIds.length !== originalAssetIds.length ||
      selectedAssetIds.some((id, index) => id !== originalAssetIds[index]));
  const canSave =
    draftName.trim().length > 0 &&
    selectedAssetIds.length > 0 &&
    health.disabledItems === 0 &&
    !savingDataset &&
    (!activeDataset || dirty);
  const renameCaptionHasDraft = renameCaptionDraftItems.length > 0;
  const renameCaptionHasInvalidDraft = renameCaptionDraftItems.some(
    (item) => !item.itemId.trim() || !item.fileStem.trim() || !item.displayName.trim(),
  );
  const canSaveRenameCaption =
    Boolean(activeDataset?.id) && renameCaptionHasDraft && !renameCaptionHasInvalidDraft && !savingRenameCaption;
  const firstTarget = trainingTargets[0] ?? null;
  const selectedTarget = useMemo(
    () => trainingTargets.find((target) => target.id === selectedTargetId) ?? firstTarget,
    [firstTarget, selectedTargetId, trainingTargets],
  );
  const qualityPresets = rangeOptions(selectedTarget?.limits, "qualityPresets");
  const outputScopes = rangeOptions(selectedTarget?.limits, "outputScopes");
  const resolutionOptions = rangeOptions(selectedTarget?.limits, "resolutions");
  const gpuOptionsKey = gpuOptions.join("\u0000");
  const configWarnings = configValidation({ activeDataset, configDraft, selectedTarget });
  const canPrepareConfig = configWarnings.length === 0 && !preparingConfig;

  useEffect(() => {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setSelectedDatasetId("");
    setRenamePrefix("");
    setRenameCaptionDraftItems([]);
    setConfigDraft({});
    setConfigSnapshot(null);
    setConfigMessage("");
    setConfigError("");
    configBasisRef.current = "";
  }, [activeProject?.id]);

  useEffect(() => {
    setRenameCaptionDraftItems(renameCaptionDrafts(activeDataset));
    setRenamePrefix(safeSlug(activeDataset?.name, "item"));
  }, [activeDataset]);

  useEffect(() => {
    if (!selectedTargetId && firstTarget?.id) {
      setSelectedTargetId(firstTarget.id);
    }
  }, [firstTarget?.id, selectedTargetId]);

  useEffect(() => {
    if (!selectedTarget) {
      if (configBasisRef.current) {
        configBasisRef.current = "";
        setConfigDraft({});
      }
      return;
    }
    const basis = `${selectedTarget.id}\u0000${activeDataset?.id ?? ""}`;
    if (configBasisRef.current === basis) {
      return;
    }
    configBasisRef.current = basis;
    setConfigDraft(configDraftFromTarget(selectedTarget, activeDataset, gpuOptions));
    setConfigSnapshot(null);
    setConfigMessage("");
    setConfigError("");
  }, [activeDataset?.id, selectedTarget?.id]);

  useEffect(() => {
    setConfigDraft((current) => {
      if (!current.requestedGpu || gpuOptions.includes(current.requestedGpu)) {
        return current;
      }
      return { ...current, requestedGpu: gpuOptions[0] ?? "" };
    });
  }, [gpuOptionsKey]);

  function focusTab(index) {
    const next = tabs[(index + tabs.length) % tabs.length];
    setActiveTab(next.id);
    window.requestAnimationFrame(() => tabRefs.current[next.id]?.focus());
  }

  function onTabKeyDown(event) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTab(activeIndex + 1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTab(activeIndex - 1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusTab(0);
    }
    if (event.key === "End") {
      event.preventDefault();
      focusTab(tabs.length - 1);
    }
  }

  async function openDataset(datasetId) {
    if (!datasetId) {
      setActiveDataset(null);
      setDraftName("");
      setSelectedAssetIds([]);
      setSelectedDatasetId("");
      return;
    }
    setBusyDatasetId(datasetId);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const dataset = await loadDataset(datasetId);
      setActiveDataset(dataset);
      setDraftName(dataset?.name ?? "");
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset));
      setSelectedDatasetId(dataset?.id ?? datasetId);
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setBusyDatasetId("");
    }
  }

  function toggleAsset(assetId) {
    setDatasetMessage("");
    setSelectedAssetIds((current) =>
      current.includes(assetId) ? current.filter((id) => id !== assetId) : [...current, assetId],
    );
  }

  function removeUnavailableAsset(assetId) {
    setDatasetMessage("");
    setSelectedAssetIds((current) => current.filter((id) => id !== assetId));
  }

  function startNewDataset() {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setSelectedDatasetId("");
  }

  function updateRenameCaptionDraft(originalItemId, patch) {
    setDatasetMessage("");
    setRenameCaptionDraftItems((current) =>
      current.map((item) => (item.originalItemId === originalItemId ? { ...item, ...patch } : item)),
    );
  }

  function updateConfigDraft(field, value) {
    setConfigMessage("");
    setConfigError("");
    setConfigSnapshot(null);
    setConfigDraft((current) => ({ ...current, [field]: value }));
  }

  function resetConfigDefaults() {
    if (!selectedTarget) {
      return;
    }
    setConfigDraft(configDraftFromTarget(selectedTarget, activeDataset, gpuOptions));
    setConfigSnapshot(null);
    setConfigMessage("Defaults restored");
    setConfigError("");
  }

  function applyOrderedNames() {
    setDatasetMessage("");
    setRenameCaptionDraftItems((current) =>
      current.map((item, index) => {
        const nextName = orderedName(index, renamePrefix || activeDataset?.name);
        const extension = String(item.path).split(".").pop() || "png";
        return {
          ...item,
          itemId: nextName,
          fileStem: nextName,
          displayName: `${nextName}.${extension}`,
        };
      }),
    );
  }

  async function handleImport(event) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) {
      return;
    }
    setImportingAssets(true);
    setDatasetError("");
    try {
      const imported = [];
      for (const file of files) {
        const asset = await importAsset(file);
        if (asset?.id) {
          imported.push(asset.id);
        }
      }
      if (imported.length) {
        setSelectedAssetIds((current) => Array.from(new Set([...current, ...imported])));
      } else {
        setDatasetError("No images were imported.");
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setImportingAssets(false);
      event.target.value = "";
    }
  }

  async function saveDataset() {
    if (!canSave) {
      return;
    }
    setSavingDataset(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const payload = datasetPayload({ activeDataset, assetsById, name: draftName, selectedAssetIds });
      const dataset = activeDataset
        ? await updateDataset(activeDataset.id, payload)
        : await createDataset(payload);
      setActiveDataset(dataset);
      setDraftName(dataset?.name ?? draftName.trim());
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset));
      setSelectedDatasetId(dataset?.id ?? "");
      setDatasetMessage(activeDataset ? "Dataset changes saved" : "Dataset created");
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setSavingDataset(false);
    }
  }

  async function saveRenameCaption() {
    if (!canSaveRenameCaption) {
      return;
    }
    setSavingRenameCaption(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      let dataset = activeDataset;
      const renameItems = renameCaptionDraftItems.map((item) => ({
        itemId: item.originalItemId,
        newItemId: item.itemId.trim(),
        fileStem: item.fileStem.trim(),
        displayName: item.displayName.trim(),
      }));
      if (renameFieldsDirty(renameCaptionDraftItems, activeDataset)) {
        dataset = await batchRenameDataset(activeDataset.id, { items: renameItems });
      }
      const result = await writeCaptionSidecars(activeDataset.id, {
        items: renameCaptionDraftItems.map((item) => ({
          itemId: item.itemId.trim(),
          caption: {
            text: item.captionText,
            source: item.captionSource,
            triggerWords: parseTriggerWords(item.triggerWords),
          },
        })),
      });
      const nextDataset = result?.dataset ?? dataset;
      setActiveDataset(nextDataset);
      setDraftName(nextDataset?.name ?? draftName);
      setSelectedAssetIds(normalizeDatasetAssetIds(nextDataset));
      setSelectedDatasetId(nextDataset?.id ?? activeDataset.id);
      setDatasetMessage(`Caption sidecars written${result?.sidecars?.length ? ` (${result.sidecars.length})` : ""}`);
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setSavingRenameCaption(false);
    }
  }

  async function prepareConfig() {
    if (!canPrepareConfig) {
      return;
    }
    setPreparingConfig(true);
    setConfigError("");
    setConfigMessage("");
    try {
      const snapshot = trainingConfigSnapshot({ activeDataset, configDraft, selectedTarget });
      const result = await prepareTrainingConfig(snapshot);
      setConfigSnapshot(result ?? snapshot);
      setConfigMessage("Config snapshot ready");
    } catch (err) {
      setConfigError(err.message);
    } finally {
      setPreparingConfig(false);
    }
  }

  async function submitTrainingJob() {
    if (!canPrepareConfig || submittingJob) {
      return;
    }
    setSubmittingJob(true);
    setConfigError("");
    setConfigMessage("");
    try {
      const snapshot = trainingConfigSnapshot({ activeDataset, configDraft, selectedTarget });
      const job = await createTrainingJob({
        targetId: snapshot.targetId,
        datasetId: snapshot.datasetId,
        datasetVersion: snapshot.datasetVersion,
        outputName: snapshot.outputName,
        dryRun: true,
        config: snapshot.config,
      });
      setConfigSnapshot(snapshot);
      setConfigMessage(`Queued dry-run job ${job?.id ?? ""}`.trim() + ". Track it in the Queue.");
    } catch (err) {
      setConfigError(err.message);
    } finally {
      setSubmittingJob(false);
    }
  }

  return (
    <section className="main-surface training-studio">
      <div className="training-studio-shell">
        <div className="training-summary-band">
          <div className="section-heading">
            <p className="eyebrow">Training Studio</p>
            <h2>Native LoRA training workflow</h2>
            <p className="view-copy">
              Build datasets, normalize captions, and prepare a Rust-owned training plan before any ML runtime work begins.
            </p>
          </div>
          <div className="training-metrics" aria-label="Training workspace summary">
            <div>
              <strong>{activeProject?.name ?? "No workspace"}</strong>
              <span>Project</span>
            </div>
            <div>
              <strong>{datasets.length}</strong>
              <span>Datasets</span>
            </div>
            <div>
              <strong>{datasetSummary.items}</strong>
              <span>Items</span>
            </div>
          </div>
        </div>

        {!authenticated ? (
          <div className="training-empty-state" role="status">
            <Icon.Train size={24} />
            <div>
              <strong>Pairing required</strong>
              <span>Unlock SceneWorks to load project training datasets.</span>
            </div>
          </div>
        ) : !activeProject ? (
          <div className="training-empty-state" role="status">
            <Icon.Folder size={24} />
            <div>
              <strong>No workspace open</strong>
              <span>Create or select a workspace before building a training dataset.</span>
            </div>
          </div>
        ) : (
          <>
            <div className="training-tabs" role="tablist" aria-label="Training workflow">
              {tabs.map((tab) => (
                <button
                  aria-controls={activeTab === tab.id ? `training-panel-${tab.id}` : undefined}
                  aria-selected={activeTab === tab.id}
                  className={activeTab === tab.id ? "active" : ""}
                  id={`training-tab-${tab.id}`}
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  onKeyDown={onTabKeyDown}
                  ref={(node) => {
                    tabRefs.current[tab.id] = node;
                  }}
                  role="tab"
                  tabIndex={activeTab === tab.id ? 0 : -1}
                  type="button"
                >
                  <span>{tab.label}</span>
                  <small>{tab.status}</small>
                </button>
              ))}
            </div>

            <section
              aria-labelledby={`training-tab-${active.id}`}
              className="training-panel"
              id={`training-panel-${active.id}`}
              role="tabpanel"
            >
              {activeTab === "dataset" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Dataset</p>
                      <h3>{active.title}</h3>
                    </div>
                    <div className="training-head-actions">
                      <button className="secondary-action" disabled={loadingDatasets} onClick={onRefreshDatasets} type="button">
                        <Icon.Search size={14} />
                        {loadingDatasets ? "Refreshing" : "Refresh"}
                      </button>
                      <button className="primary-action" onClick={startNewDataset} type="button">
                        <Icon.Plus size={14} />
                        New
                      </button>
                    </div>
                  </div>
                  {datasetsError ? <p className="inline-warning">{datasetsError}</p> : null}
                  {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
                  {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}
                  <div className="training-dataset-workspace">
                    <aside className="training-dataset-list-panel">
                      {loadingDatasets ? <div className="empty-panel compact-panel">Loading training datasets</div> : null}
                      {!loadingDatasets && datasets.length === 0 ? <div className="empty-panel compact-panel">No training datasets yet</div> : null}
                      {datasets.map((dataset) => {
                        const itemCount = datasetItemCount(dataset);
                        return (
                          <button
                            aria-pressed={selectedDatasetId === dataset.id}
                            className={selectedDatasetId === dataset.id ? "training-dataset-row active" : "training-dataset-row"}
                            disabled={busyDatasetId === dataset.id}
                            key={dataset.id}
                            onClick={() => openDataset(dataset.id)}
                            type="button"
                          >
                            <div>
                              <strong>{dataset.name ?? dataset.id}</strong>
                              <span>{formatDatasetModality(dataset)} dataset</span>
                            </div>
                            <span>{busyDatasetId === dataset.id ? "Opening" : `${itemCount} item${itemCount === 1 ? "" : "s"}`}</span>
                          </button>
                        );
                      })}
                    </aside>

                    <div className="training-dataset-editor">
                      <div className="training-dataset-form">
                        <label>
                          Dataset name
                          <input
                            onChange={(event) => setDraftName(event.target.value)}
                            placeholder="Character portrait set"
                            value={draftName}
                          />
                        </label>
                        <label>
                          Modality
                          <select disabled value="image">
                            <option value="image">Image</option>
                          </select>
                        </label>
                        <label className="file-upload-button training-import-button">
                          <input accept="image/*" disabled={importingAssets} multiple onChange={handleImport} type="file" />
                          {importingAssets ? "Importing" : "Import images"}
                        </label>
                      </div>
                      <DatasetHealth health={health} />
                      <div className="training-validity">
                        <span className={health.valid ? "training-valid-dot valid" : "training-valid-dot"} />
                        <span>{health.valid ? "Dataset is ready for downstream steps" : "Add image assets and remove disabled items"}</span>
                      </div>
                      {unavailableAssetIds.length ? (
                        <div className="training-unavailable-list" aria-label="Unavailable dataset items">
                          {unavailableAssetIds.map((assetId) => (
                            <div className="training-unavailable-item" key={assetId}>
                              <div>
                                <strong>{assetId}</strong>
                                <span>Asset is no longer available</span>
                              </div>
                              <button className="secondary-action" onClick={() => removeUnavailableAsset(assetId)} type="button">
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : null}
                      <div className="training-asset-picker" aria-label="Training dataset image assets">
                        {imageAssets.length ? (
                          imageAssets.map((asset) => {
                            const selected = selectedAssetIds.includes(asset.id);
                            const disabled = asset.status?.rejected || asset.status?.trashed;
                            return (
                              <article
                                className={[
                                  "training-asset-card",
                                  selected ? "selected" : "",
                                  disabled ? "disabled" : "",
                                ]
                                  .filter(Boolean)
                                  .join(" ")}
                                key={asset.id}
                              >
                                <button onClick={() => onPreview(asset)} type="button">
                                  <AssetThumbnail asset={asset} />
                                </button>
                                <label>
                                  <input checked={selected} onChange={() => toggleAsset(asset.id)} type="checkbox" />
                                  <span>{asset.displayName ?? imageAssetName(asset)}</span>
                                </label>
                                {disabled ? (
                                  <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
                                ) : null}
                              </article>
                            );
                          })
                        ) : (
                          <div className="empty-panel compact-panel">Import or create project images before building a dataset</div>
                        )}
                      </div>
                      <div className="training-dataset-actions">
                        <button className="primary-action" disabled={!canSave} onClick={saveDataset} type="button">
                          {savingDataset ? "Saving" : activeDataset ? "Save dataset" : "Create dataset"}
                        </button>
                        <span>{dirty ? "Unsaved changes" : activeDataset ? `Version ${activeDataset.version}` : "Draft"}</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : null}

              {activeTab === "rename-caption" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Rename & Caption</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">{activeDataset ? `Version ${activeDataset.version}` : "Select dataset"}</span>
                  </div>
                  {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
                  {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}
                  {!activeDataset ? (
                    <div className="empty-panel compact-panel">Open a saved dataset to edit captions.</div>
                  ) : (
                    <div className="training-caption-editor">
                      <div className="training-caption-toolbar">
                        <label>
                          Rename prefix
                          <input onChange={(event) => setRenamePrefix(event.target.value)} value={renamePrefix} />
                        </label>
                        <button className="secondary-action" onClick={applyOrderedNames} type="button">
                          <Icon.Sliders size={14} />
                          Apply ordered names
                        </button>
                        <button
                          className="primary-action"
                          disabled={!canSaveRenameCaption}
                          onClick={saveRenameCaption}
                          type="button"
                        >
                          {savingRenameCaption ? "Writing" : "Write sidecars"}
                        </button>
                      </div>
                      <div className="training-caption-list" aria-label="Rename and caption dataset items">
                        {renameCaptionDraftItems.map((item, index) => {
                          const asset = assetsById.get(item.assetId);
                          return (
                            <article className="training-caption-row" key={item.originalItemId}>
                              <div className="training-caption-preview">
                                <button disabled={!asset} onClick={() => asset && onPreview(asset)} type="button">
                                  {asset ? <AssetThumbnail asset={asset} /> : <Icon.Image size={22} />}
                                </button>
                                <span>{String(index + 1).padStart(2, "0")}</span>
                              </div>
                              <div className="training-caption-fields">
                                <label>
                                  Item ID
                                  <input
                                    onChange={(event) => updateRenameCaptionDraft(item.originalItemId, { itemId: event.target.value })}
                                    value={item.itemId}
                                  />
                                </label>
                                <label>
                                  File stem
                                  <input
                                    onChange={(event) => updateRenameCaptionDraft(item.originalItemId, { fileStem: event.target.value })}
                                    value={item.fileStem}
                                  />
                                </label>
                                <label>
                                  Display name
                                  <input
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { displayName: event.target.value })
                                    }
                                    value={item.displayName}
                                  />
                                </label>
                                <label className="training-caption-text">
                                  Caption
                                  <textarea
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { captionText: event.target.value })
                                    }
                                    rows={3}
                                    value={item.captionText}
                                  />
                                </label>
                                <label>
                                  Trigger words
                                  <input
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { triggerWords: event.target.value })
                                    }
                                    value={item.triggerWords}
                                  />
                                </label>
                                <label>
                                  Source
                                  <select
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { captionSource: event.target.value })
                                    }
                                    value={item.captionSource}
                                  >
                                    <option value="manual">Manual</option>
                                    <option value="imported">Imported</option>
                                    <option value="auto">Auto</option>
                                  </select>
                                </label>
                              </div>
                            </article>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </>
              ) : null}

              {activeTab === "configure" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Configure Job</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">{canPrepareConfig ? "Ready" : "Needs input"}</span>
                  </div>
                  {trainingTargetsError ? <p className="inline-warning">{trainingTargetsError}</p> : null}
                  {configError ? <p className="inline-warning">{configError}</p> : null}
                  {configMessage ? <p className="inline-success">{configMessage}</p> : null}
                  {!selectedTarget ? (
                    <div className="empty-panel compact-panel">Training target registry unavailable</div>
                  ) : (
                    <div className="training-config-form" aria-label="Training job configuration">
                      <div className="training-config-grid">
                        <label>
                          Target
                          <select onChange={(event) => setSelectedTargetId(event.target.value)} value={selectedTarget.id}>
                            {trainingTargets.map((target) => (
                              <option key={target.id} value={target.id}>
                                {target.ui?.label ?? target.name}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Base model
                          <input disabled readOnly value={selectedTarget.baseModel ?? ""} />
                        </label>
                        <label>
                          Dataset
                          <select onChange={(event) => openDataset(event.target.value)} value={activeDataset?.id ?? ""}>
                            <option value="">Select a saved dataset</option>
                            {datasets.map((dataset) => (
                              <option key={dataset.id} value={dataset.id}>
                                {dataset.name ?? dataset.id}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          LoRA name
                          <input onChange={(event) => updateConfigDraft("outputName", event.target.value)} value={configDraft.outputName ?? ""} />
                        </label>
                        <label>
                          Trigger phrase
                          <input onChange={(event) => updateConfigDraft("triggerWord", event.target.value)} value={configDraft.triggerWord ?? ""} />
                        </label>
                        <label>
                          Output scope
                          <select onChange={(event) => updateConfigDraft("outputScope", event.target.value)} value={configDraft.outputScope ?? ""}>
                            {outputScopes.length ? null : <option value={configDraft.outputScope ?? ""}>{configDraft.outputScope || "Default"}</option>}
                            {outputScopes.map((scope) => (
                              <option key={scope} value={scope}>
                                {scope}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Quality preset
                          <select
                            onChange={(event) => updateConfigDraft("qualityPreset", event.target.value)}
                            value={configDraft.qualityPreset ?? ""}
                          >
                            {qualityPresets.length ? null : (
                              <option value={configDraft.qualityPreset ?? ""}>{configDraft.qualityPreset || "Default"}</option>
                            )}
                            {qualityPresets.map((preset) => (
                              <option key={preset} value={preset}>
                                {preset}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Requested GPU
                          <select onChange={(event) => updateConfigDraft("requestedGpu", event.target.value)} value={configDraft.requestedGpu ?? ""}>
                            {gpuOptions.map((gpu) => (
                              <option key={gpu} value={gpu}>
                                {gpu === "auto" ? "Auto" : `GPU ${gpu}`}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>

                      <details
                        className="training-advanced-panel"
                        onToggle={(event) => setShowAdvancedConfig(event.currentTarget.open)}
                        open={showAdvancedConfig}
                      >
                        <summary>
                          <Icon.Sliders size={14} />
                          Advanced
                        </summary>
                        <div className="training-advanced-grid">
                          <label>
                            Rank
                            <input onChange={(event) => updateConfigDraft("rank", event.target.value)} type="number" value={configDraft.rank ?? ""} />
                          </label>
                          <label>
                            Alpha
                            <input onChange={(event) => updateConfigDraft("alpha", event.target.value)} type="number" value={configDraft.alpha ?? ""} />
                          </label>
                          <label>
                            Optimizer
                            <input onChange={(event) => updateConfigDraft("optimizer", event.target.value)} value={configDraft.optimizer ?? ""} />
                          </label>
                          <label>
                            Learning rate
                            <input
                              onChange={(event) => updateConfigDraft("learningRate", event.target.value)}
                              step="0.00001"
                              type="number"
                              value={configDraft.learningRate ?? ""}
                            />
                          </label>
                          <label>
                            Scheduler
                            <input onChange={(event) => updateConfigDraft("scheduler", event.target.value)} value={configDraft.scheduler ?? ""} />
                          </label>
                          <label>
                            Steps
                            <input onChange={(event) => updateConfigDraft("steps", event.target.value)} type="number" value={configDraft.steps ?? ""} />
                          </label>
                          <label>
                            Epochs
                            <input onChange={(event) => updateConfigDraft("epochs", event.target.value)} type="number" value={configDraft.epochs ?? ""} />
                          </label>
                          <label>
                            Repeats
                            <input onChange={(event) => updateConfigDraft("repeats", event.target.value)} type="number" value={configDraft.repeats ?? ""} />
                          </label>
                          <label>
                            Resolution
                            <select onChange={(event) => updateConfigDraft("resolution", event.target.value)} value={configDraft.resolution ?? ""}>
                              {resolutionOptions.length ? null : <option value={configDraft.resolution ?? ""}>{configDraft.resolution ?? ""}</option>}
                              {resolutionOptions.map((resolution) => (
                                <option key={resolution} value={resolution}>
                                  {resolution}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            Buckets
                            <input
                              onChange={(event) => updateConfigDraft("bucketStrategy", event.target.value)}
                              value={configDraft.bucketStrategy ?? ""}
                            />
                          </label>
                          <label>
                            Precision
                            <input onChange={(event) => updateConfigDraft("precision", event.target.value)} value={configDraft.precision ?? ""} />
                          </label>
                          <label>
                            Checkpoint cadence
                            <input
                              onChange={(event) => updateConfigDraft("saveEvery", event.target.value)}
                              type="number"
                              value={configDraft.saveEvery ?? ""}
                            />
                          </label>
                          <label>
                            Sample cadence
                            <input
                              onChange={(event) => updateConfigDraft("sampleEvery", event.target.value)}
                              type="number"
                              value={configDraft.sampleEvery ?? ""}
                            />
                          </label>
                        </div>
                      </details>

                      {configWarnings.length ? (
                        <div className="training-config-warnings" aria-label="Configuration warnings">
                          {configWarnings.map((warning) => (
                            <span key={warning}>{warning}</span>
                          ))}
                        </div>
                      ) : null}

                      <div className="training-config-actions">
                        <button className="secondary-action" onClick={resetConfigDefaults} type="button">
                          Reset defaults
                        </button>
                        <button className="secondary-action" disabled={!canPrepareConfig} onClick={prepareConfig} type="button">
                          {preparingConfig ? "Preparing" : "Prepare config"}
                        </button>
                        <button
                          className="primary-action"
                          disabled={!canPrepareConfig || submittingJob}
                          onClick={submitTrainingJob}
                          type="button"
                        >
                          {submittingJob ? "Queuing" : "Queue dry-run job"}
                        </button>
                      </div>
                      {configSnapshot ? <pre className="training-config-snapshot">{JSON.stringify(configSnapshot, null, 2)}</pre> : null}
                    </div>
                  )}
                  <p className="view-copy">
                    Queuing a dry run validates the Rust-resolved training plan and dataset on a GPU worker without training; real
                    training execution arrives in a later story.
                  </p>
                </>
              ) : null}
            </section>
          </>
        )}
      </div>
    </section>
  );
}
