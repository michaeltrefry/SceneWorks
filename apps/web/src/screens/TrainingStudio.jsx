import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";

const tabs = [
  { id: "dataset", label: "Dataset", title: "Dataset intake", status: "Rust dataset store" },
  { id: "rename-caption", label: "Rename & Caption", title: "Rename and caption pass", status: "Needs valid dataset" },
  { id: "configure", label: "Configure Job", title: "Configure training job", status: "Queue disabled" },
];

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
  const valid = selectedAssetIds.length > 0 && disabledItems === 0 && duplicateFilenames === 0;

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
  createDataset = async () => null,
  datasets = [],
  datasetsError = "",
  importAsset = async () => null,
  loadDataset = async () => null,
  loadingDatasets = false,
  onPreview = () => {},
  onRefreshDatasets = () => {},
  updateDataset = async () => null,
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
  const tabRefs = useRef({});

  const activeIndex = tabs.findIndex((tab) => tab.id === activeTab);
  const active = tabs[activeIndex] ?? tabs[0];
  const datasetSummary = useMemo(() => summarizeDatasets(datasets), [datasets]);
  const imageAssets = useMemo(
    () => assets.filter((asset) => assetCanRenderAsImage(asset) && !asset.status?.trashed),
    [assets],
  );
  const assetsById = useMemo(() => new Map(imageAssets.map((asset) => [asset.id, asset])), [imageAssets]);
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
  const canSave = draftName.trim().length > 0 && selectedAssetIds.length > 0 && health.disabledItems === 0 && !savingDataset;

  useEffect(() => {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setSelectedDatasetId("");
  }, [activeProject?.id]);

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

  function startNewDataset() {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setSelectedDatasetId("");
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
                        <span>{health.valid ? "Dataset is ready for downstream steps" : "Add image assets and resolve flagged items"}</span>
                      </div>
                      <div className="training-asset-picker" aria-label="Training dataset image assets">
                        {imageAssets.length ? (
                          imageAssets.map((asset) => {
                            const selected = selectedAssetIds.includes(asset.id);
                            return (
                              <article className={selected ? "training-asset-card selected" : "training-asset-card"} key={asset.id}>
                                <button onClick={() => onPreview(asset)} type="button">
                                  <AssetThumbnail asset={asset} />
                                </button>
                                <label>
                                  <input checked={selected} onChange={() => toggleAsset(asset.id)} type="checkbox" />
                                  <span>{asset.displayName ?? imageAssetName(asset)}</span>
                                </label>
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
                    <span className="training-status-pill">{health.valid ? "Dataset ready" : "Blocked"}</span>
                  </div>
                  <div className="training-workflow-grid">
                    <div className="training-step-block">
                      <strong>Batch rename</strong>
                      <span>Stable filenames and ordered item ids will be prepared from the selected dataset.</span>
                    </div>
                    <div className="training-step-block">
                      <strong>Caption sidecars</strong>
                      <span>Caption metadata will stay attached to SceneWorks dataset items before sidecar export.</span>
                    </div>
                  </div>
                </>
              ) : null}

              {activeTab === "configure" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Configure Job</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">{health.valid ? "Dry run pending" : "Needs dataset"}</span>
                  </div>
                  <div className="training-config-preview" aria-label="Training job placeholder settings">
                    <label>
                      Target
                      <select defaultValue="z_image_turbo" disabled>
                        <option value="z_image_turbo">Z-Image Turbo LoRA</option>
                      </select>
                    </label>
                    <label>
                      Dataset
                      <select value={activeDataset?.id ?? ""} disabled>
                        <option value="">{health.valid ? "Save to select dataset" : "Valid dataset required"}</option>
                        {activeDataset ? <option value={activeDataset.id}>{activeDataset.name}</option> : null}
                      </select>
                    </label>
                    <label>
                      Preset
                      <select defaultValue="simple" disabled>
                        <option value="simple">Simple defaults</option>
                      </select>
                    </label>
                  </div>
                  <p className="inline-warning">Training submission is disabled until the dry-run plan story wires queue semantics.</p>
                </>
              ) : null}
            </section>
          </>
        )}
      </div>
    </section>
  );
}
