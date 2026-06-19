import React, { useState } from "react";
import { apiFetch } from "../api.js";
import { foldUpscaledAssetVariants } from "../assetVariants.js";
import { batchEligibleAssets, batchItemStatus, buildBatchJob, summarizeBatchProgress } from "../batchOps.js";
import { AssetDetail, AssetGrid, emptyTrash } from "../components/assetPanels.jsx";
import { assetUrl } from "../components/assetMedia.jsx";
import { BatchOperationsPanel } from "../components/BatchOperationsPanel.jsx";
import { terminalStatuses } from "../constants.js";
import { useAppContext } from "../context/AppContext.js";
import { detailCapableModels, editCapableModels, UPSCALE_ENGINES } from "../imageJobs.js";
import { DEFAULT_MAC_CAPABILITIES, macUpscaleEngineBlocked } from "../macGating.js";

export function LibraryScreen() {
  const {
    activeProject,
    assets,
    jobs = [],
    imageModels = [],
    createVqaJob,
    deleteAsset,
    purgeAsset,
    characters = [],
    addCharacterReference,
    importAsset,
    setPreviewAsset,
    sendAssetToImage,
    sendAssetToVideo,
    selectedAsset,
    setSelectedAssetId,
    setActiveView,
    updateAssetStatus,
    updateAssetTags,
    token = "",
    requestedGpu = "auto",
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  // Bind the fullscreen preview to the currently filtered library view so
  // navigation stays inside the same type/tag/trash scope the user is browsing.
  const onPreview = (asset) => setPreviewAsset(asset, visibleAssets);
  const onSendImage = (asset) => sendAssetToImage(asset);
  const onSendVideo = (asset) => sendAssetToVideo(asset);
  const onSendEditor = (asset) => {
    setSelectedAssetId(asset.id);
    setActiveView("Editor");
  };
  const vqaEnabled = Boolean(createVqaJob) && imageModels.some((model) => (model.capabilities ?? []).includes("vqa"));
  const assetVqaJobs = selectedAsset
    ? jobs.filter((job) => job.type === "image_vqa" && job.payload?.sourceAssetId === selectedAsset.id)
    : [];
  const vqaEntries = assetVqaJobs
    .filter((job) => job.status === "completed" && job.result?.answer)
    .map((job) => ({
      jobId: job.id,
      question: job.result?.question ?? job.payload?.question ?? "",
      answer: job.result.answer,
    }));
  const vqaPending = assetVqaJobs.some((job) => !terminalStatuses.has(job.status));
  const [typeFilter, setTypeFilter] = useState("all");
  const [tagFilter, setTagFilter] = useState("all");
  const [showRejected, setShowRejected] = useState(false);
  const [assetMode, setAssetMode] = useState("assets");
  const [isImporting, setIsImporting] = useState(false);
  // Batch operations (sc-6112): a multi-asset selection + a fan-out of one job per asset.
  const [selectedAssetIds, setSelectedAssetIds] = useState(() => new Set());
  const [batchOpen, setBatchOpen] = useState(false);
  // While/after a batch runs: { op, items: [{ asset, jobId }], submitting }.
  const [batch, setBatch] = useState(null);
  // Asset Library hygiene (sc-2024): show only studio-generated and uploaded
  // media. Character Studio test outputs (origin "character_studio") live under
  // the character, not here. The backend also exposes `?scope=library`; we filter
  // on the authoritative `origin` field client-side to reuse the shared asset
  // feed instead of loading a second scoped copy. Dataset images never reach the
  // Library — they are stored outside the indexed asset folders.
  const libraryAssets = assets.filter((asset) => asset.origin !== "character_studio");
  const visibleAssets = foldUpscaledAssetVariants(libraryAssets.filter((asset) => {
    if (typeFilter !== "all" && asset.type !== typeFilter) {
      return false;
    }
    if (!showRejected && asset.status?.rejected) {
      return false;
    }
    if (assetMode === "trashcan" && !asset.status?.trashed) {
      return false;
    }
    if (assetMode === "assets" && asset.status?.trashed) {
      return false;
    }
    if (tagFilter !== "all" && !(asset.tags ?? []).includes(tagFilter)) {
      return false;
    }
    return true;
  }));
  // All discarded assets within the current type/tag filters — the exact scope
  // the "Empty Trash" button purges (unfolded, so folded variants go too).
  const trashedInView = libraryAssets.filter((asset) => {
    if (typeFilter !== "all" && asset.type !== typeFilter) {
      return false;
    }
    if (tagFilter !== "all" && !(asset.tags ?? []).includes(tagFilter)) {
      return false;
    }
    return Boolean(asset.status?.trashed);
  });

  async function handleImport(event) {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    setIsImporting(true);
    await importAsset(file);
    setIsImporting(false);
    event.target.value = "";
  }

  async function moveAssetToCharacter(asset, characterId) {
    const updated = await addCharacterReference?.(characterId, {
      assetId: asset.id,
      approved: false,
      role: "asset",
      notes: "Added from Asset Library.",
    });
    if (!updated) {
      throw new Error("Could not add this asset to the character.");
    }
    return updated;
  }

  const imageCount = libraryAssets.filter((asset) => asset.type === "image").length;
  const videoCount = libraryAssets.filter((asset) => asset.type === "video").length;
  const uploadCount = libraryAssets.filter((asset) => asset.type === "upload").length;
  const availableTags = [...new Set(libraryAssets.flatMap((asset) => (Array.isArray(asset.tags) ? asset.tags : [])))].sort();

  // ── Batch operations (sc-6112) ─────────────────────────────────────────────
  // The current multi-selection, narrowed to the raster images a batch op can run on,
  // and the upscale engines this platform actually supports.
  const selectedAssetList = assets.filter((asset) => selectedAssetIds.has(asset.id));
  const eligibleSelected = batchEligibleAssets(selectedAssetList);
  const availableUpscaleEngines = UPSCALE_ENGINES.filter((engine) => !macUpscaleEngineBlocked(macCapabilities, engine.key));
  // Per-item + aggregate progress for an in-flight/just-finished batch, read off the jobs feed.
  const batchItems = batch
    ? batch.items.map((item) => ({ asset: item.asset, status: batchItemStatus(item.jobId, jobs) }))
    : null;
  const batchProgress = batch ? summarizeBatchProgress(batch.items, jobs) : null;

  const toggleSelect = (id) =>
    setSelectedAssetIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const clearSelection = () => setSelectedAssetIds(new Set());

  // Decode an asset's native pixel size (needed for an edit job — the worker fits the
  // source to width×height). Resolves null on a load failure so that item fails alone.
  function loadImageDims(asset) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve(null);
      img.src = assetUrl(asset);
    });
  }

  // Fan out one job per selected image (NOT one mega-job): each posts independently so
  // the worker processes them serially with its between-item cache release (sc-5567).
  // Library assets are persistent ids → no scratch upload and results persist as new
  // assets (the auto asset-refresh on job completion surfaces them in the grid).
  async function runBatch(op, params) {
    if (!activeProject || !eligibleSelected.length) return;
    const targets = eligibleSelected;
    setBatch({ op, submitting: true, items: targets.map((asset) => ({ asset, jobId: null })) });
    const items = [];
    for (const asset of targets) {
      try {
        let dims = null;
        if (op === "edit") {
          dims = await loadImageDims(asset);
          if (!dims) {
            items.push({ asset, jobId: null });
            continue;
          }
        }
        const { endpoint, body } = buildBatchJob({ op, asset, params, project: activeProject, requestedGpu, dims });
        const job = await apiFetch(endpoint, token, { method: "POST", body: JSON.stringify(body) });
        items.push({ asset, jobId: job?.id ?? null });
      } catch {
        items.push({ asset, jobId: null });
      }
    }
    setBatch({ op, submitting: false, items });
  }

  function closeBatch() {
    setBatchOpen(false);
    // Closing after a run clears the spent selection + progress; cancelling the form keeps it.
    if (batch) {
      setBatch(null);
      clearSelection();
    }
  }

  return (
    <section className="main-surface library-surface">
      <div className="surface-header hero">
        <div className="section-heading">
          <p className="eyebrow">Project assets</p>
          <h2>Assets</h2>
          <p className="hero-blurb">
            Browse stills and clips across {activeProject?.name ?? "your project"} — pick a recent render to drop into the editor, or send one back to a studio for another pass.
          </p>
        </div>
        <div className="toolbar">
          <label className="file-upload-button">
            <input accept="image/*,video/*" disabled={isImporting} onChange={handleImport} type="file" />
            {isImporting ? "Importing..." : "Import"}
          </label>
          <select aria-label="Asset type" onChange={(event) => setTypeFilter(event.target.value)} value={typeFilter}>
            <option value="all">All media</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="upload">Uploads</option>
            <option value="render">Renders</option>
          </select>
          <select aria-label="Asset tag" onChange={(event) => setTagFilter(event.target.value)} value={tagFilter}>
            <option value="all">All tags</option>
            {availableTags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
          <label className="checkline">
            <input checked={showRejected} onChange={(event) => setShowRejected(event.target.checked)} type="checkbox" />
            Rejected
          </label>
          <div className="segmented-control" role="group" aria-label="Asset collection">
            <button className={assetMode === "assets" ? "active" : ""} onClick={() => setAssetMode("assets")} type="button">
              Assets
            </button>
            <button className={assetMode === "trashcan" ? "active" : ""} onClick={() => setAssetMode("trashcan")} type="button">
              Trashcan
            </button>
          </div>
          {assetMode === "trashcan" ? (
            <button
              className="danger-action empty-trash-button"
              disabled={!trashedInView.length}
              onClick={() => emptyTrash(trashedInView, purgeAsset)}
              type="button"
            >
              Empty Trash ({trashedInView.length})
            </button>
          ) : null}
        </div>
        <div className="hero-stats">
          <div className="hero-stat">
            <span className="hero-stat-label">Project</span>
            <span className="hero-stat-value">{activeProject?.name ?? "—"}</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-label">Assets</span>
            <span className="hero-stat-value">{libraryAssets.length} total</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-label">Images</span>
            <span className="hero-stat-value">{imageCount}</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-label">Clips</span>
            <span className="hero-stat-value">{videoCount + uploadCount}</span>
          </div>
        </div>
      </div>

      {selectedAssetIds.size > 0 ? (
        <div className="batch-selection-bar">
          <span>
            {selectedAssetIds.size} selected
            {eligibleSelected.length !== selectedAssetIds.size
              ? ` · ${eligibleSelected.length} image${eligibleSelected.length === 1 ? "" : "s"}`
              : ""}
          </span>
          <button className="primary" disabled={!eligibleSelected.length} onClick={() => setBatchOpen(true)} type="button">
            Batch…
          </button>
          <button onClick={clearSelection} type="button">
            Clear
          </button>
        </div>
      ) : null}

      <div className="library-layout">
        <AssetGrid
          assets={visibleAssets}
          onPreview={onPreview}
          selectedAsset={selectedAsset}
          setSelectedAssetId={setSelectedAssetId}
          selectedIds={selectedAssetIds}
          onToggleSelect={toggleSelect}
        />
        <AssetDetail
          asset={selectedAsset}
          deleteAsset={deleteAsset}
          purgeAsset={purgeAsset}
          onPreview={onPreview}
          onSendImage={onSendImage}
          onSendVideo={onSendVideo}
          onSendEditor={onSendEditor}
          characters={characters}
          onMoveToCharacter={addCharacterReference ? moveAssetToCharacter : null}
          updateAssetStatus={updateAssetStatus}
          updateAssetTags={updateAssetTags}
          availableTags={availableTags}
          vqaEnabled={vqaEnabled}
          vqaEntries={vqaEntries}
          vqaPending={vqaPending}
          createVqaJob={createVqaJob}
        />
      </div>

      {batchOpen ? (
        <BatchOperationsPanel
          assets={eligibleSelected}
          editModels={editCapableModels(imageModels)}
          detailModels={detailCapableModels(imageModels)}
          upscaleEngines={availableUpscaleEngines}
          busy={Boolean(batch?.submitting)}
          items={batchItems}
          progress={batchProgress}
          onRun={runBatch}
          onClose={closeBatch}
        />
      ) : null}
    </section>
  );
}
