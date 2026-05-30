import React, { useState } from "react";
import { foldUpscaledAssetVariants } from "../assetVariants.js";
import { AssetDetail, AssetGrid, emptyTrash } from "../components/assetPanels.jsx";
import { terminalStatuses } from "../constants.js";
import { useAppContext } from "../context/AppContext.js";

export function LibraryScreen() {
  const {
    activeProject,
    assets,
    jobs = [],
    imageModels = [],
    createVqaJob,
    deleteAsset,
    purgeAsset,
    importAsset,
    setPreviewAsset,
    sendAssetToImage,
    sendAssetToVideo,
    selectedAsset,
    setSelectedAssetId,
    setActiveView,
    updateAssetStatus,
    updateAssetTags,
  } = useAppContext();
  const onPreview = setPreviewAsset;
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

  const imageCount = libraryAssets.filter((asset) => asset.type === "image").length;
  const videoCount = libraryAssets.filter((asset) => asset.type === "video").length;
  const uploadCount = libraryAssets.filter((asset) => asset.type === "upload").length;
  const availableTags = [...new Set(libraryAssets.flatMap((asset) => (Array.isArray(asset.tags) ? asset.tags : [])))].sort();

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

      <div className="library-layout">
        <AssetGrid
          assets={visibleAssets}
          onPreview={onPreview}
          selectedAsset={selectedAsset}
          setSelectedAssetId={setSelectedAssetId}
        />
        <AssetDetail
          asset={selectedAsset}
          deleteAsset={deleteAsset}
          purgeAsset={purgeAsset}
          onPreview={onPreview}
          onSendImage={onSendImage}
          onSendVideo={onSendVideo}
          onSendEditor={onSendEditor}
          updateAssetStatus={updateAssetStatus}
          updateAssetTags={updateAssetTags}
          availableTags={availableTags}
          vqaEnabled={vqaEnabled}
          vqaEntries={vqaEntries}
          vqaPending={vqaPending}
          createVqaJob={createVqaJob}
        />
      </div>
    </section>
  );
}
