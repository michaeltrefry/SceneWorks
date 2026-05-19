import React, { useState } from "react";
import { AssetDetail, AssetGrid } from "../components/assetPanels.jsx";

export function LibraryScreen({
  activeProject,
  assets,
  deleteAsset,
  purgeAsset,
  importAsset,
  onPreview,
  onSendImage,
  onSendVideo,
  onSendEditor,
  selectedAsset,
  setSelectedAssetId,
  updateAssetStatus,
}) {
  const [typeFilter, setTypeFilter] = useState("all");
  const [showRejected, setShowRejected] = useState(false);
  const [showTrashed, setShowTrashed] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const visibleAssets = assets.filter((asset) => {
    if (typeFilter !== "all" && asset.type !== typeFilter) {
      return false;
    }
    if (!showRejected && asset.status?.rejected) {
      return false;
    }
    if (!showTrashed && asset.status?.trashed) {
      return false;
    }
    return true;
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

  const imageCount = assets.filter((asset) => asset.type === "image").length;
  const videoCount = assets.filter((asset) => asset.type === "video").length;
  const uploadCount = assets.filter((asset) => asset.type === "upload").length;

  return (
    <section className="main-surface library-surface">
      <div className="surface-header hero">
        <div className="section-heading">
          <p className="eyebrow">Project assets</p>
          <h2>Library</h2>
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
          <label className="checkline">
            <input checked={showRejected} onChange={(event) => setShowRejected(event.target.checked)} type="checkbox" />
            Rejected
          </label>
          <label className="checkline">
            <input checked={showTrashed} onChange={(event) => setShowTrashed(event.target.checked)} type="checkbox" />
            Trash
          </label>
        </div>
        <div className="hero-stats">
          <div className="hero-stat">
            <span className="hero-stat-label">Project</span>
            <span className="hero-stat-value">{activeProject?.name ?? "—"}</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat-label">Assets</span>
            <span className="hero-stat-value">{assets.length} total</span>
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
        />
      </div>
    </section>
  );
}
