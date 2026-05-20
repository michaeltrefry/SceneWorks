import React from "react";
import { AssetMedia } from "./assetMedia.jsx";
import { Icon } from "./Icons.jsx";

function RatingControl({ asset, updateAssetStatus }) {
  const currentRating = Number(asset.status?.rating) || 0;

  return (
    <div aria-label="Rating" className="rating-control" role="group">
      <span className="rating-label">Rating</span>
      <div className="rating-stars">
        {[1, 2, 3, 4, 5].map((rating) => {
          const active = currentRating >= rating;
          return (
            <button
              aria-label={`Set rating to ${rating} ${rating === 1 ? "star" : "stars"}`}
              aria-pressed={active}
              className={active ? "star-rating-button active" : "star-rating-button"}
              key={rating}
              onClick={() => updateAssetStatus(asset, { rating })}
              title={`${rating} ${rating === 1 ? "star" : "stars"}`}
              type="button"
            >
              <Icon.Star filled={active} size={18} />
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function AssetGrid({ assets, onPreview, selectedAsset, setSelectedAssetId }) {
  if (!assets.length) {
    return <div className="empty-panel">No assets in this view</div>;
  }

  return (
    <div className="asset-grid">
      {assets.map((asset) => (
        <button
          className={selectedAsset?.id === asset.id ? "asset-tile active" : "asset-tile"}
          key={asset.id}
          onClick={() => setSelectedAssetId(asset.id)}
          onDoubleClick={() => onPreview(asset)}
          type="button"
        >
          <AssetMedia asset={asset} />
          <strong>{asset.displayName}</strong>
        </button>
      ))}
    </div>
  );
}

export function AssetDetail({ asset, deleteAsset, purgeAsset, onPreview, onSendImage, onSendVideo, onSendEditor, updateAssetStatus }) {
  if (!asset) {
    return <aside className="asset-detail empty-panel">No asset selected</aside>;
  }

  return (
    <aside className="asset-detail">
      <button className="preview-button" onClick={() => onPreview(asset)} type="button">
        <AssetMedia asset={asset} />
      </button>
      <h3>{asset.displayName}</h3>
      <p>{asset.recipe?.prompt ?? "No prompt"}</p>
      <RatingControl asset={asset} updateAssetStatus={updateAssetStatus} />
      <div className="detail-actions">
        <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
          {asset.status?.favorite ? "Unfavorite" : "Favorite"}
        </button>
        <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
          {asset.status?.rejected ? "Restore" : "Reject"}
        </button>
        {asset.type === "image" ? (
          <button onClick={() => onSendImage(asset)} type="button">
            Send to Image
          </button>
        ) : null}
        {asset.type === "image" ? (
          <button onClick={() => onSendVideo(asset)} type="button">
            Send to Video
          </button>
        ) : null}
        {["image", "video", "upload", "frame"].includes(asset.type) ? (
          <button onClick={() => onSendEditor(asset)} type="button">
            Send to Editor
          </button>
        ) : null}
        {asset.status?.trashed ? (
          <button onClick={() => purgeAsset(asset)} type="button">
            Purge
          </button>
        ) : (
          <button onClick={() => deleteAsset(asset)} type="button">
            Discard
          </button>
        )}
      </div>
      <dl>
        <div>
          <dt>Model</dt>
          <dd>{asset.recipe?.model ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Duration</dt>
          <dd>{asset.file?.duration ? `${asset.file.duration}s` : "Still"}</dd>
        </div>
        <div>
          <dt>Generation set</dt>
          <dd>{asset.generationSetId ?? "None"}</dd>
        </div>
      </dl>
    </aside>
  );
}

export function AssetCard({ asset, deleteAsset, purgeAsset, onPreview, updateAssetStatus }) {
  const classes = ["review-card", asset.status?.rejected ? "rejected" : "", asset.status?.trashed ? "trashed" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <article className={classes}>
      <button className="preview-button" onClick={() => onPreview(asset)} type="button">
        <AssetMedia asset={asset} />
      </button>
      <div className="review-actions">
        <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
          {asset.status?.favorite ? "Saved" : "Favorite"}
        </button>
        <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
          {asset.status?.rejected ? "Restore" : "Reject"}
        </button>
        {asset.status?.trashed ? (
          <button onClick={() => purgeAsset(asset)} type="button">
            Purge
          </button>
        ) : (
          <button onClick={() => deleteAsset(asset)} type="button">
            Discard
          </button>
        )}
      </div>
    </article>
  );
}

export function FullscreenPreview({ asset, onClose }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="preview-modal">
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
        <AssetMedia asset={asset} />
        <footer>
          <strong>{asset.displayName}</strong>
          <span>{asset.recipe?.model}</span>
        </footer>
      </div>
    </div>
  );
}
