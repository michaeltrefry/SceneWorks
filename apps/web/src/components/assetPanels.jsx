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

function VqaPanel({ asset, entries, pending, onAsk }) {
  const [question, setQuestion] = React.useState("");
  const trimmed = question.trim();
  const submit = () => {
    if (!trimmed) {
      return;
    }
    onAsk(asset, trimmed);
    setQuestion("");
  };

  return (
    <div className="vqa-panel">
      <label className="vqa-ask">
        Ask about this image
        <textarea
          aria-label="Ask about this image"
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="e.g. What is the person wearing?"
          rows={2}
          value={question}
        />
      </label>
      <button disabled={!trimmed || pending} onClick={submit} type="button">
        {pending ? "Asking…" : "Ask"}
      </button>
      {entries.length ? (
        <ul className="vqa-history">
          {entries.map((entry) => (
            <li key={entry.jobId}>
              <strong>{entry.question}</strong>
              <p>{entry.answer}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function AssetDetail({
  asset,
  deleteAsset,
  purgeAsset,
  onPreview,
  onSendImage,
  onSendVideo,
  onSendEditor,
  updateAssetStatus,
  vqaEnabled = false,
  vqaEntries = [],
  vqaPending = false,
  createVqaJob,
}) {
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
      {vqaEnabled && asset.type === "image" ? (
        <VqaPanel asset={asset} entries={vqaEntries} pending={vqaPending} onAsk={createVqaJob} />
      ) : null}
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

export function FullscreenPreview({
  asset,
  deleteAsset,
  nextAsset,
  onClose,
  onPreviewAsset,
  previousAsset,
  purgeAsset,
  updateAssetStatus,
}) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="preview-modal">
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
        <div className="preview-modal-stage">
          <button
            aria-label="Previous asset"
            className="preview-nav-button previous"
            disabled={!previousAsset}
            onClick={() => previousAsset && onPreviewAsset(previousAsset)}
            type="button"
          >
            <Icon.ArrowLeft size={18} />
          </button>
          <AssetMedia asset={asset} />
          <button
            aria-label="Next asset"
            className="preview-nav-button next"
            disabled={!nextAsset}
            onClick={() => nextAsset && onPreviewAsset(nextAsset)}
            type="button"
          >
            <Icon.ArrowRight size={18} />
          </button>
        </div>
        <footer>
          <div className="preview-modal-meta">
            <strong>{asset.displayName}</strong>
            <span>{asset.recipe?.model}</span>
          </div>
          <div className="preview-actions">
            <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
              {asset.status?.favorite ? "Saved" : "Favorite"}
            </button>
            <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
              {asset.status?.rejected ? "Restore" : "Reject"}
            </button>
            {asset.status?.trashed ? (
              <button className="danger-action" onClick={() => purgeAsset(asset)} type="button">
                Purge
              </button>
            ) : (
              <button className="danger-action" onClick={() => deleteAsset(asset)} type="button">
                Discard
              </button>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}
