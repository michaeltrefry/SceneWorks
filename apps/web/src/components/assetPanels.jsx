import React from "react";
import { isAbortError } from "../api.js";
import { AssetMedia, assetUrl } from "./assetMedia.jsx";
import { DocumentView } from "./DocumentView.jsx";
import { Icon } from "./Icons.jsx";
import { Modal } from "./Modal.jsx";

// Reopens a saved interleaved document: the asset's file points to the segments
// JSON in assets/documents/, fetched here (served like any project file).
function DocumentReader({ asset }) {
  const url = assetUrl(asset);
  const [segments, setSegments] = React.useState(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    if (!url) {
      setError("Document file is unavailable.");
      return undefined;
    }
    const controller = new AbortController();
    setSegments(null);
    setError("");
    fetch(url, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Request failed (${response.status})`);
        }
        return response.json();
      })
      .then((document) => {
        setSegments(Array.isArray(document?.segments) ? document.segments : []);
      })
      .catch((err) => {
        if (isAbortError(err)) return;
        setError(String(err?.message ?? err));
      });
    return () => controller.abort();
  }, [url]);

  if (error) {
    return <p className="empty-panel error-text">Couldn't load document: {error}</p>;
  }
  if (segments === null) {
    return <p className="empty-panel">Loading document…</p>;
  }
  return <DocumentView projectId={asset.projectId} segments={segments} />;
}

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

const VQA_LENGTHS = [
  { value: 256, label: "Short (256)" },
  { value: 512, label: "Medium (512)" },
  { value: 1024, label: "Long (1024)" },
];

function VqaPanel({ asset, entries, pending, onAsk }) {
  const [question, setQuestion] = React.useState("");
  const [maxNewTokens, setMaxNewTokens] = React.useState(256);
  const trimmed = question.trim();
  const submit = () => {
    if (!trimmed) {
      return;
    }
    onAsk(asset, trimmed, maxNewTokens);
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
      <div className="vqa-controls">
        <label className="vqa-length">
          Response length
          <select
            aria-label="Response length"
            onChange={(event) => setMaxNewTokens(Number(event.target.value))}
            value={maxNewTokens}
          >
            {VQA_LENGTHS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <button disabled={!trimmed || pending} onClick={submit} type="button">
          {pending ? "Asking…" : "Ask"}
        </button>
      </div>
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
      {asset.type === "document" ? (
        <DocumentReader asset={asset} />
      ) : (
        <button className="preview-button" onClick={() => onPreview(asset)} type="button">
          <AssetMedia asset={asset} />
        </button>
      )}
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
    <Modal className="preview-modal" label="Asset preview" onClose={onClose}>
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
    </Modal>
  );
}
