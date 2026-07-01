import React from "react";
import { isAbortError } from "../api.js";
import { AssetMedia, assetCanRenderAsVideo, assetUrl, suppressThumbnailContextMenu } from "./assetMedia.jsx";
import { DocumentView } from "./DocumentView.jsx";
import { Icon } from "./Icons.jsx";
import { LikenessBadge } from "./LikenessBadge.jsx";
import { Modal } from "./Modal.jsx";

// Zoom/pan model for the fullscreen image preview (sc-8728). Mirrors the Image
// Editor canvas convention (`view = { scale, x, y }`, ImageEditor.jsx): `scale`
// multiplies the image and `x`/`y` translate its top-left corner within the
// stage, applied as `translate(x, y) scale(scale)` with `transform-origin: 0 0`.
// `scale === 1` is fit-to-view here (the <img> is already sized to contain the
// stage at 100% width), so fit resets to the identity view.
export const PREVIEW_MIN_SCALE = 1;
export const PREVIEW_MAX_SCALE = 8;
export const PREVIEW_ZOOM_STEP = 1.2;
export const PREVIEW_FIT_VIEW = { scale: 1, x: 0, y: 0 };

const clampScale = (value) => Math.min(PREVIEW_MAX_SCALE, Math.max(PREVIEW_MIN_SCALE, value));

// Cursor-anchored zoom: keep the image point under `pointer` (stage-relative px)
// stationary while multiplying the scale by `factor`. Pure so the anchoring math
// is unit-testable in isolation; mirrors ImageEditor's handleWheel/zoomAtCenter.
export function zoomView(view, pointer, factor) {
  const oldScale = view.scale;
  const newScale = clampScale(oldScale * factor);
  if (newScale === oldScale) {
    return view;
  }
  const imagePoint = { x: (pointer.x - view.x) / oldScale, y: (pointer.y - view.y) / oldScale };
  return {
    scale: newScale,
    x: pointer.x - imagePoint.x * newScale,
    y: pointer.y - imagePoint.y * newScale,
  };
}

// Clamp the pan offset so the (scaled) image can't be dragged fully off the
// stage: at scale 1 it stays centered (offset 0); when zoomed the visible edge
// can travel to the stage edge but no further. Pure for testing.
export function clampPan(view, stageWidth, stageHeight) {
  const scaledW = stageWidth * view.scale;
  const scaledH = stageHeight * view.scale;
  const minX = Math.min(0, stageWidth - scaledW);
  const minY = Math.min(0, stageHeight - scaledH);
  return {
    scale: view.scale,
    x: Math.min(0, Math.max(minX, view.x)),
    y: Math.min(0, Math.max(minY, view.y)),
  };
}

// Permanently purge every discarded asset in a single Trashcan view. The caller
// passes only the assets visible in that view, so per-character and per-filter
// trashcans empty their own scope and nothing else. Confirms first (guarded so
// headless/test environments skip the prompt) and purges sequentially.
export async function emptyTrash(trashedAssets, purgeAsset) {
  const items = (trashedAssets ?? []).filter((asset) => asset?.status?.trashed);
  if (!items.length || typeof purgeAsset !== "function") {
    return;
  }
  if (
    typeof window !== "undefined" &&
    typeof window.confirm === "function" &&
    !window.confirm(
      `Permanently delete ${items.length} discarded item${items.length === 1 ? "" : "s"}? This cannot be undone.`,
    )
  ) {
    return;
  }
  for (const asset of items) {
    // eslint-disable-next-line no-await-in-loop -- sequential keeps state updates predictable
    await purgeAsset(asset);
  }
}

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

function normalizeTagInput(value) {
  return String(value ?? "").trim().toLowerCase();
}

function AssetTags({ asset, availableTags = [], updateAssetTags = () => {} }) {
  const [draft, setDraft] = React.useState("");
  const tags = Array.isArray(asset.tags) ? asset.tags : [];
  const suggestions = availableTags.filter((tag) => !tags.includes(tag));

  function submit(event) {
    event.preventDefault();
    const tag = normalizeTagInput(draft);
    if (!tag || tags.includes(tag)) {
      setDraft("");
      return;
    }
    updateAssetTags(asset, [...tags, tag]);
    setDraft("");
  }

  return (
    <div className="asset-tags">
      <div className="asset-tag-list" aria-label="Asset tags">
        {tags.length ? (
          tags.map((tag) => (
            <span className="asset-tag" key={tag}>
              {tag}
              <button aria-label={`Remove ${tag} tag`} onClick={() => updateAssetTags(asset, tags.filter((item) => item !== tag))} type="button">
                Remove
              </button>
            </span>
          ))
        ) : (
          <span className="asset-tag-empty">No tags</span>
        )}
      </div>
      <form className="asset-tag-form" onSubmit={submit}>
        <input
          aria-label="Add asset tag"
          list="asset-tag-suggestions"
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Add tag"
          value={draft}
        />
        <datalist id="asset-tag-suggestions">
          {suggestions.map((tag) => (
            <option key={tag} value={tag} />
          ))}
        </datalist>
        <button type="submit">Add</button>
      </form>
    </div>
  );
}

export function assetSupportsCharacterLink(asset) {
  return (
    ["image", "frame", "upload", "video"].includes(asset?.type) ||
    asset?.file?.mimeType?.startsWith("image/") ||
    asset?.file?.mimeType?.startsWith("video/")
  );
}

function assetLinkedToCharacter(asset, character) {
  const characterId = character?.id;
  if (!asset?.id || !characterId) {
    return false;
  }
  if (asset.recipe?.normalizedSettings?.characterId === characterId) {
    return true;
  }
  if ((asset.metadata?.characterReferences ?? []).some((reference) => reference?.characterId === characterId)) {
    return true;
  }
  return (character.references ?? []).some((reference) => (reference?.assetId ?? reference?.id) === asset.id);
}

function CharacterAssetLinker({ asset, characters = [], onMoveToCharacter }) {
  const availableCharacters = React.useMemo(
    () => characters.filter((character) => !character?.archived),
    [characters],
  );
  const [characterId, setCharacterId] = React.useState(availableCharacters[0]?.id ?? "");
  const [message, setMessage] = React.useState("");
  const [moving, setMoving] = React.useState(false);

  React.useEffect(() => {
    setMessage("");
    setMoving(false);
  }, [asset?.id]);

  React.useEffect(() => {
    if (!availableCharacters.some((character) => character.id === characterId)) {
      setCharacterId(availableCharacters[0]?.id ?? "");
    }
  }, [availableCharacters, characterId]);

  if (!assetSupportsCharacterLink(asset) || !availableCharacters.length || typeof onMoveToCharacter !== "function") {
    return null;
  }

  const selectedCharacter = availableCharacters.find((character) => character.id === characterId) ?? null;
  const alreadyLinked = assetLinkedToCharacter(asset, selectedCharacter);

  async function submit(event) {
    event.preventDefault();
    if (!selectedCharacter || alreadyLinked || moving) {
      return;
    }
    setMoving(true);
    setMessage("");
    try {
      await onMoveToCharacter(asset, selectedCharacter.id);
      setMessage(`Added to ${selectedCharacter.name}'s assets.`);
    } catch (err) {
      setMessage(err?.message ?? "Could not add this asset to the character.");
    } finally {
      setMoving(false);
    }
  }

  return (
    <form className="character-asset-linker" onSubmit={submit}>
      <label>
        Character
        <select aria-label="Target character" onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
          {availableCharacters.map((character) => (
            <option key={character.id} value={character.id}>
              {character.name}
            </option>
          ))}
        </select>
      </label>
      <button
        disabled={!selectedCharacter || alreadyLinked || moving}
        title={alreadyLinked ? "Already in this character's assets" : undefined}
        type="submit"
      >
        {moving ? "Moving..." : alreadyLinked ? "Already added" : "Move to Character"}
      </button>
      {message ? <p aria-live="polite">{message}</p> : null}
    </form>
  );
}

// `onToggleSelect` (sc-6112) opts the grid into multi-select: each tile gets a
// checkbox (a sibling of the tile button, so no interactive element nests inside the
// button) toggling membership in `selectedIds`. The tile-body click still drives the
// single-select detail flow unchanged, so the default Library behavior is preserved.
export function AssetGrid({ assets, onPreview, selectedAsset, setSelectedAssetId, selectedIds = null, onToggleSelect = null }) {
  if (!assets.length) {
    return <div className="empty-panel">No assets in this view</div>;
  }
  const multi = typeof onToggleSelect === "function";

  return (
    <div className="asset-grid">
      {assets.map((asset) => {
        const tile = (
          <button
            className={selectedAsset?.id === asset.id ? "asset-tile active" : "asset-tile"}
            onClick={() => setSelectedAssetId(asset.id)}
            onContextMenu={suppressThumbnailContextMenu}
            onDoubleClick={() => onPreview(asset)}
            type="button"
          >
            <AssetMedia asset={asset} />
            <strong>{asset.displayName}</strong>
            {Array.isArray(asset.tags) && asset.tags.length ? (
              <div className="asset-tile-tags">
                {asset.tags.map((tag) => (
                  <span className="asset-tag compact" key={tag}>
                    {tag}
                  </span>
                ))}
              </div>
            ) : null}
          </button>
        );
        if (!multi) {
          return React.cloneElement(tile, { key: asset.id });
        }
        return (
          <div className={selectedIds?.has(asset.id) ? "asset-tile-wrap selected" : "asset-tile-wrap"} key={asset.id}>
            <label className="asset-tile-check">
              <input
                aria-label={`Select ${asset.displayName}`}
                checked={Boolean(selectedIds?.has(asset.id))}
                onChange={() => onToggleSelect(asset.id)}
                type="checkbox"
              />
            </label>
            {tile}
          </div>
        );
      })}
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
  characters = [],
  onMoveToCharacter,
  updateAssetStatus,
  updateAssetTags,
  availableTags,
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
      <AssetTags asset={asset} availableTags={availableTags} updateAssetTags={updateAssetTags} />
      <RatingControl asset={asset} updateAssetStatus={updateAssetStatus} />
      <CharacterAssetLinker asset={asset} characters={characters} onMoveToCharacter={onMoveToCharacter} />
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
          <>
            <button onClick={() => updateAssetStatus(asset, { trashed: false })} type="button">
              Restore
            </button>
            <button onClick={() => purgeAsset(asset)} type="button">
              Purge
            </button>
          </>
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
      <button className="preview-button" onClick={() => onPreview(asset)} onContextMenu={suppressThumbnailContextMenu} type="button">
        <AssetMedia asset={asset} />
        <LikenessBadge asset={asset} />
      </button>
      <div className="review-actions">
        <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
          {asset.status?.favorite ? "Saved" : "Favorite"}
        </button>
        <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
          {asset.status?.rejected ? "Restore" : "Reject"}
        </button>
        {asset.status?.trashed ? (
          <>
            <button onClick={() => updateAssetStatus(asset, { trashed: false })} type="button">
              Restore
            </button>
            <button onClick={() => purgeAsset(asset)} type="button">
              Purge
            </button>
          </>
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
  onEditImage,
  onPreviewAsset,
  onUseRecipe,
  previousAsset,
  purgeAsset,
  updateAssetStatus,
}) {
  const hasUpscaleVariants = Boolean(asset.variants?.original && asset.variants?.upscaled);
  const [variantMode, setVariantMode] = React.useState("upscaled");
  React.useEffect(() => {
    setVariantMode(asset.variants?.upscaled ? "upscaled" : "original");
  }, [asset.id, asset.variants?.upscaled?.id]);
  const displayedAsset = hasUpscaleVariants ? asset.variants[variantMode] : asset;

  // Zoom/pan is IMAGES ONLY — video keeps its native <video> controls and gets no
  // zoom overlay/UI (sc-8728). `isVideo` gates the whole zoom surface.
  const isVideo = assetCanRenderAsVideo(displayedAsset);

  const viewportRef = React.useRef(null);
  const [view, setView] = React.useState(PREVIEW_FIT_VIEW);
  const dragRef = React.useRef(null);

  const fitToView = React.useCallback(() => setView(PREVIEW_FIT_VIEW), []);

  // Reset to fit whenever the displayed image changes: prev/next navigation
  // (asset.id) OR a variant toggle (variantMode / the variant's own id).
  React.useEffect(() => {
    fitToView();
  }, [asset.id, variantMode, displayedAsset?.id, fitToView]);

  const stageSize = React.useCallback(() => {
    const rect = viewportRef.current?.getBoundingClientRect?.();
    return { width: rect?.width || 0, height: rect?.height || 0 };
  }, []);

  const zoomAtCenter = React.useCallback(
    (factor) => {
      const { width, height } = stageSize();
      const pointer = { x: width / 2, y: height / 2 };
      setView((current) => clampPan(zoomView(current, pointer, factor), width, height));
    },
    [stageSize],
  );

  const zoomIn = React.useCallback(() => zoomAtCenter(PREVIEW_ZOOM_STEP), [zoomAtCenter]);
  const zoomOut = React.useCallback(() => zoomAtCenter(1 / PREVIEW_ZOOM_STEP), [zoomAtCenter]);

  // Wheel-to-zoom anchored at the cursor. Native (non-passive) listener so we can
  // preventDefault the page scroll; React's onWheel is passive in some browsers.
  React.useEffect(() => {
    const node = viewportRef.current;
    if (!node || isVideo) {
      return undefined;
    }
    const onWheel = (event) => {
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      const pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      const factor = event.deltaY > 0 ? 1 / PREVIEW_ZOOM_STEP : PREVIEW_ZOOM_STEP;
      setView((current) => clampPan(zoomView(current, pointer, factor), rect.width, rect.height));
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [isVideo, displayedAsset?.id]);

  const onPointerDown = (event) => {
    if (view.scale <= PREVIEW_MIN_SCALE) {
      return; // no pan at fit scale
    }
    dragRef.current = { startX: event.clientX, startY: event.clientY, originX: view.x, originY: view.y };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const { width, height } = stageSize();
    const next = { scale: view.scale, x: drag.originX + (event.clientX - drag.startX), y: drag.originY + (event.clientY - drag.startY) };
    setView(clampPan(next, width, height));
  };

  const endDrag = (event) => {
    if (dragRef.current) {
      dragRef.current = null;
      event.currentTarget?.releasePointerCapture?.(event.pointerId);
    }
  };

  const zoomed = view.scale > PREVIEW_MIN_SCALE;

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
            onClick={() => previousAsset && onPreviewAsset(previousAsset, "previous")}
            type="button"
          >
            <Icon.ArrowLeft size={18} />
          </button>
          {isVideo ? (
            <AssetMedia asset={displayedAsset} />
          ) : (
            <div
              className={`preview-zoom-viewport${zoomed ? " zoomed" : ""}`}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              ref={viewportRef}
            >
              <div
                className="preview-zoom-inner"
                style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`, transformOrigin: "0 0" }}
              >
                <AssetMedia asset={displayedAsset} />
              </div>
            </div>
          )}
          <LikenessBadge asset={asset} />
          <button
            aria-label="Next asset"
            className="preview-nav-button next"
            disabled={!nextAsset}
            onClick={() => nextAsset && onPreviewAsset(nextAsset, "next")}
            type="button"
          >
            <Icon.ArrowRight size={18} />
          </button>
          {isVideo ? null : (
            <div className="preview-zoom-controls" role="group" aria-label="Zoom controls">
              <button
                aria-label="Zoom out"
                disabled={view.scale <= PREVIEW_MIN_SCALE}
                onClick={zoomOut}
                title="Zoom out"
                type="button"
              >
                <Icon.Minus size={16} />
              </button>
              <button aria-label="Fit to view" onClick={fitToView} title="Fit to view" type="button">
                Fit
              </button>
              <button
                aria-label="Zoom in"
                disabled={view.scale >= PREVIEW_MAX_SCALE}
                onClick={zoomIn}
                title="Zoom in"
                type="button"
              >
                <Icon.Plus size={16} />
              </button>
            </div>
          )}
        </div>
        <footer>
          <div className="preview-modal-meta">
            <strong>{asset.displayName}</strong>
            <span>{asset.recipe?.model}</span>
          </div>
          {hasUpscaleVariants ? (
            <div className="segmented-control compact-segment preview-variant-toggle" aria-label="Image variant">
              <button
                className={variantMode === "original" ? "active" : ""}
                onClick={() => setVariantMode("original")}
                type="button"
              >
                Original
              </button>
              <button
                className={variantMode === "upscaled" ? "active" : ""}
                onClick={() => setVariantMode("upscaled")}
                type="button"
              >
                Upscaled
              </button>
            </div>
          ) : null}
          <div className="preview-actions">
            {onUseRecipe && asset.type === "image" && (asset.generationSet?.recipe || asset.recipe) ? (
              <button onClick={() => onUseRecipe(asset)} type="button">
                Use this recipe
              </button>
            ) : null}
            {onEditImage && asset.type === "image" ? (
              <button onClick={() => onEditImage(asset)} type="button">
                Edit
              </button>
            ) : null}
            <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
              {asset.status?.favorite ? "Saved" : "Favorite"}
            </button>
            <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
              {asset.status?.rejected ? "Restore" : "Reject"}
            </button>
            {asset.status?.trashed ? (
              <>
                <button onClick={() => updateAssetStatus(asset, { trashed: false })} type="button">
                  Restore
                </button>
                <button className="danger-action" onClick={() => purgeAsset(asset)} type="button">
                  Purge
                </button>
              </>
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
