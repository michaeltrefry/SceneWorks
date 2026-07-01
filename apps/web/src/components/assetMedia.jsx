import React from "react";
import { API_BASE_URL } from "../api.js";

export function assetUrl(asset) {
  if (asset?.url) {
    return API_BASE_URL + asset.url;
  }
  if (asset?.projectId && asset?.file?.path) {
    const normalizedPath = String(asset.file.path)
      .replaceAll("\\", "/")
      .split("/")
      .filter(Boolean)
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    return `${API_BASE_URL}/api/v1/projects/${asset.projectId}/files/${normalizedPath}`;
  }
  return "";
}

export function assetCanRenderAsImage(asset) {
  return asset?.type === "image" || asset?.file?.mimeType?.startsWith("image/");
}

export function assetCanRenderAsVideo(asset) {
  return asset?.type === "video" || asset?.file?.mimeType?.startsWith("video/");
}

// Suppress the native WKWebView context menu on grid thumbnails (sc-8731). Right-
// clicking a thumbnail image in a Tauri webview otherwise pops the OS "Download
// Image / Copy Image / Share" menu; thumbnails have no custom menu, so we just
// swallow the default. Only preventDefault the contextmenu event — never
// stopPropagation of clicks — so left-click selection / open-preview stay intact.
// Applied at the shared AssetThumbnail seam so every grid that renders it (Queue's
// WorkerProgressCard, pickers, studios) inherits the suppression from one place.
// The Library grid renders AssetMedia directly (assetPanels.jsx), so AssetGrid
// imports this and attaches it at the tile-cell level. The full-size
// FullscreenPreview renderer is intentionally left alone: it gets its own custom
// right-click menu in sc-8729.
export function suppressThumbnailContextMenu(event) {
  event.preventDefault();
}

// Generated videos get a sibling `<name>.poster.jpg` (the worker extracts frame 0).
// WKWebView won't paint a <video>'s own first frame as a poster, so the UI shows
// this real image instead — as the thumbnail and as the player's poster attribute.
export function posterUrl(asset) {
  const src = assetUrl(asset);
  if (!src || !assetCanRenderAsVideo(asset)) {
    return "";
  }
  return src.replace(/\.\w+$/, ".poster.jpg");
}

// Placeholder shown when an asset's underlying file can't be loaded — e.g. it
// was purged from disk after the job ran, so the URL now 404s. Replaces the
// browser's broken-image glyph with a clear "deleted" marker (a red X) so queue
// thumbnails for purged outputs read as removed rather than broken.
export function MissingMedia({ className = "" }) {
  return (
    <span
      aria-label="Deleted asset"
      className={`asset-thumb-missing ${className}`.trim()}
      onContextMenu={suppressThumbnailContextMenu}
      role="img"
      title="Deleted"
    >
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M6 6l12 12M18 6L6 18" />
      </svg>
    </span>
  );
}

// Image thumbnail that falls back to the deleted-asset placeholder once the
// source fails to load (the file is gone), rather than leaving a broken image.
function ImageThumb({ src, className }) {
  const [failed, setFailed] = React.useState(false);
  if (failed) {
    return <MissingMedia className={className} />;
  }
  return <img alt="" className={className} onContextMenu={suppressThumbnailContextMenu} onError={() => setFailed(true)} src={src} />;
}

export function AssetThumbnail({ asset, className = "" }) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (!src) {
    return <span className={className} onContextMenu={suppressThumbnailContextMenu}>{asset.type ?? "asset"}</span>;
  }
  if (assetCanRenderAsVideo(asset)) {
    return <VideoPoster asset={asset} className={className} />;
  }
  if (assetCanRenderAsImage(asset)) {
    return <ImageThumb src={src} className={className} />;
  }
  return <span className={className}>{asset.type ?? "asset"}</span>;
}

function VideoPoster({ asset, className }) {
  const [failed, setFailed] = React.useState(false);
  const poster = posterUrl(asset);
  if (!poster) {
    return <span className={className} onContextMenu={suppressThumbnailContextMenu}>{asset.type ?? "video"}</span>;
  }
  if (failed) {
    return <MissingMedia className={className} />;
  }
  return <img alt="" className={className} onContextMenu={suppressThumbnailContextMenu} onError={() => setFailed(true)} src={poster} />;
}

export const AssetMedia = React.forwardRef(function AssetMedia({ asset, className = "", controls = true, ...mediaProps }, ref) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (!src) {
    return <span className={className}>{asset.type ?? "asset"}</span>;
  }
  if (assetCanRenderAsVideo(asset)) {
    return (
      <video
        className={className}
        controls={controls}
        muted
        playsInline
        poster={posterUrl(asset)}
        preload="metadata"
        ref={ref}
        src={src}
        {...mediaProps}
      />
    );
  }
  if (assetCanRenderAsImage(asset)) {
    return <img alt="" className={className} ref={ref} src={src} />;
  }
  return <span className={className}>{asset.type}</span>;
});
