// Shared "Save As" / "Reveal in Finder/Explorer" action layer for assets (sc-8727).
//
// Both the Save As button (sc-8728) and the asset context menu (sc-8729) call these
// functions, so the desktop-vs-browser branching lives in exactly one place:
//
//   - Desktop shell (Tauri): resolve the asset's absolute on-disk path via the
//     `resolve_asset_path` command, then hand it to `save_asset_as` (native save
//     dialog + copy) or `reveal_in_os` (Finder/Explorer select).
//   - Remote browser / LAN mode (no `window.__TAURI__`): fall back to a normal
//     browser download of the asset's served URL. Reveal has no browser analogue,
//     so callers hide it and `revealAsset` throws if invoked anyway.
//
// Content-agnostic: works identically for image and video assets.
import { isDesktop, tauriInvoke } from "./runtime.js";
import { assetUrl } from "./components/assetMedia.jsx";

// Map the common asset MIME types to a file extension, for when the source path
// carries no extension we can reuse. Kept small and explicit rather than pulling a
// mime-db dependency — these are the formats the worker actually emits.
const MIME_EXTENSIONS = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/jpg": "jpg",
  "image/webp": "webp",
  "image/gif": "gif",
  "image/avif": "avif",
  "image/bmp": "bmp",
  "image/tiff": "tiff",
  "video/mp4": "mp4",
  "video/webm": "webm",
  "video/quicktime": "mov",
  "video/x-matroska": "mkv",
};

// Pull the lowercase extension (no dot) off a path/filename, or "" if it has none.
// Guards against a trailing dot and a leading-dot "dotfile" with no real extension.
function extensionFromPath(pathOrName) {
  const name = String(pathOrName ?? "")
    .replaceAll("\\", "/")
    .split("/")
    .pop();
  const dot = name.lastIndexOf(".");
  if (dot <= 0 || dot === name.length - 1) {
    return "";
  }
  return name.slice(dot + 1).toLowerCase();
}

// The extension we believe the asset's bytes actually are: prefer the source file
// path's extension (that's the real on-disk format), then fall back to the MIME type.
function assetExtension(asset) {
  const fromPath = extensionFromPath(asset?.file?.path);
  if (fromPath) {
    return fromPath;
  }
  const mime = String(asset?.file?.mimeType ?? "").toLowerCase().split(";")[0].trim();
  return MIME_EXTENSIONS[mime] ?? "";
}

// The filename we suggest in the save dialog / download: the asset's display name,
// with a correct extension appended when it's missing (or mismatched from the real
// format). Falls back to the asset id, then a generic "asset", so we never suggest an
// empty name.
export function suggestedFilename(asset) {
  const base = String(asset?.displayName ?? asset?.id ?? "asset").trim() || "asset";
  const ext = assetExtension(asset);
  if (!ext) {
    return base;
  }
  // Already ends with the right extension — leave it as-is.
  if (extensionFromPath(base) === ext) {
    return base;
  }
  return `${base}.${ext}`;
}

// Resolve an asset to its absolute on-disk path via the desktop command. Only valid
// when `isDesktop`; the two desktop-only actions below call this.
function resolveAbsolutePath(asset) {
  return tauriInvoke("resolve_asset_path", {
    projectId: asset?.projectId,
    relativePath: asset?.file?.path,
  });
}

// Trigger a normal browser download of the asset's served URL. Uses a transient
// `<a download>` click, which is the portable way to name a cross-origin-safe,
// same-origin download without a fetch/blob round-trip. Content-agnostic.
function browserDownload(asset) {
  const href = assetUrl(asset);
  if (!href) {
    throw new Error("This asset has no downloadable URL.");
  }
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = suggestedFilename(asset);
  anchor.rel = "noopener";
  // Some browsers require the anchor to be in the document for the click to count.
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

// Save an asset to a user-chosen location.
//   - Desktop: resolve the absolute path, then open the native save dialog pre-filled
//     with the suggested filename and copy the bytes there. A user-cancel (the command
//     returns null) is handled quietly — no error. Returns the destination path on
//     success, or null on cancel.
//   - Browser/LAN: trigger a normal download of the asset URL. Returns null (there's no
//     destination path to report).
// Works for both image and video assets.
export async function saveAssetAs(asset) {
  if (!asset) {
    throw new Error("An asset is required.");
  }
  if (!isDesktop) {
    browserDownload(asset);
    return null;
  }
  const absolutePath = await resolveAbsolutePath(asset);
  // `save_asset_as` returns the destination path, or null when the user cancels.
  const destination = await tauriInvoke("save_asset_as", {
    sourcePath: absolutePath,
    suggestedFilename: suggestedFilename(asset),
  });
  return destination ?? null;
}

// Reveal an asset in Finder/Explorer with the file selected. Desktop-only — there is
// no browser equivalent, so callers hide/disable this in browser mode. Guarded here
// too: throws a clear error rather than dereferencing the absent Tauri bridge.
export async function revealAsset(asset) {
  if (!isDesktop) {
    throw new Error("Reveal in Finder/Explorer is only available in the desktop app.");
  }
  if (!asset) {
    throw new Error("An asset is required.");
  }
  const absolutePath = await resolveAbsolutePath(asset);
  await tauriInvoke("reveal_in_os", { path: absolutePath });
}
