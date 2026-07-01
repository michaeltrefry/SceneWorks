// Resolve the API base URL:
// - explicit non-empty VITE_API_BASE_URL (Docker/server, separate origins) → use it
// - explicit empty string (desktop embedded build, served by the API itself) →
//   same-origin, so fetches hit the API's actual port with no CORS
// - unset (local Vite dev) → the historical localhost:8000 default
const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
export const API_BASE_URL =
  configuredApiBaseUrl === undefined
    ? "http://localhost:8000"
    : configuredApiBaseUrl === "" && typeof window !== "undefined"
      ? window.location.origin
      : configuredApiBaseUrl;

// True for the DOMException fetch raises when an AbortController aborts a
// request. Callers treat this as "request superseded", not a real error.
export function isAbortError(err) {
  return err?.name === "AbortError";
}

// `options` is forwarded to fetch, so callers can pass an AbortController
// `signal` to cancel a request (e.g. a stale project-scoped load).
export async function apiFetch(path, token, options = {}) {
  const headers = new Headers(options.headers ?? {});
  const isFormData = options.body instanceof FormData;
  if (options.body && !isFormData) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed with ${response.status}`);
  }
  return response.json();
}

export function eventUrl(path, ticket) {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (ticket) {
    url.searchParams.set("ticket", ticket);
  }
  return url.toString();
}

// ---- Media tickets (sc-8810) ---------------------------------------------------
// Element-driven requests (<img src>, <video src>, <a download>) cannot attach the
// X-SceneWorks-Token header, so in remote-auth mode every bare media URL would 401.
// Mirroring the SSE ticket, App.jsx mints a short-lived reusable ticket from
// POST /api/v1/files/ticket (header-authenticated) and stores it here; every media
// URL producer routes through `withMediaTicket` so the ticket rides along as a
// query param that the API honors on the project-files and pose-preview routes.
// Module-level (not React state) because URL producers like assetUrl() are plain
// functions called from render paths and non-component code (browserDownload).
let mediaTicket = "";

export function setMediaTicket(ticket) {
  mediaTicket = typeof ticket === "string" ? ticket : "";
}

export function getMediaTicket() {
  return mediaTicket;
}

// Append the current media ticket to a media URL. No-op while no ticket is set
// (desktop/loopback or auth-off deployments), so local mode is untouched.
export function withMediaTicket(url) {
  if (!url || !mediaTicket) {
    return url;
  }
  return `${url}${url.includes("?") ? "&" : "?"}ticket=${encodeURIComponent(mediaTicket)}`;
}
