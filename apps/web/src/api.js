export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

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
