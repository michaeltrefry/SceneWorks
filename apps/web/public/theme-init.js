// Apply the persisted theme before first paint to avoid a flash. Kept as an
// external script (not inline) so the served CSP can use a strict script-src 'self'.
try {
  const saved = window.localStorage.getItem("sceneworks-theme");
  if (saved === "dark" || saved === "light") {
    document.documentElement.setAttribute("data-theme", saved);
  }
} catch (_) {
  // ignore (private mode etc.)
}
