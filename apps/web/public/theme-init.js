// Apply the persisted theme + accent before first paint to avoid a flash. Kept
// as an external script (not inline) so the served CSP can use a strict
// script-src 'self'.
try {
  const root = document.documentElement;

  const savedTheme = window.localStorage.getItem("sceneworks-theme");
  if (savedTheme === "dark" || savedTheme === "light") {
    root.setAttribute("data-theme", savedTheme);
  }

  // Keep this list in sync with web/src/accents.js (ACCENTS[].id).
  const ACCENT_IDS = ["teal", "indigo", "cobalt", "violet", "coral", "amber", "emerald"];
  const savedAccent = window.localStorage.getItem("sceneworks-accent");
  if (savedAccent && ACCENT_IDS.indexOf(savedAccent) !== -1) {
    root.setAttribute("data-accent", savedAccent);
  }
} catch (_) {
  // ignore (private mode etc.)
}
