// User-selectable accent palettes (sc-accent). Each entry maps to a
// `[data-accent="<id>"]` block in styles.css that sets the accent + secondary
// hue; the light/dark ramps in :root / [data-theme="dark"] consume those hues,
// so flipping the attribute recolors the whole app. `swatch` is the dot shown
// in the topbar picker. "teal" is the default brand accent.
export const ACCENTS = [
  { id: "teal", name: "Teal", swatch: "oklch(0.60 0.13 178)" },
  { id: "indigo", name: "Indigo", swatch: "oklch(0.55 0.16 274)" },
  { id: "cobalt", name: "Cobalt", swatch: "oklch(0.55 0.16 252)" },
  { id: "violet", name: "Violet", swatch: "oklch(0.55 0.18 305)" },
  { id: "coral", name: "Coral", swatch: "oklch(0.64 0.16 28)" },
  { id: "amber", name: "Amber", swatch: "oklch(0.72 0.13 80)" },
  { id: "emerald", name: "Emerald", swatch: "oklch(0.58 0.14 152)" },
];

export const DEFAULT_ACCENT = "teal";

const ACCENT_IDS = new Set(ACCENTS.map((accent) => accent.id));

export function isAccentId(value) {
  return typeof value === "string" && ACCENT_IDS.has(value);
}
