import React from "react";

// Static "look exemplar" art: one canonical subject (a cabin at dusk) rendered
// as an inline SVG whose palette is driven by per-look CSS variables (see the
// [data-ui-mode="simple"] .look-* rules in styles.css). This is the Phase-7
// placeholder — TODO(sc-simple-looks): replace with engine-rendered, cached
// thumbnails (one canonical prompt per look) plus a "Refresh looks" re-roll.
export function LookScene({ className = "" }) {
  return (
    <svg
      className={`sw-scene ${className}`.trim()}
      viewBox="0 0 120 80"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <rect x="0" y="0" width="120" height="80" fill="var(--s-sky, #2e3a55)" />
      <circle cx="92" cy="20" r="9" fill="var(--s-moon, #f3ead0)" />
      <path d="M0 80 L34 34 L66 80 Z" fill="var(--s-mtnb, #2b3850)" />
      <path d="M40 80 L82 28 L120 80 Z" fill="var(--s-mtna, #3b4a63)" />
      <rect x="48" y="54" width="22" height="18" fill="var(--s-cabin, #3a2f2a)" />
      <path d="M45 54 L59 44 L73 54 Z" fill="var(--s-roof, #2a221e)" />
      <rect x="55" y="59" width="7" height="7" fill="var(--s-win, #ffd27a)" />
      <rect x="0" y="72" width="120" height="8" fill="var(--s-ground, #cdd6e0)" />
    </svg>
  );
}
