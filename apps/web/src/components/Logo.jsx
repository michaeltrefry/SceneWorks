import React, { useId } from "react";

/**
 * SceneWorks mark — "scene cut" (D1). A rounded square split by a diagonal
 * seam: teal triangle over a solid ground. Colors come from CSS variables
 * (--logo-ground / --logo-seam) so it tracks the active theme; teal is fixed
 * brand color and does not follow the user-selectable accent.
 */
export function Logo({ size = 32, title = "SceneWorks", className }) {
  const clipId = useId();
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      role="img"
      aria-label={title}
      className={className}
    >
      <title>{title}</title>
      <defs>
        <clipPath id={clipId}>
          <rect x="10" y="10" width="80" height="80" rx="14" />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <rect x="10" y="10" width="80" height="80" fill="var(--logo-ground)" />
        <polygon points="10,10 90,10 90,90" fill="var(--teal)" />
        <line
          x1="10"
          y1="90"
          x2="90"
          y2="10"
          stroke="var(--logo-seam)"
          strokeWidth="2.5"
          strokeLinecap="square"
        />
      </g>
    </svg>
  );
}

export default Logo;
