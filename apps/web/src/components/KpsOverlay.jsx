import React from "react";

// Renders a face-angle preset: the 5 normalized landmarks (green-dot style) over the preset's
// source image, or — for built-ins, which have no captured photo — over a neutral canvas
// (epic 4422, sc-4435). kps are normalized to a SQUARE letterbox `[0,1]`, so the source image
// is drawn with preserveAspectRatio "xMidYMid meet" (centered letterbox) and the dots overlay
// in the same square space, keeping image and landmarks aligned for any source aspect ratio.
//
// `kps`: array of [x, y] in [0,1] (order [left_eye, right_eye, nose, mouth_left, mouth_right]).
// `imageUrl`: optional source image URL; absent → neutral canvas (built-in presets).
// `label`: accessible description for the figure.

const POINT_LABELS = ["left eye", "right eye", "nose", "mouth left", "mouth right"];

export function KpsOverlay({ kps = [], imageUrl, label = "face landmark preset" }) {
  const points = Array.isArray(kps)
    ? kps
        .map((point) => (Array.isArray(point) ? [Number(point[0]), Number(point[1])] : null))
        .filter((point) => point && Number.isFinite(point[0]) && Number.isFinite(point[1]))
    : [];

  return (
    <svg
      className="kps-overlay"
      viewBox="0 0 1 1"
      role="img"
      aria-label={label}
      preserveAspectRatio="xMidYMid meet"
    >
      {imageUrl ? (
        <image
          href={imageUrl}
          x="0"
          y="0"
          width="1"
          height="1"
          preserveAspectRatio="xMidYMid meet"
        />
      ) : (
        <rect x="0" y="0" width="1" height="1" className="kps-overlay-canvas" />
      )}
      {points.map(([x, y], index) => (
        <g key={index}>
          <circle cx={x} cy={y} r="0.03" className="kps-overlay-halo" />
          <circle cx={x} cy={y} r="0.018" className="kps-overlay-dot">
            <title>{POINT_LABELS[index] ?? `point ${index + 1}`}</title>
          </circle>
        </g>
      ))}
    </svg>
  );
}
