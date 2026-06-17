import React from "react";
import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { LookScene } from "./LookScene.jsx";

// The inner art of a "pick a look" tile: the engine-rendered exemplar once it
// exists, a shimmer while it's rendering, or the SVG placeholder otherwise
// (the placeholder is tinted by the tile's per-look CSS vars).
export function LookTile({ asset, pending = false }) {
  if (asset) {
    return <AssetThumbnail asset={asset} className="sw-scene sw-look-img" />;
  }
  if (pending) {
    return <span className="sw-scene sw-look-pending" aria-hidden="true" />;
  }
  return <LookScene />;
}
