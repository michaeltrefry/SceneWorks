import React from "react";
import { API_BASE_URL } from "../api.js";

export function assetUrl(asset) {
  return asset?.url ? API_BASE_URL + asset.url : "";
}

export function assetCanRenderAsImage(asset) {
  return asset?.type === "image" || asset?.file?.mimeType?.startsWith("image/");
}

export function AssetMedia({ asset, className = "" }) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (asset.file?.mimeType?.startsWith("video/")) {
    return <video className={className} controls muted playsInline src={src} />;
  }
  if (assetCanRenderAsImage(asset)) {
    return <img alt="" className={className} src={src} />;
  }
  return <span className={className}>{asset.type}</span>;
}
