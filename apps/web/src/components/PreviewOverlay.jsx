import React from "react";
import { FullscreenPreview } from "./assetPanels.jsx";
import { usePreviewContext } from "../context/PreviewContext.js";

export function PreviewOverlay() {
  const {
    previewedAsset,
    previewNavigation,
    previewDirectionRef,
    setPreviewAsset,
    closePreview,
    deleteAsset,
    purgeAsset,
    updateAssetStatus,
    sendAssetToImageEdit,
    sendAssetRecipeToImage,
  } = usePreviewContext();

  if (!previewedAsset) {
    return null;
  }

  return (
    <FullscreenPreview
      asset={previewedAsset}
      deleteAsset={async (asset) => {
        // Stay in the preview and advance within the launch collection.
        const { previous, next } = previewNavigation;
        const target = previewDirectionRef.current === "previous" ? previous ?? next : next ?? previous;
        await deleteAsset(asset);
        if (target) {
          setPreviewAsset(target);
        } else {
          closePreview();
        }
      }}
      nextAsset={previewNavigation.next}
      onClose={closePreview}
      onEditImage={sendAssetToImageEdit}
      onPreviewAsset={(asset, direction) => {
        if (direction) {
          previewDirectionRef.current = direction;
        }
        setPreviewAsset(asset);
      }}
      onUseRecipe={sendAssetRecipeToImage}
      previousAsset={previewNavigation.previous}
      purgeAsset={async (asset) => {
        await purgeAsset(asset);
        closePreview();
      }}
      updateAssetStatus={updateAssetStatus}
    />
  );
}
