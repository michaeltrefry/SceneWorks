export function isUpscaledAsset(asset) {
  return Boolean(asset?.extra?.isUpscaled || asset?.extra?.upscaledFromAssetId);
}

export function upscaledFromAssetId(asset) {
  return asset?.extra?.upscaledFromAssetId ?? (isUpscaledAsset(asset) ? asset?.lineage?.sourceAssetId : null);
}

function attachUpscaleVariants(representative, original, upscaled) {
  if (!original || !upscaled) {
    return representative;
  }
  return {
    ...representative,
    variants: {
      ...(representative.variants ?? {}),
      original,
      upscaled,
    },
  };
}

export function foldUpscaledAssetVariants(assets = []) {
  const byId = new Map(assets.map((asset) => [asset.id, asset]));
  const upscaledByOriginalId = new Map();
  for (const asset of assets) {
    const originalId = upscaledFromAssetId(asset);
    if (originalId && byId.has(originalId) && !upscaledByOriginalId.has(originalId)) {
      upscaledByOriginalId.set(originalId, asset);
    }
  }

  return assets
    .filter((asset) => !upscaledByOriginalId.has(asset.id))
    .map((asset) => {
      const originalId = upscaledFromAssetId(asset);
      if (!originalId) {
        return asset;
      }
      const original = byId.get(originalId);
      return attachUpscaleVariants(asset, original, asset);
    });
}

// The studios' "Recent Assets" / Recent Batches list shows freshly generated
// assets. An upscale shares its generation with the original image, so listing
// both makes every generation look duplicated. Drop the upscaled variant when its
// original is present and keep the original as the visible tile — the fullscreen
// preview still exposes the upscaled image via foldUpscaledAssetVariants and the
// Original/Upscaled toggle. (An upscale whose original is gone stays, so it never
// vanishes entirely.)
export function dropUpscaledVariants(assets = []) {
  const presentIds = new Set(assets.map((asset) => asset.id));
  return assets.filter((asset) => {
    const originalId = upscaledFromAssetId(asset);
    return !(originalId && presentIds.has(originalId));
  });
}

export function findFoldedAssetById(foldedAssets, assetId) {
  return foldedAssets.find(
    (asset) =>
      asset.id === assetId ||
      asset.variants?.original?.id === assetId ||
      asset.variants?.upscaled?.id === assetId,
  );
}
