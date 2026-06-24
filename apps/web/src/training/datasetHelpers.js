// Pure dataset helpers for the Training Studio (sc-4199). Extracted verbatim
// from TrainingStudio.jsx: dataset summaries, selection-key derivation, the
// owned-asset/asset-id normalizers, dataset-health math, and the save payload
// builder. No React, no app state — just data shaping over dataset records.

export function imageAssetName(asset) {
  const path = asset?.file?.path ?? asset?.path ?? asset?.displayName ?? asset?.id ?? "asset";
  return String(path).replaceAll("\\", "/").split("/").pop() || "asset";
}

export function datasetItemCount(dataset) {
  const value = Number(dataset.itemCount ?? dataset.items?.length ?? 0);
  return Number.isFinite(value) ? value : 0;
}

export function summarizeDatasets(datasets) {
  return datasets.reduce((summary, dataset) => ({ items: summary.items + datasetItemCount(dataset) }), { items: 0 });
}

export function captionText(item) {
  return String(item?.caption?.text ?? "").trim();
}

export function datasetItemSelectionKey(dataset, item, index = 0) {
  return item?.assetId || `dataset-item:${dataset?.id ?? "draft"}:${item?.id ?? index}`;
}

export function datasetItemProjectPath(dataset, item) {
  const path = String(item?.path ?? "").replaceAll("\\", "/");
  if (!dataset?.id || !path) {
    return "";
  }
  return `training/datasets/${dataset.id}/${path}`;
}

// Caption edit state keyed by selection id (sc-2025): the single source of
// truth for the unified caption cards, seeded from the saved dataset items and
// updated as the user edits or imports captions.
export function captionDraftsFromDataset(dataset) {
  const map = {};
  (dataset?.items ?? []).forEach((item, index) => {
    map[datasetItemSelectionKey(dataset, item, index)] = {
      text: item.caption?.text ?? "",
      source: item.caption?.source ?? "manual",
    };
  });
  return map;
}

export function datasetOwnedAssets(dataset, projectId, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  return (dataset?.items ?? [])
    .map((item, index) => {
      if (item.assetId && catalogIds.has(item.assetId)) {
        return null;
      }
      const path = datasetItemProjectPath(dataset, item);
      if (!path) {
        return null;
      }
      const id = datasetItemSelectionKey(dataset, item, index);
      return {
        id,
        assetId: item.assetId ?? null,
        datasetOwned: true,
        projectId,
        type: "image",
        displayName: item.displayName ?? imageAssetName(item),
        file: {
          path,
          mimeType: `image/${String(path).split(".").pop() || "png"}`,
          width: item.width ?? null,
          height: item.height ?? null,
        },
      };
    })
    .filter(Boolean);
}

export function normalizeDatasetAssetIds(dataset, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  return (dataset?.items ?? [])
    .map((item, index) => {
      if (item.assetId && catalogIds.has(item.assetId)) {
        return item.assetId;
      }
      return datasetItemSelectionKey(dataset, item, index);
    })
    .filter(Boolean);
}

// Map the readiness report's duplicate item ids (server item ids) to the selection keys to drop, and
// return the dataset's selection with those removed (sc-6539 one-tap dedupe). Pure so the mutating
// apply path is testable without rendering the studio: callers pass the freshly-saved dataset (for a
// stable id→key mapping) and its current selection. `removedCount` is the number actually matched —
// 0 when the plan's ids are no longer present, so the caller can skip a no-op write.
export function selectionAfterDuplicateRemoval({ dataset, currentSelection = [], removeIds = [] }) {
  const removeSet = new Set(removeIds);
  const removeKeys = new Set();
  (dataset?.items ?? []).forEach((item, index) => {
    if (removeSet.has(item.id)) {
      removeKeys.add(datasetItemSelectionKey(dataset, item, index));
    }
  });
  return {
    nextSelection: (currentSelection ?? []).filter((key) => !removeKeys.has(key)),
    removedCount: removeKeys.size,
  };
}

export function datasetHealth({ activeDataset, imageAssets, selectedAssetIds }) {
  const assetsById = new Map(imageAssets.map((asset) => [asset.id, asset]));
  const selectedAssets = selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean);
  const missingAssets = selectedAssetIds.filter((id) => !assetsById.has(id)).length;
  const disabledItems = selectedAssets.filter((asset) => asset.status?.rejected || asset.status?.trashed).length + missingAssets;
  const names = selectedAssets.map((asset) => imageAssetName(asset).toLowerCase());
  const duplicateFilenames = names.filter((name, index) => names.indexOf(name) !== index).length;
  const captionsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), captionText(item)]),
  );
  const missingCaptions = selectedAssetIds.filter((id) => !captionsByAssetId.get(id)).length;
  const valid = selectedAssetIds.length > 0 && disabledItems === 0;

  return {
    disabledItems,
    duplicateFilenames,
    itemCount: selectedAssetIds.length,
    missingCaptions,
    valid,
  };
}

export function datasetPayload({ activeDataset, assetsById, associatedCharacterId, captionDraftById = {}, name, selectedAssetIds }) {
  const itemsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), item]),
  );
  return {
    name: name.trim(),
    modality: "image",
    // sc-2022: associate the dataset with a character when one is set (created
    // from a character's images, or images imported from the Character tab).
    ...(associatedCharacterId ? { characterId: associatedCharacterId } : {}),
    items: selectedAssetIds
      .map((selectionId) => {
        const asset = assetsById.get(selectionId);
        if (!asset) {
          return null;
        }
        const previous = itemsByAssetId.get(selectionId);
        const draft = captionDraftById[selectionId];
        let caption;
        if (draft && (String(draft.text ?? "").length || draft.source)) {
          caption = {
            text: draft.text ?? "",
            source: draft.source ?? "manual",
            triggerWords: previous?.caption?.triggerWords ?? [],
          };
        } else if (previous?.caption) {
          caption = {
            text: previous.caption.text ?? "",
            source: previous.caption.source ?? "manual",
            triggerWords: previous.caption.triggerWords ?? [],
          };
        }
        const source = asset.datasetOwned || asset.datasetOnly ? { path: asset.file?.path } : { assetId: asset.id };
        return {
          ...source,
          displayName: asset.displayName ?? imageAssetName(asset),
          caption,
        };
      })
      .filter(Boolean),
  };
}
