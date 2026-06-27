import React, { useContext, useState } from "react";
import { apiFetch } from "./api.js";
import { batchEligibleAssets, batchItemStatus, buildBatchJob, summarizeBatchProgress } from "./batchOps.js";
import { assetSupportsCharacterLink } from "./components/assetPanels.jsx";
import { assetUrl } from "./components/assetMedia.jsx";
import { BatchOperationsPanel } from "./components/BatchOperationsPanel.jsx";
import { AppContext } from "./context/AppContext.js";
import { detailCapableModels, editCapableModels, UPSCALE_ENGINES } from "./imageJobs.js";
import { DEFAULT_MAC_CAPABILITIES, macUpscaleEngineBlocked } from "./macGating.js";

// Sentinel "Move" target (sc-8341): selecting it promotes the assets into the Main Asset
// Library (a true move) instead of linking them to a character. Namespaced so it can't
// collide with a real character id.
export const LIBRARY_MOVE_TARGET = "__sceneworks_library__";

// Shared multi-asset batch selection (sc-6112) — selection state, the upscale/detail/edit
// fan-out, and the bulk Discard / Move-to-character actions. Lifted out of LibraryScreen so
// the Assets page and the Character Assets page drive the identical toolbar from one source.
// The hook owns the selection; callers wire `selectedAssetIds`/`toggleSelect` into their grid
// and render <AssetSelectionBar batch={…}/> + <AssetBatchModal batch={…}/>.
export function useAssetBatch() {
  // Read the app context optionally: the toolbar should stay inert (not crash) when a host
  // component is rendered in isolation without an <AppContext.Provider> (e.g. unit tests).
  const {
    activeProject,
    assets = [],
    jobs = [],
    imageModels = [],
    characters = [],
    deleteAsset,
    addCharacterReference,
    moveAssetToLibrary,
    token = "",
    requestedGpu = "auto",
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useContext(AppContext) ?? {};

  const [selectedAssetIds, setSelectedAssetIds] = useState(() => new Set());
  const [batchOpen, setBatchOpen] = useState(false);
  // While/after a batch runs: { op, items: [{ asset, jobId }], submitting }.
  const [batch, setBatch] = useState(null);
  // Bulk Discard / Move-to-character on the current selection. `bulkAction` gates the
  // buttons while a fan-out is in flight; `moveOpen` reveals the inline character picker.
  const [bulkAction, setBulkAction] = useState(null);
  const [moveOpen, setMoveOpen] = useState(false);
  const [moveCharacterId, setMoveCharacterId] = useState("");

  // The current multi-selection, narrowed to the raster images a batch op can run on,
  // and the upscale engines this platform actually supports.
  const selectedAssetList = assets.filter((asset) => selectedAssetIds.has(asset.id));
  const eligibleSelected = batchEligibleAssets(selectedAssetList);
  const availableUpscaleEngines = UPSCALE_ENGINES.filter((engine) => !macUpscaleEngineBlocked(macCapabilities, engine.key));
  const editModels = editCapableModels(imageModels);
  const detailModels = detailCapableModels(imageModels);
  // Move targets the same "Character Assets" link the per-asset panel uses (role "asset",
  // unapproved) — NOT the character's reference images — so only link-capable media counts.
  const availableCharacters = characters.filter((character) => !character?.archived);
  const movableSelected = selectedAssetList.filter(assetSupportsCharacterLink);

  // Per-item + aggregate progress for an in-flight/just-finished batch, read off the jobs feed.
  const batchItems = batch
    ? batch.items.map((item) => ({ asset: item.asset, status: batchItemStatus(item.jobId, jobs) }))
    : null;
  const batchProgress = batch ? summarizeBatchProgress(batch.items, jobs) : null;

  const toggleSelect = (id) =>
    setSelectedAssetIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const clearSelection = () => {
    setSelectedAssetIds(new Set());
    setMoveOpen(false);
  };

  // Decode an asset's native pixel size (needed for an edit job — the worker fits the
  // source to width×height). Resolves null on a load failure so that item fails alone.
  function loadImageDims(asset) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve(null);
      img.src = assetUrl(asset);
    });
  }

  // Fan out one job per selected image (NOT one mega-job): each posts independently so
  // the worker processes them serially with its between-item cache release (sc-5567).
  async function runBatch(op, params) {
    if (!activeProject || !eligibleSelected.length) return;
    const targets = eligibleSelected;
    setBatch({ op, submitting: true, items: targets.map((asset) => ({ asset, jobId: null })) });
    const items = [];
    for (const asset of targets) {
      try {
        let dims = null;
        if (op === "edit") {
          dims = await loadImageDims(asset);
          if (!dims) {
            items.push({ asset, jobId: null });
            continue;
          }
        }
        const { endpoint, body } = buildBatchJob({ op, asset, params, project: activeProject, requestedGpu, dims });
        const job = await apiFetch(endpoint, token, { method: "POST", body: JSON.stringify(body) });
        items.push({ asset, jobId: job?.id ?? null });
      } catch {
        items.push({ asset, jobId: null });
      }
    }
    setBatch({ op, submitting: false, items });
  }

  function closeBatch() {
    setBatchOpen(false);
    // Closing after a run clears the spent selection + progress; cancelling the form keeps it.
    if (batch) {
      setBatch(null);
      clearSelection();
    }
  }

  // Send every selected asset to the Trash (reversible — the backend just flags `trashed`).
  async function discardSelected() {
    if (!selectedAssetList.length || bulkAction) return;
    setBulkAction("discard");
    try {
      for (const asset of selectedAssetList) {
        await deleteAsset?.(asset);
      }
      clearSelection();
    } finally {
      setBulkAction(null);
    }
  }

  // Fan out the move across every movable selection. The target is either the Main Asset
  // Library (a true move — sc-8341) or a character (the per-asset link AssetDetail uses:
  // role "asset", unapproved, library-member note).
  async function moveSelectedToCharacter() {
    if (!moveCharacterId || !movableSelected.length || bulkAction) return;
    const toLibrary = moveCharacterId === LIBRARY_MOVE_TARGET;
    if (toLibrary ? !moveAssetToLibrary : !addCharacterReference) return;
    setBulkAction("move");
    try {
      for (const asset of movableSelected) {
        try {
          if (toLibrary) {
            await moveAssetToLibrary(asset);
          } else {
            await addCharacterReference(moveCharacterId, {
              assetId: asset.id,
              approved: false,
              role: "asset",
              notes: "Added from Asset Library.",
            });
          }
        } catch {
          // One asset failing (e.g. already linked) shouldn't abort the rest.
        }
      }
      clearSelection();
    } finally {
      setBulkAction(null);
    }
  }

  return {
    selectedAssetIds,
    toggleSelect,
    clearSelection,
    selectedAssetList,
    eligibleSelected,
    movableSelected,
    availableCharacters,
    batchOpen,
    setBatchOpen,
    batch,
    batchItems,
    batchProgress,
    runBatch,
    closeBatch,
    bulkAction,
    discardSelected,
    moveSelectedToCharacter,
    moveOpen,
    setMoveOpen,
    moveCharacterId,
    setMoveCharacterId,
    editModels,
    detailModels,
    availableUpscaleEngines,
  };
}

// The selection toolbar: appears once anything is selected. `showDiscard` lets a host hide
// the trash action where it doesn't apply (e.g. a Trashcan view, where items are already
// discarded). `allowLibraryTarget` adds "Assets Library" as a Move destination (sc-8341) —
// used on the Character Assets page to promote media back into the Main Library.
export function AssetSelectionBar({ batch, showDiscard = true, allowLibraryTarget = false }) {
  const {
    selectedAssetIds,
    eligibleSelected,
    movableSelected,
    availableCharacters,
    setBatchOpen,
    bulkAction,
    discardSelected,
    moveOpen,
    setMoveOpen,
    moveCharacterId,
    setMoveCharacterId,
    moveSelectedToCharacter,
    clearSelection,
  } = batch;

  if (selectedAssetIds.size === 0) return null;

  // Move destinations: the Main Library (optional) followed by every non-archived character.
  const moveTargets = [
    ...(allowLibraryTarget ? [{ id: LIBRARY_MOVE_TARGET, name: "Assets Library" }] : []),
    ...availableCharacters,
  ];
  const toLibrary = moveCharacterId === LIBRARY_MOVE_TARGET;

  return (
    <div className="batch-selection-bar">
      <span>
        {selectedAssetIds.size} selected
        {eligibleSelected.length !== selectedAssetIds.size
          ? ` · ${eligibleSelected.length} image${eligibleSelected.length === 1 ? "" : "s"}`
          : ""}
      </span>
      <button className="primary" disabled={!eligibleSelected.length} onClick={() => setBatchOpen(true)} type="button">
        Batch…
      </button>
      {showDiscard ? (
        <button className="danger-action" disabled={Boolean(bulkAction)} onClick={discardSelected} type="button">
          {bulkAction === "discard" ? "Discarding…" : "Discard"}
        </button>
      ) : null}
      {moveTargets.length ? (
        <button
          disabled={!movableSelected.length || Boolean(bulkAction)}
          onClick={() =>
            setMoveOpen((open) => {
              const next = !open;
              if (next && !moveCharacterId) {
                setMoveCharacterId(moveTargets[0]?.id ?? "");
              }
              return next;
            })
          }
          title={movableSelected.length ? undefined : "No movable media selected"}
          type="button"
        >
          Move
        </button>
      ) : null}
      <button onClick={clearSelection} type="button">
        Clear
      </button>
      {moveOpen && moveTargets.length ? (
        <div className="batch-move-picker">
          <select
            aria-label="Move target"
            onChange={(event) => setMoveCharacterId(event.target.value)}
            value={moveCharacterId}
          >
            {moveTargets.map((target) => (
              <option key={target.id} value={target.id}>
                {target.name}
              </option>
            ))}
          </select>
          <button
            className="primary"
            disabled={!moveCharacterId || !movableSelected.length || Boolean(bulkAction)}
            onClick={moveSelectedToCharacter}
            type="button"
          >
            {bulkAction === "move"
              ? "Moving…"
              : `Move ${movableSelected.length} to ${toLibrary ? "library" : "assets"}`}
          </button>
          <button onClick={() => setMoveOpen(false)} type="button">
            Cancel
          </button>
        </div>
      ) : null}
    </div>
  );
}

// The batch-operations modal (upscale / detail / edit). Renders only while open.
export function AssetBatchModal({ batch }) {
  if (!batch.batchOpen) return null;
  return (
    <BatchOperationsPanel
      assets={batch.eligibleSelected}
      editModels={batch.editModels}
      detailModels={batch.detailModels}
      upscaleEngines={batch.availableUpscaleEngines}
      busy={Boolean(batch.batch?.submitting)}
      items={batch.batchItems}
      progress={batch.batchProgress}
      onRun={batch.runBatch}
      onClose={batch.closeBatch}
    />
  );
}
