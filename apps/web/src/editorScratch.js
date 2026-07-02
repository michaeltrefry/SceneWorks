// Survivor-side registry + helpers for purging Image-Editor AI-op scratch/result assets
// when the editor is no longer around to do it itself (sc-8850).
//
// The Image Editor stages the working bitmap as an ephemeral "scratch" asset, runs a
// worker job against it, loads the result back, then purges scratch + result so only
// Save (sc-2434) ever lands a Library asset. That purge used to live entirely in an
// in-component effect — which never fires if the user navigates away mid-job and the
// editor unmounts (starting an AI op doesn't set `dirty`, so the leave guard wasn't even
// registered). The result was silently lost AND the scratch upload + result asset
// permanently orphaned in the Library.
//
// The fix keeps this registry at the App level (it outlives the editor). While the editor
// is mounted it "claims" the jobIds it is tracking and owns loading the result back into
// the canvas, then releases the op (which purges). If the editor unmounts first, it never
// releases; the survivor sweep then purges the tracked scratch/mask + result assets the
// moment the job reaches a terminal status (or right after the claim is dropped, for an op
// that had already terminated). Either way nothing is orphaned.
//
// The registry is a plain factory (no React) so the full survivor behaviour — including
// "purges after the editor unmounts" — is unit-testable without mounting App.

import { terminalStatuses } from "./jobTypes.js";

// A terminal job's result assets as `{ id, projectId }` purge descriptors. Handles the
// two result shapes the worker emits (`result.assets` full objects, `result.assetIds`
// bare ids) and falls back to the job's projectId when a bare id carries none.
export function resultAssetsToPurge(job) {
  if (!job) return [];
  const projectFallback = job.projectId ?? null;
  const assets = job.result?.assets;
  if (Array.isArray(assets) && assets.length) {
    return assets
      .filter((asset) => asset && asset.id)
      .map((asset) => ({ id: asset.id, projectId: asset.projectId ?? projectFallback }));
  }
  const assetIds = job.result?.assetIds;
  if (Array.isArray(assetIds) && assetIds.length) {
    return assetIds.filter(Boolean).map((id) => ({ id, projectId: projectFallback }));
  }
  return [];
}

// The full set of assets to purge for a completed editor scratch op: the tracked scratch
// source + mask (from the registry entry) plus the job's result assets. De-duped by asset
// id. `entry.assets` is the array of scratch/mask asset objects the editor staged.
export function scratchOpAssetsToPurge(entry, job) {
  const out = [];
  const seen = new Set();
  const push = (asset) => {
    if (!asset || !asset.id || seen.has(asset.id)) return;
    seen.add(asset.id);
    out.push(asset);
  };
  for (const asset of entry?.assets ?? []) push(asset);
  for (const asset of resultAssetsToPurge(job)) push(asset);
  return out;
}

// The App-level scratch-op registry. `purgeAsset(asset)` performs the actual DELETE and
// may return a promise; failures are swallowed (best-effort cleanup). `isTerminal` is
// injectable for tests but defaults to the shared terminal-status set.
export function createEditorScratchRegistry({ purgeAsset, isTerminal = (status) => terminalStatuses.has(status) }) {
  const ops = new Map(); // jobId -> { assets }
  let claimGetter = null; // () => Set<jobId> the mounted editor is actively tracking

  function purgeOp(jobId, job) {
    const entry = ops.get(jobId);
    if (!entry) return; // idempotent: whoever purges first wins
    ops.delete(jobId);
    for (const asset of scratchOpAssetsToPurge(entry, job)) {
      if (asset?.id && asset?.projectId) {
        try {
          purgeAsset(asset)?.catch?.(() => {});
        } catch {
          // best-effort cleanup
        }
      }
    }
  }

  return {
    // Editor starts an AI op: remember its scratch/mask assets so they can be purged even
    // if the editor unmounts before its own watcher runs.
    track(jobId, assets) {
      if (!jobId) return;
      ops.set(jobId, { assets: (assets ?? []).filter(Boolean) });
    },
    // Editor's in-component watcher loaded the result back into the canvas — hand the
    // purge (scratch + mask + result) to the registry so there's a single purge path.
    release(jobId, job) {
      purgeOp(jobId, job);
    },
    // The mounted editor registers a getter for the jobIds it is actively tracking. On
    // unmount the returned unregister clears the claim AND sweeps once, so an op that had
    // already terminated while claimed (and was therefore skipped) is not orphaned.
    registerClaim(getClaimedIds, getJobs) {
      claimGetter = getClaimedIds;
      return () => {
        if (claimGetter === getClaimedIds) claimGetter = null;
        this.sweep(getJobs?.() ?? []);
      };
    },
    // Purge every tracked op whose job is terminal and which the editor is no longer
    // claiming (unmounted or already released). Runs on each jobs tick and post-unmount.
    sweep(jobsList) {
      if (ops.size === 0) return;
      const claimed = claimGetter ? claimGetter() : null;
      for (const jobId of [...ops.keys()]) {
        if (claimed && claimed.has(jobId)) continue; // editor still owns this op
        const job = jobsList?.find((item) => item.id === jobId);
        if (!job || !isTerminal(job.status)) continue;
        purgeOp(jobId, job);
      }
    },
    // Test/introspection helper.
    _size() {
      return ops.size;
    },
  };
}
