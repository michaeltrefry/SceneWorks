// Batch operations across multiple assets (sc-6112, Workstream F of epic 6087).
// Pure orchestration: turn a multi-asset selection + one op (upscale / detail / edit)
// + shared params into one job body PER asset, and summarize the fan-out's progress
// from the global jobs feed. React/DOM-free (reuses the konva-free imageJobs.js
// builders) so the fan-out math is unit-tested in isolation and the Library bundle
// never pulls in the editor / react-konva.

import { terminalStatuses } from "./constants.js";
import { buildDetailJobBody, buildEditJobBody, buildUpscaleJobBody } from "./imageJobs.js";

// The three batch-capable ops + the endpoint each posts to (mirrors the editor:
// upscale/detail are generic jobs, edit is the image-jobs route). `needsPrompt`
// flags the op whose shared params include a text prompt.
export const BATCH_OPS = [
  { key: "upscale", label: "Upscale", endpoint: "/api/v1/jobs", needsPrompt: false },
  { key: "detail", label: "Detail enhance", endpoint: "/api/v1/jobs", needsPrompt: false },
  { key: "edit", label: "AI edit", endpoint: "/api/v1/image/jobs", needsPrompt: true },
];

export function batchOpByKey(key) {
  return BATCH_OPS.find((op) => op.key === key) ?? null;
}

// Assets a raster op can run on — images (by type or image/* mime), never clips.
export function batchEligibleAssets(assets) {
  return (assets ?? []).filter((asset) => {
    if (asset?.type === "video") return false;
    const mime = asset?.file?.mimeType ?? "";
    return asset?.type === "image" || mime.startsWith("image/");
  });
}

// Build the `{ endpoint, body }` for one asset under the chosen op + shared params.
// `dims` (the asset's native width/height) is REQUIRED for edit — the worker fits the
// source to width×height — and ignored by upscale/detail. Throws on an unknown op or
// missing edit dims so a bad fan-out fails loudly rather than posting a malformed job.
export function buildBatchJob({ op, asset, params = {}, project, requestedGpu, dims = null }) {
  const sourceAssetId = asset.id;
  const displayName = asset.displayName ?? asset.id;
  if (op === "upscale") {
    return {
      endpoint: "/api/v1/jobs",
      body: buildUpscaleJobBody({
        project,
        requestedGpu,
        sourceAssetId,
        factor: params.factor,
        engine: params.engine,
        displayName,
        softness: params.softness,
      }),
    };
  }
  if (op === "detail") {
    return {
      endpoint: "/api/v1/jobs",
      body: buildDetailJobBody({
        project,
        requestedGpu,
        sourceAssetId,
        model: params.model,
        strength: params.strength,
        cnScale: params.cnScale,
        displayName,
      }),
    };
  }
  if (op === "edit") {
    if (!dims?.width || !dims?.height) {
      throw new Error("Batch edit needs the source image dimensions.");
    }
    return {
      endpoint: "/api/v1/image/jobs",
      body: buildEditJobBody({
        project,
        requestedGpu,
        sourceAssetId,
        model: params.model,
        prompt: params.prompt,
        seed: params.seed,
        width: dims.width,
        height: dims.height,
        // Same-size edit (no canvas extend) — each image is edited at its native size.
        fitMode: "crop",
      }),
    };
  }
  throw new Error(`Unknown batch op: ${op}`);
}

// One batch item's status, derived from the global jobs feed: "queued" until the job
// surfaces (or while queued), "running" mid-flight, "completed"/"failed" once terminal.
export function batchItemStatus(jobId, jobs) {
  if (!jobId) return "queued";
  const job = (jobs ?? []).find((item) => item.id === jobId);
  if (!job) return "queued"; // submitted; not yet in the feed
  if (job.status === "completed") return "completed";
  if (terminalStatuses.has(job.status)) return "failed"; // failed / canceled / interrupted
  if (job.status === "running") return "running";
  return "queued";
}

// The completed result asset for a batch item (the worker returns it on `result.assets[0]`),
// or null while pending / on failure.
export function batchItemResultAsset(jobId, jobs) {
  const job = (jobs ?? []).find((item) => item.id === jobId);
  if (!job || job.status !== "completed") return null;
  return job.result?.assets?.[0] ?? null;
}

// Aggregate a fan-out's progress. `items` = [{ assetId, jobId }]; returns the per-status
// tallies plus `done` (terminal) and `allDone`. Items with no jobId (submission failed)
// count as failed so the aggregate never claims more progress than really happened.
export function summarizeBatchProgress(items, jobs) {
  const summary = { total: items?.length ?? 0, queued: 0, running: 0, completed: 0, failed: 0 };
  for (const item of items ?? []) {
    if (!item.jobId) {
      summary.failed += 1;
      continue;
    }
    summary[batchItemStatus(item.jobId, jobs)] += 1;
  }
  summary.done = summary.completed + summary.failed;
  summary.allDone = summary.total > 0 && summary.done === summary.total;
  return summary;
}
