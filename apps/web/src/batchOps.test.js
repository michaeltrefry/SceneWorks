import { describe, expect, it } from "vitest";
import {
  BATCH_OPS,
  batchOpByKey,
  batchEligibleAssets,
  buildBatchJob,
  batchItemStatus,
  batchItemResultAsset,
  summarizeBatchProgress,
} from "./batchOps.js";

const project = { id: "project_1", name: "My Project" };
const asset = { id: "asset_1", displayName: "shot.png" };

describe("batch ops (sc-6112)", () => {
  it("exposes the three ops + their endpoints", () => {
    expect(BATCH_OPS.map((op) => op.key)).toEqual(["upscale", "detail", "edit"]);
    expect(batchOpByKey("edit").endpoint).toBe("/api/v1/image/jobs");
    expect(batchOpByKey("upscale").endpoint).toBe("/api/v1/jobs");
    expect(batchOpByKey("nope")).toBeNull();
  });

  it("keeps only raster images as eligible (by type or image/* mime), never clips", () => {
    const assets = [
      { id: "a", type: "image" },
      { id: "b", type: "video" },
      { id: "c", type: "upload", file: { mimeType: "image/png" } },
      { id: "d", type: "upload", file: { mimeType: "video/mp4" } },
      { id: "e", type: "render", file: { mimeType: "image/jpeg" } },
    ];
    expect(batchEligibleAssets(assets).map((a) => a.id)).toEqual(["a", "c", "e"]);
    expect(batchEligibleAssets(null)).toEqual([]);
  });

  it("builds an upscale job from an asset id (softness only for engines that take it)", () => {
    const real = buildBatchJob({ op: "upscale", asset, params: { engine: "real-esrgan", factor: 4, softness: 0.5 }, project, requestedGpu: "auto" });
    expect(real.endpoint).toBe("/api/v1/jobs");
    expect(real.body.type).toBe("image_upscale");
    expect(real.body.payload).toEqual({ projectId: "project_1", sourceAssetId: "asset_1", factor: 4, engine: "real-esrgan", displayName: "shot.png" });
    // Real-ESRGAN ignores softness — it must be omitted.
    expect(real.body.payload).not.toHaveProperty("softness");

    const seed = buildBatchJob({ op: "upscale", asset, params: { engine: "seedvr2", factor: 2, softness: 0.3 }, project, requestedGpu: "auto" });
    expect(seed.body.payload.softness).toBe(0.3);
  });

  it("builds a detail job with the recipe advanced params", () => {
    const { endpoint, body } = buildBatchJob({
      op: "detail",
      asset,
      params: { model: "realvisxl", strength: 0.55, cnScale: 0.7 },
      project,
      requestedGpu: "auto",
    });
    expect(endpoint).toBe("/api/v1/jobs");
    expect(body.type).toBe("image_detail");
    expect(body.payload.advanced).toEqual({ strength: 0.55, cnScale: 0.7 });
    expect(body.payload.sourceAssetId).toBe("asset_1");
  });

  it("builds an edit job at the asset's native dims (fitMode crop), and requires dims", () => {
    const { endpoint, body } = buildBatchJob({
      op: "edit",
      asset,
      params: { model: "qwen_image_edit", prompt: "make it night", seed: 7 },
      project,
      requestedGpu: "auto",
      dims: { width: 768, height: 1024 },
    });
    expect(endpoint).toBe("/api/v1/image/jobs");
    expect(body.mode).toBe("edit_image");
    expect(body.sourceAssetId).toBe("asset_1");
    expect(body.prompt).toBe("make it night");
    expect(body.width).toBe(768);
    expect(body.height).toBe(1024);
    expect(body.fitMode).toBe("crop");
    expect(body.seed).toBe(7);
    // No dims → loud failure rather than a malformed job.
    expect(() => buildBatchJob({ op: "edit", asset, params: { model: "m", prompt: "x" }, project, requestedGpu: "auto" })).toThrow();
  });

  it("throws on an unknown op", () => {
    expect(() => buildBatchJob({ op: "frobnicate", asset, project, requestedGpu: "auto" })).toThrow();
  });

  it("maps a job's feed status to a batch item status", () => {
    const jobs = [
      { id: "j1", status: "completed" },
      { id: "j2", status: "running" },
      { id: "j3", status: "failed" },
      { id: "j4", status: "queued" },
    ];
    expect(batchItemStatus("j1", jobs)).toBe("completed");
    expect(batchItemStatus("j2", jobs)).toBe("running");
    expect(batchItemStatus("j3", jobs)).toBe("failed");
    expect(batchItemStatus("j4", jobs)).toBe("queued");
    // Submitted but not yet in the feed → queued; no job id → queued.
    expect(batchItemStatus("missing", jobs)).toBe("queued");
    expect(batchItemStatus(null, jobs)).toBe("queued");
  });

  it("returns a completed item's result asset, else null", () => {
    const jobs = [
      { id: "j1", status: "completed", result: { assets: [{ id: "out_1" }] } },
      { id: "j2", status: "running" },
    ];
    expect(batchItemResultAsset("j1", jobs)).toEqual({ id: "out_1" });
    expect(batchItemResultAsset("j2", jobs)).toBeNull();
    expect(batchItemResultAsset("missing", jobs)).toBeNull();
  });

  it("aggregates fan-out progress (null jobId counts as failed)", () => {
    const items = [
      { asset: { id: "a" }, jobId: "j1" },
      { asset: { id: "b" }, jobId: "j2" },
      { asset: { id: "c" }, jobId: "j3" },
      { asset: { id: "d" }, jobId: null }, // submission failed
    ];
    const jobs = [
      { id: "j1", status: "completed" },
      { id: "j2", status: "running" },
      { id: "j3", status: "failed" },
    ];
    const summary = summarizeBatchProgress(items, jobs);
    expect(summary).toMatchObject({ total: 4, completed: 1, running: 1, failed: 2, done: 3, allDone: false });

    const allTerminal = summarizeBatchProgress(
      [{ asset: { id: "a" }, jobId: "j1" }],
      [{ id: "j1", status: "completed" }],
    );
    expect(allTerminal).toMatchObject({ total: 1, completed: 1, done: 1, allDone: true });
  });
});
