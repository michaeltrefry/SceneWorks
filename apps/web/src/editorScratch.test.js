import { describe, expect, it, vi } from "vitest";
import {
  resultAssetsToPurge,
  scratchOpAssetsToPurge,
  createEditorScratchRegistry,
} from "./editorScratch.js";

const scratchAsset = { id: "scratch-1", projectId: "project-1" };
const maskAsset = { id: "mask-1", projectId: "project-1" };
const completedJob = (extra = {}) => ({
  id: "job-1",
  projectId: "project-1",
  status: "completed",
  result: { assets: [{ id: "result-1", projectId: "project-1" }] },
  ...extra,
});

describe("resultAssetsToPurge (sc-8850)", () => {
  it("reads full result.assets objects", () => {
    expect(resultAssetsToPurge(completedJob())).toEqual([{ id: "result-1", projectId: "project-1" }]);
  });

  it("falls back to the job's projectId for bare result.assetIds", () => {
    const job = { id: "j", projectId: "project-9", status: "completed", result: { assetIds: ["a", "b"] } };
    expect(resultAssetsToPurge(job)).toEqual([
      { id: "a", projectId: "project-9" },
      { id: "b", projectId: "project-9" },
    ]);
  });

  it("returns [] for a job with no result assets (e.g. a failed job)", () => {
    expect(resultAssetsToPurge({ id: "j", status: "failed", result: {} })).toEqual([]);
    expect(resultAssetsToPurge(null)).toEqual([]);
  });
});

describe("scratchOpAssetsToPurge (sc-8850)", () => {
  it("combines tracked scratch/mask with the job's result assets, de-duped", () => {
    const entry = { assets: [scratchAsset, maskAsset, scratchAsset] };
    expect(scratchOpAssetsToPurge(entry, completedJob())).toEqual([
      scratchAsset,
      maskAsset,
      { id: "result-1", projectId: "project-1" },
    ]);
  });

  it("purges the scratch even when the job failed (no result)", () => {
    const entry = { assets: [scratchAsset] };
    expect(scratchOpAssetsToPurge(entry, { id: "job-1", status: "failed", result: {} })).toEqual([scratchAsset]);
  });
});

describe("createEditorScratchRegistry survivor sweep (sc-8850)", () => {
  function setup() {
    const purgeAsset = vi.fn().mockResolvedValue(undefined);
    const registry = createEditorScratchRegistry({ purgeAsset });
    return { purgeAsset, registry };
  }

  it("does NOT purge while the editor still claims the in-flight op", () => {
    const { purgeAsset, registry } = setup();
    registry.track("job-1", [scratchAsset]);
    // Editor mounted and actively tracking job-1.
    registry.registerClaim(() => new Set(["job-1"]), () => [completedJob()]);
    // The job terminates, but the mounted editor owns loading the result back first.
    registry.sweep([completedJob()]);
    expect(purgeAsset).not.toHaveBeenCalled();
  });

  it("purges scratch + result when a claimed op is released (editor loaded the result)", () => {
    const { purgeAsset, registry } = setup();
    registry.track("job-1", [scratchAsset]);
    registry.registerClaim(() => new Set(["job-1"]), () => [completedJob()]);
    registry.release("job-1", completedJob());
    expect(purgeAsset).toHaveBeenCalledWith(scratchAsset);
    expect(purgeAsset).toHaveBeenCalledWith({ id: "result-1", projectId: "project-1" });
    expect(registry._size()).toBe(0);
  });

  it("purges an op whose job terminates AFTER the editor unmounts (result lands late)", () => {
    const { purgeAsset, registry } = setup();
    registry.track("job-1", [scratchAsset, maskAsset]);
    let jobs = [{ id: "job-1", status: "running", projectId: "project-1" }]; // still running at unmount
    const unregister = registry.registerClaim(() => new Set(["job-1"]), () => jobs);
    // Editor unmounts mid-job — claim cleared; sweep runs but the job isn't terminal yet.
    unregister();
    expect(purgeAsset).not.toHaveBeenCalled();
    expect(registry._size()).toBe(1);
    // The job completes later; the App-level jobs tick sweeps it — nothing orphaned.
    jobs = [completedJob()];
    registry.sweep(jobs);
    expect(purgeAsset).toHaveBeenCalledWith(scratchAsset);
    expect(purgeAsset).toHaveBeenCalledWith(maskAsset);
    expect(purgeAsset).toHaveBeenCalledWith({ id: "result-1", projectId: "project-1" });
    expect(registry._size()).toBe(0);
  });

  it("purges an op that had ALREADY terminated when the editor unmounts (claim-release sweep)", () => {
    const { purgeAsset, registry } = setup();
    registry.track("job-1", [scratchAsset]);
    const jobs = [completedJob()];
    // Editor claimed the op right up to unmount, so the periodic sweep skipped it while
    // claimed. The unregister sweep must catch it now that the claim is gone.
    const unregister = registry.registerClaim(() => new Set(["job-1"]), () => jobs);
    unregister();
    expect(purgeAsset).toHaveBeenCalledWith(scratchAsset);
    expect(purgeAsset).toHaveBeenCalledWith({ id: "result-1", projectId: "project-1" });
    expect(registry._size()).toBe(0);
  });

  it("purges scratch even for a FAILED op after unmount (no silent orphan)", () => {
    const { purgeAsset, registry } = setup();
    registry.track("job-1", [scratchAsset]);
    const jobs = [{ id: "job-1", status: "failed", projectId: "project-1", result: {} }];
    registry.registerClaim(() => new Set(), () => jobs)(); // register then immediately unregister
    expect(purgeAsset).toHaveBeenCalledWith(scratchAsset);
    expect(registry._size()).toBe(0);
  });

  it("is idempotent — a second sweep after purge does nothing", () => {
    const { purgeAsset, registry } = setup();
    registry.track("job-1", [scratchAsset]);
    registry.sweep([completedJob()]); // no claim registered → purges immediately
    expect(purgeAsset).toHaveBeenCalledTimes(2); // scratch + result
    registry.sweep([completedJob()]);
    expect(purgeAsset).toHaveBeenCalledTimes(2); // entry already gone
  });
});
