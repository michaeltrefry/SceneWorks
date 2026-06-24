import { describe, expect, it } from "vitest";

import { selectionAfterDuplicateRemoval } from "./datasetHelpers.js";

describe("selectionAfterDuplicateRemoval (sc-6539 one-tap dedupe mapping)", () => {
  // Mix of catalog-backed items (selection key = assetId) and a dataset-owned item (no assetId, so the
  // key is the synthesized `dataset-item:<dsid>:<itemid>` — the case most at risk of a mapping miss).
  const dataset = {
    id: "ds1",
    items: [
      { id: "item_0001", assetId: "asset-a" },
      { id: "item_0002", assetId: "asset-b" },
      { id: "item_0003" },
    ],
  };
  const currentSelection = ["asset-a", "asset-b", "dataset-item:ds1:item_0003"];

  it("drops a catalog-backed duplicate by its asset-id key, keeping the rest", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_0002"],
    });
    expect(removedCount).toBe(1);
    expect(nextSelection).toEqual(["asset-a", "dataset-item:ds1:item_0003"]);
  });

  it("drops a dataset-owned (non-catalog) duplicate by its synthesized selection key", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_0003"],
    });
    expect(removedCount).toBe(1);
    expect(nextSelection).toEqual(["asset-a", "asset-b"]);
  });

  it("removes every planned duplicate at once", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_0001", "item_0002"],
    });
    expect(removedCount).toBe(2);
    expect(nextSelection).toEqual(["dataset-item:ds1:item_0003"]);
  });

  it("is a no-op when the planned ids are no longer in the dataset (stale report)", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_9999"],
    });
    expect(removedCount).toBe(0);
    expect(nextSelection).toEqual(currentSelection);
  });

  it("handles empty / missing inputs without throwing", () => {
    expect(selectionAfterDuplicateRemoval({})).toEqual({ nextSelection: [], removedCount: 0 });
    expect(
      selectionAfterDuplicateRemoval({ dataset, currentSelection, removeIds: [] }),
    ).toEqual({ nextSelection: currentSelection, removedCount: 0 });
  });
});
