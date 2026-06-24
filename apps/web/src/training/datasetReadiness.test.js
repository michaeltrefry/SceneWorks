import { describe, expect, it } from "vitest";

import {
  aestheticScore,
  alignmentPercent,
  badgeForSeverity,
  captionAlignmentFlaggedItemIds,
  captionHash,
  datasetDoctorSummary,
  dismissedChecks,
  diversityPercent,
  duplicateRemovalItemIds,
  flagCountsByCheck,
  flagMetric,
  flagReason,
  gateMeta,
  itemBadge,
  nextDismissedChecks,
  primaryReason,
  readinessBySelectionKey,
  readinessQueryParams,
  technicalPercent,
  trainBlockedByReadiness,
  worstFlag,
} from "./datasetReadiness.js";

function report(overrides = {}) {
  return {
    gate: "needs_attention",
    subScores: { technical: 0.8 },
    counts: { info: 0, warn: 2, fatal: 0 },
    itemCount: 5,
    items: [],
    datasetFlags: [],
    ...overrides,
  };
}

describe("captionHash", () => {
  it("matches lowercase SHA-256 hex for the exact caption text", async () => {
    expect(await captionHash("tiny test image")).toBe(
      "d93dcd03ca0197eb0ae2041bfc1b3c3b78399bb19e904c901713bcc85cc39cf7",
    );
  });
});

describe("badges", () => {
  it("maps worst severity to the three states, treating info/clean as good", () => {
    expect(badgeForSeverity("fatal").tone).toBe("fatal");
    expect(badgeForSeverity("warn").tone).toBe("warn");
    expect(badgeForSeverity("info").tone).toBe("good");
    expect(badgeForSeverity(undefined).tone).toBe("good");
  });

  it("renders an uncovered or loading item as pending, never good", () => {
    expect(itemBadge(undefined).tone).toBe("pending");
    expect(itemBadge(null).tone).toBe("pending");
    expect(itemBadge({ severity: undefined }, { loading: true }).tone).toBe("pending");
    // A covered, clean item is genuinely good.
    expect(itemBadge({ severity: undefined }).tone).toBe("good");
  });
});

describe("flagReason", () => {
  it("has plain-language copy for every Tier-0 check", () => {
    for (const check of [
      "resolution",
      "crop_loss",
      "blur",
      "exposure",
      "exact_duplicate",
      "near_duplicate",
      "count",
      "decode",
    ]) {
      const text = flagReason({ check });
      expect(text).toBeTruthy();
      expect(text).not.toBe("Flagged for review");
    }
  });

  it("has plain-language copy for the Tier-1 embedding checks (sc-6535)", () => {
    for (const check of ["near_duplicate_embedding", "low_diversity", "caption_alignment"]) {
      const text = flagReason({ check });
      expect(text).toBeTruthy();
      expect(text).not.toBe("Flagged for review");
    }
  });

  it("degrades an unknown check to generic copy, never undefined", () => {
    expect(flagReason({ check: "some_future_check" })).toBe("Flagged for review");
    expect(flagReason(null)).toBe("");
  });

  it("never implies Tier-1 identity ('different person') which does not exist yet", () => {
    const all = ["resolution", "crop_loss", "blur", "exposure", "exact_duplicate", "near_duplicate", "count", "decode"]
      .map((check) => flagReason({ check }))
      .join(" ")
      .toLowerCase();
    expect(all).not.toContain("person");
  });
});

describe("worstFlag / primaryReason", () => {
  it("picks the most severe flag for the badge reason", () => {
    const entry = {
      flags: [
        { check: "blur", severity: "warn" },
        { check: "decode", severity: "fatal" },
        { check: "resolution", severity: "info" },
      ],
    };
    expect(worstFlag(entry).check).toBe("decode");
    expect(primaryReason(entry)).toBe("This image couldn't be read");
  });

  it("returns empty for a clean item", () => {
    expect(worstFlag({ flags: [] })).toBeNull();
    expect(primaryReason({ flags: [] })).toBe("");
  });
});

describe("flagCountsByCheck", () => {
  it("counts each item once per check", () => {
    const counts = flagCountsByCheck(
      report({
        items: [
          { itemId: "a", flags: [{ check: "blur" }, { check: "blur" }] },
          { itemId: "b", flags: [{ check: "blur" }, { check: "near_duplicate" }] },
          { itemId: "c", flags: [] },
        ],
      }),
    );
    expect(counts.blur).toBe(2);
    expect(counts.near_duplicate).toBe(1);
  });
});

describe("datasetDoctorSummary", () => {
  it("leads with the photo count and lists warnings in plain language", () => {
    const text = datasetDoctorSummary(
      report({
        itemCount: 18,
        items: [
          { itemId: "a", flags: [{ check: "blur", severity: "warn" }] },
          { itemId: "b", flags: [{ check: "blur", severity: "warn" }] },
          { itemId: "c", flags: [{ check: "near_duplicate", severity: "warn" }] },
          { itemId: "d", flags: [{ check: "near_duplicate", severity: "warn" }] },
          { itemId: "e", flags: [{ check: "near_duplicate", severity: "warn" }] },
        ],
      }),
    );
    expect(text).toContain("18 photos.");
    expect(text).toContain("2 look blurry");
    expect(text).toContain("3 are near-duplicates");
    expect(text).toContain("stronger");
  });

  it("phrases embedding near-duplicates and mentions low set diversity (sc-6535)", () => {
    const text = datasetDoctorSummary(
      report({
        gate: "needs_attention",
        itemCount: 6,
        items: [
          { itemId: "a", flags: [{ check: "near_duplicate_embedding", severity: "warn" }] },
          { itemId: "b", flags: [{ check: "near_duplicate_embedding", severity: "warn" }] },
        ],
        datasetFlags: [{ check: "low_diversity", severity: "warn" }],
      }),
    );
    expect(text).toContain("2 look very similar to others");
    expect(text).toContain("more variety");
  });

  it("phrases caption alignment warnings as re-captioning advice (sc-6537)", () => {
    const text = datasetDoctorSummary(
      report({
        gate: "needs_attention",
        itemCount: 6,
        items: [
          { itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] },
          { itemId: "b", flags: [{ check: "caption_alignment", severity: "warn" }] },
        ],
      }),
    );
    expect(text).toContain("2 captions may not match their images");
    expect(text).toContain("Re-captioning");
  });

  it("uses singular copy for one caption alignment warning (sc-6537)", () => {
    const text = datasetDoctorSummary(
      report({
        gate: "needs_attention",
        itemCount: 6,
        items: [{ itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] }],
      }),
    );
    expect(text).toContain("1 caption may not match its image");
    expect(text).toContain("Re-captioning this can improve training");
  });

  it("surfaces the style-only aesthetic advisory as a non-blocking heads-up (sc-6537)", () => {
    const text = datasetDoctorSummary(
      report({
        gate: "needs_attention",
        itemCount: 12,
        items: [],
        datasetFlags: [{ check: "low_aesthetic", severity: "warn" }],
      }),
    );
    expect(text).toContain("heads-up");
    expect(text).not.toContain("ready to train");
  });

  it("says a ready set is ready", () => {
    const text = datasetDoctorSummary(report({ gate: "ready", itemCount: 20, items: [] }));
    expect(text).toBe("20 photos. This set looks ready to train.");
  });

  it("leads with the shortfall when blocked on too few photos", () => {
    const text = datasetDoctorSummary(
      report({
        gate: "blocked",
        itemCount: 3,
        items: [],
        datasetFlags: [{ check: "count", severity: "fatal", value: 3, threshold: 15 }],
      }),
    );
    expect(text).toBe("3 photos. You need at least 15 to train — add 12 more.");
  });

  it("nudges toward the recommended count when count is a soft warning", () => {
    const text = datasetDoctorSummary(
      report({
        gate: "needs_attention",
        itemCount: 8,
        items: [],
        datasetFlags: [{ check: "count", severity: "warn", value: 8, threshold: 15 }],
      }),
    );
    expect(text).toContain("8 photos.");
    expect(text).toContain("Aim for 15+ photos");
  });

  it("handles the singular case", () => {
    const text = datasetDoctorSummary(report({ gate: "ready", itemCount: 1, items: [] }));
    expect(text.startsWith("1 photo.")).toBe(true);
  });

  it("returns empty when there is no report", () => {
    expect(datasetDoctorSummary(null)).toBe("");
  });
});

describe("gate + sub-scores", () => {
  it("labels each gate with a tone", () => {
    expect(gateMeta("ready").tone).toBe("good");
    expect(gateMeta("needs_attention").tone).toBe("warn");
    expect(gateMeta("blocked").tone).toBe("fatal");
    expect(gateMeta(undefined).tone).toBe("pending");
  });

  it("renders the technical sub-score as a percentage", () => {
    expect(technicalPercent(report({ subScores: { technical: 0.8 } }))).toBe(80);
    expect(technicalPercent(report({ subScores: {} }))).toBeNull();
    expect(technicalPercent(null)).toBeNull();
  });

  it("renders the diversity sub-score as a percentage, absent until the embedding job runs (sc-6535)", () => {
    expect(diversityPercent(report({ subScores: { technical: 0.8, diversity: 0.42 } }))).toBe(42);
    expect(diversityPercent(report({ subScores: { technical: 0.8 } }))).toBeNull();
    expect(diversityPercent(null)).toBeNull();
  });

  it("renders the aesthetic sub-score (style-only) rounded to one decimal (sc-6537)", () => {
    expect(aestheticScore(report({ subScores: { technical: 0.8, aesthetic: 5.34 } }))).toBe(5.3);
    expect(aestheticScore(report({ subScores: { technical: 0.8 } }))).toBeNull();
    expect(aestheticScore(null)).toBeNull();
  });

  it("rescales the alignment sub-score from raw CLIP cosine to a meter % (sc-6537)", () => {
    // Raw image↔text cosines are low/compressed; the meter rescales [0.05, 0.18] → [0, 100], so a
    // good caption (~0.15) reads ~77% instead of an alarming 15%, and a mismatch (~0.05) reads 0%.
    expect(alignmentPercent(report({ subScores: { technical: 0.8, alignment: 0.15 } }))).toBe(77);
    expect(alignmentPercent(report({ subScores: { technical: 0.8, alignment: 0.05 } }))).toBe(0);
    expect(alignmentPercent(report({ subScores: { technical: 0.8, alignment: 0.3 } }))).toBe(100);
    expect(alignmentPercent(report({ subScores: { technical: 0.8 } }))).toBeNull();
    expect(alignmentPercent(null)).toBeNull();
  });

  it("collects active caption-alignment-flagged item IDs for the re-caption action (sc-6537)", () => {
    const flagged = report({
      items: [
        { itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] },
        { itemId: "b", flags: [{ check: "blur", severity: "warn" }] },
        {
          itemId: "c",
          flags: [{ check: "caption_alignment", severity: "warn", acknowledged: true }],
        },
        { itemId: "d", flags: [{ check: "caption_alignment", severity: "warn" }] },
      ],
    });
    expect(captionAlignmentFlaggedItemIds(flagged)).toEqual(["a", "d"]);
    expect(captionAlignmentFlaggedItemIds(report({ items: [] }))).toEqual([]);
    expect(captionAlignmentFlaggedItemIds(null)).toEqual([]);
  });

  it("flattens the duplicate-removal plan to the item IDs to drop (sc-6539)", () => {
    const withPlan = report({
      duplicateRemoval: {
        groups: [
          { keep: "a", remove: ["b"] },
          { keep: "c", remove: ["d", "e"] },
        ],
      },
    });
    expect(duplicateRemovalItemIds(withPlan)).toEqual(["b", "d", "e"]);
    // No plan / no report → nothing to remove.
    expect(duplicateRemovalItemIds(report({}))).toEqual([]);
    expect(duplicateRemovalItemIds(null)).toEqual([]);
  });
});

describe("trainBlockedByReadiness", () => {
  it("blocks only when a report exists and is blocked", () => {
    expect(trainBlockedByReadiness(report({ gate: "blocked" }))).toBe(true);
    expect(trainBlockedByReadiness(report({ gate: "needs_attention" }))).toBe(false);
    expect(trainBlockedByReadiness(report({ gate: "ready" }))).toBe(false);
    // No report (unsaved/loading/older dataset) must never hard-block.
    expect(trainBlockedByReadiness(null)).toBe(false);
    expect(trainBlockedByReadiness(undefined)).toBe(false);
  });
});

describe("readinessBySelectionKey", () => {
  it("maps report item ids onto member-asset selection keys", () => {
    const dataset = {
      id: "ds-1",
      items: [
        { id: "item-1", assetId: "asset-1" },
        { id: "item-2" },
      ],
    };
    const rpt = report({
      items: [
        { itemId: "item-1", severity: "warn", flags: [{ check: "blur", severity: "warn" }] },
        { itemId: "item-2", severity: undefined, flags: [] },
      ],
    });
    const map = readinessBySelectionKey(dataset, rpt);
    // assetId-backed item keys on its assetId; dataset-owned item keys on the synthetic id.
    expect(map.get("asset-1").severity).toBe("warn");
    expect(map.get("dataset-item:ds-1:item-2")).toBeTruthy();
  });

  it("is empty without a report", () => {
    expect(readinessBySelectionKey({ items: [{ id: "x" }] }, null).size).toBe(0);
  });
});

describe("flagMetric", () => {
  it("exposes the raw value and threshold for the Advanced surface", () => {
    const metric = flagMetric({ check: "blur", value: 42.5, threshold: 100, peers: [] });
    expect(metric.label).toContain("sharpness");
    expect(metric.value).toBe(42.5);
    expect(metric.threshold).toBe(100);
  });
});

describe("acknowledged (dismissed) findings", () => {
  it("drops dismissed flags from counts, summary, and the badge reason", () => {
    const rpt = report({
      itemCount: 2,
      items: [
        { itemId: "a", severity: undefined, flags: [{ check: "blur", severity: "warn", acknowledged: true }] },
        { itemId: "b", severity: "warn", flags: [{ check: "blur", severity: "warn" }] },
      ],
    });
    // Only b's blur still counts — the readout must match the badge and gate.
    expect(flagCountsByCheck(rpt).blur).toBe(1);
    expect(datasetDoctorSummary(rpt)).toContain("1 looks blurry");
    expect(datasetDoctorSummary(rpt)).not.toContain("2 look blurry");
    // a's only finding is dismissed → no active worst flag → clean badge, and the server-supplied
    // severity (undefined) badges it good.
    const a = rpt.items[0];
    expect(worstFlag(a)).toBeNull();
    expect(primaryReason(a)).toBe("");
    expect(itemBadge(a).tone).toBe("good");
  });

  it("derives the dismissed-check set and the next set on toggle", () => {
    const entry = {
      flags: [
        { check: "blur", severity: "warn", acknowledged: true },
        { check: "near_duplicate", severity: "warn" },
      ],
    };
    expect(dismissedChecks(entry)).toEqual(["blur"]);
    // Dismiss adds (deduped); undo removes.
    expect(nextDismissedChecks(entry, "near_duplicate", true).sort()).toEqual(["blur", "near_duplicate"]);
    expect(nextDismissedChecks(entry, "blur", true)).toEqual(["blur"]);
    expect(nextDismissedChecks(entry, "blur", false)).toEqual([]);
  });
});

describe("readinessQueryParams", () => {
  it("serializes the chosen target context", () => {
    const qs = readinessQueryParams({
      resolution: 768,
      recommendedFor: ["character", "style"],
      characterType: "person",
    });
    const params = new URLSearchParams(qs);
    expect(params.get("targetResolution")).toBe("768");
    expect(params.get("recommendedFor")).toBe("character,style");
    expect(params.get("characterType")).toBe("person");
  });

  it("omits empty/zero values", () => {
    const qs = readinessQueryParams({ resolution: 0, recommendedFor: [], characterType: "" });
    expect(qs).toBe("");
  });
});
