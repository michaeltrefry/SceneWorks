// Dataset Doctor presentation helpers (sc-6534). Pure, view-model shaping over
// the server's readiness report (sc-6533, GET …/training/datasets/:id/readiness):
// the thumbnail badge, the plain-language per-flag reason, the friendly readout
// sentence, the Train gate, and the fetch query. No React, no fetch — so the copy
// and the gating logic are unit-tested in isolation (datasetReadiness.test.js).
//
// Report payload (camelCase): {
//   gate: "ready" | "needs_attention" | "blocked",
//   subScores: { technical, diversity?, identity?, alignment? },
//   counts: { info, warn, fatal },
//   itemCount,
//   items: [{ itemId, severity?, flags: [{ check, severity, value?, threshold?, peers[] }] }],
//   datasetFlags: [{ check, severity, value?, threshold?, peers[] }],
// }

import { datasetItemSelectionKey } from "./datasetHelpers.js";

// Thumbnail badge per worst-severity (sc-6534 acceptance): ✓ good / ⚠ needs
// attention / ✕ likely to hurt — plus a neutral "pending" for items the report
// hasn't covered yet (just-added/unsaved, or first fetch in flight). A map miss
// must NEVER read as ✓ "good": that would imply an unassessed photo is fine.
export const READINESS_BADGE = {
  good: { symbol: "✓", tone: "good", label: "Looks good" },
  warn: { symbol: "⚠", tone: "warn", label: "Needs attention" },
  fatal: { symbol: "✕", tone: "fatal", label: "Likely to hurt training" },
  pending: { symbol: "·", tone: "pending", label: "Not assessed yet" },
};

// Severity ranking for "worst flag" selection (matches core's Info < Warn < Fatal).
const SEVERITY_RANK = { info: 1, warn: 2, fatal: 3 };

export function severityRank(severity) {
  return SEVERITY_RANK[severity] ?? 0;
}

// A report entry's worst severity → badge. `undefined`/`info` on a covered item
// means "assessed, nothing wrong" → ✓.
export function badgeForSeverity(severity) {
  if (severity === "fatal") {
    return READINESS_BADGE.fatal;
  }
  if (severity === "warn") {
    return READINESS_BADGE.warn;
  }
  return READINESS_BADGE.good;
}

// The single source of truth for "no entry / still loading → pending, never ✓".
// `entry` is the per-item readiness object (or undefined when the report doesn't
// cover this image yet).
export function itemBadge(entry, { loading = false } = {}) {
  if (loading || !entry) {
    return READINESS_BADGE.pending;
  }
  return badgeForSeverity(entry.severity);
}

// Plain-language reason per Tier-0 check (sc-6530 catalog). Only the eight checks
// that exist today — no Tier-1 identity copy ("different person") until the
// embedding job (sc-6535) actually computes it. `string_enum` can emit an unknown
// check, so anything unrecognized degrades to generic copy, never `undefined`.
const CHECK_REASON = {
  resolution: "Low resolution — may look soft at the training size",
  crop_loss: "A lot of this photo is cropped away at the training size",
  blur: "This photo looks a little blurry",
  exposure: "Over- or under-exposed — clipped highlights or shadows",
  exact_duplicate: "Exact duplicate of another photo",
  near_duplicate: "Nearly identical to another photo",
  near_duplicate_embedding: "Looks very similar to other photos",
  low_diversity: "The set isn't varied enough — too many similar-looking shots",
  low_aesthetic: "Lower aesthetic score for a style set — advisory only",
  count: "Not enough photos to train well yet",
  decode: "This image couldn't be read",
};

export function flagReason(flag) {
  if (!flag) {
    return "";
  }
  return CHECK_REASON[flag.check] ?? "Flagged for review";
}

// An item's findings that still count — dismissed ones (sc-6534) are excluded from every derivation
// (badge reason, summary, counts) the same way the server drops them from the rollup. They are still
// present in `entry.flags` so the Advanced surface can show them struck-through.
export function activeFlags(entry) {
  return (entry?.flags ?? []).filter((flag) => !flag.acknowledged);
}

// The checks the user has dismissed on this item (the acked flags currently in play).
export function dismissedChecks(entry) {
  return (entry?.flags ?? []).filter((flag) => flag.acknowledged).map((flag) => flag.check);
}

// The full dismissed-check set after toggling one check — the body the ack endpoint expects (it
// replaces the stored set). Dedupes on add; drops on undo.
export function nextDismissedChecks(entry, check, dismissed) {
  const current = dismissedChecks(entry);
  if (dismissed) {
    return current.includes(check) ? current : [...current, check];
  }
  return current.filter((existing) => existing !== check);
}

// The worst-severity *active* flag on an item, for the badge's one-line reason.
export function worstFlag(entry) {
  return activeFlags(entry).reduce((worst, flag) => {
    if (!worst || severityRank(flag.severity) > severityRank(worst.severity)) {
      return flag;
    }
    return worst;
  }, null);
}

export function primaryReason(entry) {
  return flagReason(worstFlag(entry));
}

// Raw metric for the Advanced surface — the measured value behind a flag and the
// threshold it was judged against (both already in the report for flagged items).
const METRIC_LABEL = {
  resolution: "short edge (px)",
  crop_loss: "cropped fraction",
  blur: "sharpness (Laplacian variance)",
  exposure: "clipped fraction",
  near_duplicate: "Hamming distance",
  exact_duplicate: "Hamming distance",
  near_duplicate_embedding: "cosine similarity",
  low_diversity: "diversity score",
  low_aesthetic: "aesthetic score",
  count: "item count",
  decode: "decoded",
};

export function flagMetric(flag) {
  return {
    check: flag.check,
    label: METRIC_LABEL[flag.check] ?? flag.check,
    value: flag.value ?? null,
    threshold: flag.threshold ?? null,
    peers: flag.peers ?? [],
  };
}

// Per-check item tallies for the readout (each item counted once per check, even
// if it carries the same check twice). Dataset-level checks (count) are handled
// separately in the summary.
export function flagCountsByCheck(report) {
  const counts = {};
  for (const item of report?.items ?? []) {
    const seen = new Set();
    // Dismissed findings (sc-6534) are excluded — the readout must match the badge and gate.
    for (const flag of activeFlags(item)) {
      if (seen.has(flag.check)) {
        continue;
      }
      seen.add(flag.check);
      counts[flag.check] = (counts[flag.check] ?? 0) + 1;
    }
  }
  return counts;
}

// Friendly sentence-fragments per check, e.g. "2 look blurry".
const CHECK_PHRASE = {
  blur: (n) => `${n} look${n === 1 ? "s" : ""} blurry`,
  near_duplicate: (n) => (n === 1 ? "1 is a near-duplicate" : `${n} are near-duplicates`),
  exact_duplicate: (n) => (n === 1 ? "1 is an exact duplicate" : `${n} are exact duplicates`),
  near_duplicate_embedding: (n) =>
    n === 1 ? "1 looks very similar to others" : `${n} look very similar to others`,
  resolution: (n) => `${n} ${n === 1 ? "is" : "are"} low-resolution`,
  crop_loss: (n) => `${n} lose${n === 1 ? "s" : ""} a lot to cropping`,
  exposure: (n) => `${n} ${n === 1 ? "is" : "are"} over- or under-exposed`,
  decode: (n) => `${n} couldn't be read`,
};
// `low_diversity` and `low_aesthetic` are omitted: they're dataset-level findings (not per-image
// counts), surfaced separately in the summary + their sub-score meters.
const PHRASE_ORDER = [
  "blur",
  "near_duplicate",
  "near_duplicate_embedding",
  "exact_duplicate",
  "resolution",
  "crop_loss",
  "exposure",
  "decode",
];

function joinList(parts) {
  if (parts.length <= 1) {
    return parts.join("");
  }
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

function capitalizeFirst(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function countFlagFor(report) {
  return (report?.datasetFlags ?? []).find((flag) => flag.check === "count") ?? null;
}

// The Dataset Doctor readout (sc-6534): "18 photos. 2 look blurry and 3 are
// near-duplicates. Replacing these would make the LoRA stronger." Bias to warn:
// the only hard-stop wording is "not enough photos".
export function datasetDoctorSummary(report) {
  if (!report) {
    return "";
  }
  const n = report.itemCount ?? (report.items?.length ?? 0);
  const parts = [`${n} photo${n === 1 ? "" : "s"}.`];

  const countFlag = countFlagFor(report);
  // Too few images is the one genuinely blocking case — lead with it.
  if (countFlag && report.gate === "blocked") {
    const need = Math.round(countFlag.threshold ?? 0);
    const have = Math.round(countFlag.value ?? n);
    const more = Math.max(need - have, 0);
    parts.push(`You need at least ${need} to train${more ? ` — add ${more} more` : ""}.`);
    return parts.join(" ");
  }

  const counts = flagCountsByCheck(report);
  const fragments = PHRASE_ORDER.filter((check) => counts[check]).map((check) => CHECK_PHRASE[check](counts[check]));
  if (fragments.length) {
    parts.push(`${capitalizeFirst(joinList(fragments))}.`);
  }

  // Low diversity is a set-level property (not a per-image count), so it's surfaced separately
  // from the per-item fragments — and from the diversity sub-score meter in the readout.
  const lowDiversity = (report.datasetFlags ?? []).some(
    (flag) => flag.check === "low_diversity" && !flag.acknowledged,
  );
  if (lowDiversity) {
    parts.push("The photos are quite similar — adding more variety would help.");
  }

  // Aesthetic is a STYLE-only advisory (sc-6537) — surfaced as a gentle heads-up, never a blocker.
  const lowAesthetic = (report.datasetFlags ?? []).some(
    (flag) => flag.check === "low_aesthetic" && !flag.acknowledged,
  );
  if (lowAesthetic) {
    parts.push(
      "These score a little lower on aesthetics for a style set — just a heads-up, not a problem.",
    );
  }

  if (report.gate === "ready") {
    parts.push("This set looks ready to train.");
  } else if (countFlag) {
    const need = Math.round(countFlag.threshold ?? 0);
    parts.push(`Aim for ${need}+ photos with some variety to make it stronger.`);
  } else if (fragments.length) {
    parts.push("Replacing or removing these would make the LoRA stronger.");
  } else if (!lowDiversity && !lowAesthetic) {
    parts.push("This set looks ready to train.");
  }
  return parts.join(" ");
}

// Gate → headline + tone for the readiness meter.
export const GATE_META = {
  ready: { label: "Ready to train", tone: "good" },
  needs_attention: { label: "Trainable — a few things to check", tone: "warn" },
  blocked: { label: "Not ready to train", tone: "fatal" },
};

export function gateMeta(gate) {
  return GATE_META[gate] ?? { label: "Assessing…", tone: "pending" };
}

// Technical sub-score as a 0–100 percentage for the meter (share of items with no
// technical-quality warning). Tier-1 sub-scores stay absent until sc-6535.
export function technicalPercent(report) {
  const value = report?.subScores?.technical;
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.round(value * 100);
}

// Diversity sub-score (1 − mean pairwise CLIP cosine) as a 0–100 percentage for the Variety
// meter. Absent until the embedding job (sc-6535) computes it — `null` hides the meter.
export function diversityPercent(report) {
  const value = report?.subScores?.diversity;
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.round(value * 100);
}

// Aesthetic sub-score — the mean LAION-Aesthetics score (~[1, 10]) for STYLE datasets only; `null`
// for person/object (never computed) or until the embedding job runs. Surfaced as an advisory
// "Aesthetic" readout (rounded to one decimal), never a gate.
export function aestheticScore(report) {
  const value = report?.subScores?.aesthetic;
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.round(value * 10) / 10;
}

// Train is disabled ONLY when a report exists AND its gate is "blocked". No report
// (unsaved draft, fetch in flight, older dataset) must never hard-block — the
// spike's bias-to-warn policy (sc-6530 §4).
export function trainBlockedByReadiness(report) {
  return report?.gate === "blocked";
}

// Map the report's per-item entries (keyed by the server item id) onto the web's
// member-asset selection keys, so the editor can badge each thumbnail. The bridge
// is `datasetItemSelectionKey`, the same id the caption grid renders under.
export function readinessBySelectionKey(dataset, report) {
  const byItemId = new Map((report?.items ?? []).map((entry) => [entry.itemId, entry]));
  const map = new Map();
  (dataset?.items ?? []).forEach((item, index) => {
    const entry = byItemId.get(item.id);
    if (entry) {
      map.set(datasetItemSelectionKey(dataset, item, index), entry);
    }
  });
  return map;
}

// Build the readiness fetch query string from what the training screen has chosen.
// All optional — the server falls back to per-kind defaults (sc-6533).
export function readinessQueryParams({ resolution, recommendedFor, characterType, minItems } = {}) {
  const params = new URLSearchParams();
  const res = Number(resolution);
  if (Number.isFinite(res) && res > 0) {
    params.set("targetResolution", String(Math.round(res)));
  }
  const tags = Array.isArray(recommendedFor)
    ? recommendedFor.filter(Boolean).join(",")
    : String(recommendedFor ?? "").trim();
  if (tags) {
    params.set("recommendedFor", tags);
  }
  if (characterType) {
    params.set("characterType", String(characterType));
  }
  const min = Number(minItems);
  if (Number.isFinite(min) && min > 0) {
    params.set("minItems", String(Math.round(min)));
  }
  return params.toString();
}
