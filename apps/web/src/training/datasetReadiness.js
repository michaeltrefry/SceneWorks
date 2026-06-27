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

export async function captionHash(caption) {
  const bytes = new TextEncoder().encode(caption ?? "");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

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

// Plain-language reason per check (sc-6530 catalog + the Tier-1/face advisories layered on later). The
// face-stack checks (identity_mismatch/no_face/small_subject, sc-6538) only ever appear on Person
// datasets. `string_enum` can emit an unknown check, so anything unrecognized degrades to generic copy,
// never `undefined`.
const CHECK_REASON = {
  resolution: "Low resolution — may look soft at the training size",
  crop_loss: "A lot of this photo is cropped away at the training size",
  blur: "This photo looks a little blurry",
  exposure: "Over- or under-exposed — clipped highlights or shadows",
  exact_duplicate: "Exact duplicate of another photo",
  near_duplicate: "Nearly identical to another photo",
  near_duplicate_embedding: "Looks very similar to other photos",
  low_diversity: "The set isn't varied enough — too many similar-looking shots",
  caption_alignment: "Caption may not match this image — re-captioning can help",
  low_aesthetic: "Lower aesthetic score for a style set — advisory only",
  embedding_outlier: "Doesn't match the rest of the set — off-style, or missing the subject",
  identity_mismatch: "Looks like a different person than the rest of the set",
  no_face: "No face detected — won't help a character LoRA",
  small_subject: "The face is small in the frame — may be hard to learn from",
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
  caption_alignment: "caption-image cosine similarity",
  low_aesthetic: "aesthetic score",
  embedding_outlier: "cosine to set centroid",
  identity_mismatch: "identity cosine",
  small_subject: "face size (fraction of frame)",
  no_face: "face detected",
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
  caption_alignment: (n) =>
    n === 1 ? "1 caption may not match its image" : `${n} captions may not match their images`,
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
  "caption_alignment",
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
  } else if (counts.caption_alignment && fragments.length === 1) {
    parts.push(`Re-captioning ${counts.caption_alignment === 1 ? "this" : "these"} can improve training.`);
  } else if (counts.caption_alignment && fragments.length) {
    parts.push("Re-captioning the mismatched captions and replacing weak images would make the LoRA stronger.");
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

// Caption↔image CLIP alignment as a 0–100 meter score. The source value is mean CLIP cosine,
// clamped by the server for display; absent until text embeddings exist for the current caption hash.
// The caption-alignment sub-score is the mean RAW CLIP image↔text cosine — low and compressed
// (matched pairs land ~0.13–0.16, clear mismatches ~0.03–0.05). Showing it ×100 reads alarmingly (a
// good caption as "15%"), so the meter rescales the meaningful range [0.05, 0.18] → [0, 100]. This is
// display-only — the stored sub-score stays the raw cosine. sc-6537.
const ALIGNMENT_DISPLAY_LO = 0.05;
const ALIGNMENT_DISPLAY_HI = 0.18;
export function alignmentPercent(report) {
  const value = report?.subScores?.alignment;
  if (!Number.isFinite(value)) {
    return null;
  }
  const scaled =
    ((value - ALIGNMENT_DISPLAY_LO) / (ALIGNMENT_DISPLAY_HI - ALIGNMENT_DISPLAY_LO)) * 100;
  return Math.round(Math.min(100, Math.max(0, scaled)));
}

// Item IDs whose active (non-dismissed) flags include `caption_alignment` — the targets for the
// "Re-caption flagged" action surfaced from the readout. sc-6537.
export function captionAlignmentFlaggedItemIds(report) {
  return (report?.items ?? [])
    .filter((item) => activeFlags(item).some((flag) => flag.check === "caption_alignment"))
    .map((item) => item.itemId);
}

// Item IDs the one-tap "drop duplicates" action would remove (sc-6539): every group's `remove` list
// from the server-computed plan, which keeps the sharpest copy of each exact/near-duplicate cluster.
// Empty when nothing is safely removable. Only exact + pHash duplicates reach here — CLIP
// near-duplicates are deliberately left in place (legitimate training variety), so this never strips
// the diversity the sibling Variety check rewards.
export function duplicateRemovalItemIds(report) {
  return (report?.duplicateRemoval?.groups ?? []).flatMap((group) => group?.remove ?? []);
}

// Item IDs whose active (non-dismissed) flags include `resolution` — the sub-target images the
// one-tap "Upscale low-res" action targets (sc-6539). Real-ESRGAN upscales each and re-points the
// item; the originals stay in the library.
export function lowResolutionFlaggedItemIds(report) {
  return (report?.items ?? [])
    .filter((item) => activeFlags(item).some((flag) => flag.check === "resolution"))
    .map((item) => item.itemId);
}

// Item IDs whose active flags include `crop_loss` — extreme-aspect images that lose too much to the
// training-bucket crop, which the one-tap "Smart-crop" action trims toward a trainable aspect (sc-6539).
export function cropLossFlaggedItemIds(report) {
  return (report?.items ?? [])
    .filter((item) => activeFlags(item).some((flag) => flag.check === "crop_loss"))
    .map((item) => item.itemId);
}

// Item IDs whose stored file can still carry embedded metadata (EXIF/GPS/ICC) — i.e. it is not yet a
// normalized metadata-free PNG (sc-6539). These are the only targets the one-tap "Strip metadata"
// action would actually change (it re-encodes to a clean PNG). Once an item is stripped the server
// reports `metadataStrippable: false`, so it drops out and the action stops re-appearing for
// already-clean photos. The backend sets the flag from each item's format, so a missing field (older
// payload) reads as not-strippable.
export function metadataStrippableItemIds(report) {
  return (report?.items ?? [])
    .filter((item) => item.metadataStrippable)
    .map((item) => item.itemId);
}

// --- Kind-aware recommendations (sc-6540) ---------------------------------------------------------
// Concrete next-steps that change with the training kind (person/style/object) the user picked in
// Teach. These are the MANUAL moves only — acquiring or replacing images — not the one-tap fixes that
// already have buttons (dedupe/upscale/smart-crop/strip-exif/re-caption), so the list stays a short
// "what only you can do" rather than restating the actions above it.

const KIND_NOUN = { person: "character", style: "style", object: "object" };

function kindNoun(kind) {
  return KIND_NOUN[kind] ?? "LoRA";
}

function diversityAdvice(kind) {
  switch (kind) {
    case "person":
      return "Add photos from other poses, expressions, angles, and lighting — the set looks too similar, so the LoRA may overfit to one look.";
    case "object":
      return "Add shots from more angles with varied backgrounds, so the LoRA binds to the object rather than the setting.";
    case "style":
      return "Apply the style to more varied subjects — the set clusters too tightly to teach the style broadly.";
    default:
      return "Add more variety (angles, lighting, subjects) so the LoRA generalizes.";
  }
}

function outlierAdvice(kind, count) {
  const n = `${count} image${count === 1 ? "" : "s"}`;
  if (kind === "object") {
    return `${n} look like ${count === 1 ? "it doesn't" : "they don't"} contain the object — remove or replace ${count === 1 ? "it" : "them"}.`;
  }
  return `${n} ${count === 1 ? "doesn't" : "don't"} match the rest of your style — remove or replace ${count === 1 ? "it" : "them"} so ${count === 1 ? "it doesn't" : "they don't"} dilute it.`;
}

function kindTip(kind) {
  switch (kind) {
    case "person":
      return "Keep captions about the scene, clothing, and pose — not the face. Over-describing the invariant identity weakens what the LoRA learns.";
    case "object":
      return "Vary backgrounds and angles, but keep the object clearly the subject in every shot.";
    case "style":
      return "Aim for diverse subjects in one consistent style — variety of what, consistency of how.";
    default:
      return "";
  }
}

// Build the ordered recommendation list for the readout. Each entry is `{ id, tone, text }`. Pure
// over the report's kind + existing signals, so the same set yields different guidance per kind
// (the acceptance criterion). Items the user must acquire/replace by hand only — never a one-tap fix.
export function datasetRecommendations(report) {
  if (!report) {
    return [];
  }
  const kind = report.kind ?? null;
  const datasetFlags = report.dataset_flags ?? report.datasetFlags ?? [];
  const datasetHas = (check) => datasetFlags.some((flag) => flag.check === check);
  const itemsWithActive = (check) =>
    (report.items ?? []).filter((item) => activeFlags(item).some((flag) => flag.check === check));

  const recs = [];

  // Too few images → acquire more (kind sets the target via the Count flag's threshold).
  const countFlag = datasetFlags.find((flag) => flag.check === "count");
  if (countFlag) {
    const min = Math.max(0, Math.round(countFlag.threshold ?? 0));
    const need = Math.max(1, min - (report.itemCount ?? 0));
    recs.push({
      id: "count",
      tone: "warn",
      text: `Add ${need} more image${need === 1 ? "" : "s"} — aim for at least ${min} for a reliable ${kindNoun(kind)} LoRA.`,
    });
  }

  // Too uniform → acquire variety, framed by what this kind needs.
  if (datasetHas("low_diversity")) {
    recs.push({ id: "diversity", tone: "warn", text: diversityAdvice(kind) });
  }

  // Off-style / object-absent outliers → remove or replace. Style/Object only: the detector is off for
  // Person (the backend never raises it), and this guard also keeps the copy from mis-firing as "style".
  const outliers =
    kind === "style" || kind === "object" ? itemsWithActive("embedding_outlier") : [];
  if (outliers.length) {
    recs.push({ id: "outlier", tone: "warn", text: outlierAdvice(kind, outliers.length) });
  }

  // Person face findings (sc-6538): wrong-person / no-face / small-face. PERSON only — the backend
  // raises these only for a character set, and the explicit gate keeps the copy correct (and matches
  // the outlier gate above). Every one is a manual remove/replace/crop-closer move — no one-tap fix
  // targets the face (smart-crop trims aspect, not framing) — so they belong in this list.
  if (kind === "person") {
    const mismatched = itemsWithActive("identity_mismatch").length;
    if (mismatched) {
      recs.push({
        id: "identity",
        tone: "warn",
        text: `${mismatched} image${mismatched === 1 ? "" : "s"} look${mismatched === 1 ? "s" : ""} like a different person — remove or replace ${mismatched === 1 ? "it" : "them"} so the LoRA learns one identity.`,
      });
    }
    const faceless = itemsWithActive("no_face").length;
    if (faceless) {
      recs.push({
        id: "no_face",
        tone: "warn",
        text: `${faceless} image${faceless === 1 ? " has" : "s have"} no detectable face — ${faceless === 1 ? "it won't" : "they won't"} teach the character; remove or replace ${faceless === 1 ? "it" : "them"}.`,
      });
    }
    const small = itemsWithActive("small_subject").length;
    if (small) {
      recs.push({
        id: "small_subject",
        tone: "info",
        text: `${small} image${small === 1 ? " has" : "s have"} a very small face — crop in closer or replace ${small === 1 ? "it" : "them"} so the identity is clear.`,
      });
    }
  }

  // Weak aesthetics → replace (Style only).
  if (kind === "style" && datasetHas("low_aesthetic")) {
    recs.push({
      id: "aesthetic",
      tone: "info",
      text: "Several images score low on aesthetics — swap in stronger examples so the style reads clearly.",
    });
  }

  // Blur / exposure → replace by hand (resolution + crop have one-tap fixes; these don't).
  const unfixable = (report.items ?? []).filter((item) =>
    activeFlags(item).some((flag) => flag.check === "blur" || flag.check === "exposure"),
  ).length;
  if (unfixable) {
    recs.push({
      id: "unfixable",
      tone: "warn",
      text: `${unfixable} image${unfixable === 1 ? " is" : "s are"} blurry or poorly exposed — replace ${unfixable === 1 ? "it" : "them"} (no one-tap fix).`,
    });
  }

  // A standing kind-specific tip — the "what this kind needs" nudge, so guidance differs by kind even
  // on a clean set.
  const tip = kindTip(kind);
  if (tip) {
    recs.push({ id: "tip", tone: "tip", text: tip });
  }

  return recs;
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
