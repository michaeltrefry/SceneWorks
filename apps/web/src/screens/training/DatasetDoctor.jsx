import React from "react";

import {
  aestheticScore,
  alignmentPercent,
  captionAlignmentFlaggedItemIds,
  cropLossFlaggedItemIds,
  datasetDoctorSummary,
  datasetRecommendations,
  diversityPercent,
  duplicateRemovalItemIds,
  flagMetric,
  flagReason,
  gateMeta,
  itemBadge,
  lowResolutionFlaggedItemIds,
  metadataStrippableItemIds,
  primaryReason,
  technicalPercent,
} from "../../training/datasetReadiness.js";

// Dataset Doctor surface (sc-6534): the plain-language readiness readout, the
// per-thumbnail quality badge, and the Advanced raw-metric list. All presentational
// — it renders the server's readiness report (sc-6533); the view-model shaping and
// copy live in datasetReadiness.js so they stay unit-tested.

// Per-thumbnail badge: ✓ good / ⚠ needs attention / ✕ likely to hurt, plus a neutral
// "·" for an item the report hasn't covered yet (just added, or first fetch in flight).
// The tooltip carries the one plain-language reason behind the worst flag.
export function ReadinessBadge({ entry, loading = false }) {
  const badge = itemBadge(entry, { loading });
  let reason;
  if (loading && !entry) {
    reason = "Checking…";
  } else if (!entry) {
    reason = "Not assessed yet";
  } else {
    reason = primaryReason(entry) || "Looks good";
  }
  return (
    <span
      className={`dataset-doctor-badge tone-${badge.tone}`}
      title={reason}
      aria-label={reason === badge.label ? badge.label : `${badge.label} — ${reason}`}
    >
      {badge.symbol}
    </span>
  );
}

// Findings that can't be dismissed: a decode failure is untrainable, and count is dataset-level
// (sc-6534). The server also strips these from any ack, but the UI shouldn't offer the affordance.
const NON_DISMISSABLE = new Set(["decode", "count"]);

// The Advanced raw-metric list for one flagged item (the editor IS the advanced
// surface). Shows each flag's plain reason, the measured value vs. threshold, and a per-image
// override (sc-6534): Dismiss a finding to drop it from the badge/readout/gate, or Undo to restore
// it. A dismissed finding stays listed, struck-through, so the choice is visible.
export function ReadinessFlagDetails({ entry, onToggle }) {
  const flags = entry?.flags ?? [];
  if (!flags.length) {
    return null;
  }
  return (
    <ul className="dataset-doctor-flags" aria-label="Quality findings">
      {flags.map((flag, index) => {
        const metric = flagMetric(flag);
        const hasValue = Number.isFinite(metric.value);
        const hasThreshold = Number.isFinite(metric.threshold);
        const dismissed = Boolean(flag.acknowledged);
        const tone = dismissed
          ? "dismissed"
          : flag.severity === "fatal"
            ? "fatal"
            : flag.severity === "warn"
              ? "warn"
              : "good";
        const canToggle = typeof onToggle === "function" && !NON_DISMISSABLE.has(flag.check);
        return (
          <li className={`tone-${tone}${dismissed ? " dismissed" : ""}`} key={`${flag.check}-${index}`}>
            <div className="dataset-doctor-flag-text">
              <span className="dataset-doctor-flag-reason">{flagReason(flag)}</span>
              {hasValue || hasThreshold ? (
                <span className="dataset-doctor-flag-metric">
                  {metric.label}: {hasValue ? round(metric.value) : "—"}
                  {hasThreshold ? ` (threshold ${round(metric.threshold)})` : ""}
                </span>
              ) : null}
            </div>
            {canToggle ? (
              <button
                className="dataset-doctor-flag-toggle"
                onClick={() => onToggle(flag.check, !dismissed)}
                type="button"
              >
                {dismissed ? "Undo" : "Dismiss"}
              </button>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

// Per-metric distributions for the Advanced surface (sc-6534), drawn from the report's
// `distributions` payload — sharpness and the two clip fractions, each a small histogram with the
// threshold marked. Renders nothing until any item has scalars.
const DISTRIBUTION_METRICS = [
  { key: "blurVariance", label: "Sharpness" },
  { key: "shadowClip", label: "Shadow clipping" },
  { key: "highlightClip", label: "Highlight clipping" },
];

export function DatasetDoctorDistributions({ report }) {
  const distributions = report?.distributions;
  if (!distributions) {
    return null;
  }
  const charts = DISTRIBUTION_METRICS.filter(({ key }) => distributions[key]?.values?.length);
  if (!charts.length) {
    return null;
  }
  return (
    <div className="dataset-doctor-distributions" aria-label="Metric distributions">
      {charts.map(({ key, label }) => (
        <MetricHistogram key={key} label={label} metric={distributions[key]} />
      ))}
    </div>
  );
}

const HISTOGRAM_BINS = 10;

function MetricHistogram({ label, metric }) {
  const values = metric.values ?? [];
  const threshold = Number.isFinite(metric.threshold) ? metric.threshold : null;
  const lo = Math.min(...values, threshold ?? Infinity);
  const hi = Math.max(...values, threshold ?? -Infinity);
  const span = hi - lo || 1;
  const counts = new Array(HISTOGRAM_BINS).fill(0);
  for (const value of values) {
    const idx = Math.min(HISTOGRAM_BINS - 1, Math.max(0, Math.floor(((value - lo) / span) * HISTOGRAM_BINS)));
    counts[idx] += 1;
  }
  const peak = Math.max(...counts, 1);
  const thresholdPct = threshold == null ? null : Math.max(0, Math.min(100, ((threshold - lo) / span) * 100));
  return (
    <div className="dataset-doctor-histogram">
      <span className="dataset-doctor-histogram-label">
        {label}
        <span className="dataset-doctor-histogram-hint">{metric.higherIsBetter ? "higher is better" : "lower is better"}</span>
      </span>
      <div className="dataset-doctor-histogram-bars">
        {counts.map((count, index) => (
          <span
            className="dataset-doctor-histogram-bar"
            key={index}
            style={{ height: `${(count / peak) * 100}%` }}
          />
        ))}
        {thresholdPct == null ? null : (
          <span
            className="dataset-doctor-histogram-threshold"
            style={{ left: `${thresholdPct}%` }}
            title={`threshold ${metric.threshold}`}
          />
        )}
      </div>
    </div>
  );
}

// The friendly readout shown before the build/train actions: gate headline, a
// technical-quality meter, the summary sentence, and severity-count chips. Renders
// nothing for an unsaved draft (no report, not loading) so the panel isn't noisy
// before there's anything to assess.
export function DatasetDoctorReadout({
  report,
  loading = false,
  compact = false,
  onRecaptionFlagged,
  onRemoveDuplicates,
  onUpscaleLowRes,
  onSmartCrop,
  onStripExif,
  onAnalyzeDataset,
  onAnalyzeFaces,
}) {
  if (!report) {
    if (loading) {
      return (
        <div className="dataset-doctor dataset-doctor-pending" aria-label="Dataset Doctor">
          Checking dataset…
        </div>
      );
    }
    return null;
  }
  const gate = gateMeta(report.gate);
  const technical = technicalPercent(report);
  // Tier-1 (sc-6535): absent until the embedding job has run, so the Variety meter only appears
  // once the report carries a diversity sub-score.
  const diversity = diversityPercent(report);
  // Caption alignment (sc-6537): absent until current-caption text embeddings are available.
  const alignment = alignmentPercent(report);
  // Aesthetic (sc-6537): STYLE datasets only — `null` for person/object, so the meter is style-scoped.
  const aesthetic = aestheticScore(report);
  const counts = report.counts ?? {};
  // sc-6537: items whose caption looks misaligned — the targets for the re-caption action. Empty
  // unless a consumer wired `onRecaptionFlagged` (so the compact readout never shows the button).
  const recaptionFlaggedIds =
    typeof onRecaptionFlagged === "function" ? captionAlignmentFlaggedItemIds(report) : [];
  // sc-6539: exact/near-duplicate copies the one-tap action would drop (keeping the sharpest of each).
  // Empty unless a consumer wired `onRemoveDuplicates`, so the compact readout stays button-free there.
  const removeDuplicateIds =
    typeof onRemoveDuplicates === "function" ? duplicateRemovalItemIds(report) : [];
  // sc-6539: sub-target images the one-tap upscale would enlarge (Real-ESRGAN). Empty unless a
  // consumer wired `onUpscaleLowRes`.
  const lowResIds =
    typeof onUpscaleLowRes === "function" ? lowResolutionFlaggedItemIds(report) : [];
  // sc-6539: extreme-aspect images the one-tap smart-crop would trim toward a trainable aspect.
  const cropLossIds =
    typeof onSmartCrop === "function" ? cropLossFlaggedItemIds(report) : [];
  // sc-6539: EXIF-strip targets only items whose stored file can still carry metadata (not yet a
  // normalized metadata-free PNG) — so once an item is stripped it drops out and the action stops
  // re-appearing. Empty unless a consumer wired `onStripExif`.
  const stripExifIds =
    typeof onStripExif === "function" ? metadataStrippableItemIds(report) : [];
  const stripExifCount = stripExifIds.length;
  // sc-6535: the CLIP analysis is the kind-agnostic analysis trigger — it embeds every photo to light
  // up the Variety / aesthetic / off-style-outlier / caption-alignment readout. An analysis prerequisite,
  // not a fix on flagged items, so it shows whenever wired (re-running is valid after a dataset edit).
  const canAnalyzeDataset = typeof onAnalyzeDataset === "function";
  // sc-6538: the face check is a PERSON-only analysis trigger, not a fix on flagged items — it runs the
  // SCRFD+ArcFace pass that produces the face sidecar the identity / no-face / small-subject checks read.
  // So it shows for a person dataset whenever wired, regardless of whether face findings exist yet.
  const canAnalyzeFaces = report.kind === "person" && typeof onAnalyzeFaces === "function";
  // sc-6540: kind-aware next-steps the user must do by hand (acquire/replace) — distinct from the
  // one-tap action buttons above.
  const recommendations = compact ? [] : datasetRecommendations(report);
  return (
    <div className={`dataset-doctor tone-${gate.tone}${compact ? " compact" : ""}`} aria-label="Dataset Doctor">
      <div className="dataset-doctor-head">
        <span className="dataset-doctor-gate">{gate.label}</span>
        {Number.isFinite(technical) ? (
          <span
            className="dataset-doctor-meter"
            role="meter"
            aria-label="Technical pass rate"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={technical}
            title={`${technical}% of photos pass the technical checks`}
          >
            <span className="dataset-doctor-meter-fill" style={{ width: `${technical}%` }} />
          </span>
        ) : null}
        {Number.isFinite(diversity) ? (
          <span
            className="dataset-doctor-meter dataset-doctor-meter-variety"
            role="meter"
            aria-label="Variety score"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={diversity}
            title={`Variety ${diversity}% — how visually varied the set is`}
          >
            <span className="dataset-doctor-meter-fill" style={{ width: `${diversity}%` }} />
          </span>
        ) : null}
        {Number.isFinite(alignment) ? (
          <span
            className="dataset-doctor-meter dataset-doctor-meter-alignment"
            role="meter"
            aria-label="Caption match score"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={alignment}
            title={`Caption match score ${alignment}% — based on current-caption image/text similarity`}
          >
            <span className="dataset-doctor-meter-fill" style={{ width: `${alignment}%` }} />
          </span>
        ) : null}
        {Number.isFinite(aesthetic) ? (
          <span
            className="dataset-doctor-meter dataset-doctor-meter-aesthetic"
            role="meter"
            aria-label="Aesthetic score"
            aria-valuemin={0}
            aria-valuemax={10}
            aria-valuenow={aesthetic}
            title={`Aesthetic ${aesthetic} / 10 — visual polish (style sets, advisory)`}
          >
            <span
              className="dataset-doctor-meter-fill"
              style={{ width: `${Math.round(aesthetic * 10)}%` }}
            />
          </span>
        ) : null}
      </div>
      <p className="dataset-doctor-summary">{datasetDoctorSummary(report)}</p>
      {!compact && (counts.warn || counts.fatal) ? (
        <div className="dataset-doctor-counts">
          {counts.fatal ? <span className="tone-fatal">{counts.fatal} blocking</span> : null}
          {counts.warn ? <span className="tone-warn">{counts.warn} to review</span> : null}
        </div>
      ) : null}
      {recaptionFlaggedIds.length ||
      removeDuplicateIds.length ||
      lowResIds.length ||
      cropLossIds.length ||
      stripExifCount ||
      canAnalyzeDataset ||
      canAnalyzeFaces ? (
        <div className="dataset-doctor-actions">
          {removeDuplicateIds.length ? (
            <button
              type="button"
              className="secondary-action"
              onClick={() => onRemoveDuplicates(removeDuplicateIds)}
            >
              Remove {removeDuplicateIds.length}{" "}
              {removeDuplicateIds.length === 1 ? "duplicate" : "duplicates"} (keeps the sharpest)
            </button>
          ) : null}
          {lowResIds.length ? (
            <button
              type="button"
              className="secondary-action"
              onClick={() => onUpscaleLowRes(lowResIds)}
            >
              Upscale {lowResIds.length} low-res{" "}
              {lowResIds.length === 1 ? "image" : "images"}
            </button>
          ) : null}
          {cropLossIds.length ? (
            <button
              type="button"
              className="secondary-action"
              onClick={() => onSmartCrop(cropLossIds)}
            >
              Smart-crop {cropLossIds.length}{" "}
              {cropLossIds.length === 1 ? "image" : "images"}
            </button>
          ) : null}
          {stripExifCount ? (
            <button type="button" className="secondary-action" onClick={() => onStripExif(stripExifIds)}>
              Strip metadata from {stripExifCount}{" "}
              {stripExifCount === 1 ? "image" : "images"}
            </button>
          ) : null}
          {recaptionFlaggedIds.length ? (
            <button
              type="button"
              className="secondary-action"
              onClick={() => onRecaptionFlagged(recaptionFlaggedIds)}
            >
              Re-caption {recaptionFlaggedIds.length}{" "}
              {recaptionFlaggedIds.length === 1 ? "flagged image" : "flagged images"}
            </button>
          ) : null}
          {canAnalyzeDataset ? (
            <button type="button" className="secondary-action" onClick={() => onAnalyzeDataset()}>
              Analyze photos
            </button>
          ) : null}
          {canAnalyzeFaces ? (
            <button type="button" className="secondary-action" onClick={() => onAnalyzeFaces()}>
              Check faces
            </button>
          ) : null}
        </div>
      ) : null}
      {recommendations.length ? (
        <ul className="dataset-doctor-recommendations" aria-label="Recommendations">
          {recommendations.map((rec) => (
            <li key={rec.id} className={`dataset-doctor-rec tone-${rec.tone}`}>
              {rec.text}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

// Round a raw metric for display — duplicates keep many decimals (Hamming is an int),
// so keep it tidy without hiding meaningful precision on the sub-one fractions.
function round(value) {
  if (!Number.isFinite(value)) {
    return value;
  }
  if (Number.isInteger(value)) {
    return value;
  }
  return Math.abs(value) < 1 ? Number(value.toFixed(3)) : Math.round(value);
}
