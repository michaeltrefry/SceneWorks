import React, { useEffect, useMemo, useState } from "react";
import { actionStatuses, terminalStatuses } from "../jobTypes.js";
import { formatSeconds, liveElapsedSeconds, percent } from "../formatting.js";
import { useAppContext } from "../context/AppContext.js";
import { deriveWorkerHardware, findWorkerForJob, liveMeters } from "../workers.js";
import { AssetMedia, AssetThumbnail, assetUrl, posterUrl } from "./assetMedia.jsx";

// Live-ticking elapsed seconds for in-flight jobs. Re-exported here after the
// sc-2093 cleanup deleted the legacy JobProgress.jsx that originally housed it.
export function useLiveJobElapsedSeconds(job) {
  const active = !terminalStatuses.has(job.status) && Boolean(job.startedAt);
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!active) {
      return undefined;
    }
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active, job.startedAt]);

  return liveElapsedSeconds(job, nowMs);
}

// WorkerProgressCard — unified worker/job progress component (sc-2083).
// Renders the same 6-row skeleton everywhere (Image Studio, Video Studio,
// Queue, Training Studio, Character Angle Set, Model/LoRA Import). The
// thumbnail region is added in sc-2084; this slice covers everything above
// it. See docs/design/worker-progress-card.md.

const MAX_ATTEMPTS = 5;
const PROMPT_TITLE_TRUNCATE = 80;
const SHORT_PROMPT_TITLE_TRUNCATE = 60;

// Human label for the Job Type chip. Keep this aligned with the design spec
// table at docs/design/worker-progress-card.md and the enum in jobTypes.js.
const JOB_TYPE_CHIP = {
  image_generate: "Generate Image",
  image_edit: "Generate Image",
  image_vqa: "Generate Image",
  image_interleave: "Generate Image",
  video_generate: "Generate Video",
  video_extend: "Generate Video",
  video_bridge: "Generate Video",
  person_replace: "Person Replace",
  lora_train: "Training Run",
  training_caption: "Dataset Captioning",
  model_download: "Model Import",
  model_import: "Model Import",
  model_convert: "Model Import",
  lora_import: "LoRA Import",
  prompt_refine: "Prompt Refine",
  person_detect: "Person Detect",
  person_track: "Person Track",
  person_segment: "Person Segment",
};

// CSS modifier slug per type (lowercased + dash-cased) for the chip color.
function chipModifier(type) {
  return String(type ?? "job").replaceAll("_", "-");
}

// Status badge label per design spec.
const STATUS_LABEL = {
  queued: "Queued",
  running: "Running",
  completed: "Complete",
  canceled: "Cancelled",
  failed: "Failed",
  interrupted: "Interrupted",
};

export function getJobTypeChip(type) {
  return JOB_TYPE_CHIP[type] ?? defaultChipLabel(type);
}

function defaultChipLabel(type) {
  const label = String(type ?? "job").replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function truncatePrompt(text, max = PROMPT_TITLE_TRUNCATE) {
  if (!text) return "";
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}

// Title derivation. After sc-2087 lands the server writes `job.title` directly.
// Until then, derive client-side from payload fields per the design spec. The
// fallbacks here intentionally mirror the table at docs/design/worker-progress-card.md
// so behavior matches once the server-side enrichment ships.
export function deriveJobTitle(job) {
  if (job?.title) return job.title;
  const payload = job?.payload ?? {};
  switch (job?.type) {
    case "lora_train":
      return `Training Run — ${payload.loraName ?? payload.outputName ?? payload.targetName ?? payload.plan?.output?.loraId ?? payload.loraId ?? "(unnamed LoRA)"}`;
    case "training_caption":
      return `Dataset Captioning — ${payload.datasetName ?? payload.datasetId ?? "(unnamed dataset)"}`;
    case "model_download":
    case "model_import":
    case "model_convert":
      return `Model Import — ${payload.modelName ?? payload.filename ?? payload.modelId ?? "(unnamed)"}`;
    case "lora_import":
      return `LoRA Import — ${payload.loraName ?? payload.filename ?? payload.loraId ?? "(unnamed)"}`;
    case "prompt_refine":
      return `Prompt Refine — ${truncatePrompt(payload.prompt ?? "", SHORT_PROMPT_TITLE_TRUNCATE) || "(empty prompt)"}`;
    case "image_generate":
    case "image_edit":
    case "image_vqa":
    case "image_interleave":
      if (payload.characterId && payload.characterName) {
        return `Character Turnaround — ${payload.characterName}`;
      }
      return `Generate Image — ${truncatePrompt(payload.prompt ?? "") || "(no prompt)"}`;
    case "video_generate":
    case "video_extend":
    case "video_bridge":
      return `Generate Video — ${truncatePrompt(payload.prompt ?? "") || "(no prompt)"}`;
    case "person_replace":
      return `Person Replace — ${truncatePrompt(payload.prompt ?? "") || "(no prompt)"}`;
    default:
      return defaultChipLabel(job?.type ?? "job");
  }
}

function shortJobId(id) {
  if (!id || id.length <= 12) return id;
  return `${id.slice(0, 6)}…${id.slice(-4)}`;
}

// Per-job static peak meters (set by sc-2086). When absent, in-flight jobs
// fall back to live worker utilization (sc-2082); completed jobs simply hide
// the meters.
function pickMeters(job, worker) {
  const isTerminal = terminalStatuses.has(job.status);
  if (isTerminal) {
    const memUsedPct = Number(job.peakGpuMemoryPct);
    const loadPct = Number(job.peakGpuLoadPct);
    return {
      memUsedPct: Number.isFinite(memUsedPct) ? memUsedPct : null,
      loadPct: Number.isFinite(loadPct) ? loadPct : null,
      source: "peak",
    };
  }
  return { ...liveMeters(worker), source: "live" };
}

function StatusBadge({ status }) {
  const label = STATUS_LABEL[status] ?? defaultChipLabel(status);
  return <span className={`status-badge worker-progress-card__status ${status}`}>{label}</span>;
}

function HardwarePills({ device, gpuLabel, architecture }) {
  if (!device) return null;
  return (
    <div className="worker-progress-card__hw-info">
      <span className="worker-progress-card__pill device">{device}</span>
      {gpuLabel ? <span className="worker-progress-card__hw-name" title={gpuLabel}>{gpuLabel}</span> : null}
      {architecture ? (
        <span className={`worker-progress-card__pill arch arch-${architecture}`}>{architecture}</span>
      ) : null}
    </div>
  );
}

function MeterBar({ label, value }) {
  if (value === null || value === undefined) {
    return (
      <div className="worker-progress-card__meter empty" aria-label={`${label} unknown`}>
        <span className="worker-progress-card__meter-label">{label}</span>
        <span className="worker-progress-card__meter-bar" />
        <span className="worker-progress-card__meter-value">—</span>
      </div>
    );
  }
  const rounded = Math.round(value);
  return (
    <div className="worker-progress-card__meter" aria-label={`${label} ${rounded}%`}>
      <span className="worker-progress-card__meter-label">{label}</span>
      <span className="worker-progress-card__meter-bar">
        <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
      </span>
      <span className="worker-progress-card__meter-value">{rounded}%</span>
    </div>
  );
}

function ProgressBar({ status, progress }) {
  const isTerminal = terminalStatuses.has(status);
  const isRunning = status === "running";
  const numeric = Number(progress);
  const hasValue = Number.isFinite(numeric) && numeric > 0;
  if (!hasValue && isRunning) {
    return <div className="progress-track worker-progress-card__progress indeterminate" aria-label="Working" />;
  }
  if (isTerminal && !hasValue) {
    return null;
  }
  const label = percent(progress);
  return (
    <div className="progress-track worker-progress-card__progress" aria-label={`${label} complete`}>
      <span style={{ width: label }} />
    </div>
  );
}

// Thumbnails region (sc-2084). Four variants — image-grid, video-player,
// small-row, hidden — covering Image Studio (large grid), Video Studio
// (single player), Queue + batch (compact row), and Caption/Import (no
// thumbnails). Interim thumbnails are emitted by image-producing workers in
// sc-2085; until that ships interimAssets is just empty.
const THUMBNAIL_VARIANTS = new Set(["image-grid", "video-player", "small-row", "hidden"]);

function mergeThumbnails(finalAssets, interimAssets) {
  const finalArray = Array.isArray(finalAssets) ? finalAssets : [];
  const interimArray = Array.isArray(interimAssets) ? interimAssets : [];
  if (interimArray.length === 0) return finalArray;
  if (finalArray.length === 0) return interimArray;
  // Dedupe by id; finals supersede interims.
  const finalIds = new Set(finalArray.map((asset) => asset?.id).filter(Boolean));
  const survivingInterim = interimArray.filter((asset) => !finalIds.has(asset?.id));
  return [...finalArray, ...survivingInterim];
}

function ThumbnailGrid({ assets, variant, onThumbnailClick, isRunning, expectedCount }) {
  const items = Array.isArray(assets) ? assets : [];
  const cellClass = variant === "small-row" ? "worker-progress-card__thumb-cell small" : "worker-progress-card__thumb-cell";
  const skeletonCount = Math.max(0, (expectedCount ?? 0) - items.length);
  const showSkeletons = isRunning && skeletonCount > 0;
  if (items.length === 0 && !showSkeletons) {
    return null;
  }
  return (
    <div
      className={`worker-progress-card__thumbnails worker-progress-card__thumbnails--${variant}`}
      role="group"
      aria-label="Job output"
    >
      {items.map((asset, index) => {
        const interactive = !!onThumbnailClick;
        const inner = <AssetThumbnail asset={asset} className="worker-progress-card__thumb-media" />;
        const label = asset.displayName && variant === "image-grid" ? (
          <small className="worker-progress-card__thumb-label">{asset.displayName}</small>
        ) : null;
        const key = asset.id ?? `interim-${index}`;
        const isInterim = asset.__interim === true;
        // Discarded (trashed-but-not-purged) assets stay in the catalog, so the
        // resolved record carries the live status. Dim them so they read as
        // "set aside" without hiding what the run produced.
        const isDiscarded = asset.status?.trashed === true;
        const cellClasses = `${cellClass}${isInterim ? " interim" : ""}${isDiscarded ? " discarded" : ""}`;
        return interactive ? (
          <button
            key={key}
            className={cellClasses}
            type="button"
            onClick={() => onThumbnailClick(asset)}
            aria-label={`${asset.displayName ?? "Open asset"}${isDiscarded ? " (discarded)" : ""}`}
          >
            {inner}
            {label}
          </button>
        ) : (
          <span key={key} className={cellClasses}>
            {inner}
            {label}
          </span>
        );
      })}
      {showSkeletons
        ? Array.from({ length: skeletonCount }, (_, i) => (
            <span key={`skel-${i}`} className={`${cellClass} skeleton`} aria-hidden="true" />
          ))
        : null}
    </div>
  );
}

function ThumbnailGroups({ groups, variant, onThumbnailClick }) {
  const sections = Array.isArray(groups) ? groups.filter((group) => Array.isArray(group?.assets) && group.assets.length) : [];
  if (!sections.length) {
    return null;
  }
  return (
    <div className="worker-progress-card__thumbnail-groups" role="group" aria-label="Job output samples">
      {sections.map((group, index) => (
        <section
          className="worker-progress-card__thumbnail-group"
          key={group.id ?? `${group.label ?? "sample"}-${index}`}
        >
          {group.label ? (
            <div className="worker-progress-card__thumbnail-group-label">{group.label}</div>
          ) : null}
          <ThumbnailGrid
            assets={group.assets}
            variant={variant}
            onThumbnailClick={onThumbnailClick}
            isRunning={false}
            expectedCount={0}
          />
        </section>
      ))}
    </div>
  );
}

function VideoThumbnail({ assets, onThumbnailClick }) {
  const asset = Array.isArray(assets) ? assets[0] : null;
  if (!asset) {
    return (
      <div
        className="worker-progress-card__thumbnails worker-progress-card__thumbnails--video-player empty"
        role="group"
        aria-label="Video output pending"
      >
        <span className="worker-progress-card__video-placeholder">Rendering…</span>
      </div>
    );
  }
  // While encoding, the asset may have a poster but no playable src yet; the
  // AssetMedia component handles both cases. Click-through opens the modal.
  const interactive = !!onThumbnailClick;
  const src = assetUrl(asset);
  const poster = posterUrl(asset);
  return (
    <div
      className="worker-progress-card__thumbnails worker-progress-card__thumbnails--video-player"
      role="group"
      aria-label="Job output"
    >
      {src ? (
        <AssetMedia asset={asset} className="worker-progress-card__video" />
      ) : poster ? (
        <img alt="" className="worker-progress-card__video" src={poster} />
      ) : (
        <span className="worker-progress-card__video-placeholder">Rendering…</span>
      )}
      {interactive ? (
        <button
          type="button"
          className="worker-progress-card__video-open"
          onClick={() => onThumbnailClick(asset)}
        >
          Open
        </button>
      ) : null}
    </div>
  );
}

function ThumbnailsRegion({ variant, finalAssets, interimAssets, thumbnailGroups, onThumbnailClick, isRunning, expectedCount }) {
  if (variant === "hidden") return null;
  if (!THUMBNAIL_VARIANTS.has(variant)) return null;
  if (variant === "video-player") {
    return <VideoThumbnail assets={finalAssets} onThumbnailClick={onThumbnailClick} />;
  }
  if (Array.isArray(thumbnailGroups) && thumbnailGroups.some((group) => Array.isArray(group?.assets) && group.assets.length)) {
    return <ThumbnailGroups groups={thumbnailGroups} variant={variant} onThumbnailClick={onThumbnailClick} />;
  }
  const merged = mergeThumbnails(finalAssets, interimAssets);
  return (
    <ThumbnailGrid
      assets={merged}
      variant={variant}
      onThumbnailClick={onThumbnailClick}
      isRunning={isRunning}
      expectedCount={expectedCount}
    />
  );
}

export function WorkerProgressCard({
  job,
  onCancel,
  onRetry,
  onFreshRetry,
  onDuplicate,
  onOpenQueue,
  hideOpenQueue = false,
  className,
  thumbnailsVariant = "hidden",
  thumbnailAssets,
  thumbnailGroups,
  interimThumbnailAssets,
  expectedThumbnailCount,
  onThumbnailClick,
}) {
  const { workersById, visibleWorkers } = useAppContext();
  const worker = useMemo(() => {
    if (job.workerId && workersById?.get) {
      const direct = workersById.get(job.workerId);
      if (direct) return direct;
    }
    return findWorkerForJob(job, visibleWorkers ?? []);
  }, [job, workersById, visibleWorkers]);

  // Architecture pill prefers the job's reported `backend` (the actual runtime
  // — mlx vs mps vs cuda) when present; falls back to a heuristic on the
  // worker's gpuName for legacy / pre-backend snapshots. This lets an MLX
  // adapter run show "mlx" while a Diffusers-on-MPS run on the same worker
  // shows "mps".
  const hardware = useMemo(() => {
    const derived = deriveWorkerHardware(worker);
    if (job.backend) {
      return { ...derived, architecture: job.backend };
    }
    return derived;
  }, [worker, job.backend]);
  const meters = useMemo(() => pickMeters(job, worker), [job, worker]);
  const elapsedSeconds = useLiveJobElapsedSeconds(job);

  const isTerminal = terminalStatuses.has(job.status);
  const attempts = job.attempts ?? 1;
  const isModelDownload = job.type === "model_download";
  const canCancel = onCancel && !isTerminal && !job.cancelRequested;
  const canRetryBase = actionStatuses.has(job.status) && attempts < MAX_ATTEMPTS;
  const canRetry = onRetry && canRetryBase && !isModelDownload;
  const canResumeDownload = onRetry && canRetryBase && isModelDownload && job.status !== "completed";
  const canFreshDownload =
    onFreshRetry &&
    canRetryBase &&
    isModelDownload &&
    job.status !== "completed" &&
    (attempts > 1 || ["resume", "fresh"].includes(job.payload?.downloadAction));
  const canDuplicate = onDuplicate && canRetryBase && !isModelDownload;
  const showOpenQueue = !!onOpenQueue && !hideOpenQueue;

  const chipLabel = getJobTypeChip(job.type);
  const title = deriveJobTitle(job);
  const idShort = shortJobId(job.id);

  return (
    <article className={`worker-progress-card ${job.status}${className ? ` ${className}` : ""}`}>
      <header className="worker-progress-card__header">
        <span className={`worker-progress-card__type chip-${chipModifier(job.type)}`}>{chipLabel}</span>
        <StatusBadge status={job.status} />
      </header>
      <div className="worker-progress-card__title-row">
        <h3 className="worker-progress-card__title" title={title}>{title}</h3>
        <code className="worker-progress-card__id" title={job.id}>{idShort}</code>
      </div>
      <div className="worker-progress-card__hardware">
        <HardwarePills {...hardware} />
        {hardware.device === "GPU" ? (
          <div className="worker-progress-card__meters" data-meter-source={meters.source}>
            <MeterBar label="Mem:" value={meters.memUsedPct} />
            <MeterBar label="Load:" value={meters.loadPct} />
          </div>
        ) : null}
      </div>
      <div className="worker-progress-card__status-row">
        <span className="worker-progress-card__status-stage">
          <small>Stage:</small>
          <strong>{defaultChipLabel(job.stage ?? job.status)}</strong>
        </span>
        {job.message ? (
          <span className="worker-progress-card__status-message" title={job.message}>
            {job.message}
          </span>
        ) : (
          <span aria-hidden="true" />
        )}
        <span className="worker-progress-card__status-right">
          {attempts > 1 || actionStatuses.has(job.status) ? (
            <span className="worker-progress-card__status-attempt">
              <small>Attempt:</small>
              <strong>{attempts}/{MAX_ATTEMPTS}</strong>
            </span>
          ) : null}
          <span>
            <small>Elapsed:</small>
            <strong>{formatSeconds(elapsedSeconds)}</strong>
          </span>
        </span>
      </div>
      <ProgressBar status={job.status} progress={job.progress} />
      {job.error ? (
        <p className="worker-progress-card__message error">{job.error}</p>
      ) : null}
      <div className="worker-progress-card__actions">
        {canCancel ? (
          <button
            className="secondary-action danger"
            disabled={job.cancelRequested}
            onClick={() => onCancel(job)}
            type="button"
          >
            {job.cancelRequested ? "Canceling…" : "Cancel"}
          </button>
        ) : null}
        {canResumeDownload ? (
          <button
            className="secondary-action"
            onClick={() => onRetry(job, { payloadChanges: { downloadAction: "resume" } })}
            type="button"
          >
            Resume Download
          </button>
        ) : null}
        {canFreshDownload ? (
          <button
            className="secondary-action"
            onClick={() => onFreshRetry(job, { payloadChanges: { downloadAction: "fresh" } })}
            type="button"
          >
            Retry Download
          </button>
        ) : null}
        {canRetry ? (
          <button className="secondary-action" onClick={() => onRetry(job)} type="button">Retry</button>
        ) : null}
        {canDuplicate ? (
          <button className="secondary-action" onClick={() => onDuplicate(job)} type="button">Duplicate</button>
        ) : null}
        {showOpenQueue ? (
          <button className="secondary-action" onClick={() => onOpenQueue(job)} type="button">View in Queue</button>
        ) : null}
      </div>
      <ThumbnailsRegion
        variant={thumbnailsVariant}
        finalAssets={thumbnailAssets}
        thumbnailGroups={thumbnailGroups}
        interimAssets={interimThumbnailAssets}
        onThumbnailClick={onThumbnailClick}
        isRunning={!isTerminal}
        expectedCount={expectedThumbnailCount}
      />
    </article>
  );
}
