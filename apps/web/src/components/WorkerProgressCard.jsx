import React, { useMemo } from "react";
import { useLiveJobElapsedSeconds } from "./JobProgress.jsx";
import { actionStatuses, terminalStatuses } from "../jobTypes.js";
import { formatSeconds, percent } from "../formatting.js";
import { useAppContext } from "../context/AppContext.js";
import { deriveWorkerHardware, findWorkerForJob, liveMeters } from "../workers.js";

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
      return `Training Run — ${payload.loraName ?? payload.targetName ?? payload.loraId ?? "(unnamed LoRA)"}`;
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

function HardwarePills({ device, vendor, architecture }) {
  if (!device) return null;
  return (
    <div className="worker-progress-card__hw-pills">
      <span className="worker-progress-card__pill device">{device}</span>
      {vendor ? <span className="worker-progress-card__pill vendor">{vendor}</span> : null}
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

export function WorkerProgressCard({
  job,
  onCancel,
  onRetry,
  onDuplicate,
  onOpenQueue,
  hideOpenQueue = false,
  className,
}) {
  const { workersById, visibleWorkers } = useAppContext();
  const worker = useMemo(() => {
    if (job.workerId && workersById?.get) {
      const direct = workersById.get(job.workerId);
      if (direct) return direct;
    }
    return findWorkerForJob(job, visibleWorkers ?? []);
  }, [job, workersById, visibleWorkers]);

  const hardware = useMemo(() => deriveWorkerHardware(worker), [worker]);
  const meters = useMemo(() => pickMeters(job, worker), [job, worker]);
  const elapsedSeconds = useLiveJobElapsedSeconds(job);

  const isTerminal = terminalStatuses.has(job.status);
  const attempts = job.attempts ?? 1;
  const canCancel = onCancel && !isTerminal && !job.cancelRequested;
  const canRetry = onRetry && actionStatuses.has(job.status) && attempts < MAX_ATTEMPTS;
  const canDuplicate = onDuplicate && actionStatuses.has(job.status) && attempts < MAX_ATTEMPTS;
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
            <MeterBar label="Mem" value={meters.memUsedPct} />
            <MeterBar label="Load" value={meters.loadPct} />
          </div>
        ) : null}
      </div>
      <div className="worker-progress-card__status-row">
        <span>
          <small>Stage</small>
          <strong>{defaultChipLabel(job.stage ?? job.status)}</strong>
        </span>
        <span>
          <small>Elapsed</small>
          <strong>{formatSeconds(elapsedSeconds)}</strong>
        </span>
        {attempts > 1 || actionStatuses.has(job.status) ? (
          <span>
            <small>Attempt</small>
            <strong>{attempts}/{MAX_ATTEMPTS}</strong>
          </span>
        ) : null}
      </div>
      <ProgressBar status={job.status} progress={job.progress} />
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
    </article>
  );
}
