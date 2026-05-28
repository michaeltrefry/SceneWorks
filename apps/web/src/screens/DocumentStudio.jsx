import React, { useEffect, useMemo, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { DocumentView } from "../components/DocumentView.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import {
  DEFAULT_INTERLEAVE_RESOLUTION,
  DEFAULT_INTERLEAVE_SYSTEM_MESSAGE,
  INTERLEAVE_RESOLUTION_OPTIONS,
} from "../constants.js";
import { useAppContext } from "../context/AppContext.js";
import { selectStackedJobs } from "./generationStudio.jsx";

const MAX_IMAGES_DEFAULT = 6;
const MAX_IMAGES_LIMIT = 10;

function modelSupportsInterleave(model) {
  return Array.isArray(model?.capabilities) && model.capabilities.includes("interleave");
}

function formatResolutionLabel(value) {
  const [width, height] = String(value).split("x");
  return height ? `${width} × ${height}` : value;
}

function documentSegments(job) {
  const segments = job.result?.segments;
  return Array.isArray(segments) && segments.length ? segments : null;
}

export function DocumentStudio() {
  const {
    activeProject,
    assets,
    createInterleaveJob,
    documentLocalJobs = [],
    gpuOptions,
    imageModels,
    jobAction,
    rememberLocalGenerationJob,
    setActiveView,
    requestedGpu,
    setRequestedGpu,
  } = useAppContext();
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onOpenQueue = () => setActiveView("Queue");
  const interleaveModels = useMemo(
    () => (imageModels ?? []).filter(modelSupportsInterleave),
    [imageModels],
  );
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [sourceAssetIds, setSourceAssetIds] = useState([]);
  const [maxImages, setMaxImages] = useState(MAX_IMAGES_DEFAULT);
  const [resolution, setResolution] = useState(DEFAULT_INTERLEAVE_RESOLUTION);
  const [systemMessage, setSystemMessage] = useState(DEFAULT_INTERLEAVE_SYSTEM_MESSAGE);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (interleaveModels.length && !interleaveModels.some((item) => item.id === model)) {
      setModel(interleaveModels[0].id);
    }
  }, [interleaveModels, model]);

  const sourceImageAssets = useMemo(
    () => (assets ?? []).filter((asset) => asset.type === "image" || asset.type === "frame" || asset.type === "upload"),
    [assets],
  );

  // Running and queued compose runs stack (oldest/active on top, queued below) and
  // each run streams its output beneath it, mirroring the Image and Video studios.
  const localJobs = useMemo(() => selectStackedJobs(documentLocalJobs), [documentLocalJobs]);

  const ready = Boolean(activeProject) && interleaveModels.length > 0;
  const canSubmit = ready && prompt.trim().length > 0 && !submitting;

  async function submit(event) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setSubmitting(true);
    const [width, height] = resolution.split("x").map((value) => Number(value));
    const trimmedSystem = systemMessage.trim();
    const job = await createInterleaveJob({
      prompt: prompt.trim(),
      model: model || undefined,
      maxImages: Number(maxImages) || MAX_IMAGES_DEFAULT,
      width,
      height,
      sourceAssetIds,
      // Only send the system prompt when edited; blank/default lets the worker use
      // its own _INTERLEAVE_SYSTEM_MESSAGE.
      advanced:
        trimmedSystem && trimmedSystem !== DEFAULT_INTERLEAVE_SYSTEM_MESSAGE
          ? { systemMessage: trimmedSystem }
          : {},
    });
    setSubmitting(false);
    if (job) {
      // Stack the run in the studio instead of routing to the Queue, so its output
      // streams in below the prompt as it composes.
      rememberLocalGenerationJob?.("document", job);
    }
  }

  return (
    <section className="main-surface document-studio">
      <form className="studio-form" onSubmit={submit}>
        {interleaveModels.length ? null : (
          <p className="empty-panel compact-panel">
            Install a SenseNova-U1 model to generate interleaved text-image documents.
          </p>
        )}
        <label className="field">
          <span>Prompt</span>
          <textarea
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Write an illustrated guide, storyboard, or tutorial…"
            rows={4}
            value={prompt}
          />
        </label>

        <div className="field-row">
          <label className="field">
            <span>Model</span>
            <select onChange={(event) => setModel(event.target.value)} value={model}>
              {interleaveModels.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name ?? item.id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Size</span>
            <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
              {INTERLEAVE_RESOLUTION_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {formatResolutionLabel(option)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Max images</span>
            <input
              max={MAX_IMAGES_LIMIT}
              min={1}
              onChange={(event) => setMaxImages(event.target.value)}
              type="number"
              value={maxImages}
            />
          </label>
          {gpuOptions?.length ? (
            <label className="field">
              <span>GPU</span>
              <select onChange={(event) => setRequestedGpu?.(event.target.value)} value={requestedGpu}>
                {gpuOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>

        <AssetPickerField
          assets={sourceImageAssets}
          buttonLabel="Add reference images"
          changeLabel="Change references"
          emptyLabel="No reference images (optional)"
          label="Reference images (optional)"
          multiple
          onChange={setSourceAssetIds}
          values={sourceAssetIds}
        />

        <label className="field document-system-prompt">
          <span>System prompt</span>
          <small>Steers the model's think / no-think composition. Prefilled with the default — edit to change behavior.</small>
          <textarea
            onChange={(event) => setSystemMessage(event.target.value)}
            rows={6}
            value={systemMessage}
          />
          {systemMessage !== DEFAULT_INTERLEAVE_SYSTEM_MESSAGE ? (
            <button
              className="secondary-action"
              onClick={() => setSystemMessage(DEFAULT_INTERLEAVE_SYSTEM_MESSAGE)}
              type="button"
            >
              Reset to default
            </button>
          ) : null}
        </label>

        <button className="primary-action" disabled={!canSubmit} type="submit">
          {submitting ? "Submitting…" : "Compose document"}
        </button>
      </form>

      <section className="studio-results">
        {localJobs.length ? (
          <div className="local-job-stack">
            {localJobs.map((job) => {
              const segments = job.status === "completed" ? documentSegments(job) : null;
              return (
                <article className="local-job-group" key={job.id}>
                  {segments ? (
                    <DocumentView assets={assets ?? []} projectId={activeProject?.id} segments={segments} />
                  ) : (
                    <WorkerProgressCard job={job} onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                  )}
                </article>
              );
            })}
          </div>
        ) : (
          <p className="empty-panel">Your generated document will appear here.</p>
        )}
      </section>
    </section>
  );
}
