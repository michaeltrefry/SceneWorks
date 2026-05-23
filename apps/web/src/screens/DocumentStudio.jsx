import React, { useEffect, useMemo, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { assetUrl } from "../components/assetMedia.jsx";
import { JobProgressCard } from "../components/JobProgress.jsx";

const MAX_IMAGES_DEFAULT = 6;
const MAX_IMAGES_LIMIT = 10;

function modelSupportsInterleave(model) {
  return Array.isArray(model?.capabilities) && model.capabilities.includes("interleave");
}

function DocumentResult({ job, assets, projectId, onOpenQueue }) {
  const segments = job.result?.segments;
  if (job.status !== "completed" || !Array.isArray(segments) || !segments.length) {
    return <JobProgressCard job={job} label="Interleaved document" onOpenQueue={onOpenQueue} />;
  }
  return (
    <article className="document-view" aria-label="Generated document">
      {segments.map((segment, index) => {
        if (segment.type === "text") {
          return (
            <p className="document-text" key={`segment-${index}`}>
              {segment.text}
            </p>
          );
        }
        const asset = assets.find((item) => item.id === segment.assetId);
        const src = assetUrl(asset ?? { projectId, file: { path: segment.path } });
        return src ? <img alt="" className="document-image" key={`segment-${index}`} src={src} /> : null;
      })}
    </article>
  );
}

export function DocumentStudio({
  activeProject,
  assets,
  createInterleaveJob,
  gpuOptions,
  imageModels,
  jobs,
  onOpenQueue,
  requestedGpu,
  setRequestedGpu,
}) {
  const interleaveModels = useMemo(
    () => (imageModels ?? []).filter(modelSupportsInterleave),
    [imageModels],
  );
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [sourceAssetIds, setSourceAssetIds] = useState([]);
  const [maxImages, setMaxImages] = useState(MAX_IMAGES_DEFAULT);
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

  const latestJob = useMemo(() => {
    return (jobs ?? []).find((job) => job.type === "image_interleave") ?? null;
  }, [jobs]);

  const ready = Boolean(activeProject) && interleaveModels.length > 0;
  const canSubmit = ready && prompt.trim().length > 0 && !submitting;

  async function submit(event) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setSubmitting(true);
    const job = await createInterleaveJob({
      prompt: prompt.trim(),
      model: model || undefined,
      maxImages: Number(maxImages) || MAX_IMAGES_DEFAULT,
      sourceAssetIds,
    });
    setSubmitting(false);
    if (job) {
      onOpenQueue?.();
    }
  }

  return (
    <div className="studio document-studio">
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

        <button className="primary-action" disabled={!canSubmit} type="submit">
          {submitting ? "Submitting…" : "Compose document"}
        </button>
      </form>

      <section className="studio-results">
        {latestJob ? (
          <DocumentResult
            assets={assets ?? []}
            job={latestJob}
            onOpenQueue={onOpenQueue}
            projectId={activeProject?.id}
          />
        ) : (
          <p className="empty-panel">Your generated document will appear here.</p>
        )}
      </section>
    </div>
  );
}
