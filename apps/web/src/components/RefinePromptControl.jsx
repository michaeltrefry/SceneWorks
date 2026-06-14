import React, { useEffect, useState } from "react";
import { Icon } from "./Icons.jsx";

// Humanize a byte count to a "7.2 GB" label; null when the size is unknown.
function formatGb(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return null;
  }
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

// Detect the "refinement model isn't provisioned" failure (sc-5605) so we can offer a
// one-click download instead of a raw worker error. The native worker resolves an
// already-cached snapshot and fast-fails with "…snapshot is not cached…" when the model
// is absent (the retired Python path auto-downloaded). Prefer the catalog install state
// when the entry is supplied; fall back to the worker message otherwise.
function isModelMissing(refineModel, errorMessage) {
  if (refineModel?.installState) {
    return refineModel.installState === "missing";
  }
  return /not cached|not installed|snapshot is not/i.test(errorMessage ?? "");
}

// "Refine my prompt" affordance shared by Image and Video Studio (sc-2041).
// Sends the current prompt + the selected model's guide to the refinement worker
// (via the context `refinePrompt`), then shows the rewrite for review. The
// original prompt is never changed until the user clicks Apply.
//
// When the refinement model isn't provisioned, the worker fails fast; instead of a
// raw error we surface a download affordance (sc-5605). `refineModel` is the catalog
// entry for the refinement LLM (PROMPT_REFINE_MODEL_ID) and `onDownloadRefineModel`
// enqueues its ModelDownload job; both are optional so older callers still work.
export function RefinePromptControl({
  prompt,
  guidePath,
  modelId,
  workflow,
  refinePrompt,
  onApply,
  refineModel,
  onDownloadRefineModel,
}) {
  const [status, setStatus] = useState("idle"); // idle | loading | review | error
  const [refined, setRefined] = useState("");
  const [error, setError] = useState("");
  const [downloadRequested, setDownloadRequested] = useState(false);

  const trimmed = (prompt ?? "").trim();
  const busy = status === "loading";
  const disabled = busy || !trimmed || typeof refinePrompt !== "function";

  const modelMissing = status === "error" && isModelMissing(refineModel, error);
  const installState = refineModel?.installState;

  // When the refinement model finishes downloading (a catalog refresh flips its
  // installState to "installed"), clear the missing-model error so the user can retry.
  useEffect(() => {
    if (installState === "installed" && downloadRequested) {
      setDownloadRequested(false);
      setStatus((current) => (current === "error" ? "idle" : current));
      setError((current) => (current ? "" : current));
    }
  }, [installState, downloadRequested]);

  async function handleRefine() {
    setStatus("loading");
    setError("");
    try {
      // The guide is first-party context for the rewrite; fetch it best-effort and
      // refine generically if it can't be loaded.
      let guide = "";
      if (guidePath) {
        try {
          const response = await fetch(guidePath);
          if (response.ok) guide = await response.text();
        } catch {
          guide = "";
        }
      }
      const result = await refinePrompt({ prompt: trimmed, modelId, workflow, guide });
      setRefined(result);
      setStatus("review");
    } catch (err) {
      setError(err?.message || "Prompt refinement failed.");
      setStatus("error");
    }
  }

  async function handleDownloadModel() {
    if (typeof onDownloadRefineModel !== "function") return;
    try {
      // The download enqueuer resolves to the job, or null when it failed (it surfaces
      // its own error). Only switch to the "downloading" note on a real job.
      const job = await onDownloadRefineModel();
      if (job) {
        setDownloadRequested(true);
      }
    } catch (err) {
      // Surface an unexpected enqueue failure inline; the Models screen owns progress.
      setError(err?.message || "Could not start the refinement model download.");
    }
  }

  const sizeLabel = formatGb(refineModel?.downloadSizeBytes);
  const modelName = refineModel?.name || "Prompt Refiner";

  return (
    <div className="refine-control">
      <button className="hero-link refine-button" disabled={disabled} onClick={handleRefine} type="button">
        <Icon.Wand size={14} /> {busy ? "Refining…" : "Refine my prompt"}
      </button>

      {status === "error" && modelMissing ? (
        <div className="refine-missing-model" role="alert">
          {downloadRequested ? (
            <p className="refine-error">
              Downloading the prompt refinement model… track progress on the Models screen, then try Refine
              again.
            </p>
          ) : (
            <>
              <p className="refine-error">
                The prompt refinement model{sizeLabel ? ` (${sizeLabel})` : ""} isn’t installed yet.
              </p>
              {typeof onDownloadRefineModel === "function" ? (
                <button className="secondary-action" onClick={handleDownloadModel} type="button">
                  Download refinement model
                </button>
              ) : (
                <p className="refine-error">Open the Models screen to download “{modelName}”.</p>
              )}
            </>
          )}
        </div>
      ) : status === "error" ? (
        <p className="refine-error" role="alert">
          {error}
        </p>
      ) : null}

      {status === "review" ? (
        <div className="refine-review">
          <p className="refine-review-label">Suggested rewrite</p>
          <p className="refine-review-text">{refined}</p>
          <div className="refine-review-actions">
            <button
              className="secondary-action"
              onClick={() => {
                onApply(refined);
                setStatus("idle");
              }}
              type="button"
            >
              Apply
            </button>
            <button className="secondary-action" onClick={() => setStatus("idle")} type="button">
              Keep original
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
