import React, { useEffect, useMemo, useRef, useState } from "react";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { Logo } from "../components/Logo.jsx";
import { terminalStatuses } from "../constants.js";

// Models flagged "Recommended" in the wizard: the fast image target and LTX-2.3
// as the headline video model. Anything not present in the live catalog is
// silently ignored, so this stays correct as the catalog evolves.
const RECOMMENDED_MODEL_IDS = new Set(["z_image_turbo", "ltx_2_3"]);
// Recommended models are pre-checked — except very large ones (e.g. LTX-2.3 is
// ~146 GB). Those stay recommended but unchecked so a new user opts into the big
// download deliberately instead of having it auto-queued on first launch.
const AUTO_SELECT_MAX_BYTES = 50 * 1024 * 1024 * 1024;

function isDownloadable(model) {
  return model.downloadable !== false && Boolean(model.downloads?.[0]?.repo ?? model.repo);
}

function isHugeDownload(model) {
  return typeof model.downloadSizeBytes === "number" && model.downloadSizeBytes > AUTO_SELECT_MAX_BYTES;
}

function downloadSizeText(model) {
  if (!model.downloadSizeLabel) {
    return "Size unavailable";
  }
  return model.downloadSizeEstimated ? `~${model.downloadSizeLabel}` : model.downloadSizeLabel;
}

function defaultSelection(models) {
  return new Set(
    models
      .filter(
        (model) =>
          isDownloadable(model) &&
          model.installState !== "installed" &&
          RECOMMENDED_MODEL_IDS.has(model.id) &&
          !isHugeDownload(model),
      )
      .map((model) => model.id),
  );
}

const TYPE_LABELS = { image: "Image models", video: "Video models" };

export function SetupWizard({ models, jobs, onDownloadModel, onCreateProject, onComplete, onOpenQueue }) {
  const [step, setStep] = useState("models");
  const [selected, setSelected] = useState(() => defaultSelection(models));
  const [started, setStarted] = useState(() => new Set());
  const [projectName, setProjectName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const initializedRef = useRef(false);

  // The catalog may arrive a tick after mount; seed the recommended selection
  // once it does (without clobbering a choice the user already made).
  useEffect(() => {
    if (!initializedRef.current && models.length) {
      setSelected(defaultSelection(models));
      initializedRef.current = true;
    }
  }, [models]);

  const downloadable = useMemo(() => models.filter(isDownloadable), [models]);
  const grouped = useMemo(() => {
    const byType = new Map();
    for (const model of downloadable) {
      const key = model.type ?? "other";
      if (!byType.has(key)) {
        byType.set(key, []);
      }
      byType.get(key).push(model);
    }
    return byType;
  }, [downloadable]);

  const activeDownloadJobs = useMemo(
    () => jobs.filter((job) => job.type === "model_download" && !terminalStatuses.has(job.status)),
    [jobs],
  );

  const pendingSelection = useMemo(
    () =>
      downloadable.filter(
        (model) => selected.has(model.id) && model.installState !== "installed" && !started.has(model.id),
      ),
    [downloadable, selected, started],
  );

  function toggle(model) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(model.id)) {
        next.delete(model.id);
      } else {
        next.add(model.id);
      }
      return next;
    });
  }

  function downloadSelected() {
    pendingSelection.forEach((model) => onDownloadModel(model));
    setStarted((current) => new Set([...current, ...pendingSelection.map((model) => model.id)]));
  }

  async function finish(event) {
    event.preventDefault();
    const trimmed = projectName.trim();
    if (!trimmed || submitting) {
      return;
    }
    setSubmitting(true);
    try {
      const created = await onCreateProject(trimmed);
      if (created) {
        await onComplete();
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="setup-wizard">
      <div className="setup-wizard-card">
        <span className="setup-wizard-mark" aria-hidden="true">
          <Logo size={48} />
        </span>
        <ol className="setup-wizard-steps" aria-hidden="true">
          <li className={step === "models" ? "active" : "done"}>Models</li>
          <li className={step === "project" ? "active" : ""}>First project</li>
        </ol>

        {step === "models" ? (
          <>
            <h2>Download starter models</h2>
            <p className="setup-wizard-lede">
              Pick the models to download now. Recommended ones are pre-selected — you can add
              more later from Models. Downloads run in the background, so you can keep going.
            </p>

            <div className="setup-wizard-models">
              {downloadable.length === 0 ? (
                <p className="setup-wizard-empty">No downloadable models in the catalog yet.</p>
              ) : (
                [...grouped.entries()].map(([type, items]) => (
                  <div className="setup-wizard-group" key={type}>
                    <h3>{TYPE_LABELS[type] ?? type}</h3>
                    {items.map((model) => {
                      const installed = model.installState === "installed";
                      const downloading = started.has(model.id) && !installed;
                      const recommended = RECOMMENDED_MODEL_IDS.has(model.id);
                      return (
                        <label className={`setup-wizard-model${installed ? " installed" : ""}`} key={model.id}>
                          <input
                            type="checkbox"
                            checked={installed || selected.has(model.id)}
                            disabled={installed || downloading}
                            onChange={() => toggle(model)}
                          />
                          <span className="setup-wizard-model-main">
                            <span className="setup-wizard-model-name">
                              {model.name}
                              {recommended && !installed ? <span className="setup-wizard-tag">Recommended</span> : null}
                            </span>
                            <span className="setup-wizard-model-meta">
                              {installed ? "Already installed" : downloading ? "Download started" : downloadSizeText(model)}
                            </span>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                ))
              )}
            </div>

            {activeDownloadJobs.length ? (
              <div className="setup-wizard-progress">
                {activeDownloadJobs.map((job) => (
                  <WorkerProgressCard job={job} key={job.id} onOpenQueue={onOpenQueue} />
                ))}
              </div>
            ) : null}

            <div className="setup-wizard-actions">
              <button
                className="setup-wizard-secondary"
                disabled={pendingSelection.length === 0}
                onClick={downloadSelected}
                type="button"
              >
                {pendingSelection.length ? `Download ${pendingSelection.length} selected` : "Download selected"}
              </button>
              <button className="setup-wizard-cta" onClick={() => setStep("project")} type="button">
                Continue
              </button>
            </div>
          </>
        ) : (
          <>
            <h2>Create your first project</h2>
            <p className="setup-wizard-lede">
              SceneWorks keeps your images, videos, characters, and timelines inside a project.
              Name your first one to jump into the studio.
            </p>
            <form className="setup-wizard-form" onSubmit={finish}>
              <input
                aria-label="Project name"
                autoFocus
                disabled={submitting}
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="e.g. My First Project"
                value={projectName}
              />
              <button className="setup-wizard-cta" disabled={submitting || !projectName.trim()} type="submit">
                {submitting ? "Setting up…" : "Finish setup"}
              </button>
            </form>
            <button className="setup-wizard-back" onClick={() => setStep("models")} type="button">
              ← Back to models
            </button>
          </>
        )}
      </div>
    </section>
  );
}
