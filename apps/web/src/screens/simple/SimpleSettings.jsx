import React, { useMemo } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { Icon } from "../../components/Icons.jsx";
import { terminalStatuses } from "../../constants.js";

// Plain-language fallback when a catalog model has no ui.description.
function describeModel(model) {
  if (model.ui?.description) return model.ui.description;
  if (model.type === "video") return "For making videos.";
  if (model.type === "image") return "For making pictures.";
  return "Helper model.";
}

export function SimpleSettings() {
  const {
    uiMode,
    setUiMode,
    setActiveView,
    models = [],
    jobs = [],
    createModelDownloadJob,
  } = useAppContext();

  // The curated getting-started set (manifest `recommended`). Image first.
  const recommended = useMemo(() => {
    const order = { image: 0, video: 1, utility: 2 };
    return models
      .filter((model) => model.recommended)
      .slice()
      .sort((a, b) => (order[a.type] ?? 9) - (order[b.type] ?? 9));
  }, [models]);

  function downloadJobFor(model) {
    return jobs.find(
      (job) => job.type === "model_download" && job.payload?.modelId === model.id && !terminalStatuses.has(job.status),
    );
  }

  return (
    <section className="main-surface sw-make">
      <div className="sw-settings">
        <section className="sw-card sw-sect">
          <h2 className="sw-sect-h">Mode</h2>
          <div className="sw-line">
            <div className="sw-line-label">
              <b>Interface</b>
              <small>Simple is friendly and guided. Advanced unlocks every control.</small>
            </div>
            <div className="sw-toggle" role="group" aria-label="Interface mode">
              <button
                type="button"
                className={uiMode === "simple" ? "on" : ""}
                aria-pressed={uiMode === "simple"}
                onClick={() => setUiMode?.("simple")}
              >
                Simple
              </button>
              <button
                type="button"
                className={uiMode === "advanced" ? "on" : ""}
                aria-pressed={uiMode === "advanced"}
                onClick={() => setUiMode?.("advanced")}
              >
                Advanced
              </button>
            </div>
          </div>
        </section>

        <section className="sw-card sw-sect">
          <h2 className="sw-sect-h">Models</h2>
          {recommended.length === 0 ? (
            <p className="sw-rendering">Models appear here once the engine is running.</p>
          ) : (
            recommended.map((model) => {
              const installed = model.installState === "installed";
              const job = downloadJobFor(model);
              const pct = Number.isFinite(job?.progress) ? Math.round(job.progress * 100) : null;
              return (
                <div className="sw-mrow" key={model.id}>
                  <span className="sw-mi"><Icon.Model /></span>
                  <div className="sw-mmeta">
                    <b>{model.name}</b>
                    <small>{describeModel(model)}</small>
                  </div>
                  {installed ? (
                    <span className="sw-installed"><Icon.Star filled /> Installed</span>
                  ) : job ? (
                    <div className="sw-dl">
                      <div className="sw-bar"><i style={{ width: `${pct ?? 8}%` }} /></div>
                      <small>{pct != null ? `Downloading… ${pct}%` : "Downloading…"}</small>
                    </div>
                  ) : (
                    <>
                      {model.downloadSizeLabel ? <span className="sw-msize">{model.downloadSizeLabel}</span> : null}
                      <button type="button" className="sw-btn-primary" onClick={() => createModelDownloadJob?.(model)}>
                        Add
                      </button>
                    </>
                  )}
                </div>
              );
            })
          )}
          <button
            type="button"
            className="sw-advlink"
            onClick={() => {
              setUiMode?.("advanced");
              setActiveView?.("Models");
            }}
          >
            See all models in Advanced →
          </button>
        </section>
      </div>
    </section>
  );
}
