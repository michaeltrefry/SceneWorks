import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const navItems = ["Library", "Image", "Video", "Characters", "Editor", "Queue"];
const terminalStatuses = new Set(["completed", "failed", "canceled", "interrupted"]);
const actionStatuses = new Set(["failed", "canceled", "interrupted", "completed"]);
const fallbackModels = [
  {
    id: "z_image_turbo",
    name: "Z-Image-Turbo",
    type: "image",
    capabilities: ["text_to_image", "style_variations", "character_image"],
    ui: { description: "Fast local text-to-image target." },
  },
  {
    id: "z_image_edit",
    name: "Z-Image-Edit",
    type: "image",
    capabilities: ["edit_image"],
    ui: { description: "Image edit target." },
  },
];

async function apiFetch(path, token, options = {}) {
  const headers = new Headers(options.headers ?? {});
  headers.set("Content-Type", "application/json");
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed with ${response.status}`);
  }
  return response.json();
}

function eventUrl(path, token) {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (token) {
    url.searchParams.set("token", token);
  }
  return url.toString();
}

function assetUrl(asset) {
  return asset?.url ? `${API_BASE_URL}${asset.url}` : "";
}

function StatusDot({ ok }) {
  return <span className={ok ? "status-dot ok" : "status-dot"} aria-hidden="true" />;
}

function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) {
    return "0s";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function App() {
  const [health, setHealth] = useState(null);
  const [access, setAccess] = useState({ authRequired: false });
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  const [projects, setProjects] = useState([]);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [projectName, setProjectName] = useState("");
  const [jobs, setJobs] = useState([]);
  const [workers, setWorkers] = useState([]);
  const [models, setModels] = useState([]);
  const [assets, setAssets] = useState([]);
  const [selectedAssetId, setSelectedAssetId] = useState(null);
  const [projectFilter, setProjectFilter] = useState("all");
  const [requestedGpu, setRequestedGpu] = useState("auto");
  const [jobPrompt, setJobPrompt] = useState("Placeholder generation");
  const [latestGenerationSetId, setLatestGenerationSetId] = useState(null);
  const [previewAsset, setPreviewAsset] = useState(null);
  const [error, setError] = useState("");

  const authenticated = useMemo(() => !access.authRequired || token.length > 0, [access, token]);
  const imageModels = useMemo(() => {
    const items = models.filter((model) => model.type === "image");
    return items.length ? items : fallbackModels;
  }, [models]);
  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.id === selectedAssetId) ?? assets[0] ?? null,
    [assets, selectedAssetId],
  );
  const latestAssets = useMemo(
    () => assets.filter((asset) => asset.generationSetId === latestGenerationSetId),
    [assets, latestGenerationSetId],
  );
  const queueCounts = useMemo(() => {
    return jobs.reduce(
      (counts, job) => {
        counts[job.status] = (counts[job.status] ?? 0) + 1;
        if (!terminalStatuses.has(job.status)) {
          counts.active += 1;
        }
        return counts;
      },
      { active: 0 },
    );
  }, [jobs]);
  const filteredJobs = useMemo(() => {
    if (projectFilter === "all") {
      return jobs;
    }
    return jobs.filter((job) => job.projectId === projectFilter);
  }, [jobs, projectFilter]);
  const gpuOptions = useMemo(() => {
    const ids = workers.map((worker) => worker.gpuId).filter(Boolean);
    return ["auto", ...Array.from(new Set(ids))];
  }, [workers]);

  useEffect(() => {
    apiFetch("/api/v1/health", "")
      .then(setHealth)
      .catch((err) => setError(err.message));

    apiFetch("/api/v1/access", "")
      .then(setAccess)
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!authenticated) {
      return;
    }
    refreshData();
  }, [authenticated, token]);

  useEffect(() => {
    if (!activeProject || !authenticated) {
      setAssets([]);
      return;
    }
    refreshAssets(activeProject.id);
  }, [activeProject?.id, authenticated, token]);

  useEffect(() => {
    if (!authenticated) {
      return undefined;
    }

    const events = new EventSource(eventUrl("/api/v1/jobs/events", token));
    events.addEventListener("job.updated", (event) => {
      const job = JSON.parse(event.data);
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      if (job.status === "completed" && job.result?.generationSetId) {
        setLatestGenerationSetId(job.result.generationSetId);
        if (job.projectId) {
          refreshAssets(job.projectId);
        }
      }
    });
    events.addEventListener("worker.updated", (event) => {
      const worker = JSON.parse(event.data);
      setWorkers((items) => [worker, ...items.filter((item) => item.id !== worker.id)].sort(sortWorkers));
    });
    events.onerror = () => {
      events.close();
    };

    return () => events.close();
  }, [authenticated, token]);

  async function refreshData() {
    try {
      const [projectItems, jobItems, workerItems, modelItems] = await Promise.all([
        apiFetch("/api/v1/projects", token),
        apiFetch("/api/v1/jobs", token),
        apiFetch("/api/v1/workers", token),
        apiFetch("/api/v1/models", token),
      ]);
      setProjects(projectItems);
      setActiveProject((current) => current ?? projectItems[0] ?? null);
      setJobs(jobItems.sort(sortNewest));
      setWorkers(workerItems.sort(sortWorkers));
      setModels(modelItems);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshAssets(projectId = activeProject?.id) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/assets?includeRejected=true`, token);
      setAssets(items);
      setSelectedAssetId((current) => current ?? items[0]?.id ?? null);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  function saveToken(event) {
    event.preventDefault();
    window.localStorage.setItem("sceneworks-token", token);
    setError("");
    refreshData();
  }

  async function createProject(event) {
    event.preventDefault();
    if (!projectName.trim()) {
      return;
    }

    try {
      const created = await apiFetch("/api/v1/projects", token, {
        method: "POST",
        body: JSON.stringify({ name: projectName }),
      });
      setProjects((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      setActiveProject(created);
      setProjectName("");
      setActiveView("Image");
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function createPlaceholderJob(event) {
    event.preventDefault();
    try {
      await apiFetch("/api/v1/jobs", token, {
        method: "POST",
        body: JSON.stringify({
          type: "placeholder",
          projectId: activeProject?.id ?? null,
          projectName: activeProject?.name ?? null,
          requestedGpu,
          payload: {
            prompt: jobPrompt,
            createdFrom: activeView,
          },
        }),
      });
      setActiveView("Queue");
      setError("");
      refreshData();
    } catch (err) {
      setError(err.message);
    }
  }

  async function createImageJob(payload) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return;
    }
    try {
      await apiFetch("/api/v1/image/jobs", token, {
        method: "POST",
        body: JSON.stringify({
          ...payload,
          projectId: activeProject.id,
          projectName: activeProject.name,
          requestedGpu,
        }),
      });
      setActiveView("Queue");
      setError("");
      refreshData();
    } catch (err) {
      setError(err.message);
    }
  }

  async function updateAssetStatus(asset, changes) {
    try {
      const updated = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/status`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function deleteAsset(asset) {
    try {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}`, token, { method: "DELETE" });
      setAssets((items) => items.filter((item) => item.id !== asset.id));
      setSelectedAssetId((current) => (current === asset.id ? null : current));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function jobAction(job, action) {
    try {
      const path = action === "duplicate" ? `/api/v1/jobs/${job.id}/duplicate` : `/api/v1/jobs/${job.id}/${action}`;
      const body = action === "duplicate" ? { payloadChanges: { duplicatedAt: new Date().toISOString() } } : {};
      await apiFetch(path, token, { method: "POST", body: JSON.stringify(body) });
      setError("");
      refreshData();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark">SW</span>
          <div>
            <h1>SceneWorks</h1>
            <p>Local creative studio</p>
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              className={activeView === item ? "nav-item active" : "nav-item"}
              key={item}
              onClick={() => setActiveView(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Project</p>
            <strong>{activeProject?.name ?? "No project open"}</strong>
          </div>
          <div className="topbar-status">
            <span>
              <StatusDot ok={health?.status === "ok"} />
              API
            </span>
            <span>{workers.length ? `${workers.length} worker${workers.length === 1 ? "" : "s"}` : "No workers"}</span>
            <span>{gpuOptions.length > 1 ? `${gpuOptions.length - 1} GPU slot${gpuOptions.length === 2 ? "" : "s"}` : "GPU auto"}</span>
            <button className="queue-chip" onClick={() => setActiveView("Queue")} type="button">
              Queue {queueCounts.active}
            </button>
          </div>
        </header>

        {error ? <p className="notice error">{error}</p> : null}

        {access.authRequired && !window.localStorage.getItem("sceneworks-token") ? (
          <section className="auth-band">
            <form onSubmit={saveToken}>
              <label htmlFor="token">Pairing token</label>
              <div className="form-row">
                <input
                  id="token"
                  onChange={(event) => setToken(event.target.value)}
                  placeholder="Enter local token"
                  type="password"
                  value={token}
                />
                <button type="submit">Unlock</button>
              </div>
            </form>
          </section>
        ) : null}

        <section className="project-band">
          <div className="project-list">
            <div className="section-heading">
              <p className="eyebrow">Recent projects</p>
              <h2>Open a workspace</h2>
            </div>
            <div className="project-buttons">
              {projects.length === 0 ? (
                <span className="empty-state">No projects yet</span>
              ) : (
                projects.map((project) => (
                  <button
                    className={activeProject?.id === project.id ? "project-pill active" : "project-pill"}
                    key={project.id}
                    onClick={() => setActiveProject(project)}
                    type="button"
                  >
                    {project.name}
                  </button>
                ))
              )}
            </div>
          </div>

          <form className="create-project" onSubmit={createProject}>
            <label htmlFor="project-name">New project</label>
            <div className="form-row">
              <input
                id="project-name"
                onChange={(event) => setProjectName(event.target.value)}
                placeholder="Noir Alley"
                value={projectName}
              />
              <button disabled={!authenticated} type="submit">
                Create
              </button>
            </div>
          </form>
        </section>

        {activeView === "Library" ? (
          <LibraryScreen
            assets={assets}
            deleteAsset={deleteAsset}
            onPreview={setPreviewAsset}
            onSendImage={(asset) => {
              setSelectedAssetId(asset.id);
              setActiveView("Image");
            }}
            selectedAsset={selectedAsset}
            setSelectedAssetId={setSelectedAssetId}
            updateAssetStatus={updateAssetStatus}
          />
        ) : null}

        {activeView === "Image" ? (
          <ImageStudio
            activeProject={activeProject}
            assets={assets}
            createImageJob={createImageJob}
            gpuOptions={gpuOptions}
            imageModels={imageModels}
            latestAssets={latestAssets}
            onPreview={setPreviewAsset}
            requestedGpu={requestedGpu}
            selectedAsset={selectedAsset}
            setRequestedGpu={setRequestedGpu}
            updateAssetStatus={updateAssetStatus}
            deleteAsset={deleteAsset}
          />
        ) : null}

        {activeView === "Queue" ? (
          <QueueScreen
            activeProject={activeProject}
            createJob={createPlaceholderJob}
            filteredJobs={filteredJobs}
            gpuOptions={gpuOptions}
            jobAction={jobAction}
            jobPrompt={jobPrompt}
            projectFilter={projectFilter}
            projects={projects}
            requestedGpu={requestedGpu}
            setJobPrompt={setJobPrompt}
            setProjectFilter={setProjectFilter}
            setRequestedGpu={setRequestedGpu}
            workers={workers}
          />
        ) : null}

        {["Video", "Characters", "Editor"].includes(activeView) ? (
          <PlaceholderSurface activeView={activeView} assets={assets} createJob={createPlaceholderJob} />
        ) : null}
      </section>

      {previewAsset ? <FullscreenPreview asset={previewAsset} onClose={() => setPreviewAsset(null)} /> : null}
    </main>
  );
}

function LibraryScreen({
  assets,
  deleteAsset,
  onPreview,
  onSendImage,
  selectedAsset,
  setSelectedAssetId,
  updateAssetStatus,
}) {
  const [typeFilter, setTypeFilter] = useState("all");
  const [showRejected, setShowRejected] = useState(false);
  const visibleAssets = assets.filter((asset) => {
    if (typeFilter !== "all" && asset.type !== typeFilter) {
      return false;
    }
    if (!showRejected && asset.status?.rejected) {
      return false;
    }
    return true;
  });

  return (
    <section className="main-surface library-surface">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Project assets</p>
          <h2>Library</h2>
        </div>
        <div className="toolbar">
          <select aria-label="Asset type" onChange={(event) => setTypeFilter(event.target.value)} value={typeFilter}>
            <option value="all">All media</option>
            <option value="image">Images</option>
            <option value="video">Videos</option>
            <option value="render">Renders</option>
          </select>
          <label className="checkline">
            <input checked={showRejected} onChange={(event) => setShowRejected(event.target.checked)} type="checkbox" />
            Rejected
          </label>
        </div>
      </div>

      <div className="library-layout">
        <AssetGrid
          assets={visibleAssets}
          onPreview={onPreview}
          selectedAsset={selectedAsset}
          setSelectedAssetId={setSelectedAssetId}
        />
        <AssetDetail
          asset={selectedAsset}
          deleteAsset={deleteAsset}
          onPreview={onPreview}
          onSendImage={onSendImage}
          updateAssetStatus={updateAssetStatus}
        />
      </div>
    </section>
  );
}

function ImageStudio({
  activeProject,
  assets,
  createImageJob,
  deleteAsset,
  gpuOptions,
  imageModels,
  latestAssets,
  onPreview,
  requestedGpu,
  selectedAsset,
  setRequestedGpu,
  updateAssetStatus,
}) {
  const [mode, setMode] = useState("text_to_image");
  const [prompt, setPrompt] = useState("A cinematic frame of a neon street at midnight");
  const [stylePreset, setStylePreset] = useState("cinematic");
  const [count, setCount] = useState(4);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(imageModels[0]?.id ?? "z_image_turbo");
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [resolution, setResolution] = useState("1024x1024");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");

  useEffect(() => {
    if (!imageModels.some((item) => item.id === model)) {
      setModel(imageModels[0]?.id ?? "z_image_turbo");
    }
  }, [imageModels, model]);

  useEffect(() => {
    if (mode === "edit_image" && selectedAsset?.id) {
      setSourceAssetId(selectedAsset.id);
    }
  }, [mode, selectedAsset?.id]);

  const availableModels = imageModels.filter((item) => {
    const caps = item.capabilities ?? [];
    if (mode === "edit_image") {
      return caps.includes("edit_image") || caps.includes("image_edit");
    }
    return item.type === "image";
  });
  const [width, height] = resolution.split("x").map((value) => Number(value));

  function submit(event) {
    event.preventDefault();
    createImageJob({
      mode,
      prompt,
      negativePrompt,
      model,
      count,
      seed: seed === "" ? null : Number(seed),
      width,
      height,
      stylePreset,
      sourceAssetId: mode === "edit_image" ? sourceAssetId || null : null,
      loras: [],
      advanced: { resolution },
    });
  }

  return (
    <section className="main-surface image-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Image Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <div className="segmented-control" role="tablist" aria-label="Image mode">
          {[
            ["text_to_image", "Text"],
            ["edit_image", "Edit"],
            ["character_image", "Character"],
            ["style_variations", "Variations"],
          ].map(([value, label]) => (
            <button className={mode === value ? "active" : ""} key={value} onClick={() => setMode(value)} type="button">
              {label}
            </button>
          ))}
        </div>
      </div>

      <form className="studio-layout" onSubmit={submit}>
        <section className="studio-controls">
          {mode === "edit_image" ? (
            <label>
              Source
              <select onChange={(event) => setSourceAssetId(event.target.value)} value={sourceAssetId}>
                <option value="">Select image</option>
                {assets
                  .filter((asset) => asset.type === "image")
                  .map((asset) => (
                    <option key={asset.id} value={asset.id}>
                      {asset.displayName}
                    </option>
                  ))}
              </select>
            </label>
          ) : null}

          <label className="prompt-field">
            Prompt
            <textarea onChange={(event) => setPrompt(event.target.value)} value={prompt} />
          </label>

          <div className="control-grid">
            <label>
              Style
              <select onChange={(event) => setStylePreset(event.target.value)} value={stylePreset}>
                <option value="cinematic">Cinematic</option>
                <option value="photoreal">Photoreal</option>
                <option value="anime">Anime</option>
                <option value="fantasy">Fantasy</option>
                <option value="product">Product Shot</option>
              </select>
            </label>
            <label>
              Count
              <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
            </label>
            <label>
              GPU
              <select onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
                {gpuOptions.map((gpu) => (
                  <option key={gpu} value={gpu}>
                    {gpu === "auto" ? "Auto" : gpu}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
            {advancedOpen ? "Hide advanced" : "Advanced"}
          </button>

          {advancedOpen ? (
            <div className="advanced-panel">
              <label>
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {(availableModels.length ? availableModels : imageModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
              </label>
              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  <option value="768x768">768 x 768</option>
                  <option value="1024x1024">1024 x 1024</option>
                  <option value="1280x720">1280 x 720</option>
                  <option value="720x1280">720 x 1280</option>
                </select>
              </label>
              <label className="prompt-field">
                Negative prompt
                <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
              </label>
            </div>
          ) : null}

          <button className="primary-action" disabled={!activeProject || !prompt.trim()} type="submit">
            Generate
          </button>
        </section>

        <section className="review-panel">
          <div className="section-heading">
            <p className="eyebrow">Fresh batch</p>
            <h2>Review</h2>
          </div>
          {latestAssets.length ? (
            <div className="review-grid">
              {latestAssets.map((asset) => (
                <AssetCard
                  asset={asset}
                  deleteAsset={deleteAsset}
                  key={asset.id}
                  onPreview={onPreview}
                  updateAssetStatus={updateAssetStatus}
                />
              ))}
            </div>
          ) : (
            <div className="empty-panel">No fresh image batch</div>
          )}
        </section>
      </form>
    </section>
  );
}

function AssetGrid({ assets, onPreview, selectedAsset, setSelectedAssetId }) {
  if (!assets.length) {
    return <div className="empty-panel">No assets in this view</div>;
  }

  return (
    <div className="asset-grid">
      {assets.map((asset) => (
        <button
          className={selectedAsset?.id === asset.id ? "asset-tile active" : "asset-tile"}
          key={asset.id}
          onClick={() => setSelectedAssetId(asset.id)}
          onDoubleClick={() => onPreview(asset)}
          type="button"
        >
          {asset.type === "image" ? <img alt="" src={assetUrl(asset)} /> : <span>{asset.type}</span>}
          <strong>{asset.displayName}</strong>
        </button>
      ))}
    </div>
  );
}

function AssetDetail({ asset, deleteAsset, onPreview, onSendImage, updateAssetStatus }) {
  if (!asset) {
    return <aside className="asset-detail empty-panel">No asset selected</aside>;
  }

  return (
    <aside className="asset-detail">
      <button className="preview-button" onClick={() => onPreview(asset)} type="button">
        {asset.type === "image" ? <img alt="" src={assetUrl(asset)} /> : <span>{asset.type}</span>}
      </button>
      <h3>{asset.displayName}</h3>
      <p>{asset.recipe?.prompt ?? "No prompt"}</p>
      <div className="rating-row">
        {[1, 2, 3, 4, 5].map((rating) => (
          <button
            className={asset.status?.rating >= rating ? "active" : ""}
            key={rating}
            onClick={() => updateAssetStatus(asset, { rating })}
            type="button"
          >
            {rating}
          </button>
        ))}
      </div>
      <div className="detail-actions">
        <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
          {asset.status?.favorite ? "Unfavorite" : "Favorite"}
        </button>
        <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
          {asset.status?.rejected ? "Restore" : "Reject"}
        </button>
        <button onClick={() => onSendImage(asset)} type="button">
          Send to Image
        </button>
        <button onClick={() => deleteAsset(asset)} type="button">
          Discard
        </button>
      </div>
      <dl>
        <div>
          <dt>Model</dt>
          <dd>{asset.recipe?.model ?? "Unknown"}</dd>
        </div>
        <div>
          <dt>Generation set</dt>
          <dd>{asset.generationSetId ?? "None"}</dd>
        </div>
      </dl>
    </aside>
  );
}

function AssetCard({ asset, deleteAsset, onPreview, updateAssetStatus }) {
  return (
    <article className={asset.status?.rejected ? "review-card rejected" : "review-card"}>
      <button className="preview-button" onClick={() => onPreview(asset)} type="button">
        <img alt="" src={assetUrl(asset)} />
      </button>
      <div className="review-actions">
        <button onClick={() => updateAssetStatus(asset, { favorite: !asset.status?.favorite })} type="button">
          {asset.status?.favorite ? "Saved" : "Favorite"}
        </button>
        <button onClick={() => updateAssetStatus(asset, { rejected: !asset.status?.rejected })} type="button">
          {asset.status?.rejected ? "Restore" : "Reject"}
        </button>
        <button onClick={() => deleteAsset(asset)} type="button">
          Discard
        </button>
      </div>
    </article>
  );
}

function FullscreenPreview({ asset, onClose }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="preview-modal">
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
        {asset.type === "image" ? <img alt="" src={assetUrl(asset)} /> : <div>{asset.type}</div>}
        <footer>
          <strong>{asset.displayName}</strong>
          <span>{asset.recipe?.model}</span>
        </footer>
      </div>
    </div>
  );
}

function QueueScreen({
  activeProject,
  createJob,
  filteredJobs,
  gpuOptions,
  jobAction,
  jobPrompt,
  projectFilter,
  projects,
  requestedGpu,
  setJobPrompt,
  setProjectFilter,
  setRequestedGpu,
  workers,
}) {
  return (
    <section className="main-surface queue-surface">
      <div className="queue-header">
        <div className="section-heading">
          <p className="eyebrow">Jobs and GPUs</p>
          <h2>Queue</h2>
        </div>
        <form className="job-composer" onSubmit={createJob}>
          <label htmlFor="queue-job-prompt">Prompt</label>
          <input id="queue-job-prompt" onChange={(event) => setJobPrompt(event.target.value)} value={jobPrompt} />
          <label htmlFor="queue-gpu">GPU</label>
          <select id="queue-gpu" onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
            {gpuOptions.map((gpu) => (
              <option key={gpu} value={gpu}>
                {gpu === "auto" ? "Auto" : gpu}
              </option>
            ))}
          </select>
          <button disabled={!activeProject} type="submit">
            Add job
          </button>
        </form>
      </div>

      <div className="queue-tools">
        <label htmlFor="project-filter">Project</label>
        <select id="project-filter" onChange={(event) => setProjectFilter(event.target.value)} value={projectFilter}>
          <option value="all">All projects</option>
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
      </div>

      <div className="worker-grid">
        {workers.length === 0 ? (
          <div className="worker-card">
            <strong>No workers registered</strong>
            <span>Start the worker service to claim queued jobs.</span>
          </div>
        ) : (
          workers.map((worker) => (
            <div className="worker-card" key={worker.id}>
              <strong>{worker.gpuName ?? worker.gpuId}</strong>
              <span>{worker.status}</span>
              <small>{worker.currentJobId ?? "Idle"}</small>
            </div>
          ))
        )}
      </div>

      <div className="job-list">
        {filteredJobs.length === 0 ? (
          <div className="empty-panel">No jobs in this view</div>
        ) : (
          filteredJobs.map((job) => <JobRow job={job} jobAction={jobAction} key={job.id} />)
        )}
      </div>
    </section>
  );
}

function JobRow({ job, jobAction }) {
  const canCancel = !terminalStatuses.has(job.status);
  const canRepeat = actionStatuses.has(job.status);
  return (
    <article className={`job-row ${job.status}`}>
      <div className="job-main">
        <div>
          <p className="eyebrow">{job.type}</p>
          <h3>{job.payload.prompt ?? job.id}</h3>
        </div>
        <span className="status-badge">{job.status}</span>
      </div>
      <div className="job-meta">
        <span>{job.projectName ?? "Global"}</span>
        <span>Stage {job.stage}</span>
        <span>Elapsed {formatSeconds(job.elapsedSeconds)}</span>
        <span>GPU {job.assignedGpu ?? job.requestedGpu}</span>
      </div>
      <div className="progress-track" aria-label={`${percent(job.progress)} complete`}>
        <span style={{ width: percent(job.progress) }} />
      </div>
      <p className={job.error ? "job-message error-text" : "job-message"}>{job.error ?? job.message}</p>
      <div className="job-actions">
        <button disabled={!canCancel || job.cancelRequested} onClick={() => jobAction(job, "cancel")} type="button">
          Cancel
        </button>
        <button disabled={!canRepeat} onClick={() => jobAction(job, "retry")} type="button">
          Retry
        </button>
        <button disabled={!canRepeat} onClick={() => jobAction(job, "duplicate")} type="button">
          Duplicate
        </button>
      </div>
    </article>
  );
}

function PlaceholderSurface({ activeView, assets, createJob }) {
  return (
    <section className="main-surface">
      <div className="section-heading">
        <p className="eyebrow">{activeView}</p>
        <h2>{activeView}</h2>
      </div>
      <form className="job-composer compact" onSubmit={createJob}>
        <label htmlFor="surface-job-prompt">Prompt</label>
        <input id="surface-job-prompt" defaultValue={`${activeView} placeholder`} />
        <button type="submit">Start job</button>
      </form>
      <div className="media-grid" aria-label={`${activeView} assets`}>
        <div className="media-tile wide">
          <span>{assets.length} assets</span>
        </div>
        <div className="media-tile accent">
          <span>{assets.filter((asset) => asset.status?.favorite).length} favorites</span>
        </div>
        <div className="media-tile warm">
          <span>{assets.filter((asset) => asset.type === "image").length} images</span>
        </div>
      </div>
    </section>
  );
}

function sortNewest(a, b) {
  return b.createdAt.localeCompare(a.createdAt);
}

function sortWorkers(a, b) {
  return `${a.gpuId}-${a.id}`.localeCompare(`${b.gpuId}-${b.id}`);
}

createRoot(document.getElementById("root")).render(<App />);
