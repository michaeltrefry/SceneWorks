import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const navItems = ["Library", "Image", "Video", "Models", "Characters", "Editor", "Queue"];
const terminalStatuses = new Set(["completed", "failed", "canceled", "interrupted"]);
const actionStatuses = new Set(["failed", "canceled", "interrupted", "completed"]);

const placeholders = {
  Library: ["Project assets", "Imported and generated media"],
  Image: ["Image Studio", "Text, edit, character, variations"],
  Video: ["Video Studio", "Short generated shots"],
  Models: ["Model Manager", "Models and LoRAs"],
  Characters: ["Character Studio", "Reusable identities"],
  Editor: ["Editor", "Timeline assembly"],
};

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

function modelStatusLabel(status) {
  return status === "downloadable" ? "Downloadable" : status;
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
  const [modelManager, setModelManager] = useState({ models: [], storage: null });
  const [loras, setLoras] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [showAdvancedLoras, setShowAdvancedLoras] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [importCategory, setImportCategory] = useState("style");
  const [importFamilies, setImportFamilies] = useState("");
  const [importScope, setImportScope] = useState("global");
  const [projectFilter, setProjectFilter] = useState("all");
  const [requestedGpu, setRequestedGpu] = useState("auto");
  const [jobPrompt, setJobPrompt] = useState("Placeholder generation");
  const [error, setError] = useState("");

  const authenticated = useMemo(() => !access.authRequired || token.length > 0, [access, token]);
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
  const selectedModel = useMemo(() => {
    return modelManager.models.find((model) => model.id === selectedModelId) ?? modelManager.models[0] ?? null;
  }, [modelManager.models, selectedModelId]);

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
    if (!authenticated) {
      return undefined;
    }

    const events = new EventSource(eventUrl("/api/v1/jobs/events", token));
    events.addEventListener("job.updated", (event) => {
      const job = JSON.parse(event.data);
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
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
        apiFetch(`/api/v1/models${activeProject?.id ? `?projectId=${activeProject.id}` : ""}`, token),
      ]);
      setProjects(projectItems);
      setActiveProject((current) => current ?? projectItems[0] ?? null);
      setJobs(jobItems.sort(sortNewest));
      setWorkers(workerItems.sort(sortWorkers));
      setModelManager(modelItems);
      setSelectedModelId((current) => current || modelItems.models[0]?.id || "");
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    if (!authenticated || !selectedModel) {
      return;
    }
    const query = new URLSearchParams();
    if (activeProject?.id) {
      query.set("projectId", activeProject.id);
    }
    query.set("modelId", selectedModel.id);
    if (showAdvancedLoras) {
      query.set("includeUnknown", "true");
      query.set("includeIncompatible", "true");
    }
    apiFetch(`/api/v1/loras?${query.toString()}`, token)
      .then((payload) => setLoras(payload.loras))
      .catch((err) => setError(err.message));
  }, [activeProject?.id, authenticated, selectedModel?.id, showAdvancedLoras, token]);

  useEffect(() => {
    if (!authenticated) {
      return;
    }
    const query = activeProject?.id ? `?projectId=${activeProject.id}` : "";
    apiFetch(`/api/v1/models${query}`, token)
      .then((payload) => {
        setModelManager(payload);
        setSelectedModelId((current) => current || payload.models[0]?.id || "");
      })
      .catch((err) => setError(err.message));
  }, [activeProject?.id, authenticated, token]);

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
      setActiveView("Library");
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function createJob(event) {
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
            modelId: selectedModel?.id,
            loras: loras
              .filter((lora) => lora.compatibility === "compatible")
              .slice(0, 2)
              .map((lora) => ({ id: lora.id, weight: lora.defaultWeight })),
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

  async function downloadModel(model) {
    try {
      await apiFetch(`/api/v1/models/${model.id}/download`, token, {
        method: "POST",
        body: JSON.stringify({ projectId: activeProject?.id ?? null }),
      });
      setActiveView("Queue");
      setError("");
      refreshData();
    } catch (err) {
      setError(err.message);
    }
  }

  async function importLora(event) {
    event.preventDefault();
    if (!importPath.trim()) {
      return;
    }
    try {
      await apiFetch("/api/v1/loras/import", token, {
        method: "POST",
        body: JSON.stringify({
          path: importPath,
          projectId: activeProject?.id ?? null,
          scope: importScope,
          category: importCategory,
          compatibleFamilies: importFamilies
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
        }),
      });
      setImportPath("");
      setError("");
      refreshData();
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

  const [viewTitle, viewEyebrow] = placeholders[activeView] ?? ["Queue", "Jobs and GPUs"];

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

        {activeView === "Queue" ? (
          <QueueScreen
            activeProject={activeProject}
            createJob={createJob}
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
        ) : activeView === "Models" ? (
          <ModelManager
            activeProject={activeProject}
            downloadModel={downloadModel}
            importCategory={importCategory}
            importFamilies={importFamilies}
            importLora={importLora}
            importPath={importPath}
            importScope={importScope}
            loras={loras}
            modelManager={modelManager}
            selectedModel={selectedModel}
            selectedModelId={selectedModelId}
            setImportCategory={setImportCategory}
            setImportFamilies={setImportFamilies}
            setImportPath={setImportPath}
            setImportScope={setImportScope}
            setSelectedModelId={setSelectedModelId}
            setShowAdvancedLoras={setShowAdvancedLoras}
            showAdvancedLoras={showAdvancedLoras}
          />
        ) : (
          <section className="main-surface">
            <div className="section-heading">
              <p className="eyebrow">{viewEyebrow}</p>
              <h2>{viewTitle}</h2>
            </div>

            <form className="job-composer compact" onSubmit={createJob}>
              <label htmlFor="surface-job-prompt">Prompt</label>
              <input
                id="surface-job-prompt"
                onChange={(event) => setJobPrompt(event.target.value)}
                value={jobPrompt}
              />
              <label htmlFor="surface-gpu">GPU</label>
              <select id="surface-gpu" onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
                {gpuOptions.map((gpu) => (
                  <option key={gpu} value={gpu}>
                    {gpu === "auto" ? "Auto" : gpu}
                  </option>
                ))}
              </select>
              <select
                aria-label="Model"
                onChange={(event) => setSelectedModelId(event.target.value)}
                value={selectedModel?.id ?? ""}
              >
                {modelManager.models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
              <button disabled={!authenticated} type="submit">
                Start job
              </button>
            </form>

            <div className="media-grid" aria-label={`${viewTitle} preview placeholders`}>
              <div className="media-tile wide">
                <span>{activeProject ? activeProject.name : "Create a project"}</span>
              </div>
              <div className="media-tile accent">
                <span>{queueCounts.running ? `${queueCounts.running} running` : "Idle"}</span>
              </div>
              <div className="media-tile warm">
                <span>{workers[0]?.gpuName ?? "GPU auto"}</span>
              </div>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

function ModelManager({
  activeProject,
  downloadModel,
  importCategory,
  importFamilies,
  importLora,
  importPath,
  importScope,
  loras,
  modelManager,
  selectedModel,
  selectedModelId,
  setImportCategory,
  setImportFamilies,
  setImportPath,
  setImportScope,
  setSelectedModelId,
  setShowAdvancedLoras,
  showAdvancedLoras,
}) {
  return (
    <section className="main-surface model-surface">
      <div className="model-header">
        <div className="section-heading">
          <p className="eyebrow">Manifest driven</p>
          <h2>Model Manager</h2>
        </div>
        <div className="storage-strip">
          <span>Models {modelManager.storage?.appManagedModels}</span>
          <span>HF cache {modelManager.storage?.hfCache}</span>
          <span>{modelManager.storage?.hfTokenConfigured ? "HF token configured" : "HF token optional"}</span>
        </div>
      </div>

      <div className="model-grid">
        {modelManager.models.map((model) => (
          <article className={selectedModelId === model.id ? "model-card active" : "model-card"} key={model.id}>
            <div className="model-card-main">
              <div>
                <p className="eyebrow">{model.family} / {model.type}</p>
                <h3>{model.name}</h3>
              </div>
              <span className={`status-badge ${model.status}`}>{modelStatusLabel(model.status)}</span>
            </div>
            <p>{model.ui?.description ?? model.adapter}</p>
            <div className="job-meta">
              <span>{model.downloadSizeLabel ?? "Size unknown"}</span>
              <span>{model.source}</span>
              <span>{model.capabilities.join(", ") || "No capabilities"}</span>
            </div>
            <div className="model-paths">
              {Object.entries(model.resolvedPaths).map(([key, value]) => (
                <small key={key}>{key}: {value}</small>
              ))}
            </div>
            <div className="job-actions">
              <button onClick={() => setSelectedModelId(model.id)} type="button">
                Select
              </button>
              <button
                disabled={model.status === "installed" || !model.downloads.length}
                onClick={() => downloadModel(model)}
                type="button"
              >
                Predownload
              </button>
            </div>
          </article>
        ))}
      </div>

      <div className="lora-panel">
        <div className="section-heading">
          <p className="eyebrow">{selectedModel?.name ?? "No model selected"}</p>
          <h2>LoRAs</h2>
        </div>
        <label className="toggle-row" htmlFor="advanced-loras">
          <input
            checked={showAdvancedLoras}
            id="advanced-loras"
            onChange={(event) => setShowAdvancedLoras(event.target.checked)}
            type="checkbox"
          />
          Show unknown and incompatible
        </label>
        <div className="lora-list">
          {loras.length === 0 ? (
            <div className="empty-panel">No compatible LoRAs for this model</div>
          ) : (
            loras.map((lora) => (
              <article className="lora-row" key={lora.id}>
                <div>
                  <strong>{lora.name}</strong>
                  <span>{lora.category} / {lora.scope}</span>
                </div>
                <span className={`status-badge ${lora.compatibility}`}>{lora.compatibility}</span>
                <small>{lora.compatibleFamilies.length ? lora.compatibleFamilies.join(", ") : "Manual compatibility needed"}</small>
              </article>
            ))
          )}
        </div>
      </div>

      <form className="import-panel" onSubmit={importLora}>
        <div className="section-heading">
          <p className="eyebrow">{activeProject ? activeProject.name : "Global"}</p>
          <h2>Import LoRA</h2>
        </div>
        <input
          aria-label="LoRA path"
          onChange={(event) => setImportPath(event.target.value)}
          placeholder="C:\\models\\character.safetensors or loras/project/character.safetensors"
          value={importPath}
        />
        <select aria-label="Category" onChange={(event) => setImportCategory(event.target.value)} value={importCategory}>
          <option value="style">Style</option>
          <option value="enhance">Enhance</option>
          <option value="character">Character</option>
          <option value="motion">Motion</option>
        </select>
        <input
          aria-label="Compatible families"
          onChange={(event) => setImportFamilies(event.target.value)}
          placeholder="ltx, qwen"
          value={importFamilies}
        />
        <select aria-label="Scope" onChange={(event) => setImportScope(event.target.value)} value={importScope}>
          <option value="global">Global</option>
          <option value="project">Project</option>
        </select>
        <button type="submit">Import</button>
      </form>
    </section>
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
        <span>{job.requiresGpu ? `GPU ${job.assignedGpu ?? job.requestedGpu}` : "CPU/network"}</span>
        {job.dependsOnJobId ? <span>Depends {job.dependsOnJobId}</span> : null}
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

function sortNewest(a, b) {
  return b.createdAt.localeCompare(a.createdAt);
}

function sortWorkers(a, b) {
  return `${a.gpuId}-${a.id}`.localeCompare(`${b.gpuId}-${b.id}`);
}

createRoot(document.getElementById("root")).render(<App />);
