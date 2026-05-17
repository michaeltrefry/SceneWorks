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
  {
    id: "ltx_2_3",
    name: "LTX-2.3",
    type: "video",
    capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
    defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
    limits: {
      durations: [4, 6, 8, 10, 12, 15],
      recommendedMaxDuration: 10,
      fps: [24, 25, 30],
      resolutions: ["768x512", "512x768", "640x640", "1280x720", "720x1280"],
    },
    ui: {
      description: "First-class short-shot video target.",
      durationHint: "Best at 10s or less for the current workflow.",
    },
  },
  {
    id: "wan_2_2",
    name: "Wan2.2",
    type: "video",
    capabilities: ["image_to_video", "text_to_video", "first_last_frame", "replace_person"],
    defaults: { duration: 5, fps: 24, resolution: "1280x720", quality: "balanced" },
    limits: {
      durations: [4, 5, 6, 7, 8],
      recommendedMaxDuration: 7,
      fps: [16, 24],
      resolutions: ["832x480", "1280x720", "720x1280"],
    },
    ui: {
      description: "Fallback video family.",
      durationHint: "Keep clips short until local looping behavior is validated.",
    },
  },
];

async function apiFetch(path, token, options = {}) {
  const headers = new Headers(options.headers ?? {});
  if (options.body) {
    headers.set("Content-Type", "application/json");
  }
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

function assetCanRenderAsImage(asset) {
  return asset?.type === "image" || asset?.file?.mimeType?.startsWith("image/");
}

function AssetMedia({ asset, className = "" }) {
  if (!asset) {
    return null;
  }
  const src = assetUrl(asset);
  if (asset.file?.mimeType?.startsWith("video/")) {
    return <video className={className} controls muted playsInline src={src} />;
  }
  if (assetCanRenderAsImage(asset)) {
    return <img alt="" className={className} src={src} />;
  }
  return <span className={className}>{asset.type}</span>;
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

const aspectOptions = {
  "16:9": { width: 1280, height: 720, label: "16:9" },
  "9:16": { width: 720, height: 1280, label: "9:16" },
  "1:1": { width: 1024, height: 1024, label: "1:1" },
};
const transitionOptions = ["cut", "crossfade", "fade_from_black", "fade_to_black"];
const speedPresets = [0.25, 0.5, 1, 2];

function createLocalTimeline(project, name = "Main timeline", aspectRatio = "16:9") {
  const dimensions = aspectOptions[aspectRatio] ?? aspectOptions["16:9"];
  return {
    schemaVersion: 1,
    id: `timeline_${crypto.randomUUID().replaceAll("-", "")}`,
    projectId: project.id,
    name,
    aspectRatio,
    width: dimensions.width,
    height: dimensions.height,
    fps: 30,
    duration: 0,
    tracks: [
      { id: "track_main", name: "Main", kind: "video", locked: false, muted: false, items: [] },
      { id: "track_overlay", name: "Overlay", kind: "overlay", locked: false, muted: false, items: [] },
      { id: "track_audio", name: "Audio", kind: "audio", locked: false, muted: false, items: [] },
    ],
    transitions: [],
    createdAt: null,
    updatedAt: null,
  };
}

function timelineDuration(timeline) {
  return Math.max(0, ...timeline.tracks.flatMap((track) => track.items.map((item) => Number(item.timelineEnd) || 0)));
}

function itemDuration(item) {
  return Math.max(0.1, Number(item.timelineEnd) - Number(item.timelineStart));
}

function trackItems(track) {
  return [...track.items].sort((a, b) => a.timelineStart - b.timelineStart);
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
  const [timelines, setTimelines] = useState([]);
  const [selectedTimelineId, setSelectedTimelineId] = useState(null);
  const [activeTimeline, setActiveTimeline] = useState(null);
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
    return items.length ? items : fallbackModels.filter((model) => model.type === "image");
  }, [models]);
  const videoModels = useMemo(() => {
    const items = models.filter((model) => model.type === "video");
    return items.length ? items : fallbackModels.filter((model) => model.type === "video");
  }, [models]);
  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.id === selectedAssetId) ?? assets[0] ?? null,
    [assets, selectedAssetId],
  );
  const latestAssets = useMemo(
    () => assets.filter((asset) => asset.generationSetId === latestGenerationSetId),
    [assets, latestGenerationSetId],
  );
  const latestImageAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "image"), [latestAssets]);
  const latestVideoAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "video"), [latestAssets]);
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
  const mediaAssets = useMemo(
    () => assets.filter((asset) => ["image", "video", "upload", "frame", "render"].includes(asset.type)),
    [assets],
  );

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
      setTimelines([]);
      setSelectedTimelineId(null);
      setActiveTimeline(null);
      return;
    }
    refreshAssets(activeProject.id);
    refreshTimelines(activeProject.id);
  }, [activeProject?.id, authenticated, token]);

  useEffect(() => {
    if (!activeProject || !selectedTimelineId) {
      return;
    }
    loadTimeline(activeProject.id, selectedTimelineId);
  }, [activeProject?.id, selectedTimelineId]);

  useEffect(() => {
    if (!authenticated) {
      return undefined;
    }

    const events = new EventSource(eventUrl("/api/v1/jobs/events", token));
    events.addEventListener("job.updated", (event) => {
      const job = JSON.parse(event.data);
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      if (job.status === "completed" && (job.result?.generationSetId || job.result?.assetIds?.length)) {
        if (job.result?.generationSetId) {
          setLatestGenerationSetId(job.result.generationSetId);
        }
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

  async function refreshTimelines(projectId = activeProject?.id) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/timelines`, token);
      setTimelines(items);
      setSelectedTimelineId((current) => current ?? items[0]?.id ?? null);
      if (!items.length) {
        setActiveTimeline(null);
      }
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadTimeline(projectId, timelineId) {
    try {
      const timeline = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token);
      setActiveTimeline(timeline);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function createTimeline(payload) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const created = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setTimelines((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      setSelectedTimelineId(created.id);
      setActiveTimeline(created);
      setError("");
      return created;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function saveTimeline(timeline) {
    if (!activeProject || !timeline) {
      return null;
    }
    try {
      const saved = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${timeline.id}`, token, {
        method: "PUT",
        body: JSON.stringify({ timeline }),
      });
      setActiveTimeline(saved);
      refreshTimelines(activeProject.id);
      setError("");
      return saved;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function exportTimeline(timeline, options) {
    if (!activeProject || !timeline) {
      return;
    }
    const saved = await saveTimeline(timeline);
    if (!saved) {
      return;
    }
    try {
      await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${saved.id}/exports`, token, {
        method: "POST",
        body: JSON.stringify({ ...options, requestedGpu }),
      });
      setActiveView("Queue");
      setError("");
      refreshData();
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

  async function createVideoJob(payload) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return;
    }
    try {
      await apiFetch("/api/v1/video/jobs", token, {
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
            onSendVideo={(asset) => {
              setSelectedAssetId(asset.id);
              setActiveView("Video");
            }}
            onSendEditor={(asset) => {
              setSelectedAssetId(asset.id);
              setActiveView("Editor");
            }}
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
            latestAssets={latestImageAssets}
            onPreview={setPreviewAsset}
            requestedGpu={requestedGpu}
            selectedAsset={selectedAsset}
            setRequestedGpu={setRequestedGpu}
            updateAssetStatus={updateAssetStatus}
            deleteAsset={deleteAsset}
          />
        ) : null}

        {activeView === "Video" ? (
          <VideoStudio
            activeProject={activeProject}
            assets={assets}
            createVideoJob={createVideoJob}
            deleteAsset={deleteAsset}
            gpuOptions={gpuOptions}
            latestAssets={latestVideoAssets}
            onPreview={setPreviewAsset}
            requestedGpu={requestedGpu}
            selectedAsset={selectedAsset}
            setRequestedGpu={setRequestedGpu}
            updateAssetStatus={updateAssetStatus}
            videoModels={videoModels}
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

        {activeView === "Editor" ? (
          <EditorScreen
            activeProject={activeProject}
            activeTimeline={activeTimeline}
            assets={mediaAssets}
            createTimeline={createTimeline}
            exportTimeline={exportTimeline}
            onPreview={setPreviewAsset}
            refreshAssets={refreshAssets}
            saveTimeline={saveTimeline}
            selectedTimelineId={selectedTimelineId}
            setActiveTimeline={setActiveTimeline}
            setSelectedTimelineId={setSelectedTimelineId}
            timelines={timelines}
          />
        ) : null}

        {activeView === "Characters" ? (
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
  onSendVideo,
  onSendEditor,
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
          onSendVideo={onSendVideo}
          onSendEditor={onSendEditor}
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

function VideoStudio({
  activeProject,
  assets,
  createVideoJob,
  deleteAsset,
  gpuOptions,
  latestAssets,
  onPreview,
  requestedGpu,
  selectedAsset,
  setRequestedGpu,
  updateAssetStatus,
  videoModels,
}) {
  const imageAssets = assets.filter((asset) => asset.type === "image" || asset.type === "frame");
  const videoAssets = assets.filter((asset) => asset.type === "video");
  const [mode, setMode] = useState("image_to_video");
  const [prompt, setPrompt] = useState("Camera slowly pushes in while the scene comes alive");
  const [quality, setQuality] = useState("balanced");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(videoModels[0]?.id ?? "ltx_2_3");
  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];
  const [duration, setDuration] = useState(selectedModel?.defaults?.duration ?? 6);
  const [resolution, setResolution] = useState(selectedModel?.defaults?.resolution ?? "768x512");
  const [fps, setFps] = useState(selectedModel?.defaults?.fps ?? 25);
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.type === "image" ? selectedAsset.id : "");
  const [lastFrameAssetId, setLastFrameAssetId] = useState("");
  const [sourceClipAssetId, setSourceClipAssetId] = useState(selectedAsset?.type === "video" ? selectedAsset.id : "");

  useEffect(() => {
    if (!videoModels.some((item) => item.id === model)) {
      setModel(videoModels[0]?.id ?? "ltx_2_3");
    }
  }, [videoModels, model]);

  useEffect(() => {
    if (selectedAsset?.type === "image") {
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
  }, [selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    setDuration((current) => {
      const options = selectedModel.limits?.durations ?? [4, 6, 8, 10];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.duration ?? options[0];
    });
    setResolution((current) => {
      const options = selectedModel.limits?.resolutions ?? ["768x512"];
      return options.includes(current) ? current : selectedModel.defaults?.resolution ?? options[0];
    });
    setFps((current) => {
      const options = selectedModel.limits?.fps ?? [24, 25, 30];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.fps ?? options[0];
    });
  }, [selectedModel?.id]);

  const modeOptions = [
    ["image_to_video", "Image to Video"],
    ["text_to_video", "Text to Video"],
    ["first_last_frame", "First/Last Frame"],
    ["extend_clip", "Extend Clip"],
    ["replace_person", "Replace Person"],
  ];
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  const implementedMode = ["image_to_video", "text_to_video", "first_last_frame"].includes(mode);
  const hasInputs =
    mode === "text_to_video" ||
    (mode === "image_to_video" && sourceAssetId) ||
    (mode === "first_last_frame" && sourceAssetId && lastFrameAssetId) ||
    (mode === "extend_clip" && sourceClipAssetId) ||
    mode === "replace_person";
  const canSubmit = Boolean(activeProject && prompt.trim() && supportsMode && implementedMode && hasInputs);
  const [width, height] = resolution.split("x").map((value) => Number(value));
  const durationOptions = selectedModel?.limits?.durations ?? [4, 6, 8, 10];
  const resolutionOptions = selectedModel?.limits?.resolutions ?? ["768x512", "640x640", "1280x720", "720x1280"];
  const fpsOptions = selectedModel?.limits?.fps ?? [24, 25, 30];
  const durationHint =
    selectedModel?.ui?.durationHint ??
    (selectedModel?.limits?.recommendedMaxDuration ? `Recommended: ${selectedModel.limits.recommendedMaxDuration}s or less.` : "");
  const blockedMessage = !supportsMode
    ? `${selectedModel?.name ?? "Selected model"} does not support this mode.`
    : !implementedMode
      ? "This entry point is reserved for the next runtime slice."
      : !hasInputs
        ? "Required inputs are missing."
        : "";

  function submit(event) {
    event.preventDefault();
    createVideoJob({
      mode,
      prompt,
      negativePrompt,
      model,
      duration: Number(duration),
      fps: Number(fps),
      width,
      height,
      quality,
      seed: seed === "" ? null : Number(seed),
      sourceAssetId: ["image_to_video", "first_last_frame"].includes(mode) ? sourceAssetId || null : null,
      lastFrameAssetId: mode === "first_last_frame" ? lastFrameAssetId || null : null,
      sourceClipAssetId: mode === "extend_clip" ? sourceClipAssetId || null : null,
      loras: [],
      advanced: { resolution, durationHint },
    });
  }

  return (
    <section className="main-surface video-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Video Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <div className="segmented-control mode-control" role="tablist" aria-label="Video mode">
          {modeOptions.map(([value, label]) => (
            <button className={mode === value ? "active" : ""} key={value} onClick={() => setMode(value)} type="button">
              {label}
            </button>
          ))}
        </div>
      </div>

      <form className="studio-layout video-layout" onSubmit={submit}>
        <section className="studio-controls">
          {mode === "image_to_video" || mode === "first_last_frame" ? (
            <label>
              First frame
              <select onChange={(event) => setSourceAssetId(event.target.value)} value={sourceAssetId}>
                <option value="">Select image</option>
                {imageAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.displayName}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {mode === "first_last_frame" ? (
            <label>
              Last frame
              <select onChange={(event) => setLastFrameAssetId(event.target.value)} value={lastFrameAssetId}>
                <option value="">Select image</option>
                {imageAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.displayName}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {mode === "extend_clip" ? (
            <label>
              Source clip
              <select onChange={(event) => setSourceClipAssetId(event.target.value)} value={sourceClipAssetId}>
                <option value="">Select clip</option>
                {videoAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.displayName}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {mode === "replace_person" ? (
            <div className="empty-panel compact-panel">Replacement is staged as a placeholder mode.</div>
          ) : null}

          <label className="prompt-field">
            Prompt
            <textarea onChange={(event) => setPrompt(event.target.value)} value={prompt} />
          </label>

          <div className="control-grid">
            <label>
              Duration
              <select onChange={(event) => setDuration(Number(event.target.value))} value={duration}>
                {durationOptions.map((value) => (
                  <option key={value} value={value}>
                    {value}s
                  </option>
                ))}
              </select>
            </label>
            <label>
              Quality
              <select onChange={(event) => setQuality(event.target.value)} value={quality}>
                <option value="fast">Fast</option>
                <option value="balanced">Balanced</option>
                <option value="best">Best</option>
              </select>
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

          <div className="guidance-strip">
            <strong>{selectedModel?.name ?? "Video model"}</strong>
            <span>{durationHint}</span>
          </div>

          <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
            {advancedOpen ? "Hide advanced" : "Advanced"}
          </button>

          {advancedOpen ? (
            <div className="advanced-panel">
              <label>
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {videoModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " x ")}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                FPS
                <select onChange={(event) => setFps(Number(event.target.value))} value={fps}>
                  {fpsOptions.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
              </label>
              <label className="prompt-field">
                Negative prompt
                <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
              </label>
            </div>
          ) : null}

          {blockedMessage ? <p className="inline-warning">{blockedMessage}</p> : null}
          <button className="primary-action" disabled={!canSubmit} type="submit">
            Generate Clip
          </button>
        </section>

        <section className="review-panel video-review">
          <div className="section-heading">
            <p className="eyebrow">Fresh clip</p>
            <h2>Review</h2>
          </div>
          {latestAssets.length ? (
            <div className="review-grid video-review-grid">
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
            <div className="empty-panel">No fresh video clip</div>
          )}

          <div className="asset-tray">
            <div className="section-heading">
              <p className="eyebrow">Asset tray</p>
              <h2>Recent videos</h2>
            </div>
            <div className="tray-grid">
              {videoAssets.slice(0, 4).map((asset) => (
                <button className="tray-item" key={asset.id} onClick={() => onPreview(asset)} type="button">
                  <AssetMedia asset={asset} />
                  <span>{asset.displayName}</span>
                </button>
              ))}
              {videoAssets.length === 0 ? <div className="empty-panel compact-panel">No video assets</div> : null}
            </div>
          </div>
        </section>
      </form>
    </section>
  );
}

function EditorScreen({
  activeProject,
  activeTimeline,
  assets,
  createTimeline,
  exportTimeline,
  onPreview,
  saveTimeline,
  selectedTimelineId,
  setActiveTimeline,
  setSelectedTimelineId,
  timelines,
}) {
  const [newTimelineName, setNewTimelineName] = useState("Main timeline");
  const [newAspectRatio, setNewAspectRatio] = useState("16:9");
  const [selectedItemId, setSelectedItemId] = useState(null);
  const [addDuration, setAddDuration] = useState(4);
  const [exportResolution, setExportResolution] = useState(720);
  const [history, setHistory] = useState([]);
  const [future, setFuture] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);

  const selectedItem = useMemo(() => {
    if (!activeTimeline) {
      return null;
    }
    return activeTimeline.tracks.flatMap((track) => track.items).find((item) => item.id === selectedItemId) ?? null;
  }, [activeTimeline, selectedItemId]);
  const selectedAsset = useMemo(() => assets.find((asset) => asset.id === selectedItem?.assetId) ?? null, [assets, selectedItem]);
  const duration = activeTimeline ? timelineDuration(activeTimeline) : 0;
  const timelineScale = Math.max(12, duration + 4);
  const mainAssets = assets.filter((asset) => asset.type === "video" || asset.file?.mimeType?.startsWith("video/"));
  const stillAssets = assets.filter((asset) => assetCanRenderAsImage(asset));

  useEffect(() => {
    setHistory([]);
    setFuture([]);
    setSelectedItemId(null);
  }, [activeTimeline?.id]);

  useEffect(() => {
    function onKeyDown(event) {
      const target = event.target;
      const isTyping = ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName);
      if (isTyping) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        setIsPlaying((value) => !value);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        if (selectedItemId) {
          event.preventDefault();
          removeSelectedItem();
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTimeline, history, future, selectedItemId]);

  async function submitNewTimeline(event) {
    event.preventDefault();
    await createTimeline({ name: newTimelineName, aspectRatio: newAspectRatio, fps: 30 });
  }

  function commit(nextTimeline) {
    if (!activeTimeline) {
      return;
    }
    setHistory((items) => [...items.slice(-24), activeTimeline]);
    setFuture([]);
    setActiveTimeline({ ...nextTimeline, duration: timelineDuration(nextTimeline) });
  }

  function updateTimelineItem(itemId, changes) {
    if (!activeTimeline) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => ({
        ...track,
        items: track.items.map((item) => (item.id === itemId ? normalizeTimelineItem({ ...item, ...changes }) : item)),
      })),
    });
  }

  function undo() {
    if (!history.length || !activeTimeline) {
      return;
    }
    const previous = history[history.length - 1];
    setHistory((items) => items.slice(0, -1));
    setFuture((items) => [activeTimeline, ...items]);
    setActiveTimeline(previous);
  }

  function redo() {
    if (!future.length || !activeTimeline) {
      return;
    }
    const next = future[0];
    setFuture((items) => items.slice(1));
    setHistory((items) => [...items, activeTimeline]);
    setActiveTimeline(next);
  }

  function addAssetToTrack(asset, trackId = "track_main") {
    if (!activeTimeline) {
      return;
    }
    const isStill = asset.type !== "video" && assetCanRenderAsImage(asset);
    const track = activeTimeline.tracks.find((item) => item.id === trackId) ?? activeTimeline.tracks[0];
    const start = Math.max(0, ...track.items.map((item) => item.timelineEnd));
    const sourceDuration = Number(asset.file?.duration) || Number(addDuration) || 4;
    const durationSeconds = isStill ? Number(addDuration) || 4 : sourceDuration;
    const item = normalizeTimelineItem({
      id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
      trackId: track.id,
      assetId: asset.id,
      type: isStill ? "image" : "video",
      displayName: asset.displayName,
      sourceIn: 0,
      sourceOut: Math.max(0.1, sourceDuration),
      timelineStart: start,
      timelineEnd: start + Math.max(0.1, durationSeconds),
      speed: 1,
      fit: "fit",
      volume: 1,
      versionAssetIds: [asset.id],
      transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
      transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
    });
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((current) =>
        current.id === track.id ? { ...current, items: [...current.items, item] } : current,
      ),
    });
    setSelectedItemId(item.id);
  }

  function removeSelectedItem() {
    if (!activeTimeline || !selectedItemId) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => ({
        ...track,
        items: track.items.filter((item) => item.id !== selectedItemId),
      })),
    });
    setSelectedItemId(null);
  }

  function changeItemTrack(trackId) {
    if (!activeTimeline || !selectedItem) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => {
        if (track.id === selectedItem.trackId) {
          return { ...track, items: track.items.filter((item) => item.id !== selectedItem.id) };
        }
        if (track.id === trackId) {
          return { ...track, items: [...track.items, normalizeTimelineItem({ ...selectedItem, trackId })] };
        }
        return track;
      }),
    });
  }

  function normalizeTimelineItem(item) {
    const start = Number(item.timelineStart) || 0;
    const end = Math.max(start + 0.1, Number(item.timelineEnd) || start + 0.1);
    const sourceIn = Number(item.sourceIn) || 0;
    const sourceOut = Math.max(sourceIn + 0.1, Number(item.sourceOut) || sourceIn + itemDuration(item));
    return {
      ...item,
      sourceIn,
      sourceOut,
      timelineStart: Math.max(0, start),
      timelineEnd: end,
      speed: Math.max(0.1, Number(item.speed) || 1),
    };
  }

  if (!activeProject) {
    return (
      <section className="main-surface">
        <div className="section-heading">
          <p className="eyebrow">Editor</p>
          <h2>Create a project</h2>
        </div>
        <div className="empty-panel">Open a project before assembling a timeline.</div>
      </section>
    );
  }

  return (
    <section className="main-surface editor-surface">
      <div className="surface-header editor-header">
        <div className="section-heading">
          <p className="eyebrow">Editor</p>
          <h2>{activeTimeline?.name ?? "Timelines"}</h2>
        </div>
        <div className="editor-actions">
          <select onChange={(event) => setSelectedTimelineId(event.target.value)} value={selectedTimelineId ?? ""}>
            <option value="">Select timeline</option>
            {timelines.map((timeline) => (
              <option key={timeline.id} value={timeline.id}>
                {timeline.name}
              </option>
            ))}
          </select>
          <button disabled={!activeTimeline} onClick={() => saveTimeline(activeTimeline)} type="button">
            Save
          </button>
          <button disabled={!history.length} onClick={undo} type="button">
            Undo
          </button>
          <button disabled={!future.length} onClick={redo} type="button">
            Redo
          </button>
        </div>
      </div>

      <form className="timeline-create" onSubmit={submitNewTimeline}>
        <label>
          Timeline
          <input onChange={(event) => setNewTimelineName(event.target.value)} value={newTimelineName} />
        </label>
        <label>
          Aspect
          <select onChange={(event) => setNewAspectRatio(event.target.value)} value={newAspectRatio}>
            {Object.entries(aspectOptions).map(([value, option]) => (
              <option key={value} value={value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <button type="submit">New Timeline</button>
      </form>

      {activeTimeline ? (
        <div className="editor-layout">
          <section className="editor-preview">
            <div className={`preview-canvas aspect-${activeTimeline.aspectRatio.replace(":", "-")}`}>
              {selectedAsset ? <AssetMedia asset={selectedAsset} /> : <span>Select a timeline item</span>}
            </div>
            <div className="playback-bar">
              <button onClick={() => setIsPlaying((value) => !value)} type="button">
                {isPlaying ? "Pause" : "Play"}
              </button>
              <span>{formatSeconds(Math.round(duration))}</span>
              <span>{activeTimeline.aspectRatio}</span>
              <span>{activeTimeline.fps} fps</span>
            </div>
          </section>

          <aside className="editor-inspector">
            {selectedItem ? (
              <>
                <div className="section-heading">
                  <p className="eyebrow">Clip</p>
                  <h2>{selectedItem.displayName}</h2>
                </div>
                <label>
                  Track
                  <select onChange={(event) => changeItemTrack(event.target.value)} value={selectedItem.trackId}>
                    {activeTimeline.tracks.map((track) => (
                      <option key={track.id} value={track.id}>
                        {track.name}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="control-grid compact-controls">
                  <label>
                    Start
                    <input
                      min="0"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { timelineStart: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.timelineStart}
                    />
                  </label>
                  <label>
                    End
                    <input
                      min="0.1"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { timelineEnd: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.timelineEnd}
                    />
                  </label>
                  <label>
                    Source In
                    <input
                      min="0"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { sourceIn: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.sourceIn}
                    />
                  </label>
                  <label>
                    Source Out
                    <input
                      min="0.1"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { sourceOut: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.sourceOut}
                    />
                  </label>
                </div>
                <label>
                  Speed
                  <select onChange={(event) => updateTimelineItem(selectedItem.id, { speed: Number(event.target.value) })} value={selectedItem.speed}>
                    {speedPresets.map((speed) => (
                      <option key={speed} value={speed}>
                        {speed}x
                      </option>
                    ))}
                    {!speedPresets.includes(Number(selectedItem.speed)) ? <option value={selectedItem.speed}>Custom {selectedItem.speed}x</option> : null}
                  </select>
                </label>
                <label>
                  Custom speed
                  <input
                    min="0.1"
                    onChange={(event) => updateTimelineItem(selectedItem.id, { speed: Number(event.target.value) })}
                    step="0.05"
                    type="number"
                    value={selectedItem.speed}
                  />
                </label>
                <label>
                  Transition in
                  <select
                    onChange={(event) =>
                      updateTimelineItem(selectedItem.id, {
                        transitionIn: { ...(selectedItem.transitionIn ?? {}), type: event.target.value },
                      })
                    }
                    value={selectedItem.transitionIn?.type ?? "cut"}
                  >
                    {transitionOptions.map((transition) => (
                      <option key={transition} value={transition}>
                        {transition.replaceAll("_", " ")}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Transition out
                  <select
                    onChange={(event) =>
                      updateTimelineItem(selectedItem.id, {
                        transitionOut: { ...(selectedItem.transitionOut ?? {}), type: event.target.value },
                      })
                    }
                    value={selectedItem.transitionOut?.type ?? "cut"}
                  >
                    {transitionOptions.map((transition) => (
                      <option key={transition} value={transition}>
                        {transition.replaceAll("_", " ")}
                      </option>
                    ))}
                  </select>
                </label>
                <button className="danger-action" onClick={removeSelectedItem} type="button">
                  Delete Clip
                </button>
              </>
            ) : (
              <div className="empty-panel compact-panel">No clip selected</div>
            )}
          </aside>

          <section className="timeline-panel">
            <div className="timeline-ruler">
              <span>0s</span>
              <span>{formatSeconds(Math.ceil(timelineScale / 2))}</span>
              <span>{formatSeconds(Math.ceil(timelineScale))}</span>
            </div>
            <div className="timeline-tracks">
              {activeTimeline.tracks.map((track) => (
                <div className="timeline-track" key={track.id}>
                  <strong>{track.name}</strong>
                  <div className="track-lane">
                    {trackItems(track).map((item) => (
                      <button
                        className={selectedItemId === item.id ? "timeline-item active" : "timeline-item"}
                        key={item.id}
                        onClick={() => setSelectedItemId(item.id)}
                        style={{
                          left: `${(item.timelineStart / timelineScale) * 100}%`,
                          width: `${Math.max(4, (itemDuration(item) / timelineScale) * 100)}%`,
                        }}
                        type="button"
                      >
                        <span>{item.displayName}</span>
                        <small>{item.speed}x</small>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <aside className="asset-bin">
            <div className="bin-controls">
              <label>
                Still duration
                <input min="0.5" onChange={(event) => setAddDuration(Number(event.target.value))} step="0.5" type="number" value={addDuration} />
              </label>
            </div>
            <div className="asset-bin-list">
              {[...mainAssets, ...stillAssets].slice(0, 18).map((asset) => (
                <article className="bin-asset" key={asset.id}>
                  <button onClick={() => onPreview(asset)} type="button">
                    <AssetMedia asset={asset} />
                  </button>
                  <strong>{asset.displayName}</strong>
                  <div className="bin-actions">
                    <button onClick={() => addAssetToTrack(asset, "track_main")} type="button">
                      Main
                    </button>
                    <button onClick={() => addAssetToTrack(asset, "track_overlay")} type="button">
                      Overlay
                    </button>
                  </div>
                </article>
              ))}
              {assets.length === 0 ? <div className="empty-panel compact-panel">No media assets</div> : null}
            </div>
          </aside>

          <form
            className="export-strip"
            onSubmit={(event) => {
              event.preventDefault();
              exportTimeline(activeTimeline, { resolution: Number(exportResolution), fps: activeTimeline.fps });
            }}
          >
            <label>
              MP4 height
              <select onChange={(event) => setExportResolution(Number(event.target.value))} value={exportResolution}>
                {[640, 720, 1024, 1280].map((resolution) => (
                  <option key={resolution} value={resolution}>
                    {resolution}
                  </option>
                ))}
              </select>
            </label>
            <button className="primary-action" disabled={!activeTimeline.tracks.some((track) => track.items.length)} type="submit">
              Export MP4
            </button>
          </form>
        </div>
      ) : (
        <div className="empty-panel">Create a timeline to start editing.</div>
      )}
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
          <AssetMedia asset={asset} />
          <strong>{asset.displayName}</strong>
        </button>
      ))}
    </div>
  );
}

function AssetDetail({ asset, deleteAsset, onPreview, onSendImage, onSendVideo, onSendEditor, updateAssetStatus }) {
  if (!asset) {
    return <aside className="asset-detail empty-panel">No asset selected</aside>;
  }

  return (
    <aside className="asset-detail">
      <button className="preview-button" onClick={() => onPreview(asset)} type="button">
        <AssetMedia asset={asset} />
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
        {asset.type === "image" ? (
          <button onClick={() => onSendImage(asset)} type="button">
            Send to Image
          </button>
        ) : null}
        {asset.type === "image" ? (
          <button onClick={() => onSendVideo(asset)} type="button">
            Send to Video
          </button>
        ) : null}
        {["image", "video", "upload", "frame"].includes(asset.type) ? (
          <button onClick={() => onSendEditor(asset)} type="button">
            Send to Editor
          </button>
        ) : null}
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
          <dt>Duration</dt>
          <dd>{asset.file?.duration ? `${asset.file.duration}s` : "Still"}</dd>
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
        <AssetMedia asset={asset} />
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
        <AssetMedia asset={asset} />
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
