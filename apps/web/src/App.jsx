import React, { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, eventUrl } from "./api.js";
import { Icon } from "./components/Icons.jsx";
import { StatusDot } from "./components/StatusDot.jsx";
import { FullscreenPreview } from "./components/assetPanels.jsx";
import { fallbackModels, terminalStatuses } from "./constants.js";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { TrainingStudio } from "./screens/TrainingStudio.jsx";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { EditorScreen } from "./screens/EditorScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { sortNewest, sortWorkers } from "./sorters.js";
import { ensureItemVersionFields } from "./timeline.js";

function isActiveWorker(worker) {
  return worker.status !== "offline";
}

function hasCapability(worker, capability) {
  return Array.isArray(worker.capabilities) && worker.capabilities.includes(capability);
}

function isPlaceholderOnlyGpuWorker(worker) {
  if (!hasCapability(worker, "gpu")) {
    return false;
  }
  const capabilities = Array.isArray(worker.capabilities) ? worker.capabilities : [];
  return capabilities.every((capability) => ["placeholder", "gpu", "nvidia"].includes(capability));
}

function isSelectableGpuWorker(worker) {
  return worker.gpuId && worker.gpuId !== "cpu" && hasCapability(worker, "gpu") && !isPlaceholderOnlyGpuWorker(worker);
}

function failedJobNotice(job) {
  const label = String(job.type ?? "job").replaceAll("_", " ");
  const detail = job.error || job.message || "Failed without additional worker detail.";
  return `${label}: ${detail}`;
}

function isImageGenerationJob(job) {
  return ["image_generate", "image_edit"].includes(job.type);
}

function isLoraImportNotice(message) {
  return String(message ?? "").startsWith("lora import: ");
}

function jobFreshnessMs(job) {
  const timestamp = job?.updatedAt ?? job?.completedAt ?? job?.canceledAt ?? job?.startedAt ?? job?.createdAt;
  const parsed = Date.parse(timestamp ?? "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function mergeFreshJobs(currentJobs, serverJobs) {
  const merged = new Map();
  for (const job of serverJobs) {
    merged.set(job.id, job);
  }
  for (const current of currentJobs) {
    const server = merged.get(current.id);
    if (!server || jobFreshnessMs(current) > jobFreshnessMs(server)) {
      merged.set(current.id, current);
    }
  }
  return [...merged.values()].sort(sortNewest);
}

function generatedResultAssetCount(job) {
  if (Array.isArray(job.result?.assetIds)) {
    return job.result.assetIds.length;
  }
  if (Array.isArray(job.result?.assets)) {
    return job.result.assets.length;
  }
  return 0;
}

const localJobStackLimit = 4;
const maxLoraUploadBytes = 2 * 1024 * 1024 * 1024;
const maxModelUploadBytes = 256 * 1024 * 1024 * 1024;

const navSections = [
  {
    label: "Workspace",
    items: [
      { id: "Library", icon: Icon.Library },
      { id: "Image", icon: Icon.Image },
      { id: "Video", icon: Icon.Video },
      { id: "Train", icon: Icon.Train },
      { id: "Editor", icon: Icon.Editor },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "Characters", icon: Icon.Character },
      { id: "Presets", icon: Icon.Preset },
      { id: "Models", icon: Icon.Model },
    ],
  },
  {
    label: "System",
    items: [{ id: "Queue", icon: Icon.Queue }],
  },
];

const viewTitles = {
  Library: { title: "Library", blurb: "Browse stills and clips across all your projects." },
  Image: { title: "Image Studio", blurb: "Describe what you want — we'll render variations side by side." },
  Video: { title: "Video Studio", blurb: "Bring stills to life, or render new clips from scratch." },
  Train: { title: "Training Studio", blurb: "Build datasets and prepare LoRA training plans." },
  Editor: { title: "Editor", blurb: "Cut, sequence and export your timeline." },
  Characters: { title: "Characters", blurb: "Keep the same face across every shot." },
  Presets: { title: "Presets", blurb: "Save and share recurring generation setups." },
  Models: { title: "Models", blurb: "Download, import and manage local checkpoints." },
  Queue: { title: "Queue", blurb: "All running and recent jobs across workers." },
};

function readStoredTheme() {
  if (typeof window === "undefined") {
    return "light";
  }
  try {
    const saved = window.localStorage.getItem("sceneworks-theme");
    return saved === "dark" || saved === "light" ? saved : "light";
  } catch {
    return "light";
  }
}

function ProjectSwitcher({ activeProject, projects, onSelect, onCreate, disabled }) {
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function onDocMouseDown(event) {
      if (!containerRef.current?.contains(event.target)) {
        setOpen(false);
        setCreating(false);
        setName("");
      }
    }
    function onDocKey(event) {
      if (event.key === "Escape") {
        setOpen(false);
        setCreating(false);
        setName("");
      }
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onDocKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onDocKey);
    };
  }, [open]);

  useEffect(() => {
    if (creating) {
      inputRef.current?.focus();
    }
  }, [creating]);

  async function submitNew(event) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || submitting) {
      return;
    }
    setSubmitting(true);
    const created = await onCreate(trimmed);
    setSubmitting(false);
    if (created) {
      setName("");
      setCreating(false);
      setOpen(false);
    }
  }

  return (
    <div className="project-switcher" ref={containerRef}>
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        className="project-pill"
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        title={activeProject?.name ?? "Pick a workspace"}
        type="button"
      >
        <span className="project-pill-thumb" aria-hidden="true" />
        <span className="project-pill-meta">
          <strong>{activeProject?.name ?? "No workspace open"}</strong>
          <span>
            {projects.length} workspace{projects.length === 1 ? "" : "s"}
          </span>
        </span>
        <Icon.ChevDown className="chev" />
      </button>

      {open ? (
        <div className="project-menu" role="listbox">
          {projects.length === 0 ? (
            <p className="project-menu-empty">No workspaces yet — create the first one below.</p>
          ) : (
            projects.map((project) => (
              <button
                aria-selected={project.id === activeProject?.id}
                className={project.id === activeProject?.id ? "project-menu-item active" : "project-menu-item"}
                key={project.id}
                onClick={() => {
                  onSelect(project);
                  setOpen(false);
                  setCreating(false);
                  setName("");
                }}
                role="option"
                type="button"
              >
                <span className="project-menu-thumb" aria-hidden="true" />
                <span className="project-menu-label">{project.name}</span>
              </button>
            ))
          )}

          {creating ? (
            <form className="project-menu-create" onSubmit={submitNew}>
              <input
                aria-label="New workspace name"
                disabled={submitting}
                onChange={(event) => setName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    event.preventDefault();
                    setCreating(false);
                    setName("");
                  }
                }}
                placeholder="Workspace name"
                ref={inputRef}
                value={name}
              />
              <button disabled={!name.trim() || submitting} type="submit">
                {submitting ? "Creating…" : "Create"}
              </button>
            </form>
          ) : (
            <button
              className="project-menu-item project-menu-item-new"
              disabled={disabled}
              onClick={() => setCreating(true)}
              type="button"
            >
              <Icon.Plus />
              <span className="project-menu-label">New workspace</span>
            </button>
          )}
        </div>
      ) : null}
    </div>
  );
}

function FirstRunProjectGate({ onCreate, disabled }) {
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || submitting) {
      return;
    }
    setSubmitting(true);
    try {
      await onCreate(trimmed);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="first-run-gate">
      <div className="first-run-card">
        <span className="first-run-mark" aria-hidden="true">
          <img src="/sceneworks-logo.svg" alt="" />
        </span>
        <h2>Create your first workspace</h2>
        <p className="first-run-lede">
          SceneWorks keeps your images, videos, characters, and timelines inside a
          workspace. Create one to start generating.
        </p>
        <form className="first-run-form" onSubmit={submit}>
          <input
            aria-label="Workspace name"
            autoFocus
            disabled={disabled || submitting}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. My First Project"
            value={name}
          />
          <button className="first-run-cta" disabled={disabled || submitting || !name.trim()} type="submit">
            {submitting ? "Creating…" : "Create workspace"}
          </button>
        </form>
      </div>
    </section>
  );
}

function uploadLimitLabel(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  return Number.isInteger(gib) ? `${gib}GB` : `${gib.toFixed(1)}GB`;
}

export function App() {
  const [health, setHealth] = useState(null);
  const [access, setAccess] = useState({ authRequired: false });
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  const [projects, setProjects] = useState([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [jobs, setJobs] = useState([]);
  const [localGenerationJobIds, setLocalGenerationJobIds] = useState({ image: [], video: [] });
  const [workers, setWorkers] = useState([]);
  const [queueSummary, setQueueSummary] = useState(null);
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);
  const [presets, setPresets] = useState([]);
  const [trainingDatasets, setTrainingDatasets] = useState([]);
  const [trainingDatasetsProjectId, setTrainingDatasetsProjectId] = useState(null);
  const [loadingTrainingDatasets, setLoadingTrainingDatasets] = useState(false);
  const [trainingDatasetsError, setTrainingDatasetsError] = useState("");
  const [assets, setAssets] = useState([]);
  const [characters, setCharacters] = useState([]);
  const [personTracks, setPersonTracks] = useState([]);
  const [timelines, setTimelines] = useState([]);
  const [timelinesProjectId, setTimelinesProjectId] = useState(null);
  const [selectedTimelineId, setSelectedTimelineId] = useState(null);
  const [activeTimeline, setActiveTimeline] = useState(null);
  const [selectedAssetId, setSelectedAssetId] = useState(null);
  const [projectFilter, setProjectFilter] = useState("all");
  const [requestedGpu, setRequestedGpu] = useState("auto");
  const [jobPrompt, setJobPrompt] = useState("Placeholder generation");
  const [latestGenerationSetId, setLatestGenerationSetId] = useState(null);
  const [previewAsset, setPreviewAsset] = useState(null);
  const [studioLaunch, setStudioLaunch] = useState(null);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(readStoredTheme);
  const activeProjectRef = useRef(null);
  const activeViewRef = useRef(activeView);
  const localGenerationJobIdsRef = useRef(localGenerationJobIds);
  const selectedTimelineIdRef = useRef(null);
  const timelineApplyQueueRef = useRef(Promise.resolve());
  const generatedAssetRefreshesRef = useRef(new Map());

  const authenticated = useMemo(() => !access.authRequired || token.length > 0, [access, token]);
  const imageModels = useMemo(() => {
    const items = models.filter((model) => model.type === "image" && model.installState !== "missing");
    return items.length || models.length ? items : fallbackModels.filter((model) => model.type === "image");
  }, [models]);
  const videoModels = useMemo(() => {
    const items = models.filter((model) => model.type === "video" && model.installState !== "missing");
    return items.length || models.length ? items : fallbackModels.filter((model) => model.type === "video");
  }, [models]);
  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.id === selectedAssetId) ?? assets[0] ?? null,
    [assets, selectedAssetId],
  );
  const previewedAsset = useMemo(
    () => (previewAsset ? assets.find((asset) => asset.id === previewAsset.id) ?? previewAsset : null),
    [assets, previewAsset],
  );
  const previewNavigation = useMemo(() => {
    if (!previewedAsset || assets.length < 2) {
      return { previous: null, next: null };
    }
    const currentIndex = assets.findIndex((asset) => asset.id === previewedAsset.id);
    if (currentIndex < 0) {
      return { previous: null, next: null };
    }
    return {
      previous: currentIndex > 0 ? assets[currentIndex - 1] : null,
      next: currentIndex < assets.length - 1 ? assets[currentIndex + 1] : null,
    };
  }, [assets, previewedAsset]);
  const latestAssets = useMemo(
    () => assets.filter((asset) => asset.generationSetId === latestGenerationSetId),
    [assets, latestGenerationSetId],
  );
  const latestImageAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "image"), [latestAssets]);
  const latestVideoAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "video"), [latestAssets]);
  const imageLocalJobs = useMemo(() => {
    const localJobs = localGenerationJobIds.image.map((id) => jobs.find((job) => job.id === id)).filter(Boolean);
    const projectJobs = jobs
      .filter(
        (job) =>
          activeProject?.id &&
          job.projectId === activeProject.id &&
          isImageGenerationJob(job) &&
          !terminalStatuses.has(job.status),
      )
      .sort(sortNewest);
    const byId = new Map();
    [...localJobs, ...projectJobs].forEach((job) => {
      if (job?.id && !byId.has(job.id)) {
        byId.set(job.id, job);
      }
    });
    return Array.from(byId.values()).slice(0, localJobStackLimit);
  }, [activeProject?.id, jobs, localGenerationJobIds.image]);
  const videoLocalJobs = useMemo(
    () => localGenerationJobIds.video.map((id) => jobs.find((job) => job.id === id)).filter(Boolean),
    [jobs, localGenerationJobIds.video],
  );
  const queueCounts = useMemo(() => {
    if (queueSummary?.counts) {
      return {
        ...queueSummary.counts,
        active: queueSummary.activeJobs?.length ?? jobs.filter((job) => !terminalStatuses.has(job.status)).length,
      };
    }
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
  const visibleWorkers = useMemo(
    () => workers.filter((worker) => isActiveWorker(worker) && !isPlaceholderOnlyGpuWorker(worker)),
    [workers],
  );
  const gpuOptions = useMemo(() => {
    const ids = visibleWorkers.filter(isSelectableGpuWorker).map((worker) => worker.gpuId);
    return ["auto", ...Array.from(new Set(ids))];
  }, [visibleWorkers]);
  const mediaAssets = useMemo(
    () => assets.filter((asset) => ["image", "video", "upload", "frame", "render"].includes(asset.type)),
    [assets],
  );

  useEffect(() => {
    activeViewRef.current = activeView;
  }, [activeView]);

  useEffect(() => {
    activeProjectRef.current = activeProject;
  }, [activeProject]);

  useEffect(() => {
    localGenerationJobIdsRef.current = localGenerationJobIds;
  }, [localGenerationJobIds]);

  useEffect(() => {
    selectedTimelineIdRef.current = selectedTimelineId;
  }, [selectedTimelineId]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem("sceneworks-theme", theme);
    } catch {
      // ignore (private mode etc.)
    }
  }, [theme]);

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
      setCharacters([]);
      setPersonTracks([]);
      setTimelines([]);
      setTimelinesProjectId(null);
      setPresets([]);
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(null);
      setTrainingDatasetsError("");
      setSelectedTimelineId(null);
      setActiveTimeline(null);
      return;
    }
    refreshAssets(activeProject.id);
    refreshCharacters(activeProject.id);
    refreshLoras(activeProject.id);
    refreshPresets(activeProject.id);
    refreshTrainingDatasets(activeProject.id);
    refreshPersonTracks(activeProject.id);
    refreshTimelines(activeProject.id);
  }, [activeProject?.id, authenticated, token]);

  useEffect(() => {
    if (!activeProject || !selectedTimelineId || timelinesProjectId !== activeProject.id) {
      return;
    }
    loadTimeline(activeProject.id, selectedTimelineId);
  }, [activeProject?.id, selectedTimelineId, timelinesProjectId]);

  useEffect(() => {
    if (!authenticated) {
      return undefined;
    }

    let events = null;
    let reconnectTimer = null;
    let reconnectAttempt = 0;
    let closed = false;

    function handleJobUpdated(event) {
      const job = JSON.parse(event.data);
      const hasGeneratedAssets = Boolean(job.result?.generationSetId || job.result?.assetIds?.length || job.result?.assets?.length);
      const resultAssetCount = generatedResultAssetCount(job);
      const generationSetId = job.result?.generationSetId ?? "";
      const refreshKey = job.id ?? generationSetId;
      const previousRefresh = generatedAssetRefreshesRef.current.get(refreshKey) ?? { assetCount: 0, generationSetId: "" };
      const shouldRefreshGeneratedAssets =
        Boolean(job.projectId) &&
        hasGeneratedAssets &&
        (resultAssetCount > previousRefresh.assetCount ||
          (resultAssetCount === 0 && generationSetId && generationSetId !== previousRefresh.generationSetId));
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      if (hasGeneratedAssets) {
        if (job.result?.generationSetId) {
          setLatestGenerationSetId(job.result.generationSetId);
        }
        generatedAssetRefreshesRef.current.set(refreshKey, {
          assetCount: Math.max(resultAssetCount, previousRefresh.assetCount),
          generationSetId: generationSetId || previousRefresh.generationSetId,
        });
        if (shouldRefreshGeneratedAssets) {
          refreshAssets(job.projectId);
        }
      }
      if (job.status === "completed" && hasGeneratedAssets) {
        enqueueTimelineGenerationApply(job);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_track") {
        refreshPersonTracks(job.projectId);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_detect") {
        refreshAssets(job.projectId);
      }
      if (job.status === "completed" && job.type === "model_download") {
        refreshData();
      }
      if (job.status === "completed" && job.type === "lora_import") {
        setError((current) => (isLoraImportNotice(current) ? "" : current));
        refreshDataWithLoraOverlay(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "failed" && !hasVisibleLocalFailure(job)) {
        setError(failedJobNotice(job));
      }
    }

    function handleWorkerUpdated(event) {
      const worker = JSON.parse(event.data);
      setWorkers((items) => [worker, ...items.filter((item) => item.id !== worker.id)].sort(sortWorkers));
    }

    function handleQueueUpdated(event) {
      const summary = JSON.parse(event.data);
      setQueueSummary(summary);
      if (Array.isArray(summary.workers)) {
        setWorkers(summary.workers.sort(sortWorkers));
      }
    }

    async function connect() {
      let ticket = "";
      try {
        if (access.authRequired) {
          const response = await apiFetch("/api/v1/jobs/events/ticket", token, { method: "POST" });
          ticket = response.ticket;
        }
      } catch (err) {
        setError(err.message);
        if (!closed) {
          const delay = Math.min(30000, 1000 * 2 ** reconnectAttempt);
          reconnectAttempt += 1;
          reconnectTimer = window.setTimeout(connect, delay);
        }
        return;
      }

      if (closed) {
        return;
      }

      const source = new EventSource(eventUrl("/api/v1/jobs/events", ticket));
      events = source;
      source.addEventListener("job.updated", handleJobUpdated);
      source.addEventListener("worker.updated", handleWorkerUpdated);
      source.addEventListener("queue.updated", handleQueueUpdated);
      source.onopen = () => {
        reconnectAttempt = 0;
      };
      source.onerror = () => {
        source.close();
        if (closed) {
          return;
        }
        const delay = Math.min(30000, 1000 * 2 ** reconnectAttempt);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      events?.close();
    };
  }, [access.authRequired, authenticated, token]);

  async function refreshData() {
    const fetchInitial = async (label, path, fallback, optional = false) => {
      try {
        return { label, value: await apiFetch(path, token), error: "" };
      } catch (err) {
        return { label, value: fallback, error: optional ? "" : `${label}: ${err.message}` };
      }
    };
    const [projectsResult, jobsResult, workersResult, modelsResult, lorasResult, presetsResult] = await Promise.all([
      fetchInitial("Projects", "/api/v1/projects", []),
      fetchInitial("Jobs", "/api/v1/jobs", []),
      fetchInitial("Workers", "/api/v1/workers", []),
      fetchInitial("Models", "/api/v1/models", []),
      fetchInitial("LoRAs", "/api/v1/loras", []),
      fetchInitial("Presets", "/api/v1/recipe-presets", [], true),
    ]);
    const projectItems = projectsResult.value;
    setProjects(projectItems);
    setProjectsLoaded(true);
    setActiveProject((current) => current ?? projectItems[0] ?? null);
    setJobs((current) => mergeFreshJobs(current, jobsResult.value));
    setWorkers(workersResult.value.sort(sortWorkers));
    setQueueSummary(null);
    setModels(modelsResult.value);
    setLoras(lorasResult.value);
    setPresets(presetsResult.value);
    setError([projectsResult, jobsResult, workersResult, modelsResult, lorasResult, presetsResult].map((result) => result.error).filter(Boolean).join("; "));
  }

  async function refreshAssets(projectId = activeProject?.id) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/assets?includeRejected=true&includeTrashed=true`, token);
      setAssets(items);
      const defaultAsset = items.find((asset) => !asset.status?.trashed && !asset.status?.rejected) ?? items[0] ?? null;
      setSelectedAssetId((current) => current ?? defaultAsset?.id ?? null);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshCharacters(projectId = activeProject?.id) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/characters`, token);
      setCharacters(items);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshLoras(projectId = activeProject?.id) {
    try {
      const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
      const items = await apiFetch(`/api/v1/loras${query}`, token);
      setLoras(items);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  function refreshDataWithLoraOverlay(projectId = activeProjectRef.current?.id) {
    refreshData()
      .then(() => {
        if (projectId) {
          refreshLoras(projectId);
        }
      })
      .catch(() => {});
  }

  async function refreshPresets(projectId = activeProject?.id) {
    try {
      const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
      const items = await apiFetch(`/api/v1/recipe-presets${query}`, token);
      setPresets(items);
      setError("");
      return items;
    } catch (err) {
      setError(err.message);
      return [];
    }
  }

  async function refreshTrainingDatasets(projectId = activeProject?.id) {
    if (!projectId) {
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(null);
      setTrainingDatasetsError("");
      return [];
    }
    setLoadingTrainingDatasets(true);
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/training/datasets`, token);
      setTrainingDatasets(items);
      setTrainingDatasetsProjectId(projectId);
      setTrainingDatasetsError("");
      return items;
    } catch (err) {
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(projectId);
      setTrainingDatasetsError(err.message);
      return [];
    } finally {
      setLoadingTrainingDatasets(false);
    }
  }

  async function loadTrainingDataset(datasetId, projectId = activeProject?.id) {
    if (!projectId || !datasetId) {
      return null;
    }
    return apiFetch(`/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}`, token);
  }

  async function createTrainingDataset(payload, projectId = activeProject?.id) {
    if (!projectId) {
      throw new Error("Create or open a project first.");
    }
    const created = await apiFetch(`/api/v1/projects/${projectId}/training/datasets`, token, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refreshTrainingDatasets(projectId);
    return created;
  }

  async function updateTrainingDataset(datasetId, payload, projectId = activeProject?.id) {
    if (!projectId || !datasetId) {
      throw new Error("Select a training dataset first.");
    }
    const updated = await apiFetch(`/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}`, token, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    await refreshTrainingDatasets(projectId);
    return updated;
  }

  function presetQuery(scope = null) {
    const params = new URLSearchParams();
    if (scope) {
      params.set("scope", scope);
    }
    if (scope === "project" && activeProject?.id) {
      params.set("projectId", activeProject.id);
    }
    const value = params.toString();
    return value ? `?${value}` : "";
  }

  async function createPreset(payload) {
    if (payload.scope === "project" && !activeProject) {
      throw new Error("Create or open a project first.");
    }
    const created = await apiFetch(`/api/v1/recipe-presets${presetQuery(payload.scope)}`, token, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refreshPresets(activeProject?.id);
    return created;
  }

  async function updatePreset(presetId, payload, scope = payload.scope) {
    const updated = await apiFetch(`/api/v1/recipe-presets/${encodeURIComponent(presetId)}${presetQuery(scope)}`, token, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    await refreshPresets(activeProject?.id);
    return updated;
  }

  async function duplicatePreset(presetId, scope = null) {
    const duplicated = await apiFetch(`/api/v1/recipe-presets/${encodeURIComponent(presetId)}/duplicate${presetQuery(scope)}`, token, {
      method: "POST",
      body: JSON.stringify({}),
    });
    await refreshPresets(activeProject?.id);
    return duplicated;
  }

  async function deletePreset(presetId, scope = null) {
    const archived = await apiFetch(`/api/v1/recipe-presets/${encodeURIComponent(presetId)}${presetQuery(scope)}`, token, {
      method: "DELETE",
    });
    await refreshPresets(activeProject?.id);
    return archived;
  }

  async function deleteModel(model) {
    const result = await apiFetch(`/api/v1/models/${encodeURIComponent(model.id)}`, token, {
      method: "DELETE",
    });
    if (result.removedManifestEntry) {
      setModels((items) => items.filter((item) => item.id !== model.id));
    }
    setError("");
    await refreshData();
    return result;
  }

  async function deleteLora(lora) {
    const params = new URLSearchParams();
    if (lora.scope) {
      params.set("scope", lora.scope);
    }
    if (lora.scope === "project" && activeProject?.id) {
      params.set("projectId", activeProject.id);
    }
    const query = params.toString() ? `?${params.toString()}` : "";
    const result = await apiFetch(`/api/v1/loras/${encodeURIComponent(lora.id)}${query}`, token, {
      method: "DELETE",
    });
    if (result.removedManifestEntry) {
      setLoras((items) => items.filter((item) => item.id !== lora.id || item.scope !== lora.scope));
    }
    setError("");
    await refreshDataWithLoraOverlay(activeProject?.id);
    return result;
  }

  async function createModelImportJob(payload, options = {}) {
    const { file, ...metadata } = payload;
    if (file?.size > maxModelUploadBytes) {
      throw new Error(`Uploaded model file exceeds the ${uploadLimitLabel(maxModelUploadBytes)} limit`);
    }
    let body;
    if (file) {
      body = new FormData();
      Object.entries(metadata).forEach(([key, value]) => {
        if (value != null && value !== "") {
          body.append(key, value);
        }
      });
      body.append("file", file);
    } else {
      body = JSON.stringify(metadata);
    }
    const job = await apiFetch("/api/v1/models/import", token, {
      method: "POST",
      body,
    });
    setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
    if (options.navigateToQueue ?? false) {
      setActiveView("Queue");
    }
    setError("");
    refreshData();
    return job;
  }

  async function createLoraImportJob(payload, options = {}) {
    if (payload.scope === "project" && !activeProject) {
      throw new Error("Create or open a project first.");
    }
    const { file, ...metadata } = payload;
    if (file?.size > maxLoraUploadBytes) {
      throw new Error("Uploaded LoRA file exceeds the 2GB limit");
    }
    let body;
    if (file) {
      body = new FormData();
      Object.entries({
        ...metadata,
        projectId: metadata.scope === "project" ? activeProject.id : null,
        projectName: metadata.scope === "project" ? activeProject.name : null,
      }).forEach(([key, value]) => {
        if (value != null && value !== "") {
          body.append(key, value);
        }
      });
      body.append("file", file);
    } else {
      body = JSON.stringify({
        ...metadata,
        projectId: metadata.scope === "project" ? activeProject.id : null,
        projectName: metadata.scope === "project" ? activeProject.name : null,
      });
    }
    const job = await apiFetch("/api/v1/loras/import", token, {
      method: "POST",
      body,
    });
    setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
    if (options.navigateToQueue ?? false) {
      setActiveView("Queue");
    }
    setError("");
    refreshDataWithLoraOverlay(activeProject?.id);
    return job;
  }

  async function refreshPersonTracks(projectId = activeProject?.id) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/person-tracks`, token);
      setPersonTracks(items);
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
      if (activeProjectRef.current?.id && activeProjectRef.current.id !== projectId) {
        return;
      }
      setTimelines(items);
      setTimelinesProjectId(projectId);
      setSelectedTimelineId((current) => (items.some((item) => item.id === current) ? current : items[0]?.id ?? null));
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
      if (activeProjectRef.current?.id !== projectId || selectedTimelineIdRef.current !== timelineId) {
        return;
      }
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
      setTimelinesProjectId(activeProject.id);
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

  async function createProject(name) {
    const trimmed = String(name ?? "").trim();
    if (!trimmed) {
      return null;
    }
    try {
      const created = await apiFetch("/api/v1/projects", token, {
        method: "POST",
        body: JSON.stringify({ name: trimmed }),
      });
      setProjects((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      setActiveProject(created);
      setActiveView("Image");
      setError("");
      return created;
    } catch (err) {
      setError(err.message);
      return null;
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
      return null;
    }
    try {
      const job = await apiFetch("/api/v1/image/jobs", token, {
        method: "POST",
        body: JSON.stringify({
          ...payload,
          projectId: activeProject.id,
          projectName: activeProject.name,
          requestedGpu,
        }),
      });
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  function rememberLocalGenerationJob(kind, job) {
    if (!job?.id) {
      return;
    }
    setLocalGenerationJobIds((current) => ({
      ...current,
      // Keep the local review stack compact for burst submissions.
      [kind]: [job.id, ...current[kind].filter((id) => id !== job.id)].slice(0, localJobStackLimit),
    }));
  }

  function hasVisibleLocalFailure(job) {
    const active = activeViewRef.current;
    const localIds = localGenerationJobIdsRef.current;
    if (active === "Image" && localIds.image.includes(job.id)) {
      return true;
    }
    if (active === "Video" && localIds.video.includes(job.id)) {
      return true;
    }
    return active === "Models" && job.type === "model_download";
  }

  async function withCharacterApi(callback) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const result = await callback(activeProject.id);
      setError("");
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function createCharacter(payload) {
    return withCharacterApi(async (projectId) => {
      const created = await apiFetch(`/api/v1/projects/${projectId}/characters`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setCharacters((items) => [created, ...items.filter((item) => item.id !== created.id)]);
      return created;
    });
  }

  async function updateCharacter(characterId, changes) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function archiveCharacter(characterId) {
    return withCharacterApi(async (projectId) => {
      await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/archive`, token, { method: "POST" });
      setCharacters((items) => items.filter((item) => item.id !== characterId));
      return { id: characterId, status: "archived" };
    });
  }

  async function addCharacterReference(characterId, payload) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/references`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function updateCharacterReference(characterId, assetId, changes) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/references/${assetId}`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function removeCharacterReference(characterId, assetId) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/references/${assetId}`, token, {
        method: "DELETE",
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function createCharacterLook(characterId, payload) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/looks`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function updateCharacterLook(characterId, lookId, changes) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/looks/${lookId}`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function deleteCharacterLook(characterId, lookId) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/looks/${lookId}`, token, {
        method: "DELETE",
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function attachCharacterLora(characterId, payload) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/loras`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function updateCharacterLora(characterId, linkId, changes) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/loras/${linkId}`, token, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function detachCharacterLora(characterId, linkId) {
    return withCharacterApi(async (projectId) => {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/loras/${linkId}`, token, {
        method: "DELETE",
      });
      setCharacters((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      return updated;
    });
  }

  async function createCharacterTestJob(characterId, payload) {
    return withCharacterApi(async (projectId) => {
      await apiFetch(`/api/v1/projects/${projectId}/characters/${characterId}/test-jobs`, token, {
        method: "POST",
        body: JSON.stringify({ ...payload, requestedGpu }),
      });
      setActiveView("Queue");
      refreshData();
      return { id: characterId, status: "queued" };
    });
  }

  async function createVideoJob(payload, options = {}) {
    const { navigateToQueue = false } = options;
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch("/api/v1/video/jobs", token, {
        method: "POST",
        body: JSON.stringify({
          ...payload,
          projectId: activeProject.id,
          projectName: activeProject.name,
          requestedGpu,
        }),
      });
      if (navigateToQueue) {
        setActiveView("Queue");
      }
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  function sendAssetToImage(asset, mode = null) {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Image", assetId: asset.id, mode });
    }
    setActiveView("Image");
  }

  function sendAssetToVideo(asset, mode = null) {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Video", assetId: asset.id, mode });
    }
    setActiveView("Video");
  }

  async function createPersonDetectionJob(payload, options = {}) {
    const { navigateToQueue = false } = options;
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch(`/api/v1/projects/${activeProject.id}/person-tracks/detections`, token, {
        method: "POST",
        body: JSON.stringify({ ...payload, requestedGpu }),
      });
      if (navigateToQueue) {
        setActiveView("Queue");
      }
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function createPersonTrackJob(payload, options = {}) {
    const { navigateToQueue = false } = options;
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch(`/api/v1/projects/${activeProject.id}/person-tracks/jobs`, token, {
        method: "POST",
        body: JSON.stringify({ ...payload, requestedGpu }),
      });
      if (navigateToQueue) {
        setActiveView("Queue");
      }
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  function sendCharacterToImage(character, lookId = null) {
    if (!character) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "Image", characterId: character.id, lookId, mode: "character_image" });
    setActiveView("Image");
  }

  function sendCharacterToVideo(character, lookId = null) {
    if (!character) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "Video", characterId: character.id, lookId, mode: "text_to_video" });
    setActiveView("Video");
  }

  async function extractTimelineFrame({ timeline, item, playheadSeconds, intendedUse }) {
    if (!activeProject || !timeline || !item) {
      return null;
    }
    try {
      const job = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${timeline.id}/items/${item.id}/frames`, token, {
        method: "POST",
        body: JSON.stringify({ playheadSeconds, intendedUse, requestedGpu }),
      });
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function queueTimelineVideoJob(payload) {
    return createVideoJob(payload, { navigateToQueue: false });
  }

  function applyTimelineGenerationResult(timeline, job) {
    const payload = job.payload ?? {};
    const action = payload.advanced?.timelineAction;
    const context = payload.advanced?.timelineContext ?? {};
    const assetId = job.result?.assetIds?.[0];
    if (!action || !assetId || context.timelineId !== timeline.id) {
      return timeline;
    }
    const resultAsset = job.result?.assets?.[0];
    const displayName = resultAsset?.displayName ?? "Generated clip";
    const createdAt = resultAsset?.createdAt ?? new Date().toISOString();
    const tracks = timeline.tracks.map((track) => {
      if (track.id !== context.trackId) {
        return track;
      }
      if (action === "bridge") {
        const bridgeItem = ensureItemVersionFields({
          id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
          trackId: track.id,
          assetId,
          type: "video",
          displayName,
          sourceIn: 0,
          sourceOut: Number(payload.duration) || Math.max(0.1, Number(context.timelineEnd) - Number(context.timelineStart)),
          timelineStart: Number(context.timelineStart),
          timelineEnd: Number(context.timelineEnd),
          speed: 1,
          fit: "fit",
          volume: 1,
          versionAssetIds: [assetId],
          currentVersionAssetId: assetId,
          versionHistory: [{ assetId, createdAt, source: "bridge", jobId: job.id, note: "Generated bridge clip" }],
          transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
          transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
        });
        return { ...track, items: [...track.items, bridgeItem] };
      }
      if (action === "extend") {
        const start = Number(context.timelineStart);
        const duration = Number(payload.duration) || 4;
        const extensionItem = ensureItemVersionFields({
          id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
          trackId: track.id,
          assetId,
          type: "video",
          displayName,
          sourceIn: 0,
          sourceOut: duration,
          timelineStart: start,
          timelineEnd: start + duration,
          speed: 1,
          fit: "fit",
          volume: 1,
          versionAssetIds: [assetId],
          currentVersionAssetId: assetId,
          versionHistory: [{ assetId, createdAt, source: "extension", jobId: job.id, note: "Generated extension" }],
          transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
          transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
        });
        return { ...track, items: [...track.items, extensionItem] };
      }
      if (action === "replace") {
        return {
          ...track,
          items: track.items.map((item) => {
            if (item.id !== context.itemId) {
              return item;
            }
            const current = ensureItemVersionFields(item);
            return {
              ...current,
              assetId,
              currentVersionAssetId: assetId,
              type: "video",
              displayName,
              versionAssetIds: Array.from(new Set([...current.versionAssetIds, assetId])),
              versionHistory: [
                ...current.versionHistory,
                { assetId, createdAt, source: "replacement", jobId: job.id, note: "Generated replacement" },
              ],
            };
          }),
        };
      }
      return track;
    });
    return { ...timeline, tracks };
  }

  function enqueueTimelineGenerationApply(job) {
    timelineApplyQueueRef.current = timelineApplyQueueRef.current
      .then(() => applyCompletedTimelineGeneration(job))
      .catch((err) => setError(err.message));
  }

  async function applyCompletedTimelineGeneration(job) {
    const timelineId = job.payload?.advanced?.timelineContext?.timelineId;
    const projectId = job.projectId;
    if (!projectId || !timelineId || !job.result?.assetIds?.length) {
      return;
    }
    try {
      const timeline = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token);
      const updated = applyTimelineGenerationResult(timeline, job);
      if (updated === timeline) {
        return;
      }
      const saved = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token, {
        method: "PUT",
        body: JSON.stringify({ timeline: updated }),
      });
      if (selectedTimelineIdRef.current === timelineId) {
        setActiveTimeline(saved);
      }
      refreshTimelines(projectId);
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

  async function purgeAsset(asset) {
    try {
      await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/purge`, token, { method: "DELETE" });
      setAssets((items) => items.filter((item) => item.id !== asset.id));
      setSelectedAssetId((current) => (current === asset.id ? null : current));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function importAsset(file, options = {}) {
    if (!activeProject || !file) {
      const error = new Error("Create or open a project first.");
      if (options.throwOnError) {
        throw error;
      }
      setError(error.message);
      return;
    }
    const body = new FormData();
    body.append("file", file);
    try {
      const imported = await apiFetch(`/api/v1/projects/${activeProject.id}/assets`, token, {
        method: "POST",
        body,
      });
      setAssets((items) => [imported, ...items.filter((item) => item.id !== imported.id)]);
      setSelectedAssetId(imported.id);
      setError("");
      return imported;
    } catch (err) {
      if (options.throwOnError) {
        throw err;
      }
      setError(err.message);
      return null;
    }
  }

  async function createModelDownloadJob(model) {
    try {
      const job = await apiFetch(`/api/v1/models/${model.id}/download`, token, {
        method: "POST",
        body: JSON.stringify({ requestedGpu: "auto" }),
      });
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function jobAction(job, action) {
    try {
      const path = action === "duplicate" ? `/api/v1/jobs/${job.id}/duplicate` : `/api/v1/jobs/${job.id}/${action}`;
      const body = action === "duplicate" ? { payloadChanges: { duplicatedAt: new Date().toISOString() } } : {};
      const updatedJob = await apiFetch(path, token, { method: "POST", body: JSON.stringify(body) });
      setJobs((items) => [updatedJob, ...items.filter((item) => item.id !== updatedJob.id)].sort(sortNewest));
      setError("");
      await refreshData();
    } catch (err) {
      setError(err.message);
    }
  }

  const titleInfo = viewTitles[activeView] ?? { title: activeView, blurb: "" };
  // Activity dots only — counts live in the topbar so nav button textContent stays clean.
  const activeIndicators = {
    Editor: timelines.length > 0,
    Queue: queueCounts.active > 0,
  };
  // First-run gate: until at least one workspace exists, replace the studio area
  // with a create prompt so navigation never lands on dead, project-scoped controls.
  const needsFirstProject = authenticated && projectsLoaded && projects.length === 0;

  return (
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <img src="/sceneworks-logo.svg" alt="" />
          </span>
          <div>
            <h1>SceneWorks</h1>
            <p>Local creative studio</p>
          </div>
        </div>

        <ProjectSwitcher
          activeProject={activeProject}
          disabled={!authenticated}
          onCreate={createProject}
          onSelect={setActiveProject}
          projects={projects}
        />

        {navSections.map((section) => (
          <div className="sidebar-section" key={section.label}>
            <div className="sidebar-section-title">{section.label}</div>
            <nav className="nav-list">
              {section.items.map((item) => {
                const IconComponent = item.icon;
                const active = activeIndicators[item.id];
                return (
                  <button
                    className={activeView === item.id ? "nav-item active" : "nav-item"}
                    key={item.id}
                    onClick={() => setActiveView(item.id)}
                    title={item.id}
                    type="button"
                  >
                    <IconComponent />
                    <span className="nav-label">{item.id}</span>
                    {active ? <span aria-hidden="true" className="nav-pulse" /> : null}
                  </button>
                );
              })}
            </nav>
          </div>
        ))}

      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{titleInfo.title}</h1>
            <p>{titleInfo.blurb}</p>
          </div>
          <span className="topbar-spacer" />
          <div className="topbar-status">
            <span className={health?.status === "ok" ? "status-pill" : "status-pill warning"}>
              <StatusDot ok={health?.status === "ok"} />
              {health?.status === "ok" ? "API ready" : "API offline"}
            </span>
            <span className="status-pill">
              <span className={visibleWorkers.length ? "dot" : "dot idle"} />
              {visibleWorkers.length ? `${visibleWorkers.length} worker${visibleWorkers.length === 1 ? "" : "s"}` : "No workers"}
            </span>
            <span className="status-pill">
              {gpuOptions.length > 1 ? `${gpuOptions.length - 1} GPU slot${gpuOptions.length === 2 ? "" : "s"}` : "GPU auto"}
            </span>
            <button className="queue-chip" onClick={() => setActiveView("Queue")} type="button">
              Queue {queueCounts.active}
            </button>
          </div>
          <button className="icon-btn" title="Notifications" type="button">
            <Icon.Bell />
          </button>
          <button
            className="icon-btn"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            type="button"
          >
            {theme === "light" ? <Icon.Moon /> : <Icon.Sun />}
          </button>
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

        {needsFirstProject ? (
          <FirstRunProjectGate disabled={!authenticated} onCreate={createProject} />
        ) : (
          <>
        {activeView === "Library" ? (
          <LibraryScreen
            activeProject={activeProject}
            assets={assets}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
            importAsset={importAsset}
            onPreview={setPreviewAsset}
            onSendImage={(asset) => sendAssetToImage(asset)}
            selectedAsset={selectedAsset}
            setSelectedAssetId={setSelectedAssetId}
            onSendVideo={(asset) => sendAssetToVideo(asset)}
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
            characters={characters}
            createImageJob={createImageJob}
            gpuOptions={gpuOptions}
            imageModels={imageModels}
            latestAssets={latestImageAssets}
            launchRequest={studioLaunch}
            loras={loras}
            localJobs={imageLocalJobs}
            onLocalJobCreated={(job) => rememberLocalGenerationJob("image", job)}
            onOpenPresets={() => setActiveView("Presets")}
            onOpenQueue={() => setActiveView("Queue")}
            onPreview={setPreviewAsset}
            presets={presets}
            requestedGpu={requestedGpu}
            selectedAsset={selectedAsset}
            setRequestedGpu={setRequestedGpu}
            updateAssetStatus={updateAssetStatus}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
          />
        ) : null}

        {activeView === "Video" ? (
          <VideoStudio
            activeProject={activeProject}
            assets={assets}
            characters={characters}
            createPersonDetectionJob={createPersonDetectionJob}
            createPersonTrackJob={createPersonTrackJob}
            createVideoJob={createVideoJob}
            deleteAsset={deleteAsset}
            purgeAsset={purgeAsset}
            gpuOptions={gpuOptions}
            latestAssets={latestVideoAssets}
            launchRequest={studioLaunch}
            loras={loras}
            jobs={jobs}
            localJobs={videoLocalJobs}
            onLocalJobCreated={(job) => rememberLocalGenerationJob("video", job)}
            onOpenPresets={() => setActiveView("Presets")}
            onOpenQueue={() => setActiveView("Queue")}
            onPreview={setPreviewAsset}
            onSendToEditor={(asset) => {
              if (asset?.id) {
                setSelectedAssetId(asset.id);
              }
              setActiveView("Editor");
            }}
            personTracks={personTracks}
            presets={presets}
            requestedGpu={requestedGpu}
            selectedAsset={selectedAsset}
            setRequestedGpu={setRequestedGpu}
            updateAssetStatus={updateAssetStatus}
            videoModels={videoModels}
          />
        ) : null}

        {activeView === "Train" ? (
          <TrainingStudio
            activeProject={activeProject}
            authenticated={authenticated}
            assets={assets}
            createDataset={createTrainingDataset}
            datasets={trainingDatasetsProjectId === activeProject?.id ? trainingDatasets : []}
            datasetsError={trainingDatasetsError}
            importAsset={(file) => importAsset(file, { throwOnError: true })}
            loadDataset={loadTrainingDataset}
            loadingDatasets={loadingTrainingDatasets}
            onPreview={setPreviewAsset}
            onRefreshDatasets={() => refreshTrainingDatasets(activeProject?.id)}
            updateDataset={updateTrainingDataset}
          />
        ) : null}

        {activeView === "Presets" ? (
          <PresetManagerScreen
            activeProject={activeProject}
            createPreset={createPreset}
            deletePreset={deletePreset}
            duplicatePreset={duplicatePreset}
            imageModels={imageModels}
            loras={loras}
            onOpenModels={() => setActiveView("Models")}
            presets={presets}
            updatePreset={updatePreset}
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
            jobs={jobs}
            jobPrompt={jobPrompt}
            projectFilter={projectFilter}
            projects={projects}
            requestedGpu={requestedGpu}
            setJobPrompt={setJobPrompt}
            setProjectFilter={setProjectFilter}
            setRequestedGpu={setRequestedGpu}
            workers={visibleWorkers}
          />
        ) : null}

        {activeView === "Models" ? (
          <ModelManagerScreen
            activeProject={activeProject}
            jobs={jobs}
            loras={loras}
            models={models}
            onDeleteLora={deleteLora}
            onDeleteModel={deleteModel}
            onDownloadModel={createModelDownloadJob}
            onImportLora={createLoraImportJob}
            onImportModel={createModelImportJob}
            onOpenQueue={() => setActiveView("Queue")}
            presets={presets}
          />
        ) : null}

        {activeView === "Editor" ? (
          <EditorScreen
            activeProject={activeProject}
            activeTimeline={activeTimeline}
            assets={mediaAssets}
            createTimeline={createTimeline}
            extractTimelineFrame={extractTimelineFrame}
            exportTimeline={exportTimeline}
            onPreview={setPreviewAsset}
            onSendImage={(asset) => sendAssetToImage(asset, "edit_image")}
            onSendVideo={(asset) => sendAssetToVideo(asset, asset?.type === "video" ? "extend_clip" : "image_to_video")}
            queueTimelineVideoJob={queueTimelineVideoJob}
            refreshAssets={refreshAssets}
            saveTimeline={saveTimeline}
            selectedTimelineId={selectedTimelineId}
            setActiveTimeline={setActiveTimeline}
            setSelectedTimelineId={setSelectedTimelineId}
            timelines={timelines}
          />
        ) : null}

        {activeView === "Characters" ? (
          <CharacterStudio
            activeProject={activeProject}
            addCharacterReference={addCharacterReference}
            archiveCharacter={archiveCharacter}
            assets={assets}
            attachCharacterLora={attachCharacterLora}
            characters={characters}
            createCharacter={createCharacter}
            createCharacterLook={createCharacterLook}
            createCharacterTestJob={createCharacterTestJob}
            deleteAsset={deleteAsset}
            deleteCharacterLook={deleteCharacterLook}
            detachCharacterLora={detachCharacterLora}
            imageModels={imageModels}
            latestAssets={latestImageAssets}
            loras={loras}
            onPreview={setPreviewAsset}
            onSendImage={sendCharacterToImage}
            onSendVideo={sendCharacterToVideo}
            purgeAsset={purgeAsset}
            removeCharacterReference={removeCharacterReference}
            updateAssetStatus={updateAssetStatus}
            updateCharacter={updateCharacter}
            updateCharacterLook={updateCharacterLook}
            updateCharacterLora={updateCharacterLora}
            updateCharacterReference={updateCharacterReference}
          />
        ) : null}
          </>
        )}
      </section>

      {previewedAsset ? (
        <FullscreenPreview
          asset={previewedAsset}
          deleteAsset={async (asset) => {
            await deleteAsset(asset);
            setPreviewAsset(null);
          }}
          nextAsset={previewNavigation.next}
          onClose={() => setPreviewAsset(null)}
          onPreviewAsset={setPreviewAsset}
          previousAsset={previewNavigation.previous}
          purgeAsset={async (asset) => {
            await purgeAsset(asset);
            setPreviewAsset(null);
          }}
          updateAssetStatus={updateAssetStatus}
        />
      ) : null}
    </main>
  );
}
