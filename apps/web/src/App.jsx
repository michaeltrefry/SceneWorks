import React, { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, eventUrl, isAbortError } from "./api.js";
import { Icon } from "./components/Icons.jsx";
import { Logo } from "./components/Logo.jsx";
import { StatusDot } from "./components/StatusDot.jsx";
import { FullscreenPreview } from "./components/assetPanels.jsx";
import { fallbackModels, terminalStatuses } from "./constants.js";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { DocumentStudio } from "./screens/DocumentStudio.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { TrainingDataSetsLibrary, TrainingStudio } from "./screens/TrainingStudio.jsx";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { EditorScreen } from "./screens/EditorScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
import { PresetManagerScreen } from "./screens/PresetManagerScreen.jsx";
import { SettingsScreen } from "./screens/SettingsScreen.jsx";
import { SetupWizard } from "./screens/SetupWizard.jsx";
import { sortNewest, sortWorkers } from "./sorters.js";
import { useCharacters } from "./hooks/useCharacters.js";
import { usePresets } from "./hooks/usePresets.js";
import { useTraining } from "./hooks/useTraining.js";
import { useModelsAndLoras } from "./hooks/useModelsAndLoras.js";
import { usePersonTracks } from "./hooks/usePersonTracks.js";
import { useTimelines } from "./hooks/useTimelines.js";
import { AppContext } from "./context/AppContext.js";

// Desktop (Tauri) shell detection. The first-run setup wizard is desktop-only;
// web/Docker keep the existing first-run project gate. Tauri commands persist the
// wizard state (the API binds a random port each launch, so localStorage — keyed
// to the origin — can't be relied on across launches).
const isDesktopShell = typeof window !== "undefined" && !!window.__TAURI__;
const tauriInvoke = (command, args) => window.__TAURI__.core.invoke(command, args);

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

function isLoraTrainingNotice(message) {
  return String(message ?? "").startsWith("lora training: ");
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

// Studios show only the most recent run's progress card; starting a new run
// replaces the previous one rather than stacking cards.
const localJobStackLimit = 1;

const navSections = [
  {
    label: "Workspace",
    items: [
      { id: "Image", icon: Icon.Image },
      { id: "Video", icon: Icon.Video },
      { id: "Document", icon: Icon.Wand },
      { id: "Train", icon: Icon.Train },
      { id: "Editor", icon: Icon.Editor },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "Library", label: "Assets", icon: Icon.Library },
      { id: "LibraryDataSets", label: "Data Sets", icon: Icon.Train },
      { id: "Characters", icon: Icon.Character },
      { id: "Presets", icon: Icon.Preset },
      { id: "Models", icon: Icon.Model },
    ],
  },
  {
    label: "System",
    items: [
      { id: "Queue", icon: Icon.Queue },
      { id: "Settings", icon: Icon.Sliders },
    ],
  },
];

const viewTitles = {
  Library: { title: "Assets", blurb: "Browse stills and clips across all your projects." },
  LibraryDataSets: { title: "Data Sets", blurb: "Create and caption training datasets." },
  Image: { title: "Image Studio", blurb: "Describe what you want — we'll render variations side by side." },
  Video: { title: "Video Studio", blurb: "Bring stills to life, or render new clips from scratch." },
  Document: { title: "Document Studio", blurb: "Generate interleaved text-image documents — guides, storyboards, tutorials." },
  Train: { title: "Training Studio", blurb: "Build datasets and prepare LoRA training plans." },
  Editor: { title: "Editor", blurb: "Cut, sequence and export your timeline." },
  Characters: { title: "Characters", blurb: "Keep the same face across every shot." },
  Presets: { title: "Presets", blurb: "Save and share recurring generation setups." },
  Models: { title: "Models", blurb: "Download, import and manage local checkpoints." },
  Queue: { title: "Queue", blurb: "All running and recent jobs across workers." },
  Settings: { title: "Settings", blurb: "Paths, Hugging Face token, and detected GPU." },
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
          <Logo size={52} />
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

export function App() {
  const [health, setHealth] = useState(null);
  const [access, setAccess] = useState({ authRequired: false });
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  const [projects, setProjects] = useState([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  // Desktop first-run wizard gate: null = unknown (still reading on desktop),
  // true = no wizard needed (web, or already completed), false = show the wizard.
  const [setupCompleted, setSetupCompleted] = useState(isDesktopShell ? null : true);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [jobs, setJobs] = useState([]);
  const [localGenerationJobIds, setLocalGenerationJobIds] = useState({ image: [], video: [] });
  const [workers, setWorkers] = useState([]);
  const [queueSummary, setQueueSummary] = useState(null);
  const [trainingTargets, setTrainingTargets] = useState({ schemaVersion: 1, targets: [] });
  const [trainingPresets, setTrainingPresets] = useState({ schemaVersion: 1, presets: [] });
  const [trainingTargetsError, setTrainingTargetsError] = useState("");
  const [trainingPresetsError, setTrainingPresetsError] = useState("");
  const [assets, setAssets] = useState([]);
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
  const generatedAssetRefreshesRef = useRef(new Map());

  const {
    characters,
    setCharacters,
    refreshCharacters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    addCharacterReference,
    updateCharacterReference,
    removeCharacterReference,
    createCharacterLook,
    updateCharacterLook,
    deleteCharacterLook,
    attachCharacterLora,
    updateCharacterLora,
    detachCharacterLora,
    createCharacterTestJob,
  } = useCharacters({ token, activeProject, setError, requestedGpu, setActiveView });

  const {
    presets,
    setPresets,
    refreshPresets,
    createPreset,
    updatePreset,
    duplicatePreset,
    deletePreset,
  } = usePresets({ token, activeProject, setError });

  const {
    trainingDatasets,
    setTrainingDatasets,
    trainingDatasetsProjectId,
    setTrainingDatasetsProjectId,
    loadingTrainingDatasets,
    trainingDatasetsError,
    setTrainingDatasetsError,
    refreshTrainingDatasets,
    loadTrainingDataset,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingJob,
  } = useTraining({ token, activeProject, setError, setJobs });

  const {
    models,
    setModels,
    loras,
    setLoras,
    refreshLoras,
    deleteModel,
    deleteLora,
    createModelImportJob,
    createLoraImportJob,
    createModelDownloadJob,
    createModelConvertJob,
  } = useModelsAndLoras({
    token,
    activeProject,
    setError,
    setJobs,
    setActiveView,
    refreshData,
    refreshDataWithLoraOverlay,
  });

  const {
    personTracks,
    setPersonTracks,
    refreshPersonTracks,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections,
  } = usePersonTracks({ token, activeProject, setError, requestedGpu, setActiveView });

  const {
    timelines,
    setTimelines,
    setTimelinesProjectId,
    selectedTimelineId,
    setSelectedTimelineId,
    activeTimeline,
    setActiveTimeline,
    refreshTimelines,
    createTimeline,
    saveTimeline,
    exportTimeline,
    extractTimelineFrame,
    queueTimelineVideoJob,
    enqueueTimelineGenerationApply,
  } = useTimelines({
    token,
    activeProject,
    activeProjectRef,
    setError,
    requestedGpu,
    setActiveView,
    createVideoJob,
  });

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
  // Person-workflow readiness, derived from the live (non-offline) workers so it
  // tracks SSE worker registration/offline transitions instantly. Mirrors the
  // server's GET /api/v1/capabilities/person (person_readiness_from_workers); the
  // worker SSE handlers keep `workers` current, so this never goes stale.
  const personReadiness = useMemo(() => {
    const live = workers.filter((worker) => worker.status !== "offline");
    const ready = (capability) => live.some((worker) => (worker.capabilities ?? []).includes(capability));
    return {
      detect: { capability: "person_detect", ready: ready("person_detect") },
      track: { capability: "person_track", ready: ready("person_track") },
      segment: { capability: "person_segment", ready: ready("person_segment") },
      replace: { capability: "person_replace", ready: ready("person_replace") },
      detectPreview: { capability: "person_detect_preview", ready: ready("person_detect_preview") },
      trackPreview: { capability: "person_track_preview", ready: ready("person_track_preview") },
    };
  }, [workers]);
  const gpuOptions = useMemo(() => {
    const ids = visibleWorkers.filter(isSelectableGpuWorker).map((worker) => worker.gpuId);
    return ["auto", ...Array.from(new Set(ids))];
  }, [visibleWorkers]);
  const mediaAssets = useMemo(
    () => assets.filter((asset) => ["image", "video", "upload", "frame", "render", "document"].includes(asset.type)),
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
    if (!isDesktopShell) {
      return;
    }
    tauriInvoke("get_storage_setup")
      .then((setup) => setSetupCompleted(Boolean(setup?.setupCompleted)))
      // Never block the app on a storage-state read failure; fall through to the studio.
      .catch(() => setSetupCompleted(true));
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
      setTrainingTargetsError("");
      setTrainingDatasets([]);
      setTrainingDatasetsProjectId(null);
      setTrainingDatasetsError("");
      setSelectedTimelineId(null);
      setActiveTimeline(null);
      return;
    }
    // Switching projects (or unmounting) aborts the previous project's in-flight
    // loads so a slow response can't overwrite the newly-selected project's data.
    const controller = new AbortController();
    const { signal } = controller;
    refreshAssets(activeProject.id, { signal });
    refreshCharacters(activeProject.id, { signal });
    refreshLoras(activeProject.id, { signal });
    refreshPresets(activeProject.id, { signal });
    refreshTrainingDatasets(activeProject.id, { signal });
    refreshPersonTracks(activeProject.id, { signal });
    refreshTimelines(activeProject.id, { signal });
    return () => controller.abort();
  }, [activeProject?.id, authenticated, token]);

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
      if (job.status === "completed" && job.type === "lora_train" && job.payload?.dryRun === false) {
        if (job.result?.loraRegistered === false) {
          setError(`lora training: ${job.result?.loraRegistrationError ?? "Completed training but could not register the LoRA."}`);
        } else {
          setError((current) => (isLoraTrainingNotice(current) ? "" : current));
          refreshDataWithLoraOverlay(job.projectId ?? activeProjectRef.current?.id);
        }
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
    const [
      projectsResult,
      jobsResult,
      workersResult,
      modelsResult,
      lorasResult,
      presetsResult,
      trainingTargetsResult,
      trainingPresetsResult,
    ] =
      await Promise.all([
        fetchInitial("Projects", "/api/v1/projects", []),
        fetchInitial("Jobs", "/api/v1/jobs", []),
        fetchInitial("Workers", "/api/v1/workers", []),
        fetchInitial("Models", "/api/v1/models", []),
        fetchInitial("LoRAs", "/api/v1/loras", []),
        fetchInitial("Presets", "/api/v1/recipe-presets", [], true),
        fetchInitial("Training targets", "/api/v1/training/targets", { schemaVersion: 1, targets: [] }),
        fetchInitial("Training presets", "/api/v1/training/presets", { schemaVersion: 1, presets: [] }),
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
    setTrainingTargets(trainingTargetsResult.value);
    setTrainingTargetsError(trainingTargetsResult.error);
    setTrainingPresets(trainingPresetsResult.value);
    setTrainingPresetsError(trainingPresetsResult.error);
    setError(
      [
        projectsResult,
        jobsResult,
        workersResult,
        modelsResult,
        lorasResult,
        presetsResult,
        trainingTargetsResult,
        trainingPresetsResult,
      ]
        .map((result) => result.error)
        .filter(Boolean)
        .join("; "),
    );
  }

  async function refreshAssets(projectId = activeProject?.id, { signal } = {}) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/assets?includeRejected=true&includeTrashed=true`, token, { signal });
      setAssets(items);
      const defaultAsset = items.find((asset) => !asset.status?.trashed && !asset.status?.rejected) ?? items[0] ?? null;
      setSelectedAssetId((current) => current ?? defaultAsset?.id ?? null);
      setError("");
    } catch (err) {
      if (isAbortError(err)) return;
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


  function saveToken(event) {
    event.preventDefault();
    window.localStorage.setItem("sceneworks-token", token);
    setError("");
    refreshData();
  }

  async function completeSetupWizard() {
    try {
      await tauriInvoke("complete_setup");
    } catch {
      // Persisting the marker failed; still dismiss the wizard so the user isn't
      // trapped. Worst case it re-appears next launch.
    }
    setSetupCompleted(true);
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
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function createVqaJob(asset, question, maxNewTokens) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch("/api/v1/image/vqa/jobs", token, {
        method: "POST",
        body: JSON.stringify({
          projectId: activeProject.id,
          projectName: activeProject.name,
          sourceAssetId: asset.id,
          question,
          maxNewTokens,
          requestedGpu,
        }),
      });
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function createInterleaveJob(payload) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch("/api/v1/image/interleave/jobs", token, {
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
      // Only the latest run is shown, so a new submission replaces the previous id.
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

  async function updateAssetTags(asset, tags) {
    try {
      const updated = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/tags`, token, {
        method: "PATCH",
        body: JSON.stringify({ tags }),
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
      setAssets((items) =>
        items.map((item) =>
          item.id === asset.id ? { ...item, status: { ...item.status, trashed: true } } : item,
        ),
      );
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

  async function jobAction(job, action) {
    try {
      const path = action === "duplicate" ? `/api/v1/jobs/${job.id}/duplicate` : `/api/v1/jobs/${job.id}/${action}`;
      const body = action === "duplicate" ? { payloadChanges: { duplicatedAt: new Date().toISOString() } } : {};
      const updatedJob = await apiFetch(path, token, { method: "POST", body: JSON.stringify(body) });
      setJobs((items) => [updatedJob, ...items.filter((item) => item.id !== updatedJob.id)].sort(sortNewest));
      setError("");
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
  // Desktop first-run wizard (sc-1473): supersedes the project gate while the
  // completion marker is unset. `null` means we're still reading the marker on
  // desktop — hold the studio/gate back briefly to avoid a flash.
  const setupGateLoading = isDesktopShell && setupCompleted === null;
  const showSetupWizard = isDesktopShell && setupCompleted === false && authenticated;

  // sc-1651 Phase B: shared primitives screens read via useAppContext() instead of
  // drilled props. Screens build any screen-specific wrappers from these (e.g. a
  // send-to-studio action with a mode). Grown one screen at a time as screens convert.
  const appContextValue = {
    activeProject,
    mediaAssets,
    setPreviewAsset,
    sendAssetToImage,
    sendAssetToVideo,
    activeTimeline,
    timelines,
    selectedTimelineId,
    setSelectedTimelineId,
    setActiveTimeline,
    createTimeline,
    saveTimeline,
    exportTimeline,
    extractTimelineFrame,
    queueTimelineVideoJob,
    // Assets / library (sc-1651 Phase B batch 1)
    assets,
    selectedAsset,
    setSelectedAssetId,
    deleteAsset,
    purgeAsset,
    importAsset,
    updateAssetStatus,
    updateAssetTags,
    latestImageAssets,
    // Jobs / queue
    jobs,
    jobAction,
    createVqaJob,
    createInterleaveJob,
    // Queue screen (sc-1651 Phase B batch 2)
    createPlaceholderJob,
    filteredJobs,
    jobPrompt,
    setJobPrompt,
    projectFilter,
    setProjectFilter,
    projects,
    visibleWorkers,
    // Generation studios (sc-1651 Phase B batch 3)
    createVideoJob,
    createImageJob,
    latestVideoAssets,
    videoLocalJobs,
    imageLocalJobs,
    studioLaunch,
    rememberLocalGenerationJob,
    // Person tracks (Video Studio + Replace Person)
    personTracks,
    personReadiness,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections,
    // Models / GPU
    imageModels,
    videoModels,
    models,
    loras,
    deleteLora,
    deleteModel,
    createModelDownloadJob,
    createModelConvertJob,
    createLoraImportJob,
    createModelImportJob,
    gpuOptions,
    requestedGpu,
    setRequestedGpu,
    // Presets
    presets,
    createPreset,
    updatePreset,
    deletePreset,
    duplicatePreset,
    // Training (sc-1651 Phase B batch 7)
    authenticated,
    trainingDatasets,
    trainingDatasetsProjectId,
    trainingDatasetsError,
    loadingTrainingDatasets,
    refreshTrainingDatasets,
    loadTrainingDataset,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingJob,
    trainingPresets,
    trainingPresetsError,
    trainingTargets,
    trainingTargetsError,
    // Navigation
    setActiveView,
    // Characters
    characters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    addCharacterReference,
    updateCharacterReference,
    removeCharacterReference,
    createCharacterLook,
    updateCharacterLook,
    deleteCharacterLook,
    attachCharacterLora,
    updateCharacterLora,
    detachCharacterLora,
    createCharacterTestJob,
    sendCharacterToImage,
    sendCharacterToVideo,
  };

  return (
    <AppContext.Provider value={appContextValue}>
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <Logo size={32} />
          </span>
          <div>
            <h1>Scene<span className="light">Works</span></h1>
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
                const label = item.label ?? item.id;
                return (
                  <button
                    className={activeView === item.id ? "nav-item active" : "nav-item"}
                    key={item.id}
                    onClick={() => setActiveView(item.id)}
                    title={label}
                    type="button"
                  >
                    <IconComponent />
                    <span className="nav-label">{label}</span>
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

        {showSetupWizard ? (
          <SetupWizard
            jobs={jobs}
            models={models}
            onComplete={completeSetupWizard}
            onCreateProject={createProject}
            onDownloadModel={createModelDownloadJob}
            onOpenQueue={() => setActiveView("Queue")}
          />
        ) : setupGateLoading ? null : needsFirstProject ? (
          <FirstRunProjectGate disabled={!authenticated} onCreate={createProject} />
        ) : (
          <>
        {activeView === "Library" ? (
          <LibraryScreen />
        ) : null}

        {activeView === "LibraryDataSets" ? (
          <TrainingDataSetsLibrary />
        ) : null}

        {activeView === "Image" ? (
          <ImageStudio />
        ) : null}

        {activeView === "Video" ? (
          <VideoStudio />
        ) : null}

        {activeView === "Document" ? (
          <DocumentStudio />
        ) : null}

        {activeView === "Train" ? (
          <TrainingStudio />
        ) : null}

        {activeView === "Presets" ? (
          <PresetManagerScreen />
        ) : null}

        {activeView === "Queue" ? (
          <QueueScreen />
        ) : null}

        {activeView === "Models" ? (
          <ModelManagerScreen />
        ) : null}

        {activeView === "Editor" ? (
          <EditorScreen />
        ) : null}

        {activeView === "Characters" ? (
          <CharacterStudio />
        ) : null}
        {activeView === "Settings" ? <SettingsScreen /> : null}
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
    </AppContext.Provider>
  );
}
