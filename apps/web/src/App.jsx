import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, eventUrl, isAbortError } from "./api.js";
import { Icon } from "./components/Icons.jsx";
import { Logo } from "./components/Logo.jsx";
import { StatusDot } from "./components/StatusDot.jsx";
import { FullscreenPreview } from "./components/assetPanels.jsx";
import { fallbackModels, terminalStatuses } from "./constants.js";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { PoseLibraryScreen } from "./screens/PoseLibraryScreen.jsx";
import { KeyPointLibraryScreen } from "./screens/KeyPointLibraryScreen.jsx";
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
import { LogsScreen } from "./screens/LogsScreen.jsx";
import { LicensesScreen } from "./screens/LicensesScreen.jsx";
import { SetupWizard } from "./screens/SetupWizard.jsx";
import { editModelForAsset } from "./presetUtils.js";
import { sortNewest, sortOldest, sortWorkers } from "./sorters.js";
import { useCharacters } from "./hooks/useCharacters.js";
import { usePresets } from "./hooks/usePresets.js";
import { useTraining } from "./hooks/useTraining.js";
import { useModelsAndLoras } from "./hooks/useModelsAndLoras.js";
import { usePersonTracks } from "./hooks/usePersonTracks.js";
import { useTimelines } from "./hooks/useTimelines.js";
import { AppContext } from "./context/AppContext.js";
import { DEFAULT_MAC_CAPABILITIES } from "./macGating.js";
import { ACCENTS, DEFAULT_ACCENT, isAccentId } from "./accents.js";
import {
  dropUpscaledVariants,
  findFoldedAssetById,
  foldUpscaledAssetVariants,
  restrictFoldedToScope,
} from "./assetVariants.js";
import { buildWorkersById } from "./workers.js";
import { isDesktop as isDesktopShell, tauriInvoke } from "./runtime.js";

// Desktop (Tauri) shell detection (unified helper, epic 4484 story 6). The first-run
// setup wizard is desktop-only; web/Docker (and a remote LAN browser) keep the
// existing first-run project gate. Tauri commands persist the wizard state (the API
// binds a random port each launch, so localStorage — keyed to the origin — can't be
// relied on across launches).

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

function isVideoGenerationJob(job) {
  return ["video_generate", "video_extend", "video_bridge"].includes(job.type);
}

function isInterleaveJob(job) {
  return job.type === "image_interleave";
}

function parseSseJson(event, label) {
  try {
    return JSON.parse(event.data);
  } catch (err) {
    console.warn(`Ignoring malformed ${label} SSE event`, err);
    return null;
  }
}

function abortableDelay(ms, signal) {
  if (signal?.aborted) {
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

// sc-4198: notice kind for a job-failure banner. LoRA import/train failures get
// their own kind so the matching job's later completion dismisses exactly that
// banner (replacing the old "lora import:"/"lora training:" startsWith protocol);
// everything else is a general error.
function noticeKindForJob(job) {
  if (job?.type === "lora_import") return "lora-import";
  if (job?.type === "lora_train") return "lora-train";
  return "general";
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

// Studios stack every running and queued run (plus the most recent finished run
// until its successor starts), so a new submission no longer evicts the prior
// progress card. Capped so a long session can't grow the visible stack unbounded.
const localJobStackLimit = 25;

// Build a studio's local-job stack: the runs it explicitly remembered plus any
// still-active generation jobs for the open project, de-duped and ordered
// oldest-first (running run on top, queued runs following in execution order),
// keeping only the most recent `localJobStackLimit` entries.
function buildLocalJobStack(rememberedIds, jobs, activeProjectId, isGenerationJob) {
  const remembered = rememberedIds.map((id) => jobs.find((job) => job.id === id)).filter(Boolean);
  const projectJobs = jobs.filter(
    (job) =>
      activeProjectId &&
      job.projectId === activeProjectId &&
      isGenerationJob(job) &&
      !terminalStatuses.has(job.status),
  );
  const byId = new Map();
  [...remembered, ...projectJobs].forEach((job) => {
    if (job?.id && !byId.has(job.id)) {
      byId.set(job.id, job);
    }
  });
  return Array.from(byId.values()).sort(sortOldest).slice(-localJobStackLimit);
}

// Lazy-load the canvas editor so Konva (canvas-based, heavy) stays out of the
// initial bundle and the jsdom test path — it only loads when the view is opened.
const ImageEditor = React.lazy(() =>
  import("./screens/ImageEditor.jsx").then((module) => ({ default: module.ImageEditor })),
);

const navSections = [
  {
    label: "Workspace",
    items: [
      { id: "Image", icon: Icon.Image },
      { id: "Video", icon: Icon.Video },
      // Character Studio is a generative studio (sc-2300) — it sits with Image/Video,
      // below Video and above Training, not in the Library section.
      { id: "Characters", icon: Icon.Character },
      { id: "Document", icon: Icon.Wand },
      { id: "Train", icon: Icon.Train },
      { id: "ImageEditor", label: "Image Editor", icon: Icon.ImageEditor },
      { id: "Editor", label: "Video Editor", icon: Icon.Editor },
    ],
  },
  {
    label: "Library",
    items: [
      { id: "Library", label: "Assets", icon: Icon.Library },
      { id: "LibraryDataSets", label: "Data Sets", icon: Icon.Train },
      { id: "Poses", label: "Pose Library", icon: Icon.Character },
      { id: "Keypoints", label: "Key Point Library", icon: Icon.Character },
      { id: "Presets", icon: Icon.Preset },
      { id: "Models", icon: Icon.Model },
    ],
  },
  {
    label: "System",
    items: [
      { id: "Queue", icon: Icon.Queue },
      { id: "Logs", icon: Icon.Logs },
      { id: "Settings", icon: Icon.Sliders },
      { id: "Licenses", icon: Icon.Info },
    ],
  },
];

const viewTitles = {
  Library: { title: "Assets", blurb: "Browse stills and clips across all your projects." },
  LibraryDataSets: { title: "Data Sets", blurb: "Create and caption training datasets." },
  Poses: { title: "Pose Library", blurb: "Manage whole-body pose skeletons and create new ones from photos." },
  Keypoints: {
    title: "Key Point Library",
    blurb: "Capture face-angle framing presets and compose angle-set collections for character turnarounds.",
  },
  Image: { title: "Image Studio", blurb: "Describe what you want — we'll render variations side by side." },
  Video: { title: "Video Studio", blurb: "Bring stills to life, or render new clips from scratch." },
  Document: { title: "Document Studio", blurb: "Generate interleaved text-image documents — guides, storyboards, tutorials." },
  Train: { title: "Training Studio", blurb: "Build datasets and prepare LoRA training plans." },
  Editor: { title: "Video Editor", blurb: "Cut, sequence and export your timeline." },
  ImageEditor: { title: "Image Editor", blurb: "Crop, upscale and refine a single image on a canvas." },
  Characters: { title: "Characters", blurb: "Keep the same face across every shot." },
  Presets: { title: "Presets", blurb: "Save and share recurring generation setups." },
  Models: { title: "Models", blurb: "Download, import and manage local checkpoints." },
  Queue: { title: "Queue", blurb: "All running and recent jobs across workers." },
  Logs: { title: "Logs", blurb: "This session's activity — routing decisions, worker phases and errors." },
  Settings: { title: "Settings", blurb: "Paths, service tokens, and detected GPU." },
  Licenses: {
    title: "Licenses",
    blurb: "Third-party components bundled with SceneWorks and their license notices.",
  },
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

function readStoredAccent() {
  if (typeof window === "undefined") {
    return DEFAULT_ACCENT;
  }
  try {
    const saved = window.localStorage.getItem("sceneworks-accent");
    return isAccentId(saved) ? saved : DEFAULT_ACCENT;
  } catch {
    return DEFAULT_ACCENT;
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

          {projects.length ? <div className="project-menu-divider" role="separator" /> : null}

          {projects.length === 0 ? (
            <p className="project-menu-empty">No workspaces yet — create the first one above.</p>
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
  // Whether GET /api/v1/access has answered yet. Until it does we don't know if a
  // password is required, so a remote browser must hold its protected data loads —
  // otherwise they fire optimistically, 401, and bury the password prompt under a band
  // of "access token required" errors (epic 4484).
  const [accessResolved, setAccessResolved] = useState(false);
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  // What the user is typing into the login gate (sc-8808). Kept separate from the
  // live `token` so keystrokes never flip `authenticated` or churn the data/SSE
  // effects; `token` only changes once /api/v1/auth/verify accepts the draft.
  const [passwordDraft, setPasswordDraft] = useState("");
  // Wrong-password feedback for the remote-browser login gate (epic 4484 story 7).
  const [authError, setAuthError] = useState("");
  const [projects, setProjects] = useState([]);
  const [projectsLoaded, setProjectsLoaded] = useState(false);
  // Desktop first-run wizard gate: null = unknown (still reading on desktop),
  // true = no wizard needed (web, or already completed), false = show the wizard.
  const [setupCompleted, setSetupCompleted] = useState(isDesktopShell ? null : true);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [jobs, setJobs] = useState([]);
  const [localGenerationJobIds, setLocalGenerationJobIds] = useState({ image: [], video: [], document: [] });
  const [workers, setWorkers] = useState([]);
  const [queueSummary, setQueueSummary] = useState(null);
  // Mac UI gating (sc-3486): inert until the capabilities endpoint reports macGatingActive.
  const [macCapabilities, setMacCapabilities] = useState(DEFAULT_MAC_CAPABILITIES);
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
  // The collection the fullscreen preview was launched from, as an ordered list
  // of asset ids. Navigation (next/previous and the discard-advance) stays bound
  // to this set so scrolling never escapes into the Library or another
  // character's assets. `null` falls back to "all assets" for any legacy caller
  // that opens the preview without a scope.
  const [previewScopeIds, setPreviewScopeIds] = useState(null);
  // Which way the user last scrolled in the fullscreen preview, so discarding an
  // asset advances in that same direction.
  const previewDirectionRef = useRef("next");
  // Open the fullscreen preview bound to the collection it was launched from.
  // `scopeAssets` is the exact list the calling gallery rendered (folded or not);
  // we snapshot its ids so navigation tracks the live asset state but never
  // wanders outside that collection. Passing no scope clears it (global nav).
  // sc-4194: useCallback so the context-exposed setPreviewAsset identity is stable.
  const openPreview = useCallback((asset, scopeAssets) => {
    if (!asset) {
      setPreviewScopeIds(null);
      setPreviewAsset(null);
      return;
    }
    setPreviewScopeIds(
      Array.isArray(scopeAssets) && scopeAssets.length
        ? scopeAssets.map((item) => item.id)
        : null,
    );
    setPreviewAsset(asset);
  }, []);
  const closePreview = () => {
    setPreviewScopeIds(null);
    setPreviewAsset(null);
  };
  const [studioLaunch, setStudioLaunch] = useState(null);
  // sc-8730: the launch channel INTO the Image Editor canvas (crop/upscale/refine
  // screen, activeView === "ImageEditor"). Mirrors studioLaunch but targets the
  // editor's openAsset(assetId) rather than Image Studio. `id` is a fresh UUID per
  // launch so relaunching the same asset still fires the editor's id-keyed effect.
  const [editorLaunch, setEditorLaunch] = useState(null);
  // sc-4198: a small notices store replaces the single `error` string that used to
  // double as a message bus — the fragile "lora import:"/"lora training:" startsWith
  // protocol. Each notice has a stable `kind`; pushing a kind replaces only that kind
  // and dismissing clears only that kind, so an unrelated success (or a background SSE
  // refresh) no longer wipes an unread, still-relevant notice of a different kind.
  const [notices, setNotices] = useState([]);
  const pushNotice = useCallback((kind, message) => {
    const text = String(message ?? "");
    setNotices((current) => {
      const others = current.filter((notice) => notice.kind !== kind);
      return text ? [...others, { kind, message: text }] : others;
    });
  }, []);
  const dismissNoticeKind = useCallback((kind) => {
    setNotices((current) => current.filter((notice) => notice.kind !== kind));
  }, []);
  // Back-compat: the existing setError(msg)/setError("") call sites map onto the
  // "general" notice kind — a truthy message replaces it, "" dismisses only it.
  const setError = useCallback((message) => pushNotice("general", message), [pushNotice]);
  const [theme, setTheme] = useState(readStoredTheme);
  // Apply a theme and persist it through the API. localStorage gives an instant
  // initial paint, but on the desktop shell the UI runs at the API's per-launch
  // http://127.0.0.1:<port> origin, where both localStorage and Tauri IPC are
  // unreliable across launches — so the durable copy lives server-side.
  const changeTheme = (next) => {
    setTheme(next);
    apiFetch("/api/v1/ui-preferences", "", {
      method: "PUT",
      body: JSON.stringify({ theme: next }),
    }).catch(() => {});
  };
  const [accent, setAccent] = useState(readStoredAccent);
  // Same persistence contract as theme: instant localStorage cache + durable
  // server copy. The PUT sends only the changed field, so the endpoint must
  // MERGE partial updates (theme writes already rely on this).
  const changeAccent = (next) => {
    setAccent(next);
    apiFetch("/api/v1/ui-preferences", "", {
      method: "PUT",
      body: JSON.stringify({ accent: next }),
    }).catch(() => {});
  };
  const activeProjectRef = useRef(null);
  const activeViewRef = useRef(activeView);
  const localGenerationJobIdsRef = useRef(localGenerationJobIds);
  const generatedAssetRefreshesRef = useRef(new Map());
  const refreshDataRef = useRef(null);
  const refreshAssetsRef = useRef(null);
  const refreshCharactersRef = useRef(null);
  const refreshLorasRef = useRef(null);
  const refreshPresetsRef = useRef(null);
  const refreshTrainingDatasetsRef = useRef(null);
  const refreshPersonTracksRef = useRef(null);
  const refreshTimelinesRef = useRef(null);
  const refreshDataWithLoraOverlayRef = useRef(null);
  // A screen (the Image Editor, sc-2434) can register a guard that runs before a
  // user-initiated navigation leaves it — e.g. to confirm discarding unsaved edits.
  // Programmatic setActiveView calls (post-generation hops) deliberately bypass it.
  const leaveGuardRef = useRef(null);
  const registerLeaveGuard = useCallback((guard) => {
    leaveGuardRef.current = guard;
    return () => {
      if (leaveGuardRef.current === guard) leaveGuardRef.current = null;
    };
  }, []);
  const navTo = useCallback((viewId) => {
    if (viewId === activeViewRef.current) return;
    const guard = leaveGuardRef.current;
    if (guard && !guard()) return; // guard returned false → user cancelled the leave
    setActiveView(viewId);
  }, []);

  // sc-4194: defined here (above the data hooks) because useTimelines takes it as a
  // dependency; a stable identity keeps the timeline hook's queue action stable too.
  const createVideoJob = useCallback(
    async (payload, options = {}) => {
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
    },
    [token, activeProject, requestedGpu],
  );

  const {
    characters,
    setCharacters,
    refreshCharacters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    unarchiveCharacter,
    listArchivedCharacters,
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
    loadTrainingDatasetReadiness,
    setTrainingDatasetItemQualityAck,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingDatasetUpscaleJob,
    createTrainingDatasetAnalysisJob,
    createTrainingDatasetFaceAnalysisJob,
    smartCropTrainingDataset,
    stripExifTrainingDataset,
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
    createLoraDownloadJob,
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

  // The desktop shell reaches its own API over loopback, which the API trusts
  // (SCENEWORKS_TRUST_LOOPBACK), so it's authenticated without a password — never prompt
  // for one locally (epic 4484). A remote browser must wait for GET /api/v1/access before
  // it knows whether a password is needed; until then it holds its protected loads rather
  // than firing them unauthenticated.
  const authenticated = useMemo(
    () =>
      isDesktopShell ||
      (accessResolved && (!access.authRequired || token.length > 0)),
    [accessResolved, access, token],
  );
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
  // Discarded (trashed) assets are excluded from the fullscreen navigation so
  // they don't show up while scrolling; purged assets are already dropped from
  // `assets` entirely.
  const foldedPreviewAssets = useMemo(
    () => foldUpscaledAssetVariants(assets.filter((asset) => !asset.status?.trashed)),
    [assets],
  );
  const previewedAsset = useMemo(
    () => (previewAsset ? findFoldedAssetById(foldedPreviewAssets, previewAsset.id) ?? previewAsset : null),
    [foldedPreviewAssets, previewAsset],
  );
  // The ordered, folded set the preview can navigate — restricted to the launch
  // collection so scrolling never escapes into the Library or another character.
  const previewScopeAssets = useMemo(
    () => restrictFoldedToScope(foldedPreviewAssets, previewScopeIds),
    [foldedPreviewAssets, previewScopeIds],
  );
  const previewNavigation = useMemo(() => {
    if (!previewedAsset || previewScopeAssets.length < 2) {
      return { previous: null, next: null };
    }
    const currentIndex = previewScopeAssets.findIndex((asset) => asset.id === previewedAsset.id);
    if (currentIndex < 0) {
      return { previous: null, next: null };
    }
    return {
      previous: currentIndex > 0 ? previewScopeAssets[currentIndex - 1] : null,
      next: currentIndex < previewScopeAssets.length - 1 ? previewScopeAssets[currentIndex + 1] : null,
    };
  }, [previewScopeAssets, previewedAsset]);
  const latestAssets = useMemo(
    () => assets.filter((asset) => asset.generationSetId === latestGenerationSetId),
    [assets, latestGenerationSetId],
  );
  const latestImageAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "image"), [latestAssets]);
  const latestVideoAssets = useMemo(() => latestAssets.filter((asset) => asset.type === "video"), [latestAssets]);
  // Recent Assets (sc-2088 / sc-2089) — the 20 most recent image/video assets
  // generated in the active project. Replaces `latestImageAssets`/
  // `latestVideoAssets` (which only ever showed the latest single generation
  // set) as the studio "what just came out" list. Sorted newest-first.
  const recentImageAssets = useMemo(
    () =>
      dropUpscaledVariants(
        assets.filter((asset) => asset.type === "image" && (!activeProject?.id || asset.projectId === activeProject.id)),
      )
        .sort(sortNewest)
        .slice(0, 20),
    [assets, activeProject?.id],
  );
  const recentVideoAssets = useMemo(
    () =>
      assets
        .filter((asset) => asset.type === "video" && (!activeProject?.id || asset.projectId === activeProject.id))
        .slice()
        .sort(sortNewest)
        .slice(0, 20),
    [assets, activeProject?.id],
  );
  const imageLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.image, jobs, activeProject?.id, isImageGenerationJob),
    [activeProject?.id, jobs, localGenerationJobIds.image],
  );
  const videoLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.video, jobs, activeProject?.id, isVideoGenerationJob),
    [activeProject?.id, jobs, localGenerationJobIds.video],
  );
  const documentLocalJobs = useMemo(
    () => buildLocalJobStack(localGenerationJobIds.document, jobs, activeProject?.id, isInterleaveJob),
    [activeProject?.id, jobs, localGenerationJobIds.document],
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
  }, [jobs, queueSummary]);
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
  // O(1) lookup by worker.id so every WorkerProgressCard consumer reads live
  // worker state without rebuilding the map per screen (sc-2082).
  const workersById = useMemo(() => buildWorkersById(workers), [workers]);
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
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.setAttribute("data-accent", accent);
    try {
      window.localStorage.setItem("sceneworks-accent", accent);
    } catch {
      // ignore (private mode etc.)
    }
  }, [accent]);

  // Seed the theme from the server on launch (the durable copy; localStorage is
  // only an instant-paint cache). Each toggle persists itself via changeTheme,
  // so there's no save effect to race with this read.
  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/v1/ui-preferences", "")
      .then((prefs) => {
        if (cancelled) {
          return;
        }
        if (prefs?.theme === "dark" || prefs?.theme === "light") {
          setTheme(prefs.theme);
        }
        if (isAccentId(prefs?.accent)) {
          setAccent(prefs.accent);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    apiFetch("/api/v1/health", "")
      .then(setHealth)
      .catch((err) => setError(err.message));

    apiFetch("/api/v1/access", "")
      .then(setAccess)
      .catch((err) => setError(err.message))
      // Either way the auth state is now as resolved as it'll get; release the gate so
      // an authenticated client (or one not requiring auth) can load its data.
      .finally(() => setAccessResolved(true));
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
    refreshDataRef.current?.();
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
    refreshAssetsRef.current?.(activeProject.id, { signal });
    refreshCharactersRef.current?.(activeProject.id, { signal });
    refreshLorasRef.current?.(activeProject.id, { signal });
    refreshPresetsRef.current?.(activeProject.id, { signal });
    refreshTrainingDatasetsRef.current?.(activeProject.id, { signal });
    refreshPersonTracksRef.current?.(activeProject.id, { signal });
    refreshTimelinesRef.current?.(activeProject.id, { signal });
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
      const job = parseSseJson(event, "job");
      if (!job) {
        return;
      }
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
          refreshAssetsRef.current?.(job.projectId);
        }
      }
      if (job.status === "completed" && hasGeneratedAssets) {
        enqueueTimelineGenerationApply(job);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_track") {
        refreshPersonTracksRef.current?.(job.projectId);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_detect") {
        refreshAssetsRef.current?.(job.projectId);
      }
      if (job.status === "completed" && job.type === "model_download") {
        refreshDataRef.current?.();
      }
      // A completed built-in LoRA download (sc-5944) flips the catalog entry to
      // installed; refresh models+loras so the Models row and any Studio gate update.
      if (job.status === "completed" && job.type === "lora_download") {
        refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_import") {
        dismissNoticeKind("lora-import");
        refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_train" && job.payload?.dryRun === false) {
        if (job.result?.loraRegistered === false) {
          pushNotice("lora-train", `lora training: ${job.result?.loraRegistrationError ?? "Completed training but could not register the LoRA."}`);
        } else {
          dismissNoticeKind("lora-train");
          refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
        }
      }
      if (job.status === "failed" && !hasVisibleLocalFailure(job)) {
        pushNotice(noticeKindForJob(job), failedJobNotice(job));
      }
    }

    function handleWorkerUpdated(event) {
      const worker = parseSseJson(event, "worker");
      if (!worker) {
        return;
      }
      setWorkers((items) => [worker, ...items.filter((item) => item.id !== worker.id)].sort(sortWorkers));
    }

    function handleQueueUpdated(event) {
      const summary = parseSseJson(event, "queue");
      if (!summary) {
        return;
      }
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
    // Mac UI gating (sc-3486): optional + non-fatal — a fetch failure leaves gating inert.
    fetchInitial("Mac capabilities", "/api/v1/capabilities/mac", DEFAULT_MAC_CAPABILITIES, true)
      .then((result) => setMacCapabilities(result.value ?? DEFAULT_MAC_CAPABILITIES))
      .catch(() => {});
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

  refreshDataRef.current = refreshData;
  refreshAssetsRef.current = refreshAssets;
  refreshCharactersRef.current = refreshCharacters;
  refreshLorasRef.current = refreshLoras;
  refreshPresetsRef.current = refreshPresets;
  refreshTrainingDatasetsRef.current = refreshTrainingDatasets;
  refreshPersonTracksRef.current = refreshPersonTracks;
  refreshTimelinesRef.current = refreshTimelines;
  refreshDataWithLoraOverlayRef.current = refreshDataWithLoraOverlay;


  // Remote-browser login (epic 4484 story 7): the password IS the API access token.
  // Verify the typed draft against the public /api/v1/auth/verify endpoint BEFORE
  // promoting it to the live `token`, so a wrong password keeps the gate up with an
  // inline error (instead of saving a bad token and silently failing every subsequent
  // request). A correct password is stored to localStorage and unlocks the app; it
  // persists across reloads. Promoting the token here flips `authenticated`, and the
  // [authenticated, token] effects perform the initial data load and SSE connect
  // exactly once — no explicit refreshData() call, or it would double-fetch (sc-8808).
  async function saveToken(event) {
    event.preventDefault();
    const candidate = passwordDraft.trim();
    if (!candidate) {
      setAuthError("Enter the password.");
      return;
    }
    try {
      const result = await apiFetch("/api/v1/auth/verify", candidate, { method: "POST" });
      if (!result?.ok) {
        setAuthError("Incorrect password. Try again.");
        return;
      }
    } catch {
      setAuthError("Couldn't reach the host to verify the password.");
      return;
    }
    window.localStorage.setItem("sceneworks-token", candidate);
    setToken(candidate);
    setPasswordDraft("");
    setAuthError("");
    setError("");
  }

  // Clear the stored password and re-show the login gate ("lock"/forget affordance,
  // epic 4484 story 7). Setting the token state to "" re-renders the gate, which
  // keys off the token state (sc-8808).
  function lockRemote() {
    window.localStorage.removeItem("sceneworks-token");
    setToken("");
    setPasswordDraft("");
    setAuthError("");
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

  const createPlaceholderJob = useCallback(
    async (event) => {
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
    },
    [token, activeProject, requestedGpu, jobPrompt, activeView],
  );

  const createImageJob = useCallback(
    async (payload) => {
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
    },
    [token, activeProject, requestedGpu],
  );

  // Standalone video upscale (epic 4811 / sc-4816): the net-new `video_upscale` job runs
  // on the generic /api/v1/jobs endpoint (like image_upscale), not the generation video
  // endpoint. `payload` carries { sourceAssetId, factor, engine, softness, model, displayName }.
  const createVideoUpscaleJob = useCallback(
    async (payload) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await apiFetch("/api/v1/jobs", token, {
          method: "POST",
          body: JSON.stringify({
            type: "video_upscale",
            projectId: activeProject.id,
            projectName: activeProject.name,
            requestedGpu,
            payload: { ...payload, projectId: activeProject.id },
          }),
        });
        setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, requestedGpu],
  );

  // Refine a prompt via the prompt_refine worker job: POST creates the job, then
  // poll until it reaches a terminal state and return the rewritten prompt. Project-
  // independent (no activeProject gate); throws on failure so the studio can surface
  // the message inline without clobbering the original prompt.
  const refinePrompt = useCallback(
    async ({ prompt, modelId, workflow, guide, signal }) => {
      const created = await apiFetch("/api/v1/prompts/refine", token, {
        method: "POST",
        signal,
        body: JSON.stringify({ prompt, modelId, workflow, guide }),
      });
      const jobId = created?.id;
      if (!jobId) {
        throw new Error("Could not start prompt refinement.");
      }
      const deadline = Date.now() + 120000;
      while (Date.now() < deadline) {
        await abortableDelay(1000, signal);
        const job = await apiFetch(`/api/v1/jobs/${jobId}`, token, { signal });
        if (job.status === "completed") {
          const refined = job.result?.refinedPrompt;
          if (!refined) {
            throw new Error("Refinement returned an empty prompt.");
          }
          return refined;
        }
        if (job.status === "failed" || job.status === "canceled" || job.status === "interrupted") {
          throw new Error(job.message || job.error || "Prompt refinement failed.");
        }
      }
      throw new Error("Prompt refinement timed out. Is the refinement runtime running?");
    },
    [token],
  );

  // Magic-prompt expansion (epic 4725, sc-5997): same `prompts/refine` endpoint + native
  // utility model, but `task: "magic_prompt"` swaps in Ideogram's caption system prompt and
  // returns a JSON caption string (the caller parses + validates it). Reuses the refine job's
  // poll-to-completion contract; captions can take longer, so the deadline is generous.
  const magicPrompt = useCallback(
    async ({ prompt, modelId, aspectRatio, guide, signal }) => {
      const created = await apiFetch("/api/v1/prompts/refine", token, {
        method: "POST",
        signal,
        body: JSON.stringify({ prompt, modelId, task: "magic_prompt", aspectRatio, guide }),
      });
      const jobId = created?.id;
      if (!jobId) {
        throw new Error("Could not start magic-prompt.");
      }
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        await abortableDelay(1000, signal);
        const job = await apiFetch(`/api/v1/jobs/${jobId}`, token, { signal });
        if (job.status === "completed") {
          const caption = job.result?.refinedPrompt;
          if (!caption) {
            throw new Error("Magic-prompt returned an empty caption.");
          }
          return caption;
        }
        if (job.status === "failed" || job.status === "canceled" || job.status === "interrupted") {
          throw new Error(job.message || job.error || "Magic-prompt failed.");
        }
      }
      throw new Error("Magic-prompt timed out. Is the refinement runtime running?");
    },
    [token],
  );

  // Reference-image → JSON caption (epic 8102, sc-8108): same `prompts/refine` endpoint + poll-to-
  // completion contract as magic-prompt, but `task: "image_caption"` drives the worker's `core_llm`
  // VISION path (sc-8105). The reference image is supplied as a project `sourceAssetId` (+ `projectId`),
  // which the API resolves to a confined on-disk `imagePath`; the vision model is named by its HF repo
  // string in `model` (the worker resolves it by repo, like the refiner). The caller parses + validates
  // the returned JSON with `parseVisionCaption` (aspect_ratio stripped, bboxes KEPT). C1: the image is
  // consumed only to produce the caption — it is NEVER passed to generation as img2img conditioning.
  const imageCaption = useCallback(
    async ({ sourceAssetId, projectId, model, signal }) => {
      const created = await apiFetch("/api/v1/prompts/refine", token, {
        method: "POST",
        signal,
        body: JSON.stringify({ task: "image_caption", sourceAssetId, projectId, model }),
      });
      const jobId = created?.id;
      if (!jobId) {
        throw new Error("Could not start image captioning.");
      }
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        await abortableDelay(1000, signal);
        const job = await apiFetch(`/api/v1/jobs/${jobId}`, token, { signal });
        if (job.status === "completed") {
          const caption = job.result?.refinedPrompt;
          if (!caption) {
            throw new Error("Image captioning returned an empty caption.");
          }
          return caption;
        }
        if (job.status === "failed" || job.status === "canceled" || job.status === "interrupted") {
          throw new Error(job.message || job.error || "Image captioning failed.");
        }
      }
      throw new Error("Image captioning timed out. Is the captioning runtime running?");
    },
    [token],
  );

  // Reference-image → plain-text description (epic 8203, sc-8208): the sibling of `imageCaption` for
  // NON-structured text-to-image models. Same `prompts/refine` endpoint + poll contract, but
  // `task: "image_describe"` drives the worker's prose/tags vision path (sc-8204/8205). `captionStyle`
  // (from the catalog, default prose) selects natural-language prose vs booru tags. Resolves to the raw
  // text the caller drops into the prompt box. C1: the image is consumed only to produce the prompt — it
  // is NEVER passed to generation as img2img conditioning.
  const imageDescribe = useCallback(
    async ({ sourceAssetId, projectId, model, captionStyle, signal }) => {
      const created = await apiFetch("/api/v1/prompts/refine", token, {
        method: "POST",
        signal,
        body: JSON.stringify({
          task: "image_describe",
          sourceAssetId,
          projectId,
          model,
          captionStyle,
        }),
      });
      const jobId = created?.id;
      if (!jobId) {
        throw new Error("Could not start image description.");
      }
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        await abortableDelay(1000, signal);
        const job = await apiFetch(`/api/v1/jobs/${jobId}`, token, { signal });
        if (job.status === "completed") {
          const description = job.result?.refinedPrompt;
          if (!description) {
            throw new Error("Image description returned empty text.");
          }
          return description;
        }
        if (job.status === "failed" || job.status === "canceled" || job.status === "interrupted") {
          throw new Error(job.message || job.error || "Image description failed.");
        }
      }
      throw new Error("Image description timed out. Is the captioning runtime running?");
    },
    [token],
  );

  // On-demand "compare image to another" likeness (epic 4406, sc-4415): score a CANDIDATE asset
  // against a SOURCE identity reference asset through the shared SCRFD+ArcFace scorer in the worker.
  // Same poll-to-completion contract as the refine/describe runners, but `/face-likeness/compare`
  // enqueues the GPU-routed `face_likeness_compare` job and returns the full result object
  // (`{ score, detected, method, sourceRef, reason? }`) so the caller can render the band + N/A framing
  // via `classifyLikeness` / `LikenessBadge`. Non-fatal end to end: a no-face / non-frontal candidate
  // is an honest detected:false result (NOT an error); only a hard failure throws.
  const compareFaceLikeness = useCallback(
    async ({ sourceAssetId, candidateAssetId, projectId, signal }) => {
      const created = await apiFetch("/api/v1/face-likeness/compare", token, {
        method: "POST",
        signal,
        body: JSON.stringify({ sourceAssetId, candidateAssetId, projectId }),
      });
      const jobId = created?.id;
      if (!jobId) {
        throw new Error("Could not start the likeness compare.");
      }
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        await abortableDelay(1000, signal);
        const job = await apiFetch(`/api/v1/jobs/${jobId}`, token, { signal });
        if (job.status === "completed") {
          // A completed compare always carries a result block (a detected:false N/A is a valid,
          // non-error outcome). Surface the whole block so the UI can band/N-A it.
          if (!job.result) {
            throw new Error("Likeness compare returned no result.");
          }
          return job.result;
        }
        if (job.status === "failed" || job.status === "canceled" || job.status === "interrupted") {
          throw new Error(job.message || job.error || "Likeness compare failed.");
        }
      }
      throw new Error("Likeness compare timed out. Is the worker running?");
    },
    [token],
  );

  const createVqaJob = useCallback(
    async (asset, question, maxNewTokens) => {
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
    },
    [token, activeProject, requestedGpu],
  );

  const createInterleaveJob = useCallback(
    async (payload) => {
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
    },
    [token, activeProject, requestedGpu],
  );

  const rememberLocalGenerationJob = useCallback((kind, job) => {
    if (!job?.id) {
      return;
    }
    setLocalGenerationJobIds((current) => ({
      ...current,
      // Remember every submitted run (newest first, capped) so running and queued
      // runs stack in the studio instead of the latest run evicting the previous one.
      [kind]: [job.id, ...current[kind].filter((id) => id !== job.id)].slice(0, localJobStackLimit),
    }));
  }, []);

  function hasVisibleLocalFailure(job) {
    const active = activeViewRef.current;
    const localIds = localGenerationJobIdsRef.current;
    if (active === "Image" && localIds.image.includes(job.id)) {
      return true;
    }
    if (active === "Video" && localIds.video.includes(job.id)) {
      return true;
    }
    if (active === "Document" && localIds.document.includes(job.id)) {
      return true;
    }
    return active === "Models" && job.type === "model_download";
  }

  const sendAssetToImage = useCallback((asset, mode = null) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Image", assetId: asset.id, mode });
    }
    setActiveView("Image");
  }, []);

  // Open Image Studio in edit mode with this image as the source, preselecting the
  // family-matched edit model when possible. sc-8730: this is the model-based path,
  // no longer wired to the preview Edit button — it stays on the context so S4
  // (sc-8729) can offer it as an "Edit in > Image Studio" context-menu item.
  const sendAssetToImageEdit = useCallback((asset) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      assetId: asset.id,
      mode: "edit_image",
      model: editModelForAsset(asset, imageModels),
    });
    setActiveView("Image");
  }, [imageModels]);

  // sc-8730: "Edit" from the fullscreen preview now opens the Image Editor canvas
  // (crop/upscale/refine) with this asset loaded via the editor's openAsset. This is
  // the model-free path; the model-based Image Studio edit_image path lives in
  // sendAssetToImageEdit above (kept for the S4 "Edit in > Image Studio" menu item).
  const sendAssetToImageEditor = useCallback((asset) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    setEditorLaunch({ id: crypto.randomUUID(), assetId: asset.id });
    setActiveView("ImageEditor");
  }, []);

  // The editor consumes editorLaunch and calls this to drop it, so navigating away
  // and back into the Image Editor without a fresh launch doesn't re-open a stale asset.
  const clearEditorLaunch = useCallback(() => setEditorLaunch(null), []);

  function recipeForAsset(asset) {
    return asset?.generationSet?.recipe ?? asset?.recipe ?? null;
  }

  function sendAssetRecipeToImage(asset) {
    const recipe = recipeForAsset(asset);
    if (!asset || !recipe) {
      return;
    }
    setSelectedAssetId(asset.id);
    closePreview();
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      assetId: asset.id,
      sourceAssetId: asset.lineage?.sourceAssetId ?? null,
      recipe,
    });
    setActiveView("Image");
  }

  const sendAssetToVideo = useCallback((asset, mode = null) => {
    if (!asset) {
      return;
    }
    setSelectedAssetId(asset.id);
    if (mode) {
      setStudioLaunch({ id: crypto.randomUUID(), view: "Video", assetId: asset.id, mode });
    }
    setActiveView("Video");
  }, []);

  const sendCharacterToImage = useCallback((character, lookId = null, referenceAssetId = null) => {
    if (!character) {
      return;
    }
    setStudioLaunch({
      id: crypto.randomUUID(),
      view: "Image",
      characterId: character.id,
      lookId,
      referenceAssetId,
      mode: "character_image",
    });
    setActiveView("Image");
  }, []);

  const sendCharacterToVideo = useCallback((character, lookId = null) => {
    if (!character) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "Video", characterId: character.id, lookId, mode: "text_to_video" });
    setActiveView("Video");
  }, []);

  // sc-2022: open a specific dataset in the Dataset editor (Character Studio's
  // "Open" action on an associated dataset). The editor consumes studioLaunch.
  const openDatasetInLibrary = useCallback((datasetId) => {
    if (!datasetId) {
      return;
    }
    setStudioLaunch({ id: crypto.randomUUID(), view: "LibraryDataSets", datasetId });
    setActiveView("LibraryDataSets");
  }, []);

  const updateAssetStatus = useCallback(
    async (asset, changes) => {
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
    },
    [token],
  );

  const updateAssetTags = useCallback(
    async (asset, tags) => {
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
    },
    [token],
  );

  // Promote a character asset into the Main Asset Library (sc-8341): a true move —
  // the backend flips origin + detaches the character, so refresh characters too.
  const moveAssetToLibrary = useCallback(
    async (asset) => {
      try {
        const updated = await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/move-to-library`, token, {
          method: "POST",
        });
        setAssets((items) => items.map((item) => (item.id === updated.id ? updated : item)));
        refreshCharactersRef.current?.(asset.projectId);
        setError("");
        return updated;
      } catch (err) {
        setError(err.message);
        throw err;
      }
    },
    [token],
  );

  const deleteAsset = useCallback(
    async (asset) => {
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
    },
    [token],
  );

  const purgeAsset = useCallback(
    async (asset) => {
      try {
        await apiFetch(`/api/v1/projects/${asset.projectId}/assets/${asset.id}/purge`, token, { method: "DELETE" });
        setAssets((items) => items.filter((item) => item.id !== asset.id));
        setSelectedAssetId((current) => (current === asset.id ? null : current));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

  const importAsset = useCallback(
    async (file, options = {}) => {
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
      // Optional lineage for derived imports (Image Editor Save, sc-2434): link the
      // new asset to the source it was opened from + record the edit-chain provenance.
      if (options.sourceAssetId) body.append("sourceAssetId", options.sourceAssetId);
      if (options.provenance) body.append("provenance", JSON.stringify(options.provenance));
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
    },
    [token, activeProject],
  );

  const jobAction = useCallback(
    async (job, action, options = {}) => {
      try {
        const path = action === "duplicate" ? `/api/v1/jobs/${job.id}/duplicate` : `/api/v1/jobs/${job.id}/${action}`;
        const body =
          action === "duplicate"
            ? { payloadChanges: { duplicatedAt: new Date().toISOString() } }
            : (options.body ?? {});
        const updatedJob = await apiFetch(path, token, { method: "POST", body: JSON.stringify(body) });
        setJobs((items) => [updatedJob, ...items.filter((item) => item.id !== updatedJob.id)].sort(sortNewest));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token],
  );

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
  // sc-4194: memoized so the provider value only changes identity when one of its
  // entries actually changes, instead of being a fresh ~120-key literal on every App
  // render (SSE job/worker/queue ticks re-render App continuously). The actions above
  // and the data-hook actions are useCallback-stable, so this holds across renders
  // that don't change data. NOTE: this dependency array must mirror the object below —
  // every value referenced here is a dependency.
  const appContextValue = useMemo(() => ({
    activeProject,
    mediaAssets,
    setPreviewAsset: openPreview,
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
    moveAssetToLibrary,
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
    workersById,
    // Generation studios (sc-1651 Phase B batch 3)
    createVideoJob,
    createVideoUpscaleJob,
    createImageJob,
    refinePrompt,
    magicPrompt,
    imageCaption,
    imageDescribe,
    compareFaceLikeness,
    latestVideoAssets,
    recentImageAssets,
    recentVideoAssets,
    videoLocalJobs,
    imageLocalJobs,
    documentLocalJobs,
    studioLaunch,
    // sc-8730: Image Editor launch channel + the two Edit paths. sendAssetToImageEditor
    // routes to the editor canvas (FullscreenPreview Edit button); sendAssetToImageEdit
    // routes to Image Studio edit_image (exposed for the S4 sc-8729 context menu).
    editorLaunch,
    clearEditorLaunch,
    sendAssetToImageEditor,
    sendAssetToImageEdit,
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
    // Mac UI gating (sc-3486)
    macCapabilities,
    loras,
    deleteLora,
    deleteModel,
    createModelDownloadJob,
    createLoraDownloadJob,
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
    // Auth (sc-4168): pairing token for screens that call apiFetch directly
    // (Image Editor, Logs, Pose Library, useUserPoseLoader). Empty string when
    // the deployment doesn't require auth.
    token,
    // Training (sc-1651 Phase B batch 7)
    authenticated,
    trainingDatasets,
    trainingDatasetsProjectId,
    trainingDatasetsError,
    loadingTrainingDatasets,
    refreshTrainingDatasets,
    loadTrainingDataset,
    loadTrainingDatasetReadiness,
    setTrainingDatasetItemQualityAck,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingDatasetUpscaleJob,
    createTrainingDatasetAnalysisJob,
    createTrainingDatasetFaceAnalysisJob,
    smartCropTrainingDataset,
    stripExifTrainingDataset,
    createTrainingJob,
    trainingPresets,
    trainingPresetsError,
    trainingTargets,
    trainingTargetsError,
    // Navigation
    setActiveView,
    registerLeaveGuard,
    // Characters
    characters,
    createCharacter,
    updateCharacter,
    archiveCharacter,
    unarchiveCharacter,
    listArchivedCharacters,
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
    openDatasetInLibrary,
  }), [
    activeProject, mediaAssets, openPreview, sendAssetToImage, sendAssetToVideo,
    activeTimeline, timelines, selectedTimelineId, setSelectedTimelineId, setActiveTimeline,
    createTimeline, saveTimeline, exportTimeline, extractTimelineFrame, queueTimelineVideoJob,
    assets, selectedAsset, setSelectedAssetId, deleteAsset, purgeAsset, moveAssetToLibrary, importAsset,
    updateAssetStatus, updateAssetTags, latestImageAssets,
    jobs, jobAction, createVqaJob, createInterleaveJob, createPlaceholderJob, filteredJobs,
    jobPrompt, setJobPrompt, projectFilter, setProjectFilter, projects, visibleWorkers, workersById,
    createVideoJob, createVideoUpscaleJob, createImageJob, refinePrompt, magicPrompt, imageCaption, imageDescribe, compareFaceLikeness, latestVideoAssets, recentImageAssets,
    recentVideoAssets, videoLocalJobs, imageLocalJobs, documentLocalJobs, studioLaunch,
    editorLaunch, clearEditorLaunch, sendAssetToImageEditor, sendAssetToImageEdit,
    rememberLocalGenerationJob, personTracks, personReadiness, createPersonDetectionJob,
    createPersonTrackJob, saveTrackCorrections, imageModels, videoModels, models, macCapabilities,
    loras, deleteLora, deleteModel, createModelDownloadJob, createLoraDownloadJob, createModelConvertJob,
    createLoraImportJob, createModelImportJob, gpuOptions, requestedGpu, setRequestedGpu,
    presets, createPreset, updatePreset, deletePreset, duplicatePreset, token, authenticated,
    trainingDatasets, trainingDatasetsProjectId, trainingDatasetsError, loadingTrainingDatasets,
    refreshTrainingDatasets, loadTrainingDataset, loadTrainingDatasetReadiness, setTrainingDatasetItemQualityAck, createTrainingDataset, uploadTrainingDatasetItem,
    updateTrainingDataset, batchRenameTrainingDataset, writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob, createTrainingDatasetUpscaleJob, createTrainingDatasetAnalysisJob, createTrainingDatasetFaceAnalysisJob, smartCropTrainingDataset, stripExifTrainingDataset, createTrainingJob, trainingPresets, trainingPresetsError,
    trainingTargets, trainingTargetsError, setActiveView, registerLeaveGuard, characters,
    createCharacter, updateCharacter, archiveCharacter, unarchiveCharacter, listArchivedCharacters,
    addCharacterReference, updateCharacterReference,
    removeCharacterReference, createCharacterLook, updateCharacterLook, deleteCharacterLook,
    attachCharacterLora, updateCharacterLora, detachCharacterLora, createCharacterTestJob,
    sendCharacterToImage, sendCharacterToVideo, openDatasetInLibrary,
  ]);

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
                    onClick={() => navTo(item.id)}
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
          <div className="accent-picker" role="group" aria-label="Accent color">
            {ACCENTS.map((option) => (
              <button
                aria-label={option.name}
                aria-pressed={accent === option.id}
                className={accent === option.id ? "accent-swatch active" : "accent-swatch"}
                key={option.id}
                onClick={() => changeAccent(option.id)}
                style={{ "--sw": option.swatch }}
                title={option.name}
                type="button"
              />
            ))}
          </div>
          <button
            className="icon-btn"
            onClick={() => changeTheme(theme === "light" ? "dark" : "light")}
            title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            type="button"
          >
            {theme === "light" ? <Icon.Moon /> : <Icon.Sun />}
          </button>
          {/* Lock/forget the saved password (epic 4484 story 7) — only meaningful in a
              remote browser where a password was entered to unlock the host. */}
          {access.authRequired && !isDesktopShell && token ? (
            <button
              className="icon-btn"
              onClick={lockRemote}
              title="Lock — forget the saved password"
              type="button"
            >
              Lock
            </button>
          ) : null}
        </header>

        {notices.map((notice) => (
          <p className="notice error" key={notice.kind}>{notice.message}</p>
        ))}

        {/* Gate visibility keys off the token STATE (not a render-time localStorage
            read), and the input edits a local draft — never the live token — so
            typing can't flip `authenticated` or fire API/SSE traffic (sc-8808). */}
        {access.authRequired && !isDesktopShell && !token ? (
          <section className="auth-band">
            <form onSubmit={saveToken}>
              <label htmlFor="token">Password</label>
              <div className="form-row">
                <input
                  id="token"
                  onChange={(event) => setPasswordDraft(event.target.value)}
                  placeholder="Enter the access password"
                  type="password"
                  value={passwordDraft}
                />
                <button type="submit">Unlock</button>
              </div>
              {authError ? <p className="notice error">{authError}</p> : null}
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

        {activeView === "Poses" ? (
          <PoseLibraryScreen />
        ) : null}

        {activeView === "Keypoints" ? (
          <KeyPointLibraryScreen />
        ) : null}

        {activeView === "Image" ? (
          <ImageStudio key={activeProject?.id ?? "default"} />
        ) : null}

        {activeView === "Video" ? (
          <VideoStudio key={activeProject?.id ?? "default"} />
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

        {activeView === "ImageEditor" ? (
          <React.Suspense fallback={<section className="main-surface">Loading editor…</section>}>
            <ImageEditor key={activeProject?.id ?? "default"} />
          </React.Suspense>
        ) : null}

        {activeView === "Characters" ? (
          <CharacterStudio key={activeProject?.id ?? "default"} />
        ) : null}
        {activeView === "Settings" ? <SettingsScreen /> : null}
        {activeView === "Logs" ? <LogsScreen /> : null}
        {activeView === "Licenses" ? <LicensesScreen /> : null}
          </>
        )}
      </section>

      {previewedAsset ? (
        <FullscreenPreview
          asset={previewedAsset}
          deleteAsset={async (asset) => {
            // Stay in the preview and advance to the neighbour in the direction
            // the user was scrolling (falling back to the other side, then to
            // closing once nothing is left).
            const { previous, next } = previewNavigation;
            const target =
              previewDirectionRef.current === "previous" ? previous ?? next : next ?? previous;
            await deleteAsset(asset);
            // Advance within the launch collection; close (and drop the scope)
            // once it is exhausted.
            if (target) {
              setPreviewAsset(target);
            } else {
              closePreview();
            }
          }}
          nextAsset={previewNavigation.next}
          onClose={closePreview}
          onEditImage={sendAssetToImageEditor}
          onEditInStudio={sendAssetToImageEdit}
          onPreviewAsset={(asset, direction) => {
            if (direction) {
              previewDirectionRef.current = direction;
            }
            setPreviewAsset(asset);
          }}
          onUseRecipe={sendAssetRecipeToImage}
          previousAsset={previewNavigation.previous}
          purgeAsset={async (asset) => {
            await purgeAsset(asset);
            closePreview();
          }}
          updateAssetStatus={updateAssetStatus}
        />
      ) : null}
    </main>
    </AppContext.Provider>
  );
}
