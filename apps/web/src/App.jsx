import React, { useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, eventUrl } from "./api.js";
import { StatusDot } from "./components/StatusDot.jsx";
import { FullscreenPreview } from "./components/assetPanels.jsx";
import { fallbackModels, navItems, terminalStatuses } from "./constants.js";
import { LibraryScreen } from "./screens/LibraryScreen.jsx";
import { ModelManagerScreen } from "./screens/ModelManagerScreen.jsx";
import { ImageStudio } from "./screens/ImageStudio.jsx";
import { VideoStudio } from "./screens/VideoStudio.jsx";
import { CharacterStudio } from "./screens/CharacterStudio.jsx";
import { EditorScreen } from "./screens/EditorScreen.jsx";
import { QueueScreen } from "./screens/QueueScreen.jsx";
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

export function App() {
  const [health, setHealth] = useState(null);
  const [access, setAccess] = useState({ authRequired: false });
  const [token, setToken] = useState(() => window.localStorage.getItem("sceneworks-token") ?? "");
  const [projects, setProjects] = useState([]);
  const [activeProject, setActiveProject] = useState(null);
  const [activeView, setActiveView] = useState("Library");
  const [projectName, setProjectName] = useState("");
  const [jobs, setJobs] = useState([]);
  const [workers, setWorkers] = useState([]);
  const [queueSummary, setQueueSummary] = useState(null);
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);
  const [recipePresets, setRecipePresets] = useState([]);
  const [assets, setAssets] = useState([]);
  const [characters, setCharacters] = useState([]);
  const [personTracks, setPersonTracks] = useState([]);
  const [timelines, setTimelines] = useState([]);
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
  const selectedTimelineIdRef = useRef(null);
  const timelineApplyQueueRef = useRef(Promise.resolve());

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
    selectedTimelineIdRef.current = selectedTimelineId;
  }, [selectedTimelineId]);

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
      setRecipePresets([]);
      setSelectedTimelineId(null);
      setActiveTimeline(null);
      return;
    }
    refreshAssets(activeProject.id);
    refreshCharacters(activeProject.id);
    refreshLoras(activeProject.id);
    refreshRecipePresets(activeProject.id);
    refreshPersonTracks(activeProject.id);
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

    let events = null;
    let reconnectTimer = null;
    let reconnectAttempt = 0;
    let closed = false;

    function handleJobUpdated(event) {
      const job = JSON.parse(event.data);
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      if (job.status === "completed" && (job.result?.generationSetId || job.result?.assetIds?.length)) {
        if (job.result?.generationSetId) {
          setLatestGenerationSetId(job.result.generationSetId);
        }
        enqueueTimelineGenerationApply(job);
        if (job.projectId) {
          refreshAssets(job.projectId);
        }
      }
      if (job.status === "completed" && job.projectId && job.type === "person_track") {
        refreshPersonTracks(job.projectId);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_detect") {
        refreshAssets(job.projectId);
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
    const [projectsResult, jobsResult, workersResult, modelsResult, lorasResult, recipePresetsResult] = await Promise.all([
      fetchInitial("Projects", "/api/v1/projects", []),
      fetchInitial("Jobs", "/api/v1/jobs", []),
      fetchInitial("Workers", "/api/v1/workers", []),
      fetchInitial("Models", "/api/v1/models", []),
      fetchInitial("LoRAs", "/api/v1/loras", []),
      fetchInitial("Recipe presets", "/api/v1/recipe-presets", [], true),
    ]);
    const projectItems = projectsResult.value;
    setProjects(projectItems);
    setActiveProject((current) => current ?? projectItems[0] ?? null);
    setJobs(jobsResult.value.sort(sortNewest));
    setWorkers(workersResult.value.sort(sortWorkers));
    setQueueSummary(null);
    setModels(modelsResult.value);
    setLoras(lorasResult.value);
    setRecipePresets(recipePresetsResult.value);
    setError([projectsResult, jobsResult, workersResult, modelsResult, lorasResult, recipePresetsResult].map((result) => result.error).filter(Boolean).join("; "));
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

  async function refreshRecipePresets(projectId = activeProject?.id) {
    try {
      const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
      const items = await apiFetch(`/api/v1/recipe-presets${query}`, token);
      setRecipePresets(items);
      setError("");
    } catch (err) {
      setError(err.message);
    }
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
    const { navigateToQueue = true } = options;
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

  async function importAsset(file) {
    if (!activeProject || !file) {
      setError("Create or open a project first.");
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
    } catch (err) {
      setError(err.message);
    }
  }

  async function createModelDownloadJob(model) {
    try {
      await apiFetch(`/api/v1/models/${model.id}/download`, token, {
        method: "POST",
        body: JSON.stringify({ requestedGpu: "auto" }),
      });
      setActiveView("Queue");
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
            <span>
              {visibleWorkers.length ? `${visibleWorkers.length} worker${visibleWorkers.length === 1 ? "" : "s"}` : "No workers"}
            </span>
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
            onPreview={setPreviewAsset}
            recipePresets={recipePresets}
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
            jobs={jobs}
            onPreview={setPreviewAsset}
            personTracks={personTracks}
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
          <ModelManagerScreen jobs={jobs} loras={loras} models={models} onDownloadModel={createModelDownloadJob} />
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
      </section>

      {previewAsset ? <FullscreenPreview asset={previewAsset} onClose={() => setPreviewAsset(null)} /> : null}
    </main>
  );
}
