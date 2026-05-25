import { useEffect, useRef, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { ensureItemVersionFields } from "../timeline.js";

// Owns the editor's timeline state (list, selection, the loaded timeline) plus every
// timeline mutation, frame extraction, and the SSE-driven "apply generated clip to the
// timeline" pipeline. Extracted from App.jsx (sc-1651) — the largest, most coupled
// slice. App keeps the SSE job.updated handler and calls the returned
// enqueueTimelineGenerationApply; the bulk reset/project-load effects use the returned
// setters/refreshTimelines. createVideoJob (App-owned) is injected for the timeline's
// generate-clip action. The two timeline-specific effects (selectedTimelineId ref sync,
// then the load-on-selection effect) live here, in that order, matching App's prior
// behavior.
export function useTimelines({
  token,
  activeProject,
  activeProjectRef,
  setError,
  requestedGpu,
  setActiveView,
  createVideoJob,
}) {
  const [timelines, setTimelines] = useState([]);
  const [timelinesProjectId, setTimelinesProjectId] = useState(null);
  const [selectedTimelineId, setSelectedTimelineId] = useState(null);
  const [activeTimeline, setActiveTimeline] = useState(null);
  const selectedTimelineIdRef = useRef(null);
  const timelineApplyQueueRef = useRef(Promise.resolve());

  useEffect(() => {
    selectedTimelineIdRef.current = selectedTimelineId;
  }, [selectedTimelineId]);

  useEffect(() => {
    if (!activeProject || !selectedTimelineId || timelinesProjectId !== activeProject.id) {
      return;
    }
    loadTimeline(activeProject.id, selectedTimelineId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject?.id, selectedTimelineId, timelinesProjectId]);

  async function refreshTimelines(projectId = activeProject?.id, { signal } = {}) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/timelines`, token, { signal });
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
      if (isAbortError(err)) return;
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
    } catch (err) {
      setError(err.message);
    }
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

  return {
    timelines,
    setTimelines,
    timelinesProjectId,
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
  };
}
