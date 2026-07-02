import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";

// Owns the project's person-track state plus detection/track job creation and manual
// track corrections. Extracted from App.jsx (sc-1651). personTracks is project-scoped
// (loaded by the project-load effect, not the bulk refreshData), so the hook just
// takes the shared concerns it needs; detection/track jobs surface live via the SSE
// job stream (the create endpoints publish job.updated), so no post-create refetch.
export function usePersonTracks({ token, activeProject, activeProjectRef, setError, requestedGpu, setActiveView }) {
  const [personTracks, setPersonTracks] = useState([]);

  // sc-4194: every action is wrapped in useCallback so its identity is stable
  // across the SSE-driven re-renders of App, letting appContextValue memoize.
  const refreshPersonTracks = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId) {
        return;
      }
      try {
        const items = await apiFetch(`/api/v1/projects/${projectId}/person-tracks`, token, { signal });
        // sc-8858: an SSE-triggered refresh for the just-active project can resolve
        // after the user switches away; committing then would clobber the new
        // project's tracks with the old one's. Drop the stale response — mirrors
        // refreshTimelines' guard (useTimelines.js).
        if (activeProjectRef?.current?.id && activeProjectRef.current.id !== projectId) {
          return;
        }
        setPersonTracks(items);
        setError("");
      } catch (err) {
        if (isAbortError(err)) return;
        setError(err.message);
      }
    },
    [token, activeProject, activeProjectRef, setError],
  );

  const createPersonDetectionJob = useCallback(
    async (payload, options = {}) => {
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
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, requestedGpu, setActiveView],
  );

  const createPersonTrackJob = useCallback(
    async (payload, options = {}) => {
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
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, requestedGpu, setActiveView],
  );

  const saveTrackCorrections = useCallback(
    async (trackId, corrections) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const track = await apiFetch(
          `/api/v1/projects/${activeProject.id}/person-tracks/${trackId}/corrections`,
          token,
          {
            method: "POST",
            body: JSON.stringify({ corrections }),
          },
        );
        setPersonTracks((items) => items.map((item) => (item.id === track.id ? track : item)));
        setError("");
        return track;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError],
  );

  return {
    personTracks,
    setPersonTracks,
    refreshPersonTracks,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections,
  };
}
