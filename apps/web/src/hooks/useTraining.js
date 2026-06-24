import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { sortNewest } from "../sorters.js";

// Owns the project's training-dataset state plus dataset CRUD, caption sidecar/job,
// and training-job creation. Extracted from App.jsx (sc-1651). The training target/
// preset catalogs stay in App (bulk-loaded by refreshData) — only the project-scoped
// dataset workflow lives here. Shared concerns (token, activeProject, error, jobs) are
// passed in; caption/training jobs push onto the shared jobs list via setJobs.
//
// sc-4194: actions are wrapped in useCallback so their identity is stable across App's
// SSE-driven re-renders, letting appContextValue memoize.
export function useTraining({ token, activeProject, setError, setJobs }) {
  const [trainingDatasets, setTrainingDatasets] = useState([]);
  const [trainingDatasetsProjectId, setTrainingDatasetsProjectId] = useState(null);
  const [loadingTrainingDatasets, setLoadingTrainingDatasets] = useState(false);
  const [trainingDatasetsError, setTrainingDatasetsError] = useState("");

  const refreshTrainingDatasets = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId) {
        setTrainingDatasets([]);
        setTrainingDatasetsProjectId(null);
        setTrainingDatasetsError("");
        return [];
      }
      setLoadingTrainingDatasets(true);
      try {
        const items = await apiFetch(`/api/v1/projects/${projectId}/training/datasets`, token, { signal });
        setTrainingDatasets(items);
        setTrainingDatasetsProjectId(projectId);
        setTrainingDatasetsError("");
        return items;
      } catch (err) {
        if (isAbortError(err)) return [];
        setTrainingDatasets([]);
        setTrainingDatasetsProjectId(projectId);
        setTrainingDatasetsError(err.message);
        return [];
      } finally {
        // A superseded load must not clear the loading flag the new load just set.
        if (!signal?.aborted) {
          setLoadingTrainingDatasets(false);
        }
      }
    },
    [token, activeProject],
  );

  const loadTrainingDataset = useCallback(
    async (datasetId, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        return null;
      }
      return apiFetch(`/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}`, token);
    },
    [token, activeProject],
  );

  // Dataset Doctor readiness report (sc-6533/6534). A read over the SAVED dataset
  // — pixel metrics (blur/dup/exposure) need the files on disk — so callers fetch
  // it keyed on the dataset version, not the live unsaved selection. `query` is the
  // pre-built query string (target resolution / kind / min) from readinessQueryParams;
  // `signal` lets a stale fetch be aborted when the target or dataset changes.
  const loadTrainingDatasetReadiness = useCallback(
    async (datasetId, query = "", projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId || !datasetId) {
        return null;
      }
      const suffix = query ? `?${query}` : "";
      return apiFetch(
        `/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}/readiness${suffix}`,
        token,
        { signal },
      );
    },
    [token, activeProject],
  );

  // Persist (or clear) a per-image quality override (sc-6534). `checks` is the full set of dismissed
  // findings for the item (the endpoint replaces the stored set); an empty array clears it. A
  // metadata write — it does not bump the dataset version — so callers refetch readiness, not the
  // dataset, afterward.
  const setTrainingDatasetItemQualityAck = useCallback(
    async (datasetId, itemId, checks, optionsOrProjectId = {}, explicitProjectId = activeProject?.id) => {
      const options = typeof optionsOrProjectId === "string" ? {} : (optionsOrProjectId ?? {});
      const projectId = typeof optionsOrProjectId === "string" ? optionsOrProjectId : explicitProjectId;
      if (!projectId || !datasetId || !itemId) {
        throw new Error("Select a training dataset item first.");
      }
      const body = {
        checks,
        ...(options.expectedContentHash ? { expectedContentHash: options.expectedContentHash } : {}),
        ...(options.expectedCaptionHash ? { expectedCaptionHash: options.expectedCaptionHash } : {}),
      };
      return apiFetch(
        `/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(itemId)}/quality-ack`,
        token,
        { method: "POST", body: JSON.stringify(body) },
      );
    },
    [token, activeProject],
  );

  const createTrainingDataset = useCallback(
    async (payload, projectId = activeProject?.id) => {
      if (!projectId) {
        throw new Error("Create or open a project first.");
      }
      const created = await apiFetch(`/api/v1/projects/${projectId}/training/datasets`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await refreshTrainingDatasets(projectId);
      return created;
    },
    [token, activeProject, refreshTrainingDatasets],
  );

  const uploadTrainingDatasetItem = useCallback(
    async (file, projectId = activeProject?.id) => {
      if (!projectId) {
        throw new Error("Create or open a project first.");
      }
      const form = new FormData();
      form.append("file", file);
      return apiFetch(`/api/v1/projects/${projectId}/training/uploads`, token, {
        method: "POST",
        body: form,
      });
    },
    [token, activeProject],
  );

  const updateTrainingDataset = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const updated = await apiFetch(`/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}`, token, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      await refreshTrainingDatasets(projectId);
      return updated;
    },
    [token, activeProject, refreshTrainingDatasets],
  );

  const batchRenameTrainingDataset = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const updated = await apiFetch(
        `/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}/batch-rename`,
        token,
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
      await refreshTrainingDatasets(projectId);
      return updated;
    },
    [token, activeProject, refreshTrainingDatasets],
  );

  const writeTrainingDatasetCaptionSidecars = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const result = await apiFetch(
        `/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}/caption-sidecars`,
        token,
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
      await refreshTrainingDatasets(projectId);
      return result;
    },
    [token, activeProject, refreshTrainingDatasets],
  );

  const createTrainingDatasetCaptionJob = useCallback(
    async (datasetId, payload, projectId = activeProject?.id) => {
      if (!projectId || !datasetId) {
        throw new Error("Select a training dataset first.");
      }
      const job = await apiFetch(
        `/api/v1/projects/${projectId}/training/datasets/${encodeURIComponent(datasetId)}/caption-jobs`,
        token,
        {
          method: "POST",
          body: JSON.stringify(payload),
        },
      );
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  const createTrainingJob = useCallback(
    async (request, projectId = activeProject?.id) => {
      if (!projectId) {
        throw new Error("Select a workspace before creating a training job.");
      }
      const job = await apiFetch(`/api/v1/projects/${projectId}/training/jobs`, token, {
        method: "POST",
        body: JSON.stringify(request),
      });
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      return job;
    },
    [token, activeProject, setJobs, setError],
  );

  return {
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
    createTrainingJob,
  };
}
