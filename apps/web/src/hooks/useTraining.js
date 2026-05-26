import { useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { sortNewest } from "../sorters.js";

// Owns the project's training-dataset state plus dataset CRUD, caption sidecar/job,
// and training-job creation. Extracted from App.jsx (sc-1651). The training target/
// preset catalogs stay in App (bulk-loaded by refreshData) — only the project-scoped
// dataset workflow lives here. Shared concerns (token, activeProject, error, jobs) are
// passed in; caption/training jobs push onto the shared jobs list via setJobs.
export function useTraining({ token, activeProject, setError, setJobs }) {
  const [trainingDatasets, setTrainingDatasets] = useState([]);
  const [trainingDatasetsProjectId, setTrainingDatasetsProjectId] = useState(null);
  const [loadingTrainingDatasets, setLoadingTrainingDatasets] = useState(false);
  const [trainingDatasetsError, setTrainingDatasetsError] = useState("");

  async function refreshTrainingDatasets(projectId = activeProject?.id, { signal } = {}) {
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

  async function uploadTrainingDatasetItem(file, projectId = activeProject?.id) {
    if (!projectId) {
      throw new Error("Create or open a project first.");
    }
    const form = new FormData();
    form.append("file", file);
    return apiFetch(`/api/v1/projects/${projectId}/training/uploads`, token, {
      method: "POST",
      body: form,
    });
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

  async function batchRenameTrainingDataset(datasetId, payload, projectId = activeProject?.id) {
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
  }

  async function writeTrainingDatasetCaptionSidecars(datasetId, payload, projectId = activeProject?.id) {
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
  }

  async function createTrainingDatasetCaptionJob(datasetId, payload, projectId = activeProject?.id) {
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
  }

  async function createTrainingJob(request, projectId = activeProject?.id) {
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
  }

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
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingJob,
  };
}
