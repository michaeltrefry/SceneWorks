import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { upsertJobNewest } from "../sorters.js";

const maxLoraUploadBytes = 2 * 1024 * 1024 * 1024;
const maxModelUploadBytes = 256 * 1024 * 1024 * 1024;

function uploadLimitLabel(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  return Number.isInteger(gib) ? `${gib}GB` : `${gib.toFixed(1)}GB`;
}

// Owns the model + LoRA catalogs and their import/download/convert/delete actions.
// Extracted from App.jsx (sc-1651). Models and LoRAs are coupled (a LoRA delete/
// import re-pulls both via the lora overlay), so they share one hook. App keeps the
// cross-cutting orchestrators — refreshData (bulk loader; seeds models+loras through
// the returned setters) and refreshDataWithLoraOverlay (refreshData + refreshLoras,
// also called by the SSE handler) — and passes them in. Both props MUST be
// identity-stable (sc-8811): they are useCallback deps of deleteModel/deleteLora,
// which sit in appContextValue's dependency array, so an unstable prop rebuilds the
// context value every App render and defeats the sc-4194 memoization. App passes
// ref-delegating useCallbacks for both.
export function useModelsAndLoras({
  token,
  activeProject,
  activeProjectRef,
  setError,
  setJobs,
  setActiveView,
  refreshData,
  refreshDataWithLoraOverlay,
}) {
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);

  // sc-4194: actions wrapped in useCallback so their identity is stable across App's
  // SSE-driven re-renders, letting appContextValue memoize.
  const refreshLoras = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      try {
        const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
        const items = await apiFetch(`/api/v1/loras${query}`, token, { signal });
        // sc-8858: an SSE-triggered refresh for a specific project can resolve after
        // the user switches away; committing then would clobber the new project's
        // LoRA overlay with the old one's. Drop the stale response — mirrors
        // refreshTimelines' guard (useTimelines.js). Only project-scoped refreshes
        // are guarded; a global refresh (projectId undefined) still commits.
        if (projectId && activeProjectRef?.current?.id && activeProjectRef.current.id !== projectId) {
          return;
        }
        setLoras(items);
        setError("");
      } catch (err) {
        if (isAbortError(err)) return;
        setError(err.message);
      }
    },
    [token, activeProject, activeProjectRef, setError],
  );

  const deleteModel = useCallback(
    async (model) => {
      const result = await apiFetch(`/api/v1/models/${encodeURIComponent(model.id)}`, token, {
        method: "DELETE",
      });
      if (result.removedManifestEntry) {
        setModels((items) => items.filter((item) => item.id !== model.id));
      }
      setError("");
      await refreshData();
      return result;
    },
    [token, setError, refreshData],
  );

  const deleteLora = useCallback(
    async (lora) => {
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
    },
    [token, activeProject, setError, refreshDataWithLoraOverlay],
  );

  const createModelImportJob = useCallback(
    async (payload, options = {}) => {
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
    setJobs((items) => upsertJobNewest(items, job));
    if (options.navigateToQueue ?? false) {
      setActiveView("Queue");
    }
    setError("");
    return job;
    },
    [token, setJobs, setActiveView, setError],
  );

  const createLoraImportJob = useCallback(
    async (payload, options = {}) => {
    if (payload.scope === "project" && !activeProject) {
      throw new Error("Create or open a project first.");
    }
    const { file, secondaryFile, ...metadata } = payload;
    if (file?.size > maxLoraUploadBytes) {
      throw new Error("Uploaded LoRA file exceeds the 2GB limit");
    }
    if (secondaryFile?.size > maxLoraUploadBytes) {
      throw new Error("Uploaded low-noise expert file exceeds the 2GB limit");
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
      // Wan A14B MoE pair (sc-1991): the low-noise expert half rides along as a
      // second file part the API stages under the high/low_noise convention.
      if (secondaryFile) {
        body.append("secondaryFile", secondaryFile);
      }
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
    setJobs((items) => upsertJobNewest(items, job));
    if (options.navigateToQueue ?? false) {
      setActiveView("Queue");
    }
    setError("");
    return job;
    },
    [token, activeProject, setJobs, setActiveView, setError],
  );

  const createModelDownloadJob = useCallback(
    async (model, options = {}) => {
      try {
        // sc-8509: install a specific quant tier when the caller passes one (the Models-page tier
        // picker for a quant-matrix model). Absent `variant` installs the model's default tier —
        // the back-compat single-download behavior every other caller relies on.
        const body = { requestedGpu: "auto" };
        if (options.variant) {
          body.variant = options.variant;
        }
        const job = await apiFetch(`/api/v1/models/${model.id}/download`, token, {
          method: "POST",
          body: JSON.stringify(body),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, setJobs, setError],
  );

  const createModelConvertJob = useCallback(
    async (model) => {
      try {
        const job = await apiFetch(`/api/v1/models/${model.id}/convert`, token, {
          method: "POST",
          body: JSON.stringify({ requestedGpu: "auto" }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, setJobs, setError],
  );

  // Built-in LoRA explicit download (sc-5944): queues a `lora_download` job that fetches
  // the catalog LoRA's HF files into the cache, flipping its installState to "installed".
  // Mirrors createModelDownloadJob.
  const createLoraDownloadJob = useCallback(
    async (lora) => {
      try {
        const job = await apiFetch(`/api/v1/loras/${encodeURIComponent(lora.id)}/download`, token, {
          method: "POST",
          body: JSON.stringify({ requestedGpu: "auto" }),
        });
        setJobs((items) => upsertJobNewest(items, job));
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, setJobs, setError],
  );

  return {
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
  };
}
