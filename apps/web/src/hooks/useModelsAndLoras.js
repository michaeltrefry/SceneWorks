import { useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { sortNewest } from "../sorters.js";

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
// also called by the SSE handler) — and passes them in.
export function useModelsAndLoras({
  token,
  activeProject,
  setError,
  setJobs,
  setActiveView,
  refreshData,
  refreshDataWithLoraOverlay,
}) {
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);

  async function refreshLoras(projectId = activeProject?.id, { signal } = {}) {
    try {
      const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
      const items = await apiFetch(`/api/v1/loras${query}`, token, { signal });
      setLoras(items);
      setError("");
    } catch (err) {
      if (isAbortError(err)) return;
      setError(err.message);
    }
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
    return job;
  }

  async function createModelDownloadJob(model) {
    try {
      const job = await apiFetch(`/api/v1/models/${model.id}/download`, token, {
        method: "POST",
        body: JSON.stringify({ requestedGpu: "auto" }),
      });
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function createModelConvertJob(model) {
    try {
      const job = await apiFetch(`/api/v1/models/${model.id}/convert`, token, {
        method: "POST",
        body: JSON.stringify({ requestedGpu: "auto" }),
      });
      setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortNewest));
      setError("");
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

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
    createModelConvertJob,
  };
}
