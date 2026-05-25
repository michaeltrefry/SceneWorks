import { useState } from "react";
import { apiFetch, isAbortError } from "../api.js";

// Owns the project's character roster plus every character CRUD/reference/look/LoRA
// mutation. Extracted from App.jsx (sc-1651) to shrink the god module; behavior is
// unchanged — shared concerns (token, active project, error/navigation) are passed in
// so the hook stays a thin data layer. Character test jobs surface live via the SSE
// job stream (the create endpoint publishes job.updated), so no post-create refetch.
export function useCharacters({ token, activeProject, setError, requestedGpu, setActiveView }) {
  const [characters, setCharacters] = useState([]);

  async function refreshCharacters(projectId = activeProject?.id, { signal } = {}) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/characters`, token, { signal });
      setCharacters(items);
      setError("");
    } catch (err) {
      if (isAbortError(err)) return;
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
      return { id: characterId, status: "queued" };
    });
  }

  return {
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
  };
}
