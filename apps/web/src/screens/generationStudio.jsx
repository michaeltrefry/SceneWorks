import React, { useEffect, useMemo, useState } from "react";
import {
  noPresetId,
  presetLoraDetails as buildPresetLoraDetails,
  presetMatchesModel,
  presetMatchesWorkflow,
  presetPromptParts as buildPresetPromptParts,
  presetValidation,
} from "../presetUtils.js";

const completedResultFallbackMs = 30000;

// Cmd/Ctrl+Enter submits the studio form from the prompt textarea.
export function onPromptKeyDown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }
}

// Shared state/derivations for the Image and Video studios: preset selection and
// validation, the catalog-driven model/character resets, and the completed-job
// "keep the progress card until the asset renders" machinery. The studios keep
// their own divergent pieces (preset-default field application, launch-request
// handling, submit payloads) and pass the bits this hook needs as arguments.
export function useGenerationStudio({
  mode,
  presets,
  selectedModel,
  loras,
  models,
  model,
  setModel,
  fallbackModelId,
  characters,
  characterId,
  setCharacterId,
  setCharacterLookId,
  assets,
  latestAssets,
  trackedLocalJobs,
}) {
  const [selectedPresetId, setSelectedPresetId] = useState(null);
  const [resultFallbackTick, setResultFallbackTick] = useState(0);

  // Snap the model back into range when the catalog changes out from under it.
  useEffect(() => {
    if (!models.some((item) => item.id === model)) {
      setModel(models[0]?.id ?? fallbackModelId);
    }
  }, [models, model]);

  // Drop a character selection that's no longer in the catalog.
  useEffect(() => {
    if (characterId && !characters.some((character) => character.id === characterId)) {
      setCharacterId("");
      setCharacterLookId("");
    }
  }, [characters, characterId]);

  const availablePresets = useMemo(
    () => presets.filter((preset) => presetMatchesWorkflow(preset, mode) && presetMatchesModel(preset, selectedModel)),
    [mode, presets, selectedModel?.id],
  );
  const selectedPreset =
    selectedPresetId === noPresetId
      ? null
      : selectedPresetId
        ? availablePresets.find((preset) => preset.id === selectedPresetId) ?? null
        : availablePresets[0] ?? null;
  const presetPromptParts = buildPresetPromptParts(selectedPreset);
  const presetLoraDetails = buildPresetLoraDetails(selectedPreset, loras);
  const presetValidationResult = useMemo(
    () => presetValidation(selectedPreset, loras, selectedModel),
    [selectedPreset, loras, selectedModel],
  );

  // An explicitly chosen preset that drops out of the available set falls back to
  // the first available preset (or None) rather than showing stale config.
  useEffect(() => {
    if (!selectedPresetId || selectedPresetId === noPresetId) {
      return;
    }
    if (!selectedPreset) {
      setSelectedPresetId(availablePresets[0]?.id ?? noPresetId);
    }
  }, [availablePresets, selectedPresetId, selectedPreset]);

  function resultVisible(job) {
    if (job.result?.generationSetId) {
      return latestAssets.some((asset) => asset.generationSetId === job.result.generationSetId);
    }
    const assetIds = job.result?.assetIds ?? [];
    return assetIds.length > 0 && assetIds.every((id) => assets.some((asset) => asset.id === id));
  }

  function completedAnchorMs(job) {
    return Date.parse(job.completedAt ?? job.updatedAt ?? "");
  }

  function completedWaitExpired(job, nowMs = Date.now()) {
    const anchorMs = completedAnchorMs(job);
    return Number.isFinite(anchorMs) && nowMs - anchorMs > completedResultFallbackMs;
  }

  // A completed job's assets can lag its SSE result by a beat; keep the progress
  // card until the asset renders or the fallback window expires, re-checking on a
  // timer so a card never lingers forever when the asset never arrives.
  useEffect(() => {
    const nowMs = Date.now();
    const pendingCompletedJobs = trackedLocalJobs.filter(
      (job) =>
        job.status === "completed" &&
        Number.isFinite(completedAnchorMs(job)) &&
        !resultVisible(job) &&
        !completedWaitExpired(job, nowMs),
    );
    if (!pendingCompletedJobs.length) {
      return undefined;
    }
    const nextDelay = Math.min(
      ...pendingCompletedJobs.map((job) => Math.max(0, completedResultFallbackMs - (nowMs - completedAnchorMs(job)))),
    );
    const timer = window.setTimeout(() => setResultFallbackTick((value) => value + 1), nextDelay + 50);
    return () => window.clearTimeout(timer);
  }, [assets, latestAssets, trackedLocalJobs, resultFallbackTick]);

  const localJobs = useMemo(
    () =>
      trackedLocalJobs.filter(
        (job) =>
          // Canceled runs produce no output, so drop them instead of leaving a
          // "Canceled" progress card behind.
          job.status !== "canceled" &&
          (job.status !== "completed" || (!resultVisible(job) && !completedWaitExpired(job))),
      ),
    [assets, latestAssets, trackedLocalJobs, resultFallbackTick],
  );

  return {
    availablePresets,
    selectedPreset,
    selectedPresetId,
    setSelectedPresetId,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
  };
}

// The "what this preset adds" strip shown under the preset picker in both studios.
export function PresetGuidanceStrip({ selectedPreset, presetPromptParts, presetLoraDetails, noPresetHint }) {
  if (!selectedPreset) {
    return (
      <div className="guidance-strip">
        <strong>No preset selected</strong>
        <span>{noPresetHint}</span>
      </div>
    );
  }
  return (
    <div className="guidance-strip">
      <strong>{selectedPreset.ui?.description ?? "Preset defaults active"}</strong>
      <span>
        {presetPromptParts.length ? `Adds: ${presetPromptParts.join(", ")}` : "No prompt fragments"}
        {presetLoraDetails.length
          ? ` | Preset LoRA applied at generation: ${presetLoraDetails.map((lora) => lora.name ?? lora.id).join(", ")}`
          : " | No preset LoRAs"}
        {presetLoraDetails.some((lora) => lora.missing) ? " | Import still pending" : ""}
      </span>
    </div>
  );
}

// The preset "missing"/"incompatible" inline warnings shared by both studios.
export function PresetValidationWarnings({ presetValidationResult, selectedModel }) {
  return (
    <>
      {presetValidationResult.missing.length ? (
        <p className="inline-warning">
          Preset cannot run until LoRA import finishes: {presetValidationResult.missing.join(", ")}. Wait for the Queue or choose another preset.
        </p>
      ) : null}
      {presetValidationResult.incompatible.length ? (
        <p className="inline-warning">
          Preset cannot run with {selectedModel?.name ?? "the selected model"} because these LoRAs are incompatible: {presetValidationResult.incompatible.join(", ")}. Choose another preset or model.
        </p>
      ) : null}
    </>
  );
}
