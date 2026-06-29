import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { terminalStatuses } from "../jobTypes.js";
import {
  loraMatchesModel,
  loraWeight,
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

function jobCreatedMs(job) {
  const parsed = Date.parse(job?.createdAt ?? "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function completedAnchorMs(job) {
  return Date.parse(job.completedAt ?? job.updatedAt ?? "");
}

// Pick the runs that belong in a studio's live stack from the tracked set, shared
// by every studio so stacking behaves identically. Runs are kept in the order they
// arrive (callers order oldest-first, so the active run sits on top and queued runs
// follow). Rules:
//   - canceled runs drop immediately (no output to show),
//   - running and queued runs always stack,
//   - a finished run slides out the moment a strictly-newer run starts (leaves the
//     queue), so the next run takes its place,
//   - a finished run with a run still queued behind it stays on top until that run
//     starts.
// `resultRendered(job)` reports whether a lone completed run's output has surfaced
// elsewhere (e.g. the Image studio's latest-batch grid); when it has, the run
// collapses out of the stack. Omit it for studios whose output ships in the job
// result itself (documents), where a lone completed run simply stays as the output.
export function selectStackedJobs(trackedLocalJobs, resultRendered) {
  const successorStarted = (job) =>
    trackedLocalJobs.some(
      (other) =>
        other.id !== job.id &&
        other.status !== "queued" &&
        other.status !== "canceled" &&
        jobCreatedMs(other) > jobCreatedMs(job),
    );
  const hasPendingSuccessor = (job) =>
    trackedLocalJobs.some(
      (other) => other.id !== job.id && other.status !== "canceled" && jobCreatedMs(other) > jobCreatedMs(job),
    );
  return trackedLocalJobs.filter((job) => {
    if (job.status === "canceled") {
      return false;
    }
    if (!terminalStatuses.has(job.status)) {
      return true;
    }
    if (successorStarted(job)) {
      return false;
    }
    if (job.status === "completed") {
      if (hasPendingSuccessor(job)) {
        return true;
      }
      return resultRendered ? !resultRendered(job) : true;
    }
    // Failed/interrupted runs with nothing started behind them stay visible so the
    // outcome is clear until the user moves on.
    return true;
  });
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
  initialPresetId = null,
  // sc-4196: LoRA selection state + validation, formerly duplicated in both studios.
  // Seeded from the persisted studio snapshot; advancedOpen/setAdvancedOpen are the
  // studio's own advanced-panel toggle (the hook auto-opens it when an incompatible
  // LoRA is selected so the blocking warning is visible).
  advancedOpen = false,
  setAdvancedOpen = () => {},
  initialSelectedLoraIds = [],
  initialLoraWeights = {},
  initialShowIncompatibleLoras = false,
}) {
  const [selectedPresetId, setSelectedPresetId] = useState(initialPresetId);
  const [resultFallbackTick, setResultFallbackTick] = useState(0);
  const [selectedLoraIds, setSelectedLoraIds] = useState(initialSelectedLoraIds);
  const [loraWeights, setLoraWeights] = useState(initialLoraWeights);
  const [showIncompatibleLoras, setShowIncompatibleLoras] = useState(initialShowIncompatibleLoras);

  // Snap the model back into range when the catalog changes out from under it.
  useEffect(() => {
    if (!models.some((item) => item.id === model)) {
      setModel(models[0]?.id ?? fallbackModelId);
    }
  }, [models, model, setModel, fallbackModelId]);

  // Drop a character selection that's no longer in the catalog.
  useEffect(() => {
    if (characterId && !characters.some((character) => character.id === characterId)) {
      setCharacterId("");
      setCharacterLookId("");
    }
  }, [characters, characterId, setCharacterId, setCharacterLookId]);

  const availablePresets = useMemo(
    () => presets.filter((preset) => presetMatchesWorkflow(preset, mode) && presetMatchesModel(preset, selectedModel, models)),
    [mode, presets, selectedModel?.id, models],
  );
  // sc-5875: presets are opt-in. With no explicit selection (fresh screen / None),
  // resolve to no preset so an unchosen preset's LoRA/resolution/prompt are never
  // silently applied. The dropdown's "None" and the applied config stay in agreement.
  const selectedPreset =
    selectedPresetId && selectedPresetId !== noPresetId
      ? availablePresets.find((preset) => preset.id === selectedPresetId) ?? null
      : null;
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

  const resultVisible = useCallback((job) => {
    if (job.result?.generationSetId) {
      return latestAssets.some((asset) => asset.generationSetId === job.result.generationSetId);
    }
    const assetIds = job.result?.assetIds ?? [];
    return assetIds.length > 0 && assetIds.every((id) => assets.some((asset) => asset.id === id));
  }, [assets, latestAssets]);

  const completedWaitExpired = useCallback((job, nowMs = Date.now()) => {
    const anchorMs = completedAnchorMs(job);
    return Number.isFinite(anchorMs) && nowMs - anchorMs > completedResultFallbackMs;
  }, []);

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
  }, [trackedLocalJobs, resultVisible, completedWaitExpired, resultFallbackTick]);

  // The visible stack (see selectStackedJobs). A lone completed run collapses once
  // its batch renders in the latest-batch grid or the SSE-lag window expires, so a
  // stale progress card never lingers.
  const localJobs = useMemo(
    () => selectStackedJobs(trackedLocalJobs, (job) => resultVisible(job) || completedWaitExpired(job)),
    [trackedLocalJobs, resultVisible, completedWaitExpired, resultFallbackTick],
  );

  // ---- LoRA selection (sc-4196: shared by Image + Video studios) ----
  const compatibleLoras = useMemo(() => loras.filter((lora) => {
    if (lora.presetManaged) {
      return false;
    }
    if (lora.installState === "missing") {
      return false;
    }
    if (showIncompatibleLoras) {
      return true;
    }
    return loraMatchesModel(lora, selectedModel);
  }), [loras, selectedModel, showIncompatibleLoras]);
  const compatibleLoraKey = useMemo(() => compatibleLoras.map((lora) => lora.id).join("|"), [compatibleLoras]);
  const selectedLoras = selectedLoraIds.map((id) => compatibleLoras.find((lora) => lora.id === id)).filter(Boolean);
  const userSelectedLoraCount = selectedLoras.filter((lora) => lora.scope !== "builtin").length;
  const selectedLoraValidationResult = useMemo(() => {
    const incompatible = selectedLoras.filter((lora) => !loraMatchesModel(lora, selectedModel)).map((lora) => lora.name ?? lora.id);
    return {
      incompatible,
      ok: incompatible.length === 0,
    };
  }, [selectedLoras, selectedModel]);
  const hasPendingCompatibleLoras = Boolean(selectedModel) && loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  const loraEmptyMessage = !selectedModel
    ? "No model selected"
    : hasPendingCompatibleLoras
      ? "No installed compatible LoRAs. Imports appear after the Queue completes."
      : showIncompatibleLoras
        ? "No installed LoRAs in the library."
        : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  // Drop selections that fall out of the compatible set (model/filter change).
  useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
  }, [compatibleLoraKey]);
  // Auto-open the advanced panel when an incompatible LoRA is selected so the
  // generate-blocking warning is visible.
  useEffect(() => {
    if (selectedLoraValidationResult.incompatible.length && !advancedOpen) {
      setAdvancedOpen(true);
    }
  }, [advancedOpen, selectedLoraValidationResult.incompatible.length]);

  function toggleLora(lora) {
    setSelectedLoraIds((ids) => {
      if (ids.includes(lora.id)) {
        return ids.filter((id) => id !== lora.id);
      }
      const selected = ids.map((id) => compatibleLoras.find((item) => item.id === id)).filter(Boolean);
      const userCount = selected.filter((item) => item.scope !== "builtin").length;
      if (lora.scope !== "builtin" && userCount >= 4) {
        return ids;
      }
      return [...ids, lora.id];
    });
  }

  // Per-LoRA strength: the override map falls back to the LoRA's default weight.
  // Order of application is intentionally not exposed — the worker combines
  // adapters additively (set_adapters / dequant-to-bf16 merge), so order has no
  // effect on output.
  function effectiveLoraWeight(lora) {
    const override = loraWeights[lora.id];
    return Number.isFinite(override) ? override : loraWeight(lora);
  }

  function setLoraWeight(id, value) {
    setLoraWeights((current) => ({ ...current, [id]: value }));
  }

  return {
    availablePresets,
    selectedPreset,
    selectedPresetId,
    setSelectedPresetId,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
    // LoRA selection bundle (sc-4196).
    selectedLoraIds,
    setSelectedLoraIds,
    loraWeights,
    setLoraWeights,
    showIncompatibleLoras,
    setShowIncompatibleLoras,
    compatibleLoras,
    selectedLoras,
    userSelectedLoraCount,
    selectedLoraValidationResult,
    loraEmptyMessage,
    toggleLora,
    effectiveLoraWeight,
    setLoraWeight,
  };
}

// The LoRA picker shared by both studios (sc-4196): the compatible-LoRA checklist
// with per-LoRA weight sliders, the "Show incompatible" toggle, and the empty state.
// All state lives in useGenerationStudio; this is a pure presentation of its bundle.
export function LoraPickerSection({
  selectedModel,
  selectedLoras,
  selectedLoraIds,
  compatibleLoras,
  userSelectedLoraCount,
  showIncompatibleLoras,
  setShowIncompatibleLoras,
  toggleLora,
  effectiveLoraWeight,
  setLoraWeight,
  loraEmptyMessage,
}) {
  return (
    <section className="lora-picker" aria-label="LoRA selection">
      <div>
        <strong>LoRAs</strong>
        <span>{selectedLoras.length ? `${selectedLoras.length} selected` : selectedModel ? "Installed and compatible" : "Choose a model"}</span>
      </div>
      <label className="checkline">
        <input
          checked={showIncompatibleLoras}
          onChange={(event) => setShowIncompatibleLoras(event.target.checked)}
          type="checkbox"
        />
        Show incompatible
      </label>
      {compatibleLoras.length ? (
        <div className="lora-choice-list">
          {compatibleLoras.map((lora) => {
            const checked = selectedLoraIds.includes(lora.id);
            const userLimitReached = lora.scope !== "builtin" && !checked && userSelectedLoraCount >= 4;
            const weight = effectiveLoraWeight(lora);
            return (
              <div className="lora-choice-item" key={lora.id}>
                <label className={checked ? "lora-choice active" : "lora-choice"}>
                  <input
                    checked={checked}
                    disabled={userLimitReached}
                    onChange={() => toggleLora(lora)}
                    type="checkbox"
                  />
                  <span>
                    <strong>{lora.name ?? lora.id}</strong>
                    <small>
                      {lora.scope ?? "global"} {lora.family ? `| ${lora.family}` : ""}
                    </small>
                  </span>
                </label>
                {checked ? (
                  <div className="lora-weight-row">
                    <span>Weight</span>
                    <input
                      aria-label={`${lora.name ?? lora.id} weight`}
                      max="2"
                      min="0"
                      onChange={(event) => setLoraWeight(lora.id, Number(event.target.value))}
                      step="0.05"
                      type="range"
                      value={weight}
                    />
                    <span className="lora-weight-value">{weight.toFixed(2)}</span>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty-panel compact-panel">{loraEmptyMessage}</div>
      )}
    </section>
  );
}

// The "Save as Preset" panel shared by both studios (sc-4196): name field, save
// button, project/global scope segment, and the inline save message. The actual
// save handler differs per studio (different payloads), so it's passed as onSave.
export function SavePresetPanel({
  presetName,
  setPresetName,
  savingPreset,
  presetSaveMessage,
  setPresetSaveMessage,
  onSave,
  presetScope,
  setPresetScope,
  activeProject,
  // Video studio gates saving to a subset of modes; pass an extra disable + a
  // tooltip explaining why. Image studio omits both (always saveable).
  saveDisabled = false,
  saveTitle = undefined,
}) {
  return (
    <div className="save-preset">
      <div className="save-preset-row">
        <input
          aria-label="Preset name"
          className="save-preset-name"
          disabled={savingPreset}
          onChange={(event) => {
            setPresetName(event.target.value);
            if (presetSaveMessage.text) {
              setPresetSaveMessage({ tone: "neutral", text: "" });
            }
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onSave();
            }
          }}
          placeholder="Name this setup…"
          value={presetName}
        />
        <button
          className="save-preset-btn"
          disabled={savingPreset || !presetName.trim() || saveDisabled}
          onClick={onSave}
          title={saveTitle}
          type="button"
        >
          <Icon.Preset size={14} /> {savingPreset ? "Saving…" : "Save as Preset"}
        </button>
      </div>
      <div className="save-preset-scope scope-segment" role="radiogroup" aria-label="Preset scope">
        <button
          aria-checked={presetScope === "project"}
          className={presetScope === "project" ? "active" : ""}
          disabled={!activeProject}
          onClick={() => setPresetScope("project")}
          role="radio"
          type="button"
        >
          <Icon.Folder size={13} /> This project
        </button>
        <button
          aria-checked={presetScope === "global"}
          className={presetScope === "global" ? "active" : ""}
          onClick={() => setPresetScope("global")}
          role="radio"
          type="button"
        >
          <Icon.Stars size={13} /> All projects
        </button>
      </div>
      {presetSaveMessage.text ? (
        <p className={presetSaveMessage.tone === "success" ? "inline-success" : "inline-warning"}>
          {presetSaveMessage.text}
        </p>
      ) : null}
    </div>
  );
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
