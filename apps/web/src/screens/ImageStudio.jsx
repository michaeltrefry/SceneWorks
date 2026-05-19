import React, { useEffect, useMemo, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { JobProgressCard } from "../components/JobProgress.jsx";
import {
  loraMatchesModel,
  loraWeight,
  presetLoraDetails as buildPresetLoraDetails,
  presetMatchesModel,
  presetMatchesWorkflow,
  presetPromptParts as buildPresetPromptParts,
  presetValidation,
} from "../presetUtils.js";

const completedResultFallbackMs = 30000;
const localErrorStatuses = new Set(["failed", "canceled", "interrupted"]);
const localErrorLabels = {
  failed: "Failed",
  canceled: "Canceled",
  interrupted: "Interrupted",
};

function jobResultAssets(job, assets) {
  const catalogById = new Map(assets.map((asset) => [asset.id, asset]));
  const resultAssets = (job.result?.assets ?? []).filter((asset) => asset?.type === "image");
  const resultById = new Map(resultAssets.map((asset) => [asset.id, catalogById.get(asset.id) ?? asset]));
  const assetIds = job.result?.assetIds ?? [];
  if (assetIds.length) {
    return assetIds
      .map((id) => resultById.get(id) ?? catalogById.get(id))
      .filter((asset) => asset?.type === "image");
  }
  if (resultAssets.length) {
    return resultAssets.map((asset) => catalogById.get(asset.id) ?? asset);
  }
  if (job.result?.generationSetId) {
    return assets
      .filter((asset) => asset.type === "image" && asset.generationSetId === job.result.generationSetId)
      .sort((left, right) => assetBatchIndex(left) - assetBatchIndex(right));
  }
  return [];
}

function jobExpectedCount(job, completedCount) {
  const expected = Number(job.result?.expectedCount ?? job.result?.count ?? job.payload?.count);
  return Number.isFinite(expected) && expected > 0 ? Math.max(expected, completedCount) : completedCount;
}

function assetBatchIndex(asset) {
  const candidates = [
    asset?.batchIndex,
    asset?.recipe?.batchIndex,
    asset?.recipe?.normalizedSettings?.batchIndex,
    asset?.lineage?.batchIndex,
  ];
  for (const candidate of candidates) {
    const value = Number(candidate);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  const fileMatch = String(asset?.file?.path ?? "").match(/_(\d{4})\.[^./\\]+$/);
  if (fileMatch) {
    return Number(fileMatch[1]) - 1;
  }
  const nameMatch = String(asset?.displayName ?? "").match(/#(\d+)\s*$/);
  return nameMatch ? Number(nameMatch[1]) - 1 : Number.POSITIVE_INFINITY;
}

function jobPendingSlotLabel(job, index) {
  if (localErrorStatuses.has(job.status)) {
    return `${localErrorLabels[job.status] ?? "Failed"} #${index + 1}`;
  }
  return `Pending #${index + 1}`;
}

export function ImageStudio({
  activeProject,
  assets,
  characters,
  createImageJob,
  deleteAsset,
  purgeAsset,
  gpuOptions,
  imageModels,
  latestAssets,
  launchRequest,
  localJobs: trackedLocalJobs = [],
  loras = [],
  onLocalJobCreated,
  onOpenQueue,
  onPreview,
  recipePresets = [],
  requestedGpu,
  selectedAsset,
  setRequestedGpu,
  updateAssetStatus,
}) {
  const [mode, setMode] = useState("text_to_image");
  const [prompt, setPrompt] = useState("A cinematic frame of a neon street at midnight");
  const [stylePreset, setStylePreset] = useState("cinematic");
  const [count, setCount] = useState(4);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(imageModels[0]?.id ?? "z_image_turbo");
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [resolution, setResolution] = useState("1024x1024");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  const [selectedLoraIds, setSelectedLoraIds] = useState([]);
  const [showIncompatibleLoras, setShowIncompatibleLoras] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [resultFallbackTick, setResultFallbackTick] = useState(0);
  const editImageAssets = useMemo(
    () => assets.filter((asset) => asset.type === "image" || asset.type === "frame"),
    [assets],
  );

  function serializeLora(lora, override = {}) {
    return {
      id: lora.id,
      name: lora.name ?? lora.id,
      scope: lora.scope ?? "global",
      weight: Number.isFinite(Number(override.weight)) ? Number(override.weight) : loraWeight(lora),
      triggerWords: lora.triggerWords ?? [],
      compatibility: lora.compatibility ?? {},
      presetManaged: Boolean(lora.presetManaged),
    };
  }

  useEffect(() => {
    if (!imageModels.some((item) => item.id === model)) {
      setModel(imageModels[0]?.id ?? "z_image_turbo");
    }
  }, [imageModels, model]);

  useEffect(() => {
    if (mode === "edit_image" && selectedAsset?.id) {
      setSourceAssetId(selectedAsset.id);
    }
  }, [mode, selectedAsset?.id]);

  useEffect(() => {
    if (launchRequest?.view !== "Image") {
      return;
    }
    if (launchRequest.characterId) {
      setMode(launchRequest.mode ?? "character_image");
      setCharacterId(launchRequest.characterId);
      setCharacterLookId(launchRequest.lookId ?? "");
      return;
    }
    if (launchRequest.assetId !== selectedAsset?.id) {
      return;
    }
    setMode(launchRequest.mode);
    if (launchRequest.mode === "edit_image" && selectedAsset?.id) {
      setSourceAssetId(selectedAsset.id);
    }
  }, [launchRequest?.id, selectedAsset?.id]);

  useEffect(() => {
    if (characterId && !characters.some((character) => character.id === characterId)) {
      setCharacterId("");
      setCharacterLookId("");
    }
  }, [characters, characterId]);

  const availableModels = imageModels.filter((item) => {
    const caps = item.capabilities ?? [];
    if (mode === "edit_image") {
      return caps.includes("edit_image") || caps.includes("image_edit");
    }
    return item.type === "image";
  });
  const selectedModel = imageModels.find((item) => item.id === model);
  const availableRecipePresets = useMemo(() => {
    return recipePresets.filter((preset) => presetMatchesWorkflow(preset, mode) && presetMatchesModel(preset, selectedModel));
  }, [mode, recipePresets, selectedModel?.id]);
  const selectedRecipePreset = availableRecipePresets.find((preset) => preset.id === stylePreset) ?? availableRecipePresets[0] ?? null;
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
  const presetLoraDetails = buildPresetLoraDetails(selectedRecipePreset, loras);
  const presetPromptParts = buildPresetPromptParts(selectedRecipePreset);
  const presetValidationResult = useMemo(
    () => presetValidation(selectedRecipePreset, loras, selectedModel),
    [selectedRecipePreset, loras, selectedModel],
  );
  const hasPendingCompatibleLoras = Boolean(selectedModel) && loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  const loraEmptyMessage = !selectedModel
    ? "No model selected"
    : hasPendingCompatibleLoras
      ? "No installed compatible LoRAs. Imports appear after the Queue completes."
      : showIncompatibleLoras
        ? "No installed LoRAs in the library."
        : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;
  const [width, height] = resolution.split("x").map((value) => Number(value));

  useEffect(() => {
    if (!availableRecipePresets.some((preset) => preset.id === stylePreset)) {
      setStylePreset(availableRecipePresets[0]?.id ?? "");
    }
  }, [availableRecipePresets, stylePreset]);

  useEffect(() => {
    if (!selectedRecipePreset) {
      return;
    }
    const defaults = selectedRecipePreset.defaults ?? {};
    if (defaults.count) {
      setCount(Number(defaults.count));
    }
    if (defaults.resolution) {
      setResolution(defaults.resolution);
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "negativePrompt")) {
      setNegativePrompt(defaults.negativePrompt ?? "");
    }
  }, [selectedRecipePreset?.id]);

  useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
  }, [compatibleLoraKey]);

  function toggleLora(lora) {
    setSelectedLoraIds((ids) => {
      if (ids.includes(lora.id)) {
        return ids.filter((id) => id !== lora.id);
      }
      const selected = ids.map((id) => compatibleLoras.find((item) => item.id === id)).filter(Boolean);
      const userCount = selected.filter((item) => item.scope !== "builtin").length;
      if (lora.scope !== "builtin" && userCount >= 2) {
        return ids;
      }
      return [...ids, lora.id];
    });
  }

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
    () => trackedLocalJobs.filter((job) => job.status !== "completed" || (!resultVisible(job) && !completedWaitExpired(job))),
    [assets, latestAssets, trackedLocalJobs, resultFallbackTick],
  );
  const reviewSlots = useMemo(() => {
    if (!localJobs.length) {
      return latestAssets.map((asset) => ({ type: "asset", id: asset.id, asset }));
    }
    return localJobs.flatMap((job) => {
      const completedAssets = jobResultAssets(job, assets);
      const expectedCount = jobExpectedCount(job, completedAssets.length);
      return Array.from({ length: expectedCount }, (_, index) => {
        const asset = completedAssets[index];
        if (asset) {
          return { type: "asset", id: `${job.id}:${asset.id}`, asset };
        }
        return {
          type: "placeholder",
          id: `${job.id}:slot-${index}`,
          label: jobPendingSlotLabel(job, index),
          isError: localErrorStatuses.has(job.status),
        };
      });
    });
  }, [assets, latestAssets, localJobs]);
  const hasReviewContent = Boolean(localJobs.length || reviewSlots.length);

  async function submit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    setSubmitting(true);
    try {
      const job = await createImageJob({
        mode,
        prompt,
        negativePrompt,
        model,
        count,
        seed: seed === "" ? null : Number(seed),
        width,
        height,
        stylePreset,
        recipePresetId: selectedRecipePreset?.id ?? null,
        characterId: mode === "character_image" ? characterId || null : null,
        characterLookId: mode === "character_image" ? characterLookId || null : null,
        sourceAssetId: mode === "edit_image" ? sourceAssetId || null : null,
        loras: selectedLoras.map((lora) => serializeLora(lora)),
        advanced: {
          resolution,
        },
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="main-surface image-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Image Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <div className="segmented-control" role="tablist" aria-label="Image mode">
          {[
            ["text_to_image", "Text"],
            ["edit_image", "Edit"],
            ["character_image", "Character"],
            ["style_variations", "Variations"],
          ].map(([value, label]) => (
            <button className={mode === value ? "active" : ""} key={value} onClick={() => setMode(value)} type="button">
              {label}
            </button>
          ))}
        </div>
      </div>

      <form className="studio-layout" onSubmit={submit}>
        <section className="studio-controls">
          {mode === "edit_image" ? (
            <AssetPickerField
              assets={editImageAssets}
              buttonLabel="Select image"
              emptyLabel="No source image selected"
              label="Source"
              onChange={setSourceAssetId}
              value={sourceAssetId}
            />
          ) : null}

          {mode === "character_image" ? (
            <>
              <div className="control-grid compact-controls">
                <label>
                  Character
                  <select onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
                    <option value="">Select character</option>
                    {characters.map((character) => (
                      <option key={character.id} value={character.id}>
                        {character.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Look
                  <select onChange={(event) => setCharacterLookId(event.target.value)} value={characterLookId}>
                    <option value="">Default look</option>
                    {(characters.find((character) => character.id === characterId)?.looks ?? []).map((look) => (
                      <option key={look.id} value={look.id}>
                        {look.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="guidance-strip">
                <strong>Recipe-only character</strong>
                <span>Character and look are saved with the recipe; adapter-level reference and LoRA conditioning are not active yet.</span>
              </div>
            </>
          ) : null}

          <label className="prompt-field">
            Prompt
            <textarea onChange={(event) => setPrompt(event.target.value)} value={prompt} />
          </label>

          <div className="control-grid">
            <label>
              Style
              <select disabled={!availableRecipePresets.length} onChange={(event) => setStylePreset(event.target.value)} value={stylePreset}>
                {availableRecipePresets.length ? (
                  availableRecipePresets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name ?? preset.id}
                    </option>
                  ))
                ) : (
                  <option value="">Presets unavailable</option>
                )}
              </select>
            </label>
            <label>
              Count
              <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
            </label>
            <label>
              GPU
              <select onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
                {gpuOptions.map((gpu) => (
                  <option key={gpu} value={gpu}>
                    {gpu === "auto" ? "Auto" : gpu}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {selectedRecipePreset ? (
            <div className="guidance-strip">
              <strong>{selectedRecipePreset.ui?.description ?? "Preset defaults active"}</strong>
              <span>
                {presetPromptParts.length ? `Adds: ${presetPromptParts.join(", ")}` : "No prompt fragments"}
                {presetLoraDetails.length
                  ? ` | Preset LoRA applied at generation: ${presetLoraDetails.map((lora) => lora.name ?? lora.id).join(", ")}`
                  : " | No preset LoRAs"}
                {presetLoraDetails.some((lora) => lora.missing) ? " | Import still pending" : ""}
              </span>
            </div>
          ) : (
            <div className="guidance-strip">
              <strong>Presets unavailable</strong>
              <span>Generation can continue, but no preset defaults or managed LoRAs will be applied.</span>
            </div>
          )}

          <section className="lora-picker" aria-label="LoRA selection">
            <div>
              <strong>LoRAs</strong>
              <span>{selectedLoras.length ? `${selectedLoras.length} selected` : selectedModel ? "Installed and compatible" : "Choose a model"}</span>
            </div>
            {advancedOpen ? (
              <label className="checkline">
                <input
                  checked={showIncompatibleLoras}
                  onChange={(event) => setShowIncompatibleLoras(event.target.checked)}
                  type="checkbox"
                />
                Show incompatible
              </label>
            ) : null}
            {compatibleLoras.length ? (
              <div className="lora-choice-list">
                {compatibleLoras.map((lora) => {
                  const checked = selectedLoraIds.includes(lora.id);
                  const userLimitReached = lora.scope !== "builtin" && !checked && userSelectedLoraCount >= 2;
                  return (
                    <label className={checked ? "lora-choice active" : "lora-choice"} key={lora.id}>
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
                  );
                })}
              </div>
            ) : (
              <div className="empty-panel compact-panel">{loraEmptyMessage}</div>
            )}
          </section>

          <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
            {advancedOpen ? "Hide advanced" : "Advanced"}
          </button>

          {advancedOpen ? (
            <div className="advanced-panel">
              <label>
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {(availableModels.length ? availableModels : imageModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
              </label>
              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  <option value="768x768">768 x 768</option>
                  <option value="1024x1024">1024 x 1024</option>
                  <option value="1280x720">1280 x 720</option>
                  <option value="720x1280">720 x 1280</option>
                </select>
              </label>
              <label className="prompt-field">
                Negative prompt
                <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
              </label>
            </div>
          ) : null}

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
          <button
            className="primary-action"
            disabled={submitting || !activeProject || !prompt.trim() || (mode === "character_image" && !characterId) || !presetValidationResult.ok}
            type="submit"
          >
            {submitting ? "Queueing..." : "Generate"}
          </button>
        </section>

        <section className="review-panel">
          <div className="section-heading">
            <p className="eyebrow">Fresh batch</p>
            <h2>Review</h2>
          </div>
          {localJobs.length ? (
            <div className="local-job-stack">
              {localJobs.map((job) => (
                <JobProgressCard job={job} key={job.id} label="Image generation" onOpenQueue={onOpenQueue} />
              ))}
            </div>
          ) : null}
          {reviewSlots.length ? (
            <div className="review-grid">
              {reviewSlots.map((slot) => (
                slot.type === "asset" ? (
                  <AssetCard
                    asset={slot.asset}
                    deleteAsset={deleteAsset}
                    key={slot.id}
                    onPreview={onPreview}
                    purgeAsset={purgeAsset}
                    updateAssetStatus={updateAssetStatus}
                  />
                ) : (
                  <div className={slot.isError ? "review-placeholder failed" : "review-placeholder"} key={slot.id}>
                    <span>{slot.label}</span>
                  </div>
                )
              ))}
            </div>
          ) : hasReviewContent ? null : (
            <div className="empty-panel">No fresh image batch</div>
          )}
        </section>
      </form>
    </section>
  );
}
