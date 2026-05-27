import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { JobProgressCard } from "../components/JobProgress.jsx";

const PROMPT_SUGGESTION_POOL = [
  "Barista pouring espresso, morning light",
  "Runner cresting a dune at dawn",
  "Dewdrop on a fern, soft bokeh",
  "Watchmaker at her bench, warm tungsten",
  "Cyclist on a wet cobblestone street, neon reflections",
  "Cellist mid-bow, theater spotlight from above",
  "Glassblower shaping a vessel, kiln glow",
  "Fox watching from the edge of a snowy forest",
  "Surfer at golden hour, backlit spray",
  "Quiet kitchen window, herbs in low afternoon light",
  "Vintage typewriter on a roll-top desk, dust motes",
  "Lighthouse beam slicing through coastal fog",
];

// Character (IP-Adapter) variations: the reference image supplies identity, so
// these describe scene / pose / lighting to vary rather than a standalone subject.
const CHARACTER_SUGGESTION_POOL = [
  "studio portrait, soft key light",
  "in a sunlit park, candid",
  "city street at dusk, cinematic",
  "seated at a wooden desk, warm light",
  "walking through a busy market, natural light",
  "close-up, dramatic side lighting",
  "outdoors at golden hour, backlit",
  "neutral grey backdrop, even studio lighting",
];

function pickSuggestions(count, pool = PROMPT_SUGGESTION_POOL) {
  const available = [...pool];
  const result = [];
  for (let index = 0; index < count && available.length; index += 1) {
    const pick = Math.floor(Math.random() * available.length);
    result.push(available.splice(pick, 1)[0]);
  }
  return result;
}

// Seeded into the prompt when entering character mode (only when untouched). The
// character's own notes win if present; otherwise a neutral, type-appropriate
// variation prompt — identity still comes from the reference image, not this text.
function defaultCharacterPrompt(character) {
  const note = (character?.description ?? "").trim();
  if (note) {
    return note;
  }
  switch (character?.type) {
    case "creature":
      return "The creature in a new setting, varied pose, natural lighting";
    case "object":
      return "The object from a fresh angle and setting, studio lighting";
    default:
      return "Portrait of the character, varied pose and expression, natural lighting";
  }
}
import {
  loraMatchesModel,
  loraWeight,
  serializeLora,
  clearPresetDefault,
  noPresetId,
  rememberPresetDefault,
} from "../presetUtils.js";
import {
  onPromptKeyDown,
  PresetGuidanceStrip,
  PresetValidationWarnings,
  useGenerationStudio,
} from "./generationStudio.jsx";
import { useAppContext } from "../context/AppContext.js";
import { errorStatuses } from "../jobTypes.js";

// Used only for models that don't declare limits.resolutions (e.g. user-imported).
const DEFAULT_RESOLUTION_OPTIONS = ["768x768", "1024x1024", "1280x720", "720x1280"];
const UPSCALE_ENGINES = [
  { id: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
  { id: "aura-sr", label: "AuraSR", factors: [4] },
];

function formatResolutionLabel(value) {
  const [width, height] = String(value).split("x");
  return height ? `${width} × ${height}` : value;
}

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
    // The worker emits assetIds in batch-slot order, so preserve this array order when filling review slots.
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
  const basename = String(asset?.file?.path ?? "").split(/[\\/]/).pop() ?? "";
  const fileMatch = basename.match(/_(\d{4})\.[^.]+$/);
  if (fileMatch) {
    return Number(fileMatch[1]) - 1;
  }
  const nameMatch = String(asset?.displayName ?? "").match(/#(\d+)\s*$/);
  return nameMatch ? Number(nameMatch[1]) - 1 : Number.POSITIVE_INFINITY;
}

function jobPendingSlotLabel(job, index) {
  if (errorStatuses.has(job.status)) {
    return `${localErrorLabels[job.status] ?? "Failed"} #${index + 1}`;
  }
  return `Pending #${index + 1}`;
}

export function ImageStudio() {
  const {
    activeProject,
    assets,
    characters,
    createImageJob,
    deleteAsset,
    purgeAsset,
    gpuOptions,
    imageModels,
    latestImageAssets,
    studioLaunch,
    imageLocalJobs = [],
    loras = [],
    jobAction,
    rememberLocalGenerationJob,
    setActiveView,
    setPreviewAsset,
    presets = [],
    requestedGpu,
    selectedAsset,
    setRequestedGpu,
    updateAssetStatus,
  } = useAppContext();
  const latestAssets = latestImageAssets;
  const launchRequest = studioLaunch;
  const trackedLocalJobs = imageLocalJobs;
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onLocalJobCreated = (job) => rememberLocalGenerationJob("image", job);
  const onOpenPresets = () => setActiveView("Presets");
  const onOpenQueue = () => setActiveView("Queue");
  const onPreview = setPreviewAsset;
  const [sceneSuggestions] = useState(() => pickSuggestions(4));
  const [characterSuggestions] = useState(() => pickSuggestions(4, CHARACTER_SUGGESTION_POOL));
  const [mode, setMode] = useState("text_to_image");
  const [prompt, setPrompt] = useState("A cinematic frame of a neon street at midnight");
  // True once the user types or picks a suggestion, so the character-mode default
  // prompt never clobbers their own wording.
  const promptEdited = useRef(false);
  const setPromptFromUser = (value) => {
    promptEdited.current = true;
    setPrompt(value);
  };
  const suggestions = mode === "character_image" ? characterSuggestions : sceneSuggestions;
  const [count, setCount] = useState(4);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(imageModels[0]?.id ?? "z_image_turbo");
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [resolution, setResolution] = useState("1024x1024");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  // Character reference (IP-Adapter / InstantID) — the approved reference image whose
  // identity is carried across variations. `ipAdapterScale` rides in `advanced`; for
  // InstantID, `controlnetScale` (IdentityNet landmark lock) rides there too.
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [ipAdapterScale, setIpAdapterScale] = useState(0.6);
  const [controlnetScale, setControlnetScale] = useState(0.8);
  const [upscaleEnabled, setUpscaleEnabled] = useState(false);
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [upscaleEngine, setUpscaleEngine] = useState("real-esrgan");
  const [selectedLoraIds, setSelectedLoraIds] = useState([]);
  const [loraWeights, setLoraWeights] = useState({});
  const [showIncompatibleLoras, setShowIncompatibleLoras] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const presetDefaultSnapshots = useRef({});
  const editImageAssets = useMemo(
    () => assets.filter((asset) => asset.type === "image" || asset.type === "frame"),
    [assets],
  );

  function handleModeChange(nextMode) {
    if (nextMode === "edit_image") {
      setCount(1);
    } else if (nextMode === "text_to_image" || nextMode === "character_image") {
      setCount(4);
    }
    setMode(nextMode);
  }

  function handleUpscaleEngineChange(nextEngine) {
    setUpscaleEngine(nextEngine);
    const option = UPSCALE_ENGINES.find((candidate) => candidate.id === nextEngine);
    if (option && !option.factors.includes(upscaleFactor)) {
      setUpscaleFactor(option.factors[0]);
    }
  }

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
      if (launchRequest.referenceAssetId) {
        setReferenceAssetId(launchRequest.referenceAssetId);
      }
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

  const availableModels = useMemo(
    () =>
      imageModels.filter((item) => {
        const caps = item.capabilities ?? [];
        if (mode === "edit_image") {
          return caps.includes("edit_image") || caps.includes("image_edit");
        }
        if (mode === "character_image") {
          // Only models with a reference-image (IP-Adapter) engine can preserve a
          // character's identity from one reference; gate the picker to them.
          return caps.includes("character_image");
        }
        return item.type === "image";
      }),
    [imageModels, mode],
  );
  // When the mode change filters out the current model (e.g. Lens-Turbo is the
  // text default but isn't edit-capable), snap to the first available model so
  // the dropdown's displayed option matches the value actually submitted.
  useEffect(() => {
    if (availableModels.length && !availableModels.some((item) => item.id === model)) {
      setModel(availableModels[0].id);
    }
  }, [availableModels, model]);
  const selectedModel = imageModels.find((item) => item.id === model);
  // Reference-tuning hints declared by the model (ui.*). InstantID raises the
  // reference-strength default and exposes a second "Identity structure" slider
  // (controlnetConditioningScale); models without these keys (e.g. Kolors) keep the
  // single reference-strength slider at the global default.
  const identityStructure = selectedModel?.ui?.identityStructure;
  // Reset the reference sliders to the selected model's declared defaults whenever the
  // model changes, so InstantID starts at its tuned 0.8/0.8 and Kolors at 0.6.
  useEffect(() => {
    const ui = imageModels.find((item) => item.id === model)?.ui ?? {};
    setIpAdapterScale(typeof ui.referenceStrengthDefault === "number" ? ui.referenceStrengthDefault : 0.6);
    setControlnetScale(typeof ui.identityStructure?.default === "number" ? ui.identityStructure.default : 0.8);
  }, [model]);
  // Approved reference images for the selected character (the IP-Adapter identity
  // source). Resolve the full asset from the catalog so thumbnails render even when
  // the character payload only carries assetIds.
  const characterReferences = useMemo(() => {
    const character = characters.find((item) => item.id === characterId);
    return (character?.approvedReferences ?? []).map((reference) => ({
      assetId: reference.assetId,
      role: reference.role ?? null,
      asset: reference.asset ?? assets.find((item) => item.id === reference.assetId) ?? null,
    }));
  }, [characters, characterId, assets]);
  // Keep the selected reference valid: default to the first approved reference when
  // none is chosen or the current one no longer belongs to this character.
  useEffect(() => {
    if (mode !== "character_image") {
      return;
    }
    if (characterReferences.some((reference) => reference.assetId === referenceAssetId)) {
      return;
    }
    setReferenceAssetId(characterReferences[0]?.assetId ?? "");
  }, [mode, characterReferences, referenceAssetId]);
  // Seed a character-appropriate default prompt when entering character mode, unless
  // the user has already typed/picked their own. The generic text-to-image default
  // ("neon street at midnight") makes no sense for character variations.
  useEffect(() => {
    if (mode !== "character_image" || !characterId || promptEdited.current) {
      return;
    }
    const character = characters.find((item) => item.id === characterId);
    if (character) {
      setPrompt(defaultCharacterPrompt(character));
    }
  }, [mode, characterId, characters]);
  const resolutionOptions = useMemo(
    () =>
      selectedModel?.limits?.resolutions?.length
        ? selectedModel.limits.resolutions
        : DEFAULT_RESOLUTION_OPTIONS,
    [selectedModel],
  );
  // Keep the selected resolution valid for the current model's buckets. Switching
  // to a model whose options exclude the current value snaps to its default (or
  // 1024x1024, then the first option) rather than leaving a stale, unselectable value.
  useEffect(() => {
    if (resolutionOptions.includes(resolution)) {
      return;
    }
    const modelDefault = selectedModel?.defaults?.resolution;
    const preferred = resolutionOptions.includes(modelDefault)
      ? modelDefault
      : resolutionOptions.includes("1024x1024")
        ? "1024x1024"
        : resolutionOptions[0];
    setResolution(preferred);
  }, [resolutionOptions, resolution, selectedModel?.defaults?.resolution]);
  const {
    availablePresets,
    selectedPreset,
    setSelectedPresetId,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
  } = useGenerationStudio({
    mode,
    presets,
    selectedModel,
    loras,
    models: imageModels,
    model,
    setModel,
    fallbackModelId: "z_image_turbo",
    characters,
    characterId,
    setCharacterId,
    setCharacterLookId,
    assets,
    latestAssets,
    trackedLocalJobs,
  });
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
  useEffect(() => {
    if (selectedLoraValidationResult.incompatible.length && !advancedOpen) {
      setAdvancedOpen(true);
    }
  }, [advancedOpen, selectedLoraValidationResult.incompatible.length]);
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
    if (!selectedPreset) {
      clearPresetDefault(setCount, presetDefaultSnapshots, "count");
      clearPresetDefault(setResolution, presetDefaultSnapshots, "resolution");
      clearPresetDefault(setNegativePrompt, presetDefaultSnapshots, "negativePrompt");
      return;
    }
    const defaults = selectedPreset.defaults ?? {};
    if (defaults.count) {
      const appliedValue = Number(defaults.count);
      setCount((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "count", current, appliedValue);
        return appliedValue;
      });
    }
    if (defaults.resolution) {
      const appliedValue = defaults.resolution;
      setResolution((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "resolution", current, appliedValue);
        return appliedValue;
      });
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "negativePrompt")) {
      const appliedValue = defaults.negativePrompt ?? "";
      setNegativePrompt((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "negativePrompt", current, appliedValue);
        return appliedValue;
      });
    }
  }, [selectedPreset?.id]);

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
          isError: errorStatuses.has(job.status),
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
        recipePresetId: selectedPreset?.id ?? null,
        characterId: mode === "character_image" ? characterId || null : null,
        characterLookId: mode === "character_image" ? characterLookId || null : null,
        sourceAssetId: mode === "edit_image" ? sourceAssetId || null : null,
        referenceAssetId: mode === "character_image" ? referenceAssetId || null : null,
        loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
        ...(upscaleEnabled
          ? {
              upscale: {
                enabled: true,
                factor: upscaleFactor,
                engine: upscaleEngine,
              },
            }
          : {}),
        advanced: {
          resolution,
          // IP-Adapter / InstantID reference strength only applies when a character
          // reference is attached; the worker reads advanced.ipAdapterScale.
          ...(mode === "character_image" && referenceAssetId ? { ipAdapterScale } : {}),
          // Identity structure (controlnetConditioningScale) is InstantID-only — sent
          // only when the model exposes the control and a reference is attached.
          ...(mode === "character_image" && referenceAssetId && identityStructure
            ? { controlnetConditioningScale: controlnetScale }
            : {}),
        },
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  const generateDisabled =
    submitting ||
    !activeProject ||
    !prompt.trim() ||
    (mode === "character_image" && !characterId) ||
    !presetValidationResult.ok ||
    !selectedLoraValidationResult.ok;

  return (
    <section className="main-surface image-studio">
      <form className="studio-shell" onSubmit={submit}>
        <div className="surface-header hero studio-prompt-hero">
          <div className="prompt-hero-top">
            <div className="segmented-control" role="tablist" aria-label="Image mode">
              {[
                ["text_to_image", "Text"],
                ["edit_image", "Edit"],
                ["character_image", "With character"],
                ["style_variations", "Variations"],
              ].map(([value, label]) => (
                <button
                  className={mode === value ? "active" : ""}
                  key={value}
                  onClick={() => handleModeChange(value)}
                  type="button"
                >
                  {value === "text_to_image" ? <Icon.Sparkle size={13} /> : null}
                  {label}
                </button>
              ))}
            </div>
            {onOpenPresets ? (
              <button className="hero-link" onClick={onOpenPresets} type="button">
                <Icon.Folder size={14} /> Saved presets
              </button>
            ) : null}
          </div>

          <div className="prompt-input-row">
            <textarea
              aria-label="Prompt"
              className="prompt-input"
              onChange={(event) => setPromptFromUser(event.target.value)}
              onKeyDown={onPromptKeyDown}
              placeholder="Describe your shot — subject, lighting, mood, lens…"
              value={prompt}
            />
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? "Queueing…" : "Generate"}
            </button>
          </div>

          <div className="suggestion-row">
            <span className="suggestion-row-label">Try:</span>
            {suggestions.map((suggestion) => (
              <button
                className="suggestion"
                key={suggestion}
                onClick={() => setPromptFromUser(suggestion)}
                type="button"
              >
                <Icon.Sparkle size={11} />
                {suggestion}
              </button>
            ))}
          </div>
        </div>

        {mode === "edit_image" || mode === "character_image" ? (
          <div className="studio-source-band">
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
                {characterId ? (
                  characterReferences.length ? (
                    <div className="character-reference-picker">
                      <span className="reference-picker-label">Reference identity</span>
                      <div className="reference-thumb-row">
                        {characterReferences.map((reference) => (
                          <button
                            aria-label={`Use ${reference.asset?.displayName ?? reference.assetId} as reference`}
                            aria-pressed={reference.assetId === referenceAssetId}
                            className={reference.assetId === referenceAssetId ? "reference-thumb active" : "reference-thumb"}
                            key={reference.assetId}
                            onClick={() => setReferenceAssetId(reference.assetId)}
                            title={reference.asset?.displayName ?? reference.assetId}
                            type="button"
                          >
                            {reference.asset ? <AssetMedia asset={reference.asset} controls={false} /> : <span>Missing asset</span>}
                          </button>
                        ))}
                      </div>
                      <label className="reference-strength">
                        {identityStructure ? "Identity strength" : "Reference strength"}
                        <input
                          max="1"
                          min="0"
                          onChange={(event) => setIpAdapterScale(Number(event.target.value))}
                          step="0.05"
                          type="range"
                          value={ipAdapterScale}
                        />
                        <span>{ipAdapterScale.toFixed(2)}</span>
                      </label>
                      {identityStructure ? (
                        <label className="reference-strength">
                          {identityStructure.label ?? "Identity structure"}
                          <input
                            max={identityStructure.max ?? 1}
                            min={identityStructure.min ?? 0}
                            onChange={(event) => setControlnetScale(Number(event.target.value))}
                            step={identityStructure.step ?? 0.05}
                            type="range"
                            value={controlnetScale}
                          />
                          <span>{controlnetScale.toFixed(2)}</span>
                        </label>
                      ) : null}
                      <div className="guidance-strip">
                        <strong>Identity from reference</strong>
                        <span>
                          {identityStructure
                            ? "InstantID holds this person's face from the reference while the prompt drives the scene. Identity strength tunes likeness; Identity structure locks face geometry and pose (lower = more pose freedom). Raise Variations and leave the seed blank to explore takes."
                            : "This reference's identity is carried across every variation. Raise Variations and leave the seed blank to explore different takes."}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="guidance-strip">
                      <strong>No approved reference</strong>
                      <span>Approve a reference image for this character in Character Studio to generate identity-preserving variations. Generating now uses the prompt only.</span>
                    </div>
                  )
                ) : (
                  <div className="guidance-strip">
                    <strong>Select a character</strong>
                    <span>Choose a character with an approved reference image to copy its identity across variations.</span>
                  </div>
                )}
              </>
            ) : null}
          </div>
        ) : null}

        <div className="studio-results">
          <section className="review-panel">
            <div className="review-panel-head">
              <h2>Latest batch</h2>
              <span className="kbd-hint">
                <kbd>⌘</kbd>
                <kbd>↵</kbd>
                to generate
              </span>
            </div>
            {localJobs.length ? (
              <div className="local-job-stack">
                {localJobs.map((job) => (
                  <JobProgressCard job={job} key={job.id} label="Image generation" onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                ))}
              </div>
            ) : null}
            {reviewSlots.length ? (
              <div className="review-grid">
                {reviewSlots.map((slot) =>
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
                  ),
                )}
              </div>
            ) : hasReviewContent ? null : (
              <div className="empty-panel">No fresh image batch</div>
            )}
          </section>

          <section className="studio-controls preset-rail">
            <div className="preset-rail-head">
              <h3>Preset</h3>
              <span className="preset-rail-model-tag">{selectedModel?.name ?? "—"}</span>
            </div>

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

            <div className="style-preset-strip">
              <span className="style-preset-label">Style preset</span>
              <div className="preset-chips">
                <button
                  className={!selectedPreset ? "preset-chip active" : "preset-chip"}
                  onClick={() => setSelectedPresetId(noPresetId)}
                  type="button"
                >
                  None
                </button>
                {availablePresets.map((preset) => (
                  <button
                    className={selectedPreset?.id === preset.id ? "preset-chip active" : "preset-chip"}
                    key={preset.id}
                    onClick={() => setSelectedPresetId(preset.id)}
                    type="button"
                  >
                    {preset.name ?? preset.id}
                  </button>
                ))}
              </div>
            </div>

            <div className="control-grid preset-rail-row">
              <label>
                Variations
                <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
              </label>
              <label>
                Aspect
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((option) => (
                    <option key={option} value={option}>{formatResolutionLabel(option)}</option>
                  ))}
                </select>
              </label>
            </div>

            <PresetGuidanceStrip
              selectedPreset={selectedPreset}
              presetPromptParts={presetPromptParts}
              presetLoraDetails={presetLoraDetails}
              noPresetHint="Generation uses only the prompt, model, and visible preset settings."
            />

            <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
              <Icon.ChevDown className={advancedOpen ? "chev-rotate open" : "chev-rotate"} size={14} />
              {advancedOpen ? "Hide advanced" : "Advanced"}
            </button>

            {advancedOpen ? (
              <div className="advanced-panel">
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
                <label>
                  Seed
                  <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
                </label>
                <label className="checkline upscale-toggle">
                  <input
                    checked={upscaleEnabled}
                    onChange={(event) => setUpscaleEnabled(event.target.checked)}
                    type="checkbox"
                  />
                  Upscale
                </label>
                <label>
                  Scale
                  <select disabled={!upscaleEnabled} onChange={(event) => setUpscaleFactor(Number(event.target.value))} value={upscaleFactor}>
                    {(UPSCALE_ENGINES.find((engine) => engine.id === upscaleEngine)?.factors ?? [2, 4]).map((factor) => (
                      <option key={factor} value={factor}>
                        {factor}x
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Engine
                  <select disabled={!upscaleEnabled} onChange={(event) => handleUpscaleEngineChange(event.target.value)} value={upscaleEngine}>
                    {UPSCALE_ENGINES.map((engine) => (
                      <option key={engine.id} value={engine.id}>
                        {engine.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="prompt-field">
                  Negative prompt
                  <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
                </label>
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
                        const userLimitReached = lora.scope !== "builtin" && !checked && userSelectedLoraCount >= 2;
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
              </div>
            ) : null}

            <PresetValidationWarnings presetValidationResult={presetValidationResult} selectedModel={selectedModel} />
            {selectedLoraValidationResult.incompatible.length ? (
              <p className="inline-warning">
                Generate is blocked because these selected LoRAs are incompatible with {selectedModel?.name ?? "the selected model"}: {selectedLoraValidationResult.incompatible.join(", ")}.
              </p>
            ) : null}
          </section>
        </div>
      </form>
    </section>
  );
}
