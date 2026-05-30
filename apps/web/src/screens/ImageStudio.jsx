import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { PoseLibraryPicker } from "../components/PoseLibraryPicker.jsx";
import { RefinePromptControl } from "../components/RefinePromptControl.jsx";
import { usePoseLibrary, useUserPoseLoader } from "../poseLibrary.js";

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
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import {
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  schedulerShiftDefaultFromModel,
  stepsDefaultFromModel,
} from "../samplerOptions.js";

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

export function ImageStudio() {
  const {
    activeProject,
    assets,
    characters,
    createImageJob,
    refinePrompt,
    deleteAsset,
    purgeAsset,
    gpuOptions,
    imageModels,
    latestImageAssets,
    recentImageAssets,
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
  // Recent Assets list (sc-2088). When the new context value is available, use
  // the bounded 20-most-recent list; fall back to the legacy single-generation
  // list for test contexts that haven't migrated. The existing useGenerationStudio
  // selectStackedJobs() machinery collapses a completed job out of the stack as
  // soon as its assets surface here, so the worker card disappearing matches the
  // spec ("when the current worker completes its assets are added to recent
  // assets, the worker disappears").
  const latestAssets = recentImageAssets ?? latestImageAssets;
  const launchRequest = studioLaunch;
  const trackedLocalJobs = imageLocalJobs;
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onLocalJobCreated = (job) => rememberLocalGenerationJob("image", job);
  const onOpenPresets = () => setActiveView("Presets");
  const onOpenQueue = () => setActiveView("Queue");
  const onPreview = setPreviewAsset;
  // Last-used settings for this workspace, restored on mount. The component is keyed
  // by workspace in App.jsx, so this reads the right snapshot per workspace.
  const saved = useMemo(() => loadStudioSettings("image", activeProject?.id ?? null), [activeProject?.id]);
  const [sceneSuggestions] = useState(() => pickSuggestions(4));
  const [characterSuggestions] = useState(() => pickSuggestions(4, CHARACTER_SUGGESTION_POOL));
  const [mode, setMode] = useState(saved.mode ?? "text_to_image");
  const [prompt, setPrompt] = useState(saved.prompt ?? "A cinematic frame of a neon street at midnight");
  // True once the user types or picks a suggestion, so the character-mode default
  // prompt never clobbers their own wording. A restored prompt counts as edited so
  // re-entering character mode doesn't overwrite it.
  const promptEdited = useRef(saved.prompt != null);
  const setPromptFromUser = (value) => {
    promptEdited.current = true;
    setPrompt(value);
  };
  const suggestions = mode === "character_image" ? characterSuggestions : sceneSuggestions;
  const [count, setCount] = useState(saved.count ?? 4);
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  const [model, setModel] = useState(saved.model ?? imageModels[0]?.id ?? "z_image_turbo");
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");
  const [resolution, setResolution] = useState(saved.resolution ?? "1024x1024");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  // Character reference (IP-Adapter / InstantID) — the approved reference image whose
  // identity is carried across variations. `ipAdapterScale` rides in `advanced`; for
  // InstantID, `controlnetScale` (IdentityNet landmark lock) rides there too.
  const [referenceAssetId, setReferenceAssetId] = useState("");
  const [ipAdapterScale, setIpAdapterScale] = useState(saved.ipAdapterScale ?? 0.6);
  const [controlnetScale, setControlnetScale] = useState(saved.controlnetScale ?? 0.8);
  // Variation knob for backbones whose CFG is decoupled from IP-Adapter:
  // FLUX (true_cfg_scale alongside ipAdapterScale) and Qwen-Image-Edit (true_cfg_scale
  // *replaces* the IP-Adapter slider because Qwen's edit pipeline doesn't use one).
  // Per-model spec rides in ui.variationStrength; resets to the model default on
  // model change like the other tuning knobs (sc-2017).
  const [trueCfgScale, setTrueCfgScale] = useState(saved.trueCfgScale ?? 4.0);
  // InstantID canonical head angle ("" = match the reference's own angle). Rides advanced.viewAngle.
  const [viewAngle, setViewAngle] = useState(saved.viewAngle ?? "");
  // Pose library: selected pose ids. When non-empty, the job carries advanced.poses
  // (one image per pose) instead of the normal variations count. Transient (not saved).
  const [selectedPoseIds, setSelectedPoseIds] = useState([]);
  // Configurable sampler / scheduler (epic 1753). Restored from per-workspace
  // settings; reset to the selected model's manifest defaults whenever the
  // model changes and the saved value is no longer offered by limits.
  const [sampler, setSampler] = useState(saved.sampler ?? "default");
  const [scheduler, setScheduler] = useState(saved.scheduler ?? "default");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? 3.0);
  // Steps / guidance: previously worker-only knobs surfaced via this same
  // advanced panel. "" represents "use the model default" so the user can
  // clear the override.
  const [stepsOverride, setStepsOverride] = useState(saved.steps ?? "");
  const [guidanceOverride, setGuidanceOverride] = useState(saved.guidanceScale ?? "");
  const [faceRestore, setFaceRestore] = useState(false);
  // User-created poses (reserved global project) join the built-in library in both
  // the picker and the id→keypoints resolver below, so saved poses can generate.
  const loadUserPoses = useUserPoseLoader();
  const { byId: poseById } = usePoseLibrary({ loadUserPoses });
  const [upscaleEnabled, setUpscaleEnabled] = useState(saved.upscaleEnabled ?? false);
  const [upscaleFactor, setUpscaleFactor] = useState(saved.upscaleFactor ?? 2);
  const [upscaleEngine, setUpscaleEngine] = useState(saved.upscaleEngine ?? "real-esrgan");
  const [selectedLoraIds, setSelectedLoraIds] = useState(saved.selectedLoraIds ?? []);
  const [loraWeights, setLoraWeights] = useState(saved.loraWeights ?? {});
  const [showIncompatibleLoras, setShowIncompatibleLoras] = useState(saved.showIncompatibleLoras ?? false);
  const [submitting, setSubmitting] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
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
    // Preselect the family-matched edit model resolved at launch time (App.jsx). It's
    // edit-capable by construction, so the availableModels snap-to-first effect leaves
    // it in place; when absent the snap falls back to the default edit model.
    if (launchRequest.model) {
      setModel(launchRequest.model);
    }
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
  // Prompt guide for the selected model; fall back to the generic image guide
  // when a model declares none, so the button is always useful (sc-1817).
  const promptGuide = selectedModel?.ui?.promptGuide ?? {
    title: "Image Prompt Guide",
    path: "/prompt-guides/generic-image.md",
  };
  // Reference-tuning hints declared by the model (ui.*). InstantID raises the
  // reference-strength default and exposes a second "Identity structure" slider
  // (controlnetConditioningScale); models without these keys (e.g. Kolors) keep the
  // single reference-strength slider at the global default.
  const identityStructure = selectedModel?.ui?.identityStructure;
  // Canonical head angles the model can render from a frontal reference (InstantID).
  const viewAngles = Array.isArray(selectedModel?.ui?.viewAngles) ? selectedModel.ui.viewAngles : null;
  // Whether the model supports the OpenPose pose library (InstantID).
  const poseLibrary = Boolean(selectedModel?.ui?.poseLibrary);
  // Variation slider spec (FLUX / Qwen). When declared, the model exposes a
  // trueCfgScale knob alongside (FLUX) or instead of (Qwen, via hideReferenceStrength)
  // the IP-Adapter reference-strength slider (sc-2017).
  const variationStrength = selectedModel?.ui?.variationStrength;
  const hideReferenceStrength = Boolean(selectedModel?.ui?.hideReferenceStrength);
  // Reset the reference tuning to the selected model's declared defaults whenever the
  // model changes, so InstantID starts at its tuned 0.8/0.8 and Kolors at 0.6, and the
  // view angle never carries over to a model that doesn't support it. Skip the mount
  // run when restoring a snapshot so the user's saved tuning survives.
  const skipReferenceTuningReset = useRef(saved.ipAdapterScale != null);
  useEffect(() => {
    if (skipReferenceTuningReset.current) {
      skipReferenceTuningReset.current = false;
      return;
    }
    const ui = imageModels.find((item) => item.id === model)?.ui ?? {};
    setIpAdapterScale(typeof ui.referenceStrengthDefault === "number" ? ui.referenceStrengthDefault : 0.6);
    setControlnetScale(typeof ui.identityStructure?.default === "number" ? ui.identityStructure.default : 0.8);
    setTrueCfgScale(typeof ui.variationStrength?.default === "number" ? ui.variationStrength.default : 4.0);
    setViewAngle("");
    setSelectedPoseIds([]);
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
  // Sampler / scheduler menus declared by the model. The advanced panel hides
  // the dropdowns when the menu has fewer than 2 options (epic 1753 §7.4).
  const samplerOptions = useMemo(() => samplerOptionsFromModel(selectedModel), [selectedModel]);
  const schedulerOptions = useMemo(() => schedulerOptionsFromModel(selectedModel), [selectedModel]);
  const showSamplerPicker = samplerOptions.length > 1;
  const showSchedulerPicker = schedulerOptions.length > 1;
  // Snap the sampler / scheduler back to the model's declared default when the
  // current value is no longer in the menu (e.g. user switched to a sealed
  // model whose only option is "default"). Mirrors the resolution-snap effect.
  useEffect(() => {
    if (samplerOptions.includes(sampler)) {
      return;
    }
    const preferred = samplerOptions.includes(samplerDefaultFromModel(selectedModel))
      ? samplerDefaultFromModel(selectedModel)
      : samplerOptions[0];
    setSampler(preferred);
  }, [samplerOptions, sampler, selectedModel]);
  useEffect(() => {
    if (schedulerOptions.includes(scheduler)) {
      return;
    }
    const preferred = schedulerOptions.includes(schedulerDefaultFromModel(selectedModel))
      ? schedulerDefaultFromModel(selectedModel)
      : schedulerOptions[0];
    setScheduler(preferred);
  }, [schedulerOptions, scheduler, selectedModel]);
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
    selectedPresetId,
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
    initialPresetId: saved.selectedPresetId ?? null,
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

  // When restoring a snapshot, the saved count/resolution/negativePrompt already
  // reflect the user's last state — skip the one preset-default pass that fires as the
  // restored preset resolves so it doesn't overwrite them. "None" applies no defaults,
  // so no guard is needed there.
  const skipPresetDefaultsOnHydrate = useRef(
    Object.keys(saved).length > 0 && saved.selectedPresetId !== noPresetId,
  );
  useEffect(() => {
    if (skipPresetDefaultsOnHydrate.current && selectedPreset) {
      skipPresetDefaultsOnHydrate.current = false;
      return;
    }
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

  useStudioSettingsWriter("image", activeProject?.id ?? null, {
    mode,
    prompt,
    count,
    advancedOpen,
    model,
    seed,
    negativePrompt,
    resolution,
    ipAdapterScale,
    controlnetScale,
    trueCfgScale,
    viewAngle,
    upscaleEnabled,
    upscaleFactor,
    upscaleEngine,
    selectedLoraIds,
    loraWeights,
    showIncompatibleLoras,
    selectedPresetId,
    sampler,
    scheduler,
    schedulerShift,
    steps: stepsOverride,
    guidanceScale: guidanceOverride,
  });

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

  // Each stacked run carries its already-resolved completed assets + the
  // expected count, which the WorkerProgressCard image-grid variant uses to
  // render thumbnails + skeleton cells (sc-2088 — replaces the explicit slot
  // construction the legacy JobProgressCard wrapper needed).
  const localJobGroups = useMemo(
    () =>
      localJobs.map((job) => {
        const completedAssets = jobResultAssets(job, assets);
        const expectedCount = jobExpectedCount(job, completedAssets.length);
        return { job, completedAssets, expectedCount };
      }),
    [assets, localJobs],
  );

  async function submit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    setSubmitting(true);
    try {
      // Pose library: when poses are selected, the job emits one image per pose
      // (advanced.poses) instead of `count` variations.
      const posePayload =
        mode === "character_image" && referenceAssetId && poseLibrary && selectedPoseIds.length
          ? selectedPoseIds.map((id) => poseById[id]).filter(Boolean).map((pose) => ({ id: pose.id, keypoints: pose.keypoints }))
          : [];
      const job = await createImageJob({
        mode,
        prompt,
        negativePrompt,
        model,
        count: posePayload.length ? 1 : count,
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
          // Configurable sampler / scheduler (epic 1753). Worker registry
          // falls back to model-native when given "default", so emitting the
          // values unconditionally is safe — invalid values are ignored.
          ...(sampler && sampler !== "default" ? { sampler } : {}),
          ...(scheduler && scheduler !== "default" ? { scheduler } : {}),
          ...(scheduler === "shift" && Number.isFinite(Number(schedulerShift))
            ? { schedulerShift: Number(schedulerShift) }
            : {}),
          // Step / guidance overrides — empty string means "use the model
          // default", which the worker reads off MODEL_TARGETS.
          ...(stepsOverride !== "" && Number.isFinite(Number(stepsOverride))
            ? { steps: Number(stepsOverride) }
            : {}),
          ...(guidanceOverride !== "" && Number.isFinite(Number(guidanceOverride))
            ? { guidanceScale: Number(guidanceOverride) }
            : {}),
          // IP-Adapter / InstantID reference strength only applies when a character
          // reference is attached AND the model uses the IP-Adapter knob; Qwen's
          // edit pipeline ignores this scalar (hideReferenceStrength gates it out).
          ...(mode === "character_image" && referenceAssetId && !hideReferenceStrength
            ? { ipAdapterScale }
            : {}),
          // Identity structure (controlnetConditioningScale) is InstantID-only — sent
          // only when the model exposes the control and a reference is attached.
          ...(mode === "character_image" && referenceAssetId && identityStructure
            ? { controlnetConditioningScale: controlnetScale }
            : {}),
          // Variation knob (trueCfgScale) — FLUX uses it alongside ipAdapterScale,
          // Qwen uses it as the only variation lever. Sent only when the model
          // declares a variationStrength slider AND a reference is attached.
          ...(mode === "character_image" && referenceAssetId && variationStrength
            ? { trueCfgScale }
            : {}),
          // View angle (InstantID) — only when a specific angle is chosen and no pose is
          // selected (a library pose drives the whole body, superseding the head angle).
          ...(mode === "character_image" && referenceAssetId && viewAngles && viewAngle && !posePayload.length
            ? { viewAngle }
            : {}),
          // Pose library (InstantID) — one image per selected pose; faceRestore toggles
          // the full-body face-restoration pass.
          ...(posePayload.length ? { poses: posePayload, faceRestore } : {}),
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
            <div className="prompt-hero-links">
              <button className="hero-link" onClick={() => setGuideOpen(true)} type="button">
                <Icon.Book size={14} /> Prompt guide
              </button>
              {onOpenPresets ? (
                <button className="hero-link" onClick={onOpenPresets} type="button">
                  <Icon.Folder size={14} /> Saved presets
                </button>
              ) : null}
            </div>
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

          <RefinePromptControl
            guidePath={promptGuide.path}
            modelId={model}
            onApply={setPromptFromUser}
            prompt={prompt}
            refinePrompt={refinePrompt}
            workflow="image"
          />

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
                      {hideReferenceStrength ? null : (
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
                      )}
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
                      {variationStrength ? (
                        <label className="reference-strength">
                          {variationStrength.label ?? "Variation"}
                          <input
                            max={variationStrength.max ?? 10}
                            min={variationStrength.min ?? 1}
                            onChange={(event) => setTrueCfgScale(Number(event.target.value))}
                            step={variationStrength.step ?? 0.5}
                            type="range"
                            value={trueCfgScale}
                          />
                          <span>{trueCfgScale.toFixed(2)}</span>
                        </label>
                      ) : null}
                      {viewAngles ? (
                        <label className="reference-strength">
                          View angle
                          <select onChange={(event) => setViewAngle(event.target.value)} value={viewAngle}>
                            <option value="">Match reference</option>
                            {viewAngles.map((angle) => (
                              <option key={angle.id} value={angle.id}>
                                {angle.label}
                              </option>
                            ))}
                          </select>
                        </label>
                      ) : null}
                      {poseLibrary ? (
                        <details className="pose-library-details">
                          <summary>
                            Pose library{selectedPoseIds.length ? ` · ${selectedPoseIds.length} selected` : ""}
                          </summary>
                          <PoseLibraryPicker
                            loadUserPoses={loadUserPoses}
                            onClear={() => setSelectedPoseIds([])}
                            onToggle={(id) =>
                              setSelectedPoseIds((ids) =>
                                ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id],
                              )
                            }
                            selectedIds={selectedPoseIds}
                          />
                          <label className="checkline">
                            <input checked={faceRestore} onChange={(event) => setFaceRestore(event.target.checked)} type="checkbox" />
                            Restore face (sharper identity; off keeps the raw render)
                          </label>
                          <p className="muted">Selecting poses generates one image per pose (overrides Variations).</p>
                        </details>
                      ) : null}
                      <div className="guidance-strip">
                        <strong>Identity from reference</strong>
                        <span>
                          {identityStructure
                            ? "InstantID holds this person's face from the reference while the prompt drives the scene. Identity strength tunes likeness; Identity structure locks face geometry. Set a View angle to rotate the head (profiles, up/down, diagonals) with identity preserved. Raise Variations and leave the seed blank to explore takes."
                            : variationStrength && hideReferenceStrength
                            ? "Qwen's dual-control architecture (semantic + appearance) carries this reference's subject across new scenes and poses. Variation steers prompt-vs-reference balance: higher = more prompt-driven, lower = closer to the reference. Raise Variations and leave the seed blank to explore takes."
                            : variationStrength
                            ? "This reference's identity is carried across every variation. Reference strength tunes how strongly the reference conditions the result; Variation steers prompt adherence (raise for more variety, lower for closer to the reference). Raise Variations and leave the seed blank to explore takes."
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
            {localJobGroups.length ? (
              <div className="worker-progress-card-stack local-job-stack">
                {localJobGroups.map(({ job, completedAssets, expectedCount }) => (
                  <WorkerProgressCard
                    key={job.id}
                    job={job}
                    thumbnailsVariant="image-grid"
                    thumbnailAssets={completedAssets}
                    expectedThumbnailCount={expectedCount}
                    onThumbnailClick={onPreview}
                    onCancel={onCancelJob}
                    onOpenQueue={onOpenQueue}
                  />
                ))}
              </div>
            ) : null}
            {latestAssets.length ? (
              <div className="recent-assets">
                {localJobGroups.length ? <h3 className="recent-assets__title">Recent Assets</h3> : null}
                <div className="review-grid">
                  {latestAssets.map((asset) => (
                    <AssetCard
                      asset={asset}
                      deleteAsset={deleteAsset}
                      key={asset.id}
                      onPreview={onPreview}
                      purgeAsset={purgeAsset}
                      updateAssetStatus={updateAssetStatus}
                    />
                  ))}
                </div>
              </div>
            ) : localJobGroups.length ? null : (
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
                {showSamplerPicker ? (
                  <label>
                    Sampler
                    <select onChange={(event) => setSampler(event.target.value)} value={sampler}>
                      {samplerOptions.map((key) => (
                        <option key={key} value={key}>
                          {SAMPLER_LABELS[key] ?? key}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {showSchedulerPicker ? (
                  <label>
                    Scheduler
                    <select onChange={(event) => setScheduler(event.target.value)} value={scheduler}>
                      {schedulerOptions.map((key) => (
                        <option key={key} value={key}>
                          {SCHEDULER_LABELS[key] ?? key}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {scheduler === "shift" ? (
                  <label>
                    Schedule shift
                    <input
                      max="10"
                      min="0.1"
                      onChange={(event) => setSchedulerShift(Number(event.target.value))}
                      step="0.1"
                      type="number"
                      value={schedulerShift}
                    />
                  </label>
                ) : null}
                <label>
                  Steps
                  <input
                    min="1"
                    max="80"
                    onChange={(event) => setStepsOverride(event.target.value)}
                    placeholder={String(stepsDefaultFromModel(selectedModel) ?? "")}
                    type="number"
                    value={stepsOverride}
                  />
                </label>
                <label>
                  Guidance
                  <input
                    min="0"
                    max="30"
                    onChange={(event) => setGuidanceOverride(event.target.value)}
                    placeholder={(() => {
                      const value = guidanceDefaultFromModel(selectedModel);
                      return value == null ? "" : String(value);
                    })()}
                    step="0.1"
                    type="number"
                    value={guidanceOverride}
                  />
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
      {guideOpen ? (
        <PromptGuideModal guide={promptGuide} modelName={selectedModel?.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </section>
  );
}
