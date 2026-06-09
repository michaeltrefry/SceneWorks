import React, { useEffect, useMemo, useRef, useState } from "react";
import { pickClosestResolution } from "../resolutionMatch.js";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { PromptGuideModal } from "../components/PromptGuideModal.jsx";
import { RefinePromptControl } from "../components/RefinePromptControl.jsx";

const MOTIONS = [
  "static",
  "slow push-in",
  "pull out",
  "pan left",
  "pan right",
  "tilt up",
  "tilt down",
  "handheld",
];

function formatGpuLabel(requestedGpu) {
  if (!requestedGpu || requestedGpu === "auto") {
    return "auto GPU";
  }
  return `GPU ${requestedGpu}`;
}

function estimateRenderSeconds(durationSeconds, quality) {
  // Rough heuristic: every clip second ~3s on Balanced, ±50% for Draft/Final.
  const base = Math.max(1, Number(durationSeconds) || 6) * 3;
  if (quality === "fast") return Math.round(base * 0.5);
  if (quality === "best") return Math.round(base * 1.5);
  return Math.round(base);
}

function formatPlaybackTime(seconds) {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  return `${minutes}:${String(safeSeconds % 60).padStart(2, "0")}`;
}

// Resolve a video job's result assets against the live catalog so the
// WorkerProgressCard video-player variant can play the finished clip (sc-2089).
// Mirrors the image-side `jobResultAssets` helper in ImageStudio.jsx.
function jobVideoResultAssets(job, assets) {
  const catalogById = new Map(assets.map((asset) => [asset.id, asset]));
  const resultAssets = (job.result?.assets ?? []).filter((asset) => asset?.type === "video");
  const resultById = new Map(resultAssets.map((asset) => [asset.id, catalogById.get(asset.id) ?? asset]));
  const assetIds = job.result?.assetIds ?? [];
  if (assetIds.length) {
    return assetIds.map((id) => resultById.get(id) ?? catalogById.get(id)).filter((asset) => asset?.type === "video");
  }
  if (resultAssets.length) {
    return resultAssets.map((asset) => catalogById.get(asset.id) ?? asset);
  }
  if (job.result?.generationSetId) {
    return assets.filter((asset) => asset.type === "video" && asset.generationSetId === job.result.generationSetId);
  }
  return [];
}
import {
  applyPresetDefault,
  buildStudioPresetPayload,
  clearPresetDefault,
  finiteNumberOrUndefined,
  loraLooksLikeIcLora,
  loraMatchesModel,
  loraWeight,
  noPresetId,
  presetNameTaken,
  rememberPresetDefault,
  serializeLora,
  slugifyPresetId,
} from "../presetUtils.js";
import {
  onPromptKeyDown,
  PresetGuidanceStrip,
  PresetValidationWarnings,
  useGenerationStudio,
} from "./generationStudio.jsx";
import { ReplacePersonPanel, findReplacementModel } from "./ReplacePersonPanel.jsx";
import { useAppContext } from "../context/AppContext.js";
import {
  DEFAULT_MAC_CAPABILITIES,
  macAvailableModels,
  macBlockedModels,
  macVideoModeBlock,
} from "../macGating.js";
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { qualityChoices } from "../jobTypes.js";
import {
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  stepsDefaultFromModel,
} from "../samplerOptions.js";

const ltxVideoModelId = "ltx_2_3";
const ltxIcLoraRequiredModes = new Set(["extend_clip", "video_bridge"]);

// Video sub-modes that map onto a recipe workflow. extend_clip / replace_person
// aren't recipe workflows, so "Save as Preset" is gated to these.
const VIDEO_PRESET_MODES = ["image_to_video", "text_to_video", "first_last_frame"];

export function VideoStudio() {
  const {
    activeProject,
    assets,
    characters,
    createPersonDetectionJob,
    createPersonTrackJob,
    createVideoJob,
    createPreset,
    refinePrompt,
    deleteAsset,
    purgeAsset,
    gpuOptions,
    latestVideoAssets,
    recentVideoAssets,
    studioLaunch,
    loras = [],
    jobs = [],
    videoLocalJobs = [],
    jobAction,
    rememberLocalGenerationJob,
    setActiveView,
    setSelectedAssetId,
    setPreviewAsset,
    personTracks = [],
    personReadiness = {},
    presets = [],
    requestedGpu,
    saveTrackCorrections,
    selectedAsset,
    setRequestedGpu,
    updateAssetStatus,
    videoModels,
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  // Recent Assets (sc-2089) — 20 most recent video assets in the active
  // project. Falls back to the legacy single-generation list for test
  // contexts that haven't migrated.
  const latestAssets = recentVideoAssets ?? latestVideoAssets;
  const launchRequest = studioLaunch;
  const trackedLocalJobs = videoLocalJobs;
  const onCancelJob = (job) => jobAction(job, "cancel");
  const onLocalJobCreated = (job) => rememberLocalGenerationJob("video", job);
  const onOpenPresets = () => setActiveView("Presets");
  const onOpenQueue = () => setActiveView("Queue");
  const onPreview = setPreviewAsset;
  const onSendToEditor = (asset) => {
    if (asset?.id) {
      setSelectedAssetId(asset.id);
    }
    setActiveView("Editor");
  };
  // Last-used settings for this workspace, restored on mount. The component is keyed
  // by workspace in App.jsx, so this reads the right snapshot per workspace.
  const saved = useMemo(() => loadStudioSettings("video", activeProject?.id ?? null), [activeProject?.id]);
  const [motion, setMotion] = useState(saved.motion ?? "slow push-in");
  const imageAssets = assets.filter((asset) => asset.type === "image" || asset.type === "frame");
  const videoAssets = assets.filter((asset) => asset.type === "video");
  const [mode, setMode] = useState(saved.mode ?? "image_to_video");
  const [prompt, setPrompt] = useState(saved.prompt ?? "Camera slowly pushes in while the scene comes alive");
  const [quality, setQuality] = useState(saved.quality ?? "balanced");
  const [ltxPipeline, setLtxPipeline] = useState(saved.ltxPipeline ?? "auto");
  const [distilledVariant, setDistilledVariant] = useState(saved.distilledVariant ?? "1.1");
  const [precision, setPrecision] = useState(saved.precision ?? "fp8");
  const [quantization, setQuantization] = useState(saved.quantization ?? "auto");
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  const [selectedLoraIds, setSelectedLoraIds] = useState(saved.selectedLoraIds ?? []);
  const [loraWeights, setLoraWeights] = useState(saved.loraWeights ?? {});
  const [showIncompatibleLoras, setShowIncompatibleLoras] = useState(saved.showIncompatibleLoras ?? false);
  const [model, setModel] = useState(saved.model ?? videoModels[0]?.id ?? ltxVideoModelId);
  const [guideOpen, setGuideOpen] = useState(false);
  // Mac UI gating (sc-3486): hide torch-only video models (e.g. SVD) and snap off one if selected.
  const macVideoModels = useMemo(
    () => macAvailableModels(videoModels, macCapabilities),
    [videoModels, macCapabilities],
  );
  const macHiddenVideoModels = useMemo(
    () => macBlockedModels(videoModels, macCapabilities),
    [videoModels, macCapabilities],
  );
  useEffect(() => {
    if (macVideoModels.length && !macVideoModels.some((item) => item.id === model)) {
      setModel(macVideoModels[0].id);
    }
  }, [macVideoModels, model]);
  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];
  // Prompt guide for the selected model; fall back to the generic video guide
  // when a model declares none, so the button is always useful (sc-1817).
  const promptGuide = selectedModel?.ui?.promptGuide ?? {
    title: "Video Prompt Guide",
    path: "/prompt-guides/generic-video.md",
  };
  const [duration, setDuration] = useState(saved.duration ?? selectedModel?.defaults?.duration ?? 6);
  const [resolution, setResolution] = useState(saved.resolution ?? selectedModel?.defaults?.resolution ?? "768x512");
  const [fps, setFps] = useState(saved.fps ?? selectedModel?.defaults?.fps ?? 25);
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");
  // Configurable sampler / scheduler (epic 1753). The Wan diffusers (torch)
  // adapter applies these; MLX-backed video paths advertise default-only via
  // mlx.limits and the picker hides itself there.
  const [sampler, setSampler] = useState(saved.sampler ?? "default");
  const [scheduler, setScheduler] = useState(saved.scheduler ?? "default");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? 3.0);
  const [stepsOverride, setStepsOverride] = useState(saved.steps ?? "");
  const [guidanceOverride, setGuidanceOverride] = useState(saved.guidanceScale ?? "");
  // LTX-2.3 native guidance knobs (epic 1753 sc-1769). The native ltx-core
  // path has no diffusers scheduler to swap — these three values (cfg + STG +
  // rescale) drive its sealed MultiModalGuiderParams instead.
  const [ltxVideoCfg, setLtxVideoCfg] = useState(saved.videoCfgGuidanceScale ?? "");
  const [ltxVideoStg, setLtxVideoStg] = useState(saved.videoStgGuidanceScale ?? "");
  const [ltxVideoRescale, setLtxVideoRescale] = useState(saved.videoRescaleScale ?? "");
  // Clip-conditioning strengths for the LTX IC-LoRA extend/bridge paths (sc-3522,
  // sc-3755). The worker reads these from `advanced` (default 1.0 when absent):
  // the source/left clip uses videoConditioningStrength, the bridge right clip
  // uses bridgeRightVideoConditioningStrength.
  const [videoConditioningStrength, setVideoConditioningStrength] = useState(saved.videoConditioningStrength ?? "");
  const [bridgeRightVideoConditioningStrength, setBridgeRightVideoConditioningStrength] = useState(
    saved.bridgeRightVideoConditioningStrength ?? "",
  );
  const [sourceAssetId, setSourceAssetId] = useState(["image", "frame"].includes(selectedAsset?.type) ? selectedAsset.id : "");
  const [lastFrameAssetId, setLastFrameAssetId] = useState("");
  const [sourceClipAssetId, setSourceClipAssetId] = useState(selectedAsset?.type === "video" ? selectedAsset.id : "");
  const [bridgeRightClipAssetId, setBridgeRightClipAssetId] = useState("");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  const [personTrackId, setPersonTrackId] = useState("");
  const [replacementMode, setReplacementMode] = useState("face_only");
  const [selectedDetectionId, setSelectedDetectionId] = useState("");
  const [trackName, setTrackName] = useState("Selected person");
  const [comparisonMode, setComparisonMode] = useState("side_by_side");
  const [abSide, setAbSide] = useState("replacement");
  const [submitting, setSubmitting] = useState(false);
  const previewVideoRef = useRef(null);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewTime, setPreviewTime] = useState(0);
  const [previewDuration, setPreviewDuration] = useState(0);
  // "Save as Preset" sidebar control — snapshots the current config into the
  // workspace preset library. Defaults to project scope, falling back to global
  // when no project is open (project-scoped presets require a project).
  const [presetName, setPresetName] = useState("");
  const [presetScope, setPresetScope] = useState(activeProject ? "project" : "global");
  const [savingPreset, setSavingPreset] = useState(false);
  const [presetSaveMessage, setPresetSaveMessage] = useState({ tone: "neutral", text: "" });
  const presetDefaultSnapshots = useRef({});
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  // GGUF quantization variants the torch adapter can load (sc-1982). Declared in
  // the model manifest's `quantization.variants`; "auto" defers to the worker's
  // per-platform default (Q8_0 on MPS, Q4_K_M on CUDA).
  const quantVariants = Object.entries(selectedModel?.quantization?.variants ?? {});
  const supportsQuantization = quantVariants.length > 0;
  const implementedMode = ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"].includes(mode);
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
    models: videoModels,
    model,
    setModel,
    fallbackModelId: ltxVideoModelId,
    characters,
    characterId,
    setCharacterId,
    setCharacterLookId,
    assets,
    latestAssets,
    trackedLocalJobs,
    initialPresetId: saved.selectedPresetId ?? null,
  });
  // Sampler / scheduler menus declared by the model. Video Wan torch
  // declares the full menu; sealed paths (LTX native, MLX) drop to
  // default-only and the picker hides.
  const samplerOptions = useMemo(() => samplerOptionsFromModel(selectedModel), [selectedModel]);
  const schedulerOptions = useMemo(() => schedulerOptionsFromModel(selectedModel), [selectedModel]);
  const showSamplerPicker = samplerOptions.length > 1;
  const showSchedulerPicker = schedulerOptions.length > 1;
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
  const requiresLtxIcLora = selectedModel?.id === ltxVideoModelId && ltxIcLoraRequiredModes.has(mode);
  const hasLtxIcLora = presetLoraDetails.some((lora) => !lora.missing && loraLooksLikeIcLora(lora));
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

  useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
  }, [compatibleLoraKey]);

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

  // Snapshot the current working config into a named recipe preset in the
  // workspace library. Captures the literal prompt + every visible knob + the
  // selected LoRAs with their weights; the seed is intentionally left out so the
  // preset stays reusable. The backend additionally enforces id uniqueness and
  // model/workflow + LoRA compatibility, surfaced here via err.message.
  async function handleSaveAsPreset() {
    const trimmed = presetName.trim();
    if (!trimmed) {
      setPresetSaveMessage({ tone: "error", text: "Name the preset before saving." });
      return;
    }
    if (!slugifyPresetId(trimmed)) {
      setPresetSaveMessage({ tone: "error", text: "Use letters or numbers in the preset name." });
      return;
    }
    if (!VIDEO_PRESET_MODES.includes(mode)) {
      setPresetSaveMessage({ tone: "error", text: "Switch to Image, Text, or First/Last mode to save a preset." });
      return;
    }
    if (presetScope === "project" && !activeProject) {
      setPresetSaveMessage({ tone: "error", text: "Open a project first, or save to all projects." });
      return;
    }
    if (presetNameTaken(trimmed, presets)) {
      setPresetSaveMessage({ tone: "error", text: `"${trimmed}" already exists — pick a unique name.` });
      return;
    }
    const payload = buildStudioPresetPayload({
      name: trimmed,
      scope: presetScope,
      mode,
      model,
      loras: selectedLoras.map((lora) => ({ id: lora.id, weight: effectiveLoraWeight(lora) })),
      defaults: {
        prompt,
        negativePrompt,
        resolution,
        duration,
        fps,
        quality,
        mode,
        guidanceScale: finiteNumberOrUndefined(guidanceOverride),
        steps: finiteNumberOrUndefined(stepsOverride),
        sampler,
        scheduler,
        schedulerShift,
        precision,
        quantization,
        ltxPipeline,
        distilledVariant,
        motion,
        videoCfgGuidanceScale: finiteNumberOrUndefined(ltxVideoCfg),
        videoStgGuidanceScale: finiteNumberOrUndefined(ltxVideoStg),
        videoRescaleScale: finiteNumberOrUndefined(ltxVideoRescale),
      },
    });
    setSavingPreset(true);
    setPresetSaveMessage({ tone: "neutral", text: "" });
    try {
      const created = await createPreset(payload);
      setSelectedPresetId(created?.id ?? payload.id);
      setPresetName("");
      setPresetSaveMessage({
        tone: "success",
        text: `Saved "${trimmed}" to ${presetScope === "project" ? "this project" : "all projects"}.`,
      });
    } catch (err) {
      setPresetSaveMessage({ tone: "error", text: err.message });
    } finally {
      setSavingPreset(false);
    }
  }

  function setLoraWeight(id, value) {
    setLoraWeights((current) => ({ ...current, [id]: value }));
  }

  useEffect(() => {
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
  }, [selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (launchRequest?.view !== "Video") {
      return;
    }
    if (launchRequest.characterId) {
      setMode(launchRequest.mode ?? "text_to_video");
      setCharacterId(launchRequest.characterId);
      setCharacterLookId(launchRequest.lookId ?? "");
      return;
    }
    if (launchRequest.assetId !== selectedAsset?.id) {
      return;
    }
    setMode(launchRequest.mode);
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
  }, [launchRequest?.id, selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    setDuration((current) => {
      const options = selectedModel.limits?.durations ?? [4, 6, 8, 10];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.duration ?? options[0];
    });
    setResolution((current) => {
      const options = selectedModel.limits?.resolutions ?? ["768x512"];
      return options.includes(current) ? current : selectedModel.defaults?.resolution ?? options[0];
    });
    setFps((current) => {
      const options = selectedModel.limits?.fps ?? [24, 25, 30];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.fps ?? options[0];
    });
  }, [selectedModel?.id]);

  // I2V: when the user picks a source image (or first/last frame) after mount,
  // snap resolution to whichever option in the model's list best matches the
  // image's aspect ratio. The ref tracks the last-seen id so polling-driven
  // assets refreshes don't re-fire, and so the saved snapshot's resolution is
  // preserved when the asset id is just being restored on mount.
  const i2vSourceAssetId = sourceAssetId || lastFrameAssetId;
  const lastI2vAssetIdRef = useRef(i2vSourceAssetId);
  useEffect(() => {
    if (i2vSourceAssetId === lastI2vAssetIdRef.current) {
      return;
    }
    lastI2vAssetIdRef.current = i2vSourceAssetId;
    if (!i2vSourceAssetId) return;
    if (!["image_to_video", "first_last_frame"].includes(mode)) return;
    const asset = assets.find((item) => item.id === i2vSourceAssetId);
    const width = asset?.file?.width;
    const height = asset?.file?.height;
    if (!width || !height) return;
    const match = pickClosestResolution(width, height, selectedModel?.limits?.resolutions);
    if (match) setResolution(match);
  }, [i2vSourceAssetId, mode, selectedModel?.id, assets]);

  useEffect(() => {
    if (mode !== "replace_person" || supportsMode) {
      return;
    }
    const replacementModel = findReplacementModel(videoModels);
    if (replacementModel) {
      setModel(replacementModel.id);
    }
  }, [mode, supportsMode, videoModels]);

  // When restoring a snapshot, the saved length/fps/quality/resolution/negativePrompt
  // already reflect the user's last state — skip the one preset-default pass that fires
  // as the restored preset resolves so it doesn't overwrite them. "None" applies no
  // defaults, so no guard is needed there.
  const skipPresetDefaultsOnHydrate = useRef(
    Object.keys(saved).length > 0 && saved.selectedPresetId !== noPresetId,
  );
  // [defaults key, setter] pairs restored through the remember/clear snapshot
  // machinery, so switching to None (or another preset) puts the user's prior
  // value back. Only keys the preset carries are applied, so older presets keep
  // working and full-snapshot presets restore the prompt, cfg, sampler, and the
  // native LTX guidance knobs. The model is intentionally absent — presets never
  // switch the model.
  const presetDefaultFields = [
    ["prompt", setPrompt],
    ["negativePrompt", setNegativePrompt],
    ["resolution", setResolution],
    ["duration", setDuration],
    ["fps", setFps],
    ["quality", setQuality],
    ["guidanceScale", setGuidanceOverride],
    ["steps", setStepsOverride],
    ["sampler", setSampler],
    ["scheduler", setScheduler],
    ["schedulerShift", setSchedulerShift],
    ["precision", setPrecision],
    ["quantization", setQuantization],
    ["ltxPipeline", setLtxPipeline],
    ["distilledVariant", setDistilledVariant],
    ["motion", setMotion],
    ["videoCfgGuidanceScale", setLtxVideoCfg],
    ["videoStgGuidanceScale", setLtxVideoStg],
    ["videoRescaleScale", setLtxVideoRescale],
  ];
  useEffect(() => {
    if (skipPresetDefaultsOnHydrate.current && selectedPreset) {
      skipPresetDefaultsOnHydrate.current = false;
      return;
    }
    if (!selectedPreset) {
      for (const [key, setter] of presetDefaultFields) {
        clearPresetDefault(setter, presetDefaultSnapshots, key);
      }
      return;
    }
    const defaults = selectedPreset.defaults ?? {};
    for (const [key, setter] of presetDefaultFields) {
      if (Object.prototype.hasOwnProperty.call(defaults, key)) {
        applyPresetDefault(presetDefaultSnapshots, key, setter, defaults[key]);
      }
    }
    // Restore the saved sub-mode ("type") when it's a generatable video workflow.
    if (VIDEO_PRESET_MODES.includes(defaults.mode)) {
      setMode(defaults.mode);
    }
  }, [selectedPreset?.id]);

  useStudioSettingsWriter("video", activeProject?.id ?? null, {
    motion,
    mode,
    prompt,
    quality,
    ltxPipeline,
    distilledVariant,
    precision,
    quantization,
    advancedOpen,
    selectedLoraIds,
    loraWeights,
    showIncompatibleLoras,
    model,
    duration,
    resolution,
    fps,
    seed,
    negativePrompt,
    selectedPresetId,
    sampler,
    scheduler,
    schedulerShift,
    steps: stepsOverride,
    guidanceScale: guidanceOverride,
    videoCfgGuidanceScale: ltxVideoCfg,
    videoStgGuidanceScale: ltxVideoStg,
    videoRescaleScale: ltxVideoRescale,
    videoConditioningStrength,
    bridgeRightVideoConditioningStrength,
  });

  useEffect(() => {
    if (mode !== "replace_person") {
      return;
    }
    const firstMatchingTrack = personTracks.find((track) => track.sourceAssetId === sourceClipAssetId);
    if (firstMatchingTrack && !personTracks.some((track) => track.id === personTrackId)) {
      setPersonTrackId(firstMatchingTrack.id);
    }
  }, [mode, personTracks, personTrackId, sourceClipAssetId]);

  const modeOptions = [
    ["image_to_video", "Image → Video"],
    ["text_to_video", "Text → Video"],
    ["first_last_frame", "First → Last"],
    ["extend_clip", "Extend"],
    ["video_bridge", "Bridge"],
    ["replace_person", "Replace person"],
  ];
  // Mac UI gating (sc-3486, sc-3773): every video mode is disabled per-model via the selected
  // model's `macSupport.features.videoModes` — FLF on the non-Keyframe Wan MoE engines,
  // replace_person on non-replace models, and the LTX IC-LoRA clip-conditioning modes
  // (extend_clip / video_bridge) on non-LTX models. On LTX, extend/bridge stay enabled because
  // the in-process Rust worker serves them, so the old coarse global flag is gone.
  const macVideoModeBlockFor = (value) => macVideoModeBlock(selectedModel, macCapabilities, value);
  const matchingTracks = personTracks.filter((track) => track.sourceAssetId === sourceClipAssetId);
  const latestDetectionJob = jobs
    .filter(
      (job) =>
        job.type === "person_detect" &&
        job.status === "completed" &&
        job.projectId === activeProject?.id &&
        job.payload?.sourceAssetId === sourceClipAssetId,
    )
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
  const detectionResult = latestDetectionJob?.result ?? null;
  const representativeFrame = assets.find((asset) => asset.id === detectionResult?.frameAssetId);
  const selectedDetection = detectionResult?.detections?.find((item) => item.id === selectedDetectionId) ?? detectionResult?.detections?.[0];
  const selectedTrack = personTracks.find((track) => track.id === personTrackId);
  const comparisonAsset = latestAssets.find((asset) => asset.recipe?.mode === "replace_person");
  const comparisonSource = assets.find((asset) => asset.id === comparisonAsset?.lineage?.sourceClipAssetId);
  const hasInputs =
    mode === "text_to_video" ||
    (mode === "image_to_video" && sourceAssetId) ||
    (mode === "first_last_frame" && sourceAssetId && lastFrameAssetId) ||
    (mode === "extend_clip" && sourceClipAssetId) ||
    (mode === "video_bridge" && sourceClipAssetId && bridgeRightClipAssetId) ||
    (mode === "replace_person" && sourceClipAssetId && personTrackId && characterId);
  // Don't let Replace Person queue a job the readiness endpoint says no live
  // worker can run — that would sit unclaimable instead of honoring the gate.
  const replaceReady = mode !== "replace_person" || personReadiness?.replace?.ready !== false;
  // Image-conditioned models (e.g. Stable Video Diffusion) take no text prompt;
  // they animate the source image, so don't gate submission on prompt text.
  const promptless = Boolean(selectedModel?.promptless);
  const canSubmit = Boolean(
    activeProject &&
      (promptless || prompt.trim()) &&
      supportsMode &&
      implementedMode &&
      hasInputs &&
      presetValidationResult.ok &&
      selectedLoraValidationResult.ok &&
      (!requiresLtxIcLora || hasLtxIcLora) &&
      replaceReady,
  );
  const [width, height] = resolution.split("x").map((value) => Number(value));
  const durationOptions = selectedModel?.limits?.durations ?? [4, 6, 8, 10];
  const resolutionOptions = selectedModel?.limits?.resolutions ?? ["768x512", "640x640", "1280x720", "720x1280"];
  const fpsOptions = selectedModel?.limits?.fps ?? [24, 25, 30];
  const durationHint =
    selectedModel?.ui?.durationHint ??
    (selectedModel?.limits?.recommendedMaxDuration ? `Recommended: ${selectedModel.limits.recommendedMaxDuration}s or less.` : "");
  const blockedMessage = !supportsMode
    ? `${selectedModel?.name ?? "Selected model"} does not support this mode.`
    : !implementedMode
      ? "This entry point is reserved for the next runtime slice."
      : !hasInputs
        ? "Required inputs are missing."
        : requiresLtxIcLora && !hasLtxIcLora
          ? "LTX video-conditioned generation needs an installed IC-LoRA preset."
          : !replaceReady
            ? "No live GPU worker can run person replacement yet."
            : "";
  const replacementModeLabels = {
    face_only: "Face Only",
    full_person_keep_outfit: "Full Person, Keep Outfit",
    full_person_replace_outfit: "Full Person, Replace Outfit",
  };

  async function submit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    setSubmitting(true);
    try {
      const job = await createVideoJob({
        mode,
        prompt,
        negativePrompt,
        model,
        duration: Number(duration),
        fps: Number(fps),
        width,
        height,
        quality,
        seed: seed === "" ? null : Number(seed),
        recipePresetId: selectedPreset?.id ?? null,
        characterId: characterId || null,
        characterLookId: characterLookId || null,
        sourceAssetId: ["image_to_video", "first_last_frame"].includes(mode) ? sourceAssetId || null : null,
        lastFrameAssetId: mode === "first_last_frame" ? lastFrameAssetId || null : null,
        sourceClipAssetId: ["extend_clip", "replace_person", "video_bridge"].includes(mode)
          ? sourceClipAssetId || null
          : null,
        bridgeRightClipAssetId: mode === "video_bridge" ? bridgeRightClipAssetId || null : null,
        personTrackId: mode === "replace_person" ? personTrackId || null : null,
        replacementMode: mode === "replace_person" ? replacementMode : "face_only",
        loras: selectedLoras.map((lora) => serializeLora(lora, { weight: effectiveLoraWeight(lora) })),
        advanced: {
          resolution,
          durationHint,
          motion,
          selectedPersonTrack: selectedTrack ?? null,
          replacementModeLabel: replacementModeLabels[replacementMode],
          ...(model === ltxVideoModelId ? { ltxPipeline, distilledVariant, precision } : {}),
          ...(supportsQuantization && quantization !== "auto" ? { quantization } : {}),
          // Configurable sampler / scheduler (epic 1753). Sealed adapters
          // (LTX native, MLX) silently fall back to default; only the Wan
          // diffusers (torch) path actually applies these.
          ...(sampler && sampler !== "default" ? { sampler } : {}),
          ...(scheduler && scheduler !== "default" ? { scheduler } : {}),
          ...(scheduler === "shift" && Number.isFinite(Number(schedulerShift))
            ? { schedulerShift: Number(schedulerShift) }
            : {}),
          ...(stepsOverride !== "" && Number.isFinite(Number(stepsOverride))
            ? { steps: Number(stepsOverride) }
            : {}),
          ...(guidanceOverride !== "" && Number.isFinite(Number(guidanceOverride))
            ? { guidanceScale: Number(guidanceOverride) }
            : {}),
          // LTX native guidance knobs (epic 1753 sc-1769). Only emitted for
          // the LTX adapter — the worker would silently ignore them on other
          // adapters but keeping the payload tight avoids surprise overrides.
          ...(selectedModel?.adapter === "ltx_video" && ltxVideoCfg !== "" && Number.isFinite(Number(ltxVideoCfg))
            ? { videoCfgGuidanceScale: Number(ltxVideoCfg) }
            : {}),
          ...(selectedModel?.adapter === "ltx_video" && ltxVideoStg !== "" && Number.isFinite(Number(ltxVideoStg))
            ? { videoStgGuidanceScale: Number(ltxVideoStg) }
            : {}),
          ...(selectedModel?.adapter === "ltx_video" && ltxVideoRescale !== "" && Number.isFinite(Number(ltxVideoRescale))
            ? { videoRescaleScale: Number(ltxVideoRescale) }
            : {}),
          // LTX IC-LoRA clip-conditioning strengths (sc-3522, sc-3755). The worker
          // reads these from `advanced`, defaulting to 1.0 when absent — extend uses
          // the source-clip strength, bridge uses both left and right.
          ...(["extend_clip", "video_bridge"].includes(mode) &&
          videoConditioningStrength !== "" &&
          Number.isFinite(Number(videoConditioningStrength))
            ? { videoConditioningStrength: Number(videoConditioningStrength) }
            : {}),
          ...(mode === "video_bridge" &&
          bridgeRightVideoConditioningStrength !== "" &&
          Number.isFinite(Number(bridgeRightVideoConditioningStrength))
            ? { bridgeRightVideoConditioningStrength: Number(bridgeRightVideoConditioningStrength) }
            : {}),
        },
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  const generateDisabled = submitting || !canSubmit;
  const renderLabel = mode === "replace_person" ? "Replace person" : "Render clip";
  const previewAsset = latestAssets[0] ?? null;
  const estimateSeconds = estimateRenderSeconds(duration, quality);
  const gpuLabel = formatGpuLabel(requestedGpu);
  const previewCanPlay = assetCanRenderAsVideo(previewAsset);
  const previewTotalSeconds = previewDuration || Number(previewAsset?.file?.duration) || Number(duration) || 0;
  const previewProgress = previewTotalSeconds ? `${Math.min(100, (previewTime / previewTotalSeconds) * 100)}%` : "0%";

  useEffect(() => {
    setPreviewPlaying(false);
    setPreviewTime(0);
    setPreviewDuration(0);
  }, [previewAsset?.id]);

  function togglePreviewPlayback() {
    const video = previewVideoRef.current;
    if (!video || !previewCanPlay) {
      return;
    }
    if (video.paused) {
      video.play().catch(() => setPreviewPlaying(false));
      return;
    }
    video.pause();
  }

  return (
    <section className="main-surface video-studio">
      <form className="studio-shell" onSubmit={submit}>
        <div className="surface-header hero studio-prompt-hero video-prompt-hero">
          <div className="prompt-hero-top">
            <div className="segmented-control mode-control" role="tablist" aria-label="Video mode">
              {modeOptions.map(([value, label]) => {
                const macBlock = macVideoModeBlockFor(value);
                return (
                  <button
                    className={mode === value ? "active" : ""}
                    key={value}
                    onClick={() => setMode(value)}
                    type="button"
                    disabled={Boolean(macBlock)}
                    title={macBlock ? macBlock.text : undefined}
                  >
                    {label}
                  </button>
                );
              })}
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
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={onPromptKeyDown}
              placeholder={
                promptless
                  ? "No prompt needed — this model animates the source image. Pick a first frame below and generate."
                  : "Describe the motion — what moves, where the camera goes, how it feels…"
              }
              value={prompt}
            />
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? "Queueing…" : renderLabel}
            </button>
          </div>

          {promptless ? null : (
            <RefinePromptControl
              guidePath={promptGuide.path}
              modelId={model}
              onApply={setPrompt}
              prompt={prompt}
              refinePrompt={refinePrompt}
              workflow="video"
            />
          )}

          <div className="motion-row">
            <span className="motion-row-label">Motion:</span>
            {MOTIONS.map((option) => (
              <button
                className={motion === option ? "motion-chip active" : "motion-chip"}
                key={option}
                onClick={() => setMotion(option)}
                type="button"
              >
                <span aria-hidden="true" className="motion-arrow">→</span>
                {option}
              </button>
            ))}
          </div>
        </div>

        {mode !== "text_to_video" ? (
          <div className="studio-source-band">
            {mode === "image_to_video" || mode === "first_last_frame" ? (
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select image"
                emptyLabel="No first frame selected"
                label="First frame"
                onChange={setSourceAssetId}
                value={sourceAssetId}
              />
            ) : null}

            {mode === "first_last_frame" ? (
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select image"
                emptyLabel="No last frame selected"
                label="Last frame"
                onChange={setLastFrameAssetId}
                value={lastFrameAssetId}
              />
            ) : null}

            {mode === "extend_clip" ? (
              <AssetPickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                emptyLabel="No source clip selected"
                label="Source clip"
                onChange={setSourceClipAssetId}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "video_bridge" ? (
              <>
                <AssetPickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  emptyLabel="No left clip selected"
                  label="Left clip"
                  onChange={setSourceClipAssetId}
                  value={sourceClipAssetId}
                />
                <AssetPickerField
                  assets={videoAssets}
                  buttonLabel="Select clip"
                  emptyLabel="No right clip selected"
                  label="Right clip"
                  onChange={setBridgeRightClipAssetId}
                  value={bridgeRightClipAssetId}
                />
              </>
            ) : null}

            {mode === "replace_person" ? (
              <ReplacePersonPanel
                createPersonDetectionJob={createPersonDetectionJob}
                createPersonTrackJob={createPersonTrackJob}
                personReadiness={personReadiness}
                detectionResult={detectionResult}
                matchingTracks={matchingTracks}
                personTrackId={personTrackId}
                replacementMode={replacementMode}
                representativeFrame={representativeFrame}
                saveTrackCorrections={saveTrackCorrections}
                selectedDetection={selectedDetection}
                selectedTrack={selectedTrack}
                setPersonTrackId={setPersonTrackId}
                setReplacementMode={setReplacementMode}
                setSelectedDetectionId={setSelectedDetectionId}
                setSourceClipAssetId={setSourceClipAssetId}
                setTrackName={setTrackName}
                sourceClipAssetId={sourceClipAssetId}
                trackName={trackName}
                videoAssets={videoAssets}
              />
            ) : null}
          </div>
        ) : null}

        <div className="video-results">
          <div className="video-main-stack">
            {localJobs.length ? (
              <div className="worker-progress-card-stack local-job-stack">
                {localJobs.map((job) => {
                  const jobAssets = jobVideoResultAssets(job, assets);
                  return (
                    <WorkerProgressCard
                      key={job.id}
                      job={job}
                      thumbnailsVariant="video-player"
                      thumbnailAssets={jobAssets}
                      onThumbnailClick={(asset) => onPreview(asset, jobAssets)}
                      onCancel={onCancelJob}
                      onOpenQueue={onOpenQueue}
                    />
                  );
                })}
              </div>
            ) : null}

            <div className="video-preview-card">
              <div className="video-preview-stage">
                {previewAsset ? (
                  <AssetMedia
                    asset={previewAsset}
                    controls={false}
                    onEnded={() => setPreviewPlaying(false)}
                    onLoadedMetadata={(event) => setPreviewDuration(event.currentTarget.duration || 0)}
                    onPause={() => setPreviewPlaying(false)}
                    onPlay={() => setPreviewPlaying(true)}
                    onTimeUpdate={(event) => setPreviewTime(event.currentTarget.currentTime || 0)}
                    ref={previewVideoRef}
                  />
                ) : (
                  <span className="video-preview-empty">No clip rendered yet — set up the prompt above and hit Render</span>
                )}
              </div>

              <div className="video-playback-bar">
                <button
                  aria-label={previewPlaying ? "Pause preview" : "Play preview"}
                  className="play-btn"
                  disabled={!previewCanPlay}
                  onClick={togglePreviewPlayback}
                  type="button"
                >
                  {previewPlaying ? <Icon.Pause size={14} /> : <Icon.Play size={14} />}
                </button>
                <div aria-hidden="true" className="video-playback-scrub">
                  <span className="video-playback-scrub-fill" style={{ width: previewProgress }} />
                </div>
                <span className="video-playback-time">
                  {formatPlaybackTime(previewTime)} / {formatPlaybackTime(previewTotalSeconds)}
                </span>
                <span className="video-playback-estimate">~{estimateSeconds}s on {gpuLabel}</span>
                <button
                  className="send-editor-btn"
                  disabled={!previewAsset || !onSendToEditor}
                  onClick={() => previewAsset && onSendToEditor?.(previewAsset)}
                  type="button"
                >
                  <Icon.Editor size={14} /> Send to editor
                </button>
              </div>
            </div>

            {videoAssets.length ? (
              <div className="recent-clips-card">
                <div className="recent-clips-head">
                  <h3>Recent clips</h3>
                  <span className="meta">{localJobs.length || latestAssets.length} this session</span>
                </div>
                <div className="recent-clips-strip">
                  {videoAssets.slice(0, 4).map((asset) => (
                    <button className="tray-item" key={asset.id} onClick={() => onPreview(asset, videoAssets.slice(0, 4))} type="button">
                      <AssetMedia asset={asset} />
                      <span>{asset.displayName}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            {comparisonAsset?.recipe?.mode === "replace_person" && comparisonSource ? (
              <div className="comparison-panel">
                <div className="comparison-toolbar">
                  <div className="segmented-control compact-segment" aria-label="Comparison mode">
                    <button className={comparisonMode === "side_by_side" ? "active" : ""} onClick={() => setComparisonMode("side_by_side")} type="button">
                      Side by Side
                    </button>
                    <button className={comparisonMode === "ab" ? "active" : ""} onClick={() => setComparisonMode("ab")} type="button">
                      A/B
                    </button>
                  </div>
                  {comparisonMode === "ab" ? (
                    <div className="segmented-control compact-segment" aria-label="A/B source">
                      <button className={abSide === "original" ? "active" : ""} onClick={() => setAbSide("original")} type="button">
                        A
                      </button>
                      <button className={abSide === "replacement" ? "active" : ""} onClick={() => setAbSide("replacement")} type="button">
                        B
                      </button>
                    </div>
                  ) : null}
                </div>
                {comparisonMode === "side_by_side" ? (
                  <div className="comparison-grid">
                    <div>
                      <p className="eyebrow">Original</p>
                      <AssetMedia asset={comparisonSource} />
                    </div>
                    <div>
                      <p className="eyebrow">Replacement</p>
                      <AssetMedia asset={comparisonAsset} />
                    </div>
                  </div>
                ) : (
                  <div className="comparison-single">
                    <p className="eyebrow">{abSide === "original" ? "A Original" : "B Replacement"}</p>
                    <AssetMedia asset={abSide === "original" ? comparisonSource : comparisonAsset} />
                  </div>
                )}
              </div>
            ) : null}

            {latestAssets.length > 1 ? (
              <div className="review-grid video-review-grid">
                {latestAssets.slice(1).map((asset) => (
                  <AssetCard
                    asset={asset}
                    deleteAsset={deleteAsset}
                    key={asset.id}
                    onPreview={(previewed) => onPreview(previewed, latestAssets.slice(1))}
                    purgeAsset={purgeAsset}
                    updateAssetStatus={updateAssetStatus}
                  />
                ))}
              </div>
            ) : null}

            {blockedMessage ? <p className="inline-warning">{blockedMessage}</p> : null}
            <PresetValidationWarnings presetValidationResult={presetValidationResult} selectedModel={selectedModel} />
            {selectedLoraValidationResult.incompatible.length ? (
              <p className="inline-warning">
                Generate is blocked because these selected LoRAs are incompatible with {selectedModel?.name ?? "the selected model"}: {selectedLoraValidationResult.incompatible.join(", ")}.
              </p>
            ) : null}
          </div>

          <div className="video-rail">
            <aside className="render-rail">
              <div className="preset-rail-head">
                <h3>Render settings</h3>
                <span className="preset-rail-model-tag">{selectedModel?.name ?? "—"}</span>
              </div>

              <label>
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {(macVideoModels.length ? macVideoModels : videoModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              {macHiddenVideoModels.length ? (
                <p className="mac-gating-note">
                  {macHiddenVideoModels.length} model
                  {macHiddenVideoModels.length === 1 ? "" : "s"} unavailable on Mac (Rust/MLX only).
                </p>
              ) : null}

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
                        handleSaveAsPreset();
                      }
                    }}
                    placeholder="Name this setup…"
                    value={presetName}
                  />
                  <button
                    className="save-preset-btn"
                    disabled={savingPreset || !presetName.trim() || !VIDEO_PRESET_MODES.includes(mode)}
                    onClick={handleSaveAsPreset}
                    title={VIDEO_PRESET_MODES.includes(mode) ? undefined : "Presets are available in Image→Video, Text→Video, or First/Last mode."}
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

              <label>
                Quality
                <div className="quality-segment" role="radiogroup" aria-label="Quality">
                  {qualityChoices.map(([value, label]) => (
                    <button
                      aria-checked={quality === value}
                      className={quality === value ? "active" : ""}
                      key={value}
                      onClick={() => setQuality(value)}
                      role="radio"
                      type="button"
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </label>

              <div className="control-grid preset-rail-row">
                <label>
                  Duration
                  <select onChange={(event) => setDuration(Number(event.target.value))} value={duration}>
                    {durationOptions.map((value) => (
                      <option key={value} value={value}>
                        {value}s
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Frames
                  <select onChange={(event) => setFps(Number(event.target.value))} value={fps}>
                    {fpsOptions.map((value) => (
                      <option key={value} value={value}>
                        {value} fps
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " × ")}
                    </option>
                  ))}
                </select>
              </label>

              <PresetGuidanceStrip
                selectedPreset={selectedPreset}
                presetPromptParts={presetPromptParts}
                presetLoraDetails={presetLoraDetails}
                noPresetHint="Generation uses only the prompt, model, and visible render settings."
              />

              {durationHint ? <p className="helper-copy">{durationHint}</p> : null}

              <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
                <Icon.ChevDown className={advancedOpen ? "chev-rotate open" : "chev-rotate"} size={14} />
                {advancedOpen ? "Hide advanced" : "Advanced"}
              </button>

              {advancedOpen ? (
                <div className="advanced-panel">
                  {model === ltxVideoModelId ? (
                    <>
                      <label>
                        LTX pipeline
                        <select onChange={(event) => setLtxPipeline(event.target.value)} value={ltxPipeline}>
                          <option value="auto">Auto (follow quality)</option>
                          <option value="distilled">Distilled (single-stage)</option>
                          <option value="two_stage">Two-stage (dev + upscaler)</option>
                        </select>
                      </label>
                      <label>
                        Distilled variant
                        <select onChange={(event) => setDistilledVariant(event.target.value)} value={distilledVariant}>
                          <option value="1.1">1.1 (newer aesthetic + audio)</option>
                          <option value="1.0">1.0 (original)</option>
                        </select>
                      </label>
                      <label>
                        Precision
                        <select onChange={(event) => setPrecision(event.target.value)} value={precision}>
                          <option value="fp8">FP8 (lower VRAM)</option>
                          <option value="bf16">BF16 (higher quality, CPU offload)</option>
                        </select>
                      </label>
                    </>
                  ) : null}
                  {selectedModel?.adapter === "ltx_video" ? (
                    <>
                      <label>
                        Video CFG
                        <input
                          min="0"
                          max="30"
                          onChange={(event) => setLtxVideoCfg(event.target.value)}
                          placeholder="4.0"
                          step="0.1"
                          type="number"
                          value={ltxVideoCfg}
                        />
                      </label>
                      <label>
                        Video STG
                        <input
                          min="0"
                          max="10"
                          onChange={(event) => setLtxVideoStg(event.target.value)}
                          placeholder="0.0"
                          step="0.1"
                          type="number"
                          value={ltxVideoStg}
                        />
                      </label>
                      <label>
                        Video rescale
                        <input
                          min="0"
                          max="2"
                          onChange={(event) => setLtxVideoRescale(event.target.value)}
                          placeholder="0.7"
                          step="0.05"
                          type="number"
                          value={ltxVideoRescale}
                        />
                      </label>
                    </>
                  ) : null}
                  {["extend_clip", "video_bridge"].includes(mode) ? (
                    <>
                      <label>
                        {mode === "video_bridge" ? "Left clip strength" : "Clip strength"}
                        <input
                          min="0"
                          max="1"
                          onChange={(event) => setVideoConditioningStrength(event.target.value)}
                          placeholder="1.0"
                          step="0.05"
                          type="number"
                          value={videoConditioningStrength}
                        />
                      </label>
                      {mode === "video_bridge" ? (
                        <label>
                          Right clip strength
                          <input
                            min="0"
                            max="1"
                            onChange={(event) => setBridgeRightVideoConditioningStrength(event.target.value)}
                            placeholder="1.0"
                            step="0.05"
                            type="number"
                            value={bridgeRightVideoConditioningStrength}
                          />
                        </label>
                      ) : null}
                    </>
                  ) : null}
                  {supportsQuantization ? (
                    <label>
                      Quantization
                      <select onChange={(event) => setQuantization(event.target.value)} value={quantization}>
                        <option value="auto">Auto (per-platform default)</option>
                        {quantVariants.map(([id, variant]) => (
                          <option key={id} value={id}>
                            {variant?.label ?? id}
                          </option>
                        ))}
                        <option value="none">Full precision (unquantized)</option>
                      </select>
                    </label>
                  ) : null}
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
                  <label>
                    Character
                    <select onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
                      <option value="">No character</option>
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
                  {characterId ? (
                    <div className="guidance-strip">
                      <strong>Character reference</strong>
                      <span>
                        Character and look are saved with the recipe; LTX image conditioning uses IC-LoRA when the selected preset includes one.
                      </span>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </aside>

            <aside className="tips-card">
              <h3>Tips</h3>
              <ul>
                <li>Short clips (4–6s) compose better in the editor.</li>
                <li>Describe the motion, not just the scene.</li>
                <li>Pick a motion chip above to guide the camera.</li>
              </ul>
            </aside>

            <aside className="keyboard-card">
              <h3>Keyboard</h3>
              <dl>
                <div className="kbd-row">
                  <span>Render</span>
                  <span className="kbd-keys">
                    <kbd>⌘</kbd>
                    <kbd>↵</kbd>
                  </span>
                </div>
                <div className="kbd-row">
                  <span>Send to editor</span>
                  <span className="kbd-keys">
                    <kbd>⇧</kbd>
                    <kbd>E</kbd>
                  </span>
                </div>
                <div className="kbd-row">
                  <span>Loop preview</span>
                  <span className="kbd-keys">
                    <kbd>L</kbd>
                  </span>
                </div>
              </dl>
            </aside>
          </div>
        </div>
      </form>
      {guideOpen ? (
        <PromptGuideModal guide={promptGuide} modelName={selectedModel?.name} onClose={() => setGuideOpen(false)} />
      ) : null}
    </section>
  );
}
