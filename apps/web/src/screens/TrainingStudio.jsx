import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAppContext } from "../context/AppContext.js";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { downloadOffersFor } from "../modelEligibility.js";
import { DEFAULT_MAC_CAPABILITIES, macTrainingKernelBlocked } from "../macGating.js";
import { API_BASE_URL, isAbortError } from "../api.js";
import { assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { JOY_CAPTION_MODEL_ID, terminalStatuses } from "../constants.js";
import {
  buildJoyCaptionPrompt,
  defaultCaptionSettings,
  joyCaptionModel,
} from "../training/joyCaptionPrompts.js";
import { asText, boundedNumber, integerFromDraft } from "../training/drafts.js";
import {
  captionDraftsFromDataset,
  datasetHealth,
  datasetItemSelectionKey,
  datasetOwnedAssets,
  datasetPayload,
  imageAssetName,
  normalizeDatasetAssetIds,
  selectionAfterDuplicateRemoval,
  summarizeDatasets,
} from "../training/datasetHelpers.js";
import {
  captionHash,
  nextDismissedChecks,
  readinessBySelectionKey,
  readinessQueryParams,
  trainBlockedByReadiness,
} from "../training/datasetReadiness.js";
import {
  configDraftFromTarget,
  configFieldLabels,
  configValidation,
  defaultGpuOptions,
  defaultOptimizerOptions,
  defaultPresetForTarget,
  lrSchedulerOptions,
  presetsForTarget,
  rangeOptions,
  samplePromptsFromTrigger,
  trainingAdapterVersionOptions,
  trainingConfigSnapshot,
} from "../training/trainingConfig.js";
import { ConfigureJobPanel } from "./training/ConfigureJobPanel.jsx";
import { DatasetEditorPanel } from "./training/DatasetEditorPanel.jsx";

// Re-exported for callers/tests that import the pure config builders from the
// screen; the implementations now live in ../training/trainingConfig.js (sc-4199).
export { configDraftFromTarget, trainingConfigSnapshot };

const trainingTabs = [
  { id: "configure", label: "Configure Job", title: "Configure training job", status: "Queue dry run" },
];

const IMPORT_IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]);

function uploadFileBaseName(name) {
  return String(name ?? "").replaceAll("\\", "/").split("/").pop() ?? "";
}

function uploadFileExtension(name) {
  const base = uploadFileBaseName(name);
  const dotIndex = base.lastIndexOf(".");
  return dotIndex > 0 ? base.slice(dotIndex + 1).toLowerCase() : "";
}

// Lowercased stem used to pair an image upload with its caption sidecar
// (e.g. `Mira_01.png` and `Mira_01.txt` both resolve to `mira_01`).
function uploadFileStem(name) {
  const base = uploadFileBaseName(name);
  const dotIndex = base.lastIndexOf(".");
  return (dotIndex > 0 ? base.slice(0, dotIndex) : base).toLowerCase();
}

function isCaptionUpload(file) {
  return uploadFileExtension(file?.name) === "txt" || file?.type === "text/plain";
}

function isImageUpload(file) {
  return String(file?.type ?? "").startsWith("image/") || IMPORT_IMAGE_EXTENSIONS.has(uploadFileExtension(file?.name));
}

function parseTriggerWords(value) {
  return String(value ?? "")
    .split(",")
    .map((word) => word.trim())
    .filter(Boolean);
}

function triggerPhraseFromText(value) {
  return parseTriggerWords(value).join(", ");
}

function safeSlug(value, fallback = "item") {
  const slug = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return slug || fallback;
}

function orderedName(index, prefix) {
  return `${safeSlug(prefix, "item")}_${String(index + 1).padStart(4, "0")}`;
}

// Map a member to its saved dataset item id so a single-image Re-Caption can
// target it after a save. The member's id IS its selection key, so match the
// saved item that resolves to the same key (asset id for library/character
// items, dataset-item key for owned uploads); fall back to display name.
function resolveSavedItemId(dataset, member) {
  const items = dataset?.items ?? [];
  const byKey = items.find((item, index) => datasetItemSelectionKey(dataset, item, index) === member?.id);
  if (byKey) {
    return byKey.id;
  }
  const name = member?.displayName ?? imageAssetName(member);
  return items.find((item) => (item.displayName ?? imageAssetName(item)) === name)?.id ?? null;
}




function trainingCaptionJobPayload(settings) {
  const captionPrompt = String(settings.captionPrompt || buildJoyCaptionPrompt(settings)).trim();
  return {
    captioner: "joy_caption",
    modelNameOrPath: String(settings.modelNameOrPath ?? "").trim() || joyCaptionModel,
    recaption: Boolean(settings.recaption),
    requestedGpu: settings.requestedGpu || "auto",
    options: {
      captionType: settings.captionType,
      captionLength: settings.captionLength,
      extraOptions: settings.extraOptions,
      nameInput: String(settings.nameInput ?? "").trim(),
      temperature: boundedNumber(settings.temperature, 0.6, 0, 2),
      topP: boundedNumber(settings.topP, 0.9, 0, 1),
      maxNewTokens: integerFromDraft(settings.maxNewTokens, 256, 1, 1024),
      captionPrompt,
      lowVram: Boolean(settings.lowVram),
    },
  };
}

// Convert a training-sample record into an asset-shaped object that AssetThumbnail
// (via assetUrl) can render. Worker emits {url, relativePath, step, prompt}; we
// translate to either { url } (when relative) or { file: { path } } (when path-
// based) so the shared component renders identically to other thumbnails.
function trainingSampleToAsset(sample, projectId, label, key) {
  if (sample?.url) {
    let relative = sample.url;
    if (sample.url.startsWith(API_BASE_URL)) {
      relative = sample.url.slice(API_BASE_URL.length);
    }
    return { id: key, type: "image", projectId, url: relative, displayName: label };
  }
  if (sample?.relativePath) {
    return {
      id: key,
      type: "image",
      projectId,
      file: { path: sample.relativePath, mimeType: "image/png" },
      displayName: label,
    };
  }
  return null;
}

function trainingSampleIdentity(sample, index) {
  return sample?.relativePath ?? sample?.path ?? sample?.url ?? `${sample?.step ?? "sample"}-${sample?.prompt ?? ""}-${index}`;
}

function allTrainingSamples(job) {
  const samples = Array.isArray(job.result?.trainingSamples) ? job.result.trainingSamples : [];
  const latest = Array.isArray(job.result?.latestTrainingSamples) ? job.result.latestTrainingSamples : [];
  const seen = new Set();
  return [...samples, ...latest].filter((sample, index) => {
    const key = trainingSampleIdentity(sample, index);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function sampleStepKey(sample, index) {
  const numeric = Number(sample?.step);
  return Number.isFinite(numeric) && numeric > 0 ? `step-${numeric}` : `sample-${index}`;
}

function sampleStepValue(samples, fallbackIndex) {
  const numeric = Number(samples[0]?.step);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : fallbackIndex;
}

// Resolve all training-sample cycles for a job. Each sampling step gets its own
// worker-card section so newer cycles stack above older cycles instead of
// replacing them.
export function trainingSampleGroups(job, projectId) {
  const samples = allTrainingSamples(job);
  const samplePrompts = Array.isArray(job.result?.samplePrompts)
    ? job.result.samplePrompts
    : samplePromptsFromTrigger(job.payload?.plan?.config?.triggerWord);
  const grouped = [];
  const groupsByStep = new Map();
  samples.forEach((sample, index) => {
    const key = sampleStepKey(sample, index);
    if (!groupsByStep.has(key)) {
      const group = { key, firstIndex: index, samples: [] };
      groupsByStep.set(key, group);
      grouped.push(group);
    }
    groupsByStep.get(key).samples.push(sample);
  });

  return grouped
    .map((group, index) => {
      const step = sampleStepValue(group.samples, group.firstIndex);
      return { ...group, chronologicalSampleNumber: index + 1, step };
    })
    .sort((a, b) => b.step - a.step)
    .map((group) => {
      const stepLabel = Number.isFinite(Number(group.samples[0]?.step)) ? ` - Step ${group.samples[0].step}` : "";
      const assets = group.samples.map((sample, index) => {
        const prompt = sample?.prompt ?? samplePrompts[index] ?? `Sample ${index + 1}`;
        return trainingSampleToAsset(
          sample,
          projectId,
          prompt,
          `${job.id}-sample-${group.key}-${trainingSampleIdentity(sample, index)}`,
        );
      }).filter(Boolean);
      return {
        id: `${job.id}-sample-group-${group.key}`,
        label: `Sample #${group.chronologicalSampleNumber}${stepLabel}`,
        assets,
      };
    })
    .filter((group) => group.assets.length);
}

function TrainingLiveProgress({ jobs, projectId }) {
  if (!jobs.length) {
    return null;
  }
  return (
    <section className="training-live-panel" aria-label="Active training progress">
      <div className="training-live-title">
        <div>
          <p className="eyebrow">Live training</p>
          <h3>Training in progress</h3>
        </div>
        <span>{jobs.length} active</span>
      </div>
      <div className="training-live-list worker-progress-card-stack">
        {jobs.map((job) => (
          <WorkerProgressCard
            key={job.id}
            job={job}
            thumbnailsVariant="image-grid"
            thumbnailGroups={trainingSampleGroups(job, projectId)}
            expectedThumbnailCount={4}
          />
        ))}
      </div>
    </section>
  );
}

export function TrainingDataSetsLibrary() {
  return <TrainingStudio mode="datasets" />;
}

export function TrainingStudio({ mode = "training" } = {}) {
  const datasetLibraryMode = mode === "datasets";
  const {
    activeProject,
    authenticated = true,
    assets = [],
    characters = [],
    gpuOptions = defaultGpuOptions,
    jobs = [],
    setPreviewAsset,
    importAsset: importAssetRaw,
    trainingDatasets = [],
    trainingDatasetsProjectId,
    trainingDatasetsError = "",
    loadingTrainingDatasets = false,
    refreshTrainingDatasets,
    loadTrainingDataset,
    loadTrainingDatasetReadiness,
    setTrainingDatasetItemQualityAck,
    createTrainingDataset,
    uploadTrainingDatasetItem,
    updateTrainingDataset,
    batchRenameTrainingDataset,
    createTrainingDatasetCaptionJob,
    createTrainingJob,
    trainingPresets: trainingPresetsCatalog,
    trainingPresetsError = "",
    trainingTargets: trainingTargetsCatalog,
    trainingTargetsError = "",
    setActiveView,
    studioLaunch,
    models = [],
    createModelDownloadJob,
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContext();
  const datasets = trainingDatasetsProjectId === activeProject?.id ? trainingDatasets : [];
  const datasetsError = trainingDatasetsError;
  const loadingDatasets = loadingTrainingDatasets;
  const onPreview = setPreviewAsset;
  const onRefreshDatasets = () => refreshTrainingDatasets(activeProject?.id);
  const uploadDatasetItem = uploadTrainingDatasetItem ?? ((file) => importAssetRaw(file, { throwOnError: true }));
  const loadDataset = loadTrainingDataset;
  const createDataset = createTrainingDataset;
  const updateDataset = updateTrainingDataset;
  const batchRenameDataset = batchRenameTrainingDataset;
  const createCaptionJob = createTrainingDatasetCaptionJob;
  const trainingPresets = trainingPresetsCatalog?.presets ?? [];
  const trainingTargets = trainingTargetsCatalog?.targets ?? [];
  const workflowTabs = datasetLibraryMode ? [] : trainingTabs;
  const [activeTab, setActiveTab] = useState(datasetLibraryMode ? "dataset" : "configure");
  const [activeDataset, setActiveDataset] = useState(null);
  const [datasetError, setDatasetError] = useState("");
  const [datasetMessage, setDatasetMessage] = useState("");
  const [draftName, setDraftName] = useState("");
  const [busyDatasetId, setBusyDatasetId] = useState("");
  const [importingAssets, setImportingAssets] = useState(false);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [uploadedDatasetAssets, setUploadedDatasetAssets] = useState([]);
  const [savingDataset, setSavingDataset] = useState(false);
  const [selectedAssetIds, setSelectedAssetIds] = useState([]);
  // Dataset Doctor readiness report (sc-6534), fetched server-side over the saved
  // dataset. `null` until the first fetch resolves; the loading flag keeps badges in
  // a neutral "not assessed" state rather than flashing green while a fetch is in flight.
  const [readiness, setReadiness] = useState(null);
  const [readinessLoading, setReadinessLoading] = useState(false);
  // Bumped after a per-image override write (sc-6534) to re-fetch readiness — an ack is a metadata
  // write that doesn't change the dataset version, so it isn't caught by the version-keyed effect.
  const [readinessRefreshTick, setReadinessRefreshTick] = useState(0);
  // Owning character for this dataset (sc-2022). Seeded from the loaded dataset,
  // set when images are imported from the Character tab, threaded into the save
  // payload. "" = a general (unassociated) dataset.
  const [associatedCharacterId, setAssociatedCharacterId] = useState("");
  // Single source of truth for caption edits, keyed by selection id (sc-2025):
  // seeded from the saved dataset items, updated as the user edits an item's
  // caption or imports a .txt sidecar, and threaded into the dataset payload on
  // save. Replaces the old importedCaptions + rename-caption draft split.
  const [captionDraftById, setCaptionDraftById] = useState({});
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [renamePrefix, setRenamePrefix] = useState("");
  const [captionTriggerWords, setCaptionTriggerWords] = useState("");
  // Open caption settings dialog: null | { type: "all" } | { type: "item", member }.
  const [captionDialog, setCaptionDialog] = useState(null);
  const [captioning, setCaptioning] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [captionSettings, setCaptionSettings] = useState(defaultCaptionSettings);
  const [selectedTargetId, setSelectedTargetId] = useState("");
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [customizedConfigFields, setCustomizedConfigFields] = useState(new Set());
  const [configDraft, setConfigDraft] = useState({});
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [configSnapshot, setConfigSnapshot] = useState(null);
  const [configMessage, setConfigMessage] = useState("");
  const [configError, setConfigError] = useState("");
  const [configTriggerFollowsCaptions, setConfigTriggerFollowsCaptions] = useState(true);
  const [submittingJob, setSubmittingJob] = useState(false);
  // Dry run validates the Rust-resolved plan without training; a real run hands
  // the plan to the worker's Z-Image LoRA kernel. Default to the safe dry run.
  const [trainingRunMode, setTrainingRunMode] = useState("dry_run");
  const configBasisRef = useRef("");
  const tabRefs = useRef({});

  const activeIndex = workflowTabs.findIndex((tab) => tab.id === activeTab);
  const active = workflowTabs[activeIndex] ?? { id: activeTab, title: datasetLibraryMode ? "Dataset management" : "Configure training job" };
  const datasetSummary = useMemo(() => summarizeDatasets(datasets), [datasets]);
  const datasetAssets = useMemo(
    () => datasetOwnedAssets(activeDataset, activeProject?.id, assets),
    [activeDataset, activeProject?.id, assets],
  );
  const imageAssets = useMemo(() => {
    const merged = [
      ...assets.filter((asset) => assetCanRenderAsImage(asset)),
      ...uploadedDatasetAssets,
      ...datasetAssets,
    ];
    const seen = new Set();
    return merged.filter((asset) => {
      if (!asset?.id || seen.has(asset.id)) {
        return false;
      }
      seen.add(asset.id);
      return true;
    });
  }, [assets, datasetAssets, uploadedDatasetAssets]);
  const assetsById = useMemo(() => new Map(imageAssets.map((asset) => [asset.id, asset])), [imageAssets]);
  const unavailableAssetIds = useMemo(
    () => selectedAssetIds.filter((assetId) => !assetsById.has(assetId)),
    [assetsById, selectedAssetIds],
  );
  // Members shown in the editor body: the dataset's own items (available
  // assets), in selection order. Captions come from the saved dataset items or
  // freshly imported .txt sidecars, keyed by selection id.
  const memberAssets = useMemo(
    () => selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean),
    [assetsById, selectedAssetIds],
  );
  // Thumbnail for the compact dataset selector (sc-2025): the list summary's
  // server-resolved cover image (first item), falling back to the open dataset's
  // first loaded member.
  const datasetThumbAsset = (dataset) => {
    if (dataset?.coverPath) {
      return { projectId: activeProject?.id, type: "image", file: { path: dataset.coverPath } };
    }
    if (dataset?.id === selectedDatasetId && memberAssets[0]) {
      return memberAssets[0];
    }
    return null;
  };
  const health = useMemo(
    () => datasetHealth({ activeDataset, imageAssets, selectedAssetIds }),
    [activeDataset, imageAssets, selectedAssetIds],
  );
  // Per-thumbnail readiness, keyed by the same selection id the caption grid renders
  // under (sc-6534). An absent entry means "not assessed yet" — never silently good.
  const readinessByKey = useMemo(() => readinessBySelectionKey(activeDataset, readiness), [activeDataset, readiness]);
  // Train is disabled only when a saved report exists and is genuinely Blocked (too few
  // images / a fatal flag). Bias to warn: no report or a warning gate never hard-blocks.
  const readinessBlocksTraining = trainBlockedByReadiness(readiness);
  const originalAssetIds = useMemo(() => normalizeDatasetAssetIds(activeDataset, assets), [activeDataset, assets]);
  // Caption edits also make the dataset dirty so the single Save persists them.
  const captionsDirty = useMemo(
    () =>
      Boolean(activeDataset) &&
      (activeDataset.items ?? []).some((item, index) => {
        const draft = captionDraftById[datasetItemSelectionKey(activeDataset, item, index)];
        return draft && (draft.text ?? "") !== (item.caption?.text ?? "");
      }),
    [activeDataset, captionDraftById],
  );
  const dirty =
    Boolean(activeDataset) &&
    (draftName.trim() !== activeDataset.name ||
      associatedCharacterId !== (activeDataset.characterId ?? "") ||
      selectedAssetIds.length !== originalAssetIds.length ||
      selectedAssetIds.some((id, index) => id !== originalAssetIds[index]) ||
      captionsDirty);
  const canSave =
    draftName.trim().length > 0 &&
    selectedAssetIds.length > 0 &&
    health.disabledItems === 0 &&
    !savingDataset &&
    (!activeDataset || dirty);
  const displayedCaptionPrompt = captionSettings.captionPrompt || buildJoyCaptionPrompt(captionSettings);
  // JoyCaption model provisioning (sc-5620): the native captioner has no auto-download (unlike
  // the torch path), so on a gated Mac a missing model would just fail the queued caption job.
  // Surface a download affordance in the caption dialog instead. Mac-gated because Windows/Linux
  // torch still auto-downloads (missing ≠ broken there); scoped to the default/cataloged model.
  const captionModel = useMemo(
    () => models.find((entry) => entry.id === JOY_CAPTION_MODEL_ID),
    [models],
  );
  const selectedCaptionModel = String(captionSettings.modelNameOrPath ?? "").trim() || joyCaptionModel;
  const captionModelMissing =
    Boolean(macCapabilities?.macGatingActive) &&
    captionModel?.installState === "missing" &&
    selectedCaptionModel === joyCaptionModel;
  const captionModelSizeLabel = useMemo(() => {
    const bytes = Number(captionModel?.downloadSizeBytes);
    return Number.isFinite(bytes) && bytes > 0 ? `${(bytes / 1e9).toFixed(1)} GB` : "";
  }, [captionModel?.downloadSizeBytes]);
  const onDownloadCaptionModel = captionModel && typeof createModelDownloadJob === "function"
    ? () => createModelDownloadJob(captionModel)
    : undefined;
  const firstTarget = trainingTargets[0] ?? null;
  // Mac UI gating (sc-3486): a target whose kernel has no native mlx-gen Rust trainer
  // (kolors_lora / lens_lora) can't train on a gated Mac — disable it and snap off it.
  const macTargetBlocked = (target) => macTrainingKernelBlocked(macCapabilities, target?.kernel);
  // Model-availability gate (sc-5947): training needs a trainable target whose base model is
  // downloaded. A target's base counts as missing only when it's present in the catalog AND
  // installState === "missing" (so a thin test context with no `models` stays ready, and a real
  // install reads the true state). Only gated when targets exist but every trainable base is
  // missing — a target-registry error keeps its own "registry unavailable" message.
  const usableTrainingTargets = trainingTargets.filter((target) => !macTargetBlocked(target));
  const trainingBaseMissing = (target) => {
    const base = models.find((item) => item.id === target?.baseModel);
    return Boolean(base) && base.installState === "missing";
  };
  const trainingReady =
    usableTrainingTargets.length === 0 ||
    usableTrainingTargets.some((target) => !trainingBaseMissing(target));
  const trainingBaseIds = new Set(usableTrainingTargets.map((target) => target.baseModel).filter(Boolean));
  const trainingOffers = useMemo(
    () => downloadOffersFor(models, (item) => trainingBaseIds.has(item.id), macCapabilities),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [models, macCapabilities, [...trainingBaseIds].join("|")],
  );
  const trainingDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );
  const selectedTarget = useMemo(
    () => trainingTargets.find((target) => target.id === selectedTargetId) ?? firstTarget,
    [firstTarget, selectedTargetId, trainingTargets],
  );
  useEffect(() => {
    if (selectedTarget && macTargetBlocked(selectedTarget)) {
      const fallback = trainingTargets.find((target) => !macTargetBlocked(target));
      if (fallback) setSelectedTargetId(fallback.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTarget?.id, trainingTargets, macCapabilities]);
  const targetPresets = useMemo(
    () => presetsForTarget(trainingPresets, selectedTarget?.id),
    [selectedTarget?.id, trainingPresets],
  );
  const selectedPreset = useMemo(
    () => targetPresets.find((preset) => preset.id === selectedPresetId) ?? defaultPresetForTarget(targetPresets, selectedTarget?.id),
    [selectedPresetId, selectedTarget?.id, targetPresets],
  );
  const qualityPresets = rangeOptions(selectedTarget?.limits, "qualityPresets");
  const visibleQualityPresets =
    configDraft.qualityPreset && !qualityPresets.includes(configDraft.qualityPreset)
      ? [...qualityPresets, configDraft.qualityPreset]
      : qualityPresets;
  const outputScopes = rangeOptions(selectedTarget?.limits, "outputScopes");
  const resolutionOptions = rangeOptions(selectedTarget?.limits, "resolutions");
  const visibleResolutionOptions =
    configDraft.resolution && !resolutionOptions.map(String).includes(String(configDraft.resolution))
      ? [...resolutionOptions, configDraft.resolution]
      : resolutionOptions;
  const optimizerOptions = rangeOptions(selectedTarget?.limits, "optimizers");
  const optimizerSelectOptions = optimizerOptions.length ? optimizerOptions : defaultOptimizerOptions;
  const visibleOptimizerOptions =
    configDraft.optimizer && !optimizerSelectOptions.includes(configDraft.optimizer)
      ? [...optimizerSelectOptions, configDraft.optimizer]
      : optimizerSelectOptions;
  // Network type: only render the picker when the target advertises a real choice
  // (more than just lora). LoKr targets gain the picker + the LoKr factor field.
  const networkTypeOptions = rangeOptions(selectedTarget?.limits, "networkTypes");
  const showNetworkType = networkTypeOptions.length > 1;
  const isLokrNetwork = asText(configDraft.networkType).trim() === "lokr";
  // Mac UI gating (sc-3486): the mlx Wan trainer can't merge a Kronecker (LoKr) adapter, so
  // disable the LoKr network type for Wan targets on a gated Mac (LoKr on Z-Image/SDXL/LTX is fine).
  const macLokrOnWanBlocked =
    Boolean(macCapabilities?.macGatingActive) &&
    (selectedTarget?.kernel ?? "").startsWith("wan") &&
    macCapabilities?.training?.lokrOnWanSupported === false;
  useEffect(() => {
    if (macLokrOnWanBlocked && isLokrNetwork) {
      updateConfigDraft("networkType", "lora");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [macLokrOnWanBlocked, isLokrNetwork]);
  const lrSchedulerLimitOptions = rangeOptions(selectedTarget?.limits, "lrSchedulers");
  const lrSchedulerSelectOptions = lrSchedulerLimitOptions.length ? lrSchedulerLimitOptions : lrSchedulerOptions;
  const visibleLrSchedulerOptions =
    configDraft.lrScheduler && !lrSchedulerSelectOptions.includes(configDraft.lrScheduler)
      ? [...lrSchedulerSelectOptions, configDraft.lrScheduler]
      : lrSchedulerSelectOptions;
  // De-distill training adapter is Z-Image-Turbo-only: show the version selector
  // only when the resolved config declares a trainingAdapterRepo.
  const showTrainingAdapter = Boolean(asText(configDraft.trainingAdapterRepo).trim());
  const visibleTrainingAdapterVersions =
    configDraft.trainingAdapterVersion && !trainingAdapterVersionOptions.includes(configDraft.trainingAdapterVersion)
      ? [...trainingAdapterVersionOptions, configDraft.trainingAdapterVersion]
      : trainingAdapterVersionOptions;
  const activeTrainingJobs = useMemo(
    () =>
      jobs
        .filter((job) => job.type === "lora_train" && job.projectId === activeProject?.id && !terminalStatuses.has(job.status))
        .slice(0, 3),
    [activeProject?.id, jobs],
  );
  const gpuOptionsKey = gpuOptions.join("\u0000");
  const configWarnings = configValidation({ activeDataset, configDraft, selectedTarget });
  const configReady = configWarnings.length === 0;
  const customizedConfigLabels = [...customizedConfigFields].map((field) => configFieldLabels[field] ?? field);

  // The dataset's character kind (person/style/object) — refines the readiness kind
  // and thus the blur floor. "" when the dataset isn't tied to a character.
  const datasetCharacterType = useMemo(
    () => characters.find((character) => character.id === associatedCharacterId)?.type ?? "",
    [characters, associatedCharacterId],
  );
  // Readiness query string. Blur/exposure are measured at the training bucket and the
  // blur floor varies by kind, so the chosen resolution + target tags + character kind
  // are threaded in — otherwise the badges would describe a different bucket than the
  // user trains at (sc-6534). All optional; the server falls back to per-kind defaults.
  const readinessQuery = useMemo(
    () =>
      readinessQueryParams({
        resolution: configDraft.resolution,
        recommendedFor: selectedTarget?.recommendedFor ?? selectedPreset?.recommendedFor ?? [],
        characterType: datasetCharacterType,
      }),
    [configDraft.resolution, selectedTarget?.recommendedFor, selectedPreset?.recommendedFor, datasetCharacterType],
  );
  // Fetch the readiness report for the SAVED dataset whenever the saved version or the
  // training context changes. Kept entirely off the save-gate path: the first (uncached)
  // pass decodes every image, so a slow assessment must never block editing or saving.
  useEffect(() => {
    const projectId = activeProject?.id;
    const datasetId = activeDataset?.id;
    if (!projectId || !datasetId || typeof loadTrainingDatasetReadiness !== "function") {
      setReadiness(null);
      setReadinessLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    setReadinessLoading(true);
    loadTrainingDatasetReadiness(datasetId, readinessQuery, projectId, { signal: controller.signal })
      .then((report) => {
        if (controller.signal.aborted) {
          return;
        }
        setReadiness(report ?? null);
        setReadinessLoading(false);
      })
      .catch((err) => {
        if (isAbortError(err) || controller.signal.aborted) {
          return;
        }
        // Readiness is advisory — a failed assessment must not surface as a hard error
        // or block training; just drop back to "not assessed".
        setReadiness(null);
        setReadinessLoading(false);
      });
    return () => controller.abort();
  }, [
    activeProject?.id,
    activeDataset?.id,
    activeDataset?.version,
    readinessQuery,
    readinessRefreshTick,
    loadTrainingDatasetReadiness,
  ]);

  // Dismiss/undo a single quality finding on an image (sc-6534). The endpoint replaces the item's
  // full dismissed-check set, so we send the next set derived from the readiness entry, then refetch.
  async function toggleItemQualityAck(entry, check, dismissed) {
    if (!activeDataset?.id || !entry?.itemId || typeof setTrainingDatasetItemQualityAck !== "function") {
      return;
    }
    const checks = nextDismissedChecks(entry, check, dismissed);
    const item = activeDataset.items?.find((candidate) => candidate.id === entry.itemId);
    if (!item?.contentHash) {
      setDatasetError("Refresh the dataset before dismissing this finding.");
      return;
    }
    try {
      await setTrainingDatasetItemQualityAck(activeDataset.id, entry.itemId, checks, {
        expectedContentHash: item.contentHash,
        expectedCaptionHash: await captionHash(item.caption?.text ?? ""),
      });
      setReadinessRefreshTick((tick) => tick + 1);
    } catch (err) {
      setDatasetError(err.message);
    }
  }

  useEffect(() => {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setUploadedDatasetAssets([]);
    setSelectedDatasetId("");
    setRenamePrefix("");
    setCaptionTriggerWords("");
    setCaptionDraftById({});
    setCaptionSettings(defaultCaptionSettings);
    setSelectedPresetId("");
    setCustomizedConfigFields(new Set());
    setConfigDraft({});
    setConfigSnapshot(null);
    setConfigMessage("");
    setConfigError("");
    setConfigTriggerFollowsCaptions(true);
    configBasisRef.current = "";
  }, [activeProject?.id]);

  useEffect(() => {
    setActiveTab(datasetLibraryMode ? "dataset" : "configure");
  }, [datasetLibraryMode]);

  // sc-2022: open a dataset requested from elsewhere (Character Studio's
  // "Open" action hands off via studioLaunch). Keyed on the launch id so a
  // repeat request for the same dataset re-opens it.
  useEffect(() => {
    if (!datasetLibraryMode || studioLaunch?.view !== "LibraryDataSets" || !studioLaunch?.datasetId) {
      return;
    }
    if (studioLaunch.datasetId !== selectedDatasetId) {
      openDataset(studioLaunch.datasetId);
    }
  }, [datasetLibraryMode, studioLaunch?.id]);

  useEffect(() => {
    const datasetTriggerPhrase = triggerPhraseFromText(activeDataset?.name);
    // Seed caption drafts from the (re)loaded dataset; switching datasets or
    // saving (which swaps activeDataset) resets edits to the persisted state.
    setCaptionDraftById(captionDraftsFromDataset(activeDataset));
    setRenamePrefix(safeSlug(activeDataset?.name, "item"));
    setCaptionTriggerWords(datasetTriggerPhrase);
    setCaptionSettings((current) => ({ ...current, nameInput: datasetTriggerPhrase }));
  }, [activeDataset]);

  useEffect(() => {
    if (!selectedTargetId && firstTarget?.id) {
      setSelectedTargetId(firstTarget.id);
    }
  }, [firstTarget?.id, selectedTargetId]);

  useEffect(() => {
    if (!selectedTarget) {
      if (configBasisRef.current) {
        configBasisRef.current = "";
        setConfigDraft({});
      }
      return;
    }
    const defaultPreset = defaultPresetForTarget(trainingPresets, selectedTarget.id);
    const basis = `${selectedTarget.id}\u0000${activeDataset?.id ?? ""}\u0000${defaultPreset?.id ?? ""}`;
    if (configBasisRef.current === basis) {
      return;
    }
    configBasisRef.current = basis;
    setSelectedPresetId(defaultPreset?.id ?? "");
    setCustomizedConfigFields(new Set());
    setConfigDraft(configDraftFromTarget(selectedTarget, activeDataset, gpuOptions, triggerPhraseFromText(activeDataset?.name), defaultPreset));
    setConfigSnapshot(null);
    setConfigMessage("");
    setConfigError("");
    setConfigTriggerFollowsCaptions(true);
  }, [activeDataset?.id, selectedTarget?.id, trainingPresets]);

  useEffect(() => {
    if (!configTriggerFollowsCaptions) {
      return;
    }
    const nextTriggerPhrase = triggerPhraseFromText(captionTriggerWords) || asText(selectedTarget?.defaults?.triggerWord);
    setConfigDraft((current) => {
      if ((current.triggerWord ?? "") === nextTriggerPhrase) {
        return current;
      }
      return { ...current, triggerWord: nextTriggerPhrase };
    });
    setConfigSnapshot(null);
  }, [captionTriggerWords, configTriggerFollowsCaptions, selectedTarget?.id]);

  useEffect(() => {
    setConfigDraft((current) => {
      if (!current.requestedGpu || gpuOptions.includes(current.requestedGpu)) {
        return current;
      }
      return { ...current, requestedGpu: gpuOptions[0] ?? "" };
    });
    setCaptionSettings((current) => {
      if (!current.requestedGpu || gpuOptions.includes(current.requestedGpu)) {
        return current;
      }
      return { ...current, requestedGpu: gpuOptions[0] ?? "" };
    });
  }, [gpuOptionsKey]);

  function focusTab(index) {
    if (!workflowTabs.length) {
      return;
    }
    const next = workflowTabs[(index + workflowTabs.length) % workflowTabs.length];
    setActiveTab(next.id);
    window.requestAnimationFrame(() => tabRefs.current[next.id]?.focus());
  }

  function onTabKeyDown(event) {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTab(activeIndex + 1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTab(activeIndex - 1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusTab(0);
    }
    if (event.key === "End") {
      event.preventDefault();
      focusTab(workflowTabs.length - 1);
    }
  }

  async function openDataset(datasetId) {
    if (!datasetId) {
      setActiveDataset(null);
      setDraftName("");
      setSelectedAssetIds([]);
      setAssociatedCharacterId("");
      setSelectedDatasetId("");
      return;
    }
    setBusyDatasetId(datasetId);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const dataset = await loadDataset(datasetId);
      setActiveDataset(dataset);
      setDraftName(dataset?.name ?? "");
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset, assets));
      setAssociatedCharacterId(dataset?.characterId ?? "");
      setSelectedDatasetId(dataset?.id ?? datasetId);
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setBusyDatasetId("");
    }
  }

  // Drops a member (or an unavailable/orphaned id) from the dataset selection.
  function removeUnavailableAsset(assetId) {
    setDatasetMessage("");
    setSelectedAssetIds((current) => current.filter((id) => id !== assetId));
  }

  function startNewDataset() {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setAssociatedCharacterId("");
    setUploadedDatasetAssets([]);
    setSelectedDatasetId("");
  }

  // Editing a caption marks it as manually authored (sc-2025) — caption source
  // is detected, never picked by the user.
  function updateCaption(selectionId, text) {
    setDatasetMessage("");
    setCaptionDraftById((current) => ({
      ...current,
      [selectionId]: { text, source: "manual" },
    }));
  }

  function updateConfigDraft(field, value) {
    setConfigMessage("");
    setConfigError("");
    setConfigSnapshot(null);
    if (field === "triggerWord") {
      setConfigTriggerFollowsCaptions(false);
    }
    if (field === "optimizer" && customizedConfigFields.size === 0) {
      const matchingPreset = targetPresets.find((preset) => preset.optimizer === value);
      if (matchingPreset && matchingPreset.id !== selectedPreset?.id) {
        applyTrainingPreset(matchingPreset, { message: `${matchingPreset.name} applied` });
        return;
      }
    }
    setCustomizedConfigFields((current) => new Set([...current, field]));
    setConfigDraft((current) => ({ ...current, [field]: value }));
  }

  function applyTrainingPreset(preset, { message = "Preset applied" } = {}) {
    if (!selectedTarget || !preset) {
      return;
    }
    setSelectedPresetId(preset.id);
    setCustomizedConfigFields(new Set());
    setConfigDraft((current) =>
      configDraftFromTarget(
        selectedTarget,
        activeDataset,
        gpuOptions,
        current.triggerWord || triggerPhraseFromText(captionTriggerWords),
        preset,
        { outputName: current.outputName },
      ),
    );
    setConfigTriggerFollowsCaptions(false);
    setConfigSnapshot(null);
    setConfigMessage(message);
    setConfigError("");
  }

  function updateSelectedPreset(presetId) {
    const preset = targetPresets.find((item) => item.id === presetId);
    if (preset) {
      applyTrainingPreset(preset);
    }
  }

  function updateCaptionSetting(field, value) {
    setDatasetMessage("");
    setDatasetError("");
    setCaptionSettings((current) => ({ ...current, [field]: value }));
  }

  function toggleCaptionExtraOption(value) {
    setDatasetMessage("");
    setCaptionSettings((current) => {
      const values = current.extraOptions.includes(value)
        ? current.extraOptions.filter((option) => option !== value)
        : [...current.extraOptions, value];
      return { ...current, extraOptions: values };
    });
  }

  function resetConfigDefaults() {
    if (!selectedTarget) {
      return;
    }
    const defaultPreset = defaultPresetForTarget(trainingPresets, selectedTarget.id);
    setSelectedPresetId(defaultPreset?.id ?? "");
    setCustomizedConfigFields(new Set());
    setConfigDraft(configDraftFromTarget(selectedTarget, activeDataset, gpuOptions, triggerPhraseFromText(captionTriggerWords), defaultPreset));
    setConfigTriggerFollowsCaptions(true);
    setConfigSnapshot(null);
    setConfigMessage("Defaults restored");
    setConfigError("");
  }

  // Apply sequential names to the saved dataset's items server-side (sc-2025:
  // moved out of the per-item rows to a dataset-level action). Persists current
  // membership/captions first so the rename targets the live item set.
  async function applyOrderedNames() {
    if (!activeDataset?.id || renaming) {
      return;
    }
    setRenaming(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const saved = await persistDataset();
      if (!saved?.id) {
        return;
      }
      const items = (saved.items ?? []).map((item, index) => {
        const nextName = orderedName(index, renamePrefix || saved.name);
        const extension = String(item.path ?? "").split(".").pop() || "png";
        return {
          itemId: item.id,
          newItemId: nextName,
          fileStem: nextName,
          displayName: `${nextName}.${extension}`,
        };
      });
      const next = await batchRenameDataset(saved.id, { items });
      if (next) {
        setActiveDataset(next);
        setSelectedAssetIds(normalizeDatasetAssetIds(next, assets));
        setSelectedDatasetId(next.id);
        setDatasetMessage("Ordered names applied");
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setRenaming(false);
    }
  }

  function addAssets(ids, characterId = null) {
    const additions = (ids ?? []).filter(Boolean);
    if (!additions.length) {
      return;
    }
    setDatasetMessage("");
    setSelectedAssetIds((current) => Array.from(new Set([...current, ...additions])));
    // sc-2022: importing a character's images associates the dataset with it.
    if (characterId) {
      setAssociatedCharacterId(characterId);
    }
  }

  // Accepts a FileList from either the dialog's file input or a drag/drop, so
  // both entry points share the import + caption-pairing path.
  async function handleImport(fileList) {
    const files = Array.from(fileList ?? []);
    if (!files.length) {
      return;
    }
    setImportingAssets(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const imageFiles = files.filter(isImageUpload);
      const captionFiles = files.filter((file) => !isImageUpload(file) && isCaptionUpload(file));

      // Pair each caption with an image by filename stem (`Mira_01.txt` → `Mira_01.png`).
      const captionByStem = new Map();
      for (const file of captionFiles) {
        const text = (await file.text()).trim();
        if (text) {
          captionByStem.set(uploadFileStem(file.name), text);
        }
      }

      const imported = [];
      const captionsByAssetId = {};
      for (const file of imageFiles) {
        const asset = await uploadDatasetItem(file);
        if (asset?.id) {
          asset.datasetOnly = true;
          imported.push(asset.id);
          setUploadedDatasetAssets((current) => [asset, ...current.filter((item) => item.id !== asset.id)]);
          const caption = captionByStem.get(uploadFileStem(file.name));
          if (caption) {
            captionsByAssetId[asset.id] = { source: "imported", text: caption };
          }
        }
      }

      if (imported.length) {
        setSelectedAssetIds((current) => Array.from(new Set([...current, ...imported])));
        if (Object.keys(captionsByAssetId).length) {
          setCaptionDraftById((current) => ({ ...current, ...captionsByAssetId }));
        }
        const imageStems = new Set(imageFiles.map((file) => uploadFileStem(file.name)));
        const unmatchedCaptions = captionFiles.filter((file) => !imageStems.has(uploadFileStem(file.name))).length;
        const matchedCaptions = Object.keys(captionsByAssetId).length;
        const captionNote = matchedCaptions ? ` with ${matchedCaptions} caption${matchedCaptions === 1 ? "" : "s"}` : "";
        const unmatchedNote = unmatchedCaptions
          ? ` ${unmatchedCaptions} caption file${unmatchedCaptions === 1 ? "" : "s"} had no matching image.`
          : "";
        setDatasetMessage(
          `Imported ${imported.length} image${imported.length === 1 ? "" : "s"}${captionNote}. Save the dataset to keep them.${unmatchedNote}`,
        );
      } else {
        setDatasetError("No images were imported.");
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setImportingAssets(false);
    }
  }

  // Persist membership + caption edits. Returns the saved dataset (or null) and
  // re-syncs local state from it. Shared by Save, Apply ordered names, and the
  // caption dialog so captioning/rename always act on the live persisted items.
  async function persistDataset(selectionOverride, baseOverride) {
    // `baseOverride` lets a follow-up save (e.g. dedupe's second persist) act on a just-saved dataset
    // whose React state hasn't flushed yet — without it, `activeDataset`/`assetsById` are stale and a
    // re-materialized item key can miss `assetsById`, silently dropping kept images. Rebuild the
    // asset map from the override so every kept item resolves.
    const base = baseOverride ?? activeDataset;
    const baseAssetsById = baseOverride
      ? new Map([
          ...assetsById,
          ...datasetOwnedAssets(baseOverride, activeProject?.id, assets).map((asset) => [
            asset.id,
            asset,
          ]),
        ])
      : assetsById;
    const payload = datasetPayload({
      activeDataset: base,
      assetsById: baseAssetsById,
      associatedCharacterId,
      captionDraftById,
      name: draftName,
      selectedAssetIds: selectionOverride ?? selectedAssetIds,
    });
    const dataset = base
      ? await updateDataset(base.id, payload)
      : await createDataset(payload);
    if (dataset) {
      setActiveDataset(dataset);
      setDraftName(dataset.name ?? draftName.trim());
      setUploadedDatasetAssets([]);
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset, assets));
      setAssociatedCharacterId(dataset.characterId ?? "");
      setSelectedDatasetId(dataset.id ?? "");
    }
    return dataset;
  }

  async function saveDataset() {
    if (!canSave) {
      return;
    }
    setSavingDataset(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const existing = Boolean(activeDataset);
      const dataset = await persistDataset();
      if (dataset) {
        setDatasetMessage(existing ? "Dataset changes saved" : "Dataset created");
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setSavingDataset(false);
    }
  }

  // Queue a Joy Caption job for the dialog's scope: a single image (Re-Caption)
  // or every item ("Caption all" / "Re-caption all"). Saves first so the job
  // sees the live items, then targets them via the itemIds filter.
  async function runCaptionJob() {
    if (!captionDialog || captioning) {
      return;
    }
    setCaptioning(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const saved = await persistDataset();
      if (!saved?.id) {
        setDatasetError("Save the dataset before captioning.");
        return;
      }
      let itemIds;
      if (captionDialog.type === "item") {
        const id = resolveSavedItemId(saved, captionDialog.member);
        if (!id) {
          setDatasetError("Could not resolve that image to re-caption.");
          return;
        }
        itemIds = [id];
      } else if (captionDialog.type === "flagged") {
        // sc-6537: re-caption the caption-alignment-flagged items. Filter to IDs still present in the
        // just-saved dataset so a stale readout can't target removed items.
        const flagged = new Set(captionDialog.itemIds ?? []);
        itemIds = (saved.items ?? []).map((item) => item.id).filter((id) => flagged.has(id));
        if (!itemIds.length) {
          setDatasetError("No flagged images to re-caption.");
          return;
        }
      }
      const payload = {
        ...trainingCaptionJobPayload(captionSettings),
        ...(itemIds ? { itemIds, recaption: true } : {}),
      };
      const job = await createCaptionJob(saved.id, payload);
      setDatasetMessage(`Caption job queued${job?.id ? ` (${job.id})` : ""}. Track it in the Queue.`);
      setCaptionDialog(null);
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setCaptioning(false);
    }
  }

  // One-tap "drop duplicates" (sc-6539): remove the exact/near-duplicate copies the readiness report
  // chose (keeping the sharpest of each), then re-persist. Saves first so the plan's item ids line up
  // with the live dataset, then maps them to selection keys and drops them. Non-destructive: the
  // originals stay in the library/uploads and can be re-added.
  async function removeDuplicates(removeIds) {
    if (!removeIds?.length || savingDataset) {
      return;
    }
    setSavingDataset(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const saved = await persistDataset();
      if (!saved?.id) {
        setDatasetError("Save the dataset before removing duplicates.");
        return;
      }
      const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
        dataset: saved,
        currentSelection: normalizeDatasetAssetIds(saved, assets),
        removeIds,
      });
      if (!removedCount) {
        setDatasetError("Those duplicates are no longer in the dataset.");
        return;
      }
      const updated = await persistDataset(nextSelection, saved);
      if (updated) {
        setDatasetMessage(
          `Removed ${removedCount} duplicate${removedCount === 1 ? "" : "s"}, keeping the sharpest of each. The originals stay in your library.`,
        );
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setSavingDataset(false);
    }
  }

  async function submitTrainingJob() {
    if (!configReady || submittingJob || readinessBlocksTraining) {
      return;
    }
    setSubmittingJob(true);
    setConfigError("");
    setConfigMessage("");
    try {
      const dryRun = trainingRunMode === "dry_run";
      const snapshot = trainingConfigSnapshot({ activeDataset, configDraft, selectedPreset, selectedTarget, dryRun });
      const job = await createTrainingJob({
        targetId: snapshot.targetId,
        datasetId: snapshot.datasetId,
        datasetVersion: snapshot.datasetVersion,
        presetId: snapshot.presetId,
        presetVersion: snapshot.presetVersion,
        outputName: snapshot.outputName,
        dryRun,
        config: snapshot.config,
      });
      setConfigSnapshot(snapshot);
      const label = dryRun ? "dry-run" : "training";
      setConfigMessage(`Queued ${label} job ${job?.id ?? ""}`.trim() + ". Track it in the Queue.");
    } catch (err) {
      setConfigError(err.message);
    } finally {
      setSubmittingJob(false);
    }
  }

  return (
    <ModelAvailabilityGate
      ready={trainingReady}
      title="Training needs a trainable base model"
      description="LoRA training needs a downloaded base model (e.g. Z-Image-Turbo or SDXL). Download one to get started."
      offers={trainingOffers}
      downloadJobs={trainingDownloadJobs}
      onDownload={createModelDownloadJob}
      onOpenModels={() => setActiveView("Models")}
      onOpenQueue={() => setActiveView("Queue")}
    >
    <section className="main-surface training-studio">
      <div className="training-studio-shell">
        <div className="training-summary-band">
          <div className="section-heading">
            <p className="eyebrow">{datasetLibraryMode ? "Library" : "Training Studio"}</p>
            <h2>{datasetLibraryMode ? "Data Sets" : "Native LoRA training workflow"}</h2>
            <p className="view-copy">
              {datasetLibraryMode
                ? "Create training datasets, manage imported dataset images, and normalize captions in one place."
                : "Select an existing dataset and prepare a Rust-owned training plan before any ML runtime work begins."}
            </p>
          </div>
          <div className="training-metrics" aria-label="Training workspace summary">
            <div>
              <strong>{activeProject?.name ?? "No workspace"}</strong>
              <span>Project</span>
            </div>
            <div>
              <strong>{datasets.length}</strong>
              <span>Datasets</span>
            </div>
            <div>
              <strong>{datasetSummary.items}</strong>
              <span>Items</span>
            </div>
          </div>
        </div>
        {datasetLibraryMode ? null : <TrainingLiveProgress jobs={activeTrainingJobs} projectId={activeProject?.id} />}

        {!authenticated ? (
          <div className="training-empty-state" role="status">
            <Icon.Train size={24} />
            <div>
              <strong>Pairing required</strong>
              <span>Unlock SceneWorks to load project datasets.</span>
            </div>
          </div>
        ) : !activeProject ? (
          <div className="training-empty-state" role="status">
            <Icon.Folder size={24} />
            <div>
              <strong>No workspace open</strong>
              <span>Create or select a workspace before building a training dataset.</span>
            </div>
          </div>
        ) : (
          <>
            {workflowTabs.length ? (
            <div className="training-tabs" role="tablist" aria-label="Training workflow">
              {workflowTabs.map((tab) => (
                <button
                  aria-controls={activeTab === tab.id ? `training-panel-${tab.id}` : undefined}
                  aria-selected={activeTab === tab.id}
                  className={activeTab === tab.id ? "active" : ""}
                  id={`training-tab-${tab.id}`}
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  onKeyDown={onTabKeyDown}
                  ref={(node) => {
                    tabRefs.current[tab.id] = node;
                  }}
                  role="tab"
                  tabIndex={activeTab === tab.id ? 0 : -1}
                  type="button"
                >
                  <span>{tab.label}</span>
                  <small>{tab.status}</small>
                </button>
              ))}
            </div>
            ) : null}

            <section
              aria-labelledby={workflowTabs.length ? `training-tab-${active.id}` : undefined}
              className="training-panel"
              id={`training-panel-${active.id}`}
              role="tabpanel"
            >
              {datasetLibraryMode || activeTab === "dataset" ? (
                <DatasetEditorPanel
                  active={active}
                  loadingDatasets={loadingDatasets}
                  onRefreshDatasets={onRefreshDatasets}
                  busyDatasetId={busyDatasetId}
                  datasetThumbAsset={datasetThumbAsset}
                  datasets={datasets}
                  startNewDataset={startNewDataset}
                  openDataset={openDataset}
                  activeDataset={activeDataset}
                  selectedDatasetId={selectedDatasetId}
                  datasetsError={datasetsError}
                  datasetError={datasetError}
                  datasetMessage={datasetMessage}
                  draftName={draftName}
                  setDraftName={setDraftName}
                  dirty={dirty}
                  setAddDialogOpen={setAddDialogOpen}
                  renamePrefix={renamePrefix}
                  setRenamePrefix={setRenamePrefix}
                  renaming={renaming}
                  memberAssets={memberAssets}
                  applyOrderedNames={applyOrderedNames}
                  setCaptionDialog={setCaptionDialog}
                  health={health}
                  readiness={readiness}
                  readinessLoading={readinessLoading}
                  readinessByKey={readinessByKey}
                  onToggleItemAck={toggleItemQualityAck}
                  onRemoveDuplicates={removeDuplicates}
                  canSave={canSave}
                  saveDataset={saveDataset}
                  savingDataset={savingDataset}
                  unavailableAssetIds={unavailableAssetIds}
                  removeUnavailableAsset={removeUnavailableAsset}
                  captionDraftById={captionDraftById}
                  onPreview={onPreview}
                  updateCaption={updateCaption}
                  captioning={captioning}
                  addDialogOpen={addDialogOpen}
                  imageAssets={imageAssets}
                  characters={characters}
                  importingAssets={importingAssets}
                  selectedAssetIds={selectedAssetIds}
                  addAssets={addAssets}
                  handleImport={handleImport}
                  captionDialog={captionDialog}
                  gpuOptions={gpuOptions}
                  updateCaptionSetting={updateCaptionSetting}
                  runCaptionJob={runCaptionJob}
                  toggleCaptionExtraOption={toggleCaptionExtraOption}
                  displayedCaptionPrompt={displayedCaptionPrompt}
                  captionSettings={captionSettings}
                  captionModelMissing={captionModelMissing}
                  onDownloadCaptionModel={onDownloadCaptionModel}
                  captionModelSizeLabel={captionModelSizeLabel}
                  captionModelName={captionModel?.name ?? "JoyCaption"}
                />
              ) : null}

              {activeTab === "configure" ? (
                <ConfigureJobPanel
                  active={active}
                  setActiveView={setActiveView}
                  configReady={configReady}
                  trainingTargetsError={trainingTargetsError}
                  trainingPresetsError={trainingPresetsError}
                  configError={configError}
                  configMessage={configMessage}
                  selectedTarget={selectedTarget}
                  setSelectedTargetId={setSelectedTargetId}
                  trainingTargets={trainingTargets}
                  macTargetBlocked={macTargetBlocked}
                  updateSelectedPreset={updateSelectedPreset}
                  selectedPreset={selectedPreset}
                  targetPresets={targetPresets}
                  openDataset={openDataset}
                  activeDataset={activeDataset}
                  datasets={datasets}
                  updateConfigDraft={updateConfigDraft}
                  configDraft={configDraft}
                  outputScopes={outputScopes}
                  visibleQualityPresets={visibleQualityPresets}
                  gpuOptions={gpuOptions}
                  customizedConfigLabels={customizedConfigLabels}
                  showAdvancedConfig={showAdvancedConfig}
                  setShowAdvancedConfig={setShowAdvancedConfig}
                  showNetworkType={showNetworkType}
                  networkTypeOptions={networkTypeOptions}
                  macLokrOnWanBlocked={macLokrOnWanBlocked}
                  isLokrNetwork={isLokrNetwork}
                  visibleOptimizerOptions={visibleOptimizerOptions}
                  visibleLrSchedulerOptions={visibleLrSchedulerOptions}
                  showTrainingAdapter={showTrainingAdapter}
                  visibleTrainingAdapterVersions={visibleTrainingAdapterVersions}
                  visibleResolutionOptions={visibleResolutionOptions}
                  configWarnings={configWarnings}
                  trainingRunMode={trainingRunMode}
                  submittingJob={submittingJob}
                  setTrainingRunMode={setTrainingRunMode}
                  resetConfigDefaults={resetConfigDefaults}
                  submitTrainingJob={submitTrainingJob}
                  configSnapshot={configSnapshot}
                  readiness={readiness}
                  readinessLoading={readinessLoading}
                  readinessBlocksTraining={readinessBlocksTraining}
                  onRemoveDuplicates={removeDuplicates}
                />
              ) : null}
            </section>
          </>
        )}
      </div>
    </section>
    </ModelAvailabilityGate>
  );
}
