import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetThumbnail, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";

const tabs = [
  { id: "dataset", label: "Dataset", title: "Dataset intake", status: "Rust dataset store" },
  { id: "rename-caption", label: "Rename & Caption", title: "Rename and caption pass", status: "Needs valid dataset" },
  { id: "configure", label: "Configure Job", title: "Configure training job", status: "Queue dry run" },
];
const defaultGpuOptions = ["auto"];
const joyCaptionModel = "fancyfeast/llama-joycaption-beta-one-hf-llava";
const joyCaptionTypes = [
  "Descriptive",
  "Descriptive (Casual)",
  "Straightforward",
  "Stable Diffusion Prompt",
  "MidJourney",
  "Danbooru tag list",
  "e621 tag list",
  "Rule34 tag list",
  "Booru-like tag list",
  "Art Critic",
  "Product Listing",
  "Social Media Post",
];
const joyCaptionLengths = [
  "any",
  "very short",
  "short",
  "medium-length",
  "long",
  "very long",
  "20",
  "30",
  "40",
  "50",
  "60",
  "80",
  "100",
  "120",
  "160",
  "200",
  "260",
];
const joyCaptionExtraOptions = [
  { value: "If there is a person/character in the image you must refer to them as {name}.", label: "Use character name" },
  {
    value:
      "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    label: "Avoid fixed traits",
  },
  { value: "Include information about lighting.", label: "Include lighting" },
  { value: "Include information about camera angle.", label: "Include camera angle" },
  { value: "Do NOT include anything sexual; keep it PG.", label: "Keep it PG" },
  { value: "Do NOT mention the image's resolution.", label: "Skip resolution" },
  { value: "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.", label: "Composition style" },
  { value: "Do NOT mention any text that is in the image.", label: "Ignore text" },
  { value: "Specify the depth of field and whether the background is in focus or blurred.", label: "Depth of field" },
  { value: "Do NOT use any ambiguous language.", label: "No ambiguity" },
  { value: "ONLY describe the most important elements of the image.", label: "Important elements only" },
  { value: "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.", label: "Shot size" },
  { value: "Your response will be used by a text-to-image model, so avoid useless meta phrases like \"This image shows...\", \"You are looking at...\", etc.", label: "No meta phrases" },
];
const joyCaptionPromptMap = {
  Descriptive: [
    "Write a detailed description for this image.",
    "Write a detailed description for this image in {word_count} words or less.",
    "Write a {length} detailed description for this image.",
  ],
  "Descriptive (Casual)": [
    "Write a descriptive caption for this image in a casual tone.",
    "Write a descriptive caption for this image in a casual tone within {word_count} words.",
    "Write a {length} descriptive caption for this image in a casual tone.",
  ],
  Straightforward: [
    'Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
    'Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
    'Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is..." or similar phrasing.',
  ],
  "Stable Diffusion Prompt": [
    "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
    "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
  ],
  MidJourney: [
    "Write a MidJourney prompt for this image.",
    "Write a MidJourney prompt for this image within {word_count} words.",
    "Write a {length} MidJourney prompt for this image.",
  ],
  "Danbooru tag list": [
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
  ],
  "e621 tag list": [
    "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
    "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags. Keep it under {word_count} words.",
    "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
  ],
  "Rule34 tag list": [
    "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
    "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags. Keep it under {word_count} words.",
    "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
  ],
  "Booru-like tag list": [
    "Write a list of Booru-like tags for this image.",
    "Write a list of Booru-like tags for this image within {word_count} words.",
    "Write a {length} list of Booru-like tags for this image.",
  ],
  "Art Critic": [
    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
    "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
  ],
  "Product Listing": [
    "Write a caption for this image as though it were a product listing.",
    "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
    "Write a {length} caption for this image as though it were a product listing.",
  ],
  "Social Media Post": [
    "Write a caption for this image as if it were being used for a social media post.",
    "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
    "Write a {length} caption for this image as if it were being used for a social media post.",
  ],
};
const defaultCaptionSettings = {
  captioner: "joy_caption",
  modelNameOrPath: joyCaptionModel,
  recaption: false,
  requestedGpu: "auto",
  captionType: "Descriptive",
  captionLength: "long",
  extraOptions: [],
  nameInput: "",
  temperature: "0.6",
  topP: "0.9",
  maxNewTokens: "256",
  captionPrompt: "",
  lowVram: false,
};

function formatDatasetModality(dataset) {
  return String(dataset.modality ?? "image").replaceAll("_", " ");
}

function datasetItemCount(dataset) {
  const value = Number(dataset.itemCount ?? dataset.items?.length ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function summarizeDatasets(datasets) {
  return datasets.reduce((summary, dataset) => ({ items: summary.items + datasetItemCount(dataset) }), { items: 0 });
}

function imageAssetName(asset) {
  const path = asset?.file?.path ?? asset?.path ?? asset?.displayName ?? asset?.id ?? "asset";
  return String(path).replaceAll("\\", "/").split("/").pop() || "asset";
}

function captionText(item) {
  return String(item?.caption?.text ?? "").trim();
}

function itemFileStem(item) {
  const name = String(item?.path ?? item?.displayName ?? item?.id ?? "item").replaceAll("\\", "/").split("/").pop() || "item";
  const dotIndex = name.lastIndexOf(".");
  return dotIndex > 0 ? name.slice(0, dotIndex) : name;
}

function triggerWordsText(caption) {
  return (caption?.triggerWords ?? caption?.trigger_words ?? []).join(", ");
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

function captionSeedFromName(value) {
  const name = String(value ?? "")
    .replaceAll("\\", "/")
    .split("/")
    .pop()
    ?.replace(/\.[a-z0-9]+$/i, "")
    .replace(/#\s*\d+$/u, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return name || "";
}

function captionSeedForItem(item, asset) {
  const prompt = String(asset?.recipe?.prompt ?? "").trim();
  if (prompt) {
    return prompt;
  }
  return (
    captionSeedFromName(item.displayName) ||
    captionSeedFromName(item.fileStem) ||
    captionSeedFromName(item.path) ||
    "training image"
  );
}

function captionWithTriggerWords(seed, triggerWords) {
  const normalizedSeed = String(seed ?? "").trim();
  const lowerSeed = normalizedSeed.toLowerCase();
  const missingTriggerWords = triggerWords.filter((word) => !lowerSeed.includes(word.toLowerCase()));
  return [...missingTriggerWords, normalizedSeed].filter(Boolean).join(", ");
}

function captionForDraftItem(item, asset, triggerWords) {
  const text = String(item.captionText ?? "").trim();
  if (text) {
    return { source: item.captionSource, text };
  }
  return {
    source: "auto",
    text: captionWithTriggerWords(captionSeedForItem(item, asset), triggerWords),
  };
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

function renameCaptionDrafts(dataset) {
  return (dataset?.items ?? []).map((item, index) => ({
    originalItemId: item.id ?? `item_${String(index + 1).padStart(4, "0")}`,
    itemId: item.id ?? `item_${String(index + 1).padStart(4, "0")}`,
    fileStem: itemFileStem(item),
    displayName: item.displayName ?? imageAssetName(item),
    captionText: item.caption?.text ?? "",
    captionSource: item.caption?.source ?? "manual",
    assetId: item.assetId ?? "",
    path: item.path ?? "",
  }));
}

function renameFieldsDirty(drafts, dataset) {
  const items = dataset?.items ?? [];
  if (drafts.length !== items.length) {
    return true;
  }
  return drafts.some((draft, index) => {
    const item = items[index] ?? {};
    return (
      draft.itemId !== item.id ||
      draft.fileStem !== itemFileStem(item) ||
      draft.displayName !== (item.displayName ?? imageAssetName(item))
    );
  });
}

function normalizeDatasetAssetIds(dataset) {
  return (dataset?.items ?? []).map((item) => item.assetId).filter(Boolean);
}

function datasetHealth({ activeDataset, imageAssets, selectedAssetIds }) {
  const assetsById = new Map(imageAssets.map((asset) => [asset.id, asset]));
  const selectedAssets = selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean);
  const missingAssets = selectedAssetIds.filter((id) => !assetsById.has(id)).length;
  const disabledItems = selectedAssets.filter((asset) => asset.status?.rejected || asset.status?.trashed).length + missingAssets;
  const names = selectedAssets.map((asset) => imageAssetName(asset).toLowerCase());
  const duplicateFilenames = names.filter((name, index) => names.indexOf(name) !== index).length;
  const captionsByAssetId = new Map((activeDataset?.items ?? []).map((item) => [item.assetId, captionText(item)]));
  const missingCaptions = selectedAssetIds.filter((id) => !captionsByAssetId.get(id)).length;
  const valid = selectedAssetIds.length > 0 && disabledItems === 0;

  return {
    disabledItems,
    duplicateFilenames,
    itemCount: selectedAssetIds.length,
    missingCaptions,
    valid,
  };
}

function datasetPayload({ activeDataset, assetsById, name, selectedAssetIds }) {
  const itemsByAssetId = new Map((activeDataset?.items ?? []).map((item) => [item.assetId, item]));
  return {
    name: name.trim(),
    modality: "image",
    items: selectedAssetIds
      .map((assetId) => {
        const asset = assetsById.get(assetId);
        if (!asset) {
          return null;
        }
        const previous = itemsByAssetId.get(assetId);
        return {
          assetId,
          displayName: asset.displayName ?? imageAssetName(asset),
          caption: previous?.caption
            ? {
                text: previous.caption.text ?? "",
                source: previous.caption.source ?? "manual",
                triggerWords: previous.caption.triggerWords ?? [],
              }
            : undefined,
        };
      })
      .filter(Boolean),
  };
}

function asText(value) {
  return value === null || value === undefined ? "" : String(value);
}

function numericDraft(value) {
  return value === null || value === undefined ? "" : String(value);
}

function numberFromDraft(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const number = Number(trimmed);
  return Number.isFinite(number) ? number : null;
}

function boundedNumber(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

function integerFromDraft(value, fallback, min, max) {
  return Math.round(boundedNumber(value, fallback, min, max));
}

function compactObject(object) {
  return Object.fromEntries(
    Object.entries(object).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  );
}

function rangeOptions(limits, key) {
  return Array.isArray(limits?.[key]) ? limits[key] : [];
}

function configDraftFromTarget(target, dataset, gpuOptions, triggerPhrase = "") {
  const defaults = target?.defaults ?? {};
  const advanced = defaults.advanced ?? {};
  const firstGpu = gpuOptions[0] ?? "";
  const requestedGpu = asText(advanced.requestedGpu || firstGpu);
  const outputLabel = outputKindLabel(target);
  return {
    outputName: dataset?.name ? `${dataset.name} ${outputLabel}` : "",
    triggerWord: triggerPhrase || asText(defaults.triggerWord),
    outputScope: asText(advanced.outputScope),
    qualityPreset: asText(advanced.qualityPreset),
    requestedGpu: gpuOptions.includes(requestedGpu) ? requestedGpu : firstGpu,
    rank: numericDraft(defaults.rank),
    alpha: numericDraft(defaults.alpha),
    optimizer: asText(defaults.optimizer),
    learningRate: numericDraft(defaults.learningRate),
    scheduler: asText(advanced.scheduler),
    steps: numericDraft(defaults.steps),
    epochs: numericDraft(advanced.epochs),
    repeats: numericDraft(advanced.repeats),
    resolution: numericDraft(defaults.resolution),
    bucketStrategy: asText(advanced.bucketStrategy),
    precision: asText(advanced.mixedPrecision),
    saveEvery: numericDraft(defaults.saveEvery),
    sampleEvery: numericDraft(advanced.sampleEvery),
    batchSize: numericDraft(defaults.batchSize),
    gradientAccumulation: numericDraft(defaults.gradientAccumulation),
    seed: numericDraft(defaults.seed),
  };
}

function configValidation({ activeDataset, configDraft, selectedTarget }) {
  const warnings = [];
  if (!selectedTarget) {
    warnings.push("Select a training target");
  }
  if (!activeDataset?.id) {
    warnings.push("Select a saved dataset");
  }
  if (!configDraft.outputName?.trim()) {
    warnings.push(`Name the ${outputKindLabel(selectedTarget)} output`);
  }
  if (!configDraft.triggerWord?.trim()) {
    warnings.push("Add a trigger phrase");
  }
  for (const [field, label] of [
    ["rank", "Rank"],
    ["alpha", "Alpha"],
    ["learningRate", "Learning rate"],
    ["steps", "Steps"],
    ["resolution", "Resolution"],
    ["batchSize", "Batch size"],
    ["gradientAccumulation", "Gradient accumulation"],
    ["saveEvery", "Checkpoint cadence"],
  ]) {
    const value = numberFromDraft(configDraft[field]);
    if (!value || value <= 0) {
      warnings.push(`${label} must be greater than zero`);
    }
  }
  return warnings;
}

function outputKindLabel(target) {
  const kind = String(target?.outputKind ?? "output").toLowerCase();
  if (kind === "lora") {
    return "LoRA";
  }
  return kind.replaceAll("_", " ");
}

function trainingConfigSnapshot({ activeDataset, configDraft, selectedTarget, dryRun = true }) {
  const defaults = selectedTarget?.defaults ?? {};
  const advanced = compactObject({
    ...(defaults.advanced ?? {}),
    scheduler: asText(configDraft.scheduler).trim(),
    epochs: numberFromDraft(configDraft.epochs),
    repeats: numberFromDraft(configDraft.repeats),
    bucketStrategy: asText(configDraft.bucketStrategy).trim(),
    mixedPrecision: asText(configDraft.precision).trim(),
    sampleEvery: numberFromDraft(configDraft.sampleEvery),
    qualityPreset: configDraft.qualityPreset,
    outputScope: configDraft.outputScope,
    requestedGpu: configDraft.requestedGpu,
  });
  return {
    targetId: selectedTarget.id,
    datasetId: activeDataset.id,
    datasetVersion: activeDataset.version,
    outputName: configDraft.outputName.trim(),
    dryRun,
    outputScope: configDraft.outputScope,
    qualityPreset: configDraft.qualityPreset,
    requestedGpu: configDraft.requestedGpu,
    config: {
      rank: numberFromDraft(configDraft.rank),
      alpha: numberFromDraft(configDraft.alpha),
      learningRate: numberFromDraft(configDraft.learningRate),
      steps: numberFromDraft(configDraft.steps),
      batchSize: numberFromDraft(configDraft.batchSize),
      gradientAccumulation: numberFromDraft(configDraft.gradientAccumulation),
      resolution: numberFromDraft(configDraft.resolution),
      saveEvery: numberFromDraft(configDraft.saveEvery),
      seed: numberFromDraft(configDraft.seed),
      optimizer: asText(configDraft.optimizer).trim(),
      triggerWord: asText(configDraft.triggerWord).trim(),
      advanced,
    },
  };
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

function buildJoyCaptionPrompt(settings) {
  const captionLength = String(settings.captionLength || "long");
  let templateIndex = 2;
  if (captionLength === "any") {
    templateIndex = 0;
  } else if (/^\d+$/.test(captionLength)) {
    templateIndex = 1;
  }
  const templates = joyCaptionPromptMap[settings.captionType] ?? joyCaptionPromptMap.Descriptive;
  const extraOptions = Array.isArray(settings.extraOptions) ? settings.extraOptions : [];
  const prompt = [templates[templateIndex], ...extraOptions].filter(Boolean).join(" ");
  return prompt
    .replaceAll("{name}", String(settings.nameInput || "{NAME}"))
    .replaceAll("{length}", captionLength)
    .replaceAll("{word_count}", captionLength);
}

function DatasetHealth({ health }) {
  return (
    <div className="training-health-grid" aria-label="Dataset health">
      <div>
        <strong>{health.itemCount}</strong>
        <span>Items</span>
      </div>
      <div className={health.missingCaptions ? "needs-attention" : ""}>
        <strong>{health.missingCaptions}</strong>
        <span>Missing captions</span>
      </div>
      <div className={health.duplicateFilenames ? "needs-attention" : ""}>
        <strong>{health.duplicateFilenames}</strong>
        <span>Duplicate filenames</span>
      </div>
      <div className={health.disabledItems ? "needs-attention" : ""}>
        <strong>{health.disabledItems}</strong>
        <span>Disabled items</span>
      </div>
    </div>
  );
}

export function TrainingStudio({
  activeProject,
  authenticated = true,
  assets = [],
  batchRenameDataset = async () => null,
  createCaptionJob = async () => null,
  createDataset = async () => null,
  createTrainingJob = async () => null,
  datasets = [],
  datasetsError = "",
  gpuOptions = defaultGpuOptions,
  importAsset = async () => null,
  loadDataset = async () => null,
  loadingDatasets = false,
  onPreview = () => {},
  onRefreshDatasets = () => {},
  prepareTrainingConfig = async (snapshot) => snapshot,
  trainingTargets = [],
  trainingTargetsError = "",
  updateDataset = async () => null,
  writeCaptionSidecars = async () => null,
}) {
  const [activeTab, setActiveTab] = useState("dataset");
  const [activeDataset, setActiveDataset] = useState(null);
  const [datasetError, setDatasetError] = useState("");
  const [datasetMessage, setDatasetMessage] = useState("");
  const [draftName, setDraftName] = useState("");
  const [busyDatasetId, setBusyDatasetId] = useState("");
  const [importingAssets, setImportingAssets] = useState(false);
  const [savingDataset, setSavingDataset] = useState(false);
  const [selectedAssetIds, setSelectedAssetIds] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [renamePrefix, setRenamePrefix] = useState("");
  const [captionTriggerWords, setCaptionTriggerWords] = useState("");
  const [renameCaptionDraftItems, setRenameCaptionDraftItems] = useState([]);
  const [savingRenameCaption, setSavingRenameCaption] = useState(false);
  const [captionSettings, setCaptionSettings] = useState(defaultCaptionSettings);
  const [selectedTargetId, setSelectedTargetId] = useState("");
  const [configDraft, setConfigDraft] = useState({});
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [configSnapshot, setConfigSnapshot] = useState(null);
  const [configMessage, setConfigMessage] = useState("");
  const [configError, setConfigError] = useState("");
  const [configTriggerFollowsCaptions, setConfigTriggerFollowsCaptions] = useState(true);
  const [preparingConfig, setPreparingConfig] = useState(false);
  const [submittingJob, setSubmittingJob] = useState(false);
  // Dry run validates the Rust-resolved plan without training; a real run hands
  // the plan to the worker's Z-Image LoRA kernel. Default to the safe dry run.
  const [trainingRunMode, setTrainingRunMode] = useState("dry_run");
  const configBasisRef = useRef("");
  const tabRefs = useRef({});

  const activeIndex = tabs.findIndex((tab) => tab.id === activeTab);
  const active = tabs[activeIndex] ?? tabs[0];
  const datasetSummary = useMemo(() => summarizeDatasets(datasets), [datasets]);
  const imageAssets = useMemo(() => assets.filter((asset) => assetCanRenderAsImage(asset)), [assets]);
  const assetsById = useMemo(() => new Map(imageAssets.map((asset) => [asset.id, asset])), [imageAssets]);
  const unavailableAssetIds = useMemo(
    () => selectedAssetIds.filter((assetId) => !assetsById.has(assetId)),
    [assetsById, selectedAssetIds],
  );
  const health = useMemo(
    () => datasetHealth({ activeDataset, imageAssets, selectedAssetIds }),
    [activeDataset, imageAssets, selectedAssetIds],
  );
  const originalAssetIds = useMemo(() => normalizeDatasetAssetIds(activeDataset), [activeDataset]);
  const dirty =
    Boolean(activeDataset) &&
    (draftName.trim() !== activeDataset.name ||
      selectedAssetIds.length !== originalAssetIds.length ||
      selectedAssetIds.some((id, index) => id !== originalAssetIds[index]));
  const canSave =
    draftName.trim().length > 0 &&
    selectedAssetIds.length > 0 &&
    health.disabledItems === 0 &&
    !savingDataset &&
    (!activeDataset || dirty);
  const renameCaptionHasDraft = renameCaptionDraftItems.length > 0;
  const renameCaptionHasInvalidDraft = renameCaptionDraftItems.some(
    (item) => !item.itemId.trim() || !item.fileStem.trim() || !item.displayName.trim(),
  );
  const missingDraftCaptions = renameCaptionDraftItems.filter((item) => !String(item.captionText ?? "").trim()).length;
  const displayedCaptionPrompt = captionSettings.captionPrompt || buildJoyCaptionPrompt(captionSettings);
  const canSaveRenameCaption =
    Boolean(activeDataset?.id) && renameCaptionHasDraft && !renameCaptionHasInvalidDraft && !savingRenameCaption;
  const firstTarget = trainingTargets[0] ?? null;
  const selectedTarget = useMemo(
    () => trainingTargets.find((target) => target.id === selectedTargetId) ?? firstTarget,
    [firstTarget, selectedTargetId, trainingTargets],
  );
  const qualityPresets = rangeOptions(selectedTarget?.limits, "qualityPresets");
  const outputScopes = rangeOptions(selectedTarget?.limits, "outputScopes");
  const resolutionOptions = rangeOptions(selectedTarget?.limits, "resolutions");
  const gpuOptionsKey = gpuOptions.join("\u0000");
  const configWarnings = configValidation({ activeDataset, configDraft, selectedTarget });
  const canPrepareConfig = configWarnings.length === 0 && !preparingConfig;

  useEffect(() => {
    setActiveDataset(null);
    setDatasetError("");
    setDatasetMessage("");
    setDraftName("");
    setSelectedAssetIds([]);
    setSelectedDatasetId("");
    setRenamePrefix("");
    setCaptionTriggerWords("");
    setRenameCaptionDraftItems([]);
    setCaptionSettings(defaultCaptionSettings);
    setConfigDraft({});
    setConfigSnapshot(null);
    setConfigMessage("");
    setConfigError("");
    setConfigTriggerFollowsCaptions(true);
    configBasisRef.current = "";
  }, [activeProject?.id]);

  useEffect(() => {
    const datasetTriggerPhrase = triggerPhraseFromText(activeDataset?.name);
    setRenameCaptionDraftItems(renameCaptionDrafts(activeDataset));
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
    const basis = `${selectedTarget.id}\u0000${activeDataset?.id ?? ""}`;
    if (configBasisRef.current === basis) {
      return;
    }
    configBasisRef.current = basis;
    setConfigDraft(configDraftFromTarget(selectedTarget, activeDataset, gpuOptions, triggerPhraseFromText(activeDataset?.name)));
    setConfigSnapshot(null);
    setConfigMessage("");
    setConfigError("");
    setConfigTriggerFollowsCaptions(true);
  }, [activeDataset?.id, selectedTarget?.id]);

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
    const next = tabs[(index + tabs.length) % tabs.length];
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
      focusTab(tabs.length - 1);
    }
  }

  async function openDataset(datasetId) {
    if (!datasetId) {
      setActiveDataset(null);
      setDraftName("");
      setSelectedAssetIds([]);
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
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset));
      setSelectedDatasetId(dataset?.id ?? datasetId);
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setBusyDatasetId("");
    }
  }

  function toggleAsset(assetId) {
    setDatasetMessage("");
    setSelectedAssetIds((current) =>
      current.includes(assetId) ? current.filter((id) => id !== assetId) : [...current, assetId],
    );
  }

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
    setSelectedDatasetId("");
  }

  function updateRenameCaptionDraft(originalItemId, patch) {
    setDatasetMessage("");
    setRenameCaptionDraftItems((current) =>
      current.map((item) => (item.originalItemId === originalItemId ? { ...item, ...patch } : item)),
    );
  }

  function updateCaptionTriggerWords(value) {
    setDatasetMessage("");
    if (configTriggerFollowsCaptions) {
      setConfigMessage("");
      setConfigError("");
    }
    setCaptionTriggerWords(value);
  }

  function updateConfigDraft(field, value) {
    setConfigMessage("");
    setConfigError("");
    setConfigSnapshot(null);
    if (field === "triggerWord") {
      setConfigTriggerFollowsCaptions(false);
    }
    setConfigDraft((current) => ({ ...current, [field]: value }));
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
    setConfigDraft(configDraftFromTarget(selectedTarget, activeDataset, gpuOptions, triggerPhraseFromText(captionTriggerWords)));
    setConfigTriggerFollowsCaptions(true);
    setConfigSnapshot(null);
    setConfigMessage("Defaults restored");
    setConfigError("");
  }

  function applyOrderedNames() {
    setDatasetMessage("");
    setRenameCaptionDraftItems((current) =>
      current.map((item, index) => {
        const nextName = orderedName(index, renamePrefix || activeDataset?.name);
        const extension = String(item.path).split(".").pop() || "png";
        return {
          ...item,
          itemId: nextName,
          fileStem: nextName,
          displayName: `${nextName}.${extension}`,
        };
      }),
    );
  }

  async function handleImport(event) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) {
      return;
    }
    setImportingAssets(true);
    setDatasetError("");
    try {
      const imported = [];
      for (const file of files) {
        const asset = await importAsset(file);
        if (asset?.id) {
          imported.push(asset.id);
        }
      }
      if (imported.length) {
        setSelectedAssetIds((current) => Array.from(new Set([...current, ...imported])));
      } else {
        setDatasetError("No images were imported.");
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setImportingAssets(false);
      event.target.value = "";
    }
  }

  async function saveDataset() {
    if (!canSave) {
      return;
    }
    setSavingDataset(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const payload = datasetPayload({ activeDataset, assetsById, name: draftName, selectedAssetIds });
      const dataset = activeDataset
        ? await updateDataset(activeDataset.id, payload)
        : await createDataset(payload);
      setActiveDataset(dataset);
      setDraftName(dataset?.name ?? draftName.trim());
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset));
      setSelectedDatasetId(dataset?.id ?? "");
      setDatasetMessage(activeDataset ? "Dataset changes saved" : "Dataset created");
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setSavingDataset(false);
    }
  }

  async function saveRenameCaption() {
    if (!canSaveRenameCaption) {
      return;
    }
    setSavingRenameCaption(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      let dataset = activeDataset;
      const renameItems = renameCaptionDraftItems.map((item) => ({
        itemId: item.originalItemId,
        newItemId: item.itemId.trim(),
        fileStem: item.fileStem.trim(),
        displayName: item.displayName.trim(),
      }));
      if (renameFieldsDirty(renameCaptionDraftItems, activeDataset)) {
        dataset = await batchRenameDataset(activeDataset.id, { items: renameItems });
      }
      const useJoyCaption = captionSettings.captioner === "joy_caption";
      const datasetTriggerWords = parseTriggerWords(captionTriggerWords);
      const captionItems = renameCaptionDraftItems.map((item) => {
        const asset = assetsById.get(item.assetId);
        const caption = useJoyCaption
          ? { source: item.captionSource, text: String(item.captionText ?? "") }
          : captionForDraftItem(item, asset, datasetTriggerWords);
        return {
          itemId: item.itemId.trim(),
          caption: {
            text: caption.text,
            source: caption.source,
            triggerWords: datasetTriggerWords,
          },
        };
      });
      const result = await writeCaptionSidecars(activeDataset.id, {
        items: captionItems,
      });
      const nextDataset = result?.dataset ?? dataset;
      setActiveDataset(nextDataset);
      setDraftName(nextDataset?.name ?? draftName);
      setSelectedAssetIds(normalizeDatasetAssetIds(nextDataset));
      setSelectedDatasetId(nextDataset?.id ?? activeDataset.id);
      if (useJoyCaption && (captionSettings.recaption || missingDraftCaptions > 0)) {
        const job = await createCaptionJob(activeDataset.id, trainingCaptionJobPayload(captionSettings));
        setDatasetMessage(`Caption job queued${job?.id ? ` (${job.id})` : ""}. Track it in the Queue.`);
      } else {
        setDatasetMessage(`Captions created${result?.sidecars?.length ? ` (${result.sidecars.length})` : ""}`);
      }
    } catch (err) {
      setDatasetError(err.message);
    } finally {
      setSavingRenameCaption(false);
    }
  }

  async function prepareConfig() {
    if (!canPrepareConfig) {
      return;
    }
    setPreparingConfig(true);
    setConfigError("");
    setConfigMessage("");
    try {
      const dryRun = trainingRunMode === "dry_run";
      const snapshot = trainingConfigSnapshot({ activeDataset, configDraft, selectedTarget, dryRun });
      const result = await prepareTrainingConfig(snapshot);
      setConfigSnapshot(result ?? snapshot);
      setConfigMessage("Config snapshot ready");
    } catch (err) {
      setConfigError(err.message);
    } finally {
      setPreparingConfig(false);
    }
  }

  async function submitTrainingJob() {
    if (!canPrepareConfig || submittingJob) {
      return;
    }
    setSubmittingJob(true);
    setConfigError("");
    setConfigMessage("");
    try {
      const dryRun = trainingRunMode === "dry_run";
      const snapshot = trainingConfigSnapshot({ activeDataset, configDraft, selectedTarget, dryRun });
      const job = await createTrainingJob({
        targetId: snapshot.targetId,
        datasetId: snapshot.datasetId,
        datasetVersion: snapshot.datasetVersion,
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
    <section className="main-surface training-studio">
      <div className="training-studio-shell">
        <div className="training-summary-band">
          <div className="section-heading">
            <p className="eyebrow">Training Studio</p>
            <h2>Native LoRA training workflow</h2>
            <p className="view-copy">
              Build datasets, normalize captions, and prepare a Rust-owned training plan before any ML runtime work begins.
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

        {!authenticated ? (
          <div className="training-empty-state" role="status">
            <Icon.Train size={24} />
            <div>
              <strong>Pairing required</strong>
              <span>Unlock SceneWorks to load project training datasets.</span>
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
            <div className="training-tabs" role="tablist" aria-label="Training workflow">
              {tabs.map((tab) => (
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

            <section
              aria-labelledby={`training-tab-${active.id}`}
              className="training-panel"
              id={`training-panel-${active.id}`}
              role="tabpanel"
            >
              {activeTab === "dataset" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Dataset</p>
                      <h3>{active.title}</h3>
                    </div>
                    <div className="training-head-actions">
                      <button className="secondary-action" disabled={loadingDatasets} onClick={onRefreshDatasets} type="button">
                        <Icon.Search size={14} />
                        {loadingDatasets ? "Refreshing" : "Refresh"}
                      </button>
                      <button className="primary-action" onClick={startNewDataset} type="button">
                        <Icon.Plus size={14} />
                        New
                      </button>
                    </div>
                  </div>
                  {datasetsError ? <p className="inline-warning">{datasetsError}</p> : null}
                  {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
                  {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}
                  <div className="training-dataset-workspace">
                    <aside className="training-dataset-list-panel">
                      {loadingDatasets ? <div className="empty-panel compact-panel">Loading training datasets</div> : null}
                      {!loadingDatasets && datasets.length === 0 ? <div className="empty-panel compact-panel">No training datasets yet</div> : null}
                      {datasets.map((dataset) => {
                        const itemCount = datasetItemCount(dataset);
                        return (
                          <button
                            aria-pressed={selectedDatasetId === dataset.id}
                            className={selectedDatasetId === dataset.id ? "training-dataset-row active" : "training-dataset-row"}
                            disabled={busyDatasetId === dataset.id}
                            key={dataset.id}
                            onClick={() => openDataset(dataset.id)}
                            type="button"
                          >
                            <div>
                              <strong>{dataset.name ?? dataset.id}</strong>
                              <span>{formatDatasetModality(dataset)} dataset</span>
                            </div>
                            <span>{busyDatasetId === dataset.id ? "Opening" : `${itemCount} item${itemCount === 1 ? "" : "s"}`}</span>
                          </button>
                        );
                      })}
                    </aside>

                    <div className="training-dataset-editor">
                      <div className="training-dataset-form">
                        <label>
                          Dataset name
                          <input
                            onChange={(event) => setDraftName(event.target.value)}
                            placeholder="Character portrait set"
                            value={draftName}
                          />
                        </label>
                        <label>
                          Modality
                          <select disabled value="image">
                            <option value="image">Image</option>
                          </select>
                        </label>
                        <label className="file-upload-button training-import-button">
                          <input accept="image/*" disabled={importingAssets} multiple onChange={handleImport} type="file" />
                          {importingAssets ? "Importing" : "Import images"}
                        </label>
                      </div>
                      <DatasetHealth health={health} />
                      <div className="training-validity">
                        <span className={health.valid ? "training-valid-dot valid" : "training-valid-dot"} />
                        <span>{health.valid ? "Dataset is ready for downstream steps" : "Add image assets and remove disabled items"}</span>
                      </div>
                      {unavailableAssetIds.length ? (
                        <div className="training-unavailable-list" aria-label="Unavailable dataset items">
                          {unavailableAssetIds.map((assetId) => (
                            <div className="training-unavailable-item" key={assetId}>
                              <div>
                                <strong>{assetId}</strong>
                                <span>Asset is no longer available</span>
                              </div>
                              <button className="secondary-action" onClick={() => removeUnavailableAsset(assetId)} type="button">
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      ) : null}
                      <div className="training-asset-picker" aria-label="Training dataset image assets">
                        {imageAssets.length ? (
                          imageAssets.map((asset) => {
                            const selected = selectedAssetIds.includes(asset.id);
                            const disabled = asset.status?.rejected || asset.status?.trashed;
                            return (
                              <article
                                className={[
                                  "training-asset-card",
                                  selected ? "selected" : "",
                                  disabled ? "disabled" : "",
                                ]
                                  .filter(Boolean)
                                  .join(" ")}
                                key={asset.id}
                              >
                                <button onClick={() => onPreview(asset)} type="button">
                                  <AssetThumbnail asset={asset} />
                                </button>
                                <label>
                                  <input checked={selected} onChange={() => toggleAsset(asset.id)} type="checkbox" />
                                  <span>{asset.displayName ?? imageAssetName(asset)}</span>
                                </label>
                                {disabled ? (
                                  <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
                                ) : null}
                              </article>
                            );
                          })
                        ) : (
                          <div className="empty-panel compact-panel">Import or create project images before building a dataset</div>
                        )}
                      </div>
                      <div className="training-dataset-actions">
                        <button className="primary-action" disabled={!canSave} onClick={saveDataset} type="button">
                          {savingDataset ? "Saving" : activeDataset ? "Save dataset" : "Create dataset"}
                        </button>
                        <span>{dirty ? "Unsaved changes" : activeDataset ? `Version ${activeDataset.version}` : "Draft"}</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : null}

              {activeTab === "rename-caption" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Rename & Caption</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">{activeDataset ? `Version ${activeDataset.version}` : "Select dataset"}</span>
                  </div>
                  {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
                  {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}
                  {!activeDataset ? (
                    <div className="empty-panel compact-panel">Open a saved dataset to edit captions.</div>
                  ) : (
                    <div className="training-caption-workspace">
                      <div className="training-caption-editor">
                        <div className="training-caption-list" aria-label="Rename and caption dataset items">
                        {renameCaptionDraftItems.map((item, index) => {
                          const asset = assetsById.get(item.assetId);
                          return (
                            <article className="training-caption-row" key={item.originalItemId}>
                              <div className="training-caption-preview">
                                <button disabled={!asset} onClick={() => asset && onPreview(asset)} type="button">
                                  {asset ? <AssetThumbnail asset={asset} /> : <Icon.Image size={22} />}
                                </button>
                                <span>{String(index + 1).padStart(2, "0")}</span>
                              </div>
                              <div className="training-caption-fields">
                                <label>
                                  Item ID
                                  <input
                                    onChange={(event) => updateRenameCaptionDraft(item.originalItemId, { itemId: event.target.value })}
                                    value={item.itemId}
                                  />
                                </label>
                                <label>
                                  File stem
                                  <input
                                    onChange={(event) => updateRenameCaptionDraft(item.originalItemId, { fileStem: event.target.value })}
                                    value={item.fileStem}
                                  />
                                </label>
                                <label>
                                  Display name
                                  <input
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { displayName: event.target.value })
                                    }
                                    value={item.displayName}
                                  />
                                </label>
                                <label className="training-caption-text">
                                  Caption
                                  <textarea
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { captionText: event.target.value })
                                    }
                                    rows={3}
                                    value={item.captionText}
                                  />
                                </label>
                                <label>
                                  Source
                                  <select
                                    onChange={(event) =>
                                      updateRenameCaptionDraft(item.originalItemId, { captionSource: event.target.value })
                                    }
                                    value={item.captionSource}
                                  >
                                    <option value="manual">Manual</option>
                                    <option value="imported">Imported</option>
                                    <option value="auto">Auto</option>
                                  </select>
                                </label>
                              </div>
                            </article>
                          );
                        })}
                        </div>
                      </div>
                      <aside className="training-caption-sidebar" aria-label="Caption options">
                        <div className="training-caption-sidebar-section">
                          <div className="training-sidebar-heading">
                            <span>Files</span>
                            <strong>{renameCaptionDraftItems.length}</strong>
                          </div>
                          <label>
                            Rename prefix
                            <input onChange={(event) => setRenamePrefix(event.target.value)} value={renamePrefix} />
                          </label>
                          <label>
                            Trigger words
                            <input onChange={(event) => updateCaptionTriggerWords(event.target.value)} value={captionTriggerWords} />
                          </label>
                          <button className="secondary-action" onClick={applyOrderedNames} type="button">
                            <Icon.Sliders size={14} />
                            Apply ordered names
                          </button>
                        </div>
                        <div className="training-caption-sidebar-section">
                          <div className="training-sidebar-heading">
                            <span>Captioner</span>
                            <strong>{missingDraftCaptions}</strong>
                          </div>
                          <label>
                            Method
                            <select
                              onChange={(event) => updateCaptionSetting("captioner", event.target.value)}
                              value={captionSettings.captioner}
                            >
                              <option value="joy_caption">Joy Caption</option>
                              <option value="metadata">Metadata fallback</option>
                            </select>
                          </label>
                          {captionSettings.captioner === "joy_caption" ? (
                            <>
                              <label>
                                Model
                                <input
                                  onChange={(event) => updateCaptionSetting("modelNameOrPath", event.target.value)}
                                  value={captionSettings.modelNameOrPath}
                                />
                              </label>
                              <label>
                                GPU
                                <select
                                  onChange={(event) => updateCaptionSetting("requestedGpu", event.target.value)}
                                  value={captionSettings.requestedGpu}
                                >
                                  {gpuOptions.map((gpu) => (
                                    <option key={gpu} value={gpu}>
                                      {gpu}
                                    </option>
                                  ))}
                                </select>
                              </label>
                              <label className="training-toggle-line">
                                <input
                                  checked={captionSettings.recaption}
                                  onChange={(event) => updateCaptionSetting("recaption", event.target.checked)}
                                  type="checkbox"
                                />
                                <span>Recaption existing</span>
                              </label>
                              <label className="training-toggle-line">
                                <input
                                  checked={captionSettings.lowVram}
                                  onChange={(event) => updateCaptionSetting("lowVram", event.target.checked)}
                                  type="checkbox"
                                />
                                <span>Low VRAM</span>
                              </label>
                            </>
                          ) : null}
                        </div>
                        {captionSettings.captioner === "joy_caption" ? (
                          <>
                            <div className="training-caption-sidebar-section">
                              <div className="training-sidebar-heading">
                                <span>Prompt</span>
                              </div>
                              <label>
                                Type
                                <select
                                  onChange={(event) => updateCaptionSetting("captionType", event.target.value)}
                                  value={captionSettings.captionType}
                                >
                                  {joyCaptionTypes.map((type) => (
                                    <option key={type} value={type}>
                                      {type}
                                    </option>
                                  ))}
                                </select>
                              </label>
                              <label>
                                Length
                                <select
                                  onChange={(event) => updateCaptionSetting("captionLength", event.target.value)}
                                  value={captionSettings.captionLength}
                                >
                                  {joyCaptionLengths.map((length) => (
                                    <option key={length} value={length}>
                                      {length}
                                    </option>
                                  ))}
                                </select>
                              </label>
                              <label>
                                Character Name
                                <input
                                  onChange={(event) => updateCaptionSetting("nameInput", event.target.value)}
                                  value={captionSettings.nameInput}
                                />
                              </label>
                              <label>
                                Caption prompt
                                <textarea
                                  onChange={(event) => updateCaptionSetting("captionPrompt", event.target.value)}
                                  rows={8}
                                  value={displayedCaptionPrompt}
                                />
                              </label>
                            </div>
                            <div className="training-caption-sidebar-section">
                              <div className="training-sidebar-heading">
                                <span>Sampling</span>
                              </div>
                              <label>
                                Temperature
                                <input
                                  max="2"
                                  min="0"
                                  onChange={(event) => updateCaptionSetting("temperature", event.target.value)}
                                  step="0.05"
                                  type="number"
                                  value={captionSettings.temperature}
                                />
                              </label>
                              <label>
                                Top P
                                <input
                                  max="1"
                                  min="0"
                                  onChange={(event) => updateCaptionSetting("topP", event.target.value)}
                                  step="0.05"
                                  type="number"
                                  value={captionSettings.topP}
                                />
                              </label>
                              <label>
                                Max tokens
                                <input
                                  max="1024"
                                  min="1"
                                  onChange={(event) => updateCaptionSetting("maxNewTokens", event.target.value)}
                                  step="1"
                                  type="number"
                                  value={captionSettings.maxNewTokens}
                                />
                              </label>
                            </div>
                            <div className="training-caption-sidebar-section">
                              <div className="training-sidebar-heading">
                                <span>Options</span>
                              </div>
                              <div className="training-caption-option-list">
                                {joyCaptionExtraOptions.map((option) => (
                                  <label className="training-toggle-line" key={option.value}>
                                    <input
                                      checked={captionSettings.extraOptions.includes(option.value)}
                                      onChange={() => toggleCaptionExtraOption(option.value)}
                                      type="checkbox"
                                    />
                                    <span>{option.label}</span>
                                  </label>
                                ))}
                              </div>
                            </div>
                          </>
                        ) : null}
                        <button
                          className="primary-action training-caption-submit"
                          disabled={!canSaveRenameCaption}
                          onClick={saveRenameCaption}
                          type="button"
                        >
                          {savingRenameCaption ? "Creating" : "Create Captions"}
                        </button>
                      </aside>
                      </div>
                  )}
                </>
              ) : null}

              {activeTab === "configure" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Configure Job</p>
                      <h3>{active.title}</h3>
                    </div>
                    <span className="training-status-pill">{canPrepareConfig ? "Ready" : "Needs input"}</span>
                  </div>
                  {trainingTargetsError ? <p className="inline-warning">{trainingTargetsError}</p> : null}
                  {configError ? <p className="inline-warning">{configError}</p> : null}
                  {configMessage ? <p className="inline-success">{configMessage}</p> : null}
                  {!selectedTarget ? (
                    <div className="empty-panel compact-panel">Training target registry unavailable</div>
                  ) : (
                    <div className="training-config-form" aria-label="Training job configuration">
                      <div className="training-config-grid">
                        <label>
                          Target
                          <select onChange={(event) => setSelectedTargetId(event.target.value)} value={selectedTarget.id}>
                            {trainingTargets.map((target) => (
                              <option key={target.id} value={target.id}>
                                {target.ui?.label ?? target.name}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Base model
                          <input disabled readOnly value={selectedTarget.baseModel ?? ""} />
                        </label>
                        <label>
                          Dataset
                          <select onChange={(event) => openDataset(event.target.value)} value={activeDataset?.id ?? ""}>
                            <option value="">Select a saved dataset</option>
                            {datasets.map((dataset) => (
                              <option key={dataset.id} value={dataset.id}>
                                {dataset.name ?? dataset.id}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          LoRA name
                          <input onChange={(event) => updateConfigDraft("outputName", event.target.value)} value={configDraft.outputName ?? ""} />
                        </label>
                        <label>
                          Trigger phrase
                          <input onChange={(event) => updateConfigDraft("triggerWord", event.target.value)} value={configDraft.triggerWord ?? ""} />
                        </label>
                        <label>
                          Output scope
                          <select onChange={(event) => updateConfigDraft("outputScope", event.target.value)} value={configDraft.outputScope ?? ""}>
                            {outputScopes.length ? null : <option value={configDraft.outputScope ?? ""}>{configDraft.outputScope || "Default"}</option>}
                            {outputScopes.map((scope) => (
                              <option key={scope} value={scope}>
                                {scope}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Quality preset
                          <select
                            onChange={(event) => updateConfigDraft("qualityPreset", event.target.value)}
                            value={configDraft.qualityPreset ?? ""}
                          >
                            {qualityPresets.length ? null : (
                              <option value={configDraft.qualityPreset ?? ""}>{configDraft.qualityPreset || "Default"}</option>
                            )}
                            {qualityPresets.map((preset) => (
                              <option key={preset} value={preset}>
                                {preset}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Requested GPU
                          <select onChange={(event) => updateConfigDraft("requestedGpu", event.target.value)} value={configDraft.requestedGpu ?? ""}>
                            {gpuOptions.map((gpu) => (
                              <option key={gpu} value={gpu}>
                                {gpu === "auto" ? "Auto" : `GPU ${gpu}`}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>

                      <details
                        className="training-advanced-panel"
                        onToggle={(event) => setShowAdvancedConfig(event.currentTarget.open)}
                        open={showAdvancedConfig}
                      >
                        <summary>
                          <Icon.Sliders size={14} />
                          Advanced
                        </summary>
                        <div className="training-advanced-grid">
                          <label>
                            Rank
                            <input onChange={(event) => updateConfigDraft("rank", event.target.value)} type="number" value={configDraft.rank ?? ""} />
                          </label>
                          <label>
                            Alpha
                            <input onChange={(event) => updateConfigDraft("alpha", event.target.value)} type="number" value={configDraft.alpha ?? ""} />
                          </label>
                          <label>
                            Optimizer
                            <input onChange={(event) => updateConfigDraft("optimizer", event.target.value)} value={configDraft.optimizer ?? ""} />
                          </label>
                          <label>
                            Learning rate
                            <input
                              onChange={(event) => updateConfigDraft("learningRate", event.target.value)}
                              step="0.00001"
                              type="number"
                              value={configDraft.learningRate ?? ""}
                            />
                          </label>
                          <label>
                            Scheduler
                            <input onChange={(event) => updateConfigDraft("scheduler", event.target.value)} value={configDraft.scheduler ?? ""} />
                          </label>
                          <label>
                            Steps
                            <input onChange={(event) => updateConfigDraft("steps", event.target.value)} type="number" value={configDraft.steps ?? ""} />
                          </label>
                          <label>
                            Epochs
                            <input onChange={(event) => updateConfigDraft("epochs", event.target.value)} type="number" value={configDraft.epochs ?? ""} />
                          </label>
                          <label>
                            Repeats
                            <input onChange={(event) => updateConfigDraft("repeats", event.target.value)} type="number" value={configDraft.repeats ?? ""} />
                          </label>
                          <label>
                            Resolution
                            <select onChange={(event) => updateConfigDraft("resolution", event.target.value)} value={configDraft.resolution ?? ""}>
                              {resolutionOptions.length ? null : <option value={configDraft.resolution ?? ""}>{configDraft.resolution ?? ""}</option>}
                              {resolutionOptions.map((resolution) => (
                                <option key={resolution} value={resolution}>
                                  {resolution}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            Buckets
                            <input
                              onChange={(event) => updateConfigDraft("bucketStrategy", event.target.value)}
                              value={configDraft.bucketStrategy ?? ""}
                            />
                          </label>
                          <label>
                            Precision
                            <input onChange={(event) => updateConfigDraft("precision", event.target.value)} value={configDraft.precision ?? ""} />
                          </label>
                          <label>
                            Checkpoint cadence
                            <input
                              onChange={(event) => updateConfigDraft("saveEvery", event.target.value)}
                              type="number"
                              value={configDraft.saveEvery ?? ""}
                            />
                          </label>
                          <label>
                            Sample cadence
                            <input
                              onChange={(event) => updateConfigDraft("sampleEvery", event.target.value)}
                              type="number"
                              value={configDraft.sampleEvery ?? ""}
                            />
                          </label>
                        </div>
                      </details>

                      {configWarnings.length ? (
                        <div className="training-config-warnings" aria-label="Configuration warnings">
                          {configWarnings.map((warning) => (
                            <span key={warning}>{warning}</span>
                          ))}
                        </div>
                      ) : null}

                      <div className="training-config-actions">
                        <label className="training-run-mode">
                          <span>Run mode</span>
                          <select
                            aria-label="Training run mode"
                            disabled={submittingJob}
                            onChange={(event) => setTrainingRunMode(event.target.value)}
                            value={trainingRunMode}
                          >
                            <option value="dry_run">Validate (dry run)</option>
                            <option value="real">Run training (beta)</option>
                          </select>
                        </label>
                        <button className="secondary-action" onClick={resetConfigDefaults} type="button">
                          Reset defaults
                        </button>
                        <button className="secondary-action" disabled={!canPrepareConfig} onClick={prepareConfig} type="button">
                          {preparingConfig ? "Preparing" : "Prepare config"}
                        </button>
                        <button
                          className="primary-action"
                          disabled={!canPrepareConfig || submittingJob}
                          onClick={submitTrainingJob}
                          type="button"
                        >
                          {submittingJob
                            ? "Queuing"
                            : trainingRunMode === "dry_run"
                              ? "Queue dry-run job"
                              : "Start training"}
                        </button>
                      </div>
                      {configSnapshot ? <pre className="training-config-snapshot">{JSON.stringify(configSnapshot, null, 2)}</pre> : null}
                    </div>
                  )}
                  <p className="view-copy">
                    A dry run validates the Rust-resolved training plan and dataset on a GPU worker without training. Run training
                    (beta) hands the same plan to the worker's Z-Image LoRA kernel to produce a real .safetensors adapter.
                  </p>
                </>
              ) : null}
            </section>
          </>
        )}
      </div>
    </section>
  );
}
