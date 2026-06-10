import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAppContext } from "../context/AppContext.js";
import { DEFAULT_MAC_CAPABILITIES, macTrainingKernelBlocked } from "../macGating.js";
import { API_BASE_URL } from "../api.js";
import { AssetThumbnail, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { CompactSelector } from "../components/CompactSelector.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
import { DatasetCaptionDialog } from "../components/DatasetCaptionDialog.jsx";
import { Icon } from "../components/Icons.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";

const trainingTabs = [
  { id: "configure", label: "Configure Job", title: "Configure training job", status: "Queue dry run" },
];
const defaultGpuOptions = ["auto"];
const defaultOptimizerOptions = ["adamw8bit", "adamw", "adam", "prodigyopt", "rose"];
const timestepTypeOptions = ["sigmoid", "linear", "weighted"];
const timestepBiasOptions = ["balanced", "high_noise", "low_noise"];
const lossTypeOptions = ["mse", "mae"];
// Learning-rate schedulers the worker actually honors (constant holds the LR
// fixed; linear/cosine decay it over the run). Distinct from the timestep/noise
// scheduler above. The target's `limits.lrSchedulers` overrides this fallback.
const lrSchedulerOptions = ["constant", "linear", "cosine"];
const optimizerLabels = {
  adam: "Adam",
  adamw: "AdamW",
  adamw8bit: "AdamW 8-bit",
  prodigy: "Prodigy",
  prodigyopt: "Prodigy",
  rose: "Rose",
};
// Adapter network parameterization. `lora` is the universal default; `lokr`
// (LyCORIS Kronecker) is offered only on targets whose `limits.networkTypes`
// advertise it (epic 2193).
const networkTypeLabels = {
  lora: "LoRA",
  lokr: "LoKr (LyCORIS Kronecker)",
};
// Versions of the ostris de-distill training adapter (Z-Image-Turbo only). The
// worker maps these to the matching repo file; legacy "v2-default" normalizes to v2.
const trainingAdapterVersionOptions = ["v1", "v2"];
const trainingAdapterVersionLabels = {
  v1: "v1 — stable (smaller)",
  v2: "v2 — experimental (heavier de-distill)",
};
const configFieldLabels = {
  outputName: "LoRA name",
  triggerWord: "Trigger phrase",
  outputScope: "Output scope",
  qualityPreset: "Quality preset",
  requestedGpu: "Requested GPU",
  rank: "Rank",
  alpha: "Alpha",
  networkType: "Network type",
  decomposeFactor: "LoKr factor",
  optimizer: "Optimizer",
  learningRate: "Learning rate",
  weightDecay: "Weight decay",
  lrScheduler: "LR scheduler",
  lrWarmupSteps: "LR warmup steps",
  steps: "Steps",
  timestepType: "Timestep type",
  timestepBias: "Timestep bias",
  lossType: "Loss type",
  trainingAdapterVersion: "De-distill adapter",
  gradientCheckpointing: "Gradient checkpointing",
  resolution: "Resolution",
  precision: "Precision",
  saveEvery: "Checkpoint cadence",
  sampleEvery: "Sample cadence",
  sampleSteps: "Sample steps",
  sampleGuidanceScale: "Guidance scale",
  batchSize: "Batch size",
  gradientAccumulation: "Gradient accumulation",
  seed: "Seed",
};
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

// Human label for the detected caption source (sc-2025) — read-only on the card.
function captionSourceLabel(source) {
  if (source === "imported") {
    return "Imported";
  }
  if (source === "auto") {
    return "Auto";
  }
  return "Manual";
}

// Caption edit state keyed by selection id (sc-2025): the single source of
// truth for the unified caption cards, seeded from the saved dataset items and
// updated as the user edits or imports captions.
function captionDraftsFromDataset(dataset) {
  const map = {};
  (dataset?.items ?? []).forEach((item, index) => {
    map[datasetItemSelectionKey(dataset, item, index)] = {
      text: item.caption?.text ?? "",
      source: item.caption?.source ?? "manual",
    };
  });
  return map;
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

function datasetItemSelectionKey(dataset, item, index = 0) {
  return item?.assetId || `dataset-item:${dataset?.id ?? "draft"}:${item?.id ?? index}`;
}

function datasetItemProjectPath(dataset, item) {
  const path = String(item?.path ?? "").replaceAll("\\", "/");
  if (!dataset?.id || !path) {
    return "";
  }
  return `training/datasets/${dataset.id}/${path}`;
}

function datasetOwnedAssets(dataset, projectId, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  return (dataset?.items ?? [])
    .map((item, index) => {
      if (item.assetId && catalogIds.has(item.assetId)) {
        return null;
      }
      const path = datasetItemProjectPath(dataset, item);
      if (!path) {
        return null;
      }
      const id = datasetItemSelectionKey(dataset, item, index);
      return {
        id,
        assetId: item.assetId ?? null,
        datasetOwned: true,
        projectId,
        type: "image",
        displayName: item.displayName ?? imageAssetName(item),
        file: {
          path,
          mimeType: `image/${String(path).split(".").pop() || "png"}`,
          width: item.width ?? null,
          height: item.height ?? null,
        },
      };
    })
    .filter(Boolean);
}

function normalizeDatasetAssetIds(dataset, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  return (dataset?.items ?? [])
    .map((item, index) => {
      if (item.assetId && catalogIds.has(item.assetId)) {
        return item.assetId;
      }
      return datasetItemSelectionKey(dataset, item, index);
    })
    .filter(Boolean);
}

function datasetHealth({ activeDataset, imageAssets, selectedAssetIds }) {
  const assetsById = new Map(imageAssets.map((asset) => [asset.id, asset]));
  const selectedAssets = selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean);
  const missingAssets = selectedAssetIds.filter((id) => !assetsById.has(id)).length;
  const disabledItems = selectedAssets.filter((asset) => asset.status?.rejected || asset.status?.trashed).length + missingAssets;
  const names = selectedAssets.map((asset) => imageAssetName(asset).toLowerCase());
  const duplicateFilenames = names.filter((name, index) => names.indexOf(name) !== index).length;
  const captionsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), captionText(item)]),
  );
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

function datasetPayload({ activeDataset, assetsById, associatedCharacterId, captionDraftById = {}, name, selectedAssetIds }) {
  const itemsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), item]),
  );
  return {
    name: name.trim(),
    modality: "image",
    // sc-2022: associate the dataset with a character when one is set (created
    // from a character's images, or images imported from the Character tab).
    ...(associatedCharacterId ? { characterId: associatedCharacterId } : {}),
    items: selectedAssetIds
      .map((selectionId) => {
        const asset = assetsById.get(selectionId);
        if (!asset) {
          return null;
        }
        const previous = itemsByAssetId.get(selectionId);
        const draft = captionDraftById[selectionId];
        let caption;
        if (draft && (String(draft.text ?? "").length || draft.source)) {
          caption = {
            text: draft.text ?? "",
            source: draft.source ?? "manual",
            triggerWords: previous?.caption?.triggerWords ?? [],
          };
        } else if (previous?.caption) {
          caption = {
            text: previous.caption.text ?? "",
            source: previous.caption.source ?? "manual",
            triggerWords: previous.caption.triggerWords ?? [],
          };
        }
        const source = asset.datasetOwned || asset.datasetOnly ? { path: asset.file?.path } : { assetId: asset.id };
        return {
          ...source,
          displayName: asset.displayName ?? imageAssetName(asset),
          caption,
        };
      })
      .filter(Boolean),
  };
}

function asText(value) {
  return value === null || value === undefined ? "" : String(value);
}

// Normalize a training-adapter version to the worker's canonical token. Mirrors
// the worker's substring match so legacy "v2-default" shows as "v2" in the select.
function normalizeTrainingAdapterVersion(value) {
  const token = asText(value).trim();
  const lower = token.toLowerCase();
  if (lower.includes("v1")) return "v1";
  if (lower.includes("v2")) return "v2";
  return token;
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

function optimizerLabel(value) {
  return optimizerLabels[value] ?? value;
}

function networkTypeLabel(value) {
  return networkTypeLabels[value] ?? value;
}

function optionLabel(value) {
  return String(value ?? "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function presetSortValue(preset) {
  const order = Number(preset?.ui?.order);
  return Number.isFinite(order) ? order : 999;
}

function presetsForTarget(presets, targetId) {
  return (presets ?? [])
    .filter((preset) => preset.targetId === targetId)
    .slice()
    .sort((left, right) => presetSortValue(left) - presetSortValue(right) || left.name.localeCompare(right.name));
}

function defaultPresetForTarget(presets, targetId) {
  const targetPresets = presetsForTarget(presets, targetId);
  return targetPresets.find((preset) => preset.ui?.default) ?? targetPresets[0] ?? null;
}

export function configDraftFromTarget(target, dataset, gpuOptions, triggerPhrase = "", preset = null, previousDraft = {}) {
  const defaults = preset?.config ?? target?.defaults ?? {};
  const advanced = defaults.advanced ?? {};
  const firstGpu = gpuOptions[0] ?? "";
  const requestedGpu = asText(advanced.requestedGpu || firstGpu);
  const outputLabel = outputKindLabel(target);
  return {
    outputName: previousDraft.outputName ?? (dataset?.name ? `${dataset.name} ${outputLabel}` : ""),
    triggerWord: triggerPhrase || asText(defaults.triggerWord),
    outputScope: asText(advanced.outputScope),
    qualityPreset: asText(advanced.qualityPreset),
    requestedGpu: gpuOptions.includes(requestedGpu) ? requestedGpu : firstGpu,
    rank: numericDraft(defaults.rank),
    alpha: numericDraft(defaults.alpha),
    networkType: asText(advanced.networkType || "lora"),
    // LoKr block-decomposition factor; -1 = auto. Only consumed when networkType
    // is lokr (the worker ignores it otherwise).
    decomposeFactor: numericDraft(advanced.decomposeFactor ?? -1),
    optimizer: asText(defaults.optimizer),
    learningRate: numericDraft(defaults.learningRate),
    weightDecay: numericDraft(advanced.weightDecay),
    lrScheduler: asText(advanced.lrScheduler || "constant"),
    lrWarmupSteps: numericDraft(advanced.lrWarmupSteps),
    steps: numericDraft(defaults.steps),
    timestepType: asText(advanced.timestepType || "sigmoid"),
    timestepBias: asText(advanced.timestepBias || "balanced"),
    lossType: asText(advanced.lossType || "mse"),
    trainingAdapterRepo: asText(advanced.trainingAdapterRepo),
    trainingAdapterVersion: normalizeTrainingAdapterVersion(advanced.trainingAdapterVersion),
    gradientCheckpointing: advanced.gradientCheckpointing !== false,
    resolution: numericDraft(defaults.resolution),
    precision: asText(advanced.mixedPrecision),
    saveEvery: numericDraft(defaults.saveEvery),
    sampleEvery: numericDraft(advanced.sampleEvery),
    sampleSteps: numericDraft(advanced.sampleSteps),
    sampleGuidanceScale: numericDraft(advanced.sampleGuidanceScale),
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

function samplePromptsFromTrigger(triggerWord) {
  const trigger = String(triggerWord ?? "").trim() || "the trained subject";
  return [
    `${trigger}, studio portrait, soft key light, detailed face`,
    `${trigger}, full body fashion editorial photo, natural pose`,
    `${trigger}, cinematic outdoor portrait, golden hour`,
    `${trigger}, close-up character portrait, dramatic rim light`,
  ];
}

export function trainingConfigSnapshot({ activeDataset, configDraft, selectedPreset, selectedTarget, dryRun = true }) {
  const defaults = selectedTarget?.defaults ?? {};
  const networkType = asText(configDraft.networkType).trim() || "lora";
  const advanced = compactObject({
    ...(defaults.advanced ?? {}),
    networkType,
    // LoKr factor only matters for lokr; omit it otherwise so lora jobs stay clean.
    decomposeFactor: networkType === "lokr" ? numberFromDraft(configDraft.decomposeFactor) : undefined,
    weightDecay: numberFromDraft(configDraft.weightDecay),
    lrScheduler: asText(configDraft.lrScheduler).trim() || "constant",
    lrWarmupSteps: numberFromDraft(configDraft.lrWarmupSteps),
    timestepType: asText(configDraft.timestepType).trim(),
    timestepBias: asText(configDraft.timestepBias).trim(),
    lossType: asText(configDraft.lossType).trim(),
    // Preset-only advanced keys (the submit spreads target defaults, not the
    // preset), so carry the de-distill adapter through explicitly — the worker
    // only fuses it when config.advanced.trainingAdapterRepo is present.
    trainingAdapterRepo: asText(configDraft.trainingAdapterRepo).trim(),
    trainingAdapterVersion: asText(configDraft.trainingAdapterVersion).trim(),
    gradientCheckpointing: Boolean(configDraft.gradientCheckpointing),
    mixedPrecision: asText(configDraft.precision).trim(),
    sampleEvery: numberFromDraft(configDraft.sampleEvery),
    sampleSteps: numberFromDraft(configDraft.sampleSteps),
    sampleGuidanceScale: numberFromDraft(configDraft.sampleGuidanceScale),
    samplePrompts: samplePromptsFromTrigger(configDraft.triggerWord),
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
    presetId: selectedPreset?.id,
    presetVersion: selectedPreset?.version,
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

function DatasetHealth({ health, action }) {
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
      {action ? <div className="training-health-action">{action}</div> : null}
    </div>
  );
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
  const firstTarget = trainingTargets[0] ?? null;
  // Mac UI gating (sc-3486): a target whose kernel has no native mlx-gen Rust trainer
  // (kolors_lora / lens_lora) can't train on a gated Mac — disable it and snap off it.
  const macTargetBlocked = (target) => macTrainingKernelBlocked(macCapabilities, target?.kernel);
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
  async function persistDataset() {
    const payload = datasetPayload({
      activeDataset,
      assetsById,
      associatedCharacterId,
      captionDraftById,
      name: draftName,
      selectedAssetIds,
    });
    const dataset = activeDataset
      ? await updateDataset(activeDataset.id, payload)
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

  async function submitTrainingJob() {
    if (!configReady || submittingJob) {
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
                      <CompactSelector
                        busyId={busyDatasetId}
                        createLabel="New dataset"
                        getSubtitle={(dataset) => {
                          const count = datasetItemCount(dataset);
                          return `${count} item${count === 1 ? "" : "s"}`;
                        }}
                        getThumbAsset={datasetThumbAsset}
                        items={datasets}
                        label="Select dataset"
                        onCreate={startNewDataset}
                        onSelect={(dataset) => openDataset(dataset.id)}
                        placeholder={activeDataset ? activeDataset.name : "New dataset"}
                        selectedId={selectedDatasetId}
                      />
                    </div>
                  </div>
                  {datasetsError ? <p className="inline-warning">{datasetsError}</p> : null}
                  {datasetError ? <p className="inline-warning">{datasetError}</p> : null}
                  {datasetMessage ? <p className="inline-success">{datasetMessage}</p> : null}
                  <div className="training-dataset-workspace">
                    {loadingDatasets ? <div className="empty-panel compact-panel">Loading training datasets</div> : null}
                    {!loadingDatasets && datasets.length === 0 ? (
                      <div className="empty-panel compact-panel">No training datasets yet — use “New” to start one.</div>
                    ) : null}
                    <div className="training-dataset-editor">
                      <div className="training-dataset-fields">
                        <label className="field-name">
                          Dataset name
                          <input
                            onChange={(event) => setDraftName(event.target.value)}
                            placeholder="Character portrait set"
                            value={draftName}
                          />
                        </label>
                        <span className="field-version">
                          {dirty ? "Unsaved changes" : activeDataset ? `Version ${activeDataset.version}` : "Draft"}
                        </span>
                        <button className="primary-action field-add" onClick={() => setAddDialogOpen(true)} type="button">
                          <Icon.Plus size={14} />
                          Add images
                        </button>

                        <label className="field-prefix">
                          Name prefix
                          <input
                            onChange={(event) => setRenamePrefix(event.target.value)}
                            placeholder="item"
                            value={renamePrefix}
                          />
                        </label>
                        <button
                          className="primary-action field-apply"
                          disabled={!activeDataset?.id || renaming || !memberAssets.length}
                          onClick={applyOrderedNames}
                          type="button"
                        >
                          <Icon.Sliders size={14} />
                          {renaming ? "Renaming…" : "Apply ordered names"}
                        </button>
                        <button
                          className="primary-action field-caption"
                          disabled={!memberAssets.length}
                          onClick={() => setCaptionDialog({ type: "all" })}
                          type="button"
                        >
                          <Icon.Sliders size={14} />
                          Caption all
                        </button>
                      </div>
                      <DatasetHealth
                        health={health}
                        action={
                          <button className="primary-action" disabled={!canSave} onClick={saveDataset} type="button">
                            {savingDataset ? "Saving" : activeDataset ? "Save dataset" : "Create dataset"}
                          </button>
                        }
                      />
                      <div className="training-validity">
                        <span className={health.valid ? "training-valid-dot valid" : "training-valid-dot"} />
                        <span>{health.valid ? "Dataset is ready for downstream steps" : "Add image assets to build this dataset"}</span>
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
                      <div className="training-caption-grid" aria-label="Dataset images and captions">
                        {memberAssets.length ? (
                          memberAssets.map((asset) => {
                            const disabled = asset.status?.rejected || asset.status?.trashed;
                            const draft = captionDraftById[asset.id] ?? {};
                            const source = draft.source ?? "manual";
                            const name = asset.displayName ?? imageAssetName(asset);
                            return (
                              <article
                                className={["training-caption-card", disabled ? "disabled" : ""].filter(Boolean).join(" ")}
                                key={asset.id}
                              >
                                <div className="training-caption-card-head">
                                  <button className="training-caption-card-thumb" onClick={() => onPreview(asset, memberAssets)} type="button">
                                    <AssetThumbnail asset={asset} />
                                  </button>
                                  <div className="training-caption-card-meta">
                                    <strong title={name}>{name}</strong>
                                    <span className={`training-caption-source source-${source}`}>{captionSourceLabel(source)}</span>
                                    {disabled ? (
                                      <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
                                    ) : null}
                                  </div>
                                </div>
                                <textarea
                                  aria-label={`Caption for ${name}`}
                                  className="training-caption-card-text"
                                  onChange={(event) => updateCaption(asset.id, event.target.value)}
                                  placeholder="Describe this image…"
                                  rows={3}
                                  value={draft.text ?? ""}
                                />
                                <div className="training-caption-card-actions">
                                  <button
                                    aria-label={`Remove ${name}`}
                                    className="secondary-action"
                                    onClick={() => removeUnavailableAsset(asset.id)}
                                    type="button"
                                  >
                                    Remove
                                  </button>
                                  <button
                                    aria-label={`Re-caption ${name}`}
                                    className="secondary-action"
                                    disabled={captioning}
                                    onClick={() => setCaptionDialog({ type: "item", member: asset })}
                                    type="button"
                                  >
                                    Re-Caption
                                  </button>
                                </div>
                              </article>
                            );
                          })
                        ) : (
                          <div className="empty-panel compact-panel">No images yet — use “Add images” to build this dataset.</div>
                        )}
                      </div>
                      {addDialogOpen ? (
                        <DatasetAddDialog
                          assets={imageAssets}
                          characters={characters}
                          importing={importingAssets}
                          memberIds={selectedAssetIds}
                          onAdd={addAssets}
                          onClose={() => setAddDialogOpen(false)}
                          onImport={handleImport}
                        />
                      ) : null}
                      {captionDialog ? (
                        <DatasetCaptionDialog
                          captionLengths={joyCaptionLengths}
                          captionTypes={joyCaptionTypes}
                          extraOptions={joyCaptionExtraOptions}
                          gpuOptions={gpuOptions}
                          onChange={updateCaptionSetting}
                          onClose={() => setCaptionDialog(null)}
                          onRun={runCaptionJob}
                          onToggleExtra={toggleCaptionExtraOption}
                          promptValue={displayedCaptionPrompt}
                          running={captioning}
                          scope={
                            captionDialog.type === "item"
                              ? {
                                  type: "item",
                                  name: captionDialog.member.displayName ?? imageAssetName(captionDialog.member),
                                }
                              : { type: "all" }
                          }
                          settings={captionSettings}
                        />
                      ) : null}
                    </div>
                  </div>
                </>
              ) : null}

              {activeTab === "configure" ? (
                <>
                  <div className="training-panel-head">
                    <div>
                      <p className="eyebrow">Configure Job</p>
                      <h3>{active.title}</h3>
                    </div>
                    <div className="training-head-actions">
                      <button className="secondary-action" onClick={() => setActiveView?.("LibraryDataSets")} type="button">
                        <Icon.Library size={14} />
                        Data Sets
                      </button>
                      <span className="training-status-pill">{configReady ? "Ready" : "Needs input"}</span>
                    </div>
                  </div>
                  {trainingTargetsError ? <p className="inline-warning">{trainingTargetsError}</p> : null}
                  {trainingPresetsError ? <p className="inline-warning">{trainingPresetsError}</p> : null}
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
                            {trainingTargets.map((target) => {
                              const blocked = macTargetBlocked(target);
                              return (
                                <option key={target.id} value={target.id} disabled={blocked}>
                                  {target.ui?.label ?? target.name}
                                  {blocked ? " — not on Mac (Rust/MLX only)" : ""}
                                </option>
                              );
                            })}
                          </select>
                        </label>
                        <label>
                          Preset
                          <select onChange={(event) => updateSelectedPreset(event.target.value)} value={selectedPreset?.id ?? ""}>
                            {targetPresets.length ? null : <option value="">Target defaults</option>}
                            {targetPresets.map((preset) => (
                              <option key={preset.id} value={preset.id}>
                                {preset.name}
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
                            {visibleQualityPresets.length ? null : (
                              <option value={configDraft.qualityPreset ?? ""}>{configDraft.qualityPreset || "Default"}</option>
                            )}
                            {visibleQualityPresets.map((preset) => (
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
                        <label>
                          Sample cadence
                          <input
                            onChange={(event) => updateConfigDraft("sampleEvery", event.target.value)}
                            type="number"
                            value={configDraft.sampleEvery ?? ""}
                          />
                        </label>
                        <label>
                          Sample steps
                          <input
                            onChange={(event) => updateConfigDraft("sampleSteps", event.target.value)}
                            type="number"
                            value={configDraft.sampleSteps ?? ""}
                          />
                        </label>
                        <label>
                          Guidance scale
                          <input
                            onChange={(event) => updateConfigDraft("sampleGuidanceScale", event.target.value)}
                            step="0.1"
                            type="number"
                            value={configDraft.sampleGuidanceScale ?? ""}
                          />
                        </label>
                      </div>

                      {selectedPreset ? (
                        <div className="training-preset-summary" aria-label="Preset values">
                          <span>{selectedPreset.name}</span>
                          <span>Rank {configDraft.rank || "-"}</span>
                          <span>LR {configDraft.learningRate || "-"}</span>
                          <span>{optimizerLabel(configDraft.optimizer)}</span>
                          <span>{configDraft.steps || "-"} steps</span>
                          <span>{configDraft.resolution || "-"}px</span>
                          {customizedConfigLabels.length ? (
                            <span>Customized: {customizedConfigLabels.join(", ")}</span>
                          ) : null}
                        </div>
                      ) : null}

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
                          {showNetworkType ? (
                            <label title="Adapter parameterization. LoRA is the standard low-rank adapter; LoKr (LyCORIS Kronecker) trains a much smaller, often more expressive adapter (torch backends only).">
                              Network type
                              <select
                                onChange={(event) => updateConfigDraft("networkType", event.target.value)}
                                value={configDraft.networkType ?? "lora"}
                              >
                                {networkTypeOptions.map((option) => {
                                  const blocked = option === "lokr" && macLokrOnWanBlocked;
                                  return (
                                    <option key={option} value={option} disabled={blocked}>
                                      {networkTypeLabel(option)}
                                      {blocked ? " — not on Mac (Rust/MLX only)" : ""}
                                    </option>
                                  );
                                })}
                              </select>
                            </label>
                          ) : null}
                          {showNetworkType && isLokrNetwork ? (
                            <label title="LoKr block-decomposition factor. -1 lets LyCORIS pick the largest factor automatically; larger values trade adapter size for capacity.">
                              LoKr factor
                              <input
                                min="-1"
                                onChange={(event) => updateConfigDraft("decomposeFactor", event.target.value)}
                                step="1"
                                type="number"
                                value={configDraft.decomposeFactor ?? ""}
                              />
                            </label>
                          ) : null}
                          <label>
                            Optimizer
                            <select onChange={(event) => updateConfigDraft("optimizer", event.target.value)} value={configDraft.optimizer ?? ""}>
                              {visibleOptimizerOptions.map((optimizer) => (
                                <option key={optimizer} value={optimizer}>
                                  {optimizerLabel(optimizer)}
                                </option>
                              ))}
                            </select>
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
                            Weight decay
                            <input
                              onChange={(event) => updateConfigDraft("weightDecay", event.target.value)}
                              step="0.00001"
                              type="number"
                              value={configDraft.weightDecay ?? ""}
                            />
                          </label>
                          <label>
                            Steps
                            <input onChange={(event) => updateConfigDraft("steps", event.target.value)} type="number" value={configDraft.steps ?? ""} />
                          </label>
                          <label>
                            Timestep type
                            <select onChange={(event) => updateConfigDraft("timestepType", event.target.value)} value={configDraft.timestepType ?? ""}>
                              {timestepTypeOptions.map((option) => (
                                <option key={option} value={option}>
                                  {optionLabel(option)}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            Timestep bias
                            <select onChange={(event) => updateConfigDraft("timestepBias", event.target.value)} value={configDraft.timestepBias ?? ""}>
                              {timestepBiasOptions.map((option) => (
                                <option key={option} value={option}>
                                  {optionLabel(option)}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            Loss type
                            <select onChange={(event) => updateConfigDraft("lossType", event.target.value)} value={configDraft.lossType ?? ""}>
                              {lossTypeOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option === "mse" ? "Mean Squared Error" : optionLabel(option)}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label title="Learning-rate scheduler (not the timestep/noise scheduler). Constant holds the LR fixed for the whole run; linear and cosine decay it toward zero over the run.">
                            LR scheduler
                            <select onChange={(event) => updateConfigDraft("lrScheduler", event.target.value)} value={configDraft.lrScheduler ?? ""}>
                              {visibleLrSchedulerOptions.map((option) => (
                                <option key={option} value={option}>
                                  {optionLabel(option)}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label title="Optional linear warmup: number of steps to ramp the LR up from zero before the scheduler body runs. 0 disables warmup.">
                            LR warmup steps
                            <input
                              min="0"
                              onChange={(event) => updateConfigDraft("lrWarmupSteps", event.target.value)}
                              type="number"
                              value={configDraft.lrWarmupSteps ?? ""}
                            />
                          </label>
                          {showTrainingAdapter ? (
                            <label title="ostris de-distill adapter for the step-distilled Z-Image-Turbo base. Fused in for training, removed at inference. v1 is stable; v2 is a heavier, experimental de-distill.">
                              De-distill adapter
                              <select
                                onChange={(event) => updateConfigDraft("trainingAdapterVersion", event.target.value)}
                                value={configDraft.trainingAdapterVersion ?? ""}
                              >
                                {visibleTrainingAdapterVersions.map((version) => (
                                  <option key={version} value={version}>
                                    {trainingAdapterVersionLabels[version] ?? version}
                                  </option>
                                ))}
                              </select>
                            </label>
                          ) : null}
                          <label>
                            Resolution
                            <select onChange={(event) => updateConfigDraft("resolution", event.target.value)} value={configDraft.resolution ?? ""}>
                              {visibleResolutionOptions.length ? null : <option value={configDraft.resolution ?? ""}>{configDraft.resolution ?? ""}</option>}
                              {visibleResolutionOptions.map((resolution) => (
                                <option key={resolution} value={resolution}>
                                  {resolution}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label>
                            Precision
                            <input onChange={(event) => updateConfigDraft("precision", event.target.value)} value={configDraft.precision ?? ""} />
                          </label>
                          <label className="training-checkbox-field">
                            <input
                              checked={Boolean(configDraft.gradientCheckpointing)}
                              onChange={(event) => updateConfigDraft("gradientCheckpointing", event.target.checked)}
                              type="checkbox"
                            />
                            Gradient checkpointing
                          </label>
                          <label>
                            Checkpoint cadence
                            <input
                              onChange={(event) => updateConfigDraft("saveEvery", event.target.value)}
                              type="number"
                              value={configDraft.saveEvery ?? ""}
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
                        <button
                          className="primary-action"
                          disabled={!configReady || submittingJob}
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
