import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAppContext } from "../context/AppContext.js";
import { API_BASE_URL } from "../api.js";
import { AssetThumbnail, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { CompactSelector } from "../components/CompactSelector.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
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

function itemFileStem(item) {
  const name = String(item?.path ?? item?.displayName ?? item?.id ?? "item").replaceAll("\\", "/").split("/").pop() || "item";
  const dotIndex = name.lastIndexOf(".");
  return dotIndex > 0 ? name.slice(0, dotIndex) : name;
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
    assetId: datasetItemSelectionKey(dataset, item, index),
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

function datasetPayload({ activeDataset, assetsById, importedCaptions = {}, name, selectedAssetIds }) {
  const itemsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), item]),
  );
  return {
    name: name.trim(),
    modality: "image",
    items: selectedAssetIds
      .map((selectionId) => {
        const asset = assetsById.get(selectionId);
        if (!asset) {
          return null;
        }
        const previous = itemsByAssetId.get(selectionId);
        const imported = importedCaptions[selectionId];
        let caption;
        if (imported) {
          // An imported .txt sidecar takes precedence over a carried-forward
          // caption so the user can skip the manual captioning step.
          caption = {
            text: imported.text ?? "",
            source: imported.source ?? "imported",
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

function configDraftFromTarget(target, dataset, gpuOptions, triggerPhrase = "", preset = null, previousDraft = {}) {
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

function trainingConfigSnapshot({ activeDataset, configDraft, selectedPreset, selectedTarget, dryRun = true }) {
  const defaults = selectedTarget?.defaults ?? {};
  const advanced = compactObject({
    ...(defaults.advanced ?? {}),
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

function latestTrainingSamples(job) {
  const latest = Array.isArray(job.result?.latestTrainingSamples) ? job.result.latestTrainingSamples : [];
  if (latest.length) {
    return latest.slice(-4);
  }
  const samples = Array.isArray(job.result?.trainingSamples) ? job.result.trainingSamples : [];
  return samples.slice(-4);
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

// Resolve the training-sample assets for a job. Pads with placeholder labels
// (the sample-prompt list) so the image-grid skeleton cells communicate
// "this slot is coming" just like image batches.
function trainingSampleAssets(job, projectId) {
  const samples = latestTrainingSamples(job);
  const samplePrompts = Array.isArray(job.result?.samplePrompts)
    ? job.result.samplePrompts
    : samplePromptsFromTrigger(job.payload?.plan?.config?.triggerWord);
  return samplePrompts.slice(0, 4).map((prompt, index) => {
    const sample = samples[index];
    if (!sample) return null;
    const label = sample?.step ? `Step ${sample.step}` : prompt;
    return trainingSampleToAsset(sample, projectId, label, `${job.id}-sample-${index}`);
  }).filter(Boolean);
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
            thumbnailAssets={trainingSampleAssets(job, projectId)}
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
    writeTrainingDatasetCaptionSidecars,
    createTrainingDatasetCaptionJob,
    createTrainingJob,
    trainingPresets: trainingPresetsCatalog,
    trainingPresetsError = "",
    trainingTargets: trainingTargetsCatalog,
    trainingTargetsError = "",
    setActiveView,
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
  const writeCaptionSidecars = writeTrainingDatasetCaptionSidecars;
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
  // Captions parsed from .txt sidecars during import, keyed by imported asset id.
  // Threaded into the dataset payload on save, then cleared once persisted.
  const [importedCaptions, setImportedCaptions] = useState({});
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [renamePrefix, setRenamePrefix] = useState("");
  const [captionTriggerWords, setCaptionTriggerWords] = useState("");
  const [renameCaptionDraftItems, setRenameCaptionDraftItems] = useState([]);
  const [savingRenameCaption, setSavingRenameCaption] = useState(false);
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
  const memberCaptionById = useMemo(() => {
    const map = new Map(
      (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), captionText(item)]),
    );
    for (const [assetId, caption] of Object.entries(importedCaptions)) {
      if (caption?.text) {
        map.set(assetId, caption.text);
      }
    }
    return map;
  }, [activeDataset, importedCaptions]);
  // Thumbnail for the compact dataset selector (sc-2025): the open dataset's
  // first member, else a list summary's first item asset if resolvable.
  const datasetThumbAsset = (dataset) => {
    if (dataset?.id === selectedDatasetId && memberAssets[0]) {
      return memberAssets[0];
    }
    const firstItemAssetId = (dataset?.items ?? []).find((item) => item.assetId)?.assetId;
    return firstItemAssetId ? assetsById.get(firstItemAssetId) ?? null : null;
  };
  const health = useMemo(
    () => datasetHealth({ activeDataset, imageAssets, selectedAssetIds }),
    [activeDataset, imageAssets, selectedAssetIds],
  );
  const originalAssetIds = useMemo(() => normalizeDatasetAssetIds(activeDataset, assets), [activeDataset, assets]);
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
    setRenameCaptionDraftItems([]);
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

  useEffect(() => {
    const datasetTriggerPhrase = triggerPhraseFromText(activeDataset?.name);
    setRenameCaptionDraftItems(renameCaptionDrafts(activeDataset));
    setRenamePrefix(safeSlug(activeDataset?.name, "item"));
    setCaptionTriggerWords(datasetTriggerPhrase);
    setCaptionSettings((current) => ({ ...current, nameInput: datasetTriggerPhrase }));
    // Imported captions are persisted into the dataset once saved (which swaps
    // activeDataset); drop the staging map so switching datasets can't leak them.
    setImportedCaptions({});
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
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset, assets));
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
    setUploadedDatasetAssets([]);
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

  function addAssets(ids) {
    const additions = (ids ?? []).filter(Boolean);
    if (!additions.length) {
      return;
    }
    setDatasetMessage("");
    setSelectedAssetIds((current) => Array.from(new Set([...current, ...additions])));
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
          setImportedCaptions((current) => ({ ...current, ...captionsByAssetId }));
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

  async function saveDataset() {
    if (!canSave) {
      return;
    }
    setSavingDataset(true);
    setDatasetError("");
    setDatasetMessage("");
    try {
      const payload = datasetPayload({ activeDataset, assetsById, importedCaptions, name: draftName, selectedAssetIds });
      const dataset = activeDataset
        ? await updateDataset(activeDataset.id, payload)
        : await createDataset(payload);
      setActiveDataset(dataset);
      setDraftName(dataset?.name ?? draftName.trim());
      setUploadedDatasetAssets([]);
      setSelectedAssetIds(normalizeDatasetAssetIds(dataset, assets));
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
      setSelectedAssetIds(normalizeDatasetAssetIds(nextDataset, assets));
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
                      <CompactSelector
                        busyId={busyDatasetId}
                        getSubtitle={(dataset) => {
                          const count = datasetItemCount(dataset);
                          return `${count} item${count === 1 ? "" : "s"}`;
                        }}
                        getThumbAsset={datasetThumbAsset}
                        items={datasets}
                        label="Select dataset"
                        onSelect={(dataset) => openDataset(dataset.id)}
                        placeholder={activeDataset ? activeDataset.name : "New dataset"}
                        selectedId={selectedDatasetId}
                      />
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
                    {loadingDatasets ? <div className="empty-panel compact-panel">Loading training datasets</div> : null}
                    {!loadingDatasets && datasets.length === 0 ? (
                      <div className="empty-panel compact-panel">No training datasets yet — use “New” to start one.</div>
                    ) : null}
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
                        <button className="primary-action training-add-images" onClick={() => setAddDialogOpen(true)} type="button">
                          <Icon.Plus size={14} />
                          Add images
                        </button>
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
                      <div className="training-member-grid" aria-label="Dataset images">
                        {memberAssets.length ? (
                          memberAssets.map((asset) => {
                            const disabled = asset.status?.rejected || asset.status?.trashed;
                            const caption = memberCaptionById.get(asset.id);
                            return (
                              <article
                                className={["training-member-card", disabled ? "disabled" : ""].filter(Boolean).join(" ")}
                                key={asset.id}
                              >
                                <button className="training-member-thumb" onClick={() => onPreview(asset)} type="button">
                                  <AssetThumbnail asset={asset} />
                                </button>
                                <div className="training-member-meta">
                                  <strong>{asset.displayName ?? imageAssetName(asset)}</strong>
                                  <span className={caption ? "training-member-caption" : "training-member-caption muted"}>
                                    {caption || "Needs caption"}
                                  </span>
                                  {disabled ? (
                                    <span className="training-asset-badge">{asset.status?.trashed ? "Trashed" : "Rejected"}</span>
                                  ) : null}
                                </div>
                                <button
                                  aria-label={`Remove ${asset.displayName ?? imageAssetName(asset)}`}
                                  className="secondary-action training-member-remove"
                                  onClick={() => removeUnavailableAsset(asset.id)}
                                  type="button"
                                >
                                  Remove
                                </button>
                              </article>
                            );
                          })
                        ) : (
                          <div className="empty-panel compact-panel">No images yet — use “Add images” to build this dataset.</div>
                        )}
                      </div>
                      <div className="training-dataset-actions">
                        <button className="primary-action" disabled={!canSave} onClick={saveDataset} type="button">
                          {savingDataset ? "Saving" : activeDataset ? "Save dataset" : "Create dataset"}
                        </button>
                        <span>{dirty ? "Unsaved changes" : activeDataset ? `Version ${activeDataset.version}` : "Draft"}</span>
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
                    </div>
                  </div>
                </>
              ) : null}

              {datasetLibraryMode || activeTab === "rename-caption" ? (
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
                            {trainingTargets.map((target) => (
                              <option key={target.id} value={target.id}>
                                {target.ui?.label ?? target.name}
                              </option>
                            ))}
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
