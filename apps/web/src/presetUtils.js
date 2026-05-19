export const noRecipePresetId = "__no_recipe_preset__";

export function rememberPresetDefault(snapshots, key, currentValue, appliedValue) {
  const previousSnapshot = snapshots.current[key];
  snapshots.current[key] = {
    appliedValue,
    previousValue:
      previousSnapshot && Object.is(currentValue, previousSnapshot.appliedValue)
        ? previousSnapshot.previousValue
        : currentValue,
  };
}

export function clearPresetDefault(setter, snapshots, key) {
  const snapshot = snapshots.current[key];
  if (!snapshot) {
    return;
  }
  setter((current) => (Object.is(current, snapshot.appliedValue) ? snapshot.previousValue : current));
  delete snapshots.current[key];
}

export const defaultModesByWorkflow = {
  text_to_image: ["text_to_image", "character_image", "style_variations"],
  edit_image: ["edit_image"],
  image_to_video: ["image_to_video"],
  text_to_video: ["text_to_video"],
  first_last_frame: ["first_last_frame"],
};

export const modeLabels = {
  text_to_image: "Text",
  edit_image: "Edit",
  character_image: "Character",
  style_variations: "Variations",
  image_to_video: "Image Video",
  text_to_video: "Text Video",
  first_last_frame: "First/Last",
};

export function workflowModelType(workflow) {
  return workflow?.includes("video") || workflow === "first_last_frame" ? "video" : "image";
}

export function workflowModes(workflow) {
  return defaultModesByWorkflow[workflow] ?? [workflow].filter(Boolean);
}

export function compactModeList(workflow) {
  return workflowModes(workflow).map((mode) => modeLabels[mode] ?? mode).join(", ");
}

export function loraFamilies(lora) {
  const compatibility = lora?.compatibility ?? {};
  const values =
    lora?.families ??
    lora?.compatibleFamilies ??
    lora?.modelFamilies ??
    compatibility.families ??
    (lora?.family ? [lora.family] : []);
  return normalizeFamilies(values);
}

export function modelLoraFamilies(model) {
  const compatibility = model?.loraCompatibility ?? {};
  const values =
    model?.families ??
    model?.compatibleFamilies ??
    model?.modelFamilies ??
    model?.loraFamilies ??
    compatibility.families ??
    (model?.family ? [model.family] : []);
  return normalizeFamilies(values);
}

export function normalizeLoraFamily(family) {
  return String(family ?? "").trim().toLowerCase().replaceAll("_", "-");
}

export function normalizeFamilies(values) {
  return (Array.isArray(values) ? values : [values])
    .map(normalizeLoraFamily)
    .filter(Boolean);
}

export function loraMatchesModel(lora, model) {
  const modelFamilies = modelLoraFamilies(model);
  const families = loraFamilies(lora);
  return !modelFamilies.length || !families.length || families.some((family) => modelFamilies.includes(family));
}

export function presetMatchesWorkflow(preset, mode) {
  // A preset has one primary workflow for persistence, but modes describe every
  // Studio entry point where the picker should surface it.
  if (preset?.modes?.length) {
    return preset.modes.includes(mode);
  }
  return preset?.workflow === mode;
}

export function presetMatchesModel(preset, model) {
  return !preset?.model || !model?.id || preset.model === model.id;
}

export function presetLoras(preset) {
  return preset?.loras ?? preset?.builtInLoras ?? [];
}

export function presetLoraId(presetLora) {
  return typeof presetLora === "string" ? presetLora : presetLora?.id ?? presetLora?.loraId;
}

export function loraWeight(lora, presetLora = {}) {
  const value = Number(presetLora.weight ?? lora?.defaultWeight ?? lora?.weight ?? 0.8);
  return Number.isFinite(value) ? value : 0.8;
}

export function serializePresetLora(lora, presetLora = {}) {
  const id = presetLoraId(presetLora) ?? lora?.id;
  return {
    id,
    name: lora?.name ?? presetLora?.name ?? presetLora?.displayName ?? id,
    scope: lora?.scope ?? presetLora?.scope ?? "global",
    weight: loraWeight(lora, presetLora),
    triggerWords: lora?.triggerWords ?? [],
    compatibility: lora?.compatibility ?? presetLora?.compatibility ?? {},
    installedPath: lora?.installedPath ?? presetLora?.installedPath ?? null,
    source: lora?.source ?? presetLora?.source ?? null,
    presetManaged: true,
  };
}

export function presetLoraDetails(preset, loras) {
  return presetLoras(preset)
    .map((presetLora) => {
      const id = presetLoraId(presetLora);
      const lora = loras.find((item) => item.id === id);
      return lora
        ? { ...serializePresetLora(lora, presetLora), missing: lora.installState === "missing" }
        : { id, name: id, weight: loraWeight(null, presetLora), missing: true };
    })
    .filter((lora) => lora.id);
}

export function presetPromptParts(preset) {
  return [preset?.prompt?.prefix, preset?.prompt?.suffix]
    .map((part) => String(part ?? "").trim())
    .filter(Boolean);
}

export function presetValidation(preset, loras, model) {
  const details = presetLoraDetails(preset, loras);
  const missing = details.filter((lora) => lora.missing).map((lora) => lora.id);
  const incompatible = details
    .filter((detail) => {
      const lora = loras.find((item) => item.id === detail.id);
      return lora && !loraMatchesModel(lora, model);
    })
    .map((lora) => lora.id);
  return {
    missing,
    incompatible,
    ok: missing.length === 0 && incompatible.length === 0,
  };
}

export function presetValidationMessage(validation) {
  if (validation.ok) {
    return "";
  }
  const parts = [];
  if (validation.missing.length) {
    parts.push(`${validation.missing.join(", ")} has not finished importing`);
  }
  if (validation.incompatible.length) {
    parts.push(`${validation.incompatible.join(", ")} is not compatible with the selected model`);
  }
  return `Save blocked: ${parts.join("; ")}. Wait for imports to finish, remove incompatible LoRAs, or choose a matching model.`;
}
