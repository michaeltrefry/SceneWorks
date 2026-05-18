import React, { useEffect, useMemo, useState } from "react";
import {
  compactModeList,
  loraMatchesModel,
  presetLoraId,
  presetLoras,
  presetValidation,
  presetValidationMessage,
  workflowModelType,
  workflowModes,
} from "../presetUtils.js";

const workflowOptions = [
  ["text_to_image", "Text to Image"],
  ["edit_image", "Image Edit"],
  ["image_to_video", "Image to Video"],
  ["text_to_video", "Text to Video"],
  ["first_last_frame", "First/Last Frame"],
];

function slugify(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function modelOptions(models, workflow) {
  const type = workflowModelType(workflow);
  return models.filter((model) => model.type === type);
}

function formLorasFromPreset(preset) {
  return presetLoras(preset)
    .map((lora) => {
      const id = presetLoraId(lora);
      return id ? { id, weight: typeof lora === "object" && lora.weight != null ? String(lora.weight) : "" } : null;
    })
    .filter(Boolean);
}

function formFromPreset(preset, fallbackModel) {
  return {
    id: preset?.id ?? "",
    name: preset?.name ?? "",
    scope: preset?.scope === "project" ? "project" : "global",
    workflow: preset?.workflow ?? "text_to_image",
    model: preset?.model ?? fallbackModel ?? "",
    order: preset?.order ?? "",
    count: preset?.defaults?.count ?? "",
    duration: preset?.defaults?.duration ?? "",
    fps: preset?.defaults?.fps ?? "",
    quality: preset?.defaults?.quality ?? "",
    resolution: preset?.defaults?.resolution ?? "",
    negativePrompt: preset?.defaults?.negativePrompt ?? "",
    promptPrefix: preset?.prompt?.prefix ?? "",
    promptSuffix: preset?.prompt?.suffix ?? "",
    description: preset?.ui?.description ?? "",
    loras: formLorasFromPreset(preset),
  };
}

function selectedLoraIds(form) {
  return form.loras.map((lora) => lora.id);
}

function loraLabel(lora) {
  return [lora.scope, lora.family ?? "compatible"].filter(Boolean).join(" | ");
}

function presetStatusLabel(status) {
  if (status.ok) {
    return "Ready";
  }
  if (status.missing.length) {
    return `Waiting for ${status.missing.join(", ")}`;
  }
  return `Remove ${status.incompatible.join(", ")}`;
}

export function PresetManagerScreen({
  activeProject,
  createLoraImportJob,
  createRecipePreset,
  deleteRecipePreset,
  duplicateRecipePreset,
  imageModels,
  loras = [],
  recipePresets = [],
  updateRecipePreset,
  videoModels,
}) {
  const models = useMemo(() => [...imageModels, ...videoModels], [imageModels, videoModels]);
  const [selectedPresetId, setSelectedPresetId] = useState(recipePresets.find((preset) => preset.scope !== "builtin")?.id ?? "");
  const selectedPreset = recipePresets.find((preset) => preset.id === selectedPresetId) ?? null;
  const [form, setForm] = useState(() => formFromPreset(selectedPreset, models[0]?.id));
  const [saving, setSaving] = useState(false);
  const [importingLora, setImportingLora] = useState(false);
  const [message, setMessage] = useState({ tone: "neutral", text: "" });
  const [importForm, setImportForm] = useState({ mode: "url", sourceUrl: "", file: null, name: "" });
  const [fileInputKey, setFileInputKey] = useState(0);
  const editable = !selectedPreset || selectedPreset.scope !== "builtin";
  const busy = saving || importingLora;
  const availableModels = modelOptions(models, form.workflow);
  const selectedModel = models.find((model) => model.id === form.model) ?? availableModels[0] ?? null;
  const availableLoras = selectedModel ? loras.filter((lora) => lora.installState !== "missing" && loraMatchesModel(lora, selectedModel)) : [];
  const validation = presetValidation({ ...selectedPreset, loras: form.loras }, loras, selectedModel);
  const validationMessage = editable ? presetValidationMessage(validation) : "";
  const saveDisabledReason = !editable
    ? "Built-in presets are read-only."
    : !form.name.trim()
      ? "Name is required."
      : !form.model
        ? "Choose a model before saving."
        : validationMessage;
  const hasPendingCompatibleLoras = Boolean(selectedModel) && loras.some((lora) => lora.installState === "missing" && loraMatchesModel(lora, selectedModel));
  const loraEmptyMessage = !selectedModel
    ? "No model selected"
    : hasPendingCompatibleLoras
      ? "No installed compatible LoRAs. Imports appear here after the Queue completes."
      : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  useEffect(() => {
    if (selectedPreset && !recipePresets.some((preset) => preset.id === selectedPreset.id)) {
      setSelectedPresetId(recipePresets.find((preset) => preset.scope !== "builtin")?.id ?? "");
    }
  }, [recipePresets, selectedPreset?.id]);

  useEffect(() => {
    setForm(formFromPreset(selectedPreset, modelOptions(models, selectedPreset?.workflow ?? "text_to_image")[0]?.id ?? models[0]?.id));
    setMessage({ tone: "neutral", text: "" });
  }, [selectedPreset?.id, models]);

  useEffect(() => {
    if (!availableModels.length) {
      return;
    }
    if (!availableModels.some((model) => model.id === form.model)) {
      setForm((current) => ({ ...current, model: availableModels[0].id, loras: [] }));
    }
  }, [availableModels, form.model]);

  function updateField(field, value) {
    setForm((current) => {
      if (field === "workflow") {
        return { ...current, workflow: value, loras: [] };
      }
      if (field === "name" && !selectedPreset) {
        return { ...current, name: value, id: slugify(value) };
      }
      if (field === "model") {
        return { ...current, model: value, loras: current.loras.filter((selection) => {
          const lora = loras.find((item) => item.id === selection.id);
          const model = models.find((item) => item.id === value);
          return !lora || loraMatchesModel(lora, model);
        }) };
      }
      return { ...current, [field]: value };
    });
  }

  function toggleLora(id) {
    setForm((current) => {
      const hasLora = current.loras.some((lora) => lora.id === id);
      if (hasLora) {
        return { ...current, loras: current.loras.filter((lora) => lora.id !== id) };
      }
      if (current.loras.length >= 3) {
        return current;
      }
      const source = loras.find((lora) => lora.id === id);
      const weight = source?.defaultWeight ?? source?.weight ?? 0.8;
      return { ...current, loras: [...current.loras, { id, weight: String(weight) }] };
    });
  }

  function updateLoraWeight(id, weight) {
    setForm((current) => ({
      ...current,
      loras: current.loras.map((lora) => (lora.id === id ? { ...lora, weight } : lora)),
    }));
  }

  function buildPayload() {
    const defaults = {};
    if (form.count !== "") {
      defaults.count = Number(form.count);
    }
    if (form.duration !== "") {
      defaults.duration = Number(form.duration);
    }
    if (form.fps !== "") {
      defaults.fps = Number(form.fps);
    }
    if (form.quality.trim()) {
      defaults.quality = form.quality.trim();
    }
    if (form.resolution.trim()) {
      defaults.resolution = form.resolution.trim();
    }
    if (form.negativePrompt.trim()) {
      defaults.negativePrompt = form.negativePrompt.trim();
    }
    const prompt = {};
    if (form.promptPrefix.trim()) {
      prompt.prefix = form.promptPrefix.trim();
    }
    if (form.promptSuffix.trim()) {
      prompt.suffix = form.promptSuffix.trim();
    }
    const payload = {
      id: slugify(form.id || form.name),
      name: form.name.trim(),
      scope: form.scope,
      workflow: form.workflow,
      modes: workflowModes(form.workflow),
      model: form.model,
      loras: form.loras.map((lora) => ({
        id: lora.id,
        weight: Number.isFinite(Number(lora.weight)) ? Number(lora.weight) : 0.8,
      })),
      ui: { description: form.description.trim() },
    };
    if (form.order !== "") {
      payload.order = Number(form.order);
    }
    if (Object.keys(defaults).length) {
      payload.defaults = defaults;
    }
    if (Object.keys(prompt).length) {
      payload.prompt = prompt;
    }
    return payload;
  }

  async function savePreset(event) {
    event.preventDefault();
    if (saveDisabledReason) {
      setMessage({ tone: "error", text: saveDisabledReason });
      return;
    }
    setSaving(true);
    setMessage({ tone: "neutral", text: "" });
    try {
      const payload = buildPayload();
      if (selectedPreset) {
        await updateRecipePreset(selectedPreset.id, payload, selectedPreset.scope);
        setMessage({ tone: "success", text: "Preset saved." });
      } else {
        const created = await createRecipePreset(payload);
        setSelectedPresetId(created?.id ?? payload.id);
        setMessage({ tone: "success", text: "Preset created." });
      }
    } catch (err) {
      setMessage({ tone: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  }

  async function duplicateSelected() {
    if (!selectedPreset) {
      return;
    }
    setSaving(true);
    setMessage({ tone: "neutral", text: "" });
    try {
      if (selectedPreset.scope === "builtin") {
        const payload = buildPayload();
        payload.id = slugify(`${selectedPreset.id}_copy`);
        payload.name = `${selectedPreset.name ?? selectedPreset.id} Copy`;
        const created = await createRecipePreset(payload);
        setSelectedPresetId(created.id);
        setMessage({ tone: "success", text: "Preset duplicated." });
        return;
      }
      const duplicated = await duplicateRecipePreset(selectedPreset.id, form.scope);
      setSelectedPresetId(duplicated.id);
      setMessage({ tone: "success", text: "Preset duplicated." });
    } catch (err) {
      setMessage({ tone: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  }

  async function archiveSelected() {
    if (!selectedPreset || selectedPreset.scope === "builtin") {
      return;
    }
    setSaving(true);
    setMessage({ tone: "neutral", text: "" });
    try {
      await deleteRecipePreset(selectedPreset.id, selectedPreset.scope);
      setSelectedPresetId("");
      setMessage({ tone: "success", text: "Preset archived." });
    } catch (err) {
      setMessage({ tone: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  }

  async function importLora(event) {
    event.preventDefault();
    const isFileImport = importForm.mode === "file";
    if ((!isFileImport && !importForm.sourceUrl.trim()) || (isFileImport && !importForm.file)) {
      return;
    }
    setImportingLora(true);
    setMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading LoRA file before queueing import." : "",
    });
    try {
      const job = await createLoraImportJob({
        ...(isFileImport ? { file: importForm.file } : { sourceUrl: importForm.sourceUrl.trim() }),
        name: importForm.name.trim() || undefined,
        scope: form.scope,
        family: selectedModel?.family ?? undefined,
      });
      const importedId = job?.payload?.loraId;
      if (importedId) {
        setForm((current) => ({
          ...current,
          loras: current.loras.some((lora) => lora.id === importedId)
            ? current.loras
            : [...current.loras, { id: importedId, weight: "0.8" }].slice(0, 3),
        }));
      }
      setImportForm((current) => ({ ...current, sourceUrl: "", file: null, name: "" }));
      // Force a re-mount so choosing the same file again still emits a change event.
      setFileInputKey((current) => current + 1);
      setMessage({ tone: "success", text: "LoRA import queued. Save after the import finishes." });
    } catch (err) {
      setMessage({ tone: "error", text: err.message });
    } finally {
      setImportingLora(false);
    }
  }

  function startNewPreset() {
    setSelectedPresetId("");
    setForm(formFromPreset(null, modelOptions(models, "text_to_image")[0]?.id ?? models[0]?.id));
    setMessage({ tone: "neutral", text: "" });
  }

  return (
    <section className="main-surface preset-manager">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Preset Manager</p>
          <h2>{activeProject ? activeProject.name : "Global presets"}</h2>
        </div>
        <div className="toolbar">
          <button onClick={startNewPreset} type="button">
            New Preset
          </button>
          <button disabled={!selectedPreset || busy} onClick={duplicateSelected} type="button">
            Duplicate
          </button>
          <button disabled={!selectedPreset || selectedPreset.scope === "builtin" || busy} onClick={archiveSelected} type="button">
            Archive
          </button>
        </div>
      </div>

      <div className="preset-layout">
        <section className="preset-list" aria-label="Recipe presets">
          {recipePresets.length ? (
            recipePresets.map((preset) => {
              const presetModel = models.find((model) => model.id === preset.model);
              const status = presetValidation(preset, loras, presetModel);
              return (
                <button
                  className={selectedPresetId === preset.id ? "preset-row active" : "preset-row"}
                  key={`${preset.scope}-${preset.id}`}
                  onClick={() => setSelectedPresetId(preset.id)}
                  type="button"
                >
                  <span>
                    <strong>{preset.name ?? preset.id}</strong>
                    <small>
                      {preset.scope ?? "global"} | {preset.workflow}
                    </small>
                  </span>
                  <span className={status.ok ? "preset-status ok" : "preset-status error"}>{presetStatusLabel(status)}</span>
                </button>
              );
            })
          ) : (
            <div className="empty-panel compact-panel">No presets yet</div>
          )}
        </section>

        <form className="preset-editor" onSubmit={savePreset}>
          <div className="control-grid compact-controls">
            <label>
              Name
              <input disabled={!editable} onChange={(event) => updateField("name", event.target.value)} required value={form.name} />
            </label>
            <label>
              ID
              <input disabled={Boolean(selectedPreset) || !editable} onChange={(event) => updateField("id", event.target.value)} required value={form.id} />
            </label>
          </div>

          <div className="control-grid">
            <label>
              Scope
              <select disabled={!editable} onChange={(event) => updateField("scope", event.target.value)} value={form.scope}>
                <option value="global">Global</option>
                <option disabled={!activeProject} value="project">
                  Project
                </option>
              </select>
            </label>
            <label>
              Workflow
              <select disabled={!editable} onChange={(event) => updateField("workflow", event.target.value)} value={form.workflow}>
                {workflowOptions.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Order
              <input disabled={!editable} onChange={(event) => updateField("order", event.target.value)} type="number" value={form.order} />
            </label>
          </div>

          <div className="control-grid compact-controls">
            <label>
              Model
              <select disabled={!editable} onChange={(event) => updateField("model", event.target.value)} value={form.model}>
                {availableModels.length ? (
                  availableModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name ?? model.id}
                    </option>
                  ))
                ) : (
                  <option value="">No models</option>
                )}
              </select>
            </label>
            <label>
              Derived modes
              <input disabled readOnly value={compactModeList(form.workflow)} />
            </label>
          </div>

          <div className="control-grid">
            <label>
              Count
              <input disabled={!editable} min="1" max="8" onChange={(event) => updateField("count", event.target.value)} type="number" value={form.count} />
            </label>
            <label>
              Duration
              <input disabled={!editable} min="1" max="30" onChange={(event) => updateField("duration", event.target.value)} type="number" value={form.duration} />
            </label>
            <label>
              Resolution
              <input disabled={!editable} onChange={(event) => updateField("resolution", event.target.value)} placeholder="1024x1024" value={form.resolution} />
            </label>
          </div>

          <div className="control-grid compact-controls">
            <label>
              FPS
              <input disabled={!editable} min="1" max="60" onChange={(event) => updateField("fps", event.target.value)} type="number" value={form.fps} />
            </label>
            <label>
              Quality
              <input disabled={!editable} onChange={(event) => updateField("quality", event.target.value)} value={form.quality} />
            </label>
          </div>

          <label>
            Negative
            <input disabled={!editable} onChange={(event) => updateField("negativePrompt", event.target.value)} value={form.negativePrompt} />
          </label>

          <label>
            Description
            <input disabled={!editable} onChange={(event) => updateField("description", event.target.value)} value={form.description} />
          </label>

          <div className="control-grid compact-controls">
            <label>
              Prompt Prefix
              <textarea disabled={!editable} onChange={(event) => updateField("promptPrefix", event.target.value)} value={form.promptPrefix} />
            </label>
            <label>
              Prompt Suffix
              <textarea disabled={!editable} onChange={(event) => updateField("promptSuffix", event.target.value)} value={form.promptSuffix} />
            </label>
          </div>

          <section className="lora-picker" aria-label="Preset LoRAs">
            <div>
              <strong>Applied LoRAs</strong>
              <span>{form.loras.length}/3 installed and compatible</span>
            </div>
            {availableLoras.length ? (
              <div className="lora-choice-list">
                {availableLoras.map((lora) => {
                  const checked = selectedLoraIds(form).includes(lora.id);
                  const selected = form.loras.find((item) => item.id === lora.id);
                  return (
                    <div className={checked ? "lora-choice active editable-lora-choice" : "lora-choice editable-lora-choice"} key={lora.id}>
                      <label>
                        <input checked={checked} disabled={!editable || (!checked && form.loras.length >= 3)} onChange={() => toggleLora(lora.id)} type="checkbox" />
                        <span>
                          <strong>{lora.name ?? lora.id}</strong>
                          <small>{loraLabel(lora)}</small>
                        </span>
                      </label>
                      {checked ? (
                        <label>
                          Weight
                          <input
                            disabled={!editable}
                            max="2"
                            min="-2"
                            onChange={(event) => updateLoraWeight(lora.id, event.target.value)}
                            step="0.05"
                            type="number"
                            value={selected?.weight ?? ""}
                          />
                        </label>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty-panel compact-panel">{loraEmptyMessage}</div>
            )}
          </section>

          <section className="lora-import-panel" aria-label="Import LoRA">
            <div>
              <strong>Import LoRA</strong>
              <span>{selectedModel?.family ?? selectedModel?.name ?? "choose a model first"}</span>
            </div>
            <div className="segmented-control compact-segment" aria-label="LoRA import source">
              <button
                className={importForm.mode === "url" ? "active" : ""}
                disabled={!editable || importingLora}
                onClick={() => setImportForm((current) => ({ ...current, mode: "url" }))}
                type="button"
              >
                URL
              </button>
              <button
                className={importForm.mode === "file" ? "active" : ""}
                disabled={!editable || importingLora}
                onClick={() => setImportForm((current) => ({ ...current, mode: "file" }))}
                type="button"
              >
                Local File
              </button>
            </div>
            <div className="inline-create">
              {importForm.mode === "url" ? (
                <label>
                  Source URL
                  <input
                    disabled={!editable || !selectedModel || importingLora}
                    onChange={(event) => setImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                    placeholder="https://..."
                    value={importForm.sourceUrl}
                  />
                </label>
              ) : (
                <label>
                  Local File
                  <span className="file-picker-row">
                    <span className="file-upload-button">
                      Choose
                      <input
                        accept=".safetensors,.ckpt,.pt,.bin"
                        disabled={!editable || !selectedModel || importingLora}
                        key={fileInputKey}
                        onChange={(event) => setImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                        type="file"
                      />
                    </span>
                    <span className="selected-file-name">{importForm.file?.name ?? "No file selected"}</span>
                  </span>
                </label>
              )}
              <label>
                Name
                <input
                  disabled={!editable || !selectedModel || importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Optional"
                  value={importForm.name}
                />
              </label>
              <button
                disabled={!editable || busy || !selectedModel || (importForm.mode === "url" ? !importForm.sourceUrl.trim() : !importForm.file)}
                onClick={importLora}
                type="button"
              >
                {importingLora ? (importForm.mode === "file" ? "Uploading" : "Queueing...") : "Queue Import"}
              </button>
            </div>
            {!selectedModel ? <p className="helper-copy">Choose a model before importing so compatibility can be recorded.</p> : null}
          </section>

          {saveDisabledReason ? <p className="inline-warning">{saveDisabledReason}</p> : null}
          {message.text ? <p className={message.tone === "success" ? "inline-success" : "inline-warning"}>{message.text}</p> : null}
          <button className="primary-action" disabled={Boolean(saveDisabledReason) || busy} type="submit">
            {selectedPreset ? "Save Preset" : "Create Preset"}
          </button>
        </form>
      </div>
    </section>
  );
}
