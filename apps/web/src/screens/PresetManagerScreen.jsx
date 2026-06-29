import React, { useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
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
import { useAppContext } from "../context/AppContext.js";
import { qualityChoices } from "../jobTypes.js";

const workflowCards = [
  { id: "text_to_image", label: "Text → Image", desc: "Make stills from a description.", outputs: "Stills", icon: "Image" },
  { id: "edit_image", label: "Edit Image", desc: "Modify an existing still.", outputs: "Stills", icon: "Wand" },
  { id: "image_to_video", label: "Image → Video", desc: "Animate a starting frame.", outputs: "Video", icon: "Video" },
  { id: "text_to_video", label: "Text → Video", desc: "Generate a clip from prose.", outputs: "Video", icon: "Sparkle" },
  { id: "first_last_frame", label: "First/Last Frame", desc: "Render the in-betweens of two frames.", outputs: "Video", icon: "ArrowRight" },
];

const imageAspectChoices = ["1024x1024", "1536x1024", "1024x1536", "2048x1152"];
const videoResolutionChoices = ["768x512", "1280x720", "720x1280"];

function isVideoWorkflow(workflow) {
  return workflowModelType(workflow) === "video";
}

function renderSection({ step, title, help, optional, children }) {
  return (
    <section className="preset-section" key={step}>
      <header>
        <span className="step">{step}</span>
        <div>
          <h3>
            {title}
            {optional ? <span className="optional-tag">optional</span> : null}
          </h3>
          {help ? <p>{help}</p> : null}
        </div>
      </header>
      <div className="preset-section-body">{children}</div>
    </section>
  );
}

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

export function PresetManagerScreen() {
  const {
    activeProject,
    createPreset,
    deletePreset,
    duplicatePreset,
    imageModels,
    loras = [],
    presets = [],
    updatePreset,
    videoModels,
    setActiveView,
  } = useAppContext();
  const onOpenModels = () => setActiveView("Models");
  const models = useMemo(() => [...imageModels, ...videoModels], [imageModels, videoModels]);
  const [selectedPresetId, setSelectedPresetId] = useState(presets.find((preset) => preset.scope !== "builtin")?.id ?? "");
  const selectedPreset = presets.find((preset) => preset.id === selectedPresetId) ?? null;
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState(() => formFromPreset(selectedPreset, models[0]?.id));
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ tone: "neutral", text: "" });
  const [selectedLoraToAdd, setSelectedLoraToAdd] = useState("");
  const [showLoraPicker, setShowLoraPicker] = useState(false);
  const [showAdvancedMeta, setShowAdvancedMeta] = useState(false);
  const editable = !selectedPreset || selectedPreset.scope !== "builtin";
  const busy = saving;
  const availableModels = modelOptions(models, form.workflow);
  const selectedModel = models.find((model) => model.id === form.model) ?? availableModels[0] ?? null;
  const installedLoras = loras.filter((lora) => lora.installState !== "missing");
  const availableLoras = selectedModel ? installedLoras.filter((lora) => loraMatchesModel(lora, selectedModel)) : [];
  const selectedIds = selectedLoraIds(form);
  const addableLoras = availableLoras.filter((lora) => !selectedIds.includes(lora.id));
  const showLoraEmptyState = !availableLoras.length && !form.loras.length;
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
    : !installedLoras.length
      ? "No uploaded LoRAs yet. Manage LoRAs on the Models page."
      : hasPendingCompatibleLoras
        ? "No installed compatible LoRAs. Imports appear here after the Queue completes."
        : `No installed LoRAs match ${selectedModel.name ?? selectedModel.id}.`;

  useEffect(() => {
    if (selectedPreset && !presets.some((preset) => preset.id === selectedPreset.id)) {
      setSelectedPresetId(presets.find((preset) => preset.scope !== "builtin")?.id ?? "");
    }
  }, [presets, selectedPreset?.id]);

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

  useEffect(() => {
    // After adding a LoRA, select the next addable item so repeat adds stay one-click.
    if (!addableLoras.some((lora) => lora.id === selectedLoraToAdd)) {
      setSelectedLoraToAdd(addableLoras[0]?.id ?? "");
    }
  }, [addableLoras, selectedLoraToAdd]);

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

  function addLoraById(id) {
    if (!id) {
      return;
    }
    setForm((current) => {
      const hasLora = current.loras.some((lora) => lora.id === id);
      if (hasLora || current.loras.length >= 5) {
        return current;
      }
      const source = loras.find((lora) => lora.id === id);
      const weight = source?.defaultWeight ?? source?.weight ?? 0.8;
      return { ...current, loras: [...current.loras, { id, weight: String(weight) }] };
    });
  }

  function removeLora(id) {
    setForm((current) => ({ ...current, loras: current.loras.filter((lora) => lora.id !== id) }));
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
        await updatePreset(selectedPreset.id, payload, selectedPreset.scope);
        setMessage({ tone: "success", text: "Preset saved." });
      } else {
        const created = await createPreset(payload);
        setSelectedPresetId(created?.id ?? payload.id);
        setCreating(false);
        setShowLoraPicker(false);
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
        const created = await createPreset(payload);
        setSelectedPresetId(created.id);
        setMessage({ tone: "success", text: "Preset duplicated." });
        return;
      }
      const duplicated = await duplicatePreset(selectedPreset.id, form.scope);
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
      await deletePreset(selectedPreset.id, selectedPreset.scope);
      setSelectedPresetId("");
      setMessage({ tone: "success", text: "Preset archived." });
    } catch (err) {
      setMessage({ tone: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  }

  function startNewPreset() {
    setSelectedPresetId("");
    setForm(formFromPreset(null, modelOptions(models, "text_to_image")[0]?.id ?? models[0]?.id));
    setMessage({ tone: "neutral", text: "" });
    setShowLoraPicker(false);
    setShowAdvancedMeta(false);
    setCreating(true);
  }

  function cancelCreate() {
    setCreating(false);
    setShowLoraPicker(false);
    setMessage({ tone: "neutral", text: "" });
  }

  function selectPresetFromList(id) {
    setCreating(false);
    setShowLoraPicker(false);
    setSelectedPresetId(id);
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

      <div className={creating ? "preset-layout creating" : "preset-layout"}>
        <section className="preset-list" aria-label="Presets">
          {presets.length ? (
            presets.map((preset) => {
              const presetModel = models.find((model) => model.id === preset.model);
              const status = presetValidation(preset, loras, presetModel);
              return (
                <button
                  className={!creating && selectedPresetId === preset.id ? "preset-row active" : "preset-row"}
                  key={`${preset.scope}-${preset.id}`}
                  onClick={() => selectPresetFromList(preset.id)}
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

        {renderPresetForm()}
      </div>
    </section>
  );

  function renderPresetForm() {
    const isCreatingPreset = creating || !selectedPreset;
    const workflowDef = workflowCards.find((card) => card.id === form.workflow) ?? workflowCards[0];
    const isVideo = isVideoWorkflow(form.workflow);
    const promptPreviewActive = form.promptPrefix.trim() || form.promptSuffix.trim();
    const resolutionChoices = isVideo ? videoResolutionChoices : imageAspectChoices;
    const addableLorasPicker = addableLoras;
    const summaryLoraNames = form.loras
      .map((selected) => loras.find((lora) => lora.id === selected.id)?.name ?? selected.id)
      .join(", ");

    return (
      <form className="preset-create-shell" onSubmit={savePreset}>
        <div className="preset-create-head">
          {isCreatingPreset ? (
            <button className="preset-create-back" onClick={cancelCreate} type="button">
              <Icon.ArrowLeft size={14} /> Back to Presets
            </button>
          ) : (
            <div className="preset-edit-context">
              <span className="preset-edit-kicker">Editing preset</span>
              <strong>{selectedPreset.name ?? selectedPreset.id}</strong>
              <small>{selectedPreset.scope === "builtin" ? "Built-in preset" : `${selectedPreset.scope ?? "global"} preset`}</small>
            </div>
          )}
          <div className="preset-create-actions">
            {isCreatingPreset ? (
              <button className="btn-secondary" onClick={cancelCreate} type="button">
                Cancel
              </button>
            ) : null}
            <button
              className="btn-primary"
              disabled={Boolean(saveDisabledReason) || busy}
              type="submit"
            >
              {isCreatingPreset ? <Icon.Plus size={14} /> : <Icon.Preset size={14} />}
              <span>{busy ? "Saving..." : isCreatingPreset ? "Create Preset" : "Save Preset"}</span>
            </button>
          </div>
        </div>

        <div className="preset-create-body">
          <div className="preset-form-stack">
            {/* 1. Identity */}
            {renderSection({
              step: "1",
              title: "Identity",
              help: "Name it something you'll recognize in the picker.",
              children: (
                <>
                  <label className="field field-name">
                    <span>Name</span>
                    <input
                      disabled={!editable}
                      onChange={(event) => updateField("name", event.target.value)}
                      placeholder="e.g. Atrium portraits"
                      required
                      value={form.name}
                    />
                  </label>
                  <label className="field">
                    <span>ID</span>
                    <input
                      disabled={!isCreatingPreset || !editable}
                      onChange={(event) => updateField("id", event.target.value)}
                      placeholder="auto-generated from name"
                      required
                      value={form.id}
                    />
                    <small>{form.id ? `id · ${form.id}` : "id · auto"}</small>
                  </label>
                  <label className="field">
                    <span>Description</span>
                    <input
                      disabled={!editable}
                      onChange={(event) => updateField("description", event.target.value)}
                      placeholder="One line — what kind of shot this makes"
                      value={form.description}
                    />
                  </label>
                  <label className="field">
                    <span>Available in</span>
                    <div className="scope-segment" role="radiogroup" aria-label="Scope">
                      <button
                        aria-checked={form.scope === "project"}
                        className={form.scope === "project" ? "active" : ""}
                        disabled={!activeProject || !editable}
                        onClick={() => updateField("scope", "project")}
                        role="radio"
                        type="button"
                      >
                        <Icon.Folder size={14} /> This project
                      </button>
                      <button
                        aria-checked={form.scope === "global"}
                        className={form.scope === "global" ? "active" : ""}
                        disabled={!editable}
                        onClick={() => updateField("scope", "global")}
                        role="radio"
                        type="button"
                      >
                        <Icon.Stars size={14} /> All projects
                      </button>
                    </div>
                  </label>
                </>
              ),
            })}

            {/* 2. Workflow */}
            {renderSection({
              step: "2",
              title: "What does this preset do?",
              help: "Pick the kind of generation. The available models update to match.",
              children: (
                <div className="workflow-grid">
                  {workflowCards.map((card) => {
                    const IconComponent = Icon[card.icon] ?? Icon.Sparkle;
                    return (
                      <button
                        className={form.workflow === card.id ? "workflow-card active" : "workflow-card"}
                        disabled={!editable}
                        key={card.id}
                        onClick={() => updateField("workflow", card.id)}
                        type="button"
                      >
                        <span className="workflow-card-icon">
                          <IconComponent size={16} />
                        </span>
                        <strong>{card.label}</strong>
                        <small>{card.desc}</small>
                        <span className="chip">{card.outputs}</span>
                      </button>
                    );
                  })}
                </div>
              ),
            })}

            {/* 3. Model */}
            {renderSection({
              step: "3",
              title: "Which model runs it?",
              help: availableModels.length
                ? `${availableModels.length} model${availableModels.length === 1 ? "" : "s"} compatible with ${workflowDef.label}.`
                : `No models compatible with ${workflowDef.label} yet. Install one from Models.`,
              children: (
                <div className="preset-model-cards">
                  {availableModels.length ? (
                    availableModels.map((modelOption) => (
                      <button
                        className={form.model === modelOption.id ? "preset-model-row active" : "preset-model-row"}
                        disabled={!editable}
                        key={modelOption.id}
                        onClick={() => updateField("model", modelOption.id)}
                        type="button"
                      >
                        <span className={form.model === modelOption.id ? "preset-model-radio active" : "preset-model-radio"} />
                        <span className="preset-model-meta">
                          <strong>{modelOption.name ?? modelOption.id}</strong>
                          <span>{modelOption.ui?.description ?? `${modelOption.type ?? "model"} family`}</span>
                        </span>
                        <span className="preset-model-tags">
                          {modelOption.family ? <span className="chip">{modelOption.family}</span> : null}
                          <span className="chip">{modelOption.type ?? "model"}</span>
                        </span>
                      </button>
                    ))
                  ) : (
                    <div className="empty-panel compact-panel">No models</div>
                  )}
                </div>
              ),
            })}

            {/* 4. Prompt template */}
            {renderSection({
              step: "4",
              title: "Prompt template",
              help: "Text added before and after whatever the user types when running this preset.",
              children: (
                <div className="prompt-template">
                  <label className="field">
                    <span>Always prepend</span>
                    <textarea
                      disabled={!editable}
                      onChange={(event) => updateField("promptPrefix", event.target.value)}
                      placeholder="e.g. Cinematic 35mm, warm tungsten,"
                      rows={2}
                      value={form.promptPrefix}
                    />
                  </label>
                  <label className="field">
                    <span>Always append</span>
                    <textarea
                      disabled={!editable}
                      onChange={(event) => updateField("promptSuffix", event.target.value)}
                      placeholder="e.g. shallow depth of field, neutral grade"
                      rows={2}
                      value={form.promptSuffix}
                    />
                  </label>
                  <label className="field">
                    <span>Negative</span>
                    <input
                      disabled={!editable}
                      onChange={(event) => updateField("negativePrompt", event.target.value)}
                      placeholder="oversaturated, hands, text, watermark"
                      value={form.negativePrompt}
                    />
                  </label>
                  {promptPreviewActive ? (
                    <div className="prompt-preview">
                      <span className="prompt-preview-label">Final prompt</span>
                      <p>
                        {form.promptPrefix.trim() ? <span className="prefix">{form.promptPrefix.trim()} </span> : null}
                        <span className="user-input">{"{user prompt}"}</span>
                        {form.promptSuffix.trim() ? <span className="suffix"> {form.promptSuffix.trim()}</span> : null}
                      </p>
                    </div>
                  ) : null}
                </div>
              ),
            })}

            {/* 5. Defaults */}
            {renderSection({
              step: "5",
              title: "Defaults",
              help: "Pre-fill the studio with these. Anyone running the preset can still override.",
              children: (
                <div className="preset-defaults-grid">
                  <label className="field">
                    <span>{isVideo ? "Clips per batch" : "Variations"}</span>
                    <select
                      disabled={!editable}
                      onChange={(event) => updateField("count", event.target.value)}
                      value={form.count}
                    >
                      <option value="">No default</option>
                      {[1, 2, 3, 4, 6, 8].map((n) => (
                        <option key={n} value={String(n)}>
                          {n}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>{isVideo ? "Resolution" : "Aspect"}</span>
                    <select
                      disabled={!editable}
                      onChange={(event) => updateField("resolution", event.target.value)}
                      value={form.resolution}
                    >
                      <option value="">No default</option>
                      {resolutionChoices.map((value) => (
                        <option key={value} value={value}>
                          {value.replace("x", " × ")}
                        </option>
                      ))}
                    </select>
                  </label>
                  {isVideo ? (
                    <>
                      <label className="field">
                        <span>Duration</span>
                        <select
                          disabled={!editable}
                          onChange={(event) => updateField("duration", event.target.value)}
                          value={form.duration}
                        >
                          <option value="">No default</option>
                          {[3, 4, 6, 8, 10, 12].map((d) => (
                            <option key={d} value={String(d)}>
                              {d}s
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field">
                        <span>Frames</span>
                        <select
                          disabled={!editable}
                          onChange={(event) => updateField("fps", event.target.value)}
                          value={form.fps}
                        >
                          <option value="">No default</option>
                          {[24, 25, 30].map((f) => (
                            <option key={f} value={String(f)}>
                              {f} fps
                            </option>
                          ))}
                        </select>
                      </label>
                    </>
                  ) : null}
                  <label className={isVideo ? "field" : "field field-span-2"}>
                    <span>Quality</span>
                    <div className="quality-pick" role="radiogroup" aria-label="Quality">
                      {qualityChoices.map(([value, label]) => (
                        <button
                          aria-checked={form.quality === value}
                          className={form.quality === value ? "active" : ""}
                          disabled={!editable}
                          key={value}
                          onClick={() => updateField("quality", value)}
                          role="radio"
                          type="button"
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </label>
                </div>
              ),
            })}

            {/* 6. LoRAs */}
            {renderSection({
              step: "6",
              title: "Style stack",
              help: `Up to 5 LoRAs layered with this preset. Compatible with ${selectedModel?.name ?? "the chosen model"}.`,
              optional: true,
              children: (
                <section aria-label="Preset LoRAs">
                  <div className="lora-stack">
                    {form.loras.length ? (
                      <div className="lora-choice-list">
                        {form.loras.map((selected) => {
                          const lora = loras.find((item) => item.id === selected.id);
                          const missing = !lora || lora.installState === "missing";
                          const incompatible = lora && selectedModel && !loraMatchesModel(lora, selectedModel);
                          return (
                            <div className={incompatible || missing ? "lora-choice editable-lora-choice warning" : "lora-choice active editable-lora-choice"} key={selected.id}>
                              <span>
                                <strong>{lora?.name ?? selected.id}</strong>
                                <small>
                                  {missing
                                    ? "Missing or still importing"
                                    : incompatible
                                      ? `${loraLabel(lora)} | incompatible with ${selectedModel?.name ?? selectedModel?.id}`
                                      : loraLabel(lora)}
                                </small>
                              </span>
                              <div className="lora-selection-actions">
                                <label>
                                  Weight
                                  <input
                                    disabled={!editable || missing || incompatible}
                                    max="2"
                                    min="-2"
                                    onChange={(event) => updateLoraWeight(selected.id, event.target.value)}
                                    step="0.05"
                                    type="number"
                                    value={selected?.weight ?? ""}
                                  />
                                </label>
                                <button disabled={!editable} onClick={() => removeLora(selected.id)} type="button">
                                  Remove
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : null}

                    {showLoraEmptyState ? (
                      <div className="empty-panel compact-panel">
                        <span>{loraEmptyMessage}</span>
                        {onOpenModels ? (
                          <button onClick={onOpenModels} type="button">
                            Open Models
                          </button>
                        ) : null}
                      </div>
                    ) : null}

                    {form.loras.length < 5 ? (
                      showLoraPicker ? (
                        <div className="lora-picker-panel">
                          <strong>Pick a LoRA</strong>
                          {addableLorasPicker.length ? (
                            <div className="lora-picker-list">
                              {addableLorasPicker.map((lora) => (
                                <button
                                  className="lora-pick-row"
                                  key={lora.id}
                                  onClick={() => {
                                    addLoraById(lora.id);
                                    setShowLoraPicker(false);
                                  }}
                                  type="button"
                                >
                                  <span>
                                    <strong>{lora.name ?? lora.id}</strong>{" "}
                                    {lora.family ? <span className="chip">{lora.family}</span> : null}
                                  </span>
                                  <Icon.Plus size={14} />
                                </button>
                              ))}
                            </div>
                          ) : (
                            <p className="lora-pick-empty">
                              {loraEmptyMessage}
                            </p>
                          )}
                          <div className="lora-picker-actions">
                            <button onClick={() => setShowLoraPicker(false)} type="button">
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <button
                          className="lora-add"
                          data-count={`${form.loras.length}/3`}
                          disabled={!editable || !availableLoras.length}
                          onClick={() => setShowLoraPicker(true)}
                          type="button"
                        >
                          <Icon.Plus size={14} />
                          <span>Add LoRA</span>
                        </button>
                      )
                    ) : null}
                  </div>
                </section>
              ),
            })}

            <button
              className="preset-advanced-toggle"
              onClick={() => setShowAdvancedMeta((value) => !value)}
              type="button"
            >
              <Icon.ChevDown
                className={showAdvancedMeta ? "chev-rotate open" : "chev-rotate"}
                size={14}
              />
              {showAdvancedMeta ? "Hide advanced metadata" : "Show advanced metadata"}
            </button>

            {showAdvancedMeta ? renderSection({
              step: "·",
              title: "Advanced",
              children: (
                <div className="control-grid">
                  <label>
                    Sort order
                    <input
                      disabled={!editable}
                      onChange={(event) => updateField("order", event.target.value)}
                      placeholder="0"
                      type="number"
                      value={form.order}
                    />
                  </label>
                  <label>
                    Derived modes
                    <input disabled readOnly value={compactModeList(form.workflow)} />
                  </label>
                </div>
              ),
            }) : null}

            {saveDisabledReason ? <p className="inline-warning">{saveDisabledReason}</p> : null}
            {message.text ? <p className={message.tone === "success" ? "inline-success" : "inline-warning"}>{message.text}</p> : null}
          </div>

          <aside className="preset-summary">
            <div className="summary-card">
              <div className="summary-section-title"><h2>Details</h2></div>
              <dl className="detail-stack">
                <div className="detail-row">
                  <dt>Type</dt>
                  <dd>{workflowDef.label}</dd>
                </div>
                <div className="detail-row">
                  <dt>Model</dt>
                  <dd>{selectedModel?.name ?? selectedModel?.id ?? "—"}</dd>
                </div>
                <div className="detail-row">
                  <dt>Scope</dt>
                  <dd>{form.scope === "project" ? activeProject?.name ?? "Project" : "All projects"}</dd>
                </div>
                <div className="detail-row">
                  <dt>Outputs</dt>
                  <dd>{form.count || "default"} × {workflowDef.outputs.toLowerCase()}</dd>
                </div>
                <div className="detail-row">
                  <dt>Resolution</dt>
                  <dd>{form.resolution ? form.resolution.replace("x", " × ") : "default"}</dd>
                </div>
                {isVideo ? (
                  <div className="detail-row">
                    <dt>Duration</dt>
                    <dd>{form.duration ? `${form.duration}s` : "default"} @ {form.fps ? `${form.fps} fps` : "default"}</dd>
                  </div>
                ) : null}
                <div className="detail-row">
                  <dt>Quality</dt>
                  <dd>{form.quality || "default"}</dd>
                </div>
                <div className="detail-row">
                  <dt>LoRAs</dt>
                  <dd>{summaryLoraNames || "none"}</dd>
                </div>
              </dl>
            </div>

            {!form.name.trim() ? (
              <div className="summary-card summary-warning">
                <strong>Add a name to save</strong>
                <span>You can change everything else later.</span>
              </div>
            ) : null}
          </aside>
        </div>
      </form>
    );
  }
}
