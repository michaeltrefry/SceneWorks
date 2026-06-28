import React from "react";
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

// Shared advanced-generation controls for Character Studio's Angle Set and Pose
// Library panels (sc-3857). Mirrors the Image Studio advanced panel
// (screens/ImageStudio.jsx) — same labels and the same `advanced`-dict keys — so
// the worker reads identical fields and the two studios stay consistent.
//
// State lives in `useCharacterAdvancedOptions(model, { defaultNegativePrompt })`;
// `<CharacterAdvancedOptions state={…} />` is purely presentational.
// `buildAdvanced(base)` folds the chosen knobs into the job's `advanced` dict and
// emits ONLY non-default values, so leaving a control untouched preserves the
// worker's model default (no behavioural change from today's hardcoded payload).
//
// The Sampler / Scheduler dropdowns render only when the model declares
// `limits.samplers` / `limits.schedulers` with more than one option (same gate as
// Image Studio). InstantID does not declare those yet — the worker sampler
// registry is flow-matching-only and needs epsilon-aware specs before an SDXL
// model can honour a sampler swap (sc-3857 §C). The control is therefore
// forward-compatible: it lights up the moment the manifest + worker land, with no
// change here.

export function useCharacterAdvancedOptions(
  model,
  { defaultNegativePrompt = "", identityStructureMode = "single" } = {},
) {
  const ui = model?.ui ?? {};
  const referenceStrengthDefault =
    typeof ui.referenceStrengthDefault === "number" ? ui.referenceStrengthDefault : 0.8;
  const identityStructure = ui.identityStructure ?? null;
  // The Identity-structure (controlnetConditioningScale) lock defaults lower for an angle set than
  // for single-image generation (sc-8354): the softer lock sharpens the off-axis views. Track the
  // backbone's `angleSetDefault` in that mode, falling back to the single-image `default`.
  const identityStructureDefault =
    identityStructureMode === "angleSet" && typeof identityStructure?.angleSetDefault === "number"
      ? identityStructure.angleSetDefault
      : typeof identityStructure?.default === "number"
        ? identityStructure.default
        : 0.8;

  const samplerOptions = samplerOptionsFromModel(model);
  const schedulerOptions = schedulerOptionsFromModel(model);
  const showSamplerPicker = samplerOptions.length > 1;
  const showSchedulerPicker = schedulerOptions.length > 1;

  const [open, setOpen] = React.useState(false);
  const [ipAdapterScale, setIpAdapterScale] = React.useState(referenceStrengthDefault);
  const [controlnetScale, setControlnetScale] = React.useState(identityStructureDefault);
  // Guidance / Steps stay "" = "use the model default"; the placeholder shows the
  // resolved default and the worker reads it off MODEL_TARGETS when unset.
  const [guidance, setGuidance] = React.useState("");
  const [steps, setSteps] = React.useState("");
  const [sampler, setSampler] = React.useState(samplerDefaultFromModel(model));
  const [scheduler, setScheduler] = React.useState(schedulerDefaultFromModel(model));
  const [schedulerShift, setSchedulerShift] = React.useState(schedulerShiftDefaultFromModel(model));
  const [seed, setSeed] = React.useState("");
  const [negativePrompt, setNegativePrompt] = React.useState(defaultNegativePrompt);

  // Reset the model-derived tuning whenever the backbone changes so a value from a
  // different model never leaks (mirrors ImageStudio's reference-tuning reset).
  const modelId = model?.id;
  React.useEffect(() => {
    setIpAdapterScale(referenceStrengthDefault);
    setControlnetScale(identityStructureDefault);
    setSampler(samplerDefaultFromModel(model));
    setScheduler(schedulerDefaultFromModel(model));
    setSchedulerShift(schedulerShiftDefaultFromModel(model));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId]);

  // Re-seed the editable negative prompt with the curated baseline when it changes
  // (e.g. switching character/panel). The user edits from that starting point.
  React.useEffect(() => {
    setNegativePrompt(defaultNegativePrompt);
  }, [defaultNegativePrompt]);

  function buildAdvanced(base = {}) {
    const advanced = { ...base, ipAdapterScale };
    if (identityStructure) {
      advanced.controlnetConditioningScale = controlnetScale;
    }
    if (guidance !== "" && Number.isFinite(Number(guidance))) {
      advanced.guidanceScale = Number(guidance);
    }
    if (steps !== "" && Number.isFinite(Number(steps))) {
      advanced.steps = Number(steps);
    }
    if (showSamplerPicker && sampler && sampler !== "default") {
      advanced.sampler = sampler;
    }
    if (showSchedulerPicker && scheduler && scheduler !== "default") {
      advanced.scheduler = scheduler;
      if (scheduler === "shift" && Number.isFinite(Number(schedulerShift))) {
        advanced.schedulerShift = Number(schedulerShift);
      }
    }
    return advanced;
  }

  return {
    open,
    setOpen,
    ipAdapterScale,
    setIpAdapterScale,
    controlnetScale,
    setControlnetScale,
    guidance,
    setGuidance,
    steps,
    setSteps,
    sampler,
    setSampler,
    scheduler,
    setScheduler,
    schedulerShift,
    setSchedulerShift,
    seed,
    setSeed,
    negativePrompt,
    setNegativePrompt,
    model,
    identityStructure,
    // Optional label/range override for the primary reference-strength slider (sc-8278: klein maps
    // it to image-guidance over 1.0–2.5). Absent ⇒ the legacy "Reference strength" 0–1 slider.
    referenceStrength: ui.referenceStrength ?? null,
    samplerOptions,
    schedulerOptions,
    showSamplerPicker,
    showSchedulerPicker,
    buildAdvanced,
    // Top-level job-payload seed: null when blank (worker derives from the prompt).
    seedValue: seed === "" ? null : Number(seed),
  };
}

export function CharacterAdvancedOptions({ state }) {
  const {
    open,
    setOpen,
    ipAdapterScale,
    setIpAdapterScale,
    controlnetScale,
    setControlnetScale,
    guidance,
    setGuidance,
    steps,
    setSteps,
    sampler,
    setSampler,
    scheduler,
    setScheduler,
    schedulerShift,
    setSchedulerShift,
    seed,
    setSeed,
    negativePrompt,
    setNegativePrompt,
    model,
    identityStructure,
    referenceStrength: referenceStrengthCfg,
    samplerOptions,
    schedulerOptions,
    showSamplerPicker,
    showSchedulerPicker,
  } = state;

  const guidancePlaceholder = (() => {
    const value = guidanceDefaultFromModel(model);
    return value == null ? "" : String(value);
  })();
  const stepsPlaceholder = (() => {
    const value = stepsDefaultFromModel(model);
    return value == null ? "" : String(value);
  })();

  return (
    <div className="character-advanced">
      <button className="advanced-toggle" onClick={() => setOpen((value) => !value)} type="button">
        {open ? "Hide advanced" : "Advanced"}
      </button>
      {open ? (
        <div className="advanced-panel">
          <label className="reference-strength">
            {referenceStrengthCfg?.label ??
              (identityStructure ? "Identity strength" : "Reference strength")}
            <input
              max={referenceStrengthCfg?.max ?? 1}
              min={referenceStrengthCfg?.min ?? 0}
              onChange={(event) => setIpAdapterScale(Number(event.target.value))}
              step={referenceStrengthCfg?.step ?? 0.05}
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
          <label>
            Guidance
            <input
              max="30"
              min="0"
              onChange={(event) => setGuidance(event.target.value)}
              placeholder={guidancePlaceholder}
              step="0.1"
              type="number"
              value={guidance}
            />
          </label>
          <label>
            Steps
            <input
              max="80"
              min="1"
              onChange={(event) => setSteps(event.target.value)}
              placeholder={stepsPlaceholder}
              type="number"
              value={steps}
            />
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
          {showSchedulerPicker && scheduler === "shift" ? (
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
            Seed
            <input
              onChange={(event) => setSeed(event.target.value)}
              placeholder="Random"
              type="number"
              value={seed}
            />
          </label>
          <label className="prompt-field">
            Negative prompt
            <textarea onChange={(event) => setNegativePrompt(event.target.value)} rows={3} value={negativePrompt} />
          </label>
        </div>
      ) : null}
    </div>
  );
}
