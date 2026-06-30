import React from "react";

import { Icon } from "../../components/Icons.jsx";
import { DatasetDoctorReadout } from "./DatasetDoctor.jsx";
import {
  lossTypeOptions,
  networkTypeLabel,
  optimizerLabel,
  optionLabel,
  timestepBiasOptions,
  timestepTypeOptions,
  trainingAdapterVersionLabels,
} from "../../training/trainingConfig.js";

// Configure-training-job panel (sc-4199): extracted verbatim from TrainingStudio
// so the target/preset/dataset selectors, the advanced config grid, and the
// run-mode actions live in their own component. All state and handlers are owned
// by the TrainingStudio screen and passed in as props — behavior is unchanged.
export function ConfigureJobPanel({
  active,
  setActiveView,
  configReady,
  trainingTargetsError,
  trainingPresetsError,
  configError,
  configMessage,
  selectedTarget,
  setSelectedTargetId,
  trainingTargets,
  macTargetBlocked,
  updateSelectedPreset,
  selectedPreset,
  targetPresets,
  openDataset,
  activeDataset,
  datasets,
  updateConfigDraft,
  configDraft,
  outputScopes,
  visibleQualityPresets,
  gpuOptions,
  customizedConfigLabels,
  showAdvancedConfig,
  setShowAdvancedConfig,
  showNetworkType,
  networkTypeOptions,
  macLokrOnWanBlocked,
  isLokrNetwork,
  visibleOptimizerOptions,
  visibleLrSchedulerOptions,
  showTrainingAdapter,
  visibleTrainingAdapterVersions,
  visibleResolutionOptions,
  configWarnings,
  trainingRunMode,
  submittingJob,
  setTrainingRunMode,
  resetConfigDefaults,
  submitTrainingJob,
  configSnapshot,
  readiness = null,
  readinessLoading = false,
  readinessBlocksTraining = false,
  onRemoveDuplicates,
  onUpscaleLowRes,
  onSmartCrop,
  onStripExif,
  onAnalyzeDataset,
  onAnalyzeFaces,
}) {
  return (
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
            <label>
              Sample count
              <input
                min="0"
                onChange={(event) => updateConfigDraft("sampleCount", event.target.value)}
                type="number"
                value={configDraft.sampleCount ?? ""}
              />
            </label>
          </div>

          <label className="training-sample-prompts">
            Sample prompts
            <textarea
              onChange={(event) => updateConfigDraft("samplePrompts", event.target.value)}
              placeholder="One prompt per line. Leave blank to use the trigger-phrase defaults."
              rows={4}
              value={configDraft.samplePrompts ?? ""}
            />
            <span className="training-field-hint">
              One prompt per line. Previews cycle through this list up to the sample count.
            </span>
          </label>

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

          {/* Dataset Doctor readout before the Train button (sc-6534). Advisory: it
              only hard-blocks training when the gate is Blocked (too few images / a
              fatal flag); warnings stay informational. */}
          <DatasetDoctorReadout
            report={readiness}
            loading={readinessLoading}
            compact
            onRemoveDuplicates={onRemoveDuplicates}
            onUpscaleLowRes={onUpscaleLowRes}
            onSmartCrop={onSmartCrop}
            onStripExif={onStripExif}
            onAnalyzeDataset={onAnalyzeDataset}
            onAnalyzeFaces={onAnalyzeFaces}
          />
          {readinessBlocksTraining ? (
            <p className="inline-warning">
              This dataset isn’t ready to train yet — open Data Sets to add or fix images.
            </p>
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
              disabled={!configReady || submittingJob || readinessBlocksTraining}
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
  );
}
