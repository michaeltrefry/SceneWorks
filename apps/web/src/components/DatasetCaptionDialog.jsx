import React from "react";
import { Modal } from "./Modal.jsx";

// Caption settings modal (sc-2025). One dialog drives both "Caption all" and a
// single image's "Re-Caption" — the dataset editor owns the captioner settings
// state and the job submission; this component only renders the controls and a
// Run button scoped to the target.
export function DatasetCaptionDialog({
  settings,
  onChange,
  gpuOptions = [],
  captionTypes = [],
  captionLengths = [],
  extraOptions = [],
  promptValue = "",
  scope,
  running = false,
  onRun,
  onToggleExtra,
  onClose,
}) {
  const single = scope?.type === "item";
  const title = single ? `Re-caption ${scope.name ?? "image"}` : "Caption images";
  const runLabel = single ? "Re-caption image" : settings.recaption ? "Re-caption all" : "Caption missing";
  const joy = settings.captioner === "joy_caption";

  return (
    <Modal className="dataset-caption-modal" labelledBy="dataset-caption-title" onClose={onClose}>
      <header className="dataset-caption-head">
        <div>
          <p className="eyebrow">Captioner</p>
          <h2 id="dataset-caption-title">{title}</h2>
        </div>
        <button className="modal-close" onClick={onClose} type="button">
          Close
        </button>
      </header>

      <div className="dataset-caption-body">
        <label>
          Method
          <select onChange={(event) => onChange("captioner", event.target.value)} value={settings.captioner}>
            <option value="joy_caption">Joy Caption</option>
            <option value="metadata">Metadata fallback</option>
          </select>
        </label>

        {joy ? (
          <>
            <label>
              Model
              <input onChange={(event) => onChange("modelNameOrPath", event.target.value)} value={settings.modelNameOrPath} />
            </label>
            <label>
              GPU
              <select onChange={(event) => onChange("requestedGpu", event.target.value)} value={settings.requestedGpu}>
                {gpuOptions.map((gpu) => (
                  <option key={gpu} value={gpu}>
                    {gpu}
                  </option>
                ))}
              </select>
            </label>
            <div className="dataset-caption-row">
              <label>
                Type
                <select onChange={(event) => onChange("captionType", event.target.value)} value={settings.captionType}>
                  {captionTypes.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Length
                <select onChange={(event) => onChange("captionLength", event.target.value)} value={settings.captionLength}>
                  {captionLengths.map((length) => (
                    <option key={length} value={length}>
                      {length}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label>
              Character name
              <input onChange={(event) => onChange("nameInput", event.target.value)} value={settings.nameInput} />
            </label>
            <label>
              Caption prompt
              <textarea onChange={(event) => onChange("captionPrompt", event.target.value)} rows={6} value={promptValue} />
            </label>
            <div className="dataset-caption-row">
              <label>
                Temperature
                <input
                  max="2"
                  min="0"
                  onChange={(event) => onChange("temperature", event.target.value)}
                  step="0.05"
                  type="number"
                  value={settings.temperature}
                />
              </label>
              <label>
                Top P
                <input
                  max="1"
                  min="0"
                  onChange={(event) => onChange("topP", event.target.value)}
                  step="0.05"
                  type="number"
                  value={settings.topP}
                />
              </label>
              <label>
                Max tokens
                <input
                  max="1024"
                  min="1"
                  onChange={(event) => onChange("maxNewTokens", event.target.value)}
                  step="1"
                  type="number"
                  value={settings.maxNewTokens}
                />
              </label>
            </div>
            <label className="training-toggle-line">
              <input checked={settings.lowVram} onChange={(event) => onChange("lowVram", event.target.checked)} type="checkbox" />
              <span>Low VRAM</span>
            </label>
            {single ? null : (
              <label className="training-toggle-line">
                <input checked={settings.recaption} onChange={(event) => onChange("recaption", event.target.checked)} type="checkbox" />
                <span>Re-caption images that already have a caption</span>
              </label>
            )}
            <div className="dataset-caption-options">
              {extraOptions.map((option) => (
                <label className="training-toggle-line" key={option.value}>
                  <input
                    checked={settings.extraOptions.includes(option.value)}
                    onChange={() => onToggleExtra(option.value)}
                    type="checkbox"
                  />
                  <span>{option.label}</span>
                </label>
              ))}
            </div>
          </>
        ) : null}
      </div>

      <footer className="dataset-caption-footer">
        <button onClick={onClose} type="button">
          Cancel
        </button>
        <button className="primary-action" disabled={running} onClick={onRun} type="button">
          {running ? "Queuing…" : runLabel}
        </button>
      </footer>
    </Modal>
  );
}
