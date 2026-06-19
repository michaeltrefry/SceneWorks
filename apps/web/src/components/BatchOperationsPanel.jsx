import React, { useState } from "react";
import { Modal } from "./Modal.jsx";
import { BATCH_OPS } from "../batchOps.js";

// Batch operations panel (sc-6112): pick ONE op (upscale / detail / edit) + shared
// params and apply it across the selected image assets. Presentational — it owns the
// op/param form state and calls `onRun(op, params)`; the parent does the fan-out
// (decode dims for edit, POST one job per asset) and feeds back per-item `items` +
// the aggregate `progress` so this panel renders the live progress view.

const STATUS_LABEL = { queued: "Queued", running: "Running…", completed: "Done", failed: "Failed" };

export function BatchOperationsPanel({
  assets,
  editModels = [],
  detailModels = [],
  upscaleEngines = [],
  busy = false,
  items = null,
  progress = null,
  onRun,
  onClose,
}) {
  const [op, setOp] = useState("upscale");
  // Upscale params.
  const [engine, setEngine] = useState(upscaleEngines[0]?.key ?? "real-esrgan");
  const [factor, setFactor] = useState(4);
  const [softness, setSoftness] = useState(0.5);
  // Detail params.
  const [detailModel, setDetailModel] = useState(detailModels[0]?.id ?? "");
  const [strength, setStrength] = useState(0.55);
  const [cnScale, setCnScale] = useState(0.7);
  // Edit params.
  const [editModel, setEditModel] = useState(editModels[0]?.id ?? "");
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState("");

  const count = assets.length;
  const running = items != null;

  // Resolve the active selections against the (async-loaded) option lists so a stale
  // or empty default falls back to the first available entry.
  const activeEngine = upscaleEngines.find((e) => e.key === engine) ?? upscaleEngines[0] ?? null;
  const factors = activeEngine?.factors ?? [2, 4];
  const effectiveFactor = factors.includes(factor) ? factor : factors[0];
  const activeDetailModel = detailModels.find((m) => m.id === detailModel)?.id ?? detailModels[0]?.id ?? "";
  const activeEditModel = editModels.find((m) => m.id === editModel)?.id ?? editModels[0]?.id ?? "";

  const missingModel =
    (op === "detail" && !activeDetailModel) || (op === "edit" && !activeEditModel) || (op === "upscale" && !activeEngine);
  const missingPrompt = op === "edit" && !prompt.trim();
  const canRun = count > 0 && !busy && !missingModel && !missingPrompt;

  function run() {
    if (!canRun) return;
    if (op === "upscale") {
      onRun(op, { engine: activeEngine.key, factor: effectiveFactor, softness });
    } else if (op === "detail") {
      onRun(op, { model: activeDetailModel, strength, cnScale });
    } else {
      onRun(op, { model: activeEditModel, prompt: prompt.trim(), seed });
    }
  }

  return (
    <Modal className="batch-ops-modal" label="Batch operations" onClose={onClose}>
      <div className="batch-ops-head">
        <h3>Batch: {count} image{count === 1 ? "" : "s"}</h3>
        <button className="modal-close" onClick={onClose} title="Close" type="button">
          ✕
        </button>
      </div>

      {running ? (
        <div className="batch-ops-progress">
          <p className="batch-ops-summary" aria-live="polite">
            {progress?.done ?? 0} / {progress?.total ?? count} done
            {progress?.failed ? ` · ${progress.failed} failed` : ""}
          </p>
          <ul className="batch-ops-items">
            {items.map((item) => (
              <li className={`batch-ops-item batch-ops-item-${item.status}`} key={item.asset.id}>
                <span className="batch-ops-item-name">{item.asset.displayName ?? item.asset.id}</span>
                <span className="batch-ops-item-status">{STATUS_LABEL[item.status] ?? item.status}</span>
              </li>
            ))}
          </ul>
          <div className="batch-ops-actions">
            <button onClick={onClose} type="button">
              {progress?.allDone ? "Done" : "Close"}
            </button>
          </div>
        </div>
      ) : (
        <div className="batch-ops-form">
          <div className="batch-ops-tabs" role="group" aria-label="Operation">
            {BATCH_OPS.map((entry) => (
              <button
                aria-pressed={op === entry.key}
                className={op === entry.key ? "active" : ""}
                key={entry.key}
                onClick={() => setOp(entry.key)}
                type="button"
              >
                {entry.label}
              </button>
            ))}
          </div>

          {op === "upscale" ? (
            <div className="batch-ops-params">
              <label>
                Engine
                <select onChange={(e) => setEngine(e.target.value)} value={activeEngine?.key ?? ""}>
                  {upscaleEngines.map((e) => (
                    <option key={e.key} value={e.key}>
                      {e.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Factor
                <select onChange={(e) => setFactor(Number(e.target.value))} value={effectiveFactor}>
                  {factors.map((f) => (
                    <option key={f} value={f}>
                      {f}×
                    </option>
                  ))}
                </select>
              </label>
              {activeEngine?.softness ? (
                <label>
                  Softness {softness.toFixed(2)}
                  <input
                    max="1"
                    min="0"
                    onChange={(e) => setSoftness(Number(e.target.value))}
                    step="0.05"
                    type="range"
                    value={softness}
                  />
                </label>
              ) : null}
            </div>
          ) : null}

          {op === "detail" ? (
            <div className="batch-ops-params">
              <label>
                Model
                {detailModels.length ? (
                  <select onChange={(e) => setDetailModel(e.target.value)} value={activeDetailModel}>
                    {detailModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label ?? m.name ?? m.id}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="batch-ops-empty">No detail-capable model installed.</span>
                )}
              </label>
              <label>
                Strength {strength.toFixed(2)}
                <input max="1" min="0" onChange={(e) => setStrength(Number(e.target.value))} step="0.05" type="range" value={strength} />
              </label>
              <label>
                ControlNet {cnScale.toFixed(2)}
                <input max="1" min="0" onChange={(e) => setCnScale(Number(e.target.value))} step="0.05" type="range" value={cnScale} />
              </label>
            </div>
          ) : null}

          {op === "edit" ? (
            <div className="batch-ops-params">
              <label>
                Model
                {editModels.length ? (
                  <select onChange={(e) => setEditModel(e.target.value)} value={activeEditModel}>
                    {editModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label ?? m.name ?? m.id}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="batch-ops-empty">No edit-capable model installed.</span>
                )}
              </label>
              <label>
                Prompt
                <textarea
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the edit applied to every selected image…"
                  rows={2}
                  value={prompt}
                />
              </label>
              <label>
                Seed (optional)
                <input
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="random"
                  type="number"
                  value={seed}
                />
              </label>
              <p className="batch-ops-note">Each image is edited at its native size.</p>
            </div>
          ) : null}

          <div className="batch-ops-actions">
            <button onClick={onClose} type="button">
              Cancel
            </button>
            <button className="primary" disabled={!canRun} onClick={run} type="button">
              {busy ? "Starting…" : `Run on ${count} image${count === 1 ? "" : "s"}`}
            </button>
          </div>
        </div>
      )}
    </Modal>
  );
}
