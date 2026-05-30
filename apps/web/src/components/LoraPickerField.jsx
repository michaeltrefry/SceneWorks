import React from "react";
import { loraMatchesModel, loraWeight, serializeLora } from "../presetUtils.js";

// Keep in sync with the worker guard (lora_adapters.MAX_JOB_LORAS) and the Rust
// recipe-preset normalizer — generation rejects more than this many LoRAs per job.
const MAX_JOB_LORAS = 3;

// Family-filtered LoRA selection shared by the Character Studio Angle Set + Pose
// Library panels (sc-2223). Mirrors the Image/Video Studio behaviour without
// rebuilding the matcher: list only LoRAs whose family matches the active backbone
// (loraMatchesModel, sc-1927), expose a per-LoRA weight that defaults to the
// manifest weight, and emit the serialized `loras` array (serializeLora) the
// character_image payload carries top-level.
export function useLoraSelection(loras, model) {
  const compatibleLoras = React.useMemo(
    () => (Array.isArray(loras) ? loras.filter((lora) => loraMatchesModel(lora, model)) : []),
    [loras, model],
  );
  const compatibleKey = compatibleLoras.map((lora) => lora.id).join("|");
  const [selectedLoraIds, setSelectedLoraIds] = React.useState([]);
  const [weights, setWeights] = React.useState({});

  // Drop selections that no longer match when the backbone changes (a stale id from
  // another family must not ride along into the payload). Mirrors ImageStudio.
  React.useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compatibleKey]);

  const weightFor = React.useCallback(
    (lora) => {
      const override = Number(weights[lora.id]);
      return Number.isFinite(override) ? override : loraWeight(lora);
    },
    [weights],
  );

  const toggleLora = React.useCallback((lora) => {
    setSelectedLoraIds((ids) => {
      if (ids.includes(lora.id)) {
        return ids.filter((id) => id !== lora.id);
      }
      if (ids.length >= MAX_JOB_LORAS) {
        return ids; // the worker rejects more than MAX_JOB_LORAS per job
      }
      return [...ids, lora.id];
    });
  }, []);

  const setWeight = React.useCallback((id, value) => {
    setWeights((current) => ({ ...current, [id]: Number(value) }));
  }, []);

  const serializedLoras = React.useMemo(
    () =>
      selectedLoraIds
        .map((id) => compatibleLoras.find((lora) => lora.id === id))
        .filter(Boolean)
        .map((lora) => serializeLora(lora, { weight: weightFor(lora) })),
    [selectedLoraIds, compatibleLoras, weightFor],
  );

  return { compatibleLoras, selectedLoraIds, toggleLora, weightFor, setWeight, serializedLoras };
}

// Presentational picker driven by a useLoraSelection() result. Renders nothing when
// the active backbone has no compatible LoRAs (the feature is optional — no clutter).
export function LoraPickerField({ selection, label = "Style LoRAs (optional)" }) {
  const { compatibleLoras, selectedLoraIds, toggleLora, weightFor, setWeight } = selection;
  if (!compatibleLoras.length) {
    return null;
  }
  // Mirrors the ImageStudio picker markup so it reuses the existing LoRA-choice styles.
  return (
    <div className="character-lora-picker">
      <p className="eyebrow">{label}</p>
      <div className="lora-choice-list">
        {compatibleLoras.map((lora) => {
          const checked = selectedLoraIds.includes(lora.id);
          const weight = weightFor(lora);
          return (
            <div className="lora-choice-item" key={lora.id}>
              <label className={checked ? "lora-choice active" : "lora-choice"}>
                <input checked={checked} onChange={() => toggleLora(lora)} type="checkbox" />
                <span>
                  <strong>{lora.name ?? lora.id}</strong>
                  <small>
                    {lora.scope ?? "global"} {lora.family ? `| ${lora.family}` : ""}
                  </small>
                </span>
              </label>
              {checked ? (
                <div className="lora-weight-row">
                  <span>Weight</span>
                  <input
                    aria-label={`${lora.name ?? lora.id} weight`}
                    max="2"
                    min="0"
                    onChange={(event) => setWeight(lora.id, Number(event.target.value))}
                    step="0.05"
                    type="range"
                    value={weight}
                  />
                  <span className="lora-weight-value">{weight.toFixed(2)}</span>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
