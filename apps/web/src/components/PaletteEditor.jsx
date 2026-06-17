// Color-palette control for the Ideogram 4 prompt-builder (epic 4725, sc-5996).
//
// Authors an ordered list of uppercase #RRGGBB colors via swatch chips + a
// native color picker. Used for the caption's overall/document palette
// (style_description.color_palette, ≤16); the per-element palette and the
// box-tool authoring path live in epic 6087. Colors are normalized and capped
// here; the ≤max / #RRGGBB validation and order-preserving emission are the
// sc-5993 contract's job (this just produces a clean array).

import React, { useState } from "react";
import { isValidHexColor, normalizeHexColor } from "../ideogramCaption.js";

const DEFAULT_DRAFT = "#888888";

export default function PaletteEditor({ value, onChange, max, label }) {
  const colors = Array.isArray(value) ? value : [];
  const [draft, setDraft] = useState(DEFAULT_DRAFT);

  const normalizedDraft = normalizeHexColor(draft);
  const atMax = colors.length >= max;
  const duplicate = normalizedDraft != null && colors.includes(normalizedDraft);
  const canAdd = normalizedDraft != null && !atMax && !duplicate;

  function addColor() {
    if (!canAdd) return;
    onChange([...colors, normalizedDraft]);
  }
  function removeAt(index) {
    const next = colors.filter((_, i) => i !== index);
    onChange(next.length ? next : null);
  }

  return (
    <div className="palette-editor">
      <div className="palette-editor-head">
        <span>{label}</span>
        <span className="palette-editor-count">
          {colors.length}/{max}
        </span>
      </div>
      {colors.length ? (
        <ul className="palette-swatches">
          {colors.map((color, i) => (
            <li key={`${color}-${i}`} className="palette-swatch">
              <span className="palette-swatch-chip" style={{ background: color }} aria-hidden="true" />
              <span className="palette-swatch-hex">{color}</span>
              <button type="button" className="palette-swatch-remove" aria-label={`Remove ${color}`} onClick={() => removeAt(i)}>
                ×
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="palette-editor-add">
        <input
          type="color"
          aria-label="Pick color"
          value={isValidHexColor(draft) ? draft.toLowerCase() : DEFAULT_DRAFT}
          onChange={(e) => setDraft(e.target.value.toUpperCase())}
          disabled={atMax}
        />
        <input
          type="text"
          aria-label="Hex color"
          className="palette-hex-input"
          placeholder="#RRGGBB"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={atMax}
        />
        <button type="button" className="structured-link" onClick={addColor} disabled={!canAdd}>
          Add
        </button>
      </div>
      {atMax ? <p className="structured-hint">Maximum {max} colors.</p> : null}
      {!atMax && normalizedDraft == null && draft.trim() ? (
        <p className="structured-hint">Enter an uppercase #RRGGBB hex color.</p>
      ) : null}
    </div>
  );
}
