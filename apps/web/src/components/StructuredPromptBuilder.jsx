// Ideogram 4 structured prompt-builder (epic 4725, sc-5994).
//
// Ideogram 4 is prompted with a structured JSON caption, not free text, so this
// form lets the user author a valid, key-ordered caption without hand-writing
// schema. It sits on the sc-5993 contract (apps/web/src/ideogramCaption.js):
// every edit produces a plain caption object that `serializeCaption` emits in
// the model's exact training order, and the parent validates with the same
// `validateCaption`.
//
// Three modes: Builder (the form), JSON (raw edit, validated live), and Plain
// (the free-text fallback that seeds magic-prompt, sc-5997). The visual bbox
// canvas (sc-5995) and rich palette picker (sc-5996) augment this form later;
// here bbox is four numbers and a palette is comma-separated #RRGGBB.

import React, { useEffect, useRef, useState } from "react";
import {
  BBOX_MAX,
  clampBboxValue,
  ELEMENT_PALETTE_MAX,
  makeObjElement,
  makeTextElement,
  orderCaption,
  parseCaption,
  STYLE_PALETTE_MAX,
} from "../ideogramCaption.js";
import PaletteEditor from "./PaletteEditor.jsx";

const MODES = [
  ["form", "Builder"],
  ["json", "JSON"],
  ["plain", "Plain text"],
];

// Pretty-print the caption in canonical order for the read-only preview and the
// JSON editor. The actual engine payload uses the compact `serializeCaption`
// form; whitespace differs but content and key order are identical.
function prettyCaption(caption) {
  return JSON.stringify(orderCaption(caption), null, 2);
}

function getComposition(caption) {
  return caption?.compositional_deconstruction ?? { background: "", elements: [] };
}

function getElements(caption) {
  const els = getComposition(caption).elements;
  return Array.isArray(els) ? els : [];
}

// Four integer inputs for a [ymin, xmin, ymax, xmax] box (0–1000). The visual
// drag canvas is sc-5995; this is the typed fallback.
function BboxField({ bbox, onChange, onRemove }) {
  const labels = ["y min", "x min", "y max", "x max"];
  function setAt(i, raw) {
    const next = (Array.isArray(bbox) ? bbox.slice() : [0, 0, BBOX_MAX, BBOX_MAX]);
    next[i] = clampBboxValue(raw);
    onChange(next);
  }
  return (
    <div className="structured-bbox">
      <span className="structured-bbox-label">Bounding box (0–{BBOX_MAX})</span>
      <div className="structured-bbox-row">
        {labels.map((lab, i) => (
          <label key={lab}>
            <span>{lab}</span>
            <input
              type="number"
              min={0}
              max={BBOX_MAX}
              value={Array.isArray(bbox) ? bbox[i] : ""}
              onChange={(e) => setAt(i, e.target.value)}
            />
          </label>
        ))}
        <button type="button" className="structured-link" onClick={onRemove}>
          Remove box
        </button>
      </div>
    </div>
  );
}

export default function StructuredPromptBuilder({
  caption,
  onCaptionChange,
  validation,
  mode,
  onModeChange,
  plainText,
  onPlainTextChange,
  onMagicExpand,
  magicModelMissing = false,
  onDownloadMagicModel,
}) {
  const composition = getComposition(caption);
  const elements = getElements(caption);

  // Magic-prompt expansion (sc-5997): plain idea -> populated, editable caption.
  const [magicBusy, setMagicBusy] = useState(false);
  const [magicError, setMagicError] = useState("");
  const [magicDownloadRequested, setMagicDownloadRequested] = useState(false);
  const magicModelNeeded =
    magicModelMissing || /not cached|not installed|snapshot is not/i.test(magicError);

  async function handleMagicExpand() {
    if (typeof onMagicExpand !== "function" || !plainText.trim() || magicBusy) return;
    setMagicBusy(true);
    setMagicError("");
    try {
      const next = await onMagicExpand(plainText.trim());
      if (next) {
        onCaptionChange(next);
        onModeChange("form"); // drop the user into the editable builder
      }
    } catch (e) {
      setMagicError(e?.message || "Magic-prompt failed.");
    } finally {
      setMagicBusy(false);
    }
  }

  async function handleDownloadMagicModel() {
    if (typeof onDownloadMagicModel !== "function") return;
    try {
      const job = await onDownloadMagicModel();
      if (job) setMagicDownloadRequested(true);
    } catch (e) {
      setMagicError(e?.message || "Could not start the model download.");
    }
  }

  // Stable React keys for the repeatable element rows so child field state stays
  // aligned with its element across add/remove (elements carry no id of their
  // own — an id key would be an unknown schema key).
  const keyCounter = useRef(0);
  const [elementKeys, setElementKeys] = useState(() => elements.map(() => (keyCounter.current += 1)));
  useEffect(() => {
    setElementKeys((prev) => {
      if (prev.length === elements.length) return prev;
      const next = prev.slice(0, elements.length);
      while (next.length < elements.length) next.push((keyCounter.current += 1));
      return next;
    });
  }, [elements.length]);

  // JSON editor buffer — seeded on entering JSON mode, then user-controlled.
  const [jsonDraft, setJsonDraft] = useState(() => prettyCaption(caption));
  const [jsonError, setJsonError] = useState(null);
  useEffect(() => {
    if (mode === "json") {
      setJsonDraft(prettyCaption(caption));
      setJsonError(null);
    }
    // Only re-seed when switching into JSON mode, not on every caption keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // ----- immutable caption updates -----
  function patchCaption(next) {
    onCaptionChange(next);
  }
  function setHighLevel(value) {
    const next = { ...caption };
    if (value.trim()) next.high_level_description = value;
    else delete next.high_level_description;
    patchCaption(next);
  }
  function setBackground(value) {
    patchCaption({
      ...caption,
      compositional_deconstruction: { ...composition, background: value },
    });
  }
  function setElements(nextElements) {
    patchCaption({
      ...caption,
      compositional_deconstruction: { ...composition, elements: nextElements },
    });
  }
  function updateElement(index, updater) {
    setElements(elements.map((el, i) => (i === index ? updater(el) : el)));
  }
  function addElement(type) {
    const el = type === "text" ? makeTextElement({}) : makeObjElement({});
    setElementKeys((prev) => [...prev, (keyCounter.current += 1)]);
    setElements([...elements, el]);
  }
  function removeElement(index) {
    setElementKeys((prev) => prev.filter((_, i) => i !== index));
    setElements(elements.filter((_, i) => i !== index));
  }
  function setElementType(index, type) {
    updateElement(index, (el) => {
      if (el.type === type) return el;
      const base = { type };
      if ("bbox" in el) base.bbox = el.bbox;
      if (type === "text") base.text = el.text ?? "";
      base.desc = el.desc ?? "";
      if ("color_palette" in el) base.color_palette = el.color_palette;
      return base;
    });
  }

  // ----- style helpers (style_description is optional) -----
  const style = caption.style_description ?? null;
  const styleKind = style && "art_style" in style && !("photo" in style) ? "art" : "photo";
  function setStyleEnabled(on) {
    if (on) {
      patchCaption({ ...caption, style_description: { aesthetics: "", lighting: "", photo: "", medium: "" } });
    } else {
      const next = { ...caption };
      delete next.style_description;
      patchCaption(next);
    }
  }
  function setStyleField(field, value) {
    patchCaption({ ...caption, style_description: { ...style, [field]: value } });
  }
  function setStyleKind(kind) {
    if (!style) return;
    const next = { ...style };
    const current = next.photo ?? next.art_style ?? "";
    delete next.photo;
    delete next.art_style;
    if (kind === "art") next.art_style = current;
    else next.photo = current;
    patchCaption({ ...caption, style_description: next });
  }
  function setStyleDiscriminatorValue(value) {
    setStyleField(styleKind === "art" ? "art_style" : "photo", value);
  }

  // ----- JSON mode -----
  function onJsonChange(text) {
    setJsonDraft(text);
    const { caption: parsed, error } = parseCaption(text);
    if (error) {
      setJsonError(error);
      return;
    }
    setJsonError(null);
    onCaptionChange(parsed);
  }

  const errors = validation?.errors ?? [];
  const warnings = validation?.warnings ?? [];

  return (
    <div className="structured-prompt-builder">
      <div className="segmented-control structured-mode" role="tablist" aria-label="Prompt mode">
        {MODES.map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={mode === value}
            className={mode === value ? "active" : ""}
            onClick={() => onModeChange(value)}
          >
            {label}
          </button>
        ))}
      </div>

      {mode === "plain" ? (
        <div className="structured-plain">
          <textarea
            aria-label="Plain prompt"
            className="prompt-input"
            placeholder="Describe your idea in plain language…"
            value={plainText}
            onChange={(e) => onPlainTextChange(e.target.value)}
          />
          <p className="structured-hint">
            Ideogram 4 was trained on structured captions — plain text produces a coherent but
            prompt-agnostic image. Use the Builder for accurate adherence, or expand this idea into a
            caption with magic-prompt.
          </p>
          {typeof onMagicExpand === "function" ? (
            <div className="structured-magic">
              <button
                type="button"
                className="secondary-action"
                disabled={!plainText.trim() || magicBusy}
                onClick={handleMagicExpand}
              >
                {magicBusy ? "Expanding…" : "✨ Expand to caption"}
              </button>
              {magicError && magicModelNeeded ? (
                <div className="structured-magic-missing" role="alert">
                  {magicDownloadRequested ? (
                    <p className="structured-hint">
                      Downloading the prompt-refiner model… track it on the Models screen, then try again.
                    </p>
                  ) : (
                    <>
                      <p className="structured-error">The prompt-refiner model isn’t installed yet.</p>
                      {typeof onDownloadMagicModel === "function" ? (
                        <button type="button" className="secondary-action" onClick={handleDownloadMagicModel}>
                          Download prompt-refiner model
                        </button>
                      ) : (
                        <p className="structured-hint">Open the Models screen to download it.</p>
                      )}
                    </>
                  )}
                </div>
              ) : magicError ? (
                <p className="structured-error" role="alert">
                  {magicError}
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      {mode === "json" ? (
        <div className="structured-json">
          <textarea
            aria-label="JSON caption"
            className="prompt-input structured-json-input"
            spellCheck={false}
            value={jsonDraft}
            onChange={(e) => onJsonChange(e.target.value)}
          />
          {jsonError ? <p className="structured-error">Invalid JSON: {jsonError}</p> : null}
        </div>
      ) : null}

      {mode === "form" ? (
        <div className="structured-form">
          <label className="structured-field">
            <span>High-level description</span>
            <input
              type="text"
              placeholder="One sentence summarizing the whole image"
              value={caption.high_level_description ?? ""}
              onChange={(e) => setHighLevel(e.target.value)}
            />
          </label>

          <fieldset className="structured-style">
            <legend>
              <label className="structured-checkline">
                <input type="checkbox" checked={Boolean(style)} onChange={(e) => setStyleEnabled(e.target.checked)} />
                Style
              </label>
            </legend>
            {style ? (
              <div className="structured-style-fields">
                <label className="structured-field">
                  <span>Aesthetics</span>
                  <input type="text" placeholder="serene, warm, naturalistic" value={style.aesthetics ?? ""} onChange={(e) => setStyleField("aesthetics", e.target.value)} />
                </label>
                <label className="structured-field">
                  <span>Lighting</span>
                  <input type="text" placeholder="golden hour, soft backlight" value={style.lighting ?? ""} onChange={(e) => setStyleField("lighting", e.target.value)} />
                </label>
                <div className="structured-field">
                  <span>Kind</span>
                  <div className="segmented-control structured-kind" role="tablist" aria-label="Style kind">
                    <button type="button" aria-selected={styleKind === "photo"} className={styleKind === "photo" ? "active" : ""} onClick={() => setStyleKind("photo")}>
                      Photo
                    </button>
                    <button type="button" aria-selected={styleKind === "art"} className={styleKind === "art" ? "active" : ""} onClick={() => setStyleKind("art")}>
                      Art
                    </button>
                  </div>
                </div>
                <label className="structured-field">
                  <span>{styleKind === "art" ? "Art style" : "Photo"}</span>
                  <input
                    type="text"
                    placeholder={styleKind === "art" ? "watercolor illustration" : "telephoto, shallow depth of field, eye-level"}
                    value={(styleKind === "art" ? style.art_style : style.photo) ?? ""}
                    onChange={(e) => setStyleDiscriminatorValue(e.target.value)}
                  />
                </label>
                <label className="structured-field">
                  <span>Medium</span>
                  <input type="text" placeholder="photograph, oil painting" value={style.medium ?? ""} onChange={(e) => setStyleField("medium", e.target.value)} />
                </label>
                <PaletteEditor
                  label={`Color palette (max ${STYLE_PALETTE_MAX})`}
                  value={style.color_palette}
                  max={STYLE_PALETTE_MAX}
                  onChange={(colors) => {
                    const next = { ...style };
                    if (colors) next.color_palette = colors;
                    else delete next.color_palette;
                    patchCaption({ ...caption, style_description: next });
                  }}
                />
              </div>
            ) : null}
          </fieldset>

          <label className="structured-field">
            <span>Background</span>
            <textarea
              className="structured-textarea"
              placeholder="The scene behind the elements"
              value={composition.background ?? ""}
              onChange={(e) => setBackground(e.target.value)}
            />
          </label>

          <div className="structured-elements">
            <div className="structured-elements-head">
              <span>Elements</span>
              <div>
                <button type="button" className="structured-link" onClick={() => addElement("obj")}>
                  + Object
                </button>
                <button type="button" className="structured-link" onClick={() => addElement("text")}>
                  + Text
                </button>
              </div>
            </div>
            {elements.length === 0 ? <p className="structured-hint">Add objects and text blocks to place them on the canvas.</p> : null}
            {elements.map((el, i) => (
              <div className="structured-element" key={elementKeys[i] ?? i}>
                <div className="structured-element-head">
                  <div className="segmented-control structured-kind" role="tablist" aria-label={`Element ${i + 1} type`}>
                    <button type="button" aria-selected={el.type !== "text"} className={el.type !== "text" ? "active" : ""} onClick={() => setElementType(i, "obj")}>
                      Object
                    </button>
                    <button type="button" aria-selected={el.type === "text"} className={el.type === "text" ? "active" : ""} onClick={() => setElementType(i, "text")}>
                      Text
                    </button>
                  </div>
                  <button type="button" className="structured-link" onClick={() => removeElement(i)}>
                    Remove
                  </button>
                </div>
                {el.type === "text" ? (
                  <label className="structured-field">
                    <span>Text to render</span>
                    <input type="text" placeholder="The exact characters to render" value={el.text ?? ""} onChange={(e) => updateElement(i, (cur) => ({ ...cur, text: e.target.value }))} />
                  </label>
                ) : null}
                <label className="structured-field">
                  <span>Description</span>
                  <textarea className="structured-textarea" placeholder="Material, pose, expression, lighting on this element" value={el.desc ?? ""} onChange={(e) => updateElement(i, (cur) => ({ ...cur, desc: e.target.value }))} />
                </label>
                {"bbox" in el ? (
                  <BboxField
                    bbox={el.bbox}
                    onChange={(bbox) => updateElement(i, (cur) => ({ ...cur, bbox }))}
                    onRemove={() => updateElement(i, (cur) => {
                      const next = { ...cur };
                      delete next.bbox;
                      return next;
                    })}
                  />
                ) : (
                  <button type="button" className="structured-link" onClick={() => updateElement(i, (cur) => ({ ...cur, bbox: [0, 0, BBOX_MAX, BBOX_MAX] }))}>
                    + Bounding box
                  </button>
                )}
                <PaletteEditor
                  label={`Palette (max ${ELEMENT_PALETTE_MAX})`}
                  value={el.color_palette}
                  max={ELEMENT_PALETTE_MAX}
                  onChange={(colors) => updateElement(i, (cur) => {
                    const next = { ...cur };
                    if (colors) next.color_palette = colors;
                    else delete next.color_palette;
                    return next;
                  })}
                />
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {mode !== "plain" ? (
        <div className="structured-preview">
          {/* The JSON tab's textarea already shows the canonical JSON, so the
              read-only preview pane is a duplicate there — only render it in the
              form builder. Schema errors/warnings stay on both tabs (sc-8114). */}
          {mode === "form" ? (
            <>
              <div className="structured-preview-head">Generated caption (sent to the model)</div>
              <pre aria-label="Caption preview">{prettyCaption(caption)}</pre>
            </>
          ) : null}
          {errors.length ? (
            <ul className="structured-issues structured-issues-error">
              {errors.map((issue, i) => (
                <li key={`e${i}`}>{issue.message}</li>
              ))}
            </ul>
          ) : null}
          {warnings.length ? (
            <ul className="structured-issues structured-issues-warn">
              {warnings.map((issue, i) => (
                <li key={`w${i}`}>{issue.message}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
